#include <jni.h>

#include <android/log.h>

#include <atomic>
#include <cctype>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "inference_backend.h"

#if defined(OMNIINFER_BACKEND_LLAMA_CPP)
#include "backend_llama_cpp.h"
#endif

#if defined(OMNIINFER_BACKEND_MNN)
#include "backend_mnn.h"
#endif

#if defined(OMNIINFER_BACKEND_EXECUTORCH_QNN)
#include "backend_executorch_qnn.h"
#endif

namespace {

constexpr const char* kTag = "OmniInferJni";

void LogPrint(int priority, const std::string& message) {
  __android_log_print(priority, kTag, "%s", message.c_str());
}

// ---------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------

struct Session {
  int64_t handle = 0;
  std::unique_ptr<omniinfer::InferenceBackend> backend;
  std::atomic<bool> cancelled{false};
  std::atomic<bool> graceful_stop{false};  // true = /v1/cancel, false = hard cancel (disconnect)
  bool thinking_enabled = false;
};

std::mutex g_sessions_mutex;
std::unordered_map<int64_t, Session*> g_sessions;
std::atomic<int64_t> g_next_handle{1};

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

std::string JStringToStdString(JNIEnv* env, jstring value) {
  if (!value) return "";
  const char* chars = env->GetStringUTFChars(value, nullptr);
  if (!chars) return "";
  std::string result(chars);
  env->ReleaseStringUTFChars(value, chars);
  return result;
}

jstring StdStringToJString(JNIEnv* env, const std::string& value) {
  // Convert UTF-8 to UTF-16 (jchar array) for JNI. This correctly handles
  // 4-byte UTF-8 sequences (emoji, etc.) by encoding them as surrogate pairs,
  // which NewStringUTF (Modified UTF-8) cannot do.
  std::vector<jchar> utf16;
  utf16.reserve(value.size());
  size_t i = 0;
  while (i < value.size()) {
    unsigned char c = static_cast<unsigned char>(value[i]);
    uint32_t cp = 0;
    int len = 0;
    if (c < 0x80) { cp = c; len = 1; }
    else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
    else { utf16.push_back(0xFFFD); i++; continue; } // invalid byte

    // Validate and decode continuation bytes.
    bool valid = (i + len <= value.size());
    for (int j = 1; j < len && valid; j++) {
      if ((static_cast<unsigned char>(value[i + j]) & 0xC0) != 0x80) valid = false;
      else cp = (cp << 6) | (value[i + j] & 0x3F);
    }
    if (!valid) { utf16.push_back(0xFFFD); i++; continue; }
    i += len;

    // Encode codepoint as UTF-16.
    if (cp <= 0xFFFF) {
      utf16.push_back(static_cast<jchar>(cp));
    } else if (cp <= 0x10FFFF) {
      // Surrogate pair for emoji and other supplementary characters.
      cp -= 0x10000;
      utf16.push_back(static_cast<jchar>(0xD800 | (cp >> 10)));
      utf16.push_back(static_cast<jchar>(0xDC00 | (cp & 0x3FF)));
    } else {
      utf16.push_back(0xFFFD);
    }
  }
  return env->NewString(utf16.data(), static_cast<jsize>(utf16.size()));
}

std::optional<std::string> ExtractJsonString(const std::string& json, const std::string& key) {
  const std::string token = "\"" + key + "\"";
  size_t key_pos = json.find(token);
  if (key_pos == std::string::npos) return std::nullopt;
  size_t colon_pos = json.find(':', key_pos + token.size());
  if (colon_pos == std::string::npos) return std::nullopt;
  size_t pos = colon_pos + 1;
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
  if (pos >= json.size() || json.compare(pos, 4, "null") == 0) return std::nullopt;
  if (json[pos] != '"') return std::nullopt;
  ++pos;
  std::string out;
  bool esc = false;
  while (pos < json.size()) {
    char ch = json[pos++];
    if (esc) { out.push_back(ch == 'n' ? '\n' : ch == 't' ? '\t' : ch); esc = false; continue; }
    if (ch == '\\') { esc = true; continue; }
    if (ch == '"') return out;
    out.push_back(ch);
  }
  return std::nullopt;
}

// Extract a raw JSON value (array or object) for a given key.
std::optional<std::string> ExtractJsonRaw(const std::string& json, const std::string& key) {
  const std::string token = "\"" + key + "\"";
  size_t key_pos = json.find(token);
  if (key_pos == std::string::npos) return std::nullopt;
  size_t colon_pos = json.find(':', key_pos + token.size());
  if (colon_pos == std::string::npos) return std::nullopt;
  size_t pos = colon_pos + 1;
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
  if (pos >= json.size()) return std::nullopt;
  char open = json[pos];
  if (open != '[' && open != '{') return std::nullopt;
  char close = (open == '[') ? ']' : '}';
  int depth = 0;
  bool in_str = false;
  for (size_t i = pos; i < json.size(); i++) {
    char c = json[i];
    if (in_str) { if (c == '"' && json[i-1] != '\\') in_str = false; continue; }
    if (c == '"') { in_str = true; continue; }
    if (c == open) depth++;
    else if (c == close) { depth--; if (depth == 0) return json.substr(pos, i - pos + 1); }
  }
  return std::nullopt;
}

std::optional<int> ExtractJsonInt(const std::string& json, const std::string& key) {
  const std::string token = "\"" + key + "\"";
  size_t key_pos = json.find(token);
  if (key_pos == std::string::npos) return std::nullopt;
  size_t colon_pos = json.find(':', key_pos + token.size());
  if (colon_pos == std::string::npos) return std::nullopt;
  size_t pos = colon_pos + 1;
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
  size_t end = pos;
  while (end < json.size() && (std::isdigit(static_cast<unsigned char>(json[end])) || json[end] == '-')) ++end;
  if (end == pos) return std::nullopt;
  return std::stoi(json.substr(pos, end - pos));
}

std::optional<bool> ExtractJsonBool(const std::string& json, const std::string& key) {
  const std::string token = "\"" + key + "\"";
  size_t key_pos = json.find(token);
  if (key_pos == std::string::npos) return std::nullopt;
  size_t colon_pos = json.find(':', key_pos + token.size());
  if (colon_pos == std::string::npos) return std::nullopt;
  size_t pos = colon_pos + 1;
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
  if (json.compare(pos, 4, "true") == 0) return true;
  if (json.compare(pos, 5, "false") == 0) return false;
  return std::nullopt;
}

void InvokeCallbackStringMethod(JNIEnv* env, jobject callback, const char* method_name, const std::string& payload) {
  if (!callback) return;
  if (env->ExceptionCheck()) { env->ExceptionClear(); return; }
  jclass cls = env->GetObjectClass(callback);
  if (!cls) return;
  jmethodID method = env->GetMethodID(cls, method_name, "(Ljava/lang/String;)V");
  if (!method) { env->ExceptionClear(); env->DeleteLocalRef(cls); return; }
  jstring message = StdStringToJString(env, payload);
  env->CallVoidMethod(callback, method, message);
  env->DeleteLocalRef(message);
  env->DeleteLocalRef(cls);
}

// ---------------------------------------------------------------------------
// JNI native methods
// ---------------------------------------------------------------------------

jlong NativeInit(JNIEnv* env, jobject, jstring config_json) {
  const std::string config = JStringToStdString(env, config_json);
  const auto backend_name = ExtractJsonString(config, "backend").value_or("llama.cpp");
  const auto model_path = ExtractJsonString(config, "model_path");
  const auto native_lib_dir = ExtractJsonString(config, "native_lib_dir").value_or("");
  const int n_threads = ExtractJsonInt(config, "n_threads").value_or(4);
  const int n_ctx = ExtractJsonInt(config, "n_ctx").value_or(4096);

  if (!model_path.has_value()) {
    LogPrint(ANDROID_LOG_ERROR, "NativeInit: missing model_path");
    return 0;
  }

  // Create backend by name.
  std::unique_ptr<omniinfer::InferenceBackend> backend;
#if defined(OMNIINFER_BACKEND_MNN)
  if (backend_name == "mnn") {
    backend = std::make_unique<omniinfer::MnnBackend>();
  }
#endif
#if defined(OMNIINFER_BACKEND_LLAMA_CPP)
  if (!backend && (backend_name == "llama.cpp" || backend_name.empty())) {
    backend = std::make_unique<omniinfer::LlamaCppBackend>();
  }
#endif
#if defined(OMNIINFER_BACKEND_EXECUTORCH_QNN)
  if (!backend && backend_name == "executorch-qnn") {
    backend = std::make_unique<omniinfer::ExecuTorchQnnBackend>();
  }
#endif
  if (!backend) {
    LogPrint(ANDROID_LOG_ERROR, "NativeInit: unknown or disabled backend: " + backend_name);
    return 0;
  }

  LogPrint(ANDROID_LOG_INFO, "NativeInit: backend=" + backend_name + " model=" + *model_path);

  if (!backend->load(*model_path, config, native_lib_dir, n_threads, n_ctx)) {
    LogPrint(ANDROID_LOG_ERROR, "NativeInit: " + backend_name + " load failed");
    return 0;
  }

  auto* session = new Session();
  session->handle = g_next_handle.fetch_add(1);
  session->backend = std::move(backend);

  {
    std::lock_guard<std::mutex> guard(g_sessions_mutex);
    g_sessions[session->handle] = session;
  }

  LogPrint(ANDROID_LOG_INFO, "NativeInit: session " + std::to_string(session->handle) +
           " created (" + session->backend->name() + ")");
  return static_cast<jlong>(session->handle);
}

jstring NativeGenerate(JNIEnv* env, jobject, jlong handle, jstring system_prompt,
                       jstring prompt, jstring request_json, jobjectArray image_data_array, jobject callback) {
  Session* session = nullptr;
  {
    std::lock_guard<std::mutex> guard(g_sessions_mutex);
    auto it = g_sessions.find(static_cast<int64_t>(handle));
    if (it == g_sessions.end()) return StdStringToJString(env, "");
    session = it->second;
  }

  session->cancelled.store(false);
  session->graceful_stop.store(false);
  const std::string req = JStringToStdString(env, request_json);

  const bool thinking = ExtractJsonBool(req, "thinking_enabled").value_or(session->thinking_enabled);
  const int max_tokens = ExtractJsonInt(req, "max_tokens").value_or(0);
  const auto tools_json = ExtractJsonRaw(req, "tools");
  const auto tool_choice = ExtractJsonString(req, "tool_choice");
  const auto messages_json = ExtractJsonRaw(req, "messages");

  // Legacy path: use system_prompt/prompt if messages not in requestJson.
  const std::string sys = messages_json.has_value() ? "" : JStringToStdString(env, system_prompt);
  const std::string user = messages_json.has_value() ? "" : JStringToStdString(env, prompt);

  auto on_token = [&](const std::string& token) -> bool {
    InvokeCallbackStringMethod(env, callback, "onToken", token);
    return !session->cancelled.load();
  };

  // Extract image bytes from Array<ByteArray>.
  std::vector<std::vector<uint8_t>> images;
  if (image_data_array) {
    jsize n = env->GetArrayLength(image_data_array);
    for (jsize i = 0; i < n; i++) {
      auto arr = reinterpret_cast<jbyteArray>(env->GetObjectArrayElement(image_data_array, i));
      if (!arr) continue;
      jsize len = env->GetArrayLength(arr);
      if (len <= 0) { env->DeleteLocalRef(arr); continue; }
      jbyte* bytes = env->GetByteArrayElements(arr, nullptr);
      images.emplace_back(reinterpret_cast<uint8_t*>(bytes),
                          reinterpret_cast<uint8_t*>(bytes) + len);
      env->ReleaseByteArrayElements(arr, bytes, JNI_ABORT);
      env->DeleteLocalRef(arr);
    }
  }

  std::string result = session->backend->generate(sys, user, thinking, session->cancelled, on_token,
      tools_json.value_or(""), tool_choice.value_or(""), messages_json.value_or(""),
      images, max_tokens, session->graceful_stop);

  // Report metrics.
  auto m = session->backend->get_metrics();
  if (m.generated_tokens > 0) {
    double decode_tps = m.decode_us > 0 ? m.generated_tokens / (m.decode_us / 1e6) : 0;
    double prefill_tps = m.prefill_us > 0 ? m.prompt_tokens / (m.prefill_us / 1e6) : 0;
    std::ostringstream metrics;
    metrics << "prefill_tps=" << prefill_tps << ", decode_tps=" << decode_tps;
    InvokeCallbackStringMethod(env, callback, "onMetrics", metrics.str());
  }

  return StdStringToJString(env, result);
}

jboolean NativeLoadHistory(JNIEnv* env, jobject, jlong handle, jobjectArray roles, jobjectArray contents) {
  std::lock_guard<std::mutex> guard(g_sessions_mutex);
  auto it = g_sessions.find(static_cast<int64_t>(handle));
  if (it == g_sessions.end()) return JNI_FALSE;

  const jsize count = roles ? env->GetArrayLength(roles) : 0;
  if (count != (contents ? env->GetArrayLength(contents) : 0)) return JNI_FALSE;

  std::vector<std::pair<std::string, std::string>> msgs;
  for (jsize i = 0; i < count; i++) {
    auto* jr = static_cast<jstring>(env->GetObjectArrayElement(roles, i));
    auto* jc = static_cast<jstring>(env->GetObjectArrayElement(contents, i));
    msgs.push_back({JStringToStdString(env, jr), JStringToStdString(env, jc)});
    env->DeleteLocalRef(jr);
    env->DeleteLocalRef(jc);
  }
  return it->second->backend->load_history(msgs) ? JNI_TRUE : JNI_FALSE;
}

jboolean NativePrewarmImage(JNIEnv*, jobject, jlong, jbyteArray, jint) {
  return JNI_FALSE;
}

void NativeSetThinkMode(JNIEnv*, jobject, jlong handle, jboolean enabled) {
  std::lock_guard<std::mutex> guard(g_sessions_mutex);
  auto it = g_sessions.find(static_cast<int64_t>(handle));
  if (it != g_sessions.end()) it->second->thinking_enabled = enabled == JNI_TRUE;
}

void NativeReset(JNIEnv*, jobject, jlong handle) {
  std::lock_guard<std::mutex> guard(g_sessions_mutex);
  auto it = g_sessions.find(static_cast<int64_t>(handle));
  if (it != g_sessions.end()) it->second->backend->reset();
}

void NativeCancel(JNIEnv*, jobject, jlong handle) {
  std::lock_guard<std::mutex> guard(g_sessions_mutex);
  auto it = g_sessions.find(static_cast<int64_t>(handle));
  if (it != g_sessions.end()) it->second->cancelled.store(true);
}

void NativeGracefulStop(JNIEnv*, jobject, jlong handle) {
  std::lock_guard<std::mutex> guard(g_sessions_mutex);
  auto it = g_sessions.find(static_cast<int64_t>(handle));
  if (it != g_sessions.end()) {
    it->second->graceful_stop.store(true);
    it->second->cancelled.store(true);
  }
}

void NativeFree(JNIEnv*, jobject, jlong handle) {
  Session* session = nullptr;
  {
    std::lock_guard<std::mutex> guard(g_sessions_mutex);
    auto it = g_sessions.find(static_cast<int64_t>(handle));
    if (it == g_sessions.end()) return;
    session = it->second;
    g_sessions.erase(it);
  }
  delete session;
  LogPrint(ANDROID_LOG_INFO, "NativeFree: session " + std::to_string(handle) + " destroyed");
}

jstring NativeCollectDiagnosticsJson(JNIEnv* env, jobject, jlong handle) {
  std::lock_guard<std::mutex> guard(g_sessions_mutex);
  auto it = g_sessions.find(static_cast<int64_t>(handle));
  if (it == g_sessions.end()) return StdStringToJString(env, "{}");
  auto m = it->second->backend->get_metrics();
  std::ostringstream json;
  json << "{"
       << "\"backend\":\"" << it->second->backend->name() << "\","
       << "\"prompt_tokens\":" << m.prompt_tokens << ","
       << "\"generated_tokens\":" << m.generated_tokens << ","
       << "\"prefill_us\":" << m.prefill_us << ","
       << "\"decode_us\":" << m.decode_us << ","
       << "\"reasoning_tokens\":" << m.reasoning_tokens << ","
       << "\"image_tokens\":" << m.image_tokens << ","
       << "\"cached_tokens\":" << m.cached_tokens << ","
       << "\"n_threads\":" << it->second->backend->n_threads()
       << "}";
  return StdStringToJString(env, json.str());
}

// ---------------------------------------------------------------------------
// JNI registration
// ---------------------------------------------------------------------------

JNINativeMethod kMethods[] = {
    {"nativeInit", "(Ljava/lang/String;)J", reinterpret_cast<void*>(NativeInit)},
    {"nativeGenerate", "(JLjava/lang/String;Ljava/lang/String;Ljava/lang/String;[[BLcom/omniinfer/server/OmniInferStreamCallback;)Ljava/lang/String;", reinterpret_cast<void*>(NativeGenerate)},
    {"nativeLoadHistory", "(J[Ljava/lang/String;[Ljava/lang/String;)Z", reinterpret_cast<void*>(NativeLoadHistory)},
    {"nativePrewarmImage", "(J[BI)Z", reinterpret_cast<void*>(NativePrewarmImage)},
    {"nativeSetThinkMode", "(JZ)V", reinterpret_cast<void*>(NativeSetThinkMode)},
    {"nativeReset", "(J)V", reinterpret_cast<void*>(NativeReset)},
    {"nativeCancel", "(J)V", reinterpret_cast<void*>(NativeCancel)},
    {"nativeGracefulStop", "(J)V", reinterpret_cast<void*>(NativeGracefulStop)},
    {"nativeFree", "(J)V", reinterpret_cast<void*>(NativeFree)},
    {"nativeCollectDiagnosticsJson", "(J)Ljava/lang/String;", reinterpret_cast<void*>(NativeCollectDiagnosticsJson)},
};

}  // namespace

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  JNIEnv* env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK || !env) return JNI_ERR;
  jclass bridge_class = env->FindClass("com/omniinfer/server/OmniInferBridge");
  if (!bridge_class) { LogPrint(ANDROID_LOG_ERROR, "Failed to resolve JNI bridge class."); return JNI_ERR; }
  if (env->RegisterNatives(bridge_class, kMethods, sizeof(kMethods) / sizeof(kMethods[0])) != JNI_OK) {
    env->DeleteLocalRef(bridge_class);
    LogPrint(ANDROID_LOG_ERROR, "RegisterNatives failed.");
    return JNI_ERR;
  }
  env->DeleteLocalRef(bridge_class);
  return JNI_VERSION_1_6;
}
