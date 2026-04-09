// OmniInfer iOS C bridge — mirrors android omniinfer_jni.cpp.
// Compiled by CMake into the xcframework alongside llama.cpp.

#include "omniinfer_bridge.h"
#include "inference_backend.h"

#if defined(OMNIINFER_BACKEND_LLAMA_CPP)
#include "backend_llama_cpp.h"
#endif

#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Session {
  int64_t handle = 0;
  std::unique_ptr<omniinfer::InferenceBackend> backend;
  std::atomic<bool> cancelled{false};
  bool thinking_enabled = false;
};

std::mutex g_sessions_mutex;
std::unordered_map<int64_t, Session *> g_sessions;
std::atomic<int64_t> g_next_handle{1};

std::optional<std::string> ExtractJsonString(const std::string &json,
                                             const std::string &key) {
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

std::optional<int> ExtractJsonInt(const std::string &json, const std::string &key) {
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

std::optional<bool> ExtractJsonBool(const std::string &json, const std::string &key) {
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

}  // namespace

extern "C" {

OmniInferHandle omniinfer_init(const char *config_json) {
  if (!config_json) return 0;
  const std::string config(config_json);
  const auto backend_name = ExtractJsonString(config, "backend").value_or("llama.cpp");
  const auto model_path = ExtractJsonString(config, "model_path");
  const int n_threads = ExtractJsonInt(config, "n_threads").value_or(0);
  const int n_ctx = ExtractJsonInt(config, "n_ctx").value_or(4096);

  if (!model_path.has_value()) { fprintf(stderr, "[OmniInfer] init: missing model_path\n"); return 0; }

  std::unique_ptr<omniinfer::InferenceBackend> backend;
#if defined(OMNIINFER_BACKEND_LLAMA_CPP)
  if (!backend && (backend_name == "llama.cpp" || backend_name.empty()))
    backend = std::make_unique<omniinfer::LlamaCppBackend>();
#endif
  if (!backend) { fprintf(stderr, "[OmniInfer] init: unknown backend: %s\n", backend_name.c_str()); return 0; }

  if (!backend->load(*model_path, config, "", n_threads, n_ctx)) {
    fprintf(stderr, "[OmniInfer] init: load failed\n"); return 0;
  }

  auto *session = new Session();
  session->handle = g_next_handle.fetch_add(1);
  session->backend = std::move(backend);
  { std::lock_guard<std::mutex> guard(g_sessions_mutex); g_sessions[session->handle] = session; }
  fprintf(stderr, "[OmniInfer] init: session %lld (%s)\n", (long long)session->handle, session->backend->name());
  return session->handle;
}

const char *omniinfer_generate(OmniInferHandle handle, const char *system_prompt,
                               const char *user_prompt, const char *request_json,
                               OmniInferTokenCallback on_token, OmniInferMetricsCallback on_metrics,
                               void *userdata) {
  Session *session = nullptr;
  { std::lock_guard<std::mutex> guard(g_sessions_mutex); auto it = g_sessions.find(handle); if (it == g_sessions.end()) return strdup(""); session = it->second; }
  session->cancelled.store(false);
  const std::string req(request_json ? request_json : "{}");
  const bool thinking = ExtractJsonBool(req, "thinking_enabled").value_or(session->thinking_enabled);
  const std::string sys(system_prompt ? system_prompt : "");
  const std::string user(user_prompt ? user_prompt : "");
  auto token_cb = [&](const std::string &token) -> bool {
    if (on_token) return on_token(token.c_str(), userdata);
    return !session->cancelled.load();
  };
  std::string result = session->backend->generate(sys, user, thinking, session->cancelled, token_cb);
  auto m = session->backend->get_metrics();
  if (m.generated_tokens > 0 && on_metrics) {
    double decode_tps = m.decode_us > 0 ? m.generated_tokens / (m.decode_us / 1e6) : 0;
    double prefill_tps = m.prefill_us > 0 ? m.prompt_tokens / (m.prefill_us / 1e6) : 0;
    std::ostringstream metrics;
    metrics << "prefill_tps=" << prefill_tps << ", decode_tps=" << decode_tps;
    on_metrics(metrics.str().c_str(), userdata);
  }
  return strdup(result.c_str());
}

void omniinfer_free_string(const char *str) { free(const_cast<char *>(str)); }

bool omniinfer_load_history(OmniInferHandle handle, const char **roles, const char **contents, int count) {
  std::lock_guard<std::mutex> guard(g_sessions_mutex);
  auto it = g_sessions.find(handle); if (it == g_sessions.end()) return false;
  std::vector<std::pair<std::string, std::string>> msgs;
  for (int i = 0; i < count; i++) msgs.emplace_back(roles[i] ? roles[i] : "", contents[i] ? contents[i] : "");
  return it->second->backend->load_history(msgs);
}

void omniinfer_set_think_mode(OmniInferHandle handle, bool enabled) {
  std::lock_guard<std::mutex> guard(g_sessions_mutex);
  auto it = g_sessions.find(handle); if (it != g_sessions.end()) it->second->thinking_enabled = enabled;
}

void omniinfer_reset(OmniInferHandle handle) {
  std::lock_guard<std::mutex> guard(g_sessions_mutex);
  auto it = g_sessions.find(handle); if (it != g_sessions.end()) it->second->backend->reset();
}

void omniinfer_cancel(OmniInferHandle handle) {
  std::lock_guard<std::mutex> guard(g_sessions_mutex);
  auto it = g_sessions.find(handle); if (it != g_sessions.end()) it->second->cancelled.store(true);
}

void omniinfer_free(OmniInferHandle handle) {
  Session *session = nullptr;
  { std::lock_guard<std::mutex> guard(g_sessions_mutex); auto it = g_sessions.find(handle); if (it == g_sessions.end()) return; session = it->second; g_sessions.erase(it); }
  delete session;
}

const char *omniinfer_collect_diagnostics_json(OmniInferHandle handle) {
  std::lock_guard<std::mutex> guard(g_sessions_mutex);
  auto it = g_sessions.find(handle); if (it == g_sessions.end()) return strdup("{}");
  auto m = it->second->backend->get_metrics();
  std::ostringstream json;
  json << "{\"backend\":\"" << it->second->backend->name() << "\",\"prompt_tokens\":" << m.prompt_tokens
       << ",\"generated_tokens\":" << m.generated_tokens << ",\"prefill_us\":" << m.prefill_us
       << ",\"decode_us\":" << m.decode_us << "}";
  return strdup(json.str().c_str());
}

}  // extern "C"
