/*
 * ExecuTorch QNN backend for OmniInfer Android.
 * Wraps ET's qualcomm Runner<T> behind InferenceBackend interface.
 *
 * Phase 1: text-only + thinking, no tool calling, no multimodal.
 */

#pragma once

#include "et_chat_template.h"
#include "inference_backend.h"
#include "thinking_tags.h"

#include <android/log.h>
#include <dlfcn.h>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/platform/platform.h>

#define ET_LOG_TAG "OmniInferET"
#define ET_LOGI(...) __android_log_print(ANDROID_LOG_INFO, ET_LOG_TAG, __VA_ARGS__)
#define ET_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, ET_LOG_TAG, __VA_ARGS__)

namespace omniinfer {

namespace {

// Extract a string value from minimal JSON object by key.
// Handles JSON-escaped forward slashes (\/) which org.json produces.
inline std::string et_json_string(const std::string& json, const std::string& key) {
  std::string search = "\"" + key + "\"";
  auto pos = json.find(search);
  if (pos == std::string::npos) return "";
  pos = json.find(':', pos + search.size());
  if (pos == std::string::npos) return "";
  auto q1 = json.find('"', pos + 1);
  if (q1 == std::string::npos) return "";
  // Find closing quote, handling escape sequences
  std::string result;
  size_t i = q1 + 1;
  while (i < json.size()) {
    if (json[i] == '\\' && i + 1 < json.size()) {
      char c = json[i + 1];
      if (c == '/' || c == '"' || c == '\\') { result += c; i += 2; }
      else if (c == 'n') { result += '\n'; i += 2; }
      else if (c == 't') { result += '\t'; i += 2; }
      else { result += c; i += 2; }
    } else if (json[i] == '"') {
      break;
    } else {
      result += json[i]; i++;
    }
  }
  return result;
}

inline int et_json_int(const std::string& json, const std::string& key, int def) {
  std::string search = "\"" + key + "\"";
  auto pos = json.find(search);
  if (pos == std::string::npos) return def;
  pos = json.find(':', pos + search.size());
  if (pos == std::string::npos) return def;
  pos++;
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
  try { return std::stoi(json.substr(pos)); } catch (...) { return def; }
}

inline bool et_json_bool(const std::string& json, const std::string& key, bool def) {
  std::string search = "\"" + key + "\"";
  auto pos = json.find(search);
  if (pos == std::string::npos) return def;
  pos = json.find(':', pos + search.size());
  if (pos == std::string::npos) return def;
  auto rest = json.substr(pos + 1);
  if (rest.find("true") < rest.find("false")) return true;
  if (rest.find("false") < rest.find("true")) return false;
  return def;
}

// Check if a byte sequence forms valid UTF-8.
inline bool is_valid_utf8(const std::string& s) {
  const auto* p = reinterpret_cast<const unsigned char*>(s.data());
  const auto* end = p + s.size();
  while (p < end) {
    if (*p < 0x80) { p++; }
    else if ((*p & 0xE0) == 0xC0) { if (end - p < 2 || (p[1] & 0xC0) != 0x80) return false; p += 2; }
    else if ((*p & 0xF0) == 0xE0) { if (end - p < 3 || (p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80) return false; p += 3; }
    else if ((*p & 0xF8) == 0xF0) { if (end - p < 4 || (p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80 || (p[3] & 0xC0) != 0x80) return false; p += 4; }
    else return false;
  }
  return true;
}

} // anonymous namespace


class ExecuTorchQnnBackend : public InferenceBackend {
public:
  ~ExecuTorchQnnBackend() override = default;

  bool load(const std::string& model_path, const std::string& config_json,
            const std::string& native_lib_dir, int n_threads, int n_ctx) override {

    // Initialize ET platform abstraction (enables logging to logcat)
    et_pal_init();

    n_threads_ = n_threads;
    n_ctx_ = n_ctx > 0 ? n_ctx : 2048;

    // Parse config
    std::string tokenizer_path = et_json_string(config_json, "tokenizer_path");
    decoder_model_version_ = et_json_string(config_json, "decoder_model_version");
    std::string qnn_lib_dir = et_json_string(config_json, "qnn_lib_dir");
    float temperature = 0.8f;
    int eval_mode = 1; // hybrid
    bool shared_buffer = et_json_bool(config_json, "shared_buffer", false);

    if (decoder_model_version_.empty()) {
      decoder_model_version_ = "qwen3"; // default for Phase 1
      ET_LOGI("No decoder_model_version specified, defaulting to qwen3");
    }

    // Auto-discover tokenizer.json in model directory
    if (tokenizer_path.empty()) {
      auto slash = model_path.rfind('/');
      if (slash != std::string::npos) {
        tokenizer_path = model_path.substr(0, slash + 1) + "tokenizer.json";
      }
    }

    ET_LOGI("ExecuTorch QNN load: model=%s tokenizer=%s version=%s qnn_lib=%s",
            model_path.c_str(), tokenizer_path.c_str(),
            decoder_model_version_.c_str(), qnn_lib_dir.c_str());

    // QNN runtime .so files (libQnnHtp, skel, stub, etc.) are bundled in
    // jniLibs and extracted to nativeLibraryDir (requires extractNativeLibs=true).
    // IMPORTANT: Do NOT bundle libcdsprpc.so or other system FastRPC deps in
    // jniLibs — they must come from the system (/vendor/lib64/) or the app's
    // <uses-native-library> declaration. Bundling them overrides the system
    // version and breaks FastRPC session establishment (error 4000).
    std::string lib_dir = native_lib_dir;
    if (!qnn_lib_dir.empty()) lib_dir = qnn_lib_dir;

    if (!lib_dir.empty()) {
      // ADSP_LIBRARY_PATH must include the skel's location.
      // Prepend our dir, then add all standard system skel paths as fallback.
      // If the bundled skel hits SELinux restrictions, the system skel at these
      // vendor-labeled paths may work (provided QNN SDK version is compatible).
      // FastRPC Unsigned PD mode: the app process itself opens skel files
      // via apps_std_fopen_fd and passes them to the DSP. Both DSP_LIBRARY_PATH
      // and ADSP_LIBRARY_PATH must be set — DSP_LIBRARY_PATH is used by the
      // FastRPC host-side library for the fopen path, ADSP_LIBRARY_PATH is the
      // legacy name checked by some QNN SDK versions.
      setenv("DSP_LIBRARY_PATH", lib_dir.c_str(), 1);
      ET_LOGI("Set DSP_LIBRARY_PATH=%s", lib_dir.c_str());

      std::string adsp = lib_dir
          + ";/dsp"
          + ";/vendor/dsp"
          + ";/vendor/lib/rfsa/adsp"
          + ";/odm/lib/rfsa/adsp"
          + ";/system/lib/rfsa/adsp"
          + ";/system/vendor/lib/rfsa/adsp";
      setenv("ADSP_LIBRARY_PATH", adsp.c_str(), 1);
      ET_LOGI("Set ADSP_LIBRARY_PATH=%s", adsp.c_str());

      // Diagnostic: check skel accessibility from each ADSP path.
      // This helps distinguish SELinux issues from version mismatches.
      const char* skel_names[] = {"libQnnHtpV79Skel.so", "libQnnHtpV75Skel.so",
                                   "libQnnHtpV73Skel.so", nullptr};
      const char* adsp_dirs[] = {lib_dir.c_str(), "/dsp", "/vendor/dsp",
                                  "/vendor/lib/rfsa/adsp", "/odm/lib/rfsa/adsp", nullptr};
      for (int d = 0; adsp_dirs[d]; d++) {
        for (int s = 0; skel_names[s]; s++) {
          std::string path = std::string(adsp_dirs[d]) + "/" + skel_names[s];
          FILE* f = fopen(path.c_str(), "r");
          if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fclose(f);
            ET_LOGI("Skel found: %s (%ld bytes, readable)", path.c_str(), sz);
          }
        }
      }

      // Pre-load QNN runtime libraries so ET delegate can find them.
      // Use RTLD_GLOBAL so symbols are visible to subsequent dlopen calls
      // from within the ET delegate (which does dlopen("libQnnHtp.so") by bare name).
      const char* qnn_libs[] = {
        "libQnnSystem.so", "libQnnHtp.so", "libQnnHtpPrepare.so",
        "libqnn_executorch_backend.so",
        nullptr
      };
      for (int i = 0; qnn_libs[i]; i++) {
        // Try bare name first (uses linker search paths), then full path
        void* h = dlopen(qnn_libs[i], RTLD_NOW | RTLD_GLOBAL);
        if (!h) {
          std::string lib_path = lib_dir + "/" + qnn_libs[i];
          h = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        }
        if (!h) {
          ET_LOGE("dlopen %s failed: %s", qnn_libs[i], dlerror());
        } else {
          ET_LOGI("Loaded %s", qnn_libs[i]);
        }
      }
    }

    // Create ET Module
    try {
      module_ = std::make_unique<executorch::extension::Module>(
          model_path,
          executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
    } catch (const std::exception& e) {
      ET_LOGE("Failed to create Module: %s", e.what());
      return false;
    }

    // Load the module program before querying metadata
    ET_LOGI("Loading module program...");
    auto load_err = module_->load();
    ET_LOGI("Module::load() returned %d", static_cast<int>(load_err));
    if (load_err != executorch::runtime::Error::Ok) {
      ET_LOGE("Module::load() failed with error %d", static_cast<int>(load_err));
      return false;
    }

    // List available methods
    auto method_names = module_->method_names();
    if (method_names.ok()) {
      for (const auto& name : *method_names) {
        ET_LOGI("Module method: %s", name.c_str());
      }
    }

    // Detect KV bit width from model metadata
    int kv_bit_width = 8;
    try {
      auto bw_result = module_->get("get_kv_io_bit_width");
      if (bw_result.ok()) {
        kv_bit_width = bw_result->toScalar().to<int64_t>();
        ET_LOGI("KV bit width from model: %d", kv_bit_width);
      }
    } catch (...) {
      ET_LOGI("Could not read kv_io_bit_width, defaulting to 8");
    }

    ET_LOGI("KV bit width: %d, eval_mode: %d, shared_buffer: %d",
            kv_bit_width, eval_mode, shared_buffer);

    // Create Runner (type-erased via IRunner)
    if (kv_bit_width == 16) {
      auto runner = std::make_unique<example::Runner<uint16_t>>(
          std::move(module_), decoder_model_version_, model_path,
          tokenizer_path, "", "", temperature, eval_mode, shared_buffer);
      runner_iface_ = std::move(runner);
    } else {
      auto runner = std::make_unique<example::Runner<uint8_t>>(
          std::move(module_), decoder_model_version_, model_path,
          tokenizer_path, "", "", temperature, eval_mode, shared_buffer);
      runner_iface_ = std::move(runner);
    }

    // Load model methods
    auto err = runner_iface_->load();
    if (err != executorch::runtime::Error::Ok) {
      ET_LOGE("Runner::load() failed with error %d", static_cast<int>(err));
      runner_iface_.reset();
      return false;
    }

    ET_LOGI("ExecuTorch QNN model loaded successfully");
    return true;
  }

  std::string generate(
      const std::string& system_prompt,
      const std::string& user_prompt,
      bool thinking_enabled,
      std::atomic<bool>& cancelled,
      std::function<bool(const std::string& token)> on_token,
      const std::string& tools_json,
      const std::string& tool_choice,
      const std::string& messages_json,
      const std::vector<std::vector<uint8_t>>& images,
      int max_tokens,
      std::atomic<bool>& graceful_stop,
      const std::string& /*sampling_json*/ = "") override {

    if (!runner_iface_) return "";

    // Reset metrics
    last_metrics_ = {};

    // Build formatted prompt
    std::string prompt = et_template::format_prompt(
        decoder_model_version_, system_prompt, user_prompt,
        messages_json, thinking_enabled);

    ET_LOGI("generate: prompt_len=%zu thinking=%d max_tokens=%d",
            prompt.size(), thinking_enabled, max_tokens);

    // Configure generation
    executorch::extension::llm::GenerationConfig config;
    config.echo = false;
    config.seq_len = n_ctx_;
    config.temperature = 0.8f;
    if (max_tokens > 0) {
      config.max_new_tokens = max_tokens;
    }

    // State for streaming
    std::string full_output;
    std::string utf8_buf;
    int token_count = 0;
    bool stopped = false;
    bool in_thinking = false;
    int reasoning_token_count = 0;

    // Thinking tag normalizer (for non-standard tags like Gemma's)
    // Only created if the model uses non-standard thinking tags.
    const thinking_tags::NativeTagPair* non_std_tags = nullptr;
    std::unique_ptr<thinking_tags::Normalizer> tag_normalizer;

    // Detect if model uses non-standard thinking tags
    if (thinking_enabled) {
      // Check known non-standard models (e.g., Gemma uses <|channel>thought)
      if (decoder_model_version_.find("gemma") != std::string::npos) {
        non_std_tags = thinking_tags::detect_in_prompt(prompt);
        if (non_std_tags) {
          tag_normalizer = std::make_unique<thinking_tags::Normalizer>(*non_std_tags);
        }
      }
    }

    auto t_start = std::chrono::steady_clock::now();
    auto t_first_token = t_start;
    bool first_token = true;

    // Token callback adapter: ET void(string) -> OmniInfer bool(string)
    auto token_callback = [&](const std::string& piece) {
      if (stopped || cancelled.load(std::memory_order_relaxed)) {
        stopped = true;
        // Can't truly stop ET runner (stop() is no-op), but stop emitting
        return;
      }

      if (first_token) {
        t_first_token = std::chrono::steady_clock::now();
        first_token = false;
      }

      token_count++;
      full_output += piece;

      // Normalize thinking tags (e.g., Gemma's <|channel>thought -> <think>)
      std::string normalized = tag_normalizer ? tag_normalizer->process(piece) : piece;

      // Track thinking tokens
      if (thinking_enabled) {
        if (normalized.find("<think>") != std::string::npos) {
          in_thinking = true;
        }
        if (in_thinking) {
          reasoning_token_count++;
        }
        if (normalized.find("</think>") != std::string::npos) {
          in_thinking = false;
        }
      }

      // UTF-8 buffering
      utf8_buf += normalized;
      if (!utf8_buf.empty() && is_valid_utf8(utf8_buf)) {
        if (on_token && !stopped) {
          bool cont = on_token(utf8_buf);
          if (!cont) {
            stopped = true;
            if (graceful_stop.load(std::memory_order_relaxed)) {
              // Graceful stop: preserve any state
            }
          }
        }
        utf8_buf.clear();
      }
    };

    // Stats callback — capture pointer since Stats has deleted copy assignment
    const executorch::llm::Stats* captured_stats_ptr = nullptr;
    auto stats_callback = [&](const executorch::llm::Stats& stats) {
      captured_stats_ptr = &stats;
    };

    // Run generation
    auto err = runner_iface_->generate(prompt, config, token_callback, stats_callback);

    auto t_end = std::chrono::steady_clock::now();

    // Flush remaining UTF-8 buffer
    if (!utf8_buf.empty() && on_token && !stopped) {
      on_token(utf8_buf);
      utf8_buf.clear();
    }

    if (err != executorch::runtime::Error::Ok) {
      ET_LOGE("Runner::generate() failed with error %d", static_cast<int>(err));
    }

    // Collect metrics
    last_metrics_.generated_tokens = token_count;
    last_metrics_.reasoning_tokens = reasoning_token_count;

    if (captured_stats_ptr && captured_stats_ptr->num_prompt_tokens > 0) {
      last_metrics_.prompt_tokens = captured_stats_ptr->num_prompt_tokens;
      // Convert ms to us
      int64_t prefill_ms = captured_stats_ptr->prompt_eval_end_ms - captured_stats_ptr->inference_start_ms;
      int64_t decode_ms = captured_stats_ptr->inference_end_ms - captured_stats_ptr->prompt_eval_end_ms;
      last_metrics_.prefill_us = prefill_ms * 1000;
      last_metrics_.decode_us = decode_ms * 1000;
      last_metrics_.generated_tokens = captured_stats_ptr->num_generated_tokens;
    } else {
      // Fallback: compute from wall clock
      auto prefill_dur = std::chrono::duration_cast<std::chrono::microseconds>(
          t_first_token - t_start);
      auto decode_dur = std::chrono::duration_cast<std::chrono::microseconds>(
          t_end - t_first_token);
      last_metrics_.prefill_us = prefill_dur.count();
      last_metrics_.decode_us = decode_dur.count();
    }

    ET_LOGI("generate done: prompt=%d generated=%d prefill=%.1fms decode=%.1fms",
            last_metrics_.prompt_tokens, last_metrics_.generated_tokens,
            last_metrics_.prefill_us / 1000.0, last_metrics_.decode_us / 1000.0);

    return full_output;
  }

  bool load_history(
      const std::vector<std::pair<std::string, std::string>>& /*messages*/) override {
    // ET runner manages KV cache internally. No-op since we pass full
    // conversation history via messages_json each request.
    return true;
  }

  void reset() override {
    if (runner_iface_) {
      runner_iface_->reset();  // Currently no-op in ET runner
    }
  }

  InferenceMetrics get_metrics() override {
    return last_metrics_;
  }

  int n_threads() const override {
    return n_threads_;
  }

  const char* name() const override {
    return "executorch-qnn";
  }

private:
  std::unique_ptr<executorch::extension::Module> module_;
  std::unique_ptr<executorch::extension::llm::IRunner> runner_iface_;
  std::string decoder_model_version_;
  int n_threads_ = 0;
  int n_ctx_ = 2048;
  InferenceMetrics last_metrics_;
};

} // namespace omniinfer
