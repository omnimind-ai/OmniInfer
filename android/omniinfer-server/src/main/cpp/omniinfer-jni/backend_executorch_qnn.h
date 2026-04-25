/*
 * ExecuTorch QNN backend for OmniInfer Android — subprocess architecture.
 *
 * Spawns libetqnn_runner.so as a standalone process to bypass Android linker
 * namespace and SELinux restrictions that prevent JNI-based QNN/FastRPC usage.
 * Communicates via stdin/stdout JSON protocol.
 *
 * Key insight: untrusted_app processes can use QNN NPU through Unsigned PD
 * (Protection Domain) if QNN libraries are loaded in a standalone executable
 * rather than a JNI-loaded shared library. The standalone process uses
 * /system/bin/linker64 directly, avoiding app linker namespace restrictions.
 */

#pragma once

#include "et_chat_template.h"
#include "inference_backend.h"

#include <android/log.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#define ET_LOG_TAG "OmniInferET"
#define ET_LOGI(...) __android_log_print(ANDROID_LOG_INFO, ET_LOG_TAG, __VA_ARGS__)
#define ET_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, ET_LOG_TAG, __VA_ARGS__)

namespace omniinfer {

namespace {

// Minimal JSON helpers
inline std::string et_json_string(const std::string& json, const std::string& key) {
  std::string search = "\"" + key + "\"";
  auto pos = json.find(search);
  if (pos == std::string::npos) return "";
  pos = json.find(':', pos + search.size());
  if (pos == std::string::npos) return "";
  auto q1 = json.find('"', pos + 1);
  if (q1 == std::string::npos) return "";
  std::string result;
  size_t i = q1 + 1;
  while (i < json.size()) {
    if (json[i] == '\\' && i + 1 < json.size()) {
      char c = json[i + 1];
      if (c == '/' || c == '"' || c == '\\') { result += c; i += 2; }
      else if (c == 'n') { result += '\n'; i += 2; }
      else if (c == 't') { result += '\t'; i += 2; }
      else { result += c; i += 2; }
    } else if (json[i] == '"') { break; }
    else { result += json[i]; i++; }
  }
  return result;
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

// Extract "type" field from a JSON line
inline std::string json_type(const std::string& line) {
  auto pos = line.find("\"type\"");
  if (pos == std::string::npos) return "";
  pos = line.find(':', pos + 6);
  if (pos == std::string::npos) return "";
  auto q1 = line.find('"', pos + 1);
  if (q1 == std::string::npos) return "";
  auto q2 = line.find('"', q1 + 1);
  if (q2 == std::string::npos) return "";
  return line.substr(q1 + 1, q2 - q1 - 1);
}

// Extract "text" field (handles escape sequences)
inline std::string json_text(const std::string& line) {
  auto pos = line.find("\"text\"");
  if (pos == std::string::npos) return "";
  pos = line.find(':', pos + 6);
  if (pos == std::string::npos) return "";
  auto q1 = line.find('"', pos + 1);
  if (q1 == std::string::npos) return "";
  std::string result;
  size_t i = q1 + 1;
  while (i < line.size()) {
    if (line[i] == '\\' && i + 1 < line.size()) {
      char c = line[i + 1];
      if (c == '"') result += '"';
      else if (c == '\\') result += '\\';
      else if (c == 'n') result += '\n';
      else if (c == 't') result += '\t';
      else { result += '\\'; result += c; }
      i += 2;
    } else if (line[i] == '"') { break; }
    else { result += line[i]; i++; }
  }
  return result;
}

// Extract int from JSON
inline int json_int(const std::string& json, const char* key, int def) {
  std::string needle = std::string("\"") + key + "\"";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return def;
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos) return def;
  pos++;
  while (pos < json.size() && json[pos] == ' ') pos++;
  try { return std::stoi(json.substr(pos)); } catch (...) { return def; }
}

inline int64_t json_int64(const std::string& json, const char* key, int64_t def) {
  std::string needle = std::string("\"") + key + "\"";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return def;
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos) return def;
  pos++;
  while (pos < json.size() && json[pos] == ' ') pos++;
  try { return std::stoll(json.substr(pos)); } catch (...) { return def; }
}

// Escape string for JSON
inline std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n";  break;
      case '\r': out += "\\r";  break;
      case '\t': out += "\\t";  break;
      default:   out += c;      break;
    }
  }
  return out;
}

} // anonymous namespace


class ExecuTorchQnnBackend : public InferenceBackend {
public:
  ~ExecuTorchQnnBackend() override { release(); }

  bool load(const std::string& model_path, const std::string& config_json,
            const std::string& native_lib_dir, int n_threads, int n_ctx) override {

    n_threads_ = n_threads;
    n_ctx_ = n_ctx > 0 ? n_ctx : 2048;

    // Parse config
    std::string tokenizer_path = et_json_string(config_json, "tokenizer_path");
    decoder_model_version_ = et_json_string(config_json, "decoder_model_version");
    std::string qnn_lib_dir = et_json_string(config_json, "qnn_lib_dir");

    if (decoder_model_version_.empty()) decoder_model_version_ = "qwen3";

    // Auto-discover tokenizer.json in model directory
    if (tokenizer_path.empty()) {
      auto slash = model_path.rfind('/');
      if (slash != std::string::npos)
        tokenizer_path = model_path.substr(0, slash + 1) + "tokenizer.json";
    }

    std::string lib_dir = qnn_lib_dir.empty() ? native_lib_dir : qnn_lib_dir;

    ET_LOGI("ExecuTorch QNN subprocess load: model=%s tokenizer=%s lib_dir=%s",
            model_path.c_str(), tokenizer_path.c_str(), lib_dir.c_str());

    // Find the runner executable. nativeLibraryDir varies by device
    // (lib/arm64 vs lib/arm64-v8a), so try multiple paths.
    std::string runner_path;
    const std::string candidates[] = {
      lib_dir + "/libetqnn_runner.so",
      lib_dir + "-v8a/libetqnn_runner.so",  // lib/arm64 → lib/arm64-v8a
    };
    for (const auto& path : candidates) {
      if (access(path.c_str(), X_OK) == 0) {
        runner_path = path;
        // Also update lib_dir to match the actual directory
        lib_dir = path.substr(0, path.rfind('/'));
        break;
      }
    }
    if (runner_path.empty()) {
      ET_LOGE("Runner not found at any candidate path:");
      for (const auto& path : candidates) ET_LOGE("  tried: %s", path.c_str());
      return false;
    }

    // Fork + exec the subprocess
    int stdin_pipe[2], stdout_pipe[2];
    if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0) {
      ET_LOGE("pipe() failed: %s", strerror(errno));
      return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
      ET_LOGE("fork() failed: %s", strerror(errno));
      return false;
    }

    if (pid == 0) {
      // Child process
      close(stdin_pipe[1]);   // close write end of stdin pipe
      close(stdout_pipe[0]);  // close read end of stdout pipe
      dup2(stdin_pipe[0], STDIN_FILENO);
      dup2(stdout_pipe[1], STDOUT_FILENO);
      dup2(stdout_pipe[1], STDERR_FILENO);
      close(stdin_pipe[0]);
      close(stdout_pipe[1]);

      // Set environment for QNN.
      // DSP_LIBRARY_PATH: where FastRPC looks for skel files (app dir first).
      // LD_LIBRARY_PATH: must include system/vendor paths so the subprocess
      // can find libcdsprpc.so (FastRPC) and other system libraries.
      setenv("DSP_LIBRARY_PATH", lib_dir.c_str(), 1);
      std::string adsp = lib_dir + ";/odm/lib/rfsa/adsp;/system/lib/rfsa/adsp";
      setenv("ADSP_LIBRARY_PATH", adsp.c_str(), 1);
      std::string ld = lib_dir + ":/system/lib64:/vendor/lib64:/vendor/lib64/egl";
      setenv("LD_LIBRARY_PATH", ld.c_str(), 1);

      // Exec the custom OmniInfer runner (JSON stdin/stdout protocol).
      execl(runner_path.c_str(), "libetqnn_runner.so",
            "--model_path", model_path.c_str(),
            "--tokenizer_path", tokenizer_path.c_str(),
            "--decoder_model_version", decoder_model_version_.c_str(),
            "--lib_dir", lib_dir.c_str(),
            "--n_ctx", std::to_string(n_ctx_).c_str(),
            nullptr);
      // If execl returns, it failed
      _exit(1);
    }

    // Parent process
    close(stdin_pipe[0]);   // close read end
    close(stdout_pipe[1]);  // close write end
    child_pid_ = pid;
    child_stdin_ = stdin_pipe[1];
    child_stdout_ = stdout_pipe[0];

    ET_LOGI("Subprocess started: pid=%d", pid);

    // Wait for {"type":"ready"} or {"type":"error"} from the runner.
    // The runner outputs these after loading the model.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    std::string line;
    while (std::chrono::steady_clock::now() < deadline) {
      line = read_line(500);
      if (line.empty()) {
        int status;
        pid_t r = waitpid(child_pid_, &status, WNOHANG);
        if (r == child_pid_) {
          ET_LOGE("Subprocess exited during load (status=%d)", status);
          child_pid_ = -1;
          release();
          return false;
        }
        continue;
      }
      ET_LOGI("Subprocess: %s", line.c_str());
      std::string type = json_type(line);
      if (type == "ready") {
        ET_LOGI("Subprocess ready, model loaded");
        return true;
      }
      if (type == "error") {
        ET_LOGE("Subprocess error: %s", line.c_str());
        release();
        return false;
      }
    }

    ET_LOGE("Subprocess timeout (no ready in 60s)");
    release();
    return false;
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
      const std::string& sampling_json = "") override {

    if (child_pid_ <= 0) return "";
    last_metrics_ = {};

    // Format prompt using chat template (runner expects a pre-formatted prompt)
    std::string prompt = et_template::format_prompt(
        decoder_model_version_, system_prompt, user_prompt,
        messages_json, thinking_enabled);

    // Build generate command JSON
    std::ostringstream cmd;
    cmd << "{\"command\":\"generate\"";
    cmd << ",\"prompt\":\"" << json_escape(prompt) << "\"";
    cmd << ",\"max_tokens\":" << (max_tokens > 0 ? max_tokens : 512);
    cmd << "}\n";

    write_cmd(cmd.str());

    // Read response lines.
    // The runner may echo the prompt (if compiled with echo=true) and emit
    // EOS special tokens. We filter both: skip tokens that are part of the
    // prompt prefix, and suppress known special tokens.
    std::string full_output;
    size_t prompt_echo_remaining = prompt.size();  // bytes of prompt left to skip
    bool prompt_echo_done = false;

    while (true) {
      if (cancelled.load(std::memory_order_relaxed)) {
        write_cmd("{\"command\":\"cancel\"}\n");
        break;
      }

      std::string line = read_line(100);
      if (line.empty()) {
        int status;
        pid_t r = waitpid(child_pid_, &status, WNOHANG);
        if (r == child_pid_) {
          ET_LOGE("Subprocess exited during generate");
          child_pid_ = -1;
          break;
        }
        continue;
      }

      std::string type = json_type(line);
      if (type == "token") {
        std::string text = json_text(line);

        // Skip prompt echo: runner with echo=true replays the full prompt
        if (!prompt_echo_done) {
          if (text.size() <= prompt_echo_remaining) {
            prompt_echo_remaining -= text.size();
            if (prompt_echo_remaining == 0) prompt_echo_done = true;
            continue;
          } else {
            text = text.substr(prompt_echo_remaining);
            prompt_echo_done = true;
          }
        }

        // Filter EOS / special tokens
        if (text == "<|im_end|>" || text == "<|endoftext|>" ||
            text == "<end_of_turn>" || text == "<|eot_id|>" ||
            text == "</s>") {
          continue;
        }

        full_output += text;
        if (on_token) {
          bool cont = on_token(text);
          if (!cont) {
            if (graceful_stop.load(std::memory_order_relaxed))
              write_cmd("{\"command\":\"cancel\"}\n");
            break;
          }
        }
      } else if (type == "metrics") {
        last_metrics_.prompt_tokens = json_int(line, "prompt_tokens", 0);
        last_metrics_.generated_tokens = json_int(line, "generated_tokens", 0);
        last_metrics_.prefill_us = json_int64(line, "prefill_us", 0);
        last_metrics_.decode_us = json_int64(line, "decode_us", 0);
      } else if (type == "done") {
        break;
      } else if (type == "error") {
        ET_LOGE("Subprocess error: %s", line.c_str());
        break;
      }
      // Log other lines (e.g., ET warnings)
      else if (!line.empty()) {
        ET_LOGI("Subprocess: %s", line.c_str());
      }
    }

    return full_output;
  }

  bool load_history(
      const std::vector<std::pair<std::string, std::string>>& /*messages*/) override {
    return true;
  }

  void reset() override {
    if (child_pid_ > 0) write_cmd("{\"command\":\"reset\"}\n");
  }

  InferenceMetrics get_metrics() override { return last_metrics_; }
  int n_threads() const override { return n_threads_; }
  const char* name() const override { return "executorch-qnn"; }

private:
  void release() {
    if (child_pid_ > 0) {
      write_cmd("{\"command\":\"quit\"}\n");
      // Give it a moment to exit gracefully
      int status;
      for (int i = 0; i < 10; i++) {
        pid_t r = waitpid(child_pid_, &status, WNOHANG);
        if (r == child_pid_) { child_pid_ = -1; break; }
        usleep(100000); // 100ms
      }
      if (child_pid_ > 0) {
        kill(child_pid_, SIGKILL);
        waitpid(child_pid_, &status, 0);
        child_pid_ = -1;
      }
    }
    if (child_stdin_ >= 0) { close(child_stdin_); child_stdin_ = -1; }
    if (child_stdout_ >= 0) { close(child_stdout_); child_stdout_ = -1; }
  }

  void write_cmd(const std::string& cmd) {
    if (child_stdin_ < 0) return;
    const char* p = cmd.data();
    size_t remaining = cmd.size();
    while (remaining > 0) {
      ssize_t n = write(child_stdin_, p, remaining);
      if (n <= 0) break;
      p += n;
      remaining -= n;
    }
  }

  // Read one line from child stdout. Returns empty if timeout_ms expires.
  std::string read_line(int timeout_ms) {
    if (child_stdout_ < 0) return "";

    // Use select for timeout
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(child_stdout_, &fds);
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    // First drain the read buffer
    while (!read_buf_.empty()) {
      auto nl = read_buf_.find('\n');
      if (nl != std::string::npos) {
        std::string line = read_buf_.substr(0, nl);
        read_buf_.erase(0, nl + 1);
        return line;
      }
      break;
    }

    int ret = select(child_stdout_ + 1, &fds, nullptr, nullptr, &tv);
    if (ret <= 0) return "";

    char buf[4096];
    ssize_t n = read(child_stdout_, buf, sizeof(buf));
    if (n <= 0) return "";
    read_buf_.append(buf, n);

    auto nl = read_buf_.find('\n');
    if (nl != std::string::npos) {
      std::string line = read_buf_.substr(0, nl);
      read_buf_.erase(0, nl + 1);
      return line;
    }
    return "";
  }

  pid_t child_pid_ = -1;
  int child_stdin_ = -1;
  int child_stdout_ = -1;
  std::string read_buf_;

  std::string decoder_model_version_;
  int n_threads_ = 0;
  int n_ctx_ = 2048;
  InferenceMetrics last_metrics_;
};

} // namespace omniinfer
