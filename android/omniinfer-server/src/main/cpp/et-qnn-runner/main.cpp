/*
 * ExecuTorch QNN subprocess runner.
 * Runs as standalone process, communicates via stdin/stdout JSON protocol.
 * Launched by OmniInfer's backend_executorch_qnn.h via fork+exec.
 *
 * Usage:
 *   libetqnn_runner.so --model_path <.pte> --tokenizer_path <tokenizer.json>
 *                      --lib_dir <dir with QNN .so> [--decoder_model_version qwen3]
 *                      [--n_ctx 2048] [--temperature 0.8]
 *
 * Protocol (stdin → stdout):
 *   Stdin:  one JSON per line
 *     {"command":"generate","messages":"[...]","max_tokens":100,"temperature":0.8,"thinking_enabled":false}
 *     {"command":"cancel"}
 *     {"command":"quit"}
 *   Stdout: one JSON per line
 *     {"type":"ready"}
 *     {"type":"token","text":"..."}
 *     {"type":"metrics","prompt_tokens":N,"generated_tokens":N,"prefill_us":N,"decode_us":N}
 *     {"type":"done"}
 *     {"type":"error","message":"..."}
 */

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include <executorch/extension/module/module.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/runtime/platform/platform.h>

// ET Runner
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>

// Chat template
#include "../omniinfer-jni/et_chat_template.h"

// JSON escape helper
static std::string json_escape(const std::string& s) {
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

// Minimal JSON string value extractor
static std::string json_str(const std::string& json, const char* key) {
  std::string needle = std::string("\"") + key + "\"";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return "";
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos) return "";
  pos++;
  while (pos < json.size() && json[pos] == ' ') pos++;
  if (pos >= json.size() || json[pos] != '"') return "";
  pos++; // skip opening quote
  std::string result;
  while (pos < json.size()) {
    if (json[pos] == '\\' && pos + 1 < json.size()) {
      char c = json[pos + 1];
      if (c == '"') result += '"';
      else if (c == '\\') result += '\\';
      else if (c == 'n') result += '\n';
      else if (c == 't') result += '\t';
      else { result += '\\'; result += c; }
      pos += 2;
    } else if (json[pos] == '"') {
      break;
    } else {
      result += json[pos];
      pos++;
    }
  }
  return result;
}

// Minimal JSON number extractor
static double json_num(const std::string& json, const char* key, double def) {
  std::string needle = std::string("\"") + key + "\"";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return def;
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos) return def;
  pos++;
  while (pos < json.size() && json[pos] == ' ') pos++;
  try { return std::stod(json.substr(pos, 20)); }
  catch (...) { return def; }
}

static bool json_bool(const std::string& json, const char* key, bool def) {
  std::string needle = std::string("\"") + key + "\"";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return def;
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos) return def;
  pos++;
  while (pos < json.size() && json[pos] == ' ') pos++;
  if (json.substr(pos, 4) == "true") return true;
  if (json.substr(pos, 5) == "false") return false;
  return def;
}

// Extract raw JSON value (for messages array)
static std::string json_raw(const std::string& json, const char* key) {
  std::string needle = std::string("\"") + key + "\"";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return "";
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos) return "";
  pos++;
  while (pos < json.size() && json[pos] == ' ') pos++;
  if (pos >= json.size()) return "";

  // Detect type and extract raw value
  char first = json[pos];
  if (first == '[' || first == '{') {
    int depth = 1;
    size_t start = pos;
    pos++;
    char open = first, close = (first == '[') ? ']' : '}';
    bool in_str = false;
    while (pos < json.size() && depth > 0) {
      if (json[pos] == '\\' && in_str) { pos += 2; continue; }
      if (json[pos] == '"') in_str = !in_str;
      if (!in_str) {
        if (json[pos] == open) depth++;
        else if (json[pos] == close) depth--;
      }
      pos++;
    }
    return json.substr(start, pos - start);
  }
  return "";
}

static void emit(const std::string& json) {
  std::cout << json << "\n";
  std::cout.flush();
}

int main(int argc, char** argv) {
  // Parse CLI args
  std::string model_path, tokenizer_path, lib_dir, decoder_version = "qwen3";
  int n_ctx = 2048;
  float temperature = 0.8f;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--model_path" && i + 1 < argc) model_path = argv[++i];
    else if (arg == "--tokenizer_path" && i + 1 < argc) tokenizer_path = argv[++i];
    else if (arg == "--lib_dir" && i + 1 < argc) lib_dir = argv[++i];
    else if (arg == "--decoder_model_version" && i + 1 < argc) decoder_version = argv[++i];
    else if (arg == "--n_ctx" && i + 1 < argc) n_ctx = std::atoi(argv[++i]);
    else if (arg == "--temperature" && i + 1 < argc) temperature = std::atof(argv[++i]);
  }

  if (model_path.empty() || tokenizer_path.empty()) {
    emit(R"({"type":"error","message":"--model_path and --tokenizer_path required"})");
    return 1;
  }

  // Set QNN library paths
  if (!lib_dir.empty()) {
    setenv("DSP_LIBRARY_PATH", lib_dir.c_str(), 1);
    std::string adsp = lib_dir + ";/odm/lib/rfsa/adsp;/system/lib/rfsa/adsp";
    setenv("ADSP_LIBRARY_PATH", adsp.c_str(), 1);
    std::string ld = lib_dir;
    const char* old_ld = getenv("LD_LIBRARY_PATH");
    if (old_ld) { ld += ":"; ld += old_ld; }
    setenv("LD_LIBRARY_PATH", ld.c_str(), 1);
  }

  // Initialize ET platform (logging)
  et_pal_init();

  // Create Module
  std::unique_ptr<executorch::extension::Module> module;
  try {
    module = std::make_unique<executorch::extension::Module>(
        model_path,
        executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
  } catch (const std::exception& e) {
    emit(std::string(R"({"type":"error","message":"Module create failed: )") +
         json_escape(e.what()) + "\"}");
    return 1;
  }

  // Detect KV bit width from model metadata
  int kv_bit_width = 16;
  auto bw_result = module->execute("get_kv_io_bit_width");
  if (bw_result.ok()) {
    auto& values = bw_result.get();
    if (!values.empty()) {
      auto val = values[0].toScalar().to<int64_t>();
      if (val == 8 || val == 16) kv_bit_width = static_cast<int>(val);
    }
  }

  // Create Runner (template on KV bit width)
  std::unique_ptr<executorch::extension::llm::IRunner> runner;
  int eval_mode = 1; // hybrid

  if (kv_bit_width == 8) {
    runner = std::make_unique<example::Runner<uint8_t>>(
        std::move(module), decoder_version, model_path, tokenizer_path,
        "", "", temperature, eval_mode, false);
  } else {
    runner = std::make_unique<example::Runner<uint16_t>>(
        std::move(module), decoder_version, model_path, tokenizer_path,
        "", "", temperature, eval_mode, false);
  }

  // Load
  auto err = runner->load();
  if (err != executorch::runtime::Error::Ok) {
    emit(std::string(R"({"type":"error","message":"Runner::load() failed: )") +
         std::to_string(static_cast<int>(err)) + "\"}");
    return 1;
  }

  emit(R"({"type":"ready"})");

  // Command loop
  std::atomic<bool> cancel_flag{false};
  std::string line;

  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;

    std::string cmd = json_str(line, "command");

    if (cmd == "quit") {
      break;
    } else if (cmd == "cancel") {
      cancel_flag.store(true);
    } else if (cmd == "reset") {
      runner->reset();
      emit(R"({"type":"done"})");
    } else if (cmd == "generate") {
      cancel_flag.store(false);

      // Extract params
      std::string messages = json_raw(line, "messages");
      std::string system_prompt = json_str(line, "system_prompt");
      std::string user_prompt = json_str(line, "user_prompt");
      int max_tokens = static_cast<int>(json_num(line, "max_tokens", 512));
      float temp = static_cast<float>(json_num(line, "temperature", temperature));
      bool thinking = json_bool(line, "thinking_enabled", false);

      // Build prompt using chat template
      std::string prompt = omniinfer::et_template::format_prompt(
          decoder_version, system_prompt, user_prompt, messages, thinking);

      // Configure generation
      executorch::extension::llm::GenerationConfig config;
      config.seq_len = max_tokens + 512; // prompt + generation headroom
      config.echo = false;

      // Stats
      const executorch::llm::Stats* stats_ptr = nullptr;
      int prompt_tokens = 0;
      int gen_tokens = 0;

      auto t_start = std::chrono::steady_clock::now();
      auto t_first_token = t_start;
      bool first_token = true;

      // Generate with streaming
      auto gen_err = runner->generate(
          prompt,
          config,
          // Token callback
          [&](const std::string& token) {
            if (cancel_flag.load()) return;
            if (first_token) {
              t_first_token = std::chrono::steady_clock::now();
              first_token = false;
            }
            gen_tokens++;
            emit(std::string(R"({"type":"token","text":")") +
                 json_escape(token) + "\"}");
          },
          // Stats callback
          [&](const executorch::llm::Stats& s) {
            stats_ptr = &s;
            prompt_tokens = s.num_prompt_tokens;
          });

      auto t_end = std::chrono::steady_clock::now();

      if (gen_err != executorch::runtime::Error::Ok && !cancel_flag.load()) {
        emit(std::string(R"({"type":"error","message":"generate failed: )") +
             std::to_string(static_cast<int>(gen_err)) + "\"}");
      }

      // Compute timing
      auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(
          t_end - t_start).count();
      auto ttft_us = std::chrono::duration_cast<std::chrono::microseconds>(
          t_first_token - t_start).count();
      int64_t prefill_us = ttft_us;
      int64_t decode_us = total_us - prefill_us;

      // Emit metrics
      std::string metrics = R"({"type":"metrics")";
      metrics += ",\"prompt_tokens\":" + std::to_string(prompt_tokens);
      metrics += ",\"generated_tokens\":" + std::to_string(gen_tokens);
      metrics += ",\"prefill_us\":" + std::to_string(prefill_us);
      metrics += ",\"decode_us\":" + std::to_string(decode_us);
      metrics += "}";
      emit(metrics);
      emit(R"({"type":"done"})");
    }
  }

  return 0;
}
