#pragma once

#include "inference_backend.h"

#include <llm/llm.hpp>

#include <android/log.h>
#include <cstdio>

namespace omniinfer {

class MnnBackend : public InferenceBackend {
public:
  ~MnnBackend() override { release(); }

  bool load(const std::string& model_path, const std::string& config_json,
            const std::string& /*native_lib_dir*/, int n_threads, int n_ctx) override {
    llm_ = MNN::Transformer::Llm::createLLM(model_path);
    if (!llm_) return false;
    // Extract app cache dir for temp file writing (writable by the app process).
    cache_dir_ = extract_string(config_json, "cache_dir");
    if (cache_dir_.empty()) {
      auto slash = model_path.find_last_of("/\\");
      cache_dir_ = (slash != std::string::npos) ? model_path.substr(0, slash) : "/tmp";
    }

    std::ostringstream cfg;
    cfg << "{";
    bool first = true;
    if (n_threads > 0) { cfg << "\"thread_num\":" << n_threads; first = false; }
    if (n_ctx > 0) { if (!first) cfg << ","; cfg << "\"max_new_tokens\":" << n_ctx; }
    cfg << "}";
    llm_->set_config(cfg.str());

    if (!config_json.empty()) llm_->set_config(config_json);

    if (!llm_->load()) {
      MNN::Transformer::Llm::destroy(llm_); llm_ = nullptr;
      return false;
    }
    return true;
  }

  std::string generate(
      const std::string& system_prompt,
      const std::string& user_prompt,
      bool thinking_enabled,
      std::atomic<bool>& cancelled,
      std::function<bool(const std::string& token)> on_token,
      const std::string& /*tools_json*/ = "",
      const std::string& /*tool_choice*/ = "",
      const std::string& messages_json = "",
      const uint8_t* image_data = nullptr,
      size_t image_size = 0) override {

    using ChatMessages = MNN::Transformer::ChatMessages;
    ChatMessages msgs;

    // Parse messages from JSON array or use legacy single-turn path.
    if (!messages_json.empty()) {
      msgs = parse_chat_messages(messages_json);
    } else {
      if (!system_prompt.empty()) msgs.push_back({"system", system_prompt});
      if (!user_prompt.empty()) msgs.push_back({"user", user_prompt});
    }

    // Handle multimodal: save image to temp file in model directory (writable).
    std::string tmp_image_path;
    if (image_data && image_size > 0) {
      tmp_image_path = cache_dir_ + "/.omniinfer_vlm_tmp";
      FILE* f = fopen(tmp_image_path.c_str(), "wb");
      if (f) { fwrite(image_data, 1, image_size, f); fclose(f); }
    }

    // Stateless: reset all state at start of each request.
    llm_->reset();

    // Thinking control via jinja context variable (must be set before apply_chat_template).
    if (thinking_enabled) {
      llm_->set_config(R"({"jinja":{"context":{"enable_thinking":true}}})");
    } else {
      llm_->set_config(R"({"jinja":{"context":{"enable_thinking":false}}})");
    }

    // Apply chat template and tokenize.
    std::string formatted = llm_->apply_chat_template(msgs);

    // For multimodal: insert <img> tag into the formatted prompt after the last
    // user turn marker, so tokenizer_encode(MultimodalPrompt) can find and
    // process it. The chat template outputs plain text; <img> tags must be in
    // the final prompt_template for MNN's regex-based multimodal tokenizer.
    std::vector<int> input_ids;
    if (!tmp_image_path.empty()) {
      // Find last user turn in formatted prompt (e.g. "<|im_start|>user\n")
      // and insert <img> tag right after the marker.
      std::string user_marker = "user\n";
      auto pos = formatted.rfind(user_marker);
      if (pos != std::string::npos) {
        std::string img_tag = "<img>" + tmp_image_path + "</img>\n";
        formatted.insert(pos + user_marker.size(), img_tag);
      }
      MNN::Transformer::MultimodalPrompt mprompt;
      mprompt.prompt_template = formatted;
      input_ids = llm_->tokenizer_encode(mprompt);
    } else {
      input_ids = llm_->tokenizer_encode(formatted);
    }

    // Initialize generation context (no ostream — we handle output manually).
    llm_->generate_init(nullptr, "<eop>");

    // Prefill (max_tokens=0 means no generation, just fill KV cache).
    llm_->generate(input_ids, 0);

    // Decode: generate tokens one by one for streaming.
    std::string full_response;
    auto* ctx = llm_->getContext();
    int max_tokens = 2048;
    size_t prev_len = 0;

    // Prepend thinking start tag when thinking is enabled (consumed during prefill).
    if (thinking_enabled) {
      auto end = formatted.find_last_not_of(" \t\n\r");
      if (end != std::string::npos && formatted[end] == '>') {
        auto lt = formatted.rfind('<', end);
        if (lt != std::string::npos && end - lt >= 3) {
          std::string tag = formatted.substr(lt, end - lt + 1);
          if (tag[1] != '/' && tag.find('|') == std::string::npos) {
            full_response += tag + "\n";
            if (on_token) on_token(tag + "\n");
          }
        }
      }
    }

    while (!cancelled.load() && !llm_->stoped() && ctx->gen_seq_len < max_tokens) {
      llm_->generate(1);
      if (llm_->stoped()) break;

      const std::string& current = ctx->generate_str;
      if (current.size() > prev_len) {
        std::string delta = current.substr(prev_len);
        prev_len = current.size();

        if (delta.find("<eop>") != std::string::npos) break;

        full_response += delta;
        if (on_token && !delta.empty()) {
          if (!on_token(delta)) { cancelled.store(true); break; }
        }
      }
    }

    // Collect metrics.
    if (ctx) {
      last_metrics_ = {ctx->prompt_len, ctx->gen_seq_len, ctx->prefill_us, ctx->decode_us};
    }

    // Clean up temp image file.
    if (!tmp_image_path.empty()) remove(tmp_image_path.c_str());

    return full_response;
  }

  bool load_history(
      const std::vector<std::pair<std::string, std::string>>& messages) override {
    // Not used in stateless mode, but kept for interface compatibility.
    llm_->reset();
    return true;
  }

  void reset() override {
    if (llm_) llm_->reset();
  }

  InferenceMetrics get_metrics() override { return last_metrics_; }
  const char* name() const override { return "mnn"; }

private:
  void release() {
    if (llm_) { MNN::Transformer::Llm::destroy(llm_); llm_ = nullptr; }
  }

  // Parse JSON message array: [{"role":"...","content":"..."},...]
  static MNN::Transformer::ChatMessages parse_chat_messages(const std::string& json) {
    MNN::Transformer::ChatMessages msgs;
    size_t pos = 0;
    while (pos < json.size()) {
      size_t obj_start = json.find('{', pos);
      if (obj_start == std::string::npos) break;
      int depth = 0; bool in_str = false;
      size_t obj_end = obj_start;
      for (size_t i = obj_start; i < json.size(); i++) {
        char c = json[i];
        if (in_str) { if (c == '"' && json[i-1] != '\\') in_str = false; continue; }
        if (c == '"') { in_str = true; continue; }
        if (c == '{') depth++;
        else if (c == '}') { depth--; if (depth == 0) { obj_end = i; break; } }
      }
      std::string obj = json.substr(obj_start, obj_end - obj_start + 1);
      std::string role = extract_string(obj, "role");
      std::string content = extract_string(obj, "content");
      if (!role.empty()) msgs.push_back({role, content});
      pos = obj_end + 1;
    }
    return msgs;
  }

  static std::string extract_string(const std::string& json, const std::string& key) {
    std::string token = "\"" + key + "\"";
    size_t kp = json.find(token);
    if (kp == std::string::npos) return "";
    size_t cp = json.find(':', kp + token.size());
    if (cp == std::string::npos) return "";
    size_t p = cp + 1;
    while (p < json.size() && std::isspace(static_cast<unsigned char>(json[p]))) p++;
    if (p >= json.size() || json[p] != '"') return "";
    p++;
    std::string out;
    bool esc = false;
    while (p < json.size()) {
      char ch = json[p++];
      if (esc) { out.push_back(ch == 'n' ? '\n' : ch == 't' ? '\t' : ch); esc = false; continue; }
      if (ch == '\\') { esc = true; continue; }
      if (ch == '"') break;
      out.push_back(ch);
    }
    return out;
  }

  MNN::Transformer::Llm* llm_ = nullptr;
  std::string cache_dir_;
  InferenceMetrics last_metrics_;
};

}  // namespace omniinfer
