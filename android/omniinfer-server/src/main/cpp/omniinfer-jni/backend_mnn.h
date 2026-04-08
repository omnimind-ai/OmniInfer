#pragma once

#include "inference_backend.h"

#include <llm/llm.hpp>

namespace omniinfer {

class MnnBackend : public InferenceBackend {
public:
  ~MnnBackend() override { release(); }

  bool load(const std::string& model_path, const std::string& config_json,
            const std::string& /*native_lib_dir*/, int n_threads, int n_ctx) override {
    llm_ = MNN::Transformer::Llm::createLLM(model_path);
    if (!llm_) return false;

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
      bool /*thinking_enabled*/,
      std::atomic<bool>& cancelled,
      std::function<bool(const std::string& token)> on_token) override {

    using ChatMessages = MNN::Transformer::ChatMessages;
    ChatMessages msgs;
    if (!system_prompt.empty()) msgs.push_back({"system", system_prompt});
    for (auto& m : history_) msgs.push_back({m.first, m.second});
    msgs.push_back({"user", user_prompt});

    // Apply chat template and tokenize.
    std::string formatted = llm_->apply_chat_template(msgs);
    auto input_ids = llm_->tokenizer_encode(formatted);

    // Initialize generation context (no ostream — we handle output manually).
    llm_->generate_init(nullptr, "<eop>");

    // Prefill only (max_tokens=0 means no generation, just fill KV cache).
    llm_->generate(input_ids, 0);

    // Decode: generate tokens one by one for streaming.
    std::string full_response;
    auto* ctx = llm_->getContext();
    int max_tokens = 2048;
    size_t prev_len = 0;

    // If the chat template ends with a thinking start tag (e.g. Qwen3.5 "<think>\n"),
    // it is part of the input prompt, not the output. Prepend it for a complete block.
    {
      const std::string think_tag = "<think>";
      auto pos = formatted.rfind(think_tag);
      if (pos != std::string::npos && pos >= formatted.size() - think_tag.size() - 2) {
        full_response += think_tag + "\n";
        if (on_token) on_token(think_tag + "\n");
      }
    }

    while (!cancelled.load() && !llm_->stoped() && ctx->gen_seq_len < max_tokens) {
      llm_->generate(1);
      if (llm_->stoped()) break;

      // Get incremental text from generate_str.
      const std::string& current = ctx->generate_str;
      if (current.size() > prev_len) {
        std::string delta = current.substr(prev_len);
        prev_len = current.size();

        // Skip <eop> marker.
        if (delta.find("<eop>") != std::string::npos) break;

        full_response += delta;
        if (on_token && !delta.empty()) {
          if (!on_token(delta)) { cancelled.store(true); break; }
        }
      }
    }

    // Update history.
    history_.push_back({"user", user_prompt});
    history_.push_back({"assistant", full_response});

    // Collect metrics.
    if (ctx) {
      last_metrics_ = {ctx->prompt_len, ctx->gen_seq_len, ctx->prefill_us, ctx->decode_us};
    }

    return full_response;
  }

  bool load_history(
      const std::vector<std::pair<std::string, std::string>>& messages) override {
    history_ = messages;
    llm_->reset();
    return true;
  }

  void reset() override {
    history_.clear();
    if (llm_) llm_->reset();
  }

  InferenceMetrics get_metrics() override { return last_metrics_; }
  const char* name() const override { return "mnn"; }

private:
  void release() {
    if (llm_) { MNN::Transformer::Llm::destroy(llm_); llm_ = nullptr; }
  }

  MNN::Transformer::Llm* llm_ = nullptr;
  std::vector<std::pair<std::string, std::string>> history_;
  InferenceMetrics last_metrics_;
};

}  // namespace omniinfer
