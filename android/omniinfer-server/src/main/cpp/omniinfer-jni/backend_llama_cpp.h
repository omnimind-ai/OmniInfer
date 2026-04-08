#pragma once

#include "inference_backend.h"

#include "llama.h"
#include "common.h"
#include "chat.h"
#include "sampling.h"

#include <chrono>
#include <unistd.h>

namespace omniinfer {

class LlamaCppBackend : public InferenceBackend {
public:
  ~LlamaCppBackend() override { release(); }

  bool load(const std::string& model_path, const std::string& /*config_json*/,
            const std::string& native_lib_dir, int n_threads, int n_ctx) override {
    std::call_once(s_backend_init, [&]() {
      if (!native_lib_dir.empty()) {
        ggml_backend_load_all_from_path(native_lib_dir.c_str());
      } else {
        ggml_backend_load_all();
      }
      llama_backend_init();
    });

    llama_model_params mp = llama_model_default_params();
    model_ = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model_) return false;

    int eff_threads = std::max(2, n_threads > 0 ? n_threads
        : (int)sysconf(_SC_NPROCESSORS_ONLN) - 1);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = eff_threads;
    cp.n_threads_batch = eff_threads;
    ctx_ = llama_init_from_model(model_, cp);
    if (!ctx_) { llama_model_free(model_); model_ = nullptr; return false; }

    common_params_sampling sp;
    sp.temp = 0.7f;
    sampler_ = common_sampler_init(model_, sp);
    chat_templates_ = common_chat_templates_init(model_, "");
    batch_ = llama_batch_init(512, 0, 1);
    n_ctx_ = n_ctx;
    return true;
  }

  std::string generate(
      const std::string& system_prompt,
      const std::string& user_prompt,
      bool /*thinking_enabled*/,
      std::atomic<bool>& cancelled,
      std::function<bool(const std::string& token)> on_token) override {

    // Stateless: reset all state at start of each request.
    cur_pos_ = 0;
    llama_memory_clear(llama_get_memory(ctx_), false);
    common_sampler_reset(sampler_);

    // Build full messages vector.
    std::vector<common_chat_msg> messages;
    if (!system_prompt.empty()) {
      common_chat_msg sys_msg;
      sys_msg.role = "system";
      sys_msg.content = system_prompt;
      messages.push_back(std::move(sys_msg));
    }
    {
      common_chat_msg user_msg;
      user_msg.role = "user";
      user_msg.content = user_prompt;
      messages.push_back(std::move(user_msg));
    }

    // Apply chat template with all messages at once (avoids Qwen3.5 Jinja crash).
    common_chat_templates_inputs inputs;
    inputs.messages = messages;
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;

    common_chat_params params = common_chat_templates_apply(chat_templates_.get(), inputs);

    // Tokenize the full formatted prompt.
    auto prompt_toks = common_tokenize(ctx_, params.prompt, true, true);

    auto t_prefill_start = std::chrono::steady_clock::now();
    if (decode_batched(prompt_toks, 0, true) != 0) return "";
    auto t_prefill_end = std::chrono::steady_clock::now();
    int64_t prefill_us = std::chrono::duration_cast<std::chrono::microseconds>(t_prefill_end - t_prefill_start).count();
    cur_pos_ = (int)prompt_toks.size();

    // Generate tokens.
    common_sampler_reset(sampler_);
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    std::string full_response;
    std::string utf8_buf;
    auto t_decode_start = std::chrono::steady_clock::now();

    while (!cancelled.load()) {
      if (cur_pos_ >= n_ctx_ - 4) shift_context();

      llama_token tok = common_sampler_sample(sampler_, ctx_, -1);
      common_sampler_accept(sampler_, tok, true);

      if (llama_vocab_is_eog(vocab, tok)) {
        if (!utf8_buf.empty()) {
          full_response += utf8_buf;
          if (on_token) on_token(utf8_buf);
          utf8_buf.clear();
        }
        break;
      }

      common_batch_clear(batch_);
      common_batch_add(batch_, tok, cur_pos_, {0}, true);
      if (llama_decode(ctx_, batch_) != 0) break;
      cur_pos_++;

      auto piece = common_token_to_piece(ctx_, tok);
      utf8_buf += piece;

      if (is_valid_utf8(utf8_buf.c_str())) {
        full_response += utf8_buf;
        if (on_token && !on_token(utf8_buf)) { cancelled.store(true); break; }
        utf8_buf.clear();
      }
    }

    // Strip template-specific stop sequences from output.
    for (const auto& stop : params.additional_stops) {
      auto pos = full_response.find(stop);
      if (pos != std::string::npos) {
        full_response = full_response.substr(0, pos);
      }
    }

    auto t_decode_end = std::chrono::steady_clock::now();
    int64_t decode_us = std::chrono::duration_cast<std::chrono::microseconds>(t_decode_end - t_decode_start).count();
    int n_prompt = (int)prompt_toks.size();
    int n_generated = cur_pos_ - n_prompt;
    last_metrics_ = {n_prompt, n_generated, prefill_us, decode_us};
    return full_response;
  }

  bool load_history(
      const std::vector<std::pair<std::string, std::string>>& messages) override {
    cur_pos_ = 0;
    llama_memory_clear(llama_get_memory(ctx_), false);

    std::vector<common_chat_msg> msgs;
    for (const auto& m : messages) {
      common_chat_msg msg;
      msg.role = m.first;
      msg.content = m.second;
      msgs.push_back(std::move(msg));
    }

    common_chat_templates_inputs inputs;
    inputs.messages = msgs;
    inputs.add_generation_prompt = false;
    inputs.use_jinja = true;

    common_chat_params params = common_chat_templates_apply(chat_templates_.get(), inputs);
    auto toks = common_tokenize(ctx_, params.prompt, true, true);
    if (decode_batched(toks, 0) != 0) return false;
    cur_pos_ = (int)toks.size();
    return true;
  }

  void reset() override {
    cur_pos_ = 0;
    llama_memory_clear(llama_get_memory(ctx_), false);
    common_sampler_reset(sampler_);
  }

  InferenceMetrics get_metrics() override { return last_metrics_; }
  const char* name() const override { return "llama.cpp"; }

private:
  void release() {
    if (sampler_) { common_sampler_free(sampler_); sampler_ = nullptr; }
    chat_templates_.reset();
    llama_batch_free(batch_); batch_ = {};
    if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
    if (model_) { llama_model_free(model_); model_ = nullptr; }
  }

  void shift_context() {
    int n_discard = cur_pos_ / 2;
    if (n_discard <= 0) return;
    llama_memory_seq_rm(llama_get_memory(ctx_), 0, 0, n_discard);
    llama_memory_seq_add(llama_get_memory(ctx_), 0, n_discard, cur_pos_, -n_discard);
    cur_pos_ -= n_discard;
  }

  int decode_batched(const std::vector<llama_token>& toks, llama_pos start, bool last_logit = false) {
    for (int i = 0; i < (int)toks.size(); i += 512) {
      int n = std::min((int)toks.size() - i, 512);
      common_batch_clear(batch_);
      if (start + i + n >= n_ctx_ - 4) shift_context();
      for (int j = 0; j < n; j++) {
        bool want = last_logit && (i + j == (int)toks.size() - 1);
        common_batch_add(batch_, toks[i + j], start + i + j, {0}, want);
      }
      if (llama_decode(ctx_, batch_) != 0) return 1;
    }
    return 0;
  }

  static bool is_valid_utf8(const char* s) {
    if (!s) return true;
    const auto* b = reinterpret_cast<const unsigned char*>(s);
    while (*b) {
      int n = (*b & 0x80) == 0 ? 1 : (*b & 0xE0) == 0xC0 ? 2 : (*b & 0xF0) == 0xE0 ? 3 : (*b & 0xF8) == 0xF0 ? 4 : 0;
      if (!n) return false;
      b++;
      for (int i = 1; i < n; i++) { if ((*b & 0xC0) != 0x80) return false; b++; }
    }
    return true;
  }

  static std::once_flag s_backend_init;
  llama_model* model_ = nullptr;
  llama_context* ctx_ = nullptr;
  common_sampler* sampler_ = nullptr;
  common_chat_templates_ptr chat_templates_;
  llama_batch batch_ = {};
  llama_pos cur_pos_ = 0;
  int n_ctx_ = 4096;
  InferenceMetrics last_metrics_;
};

std::once_flag LlamaCppBackend::s_backend_init;

}  // namespace omniinfer
