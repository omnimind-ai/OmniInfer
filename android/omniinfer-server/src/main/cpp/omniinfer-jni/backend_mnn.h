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

    // Enable KV cache reuse for multi-turn prefix matching.
    llm_->set_config(R"({"reuse_kv": true})");

    if (!llm_->load()) {
      MNN::Transformer::Llm::destroy(llm_); llm_ = nullptr;
      return false;
    }

    // Workaround: MNN's setChatTemplate passes eos but not bos from the jinja config.
    // Read llm_config.json and inject jinja.bos as bos_token in the jinja context
    // so {{ bos_token }} renders correctly in chat templates (e.g. Gemma models).
    inject_jinja_special_tokens(model_path);

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
    bool is_multimodal = image_data && image_size > 0;
    if (is_multimodal) {
      tmp_image_path = cache_dir_ + "/.omniinfer_vlm_tmp";
      FILE* f = fopen(tmp_image_path.c_str(), "wb");
      if (f) { fwrite(image_data, 1, image_size, f); fclose(f); }
    }

    // Thinking control via jinja context variable (must be set before apply_chat_template).
    if (thinking_enabled) {
      llm_->set_config(R"({"jinja":{"context":{"enable_thinking":true}}})");
    } else {
      llm_->set_config(R"({"jinja":{"context":{"enable_thinking":false}}})");
    }

    // Apply chat template and tokenize.
    std::string formatted = llm_->apply_chat_template(msgs);
    // Save original formatted (before image insertion) for multimodal KV cache comparison.
    std::string formatted_text_only = formatted;

    std::vector<int> input_ids;
    int n_image_tokens = 0;
    if (!tmp_image_path.empty()) {
      auto text_ids = llm_->tokenizer_encode(formatted);
      std::string user_marker = "user\n";
      auto pos = formatted.find(user_marker);
      if (pos != std::string::npos) {
        std::string img_tag = "<img>" + tmp_image_path + "</img>\n";
        formatted.insert(pos + user_marker.size(), img_tag);
      }
      MNN::Transformer::MultimodalPrompt mprompt;
      mprompt.prompt_template = formatted;
      input_ids = llm_->tokenizer_encode(mprompt);
      n_image_tokens = (int)input_ids.size() - (int)text_ids.size();
      if (n_image_tokens < 0) n_image_tokens = 0;
    } else {
      input_ids = llm_->tokenizer_encode(formatted);
    }

    // KV cache prefix reuse.
    int n_cached_tokens = 0;
    if (is_multimodal) {
      // Compare conversation history (without generation prompt) for KV cache reuse.
      std::string conv_history = strip_generation_prompt(formatted_text_only);
      bool reuse_mm = has_cache_ && !prev_eval_prompt_.empty() &&
                      conv_history.size() > prev_eval_prompt_.size() &&
                      conv_history.compare(0, prev_eval_prompt_.size(), prev_eval_prompt_) == 0;

      if (reuse_mm) {
        // Reuse multimodal KV cache: trim old generation prompt + generated tokens,
        // prefill new turns + new generation prompt as text-only suffix.
        std::string suffix_text = formatted_text_only.substr(prev_eval_prompt_.size());
        auto suffix_ids = llm_->tokenizer_encode(suffix_text);
        n_cached_tokens = prev_eval_n_tokens_;
        llm_->generate_init(nullptr, "<eop>");
        llm_->eraseHistory(prev_eval_n_tokens_, 0);
        llm_->generate(suffix_ids, 0);
        __android_log_print(ANDROID_LOG_INFO, "OmniInferJni",
            "MNN multimodal KV cache reuse: %d cached, %d new text tokens",
            prev_eval_n_tokens_, (int)suffix_ids.size());
      } else {
        // Full multimodal eval (first request or new image).
        llm_->reset();
        llm_->generate_init(nullptr, "<eop>");
        llm_->generate(input_ids, 0);
      }

      // Save conversation history (stripped) and token count up to that point.
      prev_eval_prompt_ = conv_history;
      std::string gen_prompt = formatted_text_only.substr(conv_history.size());
      int gen_prompt_tokens = gen_prompt.empty() ? 0 : (int)llm_->tokenizer_encode(gen_prompt).size();
      prev_eval_n_tokens_ = (int)input_ids.size() - gen_prompt_tokens;
      prev_input_ids_.clear();
      has_cache_ = true;
    } else {
      // Find common prefix with previous request.
      int common_prefix = 0;
      if (has_cache_ && !prev_input_ids_.empty()) {
        int max_common = std::min((int)prev_input_ids_.size(), (int)input_ids.size());
        while (common_prefix < max_common &&
               prev_input_ids_[common_prefix] == input_ids[common_prefix]) {
          common_prefix++;
        }
      }

      if (common_prefix > 0 && common_prefix == (int)input_ids.size()) {
        // Exact match: erase from last prompt token, re-prefill 1 token for logits.
        n_cached_tokens = common_prefix - 1;
        llm_->generate_init(nullptr, "<eop>");
        llm_->eraseHistory(common_prefix - 1, 0);
        std::vector<int> last_tok = {input_ids.back()};
        llm_->generate(last_tok, 0);
        __android_log_print(ANDROID_LOG_INFO, "OmniInferJni",
            "MNN KV cache reuse: %d/%d tokens cached (exact, 1 re-prefilled)",
            common_prefix - 1, (int)input_ids.size());
      } else if (common_prefix > 0) {
        // Partial prefix match: reuse cached KV, prefill only suffix.
        n_cached_tokens = common_prefix;
        llm_->generate_init(nullptr, "<eop>");
        llm_->eraseHistory(common_prefix, 0);
        std::vector<int> suffix(input_ids.begin() + common_prefix, input_ids.end());
        llm_->generate(suffix, 0);
        __android_log_print(ANDROID_LOG_INFO, "OmniInferJni",
            "MNN KV cache reuse: %d/%d tokens cached, %d new",
            common_prefix, (int)input_ids.size(),
            (int)input_ids.size() - common_prefix);
      } else {
        // No cache match: full reset + prefill.
        llm_->reset();
        llm_->generate_init(nullptr, "<eop>");
        llm_->generate(input_ids, 0);
      }

      prev_input_ids_ = input_ids;
      has_cache_ = true;
    }

    // Decode: generate tokens one by one for streaming.
    std::string full_response;
    auto* ctx = llm_->getContext();
    int max_tokens = 2048;
    size_t prev_len = 0;
    int n_reasoning_tokens = 0;
    bool counting_reasoning = thinking_enabled && !detect_trailing_tag(formatted).empty();

    // MNN's generate_str doesn't include text consumed during prefill. When
    // thinking is enabled, the chat template appends a thinking start tag
    // (e.g. <think>) that gets consumed. Detect and re-emit it so the Kotlin
    // SSE layer can parse reasoning_content vs content.
    std::string thinking_tag = detect_trailing_tag(formatted);
    if (!thinking_tag.empty() && thinking_enabled) {
      full_response += thinking_tag + "\n";
      if (on_token) on_token(thinking_tag + "\n");
    }

    while (!cancelled.load() && !llm_->stoped() && ctx->gen_seq_len < max_tokens) {
      llm_->generate(1);
      if (llm_->stoped()) break;
      if (counting_reasoning) n_reasoning_tokens++;

      const std::string& current = ctx->generate_str;
      if (current.size() > prev_len) {
        std::string delta = current.substr(prev_len);
        prev_len = current.size();

        if (delta.find("<eop>") != std::string::npos) break;

        full_response += delta;
        if (on_token && !delta.empty()) {
          if (!on_token(delta)) { cancelled.store(true); break; }
        }
        if (counting_reasoning && full_response.find("</think>") != std::string::npos) {
          counting_reasoning = false;
        }
      }
    }

    // Collect metrics. prompt_tokens = cached + actually prefilled.
    if (ctx) {
      int total_prompt = n_cached_tokens + ctx->prompt_len;
      last_metrics_ = {total_prompt, ctx->gen_seq_len, ctx->prefill_us, ctx->decode_us,
                       n_reasoning_tokens, n_image_tokens, n_cached_tokens};
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
    prev_input_ids_.clear();
    prev_eval_prompt_.clear();
    prev_eval_n_tokens_ = 0;
    has_cache_ = false;
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

  // Strip the trailing generation prompt from a formatted chat template output.
  // The generation prompt starts at the last assistant/model role marker.
  // Used by multimodal KV cache to compare only conversation history.
  static std::string strip_generation_prompt(const std::string& formatted) {
    size_t cut = std::string::npos;
    for (const char* marker : {"<|im_start|>assistant", "<start_of_turn>model"}) {
      auto pos = formatted.rfind(marker);
      if (pos != std::string::npos && (cut == std::string::npos || pos > cut)) cut = pos;
    }
    return (cut != std::string::npos) ? formatted.substr(0, cut) : formatted;
  }

  // Read llm_config.json and inject jinja.bos / jinja.eos as bos_token / eos_token
  // into the jinja context. Works around MNN's setChatTemplate not passing bos.
  void inject_jinja_special_tokens(const std::string& model_path) {
    auto slash = model_path.find_last_of("/\\");
    if (slash == std::string::npos) return;
    std::string cfg_path = model_path.substr(0, slash) + "/llm_config.json";
    FILE* f = fopen(cfg_path.c_str(), "r");
    if (!f) return;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string buf(sz, '\0');
    fread(&buf[0], 1, sz, f);
    fclose(f);

    // Find the "jinja" section so we read bos/eos from it (not the top-level keys).
    size_t jinja_pos = buf.find("\"jinja\"");
    if (jinja_pos == std::string::npos) return;
    std::string jinja_section = buf.substr(jinja_pos);

    std::string bos = extract_string(jinja_section, "bos");
    std::string eos = extract_string(jinja_section, "eos");
    if (bos.empty() && eos.empty()) return;

    // Build a single set_config call to inject both tokens.
    std::ostringstream cfg;
    cfg << R"({"jinja":{"context":{)";
    bool need_comma = false;
    if (!bos.empty()) {
      cfg << R"("bos_token":")" << bos << "\"";
      need_comma = true;
    }
    if (!eos.empty()) {
      if (need_comma) cfg << ",";
      cfg << R"("eos_token":")" << eos << "\"";
    }
    cfg << "}}}";
    llm_->set_config(cfg.str());
    __android_log_print(ANDROID_LOG_INFO, "OmniInferJni",
        "Injected jinja tokens: bos=[%s] eos=[%s]", bos.c_str(), eos.c_str());
  }

  // Detect a trailing opening tag at the end of the formatted prompt.
  // Returns the tag string (e.g. "<think>") or empty if none found.
  // Filters out template role markers like <|im_start|> and closing tags.
  static std::string detect_trailing_tag(const std::string& text) {
    auto end = text.find_last_not_of(" \t\n\r");
    if (end == std::string::npos || text[end] != '>') return "";
    auto lt = text.rfind('<', end);
    if (lt == std::string::npos || end - lt < 3) return "";
    std::string tag = text.substr(lt, end - lt + 1);
    if (tag[1] == '/') return "";              // closing tag </...>
    if (tag.find('|') != std::string::npos) return "";  // role marker <|...|>
    return tag;
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
  // KV cache prefix reuse state.
  std::vector<int> prev_input_ids_;     // text-only token comparison
  std::string prev_eval_prompt_;        // multimodal string prefix comparison
  int prev_eval_n_tokens_ = 0;          // total tokens from last multimodal eval
  bool has_cache_ = false;
};

}  // namespace omniinfer
