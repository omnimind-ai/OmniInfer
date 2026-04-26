#pragma once

#include "inference_backend.h"
#include "soc_defaults.h"
#include "thinking_tags.h"
#include "tool_call_parser.h"

#include <llm/llm.hpp>

#include <android/log.h>
#include <cstdio>
#include <unistd.h>

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

    int eff_threads = n_threads > 0 ? n_threads : get_soc_default_threads();
    n_threads_ = eff_threads;
    n_ctx_ = n_ctx > 0 ? n_ctx : 16384;

    // Detect GPU backend from caller config.
    std::string backend_type = extract_string(config_json, "backend_type");
    is_gpu_ = (backend_type == "opencl" || backend_type == "vulkan");

    std::ostringstream cfg;
    if (is_gpu_) {
      // GPU: thread_num is a bitmask, not thread count.
      // MNN_GPU_MEMORY_BUFFER(64) | MNN_GPU_TUNING_WIDE(4) = 68
      cfg << "{\"thread_num\":68";
      cfg << ",\"tmp_path\":\"" << cache_dir_ << "\"";
    } else {
      cfg << "{\"thread_num\":" << eff_threads;
    }
    if (n_ctx > 0) cfg << ",\"max_new_tokens\":" << n_ctx;
    cfg << "}";
    llm_->set_config(cfg.str());

    if (!config_json.empty()) llm_->set_config(config_json);

    // KV cache reuse: enabled for CPU, disabled for GPU (not yet validated on GPU).
    llm_->set_config(is_gpu_ ? R"({"reuse_kv": false})" : R"({"reuse_kv": true})");

    if (!llm_->load()) {
      MNN::Transformer::Llm::destroy(llm_); llm_ = nullptr;
      return false;
    }

    __android_log_print(ANDROID_LOG_INFO, "OmniInferJni",
        "MNN backend loaded: type=%s, threads=%d, ctx=%d",
        is_gpu_ ? backend_type.c_str() : "cpu", eff_threads, n_ctx_);

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
      const std::string& tools_json,
      const std::string& /*tool_choice*/,
      const std::string& messages_json,
      const std::vector<std::vector<uint8_t>>& images,
      int max_tokens,
      std::atomic<bool>& graceful_stop,
      const std::string& sampling_json = "") override {

    // Apply per-request sampling parameters via set_config.
    if (!sampling_json.empty()) llm_->set_config(sampling_json);

    using ChatMessages = MNN::Transformer::ChatMessages;
    ChatMessages msgs;

    // Parse messages from JSON array or use legacy single-turn path.
    if (!messages_json.empty()) {
      msgs = parse_chat_messages(messages_json);
    } else {
      if (!system_prompt.empty()) msgs.push_back({"system", system_prompt});
      if (!user_prompt.empty()) msgs.push_back({"user", user_prompt});
    }

    // Handle multimodal: write each image to a temp file.
    std::vector<std::string> tmp_image_paths;
    bool is_multimodal = !images.empty();
    if (is_multimodal) {
      for (size_t i = 0; i < images.size(); i++) {
        std::string path = cache_dir_ + "/.omniinfer_vlm_tmp_" + std::to_string(i);
        FILE* f = fopen(path.c_str(), "wb");
        if (f) { fwrite(images[i].data(), 1, images[i].size(), f); fclose(f); }
        tmp_image_paths.push_back(path);
      }
    }

    // Thinking control via jinja context variable (must be set before apply_chat_template).
    if (thinking_enabled) {
      llm_->set_config(R"({"jinja":{"context":{"enable_thinking":true}}})");
    } else {
      llm_->set_config(R"({"jinja":{"context":{"enable_thinking":false}}})");
    }

    // Tool calling: inject tools into jinja context for template rendering.
    bool has_tools = !tools_json.empty();
    if (has_tools) {
      llm_->set_config("{\"jinja\":{\"context\":{\"tools\":" + tools_json + "}}}");
    }

    // Apply chat template and tokenize.
    std::string formatted = llm_->apply_chat_template(msgs);
    // Save original formatted (before image insertion) for multimodal KV cache comparison.
    std::string formatted_text_only = formatted;

    std::vector<int> input_ids;
    int n_image_tokens = 0;
    if (!tmp_image_paths.empty()) {
      auto text_ids = llm_->tokenizer_encode(formatted);
      // Replace each <image> placeholder with <img>path</img> tag.
      size_t img_idx = 0;
      size_t pos = 0;
      while ((pos = formatted.find("<image>", pos)) != std::string::npos && img_idx < tmp_image_paths.size()) {
        std::string img_tag = "<img>" + tmp_image_paths[img_idx] + "</img>";
        formatted.replace(pos, 7, img_tag);
        pos += img_tag.size();
        img_idx++;
      }
      MNN::Transformer::MultimodalPrompt mprompt;
      mprompt.prompt_template = formatted;
      input_ids = llm_->tokenizer_encode(mprompt);
      n_image_tokens = (int)input_ids.size() - (int)text_ids.size();
      if (n_image_tokens < 0) n_image_tokens = 0;
    } else {
      input_ids = llm_->tokenizer_encode(formatted);
    }

    // Check prompt fits in context window.
    if ((int)input_ids.size() > n_ctx_ - 4) {
      std::ostringstream err;
      err << R"({"error":"prompt_too_long","prompt_tokens":)" << (int)input_ids.size()
          << R"(,"max_context":)" << n_ctx_ << "}";
      for (const auto& p : tmp_image_paths) remove(p.c_str());
      return err.str();
    }

    // KV cache prefix reuse (CPU only; GPU reuse_kv is disabled).
    int n_cached_tokens = 0;
    if (is_gpu_) {
      // GPU mode: always full prefill (reuse_kv not yet validated on GPU).
      llm_->reset();
      llm_->generate_init(nullptr, "<eop>");
      llm_->generate(input_ids, 0);
    } else if (is_multimodal) {
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

      bool cache_reuse_ok = false;
      if (common_prefix > 0 && common_prefix == (int)input_ids.size()) {
        // Exact match: erase from last prompt token, re-prefill 1 token for logits.
        n_cached_tokens = common_prefix - 1;
        llm_->generate_init(nullptr, "<eop>");
        llm_->eraseHistory(common_prefix - 1, 0);
        std::vector<int> last_tok = {input_ids.back()};
        llm_->generate(last_tok, 0);
        cache_reuse_ok = !llm_->stoped();
        if (cache_reuse_ok) {
          __android_log_print(ANDROID_LOG_INFO, "OmniInferJni",
              "MNN KV cache reuse: %d/%d tokens cached (exact, 1 re-prefilled)",
              common_prefix - 1, (int)input_ids.size());
        }
      } else if (common_prefix > 0) {
        // Partial prefix match: reuse cached KV, prefill only suffix.
        n_cached_tokens = common_prefix;
        llm_->generate_init(nullptr, "<eop>");
        llm_->eraseHistory(common_prefix, 0);
        std::vector<int> suffix(input_ids.begin() + common_prefix, input_ids.end());
        llm_->generate(suffix, 0);
        cache_reuse_ok = !llm_->stoped();
        if (cache_reuse_ok) {
          __android_log_print(ANDROID_LOG_INFO, "OmniInferJni",
              "MNN KV cache reuse: %d/%d tokens cached, %d new",
              common_prefix, (int)input_ids.size(),
              (int)input_ids.size() - common_prefix);
        }
      }
      if (!cache_reuse_ok) {
        // No cache match, or cache reuse failed: full reset + prefill.
        n_cached_tokens = 0;
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
    int eff_max_tokens = max_tokens > 0 ? max_tokens : 4096;
    size_t prev_len = 0;
    int n_reasoning_tokens = 0;

    // Detect thinking tag from formatted prompt and set up normalization.
    // Try standard detection first, then fall back to the shared registry
    // for non-standard tags (e.g. Gemma 4's <|channel>thought contains |
    // which detect_trailing_tag() filters out).
    std::string thinking_tag = detect_trailing_tag(formatted);
    const thinking_tags::NativeTagPair* think_non_std = nullptr;
    if (thinking_tag.empty() && thinking_enabled) {
      think_non_std = thinking_tags::detect_in_prompt(formatted);
    }
    if (!think_non_std && !thinking_tag.empty()) {
      think_non_std = thinking_tags::find_non_standard(thinking_tag);
    }

    std::optional<thinking_tags::Normalizer> think_norm;
    bool counting_reasoning = false;
    if (thinking_enabled && (think_non_std || !thinking_tag.empty())) {
      counting_reasoning = true;
      if (think_non_std) {
        think_norm.emplace(*think_non_std);
        full_response += "<think>\n";
        if (on_token) on_token("<think>\n");
      } else {
        full_response += thinking_tag + "\n";
        if (on_token) on_token(thinking_tag + "\n");
      }
    }

    // Buffer for incomplete multi-byte UTF-8 sequences (e.g. emoji split across tokens).
    std::string utf8_buf;

    while (!cancelled.load() && !llm_->stoped() && ctx->gen_seq_len < eff_max_tokens) {
      llm_->generate(1);
      if (llm_->stoped()) break;
      if (counting_reasoning) n_reasoning_tokens++;

      const std::string& current = ctx->generate_str;
      if (current.size() > prev_len) {
        std::string delta = current.substr(prev_len);
        prev_len = current.size();

        if (delta.find("<eop>") != std::string::npos) break;

        utf8_buf += delta;
        if (is_valid_utf8(utf8_buf)) {
          std::string to_emit = think_norm ? think_norm->process(utf8_buf) : utf8_buf;
          if (!to_emit.empty()) {
            full_response += to_emit;
            if (on_token && !on_token(to_emit)) { cancelled.store(true); break; }
          }
          if (counting_reasoning && full_response.find("</think>") != std::string::npos) {
            counting_reasoning = false;
          }
          utf8_buf.clear();
        }
      }
    }

    if (cancelled.load()) {
      if (graceful_stop.load()) {
        // Graceful stop (/v1/cancel): keep KV cache for prefix matching.
        // Re-tokenize prompt + generated text to update tracking.
        // Multimodal tracking (prev_eval_prompt_, prev_eval_n_tokens_)
        // was already set during prefill and remains valid.
        if (!is_multimodal && !full_response.empty()) {
          prev_input_ids_ = llm_->tokenizer_encode(formatted + full_response);
        }
        // has_cache_ stays true, no llm_->reset().
      } else {
        // Hard cancel (client disconnect): invalidate cache for clean state.
        llm_->reset();
        has_cache_ = false;
        prev_input_ids_.clear();
        prev_eval_prompt_.clear();
        prev_eval_n_tokens_ = 0;
      }
    }

    // Flush any remaining buffered bytes (through normalizer if active).
    if (!utf8_buf.empty()) {
      std::string to_emit = think_norm ? think_norm->process(utf8_buf) : utf8_buf;
      if (!to_emit.empty()) { full_response += to_emit; if (on_token) on_token(to_emit); }
      utf8_buf.clear();
    }
    if (think_norm) {
      auto remaining = think_norm->flush();
      if (!remaining.empty()) { full_response += remaining; if (on_token) on_token(remaining); }
    }

    // Collect metrics. prompt_tokens = cached + actually prefilled.
    if (ctx) {
      int total_prompt = n_cached_tokens + ctx->prompt_len;
      last_metrics_ = {total_prompt, ctx->gen_seq_len, ctx->prefill_us, ctx->decode_us,
                       n_reasoning_tokens, n_image_tokens, n_cached_tokens};
    }

    // Tool calling: parse output and clear tools context.
    if (has_tools) {
      llm_->set_config(R"({"jinja":{"context":{"tools":null}}})");
      std::string tool_result = tool_parser::parse_tool_calls(full_response);
      if (!tool_result.empty()) full_response = tool_result;
    }

    // Clean up temp image files.
    for (const auto& path : tmp_image_paths) remove(path.c_str());

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
  int n_threads() const override { return n_threads_; }
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
      // Multi-turn tool use: pass full message object via "json" role for
      // assistant messages with tool_calls and tool-result messages.
      if (role == "tool" ||
          (role == "assistant" && obj.find("\"tool_calls\"") != std::string::npos)) {
        msgs.push_back({"json", obj});
      } else if (!role.empty()) {
        msgs.push_back({role, content});
      }
      pos = obj_end + 1;
    }
    return msgs;
  }

  static bool is_valid_utf8(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
      unsigned char c = static_cast<unsigned char>(s[i]);
      int len = (c < 0x80) ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 0;
      if (!len) return false;
      if (i + len > s.size()) return false; // incomplete sequence at end
      for (int j = 1; j < len; j++) {
        if ((static_cast<unsigned char>(s[i + j]) & 0xC0) != 0x80) return false;
      }
      i += len;
    }
    return true;
  }

  // Strip the trailing generation prompt from a formatted chat template output.
  // The generation prompt starts at the last assistant/model role marker.
  // Used by multimodal KV cache to compare only conversation history.
  static std::string strip_generation_prompt(const std::string& formatted) {
    size_t cut = std::string::npos;
    for (const char* marker : {"<|im_start|>assistant", "<start_of_turn>model", "<|turn>model"}) {
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
  int n_threads_ = 0;
  int n_ctx_ = 16384;
  bool is_gpu_ = false;
  InferenceMetrics last_metrics_;
  // KV cache prefix reuse state.
  std::vector<int> prev_input_ids_;     // text-only token comparison
  std::string prev_eval_prompt_;        // multimodal string prefix comparison
  int prev_eval_n_tokens_ = 0;          // total tokens from last multimodal eval
  bool has_cache_ = false;
};

}  // namespace omniinfer
