#pragma once

#include "inference_backend.h"
#include "thinking_tags.h"
#include "tool_call_parser.h"

#include "llama.h"
#include "common.h"
#include "chat.h"
#include "sampling.h"
#include "peg-parser.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <android/log.h>
#include <chrono>
#include <unistd.h>
#include <sstream>
#include <dirent.h>

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

    int eff_threads = n_threads > 0 ? n_threads : (int)sysconf(_SC_NPROCESSORS_ONLN);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = eff_threads;
    cp.n_threads_batch = eff_threads;
    cp.type_k = GGML_TYPE_F16;                          // KV cache quantization: 50% memory reduction
    cp.type_v = GGML_TYPE_F16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;  // Flash attention: faster prefill, less memory
    ctx_ = llama_init_from_model(model_, cp);
    if (!ctx_) { llama_model_free(model_); model_ = nullptr; return false; }

    common_params_sampling sp;
    sp.temp = 0.7f;
    sampler_ = common_sampler_init(model_, sp);
    chat_templates_ = common_chat_templates_init(model_, "");
    batch_ = llama_batch_init(512, 0, 1);
    n_ctx_ = n_ctx;
    n_threads_ = eff_threads;

    // Auto-discover mmproj in same directory for multimodal models.
    std::string mmproj_path = find_mmproj(model_path);
    if (!mmproj_path.empty()) {
      mtmd_context_params mparams = mtmd_context_params_default();
      mparams.use_gpu = false;
      mparams.n_threads = eff_threads;
      media_marker_ = mparams.media_marker ? mparams.media_marker : "<__media__>";
      mtmd_ctx_ = mtmd_init_from_file(mmproj_path.c_str(), model_, mparams);
      if (mtmd_ctx_) {
        __android_log_print(ANDROID_LOG_INFO, "OmniInferJni",
            "Loaded mmproj: %s", mmproj_path.c_str());
      }
    }

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
      std::atomic<bool>& graceful_stop) override {

    common_sampler_reset(sampler_);

    // Build full messages vector.
    std::vector<common_chat_msg> messages;
    if (!messages_json.empty()) {
      messages = parse_messages(messages_json);
    } else {
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
    }

    bool is_multimodal = !images.empty() && mtmd_ctx_;

    if (is_multimodal) {
      // Replace <image> placeholders with media markers for mtmd.
      // Kotlin inserts <image> at the exact position of each image_url in the content.
      for (auto& msg : messages) {
        size_t pos = 0;
        while ((pos = msg.content.find("<image>", pos)) != std::string::npos) {
          msg.content.replace(pos, 7, media_marker_ + "\n");
          pos += media_marker_.size() + 1;
        }
      }
    }

    // Apply chat template with all messages at once.
    common_chat_templates_inputs inputs;
    inputs.messages = messages;
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;
    inputs.enable_thinking = thinking_enabled;

    // Parse and set tools if provided.
    bool has_tools = false;
    if (!tools_json.empty()) {
      inputs.tools = parse_tools(tools_json);
      has_tools = !inputs.tools.empty();
      if (!tool_choice.empty()) {
        if (tool_choice == "none") inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
        else if (tool_choice == "required") inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        else inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
      }
    }

    common_chat_params params = common_chat_templates_apply(chat_templates_.get(), inputs);

    // Build parser params for tool call detection.
    common_chat_parser_params parser_params(params);
    parser_params.parse_tool_calls = has_tools;
    if (!params.parser.empty()) {
      parser_params.parser.load(params.parser);
    }

    // Prefill with KV cache prefix reuse.
    auto t_prefill_start = std::chrono::steady_clock::now();
    int n_prompt_tokens = 0;
    int n_cached_tokens = 0;
    int n_image_tokens = 0;

    if (is_multimodal) {
      // Quick check: text tokens alone (excluding image) must fit in context.
      auto text_check = common_tokenize(ctx_, params.prompt, true, true);
      if ((int)text_check.size() > n_ctx_ - 4) {
        std::ostringstream err;
        err << R"({"error":"prompt_too_long","prompt_tokens":)" << (int)text_check.size()
            << R"(,"max_context":)" << n_ctx_ << "}";
        return err.str();
      }

      // Compare conversation history (without generation prompt) for KV cache reuse.
      std::string conv_history = strip_generation_prompt(params.prompt);
      bool reuse_mm = has_cache_ && !prev_eval_prompt_.empty() &&
                      conv_history.size() > prev_eval_prompt_.size() &&
                      conv_history.compare(0, prev_eval_prompt_.size(), prev_eval_prompt_) == 0;
      std::string suffix;
      if (reuse_mm) {
        suffix = params.prompt.substr(prev_eval_prompt_.size());
        if (suffix.find(media_marker_) != std::string::npos) reuse_mm = false;
      }

      if (reuse_mm) {
        // Try KV cache reuse: trim old gen prompt + generated tokens, decode suffix.
        n_cached_tokens = prev_eval_n_tokens_;
        bool trimmed = llama_memory_seq_rm(llama_get_memory(ctx_), 0, prev_eval_n_tokens_, -1);
        if (trimmed) {
          auto suffix_toks = common_tokenize(ctx_, suffix, false, false);
          if (decode_batched(suffix_toks, prev_eval_n_tokens_, true) != 0) {
            reuse_mm = false;
            n_cached_tokens = 0;
          } else {
            cur_pos_ = prev_eval_n_tokens_ + (int)suffix_toks.size();
            n_prompt_tokens = cur_pos_;
            auto conv_text_toks = common_tokenize(ctx_, prev_eval_prompt_, true, true);
            n_image_tokens = prev_eval_n_tokens_ - (int)conv_text_toks.size();
            if (n_image_tokens < 0) n_image_tokens = 0;
            __android_log_print(ANDROID_LOG_INFO, "OmniInferJni",
                "Multimodal KV cache reuse: %d cached, %d new text tokens",
                prev_eval_n_tokens_, (int)suffix_toks.size());
          }
        } else {
          // seq_rm failed (SWA/hybrid models): fall through to full multimodal eval.
          reuse_mm = false;
          n_cached_tokens = 0;
          __android_log_print(ANDROID_LOG_INFO, "OmniInferJni",
              "Multimodal KV cache: seq_rm failed (SWA model), full re-eval");
        }
      }

      if (!reuse_mm) {
        // Full multimodal eval (first request or new image).
        cur_pos_ = 0;
        llama_memory_clear(llama_get_memory(ctx_), false);

        // Create bitmaps for all images.
        std::vector<mtmd_bitmap*> bmps;
        for (const auto& img : images) {
          auto* bmp = mtmd_helper_bitmap_init_from_buf(mtmd_ctx_, img.data(), img.size());
          if (bmp) bmps.push_back(bmp);
        }
        if (bmps.empty()) return "";

        mtmd_input_text text{params.prompt.c_str(), true, true};
        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        std::vector<const mtmd_bitmap*> bmp_ptrs(bmps.begin(), bmps.end());
        if (mtmd_tokenize(mtmd_ctx_, chunks, &text, bmp_ptrs.data(), bmp_ptrs.size()) != 0) {
          for (auto* b : bmps) mtmd_bitmap_free(b);
          mtmd_input_chunks_free(chunks);
          return "";
        }

        llama_pos n_past = 0;
        if (mtmd_helper_eval_chunks(mtmd_ctx_, ctx_, chunks, 0, 0, 512, true, &n_past) != 0) {
          for (auto* b : bmps) mtmd_bitmap_free(b);
          mtmd_input_chunks_free(chunks);
          return "";
        }

        cur_pos_ = n_past;
        n_prompt_tokens = (int)mtmd_helper_get_n_tokens(chunks);
        auto text_toks = common_tokenize(ctx_, params.prompt, true, true);
        n_image_tokens = n_prompt_tokens - (int)text_toks.size();
        if (n_image_tokens < 0) n_image_tokens = 0;
        for (auto* b : bmps) mtmd_bitmap_free(b);
        mtmd_input_chunks_free(chunks);
      }

      // Save conversation history (stripped of generation prompt) and token count.
      prev_eval_prompt_ = conv_history;
      auto gen_prompt_toks = common_tokenize(ctx_, params.prompt.substr(conv_history.size()), false, false);
      prev_eval_n_tokens_ = n_prompt_tokens - (int)gen_prompt_toks.size();
      prev_prompt_tokens_.clear();
      has_cache_ = true;
    } else {
      // Text-only path with KV cache prefix reuse.
      auto prompt_toks = common_tokenize(ctx_, params.prompt, true, true);
      n_prompt_tokens = (int)prompt_toks.size();

      if (n_prompt_tokens > n_ctx_ - 4) {
        std::ostringstream err;
        err << R"({"error":"prompt_too_long","prompt_tokens":)" << n_prompt_tokens
            << R"(,"max_context":)" << n_ctx_ << "}";
        return err.str();
      }

      // Find common prefix length with previous prompt tokens.
      int common_prefix = 0;
      if (has_cache_ && !prev_prompt_tokens_.empty()) {
        int max_common = std::min((int)prev_prompt_tokens_.size(), (int)prompt_toks.size());
        while (common_prefix < max_common &&
               prev_prompt_tokens_[common_prefix] == prompt_toks[common_prefix]) {
          common_prefix++;
        }
      }

      if (common_prefix > 0 && common_prefix == (int)prompt_toks.size()) {
        // Exact match: trim generated tokens + last prompt token, re-decode 1 token for logits.
        n_cached_tokens = common_prefix - 1;
        if (!llama_memory_seq_rm(llama_get_memory(ctx_), 0, common_prefix - 1, -1)) {
          // SWA model: can't trim tail. Full re-decode.
          n_cached_tokens = 0;
          goto full_prefill;
        }
        cur_pos_ = common_prefix - 1;
        common_batch_clear(batch_);
        common_batch_add(batch_, prompt_toks.back(), common_prefix - 1, {0}, true);
        if (llama_decode(ctx_, batch_) != 0) {
          goto full_prefill;
        }
        cur_pos_ = common_prefix;
        __android_log_print(ANDROID_LOG_INFO, "OmniInferJni",
            "KV cache reuse: %d/%d tokens cached (exact, 1 re-decoded)",
            common_prefix - 1, (int)prompt_toks.size());
      } else if (common_prefix > 0) {
        // Partial prefix match: remove old entries after common prefix, decode suffix.
        n_cached_tokens = common_prefix;
        bool trimmed = llama_memory_seq_rm(llama_get_memory(ctx_), 0, common_prefix, -1);
        if (!trimmed) {
          // SWA/hybrid model: can't trim tail. Full clear + re-decode.
          n_cached_tokens = 0;
          llama_memory_clear(llama_get_memory(ctx_), false);
          cur_pos_ = 0;
          if (decode_batched(prompt_toks, 0, true) != 0) goto full_prefill;
          cur_pos_ = (int)prompt_toks.size();
          __android_log_print(ANDROID_LOG_INFO, "OmniInferJni",
              "KV cache: seq_rm failed (SWA model), full re-decode %d tokens",
              (int)prompt_toks.size());
        } else {
          // seq_rm succeeded: decode only suffix.
          cur_pos_ = common_prefix;
          std::vector<llama_token> suffix(prompt_toks.begin() + common_prefix, prompt_toks.end());
          if (decode_batched(suffix, common_prefix, true) != 0) {
            n_cached_tokens = 0;
            goto full_prefill;
          }
          cur_pos_ = (int)prompt_toks.size();
          __android_log_print(ANDROID_LOG_INFO, "OmniInferJni",
              "KV cache reuse: %d/%d tokens cached, %d new",
              common_prefix, (int)prompt_toks.size(),
              (int)prompt_toks.size() - common_prefix);
        }
      } else {
full_prefill:
        // No cache match (or fallback from failed seq_rm/decode): full clear + full prefill.
        cur_pos_ = 0;
        llama_memory_clear(llama_get_memory(ctx_), false);
        if (decode_batched(prompt_toks, 0, true) != 0) return "";
        cur_pos_ = (int)prompt_toks.size();
      }

      // Save tokens for next request's prefix matching.
      prev_prompt_tokens_ = std::move(prompt_toks);
      has_cache_ = true;
    }

    auto t_prefill_end = std::chrono::steady_clock::now();
    int64_t prefill_us = std::chrono::duration_cast<std::chrono::microseconds>(t_prefill_end - t_prefill_start).count();

    // Generate tokens.
    common_sampler_reset(sampler_);
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    std::string full_response;
    int n_reasoning_tokens = 0;
    int eff_max_tokens = max_tokens > 0 ? max_tokens : (n_ctx_ - n_prompt_tokens - 4);
    bool counting_reasoning = thinking_enabled && params.supports_thinking;
    std::string utf8_buf;

    // Set up thinking tag normalization.
    // All non-standard thinking tags (e.g. Gemma 4's <|channel>thought) are
    // normalized to <think>/<​/think> so Kotlin only handles one format.
    std::optional<thinking_tags::Normalizer> think_norm;

    if (thinking_enabled && params.supports_thinking) {
      auto* non_std = thinking_tags::find_non_standard(params.thinking_start_tag);
      if (non_std) {
        think_norm.emplace(*non_std);
        if (on_token) on_token("<think>\n");
      } else if (!params.thinking_start_tag.empty()) {
        if (on_token) on_token(params.thinking_start_tag);
      }
    }

    // Emit helper: passes through normalizer when active, otherwise direct.
    auto emit = [&](const std::string& text) -> bool {
      std::string out = think_norm ? think_norm->process(text) : text;
      if (!out.empty()) {
        full_response += out;
        if (on_token && !on_token(out)) return false;
      }
      return true;
    };

    auto t_decode_start = std::chrono::steady_clock::now();

    std::vector<llama_token> generated_toks;
    int n_generated = 0;
    while (!cancelled.load()) {
      if (n_generated >= eff_max_tokens) break;
      if (cur_pos_ >= n_ctx_ - 4) shift_context();

      llama_token tok = common_sampler_sample(sampler_, ctx_, -1);
      common_sampler_accept(sampler_, tok, true);

      if (llama_vocab_is_eog(vocab, tok)) {
        if (!utf8_buf.empty()) {
          if (!emit(utf8_buf)) { cancelled.store(true); }
          utf8_buf.clear();
        }
        break;
      }

      n_generated++;
      generated_toks.push_back(tok);
      if (counting_reasoning) n_reasoning_tokens++;

      common_batch_clear(batch_);
      common_batch_add(batch_, tok, cur_pos_, {0}, true);
      if (llama_decode(ctx_, batch_) != 0) break;
      cur_pos_++;

      auto piece = common_token_to_piece(ctx_, tok);
      utf8_buf += piece;

      if (is_valid_utf8(utf8_buf.c_str())) {
        if (!emit(utf8_buf)) { cancelled.store(true); break; }
        utf8_buf.clear();
        // Stop counting reasoning tokens once </think> appears in output.
        if (counting_reasoning && full_response.find("</think>") != std::string::npos) {
          counting_reasoning = false;
        }
      }
    }

    // Hard cancel (client disconnect): invalidate cache for clean state.
    if (cancelled.load() && !graceful_stop.load()) {
      cur_pos_ = 0;
      llama_memory_clear(llama_get_memory(ctx_), false);
      has_cache_ = false;
      prev_prompt_tokens_.clear();
      prev_eval_prompt_.clear();
      prev_eval_n_tokens_ = 0;
    }

    // Append generated tokens to tracking for next request's prefix matching.
    // This matches llama-server's slot behavior: input + generated tokens are
    // stored together, so the next turn can prefix-match across the boundary
    // (e.g. assistant response re-encoded through template still shares most
    // tokens with the raw generation). Skipped when hard cancel cleared above.
    if (!generated_toks.empty() && has_cache_) {
      prev_prompt_tokens_.insert(prev_prompt_tokens_.end(),
          generated_toks.begin(), generated_toks.end());
    }

    // Flush thinking normalizer buffer.
    if (think_norm) {
      auto remaining = think_norm->flush();
      if (!remaining.empty()) {
        full_response += remaining;
        if (on_token) on_token(remaining);
      }
    }

    // Strip template-specific stop sequences from output.
    for (const auto& stop : params.additional_stops) {
      auto pos = full_response.find(stop);
      if (pos != std::string::npos) {
        full_response = full_response.substr(0, pos);
      }
    }

    // If tools were provided, parse output for tool calls.
    // Wrapped in try-catch: llama.cpp's PEG parser throws std::runtime_error
    // on certain multi-tool-call outputs that don't match its grammar.
    if (has_tools && parser_params.format != COMMON_CHAT_FORMAT_CONTENT_ONLY) {
      try {
        common_chat_msg parsed = common_chat_parse(full_response, false, parser_params);
        if (!parsed.tool_calls.empty()) {
          full_response = format_tool_calls_json(parsed);
        }
      } catch (const std::exception& e) {
        __android_log_print(ANDROID_LOG_WARN, "OmniInferJni",
            "PEG tool call parse failed, trying fallback parser: %s", e.what());
        std::string fallback = tool_parser::parse_tool_calls(full_response);
        if (!fallback.empty()) {
          full_response = fallback;
        } else {
          __android_log_print(ANDROID_LOG_WARN, "OmniInferJni",
              "Both PEG and fallback parser failed. Raw output (first 200): %.200s",
              full_response.c_str());
        }
      }
    }

    auto t_decode_end = std::chrono::steady_clock::now();
    int64_t decode_us = std::chrono::duration_cast<std::chrono::microseconds>(t_decode_end - t_decode_start).count();
    int n_prompt = n_prompt_tokens;
    last_metrics_ = {n_prompt, n_generated, prefill_us, decode_us,
                     n_reasoning_tokens, n_image_tokens, n_cached_tokens};
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
    prev_prompt_tokens_.clear();
    prev_eval_prompt_.clear();
    prev_eval_n_tokens_ = 0;
    has_cache_ = false;
  }

  InferenceMetrics get_metrics() override { return last_metrics_; }
  int n_threads() const override { return n_threads_; }
  const char* name() const override { return "llama.cpp"; }

private:
  void release() {
    if (mtmd_ctx_) { mtmd_free(mtmd_ctx_); mtmd_ctx_ = nullptr; }
    if (sampler_) { common_sampler_free(sampler_); sampler_ = nullptr; }
    chat_templates_.reset();
    llama_batch_free(batch_); batch_ = {};
    if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
    if (model_) { llama_model_free(model_); model_ = nullptr; }
  }

  // Scan model directory for mmproj*.gguf file.
  static std::string find_mmproj(const std::string& model_path) {
    auto slash = model_path.find_last_of("/\\");
    if (slash == std::string::npos) return "";
    std::string dir = model_path.substr(0, slash);
    DIR* dp = opendir(dir.c_str());
    if (!dp) return "";
    std::string result;
    while (auto* entry = readdir(dp)) {
      std::string name = entry->d_name;
      if (name.find("mmproj") != std::string::npos &&
          name.size() > 5 && name.substr(name.size() - 5) == ".gguf") {
        result = dir + "/" + name;
        break;
      }
    }
    closedir(dp);
    return result;
  }

  // Parse messages JSON array into common_chat_msg vector.
  // Input: [{"role":"system","content":"..."},{"role":"user","content":"..."},...]
  static std::vector<common_chat_msg> parse_messages(const std::string& json) {
    std::vector<common_chat_msg> msgs;
    size_t pos = 0;
    while (pos < json.size()) {
      // Find next object.
      size_t obj_start = json.find('{', pos);
      if (obj_start == std::string::npos) break;
      std::string obj = extract_json_object(json, obj_start);
      if (obj == "{}") { pos = obj_start + 1; continue; }

      common_chat_msg msg;
      size_t role_key = obj.find("\"role\"");
      if (role_key != std::string::npos) msg.role = extract_json_string_value(obj, role_key);
      size_t content_key = obj.find("\"content\"");
      if (content_key != std::string::npos) msg.content = extract_json_string_value(obj, content_key);

      if (!msg.role.empty()) msgs.push_back(std::move(msg));
      pos = obj_start + obj.size();
    }
    return msgs;
  }

  // Parse OpenAI tools JSON array into common_chat_tool vector.
  // Input: [{"type":"function","function":{"name":"...","description":"...","parameters":{...}}}]
  static std::vector<common_chat_tool> parse_tools(const std::string& json) {
    std::vector<common_chat_tool> tools;
    // Minimal JSON array parsing — extract function objects.
    // Find each {"type":"function","function":{...}} block.
    size_t pos = 0;
    while ((pos = json.find("\"function\"", pos)) != std::string::npos) {
      // Find the nested object after "function":
      size_t colon = json.find(':', pos + 10);
      if (colon == std::string::npos) break;
      size_t obj_start = json.find('{', colon);
      if (obj_start == std::string::npos) break;

      // Extract name
      std::string name, description, parameters;
      size_t name_key = json.find("\"name\"", obj_start);
      if (name_key != std::string::npos) {
        name = extract_json_string_value(json, name_key);
      }
      size_t desc_key = json.find("\"description\"", obj_start);
      if (desc_key != std::string::npos) {
        description = extract_json_string_value(json, desc_key);
      }
      // Extract parameters as raw JSON object
      size_t params_key = json.find("\"parameters\"", obj_start);
      if (params_key != std::string::npos) {
        size_t params_colon = json.find(':', params_key + 12);
        if (params_colon != std::string::npos) {
          size_t params_obj = json.find('{', params_colon);
          if (params_obj != std::string::npos) {
            parameters = extract_json_object(json, params_obj);
          }
        }
      }

      if (!name.empty()) {
        common_chat_tool tool;
        tool.name = name;
        tool.description = description;
        tool.parameters = parameters;
        tools.push_back(std::move(tool));
      }
      pos = colon + 1;
    }
    return tools;
  }

  // Extract a JSON string value after a key position.
  static std::string extract_json_string_value(const std::string& json, size_t key_pos) {
    size_t colon = json.find(':', key_pos);
    if (colon == std::string::npos) return "";
    size_t quote1 = json.find('"', colon + 1);
    if (quote1 == std::string::npos) return "";
    size_t quote2 = quote1 + 1;
    while (quote2 < json.size()) {
      if (json[quote2] == '"' && json[quote2 - 1] != '\\') break;
      quote2++;
    }
    return json.substr(quote1 + 1, quote2 - quote1 - 1);
  }

  // Extract a balanced JSON object starting at pos.
  static std::string extract_json_object(const std::string& json, size_t pos) {
    if (pos >= json.size() || json[pos] != '{') return "{}";
    int depth = 0;
    bool in_string = false;
    size_t start = pos;
    for (size_t i = pos; i < json.size(); i++) {
      char c = json[i];
      if (in_string) {
        if (c == '"' && json[i - 1] != '\\') in_string = false;
        continue;
      }
      if (c == '"') { in_string = true; continue; }
      if (c == '{') depth++;
      else if (c == '}') { depth--; if (depth == 0) return json.substr(start, i - start + 1); }
    }
    return "{}";
  }

  // Strip trailing generation prompt from formatted chat template output.
  static std::string strip_generation_prompt(const std::string& formatted) {
    size_t cut = std::string::npos;
    for (const char* marker : {"<|im_start|>assistant", "<start_of_turn>model", "<|turn>model"}) {
      auto pos = formatted.rfind(marker);
      if (pos != std::string::npos && (cut == std::string::npos || pos > cut)) cut = pos;
    }
    return (cut != std::string::npos) ? formatted.substr(0, cut) : formatted;
  }

  // Format parsed tool calls as JSON for the HTTP layer.
  // Returns: {"tool_calls":[{"id":"call_xxx","type":"function","function":{"name":"...","arguments":"..."}}],"content":"..."}
  static std::string format_tool_calls_json(const common_chat_msg& msg) {
    std::ostringstream ss;
    ss << "{\"tool_calls\":[";
    for (size_t i = 0; i < msg.tool_calls.size(); i++) {
      if (i > 0) ss << ",";
      const auto& tc = msg.tool_calls[i];
      std::string id = tc.id.empty() ? ("call_" + std::to_string(i)) : tc.id;
      ss << "{\"id\":\"" << id << "\","
         << "\"type\":\"function\","
         << "\"function\":{\"name\":\"" << tc.name << "\","
         << "\"arguments\":" << tc.arguments << "}}";
    }
    ss << "]";
    if (!msg.content.empty()) {
      ss << ",\"content\":\"";
      // Escape content for JSON
      for (char c : msg.content) {
        if (c == '"') ss << "\\\"";
        else if (c == '\\') ss << "\\\\";
        else if (c == '\n') ss << "\\n";
        else if (c == '\r') ss << "\\r";
        else if (c == '\t') ss << "\\t";
        else ss << c;
      }
      ss << "\"";
    }
    ss << "}";
    return ss.str();
  }

  void shift_context() {
    int n_discard = cur_pos_ / 2;
    if (n_discard <= 0) return;
    llama_memory_seq_rm(llama_get_memory(ctx_), 0, 0, n_discard);
    llama_memory_seq_add(llama_get_memory(ctx_), 0, n_discard, cur_pos_, -n_discard);
    cur_pos_ -= n_discard;
    // Invalidate cache — positions shifted, prefix matching no longer valid.
    prev_prompt_tokens_.clear();
    prev_eval_prompt_.clear();
    prev_eval_n_tokens_ = 0;
    has_cache_ = false;
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
  int n_threads_ = 4;
  mtmd_context* mtmd_ctx_ = nullptr;
  std::string media_marker_;
  InferenceMetrics last_metrics_;
  // KV cache prefix reuse state.
  std::vector<llama_token> prev_prompt_tokens_;  // text-only token comparison
  std::string prev_eval_prompt_;                 // multimodal string prefix comparison
  int prev_eval_n_tokens_ = 0;                   // total tokens from last multimodal eval
  bool has_cache_ = false;
};

std::once_flag LlamaCppBackend::s_backend_init;

}  // namespace omniinfer
