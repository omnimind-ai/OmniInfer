/*
 * Hardcoded chat template engine for ExecuTorch model families.
 * Phase 1: Qwen3 as primary target. Extend for other DecoderModelVersions as needed.
 *
 * Input:  messages (role/content pairs) + thinking_enabled + model version string
 * Output: formatted prompt string ready for ET Runner::generate()
 */

#pragma once

#include <string>
#include <utility>
#include <vector>

namespace omniinfer {
namespace et_template {

struct Message {
  std::string role;    // "system", "user", "assistant"
  std::string content;
};

// Parse a JSON array of messages into Message structs.
// Minimal parser — handles the subset produced by OmniInfer's JNI layer.
inline std::vector<Message> parse_messages_json(const std::string& json) {
  std::vector<Message> msgs;
  if (json.empty() || json[0] != '[') return msgs;

  // Walk through the JSON array looking for {"role":"...","content":"..."}
  size_t pos = 0;
  while (pos < json.size()) {
    auto role_key = json.find("\"role\"", pos);
    if (role_key == std::string::npos) break;

    // Find the role value
    auto role_colon = json.find(':', role_key + 6);
    if (role_colon == std::string::npos) break;
    auto role_quote1 = json.find('"', role_colon + 1);
    if (role_quote1 == std::string::npos) break;
    auto role_quote2 = json.find('"', role_quote1 + 1);
    if (role_quote2 == std::string::npos) break;
    std::string role = json.substr(role_quote1 + 1, role_quote2 - role_quote1 - 1);

    // Find the content value — may contain escaped quotes
    auto content_key = json.find("\"content\"", role_quote2);
    if (content_key == std::string::npos) break;
    auto content_colon = json.find(':', content_key + 9);
    if (content_colon == std::string::npos) break;

    // Skip whitespace after colon
    size_t val_start = content_colon + 1;
    while (val_start < json.size() && (json[val_start] == ' ' || json[val_start] == '\t'))
      val_start++;

    std::string content;
    if (val_start < json.size() && json[val_start] == '"') {
      // String value — handle escape sequences
      size_t i = val_start + 1;
      while (i < json.size()) {
        if (json[i] == '\\' && i + 1 < json.size()) {
          char c = json[i + 1];
          if (c == '"') content += '"';
          else if (c == '\\') content += '\\';
          else if (c == 'n') content += '\n';
          else if (c == 't') content += '\t';
          else if (c == 'r') content += '\r';
          else { content += '\\'; content += c; }
          i += 2;
        } else if (json[i] == '"') {
          break;
        } else {
          content += json[i];
          i++;
        }
      }
      pos = i + 1;
    } else if (val_start < json.size() && json.substr(val_start, 4) == "null") {
      content = "";
      pos = val_start + 4;
    } else {
      pos = val_start + 1;
      continue;
    }

    msgs.push_back({std::move(role), std::move(content)});
  }
  return msgs;
}

// Qwen3 chat template (ChatML variant with thinking support)
// Reference: https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/tokenizer_config.json
inline std::string apply_qwen3(const std::vector<Message>& msgs, bool thinking_enabled) {
  std::string result;
  bool has_system = false;

  for (const auto& msg : msgs) {
    if (msg.role == "system") {
      has_system = true;
      result += "<|im_start|>system\n" + msg.content + "<|im_end|>\n";
    } else if (msg.role == "user") {
      result += "<|im_start|>user\n" + msg.content + "<|im_end|>\n";
    } else if (msg.role == "assistant") {
      result += "<|im_start|>assistant\n" + msg.content + "<|im_end|>\n";
    }
  }

  // Add default system prompt if none provided
  if (!has_system) {
    result = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" + result;
  }

  // Generation prompt
  if (thinking_enabled) {
    result += "<|im_start|>assistant\n<think>\n";
  } else {
    result += "<|im_start|>assistant\n";
  }

  return result;
}

// Qwen2.5 chat template (ChatML, no thinking)
inline std::string apply_qwen2_5(const std::vector<Message>& msgs, bool /*thinking_enabled*/) {
  std::string result;
  bool has_system = false;

  for (const auto& msg : msgs) {
    if (msg.role == "system") {
      has_system = true;
      result += "<|im_start|>system\n" + msg.content + "<|im_end|>\n";
    } else if (msg.role == "user") {
      result += "<|im_start|>user\n" + msg.content + "<|im_end|>\n";
    } else if (msg.role == "assistant") {
      result += "<|im_start|>assistant\n" + msg.content + "<|im_end|>\n";
    }
  }

  if (!has_system) {
    result = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" + result;
  }

  result += "<|im_start|>assistant\n";
  return result;
}

// Llama 3 chat template
inline std::string apply_llama3(const std::vector<Message>& msgs, bool /*thinking_enabled*/) {
  std::string result = "<|begin_of_text|>";
  bool has_system = false;

  for (const auto& msg : msgs) {
    if (msg.role == "system") {
      has_system = true;
      result += "<|start_header_id|>system<|end_header_id|>\n\n" + msg.content + "<|eot_id|>";
    } else if (msg.role == "user") {
      result += "<|start_header_id|>user<|end_header_id|>\n\n" + msg.content + "<|eot_id|>";
    } else if (msg.role == "assistant") {
      result += "<|start_header_id|>assistant<|end_header_id|>\n\n" + msg.content + "<|eot_id|>";
    }
  }

  result += "<|start_header_id|>assistant<|end_header_id|>\n\n";
  return result;
}

// Gemma 3 chat template
inline std::string apply_gemma3(const std::vector<Message>& msgs, bool thinking_enabled) {
  std::string result;
  bool has_system = false;

  for (const auto& msg : msgs) {
    if (msg.role == "system") {
      has_system = true;
      // Gemma 3 doesn't have a system role — prepend to first user message
      continue;
    } else if (msg.role == "user") {
      result += "<start_of_turn>user\n";
      if (has_system && result.find("<start_of_turn>user\n") == result.rfind("<start_of_turn>user\n")) {
        // First user message — prepend system content
        for (const auto& m : msgs) {
          if (m.role == "system") {
            result += m.content + "\n\n";
            break;
          }
        }
      }
      result += msg.content + "<end_of_turn>\n";
    } else if (msg.role == "assistant") {
      result += "<start_of_turn>model\n" + msg.content + "<end_of_turn>\n";
    }
  }

  if (thinking_enabled) {
    result += "<start_of_turn>model\n<|channel>thought\n";
  } else {
    result += "<start_of_turn>model\n";
  }

  return result;
}

// Dispatch to the correct template based on model version string
inline std::string apply_chat_template(
    const std::string& decoder_model_version,
    const std::vector<Message>& msgs,
    bool thinking_enabled) {

  if (decoder_model_version == "qwen3") {
    return apply_qwen3(msgs, thinking_enabled);
  } else if (decoder_model_version == "qwen2_5" || decoder_model_version == "qwen2.5") {
    return apply_qwen2_5(msgs, thinking_enabled);
  } else if (decoder_model_version == "llama3") {
    return apply_llama3(msgs, thinking_enabled);
  } else if (decoder_model_version == "gemma3" || decoder_model_version == "gemma") {
    return apply_gemma3(msgs, thinking_enabled);
  }

  // Fallback: Qwen3/ChatML style (most common for ET exported models)
  return apply_qwen3(msgs, thinking_enabled);
}

// Convenience: apply template from messages_json string
inline std::string format_prompt(
    const std::string& decoder_model_version,
    const std::string& system_prompt,
    const std::string& user_prompt,
    const std::string& messages_json,
    bool thinking_enabled) {

  std::vector<Message> msgs;

  if (!messages_json.empty()) {
    msgs = parse_messages_json(messages_json);
  } else {
    // Legacy path: system_prompt + user_prompt
    if (!system_prompt.empty()) {
      msgs.push_back({"system", system_prompt});
    }
    if (!user_prompt.empty()) {
      msgs.push_back({"user", user_prompt});
    }
  }

  if (msgs.empty()) {
    msgs.push_back({"user", "Hello"});
  }

  return apply_chat_template(decoder_model_version, msgs, thinking_enabled);
}

} // namespace et_template
} // namespace omniinfer
