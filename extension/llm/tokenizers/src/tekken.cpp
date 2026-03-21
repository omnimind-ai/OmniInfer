/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @lint-ignore-every LICENSELINT
 */

#include <pytorch/tokenizers/base64.h>
#include <pytorch/tokenizers/log.h>
#include <pytorch/tokenizers/tekken.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

namespace tokenizers {

using json = nlohmann::json;
using namespace detail;

Tekken::Tekken() {}

Error Tekken::_encode(
    const std::string& input,
    std::vector<uint64_t>& ret,
    uint64_t& last_piece_token_len) const {
  std::string piece;
  assert(_regex);
  for (const auto& match : _regex->find_all(input)) {
    std::string matched_text =
        input.substr(match.start, match.end - match.start);

    // Try to find the whole word in token map first
    const auto result = token_map_->tryGetInteger(matched_text);
    if (result) {
      last_piece_token_len = 1;
      ret.push_back(*result);
      continue;
    }

    // Fall back to BPE encoding
    auto tokens_result = byte_pair_encode_(matched_text, *token_map_);
    if (!tokens_result.ok()) {
      return tokens_result.error();
    }
    auto tokens = std::move(*tokens_result);
    last_piece_token_len = tokens.size();
    ret.insert(ret.end(), tokens.begin(), tokens.end());
  }
  return Error::Ok;
}

void Tekken::_decode(const std::string& input, std::string& ret) const {
  ret += input;
}

std::string Tekken::_get_default_tekken_pattern() {
  // The Tekken pattern from the JSON - note this needs to work with RE2
  // We use the simplified version that works with RE2 (no negative lookahead)
  return R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+)";
}

std::vector<Tekken::SpecialTokenInfo> Tekken::_get_default_special_tokens() {
  return {
      {0, "<unk>", true},
      {1, "<s>", true},
      {2, "</s>", true},
      {3, "[INST]", true},
      {4, "[/INST]", true},
      {5, "[AVAILABLE_TOOLS]", true},
      {6, "[/AVAILABLE_TOOLS]", true},
      {7, "[TOOL_RESULTS]", true},
      {8, "[/TOOL_RESULTS]", true},
      {9, "[TOOL_CALLS]", true},
      {10, "[IMG]", true},
      {11, "<pad>", true},
      {12, "[IMG_BREAK]", true},
      {13, "[IMG_END]", true},
      {14, "[PREFIX]", true},
      {15, "[MIDDLE]", true},
      {16, "[SUFFIX]", true},
      {17, "[SYSTEM_PROMPT]", true},
      {18, "[/SYSTEM_PROMPT]", true},
      {19, "[TOOL_CONTENT]", true}};
}

Error Tekken::load(const std::string& tokenizer_path) {
  TK_LOG(Info, "Loading Tekken tokenizer from: %s", tokenizer_path.c_str());

  // Load and parse JSON file
  std::ifstream file(tokenizer_path);
  TK_CHECK_OR_RETURN_ERROR(
      file.is_open(),
      LoadFailure,
      "Failed to open tekken file: %s",
      tokenizer_path.c_str());

  json parsed_json;
  try {
    parsed_json = json::parse(file);
  } catch (const json::exception& e) {
    TK_CHECK_OR_RETURN_ERROR(
        false, ParseFailure, "Error parsing tekken JSON: %s", e.what());
  }

  // Parse configuration
  auto config_result = _parse_config(parsed_json["config"]);
  if (!config_result.ok()) {
    return config_result.error();
  }
  auto config = std::move(*config_result);
  _pattern = config.pattern;
  _num_special_tokens = config.default_num_special_tokens;
  _version = config.version;

  TK_LOG(
      Info,
      "Tekken version: %s, vocab_size: %zu, special_tokens: %zu",
      _version.c_str(),
      config.default_vocab_size,
      _num_special_tokens);

  // Parse special tokens if present
  std::vector<SpecialTokenInfo> special_tokens;
  if (parsed_json.contains("special_tokens") &&
      !parsed_json["special_tokens"].is_null()) {
    TK_LOG(Info, "Loading special tokens from JSON");
    for (const auto& token : parsed_json["special_tokens"]) {
      special_tokens.push_back(
          {token["rank"], token["token_str"], token.value("is_control", true)});
    }
  } else {
    // Use default special tokens for older versions
    TK_LOG(Info, "Using default special tokens");
    special_tokens = _get_default_special_tokens();
  }

  // Fill special tokens up to num_special_tokens
  special_tokens =
      _initialize_special_tokens(special_tokens, _num_special_tokens);

  // Load vocabulary (exclude special tokens from vocab size)
  size_t vocab_size = config.default_vocab_size - _num_special_tokens;
  TK_LOG(Info, "Loading %zu vocabulary tokens", vocab_size);
  auto token_map_result =
      _load_vocab_from_json(parsed_json["vocab"], vocab_size);
  if (!token_map_result.ok()) {
    return token_map_result.error();
  }
  token_map_.emplace(std::move(*token_map_result));

  // Set up special tokens
  std::vector<std::string> special_token_strings;
  std::vector<std::pair<std::string, uint64_t>> special_token_pairs;

  for (const auto& st : special_tokens) {
    special_token_strings.push_back(st.token_str);
    // Special tokens use their rank directly (0-999) to match mistral-common
    special_token_pairs.emplace_back(st.token_str, st.rank);
  }

  auto special_token_map_result = TokenMap::create(special_token_pairs);
  if (!special_token_map_result.ok()) {
    return special_token_map_result.error();
  }
  special_token_map_.emplace(std::move(*special_token_map_result));

  // Initialize regex with the pattern from config
  auto regex_result = create_regex(_pattern);
  if (!regex_result.ok()) {
    return regex_result.error();
  }
  _regex = std::move(*regex_result);
  auto special_token_regex_result =
      build_special_token_regex(*special_token_map_);
  if (!special_token_regex_result.ok()) {
    return special_token_regex_result.error();
  }
  special_token_regex_ = std::move(*special_token_regex_result);

  // Set vocab size and special token indices
  vocab_size_ = token_map_->size() + special_token_map_->size();

  // Find BOS and EOS tokens
  auto bos_result = special_token_map_->tryGetInteger("<s>");
  auto eos_result = special_token_map_->tryGetInteger("</s>");

  TK_CHECK_OR_RETURN_ERROR(
      bos_result.has_value(),
      LoadFailure,
      "BOS token <s> not found in special tokens");

  TK_CHECK_OR_RETURN_ERROR(
      eos_result.has_value(),
      LoadFailure,
      "EOS token </s> not found in special tokens");

  bos_tok_ = *bos_result;
  eos_tok_ = *eos_result;

  TK_LOG(
      Info,
      "Tekken tokenizer loaded successfully. Vocab size: %d, BOS: %llu, EOS: %llu",
      (int)vocab_size_,
      (unsigned long long)bos_tok_,
      (unsigned long long)eos_tok_);

  initialized_ = true;
  return Error::Ok;
}

Error Tekken::load_with_special_tokens(
    const std::string& tokenizer_path,
    const std::vector<SpecialTokenInfo>& explicit_special_tokens) {
  // Similar to load() but uses the provided special tokens instead of defaults
  // This would be implemented for advanced use cases
  return load(tokenizer_path); // For now, delegate to standard load
}

Result<Tekken::TekkenConfig> Tekken::_parse_config(const json& j) const {
  TekkenConfig config;

  try {
    config.pattern = j["pattern"];
    config.num_vocab_tokens = j["num_vocab_tokens"];
    config.default_vocab_size = j["default_vocab_size"];
    config.default_num_special_tokens = j["default_num_special_tokens"];
    config.version = j["version"];
  } catch (const json::exception& e) {
    TK_CHECK_OR_RETURN_ERROR(
        false, ParseFailure, "Error parsing config: %s", e.what());
  }

  return config;
}

Result<TokenMap> Tekken::_load_vocab_from_json(
    const json& vocab_json,
    size_t max_vocab) const {
  std::vector<std::pair<std::string, uint64_t>> pairs;

  size_t vocab_limit = std::min(vocab_json.size(), max_vocab);
  TK_LOG(
      Info,
      "Processing %zu vocabulary entries (limit: %zu)",
      vocab_limit,
      max_vocab);

  for (size_t i = 0; i < vocab_limit; ++i) {
    const auto& item = vocab_json[i];

    // Decode base64 token
    auto token_bytes_b64 = item["token_bytes"].get<std::string>();
    auto token_result = base64::decode(token_bytes_b64);

    TK_CHECK_OR_RETURN_ERROR(
        token_result.ok(),
        ParseFailure,
        "Failed to decode base64 token at index %zu: %s",
        i,
        token_bytes_b64.c_str());

    std::string token = token_result.get();
    uint64_t rank = item["rank"];

    // Verify rank matches index
    TK_CHECK_OR_RETURN_ERROR(
        rank == i,
        ParseFailure,
        "Invalid rank %llu at index %zu (expected %zu)",
        (unsigned long long)rank,
        i,
        i);

    // Offset regular vocabulary tokens by num_special_tokens to match
    // mistral-common This reserves IDs 0-999 for special tokens
    pairs.emplace_back(std::move(token), rank + _num_special_tokens);
  }

  TK_LOG(Info, "Built vocabulary with %zu tokens", pairs.size());
  return build_token_map(pairs);
}

std::vector<Tekken::SpecialTokenInfo> Tekken::_initialize_special_tokens(
    const std::vector<SpecialTokenInfo>& defined_tokens,
    size_t num_special_tokens) const {
  std::vector<SpecialTokenInfo> result = defined_tokens;

  // Sort by rank to ensure proper ordering
  std::sort(
      result.begin(),
      result.end(),
      [](const SpecialTokenInfo& a, const SpecialTokenInfo& b) {
        return a.rank < b.rank;
      });

  // Fill remaining slots with placeholder tokens
  for (size_t i = defined_tokens.size(); i < num_special_tokens; ++i) {
    result.push_back({i, "<SPECIAL_" + std::to_string(i) + ">", true});
  }

  TK_LOG(
      Info,
      "Initialized %zu special tokens (%zu defined, %zu placeholders)",
      result.size(),
      defined_tokens.size(),
      num_special_tokens - defined_tokens.size());

  return result;
}

} // namespace tokenizers
