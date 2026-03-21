/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <pytorch/tokenizers/hf_tokenizer.h>

// Standard
#include <algorithm>
#include <cinttypes>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Third Party
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace tokenizers {

namespace {
// Helper to extract token string from either string or object format
std::string extract_token_string(const json& token_json) {
  if (token_json.is_string()) {
    return token_json.get<std::string>();
  } else if (token_json.is_object() && token_json.contains("content")) {
    return token_json["content"].get<std::string>();
  }
  return "";
};
} // namespace

// -------------------------public method start-------------------------------

Error HFTokenizer::load(const std::string& path) {
  std::string model_json = path;
  std::string model_config_json = "";
  std::string special_tokens_map_json = "";

  if (fs::is_directory(path)) {
    const fs::path root(path);
    model_json = (root / "tokenizer.json").string();
    if (!fs::exists(model_json)) {
      TK_LOG(Info, "no tokenizer.json found in %s", path.c_str());
      return Error::LoadFailure;
    }
    const auto model_config_json_path = root / "tokenizer_config.json";
    if (fs::exists(model_config_json_path)) {
      model_config_json = model_config_json_path.string();
    }
    const auto special_tokens_map_json_path = root / "special_tokens_map.json";
    if (fs::exists(special_tokens_map_json_path)) {
      special_tokens_map_json = special_tokens_map_json_path.string();
    }
  }

  std::ifstream file(model_json);
  if (!file) {
    TK_LOG(Info, "failed to open encoder file: %s", path.c_str());
    return Error::LoadFailure;
  }
  std::string contents(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  json parsed_json;
  try {
    parsed_json = json::parse(contents);
  } catch (const std::exception& e) {
    TK_LOG(Error, "Error parsing json file: %s", e.what());
    return Error::LoadFailure;
  }

  TK_CHECK_OK_OR_RETURN_ERROR(parse_special_tokens(parsed_json));
  TK_CHECK_OK_OR_RETURN_ERROR(parse_tokens(parsed_json));

  vocab_size_ = token_map_->size() + special_token_map_->size();

  TK_CHECK_OK_OR_RETURN_ERROR(setup_normalizer(parsed_json));
  TK_CHECK_OK_OR_RETURN_ERROR(setup_pretokenizer(parsed_json));
  TK_CHECK_OK_OR_RETURN_ERROR(setup_postprocessor(parsed_json));
  TK_CHECK_OK_OR_RETURN_ERROR(setup_decoder(parsed_json));

  TK_CHECK_OK_OR_RETURN_ERROR(parse_merges(parsed_json));

  TK_CHECK_OK_OR_RETURN_ERROR(setup_special_token_ids(
      path, parsed_json, model_config_json, special_tokens_map_json));

  initialized_ = true;
  return Error::Ok;
}

Result<std::vector<uint64_t>>
HFTokenizer::encode(const std::string& input, int8_t bos, int8_t eos) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }

  auto encode_result = encode_with_special_token_(input, *special_token_map_);
  if (!encode_result.ok()) {
    return encode_result.error();
  }
  std::vector<uint64_t> tokens = std::move((*encode_result).first);

  bool add_special = (bos > 0 || eos > 0);
  if (_postprocessor) {
    tokens = _postprocessor->process(tokens, add_special);
  }

  return tokens;
}

Result<std::string> HFTokenizer::decode(
    const std::vector<uint64_t>& tokens,
    bool skip_special_tokens) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  std::vector<std::string> pieces;
  for (uint64_t token : tokens) {
    std::string_view token_bytes;
    auto regular_token_result = token_map_->tryGetString(token);
    if (regular_token_result) { // Found in regular tokens
      token_bytes = *regular_token_result;
    } else { // Not a regular token, check if it's a special token
      auto special_token_result = special_token_map_->tryGetString(token);
      if (special_token_result) { // It's a special token
        if (skip_special_tokens) {
          continue;
        }
        token_bytes = *special_token_result; // Don't skip, use its string
      } else { // Unknown token
        TK_LOG(Error, "unknown token: %" PRIu64 "\n", token);
        return Error::DecodeFailure;
      }
    }
    pieces.push_back(std::string(token_bytes));
  }

  auto decoded_pieces = _decode(pieces);

  std::string result_str;
  for (const auto& p : decoded_pieces) {
    result_str += p;
  }

  return result_str;
}

// -------------------------private method start--------------------------------

Error HFTokenizer::_encode(
    const std::string& input,
    std::vector<uint64_t>& ret,
    uint64_t& last_piece_token_len) const {
  std::string normalized_input = input;
  if (_normalizer) {
    normalized_input = _normalizer->normalize(input);
  }

  std::vector<std::string> pieces;
  if (_pretokenizer) {
    pieces = _pretokenizer->pre_tokenize(normalized_input);
  } else {
    pieces.push_back(normalized_input);
  }

  for (const auto& piece : pieces) {
    const auto result = token_map_->tryGetInteger(piece);
    if (result) {
      ret.push_back(*result);
      last_piece_token_len = 1;
      continue;
    }
    auto tokens_result = byte_pair_encode_(piece, *token_map_);
    if (!tokens_result.ok()) {
      return tokens_result.error();
    }
    auto piece_tokens = std::move(*tokens_result);
    ret.insert(ret.end(), piece_tokens.begin(), piece_tokens.end());
    last_piece_token_len = piece_tokens.size();
  }
  return Error::Ok;
}

void HFTokenizer::_decode(const std::string& input, std::string& ret) const {
  if (_decoder) {
    auto result = _decoder->decode({input});
    for (auto& piece : result) {
      ret += piece;
    }
  } else {
    ret += input;
  }
}

std::vector<std::string> HFTokenizer::_decode(
    const std::vector<std::string>& pieces) const {
  if (_decoder) {
    return _decoder->decode(pieces);
  }
  return pieces;
}

Result<std::vector<uint64_t>> HFTokenizer::byte_pair_encode_(
    const std::string& piece,
    const detail::TokenMap& token_map) const {
  if (piece.size() == 1) {
    const auto result = token_map.tryGetInteger(piece);
    if (result) {
      return std::vector<uint64_t>(1, *result);
    }
    if (byte_fallback_) {
      char hex[7];
      snprintf(
          hex, sizeof(hex), "<0x%02X>", static_cast<unsigned char>(piece[0]));
      const auto byte_result = token_map.tryGetInteger(std::string(hex));
      if (byte_result) {
        return std::vector<uint64_t>(1, *byte_result);
      }
      if (unk_token_is_configured_) {
        return std::vector<uint64_t>(1, unk_tok_);
      }
      return Error::EncodeFailure;
    } else {
      if (unk_token_is_configured_) {
        return std::vector<uint64_t>(1, unk_tok_);
      }
      return Error::EncodeFailure;
    }
  }

  const detail::TokenMap& merge_ranks =
      merge_ranks_ ? *merge_ranks_ : token_map;

  return _byte_pair_merge(
      piece,
      merge_ranks,
      [this, &piece, &token_map](uint64_t start, uint64_t stop) {
        std::string key = piece.substr(start, stop - start);
        const auto result = token_map.tryGetInteger(key);
        if (result) {
          return *result;
        }
        if (byte_fallback_) {
          return UINT64_MAX;
        }
        if (unk_token_is_configured_) {
          return unk_tok_;
        }
        return UINT64_MAX - 1;
      });
}

std::vector<uint64_t> HFTokenizer::_byte_pair_merge(
    const std::string& piece,
    const detail::TokenMap& ranks,
    std::function<uint64_t(uint64_t, uint64_t)> func) const {
  HFWord word;
  size_t i = 0;
  while (i < piece.size()) {
    size_t char_start = i;
    size_t char_len = 1;
    unsigned char byte = static_cast<unsigned char>(piece[i]);
    if ((byte & 0x80) == 0)
      char_len = 1;
    else if ((byte & 0xE0) == 0xC0)
      char_len = 2;
    else if ((byte & 0xF0) == 0xE0)
      char_len = 3;
    else if ((byte & 0xF8) == 0xF0)
      char_len = 4;
    if (char_start + char_len > piece.size())
      char_len = piece.size() - char_start;

    uint64_t token_id = func(char_start, char_start + char_len);
    if (token_id == UINT64_MAX) { // This is the sentinel for byte_fallback
      for (size_t j = 0; j < char_len; ++j) {
        char hex[7];
        snprintf(
            hex,
            sizeof(hex),
            "<0x%02X>",
            static_cast<unsigned char>(piece[char_start + j]));
        const auto byte_result = token_map_->tryGetInteger(std::string(hex));
        if (byte_result) {
          word.add(*byte_result, 1);
        } else if (unk_token_is_configured_) {
          word.add(unk_tok_, 1);
        } else {
          return {}; // Unhandled byte fallback
        }
      }
    } else if (token_id == (UINT64_MAX - 1)) { // This is the sentinel for a
                                               // generic err
      return {}; // Return empty on error
    } else { // If it's not a sentinel, it's a valid token_id (could be 0, for
             // [UNK] oranother token)
      word.add(token_id, char_len); // Add any valid token_id
    }
    i += char_len;
  }

  if (merge_ranks_ && token_map_) {
    word.merge_all(*merge_ranks_, *token_map_);
  }
  return word.tokens;
}

Error HFTokenizer::parse_special_tokens(const json& parsed_json) {
  try {
    const auto& special_tokens = parsed_json.at("added_tokens");
    auto special_token_map_result = detail::build_token_map(
        special_tokens,
        [](const auto& it) -> std::string { return it.at("content"); },
        [](const auto& it) -> std::uint64_t { return it.at("id"); });
    if (!special_token_map_result.ok()) {
      return special_token_map_result.error();
    }
    auto special_token_map = std::move(*special_token_map_result);

    auto special_token_regex_result =
        detail::build_special_token_regex(special_token_map);
    if (!special_token_regex_result.ok()) {
      return special_token_regex_result.error();
    }
    special_token_regex_ = std::move(*special_token_regex_result);
    special_token_map_.emplace(std::move(special_token_map));
  } catch (const std::exception& e) {
    TK_LOG(Info, "Could not parse special tokens: %s", e.what());
    return Error::LoadFailure;
  }
  return Error::Ok;
}

Error HFTokenizer::parse_tokens(const json& parsed_json) {
  try {
    std::vector<std::pair<std::string, std::uint64_t>> token_pairs;
    const auto& vocab = parsed_json.at("/model/vocab"_json_pointer);
    for (const auto& entry : vocab.items()) {
      const std::string token = entry.key();
      const uint64_t token_id = entry.value();
      if (!special_token_map_->tryGetString(token_id)) {
        token_pairs.emplace_back(token, token_id);
      }
    }

    auto token_map_result = detail::build_token_map(std::move(token_pairs));
    if (!token_map_result.ok()) {
      return token_map_result.error();
    }
    token_map_.emplace(std::move(*token_map_result));
  } catch (const std::exception& e) {
    TK_LOG(Info, "Could not parse tokens: %s", e.what());
    return Error::LoadFailure;
  }
  return Error::Ok;
}

Error HFTokenizer::setup_normalizer(const json& parsed_json) {
  try {
    if (parsed_json.contains("normalizer") &&
        !parsed_json.at("normalizer").is_null()) {
      const auto& normalizer_json = parsed_json.at("normalizer");
      _normalizer = NormalizerConfig().parse_json(normalizer_json).create();
    }
  } catch (const std::exception& e) {
    TK_LOG(Error, "Failed to setup normalizer: %s", e.what());
    return Error::LoadFailure;
  }
  return Error::Ok;
}

Error HFTokenizer::setup_pretokenizer(const json& parsed_json) {
  try {
    if (parsed_json.contains("pre_tokenizer") &&
        !parsed_json.at("pre_tokenizer").is_null()) {
      const auto& pretokenizer_json = parsed_json.at("pre_tokenizer");
      _pretokenizer =
          PreTokenizerConfig().parse_json(pretokenizer_json).create();
    }
  } catch (const std::exception& e) {
    TK_LOG(Error, "Failed to setup pretokenizer: %s", e.what());
    return Error::LoadFailure;
  }
  return Error::Ok;
}

Error HFTokenizer::setup_postprocessor(const json& parsed_json) {
  try {
    if (parsed_json.contains("post_processor") &&
        !parsed_json.at("post_processor").is_null()) {
      const auto& post_processor_json = parsed_json.at("post_processor");
      _postprocessor =
          PostProcessorConfig().parse_json(post_processor_json).create();
    }
  } catch (const std::exception& e) {
    TK_LOG(Error, "Failed to setup post_processor: %s", e.what());
    return Error::LoadFailure;
  }
  return Error::Ok;
}

Error HFTokenizer::setup_decoder(const json& parsed_json) {
  try {
    if (parsed_json.contains("decoder") &&
        !parsed_json.at("decoder").is_null()) {
      _decoder =
          TokenDecoderConfig().parse_json(parsed_json.at("decoder")).create();
    }
  } catch (const std::exception& e) {
    TK_LOG(Error, "Failed to setup decoder: %s", e.what());
    return Error::LoadFailure;
  }
  return Error::Ok;
}

Error HFTokenizer::parse_merges(const json& parsed_json) {
  try {
    const auto& merges = parsed_json.at("/model/merges"_json_pointer);
    std::vector<std::pair<std::string, std::string>> merge_pairs;

    for (const auto& merge : merges) {
      std::string first, second;
      if (merge.is_string()) {
        std::string merge_str = merge.get<std::string>();
        if (merge_str.rfind("#version", 0) == 0)
          continue;
        auto space_pos = merge_str.find(' ');
        if (space_pos != std::string::npos) {
          first = merge_str.substr(0, space_pos);
          second = merge_str.substr(space_pos + 1);
        }
      } else if (merge.is_array() && merge.size() == 2) {
        first = merge[0].get<std::string>();
        second = merge[1].get<std::string>();
      }
      if (!first.empty() && !second.empty()) {
        merge_pairs.emplace_back(first, second);
      }
    }

    merge_map_ = std::make_unique<detail::MergeMap>();
    for (size_t i = 0; i < merge_pairs.size(); ++i) {
      const auto& [first, second] = merge_pairs[i];
      auto first_id = token_map_->tryGetInteger(first);
      auto second_id = token_map_->tryGetInteger(second);
      if (first_id && second_id) {
        std::string merged = first + second;
        auto merged_id = token_map_->tryGetInteger(merged);
        if (merged_id) {
          merge_map_->emplace(
              std::make_pair(*first_id, *second_id),
              std::make_pair(static_cast<uint32_t>(i), *merged_id));
        }
      }
    }

    auto merge_ranks_result =
        detail::build_merge_ranks_map(*merge_map_, *token_map_);
    if (!merge_ranks_result.ok()) {
      return merge_ranks_result.error();
    }
    merge_ranks_.emplace(std::move(*merge_ranks_result));
  } catch (const std::exception& e) {
    TK_LOG(Error, "Could not parse merges: %s", e.what());
    return Error::LoadFailure;
  }
  return Error::Ok;
}

Error HFTokenizer::setup_special_token_ids(
    const std::string& /*path*/,
    const json& parsed_json,
    const std::string& model_config_json,
    const std::string& special_tokens_map_json) {
  std::string config_bos_token;
  std::string config_eos_token;
  std::string config_unk_token;
  bool explicit_unk_null = false;

  try {
    const auto& model_json = parsed_json.at("model");
    if (model_json.contains("byte_fallback")) {
      byte_fallback_ = model_json.at("byte_fallback").get<bool>();
    }
    if (model_json.contains("unk_token")) {
      if (model_json.at("unk_token").is_null()) {
        explicit_unk_null = true;
      } else {
        config_unk_token = model_json.at("unk_token").get<std::string>();
      }
    }
  } catch (...) {
  }

  auto process_config_file = [&](const std::string& file_path) {
    if (file_path.empty())
      return;
    std::ifstream f(file_path);
    if (!f)
      return;
    try {
      json j = json::parse(f);
      if (j.contains("bos_token"))
        config_bos_token = extract_token_string(j["bos_token"]);
      if (j.contains("eos_token"))
        config_eos_token = extract_token_string(j["eos_token"]);
      if (config_unk_token.empty() && !explicit_unk_null &&
          j.contains("unk_token")) {
        if (j["unk_token"].is_null()) {
          explicit_unk_null = true;
        } else {
          config_unk_token = extract_token_string(j["unk_token"]);
        }
      }
    } catch (...) {
    }
  };

  process_config_file(special_tokens_map_json);
  if (config_bos_token.empty() || config_eos_token.empty() ||
      (config_unk_token.empty() && !explicit_unk_null)) {
    process_config_file(model_config_json);
  }

  auto set_special =
      [&](const std::string& token_str, uint64_t& target_id, bool& found_flag) {
        if (!token_str.empty()) {
          auto id = special_token_map_->tryGetInteger(token_str);
          if (id) {
            target_id = *id;
            found_flag = true;
          }
        }
      };

  bool bos_found = false;
  bool eos_found = false;
  set_special(config_bos_token, bos_tok_, bos_found);
  set_special(config_eos_token, eos_tok_, eos_found);

  if (!config_unk_token.empty()) {
    auto id = special_token_map_->tryGetInteger(config_unk_token);
    if (id) {
      unk_tok_ = *id;
      unk_token_is_configured_ = true;
    }
  }

  if (!unk_token_is_configured_ && !explicit_unk_null) {
    for (const auto& name : {"<unk>", "[UNK]", "<|endoftext|>"}) {
      auto id = special_token_map_->tryGetInteger(name);
      if (id) {
        unk_tok_ = *id;
        unk_token_is_configured_ = true;
        break;
      }
    }
  }

  if (!bos_found || !eos_found) {
    std::vector<std::string_view> bos_c, eos_c;
    for (size_t i = 0; i < special_token_map_->size(); ++i) {
      const auto& [token, _] = special_token_map_->getElement(i);
      if (!bos_found &&
          (token.find("bos") != std::string::npos ||
           token.find("begin") != std::string::npos))
        bos_c.push_back(token);
      if (!eos_found &&
          (token.find("eos") != std::string::npos ||
           token.find("end") != std::string::npos))
        eos_c.push_back(token);
    }
    if (!bos_found && bos_c.size() == 1) {
      bos_tok_ = *special_token_map_->tryGetInteger(std::string(bos_c[0]));
      bos_found = true;
    }
    if (!eos_found && eos_c.size() == 1) {
      eos_tok_ = *special_token_map_->tryGetInteger(std::string(eos_c[0]));
      eos_found = true;
    }
    if (bos_found && !eos_found) {
      eos_tok_ = bos_tok_;
      eos_found = true;
    } else if (!bos_found && eos_found) {
      bos_tok_ = eos_tok_;
      bos_found = true;
    }
  }

  return Error::Ok;
}

} // namespace tokenizers
