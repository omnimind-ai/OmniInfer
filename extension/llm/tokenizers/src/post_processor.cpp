/*
 * Copyright (c) Software Mansion S.A and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <pytorch/tokenizers/log.h>
#include <pytorch/tokenizers/post_processor.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

using json = nlohmann::json;

namespace tokenizers {

namespace {

// Helper to count added tokens
size_t count_added(
    const Template& container,
    const std::map<std::string, SpecialToken>& special_tokens) {
  size_t count = 0;
  for (const auto& piece : container) {
    if (piece.is_special_token) {
      auto it = special_tokens.find(piece.id);
      if (it != special_tokens.end()) {
        count += it->second.ids.size();
      }
    }
  }
  return count;
}

// Helper to parse Piece from string or object
Piece parse_piece(const json& j) {
  if (j.is_string()) {
    std::string s = j.get<std::string>();
    // Check for $A, $B, $0, etc.
    if (s.rfind("$", 0) == 0) {
      std::string rest = s.substr(1);
      if (rest == "" || rest == "A" || rest == "a")
        return Piece::Sequence(SequenceId::A, 0);
      if (rest == "B" || rest == "b")
        return Piece::Sequence(SequenceId::B, 0);
      // Try parse number
      try {
        uint64_t type_id = std::stoul(rest);
        return Piece::Sequence(SequenceId::A, type_id);
      } catch (...) {
      }
      // Split by :
      auto colon = rest.find(':');
      if (colon != std::string::npos) {
        // handle $A:1
        std::string id_part = rest.substr(0, colon);
        std::string type_part = rest.substr(colon + 1);
        SequenceId seq =
            (id_part == "B" || id_part == "b") ? SequenceId::B : SequenceId::A;
        uint64_t type_id = 0;
        try {
          type_id = std::stoul(type_part);
        } catch (...) {
        }
        return Piece::Sequence(seq, type_id);
      }
    }
    // Else special token
    return Piece::SpecialToken(s, 0);
  } else if (j.is_object()) {
    if (j.contains("Sequence")) {
      auto& s = j["Sequence"];
      std::string id_str = s.value("id", "A");
      uint64_t type_id = s.value("type_id", 0);
      return Piece::Sequence(
          id_str == "B" ? SequenceId::B : SequenceId::A, type_id);
    } else if (j.contains("SpecialToken")) {
      auto& s = j["SpecialToken"];
      std::string id = s.value("id", "");
      uint64_t type_id = s.value("type_id", 0);
      return Piece::SpecialToken(id, type_id);
    }
  }
  return Piece::SpecialToken("", 0); // Fallback
}

Template parse_template(const json& j) {
  Template t;
  if (j.is_array()) {
    for (const auto& item : j) {
      t.push_back(parse_piece(item));
    }
  } else if (j.is_string()) {
    // Split by space
    std::string s = j.get<std::string>();
    std::string delimiter = " ";
    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
      token = s.substr(0, pos);
      if (!token.empty())
        t.push_back(parse_piece(token));
      s.erase(0, pos + delimiter.length());
    }
    if (!s.empty())
      t.push_back(parse_piece(s));
  }
  return t;
}

std::map<std::string, SpecialToken> parse_special_tokens(const json& j) {
  std::map<std::string, SpecialToken> map;
  for (auto it = j.begin(); it != j.end(); ++it) {
    std::string key = it.key();
    auto val = it.value();
    SpecialToken st;
    st.id = val.value("id", key);
    if (val.contains("ids"))
      st.ids = val["ids"].get<std::vector<uint64_t>>();
    if (val.contains("tokens"))
      st.tokens = val["tokens"].get<std::vector<std::string>>();
    map[key] = st;
  }
  return map;
}

} // namespace

// PostProcessorConfig /////////////////////////////////////////////////////////

PostProcessorConfig::PostProcessorConfig(std::string type)
    : type(std::move(type)) {
  // Set defaults for complex types if needed, though mostly handled in parsing
  sep = {"[SEP]", 102};
  cls = {"[CLS]", 101};
}

PostProcessor::Ptr PostProcessorConfig::create() const {
  if (type == "TemplateProcessing") {
    return std::make_shared<TemplateProcessing>(single, pair, special_tokens);
  } else if (type == "Sequence") {
    std::vector<PostProcessor::Ptr> ptrs;
    for (const auto& cfg : processors) {
      ptrs.push_back(cfg.create());
    }
    return std::make_shared<Sequence>(std::move(ptrs));
  } else if (type == "ByteLevel") {
    // Return no-op sequence
    return std::make_shared<Sequence>(std::vector<PostProcessor::Ptr>{});
  }

  throw std::runtime_error("Unsupported PostProcessor type: " + type);
}

PostProcessorConfig& PostProcessorConfig::parse_json(const json& j) {
  if (j.contains("type")) {
    type = j["type"].get<std::string>();
  }

  if (type == "TemplateProcessing") {
    single = j.contains("single") ? parse_template(j["single"])
                                  : parse_template("$0");
    pair = j.contains("pair") ? parse_template(j["pair"])
                              : parse_template("$A:0 $B:1");
    if (j.contains("special_tokens")) {
      special_tokens = parse_special_tokens(j["special_tokens"]);
    }
  } else if (type == "Sequence") {
    if (j.contains("processors")) {
      for (const auto& item : j["processors"]) {
        processors.push_back(PostProcessorConfig().parse_json(item));
      }
    }
  }

  return *this;
}

// TemplateProcessing //////////////////////////////////////////////////////////

TemplateProcessing::TemplateProcessing(
    Template single,
    Template pair,
    std::map<std::string, SpecialToken> special_tokens)
    : single_(std::move(single)),
      pair_(std::move(pair)),
      special_tokens_(std::move(special_tokens)) {
  added_single_ = count_added(single_, special_tokens_);
  added_pair_ = count_added(pair_, special_tokens_);
}

size_t TemplateProcessing::added_tokens(bool is_pair) const {
  return is_pair ? added_pair_ : added_single_;
}

std::vector<uint64_t> TemplateProcessing::apply_template(
    const Template& tmpl,
    const std::vector<uint64_t>& tokens_a,
    const std::vector<uint64_t>* tokens_b,
    bool add_special_tokens) const {
  std::vector<uint64_t> result;
  // Reserve estimation
  result.reserve(
      tokens_a.size() + (tokens_b ? tokens_b->size() : 0) + tmpl.size());

  for (const auto& piece : tmpl) {
    if (piece.is_special_token) {
      if (add_special_tokens) {
        auto it = special_tokens_.find(piece.id);
        if (it != special_tokens_.end()) {
          for (auto id : it->second.ids) {
            result.push_back(id);
          }
        } else {
          TK_LOG(
              Error,
              "TemplateProcessing: Special token '%s' not found",
              piece.id.c_str());
        }
      }
    } else {
      // Sequence
      if (piece.id == "A") {
        result.insert(result.end(), tokens_a.begin(), tokens_a.end());
      } else if (piece.id == "B") {
        if (tokens_b) {
          result.insert(result.end(), tokens_b->begin(), tokens_b->end());
        }
      }
    }
  }
  return result;
}

std::vector<uint64_t> TemplateProcessing::process(
    const std::vector<uint64_t>& tokens,
    bool add_special_tokens) const {
  return apply_template(single_, tokens, nullptr, add_special_tokens);
}

std::vector<uint64_t> TemplateProcessing::process(
    const std::vector<uint64_t>& tokens_a,
    const std::vector<uint64_t>& tokens_b,
    bool add_special_tokens) const {
  return apply_template(pair_, tokens_a, &tokens_b, add_special_tokens);
}

// Sequence ////////////////////////////////////////////////////////////////////

Sequence::Sequence(std::vector<PostProcessor::Ptr> processors)
    : processors_(std::move(processors)) {}

size_t Sequence::added_tokens(bool is_pair) const {
  size_t sum = 0;
  for (const auto& p : processors_) {
    sum += p->added_tokens(is_pair);
  }
  return sum;
}

std::vector<uint64_t> Sequence::process(
    const std::vector<uint64_t>& tokens,
    bool add_special_tokens) const {
  std::vector<uint64_t> current = tokens;
  for (const auto& p : processors_) {
    current = p->process(current, add_special_tokens);
  }
  return current;
}

std::vector<uint64_t> Sequence::process(
    const std::vector<uint64_t>& tokens_a,
    const std::vector<uint64_t>& tokens_b,
    bool add_special_tokens) const {
  if (processors_.empty()) {
    std::vector<uint64_t> res = tokens_a;
    res.insert(res.end(), tokens_b.begin(), tokens_b.end());
    return res;
  }

  std::vector<uint64_t> current =
      processors_[0]->process(tokens_a, tokens_b, add_special_tokens);

  for (size_t i = 1; i < processors_.size(); ++i) {
    current = processors_[i]->process(current, add_special_tokens);
  }
  return current;
}

} // namespace tokenizers
