#pragma once

// Shared thinking tag registry and stream normalizer.
// Models that use non-standard thinking tags (not <think>/<​/think>) are
// registered here. Both llama.cpp and MNN backends use this to normalize
// all thinking output to <think>/<​/think> before sending to Kotlin.
//
// Adding a new model family: add one entry to kNonStandardTags[].

#include <optional>
#include <string>

namespace omniinfer {
namespace thinking_tags {

// A pair of native thinking tags that need normalization to <think>/<​/think>.
struct NativeTagPair {
  const char* id;     // Core marker for matching (e.g. "<|channel>thought")
  const char* start;  // Full start tag including trailing whitespace (e.g. "<|channel>thought\n")
  const char* end;    // End tag (e.g. "<channel|>")
};

// ---------------------------------------------------------------------------
// Registry: one entry per model family with non-standard thinking tags.
// ---------------------------------------------------------------------------

static const NativeTagPair kNonStandardTags[] = {
    {"<|channel>thought", "<|channel>thought\n", "<channel|>"},  // Gemma 4
};

// ---------------------------------------------------------------------------
// Lookup functions
// ---------------------------------------------------------------------------

// Given a thinking start tag string (from template system or prompt detection),
// returns the matching NativeTagPair if normalization is needed, or nullptr
// if the tag is standard (<think>) or unknown.
static const NativeTagPair* find_non_standard(const std::string& tag) {
  if (tag.empty()) return nullptr;
  if (tag.find("<think>") != std::string::npos) return nullptr;
  for (const auto& pair : kNonStandardTags) {
    if (tag.find(pair.id) != std::string::npos) return &pair;
  }
  return nullptr;
}

// Scan the tail of a formatted prompt for any known non-standard thinking tag.
// Used by MNN when detect_trailing_tag() fails (e.g. Gemma 4 tags contain |).
static const NativeTagPair* detect_in_prompt(const std::string& prompt) {
  size_t tail_start = prompt.size() > 100 ? prompt.size() - 100 : 0;
  std::string tail = prompt.substr(tail_start);
  for (const auto& pair : kNonStandardTags) {
    if (tail.find(pair.id) != std::string::npos) return &pair;
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// Stream normalizer: replaces native thinking tags with <think>/<​/think>.
// Create one per generate() call when normalization is needed.
// ---------------------------------------------------------------------------

class Normalizer {
public:
  explicit Normalizer(const NativeTagPair& tags)
      : start_(tags.start), end_(tags.end),
        keep_(std::max(start_.size(), end_.size())) {}

  // Process incoming text. Returns normalized output ready to emit.
  // May buffer partial tags internally.
  std::string process(const std::string& text) {
    buf_ += text;
    std::string out;

    // Replace native start tag with <think>\n
    auto sp = buf_.find(start_);
    if (sp != std::string::npos) {
      out += buf_.substr(0, sp) + "<think>\n";
      buf_ = buf_.substr(sp + start_.size());
    }

    // Replace native end tag with </think>
    auto ep = buf_.find(end_);
    if (ep != std::string::npos) {
      out += buf_.substr(0, ep) + "</think>";
      buf_ = buf_.substr(ep + end_.size());
    }

    // Flush safe content, keep tail for partial tag matching.
    if (buf_.size() > keep_) {
      out += buf_.substr(0, buf_.size() - keep_);
      buf_ = buf_.substr(buf_.size() - keep_);
    }

    return out;
  }

  // Flush remaining buffer (call at end of generation).
  std::string flush() {
    std::string out = std::move(buf_);
    buf_.clear();
    return out;
  }

private:
  std::string start_;
  std::string end_;
  size_t keep_;
  std::string buf_;
};

}  // namespace thinking_tags
}  // namespace omniinfer
