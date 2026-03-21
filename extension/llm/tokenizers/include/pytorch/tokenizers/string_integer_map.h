/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <pytorch/tokenizers/error.h>
#include <pytorch/tokenizers/result.h>

namespace tokenizers {
namespace detail {

/**
 * StringIntegerMap is an immutable bidirectional map between strings and 64 bit
 * unsigned integers. The element data is stored in a contiguous array and is
 * shared between both the string buckets and the integer buckets, offering a
 * compact representation.
 *
 * Variable sized integers are used internally, which are sized based on the
 * data being stored. Custom hash functions are supported, with a stateful hash
 * functor being optionally provided at construction time.
 */
template <
    typename TStringHash = std::hash<std::string_view>,
    typename TIntegerHash = std::hash<std::uint64_t>,
    typename TAllocator = std::allocator<std::uint8_t>>
class StringIntegerMap {
 public:
  /// Default constructor is deleted.
  StringIntegerMap() = delete;

  /// StringIntegerMap is move-only.
  ~StringIntegerMap() = default;
  StringIntegerMap(const StringIntegerMap&) = delete;
  StringIntegerMap& operator=(const StringIntegerMap&) = delete;
  StringIntegerMap(StringIntegerMap&&) = default;
  StringIntegerMap& operator=(StringIntegerMap&&) = default;

  /**
   * Create a StringIntegerMap from a map of strings to integers,
   * detecting duplicate tokens or ranks during construction.
   * @param map map of strings to integers
   * @return Result containing the map, or an error if duplicates are detected
   */
  template <typename TMap>
  static Result<StringIntegerMap> create(const TMap& map);

  /**
   * Create a StringIntegerMap from a map of strings to integers,
   * detecting duplicate tokens or ranks during construction.
   * @param map map of strings to integers
   * @param string_hasher hasher for strings
   * @param integer_hasher hasher for integers
   * @return Result containing the map, or an error if duplicates are detected
   */
  template <typename TMap>
  static Result<StringIntegerMap> create(
      const TMap& map,
      TStringHash string_hasher,
      TIntegerHash integer_hasher);

  /// @name Accessors
  /// @{

  /**
   * Attempts to retrieve the integer mapped for the given string.
   * @param str string to lookup
   * @return a std::optional containing the integer if the string was found,
   * std::nullopt otherwise
   */
  std::optional<std::uint64_t> tryGetInteger(std::string_view str) const;

  /**
   * Attempts to retrieve the string mapped for the given integer.
   * @param integer integer to lookup
   * @return a std::optional containing the string if the integer was found,
   * std::nullopt otherwise
   */
  std::optional<std::string_view> tryGetString(std::uint64_t integer) const;

  /**
   * Retrieves the number of elements in the map.
   * @return the number of elements in the map
   */
  std::size_t size() const;

  /**
   * Retrieves the element in the map at the given index.
   * @return A pair containing the string and integer at the given index.
   */
  std::pair<std::string_view, std::uint64_t> getElement(
      std::size_t index) const;

  /// @}

 private:
  template <typename TLogical>
  class VariableSizedInteger {
   public:
    VariableSizedInteger() = default;

    explicit VariableSizedInteger(TLogical max_value) {
      while (max_value != 0) {
        ++byte_count_;
        max_value >>= 8;
      }

      mask_ = (TLogical(1) << (byte_count_ * 8)) - TLogical(1);
    }

    std::size_t getByteCount() const {
      return byte_count_;
    }

    TLogical getMask() const {
      return mask_;
    }

    std::uint8_t* write(std::uint8_t* target, TLogical value) const {
      std::memcpy(target, &value, byte_count_);
      return target + byte_count_;
    }

    TLogical read(const std::uint8_t* source) const {
      TLogical value;
      std::memcpy(&value, source, sizeof(TLogical));
      return value & mask_;
    }

   private:
    std::size_t byte_count_ = 0;
    TLogical mask_ = 0;
  };

  bool tryGetInteger(std::string_view str, std::uint64_t& result) const;

  bool tryGetString(std::uint64_t integer, std::string_view& result) const;

  std::size_t getBucketIndex(std::string_view value) const;

  std::size_t getBucketIndex(std::uint64_t value) const;

  static std::uint8_t getSmallHash(std::size_t hash);

  /// Construct with hashers only; init() must be called to complete setup.
  StringIntegerMap(TStringHash string_hasher, TIntegerHash integer_hasher);

  /// Initialize the map from a map of strings to integers. Must only be
  /// called once, on a freshly constructed (uninitialized) instance.
  /// @return Error::Ok on success, or Error::ParseFailure if duplicates found.
  template <typename TMap>
  Error init(const TMap& map);

  /// The hasher used for strings.
  TStringHash string_hasher_ = {};

  /// The hasher used for integers.
  TIntegerHash integer_hasher_ = {};

  /// String bucket references.
  std::vector<std::uint8_t, TAllocator> integer_bucket_data_;

  /// Integer bucket elements.
  /// Laid out as:
  /// struct {
  ///   std::uint64_t integer; - Physically using integer_ bytes.
  ///   std::size_t string_size; - Physically using string_size_ bytes
  ///   std::size_t string_offset; - Physically using string_offset_ bytes
  /// }
  std::vector<std::uint8_t, TAllocator> integer_element_data_;

  /// String bucket references.
  std::vector<std::uint8_t, TAllocator> string_bucket_data_;

  /// String bucket elements.
  /// Laid out as:
  /// struct {
  ///   std::uint64_t integer; - Physically using integer_ bytes.
  ///   std::size_t string_size; - Physically using string_size_ bytes
  ///   std::uint8_t small_hash; - Using std::uint8_t bytes.
  ///   char string[string_size]; - String data, not zero terminated.
  /// }
  std::vector<std::uint8_t, TAllocator> string_element_data_;

  /// Number of hash buckets to use.
  std::size_t bucket_count_ = 0;

  /// Number of elements stored in the map.
  std::size_t size_ = 0;

  /// Variable sized element offset info.
  VariableSizedInteger<std::size_t> element_offset_;

  /// Variable size string offset info.
  VariableSizedInteger<std::size_t> string_offset_;

  /// Variable sized string size info.
  VariableSizedInteger<std::size_t> string_size_;

  /// Variable sized integer info.
  VariableSizedInteger<std::uint64_t> integer_;
};

template <typename TStringHash, typename TIntegerHash, typename TAllocator>
StringIntegerMap<TStringHash, TIntegerHash, TAllocator>::StringIntegerMap(
    TStringHash string_hasher,
    TIntegerHash integer_hasher)
    : string_hasher_(std::move(string_hasher)),
      integer_hasher_(std::move(integer_hasher)) {}

template <typename TStringHash, typename TIntegerHash, typename TAllocator>
template <typename TMap>
Error StringIntegerMap<TStringHash, TIntegerHash, TAllocator>::init(
    const TMap& map) {
  TK_CHECK_OR_RETURN_ERROR(
      size_ == 0, Internal, "init() called on an already-initialized map");
  assert(map.size() <= std::numeric_limits<std::uint32_t>::max());
  bucket_count_ = size_ = map.size();

  struct BuilderElement {
    std::uint64_t integer = 0;
    std::string_view string;
    std::size_t hash = 0;
    std::size_t element_offset = 0;
    std::size_t original_index = 0;
  };

  std::vector<BuilderElement> builder_string_elements;
  std::vector<BuilderElement> builder_integer_elements;
  builder_string_elements.reserve(map.size());
  builder_integer_elements.reserve(map.size());

  //
  // Calculate various item sizes and gather the builder elements.
  //

  std::size_t largest_string_size = 0;
  std::uint64_t largest_integer = 0;
  std::size_t total_string_size = 0;

  std::size_t idx = 0;
  for (const auto& [str, integer] : map) {
    total_string_size += str.size();
    largest_string_size = std::max(largest_string_size, str.size());
    largest_integer = std::max(largest_integer, integer);
    builder_string_elements.push_back(
        {integer, str, string_hasher_(str), 0, idx});
    builder_integer_elements.push_back(
        {integer, str, integer_hasher_(integer), 0, idx});
    ++idx;
  }

  integer_ = VariableSizedInteger<std::uint64_t>(largest_integer);
  string_size_ = VariableSizedInteger<std::size_t>(largest_string_size);
  string_offset_ = VariableSizedInteger<std::size_t>(total_string_size);

  const auto string_element_data_size =
      ((integer_.getByteCount() + string_size_.getByteCount() + 1) *
       map.size()) +
      total_string_size;
  const auto integer_element_size = integer_.getByteCount() +
      string_offset_.getByteCount() + string_size_.getByteCount();
  const auto integer_element_data_size = integer_element_size * map.size();

  element_offset_ = VariableSizedInteger<std::size_t>(
      std::max(string_element_data_size, integer_element_data_size));

  string_bucket_data_.resize(
      ((bucket_count_ + 1) * element_offset_.getByteCount()) +
      sizeof(std::uint64_t));
  integer_bucket_data_.resize(
      ((bucket_count_ + 1) * element_offset_.getByteCount()) +
      sizeof(std::uint64_t));

  //
  // Set up terminal bucket indices.
  //

  element_offset_.write(
      string_bucket_data_.data() +
          (bucket_count_ * element_offset_.getByteCount()),
      string_element_data_size);
  element_offset_.write(
      integer_bucket_data_.data() +
          (bucket_count_ * element_offset_.getByteCount()),
      integer_element_data_size);
  //
  // Sort the builder elements.
  //

  // Counting sort by bucket index (O(n) vs O(n log n))
  {
    std::vector<std::size_t> counts(bucket_count_ + 1, 0);
    for (const auto& e : builder_string_elements) {
      counts[e.hash % bucket_count_]++;
    }
    std::size_t total = 0;
    for (auto& c : counts) {
      auto old = c;
      c = total;
      total += old;
    }
    std::vector<BuilderElement> sorted(builder_string_elements.size());
    for (auto& e : builder_string_elements) {
      sorted[counts[e.hash % bucket_count_]++] = std::move(e);
    }
    builder_string_elements = std::move(sorted);

    // Sort within each bucket by small_hash, then by string to ensure
    // identical strings are adjacent for duplicate detection.
    {
      std::size_t bucket_start = 0;
      for (std::size_t b = 0; b < bucket_count_; ++b) {
        std::size_t bucket_end = counts[b];
        if (bucket_end - bucket_start > 1) {
          std::sort(
              builder_string_elements.begin() + bucket_start,
              builder_string_elements.begin() + bucket_end,
              [](const BuilderElement& a, const BuilderElement& b) {
                auto ah = getSmallHash(a.hash);
                auto bh = getSmallHash(b.hash);
                if (ah != bh) {
                  return ah < bh;
                }
                return a.string < b.string;
              });
        }
        bucket_start = bucket_end;
      }
    }
  }

  // Detect duplicate tokens (adjacent after sort)
  for (std::size_t i = 1; i < builder_string_elements.size(); ++i) {
    if (builder_string_elements[i].string ==
        builder_string_elements[i - 1].string) {
      TK_LOG(
          Error,
          "duplicate token: %s",
          std::string(builder_string_elements[i].string).c_str());
      return Error::ParseFailure;
    }
  }

  // Counting sort by bucket index (O(n) vs O(n log n))
  {
    std::vector<std::size_t> counts(bucket_count_ + 1, 0);
    for (const auto& e : builder_integer_elements) {
      counts[e.hash % bucket_count_]++;
    }
    std::size_t total = 0;
    for (auto& c : counts) {
      auto old = c;
      c = total;
      total += old;
    }
    std::vector<BuilderElement> sorted(builder_integer_elements.size());
    for (auto& e : builder_integer_elements) {
      sorted[counts[e.hash % bucket_count_]++] = std::move(e);
    }
    builder_integer_elements = std::move(sorted);

    // Sort within each bucket by integer value
    {
      std::size_t bucket_start = 0;
      for (std::size_t b = 0; b < bucket_count_; ++b) {
        std::size_t bucket_end = counts[b];
        if (bucket_end - bucket_start > 1) {
          std::sort(
              builder_integer_elements.begin() + bucket_start,
              builder_integer_elements.begin() + bucket_end,
              [](const BuilderElement& a, const BuilderElement& b) {
                return a.integer < b.integer;
              });
        }
        bucket_start = bucket_end;
      }
    }
  }

  // Detect duplicate ranks (adjacent after sort)
  for (std::size_t i = 1; i < builder_integer_elements.size(); ++i) {
    if (builder_integer_elements[i].integer ==
        builder_integer_elements[i - 1].integer) {
      TK_LOG(
          Error,
          "duplicate rank: %llu",
          static_cast<unsigned long long>(builder_integer_elements[i].integer));
      return Error::ParseFailure;
    }
  }

  //
  // Lay out the string elements and record their positions.
  //

  std::vector<std::size_t> string_offsets_by_index(size_);
  string_element_data_.resize(string_element_data_size + sizeof(std::uint64_t));
  auto* string_element = string_element_data_.data();
  for (auto& builder_element : builder_string_elements) {
    builder_element.element_offset =
        string_element - string_element_data_.data();

    string_offsets_by_index[builder_element.original_index] =
        builder_element.element_offset;

    string_element = integer_.write(string_element, builder_element.integer);
    string_element =
        string_size_.write(string_element, builder_element.string.size());
    *string_element = getSmallHash(builder_element.hash);
    string_element++;
    std::memcpy(
        string_element,
        builder_element.string.data(),
        builder_element.string.size());
    string_element += builder_element.string.size();
    assert(
        string_element >= string_element_data_.data() &&
        string_element <=
            string_element_data_.data() + string_element_data_size);
  }

  //
  // Lay out the integer elements.
  //

  integer_element_data_.resize(
      integer_element_data_size + sizeof(std::uint64_t));
  auto* integer_element = integer_element_data_.data();
  for (auto& builder_element : builder_integer_elements) {
    builder_element.element_offset =
        integer_element - integer_element_data_.data();
    auto string_element_byte_offset =
        string_offsets_by_index[builder_element.original_index];
    integer_element = integer_.write(integer_element, builder_element.integer);
    integer_element =
        string_size_.write(integer_element, builder_element.string.size());
    integer_element =
        string_offset_.write(integer_element, string_element_byte_offset);
    assert(
        integer_element >= integer_element_data_.data() &&
        integer_element <=
            integer_element_data_.data() + integer_element_data_size);
  }

  //
  // Both the string elements and integer elements are laid out in order of
  // their respective hashes. Generate the hash indexes for the string elements
  // and integer elements.
  //

  auto builder_string_elements_iter = std::begin(builder_string_elements);
  auto builder_integer_elements_iter = std::begin(builder_integer_elements);

  for (std::size_t bucket_idx = 0; bucket_idx < bucket_count_; ++bucket_idx) {
    auto* string_bucket = string_bucket_data_.data() +
        (bucket_idx * element_offset_.getByteCount());
    if (builder_string_elements_iter != std::end(builder_string_elements)) {
      element_offset_.write(
          string_bucket, builder_string_elements_iter->element_offset);
    } else {
      element_offset_.write(string_bucket, string_element_data_size);
    }

    auto* integer_bucket = integer_bucket_data_.data() +
        (bucket_idx * element_offset_.getByteCount());
    if (builder_integer_elements_iter != std::end(builder_integer_elements)) {
      element_offset_.write(
          integer_bucket, builder_integer_elements_iter->element_offset);
    } else {
      element_offset_.write(integer_bucket, integer_element_data_size);
    }

    //
    // Advance the string element iterator past all string elements that map
    // into this bucket.
    //

    while (builder_string_elements_iter != std::end(builder_string_elements) &&
           getBucketIndex(builder_string_elements_iter->string) == bucket_idx) {
      ++builder_string_elements_iter;
    }

    //
    // Advance the integer element index past all integer elements that map into
    // this bucket.
    //

    while (
        builder_integer_elements_iter != std::end(builder_integer_elements) &&
        getBucketIndex(builder_integer_elements_iter->integer) == bucket_idx) {
      ++builder_integer_elements_iter;
    }
  }

  return Error::Ok;
}

template <typename TStringHash, typename TIntegerHash, typename TAllocator>
template <typename TMap>
Result<StringIntegerMap<TStringHash, TIntegerHash, TAllocator>>
StringIntegerMap<TStringHash, TIntegerHash, TAllocator>::create(
    const TMap& map) {
  return create(map, TStringHash(), TIntegerHash());
}

template <typename TStringHash, typename TIntegerHash, typename TAllocator>
template <typename TMap>
Result<StringIntegerMap<TStringHash, TIntegerHash, TAllocator>>
StringIntegerMap<TStringHash, TIntegerHash, TAllocator>::create(
    const TMap& map,
    TStringHash string_hasher,
    TIntegerHash integer_hasher) {
  StringIntegerMap result(std::move(string_hasher), std::move(integer_hasher));
  auto error = result.init(map);
  if (error != Error::Ok) {
    return error;
  }
  return std::move(result);
}

template <typename TStringHash, typename TIntegerHash, typename TAllocator>
std::optional<std::uint64_t>
StringIntegerMap<TStringHash, TIntegerHash, TAllocator>::tryGetInteger(
    std::string_view str) const {
  std::uint64_t result;
  return tryGetInteger(str, result) ? std::optional<std::uint64_t>(result)
                                    : std::nullopt;
}

template <typename TStringHash, typename TIntegerHash, typename TAllocator>
bool StringIntegerMap<TStringHash, TIntegerHash, TAllocator>::tryGetInteger(
    std::string_view str,
    std::uint64_t& result) const {
  if (size_ == 0) {
    return false;
  }

  const auto hash = string_hasher_(str);
  const auto bucket_index = hash % bucket_count_;
  const auto small_hash = getSmallHash(hash);

  const auto* bucket_data = string_bucket_data_.data() +
      (bucket_index * element_offset_.getByteCount());
  const auto lower_element_offset = element_offset_.read(bucket_data);
  const auto upper_element_offset =
      element_offset_.read(bucket_data + element_offset_.getByteCount());

  const auto integer_size = integer_.getByteCount();
  const auto string_size_size = string_size_.getByteCount();

  std::size_t element_size = 0;
  auto* element_data_end = string_element_data_.data() + upper_element_offset;
  for (auto* element_data = string_element_data_.data() + lower_element_offset;
       element_data < element_data_end;
       element_data += element_size) {
    //
    // Read the string length.
    //

    const auto element_string_length =
        string_size_.read(element_data + integer_size);
    element_size = integer_size + string_size_size + 1 + element_string_length;

    //
    // Read the string small hash.
    //

    const auto element_small_hash =
        element_data[integer_size + string_size_size];
    if (element_small_hash < small_hash) {
      continue;
    } else if (element_small_hash > small_hash) {
      break;
    }

    //
    // Get a view on the string for a full comparison.
    //

    std::string_view element_string(
        reinterpret_cast<const char*>(
            element_data + integer_size + string_size_size + 1),
        element_string_length);
    if (str == element_string) {
      result = integer_.read(element_data);
      return true;
    }
  }

  return false;
}

template <typename TStringHash, typename TIntegerHash, typename TAllocator>
std::optional<std::string_view>
StringIntegerMap<TStringHash, TIntegerHash, TAllocator>::tryGetString(
    std::uint64_t integer) const {
  std::string_view result;
  return tryGetString(integer, result) ? std::optional<std::string_view>(result)
                                       : std::nullopt;
}

template <typename TStringHash, typename TIntegerHash, typename TAllocator>
bool StringIntegerMap<TStringHash, TIntegerHash, TAllocator>::tryGetString(
    std::uint64_t integer,
    std::string_view& result) const {
  if (size_ == 0) {
    return false;
  }

  const auto bucket_index = getBucketIndex(integer);

  const auto* bucket_data = integer_bucket_data_.data() +
      (bucket_index * element_offset_.getByteCount());
  const auto lower_element_offset = element_offset_.read(bucket_data);
  const auto upper_element_offset =
      element_offset_.read(bucket_data + element_offset_.getByteCount());

  const auto integer_element_size = integer_.getByteCount() +
      string_offset_.getByteCount() + string_size_.getByteCount();
  auto* element_data_end = integer_element_data_.data() + upper_element_offset;
  for (auto* element_data = integer_element_data_.data() + lower_element_offset;
       element_data < element_data_end;
       element_data += integer_element_size) {
    const auto element_integer = integer_.read(element_data);
    if (element_integer == integer) {
      const auto element_string_size =
          string_size_.read(element_data + integer_.getByteCount());
      const auto element_string_offset = string_offset_.read(
          element_data + integer_.getByteCount() + string_size_.getByteCount());
      const auto* string_element =
          string_element_data_.data() + element_string_offset;
      const auto* string_data = reinterpret_cast<const char*>(
          string_element + integer_.getByteCount() +
          string_size_.getByteCount() + 1);
      result = std::string_view(string_data, element_string_size);
      return true;
    } else if (element_integer > integer) {
      break;
    }
  }

  return false;
}

template <typename TStringHash, typename TIntegerHash, typename TAllocator>
std::size_t StringIntegerMap<TStringHash, TIntegerHash, TAllocator>::size()
    const {
  return size_;
}

template <typename TStringHash, typename TIntegerHash, typename TAllocator>
std::pair<std::string_view, std::uint64_t>
StringIntegerMap<TStringHash, TIntegerHash, TAllocator>::getElement(
    std::size_t index) const {
  assert(index < size_);

  const auto integer_size = integer_.getByteCount();
  const auto string_offset_size = string_offset_.getByteCount();
  const auto string_size_size = string_size_.getByteCount();
  const auto element_size =
      integer_size + string_offset_size + string_size_size;
  const auto* element_data = &integer_element_data_[index * element_size];

  const auto integer = integer_.read(element_data);
  element_data += integer_size;
  const auto string_size = string_size_.read(element_data);
  element_data += string_size_size;
  const auto string_offset = string_offset_.read(element_data);
  const auto* string_data =
      &string_element_data_
          [string_offset + integer_size + string_size_size + 1];

  return std::make_pair(
      std::string_view(reinterpret_cast<const char*>(string_data), string_size),
      integer);
}

template <typename TStringHash, typename TIntegerHash, typename TAllocator>
std::size_t
StringIntegerMap<TStringHash, TIntegerHash, TAllocator>::getBucketIndex(
    std::string_view value) const {
  return string_hasher_(value) % bucket_count_;
}

template <typename TStringHash, typename TIntegerHash, typename TAllocator>
std::size_t
StringIntegerMap<TStringHash, TIntegerHash, TAllocator>::getBucketIndex(
    std::uint64_t value) const {
  return integer_hasher_(value) % bucket_count_;
}

template <typename TStringHash, typename TIntegerHash, typename TAllocator>
std::uint8_t
StringIntegerMap<TStringHash, TIntegerHash, TAllocator>::getSmallHash(
    std::size_t hash) {
  const auto shift = (sizeof(std::size_t) * 8) - 8;
  return static_cast<std::uint8_t>(hash >> shift);
}

template <
    typename TStringHash = std::hash<std::string_view>,
    typename TIntegerHash = std::hash<std::uint64_t>,
    typename TAllocator = std::allocator<std::uint8_t>>
struct StringIntegerMapTypeBuilder {
  using Map = StringIntegerMap<TStringHash, TIntegerHash, TAllocator>;

  template <typename TOtherStringHash>
  using WithStringHash =
      StringIntegerMapTypeBuilder<TOtherStringHash, TIntegerHash, TAllocator>;

  template <typename TOtherIntegerHash>
  using WithIntegerHash =
      StringIntegerMapTypeBuilder<TStringHash, TOtherIntegerHash, TAllocator>;

  template <typename TOtherAllocator>
  using WithAllocator =
      StringIntegerMapTypeBuilder<TStringHash, TIntegerHash, TOtherAllocator>;
};

} // namespace detail
} // namespace tokenizers
