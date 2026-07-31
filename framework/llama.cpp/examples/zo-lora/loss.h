#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace zo_lora {

inline float vocabulary_cross_entropy(const float * logits, int32_t n_vocab, int32_t label) {
    if (logits == nullptr || n_vocab <= 0) {
        throw std::invalid_argument("vocabulary cross-entropy requires logits");
    }
    if (label < 0 || label >= n_vocab) {
        throw std::out_of_range("vocabulary cross-entropy label is out of range");
    }
    float maximum = -std::numeric_limits<float>::infinity();
    for (int32_t token = 0; token < n_vocab; ++token) {
        maximum = std::max(maximum, logits[token]);
    }

    double sum = 0.0;
    for (int32_t token = 0; token < n_vocab; ++token) {
        sum += std::exp((double) logits[token] - (double) maximum);
    }

    return (float) (std::log(sum) + (double) maximum - (double) logits[label]);
}

} // namespace zo_lora
