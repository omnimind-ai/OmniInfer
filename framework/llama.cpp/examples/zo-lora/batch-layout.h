#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace zo_lora {

struct batch_sequence_layout {
    size_t side = 0;
    size_t sample = 0;
    size_t real_tokens = 0;
    size_t backend_tokens = 0;
    int32_t sequence_id = 0;
};

struct batch_layout {
    size_t padded_length = 0;
    int64_t real_tokens = 0;
    int64_t padding_tokens = 0;
    int64_t backend_tokens = 0;
    std::vector<batch_sequence_layout> sequences;
};

inline batch_layout make_batch_layout(
        const std::vector<size_t> & lengths,
        bool                        paired,
        bool                        right_padding) {
    batch_layout result;
    for (size_t length : lengths) {
        result.padded_length = std::max(result.padded_length, length);
    }

    const size_t sides = paired ? 2 : 1;
    result.sequences.reserve(lengths.size()*sides);
    for (size_t side = 0; side < sides; ++side) {
        for (size_t sample = 0; sample < lengths.size(); ++sample) {
            const size_t backend_tokens = right_padding ? result.padded_length : lengths[sample];
            result.sequences.push_back({
                side,
                sample,
                lengths[sample],
                backend_tokens,
                (int32_t) (side*lengths.size() + sample),
            });
            result.real_tokens += lengths[sample];
            result.backend_tokens += backend_tokens;
        }
    }
    result.padding_tokens = result.backend_tokens - result.real_tokens;
    return result;
}

} // namespace zo_lora
