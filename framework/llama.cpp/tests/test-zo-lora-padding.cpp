#include "batch-layout.h"

#include <array>
#include <cstdio>
#include <vector>

static bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "test-zo-lora-padding: %s\n", message);
    }
    return condition;
}

int main() {
    const std::vector<size_t> lengths = {5, 2, 5, 3};
    const zo_lora::batch_layout paired = zo_lora::make_batch_layout(lengths, true, true);

    bool ok = true;
    ok = check(paired.padded_length == 5, "wrong padded length") && ok;
    ok = check(paired.real_tokens == 30, "wrong real token count") && ok;
    ok = check(paired.padding_tokens == 10, "wrong padding token count") && ok;
    ok = check(paired.backend_tokens == 40, "wrong backend token count") && ok;
    ok = check(paired.sequences.size() == 8, "wrong sequence count") && ok;

    const std::array<size_t, 8> expected_sides = {0, 0, 0, 0, 1, 1, 1, 1};
    const std::array<size_t, 8> expected_samples = {0, 1, 2, 3, 0, 1, 2, 3};
    for (size_t i = 0; i < paired.sequences.size(); ++i) {
        const zo_lora::batch_sequence_layout & sequence = paired.sequences[i];
        ok = check(sequence.side == expected_sides[i], "layout is not side-major") && ok;
        ok = check(sequence.sample == expected_samples[i], "sample order changed") && ok;
        ok = check(sequence.sequence_id == (int32_t) i, "sequence ids are not contiguous") && ok;
        ok = check(sequence.backend_tokens == 5, "sequence is not rectangular") && ok;
    }

    const zo_lora::batch_layout cpu = zo_lora::make_batch_layout(lengths, true, false);
    ok = check(cpu.real_tokens == 30, "CPU real token count changed") && ok;
    ok = check(cpu.padding_tokens == 0, "CPU path gained padding") && ok;
    ok = check(cpu.backend_tokens == 30, "CPU backend token count changed") && ok;
    return ok ? 0 : 1;
}
