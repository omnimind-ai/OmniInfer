#include "loss.h"

#include <array>
#include <cmath>
#include <cstdio>

static bool check_close(float actual, double expected, const char * message) {
    const bool ok = std::fabs((double) actual - expected) <= 1e-6;
    if (!ok) {
        std::fprintf(stderr, "test-zo-lora-loss: %s: actual=%g expected=%g\n",
            message, (double) actual, expected);
    }
    return ok;
}

template<typename Exception, typename Function>
static bool check_throws(Function && function, const char * message) {
    try {
        function();
    } catch (const Exception &) {
        return true;
    } catch (...) {
    }
    std::fprintf(stderr, "test-zo-lora-loss: %s: expected exception\n", message);
    return false;
}

int main() {
    bool ok = true;

    const std::array<float, 3> logits = { 1.0f, 2.0f, 3.0f };
    const double expected = std::log(std::exp(-1.0) + 1.0 + std::exp(1.0));
    ok = check_close(zo_lora::vocabulary_cross_entropy(logits.data(), (int32_t) logits.size(), 1),
        expected, "full vocabulary normalization") && ok;
    const double last_label_expected = std::log(std::exp(-2.0) + std::exp(-1.0) + 1.0);
    ok = check_close(zo_lora::vocabulary_cross_entropy(logits.data(), (int32_t) logits.size(), 2),
        last_label_expected, "last vocabulary label") && ok;

    const std::array<float, 3> shifted = { 1001.0f, 1002.0f, 1003.0f };
    ok = check_close(zo_lora::vocabulary_cross_entropy(shifted.data(), (int32_t) shifted.size(), 1),
        expected, "large-logit shift invariance") && ok;

    const std::array<float, 3> distractor = { 2.0f, 1.0f, 6.0f };
    const double distractor_expected = std::log(1.0 + std::exp(-1.0) + std::exp(4.0));
    ok = check_close(zo_lora::vocabulary_cross_entropy(distractor.data(), (int32_t) distractor.size(), 0),
        distractor_expected, "non-verbalizer token contribution") && ok;

    const std::array<float, 1> single = { 7.0f };
    ok = check_close(zo_lora::vocabulary_cross_entropy(single.data(), (int32_t) single.size(), 0),
        0.0, "single-token vocabulary") && ok;

    ok = check_throws<std::invalid_argument>([] {
        zo_lora::vocabulary_cross_entropy(nullptr, 3, 0);
    }, "null logits") && ok;
    ok = check_throws<std::invalid_argument>([&] {
        zo_lora::vocabulary_cross_entropy(logits.data(), 0, 0);
    }, "empty vocabulary") && ok;
    ok = check_throws<std::out_of_range>([&] {
        zo_lora::vocabulary_cross_entropy(logits.data(), (int32_t) logits.size(), -1);
    }, "negative label") && ok;
    ok = check_throws<std::out_of_range>([&] {
        zo_lora::vocabulary_cross_entropy(logits.data(), (int32_t) logits.size(), (int32_t) logits.size());
    }, "label past vocabulary") && ok;

    return ok ? 0 : 1;
}
