#pragma once
// SoC → recommended thread count lookup table.
// Detects the current SoC at runtime via Android system properties
// and returns the benchmarked-optimal thread count.
//
// Adding a new SoC: add one entry to kSocThreadDefaults[].

#include <sys/system_properties.h>
#include <algorithm>
#include <string>

namespace omniinfer {

// Returns lowercase SoC platform identifier (e.g. "sm8650").
inline std::string get_soc_identifier() {
    char buf[PROP_VALUE_MAX] = {};
    const char* props[] = {"ro.soc.model", "ro.board.platform", "ro.hardware.chipname"};
    for (auto* prop : props) {
        if (__system_property_get(prop, buf) > 0) {
            std::string s(buf);
            std::transform(s.begin(), s.end(), s.begin(), ::tolower);
            return s;
        }
    }
    return "unknown";
}

struct SocThreadDefault {
    const char* prefix;   // lowercase SoC identifier prefix
    int threads;          // recommended thread count (big cores only)
};

// Lookup table: prefix → recommended threads.
// Sorted by specificity is not required; first prefix match wins.
static constexpr SocThreadDefault kSocThreadDefaults[] = {
    // Qualcomm Snapdragon
    {"sm8650", 6},  // 8 Gen 3: 1×X4 + 3×A720 + 2×A720 (+ 2×A520 efficiency)
};

// Returns recommended thread count for the current SoC.
// Falls back to `fallback` if the SoC is not in the table.
inline int get_soc_default_threads(int fallback = 6) {
    static const int cached = [fallback]() {
        std::string soc = get_soc_identifier();
        for (const auto& entry : kSocThreadDefaults) {
            if (soc.rfind(entry.prefix, 0) == 0) {  // starts_with
                return entry.threads;
            }
        }
        return fallback;
    }();
    return cached;
}

}  // namespace omniinfer
