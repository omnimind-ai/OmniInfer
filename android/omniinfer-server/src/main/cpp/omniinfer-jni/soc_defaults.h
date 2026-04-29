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
    {"sm8850", 6},  // 8 Elite Gen 5: 6×Oryon V3@3.6G + 2×Oryon V3@4.6G (decode peaks 6-7, prefill scales to 8)
    {"sm8750", 6},  // 8 Elite: 6×Oryon V2@3.5G + 2×Oryon V2@4.3G (decode 4-5≈6, prefill 6>>4; 7-8 regress)
    {"sm8650", 6},  // 8 Gen 3: 1×X4 + 3×A720 + 2×A720 (+ 2×A520 efficiency)
    {"sm8550", 5},  // 8 Gen 2: 1×X3 + 2×A715 + 2×A710 (+ 3×A510 efficiency; 6+ threads regress 30-40%)
    // MediaTek Dimensity
    {"mt6878", 4},  // 7300/7400: 4×A78 + 4×A55 (A55 drags decode, 4 big cores optimal)
};

// Returns recommended thread count for the current SoC.
// Falls back to `fallback` if the SoC is not in the table.
// Default 4: most Android SoCs have 4 big + 4 small cores;
// using >4 threads pulls in slow efficiency cores that hurt decode.
inline int get_soc_default_threads(int fallback = 4) {
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
