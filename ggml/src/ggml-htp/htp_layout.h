#pragma once

// HTP weight layout utilities:

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Restore FP16 weight file layout
// Functions to convert __fp16 weights from 32x32 tile + even/odd row interleaved layout back to regular row-major (k x n) format.
static int htp_w16_tile32_index(int i, int j, int k, int n) {
    // (i, j) -> reordered weight linear index (element-wise)
    int i0 = i / 32, i1 = i % 32;
    int j0 = j / 32, j1 = j % 32;
    int tiles_per_col = k / 32;            // must be divisible
    int tile_idx = j0 * tiles_per_col + i0; // linear tile index
    int tile_base = tile_idx * 1024;   // 1024 __fp16 per tile
    int in_tile = ((i1 & ~1) * 32) + (j1 * 2) + (i1 & 1);
    return tile_base + in_tile;
}

// Convert to C row-major (i, j) -> w_ref[i*n + j]
static void htp_w16_tile32_to_rowmajor_f16(const ggml_fp16_t * __restrict w_perm,
                                                  ggml_fp16_t       * __restrict w_ref,
                                                  size_t k, size_t n) {
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = htp_w16_tile32_index(i, j, k, n);
            w_ref[i * n + j] = w_perm[idx];
        }
    }
}

// Convert to GGML column-major (i, j) -> w_ref[j*k + i]
static void htp_w16_tile32_to_ggml_f16(const ggml_fp16_t * __restrict w_perm,
                                              ggml_fp16_t       * __restrict w_ref,
                                              size_t k, size_t n) {
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = htp_w16_tile32_index(i, j, k, n);
            w_ref[j * k + i] = w_perm[idx];
        }
    }
}

#ifdef __cplusplus
}
#endif
