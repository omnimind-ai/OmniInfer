#pragma once

#include "ggml.h"
#include "ggml-quants.h"
#include "htp_layout.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline void htp_debug_print_f32_head(const char * tag, const float * data, int count) {
    fprintf(stderr, "%s:", tag);
    for (int i = 0; i < count; ++i) {
        fprintf(stderr, " %f", data[i]);
    }
    fprintf(stderr, "\n");
}

// Print the first 'limit' elements of the 0th column of a tensor:
// - F32: read directly
// - F16: convert to F32 then print
// - Q8_0: dequantize the 0th "row" (consistent with existing implementation) then print
// Other types print placeholders
static inline void htp_debug_print_tensor_col0_head(const char * prefix, const struct ggml_tensor * t, int64_t len0, int limit) {
    if (limit > (int) len0) limit = (int) len0;

    const char * base = (const char *) t->data;
    if (t->type == GGML_TYPE_F32) {
        fprintf(stderr, "%s[:,0][:8]:", prefix);
        for (int p = 0; p < limit; ++p) {
            const float * v = (const float *) (base + (size_t)p*t->nb[0] + 0*t->nb[1]);
            fprintf(stderr, " %f", *v);
        }
        fprintf(stderr, "\n");
        return;
    }

    if (t->type == GGML_TYPE_F16) {
        fprintf(stderr, "%s[:,0][:8]:", prefix);
        for (int p = 0; p < limit; ++p) {
            const ggml_fp16_t * v = (const ggml_fp16_t *) (base + (size_t)p*t->nb[0] + 0*t->nb[1]);
            fprintf(stderr, " %f", ggml_fp16_to_fp32(*v));
        }
        fprintf(stderr, "\n");
        return;
    }

    if (t->type == GGML_TYPE_Q8_0) {
        float * tmp = (float *) malloc(sizeof(float) * (size_t) len0);
        if (tmp) {
            const void * row0 = (const void *) (base + 0*t->nb[1]);
            dequantize_row_q8_0((const block_q8_0 *) row0, tmp, len0);
            fprintf(stderr, "%s[:,0][:8] (deq q8_0):", prefix);
            for (int p = 0; p < limit; ++p) fprintf(stderr, " %f", tmp[p]);
            fprintf(stderr, "\n");
            free(tmp);
        } else {
            fprintf(stderr, "%s q8_0 deq malloc failed\n", prefix);
        }
        return;
    }

    fprintf(stderr, "%s type %s not supported for print\n", prefix, ggml_type_name(t->type));
}

// Calculate and print the first 'limit' elements of CPU reference out[:,0].
// Supports weights in F32/F16; Q8_0 will be dequantized column by column before dot product with activations.
static inline void htp_debug_print_cpu_ref_col0(const struct ggml_tensor * weight,
                                               const struct ggml_tensor * act,
                                               int64_t k, int64_t n,
                                               int limit) {
    if (limit > (int) n) limit = (int) n;
    if (act->type != GGML_TYPE_F32) {
        fprintf(stderr, "[HTP DEBUG] out[:,0][:8] (CPU ref): (skip: act=%s)\n", ggml_type_name(act->type));
        return;
    }

    const char * wbase = (const char *) weight->data;
    const char * abase = (const char *) act->data;

    fprintf(stderr, "[HTP DEBUG] out[:,0][:8] (CPU ref):");
    if (weight->type == GGML_TYPE_F32 || weight->type == GGML_TYPE_F16) {
        // If weight is F16, rearrange from HTP layout to regular layout, then do (fp16 x fp16)->fp32 accumulation
        // If weight is F32, convert to fp16 regular layout first, then participate in multiplication, 
        // to maintain consistency with w16a32 numerical path
        ggml_fp16_t * w_ref_f16 = (ggml_fp16_t *) malloc(sizeof(ggml_fp16_t) * (size_t)(k*n));
        GGML_ASSERT(w_ref_f16);

        if (weight->type == GGML_TYPE_F16) {
            // weight->data is in rearranged layout: first rearrange back
            const ggml_fp16_t * w_perm = (const ggml_fp16_t *) wbase;
            GGML_ASSERT((k % 32 == 0) && (n % 32 == 0)); // ! Test this
            if ((k % 32 == 0) && (n % 32 == 0)) {
                htp_w16_tile32_to_rowmajor_f16(w_perm, w_ref_f16, k, n);
            } else {
                // Not 32-aligned: fall back to element-by-element indexing
                for (int ii = 0; ii < k; ++ii) {
                    for (int jj = 0; jj < n; ++jj) {
                        int idx = htp_w16_tile32_index(ii, jj, k, n);
                        w_ref_f16[ii * n + jj] = w_perm[idx];
                    }
                }
            }
        } else {
            // F32 -> F16 regular layout
            for (int64_t ii = 0; ii < k; ++ii) {
                for (int64_t jj = 0; jj < n; ++jj) {
                    const float * wp = (const float *) (wbase + (size_t)ii*weight->nb[0] + (size_t)jj*weight->nb[1]);
                    w_ref_f16[(size_t)ii*(size_t)n + (size_t)jj] = ggml_fp32_to_fp16(*wp);
                }
            }
        }

        for (int i = 0; i < limit; ++i) {
            float acc = 0.0;
            for (int64_t p = 0; p < k; ++p) {
                const float * ap_f32 = (const float *) (abase + (size_t)p*act->nb[0] + 0*act->nb[1]);
                ggml_fp16_t a16 = ggml_fp32_to_fp16(*ap_f32);
                ggml_fp16_t w16 = w_ref_f16[(size_t)p*(size_t)n + (size_t)i];
                acc += (float)(ggml_fp16_to_fp32(a16) * ggml_fp16_to_fp32(w16));
            }
            fprintf(stderr, " %f", acc);
        }
        fprintf(stderr, "\n");
        free(w_ref_f16);
        return;
    }

    if (weight->type == GGML_TYPE_Q8_0) {
        float * tmp = (float *) malloc(sizeof(float) * (size_t) k);
        if (!tmp) {
            fprintf(stderr, " malloc fail\n");
            return;
        }
        for (int i = 0; i < limit; ++i) {
            const void * row_i = (const void *) (wbase + (size_t)i*weight->nb[1]);
            dequantize_row_q8_0((const block_q8_0 *) row_i, tmp, k);
            double acc = 0.0;
            for (int64_t p = 0; p < k; ++p) {
                const float * ap = (const float *) (abase + (size_t)p*act->nb[0] + 0*act->nb[1]);
                acc += (double)tmp[p] * (double)(*ap);
            }
            fprintf(stderr, " %f", (float)acc);
        }
        fprintf(stderr, "\n");
        free(tmp);
        return;
    }

    fprintf(stderr, " (skip: w=%s)\n", ggml_type_name(weight->type));
}

// Print header information for a tensor
static void htp_debug_print_tensor_head_info(const struct ggml_tensor * tensor) {
    struct ggml_tensor * weight = tensor->src[0];
    struct ggml_tensor * act    = tensor->src[1];

    const int64_t k = weight->ne[0];
    const int64_t n = weight->ne[1];
    const int64_t m = ggml_nrows(act);

    fprintf(stderr, "-------------\n[HTP DEBUG] MUL_MAT %s  dst=%s  w=%s  a=%s  m=%ld k=%ld n=%ld\n",
            tensor->name,
            ggml_type_name(tensor->type),
            ggml_type_name(weight->type),
            ggml_type_name(act->type),
            (long)m, (long)k, (long)n);
}

// Print the first 'limit' elements of the 0th column of the output tensor
static void htp_debug_print_xpu_result(struct ggml_tensor * tensor, bool use_npu) {
    if(tensor->type != GGML_TYPE_F32) {
        fprintf(stderr, "[HTP DEBUG] out type %s not printed\n", ggml_type_name(tensor->type));
        return;
    }
    const char * obase = (const char *) tensor->data;
    struct ggml_tensor * weight = tensor->src[0];
    const int64_t n = weight->ne[1];
    const int lim = (int) (n < 8 ? n : 8);
    fprintf(stderr, use_npu ? "[HTP DEBUG] out[:,0][:8] (NPU):" : "[HTP DEBUG] out[:,0][:8] (CPU):");
    for (int i = 0; i < lim; ++i) {
        const float * op = (const float *) (obase + i*tensor->nb[0] + 0*tensor->nb[1]);
        fprintf(stderr, " %f", *op);
    }
    fprintf(stderr, "\n");
}

#ifdef __cplusplus
}
#endif
