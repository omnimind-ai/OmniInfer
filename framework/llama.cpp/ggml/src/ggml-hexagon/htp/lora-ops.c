#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"

#include <AEEStdErr.h>
#include <HAP_farf.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-base.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#define HTP_LORA_NO_CPU_FALLBACK (1U << 0)
#define HTP_LORA_SIDE_CONTIGUOUS  (1U << 1)

#if __HVX_ARCH__ < 79
#define HTP_LORA_ADD(a, b) Q6_Vqf32_vadd_Vqf32Vqf32(a, b)
#define HTP_LORA_MUL(a, b) Q6_Vqf32_vmpy_VsfVsf(a, b)
#define HTP_LORA_SCALE(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#else
#define HTP_LORA_ADD(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#define HTP_LORA_MUL(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#define HTP_LORA_SCALE(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#endif

struct htp_lora_acc_context {
    const struct htp_tensor * dst;
    const struct htp_tensor * tmp;
    const struct htp_tensor * b_pair;
    float                     scale;
    uint32_t                  side_begin[2];
    uint32_t                  side_end[2];
    uint32_t                  active_side[2];
    uint32_t                  n_active_sides;
    uint32_t                  rank;
    uint32_t                  m;
    uint32_t                  n;
    uint32_t                  padded_n;
};

static inline __attribute__((always_inline)) void lora_acc_half_tile(
        const struct htp_lora_acc_context * actx,
        uint32_t                            side,
        uint32_t                            in,
        uint32_t                            ir0,
        HVX_Vector                         b0_scaled,
        HVX_Vector                         b1_scaled,
        HVX_Vector                         b2_scaled,
        HVX_Vector                         b3_scaled,
        HVX_Vector                         b4_scaled,
        HVX_Vector                         b5_scaled,
        HVX_Vector                         b6_scaled,
        HVX_Vector                         b7_scaled) {
    float * restrict dst = (float *) (uintptr_t) actx->dst->data;
    const float * restrict tmp = (const float *) (uintptr_t) actx->tmp->data;

#define HTP_LORA_ACCUMULATE(acc, b_value, coeff_value)                           \
    HTP_LORA_ADD((acc), HTP_LORA_MUL((b_value), hvx_vec_splat_f32(coeff_value)))
#define HTP_LORA_ACCUMULATE_EIGHT(b_value, coeff_index) do {                     \
    acc0 = HTP_LORA_ACCUMULATE(acc0, b_value, coeff0[coeff_index]);              \
    acc1 = HTP_LORA_ACCUMULATE(acc1, b_value, coeff1[coeff_index]);              \
    acc2 = HTP_LORA_ACCUMULATE(acc2, b_value, coeff2[coeff_index]);              \
    acc3 = HTP_LORA_ACCUMULATE(acc3, b_value, coeff3[coeff_index]);              \
    acc4 = HTP_LORA_ACCUMULATE(acc4, b_value, coeff4[coeff_index]);              \
    acc5 = HTP_LORA_ACCUMULATE(acc5, b_value, coeff5[coeff_index]);              \
    acc6 = HTP_LORA_ACCUMULATE(acc6, b_value, coeff6[coeff_index]);              \
    acc7 = HTP_LORA_ACCUMULATE(acc7, b_value, coeff7[coeff_index]);              \
} while (0)
#define HTP_LORA_ACCUMULATE_FOUR(b_value, coeff_index) do {                      \
    acc0 = HTP_LORA_ACCUMULATE(acc0, b_value, coeff0[coeff_index]);              \
    acc1 = HTP_LORA_ACCUMULATE(acc1, b_value, coeff1[coeff_index]);              \
    acc2 = HTP_LORA_ACCUMULATE(acc2, b_value, coeff2[coeff_index]);              \
    acc3 = HTP_LORA_ACCUMULATE(acc3, b_value, coeff3[coeff_index]);              \
} while (0)
    uint32_t im = actx->side_begin[side];
    const uint32_t im_end = actx->side_end[side];
    for (; im + 8 <= im_end; im += 8) {
        float * out0 = dst + (size_t) (im + 0) * actx->n + in;
        float * out1 = dst + (size_t) (im + 1) * actx->n + in;
        float * out2 = dst + (size_t) (im + 2) * actx->n + in;
        float * out3 = dst + (size_t) (im + 3) * actx->n + in;
        float * out4 = dst + (size_t) (im + 4) * actx->n + in;
        float * out5 = dst + (size_t) (im + 5) * actx->n + in;
        float * out6 = dst + (size_t) (im + 6) * actx->n + in;
        float * out7 = dst + (size_t) (im + 7) * actx->n + in;
        const float * coeff0 = tmp + (size_t) (im + 0) * actx->rank + ir0;
        const float * coeff1 = tmp + (size_t) (im + 1) * actx->rank + ir0;
        const float * coeff2 = tmp + (size_t) (im + 2) * actx->rank + ir0;
        const float * coeff3 = tmp + (size_t) (im + 3) * actx->rank + ir0;
        const float * coeff4 = tmp + (size_t) (im + 4) * actx->rank + ir0;
        const float * coeff5 = tmp + (size_t) (im + 5) * actx->rank + ir0;
        const float * coeff6 = tmp + (size_t) (im + 6) * actx->rank + ir0;
        const float * coeff7 = tmp + (size_t) (im + 7) * actx->rank + ir0;
#if __HVX_ARCH__ < 79
        const HVX_Vector zero = Q6_V_vzero();
        HVX_Vector acc0 = Q6_Vqf32_vadd_VsfVsf(*(const HVX_UVector *) out0, zero);
        HVX_Vector acc1 = Q6_Vqf32_vadd_VsfVsf(*(const HVX_UVector *) out1, zero);
        HVX_Vector acc2 = Q6_Vqf32_vadd_VsfVsf(*(const HVX_UVector *) out2, zero);
        HVX_Vector acc3 = Q6_Vqf32_vadd_VsfVsf(*(const HVX_UVector *) out3, zero);
        HVX_Vector acc4 = Q6_Vqf32_vadd_VsfVsf(*(const HVX_UVector *) out4, zero);
        HVX_Vector acc5 = Q6_Vqf32_vadd_VsfVsf(*(const HVX_UVector *) out5, zero);
        HVX_Vector acc6 = Q6_Vqf32_vadd_VsfVsf(*(const HVX_UVector *) out6, zero);
        HVX_Vector acc7 = Q6_Vqf32_vadd_VsfVsf(*(const HVX_UVector *) out7, zero);
#else
        HVX_Vector acc0 = *(const HVX_UVector *) out0;
        HVX_Vector acc1 = *(const HVX_UVector *) out1;
        HVX_Vector acc2 = *(const HVX_UVector *) out2;
        HVX_Vector acc3 = *(const HVX_UVector *) out3;
        HVX_Vector acc4 = *(const HVX_UVector *) out4;
        HVX_Vector acc5 = *(const HVX_UVector *) out5;
        HVX_Vector acc6 = *(const HVX_UVector *) out6;
        HVX_Vector acc7 = *(const HVX_UVector *) out7;
#endif

        HTP_LORA_ACCUMULATE_EIGHT(b0_scaled, 0);
        HTP_LORA_ACCUMULATE_EIGHT(b1_scaled, 1);
        HTP_LORA_ACCUMULATE_EIGHT(b2_scaled, 2);
        HTP_LORA_ACCUMULATE_EIGHT(b3_scaled, 3);
        HTP_LORA_ACCUMULATE_EIGHT(b4_scaled, 4);
        HTP_LORA_ACCUMULATE_EIGHT(b5_scaled, 5);
        HTP_LORA_ACCUMULATE_EIGHT(b6_scaled, 6);
        HTP_LORA_ACCUMULATE_EIGHT(b7_scaled, 7);

#if __HVX_ARCH__ < 79
        acc0 = Q6_Vsf_equals_Vqf32(acc0);
        acc1 = Q6_Vsf_equals_Vqf32(acc1);
        acc2 = Q6_Vsf_equals_Vqf32(acc2);
        acc3 = Q6_Vsf_equals_Vqf32(acc3);
        acc4 = Q6_Vsf_equals_Vqf32(acc4);
        acc5 = Q6_Vsf_equals_Vqf32(acc5);
        acc6 = Q6_Vsf_equals_Vqf32(acc6);
        acc7 = Q6_Vsf_equals_Vqf32(acc7);
#endif

        *(HVX_UVector *) out0 = acc0;
        *(HVX_UVector *) out1 = acc1;
        *(HVX_UVector *) out2 = acc2;
        *(HVX_UVector *) out3 = acc3;
        *(HVX_UVector *) out4 = acc4;
        *(HVX_UVector *) out5 = acc5;
        *(HVX_UVector *) out6 = acc6;
        *(HVX_UVector *) out7 = acc7;
    }

    for (; im + 4 <= im_end; im += 4) {
        float * out0 = dst + (size_t) (im + 0) * actx->n + in;
        float * out1 = dst + (size_t) (im + 1) * actx->n + in;
        float * out2 = dst + (size_t) (im + 2) * actx->n + in;
        float * out3 = dst + (size_t) (im + 3) * actx->n + in;
        const float * coeff0 = tmp + (size_t) (im + 0) * actx->rank + ir0;
        const float * coeff1 = tmp + (size_t) (im + 1) * actx->rank + ir0;
        const float * coeff2 = tmp + (size_t) (im + 2) * actx->rank + ir0;
        const float * coeff3 = tmp + (size_t) (im + 3) * actx->rank + ir0;
#if __HVX_ARCH__ < 79
        const HVX_Vector zero = Q6_V_vzero();
        HVX_Vector acc0 = Q6_Vqf32_vadd_VsfVsf(*(const HVX_UVector *) out0, zero);
        HVX_Vector acc1 = Q6_Vqf32_vadd_VsfVsf(*(const HVX_UVector *) out1, zero);
        HVX_Vector acc2 = Q6_Vqf32_vadd_VsfVsf(*(const HVX_UVector *) out2, zero);
        HVX_Vector acc3 = Q6_Vqf32_vadd_VsfVsf(*(const HVX_UVector *) out3, zero);
#else
        HVX_Vector acc0 = *(const HVX_UVector *) out0;
        HVX_Vector acc1 = *(const HVX_UVector *) out1;
        HVX_Vector acc2 = *(const HVX_UVector *) out2;
        HVX_Vector acc3 = *(const HVX_UVector *) out3;
#endif

        HTP_LORA_ACCUMULATE_FOUR(b0_scaled, 0);
        HTP_LORA_ACCUMULATE_FOUR(b1_scaled, 1);
        HTP_LORA_ACCUMULATE_FOUR(b2_scaled, 2);
        HTP_LORA_ACCUMULATE_FOUR(b3_scaled, 3);
        HTP_LORA_ACCUMULATE_FOUR(b4_scaled, 4);
        HTP_LORA_ACCUMULATE_FOUR(b5_scaled, 5);
        HTP_LORA_ACCUMULATE_FOUR(b6_scaled, 6);
        HTP_LORA_ACCUMULATE_FOUR(b7_scaled, 7);

#if __HVX_ARCH__ < 79
        acc0 = Q6_Vsf_equals_Vqf32(acc0);
        acc1 = Q6_Vsf_equals_Vqf32(acc1);
        acc2 = Q6_Vsf_equals_Vqf32(acc2);
        acc3 = Q6_Vsf_equals_Vqf32(acc3);
#endif

        *(HVX_UVector *) out0 = acc0;
        *(HVX_UVector *) out1 = acc1;
        *(HVX_UVector *) out2 = acc2;
        *(HVX_UVector *) out3 = acc3;
    }

    for (; im < im_end; ++im) {
        float * out = dst + (size_t) im * actx->n + in;
        const float * coeff = tmp + (size_t) im * actx->rank + ir0;
#if __HVX_ARCH__ < 79
        const HVX_Vector zero = Q6_V_vzero();
        HVX_Vector acc = Q6_Vqf32_vadd_VsfVsf(*(const HVX_UVector *) out, zero);
#else
        HVX_Vector acc = *(const HVX_UVector *) out;
#endif
        acc = HTP_LORA_ACCUMULATE(acc, b0_scaled, coeff[0]);
        acc = HTP_LORA_ACCUMULATE(acc, b1_scaled, coeff[1]);
        acc = HTP_LORA_ACCUMULATE(acc, b2_scaled, coeff[2]);
        acc = HTP_LORA_ACCUMULATE(acc, b3_scaled, coeff[3]);
        acc = HTP_LORA_ACCUMULATE(acc, b4_scaled, coeff[4]);
        acc = HTP_LORA_ACCUMULATE(acc, b5_scaled, coeff[5]);
        acc = HTP_LORA_ACCUMULATE(acc, b6_scaled, coeff[6]);
        acc = HTP_LORA_ACCUMULATE(acc, b7_scaled, coeff[7]);
#if __HVX_ARCH__ < 79
        acc = Q6_Vsf_equals_Vqf32(acc);
#endif
        *(HVX_UVector *) out = acc;
    }
#undef HTP_LORA_ACCUMULATE_EIGHT
#undef HTP_LORA_ACCUMULATE_FOUR
#undef HTP_LORA_ACCUMULATE
}

static void lora_acc_tile_job(unsigned int nth, unsigned int ith, void * data) {
    struct htp_lora_acc_context * actx = (struct htp_lora_acc_context *) data;
    const _Float16 * restrict b_pair = (const _Float16 *) (uintptr_t) actx->b_pair->data;
    const HVX_Vector scale = hvx_vec_splat_f32(actx->scale);

#define HTP_LORA_LOAD_B_HALF(row, half)                                          \
    HTP_LORA_SCALE(                                                              \
        Q6_V_##half##_W(hvx_vec_f16_to_f32(                                     \
            *(const HVX_UVector *) (b + (row) * actx->padded_n))),               \
        scale)

    const uint32_t n_tiles = (actx->n + VLEN_FP16 - 1) / VLEN_FP16;
    const uint32_t n_jobs = n_tiles * actx->n_active_sides;
    for (uint32_t job = ith; job < n_jobs; job += nth) {
        const uint32_t paired = actx->n_active_sides == 2;
        const uint32_t tile = paired ? job >> 1 : job;
        const uint32_t side = paired ? job & 1 : actx->active_side[0];
        const uint32_t in = tile * VLEN_FP16;

        for (uint32_t ir0 = 0; ir0 < actx->rank; ir0 += 8) {
            const _Float16 * b = b_pair +
                ((size_t) side * actx->rank + ir0) * actx->padded_n + in;
            {
                const HVX_Vector b0 = HTP_LORA_LOAD_B_HALF(0, lo);
                const HVX_Vector b1 = HTP_LORA_LOAD_B_HALF(1, lo);
                const HVX_Vector b2 = HTP_LORA_LOAD_B_HALF(2, lo);
                const HVX_Vector b3 = HTP_LORA_LOAD_B_HALF(3, lo);
                const HVX_Vector b4 = HTP_LORA_LOAD_B_HALF(4, lo);
                const HVX_Vector b5 = HTP_LORA_LOAD_B_HALF(5, lo);
                const HVX_Vector b6 = HTP_LORA_LOAD_B_HALF(6, lo);
                const HVX_Vector b7 = HTP_LORA_LOAD_B_HALF(7, lo);

                lora_acc_half_tile(
                        actx, side, in, ir0,
                        b0, b1, b2, b3, b4, b5, b6, b7);
            }

            if (in + VLEN_FP32 < actx->n) {
                const HVX_Vector b0 = HTP_LORA_LOAD_B_HALF(0, hi);
                const HVX_Vector b1 = HTP_LORA_LOAD_B_HALF(1, hi);
                const HVX_Vector b2 = HTP_LORA_LOAD_B_HALF(2, hi);
                const HVX_Vector b3 = HTP_LORA_LOAD_B_HALF(3, hi);
                const HVX_Vector b4 = HTP_LORA_LOAD_B_HALF(4, hi);
                const HVX_Vector b5 = HTP_LORA_LOAD_B_HALF(5, hi);
                const HVX_Vector b6 = HTP_LORA_LOAD_B_HALF(6, hi);
                const HVX_Vector b7 = HTP_LORA_LOAD_B_HALF(7, hi);

                lora_acc_half_tile(
                        actx, side, in + VLEN_FP32, ir0,
                        b0, b1, b2, b3, b4, b5, b6, b7);
            }
        }
    }
#undef HTP_LORA_LOAD_B_HALF
}

static int htp_lora_valid_rank(uint32_t rank) {
    return rank == 8 || rank == 16 || rank == 24 || rank == 32;
}

static int htp_lora_is_contiguous_2d(
        const struct htp_tensor * tensor,
        uint32_t                  ne0,
        uint32_t                  ne1,
        uint32_t                  element_size) {
    return tensor && tensor->ne[0] == ne0 && tensor->ne[1] == ne1 &&
           tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
           tensor->nb[0] == element_size && tensor->nb[1] == ne0 * element_size;
}

int op_lora_acc_inplace(struct htp_ops_context * octx) {
    const struct htp_tensor * base     = octx->src[0];
    const struct htp_tensor * tmp      = octx->src[1];
    const struct htp_tensor * b_pair   = octx->src[2];
    const struct htp_tensor * row_side = octx->src[3];
    const struct htp_tensor * dst      = octx->dst;

    if (!base || !tmp || !b_pair || !row_side || !dst) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    float scale = 0.0f;
    memcpy(&scale, &octx->op_params[0], sizeof(scale));
    const uint32_t flags = (uint32_t) octx->op_params[1];
    const uint32_t n = base->ne[0];
    const uint32_t m = base->ne[1];
    const uint32_t rank = tmp->ne[0];
    const uint32_t padded_n = (n + 63)/64*64;

    if (base->type != HTP_TYPE_F32 || tmp->type != HTP_TYPE_F32 ||
        b_pair->type != HTP_TYPE_F16 || row_side->type != HTP_TYPE_I32 ||
        dst->type != HTP_TYPE_F32 || !isfinite(scale)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (!(flags & HTP_LORA_NO_CPU_FALLBACK) || !(flags & HTP_LORA_SIDE_CONTIGUOUS) ||
        (flags & ~(HTP_LORA_NO_CPU_FALLBACK | HTP_LORA_SIDE_CONTIGUOUS)) != 0 ||
        !htp_lora_valid_rank(rank) || n % VLEN_FP32 != 0 ||
        !htp_lora_is_contiguous_2d(base, n, m, sizeof(float)) ||
        !htp_lora_is_contiguous_2d(tmp, rank, m, sizeof(float)) ||
        !htp_lora_is_contiguous_2d(row_side, m, 1, sizeof(int32_t)) ||
        b_pair->ne[0] != padded_n || b_pair->ne[1] != rank ||
        b_pair->ne[2] != 2 || b_pair->ne[3] != 1 ||
        b_pair->nb[0] != sizeof(_Float16) ||
        b_pair->nb[1] != padded_n * sizeof(_Float16) ||
        b_pair->nb[2] != padded_n * rank * sizeof(_Float16) ||
        !htp_lora_is_contiguous_2d(dst, n, m, sizeof(float)) ||
        base->data != dst->data) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const int32_t * sides = (const int32_t *) (uintptr_t) row_side->data;
    uint32_t split = m;
    int32_t previous = 0;
    for (uint32_t im = 0; im < m; ++im) {
        const int32_t side = sides[im];
        if ((side != 0 && side != 1) || side < previous) {
            return HTP_STATUS_INVAL_PARAMS;
        }
        if (side == 1 && split == m) {
            split = im;
        }
        previous = side;
    }

    if ((octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) || scale == 0.0f) {
        return HTP_STATUS_OK;
    }

    struct htp_lora_acc_context actx = {
        .dst        = dst,
        .tmp        = tmp,
        .b_pair     = b_pair,
        .scale      = scale,
        .side_begin = { 0, split },
        .side_end   = { split, m },
        .active_side = { 0, 1 },
        .n_active_sides = (split > 0) + (split < m),
        .rank       = rank,
        .m          = m,
        .n          = n,
        .padded_n   = padded_n,
    };

    if (split == 0) {
        actx.active_side[0] = 1;
    }

    const uint32_t n_tiles = (n + VLEN_FP16 - 1) / VLEN_FP16;
    const uint32_t n_jobs = n_tiles * actx.n_active_sides;
    const uint32_t max_threads = octx->n_threads ? MIN(octx->n_threads, (uint32_t) MAX_NUM_WORKERS) : 1;
    const uint32_t n_threads = MIN(max_threads, n_jobs);
    const AEEResult result = worker_pool_run_func(
            octx->ctx->worker_pool, lora_acc_tile_job, &actx, n_threads);

    if (result != AEE_SUCCESS) {
        FARF(ERROR, "lora_acc_inplace: accumulation failed: 0x%x", result);
        return HTP_STATUS_INTERNAL_ERR;
    }

    return HTP_STATUS_OK;
}
