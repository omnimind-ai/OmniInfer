// This source file mainly contains matrix multiplication implementation used in micro-benchmarks

#include <stdbool.h>

#include "dsp/hmx_utils.h"
#include "dsp/hvx_internal.h"
#include "dsp/ops.h"
#include "dsp/utils.h"
#include "dsp/worker_pool.h"

// Assume tiles already satisfy FP16 HMX Crouton memory layout
int hmx_mat_mul_fp16_core(__fp16 *restrict __vtcm c, const __fp16 *restrict __vtcm a, const __fp16 *restrict __vtcm b,
                          __fp16 *restrict __vtcm scales, int M, int K, int N) {
  if (M % 32 != 0 || K % 32 != 0 || N % 32 != 0) {
    return -1;
  }

  // number of tiles
  int mt = M / 32;
  int nt = N / 32;
  int kt = K / 32;

  hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));
  hmx_set_output_scales(scales);

  for (int i = 0; i < mt; ++i) {
    for (int j = 0; j < nt; ++j) {
      const __fp16 *a_tiles = a + i * kt * HMX_FP16_TILE_N_ELMS;
      const __fp16 *b_tiles = b + j * kt * HMX_FP16_TILE_N_ELMS;
      __fp16       *c_tile  = c + (i * nt + j) * HMX_FP16_TILE_N_ELMS;

      for (int k = 0; k < kt; k += 32) {
        hmx_load_tiles_fp16(a_tiles + k * HMX_FP16_TILE_N_ELMS, b_tiles + k * HMX_FP16_TILE_N_ELMS, smin(kt - k, 32));
      }
      hmx_consume_accumulator_fp16(c_tile);
    }
  }
  return 0;
}

// This assumes all operands are located in VTCM. Accumulator type: qf16
int hvx_mat_mul_fp16_core(__fp16 *restrict __vtcm c, const __fp16 *restrict __vtcm a, const __fp16 *restrict __vtcm b,
                          int M, int K, int N) {
  if (K % 64 != 0 || N % 64 != 0) {
    return -1;
  }

  const bool scalar_broadcast = false;

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; j += 64) {
      // a slice of matrix A
      uint16_t _Alignas(VLEN) tmp[VLEN / sizeof(uint16_t)];
      HVX_Vector v_tmp;

      HVX_Vector v_sum = Q6_V_vzero();
#pragma unroll 4
      for (int k = 0; k < K; ++k) {
        // option 1: use intermediate vars (this might be optimized into vextract) + scalar vsplat
        // option 2: shift + vlut broadcast
        if (k % 64 == 0) {
          v_tmp = vmem(a + i * K + k);
          if (scalar_broadcast) {
            vmem(tmp) = v_tmp;
          }
        }

        HVX_Vector v_a;
        if (scalar_broadcast) {
          v_a = Q6_Vh_vsplat_R(tmp[k % 64]);
        } else {
          HVX_Vector v_lut = Q6_V_vror_VR(v_tmp, (k % 64) * sizeof(__fp16));
          v_a              = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_vzero(), v_lut, 0));
        }

        HVX_Vector v_b = vmem(b + k * N + j);  // assume B is row-major
        v_sum          = Q6_Vqf16_vadd_Vqf16Vqf16(v_sum, Q6_Vqf16_vmpy_VhfVhf(v_a, v_b));
      }
      vmem(c + i * N + j) = Q6_Vhf_equals_Vqf16(v_sum);
    }
  }
  return 0;
}

// This assumes all operands are located in VTCM. Accumulator type: qf32
int hvx_mat_mul_fp32_core(float *restrict __vtcm c, const float *restrict __vtcm a, const float *restrict __vtcm b,
                          int M, int K, int N) {
  if (K % 32 != 0 || N % 32 != 0) {
    return -1;
  }

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; j += 32) {
      // a slice of matrix A
      uint32_t _Alignas(VLEN) tmp[VLEN / sizeof(uint32_t)];
      HVX_Vector v_tmp;

      HVX_Vector v_sum = Q6_V_vzero();
#pragma unroll 4
      for (int k = 0; k < K; ++k) {
        if (k % 32 == 0) {
          v_tmp     = vmem(a + i * K + k);
          vmem(tmp) = v_tmp;
        }

        HVX_Vector v_a = Q6_V_vsplat_R(tmp[k % 32]);
        HVX_Vector v_b = vmem(b + k * N + j);  // assume B is row-major
        v_sum          = Q6_Vqf32_vadd_Vqf32Vqf32(v_sum, Q6_Vqf32_vmpy_VsfVsf(v_a, v_b));
      }
      vmem(c + i * N + j) = Q6_Vsf_equals_Vqf32(v_sum);
    }
  }
  return 0;
}

int hvx_mat_mul_int16_core(int16_t *restrict __vtcm c, const int16_t *restrict __vtcm a,
                           const int16_t *restrict __vtcm b, int M, int K, int N) {
  if (K % 64 != 0 || N % 64 != 0) {
    return -1;
  }

  const bool scalar_broadcast = false;

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; j += 64) {
      // a slice of matrix A
      uint16_t _Alignas(VLEN) tmp[VLEN / sizeof(uint16_t)];
      HVX_Vector v_tmp;

      HVX_VectorPair vp_sum = Q6_W_vzero();
#pragma unroll 4
      for (int k = 0; k < K; ++k) {
        // option 1: use intermediate vars (this might be optimized into vextract) + scalar vsplat
        // option 2: shift + vlut broadcast
        if (k % 64 == 0) {
          v_tmp = vmem(a + i * K + k);
          if (scalar_broadcast) {
            vmem(tmp) = v_tmp;
          }
        }

        HVX_Vector v_a;
        if (scalar_broadcast) {
          v_a = Q6_Vh_vsplat_R(tmp[k % 64]);
        } else {
          HVX_Vector v_lut = Q6_V_vror_VR(v_tmp, (k % 64) * sizeof(__fp16));
          v_a              = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_vzero(), v_lut, 0));
        }

        HVX_Vector v_b = vmem(b + k * N + j);  // assume B is row-major

        vp_sum = Q6_Ww_vmpyacc_WwVhVh(vp_sum, v_a, v_b);
      }
      vmem(c + i * N + j) = Q6_Vh_vasr_VwVwR_sat(Q6_V_hi_W(vp_sum), Q6_V_lo_W(vp_sum), 15);
    }
  }
  return 0;
}

int hvx_mat_mul_int32_core(int32_t *restrict __vtcm c, const int32_t *restrict __vtcm a,
                           const int32_t *restrict __vtcm b, int M, int K, int N) {
  if (K % 32 != 0 || N % 32 != 0) {
    return -1;
  }

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; j += 32) {
      uint32_t _Alignas(VLEN) tmp[VLEN / sizeof(uint32_t)];

      HVX_Vector v_sum = Q6_V_vzero();
#pragma unroll 4
      for (int k = 0; k < K; ++k) {
        if (k % 32 == 0) {
          vmem(tmp) = vmem(a + i * K + k);
        }

        HVX_Vector v_a = Q6_V_vsplat_R(tmp[k % 32]);
        HVX_Vector v_b = vmem(b + k * N + j);
        v_sum          = Q6_Vw_vadd_VwVw_sat(v_sum, Q6_Vw_vmpy_VwVw_s1_sat(v_a, v_b));
      }
      vmem(c + i * N + j) = v_sum;
    }
  }
  return 0;
}

typedef struct {
  EXPAND_COMMON_TASK_STATE_MEMBERS
  __fp16       *c;
  const __fp16 *a, *b;
  int           K, N;
} hvx_mat_mul_fp16_task_state_t;

static void hvx_mat_mul_fp16_worker_loop(void *data, int _worker_index) {
  (void) _worker_index;
  hvx_mat_mul_fp16_task_state_t *state = (hvx_mat_mul_fp16_task_state_t *) data;

  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    if (task_id >= state->n_tasks) {
      break;
    }
    int    chunk_idx  = task_id * state->n_chunks_per_task;
    size_t chunk_size = smin(state->n_tot_chunks - chunk_idx, state->n_chunks_per_task);

    // one chunk: one row of A
    const __fp16 *a = state->a + chunk_idx * state->K;
    __fp16       *c = state->c + chunk_idx * state->N;
    hvx_mat_mul_fp16_core(c, a, state->b, chunk_size, state->K, state->N);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

int hvx_mat_mul_fp16_core_mt(__fp16 *restrict __vtcm c, const __fp16 *restrict __vtcm a,
                             const __fp16 *restrict __vtcm b, int M, int K, int N, int n_threads) {
  if (K % 64 != 0 || N % 64 != 0) {
    return -1;
  }

  int    n_workers         = n_threads;
  size_t n_tot_chunks      = M;
  size_t n_chunks_per_task = ceil_div(n_tot_chunks, n_workers);  // static partitioning

  hvx_mat_mul_fp16_task_state_t state;
  INIT_COMMON_TASK_STATE_MEMBERS(state, n_tot_chunks, n_chunks_per_task);
  state.a = a;
  state.b = b;
  state.c = c;
  state.K = K;
  state.N = N;

  worker_pool_job_t job = {
    .fptr = hvx_mat_mul_fp16_worker_loop,
    .dptr = &state,
  };

  worker_pool_synctoken_init(&(state.sync_ctx), n_workers);
  for (int i = 0; i < n_workers; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
  return 0;
}

typedef struct {
  EXPAND_COMMON_TASK_STATE_MEMBERS
  int16_t       *c;
  const int16_t *a, *b;
  int            K, N;
} hvx_mat_mul_int16_task_state_t;

static void hvx_mat_mul_int16_worker_loop(void *data, int _worker_index) {
  (void) _worker_index;
  hvx_mat_mul_int16_task_state_t *state = (hvx_mat_mul_int16_task_state_t *) data;

  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    if (task_id >= state->n_tasks) {
      break;
    }
    int    chunk_idx  = task_id * state->n_chunks_per_task;
    size_t chunk_size = smin(state->n_tot_chunks - chunk_idx, state->n_chunks_per_task);

    // one chunk: one row of A
    const int16_t *a = state->a + chunk_idx * state->K;
    int16_t       *c = state->c + chunk_idx * state->N;
    hvx_mat_mul_int16_core(c, a, state->b, chunk_size, state->K, state->N);
  }
  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

int hvx_mat_mul_int16_core_mt(int16_t *restrict __vtcm c, const int16_t *restrict __vtcm a,
                              const int16_t *restrict __vtcm b, int M, int K, int N, int n_threads) {
  if (K % 64 != 0 || N % 64 != 0) {
    return -1;
  }

  int    n_workers         = n_threads;
  size_t n_tot_chunks      = M;
  size_t n_chunks_per_task = ceil_div(n_tot_chunks, n_workers);  // static partitioning

  hvx_mat_mul_int16_task_state_t state;
  INIT_COMMON_TASK_STATE_MEMBERS(state, n_tot_chunks, n_chunks_per_task);
  state.a = a;
  state.b = b;
  state.c = c;
  state.K = K;
  state.N = N;

  worker_pool_job_t job = {
    .fptr = hvx_mat_mul_int16_worker_loop,
    .dptr = &state,
  };

  worker_pool_synctoken_init(&(state.sync_ctx), n_workers);
  for (int i = 0; i < n_workers; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
  return 0;
}
