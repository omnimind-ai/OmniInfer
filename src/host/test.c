#include <math.h>
#include <remote.h>
#include <rpcmem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "host/session.h"
#include "htp_ops.h"  // auto-generated
#include "message.h"
#include "op_reg.h"

static inline int64_t get_time_us() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000L + ts.tv_nsec / 1000;
}

static inline int align_up(size_t size, size_t align) {
  return (size + align - 1) / align * align;
}

static inline double rand_01() {
  return ((double) rand()) / RAND_MAX;
}

// assert p_buf, p_fd and size are always valid
int alloc_shared_mem_buf(void **p_buf, int *p_fd, size_t size) {
  void *buf = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, size);
  if (!buf) {
    fprintf(stderr, "alloc_shared_mem_buf: rpcmem_alloc failed\n");
    return -1;
  }

  int fd = rpcmem_to_fd(buf);
  if (fd < 0) {
    fprintf(stderr, "alloc_shared_mem_buf: rpcmem_to_fd failed\n");
    return -1;
  }

  // map buffer to the DSP
  int err = fastrpc_mmap(CDSP_DOMAIN_ID, fd, buf, 0, size, FASTRPC_MAP_FD);
  if (err) {
    fprintf(stderr, "alloc_shared_mem_buf: fastrpc_mmap failed, err: %d\n", err);
    return -1;
  }

  *p_buf = buf;
  *p_fd  = fd;
  return 0;
}

void free_shared_mem_buf(void *buf, int fd, size_t size) {
  fastrpc_munmap(CDSP_DOMAIN_ID, fd, buf, size);
  rpcmem_free(buf);
}

static void rms_norm_f32_ref(float *dst, const float *src, int ne0, int ne1) {
  const float eps = 1e-5;

  for (int j = 0; j < ne1; ++j) {
    const float *x = src + j * ne0;
    float       *y = dst + j * ne0;

    float sum = 0;
    for (int i = 0; i < ne0; ++i) {
      sum += x[i] * x[i];
    }

    float mean  = sum / ne0;
    float scale = 1.0f / sqrtf(mean + eps);
    for (int i = 0; i < ne0; ++i) {
      y[i] = x[i] * scale;
    }

    printf("%s: sum: %.5f mean: %.5f scale: %.5f\n", __func__, sum, mean, scale);
  }
}

static void test_rms_norm_f32_rpc(remote_handle64 handle, int ne0) {
  float *src, *dsp_dst, *ref_dst;
  int    fd_src, fd_dst;

  int err, passed = 0;

  src = dsp_dst = ref_dst = NULL;
  size_t size             = align_up(ne0 * sizeof(float), 128);

  if (alloc_shared_mem_buf((void **) &src, &fd_src, size)) {
    goto end;
  }
  if (alloc_shared_mem_buf((void **) &dsp_dst, &fd_dst, size)) {
    goto end;
  }
  ref_dst = (float *) malloc(size);

  // fill data, [0, 20000] -> [-20, 20]
  for (int i = 0; i < ne0; ++i) {
    src[i] = (rand() % 20000) * 2e-3f - 20.0f;
  }

  int64_t t0             = get_time_us();
  err                    = htp_ops_rms_norm_f32(handle, fd_dst, 0, fd_src, 0, ne0, 1);
  int64_t rpc_elapsed_us = get_time_us() - t0;
  fprintf(stderr, "rms_norm_f32 RPC took %ld us\n", rpc_elapsed_us);

  if (err != 0) {
    fprintf(stderr, "%s: RPC failed with %x\n", __func__, err);
    goto end;
  }
  rms_norm_f32_ref(ref_dst, src, ne0, 1);

  int   n_failed = 0;
  float tol      = 1e-5;
  for (int i = 0; i < ne0; ++i) {
    if (fabs(ref_dst[i] - dsp_dst[i]) > tol) {
      n_failed++;
      if (n_failed < 16) {
        fprintf(stderr, "%s: index %d, ref val=%.5f, dsp val=%.5f\n", __func__, i, ref_dst[i], dsp_dst[i]);
      }
    }
  }
  passed = (n_failed == 0);

end:
  if (src) {
    free_shared_mem_buf(src, fd_src, size);
  }
  if (dsp_dst) {
    free_shared_mem_buf(dsp_dst, fd_dst, size);
  }
  if (ref_dst) {
    free(ref_dst);
  }

  fprintf(stderr, passed ? "%s passed\n" : "%s failed\n", __func__);
  return;
}

static void test_rms_norm_f32_chan(void *chan, int ne0) {
  struct MessageHeader *msg = (struct MessageHeader *) chan;

  float *src, *dsp_dst, *ref_dst;
  int    fd_src, fd_dst;

  int err, passed = 0;

  src = dsp_dst = ref_dst = NULL;
  size_t size             = align_up(ne0 * sizeof(float), 128);

  if (alloc_shared_mem_buf((void **) &src, &fd_src, size)) {
    goto end;
  }
  if (alloc_shared_mem_buf((void **) &dsp_dst, &fd_dst, size)) {
    goto end;
  }
  ref_dst = (float *) malloc(size);

  // fill data, [0, 20000] -> [-20, 20]
  for (int i = 0; i < ne0; ++i) {
    src[i] = (rand() % 20000) * 2e-3f - 20.0f;
  }

  {
    struct RequestHeader req_hdr = {
      .state = 0,
      .type  = REQUEST_TYPE_OP_COMPUTE,
    };
    struct OpComputeRequest compute_req = {
      .op = HTP_OPS_RMS_NORM_F32,
    };
    struct RmsNormF32Params params = {
      .dst = { .fd = fd_dst, .offset = 0, },
      .src = { .fd = fd_src, .offset = 0, },
      .ne0 = ne0,
      .ne1 = 1,
    };

    size_t req_size     = sizeof(req_hdr) + sizeof(compute_req) + sizeof(params);
    msg->state.d        = 0;
    msg->n_reqs         = 1;
    msg->req_offsets[0] = message_header_size(msg);
    msg->req_offsets[1] = msg->req_offsets[0] + req_size;

    uint8_t *p                  = (uint8_t *) message_header_get_request_ptr(msg, 0);
    *(struct RequestHeader *) p = req_hdr;
    p += sizeof(struct RequestHeader);
    *(struct OpComputeRequest *) p = compute_req;
    p += sizeof(struct OpComputeRequest);
    *(struct RmsNormF32Params *) p = params;
    p += sizeof(struct RmsNormF32Params);
  }

  int64_t t0      = get_time_us();
  msg->state.v[0] = 1;
  while (msg->state.v[1] != 1) {
    // usleep(10);
  }
  int64_t chan_elapsed_us = get_time_us() - t0;
  fprintf(stderr, "rms_norm_f32 CHAN took %ld us\n", chan_elapsed_us);

  err = message_header_get_request_ptr(msg, 0)->state;
  if (err != 0) {
    fprintf(stderr, "%s: CHAN failed with %x\n", __func__, err);
    goto end;
  }
  rms_norm_f32_ref(ref_dst, src, ne0, 1);

  int   n_failed = 0;
  float tol      = 1e-5;
  for (int i = 0; i < ne0; ++i) {
    if (fabs(ref_dst[i] - dsp_dst[i]) > tol) {
      n_failed++;
      if (n_failed < 16) {
        fprintf(stderr, "%s: index %d, ref val=%.5f, dsp val=%.5f\n", __func__, i, ref_dst[i], dsp_dst[i]);
      }
    }
  }
  passed = (n_failed == 0);

  // extra test: trigger DSP-side mapping reclaimation
  // fprintf(stderr, "manually unmap fd %d, %d\n", fd_dst, fd_src);
  // fastrpc_munmap(CDSP_DOMAIN_ID, fd_dst, NULL, 0);
  // fastrpc_munmap(CDSP_DOMAIN_ID, fd_src, NULL, 0);
  {
    struct RequestHeader req_hdr = {
      .state = 0,
      .type  = REQUEST_TYPE_RPCMEM_MAP,
    };
    struct RpcmemMapRequest map_req = {
      .n_puts = 2,
      .n_gets = 0,
    };

    size_t req_size     = sizeof(req_hdr) + sizeof(map_req) + 2 * sizeof(int);
    msg->state.d        = 0;
    msg->n_reqs         = 1;
    msg->req_offsets[0] = message_header_size(msg);
    msg->req_offsets[1] = msg->req_offsets[0] + req_size;

    uint8_t *p                  = (uint8_t *) message_header_get_request_ptr(msg, 0);
    *(struct RequestHeader *) p = req_hdr;
    p += sizeof(struct RequestHeader);
    *(struct RpcmemMapRequest *) p = map_req;
    p += sizeof(struct RpcmemMapRequest);

    // fill in fd data
    *(int *) p = fd_dst;
    p += sizeof(int);
    *(int *) p = fd_src;
    p += sizeof(int);
  }

  msg->state.v[0] = 1;
  while (msg->state.v[1] != 1) {
    usleep(10);
  }

end:
  if (src) {
    free_shared_mem_buf(src, fd_src, size);
  }
  if (dsp_dst) {
    free_shared_mem_buf(dsp_dst, fd_dst, size);
  }
  if (ref_dst) {
    free(ref_dst);
  }

  fprintf(stderr, passed ? "%s passed\n" : "%s failed\n", __func__);
}

static void test_mat_mul_rpc(remote_handle64 handle) {
  float *activation, *output;
  __fp16 *weight;

  int output_fd, activation_fd, weight_fd;

  int m = 1;
  int k = 1024;
  // int n = 608; // 576 | 608
  int n = 1024;

  alloc_shared_mem_buf((void **) &output, &output_fd, m * n * sizeof(float));
  alloc_shared_mem_buf((void **) &activation, &activation_fd, m * k * sizeof(float));
  alloc_shared_mem_buf((void **) &weight, &weight_fd, k * n * sizeof(__fp16));

  float *weight_ref = (float *) malloc(n * k * sizeof(float));
  float *output_ref = (float *) malloc(m * n * sizeof(float));
  memset(output_ref, 0, m * n * sizeof(float));

  __fp16 *output_f16 = (__fp16 *) malloc(m * n * sizeof(__fp16));
  memset(output_f16, 0, m * n * sizeof(__fp16));

  float *output_mix = (float *) malloc(m * n * sizeof(float));
  memset(output_mix, 0, m * n * sizeof(float));

  for (int i = 0; i < m; ++i)
    for (int j = 0; j < k; ++j)
      activation[i * k + j] = rand_01();
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      float x = rand_01();

      int i0 = i / 32, i1 = i % 32;
      int j0 = j / 32, j1 = j % 32;

      int tile_idx = j0 * (k / 32) + i0;
      __fp16 *tile = weight + tile_idx * 1024;
      tile[(i1 & ~1) * 32 + j1 * 2 + (i1 & 1)] = (__fp16) x;
      weight_ref[i * n + j] = x;
    }
  }

  htp_ops_mat_mul_permuted_w16a32(handle, output_fd, 0, activation_fd, 0, weight_fd, 0, m, k, n);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int l = 0; l < k; ++l) {
        output_ref[i * n + j] += activation[i * k + l] * weight_ref[l * n + j];
        output_f16[i * n + j] += (__fp16)(((__fp16) activation[i * k + l]) * ((__fp16) weight_ref[l * n + j]));
        output_mix[i * n + j] += (float)((__fp16) activation[i * k + l] * ((__fp16) weight_ref[l * n + j]));
      }
    }
  }

  for (int i = 0; i < m * n; ++i)
    printf("#%d hmx: %g, f32: %g, f16: %g, mix: %g\n", i, output[i], output_ref[i], output_f16[i], output_mix[i]);

  free(weight_ref);
  free(output_ref);
  free(output_f16);
  free(output_mix);

  free_shared_mem_buf(output, output_fd, m * n * sizeof(float));
  free_shared_mem_buf(activation, activation_fd, m * k * sizeof(float));
  free_shared_mem_buf(weight, weight_fd, k * n * sizeof(__fp16));
}

int main(int argc, char **argv) {
  int err = open_dsp_session(CDSP_DOMAIN_ID, 1);
  if (err != 0) {
    fprintf(stderr, "Open DSP session failed\n");
    return 1;
  }

  init_htp_backend();

  // test_mat_mul_rpc(get_global_handle());

  htp_ops_test_ops(get_global_handle());

  /*
  test_rms_norm_f32_rpc(get_global_handle(), 60000);

  void        *chan;
  int          chan_fd;
  const size_t max_msg_size = 4096;

  err = alloc_shared_mem_buf(&chan, &chan_fd, max_msg_size);
  if (err) {
    fprintf(stderr, "Cannot allocate rpcmem for message channel\n");
    goto skip1;
  }

  err = htp_ops_create_channel(get_global_handle(), chan_fd, max_msg_size);
  if (err) {
    fprintf(stderr, "Create channel failed\n");
    goto skip2;
  }

  test_rms_norm_f32_chan(chan, 60000);

  htp_ops_destroy_channel(get_global_handle());

skip2:
  free_shared_mem_buf(chan, chan_fd, max_msg_size);
  */

skip1:
  close_dsp_session();
  return 0;
}
