// Communication & RPC related interfaces

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <qurt_memory.h>
#include <qurt_signal.h>
#include <qurt_thread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "dsp/hmx_mgr.h"
#include "dsp/mmap_mgr.h"
#include "dsp/op_executor.h"
#include "dsp/ops.h"
#include "dsp/power.h"
#include "dsp/vtcm_mgr.h"
#include "htp_ops.h"  // QAIC auto-generated header for FastRPC
#include "message.h"

static int dummy_handle;  // served as the global handle

struct MessageChannel {
  uint8_t      *msg;
  int           rpcmem_fd;
  size_t        max_msg_size;
  bool          msg_receiver_should_stop;
  qurt_signal_t msg_receiver_ready;
  qurt_thread_t msg_receiver_thread;
};

static struct MessageChannel global_msg_chan;

static void msg_receiver_loop(void *param) {
  struct MessageChannel *chan = (struct MessageChannel *) param;
  qurt_signal_set(&(chan->msg_receiver_ready), 1);

  const int SLEEP_TIME_US = 1;

  // TODO(hzx): using the poller thread to do computation may not be a good idea
  hmx_manager_enable_execution();

  while (1) {
    if (chan->msg_receiver_should_stop) {
      break;
    }

    struct MessageHeader *msg_hdr = (struct MessageHeader *) chan->msg;
    if (msg_hdr == NULL) {
      qurt_sleep(SLEEP_TIME_US);  // wait until shared message buffer become available
    }

    // invalidate cache
    qurt_mem_cache_clean((qurt_addr_t) msg_hdr, chan->max_msg_size, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);

    // NOTE(hzx): do we need a barrier here?
    // asm volatile("barrier" ::: "memory");

    // FIXME: memory order may not be working
    // volatile _Atomic uint64_t *d_ptr = (volatile _Atomic uint64_t *) &(msg_hdr->state.d);
    // uint64_t d_val = atomic_load_explicit(d_ptr, memory_order_acquire);

    volatile uint64_t d_val, *d_ptr = &(msg_hdr->state.d);
    asm volatile("%0 = memd_aq(%1)" : "=r"(d_val) : "r"(d_ptr) : "memory");

    // TODO(hzx): use more proper message state
    uint8_t v0 = d_val & 0xff;
    uint8_t v1 = (d_val >> 8) & 0xff;
    if (v0 == 0 || v1 != 0) {
      qurt_sleep(SLEEP_TIME_US);
      continue;
    }

    // simple checksum
    if (0) {
      uint32_t  sum   = 0;
      uint32_t *begin = ((uint32_t *) msg_hdr);
      uint32_t *end   = begin + chan->max_msg_size / 4;
      for (uint32_t *p = begin; p < end; ++p) {
        sum += *p;
      }

      if (sum != 0) {
        __builtin_trap();  // die
      }
    }

    for (int i = 0; i < msg_hdr->n_reqs; ++i) {
      struct RequestHeader *req_hdr = message_header_get_request_ptr(msg_hdr, i);
      switch (req_hdr->type) {
        case REQUEST_TYPE_OP_COMPUTE:
          {
            // TODO(hzx): use separate thread (pool) to execute op
            struct OpComputeRequest *compute_req = (struct OpComputeRequest *) req_hdr->data;

            req_hdr->state = execute_op_simple(compute_req);
          }
          break;
        case REQUEST_TYPE_RPCMEM_MAP:
          {
            struct RpcmemMapRequest *map_req = (struct RpcmemMapRequest *) req_hdr->data;
            for (int j = 0; j < map_req->n_puts; ++j) {
              mmap_manager_put_map(map_req->fds[j]);
            }
            req_hdr->state = 0;
          }
          break;
        default:
          break;
      }
    }

    // first payload cache flush
    qurt_mem_cache_clean((qurt_addr_t) msg_hdr, message_total_size(msg_hdr), QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);

    asm volatile("barrier" ::: "memory");

    // FIXME: memory order may not be working
    // atomic_uchar *v1_ptr = (atomic_uchar *) &(msg_hdr->state.v[1]);
    // atomic_store_explicit(v1_ptr, 1, memory_order_release);

    d_val = v0 | (1 << 8);
    asm volatile("memd_rl(%0):at = %1" ::"r"(d_ptr), "r"(d_val) : "memory");

    // flush cache
    qurt_mem_cache_clean((qurt_addr_t) msg_hdr, message_total_size(msg_hdr), QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);

    // TODO(hzx): estimate host's job completion time and sleep
    qurt_sleep(SLEEP_TIME_US);
  }

  hmx_manager_disable_execution();
}

// init an empty (semantically unintialized) message channel
void message_channel_init(struct MessageChannel *chan) {
  chan->msg          = NULL;
  chan->rpcmem_fd    = -1;
  chan->max_msg_size = 0;

  chan->msg_receiver_should_stop = false;
}

bool message_channel_is_active(const struct MessageChannel *chan) {
  return chan->msg != NULL;
}

int message_channel_create(struct MessageChannel *chan, int rpcmem_fd, size_t max_msg_size) {
  uint8_t *p;
  int      err = HAP_mmap_get(rpcmem_fd, (void **) &p, NULL);
  if (err) {
    FARF(ALWAYS, "%s: HAP_mmap_get failed with %x", __func__, err);
    return -1;
  }

  // clear message state
  memset(p, 0, max_msg_size);

  chan->msg_receiver_should_stop = false;
  qurt_signal_init(&(chan->msg_receiver_ready));

  const size_t stack_size = 8192;
  void        *stack      = memalign(4096, stack_size);
  if (!stack) {
    FARF(ALWAYS, "%s: failed to allocate memory for thread stack", __func__);
    qurt_signal_destroy(&(chan->msg_receiver_ready));
    return -1;
  }

  // launch message receiver thread
  qurt_thread_attr_t attr;
  qurt_thread_attr_init(&attr);
  qurt_thread_attr_set_name(&attr, "hops-msg-recv");
  qurt_thread_attr_set_priority(&attr, 64);
  qurt_thread_attr_set_stack_addr(&attr, stack);
  qurt_thread_attr_set_stack_size(&attr, stack_size);
  qurt_thread_attr_set_autostack(&attr, QURT_THREAD_AUTOSTACK_ENABLED);

  err = qurt_thread_create(&(chan->msg_receiver_thread), &attr, msg_receiver_loop, chan);
  if (err) {
    FARF(ALWAYS, "%s: qurt_thread_create failed with 0x%x", __func__, err);
    qurt_signal_destroy(&(chan->msg_receiver_ready));
    return -1;
  }

  chan->msg          = p;
  chan->rpcmem_fd    = rpcmem_fd;
  chan->max_msg_size = max_msg_size;
  // wait until msg reciever thread is ready
  qurt_signal_wait_all(&(chan->msg_receiver_ready), 1);
  return 0;
}

int message_channel_destroy(struct MessageChannel *chan) {
  if (!message_channel_is_active(chan)) {
    return 0;
  }

  // signal message receiver thread to stop
  chan->msg_receiver_should_stop = true;

  int status;
  qurt_thread_join(chan->msg_receiver_thread, &status);
  qurt_signal_destroy(&(chan->msg_receiver_ready));
  HAP_mmap_put(chan->rpcmem_fd);

  message_channel_init(chan);
  return 0;
}

// FastRPC interface
AEEResult htp_ops_open(const char *uri, remote_handle64 *handle) {
  // We may keep this function simple and leave real initialization somewhere else

  *handle = (remote_handle64) &dummy_handle;

  message_channel_init(&global_msg_chan);
  return AEE_SUCCESS;
}

// FastRPC interface
AEEResult htp_ops_close(remote_handle64 handle) {
  mmap_manager_release_all();
  message_channel_destroy(&global_msg_chan);

  hmx_manager_reset();
  vtcm_manager_reset();
  power_reset();

  return AEE_SUCCESS;
}

void init_precomputed_tables();

// FastRPC interface
AEEResult htp_ops_init_backend(remote_handle64 handle) {
  FARF(ALWAYS, "init_backend called");

  power_setup();
  vtcm_manager_setup();
  hmx_manager_setup();

  init_precomputed_tables();

  return AEE_SUCCESS;
}

// FastRPC interface
AEEResult htp_ops_create_channel(remote_handle64 handle, int32 fd, uint32 size) {
  if (message_channel_is_active(&global_msg_chan)) {
    return AEE_EALREADY;
  }

  return message_channel_create(&global_msg_chan, fd, size);
}

// FastRPC interface
AEEResult htp_ops_destroy_channel(remote_handle64 handle) {
  return message_channel_destroy(&global_msg_chan);
}

// FastRPC interface
AEEResult htp_ops_rms_norm_f32(remote_handle64 handle, int32 fd0, int32 offset0, int32 fd1, int32 offset1, int32 ne0,
                               int32 ne1) {
  int64_t t0 = HAP_perf_get_qtimer_count();

  // TODO(hzx): maybe we should cache fd -> address mapping
  uint8_t *p0, *p1;
  p0 = p1 = NULL;

  int err = HAP_mmap_get(fd0, (void **) &p0, NULL);
  if (err) {
    FARF(ALWAYS, "HAP_mmap_get failed: %d", err);
    goto bail;
  }

  err = HAP_mmap_get(fd1, (void **) &p1, NULL);
  if (err) {
    FARF(ALWAYS, "HAP_mmap_get failed: %d", err);
    goto bail;
  }

  int64_t t1 = HAP_perf_get_qtimer_count();

  const float *input      = (float *) (p1 + offset1);
  size_t       input_size = ne0 * ne1 * sizeof(float);  // This can be inaccurate
  qurt_mem_cache_clean((qurt_addr_t) input, input_size, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);

  float *output = (float *) (p0 + offset0);
  err           = hvx_rms_norm_f32(output, input, ne0, ne1);
  if (err) {
    FARF(ALWAYS, "%s: bad input or alignment", __func__);
    goto bail;
  }

  // TODO(hzx): we need a smarter way to do this
  size_t output_size = ne0 * ne1 * sizeof(float);  // This can be inaccurate
  err                = qurt_mem_cache_clean((qurt_addr_t) output, output_size, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);

  int64_t t2 = HAP_perf_get_qtimer_count();

bail:
  if (p0) {
    HAP_mmap_put(fd0);
  }
  if (p1) {
    HAP_mmap_put(fd1);
  }

  int64_t elapsed = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - t0);
  FARF(ALWAYS, "rms_norm_f32 (ne0=%d, ne1=%d) took %ld us", ne0, ne1, elapsed);
  FARF(ALWAYS, "    core + cache inv+flush: %ld us", HAP_perf_qtimer_count_to_us(t2 - t1));
  return err;
}

// FastRPC interface
AEEResult htp_ops_mat_mul_permuted_w16a32(remote_handle64 handle, int32 output_fd, int32 output_offset,
                                          int32 activation_fd, int32 activation_offset, int32 weight_fd,
                                          int32 weight_offset, int32 m, int32 k, int32 n) {
  uint8_t *p0, *p1, *p2;
  p0 = p1 = p2 = NULL;

  int err = HAP_mmap_get(output_fd, (void **) &p0, NULL);
  if (err) {
    FARF(ALWAYS, "HAP_mmap_get failed: %d", err);
    goto bail;
  }

  err = HAP_mmap_get(activation_fd, (void **) &p1, NULL);
  if (err) {
    FARF(ALWAYS, "HAP_mmap_get failed: %d", err);
    goto bail;
  }

  err = HAP_mmap_get(weight_fd, (void **) &p2, NULL);
  if (err) {
    FARF(ALWAYS, "HAP_mmap_get failed: %d", err);
    goto bail;
  }

  float        *output     = (float *) (p0 + output_offset);
  const float  *activation = (const float *) (p1 + activation_offset);
  const __fp16 *weight     = (const __fp16 *) (p2 + weight_offset);

  size_t output_size     = m * n * sizeof(float);
  size_t activation_size = m * k * sizeof(float);
  size_t weight_size     = k * n * sizeof(__fp16);

  qurt_mem_cache_clean((qurt_addr_t) activation, activation_size, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
  qurt_mem_cache_clean((qurt_addr_t) weight, weight_size, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);

  // static char print_buf[256];

  // const uint16_t *w = (const uint16_t *) weight;
  // sprintf(print_buf, "%s: weight digest %04x %04x %04x %04x | %04x %04x", __func__, w[0], w[1], w[2], w[3], w[64], w[65]);
  // FARF(ALWAYS, "%s", print_buf);

  // const float *a = activation;
  // sprintf(print_buf, "%s: activa digest %g %g %g %g", __func__, a[0], a[1], a[2], a[3]);
  // FARF(ALWAYS, "%s", print_buf);

  hmx_manager_enable_execution();
  err = hmx_mat_mul_permuted_w16a32(output, activation, weight, m, k, n);
  hmx_manager_disable_execution();

  // sprintf(print_buf, "%s: output digest %g %g %g %g", __func__, output[0], output[1], output[2], output[3]);
  // FARF(ALWAYS, "%s", print_buf);

  qurt_mem_cache_clean((qurt_addr_t) output, output_size, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);

bail:
  if (p0) {
    HAP_mmap_put(output_fd);
  }
  if (p1) {
    HAP_mmap_put(activation_fd);
  }
  if (p2) {
    HAP_mmap_put(weight_fd);
  }
  return err;
}

void internal_op_tests();

// FastRPC interface
AEEResult htp_ops_test_ops(remote_handle64 handle) {
  FARF(ALWAYS, "Op Tests!");

  internal_op_tests();

  return 0;
}
