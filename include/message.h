#pragma once

#include <stddef.h>
#include <stdint.h>

struct MessageState {
  union {
    volatile uint8_t  v[8];
    volatile uint64_t d;
  };
} __attribute__((packed));

struct MessageHeader {
  struct MessageState state;
  uint32_t            checksum;
  int32_t             n_reqs;
  int32_t             req_offsets[0];  // n_reqs + 1 entries
} __attribute__((packed));

struct RequestHeader {
  int32_t state;
  int32_t type;
  uint8_t data[0];
} __attribute__((packed));

enum RequestType {
  REQUEST_TYPE_NO_OP = 0,
  REQUEST_TYPE_RPCMEM_MAP,
  REQUEST_TYPE_OP_COMPUTE,
  REQUEST_TYPE_COUNT,
};

struct RpcmemMapRequest {
  int32_t n_puts;
  int32_t n_gets;
  int32_t fds[0];
} __attribute__((packed));

struct OpComputeRequest {
  uint32_t op;
  uint8_t  payload[0];
} __attribute__((packed));

static inline size_t message_header_size(const struct MessageHeader *h) {
  return sizeof(struct MessageHeader) + (h->n_reqs + 1) * sizeof(int32_t);
}

static inline size_t message_total_size(const struct MessageHeader *h) {
  return h->req_offsets[h->n_reqs];
}

static inline size_t message_header_get_request_size(const struct MessageHeader *h, int req_idx) {
  return h->req_offsets[req_idx + 1] - h->req_offsets[req_idx];
}

static inline struct RequestHeader *message_header_get_request_ptr(struct MessageHeader *h, int req_idx) {
  return (struct RequestHeader *) ((uintptr_t) (h) + h->req_offsets[req_idx]);
}
