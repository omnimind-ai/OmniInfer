#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void vtcm_manager_setup();
void vtcm_manager_reset();

void *vtcm_manager_get_vtcm_base();

void *vtcm_manager_reserve_area(const char *name, size_t size, size_t alignment);
void *vtcm_manager_query_area(const char *name);

static inline uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
  uint8_t *p = *vtcm_ptr;
  *vtcm_ptr += size;
  return p;
}

#ifdef __cplusplus
}
#endif
