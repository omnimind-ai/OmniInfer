#include "dsp/vtcm_mgr.h"

#include <HAP_compute_res.h>
#include <HAP_farf.h>

#include <cstring>
#include <string>
#include <unordered_map>

namespace vtcm_manager {

uint8_t *vtcm_base           = nullptr;
uint8_t *vtcm_reserved_start = nullptr;

int vtcm_mgr_ctx_id = 0;

std::unordered_map<std::string, uint8_t *> reserved_areas;

}  // namespace vtcm_manager

extern "C" {

void vtcm_manager_setup() {
  using namespace vtcm_manager;

  int err;

  unsigned int            avail_size, total_size;
  compute_res_vtcm_page_t avail_pages, total_pages;
  err = HAP_compute_res_query_VTCM(0, &total_size, &total_pages, &avail_size, &avail_pages);
  if (err) {
    FARF(ALWAYS, "HAP_compute_res_query_VTCM failed with return code 0x%x", err);
    return;
  }
  FARF(ALWAYS, "available VTCM size: %d KiB, total VTCM size: %d KiB", avail_size / 1024, total_size / 1024);

  compute_res_attr_t req;
  HAP_compute_res_attr_init(&req);

  // NOTE(hzx): here we try to request all VTCM memory in one page
  HAP_compute_res_attr_set_vtcm_param(&req, total_size, 1);

  vtcm_mgr_ctx_id = HAP_compute_res_acquire(&req, 10000);  // timeout 10ms
  if (vtcm_mgr_ctx_id == 0) {
    FARF(ALWAYS, "%s: HAP_compute_res_acquire failed", __func__);
    return;
  }

  vtcm_base = (uint8_t *) HAP_compute_res_attr_get_vtcm_ptr(&req);
  memset(vtcm_base, 0, total_size);

  vtcm_reserved_start = vtcm_base + total_size;
}

void vtcm_manager_reset() {
  using namespace vtcm_manager;

  if (vtcm_mgr_ctx_id) {
    HAP_compute_res_release(vtcm_mgr_ctx_id);
  }
}

void *vtcm_manager_get_vtcm_base() {
  return vtcm_manager::vtcm_base;
}

void *vtcm_manager_reserve_area(const char *name, size_t size, size_t alignment) {
  using namespace vtcm_manager;

  if (!name || (alignment & (alignment - 1)) != 0) {
    return nullptr;
  }

  std::string ident = name;
  auto        it    = reserved_areas.find(ident);
  if (it != reserved_areas.end()) {
    return it->second;
  }

  uintptr_t start_val = reinterpret_cast<uintptr_t>(vtcm_reserved_start - size) & ~(alignment - 1);
  uint8_t  *new_start = reinterpret_cast<uint8_t *>(start_val);
  if (new_start <= vtcm_base) {
    return nullptr;  // no enough space left
  }

  vtcm_reserved_start   = new_start;
  reserved_areas[ident] = new_start;
  return new_start;
}

void *vtcm_manager_query_area(const char *name) {
  using namespace vtcm_manager;

  if (!name) {
    return nullptr;
  }

  std::string ident = name;
  auto        it    = reserved_areas.find(ident);
  if (it == reserved_areas.end()) {
    return nullptr;
  }
  return it->second;
}
}
