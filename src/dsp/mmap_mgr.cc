#include "dsp/mmap_mgr.h"

#include <HAP_mem.h>

#include <unordered_map>

namespace mmap_manager {

std::unordered_map<int, void *> mapping;  // fd -> addr

}

extern "C" {

void *mmap_manager_get_map(int fd) {
  auto it = mmap_manager::mapping.find(fd);
  if (it != mmap_manager::mapping.end()) {
    return it->second;
  }

  void *p;
  int   err = HAP_mmap_get(fd, &p, nullptr);
  if (err) {
    return nullptr;
  }
  return mmap_manager::mapping[fd] = p;
}

int mmap_manager_put_map(int fd) {
  auto it = mmap_manager::mapping.find(fd);
  if (it != mmap_manager::mapping.end()) {
    int ret = HAP_mmap_put(it->first);
    mmap_manager::mapping.erase(it);
    return ret;
  }
  return 0;
}

void mmap_manager_release_all() {
  for (auto [fd, _] : mmap_manager::mapping) {
    HAP_mmap_put(fd);
  }
}
}
