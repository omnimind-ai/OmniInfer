#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void *mmap_manager_get_map(int fd);
int mmap_manager_put_map(int fd);
void mmap_manager_release_all();

#ifdef __cplusplus
}
#endif
