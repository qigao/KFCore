#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void trackers_test_alloc_reset(void);
void trackers_test_alloc_fail_after(size_t successful_allocations);
size_t trackers_test_alloc_count(void);
void* trackers_test_malloc(size_t size);
void* trackers_test_calloc(size_t count, size_t size);
void* trackers_test_realloc(void* data, size_t size);

#ifdef __cplusplus
}
#endif
