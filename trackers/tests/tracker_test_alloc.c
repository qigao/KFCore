#include "tracker_test_alloc.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static size_t trackers_test_alloc_count_value;
static size_t trackers_test_alloc_limit = SIZE_MAX;
static size_t trackers_test_kalman_count_value;
static size_t trackers_test_kalman_limit = SIZE_MAX;

void trackers_test_alloc_reset(void) {
    trackers_test_alloc_count_value = 0;
    trackers_test_alloc_limit = SIZE_MAX;
    trackers_test_kalman_count_value = 0;
    trackers_test_kalman_limit = SIZE_MAX;
}

void trackers_test_alloc_fail_after(size_t successful_allocations) {
    trackers_test_alloc_count_value = 0;
    trackers_test_alloc_limit = successful_allocations;
}

size_t trackers_test_alloc_count(void) {
    return trackers_test_alloc_count_value;
}

void trackers_test_kalman_fail_after(size_t successful_operations) {
    trackers_test_kalman_count_value = 0;
    trackers_test_kalman_limit = successful_operations;
}

void trackers_test_kalman_fail_next(void) {
    trackers_test_kalman_fail_after(0);
}

int trackers_test_kalman_should_fail(void) {
    if (trackers_test_kalman_count_value >= trackers_test_kalman_limit) {
        trackers_test_kalman_count_value = 0;
        trackers_test_kalman_limit = SIZE_MAX;
        return 1;
    }
    ++trackers_test_kalman_count_value;
    return 0;
}

static int trackers_test_alloc_should_fail(void) {
    if (trackers_test_alloc_count_value >= trackers_test_alloc_limit) {
        return 1;
    }
    ++trackers_test_alloc_count_value;
    return 0;
}

void* trackers_test_malloc(size_t size) {
    return trackers_test_alloc_should_fail() ? NULL : malloc(size);
}

void* trackers_test_calloc(size_t count, size_t size) {
    if (count != 0 && size > SIZE_MAX / count) {
        return NULL;
    }
    void* data = trackers_test_malloc(count * size);
    if (data) {
        memset(data, 0, count * size);
    }
    return data;
}

void* trackers_test_realloc(void* data, size_t size) {
    return trackers_test_alloc_should_fail() ? NULL : realloc(data, size);
}
