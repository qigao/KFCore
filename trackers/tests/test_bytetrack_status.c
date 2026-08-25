#include "trackers/tracker.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "tinytest.h"
#include "tracker_test_alloc.h"

static size_t trackers_test_alloc_count;
static size_t trackers_test_alloc_limit = SIZE_MAX;

void trackers_test_alloc_reset(void) {
    trackers_test_alloc_count = 0;
    trackers_test_alloc_limit = SIZE_MAX;
}

void trackers_test_alloc_fail_after(size_t successful_allocations) {
    trackers_test_alloc_count = 0;
    trackers_test_alloc_limit = successful_allocations;
}

static int trackers_test_alloc_should_fail(void) {
    if (trackers_test_alloc_count >= trackers_test_alloc_limit) {
        return 1;
    }
    ++trackers_test_alloc_count;
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

static detection_t make_detection(float x1, float y1, float x2, float y2, float confidence) {
    detection_t detection;
    detection.box.x1 = x1;
    detection.box.y1 = y1;
    detection.box.x2 = x2;
    detection.box.y2 = y2;
    detection.confidence = confidence;
    detection.has_confidence = 1;
    detection.class_id = 0;
    detection.has_class_id = 0;
    return detection;
}

suite("bytetrack update status") {
    it("does not advance state when preparation allocation fails") {
        bytetrack_config_t config = bytetrack_default_config();
        config.minimum_consecutive_frames = 1;
        bytetrack_t* tracker = bytetrack_create(&config);
        detection_t detections[1] = {
            make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.95f),
        };
        tracked_detection_ex_t output[1];
        size_t written = 99;

        check_not_null(tracker);
        trackers_test_alloc_fail_after(0);
        check_equal(bytetrack_update_ex(tracker, detections, 1, output, 1, &written),
                    TRACKER_STATUS_ALLOCATION_FAILED);
        check_equal(written, (size_t)0);

        trackers_test_alloc_reset();
        check_equal(bytetrack_update_ex(tracker, detections, 1, output, 1, &written),
                    TRACKER_STATUS_OK);
        check_equal(written, (size_t)1);
        check_equal(output[0].tracked.tracker_id, -1);
        check_equal(bytetrack_update_ex(tracker, detections, 1, output, 1, &written),
                    TRACKER_STATUS_OK);
        check_equal(output[0].tracked.tracker_id, 0);
        bytetrack_destroy(tracker);
    }
}
