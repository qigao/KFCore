#include "trackers/tracker.h"

#include "tinytest.h"
#include "tracker_test_alloc.h"

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

    it("clones tracker state without sharing later resets") {
        bytetrack_config_t config = bytetrack_default_config();
        config.minimum_consecutive_frames = 1;
        bytetrack_t* tracker = bytetrack_create(&config);
        bytetrack_t* clone = NULL;
        detection_t detections[1] = {
            make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.95f),
        };
        tracked_detection_ex_t output[1];
        size_t written = 0;

        check_not_null(tracker);
        check_equal(bytetrack_update_ex(tracker, detections, 1, output, 1, &written),
                    TRACKER_STATUS_OK);
        check_equal(bytetrack_clone(tracker, &clone), TRACKER_STATUS_OK);
        check_not_null(clone);

        bytetrack_reset(tracker);
        check_equal(bytetrack_update_ex(tracker, detections, 1, output, 1, &written),
                    TRACKER_STATUS_OK);
        check_equal(output[0].tracked.tracker_id, -1);
        check_equal(bytetrack_update_ex(clone, detections, 1, output, 1, &written),
                    TRACKER_STATUS_OK);
        check_equal(output[0].tracked.tracker_id, 0);

        bytetrack_destroy(clone);
        bytetrack_destroy(tracker);
    }

    it("leaves the source untouched when clone allocation fails") {
        bytetrack_t* tracker = bytetrack_create(NULL);
        bytetrack_t* clone = (bytetrack_t*)1;

        check_not_null(tracker);
        trackers_test_alloc_fail_after(0);
        check_equal(bytetrack_clone(tracker, &clone), TRACKER_STATUS_ALLOCATION_FAILED);
        check_null(clone);
        trackers_test_alloc_reset();
        check_equal(bytetrack_clone(tracker, &clone), TRACKER_STATUS_OK);
        check_not_null(clone);

        bytetrack_destroy(clone);
        bytetrack_destroy(tracker);
    }
}
