#include "trackers/tracker.h"

#include <limits.h>
#include <math.h>

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

suite("tracker update status") {
    it("rolls back SORT after a late Kalman failure") {
        sort_config_t config = sort_default_config();
        config.minimum_consecutive_frames = 2;
        sort_t* tracker = sort_create(&config);
        detection_t detections[2] = {
            make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.95f),
            make_detection(100.0f, 0.0f, 110.0f, 10.0f, 0.95f),
        };
        tracked_detection_ex_t output[2];
        size_t written = 0;

        check_not_null(tracker);
        trackers_test_alloc_reset();

        check_equal(sort_update_ex(tracker, detections, 2, output, 2, &written),
                    TRACKER_STATUS_OK);
        check_equal(written, (size_t)2);
        check_equal(output[0].tracked.tracker_id, -1);
        check_equal(output[1].tracked.tracker_id, -1);

        trackers_test_kalman_fail_after(3);
        check_equal(sort_update_ex(tracker, detections, 2, output, 2, &written),
                    TRACKER_STATUS_NUMERICAL_FAILURE);
        check_equal(written, (size_t)0);

        check_equal(sort_update_ex(tracker, detections, 2, output, 2, &written),
                    TRACKER_STATUS_OK);
        check_equal(written, (size_t)2);
        check_equal(output[0].detection_index, (size_t)0);
        check_equal(output[1].detection_index, (size_t)1);
        check_equal(output[0].tracked.tracker_id, 0);
        check_equal(output[1].tracked.tracker_id, 1);

        sort_destroy(tracker);
    }

    it("rolls back CBIoU after a late Kalman failure") {
        cbiou_config_t config = cbiou_default_config();
        config.minimum_consecutive_frames = 2;
        cbiou_t* tracker = cbiou_create(&config);
        detection_t detections[2] = {
            make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.95f),
            make_detection(100.0f, 0.0f, 110.0f, 10.0f, 0.95f),
        };
        tracked_detection_ex_t output[2];
        size_t written = 0;

        check_not_null(tracker);
        trackers_test_alloc_reset();

        check_equal(cbiou_update_ex(tracker, detections, 2, output, 2, &written),
                    TRACKER_STATUS_OK);
        check_equal(written, (size_t)2);
        check_equal(output[0].tracked.tracker_id, -1);
        check_equal(output[1].tracked.tracker_id, -1);

        trackers_test_kalman_fail_after(3);
        check_equal(cbiou_update_ex(tracker, detections, 2, output, 2, &written),
                    TRACKER_STATUS_NUMERICAL_FAILURE);
        check_equal(written, (size_t)0);

        check_equal(cbiou_update_ex(tracker, detections, 2, output, 2, &written),
                    TRACKER_STATUS_OK);
        check_equal(written, (size_t)2);
        check_equal(output[0].detection_index, (size_t)0);
        check_equal(output[1].detection_index, (size_t)1);
        check_equal(output[0].tracked.tracker_id, 0);
        check_equal(output[1].tracked.tracker_id, 1);

        cbiou_destroy(tracker);
    }

    it("rolls back OC-SORT after a late Kalman failure") {
        ocsort_config_t config = ocsort_default_config();
        config.minimum_consecutive_frames = 1;
        ocsort_t* tracker = ocsort_create(&config);
        detection_t detections[2] = {
            make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.95f),
            make_detection(100.0f, 0.0f, 110.0f, 10.0f, 0.95f),
        };
        tracked_detection_ex_t output[2];
        size_t written = 0;

        check_not_null(tracker);
        trackers_test_alloc_reset();

        check_equal(ocsort_update_ex(tracker, detections, 2, output, 2, &written),
                    TRACKER_STATUS_OK);
        check_equal(written, (size_t)2);
        check_equal(output[0].tracked.tracker_id, -1);
        check_equal(output[1].tracked.tracker_id, -1);

        trackers_test_kalman_fail_after(3);
        check_equal(ocsort_update_ex(tracker, detections, 2, output, 2, &written),
                    TRACKER_STATUS_NUMERICAL_FAILURE);
        check_equal(written, (size_t)0);

        check_equal(ocsort_update_ex(tracker, detections, 2, output, 2, &written),
                    TRACKER_STATUS_OK);
        check_equal(written, (size_t)2);
        check_equal(output[0].detection_index, (size_t)0);
        check_equal(output[1].detection_index, (size_t)1);
        check_equal(output[0].tracked.tracker_id, 0);
        check_equal(output[1].tracked.tracker_id, 1);

        ocsort_destroy(tracker);
    }

    it("rolls back OC-SORT when unfreeze replay fails") {
        ocsort_config_t config = ocsort_default_config();
        config.minimum_consecutive_frames = 1;
        ocsort_t* tracker = ocsort_create(&config);
        detection_t detection =
            make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.95f);
        tracked_detection_ex_t output[1];
        size_t written = 0;

        check_not_null(tracker);
        trackers_test_alloc_reset();

        check_equal(ocsort_update_ex(tracker, &detection, 1, output, 1, &written),
                    TRACKER_STATUS_OK);
        check_equal(written, (size_t)1);

        check_equal(ocsort_update_ex(tracker, NULL, 0, NULL, 0, &written),
                    TRACKER_STATUS_OK);
        check_equal(written, (size_t)0);

        trackers_test_kalman_fail_after(1);
        check_equal(ocsort_update_ex(tracker, &detection, 1, output, 1, &written),
                    TRACKER_STATUS_NUMERICAL_FAILURE);
        check_equal(written, (size_t)0);

        check_equal(ocsort_update_ex(tracker, &detection, 1, output, 1, &written),
                    TRACKER_STATUS_OK);
        check_equal(written, (size_t)1);
        check_equal(output[0].detection_index, (size_t)0);
        check_equal(output[0].tracked.tracker_id, 0);

        ocsort_destroy(tracker);
    }

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

    it("rolls back the frame when a Kalman operation fails") {
        bytetrack_config_t config = bytetrack_default_config();
        config.minimum_consecutive_frames = 1;
        bytetrack_t* tracker = bytetrack_create(&config);
        detection_t detection =
            make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.95f);
        tracked_detection_ex_t output[1];
        size_t written = 0;

        check_not_null(tracker);
        trackers_test_alloc_reset();

        check_equal(bytetrack_update_ex(tracker, &detection, 1, output, 1, &written),
                    TRACKER_STATUS_OK);
        check_equal(written, (size_t)1);
        check_equal(output[0].tracked.tracker_id, -1);

        trackers_test_kalman_fail_next();
        check_equal(bytetrack_update_ex(tracker, &detection, 1, output, 1, &written),
                    TRACKER_STATUS_NUMERICAL_FAILURE);
        check_equal(written, (size_t)0);

        check_equal(bytetrack_update_ex(tracker, &detection, 1, output, 1, &written),
                    TRACKER_STATUS_OK);
        check_equal(written, (size_t)1);
        check_equal(output[0].tracked.tracker_id, 0);

        bytetrack_destroy(tracker);
    }

    it("keeps a clone usable after its source is destroyed") {
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

        bytetrack_destroy(tracker);
        tracker = NULL;
        check_equal(bytetrack_update_ex(clone, detections, 1, output, 1, &written),
                    TRACKER_STATUS_OK);
        check_equal(output[0].tracked.tracker_id, 0);

        bytetrack_destroy(clone);
    }

    it("leaves a non-empty source usable when tracks clone allocation fails") {
        bytetrack_config_t config = bytetrack_default_config();
        detection_t detection = make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.95f);
        tracked_detection_ex_t output[1];
        size_t written = 0;
        config.minimum_consecutive_frames = 1;
        bytetrack_t* tracker = bytetrack_create(&config);
        bytetrack_t* clone = (bytetrack_t*)1;

        check_not_null(tracker);
        check_equal(bytetrack_update_ex(tracker, &detection, 1, output, 1, &written),
                    TRACKER_STATUS_OK);
        trackers_test_alloc_fail_after(1);
        check_equal(bytetrack_clone(tracker, &clone), TRACKER_STATUS_ALLOCATION_FAILED);
        check_null(clone);
        trackers_test_alloc_reset();
        check_equal(bytetrack_update_ex(tracker, &detection, 1, output, 1, &written),
                    TRACKER_STATUS_OK);
        check_equal(output[0].tracked.tracker_id, 0);

        bytetrack_destroy(tracker);
    }

    it("rejects scaled lost buffers that cannot be represented as int") {
        bytetrack_config_t config = bytetrack_default_config();
        config.lost_track_buffer = INT_MAX;
        config.frame_rate = 30.0f;
        check_null(bytetrack_create(&config));

        config.frame_rate = nextafterf(30.0f, 0.0f);
        bytetrack_t* tracker = bytetrack_create(&config);
        check_not_null(tracker);
        bytetrack_destroy(tracker);
    }
}
