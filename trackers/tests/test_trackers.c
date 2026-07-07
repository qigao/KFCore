#include "trackers/tracker.h"

#include <math.h>
#include <stddef.h>

#include "tinytest.h"

static detection_t make_detection(float x1, float y1, float x2, float y2, float conf) {
    detection_t detection;
    detection.box.x1 = x1;
    detection.box.y1 = y1;
    detection.box.x2 = x2;
    detection.box.y2 = y2;
    detection.confidence = conf;
    detection.has_confidence = 1;
    detection.class_id = 0;
    detection.has_class_id = 0;
    return detection;
}

static detection_t make_detection_without_conf(float x1, float y1, float x2, float y2) {
    detection_t detection = make_detection(x1, y1, x2, y2, 0.0f);
    detection.has_confidence = 0;
    return detection;
}

spec("trackers c") {
    it("empty initial updates return no tracks like python trackers") {
        sort_t* sort = sort_create(NULL);
        bytetrack_t* bytetrack = bytetrack_create(NULL);
        ocsort_t* ocsort = ocsort_create(NULL);
        tracked_detection_t tracked[1];

        check_size_eq(sort_update(sort, NULL, 0, tracked, 1), 0u);
        check_size_eq(bytetrack_update(bytetrack, NULL, 0, tracked, 1), 0u);
        check_size_eq(ocsort_update(ocsort, NULL, 0, tracked, 1), 0u);

        sort_destroy(sort);
        bytetrack_destroy(bytetrack);
        ocsort_destroy(ocsort);
    }

    it("sort assigns an id on the first frame when maturity is one") {
        sort_config_t config = sort_default_config();
        config.minimum_consecutive_frames = 1;
        sort_t* tracker = sort_create(&config);
        detection_t detections[1] = {
            make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.9f),
        };
        tracked_detection_t tracked[1];

        size_t count = sort_update(tracker, detections, 1, tracked, 1);

        check_size_eq(count, 1u);
        check(tracked[0].tracker_id >= 0);
        sort_destroy(tracker);
    }

    it("sort preserves ids across matching frames") {
        sort_config_t config = sort_default_config();
        config.minimum_consecutive_frames = 1;
        sort_t* tracker = sort_create(&config);
        detection_t frame1[1] = {
            make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.9f),
        };
        detection_t frame2[1] = {
            make_detection(0.5f, 0.0f, 10.5f, 10.0f, 0.9f),
        };
        tracked_detection_t tracked1[1];
        tracked_detection_t tracked2[1];

        size_t count1 = sort_update(tracker, frame1, 1, tracked1, 1);
        size_t count2 = sort_update(tracker, frame2, 1, tracked2, 1);

        check_size_eq(count1, 1u);
        check_size_eq(count2, 1u);
        check(tracked1[0].tracker_id >= 0);
        check_int_eq(tracked1[0].tracker_id, tracked2[0].tracker_id);
        sort_destroy(tracker);
    }

    it("sort accepts detections without confidence and keeps them tentative before maturity") {
        sort_t* tracker = sort_create(NULL);
        detection_t frame1[1] = {
            make_detection_without_conf(50.0f, 20.0f, 70.0f, 60.0f),
        };
        detection_t frame2[1] = {
            make_detection_without_conf(51.0f, 20.0f, 71.0f, 60.0f),
        };
        tracked_detection_t tracked1[1];
        tracked_detection_t tracked2[1];

        size_t count1 = sort_update(tracker, frame1, 1, tracked1, 1);
        size_t count2 = sort_update(tracker, frame2, 1, tracked2, 1);

        check_size_eq(count1, 1u);
        check_size_eq(count2, 1u);
        check_int_eq(tracked1[0].tracker_id, -1);
        check_int_eq(tracked2[0].tracker_id, -1);
        sort_destroy(tracker);
    }

    it("sort reset clears state and restarts ids") {
        sort_config_t config = sort_default_config();
        config.minimum_consecutive_frames = 1;
        sort_t* tracker = sort_create(&config);
        detection_t detections[1] = {
            make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.9f),
        };
        tracked_detection_t tracked[1];

        (void)sort_update(tracker, detections, 1, tracked, 1);
        sort_reset(tracker);
        size_t count = sort_update(tracker, detections, 1, tracked, 1);

        check_size_eq(count, 1u);
        check_int_eq(tracked[0].tracker_id, 0);
        sort_destroy(tracker);
    }

    it("bytetrack keeps unmatched detections between thresholds") {
        bytetrack_t* tracker = bytetrack_create(NULL);
        detection_t detections[1] = {
            make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.65f),
        };
        tracked_detection_t tracked[1];

        size_t count = bytetrack_update(tracker, detections, 1, tracked, 1);

        check_size_eq(count, 1u);
        check_int_eq(tracked[0].tracker_id, -1);
        bytetrack_destroy(tracker);
    }

    it("bytetrack promotes mature ids on the second match") {
        bytetrack_t* tracker = bytetrack_create(NULL);
        detection_t frame1[1] = {
            make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.95f),
        };
        detection_t frame2[1] = {
            make_detection(0.5f, 0.0f, 10.5f, 10.0f, 0.95f),
        };
        tracked_detection_t tracked1[1];
        tracked_detection_t tracked2[1];

        size_t count1 = bytetrack_update(tracker, frame1, 1, tracked1, 1);
        size_t count2 = bytetrack_update(tracker, frame2, 1, tracked2, 1);

        check_size_eq(count1, 1u);
        check_size_eq(count2, 1u);
        check_int_eq(tracked1[0].tracker_id, -1);
        check(tracked2[0].tracker_id >= 0);
        bytetrack_destroy(tracker);
    }

    it("bytetrack recovers an existing track with a low-confidence detection") {
        bytetrack_t* tracker = bytetrack_create(NULL);
        detection_t frame1[1] = {
            make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.95f),
        };
        detection_t frame2[1] = {
            make_detection(0.5f, 0.0f, 10.5f, 10.0f, 0.55f),
        };
        detection_t frame3[1] = {
            make_detection(1.0f, 0.0f, 11.0f, 10.0f, 0.95f),
        };
        tracked_detection_t tracked1[1];
        tracked_detection_t tracked2[1];
        tracked_detection_t tracked3[1];

        size_t count1 = bytetrack_update(tracker, frame1, 1, tracked1, 1);
        size_t count2 = bytetrack_update(tracker, frame2, 1, tracked2, 1);
        size_t count3 = bytetrack_update(tracker, frame3, 1, tracked3, 1);

        check_size_eq(count1, 1u);
        check_size_eq(count2, 1u);
        check_size_eq(count3, 1u);
        check_int_eq(tracked1[0].tracker_id, -1);
        check(tracked2[0].tracker_id >= 0);
        check_int_eq(tracked2[0].tracker_id, tracked3[0].tracker_id);
        bytetrack_destroy(tracker);
    }

    it("sort allocates ids in tracker order when detections arrive reversed") {
        sort_config_t config = sort_default_config();
        config.minimum_consecutive_frames = 2;
        sort_t* tracker = sort_create(&config);
        detection_t frame1[2] = {
            make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.95f),
            make_detection(100.0f, 0.0f, 110.0f, 10.0f, 0.95f),
        };
        detection_t frame2[2] = {
            make_detection(100.5f, 0.0f, 110.5f, 10.0f, 0.95f),
            make_detection(0.5f, 0.0f, 10.5f, 10.0f, 0.95f),
        };
        tracked_detection_t tracked[2];

        (void)sort_update(tracker, frame1, 2, tracked, 2);
        size_t count = sort_update(tracker, frame2, 2, tracked, 2);

        check_size_eq(count, 2u);
        check(tracked[0].tracker_id >= 0);
        check(tracked[1].tracker_id >= 0);
        check(tracked[0].tracker_id != tracked[1].tracker_id);
        check(tracked[0].tracker_id > tracked[1].tracker_id);
        sort_destroy(tracker);
    }

    it("bytetrack emits matched detections in tracker order when detections arrive reversed") {
        bytetrack_config_t config = bytetrack_default_config();
        config.minimum_consecutive_frames = 2;
        bytetrack_t* tracker = bytetrack_create(&config);
        detection_t frame1[2] = {
            make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.95f),
            make_detection(100.0f, 0.0f, 110.0f, 10.0f, 0.95f),
        };
        detection_t frame2[2] = {
            make_detection(100.5f, 0.0f, 110.5f, 10.0f, 0.95f),
            make_detection(0.5f, 0.0f, 10.5f, 10.0f, 0.95f),
        };
        tracked_detection_t tracked[2];

        (void)bytetrack_update(tracker, frame1, 2, tracked, 2);
        size_t count = bytetrack_update(tracker, frame2, 2, tracked, 2);

        check_size_eq(count, 2u);
        check(fabsf(tracked[0].detection.box.x1 - 0.5f) < 1e-5f);
        check(tracked[0].tracker_id >= 0);
        check(fabsf(tracked[1].detection.box.x1 - 100.5f) < 1e-5f);
        check(tracked[1].tracker_id >= 0);
        check(tracked[0].tracker_id != tracked[1].tracker_id);
        check(tracked[0].tracker_id < tracked[1].tracker_id);
        bytetrack_destroy(tracker);
    }

    it("cbiou promotes mature ids on a buffered low-score recovery match") {
        cbiou_config_t config = cbiou_default_config();
        config.minimum_consecutive_frames = 2;
        config.minimum_biou_threshold = 0.3f;
        config.first_buffer_ratio = 0.0f;
        config.second_buffer_ratio = 1.0f;
        config.fuse_detection_score = 0;
        cbiou_t* tracker = cbiou_create(&config);
        detection_t frame1[1] = {
            make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.95f),
        };
        detection_t frame2[1] = {
            make_detection(8.0f, 0.0f, 18.0f, 10.0f, 0.3f),
        };
        tracked_detection_t tracked1[1];
        tracked_detection_t tracked2[1];

        size_t count1 = cbiou_update(tracker, frame1, 1, tracked1, 1);
        size_t count2 = cbiou_update(tracker, frame2, 1, tracked2, 1);

        check_size_eq(count1, 1u);
        check_size_eq(count2, 1u);
        check_int_eq(tracked1[0].tracker_id, -1);
        check(tracked2[0].tracker_id >= 0);
        cbiou_destroy(tracker);
    }

    it("cbiou reset clears state and restarts ids") {
        cbiou_config_t config = cbiou_default_config();
        config.minimum_consecutive_frames = 1;
        config.fuse_detection_score = 0;
        cbiou_t* tracker = cbiou_create(&config);
        detection_t frame1[1] = {
            make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.95f),
        };
        detection_t frame2[1] = {
            make_detection(0.3f, 0.0f, 10.3f, 10.0f, 0.95f),
        };
        tracked_detection_t tracked[1];

        (void)cbiou_update(tracker, frame1, 1, tracked, 1);
        (void)cbiou_update(tracker, frame2, 1, tracked, 1);
        cbiou_reset(tracker);
        (void)cbiou_update(tracker, frame1, 1, tracked, 1);
        size_t count = cbiou_update(tracker, frame2, 1, tracked, 1);

        check_size_eq(count, 1u);
        check_int_eq(tracked[0].tracker_id, 0);
        cbiou_destroy(tracker);
    }

    it("ocsort accepts detections without confidence") {
        ocsort_t* tracker = ocsort_create(NULL);
        detection_t detections[1] = {
            make_detection_without_conf(0.0f, 0.0f, 10.0f, 10.0f),
        };
        tracked_detection_t tracked[1];

        size_t count = ocsort_update(tracker, detections, 1, tracked, 1);

        check_size_eq(count, 1u);
        check_int_eq(tracked[0].tracker_id, -1);
        ocsort_destroy(tracker);
    }

    it("ocsort filters detections below the high-confidence threshold") {
        ocsort_t* tracker = ocsort_create(NULL);
        detection_t detections[1] = {
            make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.5f),
        };
        tracked_detection_t tracked[1];

        size_t count = ocsort_update(tracker, detections, 1, tracked, 1);

        check_size_eq(count, 0u);
        ocsort_destroy(tracker);
    }

    it("ocsort promotes and then preserves ids across matching frames") {
        ocsort_config_t config = ocsort_default_config();
        config.minimum_consecutive_frames = 1;
        ocsort_t* tracker = ocsort_create(&config);
        detection_t frame1[1] = {
            make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.95f),
        };
        detection_t frame2[1] = {
            make_detection(0.3f, 0.0f, 10.3f, 10.0f, 0.95f),
        };
        tracked_detection_t tracked1[1];
        tracked_detection_t tracked2[1];

        size_t count1 = ocsort_update(tracker, frame1, 1, tracked1, 1);
        size_t count2 = ocsort_update(tracker, frame2, 1, tracked2, 1);

        check_size_eq(count1, 1u);
        check_size_eq(count2, 1u);
        check_int_eq(tracked1[0].tracker_id, -1);
        check(tracked2[0].tracker_id >= 0);
        ocsort_destroy(tracker);
    }

    it("ocsort emits matched detections in tracker order when detections arrive reversed") {
        ocsort_config_t config = ocsort_default_config();
        config.minimum_consecutive_frames = 1;
        ocsort_t* tracker = ocsort_create(&config);
        detection_t frame1[2] = {
            make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.95f),
            make_detection(100.0f, 0.0f, 110.0f, 10.0f, 0.95f),
        };
        detection_t frame2[2] = {
            make_detection(100.5f, 0.0f, 110.5f, 10.0f, 0.95f),
            make_detection(0.5f, 0.0f, 10.5f, 10.0f, 0.95f),
        };
        tracked_detection_t tracked[2];

        (void)ocsort_update(tracker, frame1, 2, tracked, 2);
        size_t count = ocsort_update(tracker, frame2, 2, tracked, 2);

        check_size_eq(count, 2u);
        check(fabsf(tracked[0].detection.box.x1 - 0.5f) < 1e-5f);
        check(tracked[0].tracker_id >= 0);
        check(fabsf(tracked[1].detection.box.x1 - 100.5f) < 1e-5f);
        check(tracked[1].tracker_id >= 0);
        check(tracked[0].tracker_id != tracked[1].tracker_id);
        check(tracked[0].tracker_id < tracked[1].tracker_id);
        ocsort_destroy(tracker);
    }
}
