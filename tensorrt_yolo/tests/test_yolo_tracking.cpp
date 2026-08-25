#include "kfcore/yolo/tracking.hpp"
#include "tinytest.hpp"
#include "tracker_test_alloc.h"

#include <climits>
#include <limits>

using namespace kfcore::yolo;

namespace {

DetectionFrame frame_for(std::int32_t class_id = 0) {
    return {100, 100, {{{10, 10, 20, 20}, 0.95f, class_id}}};
}

ByteTrackOptions immediate_options() {
    ByteTrackOptions options;
    options.minimum_consecutive_frames = 1;
    return options;
}

}  // namespace

spec("YOLO ByteTrack session") {
    it("isolates overlapping detections by class and preserves input order") {
        ByteTrackOptions options;
        options.minimum_consecutive_frames = 1;
        ByteTrackSession session(options);
        DetectionFrame frame{640, 480, {
            {{10, 10, 50, 50}, 0.95f, 7},
            {{10, 10, 50, 50}, 0.95f, 2},
        }};

        (void)session.update(frame);
        TrackFrame result = session.update(frame);

        check(result.detections.size() == (size_t)2);
        check(result.detections[0].detection.class_id == 7);
        check(result.detections[1].detection.class_id == 2);
        check(result.detections[0].track_id.has_value());
        check(result.detections[1].track_id.has_value());
        check(*result.detections[0].track_id != *result.detections[1].track_id);
    }

    it("advances absent classes on empty frames") {
        ByteTrackOptions options;
        options.minimum_consecutive_frames = 2;
        options.lost_track_buffer = 1;
        ByteTrackSession session(options);
        (void)session.update({640, 480, {{{0, 0, 10, 10}, 0.95f, 0}}});
        TrackFrame mature = session.update(
            {640, 480, {{{0, 0, 10, 10}, 0.95f, 0}}});
        check(*mature.detections[0].track_id == (std::uint64_t)0);
        (void)session.update({640, 480, {}});
        TrackFrame tentative = session.update(
            {640, 480, {{{0, 0, 10, 10}, 0.95f, 0}}});
        check(!tentative.detections[0].track_id.has_value());
        TrackFrame result = session.update(
            {640, 480, {{{0, 0, 10, 10}, 0.95f, 0}}});
        check(result.detections[0].track_id.has_value());
        check(*result.detections[0].track_id == (std::uint64_t)1);
    }

    it("rejects non-finite boxes without mutating the session") {
        ByteTrackSession session;
        DetectionFrame invalid{640, 480, {
            {{0, 0, std::numeric_limits<float>::quiet_NaN(), 10}, 0.9f, 0},
        }};
        check_throws_as(session.update(invalid), YoloError);
        check_nothrow(session.update({640, 480, {}}));
    }

    it("enforces configured resource limits before advancing a tracker") {
        ByteTrackOptions options;
        options.minimum_consecutive_frames = 1;
        options.max_detections_per_frame = 1;
        options.max_class_trackers = 1;
        ByteTrackSession session(options);
        DetectionFrame too_many{640, 480, {
            {{0, 0, 10, 10}, 0.95f, 0},
            {{20, 20, 30, 30}, 0.95f, 0},
        }};

        check_throws_as(session.update(too_many), YoloError);
        TrackFrame tentative = session.update(
            {640, 480, {{{0, 0, 10, 10}, 0.95f, 0}}});
        check(!tentative.detections[0].track_id.has_value());
        TrackFrame mature = session.update(
            {640, 480, {{{0, 0, 10, 10}, 0.95f, 0}}});
        check(*mature.detections[0].track_id == (std::uint64_t)0);
        check_throws_as(
            session.update({640, 480, {{{0, 0, 10, 10}, 0.95f, 1}}}),
            YoloError);
    }

    it("rejects every invalid option including unsafe lost-buffer scaling") {
        ByteTrackOptions options = immediate_options();

        options.lost_track_buffer = -1;
        check_throws_as(ByteTrackSession(options), YoloError);
        options = immediate_options();
        options.frame_rate = 0.0f;
        check_throws_as(ByteTrackSession(options), YoloError);
        options = immediate_options();
        options.frame_rate = std::numeric_limits<float>::infinity();
        check_throws_as(ByteTrackSession(options), YoloError);
        options = immediate_options();
        options.frame_rate = (std::numeric_limits<float>::max)();
        check_throws_as(ByteTrackSession(options), YoloError);
        options = immediate_options();
        options.minimum_consecutive_frames = 0;
        check_throws_as(ByteTrackSession(options), YoloError);
        options = immediate_options();
        options.track_activation_threshold = -0.1f;
        check_throws_as(ByteTrackSession(options), YoloError);
        options = immediate_options();
        options.minimum_iou_threshold = 1.1f;
        check_throws_as(ByteTrackSession(options), YoloError);
        options = immediate_options();
        options.high_conf_det_threshold = std::numeric_limits<float>::quiet_NaN();
        check_throws_as(ByteTrackSession(options), YoloError);
        options = immediate_options();
        options.max_detections_per_frame = 0;
        check_throws_as(ByteTrackSession(options), YoloError);
        options = immediate_options();
        options.max_class_trackers = 0;
        check_throws_as(ByteTrackSession(options), YoloError);
        options = immediate_options();
        options.lost_track_buffer = INT_MAX;
        options.frame_rate = 30.0f;
        check_nothrow(ByteTrackSession(options));
    }

    it("rejects invalid frames before advancing any tracker") {
        ByteTrackSession session(immediate_options());
        ByteTrackSession control(immediate_options());
        (void)session.update(frame_for());
        (void)control.update(frame_for());

        const DetectionFrame invalid_frames[] = {
            {0, 100, {}},
            {100, 0, {}},
            {100, 100, {{{10, 10, 20, 20}, -0.01f, 0}}},
            {100, 100, {{{10, 10, 20, 20}, 1.01f, 0}}},
            {100, 100, {{{10, 10, 20, 20}, std::numeric_limits<float>::quiet_NaN(), 0}}},
            {100, 100, {{{20, 10, 20, 20}, 0.95f, 0}}},
            {100, 100, {{{21, 10, 20, 20}, 0.95f, 0}}},
            {100, 100, {{{10, 20, 20, 20}, 0.95f, 0}}},
            {100, 100, {{{10, 21, 20, 20}, 0.95f, 0}}},
            {100, 100, {{{-1, 10, 20, 20}, 0.95f, 0}}},
            {100, 100, {{{10, 10, 101, 20}, 0.95f, 0}}},
            {100, 100, {{{10, -1, 20, 20}, 0.95f, 0}}},
            {100, 100, {{{10, 10, 20, 101}, 0.95f, 0}}},
        };
        for (const DetectionFrame& invalid : invalid_frames) {
            check_throws_as(session.update(invalid), YoloError);
        }

        TrackFrame actual = session.update(frame_for());
        TrackFrame expected = control.update(frame_for());
        check(actual.detections[0].track_id == expected.detections[0].track_id);
    }

    it("does not commit a multi-class frame when the second class allocation fails") {
        ByteTrackOptions options = immediate_options();
        options.minimum_consecutive_frames = 3;
        ByteTrackSession session(options);
        ByteTrackSession control(options);
        DetectionFrame classes{100, 100, {
            {{10, 10, 20, 20}, 0.95f, 0},
            {{30, 30, 40, 40}, 0.95f, 1},
        }};
        (void)session.update(classes);
        (void)control.update(classes);

        trackers_test_alloc_fail_after(16);
        check_throws_as(session.update(classes), YoloError);
        trackers_test_alloc_reset();

        TrackFrame actual = session.update(classes);
        TrackFrame expected = control.update(classes);
        check(actual.detections[0].track_id == expected.detections[0].track_id);
        check(actual.detections[1].track_id == expected.detections[1].track_id);
    }

    it("does not reserve a class when tracker construction fails") {
        ByteTrackOptions options = immediate_options();
        options.max_class_trackers = 1;
        ByteTrackSession session(options);

        trackers_test_alloc_fail_after(0);
        check_throws_as(session.update(frame_for(0)), YoloError);
        trackers_test_alloc_reset();
        check_nothrow(session.update(frame_for(1)));
        TrackFrame mature = session.update(frame_for(1));
        check(mature.detections[0].track_id.has_value());
    }

    it("supports moved-from sessions and reset starts a new ID epoch") {
        ByteTrackOptions options = immediate_options();
        options.max_class_trackers = 1;
        ByteTrackSession source(options);
        ByteTrackSession session(std::move(source));
        source.reset();
        check_throws_as(source.update(frame_for()), YoloError);

        (void)session.update(frame_for(0));
        TrackFrame before_reset = session.update(frame_for(0));
        check(*before_reset.detections[0].track_id == (std::uint64_t)0);
        session.reset();
        check_nothrow(session.update(frame_for(1)));
        TrackFrame different_class = session.update(frame_for(1));
        check(*different_class.detections[0].track_id == (UINT64_C(1) << 32U));
        session.reset();
        (void)session.update(frame_for(0));
        TrackFrame reused_epoch = session.update(frame_for(0));
        check(*reused_epoch.detections[0].track_id == (std::uint64_t)0);
    }

    it("encodes negative class IDs with the specified unsigned bit pattern") {
        ByteTrackSession session(immediate_options());
        (void)session.update(frame_for(-1));
        TrackFrame result = session.update(frame_for(-1));
        check(*result.detections[0].track_id == UINT64_C(0xffffffff00000000));
    }
}
