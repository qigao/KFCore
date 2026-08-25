#include "kfcore/yolo/tracking.hpp"
#include "tinytest.hpp"

#include <limits>

using namespace kfcore::yolo;

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
}
