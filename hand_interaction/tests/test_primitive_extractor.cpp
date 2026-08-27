#include "kfcore/hand_interaction/primitive_extractor.hpp"
#include "tinytest.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>

namespace
{

using kfcore::hand_interaction::GestureFrameContext;
using kfcore::hand_interaction::HandPrimitiveExtractor;
using kfcore::hand_interaction::HandPrimitiveOptions;
using kfcore::thig::Observation;
using kfcore::vision_models::Gesture;
using kfcore::vision_models::HandFrame;
using kfcore::vision_models::HandResult;

constexpr int kImageWidth  = 640;
constexpr int kImageHeight = 480;

GestureFrameContext frame_context(std::uint64_t serial, int elapsed_ms = 0)
{
    return { serial,
             std::chrono::steady_clock::time_point {} +
                 std::chrono::milliseconds(elapsed_ms),
             kImageWidth,
             kImageHeight };
}

void set_point(HandResult& hand, std::size_t index, float x, float y)
{
    hand.landmarks[index] = { x, y, 0.0F };
}

HandResult base_hand(int track_id = 0)
{
    HandResult hand;
    hand.track_id              = track_id;
    hand.palm.confidence       = 0.98F;
    hand.landmark_confidence   = 0.98F;
    hand.palm.box              = { 40.0F, 40.0F, 160.0F, 160.0F };
    for (auto& landmark : hand.landmarks)
    {
        landmark = { 100.0F, 180.0F, 0.0F };
    }
    set_point(hand, 0, 100.0F, 220.0F);
    set_point(hand, 5, 75.0F, 170.0F);
    set_point(hand, 9, 100.0F, 165.0F);
    set_point(hand, 13, 125.0F, 170.0F);
    set_point(hand, 17, 145.0F, 180.0F);
    return hand;
}

HandResult v_hand(int track_id = 0)
{
    HandResult hand = base_hand(track_id);
    set_point(hand, 6, 75.0F, 130.0F);
    set_point(hand, 8, 60.0F, 70.0F);
    set_point(hand, 10, 100.0F, 125.0F);
    set_point(hand, 12, 115.0F, 65.0F);
    set_point(hand, 14, 125.0F, 150.0F);
    set_point(hand, 16, 125.0F, 175.0F);
    set_point(hand, 18, 145.0F, 160.0F);
    set_point(hand, 20, 142.0F, 185.0F);
    return hand;
}

HandResult ok_hand(int track_id = 0)
{
    HandResult hand = base_hand(track_id);
    hand.palm.box = { 40.0F, 40.0F, 120.0F, 120.0F };
    set_point(hand, 0, 100.0F, 210.0F);
    set_point(hand, 2, 70.0F, 175.0F);
    set_point(hand, 4, 72.0F, 100.0F);
    set_point(hand, 5, 80.0F, 150.0F);
    set_point(hand, 8, 74.0F, 102.0F);
    set_point(hand, 9, 110.0F, 145.0F);
    set_point(hand, 10, 115.0F, 100.0F);
    set_point(hand, 12, 115.0F, 45.0F);
    set_point(hand, 13, 135.0F, 155.0F);
    set_point(hand, 14, 138.0F, 110.0F);
    set_point(hand, 16, 142.0F, 60.0F);
    set_point(hand, 17, 155.0F, 165.0F);
    set_point(hand, 18, 158.0F, 125.0F);
    set_point(hand, 20, 165.0F, 80.0F);
    return hand;
}

void translate_hand(HandResult& hand, float dx, float dy)
{
    hand.palm.box.x += dx;
    hand.palm.box.y += dy;
    for (auto& point : hand.landmarks)
    {
        point.x += dx;
        point.y += dy;
    }
}

void scale_hand(HandResult& hand, float factor, float center_x = 100.0F,
                float center_y = 170.0F)
{
    for (auto& point : hand.landmarks)
    {
        point.x = center_x + (point.x - center_x) * factor;
        point.y = center_y + (point.y - center_y) * factor;
    }
    hand.palm.box.width *= factor;
    hand.palm.box.height *= factor;
}

void rotate_hand(HandResult& hand, float degrees, float center_x = 100.0F,
                 float center_y = 170.0F)
{
    const float radians = degrees * 3.14159265358979323846F / 180.0F;
    const float cosine = std::cos(radians);
    const float sine = std::sin(radians);
    for (auto& point : hand.landmarks)
    {
        const float x = point.x - center_x;
        const float y = point.y - center_y;
        point.x = center_x + x * cosine - y * sine;
        point.y = center_y + x * sine + y * cosine;
    }
}

HandPrimitiveOptions immediate_options()
{
    HandPrimitiveOptions options;
    options.motion.movement_window_samples = 3;
    options.motion.stationary_samples = 3;
    options.spatial.minimum_samples = 3;
    return options;
}

const Observation* find_relation(const std::vector<Observation>& observations,
                                 const std::string& relation)
{
    const auto found = std::find_if(
        observations.begin(), observations.end(), [&](const Observation& item) {
            return item.relation == relation;
        });
    return found == observations.end() ? nullptr : &*found;
}

} // namespace

spec("hand primitive extractor")
{
    it("accepts KFCore track zero and maps Closed to Shape Fist")
    {
        HandPrimitiveExtractor extractor;
        HandFrame frame;
        frame.hands.push_back(base_hand(0));
        frame.hands[0].gesture = Gesture::Closed;

        const auto result = extractor.process(frame, frame_context(1));

        check_size(result.hands, 1);
        check(result.hands[0].raw_track_id == 0);
        check(result.hands[0].canonical_id == 1);
        const Observation* shape = find_relation(result.observations, "Shape Fist");
        check_not_null(shape);
        check(shape->source.kind == "hand");
        check(shape->source.id == 1);
    }

    it("publishes V from landmark geometry instead of the model label")
    {
        HandPrimitiveExtractor extractor;
        HandFrame frame;
        frame.hands.push_back(v_hand());
        frame.hands[0].gesture = Gesture::Unknown;

        const auto result = extractor.process(frame, frame_context(1));

        const Observation* shape = find_relation(result.observations, "Shape V");
        check_not_null(shape);
        check(shape->producer == "shape_geometry");
    }

    it("publishes OK only when thumb-index contact and three extensions match")
    {
        HandPrimitiveExtractor extractor;
        HandFrame frame;
        frame.hands.push_back(ok_hand());

        const auto matched = extractor.process(frame, frame_context(1));
        check_not_null(find_relation(matched.observations, "Pose OK"));

        extractor.reset();
        frame.hands[0].landmarks[8].x = 150.0F;
        const auto rejected = extractor.process(frame, frame_context(2));
        check_not_null(find_relation(rejected.observations, "Pose Not OK"));
        check_null(find_relation(rejected.observations, "Pose OK"));
    }

    it("classifies index extension and press in palm-normalized geometry")
    {
        HandPrimitiveExtractor extractor;
        HandFrame frame;
        frame.hands.push_back(v_hand());

        const auto extended = extractor.process(frame, frame_context(1));
        check_not_null(find_relation(extended.observations, "Index Extended"));

        extractor.reset();
        set_point(frame.hands[0], 8, 90.0F, 160.0F);
        const auto pressed = extractor.process(frame, frame_context(2));
        check_not_null(find_relation(pressed.observations, "Index Pressed"));
    }

    it("keeps a canonical identity when ByteTrack recreates a nearby raw id")
    {
        HandPrimitiveExtractor extractor;
        HandFrame first_frame;
        first_frame.hands.push_back(base_hand(4));
        const auto first = extractor.process(first_frame, frame_context(1));

        HandFrame second_frame;
        second_frame.hands.push_back(base_hand(19));
        translate_hand(second_frame.hands[0], 8.0F, 4.0F);
        const auto second = extractor.process(second_frame, frame_context(2, 33));

        check(first.hands[0].canonical_id > 0);
        check(second.hands[0].canonical_id == first.hands[0].canonical_id);
    }

    it("withholds canonical identities while two crossing hands are ambiguous")
    {
        HandPrimitiveExtractor extractor;
        const auto two_hands = [](int left_id, float left_x, int right_id,
                                  float right_x) {
            HandFrame frame;
            frame.hands.push_back(base_hand(left_id));
            frame.hands.push_back(base_hand(right_id));
            translate_hand(frame.hands[0], left_x - 100.0F, 0.0F);
            translate_hand(frame.hands[1], right_x - 100.0F, 0.0F);
            return frame;
        };

        const auto start = extractor.process(two_hands(10, 100.0F, 20, 300.0F),
                                             frame_context(1));
        const auto approaching = extractor.process(
            two_hands(10, 140.0F, 20, 260.0F), frame_context(2, 33));
        const auto overlapping = extractor.process(
            two_hands(20, 200.0F, 10, 200.0F), frame_context(3, 66));
        const auto separated = extractor.process(
            two_hands(20, 260.0F, 10, 140.0F), frame_context(4, 99));

        check(approaching.hands[0].canonical_id == start.hands[0].canonical_id);
        check(approaching.hands[1].canonical_id == start.hands[1].canonical_id);
        check(overlapping.hands[0].canonical_id == 0);
        check(overlapping.hands[1].canonical_id == 0);
        check(separated.hands[0].canonical_id == start.hands[0].canonical_id);
        check(separated.hands[1].canonical_id == start.hands[1].canonical_id);
        check_empty(overlapping.observations);
    }

    it("derives directional motion and stationarity from timestamped palm anchors")
    {
        HandPrimitiveExtractor extractor(immediate_options());
        HandFrame frame;
        frame.hands.push_back(base_hand(0));
        (void)extractor.process(frame, frame_context(1, 0));
        translate_hand(frame.hands[0], 20.0F, 0.0F);
        (void)extractor.process(frame, frame_context(2, 33));
        translate_hand(frame.hands[0], 35.0F, 0.0F);
        const auto moving = extractor.process(frame, frame_context(3, 66));

        check_not_null(find_relation(moving.observations, "Direction Right"));
        check_not_null(find_relation(moving.observations, "Motion Unstable"));

        extractor.reset();
        frame = {};
        frame.hands.push_back(base_hand(0));
        (void)extractor.process(frame, frame_context(10, 100));
        (void)extractor.process(frame, frame_context(11, 133));
        const auto stationary = extractor.process(frame, frame_context(12, 166));
        check_not_null(find_relation(stationary.observations,
                                     "Motion Stationary"));
        check_not_null(find_relation(stationary.observations,
                                     "Direction Neutral"));
    }

    it("derives scale and shortest-path rotation trends from palm geometry")
    {
        HandPrimitiveExtractor scale_extractor(immediate_options());
        HandFrame scale_frame;
        scale_frame.hands.push_back(base_hand(0));
        (void)scale_extractor.process(scale_frame, frame_context(1, 0));
        scale_hand(scale_frame.hands[0], 1.08F);
        (void)scale_extractor.process(scale_frame, frame_context(2, 100));
        scale_hand(scale_frame.hands[0], 1.15F);
        const auto scale_result =
            scale_extractor.process(scale_frame, frame_context(3, 200));
        check_not_null(find_relation(scale_result.observations,
                                     "Scale Increasing"));

        HandPrimitiveExtractor rotation_extractor(immediate_options());
        HandFrame rotation_frame;
        rotation_frame.hands.push_back(base_hand(0));
        (void)rotation_extractor.process(rotation_frame, frame_context(1, 0));
        rotate_hand(rotation_frame.hands[0], 10.0F);
        (void)rotation_extractor.process(rotation_frame, frame_context(2, 100));
        rotate_hand(rotation_frame.hands[0], 15.0F);
        const auto rotation_result =
            rotation_extractor.process(rotation_frame, frame_context(3, 200));
        check_not_null(find_relation(rotation_result.observations,
                                     "Rotation Clockwise"));
    }

    it("publishes normalized distance trends with both canonical identities")
    {
        HandPrimitiveExtractor extractor(immediate_options());
        const auto pair = [](float left_x, float right_x) {
            HandFrame frame;
            frame.hands.push_back(base_hand(0));
            frame.hands.push_back(base_hand(1));
            translate_hand(frame.hands[0], left_x - 100.0F, 0.0F);
            translate_hand(frame.hands[1], right_x - 100.0F, 0.0F);
            return frame;
        };
        (void)extractor.process(pair(180.0F, 300.0F), frame_context(1, 0));
        (void)extractor.process(pair(160.0F, 320.0F), frame_context(2, 100));
        const auto result =
            extractor.process(pair(100.0F, 380.0F), frame_context(3, 200));

        const Observation* distance_relation =
            find_relation(result.observations, "Hands Distance Expanding");
        check_not_null(distance_relation);
        check_true(distance_relation->target.has_value());
        check(distance_relation->source.id != distance_relation->target->id);
    }

    it("rejects non-monotonic input and capacity overflow before state changes")
    {
        HandPrimitiveOptions options = immediate_options();
        options.max_hands = 1;
        options.identity.maximum_identities = 1;
        options.spatial.max_pair_histories = 0;
        HandPrimitiveExtractor extractor(options);
        HandFrame one;
        one.hands.push_back(base_hand(0));
        (void)extractor.process(one, frame_context(1, 0));

        check_throws_as(extractor.process(one, frame_context(1, 1)),
                        std::invalid_argument);
        HandFrame two = one;
        two.hands.push_back(base_hand(1));
        check_throws_as(extractor.process(two, frame_context(2, 2)),
                        std::length_error);

        const auto valid = extractor.process(one, frame_context(2, 2));
        check(valid.hands[0].canonical_id == 1);
    }

    it("rejects invalid identity thresholds instead of silently clamping them")
    {
        HandPrimitiveOptions options;
        options.identity.reacquire_frames = 0;
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);

        options = {};
        options.identity.maximum_linear_scale_ratio = 0.5F;
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);

        options = {};
        options.identity.velocity_observation_weight = 1.5F;
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);
    }
}
