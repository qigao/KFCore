#include "kfcore/hand_interaction/primitive_extractor.hpp"
#include "tinytest.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
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

HandResult identity_hand(int track_id = 0)
{
    HandResult hand;
    hand.track_id = track_id;
    hand.palm.confidence = 0.98F;
    hand.landmark_confidence = 0.98F;
    hand.palm.box = { 40.0F, 40.0F, 160.0F, 160.0F };
    hand.handedness = kfcore::vision_models::Handedness::Left;

    constexpr std::array<kfcore::vision_models::HandLandmark, 21>
        kLandmarks = {{
            { 100.0F, 220.0F, 0.0F }, { 72.0F, 210.0F, 0.0F },
            { 55.0F, 188.0F, 0.0F },  { 42.0F, 166.0F, 0.0F },
            { 30.0F, 145.0F, 0.0F },  { 75.0F, 170.0F, 0.0F },
            { 72.0F, 130.0F, 0.0F },  { 68.0F, 95.0F, 0.0F },
            { 65.0F, 60.0F, 0.0F },   { 100.0F, 165.0F, 0.0F },
            { 100.0F, 122.0F, 0.0F }, { 100.0F, 85.0F, 0.0F },
            { 100.0F, 48.0F, 0.0F },  { 125.0F, 170.0F, 0.0F },
            { 130.0F, 132.0F, 0.0F }, { 134.0F, 99.0F, 0.0F },
            { 138.0F, 67.0F, 0.0F },  { 145.0F, 180.0F, 0.0F },
            { 153.0F, 150.0F, 0.0F }, { 158.0F, 124.0F, 0.0F },
            { 163.0F, 99.0F, 0.0F },
        }};
    hand.landmarks = kLandmarks;
    return hand;
}

void stretch_finger(HandResult& hand, std::size_t mcp_index, float factor)
{
    const auto anchor = hand.landmarks[mcp_index];
    for (std::size_t index = mcp_index + 1U; index <= mcp_index + 3U; ++index)
    {
        auto& point = hand.landmarks[index];
        point.x = anchor.x + (point.x - anchor.x) * factor;
        point.y = anchor.y + (point.y - anchor.y) * factor;
        point.z = anchor.z + (point.z - anchor.z) * factor;
    }
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

void set_palm_axis(HandResult& hand, float degrees, float center_x = 110.0F,
                   float center_y = 170.0F)
{
    constexpr float kRadius = 40.0F;
    constexpr float kRadiansPerDegree = 3.14159265358979323846F / 180.0F;
    const float radians = degrees * kRadiansPerDegree;
    const float x = std::cos(radians) * kRadius;
    const float y = std::sin(radians) * kRadius;
    set_point(hand, 5, center_x + x, center_y + y);
    set_point(hand, 17, center_x - x, center_y - y);
}

float measured_palm_axis(const HandResult& hand)
{
    constexpr float kDegreesPerRadian = 180.0F / 3.14159265358979323846F;
    const float orientation =
        std::atan2(hand.landmarks[5].y - hand.landmarks[17].y,
                   hand.landmarks[5].x - hand.landmarks[17].x) *
        kDegreesPerRadian;
    return std::fabs(orientation);
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

std::size_t count_palm_axis_relations(
    const std::vector<Observation>& observations)
{
    return static_cast<std::size_t>(std::count_if(
        observations.begin(), observations.end(), [](const Observation& item) {
            return item.relation.compare(0U, 10U, "Palm Axis ") == 0;
        }));
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

    it("keeps a canonical identity while ByteTrack is temporarily unconfirmed")
    {
        HandPrimitiveExtractor extractor;
        HandFrame first_frame;
        first_frame.hands.push_back(base_hand(4));
        const auto first = extractor.process(first_frame, frame_context(1));

        HandFrame unconfirmed_frame;
        unconfirmed_frame.hands.push_back(base_hand(-1));
        translate_hand(unconfirmed_frame.hands[0], 8.0F, 4.0F);
        const auto unconfirmed =
            extractor.process(unconfirmed_frame, frame_context(2, 33));

        check(first.hands[0].canonical_id > 0);
        check(unconfirmed.hands[0].raw_track_id == -1);
        check(unconfirmed.hands[0].canonical_id ==
              first.hands[0].canonical_id);
    }

    it("keeps distinct canonical identities when two hands alternate unconfirmed")
    {
        HandPrimitiveExtractor extractor;
        HandFrame both_hands;
        both_hands.hands.push_back(base_hand(4));
        both_hands.hands.push_back(base_hand(9));
        translate_hand(both_hands.hands[1], 240.0F, 0.0F);
        const auto initial = extractor.process(both_hands, frame_context(1));

        HandFrame right_only;
        right_only.hands.push_back(base_hand(-1));
        translate_hand(right_only.hands[0], 244.0F, 3.0F);
        const auto right =
            extractor.process(right_only, frame_context(2, 33));

        HandFrame left_only;
        left_only.hands.push_back(base_hand(-1));
        translate_hand(left_only.hands[0], 4.0F, 3.0F);
        const auto left =
            extractor.process(left_only, frame_context(3, 66));

        check(initial.hands[0].canonical_id > 0);
        check(initial.hands[1].canonical_id > 0);
        check(initial.hands[0].canonical_id != initial.hands[1].canonical_id);
        check(right.hands[0].raw_track_id == -1);
        check(right.hands[0].canonical_id == initial.hands[1].canonical_id);
        check(left.hands[0].raw_track_id == -1);
        check(left.hands[0].canonical_id == initial.hands[0].canonical_id);
    }

    it("reacquires the same hand by shape inside the retention horizon")
    {
        HandPrimitiveOptions options;
        options.identity.reacquire_frames = 2;
        HandPrimitiveExtractor extractor(options);

        HandFrame first_frame;
        first_frame.hands.push_back(identity_hand(4));
        const auto first = extractor.process(first_frame, frame_context(1));
        (void)extractor.process({}, frame_context(2));

        HandFrame reacquired_frame;
        reacquired_frame.hands.push_back(identity_hand(19));
        scale_hand(reacquired_frame.hands[0], 1.35F);
        rotate_hand(reacquired_frame.hands[0], 57.0F);
        translate_hand(reacquired_frame.hands[0], 250.0F, 120.0F);
        const auto reacquired =
            extractor.process(reacquired_frame, frame_context(3));

        check(reacquired.hands[0].canonical_id == first.hands[0].canonical_id);
    }

    it("reacquires a hand with independent small 3D landmark jitter")
    {
        HandPrimitiveOptions options;
        options.identity.reacquire_frames = 2;
        HandPrimitiveExtractor extractor(options);

        HandFrame first_frame;
        first_frame.hands.push_back(identity_hand(4));
        const auto first = extractor.process(first_frame, frame_context(1));
        (void)extractor.process({}, frame_context(2));

        HandFrame jittered_frame;
        jittered_frame.hands.push_back(identity_hand(19));
        jittered_frame.hands[0].landmarks[1].x += 0.30F;
        jittered_frame.hands[0].landmarks[6].y -= 0.25F;
        jittered_frame.hands[0].landmarks[11].z += 0.20F;
        jittered_frame.hands[0].landmarks[16].x -= 0.15F;
        jittered_frame.hands[0].landmarks[19].z -= 0.18F;
        translate_hand(jittered_frame.hands[0], 250.0F, 120.0F);
        const auto jittered = extractor.process(jittered_frame, frame_context(3));

        check(jittered.hands[0].canonical_id == first.hands[0].canonical_id);
    }

    it("allocates a new identity after the retention horizon")
    {
        HandPrimitiveOptions options;
        options.identity.reacquire_frames = 2;
        HandPrimitiveExtractor extractor(options);

        HandFrame first_frame;
        first_frame.hands.push_back(identity_hand(4));
        const auto first = extractor.process(first_frame, frame_context(1));
        (void)extractor.process({}, frame_context(2));
        (void)extractor.process({}, frame_context(3));
        (void)extractor.process({}, frame_context(4));

        HandFrame later_frame;
        later_frame.hands.push_back(identity_hand(19));
        translate_hand(later_frame.hands[0], 250.0F, 120.0F);
        const auto later = extractor.process(later_frame, frame_context(5));

        check(later.hands[0].canonical_id > 0);
        check(later.hands[0].canonical_id != first.hands[0].canonical_id);
    }

    it("rejects a materially different hand shape at the same location")
    {
        HandPrimitiveExtractor extractor;
        HandFrame first_frame;
        first_frame.hands.push_back(identity_hand(4));
        const auto first = extractor.process(first_frame, frame_context(1));

        HandFrame different_shape_frame;
        different_shape_frame.hands.push_back(identity_hand(19));
        stretch_finger(different_shape_frame.hands[0], 5U, 2.0F);
        const auto different_shape =
            extractor.process(different_shape_frame, frame_context(2));

        check(different_shape.hands[0].canonical_id > 0);
        check(different_shape.hands[0].canonical_id != first.hands[0].canonical_id);
    }

    it("preserves two shape identities when their positions swap inside the horizon")
    {
        HandPrimitiveOptions options;
        options.identity.reacquire_frames = 2;
        HandPrimitiveExtractor extractor(options);
        HandFrame initial_frame;
        initial_frame.hands.push_back(identity_hand(4));
        initial_frame.hands.push_back(identity_hand(9));
        stretch_finger(initial_frame.hands[1], 9U, 2.8F);
        translate_hand(initial_frame.hands[1], 260.0F, 0.0F);
        const auto initial = extractor.process(initial_frame, frame_context(1));

        HandFrame swapped_frame;
        swapped_frame.hands.push_back(identity_hand(19));
        stretch_finger(swapped_frame.hands[0], 9U, 2.8F);
        swapped_frame.hands.push_back(identity_hand(23));
        translate_hand(swapped_frame.hands[1], 260.0F, 0.0F);
        const auto swapped = extractor.process(swapped_frame, frame_context(2));

        check(swapped.hands[0].canonical_id == initial.hands[1].canonical_id);
        check(swapped.hands[1].canonical_id == initial.hands[0].canonical_id);
    }

    it("rejects a known handedness conflict as an identity candidate")
    {
        HandPrimitiveOptions options;
        options.max_hands = 1;
        options.identity.maximum_identities = 2;
        HandPrimitiveExtractor extractor(options);

        HandFrame left_frame;
        left_frame.hands.push_back(identity_hand(4));
        const auto left = extractor.process(left_frame, frame_context(1));

        HandFrame right_frame;
        right_frame.hands.push_back(identity_hand(19));
        right_frame.hands[0].handedness =
            kfcore::vision_models::Handedness::Right;
        const auto right = extractor.process(right_frame, frame_context(2));

        check(right.hands[0].canonical_id > 0);
        check(right.hands[0].canonical_id != left.hands[0].canonical_id);
    }

    it("withholds an indistinguishable shape match while retention evidence is fresh")
    {
        HandPrimitiveOptions options;
        options.identity.reacquire_frames = 2;
        HandPrimitiveExtractor extractor(options);
        HandFrame initial_frame;
        initial_frame.hands.push_back(identity_hand(4));
        initial_frame.hands.push_back(identity_hand(9));
        initial_frame.hands[0].handedness =
            kfcore::vision_models::Handedness::Unknown;
        initial_frame.hands[1].handedness =
            kfcore::vision_models::Handedness::Unknown;
        (void)extractor.process(initial_frame, frame_context(1));

        HandFrame ambiguous_frame;
        ambiguous_frame.hands.push_back(identity_hand(-1));
        ambiguous_frame.hands[0].handedness =
            kfcore::vision_models::Handedness::Unknown;
        const auto ambiguous =
            extractor.process(ambiguous_frame, frame_context(2));

        check(ambiguous.hands[0].canonical_id == 0);
    }

    it("withholds duplicate competitors independent of input order")
    {
        const auto resolve_duplicates = [](bool reverse_input) {
            HandPrimitiveExtractor extractor;
            HandFrame first_frame;
            first_frame.hands.push_back(identity_hand(4));
            (void)extractor.process(first_frame, frame_context(1));

            HandResult nearer = identity_hand(-1);
            HandResult farther = identity_hand(-1);
            translate_hand(nearer, 8.0F, 0.0F);
            translate_hand(farther, 9.0F, 0.0F);
            HandFrame duplicates;
            if (reverse_input)
            {
                duplicates.hands.push_back(farther);
                duplicates.hands.push_back(nearer);
            }
            else
            {
                duplicates.hands.push_back(nearer);
                duplicates.hands.push_back(farther);
            }
            return extractor.process(duplicates, frame_context(2));
        };

        const auto forward = resolve_duplicates(false);
        const auto reversed = resolve_duplicates(true);

        check(forward.hands[0].canonical_id == 0);
        check(forward.hands[1].canonical_id == 0);
        check(reversed.hands[0].canonical_id == 0);
        check(reversed.hands[1].canonical_id == 0);
    }

    it("uses the full 3D descriptor when deciding a shape match")
    {
        HandPrimitiveExtractor extractor;
        HandFrame first_frame;
        first_frame.hands.push_back(identity_hand(4));
        const auto first = extractor.process(first_frame, frame_context(1));

        HandFrame depth_changed_frame;
        depth_changed_frame.hands.push_back(identity_hand(19));
        depth_changed_frame.hands[0].landmarks[8].z = 1000.0F;
        const auto depth_changed =
            extractor.process(depth_changed_frame, frame_context(2));

        check(depth_changed.hands[0].canonical_id !=
              first.hands[0].canonical_id);
    }

    it("applies the configured shape-distance gate")
    {
        const auto resolve = [](float maximum_shape_distance) {
            HandPrimitiveOptions options;
            options.identity.maximum_shape_distance = maximum_shape_distance;
            HandPrimitiveExtractor extractor(options);

            HandFrame first_frame;
            first_frame.hands.push_back(identity_hand(4));
            const auto first = extractor.process(first_frame, frame_context(1));

            HandFrame changed_frame;
            changed_frame.hands.push_back(identity_hand(19));
            stretch_finger(changed_frame.hands[0], 5U, 2.0F);
            const auto changed = extractor.process(changed_frame, frame_context(2));
            return std::pair { first.hands[0].canonical_id,
                               changed.hands[0].canonical_id };
        };

        const auto strict = resolve(0.20F);
        const auto permissive = resolve(0.25F);
        check(strict.first != strict.second);
        check(permissive.first == permissive.second);
    }

    it("weights compatible shapes when spatial evidence disagrees")
    {
        const auto resolve = [](float shape_cost_weight) {
            HandPrimitiveOptions options;
            options.max_hands = 2;
            options.identity.maximum_identities = 2;
            options.identity.maximum_shape_distance = 0.25F;
            options.identity.shape_cost_weight = shape_cost_weight;
            HandPrimitiveExtractor extractor(options);

            HandFrame initial_frame;
            initial_frame.hands.push_back(identity_hand(4));
            initial_frame.hands.push_back(identity_hand(9));
            stretch_finger(initial_frame.hands[1], 5U, 2.0F);
            translate_hand(initial_frame.hands[1], 300.0F, 0.0F);
            const auto initial = extractor.process(initial_frame, frame_context(1));

            HandFrame query_frame;
            query_frame.hands.push_back(identity_hand(-1));
            stretch_finger(query_frame.hands[0], 5U, 2.0F);
            const auto query = extractor.process(query_frame, frame_context(2));
            return std::pair { initial.hands[0].canonical_id,
                               std::pair { initial.hands[1].canonical_id,
                                           query.hands[0].canonical_id } };
        };

        const auto spatial_only = resolve(0.0F);
        const auto shape_weighted = resolve(30.0F);
        check(spatial_only.second.second == spatial_only.first);
        check(shape_weighted.second.second == shape_weighted.second.first);
    }

    it("normalizes distance evidence by its configured saturation ratio")
    {
        const auto resolve = [](float maximum_distance_scale_ratio) {
            HandPrimitiveOptions options;
            options.max_hands = 2;
            options.identity.maximum_identities = 2;
            options.identity.maximum_distance_scale_ratio =
                maximum_distance_scale_ratio;
            options.identity.ambiguity_cost_margin = 0.20F;
            HandPrimitiveExtractor extractor(options);

            HandFrame initial_frame;
            initial_frame.hands.push_back(identity_hand(4));
            initial_frame.hands.push_back(identity_hand(9));
            translate_hand(initial_frame.hands[1], 100.0F, 0.0F);
            const auto initial = extractor.process(initial_frame, frame_context(1));

            HandFrame query_frame;
            query_frame.hands.push_back(identity_hand(-1));
            translate_hand(query_frame.hands[0], 10.0F, 0.0F);
            const auto query = extractor.process(query_frame, frame_context(2));
            return std::pair { initial.hands[0].canonical_id,
                               query.hands[0].canonical_id };
        };

        const auto discriminating = resolve(3.0F);
        const auto saturated = resolve(0.10F);
        check(discriminating.second == discriminating.first);
        check(saturated.second == 0);
    }

    it("normalizes scale evidence by its configured saturation ratio")
    {
        const auto resolve = [](float maximum_linear_scale_ratio) {
            HandPrimitiveOptions options;
            options.max_hands = 2;
            options.identity.maximum_identities = 2;
            options.identity.maximum_distance_scale_ratio = 3.0F;
            options.identity.maximum_linear_scale_ratio =
                maximum_linear_scale_ratio;
            options.identity.scale_cost_weight = 1.0F;
            options.identity.ambiguity_cost_margin = 0.20F;
            HandPrimitiveExtractor extractor(options);

            HandFrame initial_frame;
            initial_frame.hands.push_back(identity_hand(4));
            initial_frame.hands.push_back(identity_hand(9));
            scale_hand(initial_frame.hands[1], 2.0F);
            const auto initial = extractor.process(initial_frame, frame_context(1));

            HandFrame query_frame;
            query_frame.hands.push_back(identity_hand(-1));
            scale_hand(query_frame.hands[0], 1.2F);
            const auto query = extractor.process(query_frame, frame_context(2));
            return std::pair { initial.hands[0].canonical_id,
                               query.hands[0].canonical_id };
        };

        const auto discriminating = resolve(3.0F);
        const auto saturated = resolve(1.10F);
        check(discriminating.second == discriminating.first);
        check(saturated.second == 0);
    }

    it("updates the shape prototype at the configured rate")
    {
        const auto resolve = [](float shape_update_weight) {
            HandPrimitiveOptions options;
            options.max_hands = 2;
            options.identity.maximum_identities = 2;
            options.identity.maximum_shape_distance = 0.10F;
            options.identity.shape_update_weight = shape_update_weight;
            HandPrimitiveExtractor extractor(options);

            HandFrame first_frame;
            first_frame.hands.push_back(identity_hand(4));
            const auto first = extractor.process(first_frame, frame_context(1));

            HandFrame intermediate_frame;
            intermediate_frame.hands.push_back(identity_hand(19));
            stretch_finger(intermediate_frame.hands[0], 5U, 1.4F);
            (void)extractor.process(intermediate_frame, frame_context(2));

            HandFrame later_frame;
            later_frame.hands.push_back(identity_hand(23));
            stretch_finger(later_frame.hands[0], 5U, 1.8F);
            const auto later = extractor.process(later_frame, frame_context(3));
            return std::pair { first.hands[0].canonical_id,
                               later.hands[0].canonical_id };
        };

        const auto slow = resolve(0.10F);
        const auto immediate = resolve(1.0F);
        check(slow.first != slow.second);
        check(immediate.first == immediate.second);
    }

    it("does not commit matched prototype updates when a mixed frame exhausts capacity")
    {
        HandPrimitiveOptions options;
        options.max_hands = 2;
        options.identity.maximum_identities = 2;
        options.identity.maximum_shape_distance = 0.10F;
        options.identity.shape_update_weight = 1.0F;
        HandPrimitiveExtractor extractor(options);

        HandFrame initial_frame;
        initial_frame.hands.push_back(identity_hand(4));
        initial_frame.hands.push_back(identity_hand(9));
        stretch_finger(initial_frame.hands[1], 13U, 2.0F);
        translate_hand(initial_frame.hands[1], 300.0F, 0.0F);
        const auto initial = extractor.process(initial_frame, frame_context(1));

        HandFrame overflowing_frame;
        overflowing_frame.hands.push_back(identity_hand(19));
        stretch_finger(overflowing_frame.hands[0], 5U, 1.4F);
        overflowing_frame.hands.push_back(identity_hand(23));
        stretch_finger(overflowing_frame.hands[1], 9U, 2.0F);
        translate_hand(overflowing_frame.hands[1], 300.0F, 0.0F);
        check_throws_as(extractor.process(overflowing_frame, frame_context(2)),
                        std::length_error);

        HandFrame later_shape_frame;
        later_shape_frame.hands.push_back(identity_hand(29));
        stretch_finger(later_shape_frame.hands[0], 5U, 1.8F);
        check_throws_as(extractor.process(later_shape_frame, frame_context(2)),
                        std::length_error);

        HandFrame retry_frame;
        retry_frame.hands.push_back(identity_hand(31));
        const auto retry = extractor.process(retry_frame, frame_context(2));
        check(retry.hands[0].canonical_id == initial.hands[0].canonical_id);
    }

    it("withholds identity for a non-finite non-palm landmark")
    {
        HandPrimitiveExtractor extractor;
        HandFrame frame;
        frame.hands.push_back(identity_hand(4));
        frame.hands[0].landmarks[6].x =
            std::numeric_limits<float>::quiet_NaN();

        const auto result = extractor.process(frame, frame_context(1));

        check(result.hands[0].canonical_id == 0);
    }

    it("withholds identity for a fully degenerate landmark shape")
    {
        HandPrimitiveExtractor extractor;
        HandFrame frame;
        frame.hands.push_back(identity_hand(4));
        for (auto& landmark : frame.hands[0].landmarks)
        {
            landmark = { 100.0F, 180.0F, 0.0F };
        }

        const auto result = extractor.process(frame, frame_context(1));

        check(result.hands[0].canonical_id == 0);
    }

    it("allocates a new canonical identity for a far hand while an old hand is dormant")
    {
        HandPrimitiveOptions options;
        options.max_hands = 1;
        options.identity.maximum_identities = 2;
        HandPrimitiveExtractor extractor(options);
        HandFrame first_frame;
        first_frame.hands.push_back(identity_hand(4));
        const auto first = extractor.process(first_frame, frame_context(1));
        (void)extractor.process({}, frame_context(2, 33));

        HandFrame far_frame;
        far_frame.hands.push_back(identity_hand(19));
        stretch_finger(far_frame.hands[0], 5U, 2.8F);
        translate_hand(far_frame.hands[0], 350.0F, 0.0F);
        const auto far_result =
            extractor.process(far_frame, frame_context(3, 66));

        check(far_result.hands[0].canonical_id > 0);
        check(far_result.hands[0].canonical_id != first.hands[0].canonical_id);
    }

    it("rejects canonical identity exhaustion without consuming the frame")
    {
        HandPrimitiveOptions options;
        options.max_hands = 1;
        options.identity.maximum_identities = 1;
        HandPrimitiveExtractor extractor(options);
        HandFrame first_frame;
        first_frame.hands.push_back(identity_hand(4));
        (void)extractor.process(first_frame, frame_context(1));

        HandFrame far_frame;
        far_frame.hands.push_back(identity_hand(19));
        stretch_finger(far_frame.hands[0], 5U, 2.8F);
        translate_hand(far_frame.hands[0], 350.0F, 0.0F);
        check_throws_as(extractor.process(far_frame, frame_context(2, 33)),
                        std::length_error);

        const auto retry =
            extractor.process(first_frame, frame_context(2, 33));
        check(retry.hands[0].canonical_id == 1);
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

    it("publishes an undirected palm-axis angle state")
    {
        HandPrimitiveOptions options = immediate_options();
        options.spatial.palm_axis_horizontal_max_degrees = 30.0F;
        options.spatial.palm_axis_vertical_min_degrees = 60.0F;

        HandFrame frame;
        frame.hands.push_back(base_hand(0));
        HandPrimitiveExtractor horizontal_extractor(options);
        const auto horizontal =
            horizontal_extractor.process(frame, frame_context(1));
        check_not_null(find_relation(horizontal.observations,
                                     "Palm Axis Horizontal"));
        check(count_palm_axis_relations(horizontal.observations) == 1U);

        rotate_hand(frame.hands[0], 45.0F);
        HandPrimitiveExtractor diagonal_extractor(options);
        const auto diagonal =
            diagonal_extractor.process(frame, frame_context(2));
        check_not_null(find_relation(diagonal.observations,
                                     "Palm Axis Diagonal"));
        check(count_palm_axis_relations(diagonal.observations) == 1U);

        rotate_hand(frame.hands[0], 45.0F);
        HandPrimitiveExtractor vertical_extractor(options);
        const auto vertical =
            vertical_extractor.process(frame, frame_context(3));
        check_not_null(find_relation(vertical.observations,
                                     "Palm Axis Vertical"));
        check(count_palm_axis_relations(vertical.observations) == 1U);

        frame.hands[0] = base_hand(0);
        rotate_hand(frame.hands[0], 170.0F);
        HandPrimitiveExtractor reversed_extractor(options);
        const auto reversed =
            reversed_extractor.process(frame, frame_context(4));
        check_not_null(find_relation(reversed.observations,
                                     "Palm Axis Horizontal"));
        check(count_palm_axis_relations(reversed.observations) == 1U);

        frame.hands[0] = base_hand(0);
        set_palm_axis(frame.hands[0], 30.0F);
        HandPrimitiveOptions horizontal_boundary_options = immediate_options();
        horizontal_boundary_options.spatial.palm_axis_horizontal_max_degrees =
            measured_palm_axis(frame.hands[0]);
        HandPrimitiveExtractor horizontal_boundary_extractor(
            horizontal_boundary_options);
        const auto horizontal_boundary =
            horizontal_boundary_extractor.process(frame, frame_context(5));
        check_not_null(find_relation(horizontal_boundary.observations,
                                     "Palm Axis Horizontal"));

        frame.hands[0] = base_hand(0);
        set_palm_axis(frame.hands[0], 60.0F);
        HandPrimitiveOptions vertical_boundary_options = immediate_options();
        vertical_boundary_options.spatial.palm_axis_vertical_min_degrees =
            measured_palm_axis(frame.hands[0]);
        HandPrimitiveExtractor vertical_boundary_extractor(
            vertical_boundary_options);
        const auto vertical_boundary =
            vertical_boundary_extractor.process(frame, frame_context(6));
        check_not_null(find_relation(vertical_boundary.observations,
                                     "Palm Axis Vertical"));
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

    it("rejects invalid shape identity configuration")
    {
        HandPrimitiveOptions options;
        options.identity.maximum_shape_distance = 0.0F;
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);

        options = {};
        options.identity.maximum_shape_distance = -0.1F;
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);

        options = {};
        options.identity.maximum_shape_distance =
            std::numeric_limits<float>::quiet_NaN();
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);

        options = {};
        options.identity.maximum_shape_distance =
            std::numeric_limits<float>::infinity();
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);

        options = {};
        options.identity.maximum_shape_distance = 2.1F;
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);

        options = {};
        options.identity.shape_cost_weight = -0.1F;
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);

        options = {};
        options.identity.shape_cost_weight =
            std::numeric_limits<float>::quiet_NaN();
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);

        options = {};
        options.identity.shape_cost_weight =
            std::numeric_limits<float>::infinity();
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);

        options = {};
        options.identity.shape_update_weight = -0.1F;
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);

        options = {};
        options.identity.shape_update_weight = 0.0F;
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);

        options = {};
        options.identity.shape_update_weight = 1.1F;
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);

        options = {};
        options.identity.shape_update_weight =
            std::numeric_limits<float>::quiet_NaN();
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);

        options = {};
        options.identity.shape_update_weight =
            std::numeric_limits<float>::infinity();
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);

        options = {};
        options.identity.handedness_mismatch_penalty = -0.1F;
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);

        options = {};
        options.identity.handedness_mismatch_penalty =
            std::numeric_limits<float>::quiet_NaN();
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);

        options = {};
        options.identity.handedness_mismatch_penalty =
            std::numeric_limits<float>::infinity();
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);
    }

    it("accepts zero shape and handedness ranking costs")
    {
        HandPrimitiveOptions options;
        options.identity.shape_cost_weight = 0.0F;
        options.identity.handedness_mismatch_penalty = 0.0F;

        check_nothrow(HandPrimitiveExtractor { options });
    }

    it("rejects non-finite and negative geometry thresholds")
    {
        HandPrimitiveOptions options;
        options.pose.minimum_confidence =
            std::numeric_limits<float>::quiet_NaN();
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);

        options = {};
        options.pose.thumb_index_contact_ratio = -0.1F;
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);

        options = {};
        options.motion.direction_dominance_ratio =
            std::numeric_limits<float>::quiet_NaN();
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);

        options = {};
        options.spatial.scale_change_ratio =
            std::numeric_limits<float>::quiet_NaN();
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);

        options = {};
        options.spatial.palm_axis_horizontal_max_degrees = 60.0F;
        options.spatial.palm_axis_vertical_min_degrees = 30.0F;
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);

        options = {};
        options.spatial.palm_axis_vertical_min_degrees =
            std::numeric_limits<float>::infinity();
        check_throws_as(HandPrimitiveExtractor { options }, std::invalid_argument);
    }
}
