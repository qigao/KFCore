#include "kfcore/hand_gesture/feature_encoder.hpp"

#include "tinytest.hpp"

#include <cmath>

using namespace kfcore::hand_gesture;
using namespace kfcore::hand_models;

namespace
{

HandResult sample_hand(Handedness handedness)
{
    HandResult hand;
    hand.track_id = 7;
    hand.handedness = handedness;
    hand.gesture = Gesture::Open;
    hand.landmark_confidence = 0.90F;
    hand.palm.confidence = 0.95F;

    for (std::size_t index = 0U; index < hand.landmarks.size(); ++index)
    {
        hand.landmarks[index] = {
            100.0F + static_cast<float>(index) * 2.0F,
            80.0F + static_cast<float>(index),
            static_cast<float>(index) * 0.25F,
        };
    }
    return hand;
}

bool near(float left, float right, float tolerance = 1e-5F)
{
    return std::fabs(left - right) <= tolerance;
}

} // namespace

spec("temporal gesture feature encoder")
{
    it("emits the fixed 78-value first-frame contract")
    {
        const auto encoded = GestureFeatureEncoder::encode(
            sample_hand(Handedness::Right),
            GestureFrameMetadata {1'000'000'000ULL, 1000, 500});

        check_true(encoded.values.size() == kTemporalGestureFeatureCount);
        check_true(near(encoded.values[63], 0.10F));
        check_true(near(encoded.values[64], 0.16F));
        check_true(near(encoded.values[65], 0.0F));
        check_true(near(encoded.values[66], 0.0F));
        check_true(near(encoded.values[70], 1.0F));
        check_true(near(encoded.values[71], 0.90F));
        check_true(near(encoded.values[72], 0.95F));
        check_true(near(encoded.values[73], 0.0F));
        check_true(near(encoded.values[74], 0.0F));
        check_true(near(encoded.values[75], 1.0F));
        check_true(near(encoded.values[76], 0.0F));
        check_true(near(encoded.values[77], 0.0F));
    }

    it("mirrors local x for left hands without changing global wrist position")
    {
        const auto right = GestureFeatureEncoder::encode(
            sample_hand(Handedness::Right),
            GestureFrameMetadata {1'000'000'000ULL, 1000, 500});
        const auto left = GestureFeatureEncoder::encode(
            sample_hand(Handedness::Left),
            GestureFrameMetadata {1'000'000'000ULL, 1000, 500});

        check_true(near(left.values[3], -right.values[3]));
        check_true(near(left.values[4], right.values[4]));
        check_true(near(left.values[63], right.values[63]));
        check_true(near(left.values[64], right.values[64]));
        check_true(near(left.values[70], -1.0F));
    }

    it("encodes normalized velocity in units per second")
    {
        const auto first = GestureFeatureEncoder::encode(
            sample_hand(Handedness::Right),
            GestureFrameMetadata {1'000'000'000ULL, 1000, 500});

        HandResult moved = sample_hand(Handedness::Right);
        for (auto& landmark : moved.landmarks)
        {
            landmark.x += 10.0F;
            landmark.y += 5.0F;
        }
        const auto second = GestureFeatureEncoder::encode(
            moved,
            GestureFrameMetadata {1'500'000'000ULL, 1000, 500},
            first.next_state);

        check_true(near(second.values[65], 0.02F));
        check_true(near(second.values[66], 0.02F));
        check_true(near(second.values[73], 0.50F));
    }
}
