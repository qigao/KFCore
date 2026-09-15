#include "kfcore/hand_models/tracking.hpp"

#include "tinytest.hpp"

#include <cstdint>
#include <vector>

using namespace kfcore::hand_models;

namespace
{

HandResult hand_at(float x)
{
    HandResult hand;
    hand.palm.confidence = 0.95F;
    hand.palm.box = {x, 10.0F, 20.0F, 30.0F};
    hand.landmark_confidence = 0.90F;
    return hand;
}

HandTrackingOptions immediate_options()
{
    HandTrackingOptions options;
    options.max_hands = 2U;
    options.tracker.minimum_consecutive_frames = 1;
    return options;
}

} // namespace

spec("hand tracking")
{
    it("assigns stable track ids across nearby observations")
    {
        auto tracker = HandTracker::create(immediate_options());

        HandFrame first;
        first.hands.push_back(hand_at(10.0F));
        first = tracker->update(std::move(first));
        check_true(first.hands.size() == 1U);
        check_true(first.hands[0].track_id >= 0);
        const int track_id = first.hands[0].track_id;

        HandFrame second;
        second.hands.push_back(hand_at(11.0F));
        second = tracker->update(std::move(second));
        check_true(second.hands.size() == 1U);
        check(second.hands[0].track_id == track_id);
        check(second.timings.tracking_ms >= 0.0);
    }

    it("reset clears temporal identity state")
    {
        auto tracker = HandTracker::create(immediate_options());
        HandFrame first;
        first.hands.push_back(hand_at(10.0F));
        first = tracker->update(std::move(first));
        check_true(first.hands[0].track_id >= 0);

        tracker->reset();

        HandFrame second;
        second.hands.push_back(hand_at(10.0F));
        second = tracker->update(std::move(second));
        check_true(second.hands[0].track_id >= 0);
    }
}
