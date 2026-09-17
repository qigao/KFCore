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
    hand.palm.confidence     = 0.95F;
    hand.palm.box            = { x, 10.0F, 20.0F, 30.0F };
    hand.landmark_confidence = 0.90F;
    return hand;
}

HandTrackingOptions immediate_options()
{
    HandTrackingOptions options;
    options.max_hands                          = 2U;
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
        check(first.hands[0].track_id == -1);

        HandFrame second;
        second.hands.push_back(hand_at(11.0F));
        second = tracker->update(std::move(second));
        check_true(second.hands.size() == 1U);
        check_true(second.hands[0].track_id >= 0);
        const int track_id = second.hands[0].track_id;

        HandFrame third;
        third.hands.push_back(hand_at(12.0F));
        third = tracker->update(std::move(third));
        check_true(third.hands.size() == 1U);
        check(third.hands[0].track_id == track_id);
        check(third.timings.tracking_ms >= 0.0);
    }

    it("reset clears temporal identity state")
    {
        auto      tracker = HandTracker::create(immediate_options());
        HandFrame first;
        first.hands.push_back(hand_at(10.0F));
        first = tracker->update(std::move(first));
        check_true(first.hands.size() == 1U);
        check(first.hands[0].track_id == -1);

        HandFrame confirmed;
        confirmed.hands.push_back(hand_at(10.0F));
        confirmed = tracker->update(std::move(confirmed));
        check_true(confirmed.hands.size() == 1U);
        check_true(confirmed.hands[0].track_id >= 0);

        tracker->reset();

        HandFrame after_reset;
        after_reset.hands.push_back(hand_at(10.0F));
        after_reset = tracker->update(std::move(after_reset));
        check_true(after_reset.hands.size() == 1U);
        check(after_reset.hands[0].track_id == -1);

        HandFrame reconfirmed;
        reconfirmed.hands.push_back(hand_at(10.0F));
        reconfirmed = tracker->update(std::move(reconfirmed));
        check_true(reconfirmed.hands.size() == 1U);
        check_true(reconfirmed.hands[0].track_id >= 0);
    }

    it("preserves a confirmed track across a short detection dropout")
    {
        HandTrackingOptions options       = immediate_options();
        options.tracker.lost_track_buffer = 2;
        auto tracker                      = HandTracker::create(options);

        HandFrame initial;
        initial.hands.push_back(hand_at(10.0F));
        initial = tracker->update(std::move(initial));
        check_size(initial.hands, 1U);
        check(initial.hands[0].track_id == -1);

        HandFrame confirmed;
        confirmed.hands.push_back(hand_at(11.0F));
        confirmed = tracker->update(std::move(confirmed));
        check_size(confirmed.hands, 1U);
        check_true(confirmed.hands[0].track_id >= 0);
        const int track_id = confirmed.hands[0].track_id;

        HandFrame missing;
        missing = tracker->update(std::move(missing));
        check_empty(missing.hands);

        HandFrame reacquired;
        reacquired.hands.push_back(hand_at(12.0F));
        reacquired = tracker->update(std::move(reacquired));
        check_size(reacquired.hands, 1U);
        check(reacquired.hands[0].track_id == track_id);
    }

    it("starts an unconfirmed track after the lost buffer expires")
    {
        HandTrackingOptions options       = immediate_options();
        options.tracker.lost_track_buffer = 1;
        auto tracker                      = HandTracker::create(options);

        HandFrame initial;
        initial.hands.push_back(hand_at(10.0F));
        (void)tracker->update(std::move(initial));

        HandFrame confirmed;
        confirmed.hands.push_back(hand_at(11.0F));
        confirmed = tracker->update(std::move(confirmed));
        check_size(confirmed.hands, 1U);
        check_true(confirmed.hands[0].track_id >= 0);

        HandFrame first_missing;
        first_missing = tracker->update(std::move(first_missing));
        check_empty(first_missing.hands);

        HandFrame replacement;
        replacement.hands.push_back(hand_at(12.0F));
        replacement = tracker->update(std::move(replacement));
        check_size(replacement.hands, 1U);
        check(replacement.hands[0].track_id == -1);
    }
}
