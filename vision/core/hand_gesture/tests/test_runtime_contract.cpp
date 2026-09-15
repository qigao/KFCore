#include "kfcore/hand_gesture/runtime.hpp"

#include "tinytest.hpp"

#include <type_traits>

using namespace kfcore::hand_gesture;

spec("temporal gesture runtime public contract")
{
    it("freezes recurrent dimensions and non-copyable recognizer ownership")
    {
        check_true(kTemporalGestureFeatureCount == 78U);
        check_true(kTemporalGestureHiddenLayers == 2U);
        check_true(kTemporalGestureHiddenSize == 64U);
        check_true(kTemporalGestureHiddenElementCount == 128U);
        check_true(kTemporalGestureClassCount == 8U);
        check_true(kTemporalGesturePhaseCount == 4U);
        check_false(std::is_copy_constructible_v<TemporalGestureRecognizer>);
        check_false(std::is_copy_assignable_v<TemporalGestureRecognizer>);
    }

    it("freezes gesture and phase numeric identities")
    {
        check_true(static_cast<unsigned>(GestureClass::None) == 0U);
        check_true(static_cast<unsigned>(GestureClass::Wave) == 1U);
        check_true(static_cast<unsigned>(GestureClass::SwipeLeft) == 2U);
        check_true(static_cast<unsigned>(GestureClass::SwipeRight) == 3U);
        check_true(static_cast<unsigned>(GestureClass::Grab) == 4U);
        check_true(static_cast<unsigned>(GestureClass::Release) == 5U);
        check_true(static_cast<unsigned>(GestureClass::Point) == 6U);
        check_true(static_cast<unsigned>(GestureClass::Click) == 7U);
        check_true(static_cast<unsigned>(GesturePhase::Idle) == 0U);
        check_true(static_cast<unsigned>(GesturePhase::Start) == 1U);
        check_true(static_cast<unsigned>(GesturePhase::Active) == 2U);
        check_true(static_cast<unsigned>(GesturePhase::End) == 3U);
    }
}
