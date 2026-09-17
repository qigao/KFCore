#include "kfcore/hand_gesture/runtime.hpp"
#include "runtime_contract.hpp"

#include "tinytest.hpp"

#include <type_traits>
#include <vector>

using namespace kfcore::hand_gesture;

spec("temporal gesture runtime public contract")
{
    it("freezes recurrent dimensions and non-copyable recognizer ownership")
    {
        check_true(kTemporalGestureFeatureCount == 78U);
        check_true(kTemporalGestureHiddenLayers == 2U);
        check_true(kTemporalGestureHiddenSize == 64U);
        check_true(kTemporalGestureHiddenElementCount == 128U);
        check_true(kTemporalGestureClassCount == 5U);
        check_true(kTemporalGesturePhaseCount == 4U);
        check_false(std::is_copy_constructible_v<TemporalGestureRecognizer>);
        check_false(std::is_copy_assignable_v<TemporalGestureRecognizer>);
    }

    it("freezes gesture and phase numeric identities")
    {
        check_true(static_cast<unsigned>(GestureClass::None) == 0U);
        check_true(static_cast<unsigned>(GestureClass::SwipeLeft) == 1U);
        check_true(static_cast<unsigned>(GestureClass::SwipeRight) == 2U);
        check_true(static_cast<unsigned>(GestureClass::Grab) == 3U);
        check_true(static_cast<unsigned>(GestureClass::Release) == 4U);
        check_true(static_cast<unsigned>(GesturePhase::Idle) == 0U);
        check_true(static_cast<unsigned>(GesturePhase::Start) == 1U);
        check_true(static_cast<unsigned>(GesturePhase::Active) == 2U);
        check_true(static_cast<unsigned>(GesturePhase::End) == 3U);
    }

    it("rejects the former eight-class gesture output at the typed load boundary")
    {
        using kfcore::runtime::DataType;
        using kfcore::runtime::TensorDescriptor;

        const std::vector<TensorDescriptor> tensors {
            {"features", DataType::Float32, {1, 78}, true},
            {"hidden_in", DataType::Float32, {2, 1, 64}, true},
            {"gesture_logits", DataType::Float32, {1, 8}, false},
            {"phase_logits", DataType::Float32, {1, 4}, false},
            {"hidden_out", DataType::Float32, {2, 1, 64}, false},
        };

        check_throws_as(
            detail::require_temporal_gesture_tensor_contract(tensors),
            HandGestureError);
    }
}
