#pragma once

#include "kfcore/runtime/types.hpp"

#include <vector>

namespace kfcore::hand_gesture::detail
{

struct TemporalGestureTensorContract
{
    runtime::TensorDescriptor features;
    runtime::TensorDescriptor hidden_in;
    runtime::TensorDescriptor gesture_logits;
    runtime::TensorDescriptor phase_logits;
    runtime::TensorDescriptor hidden_out;
};

TemporalGestureTensorContract require_temporal_gesture_tensor_contract(
    const std::vector<runtime::TensorDescriptor>& tensors);

} // namespace kfcore::hand_gesture::detail
