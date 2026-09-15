#pragma once

#include "kfcore/hand_gesture/types.hpp"
#include "kfcore/hand_models/types.hpp"

#include <optional>

namespace kfcore::hand_gesture
{

class GestureFeatureEncoder final
{
public:
    [[nodiscard]] static EncodedGestureFeatures encode(
        const hand_models::HandResult& hand,
        const GestureFrameMetadata& metadata,
        const std::optional<GestureFeatureState>& previous = std::nullopt);
};

} // namespace kfcore::hand_gesture
