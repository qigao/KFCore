#pragma once

#include "kfcore/hand_models/types.hpp"

#include <array>
#include <cstddef>
#include <optional>

namespace kfcore::hand_interaction::detail {

inline constexpr std::size_t kHandShapeFeatureCount = 20U;

struct HandShapeDescriptor {
  std::array<float, kHandShapeFeatureCount> values {};
};

[[nodiscard]] std::optional<HandShapeDescriptor> MakeHandShapeDescriptor(
    const std::array<hand_models::HandLandmark,
                     hand_models::kHandLandmarkCount>& landmarks);
[[nodiscard]] float HandShapeDistance(const HandShapeDescriptor& lhs,
                                      const HandShapeDescriptor& rhs);
[[nodiscard]] bool NormalizeHandShapeDescriptor(HandShapeDescriptor& descriptor);

}  // namespace kfcore::hand_interaction::detail
