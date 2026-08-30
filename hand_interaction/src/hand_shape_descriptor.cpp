#include "hand_shape_descriptor.hpp"

#include <array>
#include <cmath>

namespace kfcore::hand_interaction::detail {
namespace {

constexpr std::array<std::array<std::size_t, 5U>, 5U> kFingerChains = {{
    {{0U, 1U, 2U, 3U, 4U}},
    {{0U, 5U, 6U, 7U, 8U}},
    {{0U, 9U, 10U, 11U, 12U}},
    {{0U, 13U, 14U, 15U, 16U}},
    {{0U, 17U, 18U, 19U, 20U}},
}};

bool IsFinite(const hand_models::HandLandmark& landmark) {
  return std::isfinite(landmark.x) && std::isfinite(landmark.y) &&
         std::isfinite(landmark.z);
}

}  // namespace

std::optional<HandShapeDescriptor> MakeHandShapeDescriptor(
    const std::array<hand_models::HandLandmark,
                     hand_models::kHandLandmarkCount>& landmarks) {
  for (const auto& landmark : landmarks) {
    if (!IsFinite(landmark)) {
      return std::nullopt;
    }
  }

  HandShapeDescriptor descriptor;
  std::size_t feature_index = 0U;
  for (const auto& chain : kFingerChains) {
    for (std::size_t index = 1U; index < chain.size(); ++index) {
      const auto& first = landmarks[chain[index - 1U]];
      const auto& last = landmarks[chain[index]];
      const float length = std::hypot(last.x - first.x, last.y - first.y,
                                      last.z - first.z);
      if (!std::isfinite(length)) {
        return std::nullopt;
      }
      descriptor.values[feature_index++] = length;
    }
  }
  if (!NormalizeHandShapeDescriptor(descriptor)) {
    return std::nullopt;
  }
  return descriptor;
}

float HandShapeDistance(const HandShapeDescriptor& lhs,
                        const HandShapeDescriptor& rhs) {
  float distance = 0.0F;
  for (std::size_t index = 0U; index < kHandShapeFeatureCount; ++index) {
    distance += std::fabs(lhs.values[index] - rhs.values[index]);
  }
  return distance;
}

bool NormalizeHandShapeDescriptor(HandShapeDescriptor& descriptor) {
  float total = 0.0F;
  for (const float value : descriptor.values) {
    if (!std::isfinite(value) || value < 0.0F) {
      return false;
    }
    total += value;
  }
  if (!std::isfinite(total) || total <= 0.0F) {
    return false;
  }
  for (float& value : descriptor.values) {
    value /= total;
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

}  // namespace kfcore::hand_interaction::detail
