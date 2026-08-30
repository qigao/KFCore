#pragma once

#include "geometry.hpp"
#include "kfcore/hand_models/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace kfcore::hand_models::detail
{

std::vector<PalmDetection> decode_palms(
    const float* values, std::size_t value_count, float confidence_threshold,
    std::size_t max_candidates, std::size_t max_hands,
    const image::LetterboxTransform& letterbox, std::int32_t model_extent);

std::array<HandLandmark, kHandLandmarkCount> decode_hand_landmarks(
    const float* values, std::size_t value_count, const RotatedRoi& roi,
    std::int32_t model_extent);

std::array<float, kHandLandmarkCount * 2U> make_keypoint_features(
    const std::array<HandLandmark, kHandLandmarkCount>& landmarks);

Handedness decode_handedness(float value);
Gesture    decode_gesture(std::int64_t class_id) noexcept;

} // namespace kfcore::hand_models::detail
