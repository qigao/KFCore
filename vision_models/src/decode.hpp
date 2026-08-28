#pragma once

#include "geometry.hpp"
#include "kfcore/vision_models/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace kfcore::vision_models::detail
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

std::optional<FaceDetection> decode_yolo12_face(
    const float* values, std::size_t value_count, std::int32_t face_class_id,
    float confidence_threshold, const image::LetterboxTransform& letterbox,
    std::int32_t image_width, std::int32_t image_height);

std::array<Point3f, kFaceLandmarkCount> decode_face_landmarks(
    const float* values, std::size_t value_count, const FaceRoi& roi,
    std::int32_t model_extent, bool normalized_coordinates);

} // namespace kfcore::vision_models::detail
