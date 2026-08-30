#pragma once

#include "facemesh_geometry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace kfcore::face_models::detail
{

std::optional<FaceDetection> decode_yolo12_face(
    const float* values, std::size_t value_count, std::int32_t face_class_id,
    float confidence_threshold, const image::LetterboxTransform& letterbox,
    std::int32_t image_width, std::int32_t image_height);

std::array<Point3f, kFaceMeshLandmarkCount> decode_face_landmarks(
    const float* values, std::size_t value_count, const FaceRoi& roi,
    std::int32_t model_extent, bool normalized_coordinates);

} // namespace kfcore::face_models::detail
