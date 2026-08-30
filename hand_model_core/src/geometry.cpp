#include "geometry.hpp"

#include "kfcore/hand_models/error.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace kfcore::hand_models::detail
{
namespace
{

constexpr float kPi                 = 3.14159265358979323846F;
constexpr float kHandRoiScale       = 2.9F;
constexpr float kHandCenterShift    = 0.5F;

[[noreturn]] void throw_geometry(const std::string& detail)
{
    throw HandModelError(HandModelErrorCode::ModelContractMismatch,
                           "vision geometry stage: " + detail);
}

void require_finite(float value, const char* name)
{
    if (!std::isfinite(value))
    {
        throw_geometry(std::string(name) + " must be finite");
    }
}

float normalize_radians(float angle)
{
    while (angle > kPi)
    {
        angle -= 2.0F * kPi;
    }
    while (angle < -kPi)
    {
        angle += 2.0F * kPi;
    }
    return angle;
}

Point2f model_normalized_to_source(float x, float y,
                                   const image::LetterboxTransform& letterbox,
                                   std::int32_t model_extent)
{
    return { (x * static_cast<float>(model_extent) - letterbox.pad_x) / letterbox.scale,
             (y * static_cast<float>(model_extent) - letterbox.pad_y) / letterbox.scale };
}

} // namespace

PalmDetection decode_palm_row(const PalmRow& row,
                              const image::LetterboxTransform& letterbox,
                              std::int32_t model_extent)
{
    const float values[] = { row.score, row.box_x, row.box_y, row.box_size,
                             row.keypoint0_x, row.keypoint0_y,
                             row.keypoint2_x, row.keypoint2_y };
    for (float value : values)
    {
        require_finite(value, "Palm value");
    }
    if (model_extent <= 0 || letterbox.scale <= 0.0F ||
        letterbox.source_width <= 0 || letterbox.source_height <= 0)
    {
        throw_geometry("Palm transform dimensions and scale must be positive");
    }
    if (row.box_size <= 0.0F)
    {
        throw_geometry("Palm box size must be positive");
    }

    const Point2f box_center = model_normalized_to_source(
        row.box_x, row.box_y, letterbox, model_extent);
    const Point2f keypoint0 = model_normalized_to_source(
        row.keypoint0_x, row.keypoint0_y, letterbox, model_extent);
    const Point2f keypoint2 = model_normalized_to_source(
        row.keypoint2_x, row.keypoint2_y, letterbox, model_extent);
    const float box_size = row.box_size * static_cast<float>(model_extent) / letterbox.scale;
    const float delta_x = keypoint2.x - keypoint0.x;
    const float delta_y = keypoint2.y - keypoint0.y;
    const float rotation = normalize_radians(0.5F * kPi - std::atan2(-delta_y, delta_x));

    PalmDetection result;
    result.confidence = row.score;
    result.box = { box_center.x - box_size * 0.5F,
                   box_center.y - box_size * 0.5F, box_size, box_size };
    result.wrist_keypoint = keypoint0;
    result.middle_finger_keypoint = keypoint2;
    result.roi.center = {
        box_center.x + kHandCenterShift * box_size * std::sin(rotation),
        box_center.y - kHandCenterShift * box_size * std::cos(rotation),
    };
    result.roi.size = kHandRoiScale * box_size;
    result.roi.rotation_radians = rotation;
    return result;
}

image::AffineTransform hand_roi_transform(const RotatedRoi& roi,
                                          std::int32_t destination_extent)
{
    require_finite(roi.center.x, "hand ROI center x");
    require_finite(roi.center.y, "hand ROI center y");
    require_finite(roi.size, "hand ROI size");
    require_finite(roi.rotation_radians, "hand ROI rotation");
    if (roi.size <= 0.0F || destination_extent <= 0)
    {
        throw_geometry("hand ROI size and destination extent must be positive");
    }
    const float scale  = roi.size / static_cast<float>(destination_extent);
    const float cosine = std::cos(roi.rotation_radians);
    const float sine   = std::sin(roi.rotation_radians);
    const float center = (static_cast<float>(destination_extent) - 1.0F) * 0.5F;
    image::AffineTransform transform;
    transform.destination_to_source = {
        cosine * scale,
        -sine * scale,
        roi.center.x - cosine * scale * center + sine * scale * center,
        sine * scale,
        cosine * scale,
        roi.center.y - sine * scale * center - cosine * scale * center,
    };
    return transform;
}

Point2f transform_point(const image::AffineTransform& transform, float x, float y)
{
    require_finite(x, "point x");
    require_finite(y, "point y");
    const auto& matrix = transform.destination_to_source;
    for (float coefficient : matrix)
    {
        require_finite(coefficient, "affine coefficient");
    }
    return { matrix[0] * x + matrix[1] * y + matrix[2],
             matrix[3] * x + matrix[4] * y + matrix[5] };
}

} // namespace kfcore::hand_models::detail
