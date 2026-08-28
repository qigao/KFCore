#include "decode.hpp"

#include "kfcore/vision_models/error.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace kfcore::vision_models::detail
{
namespace
{

[[noreturn]] void throw_contract(const std::string& detail)
{
    throw VisionModelError(VisionModelErrorCode::ModelContractMismatch,
                           "vision decode stage: " + detail);
}

void require_finite(float value, const char* name)
{
    if (!std::isfinite(value))
    {
        throw_contract(std::string(name) + " must be finite");
    }
}

} // namespace

std::vector<PalmDetection> decode_palms(
    const float* values, std::size_t value_count, float confidence_threshold,
    std::size_t max_candidates, std::size_t max_hands,
    const image::LetterboxTransform& letterbox, std::int32_t model_extent)
{
    if ((values == nullptr && value_count != 0U) || value_count % kPalmRowWidth != 0U)
    {
        throw_contract("Palm output must contain complete rows of eight FP32 values");
    }
    require_finite(confidence_threshold, "Palm confidence threshold");
    if (confidence_threshold < 0.0F || confidence_threshold > 1.0F ||
        max_candidates == 0U || max_hands == 0U)
    {
        throw_contract("Palm thresholds and capacities are invalid");
    }
    const std::size_t row_count = value_count / kPalmRowWidth;
    if (row_count > max_candidates)
    {
        throw VisionModelError(VisionModelErrorCode::ResourceLimitExceeded,
                               "vision decode stage: Palm candidates exceed configured limit");
    }

    std::vector<PalmDetection> results;
    results.reserve((std::min)(row_count, max_hands));
    for (std::size_t row_index = 0; row_index < row_count; ++row_index)
    {
        const float* row = values + row_index * kPalmRowWidth;
        require_finite(row[0], "Palm confidence");
        if (row[0] < confidence_threshold)
        {
            continue;
        }
        results.push_back(decode_palm_row(
            { row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7] },
            letterbox, model_extent));
    }
    std::stable_sort(results.begin(), results.end(),
                     [](const PalmDetection& left, const PalmDetection& right)
                     { return left.confidence > right.confidence; });
    if (results.size() > max_hands)
    {
        results.resize(max_hands);
    }
    return results;
}

std::array<HandLandmark, kHandLandmarkCount> decode_hand_landmarks(
    const float* values, std::size_t value_count, const RotatedRoi& roi,
    std::int32_t model_extent)
{
    constexpr std::size_t kExpectedValues = kHandLandmarkCount * 3U;
    if (values == nullptr || value_count != kExpectedValues || model_extent <= 0)
    {
        throw_contract("hand landmark output must be exactly 21x3 FP32 values");
    }
    const image::AffineTransform transform = hand_roi_transform(roi, model_extent);
    std::array<HandLandmark, kHandLandmarkCount> result {};
    const float z_scale = roi.size / static_cast<float>(model_extent);
    for (std::size_t index = 0; index < kHandLandmarkCount; ++index)
    {
        const float x = values[index * 3U];
        const float y = values[index * 3U + 1U];
        const float z = values[index * 3U + 2U];
        require_finite(x, "hand landmark x");
        require_finite(y, "hand landmark y");
        require_finite(z, "hand landmark z");
        const Point2f mapped = transform_point(transform, x, y);
        result[index] = { mapped.x, mapped.y, z * z_scale };
    }
    return result;
}

std::array<float, kHandLandmarkCount * 2U> make_keypoint_features(
    const std::array<HandLandmark, kHandLandmarkCount>& landmarks)
{
    std::array<float, kHandLandmarkCount * 2U> result {};
    const float origin_x = landmarks[0].x;
    const float origin_y = landmarks[0].y;
    require_finite(origin_x, "wrist x");
    require_finite(origin_y, "wrist y");
    float maximum = 0.0F;
    for (std::size_t index = 0; index < kHandLandmarkCount; ++index)
    {
        require_finite(landmarks[index].x, "hand landmark x");
        require_finite(landmarks[index].y, "hand landmark y");
        result[index * 2U]     = landmarks[index].x - origin_x;
        result[index * 2U + 1] = landmarks[index].y - origin_y;
        maximum = (std::max)(maximum, std::fabs(result[index * 2U]));
        maximum = (std::max)(maximum, std::fabs(result[index * 2U + 1U]));
    }
    if (maximum > 0.0F)
    {
        for (float& value : result)
        {
            value /= maximum;
        }
    }
    return result;
}

Handedness decode_handedness(float value)
{
    require_finite(value, "handedness");
    if (value < 0.0F || value > 1.0F)
    {
        throw_contract("handedness must be within [0,1]");
    }
    return value < 0.5F ? Handedness::Left : Handedness::Right;
}

Gesture decode_gesture(std::int64_t class_id) noexcept
{
    switch (class_id)
    {
    case 0:
        return Gesture::Open;
    case 1:
        return Gesture::Closed;
    case 2:
        return Gesture::Pointer;
    default:
        return Gesture::Unknown;
    }
}

std::optional<FaceDetection> decode_yolo12_face(
    const float* values, std::size_t value_count, std::int32_t face_class_id,
    float confidence_threshold, const image::LetterboxTransform& letterbox,
    std::int32_t image_width, std::int32_t image_height)
{
    constexpr std::size_t kValuesPerDetection = 6U;
    if ((values == nullptr && value_count != 0U) ||
        value_count % kValuesPerDetection != 0U)
    {
        throw_contract("YOLOv12 face output must contain complete rows of six FP32 values");
    }
    require_finite(confidence_threshold, "YOLOv12 face confidence threshold");
    require_finite(letterbox.scale, "YOLOv12 face letterbox scale");
    require_finite(letterbox.pad_x, "YOLOv12 face letterbox pad_x");
    require_finite(letterbox.pad_y, "YOLOv12 face letterbox pad_y");
    if (face_class_id < 0 || confidence_threshold < 0.0F ||
        confidence_threshold > 1.0F || letterbox.scale <= 0.0F ||
        image_width <= 0 || image_height <= 0)
    {
        throw_contract("YOLOv12 face decode configuration is invalid");
    }
    if (letterbox.source_width != image_width ||
        letterbox.source_height != image_height)
    {
        throw_contract("YOLOv12 face letterbox source dimensions do not match the image");
    }

    std::optional<FaceDetection> best;
    const std::size_t row_count = value_count / kValuesPerDetection;
    for (std::size_t index = 0; index < row_count; ++index)
    {
        const float* row = values + index * kValuesPerDetection;
        for (std::size_t value = 0; value < kValuesPerDetection; ++value)
        {
            require_finite(row[value], "YOLOv12 face output value");
        }
        const float score = row[4];
        if (score < 0.0F || score > 1.0F)
        {
            throw_contract("YOLOv12 face score must be within [0,1]");
        }
        if (score == 0.0F)
        {
            continue;
        }
        if (row[0] > row[2] || row[1] > row[3])
        {
            throw_contract("YOLOv12 face box coordinates are invalid");
        }
        if (row[5] < 0.0F ||
            row[5] > static_cast<float>((std::numeric_limits<std::int32_t>::max)()) ||
            std::trunc(row[5]) != row[5])
        {
            throw_contract("YOLOv12 face class id is invalid");
        }
        const auto class_id = static_cast<std::int32_t>(row[5]);
        if (class_id != face_class_id || score < confidence_threshold)
        {
            continue;
        }

        const float inverse_scale = 1.0F / letterbox.scale;
        const float left = std::clamp((row[0] - letterbox.pad_x) * inverse_scale,
                                      0.0F, static_cast<float>(image_width));
        const float top = std::clamp((row[1] - letterbox.pad_y) * inverse_scale,
                                     0.0F, static_cast<float>(image_height));
        const float right = std::clamp((row[2] - letterbox.pad_x) * inverse_scale,
                                       0.0F, static_cast<float>(image_width));
        const float bottom = std::clamp((row[3] - letterbox.pad_y) * inverse_scale,
                                        0.0F, static_cast<float>(image_height));
        if (left >= right || top >= bottom)
        {
            throw_contract("YOLOv12 face restored box has no positive area");
        }
        const FaceDetection candidate {
            { left, top, right - left, bottom - top }, score
        };
        if (!best || candidate.confidence > best->confidence)
        {
            best = candidate;
        }
    }
    return best;
}

std::array<Point3f, kFaceLandmarkCount> decode_face_landmarks(
    const float* values, std::size_t value_count, const FaceRoi& roi,
    std::int32_t model_extent, bool normalized_coordinates)
{
    constexpr std::size_t kExpectedValues = kFaceLandmarkCount * 3U;
    if (values == nullptr || value_count != kExpectedValues || model_extent <= 0 ||
        roi.side <= 0.0F)
    {
        throw_contract("face landmark output must be exactly 468x3 FP32 values");
    }
    std::array<Point3f, kFaceLandmarkCount> result {};
    const float coordinate_scale = normalized_coordinates
                                       ? static_cast<float>(model_extent)
                                       : 1.0F;
    const float z_scale = roi.side / static_cast<float>(model_extent);
    for (std::size_t index = 0; index < kFaceLandmarkCount; ++index)
    {
        float x = values[index * 3U];
        float y = values[index * 3U + 1U];
        float z = values[index * 3U + 2U];
        require_finite(x, "face landmark x");
        require_finite(y, "face landmark y");
        require_finite(z, "face landmark z");
        x *= coordinate_scale;
        y *= coordinate_scale;
        z *= coordinate_scale;
        const Point2f mapped = transform_point(roi.destination_to_source, x, y);
        result[index] = { mapped.x, mapped.y, z * z_scale };
    }
    return result;
}

} // namespace kfcore::vision_models::detail
