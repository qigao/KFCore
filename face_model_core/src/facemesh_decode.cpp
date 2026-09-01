#include "facemesh_decode.hpp"

#include "kfcore/face_models/error.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace kfcore::face_models::detail
{
namespace
{

[[noreturn]] void throw_contract(const std::string& detail)
{
    throw FaceModelError(FaceModelErrorCode::ModelContractMismatch,
                         "face decode stage: " + detail);
}

void require_finite(float value, const char* name)
{
    if (!std::isfinite(value))
    {
        throw_contract(std::string(name) + " must be finite");
    }
}

} // namespace

std::optional<FaceDetection> decode_yolo12_face(
    const float* values, std::size_t value_count, std::int32_t face_class_id,
    float confidence_threshold, const image::LetterboxTransform& letterbox,
    std::int32_t image_width, std::int32_t image_height)
{
    const std::vector<FaceDetection> faces = decode_yolo12_faces(
        values, value_count, face_class_id, confidence_threshold, letterbox,
        image_width, image_height);
    const auto best = std::max_element(
        faces.begin(), faces.end(),
        [](const FaceDetection& left, const FaceDetection& right)
        {
            return left.confidence < right.confidence;
        });
    return best == faces.end() ? std::nullopt
                               : std::optional<FaceDetection>(*best);
}

std::vector<FaceDetection> decode_yolo12_faces(
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
    if (letterbox.source_width != image_width ||
        letterbox.source_height != image_height)
    {
        throw_contract("YOLOv12 face letterbox source dimensions do not match the image");
    }
    if (face_class_id < 0 || confidence_threshold < 0.0F ||
        confidence_threshold > 1.0F || letterbox.scale <= 0.0F ||
        image_width <= 0 || image_height <= 0)
    {
        throw_contract("YOLOv12 face decode configuration is invalid");
    }

    std::vector<FaceDetection> faces;
    const std::size_t row_count = value_count / kValuesPerDetection;
    for (std::size_t index = 0; index < row_count; ++index)
    {
        const float* row = values + index * kValuesPerDetection;
        for (std::size_t value = 0; value < kValuesPerDetection; ++value)
        {
            require_finite(row[value], "YOLOv12 face output value");
        }
        const float score = row[4];
        if (row[0] > row[2] || row[1] > row[3])
        {
            throw_contract("YOLOv12 face output box is invalid");
        }
        if (score < 0.0F || score > 1.0F || row[5] < 0.0F ||
            row[5] > static_cast<float>((std::numeric_limits<std::int32_t>::max)()) ||
            std::trunc(row[5]) != row[5])
        {
            throw_contract("YOLOv12 face output row is invalid");
        }
        if (score == 0.0F || static_cast<std::int32_t>(row[5]) != face_class_id ||
            score < confidence_threshold)
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
        faces.push_back(FaceDetection {
            { left, top, right - left, bottom - top }, score
        });
    }
    return faces;
}

std::array<Point3f, kFaceMeshLandmarkCount> decode_face_landmarks(
    const float* values, std::size_t value_count, const FaceRoi& roi,
    std::int32_t model_extent, bool normalized_coordinates)
{
    constexpr std::size_t kExpectedValues = kFaceMeshLandmarkCount * 3U;
    if (values == nullptr || value_count != kExpectedValues || model_extent <= 0 ||
        roi.side <= 0.0F)
    {
        throw_contract("face landmark output must be exactly 468x3 FP32 values");
    }
    std::array<Point3f, kFaceMeshLandmarkCount> result {};
    const float coordinate_scale = normalized_coordinates
                                       ? static_cast<float>(model_extent)
                                       : 1.0F;
    const float z_scale = roi.side / static_cast<float>(model_extent);
    const auto& matrix = roi.destination_to_source.destination_to_source;
    for (std::size_t index = 0; index < kFaceMeshLandmarkCount; ++index)
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
        result[index] = { matrix[0] * x + matrix[1] * y + matrix[2],
                          matrix[3] * x + matrix[4] * y + matrix[5],
                          z * z_scale };
    }
    return result;
}

} // namespace kfcore::face_models::detail
