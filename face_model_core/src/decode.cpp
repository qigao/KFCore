#include "decode.hpp"

#include "kfcore/face_models/error.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace kfcore::face_models::detail
{
namespace
{

constexpr std::size_t kFace68ValuesPerImage =
    kFace68LandmarkCount * static_cast<std::size_t>(kFace68LandmarkWidth);
constexpr float kFace68CoordinateScale =
    static_cast<float>(kFace68InputExtent) / static_cast<float>(kFace68HeatmapExtent);
constexpr char kFace68ModelName[]    = "Face68";
constexpr char kArcFaceModelName[]   = "ArcFace";
constexpr char kAgeGenderModelName[] = "AgeGender";

[[noreturn]] void throw_decode(const std::string& model, const std::string& detail)
{
    throw FaceModelError(FaceModelErrorCode::RuntimeFailure,
                         model + " result decoding stage: " + detail);
}

std::size_t checked_expected_elements(std::size_t batch, std::size_t per_image,
                                      const std::string& model)
{
    if (batch == 0U)
    {
        throw_decode(model, "batch must be positive");
    }
    if (per_image != 0U && batch > (std::numeric_limits<std::size_t>::max)() / per_image)
    {
        throw_decode(model, "element count overflow");
    }
    return batch * per_image;
}

void validate_decode_storage(const float* values, std::size_t element_count,
                             std::size_t expected, const std::string& model)
{
    if (values == nullptr)
    {
        throw_decode(model, "output storage must not be null");
    }
    if (element_count != expected)
    {
        throw_decode(model, "output element count does not match the model contract");
    }
}

void validate_finite(const float* values, std::size_t count, const std::string& model,
                     const char* subject)
{
    for (std::size_t index = 0; index < count; ++index)
    {
        if (!std::isfinite(values[index]))
        {
            throw_decode(model, std::string(subject) + " must contain only finite values");
        }
    }
}

} // namespace

std::vector<Face68Result> decode_face68(const float* values, std::size_t element_count,
                                        std::size_t batch)
{
    const std::size_t expected =
        checked_expected_elements(batch, kFace68ValuesPerImage, kFace68ModelName);
    validate_decode_storage(values, element_count, expected, kFace68ModelName);
    std::vector<Face68Result> results(batch);
    for (std::size_t image = 0; image < batch; ++image)
    {
        for (std::size_t point = 0; point < kFace68LandmarkCount; ++point)
        {
            const std::size_t offset = image * kFace68ValuesPerImage + point * 3U;
            const float raw_x = values[offset];
            const float raw_y = values[offset + 1U];
            const float score = values[offset + 2U];
            if (!std::isfinite(raw_x) || !std::isfinite(raw_y) || !std::isfinite(score))
            {
                throw_decode(kFace68ModelName, "landmark x, y, and score must be finite");
            }
            const float scaled_x = raw_x * kFace68CoordinateScale;
            const float scaled_y = raw_y * kFace68CoordinateScale;
            if (!std::isfinite(scaled_x) || !std::isfinite(scaled_y))
            {
                throw_decode(kFace68ModelName,
                             "scaled landmark coordinates must remain finite");
            }
            results[image][point] = { scaled_x, scaled_y, score };
        }
    }
    return results;
}

std::vector<ArcFaceResult> decode_arcface(const float* values, std::size_t element_count,
                                          std::size_t batch)
{
    const std::size_t expected =
        checked_expected_elements(batch, kArcFaceEmbeddingLength, kArcFaceModelName);
    validate_decode_storage(values, element_count, expected, kArcFaceModelName);
    validate_finite(values, expected, kArcFaceModelName, "embedding");
    std::vector<ArcFaceResult> results(batch);
    for (std::size_t image = 0; image < batch; ++image)
    {
        std::copy_n(values + image * kArcFaceEmbeddingLength, kArcFaceEmbeddingLength,
                    results[image].begin());
    }
    return results;
}

std::vector<AgeGenderResult> decode_age_gender(const float* values,
                                               std::size_t element_count,
                                               std::size_t batch)
{
    const std::size_t expected =
        checked_expected_elements(batch, kAgeGenderLogitCount, kAgeGenderModelName);
    validate_decode_storage(values, element_count, expected, kAgeGenderModelName);
    validate_finite(values, expected, kAgeGenderModelName, "logits");
    std::vector<AgeGenderResult> results(batch);
    for (std::size_t image = 0; image < batch; ++image)
    {
        std::copy_n(values + image * kAgeGenderLogitCount, kAgeGenderLogitCount,
                    results[image].begin());
    }
    return results;
}

} // namespace kfcore::face_models::detail
