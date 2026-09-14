#include "compact_nms.hpp"

#include "checked_size.hpp"
#include "kfcore/yolo/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <utility>

namespace kfcore::yolo::detail
{
namespace
{

constexpr std::size_t kCompactDetectionValues = 6U;

[[noreturn]] void throw_resource(std::string message)
{
    throw YoloError(YoloErrorCode::ResourceLimitExceeded, std::move(message));
}

[[noreturn]] void throw_inference(std::string message)
{
    throw YoloError(YoloErrorCode::TensorRtFailure, std::move(message));
}

std::size_t checked_multiply(std::size_t left, std::size_t right)
{
    std::size_t result = 0U;
    if (!checked_multiply_size(left, right, &result))
    {
        throw_resource("output validation stage: byte count overflow");
    }
    return result;
}

float half_to_float(std::uint16_t bits) noexcept
{
    const bool negative = (bits & UINT16_C(0x8000)) != 0;
    const std::uint16_t exponent = static_cast<std::uint16_t>((bits >> 10) & 0x1fU);
    const std::uint16_t fraction = static_cast<std::uint16_t>(bits & 0x03ffU);
    float value = 0.0F;
    if (exponent == 0U)
    {
        value = std::ldexp(static_cast<float>(fraction), -24);
    }
    else if (exponent == 0x1fU)
    {
        value = fraction == 0U ? (std::numeric_limits<float>::infinity)()
                               : (std::numeric_limits<float>::quiet_NaN)();
    }
    else
    {
        value = std::ldexp(static_cast<float>(UINT16_C(0x0400) + fraction),
                           static_cast<int>(exponent) - 25);
    }
    return negative ? -value : value;
}

float floating_value(const void* values, std::size_t index, TensorDataType type)
{
    switch (type)
    {
    case TensorDataType::Float16:
        return half_to_float(static_cast<const std::uint16_t*>(values)[index]);
    case TensorDataType::Float32:
        return static_cast<const float*>(values)[index];
    case TensorDataType::Int32:
        throw_inference("output validation stage: floating output type is invalid");
    }
    throw_inference("output validation stage: floating output type is unknown");
}

} // namespace

std::vector<DetectionFrame> decode_compact_nms(
    const std::vector<ImageView>& images,
    const std::vector<LetterboxTransform>& transforms,
    const CompactNmsOutputView& outputs)
{
    try
    {
        if (images.empty() || transforms.size() != images.size())
        {
            throw_inference(
                "output validation stage: image and transform vector size is inconsistent");
        }
        if (outputs.detections == nullptr && outputs.detections_count != 0U)
        {
            throw_inference(
                "output validation stage: non-empty detections require output storage");
        }
        if (outputs.output_type != TensorDataType::Float16 &&
            outputs.output_type != TensorDataType::Float32)
        {
            throw_inference("output validation stage: floating output type is invalid");
        }
        const std::size_t expected_elements = checked_multiply(
            checked_multiply(images.size(), outputs.max_detections),
            kCompactDetectionValues);
        if (outputs.detections_count != expected_elements)
        {
            throw_inference(
                "output validation stage: detections vector size is inconsistent");
        }

        std::vector<DetectionFrame> results;
        results.reserve(images.size());
        for (std::size_t image_index = 0U; image_index < images.size(); ++image_index)
        {
            const ImageView& image = images[image_index];
            const LetterboxTransform& transform = transforms[image_index];
            if (!std::isfinite(transform.scale) || transform.scale <= 0.0F ||
                !std::isfinite(transform.pad_x) || !std::isfinite(transform.pad_y) ||
                transform.source_width != image.width ||
                transform.source_height != image.height)
            {
                throw_inference("output validation stage: letterbox transform is invalid");
            }

            DetectionFrame frame { image.width, image.height, {} };
            frame.detections.reserve(outputs.max_detections);
            const std::size_t image_base =
                image_index * outputs.max_detections * kCompactDetectionValues;
            for (std::size_t detection_index = 0U;
                 detection_index < outputs.max_detections; ++detection_index)
            {
                const std::size_t row =
                    image_base + detection_index * kCompactDetectionValues;
                const float left = floating_value(outputs.detections, row, outputs.output_type);
                const float top = floating_value(outputs.detections, row + 1U,
                                                 outputs.output_type);
                const float right = floating_value(outputs.detections, row + 2U,
                                                   outputs.output_type);
                const float bottom = floating_value(outputs.detections, row + 3U,
                                                    outputs.output_type);
                const float score = floating_value(outputs.detections, row + 4U,
                                                   outputs.output_type);
                const float class_value = floating_value(outputs.detections, row + 5U,
                                                         outputs.output_type);
                if (!std::isfinite(left) || !std::isfinite(top) || !std::isfinite(right) ||
                    !std::isfinite(bottom))
                {
                    throw_inference("output validation stage: box coordinate is not finite");
                }
                if (!std::isfinite(score) || score < 0.0F || score > 1.0F)
                {
                    throw_inference("output validation stage: score is outside the valid range");
                }
                if (!std::isfinite(class_value))
                {
                    throw_inference("output validation stage: class identifier is not finite");
                }
                if (score == 0.0F)
                {
                    continue;
                }
                if (class_value < 0.0F ||
                    static_cast<double>(class_value) >
                        static_cast<double>((std::numeric_limits<std::int32_t>::max)()) ||
                    std::trunc(class_value) != class_value)
                {
                    throw_inference(
                        "output validation stage: class identifier is outside the valid range");
                }
                if (left > right || top > bottom)
                {
                    throw_inference("output validation stage: box is inverted");
                }

                const float inverse_scale = 1.0F / transform.scale;
                const float mapped_left = (left - transform.pad_x) * inverse_scale;
                const float mapped_top = (top - transform.pad_y) * inverse_scale;
                const float mapped_right = (right - transform.pad_x) * inverse_scale;
                const float mapped_bottom = (bottom - transform.pad_y) * inverse_scale;
                if (!std::isfinite(mapped_left) || !std::isfinite(mapped_top) ||
                    !std::isfinite(mapped_right) || !std::isfinite(mapped_bottom))
                {
                    throw_inference(
                        "output validation stage: inverse box coordinate is not finite");
                }
                const float image_width = static_cast<float>(image.width);
                const float image_height = static_cast<float>(image.height);
                const BoxF restored_box {
                    (std::clamp)(mapped_left, 0.0F, image_width),
                    (std::clamp)(mapped_top, 0.0F, image_height),
                    (std::clamp)(mapped_right, 0.0F, image_width),
                    (std::clamp)(mapped_bottom, 0.0F, image_height),
                };
                if (!(restored_box.left < restored_box.right) ||
                    !(restored_box.top < restored_box.bottom))
                {
                    throw_inference(
                        "output validation stage: restored box must have positive area");
                }
                frame.detections.push_back(
                    { restored_box, score, static_cast<std::int32_t>(class_value) });
            }
            results.push_back(std::move(frame));
        }
        return results;
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("output validation stage: allocation failed");
    }
    catch (const std::length_error&)
    {
        throw_resource("output validation stage: container capacity exceeded");
    }
}

} // namespace kfcore::yolo::detail
