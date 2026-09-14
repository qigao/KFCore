#include "efficient_nms.hpp"

#include "kfcore/yolo/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <string>

namespace kfcore::yolo::detail
{
namespace
{

constexpr std::size_t kBoxCoordinates = 4U;

[[noreturn]] void throw_contract(const std::string& message)
{
    throw YoloError(YoloErrorCode::EngineContractMismatch,
                    "YOLO EfficientNMS output: " + message);
}

[[noreturn]] void throw_resource(const std::string& message)
{
    throw YoloError(YoloErrorCode::ResourceLimitExceeded,
                    "YOLO EfficientNMS output: " + message);
}

std::size_t checked_multiply(std::size_t left, std::size_t right)
{
    if (left != 0U && right > (std::numeric_limits<std::size_t>::max)() / left)
    {
        throw_resource("element count overflow");
    }
    return left * right;
}

float half_to_float(std::uint16_t bits) noexcept
{
    const bool negative = (bits & UINT16_C(0x8000)) != 0;
    const std::uint16_t exponent = static_cast<std::uint16_t>((bits >> 10U) & 0x1fU);
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

float floating_value(const void* data, std::size_t index, TensorDataType type)
{
    if (data == nullptr)
    {
        throw_contract("floating tensor pointer is null");
    }
    switch (type)
    {
    case TensorDataType::Float32:
        return static_cast<const float*>(data)[index];
    case TensorDataType::Float16:
        return half_to_float(static_cast<const std::uint16_t*>(data)[index]);
    case TensorDataType::Int32:
        throw_contract("floating tensor cannot use Int32");
    }
    throw_contract("unknown floating tensor type");
}

void require_size(std::size_t actual, std::size_t expected, const char* name)
{
    if (actual != expected)
    {
        throw_contract(std::string(name) + " element count mismatch");
    }
}

} // namespace

std::vector<DetectionFrame> decode_efficient_nms(
    const std::vector<ImageView>& images,
    const std::vector<LetterboxTransform>& transforms,
    const EfficientNmsOutputView& outputs)
{
    try
    {
        if (images.empty() || transforms.size() != images.size())
        {
            throw_contract("image and transform counts differ");
        }
        if (outputs.num_dets == nullptr || outputs.boxes == nullptr ||
            outputs.scores == nullptr || outputs.labels == nullptr ||
            outputs.max_detections == 0U)
        {
            throw_contract("output pointer is null or max_detections is zero");
        }

        const std::size_t slots = checked_multiply(images.size(), outputs.max_detections);
        require_size(outputs.num_dets_count, images.size(), "num_dets");
        require_size(outputs.boxes_count, checked_multiply(slots, kBoxCoordinates), "boxes");
        require_size(outputs.scores_count, slots, "scores");
        require_size(outputs.labels_count, slots, "labels");

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
                throw_contract("letterbox transform is invalid");
            }

            const std::int32_t count = outputs.num_dets[image_index];
            if (count < 0 || static_cast<std::size_t>(count) > outputs.max_detections)
            {
                throw_contract("num_dets is outside valid range");
            }

            DetectionFrame frame{image.width, image.height, {}};
            frame.detections.reserve(static_cast<std::size_t>(count));
            const std::size_t base = image_index * outputs.max_detections;
            for (std::size_t detection_index = 0U;
                 detection_index < static_cast<std::size_t>(count); ++detection_index)
            {
                const std::size_t slot = base + detection_index;
                const std::size_t box_base = slot * kBoxCoordinates;
                const float left = floating_value(outputs.boxes, box_base, outputs.output_type);
                const float top = floating_value(outputs.boxes, box_base + 1U, outputs.output_type);
                const float right = floating_value(outputs.boxes, box_base + 2U, outputs.output_type);
                const float bottom = floating_value(outputs.boxes, box_base + 3U, outputs.output_type);
                const float score = floating_value(outputs.scores, slot, outputs.output_type);
                const std::int32_t label = outputs.labels[slot];
                if (!std::isfinite(left) || !std::isfinite(top) || !std::isfinite(right) ||
                    !std::isfinite(bottom) || !std::isfinite(score) || label < 0 ||
                    left > right || top > bottom)
                {
                    throw_contract("detection contains invalid coordinates, score, or label");
                }

                const float inverse_scale = 1.0F / transform.scale;
                const float mapped_left = (left - transform.pad_x) * inverse_scale;
                const float mapped_top = (top - transform.pad_y) * inverse_scale;
                const float mapped_right = (right - transform.pad_x) * inverse_scale;
                const float mapped_bottom = (bottom - transform.pad_y) * inverse_scale;
                const float image_width = static_cast<float>(image.width);
                const float image_height = static_cast<float>(image.height);
                const BoxF box{
                    (std::clamp)(mapped_left, 0.0F, image_width),
                    (std::clamp)(mapped_top, 0.0F, image_height),
                    (std::clamp)(mapped_right, 0.0F, image_width),
                    (std::clamp)(mapped_bottom, 0.0F, image_height),
                };
                if (!(box.left < box.right) || !(box.top < box.bottom))
                {
                    throw_contract("restored box has no positive area");
                }
                frame.detections.push_back({box, score, label});
            }
            results.push_back(std::move(frame));
        }
        return results;
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("allocation failed");
    }
    catch (const std::length_error&)
    {
        throw_resource("container capacity exceeded");
    }
}

} // namespace kfcore::yolo::detail
