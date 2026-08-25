#include "detector_helpers.hpp"

#include "checked_size.hpp"
#include "kfcore/yolo/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <utility>

namespace kfcore::yolo::detail
{
namespace
{

    constexpr std::size_t kImageChannels = 3;
    constexpr std::size_t kBoxCoordinates = 4;

    [[noreturn]] void throw_invalid(std::string message)
    {
        throw YoloError(YoloErrorCode::InvalidArgument, std::move(message));
    }

    [[noreturn]] void throw_resource(std::string message)
    {
        throw YoloError(YoloErrorCode::ResourceLimitExceeded, std::move(message));
    }

    [[noreturn]] void throw_inference(std::string message)
    {
        throw YoloError(YoloErrorCode::TensorRtFailure, std::move(message));
    }

    std::size_t floating_element_size(TensorDataType type, const char* stage)
    {
        switch (type)
        {
        case TensorDataType::Float16:
            return sizeof(std::uint16_t);
        case TensorDataType::Float32:
            return sizeof(float);
        case TensorDataType::Int32:
            throw_invalid(std::string(stage) + " stage: floating tensor type is invalid");
        }
        throw_invalid(std::string(stage) + " stage: floating tensor type is unknown");
    }

    std::size_t checked_multiply(std::size_t left, std::size_t right, const char* stage)
    {
        std::size_t result = 0;
        if (!checked_multiply_size(left, right, &result))
        {
            throw_resource(std::string(stage) + " stage: byte count overflow");
        }
        return result;
    }

    std::size_t checked_add(std::size_t left, std::size_t right, const char* stage)
    {
        std::size_t result = 0;
        if (!checked_add_size(left, right, &result))
        {
            throw_resource(std::string(stage) + " stage: byte count overflow");
        }
        return result;
    }

    void validate_image_enums(const ImageView& image)
    {
        switch (image.pixel_format)
        {
        case PixelFormat::Bgr8:
        case PixelFormat::Rgb8:
            break;
        default:
            throw_invalid("input validation stage: unsupported pixel format");
        }
        switch (image.memory_kind)
        {
        case MemoryKind::Host:
        case MemoryKind::CudaDevice:
            break;
        default:
            throw_invalid("input validation stage: unsupported memory kind");
        }
    }

    float half_to_float(std::uint16_t bits) noexcept
    {
        const bool          negative = (bits & UINT16_C(0x8000)) != 0;
        const std::uint16_t exponent = static_cast<std::uint16_t>((bits >> 10) & 0x1fU);
        const std::uint16_t fraction = static_cast<std::uint16_t>(bits & 0x03ffU);
        float value = 0.0f;
        if (exponent == 0)
        {
            value = std::ldexp(static_cast<float>(fraction), -24);
        }
        else if (exponent == 0x1fU)
        {
            value = fraction == 0 ? (std::numeric_limits<float>::infinity)()
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

    void require_exact_size(std::size_t actual, std::size_t expected, const char* tensor)
    {
        if (actual != expected)
        {
            throw_inference(std::string("output validation stage: ") + tensor +
                            " vector size is inconsistent");
        }
    }

} // namespace

LetterboxLaunchPlan plan_letterbox_launch(std::size_t destination_width,
                                          std::size_t destination_height)
{
    if (destination_width == 0 || destination_height == 0)
    {
        throw_invalid("letterbox launch stage: dimensions must be positive");
    }

    const std::size_t total_pixels =
        checked_multiply(destination_width, destination_height, "letterbox launch");
    const std::size_t required_blocks =
        total_pixels / kLetterboxThreadsPerBlock +
        (total_pixels % kLetterboxThreadsPerBlock == 0 ? 0U : 1U);
    return {
        total_pixels,
        static_cast<std::uint32_t>((std::min)(required_blocks,
                                              std::size_t { kLetterboxMaxBlocks })),
    };
}

LetterboxTransform compute_letterbox_transform(std::int32_t source_width,
                                               std::int32_t source_height,
                                               std::int32_t destination_width,
                                               std::int32_t destination_height)
{
    if (source_width <= 0 || source_height <= 0 || destination_width <= 0 ||
        destination_height <= 0)
    {
        throw YoloError(YoloErrorCode::InvalidArgument,
                        "letterbox transform stage: dimensions must be positive");
    }

    const float scale = (std::min)(static_cast<float>(destination_width) /
                                       static_cast<float>(source_width),
                                   static_cast<float>(destination_height) /
                                       static_cast<float>(source_height));
    return {
        scale,
        (static_cast<float>(destination_width) - static_cast<float>(source_width) * scale) *
            0.5f,
        (static_cast<float>(destination_height) - static_cast<float>(source_height) * scale) *
            0.5f,
        source_width,
        source_height,
    };
}

BatchInputPlan prepare_batch(const std::vector<ImageView>& images, std::size_t min_batch,
                             std::size_t max_batch, std::int32_t input_width,
                             std::int32_t input_height, TensorDataType input_type,
                             std::size_t max_input_bytes)
{
    try
    {
        if (images.empty())
        {
            throw_invalid("input validation stage: batch must not be empty");
        }
        if (min_batch == 0 || max_batch < min_batch)
        {
            throw_invalid("input validation stage: invalid engine batch profile");
        }
        if (images.size() < min_batch)
        {
            throw_invalid("input validation stage: batch is below the engine profile minimum");
        }
        if (images.size() > max_batch)
        {
            throw_resource("input validation stage: batch exceeds the engine profile maximum");
        }
        if (max_input_bytes == 0)
        {
            throw_resource("input validation stage: input byte limit must be positive");
        }

        const std::size_t element_bytes = floating_element_size(input_type, "input validation");
        BatchInputPlan result {};
        result.images.reserve(images.size());
        std::size_t total_source_span = 0;

        for (const ImageView& image : images)
        {
            if (image.data == nullptr)
            {
                throw_invalid("input validation stage: image data must not be null");
            }
            if (image.width <= 0 || image.height <= 0)
            {
                throw_invalid("input validation stage: image dimensions must be positive");
            }
            validate_image_enums(image);

            const std::size_t width = static_cast<std::size_t>(image.width);
            const std::size_t height = static_cast<std::size_t>(image.height);
            const std::size_t row_bytes =
                checked_multiply(width, kImageChannels, "input validation");
            if (image.row_stride < row_bytes)
            {
                throw_invalid("input validation stage: row stride is smaller than packed RGB8");
            }
            const std::size_t preceding_rows =
                checked_multiply(height - 1, image.row_stride, "input validation");
            const std::size_t source_span =
                checked_add(preceding_rows, row_bytes, "input validation");
            total_source_span =
                checked_add(total_source_span, source_span, "input validation");
            if (total_source_span > max_input_bytes)
            {
                throw_resource("input validation stage: source bytes exceed configured limit");
            }

            const std::size_t packed_bytes =
                checked_multiply(row_bytes, height, "input validation");
            std::size_t staging_offset = kNoStagingOffset;
            if (image.memory_kind == MemoryKind::Host)
            {
                staging_offset = result.host_staging_bytes;
                result.host_staging_bytes = checked_add(result.host_staging_bytes, packed_bytes,
                                                        "input validation");
            }
            result.images.push_back({
                compute_letterbox_transform(image.width, image.height, input_width, input_height),
                source_span,
                packed_bytes,
                staging_offset,
            });
        }

        result.input_elements = checked_multiply(images.size(), kImageChannels, "input validation");
        result.input_elements = checked_multiply(result.input_elements,
                                                 static_cast<std::size_t>(input_height),
                                                 "input validation");
        result.input_elements = checked_multiply(result.input_elements,
                                                 static_cast<std::size_t>(input_width),
                                                 "input validation");
        result.input_bytes =
            checked_multiply(result.input_elements, element_bytes, "input validation");
        if (result.input_bytes > max_input_bytes || result.host_staging_bytes > max_input_bytes)
        {
            throw_resource("input validation stage: input bytes exceed configured limit");
        }
        return result;
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("input validation stage: allocation failed");
    }
    catch (const std::length_error&)
    {
        throw_resource("input validation stage: container capacity exceeded");
    }
}

DetectionBufferLayout compute_detection_buffer_layout(std::size_t batch,
                                                       std::size_t max_detections,
                                                       TensorDataType output_type,
                                                       std::size_t max_output_bytes)
{
    if (batch == 0 || max_detections == 0)
    {
        throw_invalid("output layout stage: batch and maximum detections must be positive");
    }
    const std::size_t floating_bytes =
        floating_element_size(output_type, "output layout");
    const std::size_t slots = checked_multiply(batch, max_detections, "output layout");

    DetectionBufferLayout result;
    result.num_dets_bytes = checked_multiply(batch, sizeof(std::int32_t), "output layout");
    result.boxes_bytes = checked_multiply(slots, kBoxCoordinates, "output layout");
    result.boxes_bytes = checked_multiply(result.boxes_bytes, floating_bytes, "output layout");
    result.scores_bytes = checked_multiply(slots, floating_bytes, "output layout");
    result.labels_bytes = checked_multiply(slots, sizeof(std::int32_t), "output layout");
    result.total_output_bytes = checked_add(result.num_dets_bytes, result.boxes_bytes,
                                            "output layout");
    result.total_output_bytes = checked_add(result.total_output_bytes, result.scores_bytes,
                                            "output layout");
    result.total_output_bytes = checked_add(result.total_output_bytes, result.labels_bytes,
                                            "output layout");
    if (result.total_output_bytes > max_output_bytes)
    {
        throw_resource("output layout stage: outputs exceed configured byte limit");
    }
    return result;
}

std::vector<DetectionFrame>
decode_efficient_nms(const std::vector<ImageView>& images,
                     const std::vector<LetterboxTransform>& transforms,
                     const EfficientNmsOutputView& outputs)
{
    try
    {
        if (images.empty() || transforms.size() != images.size())
        {
            throw_inference("output validation stage: image and transform vector size is inconsistent");
        }
        if (outputs.num_dets == nullptr || outputs.boxes == nullptr || outputs.scores == nullptr ||
            outputs.labels == nullptr || outputs.max_detections == 0)
        {
            throw_inference("output validation stage: output pointer or maximum detections is invalid");
        }

        const std::size_t slots =
            checked_multiply(images.size(), outputs.max_detections, "output validation");
        const std::size_t box_elements =
            checked_multiply(slots, kBoxCoordinates, "output validation");
        require_exact_size(outputs.num_dets_count, images.size(), "num_dets");
        require_exact_size(outputs.boxes_count, box_elements, "boxes");
        require_exact_size(outputs.scores_count, slots, "scores");
        require_exact_size(outputs.labels_count, slots, "labels");
        (void)floating_element_size(outputs.output_type, "output validation");

        std::vector<DetectionFrame> results;
        results.reserve(images.size());
        for (std::size_t image_index = 0; image_index < images.size(); ++image_index)
        {
            const ImageView& image = images[image_index];
            const LetterboxTransform& transform = transforms[image_index];
            if (!std::isfinite(transform.scale) || transform.scale <= 0.0f ||
                !std::isfinite(transform.pad_x) || !std::isfinite(transform.pad_y) ||
                transform.source_width != image.width || transform.source_height != image.height)
            {
                throw_inference("output validation stage: letterbox transform is invalid");
            }

            const std::int32_t count = outputs.num_dets[image_index];
            if (count < 0 || static_cast<std::size_t>(count) > outputs.max_detections)
            {
                throw_inference("output validation stage: num_dets is outside the valid range");
            }

            DetectionFrame frame { image.width, image.height, {} };
            frame.detections.reserve(static_cast<std::size_t>(count));
            const std::size_t detection_base = image_index * outputs.max_detections;
            for (std::size_t detection_index = 0;
                 detection_index < static_cast<std::size_t>(count); ++detection_index)
            {
                const std::size_t slot = detection_base + detection_index;
                const std::size_t box_base = slot * kBoxCoordinates;
                const float left = floating_value(outputs.boxes, box_base,
                                                  outputs.output_type);
                const float top = floating_value(outputs.boxes, box_base + 1,
                                                 outputs.output_type);
                const float right = floating_value(outputs.boxes, box_base + 2,
                                                   outputs.output_type);
                const float bottom = floating_value(outputs.boxes, box_base + 3,
                                                    outputs.output_type);
                const float score =
                    floating_value(outputs.scores, slot, outputs.output_type);
                if (!std::isfinite(score))
                {
                    throw_inference("output validation stage: score is not finite");
                }
                if (!std::isfinite(left) || !std::isfinite(top) || !std::isfinite(right) ||
                    !std::isfinite(bottom))
                {
                    throw_inference("output validation stage: box coordinate is not finite");
                }
                if (left > right || top > bottom)
                {
                    throw_inference("output validation stage: box is inverted");
                }
                const std::int32_t label = outputs.labels[slot];
                if (label < 0)
                {
                    throw_inference("output validation stage: label is outside the valid range");
                }

                const float inverse_scale = 1.0f / transform.scale;
                const float mapped_left = (left - transform.pad_x) * inverse_scale;
                const float mapped_top = (top - transform.pad_y) * inverse_scale;
                const float mapped_right = (right - transform.pad_x) * inverse_scale;
                const float mapped_bottom = (bottom - transform.pad_y) * inverse_scale;
                if (!std::isfinite(mapped_left) || !std::isfinite(mapped_top) ||
                    !std::isfinite(mapped_right) || !std::isfinite(mapped_bottom))
                {
                    throw_inference("output validation stage: inverse box coordinate is not finite");
                }
                const float image_width = static_cast<float>(image.width);
                const float image_height = static_cast<float>(image.height);
                frame.detections.push_back({
                    {
                        (std::clamp)(mapped_left, 0.0f, image_width),
                        (std::clamp)(mapped_top, 0.0f, image_height),
                        (std::clamp)(mapped_right, 0.0f, image_width),
                        (std::clamp)(mapped_bottom, 0.0f, image_height),
                    },
                    score,
                    label,
                });
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
