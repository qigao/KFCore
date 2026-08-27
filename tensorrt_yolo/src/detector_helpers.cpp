#include "detector_helpers.hpp"

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

    constexpr std::size_t kBoxCoordinates = 4;
    constexpr std::size_t kCompactDetectionValues = 6;

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

    [[noreturn]] void throw_processor_error(const kfcore::image::ImageProcessorError& error)
    {
        switch (error.code())
        {
        case kfcore::image::ImageProcessorErrorCode::InvalidArgument:
            throw YoloError(YoloErrorCode::InvalidArgument, error.what());
        case kfcore::image::ImageProcessorErrorCode::ResourceLimitExceeded:
            throw YoloError(YoloErrorCode::ResourceLimitExceeded, error.what());
        case kfcore::image::ImageProcessorErrorCode::CudaFailure:
            throw YoloError(YoloErrorCode::CudaFailure, error.what());
        }
        throw YoloError(YoloErrorCode::InvalidArgument,
                        "image processor returned an unknown error code");
    }

    kfcore::image::PixelFormat processor_pixel_format(PixelFormat format)
    {
        switch (format)
        {
        case PixelFormat::Bgr8:
            return kfcore::image::PixelFormat::Bgr8;
        case PixelFormat::Rgb8:
            return kfcore::image::PixelFormat::Rgb8;
        }
        throw_invalid("input validation stage: unsupported pixel format");
    }

    kfcore::image::MemoryKind processor_memory_kind(MemoryKind memory_kind)
    {
        switch (memory_kind)
        {
        case MemoryKind::Host:
            return kfcore::image::MemoryKind::Host;
        case MemoryKind::CudaDevice:
            return kfcore::image::MemoryKind::CudaDevice;
        }
        throw_invalid("input validation stage: unsupported memory kind");
    }

    std::size_t legacy_source_capacity(const ImageView& image) noexcept
    {
        if (image.width <= 0 || image.height <= 0)
        {
            return (std::numeric_limits<std::size_t>::max)();
        }
        const std::size_t width = static_cast<std::size_t>(image.width);
        if (width > (std::numeric_limits<std::size_t>::max)() / 3U)
        {
            return (std::numeric_limits<std::size_t>::max)();
        }
        const std::size_t row_bytes = width * 3U;
        if (image.row_stride < row_bytes)
        {
            return (std::numeric_limits<std::size_t>::max)();
        }
        const std::size_t preceding_rows = static_cast<std::size_t>(image.height - 1);
        if (preceding_rows != 0 &&
            image.row_stride > (std::numeric_limits<std::size_t>::max)() / preceding_rows)
        {
            return (std::numeric_limits<std::size_t>::max)();
        }
        const std::size_t preceding_bytes = preceding_rows * image.row_stride;
        if (row_bytes > (std::numeric_limits<std::size_t>::max)() - preceding_bytes)
        {
            return (std::numeric_limits<std::size_t>::max)();
        }
        return preceding_bytes + row_bytes;
    }

    kfcore::image::TensorElementType processor_element_type(TensorDataType type)
    {
        switch (type)
        {
        case TensorDataType::Float16:
            return kfcore::image::TensorElementType::Float16;
        case TensorDataType::Float32:
            return kfcore::image::TensorElementType::Float32;
        case TensorDataType::Int32:
            throw_invalid("input validation stage: floating tensor type is invalid");
        }
        throw_invalid("input validation stage: floating tensor type is unknown");
    }

    float half_to_float(std::uint16_t bits) noexcept
    {
        const bool          negative = (bits & UINT16_C(0x8000)) != 0;
        const std::uint16_t exponent = static_cast<std::uint16_t>((bits >> 10) & 0x1fU);
        const std::uint16_t fraction = static_cast<std::uint16_t>(bits & 0x03ffU);
        float               value    = 0.0f;
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

LetterboxTransform compute_letterbox_transform(std::int32_t source_width,
                                               std::int32_t source_height,
                                               std::int32_t destination_width,
                                               std::int32_t destination_height)
{
    try
    {
        return kfcore::image::ImageProcessor::letterbox_transform(
            source_width, source_height, destination_width, destination_height);
    }
    catch (const kfcore::image::ImageProcessorError& error)
    {
        throw_processor_error(error);
    }
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

        BatchInputPlan result {};
        result.source_images.reserve(images.size());
        for (const ImageView& image : images)
        {
            result.source_images.push_back({
                image.data,
                legacy_source_capacity(image),
                image.width,
                image.height,
                image.row_stride,
                processor_pixel_format(image.pixel_format),
                processor_memory_kind(image.memory_kind),
            });
        }

        std::byte                       destination_sentinel {};
        const kfcore::image::TensorView destination {
            &destination_sentinel,
            (std::numeric_limits<std::size_t>::max)(),
            static_cast<std::int32_t>(images.size()),
            3,
            input_height,
            input_width,
            processor_element_type(input_type),
            kfcore::image::TensorLayout::Nchw,
            kfcore::image::MemoryKind::CudaDevice,
        };
        try
        {
            result.processor = kfcore::image::ImageProcessor::plan(
                result.source_images, destination, max_input_bytes,
                (std::numeric_limits<std::size_t>::max)());
        }
        catch (const kfcore::image::ImageProcessorError& error)
        {
            throw_processor_error(error);
        }
        const std::size_t element_bytes = floating_element_size(input_type, "input validation");
        result.input_bytes              = result.processor.tensor_bytes;
        if (result.input_bytes > max_input_bytes)
        {
            throw_resource("input validation stage: input bytes exceed configured limit");
        }
        result.input_elements = result.input_bytes / element_bytes;
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

DetectionBufferLayout compute_detection_buffer_layout(std::size_t batch, std::size_t max_detections,
                                                      TensorDataType output_type,
                                                      std::size_t    max_output_bytes)
{
    if (batch == 0 || max_detections == 0)
    {
        throw_invalid("output layout stage: batch and maximum detections must be positive");
    }
    const std::size_t floating_bytes = floating_element_size(output_type, "output layout");
    const std::size_t slots          = checked_multiply(batch, max_detections, "output layout");

    DetectionBufferLayout result;
    result.num_dets_bytes = checked_multiply(batch, sizeof(std::int32_t), "output layout");
    result.boxes_bytes    = checked_multiply(slots, kBoxCoordinates, "output layout");
    result.boxes_bytes    = checked_multiply(result.boxes_bytes, floating_bytes, "output layout");
    result.scores_bytes   = checked_multiply(slots, floating_bytes, "output layout");
    result.labels_bytes   = checked_multiply(slots, sizeof(std::int32_t), "output layout");
    result.total_output_bytes =
        checked_add(result.num_dets_bytes, result.boxes_bytes, "output layout");
    result.total_output_bytes =
        checked_add(result.total_output_bytes, result.scores_bytes, "output layout");
    result.total_output_bytes =
        checked_add(result.total_output_bytes, result.labels_bytes, "output layout");
    if (result.total_output_bytes > max_output_bytes)
    {
        throw_resource("output layout stage: outputs exceed configured byte limit");
    }
    return result;
}

CompactNmsBufferLayout compute_compact_nms_buffer_layout(std::size_t batch,
                                                         std::size_t max_detections,
                                                         TensorDataType output_type,
                                                         std::size_t max_output_bytes)
{
    if (batch == 0 || max_detections == 0)
    {
        throw_invalid("output layout stage: batch and maximum detections must be positive");
    }
    const std::size_t floating_bytes = floating_element_size(output_type, "output layout");
    std::size_t detections_bytes =
        checked_multiply(batch, max_detections, "output layout");
    detections_bytes =
        checked_multiply(detections_bytes, kCompactDetectionValues, "output layout");
    detections_bytes = checked_multiply(detections_bytes, floating_bytes, "output layout");
    if (detections_bytes > max_output_bytes)
    {
        throw_resource("output layout stage: outputs exceed configured byte limit");
    }
    return { detections_bytes };
}

std::vector<DetectionFrame> decode_efficient_nms(const std::vector<ImageView>&          images,
                                                 const std::vector<LetterboxTransform>& transforms,
                                                 const EfficientNmsOutputView&          outputs)
{
    try
    {
        if (images.empty() || transforms.size() != images.size())
        {
            throw_inference(
                "output validation stage: image and transform vector size is inconsistent");
        }
        if (outputs.num_dets == nullptr || outputs.boxes == nullptr || outputs.scores == nullptr ||
            outputs.labels == nullptr || outputs.max_detections == 0)
        {
            throw_inference(
                "output validation stage: output pointer or maximum detections is invalid");
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
            const ImageView&          image     = images[image_index];
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
            for (std::size_t detection_index = 0; detection_index < static_cast<std::size_t>(count);
                 ++detection_index)
            {
                const std::size_t slot     = detection_base + detection_index;
                const std::size_t box_base = slot * kBoxCoordinates;
                const float left = floating_value(outputs.boxes, box_base, outputs.output_type);
                const float top  = floating_value(outputs.boxes, box_base + 1, outputs.output_type);
                const float right =
                    floating_value(outputs.boxes, box_base + 2, outputs.output_type);
                const float bottom =
                    floating_value(outputs.boxes, box_base + 3, outputs.output_type);
                const float score = floating_value(outputs.scores, slot, outputs.output_type);
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
                const float mapped_left   = (left - transform.pad_x) * inverse_scale;
                const float mapped_top    = (top - transform.pad_y) * inverse_scale;
                const float mapped_right  = (right - transform.pad_x) * inverse_scale;
                const float mapped_bottom = (bottom - transform.pad_y) * inverse_scale;
                if (!std::isfinite(mapped_left) || !std::isfinite(mapped_top) ||
                    !std::isfinite(mapped_right) || !std::isfinite(mapped_bottom))
                {
                    throw_inference(
                        "output validation stage: inverse box coordinate is not finite");
                }
                const float image_width  = static_cast<float>(image.width);
                const float image_height = static_cast<float>(image.height);
                const BoxF  restored_box {
                    (std::clamp)(mapped_left, 0.0f, image_width),
                    (std::clamp)(mapped_top, 0.0f, image_height),
                    (std::clamp)(mapped_right, 0.0f, image_width),
                    (std::clamp)(mapped_bottom, 0.0f, image_height),
                };
                if (!(restored_box.left < restored_box.right) ||
                    !(restored_box.top < restored_box.bottom))
                {
                    throw_inference(
                        "output validation stage: restored box must have positive area");
                }
                frame.detections.push_back({
                    restored_box,
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
        if (outputs.detections == nullptr || outputs.max_detections == 0)
        {
            throw_inference(
                "output validation stage: output pointer or maximum detections is invalid");
        }
        const std::size_t slots =
            checked_multiply(images.size(), outputs.max_detections, "output validation");
        const std::size_t expected_elements =
            checked_multiply(slots, kCompactDetectionValues, "output validation");
        require_exact_size(outputs.detections_count, expected_elements, "detections");
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

            DetectionFrame frame { image.width, image.height, {} };
            frame.detections.reserve(outputs.max_detections);
            const std::size_t image_base =
                image_index * outputs.max_detections * kCompactDetectionValues;
            for (std::size_t detection_index = 0;
                 detection_index < outputs.max_detections; ++detection_index)
            {
                const std::size_t row =
                    image_base + detection_index * kCompactDetectionValues;
                const float left = floating_value(outputs.detections, row, outputs.output_type);
                const float top = floating_value(outputs.detections, row + 1, outputs.output_type);
                const float right =
                    floating_value(outputs.detections, row + 2, outputs.output_type);
                const float bottom =
                    floating_value(outputs.detections, row + 3, outputs.output_type);
                const float score =
                    floating_value(outputs.detections, row + 4, outputs.output_type);
                const float class_value =
                    floating_value(outputs.detections, row + 5, outputs.output_type);
                if (!std::isfinite(left) || !std::isfinite(top) || !std::isfinite(right) ||
                    !std::isfinite(bottom))
                {
                    throw_inference("output validation stage: box coordinate is not finite");
                }
                if (!std::isfinite(score) || score < 0.0f || score > 1.0f)
                {
                    throw_inference("output validation stage: score is outside the valid range");
                }
                if (!std::isfinite(class_value))
                {
                    throw_inference("output validation stage: class identifier is not finite");
                }
                if (score == 0.0f)
                {
                    continue;
                }
                if (class_value < 0.0f ||
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

                const float inverse_scale = 1.0f / transform.scale;
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
                    (std::clamp)(mapped_left, 0.0f, image_width),
                    (std::clamp)(mapped_top, 0.0f, image_height),
                    (std::clamp)(mapped_right, 0.0f, image_width),
                    (std::clamp)(mapped_bottom, 0.0f, image_height),
                };
                if (!(restored_box.left < restored_box.right) ||
                    !(restored_box.top < restored_box.bottom))
                {
                    throw_inference(
                        "output validation stage: restored box must have positive area");
                }
                frame.detections.push_back({ restored_box, score,
                                             static_cast<std::int32_t>(class_value) });
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
