#include "kfcore/image_processor/image_processor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <utility>

namespace kfcore::image
{
namespace
{

    constexpr std::size_t kImageChannels = 3;

    [[noreturn]] void throw_invalid(std::string message)
    {
        throw ImageProcessorError(ImageProcessorErrorCode::InvalidArgument, std::move(message));
    }

    [[noreturn]] void throw_resource(std::string message)
    {
        throw ImageProcessorError(ImageProcessorErrorCode::ResourceLimitExceeded,
                                  std::move(message));
    }

    std::size_t checked_add(std::size_t left, std::size_t right, const char* stage)
    {
        if (right > (std::numeric_limits<std::size_t>::max)() - left)
        {
            throw_resource(std::string(stage) + " stage: byte count overflow");
        }
        return left + right;
    }

    std::size_t checked_multiply(std::size_t left, std::size_t right, const char* stage)
    {
        if (left != 0 && right > (std::numeric_limits<std::size_t>::max)() / left)
        {
            throw_resource(std::string(stage) + " stage: byte count overflow");
        }
        return left * right;
    }

    std::size_t element_size(TensorElementType type)
    {
        switch (type)
        {
        case TensorElementType::Float16:
            return sizeof(std::uint16_t);
        case TensorElementType::Float32:
            return sizeof(float);
        }
        throw_invalid("tensor validation stage: tensor element type is unsupported");
    }

    void validate_pixel_format(PixelFormat format, const char* stage)
    {
        switch (format)
        {
        case PixelFormat::Bgr8:
        case PixelFormat::Rgb8:
            return;
        }
        throw_invalid(std::string(stage) + " stage: pixel format is unsupported");
    }

    void validate_memory_kind(MemoryKind memory_kind, const char* stage)
    {
        switch (memory_kind)
        {
        case MemoryKind::Host:
        case MemoryKind::CudaDevice:
            return;
        }
        throw_invalid(std::string(stage) + " stage: memory kind is unsupported");
    }

    LetterboxTransform compute_transform(std::int32_t source_width, std::int32_t source_height,
                                         std::int32_t destination_width,
                                         std::int32_t destination_height)
    {
        const float scale =
            (std::min)(static_cast<float>(destination_width) / static_cast<float>(source_width),
                       static_cast<float>(destination_height) / static_cast<float>(source_height));
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

    std::size_t validate_tensor(const TensorView& tensor, std::size_t image_count,
                                std::size_t max_tensor_bytes)
    {
        if (tensor.data == nullptr)
        {
            throw_invalid("tensor validation stage: tensor data must not be null");
        }
        if (tensor.batch <= 0 || tensor.channels != static_cast<std::int32_t>(kImageChannels) ||
            tensor.height <= 0 || tensor.width <= 0)
        {
            throw_invalid("tensor validation stage: tensor must have positive NCHW dimensions and "
                          "3 channels");
        }
        if (static_cast<std::size_t>(tensor.batch) != image_count)
        {
            throw_invalid("tensor validation stage: tensor batch must match the image batch");
        }
        if (tensor.layout != TensorLayout::Nchw)
        {
            throw_invalid("tensor validation stage: tensor layout must be NCHW");
        }
        if (tensor.memory_kind != MemoryKind::CudaDevice)
        {
            throw_invalid(
                "tensor validation stage: destination tensor must use CUDA device memory");
        }

        std::size_t bytes = checked_multiply(image_count, kImageChannels, "tensor validation");
        bytes =
            checked_multiply(bytes, static_cast<std::size_t>(tensor.height), "tensor validation");
        bytes =
            checked_multiply(bytes, static_cast<std::size_t>(tensor.width), "tensor validation");
        bytes = checked_multiply(bytes, element_size(tensor.element_type), "tensor validation");
        if (bytes > max_tensor_bytes)
        {
            throw_resource("tensor validation stage: tensor bytes exceed configured limit");
        }
        if (tensor.byte_size < bytes)
        {
            throw_invalid(
                "tensor validation stage: tensor capacity is smaller than required bytes");
        }
        return bytes;
    }

    void validate_stage_plan(const std::vector<ImageView>& images, const BatchPlan& plan)
    {
        if (images.size() != plan.images.size())
        {
            throw_invalid("host staging stage: plan image count does not match the batch");
        }

        std::size_t expected_staging_bytes = 0;
        for (std::size_t index = 0; index < images.size(); ++index)
        {
            const ImageView& image      = images[index];
            const ImagePlan& image_plan = plan.images[index];
            if (image.data == nullptr)
            {
                throw_invalid("host staging stage: image data must not be null");
            }
            if (image.width <= 0 || image.height <= 0)
            {
                throw_invalid("host staging stage: image dimensions must be positive");
            }
            if (image.width != image_plan.transform.source_width ||
                image.height != image_plan.transform.source_height)
            {
                throw_invalid("host staging stage: image dimensions do not match the plan");
            }
            validate_pixel_format(image.pixel_format, "host staging");
            validate_memory_kind(image.memory_kind, "host staging");
            const std::size_t row_bytes = checked_multiply(static_cast<std::size_t>(image.width),
                                                           kImageChannels, "host staging");
            if (image.row_stride < row_bytes)
            {
                throw_invalid("host staging stage: row stride is smaller than packed RGB8");
            }
            const std::size_t source_span =
                checked_add(checked_multiply(static_cast<std::size_t>(image.height - 1),
                                             image.row_stride, "host staging"),
                            row_bytes, "host staging");
            if (source_span > image.byte_size)
            {
                throw_invalid(
                    "host staging stage: image capacity is smaller than required source span");
            }
            if (image_plan.source_span_bytes != source_span)
            {
                throw_invalid("host staging stage: source view changed after planning");
            }
            const bool should_stage = image.memory_kind == MemoryKind::Host;
            if (image_plan.requires_staging != should_stage)
            {
                throw_invalid("host staging stage: plan memory kind does not match the image");
            }
            if (!should_stage)
            {
                continue;
            }
            const std::size_t packed_bytes =
                checked_multiply(row_bytes, static_cast<std::size_t>(image.height), "host staging");
            if (image_plan.packed_bytes != packed_bytes ||
                image_plan.staging_offset != expected_staging_bytes)
            {
                throw_invalid("host staging stage: plan offsets are inconsistent");
            }
            expected_staging_bytes =
                checked_add(expected_staging_bytes, packed_bytes, "host staging");
        }
        if (expected_staging_bytes != plan.host_staging_bytes ||
            plan.device_staging_bytes != plan.host_staging_bytes)
        {
            throw_invalid("host staging stage: plan workspace bytes are inconsistent");
        }
    }

} // namespace

ImageProcessorError::ImageProcessorError(ImageProcessorErrorCode code, std::string message)
    : std::runtime_error(std::move(message))
    , code_(code)
{
}

ImageProcessorErrorCode ImageProcessorError::code() const noexcept
{
    return code_;
}

LetterboxTransform ImageProcessor::letterbox_transform(std::int32_t source_width,
                                                       std::int32_t source_height,
                                                       std::int32_t destination_width,
                                                       std::int32_t destination_height)
{
    if (source_width <= 0 || source_height <= 0 || destination_width <= 0 ||
        destination_height <= 0)
    {
        throw_invalid("letterbox transform stage: dimensions must be positive");
    }
    return compute_transform(source_width, source_height, destination_width, destination_height);
}

BatchPlan ImageProcessor::plan(const std::vector<ImageView>& images, const TensorView& destination,
                               std::size_t max_source_bytes, std::size_t max_tensor_bytes)
{
    try
    {
        if (images.empty())
        {
            throw_invalid("input validation stage: batch must not be empty");
        }
        if (max_source_bytes == 0 || max_tensor_bytes == 0)
        {
            throw_resource("input validation stage: byte limits must be positive");
        }

        BatchPlan result;
        result.tensor_bytes = validate_tensor(destination, images.size(), max_tensor_bytes);
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
            validate_pixel_format(image.pixel_format, "input validation");
            validate_memory_kind(image.memory_kind, "input validation");

            const std::size_t width  = static_cast<std::size_t>(image.width);
            const std::size_t height = static_cast<std::size_t>(image.height);
            const std::size_t row_bytes =
                checked_multiply(width, kImageChannels, "input validation");
            if (image.row_stride < row_bytes)
            {
                throw_invalid("input validation stage: row stride is smaller than packed RGB8");
            }
            const std::size_t source_span =
                checked_add(checked_multiply(height - 1, image.row_stride, "input validation"),
                            row_bytes, "input validation");
            if (source_span > image.byte_size)
            {
                throw_invalid(
                    "input validation stage: image capacity is smaller than required source span");
            }
            total_source_span = checked_add(total_source_span, source_span, "input validation");
            if (total_source_span > max_source_bytes)
            {
                throw_resource("input validation stage: source bytes exceed configured limit");
            }

            const std::size_t packed_bytes =
                checked_multiply(row_bytes, height, "input validation");
            const bool        requires_staging = image.memory_kind == MemoryKind::Host;
            const std::size_t staging_offset   = result.host_staging_bytes;
            if (requires_staging)
            {
                result.host_staging_bytes =
                    checked_add(result.host_staging_bytes, packed_bytes, "input validation");
                if (result.host_staging_bytes > max_source_bytes)
                {
                    throw_resource(
                        "input validation stage: host staging bytes exceed configured limit");
                }
            }
            result.images.push_back({
                letterbox_transform(image.width, image.height, destination.width,
                                    destination.height),
                source_span,
                packed_bytes,
                staging_offset,
                requires_staging,
            });
        }
        result.device_staging_bytes = result.host_staging_bytes;
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

void ImageProcessor::stage_host_inputs(const std::vector<ImageView>& images, const BatchPlan& plan,
                                       MutableBufferView pinned_host_workspace)
{
    validate_stage_plan(images, plan);
    if (plan.host_staging_bytes == 0)
    {
        return;
    }
    if (pinned_host_workspace.data == nullptr)
    {
        throw_invalid("host staging stage: workspace data must not be null");
    }
    if (pinned_host_workspace.byte_size < plan.host_staging_bytes)
    {
        throw_invalid("host staging stage: workspace capacity is smaller than required bytes");
    }

    auto* destination = static_cast<std::byte*>(pinned_host_workspace.data);
    for (std::size_t index = 0; index < images.size(); ++index)
    {
        const ImageView& image      = images[index];
        const ImagePlan& image_plan = plan.images[index];
        if (!image_plan.requires_staging)
        {
            continue;
        }
        const std::size_t row_bytes =
            image_plan.packed_bytes / static_cast<std::size_t>(image.height);
        const auto* source = static_cast<const std::byte*>(image.data);
        for (std::int32_t row = 0; row < image.height; ++row)
        {
            std::memcpy(destination + image_plan.staging_offset +
                            static_cast<std::size_t>(row) * row_bytes,
                        source + static_cast<std::size_t>(row) * image.row_stride, row_bytes);
        }
    }
}

} // namespace kfcore::image
