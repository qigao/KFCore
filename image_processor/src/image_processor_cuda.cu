#include "kfcore/image_processor/image_processor.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace kfcore::image
{
namespace
{

    constexpr float         kPixelScale      = 1.0f / 255.0f;
    constexpr std::uint32_t kThreadsPerBlock = 256;
    constexpr std::uint32_t kMaximumBlocks   = 4096;

    [[noreturn]] void throw_invalid(std::string message)
    {
        throw ImageProcessorError(ImageProcessorErrorCode::InvalidArgument, std::move(message));
    }

    [[noreturn]] void throw_cuda(std::string message)
    {
        throw ImageProcessorError(ImageProcessorErrorCode::CudaFailure, std::move(message));
    }

    void check_cuda(cudaError_t result, const char* operation, const char* stage)
    {
        if (result == cudaSuccess)
        {
            return;
        }
        const char* error_name = cudaGetErrorName(result);
        const char* error_text = cudaGetErrorString(result);
        throw_cuda(std::string(operation) + " failed during " + stage +
                   " stage: " + (error_name != nullptr ? error_name : "unknown CUDA error") + " (" +
                   (error_text != nullptr ? error_text : "no CUDA error description") + ")");
    }

    void validate_pixel_format(PixelFormat format, const char* description)
    {
        if (format != PixelFormat::Rgb8 && format != PixelFormat::Bgr8)
        {
            throw_invalid(std::string("CUDA preprocessing stage: ") + description +
                          " pixel format is unsupported");
        }
    }

    void validate_options(const PreprocessOptions& options)
    {
        validate_pixel_format(options.output_format, "output");
        for (float value : options.mean)
        {
            if (!std::isfinite(value))
            {
                throw_invalid("CUDA preprocessing stage: mean values must be finite");
            }
        }
        for (float value : options.stddev)
        {
            if (!std::isfinite(value) || value <= 0.0f)
            {
                throw_invalid(
                    "CUDA preprocessing stage: stddev values must be finite and positive");
            }
        }
        if (!std::isfinite(options.border_value) || options.border_value < 0.0f ||
            options.border_value > 255.0f)
        {
            throw_invalid("CUDA preprocessing stage: border value must be within [0, 255]");
        }
    }

    bool same_transform(const LetterboxTransform& left, const LetterboxTransform& right) noexcept
    {
        return left.scale == right.scale && left.pad_x == right.pad_x &&
               left.pad_y == right.pad_y && left.source_width == right.source_width &&
               left.source_height == right.source_height;
    }

    void validate_plan(const std::vector<ImageView>& images, const TensorView& destination,
                       const BatchPlan& supplied)
    {
        const BatchPlan expected =
            ImageProcessor::plan(images, destination, (std::numeric_limits<std::size_t>::max)(),
                                 (std::numeric_limits<std::size_t>::max)());
        if (supplied.images.size() != expected.images.size() ||
            supplied.host_staging_bytes != expected.host_staging_bytes ||
            supplied.device_staging_bytes != expected.device_staging_bytes ||
            supplied.tensor_bytes != expected.tensor_bytes)
        {
            throw_invalid("CUDA preprocessing stage: batch plan is inconsistent");
        }
        for (std::size_t index = 0; index < expected.images.size(); ++index)
        {
            const ImagePlan& actual = supplied.images[index];
            const ImagePlan& wanted = expected.images[index];
            if (!same_transform(actual.transform, wanted.transform) ||
                actual.source_span_bytes != wanted.source_span_bytes ||
                actual.packed_bytes != wanted.packed_bytes ||
                actual.staging_offset != wanted.staging_offset ||
                actual.requires_staging != wanted.requires_staging)
            {
                throw_invalid("CUDA preprocessing stage: image plan is inconsistent");
            }
        }
    }

    cudaPointerAttributes pointer_attributes(const void* pointer, const char* description)
    {
        cudaPointerAttributes attributes {};
        check_cuda(cudaPointerGetAttributes(&attributes, pointer), "cudaPointerGetAttributes",
                   description);
        return attributes;
    }

    void validate_device_pointer(const void* pointer, int current_device, const char* description)
    {
        const cudaPointerAttributes attributes = pointer_attributes(pointer, description);
        if (attributes.type != cudaMemoryTypeDevice)
        {
            throw_invalid(std::string("CUDA preprocessing stage: ") + description +
                          " is not CUDA device memory");
        }
        if (attributes.device != current_device)
        {
            throw_invalid(std::string("CUDA preprocessing stage: ") + description +
                          " belongs to a different CUDA device");
        }
    }

    void validate_workspaces(const BatchPlan& plan, MutableBufferView pinned_host_workspace,
                             MutableBufferView device_workspace, int current_device)
    {
        if (plan.host_staging_bytes == 0)
        {
            return;
        }
        if (pinned_host_workspace.data == nullptr || device_workspace.data == nullptr)
        {
            throw_invalid("CUDA preprocessing stage: host inputs require both workspaces");
        }
        if (pinned_host_workspace.byte_size < plan.host_staging_bytes ||
            device_workspace.byte_size < plan.device_staging_bytes)
        {
            throw_invalid(
                "CUDA preprocessing stage: workspace capacity is smaller than required bytes");
        }
        const cudaPointerAttributes host_attributes =
            pointer_attributes(pinned_host_workspace.data, "pinned host workspace validation");
        if (host_attributes.type != cudaMemoryTypeHost)
        {
            throw_invalid("CUDA preprocessing stage: host workspace must be CUDA pinned memory");
        }
        validate_device_pointer(device_workspace.data, current_device,
                                "device workspace validation");
    }

    template <typename Destination> __device__ Destination convert_destination(float value);

    template <> __device__ float convert_destination<float>(float value)
    {
        return value;
    }

    template <> __device__ __half convert_destination<__half>(float value)
    {
        return __float2half(value);
    }

    __device__ int rgb_channel_for_output(int output_channel, PixelFormat output_format)
    {
        return output_format == PixelFormat::Rgb8 ? output_channel : 2 - output_channel;
    }

    __device__ float pixel_channel(const std::uint8_t* source, std::size_t source_stride,
                                   std::int32_t source_width, std::int32_t source_height, int x,
                                   int y, int output_channel, PixelFormat source_format,
                                   PixelFormat output_format)
    {
        x                     = max(0, min(x, source_width - 1));
        y                     = max(0, min(y, source_height - 1));
        const int rgb_channel = rgb_channel_for_output(output_channel, output_format);
        const int source_channel =
            source_format == PixelFormat::Rgb8 ? rgb_channel : 2 - rgb_channel;
        return static_cast<float>(
            source[static_cast<std::size_t>(y) * source_stride + static_cast<std::size_t>(x) * 3U +
                   static_cast<std::size_t>(source_channel)]);
    }

    __device__ float bilinear_channel(const std::uint8_t* source, std::size_t source_stride,
                                      const LetterboxTransform& transform, int destination_x,
                                      int destination_y, int output_channel,
                                      PixelFormat source_format, PixelFormat output_format,
                                      float border_value)
    {
        const float destination_center_x = static_cast<float>(destination_x) + 0.5f;
        const float destination_center_y = static_cast<float>(destination_y) + 0.5f;
        const float content_right =
            transform.pad_x + static_cast<float>(transform.source_width) * transform.scale;
        const float content_bottom =
            transform.pad_y + static_cast<float>(transform.source_height) * transform.scale;
        if (destination_center_x < transform.pad_x || destination_center_x >= content_right ||
            destination_center_y < transform.pad_y || destination_center_y >= content_bottom)
        {
            return border_value;
        }

        float source_x = (destination_center_x - transform.pad_x) / transform.scale - 0.5f;
        float source_y = (destination_center_y - transform.pad_y) / transform.scale - 0.5f;
        source_x = fminf(fmaxf(source_x, 0.0f), static_cast<float>(transform.source_width - 1));
        source_y = fminf(fmaxf(source_y, 0.0f), static_cast<float>(transform.source_height - 1));
        const int   x0       = static_cast<int>(floorf(source_x));
        const int   y0       = static_cast<int>(floorf(source_y));
        const int   x1       = min(x0 + 1, transform.source_width - 1);
        const int   y1       = min(y0 + 1, transform.source_height - 1);
        const float x_weight = source_x - static_cast<float>(x0);
        const float y_weight = source_y - static_cast<float>(y0);

        const float top_left =
            pixel_channel(source, source_stride, transform.source_width, transform.source_height,
                          x0, y0, output_channel, source_format, output_format);
        const float top_right =
            pixel_channel(source, source_stride, transform.source_width, transform.source_height,
                          x1, y0, output_channel, source_format, output_format);
        const float bottom_left =
            pixel_channel(source, source_stride, transform.source_width, transform.source_height,
                          x0, y1, output_channel, source_format, output_format);
        const float bottom_right =
            pixel_channel(source, source_stride, transform.source_width, transform.source_height,
                          x1, y1, output_channel, source_format, output_format);
        const float top    = top_left + (top_right - top_left) * x_weight;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_weight;
        return top + (bottom - top) * y_weight;
    }

    template <typename Destination>
    __global__ void preprocess_kernel(const std::uint8_t* source, std::size_t source_stride,
                                      PixelFormat source_format, PixelFormat output_format,
                                      Destination* destination, std::int32_t destination_width,
                                      std::size_t total_pixels, LetterboxTransform transform,
                                      float mean0, float mean1, float mean2, float stddev0,
                                      float stddev1, float stddev2, float border_value)
    {
        const float       means[3]               = { mean0, mean1, mean2 };
        const float       standard_deviations[3] = { stddev0, stddev1, stddev2 };
        const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::size_t step  = static_cast<std::size_t>(gridDim.x) * blockDim.x;
        for (std::size_t pixel_index = start; pixel_index < total_pixels; pixel_index += step)
        {
            const int x =
                static_cast<int>(pixel_index % static_cast<std::size_t>(destination_width));
            const int y =
                static_cast<int>(pixel_index / static_cast<std::size_t>(destination_width));
            for (int channel = 0; channel < 3; ++channel)
            {
                const float pixel =
                    bilinear_channel(source, source_stride, transform, x, y, channel, source_format,
                                     output_format, border_value);
                const float normalized =
                    (pixel * kPixelScale - means[channel]) / standard_deviations[channel];
                destination[static_cast<std::size_t>(channel) * total_pixels + pixel_index] =
                    convert_destination<Destination>(normalized);
            }
        }
    }

    void launch_image(const std::uint8_t* source, std::size_t source_stride,
                      PixelFormat source_format, const TensorView& destination,
                      const LetterboxTransform& transform, void* destination_data,
                      const PreprocessOptions& options, cudaStream_t stream)
    {
        const std::size_t total_pixels = static_cast<std::size_t>(destination.width) *
                                         static_cast<std::size_t>(destination.height);
        const std::size_t required_blocks =
            total_pixels / kThreadsPerBlock + (total_pixels % kThreadsPerBlock == 0 ? 0U : 1U);
        const dim3 block(kThreadsPerBlock);
        const dim3 grid(static_cast<std::uint32_t>(
            (std::min)(required_blocks, std::size_t { kMaximumBlocks })));

        switch (destination.element_type)
        {
        case TensorElementType::Float16:
            preprocess_kernel<<<grid, block, 0, stream>>>(
                source, source_stride, source_format, options.output_format,
                static_cast<__half*>(destination_data), destination.width, total_pixels, transform,
                options.mean[0], options.mean[1], options.mean[2], options.stddev[0],
                options.stddev[1], options.stddev[2], options.border_value);
            break;
        case TensorElementType::Float32:
            preprocess_kernel<<<grid, block, 0, stream>>>(
                source, source_stride, source_format, options.output_format,
                static_cast<float*>(destination_data), destination.width, total_pixels, transform,
                options.mean[0], options.mean[1], options.mean[2], options.stddev[0],
                options.stddev[1], options.stddev[2], options.border_value);
            break;
        default:
            throw_invalid("CUDA preprocessing stage: tensor element type is unsupported");
        }
        check_cuda(cudaGetLastError(), "preprocess_kernel", "CUDA preprocessing");
    }

} // namespace

void ImageProcessor::enqueue(const std::vector<ImageView>& images, const TensorView& destination,
                             const BatchPlan& plan, MutableBufferView pinned_host_workspace,
                             MutableBufferView device_workspace, const PreprocessOptions& options,
                             void* stream)
{
    validate_options(options);
    validate_plan(images, destination, plan);

    int current_device = 0;
    check_cuda(cudaGetDevice(&current_device), "cudaGetDevice", "CUDA preprocessing");
    validate_device_pointer(destination.data, current_device, "destination tensor validation");
    validate_workspaces(plan, pinned_host_workspace, device_workspace, current_device);
    for (const ImageView& image : images)
    {
        validate_pixel_format(image.pixel_format, "source");
        if (image.memory_kind == MemoryKind::CudaDevice)
        {
            validate_device_pointer(image.data, current_device, "source image validation");
        }
    }

    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (plan.host_staging_bytes != 0)
    {
        check_cuda(cudaMemcpyAsync(device_workspace.data, pinned_host_workspace.data,
                                   plan.host_staging_bytes, cudaMemcpyHostToDevice, cuda_stream),
                   "cudaMemcpyAsync", "host input upload");
    }

    const std::size_t tensor_bytes_per_image = plan.tensor_bytes / images.size();
    auto*             destination_bytes      = static_cast<std::byte*>(destination.data);
    auto*             staged_device_bytes    = static_cast<std::byte*>(device_workspace.data);
    for (std::size_t index = 0; index < images.size(); ++index)
    {
        const ImageView&    image         = images[index];
        const ImagePlan&    image_plan    = plan.images[index];
        const std::uint8_t* source        = nullptr;
        std::size_t         source_stride = image.row_stride;
        if (image_plan.requires_staging)
        {
            source        = reinterpret_cast<const std::uint8_t*>(staged_device_bytes +
                                                                  image_plan.staging_offset);
            source_stride = image_plan.packed_bytes / static_cast<std::size_t>(image.height);
        }
        else
        {
            source = static_cast<const std::uint8_t*>(image.data);
        }
        launch_image(source, source_stride, image.pixel_format, destination, image_plan.transform,
                     destination_bytes + index * tensor_bytes_per_image, options, cuda_stream);
    }
}

} // namespace kfcore::image
