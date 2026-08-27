#include "kfcore/image_processor/image_processor.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
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

    __device__ float affine_pixel_channel(const std::uint8_t* source, std::size_t source_stride,
                                          std::int32_t source_width, std::int32_t source_height,
                                          int x, int y, int output_channel,
                                          PixelFormat source_format, PixelFormat output_format,
                                          float border_value)
    {
        if (x < 0 || x >= source_width || y < 0 || y >= source_height)
        {
            return border_value;
        }
        return pixel_channel(source, source_stride, source_width, source_height, x, y,
                             output_channel, source_format, output_format);
    }

    __device__ float affine_bilinear_channel(const std::uint8_t* source, std::size_t source_stride,
                                             std::int32_t source_width, std::int32_t source_height,
                                             int destination_x, int destination_y,
                                             int output_channel, PixelFormat source_format,
                                             PixelFormat output_format, float transform0,
                                             float transform1, float transform2, float transform3,
                                             float transform4, float transform5, float border_value)
    {
        const float source_x = transform0 * static_cast<float>(destination_x) +
                               transform1 * static_cast<float>(destination_y) + transform2;
        const float source_y = transform3 * static_cast<float>(destination_x) +
                               transform4 * static_cast<float>(destination_y) + transform5;
        const int   x0       = static_cast<int>(floorf(source_x));
        const int   y0       = static_cast<int>(floorf(source_y));
        const int   x1       = x0 + 1;
        const int   y1       = y0 + 1;
        const float x_weight = source_x - static_cast<float>(x0);
        const float y_weight = source_y - static_cast<float>(y0);
        const float top_left =
            affine_pixel_channel(source, source_stride, source_width, source_height, x0, y0,
                                 output_channel, source_format, output_format, border_value);
        const float top_right =
            affine_pixel_channel(source, source_stride, source_width, source_height, x1, y0,
                                 output_channel, source_format, output_format, border_value);
        const float bottom_left =
            affine_pixel_channel(source, source_stride, source_width, source_height, x0, y1,
                                 output_channel, source_format, output_format, border_value);
        const float bottom_right =
            affine_pixel_channel(source, source_stride, source_width, source_height, x1, y1,
                                 output_channel, source_format, output_format, border_value);
        const float top    = top_left + (top_right - top_left) * x_weight;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_weight;
        return top + (bottom - top) * y_weight;
    }

    template <typename Destination>
    __global__ void affine_preprocess_kernel(
        const std::uint8_t* source, std::size_t source_stride, std::int32_t source_width,
        std::int32_t source_height, PixelFormat source_format, PixelFormat output_format,
        Destination* destination, std::int32_t destination_width, std::size_t total_pixels,
        float transform0, float transform1, float transform2, float transform3, float transform4,
        float transform5, float mean0, float mean1, float mean2, float stddev0, float stddev1,
        float stddev2, float border_value)
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
                const float pixel = affine_bilinear_channel(
                    source, source_stride, source_width, source_height, x, y, channel,
                    source_format, output_format, transform0, transform1, transform2, transform3,
                    transform4, transform5, border_value);
                const float normalized =
                    (pixel * kPixelScale - means[channel]) / standard_deviations[channel];
                destination[static_cast<std::size_t>(channel) * total_pixels + pixel_index] =
                    convert_destination<Destination>(normalized);
            }
        }
    }

    void launch_affine(const std::uint8_t* source, std::size_t source_stride,
                       std::int32_t source_width, std::int32_t source_height,
                       PixelFormat source_format, const TensorView& destination,
                       const AffineTransform& transform, const PreprocessOptions& options,
                       cudaStream_t stream)
    {
        const std::size_t total_pixels = static_cast<std::size_t>(destination.width) *
                                         static_cast<std::size_t>(destination.height);
        const std::size_t required_blocks =
            total_pixels / kThreadsPerBlock + (total_pixels % kThreadsPerBlock == 0 ? 0U : 1U);
        const dim3  block(kThreadsPerBlock);
        const dim3  grid(static_cast<std::uint32_t>(
            (std::min)(required_blocks, std::size_t { kMaximumBlocks })));
        const auto& matrix = transform.destination_to_source;
        if (destination.element_type == TensorElementType::Float32)
        {
            affine_preprocess_kernel<<<grid, block, 0, stream>>>(
                source, source_stride, source_width, source_height, source_format,
                options.output_format, static_cast<float*>(destination.data), destination.width,
                total_pixels, matrix[0], matrix[1], matrix[2], matrix[3], matrix[4], matrix[5],
                options.mean[0], options.mean[1], options.mean[2], options.stddev[0],
                options.stddev[1], options.stddev[2], options.border_value);
        }
        else if (destination.element_type == TensorElementType::Float16)
        {
            affine_preprocess_kernel<<<grid, block, 0, stream>>>(
                source, source_stride, source_width, source_height, source_format,
                options.output_format, static_cast<__half*>(destination.data), destination.width,
                total_pixels, matrix[0], matrix[1], matrix[2], matrix[3], matrix[4], matrix[5],
                options.mean[0], options.mean[1], options.mean[2], options.stddev[0],
                options.stddev[1], options.stddev[2], options.border_value);
        }
        else
        {
            throw_invalid("CUDA affine preprocessing stage: tensor element type is unsupported");
        }
        check_cuda(cudaGetLastError(), "affine_preprocess_kernel", "CUDA affine preprocessing");
    }

    template <typename Value>
    __device__ float tensor_float(Value value)
    {
        return static_cast<float>(value);
    }

    template <>
    __device__ float tensor_float<__half>(__half value)
    {
        return __half2float(value);
    }

    template <typename Value>
    __global__ void validate_finite_kernel(const Value* values, std::size_t count,
                                           int* invalid_value)
    {
        const std::size_t start =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
        for (std::size_t index = start; index < count; index += step)
        {
            if (!isfinite(tensor_float(values[index])))
            {
                atomicExch(invalid_value, 1);
            }
        }
    }

    template <typename Value>
    __device__ float decoded_tensor_pixel(const Value* rgb, std::size_t plane, int x, int y,
                                          int width, int height, int rgb_channel,
                                          TensorValueRange input_range)
    {
        if (x < 0 || x >= width || y < 0 || y >= height)
        {
            return 0.0F;
        }
        float value = tensor_float(
            rgb[static_cast<std::size_t>(rgb_channel) * plane +
                static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(x)]);
        if (input_range == TensorValueRange::SignedUnit)
        {
            value = (value + 1.0F) * 0.5F;
        }
        value = fminf(fmaxf(value, 0.0F), 1.0F) * 255.0F;
        return static_cast<float>(static_cast<std::uint8_t>(value));
    }

    template <typename Value>
    __device__ float bilinear_tensor_channel(const Value* rgb, int width, int height,
                                             float source_x, float source_y, int rgb_channel,
                                             TensorValueRange input_range)
    {
        const int   x0       = static_cast<int>(floorf(source_x));
        const int   y0       = static_cast<int>(floorf(source_y));
        const int   x1       = x0 + 1;
        const int   y1       = y0 + 1;
        const float x_weight = source_x - static_cast<float>(x0);
        const float y_weight = source_y - static_cast<float>(y0);
        const std::size_t plane = static_cast<std::size_t>(width) * height;
        const float top_left =
            decoded_tensor_pixel(rgb, plane, x0, y0, width, height, rgb_channel, input_range);
        const float top_right =
            decoded_tensor_pixel(rgb, plane, x1, y0, width, height, rgb_channel, input_range);
        const float bottom_left =
            decoded_tensor_pixel(rgb, plane, x0, y1, width, height, rgb_channel, input_range);
        const float bottom_right =
            decoded_tensor_pixel(rgb, plane, x1, y1, width, height, rgb_channel, input_range);
        const float top    = top_left + (top_right - top_left) * x_weight;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_weight;
        return top + (bottom - top) * y_weight;
    }

    __device__ float mask_pixel(const float* alpha, int width, int height, int x, int y)
    {
        if (x < 0 || x >= width || y < 0 || y >= height)
        {
            return 0.0F;
        }
        return alpha[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                     static_cast<std::size_t>(x)];
    }

    __device__ float bilinear_mask(const float* alpha, int width, int height, float source_x,
                                   float source_y)
    {
        const int   x0       = static_cast<int>(floorf(source_x));
        const int   y0       = static_cast<int>(floorf(source_y));
        const int   x1       = x0 + 1;
        const int   y1       = y0 + 1;
        const float x_weight = source_x - static_cast<float>(x0);
        const float y_weight = source_y - static_cast<float>(y0);
        const float top_left = mask_pixel(alpha, width, height, x0, y0);
        const float top_right = mask_pixel(alpha, width, height, x1, y0);
        const float bottom_left = mask_pixel(alpha, width, height, x0, y1);
        const float bottom_right = mask_pixel(alpha, width, height, x1, y1);
        const float top    = top_left + (top_right - top_left) * x_weight;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_weight;
        return top + (bottom - top) * y_weight;
    }

    template <typename Value>
    __global__ void composite_affine_kernel(
        const std::uint8_t* base, std::size_t base_stride, PixelFormat base_format,
        std::int32_t destination_width, std::int32_t destination_height, const Value* aligned_rgb,
        const float* aligned_alpha, std::int32_t aligned_width, std::int32_t aligned_height,
        float transform0, float transform1, float transform2, float transform3, float transform4,
        float transform5, TensorValueRange input_range, float strength, std::uint8_t* destination)
    {
        const std::size_t total_pixels =
            static_cast<std::size_t>(destination_width) * destination_height;
        const std::size_t start =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
        const std::size_t destination_stride = static_cast<std::size_t>(destination_width) * 3U;
        for (std::size_t pixel_index = start; pixel_index < total_pixels; pixel_index += step)
        {
            const int x =
                static_cast<int>(pixel_index % static_cast<std::size_t>(destination_width));
            const int y =
                static_cast<int>(pixel_index / static_cast<std::size_t>(destination_width));
            const float source_x = transform0 * static_cast<float>(x) +
                                   transform1 * static_cast<float>(y) + transform2;
            const float source_y = transform3 * static_cast<float>(x) +
                                   transform4 * static_cast<float>(y) + transform5;
            const float alpha = fminf(fmaxf(bilinear_mask(aligned_alpha, aligned_width,
                                                          aligned_height, source_x, source_y),
                                             0.0F),
                                      1.0F) *
                                strength;
            const std::size_t base_offset =
                static_cast<std::size_t>(y) * base_stride + static_cast<std::size_t>(x) * 3U;
            const std::size_t output_offset =
                static_cast<std::size_t>(y) * destination_stride +
                static_cast<std::size_t>(x) * 3U;
            for (int channel = 0; channel < 3; ++channel)
            {
                const int rgb_channel = base_format == PixelFormat::Bgr8 ? 2 - channel : channel;
                const float foreground = bilinear_tensor_channel(
                    aligned_rgb, aligned_width, aligned_height, source_x, source_y, rgb_channel,
                    input_range);
                const float background = static_cast<float>(base[base_offset + channel]);
                const float value = alpha * foreground + (1.0F - alpha) * background;
                destination[output_offset + channel] = static_cast<std::uint8_t>(
                    fminf(fmaxf(value, 0.0F), 255.0F));
            }
        }
    }

    dim3 kernel_grid(std::size_t element_count)
    {
        const std::size_t required_blocks =
            element_count / kThreadsPerBlock +
            (element_count % kThreadsPerBlock == 0 ? 0U : 1U);
        return dim3(static_cast<std::uint32_t>(
            (std::min)(required_blocks, std::size_t { kMaximumBlocks })));
    }

    void launch_finite_validation(const TensorView& rgb, const float* alpha,
                                  std::size_t alpha_elements, int* invalid_value,
                                  cudaStream_t stream)
    {
        const std::size_t rgb_elements = static_cast<std::size_t>(rgb.batch) * rgb.channels *
                                         rgb.height * rgb.width;
        const dim3 block(kThreadsPerBlock);
        if (rgb.element_type == TensorElementType::Float32)
        {
            validate_finite_kernel<<<kernel_grid(rgb_elements), block, 0, stream>>>(
                static_cast<const float*>(rgb.data), rgb_elements, invalid_value);
        }
        else
        {
            validate_finite_kernel<<<kernel_grid(rgb_elements), block, 0, stream>>>(
                static_cast<const __half*>(rgb.data), rgb_elements, invalid_value);
        }
        validate_finite_kernel<<<kernel_grid(alpha_elements), block, 0, stream>>>(
            alpha, alpha_elements, invalid_value);
        check_cuda(cudaGetLastError(), "validate_finite_kernel", "CUDA affine composition");
    }

    void launch_composite(const ImageView& base, const TensorView& rgb, const float* alpha,
                          const AffineTransform& transform,
                          const TensorCompositeOptions& options, void* destination,
                          cudaStream_t stream)
    {
        const std::size_t pixels = static_cast<std::size_t>(base.width) * base.height;
        const dim3 block(kThreadsPerBlock);
        const dim3 grid = kernel_grid(pixels);
        const auto& matrix = transform.destination_to_source;
        if (rgb.element_type == TensorElementType::Float32)
        {
            composite_affine_kernel<<<grid, block, 0, stream>>>(
                static_cast<const std::uint8_t*>(base.data), base.row_stride, base.pixel_format,
                base.width, base.height, static_cast<const float*>(rgb.data), alpha, rgb.width,
                rgb.height, matrix[0], matrix[1], matrix[2], matrix[3], matrix[4], matrix[5],
                options.input_range, options.strength, static_cast<std::uint8_t*>(destination));
        }
        else
        {
            composite_affine_kernel<<<grid, block, 0, stream>>>(
                static_cast<const std::uint8_t*>(base.data), base.row_stride, base.pixel_format,
                base.width, base.height, static_cast<const __half*>(rgb.data), alpha, rgb.width,
                rgb.height, matrix[0], matrix[1], matrix[2], matrix[3], matrix[4], matrix[5],
                options.input_range, options.strength, static_cast<std::uint8_t*>(destination));
        }
        check_cuda(cudaGetLastError(), "composite_affine_kernel", "CUDA affine composition");
    }

    std::size_t checked_multiply(std::size_t left, std::size_t right, const char* stage)
    {
        if (left != 0 && right > (std::numeric_limits<std::size_t>::max)() / left)
        {
            throw_invalid(std::string("CUDA affine preprocessing stage: byte count overflows at ") +
                          stage);
        }
        return left * right;
    }

    std::size_t checked_add(std::size_t left, std::size_t right, const char* stage)
    {
        if (right > (std::numeric_limits<std::size_t>::max)() - left)
        {
            throw_invalid(std::string("CUDA affine preprocessing stage: byte count overflows at ") +
                          stage);
        }
        return left + right;
    }

    class OwnedCudaAllocation final
    {
    public:
        explicit OwnedCudaAllocation(bool pinned) noexcept
            : pinned_(pinned)
        {
        }
        ~OwnedCudaAllocation() noexcept
        {
            if (data_ != nullptr)
            {
                if (pinned_)
                {
                    (void)cudaFreeHost(data_);
                }
                else
                {
                    (void)cudaFree(data_);
                }
            }
        }

        OwnedCudaAllocation(const OwnedCudaAllocation&)            = delete;
        OwnedCudaAllocation& operator=(const OwnedCudaAllocation&) = delete;

        void reserve(std::size_t bytes, std::size_t hard_limit)
        {
            if (bytes <= capacity_)
            {
                return;
            }
            if (bytes > hard_limit)
            {
                throw ImageProcessorError(ImageProcessorErrorCode::ResourceLimitExceeded,
                                          "CUDA affine preprocessing stage: buffer limit exceeded");
            }
            void*             replacement = nullptr;
            const cudaError_t result =
                pinned_ ? cudaMallocHost(&replacement, bytes) : cudaMalloc(&replacement, bytes);
            check_cuda(result, pinned_ ? "cudaMallocHost" : "cudaMalloc",
                       "CUDA affine buffer allocation");
            void* old = data_;
            data_     = replacement;
            capacity_ = bytes;
            if (old != nullptr)
            {
                check_cuda(pinned_ ? cudaFreeHost(old) : cudaFree(old),
                           pinned_ ? "cudaFreeHost" : "cudaFree", "CUDA affine buffer replacement");
            }
        }

        void* data() noexcept
        {
            return data_;
        }

        void abandon() noexcept
        {
            data_     = nullptr;
            capacity_ = 0;
        }

        void release() noexcept
        {
            if (data_ != nullptr)
            {
                if (pinned_)
                {
                    (void)cudaFreeHost(data_);
                }
                else
                {
                    (void)cudaFree(data_);
                }
                data_     = nullptr;
                capacity_ = 0;
            }
        }

    private:
        bool        pinned_   = false;
        void*       data_     = nullptr;
        std::size_t capacity_ = 0;
    };

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

    struct CudaImageProcessor::Impl final
{
    explicit Impl(CudaImageProcessorOptions configured_options)
        : options(std::move(configured_options))
        , pinned_source(true)
        , device_source(false)
        , device_tensor(false)
        , device_inference_tensor(false)
        , pinned_mask(true)
        , device_mask(false)
        , device_composite(false)
        , device_status(false)
    {
        int previous_device = 0;
        check_cuda(cudaGetDevice(&previous_device), "cudaGetDevice",
                   "CUDA affine processor creation");
        const bool restore = previous_device != options.device_id;
        if (restore)
        {
            check_cuda(cudaSetDevice(options.device_id), "cudaSetDevice",
                       "CUDA affine processor creation");
        }
        try
        {
            check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate",
                       "CUDA affine processor creation");
        }
        catch (...)
        {
            if (restore)
            {
                (void)cudaSetDevice(previous_device);
            }
            throw;
        }
        if (restore)
        {
            const cudaError_t result = cudaSetDevice(previous_device);
            if (result != cudaSuccess)
            {
                (void)cudaStreamDestroy(stream);
                stream = nullptr;
                check_cuda(result, "cudaSetDevice", "CUDA affine processor device restoration");
            }
        }
    }

    ~Impl() noexcept
    {
        int previous_device = 0;
        if (cudaGetDevice(&previous_device) != cudaSuccess ||
            cudaSetDevice(options.device_id) != cudaSuccess)
        {
            pinned_source.abandon();
            device_source.abandon();
            device_tensor.abandon();
            device_inference_tensor.abandon();
            pinned_mask.abandon();
            device_mask.abandon();
            device_composite.abandon();
            device_status.abandon();
            stream = nullptr;
            return;
        }
        if (stream != nullptr)
        {
            (void)cudaStreamSynchronize(stream);
            (void)cudaStreamDestroy(stream);
            stream = nullptr;
        }
        device_tensor.release();
        device_inference_tensor.release();
        device_composite.release();
        device_mask.release();
        device_status.release();
        device_source.release();
        pinned_mask.release();
        pinned_source.release();
        if (previous_device != options.device_id)
        {
            (void)cudaSetDevice(previous_device);
        }
    }

    CudaImageProcessorOptions options;
    OwnedCudaAllocation       pinned_source;
    OwnedCudaAllocation       device_source;
    OwnedCudaAllocation       device_tensor;
    OwnedCudaAllocation       device_inference_tensor;
    OwnedCudaAllocation       pinned_mask;
    OwnedCudaAllocation       device_mask;
    OwnedCudaAllocation       device_composite;
    OwnedCudaAllocation       device_status;
    cudaStream_t              stream = nullptr;
};

CudaImageProcessor::CudaImageProcessor(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

CudaImageProcessor::~CudaImageProcessor() = default;

std::unique_ptr<CudaImageProcessor>
CudaImageProcessor::create(const CudaImageProcessorOptions& options)
{
    if (options.device_id < 0)
    {
        throw_invalid("CUDA affine processor creation stage: device id must be non-negative");
    }
    if (options.max_source_bytes == 0 || options.max_tensor_bytes == 0)
    {
        throw ImageProcessorError(
            ImageProcessorErrorCode::ResourceLimitExceeded,
            "CUDA affine processor creation stage: byte limits must be positive");
    }
    try
    {
        auto impl = std::make_unique<Impl>(options);
        return std::unique_ptr<CudaImageProcessor>(new CudaImageProcessor(std::move(impl)));
    }
    catch (const std::bad_alloc&)
    {
        throw ImageProcessorError(ImageProcessorErrorCode::ResourceLimitExceeded,
                                  "CUDA affine processor creation stage: allocation failed");
    }
}

ImageView CudaImageProcessor::stage(const ImageView& source)
{
    if (!impl_)
    {
        throw_invalid("CUDA image staging stage: processor state is unavailable");
    }
    validate_pixel_format(source.pixel_format, "staging source");
    if (source.memory_kind != MemoryKind::Host && source.memory_kind != MemoryKind::CudaDevice)
    {
        throw_invalid("CUDA image staging stage: source memory kind is unsupported");
    }
    if (source.data == nullptr || source.width <= 0 || source.height <= 0)
    {
        throw_invalid("CUDA image staging stage: pointer and dimensions must be valid");
    }
    const std::size_t row_bytes =
        checked_multiply(static_cast<std::size_t>(source.width), 3U, "staging source row");
    if (source.row_stride < row_bytes)
    {
        throw_invalid("CUDA image staging stage: source row stride is too small");
    }
    const std::size_t source_span =
        checked_add(checked_multiply(static_cast<std::size_t>(source.height - 1), source.row_stride,
                                     "staging source span"),
                    row_bytes, "staging source span");
    if (source_span > source.byte_size)
    {
        throw_invalid("CUDA image staging stage: source capacity is too small");
    }
    const std::size_t packed_bytes = checked_multiply(
        row_bytes, static_cast<std::size_t>(source.height), "staging packed source");
    if (source_span > impl_->options.max_source_bytes ||
        packed_bytes > impl_->options.max_source_bytes)
    {
        throw ImageProcessorError(ImageProcessorErrorCode::ResourceLimitExceeded,
                                  "CUDA image staging stage: source limit exceeded");
    }

    int previous_device = 0;
    check_cuda(cudaGetDevice(&previous_device), "cudaGetDevice", "CUDA image staging");
    const bool restore = previous_device != impl_->options.device_id;
    if (restore)
    {
        check_cuda(cudaSetDevice(impl_->options.device_id), "cudaSetDevice", "CUDA image staging");
    }
    bool work_pending = false;
    try
    {
        if (source.memory_kind == MemoryKind::CudaDevice)
        {
            validate_device_pointer(source.data, impl_->options.device_id,
                                    "staging source validation");
        }
        impl_->device_source.reserve(packed_bytes, impl_->options.max_source_bytes);
        if (source.data != impl_->device_source.data() || source.row_stride != row_bytes)
        {
            if (source.memory_kind == MemoryKind::Host)
            {
                impl_->pinned_source.reserve(packed_bytes, impl_->options.max_source_bytes);
                auto*       packed = static_cast<std::byte*>(impl_->pinned_source.data());
                const auto* input  = static_cast<const std::byte*>(source.data);
                for (std::int32_t row = 0; row < source.height; ++row)
                {
                    std::memcpy(packed + static_cast<std::size_t>(row) * row_bytes,
                                input + static_cast<std::size_t>(row) * source.row_stride,
                                row_bytes);
                }
                check_cuda(cudaMemcpyAsync(impl_->device_source.data(), impl_->pinned_source.data(),
                                           packed_bytes, cudaMemcpyHostToDevice, impl_->stream),
                           "cudaMemcpyAsync", "CUDA image staging upload");
            }
            else
            {
                check_cuda(cudaMemcpy2DAsync(impl_->device_source.data(), row_bytes, source.data,
                                             source.row_stride, row_bytes,
                                             static_cast<std::size_t>(source.height),
                                             cudaMemcpyDeviceToDevice, impl_->stream),
                           "cudaMemcpy2DAsync", "CUDA image staging copy");
            }
            work_pending = true;
            check_cuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize",
                       "CUDA image staging completion");
            work_pending = false;
        }
        if (restore)
        {
            check_cuda(cudaSetDevice(previous_device), "cudaSetDevice",
                       "CUDA image staging device restoration");
        }
        return { impl_->device_source.data(),
                 packed_bytes,
                 source.width,
                 source.height,
                 row_bytes,
                 source.pixel_format,
                 MemoryKind::CudaDevice };
    }
    catch (...)
    {
        if (work_pending)
        {
            (void)cudaStreamSynchronize(impl_->stream);
        }
        if (restore)
        {
            (void)cudaSetDevice(previous_device);
        }
        throw;
    }
}

TensorView CudaImageProcessor::process_affine(const ImageView&         source,
                                              std::int32_t             destination_width,
                                              std::int32_t             destination_height,
                                              const AffineTransform&   transform,
                                              const PreprocessOptions& options,
                                              TensorElementType        element_type)
{
    if (!impl_)
    {
        throw_invalid("CUDA affine preprocessing stage: processor state is unavailable");
    }
    validate_options(options);
    validate_pixel_format(source.pixel_format, "affine source");
    if (source.memory_kind != MemoryKind::Host && source.memory_kind != MemoryKind::CudaDevice)
    {
        throw_invalid("CUDA affine preprocessing stage: source memory kind is unsupported");
    }
    if (source.data == nullptr || source.width <= 0 || source.height <= 0 ||
        destination_width <= 0 || destination_height <= 0)
    {
        throw_invalid("CUDA affine preprocessing stage: pointers and dimensions must be valid");
    }
    for (float coefficient : transform.destination_to_source)
    {
        if (!std::isfinite(coefficient))
        {
            throw_invalid("CUDA affine preprocessing stage: transform must be finite");
        }
    }

    const std::size_t source_row_bytes =
        checked_multiply(static_cast<std::size_t>(source.width), 3U, "source row");
    if (source.row_stride < source_row_bytes)
    {
        throw_invalid("CUDA affine preprocessing stage: source row stride is too small");
    }
    const std::size_t source_span =
        checked_add(checked_multiply(static_cast<std::size_t>(source.height - 1), source.row_stride,
                                     "source span"),
                    source_row_bytes, "source span");
    if (source_span > source.byte_size)
    {
        throw_invalid("CUDA affine preprocessing stage: source capacity is too small");
    }
    if (source_span > impl_->options.max_source_bytes)
    {
        throw ImageProcessorError(ImageProcessorErrorCode::ResourceLimitExceeded,
                                  "CUDA affine preprocessing stage: source limit exceeded");
    }
    const std::size_t element_bytes =
        element_type == TensorElementType::Float16 ? sizeof(__half) : sizeof(float);
    if (element_type != TensorElementType::Float16 && element_type != TensorElementType::Float32)
    {
        throw_invalid("CUDA affine preprocessing stage: tensor element type is unsupported");
    }
    const std::size_t total_pixels =
        checked_multiply(static_cast<std::size_t>(destination_width),
                         static_cast<std::size_t>(destination_height), "tensor pixels");
    const std::size_t tensor_bytes = checked_multiply(
        checked_multiply(total_pixels, 3U, "tensor channels"), element_bytes, "tensor bytes");
    if (tensor_bytes > impl_->options.max_tensor_bytes)
    {
        throw ImageProcessorError(ImageProcessorErrorCode::ResourceLimitExceeded,
                                  "CUDA affine preprocessing stage: tensor limit exceeded");
    }

    int previous_device = 0;
    check_cuda(cudaGetDevice(&previous_device), "cudaGetDevice", "CUDA affine preprocessing");
    const bool restore = previous_device != impl_->options.device_id;
    if (restore)
    {
        check_cuda(cudaSetDevice(impl_->options.device_id), "cudaSetDevice",
                   "CUDA affine preprocessing");
    }
    bool work_pending = false;
    try
    {
        const std::uint8_t* device_pixels = static_cast<const std::uint8_t*>(source.data);
        std::size_t         device_stride = source.row_stride;
        if (source.memory_kind == MemoryKind::CudaDevice)
        {
            validate_device_pointer(source.data, impl_->options.device_id,
                                    "affine source validation");
        }
        else
        {
            const std::size_t packed_bytes = checked_multiply(
                source_row_bytes, static_cast<std::size_t>(source.height), "packed source");
            impl_->pinned_source.reserve(packed_bytes, impl_->options.max_source_bytes);
            impl_->device_source.reserve(packed_bytes, impl_->options.max_source_bytes);
            auto*       packed = static_cast<std::byte*>(impl_->pinned_source.data());
            const auto* input  = static_cast<const std::byte*>(source.data);
            for (std::int32_t row = 0; row < source.height; ++row)
            {
                std::memcpy(packed + static_cast<std::size_t>(row) * source_row_bytes,
                            input + static_cast<std::size_t>(row) * source.row_stride,
                            source_row_bytes);
            }
            check_cuda(cudaMemcpyAsync(impl_->device_source.data(), impl_->pinned_source.data(),
                                       packed_bytes, cudaMemcpyHostToDevice, impl_->stream),
                       "cudaMemcpyAsync", "CUDA affine source upload");
            work_pending  = true;
            device_pixels = static_cast<const std::uint8_t*>(impl_->device_source.data());
            device_stride = source_row_bytes;
        }
        impl_->device_tensor.reserve(tensor_bytes, impl_->options.max_tensor_bytes);
        TensorView result {
            impl_->device_tensor.data(),
            tensor_bytes,
            1,
            3,
            destination_height,
            destination_width,
            element_type,
            TensorLayout::Nchw,
            MemoryKind::CudaDevice,
        };
        launch_affine(device_pixels, device_stride, source.width, source.height,
                      source.pixel_format, result, transform, options, impl_->stream);
        work_pending = true;
        check_cuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize",
                   "CUDA affine preprocessing completion");
        work_pending = false;
        if (restore)
        {
            check_cuda(cudaSetDevice(previous_device), "cudaSetDevice",
                       "CUDA affine preprocessing device restoration");
        }
        return result;
    }
    catch (...)
    {
        if (work_pending)
        {
            (void)cudaStreamSynchronize(impl_->stream);
        }
        if (restore)
        {
            (void)cudaSetDevice(previous_device);
        }
        throw;
    }
}

TensorView CudaImageProcessor::acquire_tensor(std::int32_t batch, std::int32_t channels,
                                              std::int32_t height, std::int32_t width,
                                              TensorElementType element_type)
{
    if (!impl_)
    {
        throw_invalid("CUDA inference tensor acquisition stage: processor state is unavailable");
    }
    if (batch <= 0 || channels <= 0 || height <= 0 || width <= 0)
    {
        throw_invalid("CUDA inference tensor acquisition stage: dimensions must be positive");
    }
    if (element_type != TensorElementType::Float16 &&
        element_type != TensorElementType::Float32)
    {
        throw_invalid(
            "CUDA inference tensor acquisition stage: tensor element type is unsupported");
    }
    const std::size_t element_bytes =
        element_type == TensorElementType::Float16 ? sizeof(__half) : sizeof(float);
    std::size_t elements = checked_multiply(static_cast<std::size_t>(batch),
                                            static_cast<std::size_t>(channels),
                                            "inference tensor batch and channels");
    elements = checked_multiply(elements, static_cast<std::size_t>(height),
                                "inference tensor height");
    elements = checked_multiply(elements, static_cast<std::size_t>(width),
                                "inference tensor width");
    const std::size_t bytes =
        checked_multiply(elements, element_bytes, "inference tensor bytes");
    if (bytes > impl_->options.max_tensor_bytes)
    {
        throw ImageProcessorError(
            ImageProcessorErrorCode::ResourceLimitExceeded,
            "CUDA inference tensor acquisition stage: tensor limit exceeded");
    }

    int previous_device = 0;
    check_cuda(cudaGetDevice(&previous_device), "cudaGetDevice",
               "CUDA inference tensor acquisition");
    const bool restore = previous_device != impl_->options.device_id;
    if (restore)
    {
        check_cuda(cudaSetDevice(impl_->options.device_id), "cudaSetDevice",
                   "CUDA inference tensor acquisition");
    }
    try
    {
        impl_->device_inference_tensor.reserve(bytes, impl_->options.max_tensor_bytes);
        TensorView result { impl_->device_inference_tensor.data(), bytes, batch, channels, height,
                            width, element_type, TensorLayout::Nchw,
                            MemoryKind::CudaDevice };
        if (restore)
        {
            check_cuda(cudaSetDevice(previous_device), "cudaSetDevice",
                       "CUDA inference tensor acquisition device restoration");
        }
        return result;
    }
    catch (...)
    {
        if (restore)
        {
            (void)cudaSetDevice(previous_device);
        }
        throw;
    }
}

ImageView CudaImageProcessor::composite_affine(const ImageView& base,
                                               const TensorView& aligned_rgb,
                                               const TensorView& aligned_alpha,
                                               const AffineTransform& transform,
                                               const TensorCompositeOptions& options)
{
    if (!impl_)
    {
        throw_invalid("CUDA affine composition stage: processor state is unavailable");
    }
    validate_pixel_format(base.pixel_format, "composition base");
    if (base.memory_kind != MemoryKind::CudaDevice || base.data == nullptr || base.width <= 0 ||
        base.height <= 0)
    {
        throw_invalid("CUDA affine composition stage: base must be a valid CUDA image");
    }
    const std::size_t base_row_bytes =
        checked_multiply(static_cast<std::size_t>(base.width), 3U, "composition base row");
    if (base.row_stride < base_row_bytes)
    {
        throw_invalid("CUDA affine composition stage: base row stride is too small");
    }
    const std::size_t base_span = checked_add(
        checked_multiply(static_cast<std::size_t>(base.height - 1), base.row_stride,
                         "composition base span"),
        base_row_bytes, "composition base span");
    const std::size_t packed_bytes = checked_multiply(
        base_row_bytes, static_cast<std::size_t>(base.height), "composition output bytes");
    if (base_span > base.byte_size)
    {
        throw_invalid("CUDA affine composition stage: base capacity is too small");
    }
    if (base_span > impl_->options.max_source_bytes ||
        packed_bytes > impl_->options.max_source_bytes)
    {
        throw ImageProcessorError(ImageProcessorErrorCode::ResourceLimitExceeded,
                                  "CUDA affine composition stage: image limit exceeded");
    }
    if (aligned_rgb.data == nullptr || aligned_rgb.batch != 1 || aligned_rgb.channels != 3 ||
        aligned_rgb.width <= 0 || aligned_rgb.height <= 0 ||
        aligned_rgb.layout != TensorLayout::Nchw ||
        aligned_rgb.memory_kind != MemoryKind::CudaDevice ||
        (aligned_rgb.element_type != TensorElementType::Float16 &&
         aligned_rgb.element_type != TensorElementType::Float32))
    {
        throw_invalid(
            "CUDA affine composition stage: RGB input must be a CUDA NCHW float tensor [1,3,H,W]");
    }
    if (aligned_alpha.data == nullptr || aligned_alpha.batch != 1 ||
        aligned_alpha.channels != 1 || aligned_alpha.width != aligned_rgb.width ||
        aligned_alpha.height != aligned_rgb.height ||
        aligned_alpha.layout != TensorLayout::Nchw ||
        aligned_alpha.element_type != TensorElementType::Float32 ||
        (aligned_alpha.memory_kind != MemoryKind::Host &&
         aligned_alpha.memory_kind != MemoryKind::CudaDevice))
    {
        throw_invalid("CUDA affine composition stage: alpha input must be an FP32 NCHW tensor "
                      "[1,1,H,W] matching the RGB extent");
    }
    if (options.input_range != TensorValueRange::Unit &&
        options.input_range != TensorValueRange::SignedUnit)
    {
        throw_invalid("CUDA affine composition stage: tensor value range is unsupported");
    }
    if (!std::isfinite(options.strength) || options.strength < 0.0F ||
        options.strength > 1.0F)
    {
        throw_invalid("CUDA affine composition stage: strength must be within [0,1]");
    }
    for (float coefficient : transform.destination_to_source)
    {
        if (!std::isfinite(coefficient))
        {
            throw_invalid("CUDA affine composition stage: transform must be finite");
        }
    }

    const std::size_t aligned_pixels = checked_multiply(
        static_cast<std::size_t>(aligned_rgb.width),
        static_cast<std::size_t>(aligned_rgb.height), "composition aligned pixels");
    const std::size_t rgb_element_bytes =
        aligned_rgb.element_type == TensorElementType::Float16 ? sizeof(__half) : sizeof(float);
    const std::size_t rgb_bytes = checked_multiply(
        checked_multiply(aligned_pixels, 3U, "composition RGB channels"), rgb_element_bytes,
        "composition RGB bytes");
    const std::size_t alpha_bytes =
        checked_multiply(aligned_pixels, sizeof(float), "composition alpha bytes");
    if (aligned_rgb.byte_size < rgb_bytes || aligned_alpha.byte_size < alpha_bytes)
    {
        throw_invalid("CUDA affine composition stage: tensor capacity is too small");
    }
    if (rgb_bytes > impl_->options.max_tensor_bytes ||
        alpha_bytes > impl_->options.max_tensor_bytes)
    {
        throw ImageProcessorError(ImageProcessorErrorCode::ResourceLimitExceeded,
                                  "CUDA affine composition stage: tensor limit exceeded");
    }

    int previous_device = 0;
    check_cuda(cudaGetDevice(&previous_device), "cudaGetDevice", "CUDA affine composition");
    const bool restore = previous_device != impl_->options.device_id;
    if (restore)
    {
        check_cuda(cudaSetDevice(impl_->options.device_id), "cudaSetDevice",
                   "CUDA affine composition");
    }
    bool work_pending = false;
    try
    {
        validate_device_pointer(base.data, impl_->options.device_id,
                                "composition base validation");
        validate_device_pointer(aligned_rgb.data, impl_->options.device_id,
                                "composition RGB tensor validation");
        impl_->device_composite.reserve(packed_bytes, impl_->options.max_source_bytes);
        const auto overlaps_output = [&](const void* data, std::size_t bytes)
        {
            const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(data);
            if (bytes > (std::numeric_limits<std::uintptr_t>::max)() - begin)
            {
                throw_invalid("CUDA affine composition stage: address range overflows");
            }
            const std::uintptr_t end = begin + bytes;
            const std::uintptr_t output_begin =
                reinterpret_cast<std::uintptr_t>(impl_->device_composite.data());
            if (packed_bytes > (std::numeric_limits<std::uintptr_t>::max)() - output_begin)
            {
                throw_invalid("CUDA affine composition stage: output address range overflows");
            }
            const std::uintptr_t output_end = output_begin + packed_bytes;
            return begin < output_end && output_begin < end;
        };
        const bool base_overlaps_output = overlaps_output(base.data, base_span);
        if (base_overlaps_output &&
            (base.data != impl_->device_composite.data() || base.row_stride != base_row_bytes))
        {
            throw_invalid("CUDA affine composition stage: base partially overlaps output storage");
        }
        if (overlaps_output(aligned_rgb.data, rgb_bytes))
        {
            throw_invalid("CUDA affine composition stage: RGB input overlaps output storage");
        }

        const float* device_alpha = static_cast<const float*>(aligned_alpha.data);
        if (aligned_alpha.memory_kind == MemoryKind::Host)
        {
            impl_->pinned_mask.reserve(alpha_bytes, impl_->options.max_tensor_bytes);
            impl_->device_mask.reserve(alpha_bytes, impl_->options.max_tensor_bytes);
            std::memcpy(impl_->pinned_mask.data(), aligned_alpha.data, alpha_bytes);
            check_cuda(cudaMemcpyAsync(impl_->device_mask.data(), impl_->pinned_mask.data(),
                                       alpha_bytes, cudaMemcpyHostToDevice, impl_->stream),
                       "cudaMemcpyAsync", "CUDA affine alpha upload");
            device_alpha = static_cast<const float*>(impl_->device_mask.data());
            work_pending = true;
        }
        else
        {
            validate_device_pointer(aligned_alpha.data, impl_->options.device_id,
                                    "composition alpha tensor validation");
            if (overlaps_output(aligned_alpha.data, alpha_bytes))
            {
                throw_invalid("CUDA affine composition stage: alpha input overlaps output storage");
            }
        }

        impl_->device_status.reserve(sizeof(int), sizeof(int));
        check_cuda(cudaMemsetAsync(impl_->device_status.data(), 0, sizeof(int), impl_->stream),
                   "cudaMemsetAsync", "CUDA affine finite validation");
        launch_finite_validation(aligned_rgb, device_alpha, aligned_pixels,
                                 static_cast<int*>(impl_->device_status.data()), impl_->stream);
        launch_composite(base, aligned_rgb, device_alpha, transform, options,
                         impl_->device_composite.data(), impl_->stream);
        int invalid_value = 0;
        check_cuda(cudaMemcpyAsync(&invalid_value, impl_->device_status.data(), sizeof(int),
                                   cudaMemcpyDeviceToHost, impl_->stream),
                   "cudaMemcpyAsync", "CUDA affine finite validation result");
        work_pending = true;
        check_cuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize",
                   "CUDA affine composition completion");
        work_pending = false;
        if (invalid_value != 0)
        {
            throw_invalid("CUDA affine composition stage: tensor values must be finite");
        }
        if (restore)
        {
            check_cuda(cudaSetDevice(previous_device), "cudaSetDevice",
                       "CUDA affine composition device restoration");
        }
        return { impl_->device_composite.data(), packed_bytes, base.width, base.height,
                 base_row_bytes, base.pixel_format, MemoryKind::CudaDevice };
    }
    catch (...)
    {
        if (work_pending)
        {
            (void)cudaStreamSynchronize(impl_->stream);
        }
        if (restore)
        {
            (void)cudaSetDevice(previous_device);
        }
        throw;
    }
}

void CudaImageProcessor::download_bgr(const ImageView& source, MutableBufferView destination)
{
    if (!impl_)
    {
        throw_invalid("CUDA BGR download stage: processor state is unavailable");
    }
    if (source.data == nullptr || source.width <= 0 || source.height <= 0 ||
        source.pixel_format != PixelFormat::Bgr8 ||
        source.memory_kind != MemoryKind::CudaDevice)
    {
        throw_invalid("CUDA BGR download stage: source must be a valid CUDA BGR8 image");
    }
    const std::size_t row_bytes =
        checked_multiply(static_cast<std::size_t>(source.width), 3U, "BGR download row");
    if (source.row_stride < row_bytes)
    {
        throw_invalid("CUDA BGR download stage: source row stride is too small");
    }
    const std::size_t source_span = checked_add(
        checked_multiply(static_cast<std::size_t>(source.height - 1), source.row_stride,
                         "BGR download source span"),
        row_bytes, "BGR download source span");
    const std::size_t packed_bytes = checked_multiply(
        row_bytes, static_cast<std::size_t>(source.height), "BGR download bytes");
    if (source_span > source.byte_size)
    {
        throw_invalid("CUDA BGR download stage: source capacity is too small");
    }
    if (destination.data == nullptr || destination.byte_size < packed_bytes)
    {
        throw_invalid("CUDA BGR download stage: destination capacity is too small");
    }

    int previous_device = 0;
    check_cuda(cudaGetDevice(&previous_device), "cudaGetDevice", "CUDA BGR download");
    const bool restore = previous_device != impl_->options.device_id;
    if (restore)
    {
        check_cuda(cudaSetDevice(impl_->options.device_id), "cudaSetDevice", "CUDA BGR download");
    }
    try
    {
        validate_device_pointer(source.data, impl_->options.device_id,
                                "BGR download source validation");
        check_cuda(cudaMemcpy2D(destination.data, row_bytes, source.data, source.row_stride,
                                row_bytes, static_cast<std::size_t>(source.height),
                                cudaMemcpyDeviceToHost),
                   "cudaMemcpy2D", "CUDA BGR download");
        if (restore)
        {
            check_cuda(cudaSetDevice(previous_device), "cudaSetDevice",
                       "CUDA BGR download device restoration");
        }
    }
    catch (...)
    {
        if (restore)
        {
            (void)cudaSetDevice(previous_device);
        }
        throw;
    }
}

} // namespace kfcore::image
