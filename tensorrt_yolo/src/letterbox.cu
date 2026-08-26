#include "letterbox.hpp"

#include "cuda_buffer.hpp"
#include "kfcore/yolo/error.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <string>

namespace kfcore::yolo::detail
{
namespace
{

    constexpr float kPixelScale = 1.0f / 255.0f;

    template <typename Destination>
    __device__ Destination convert_destination(float value);

    template <>
    __device__ float convert_destination<float>(float value)
    {
        return value;
    }

    template <>
    __device__ __half convert_destination<__half>(float value)
    {
        return __float2half(value);
    }

    __device__ float pixel_channel(const std::uint8_t* source, std::size_t source_stride,
                                   std::int32_t source_width, std::int32_t source_height,
                                   int x, int y, int channel, PixelFormat source_format)
    {
        x = max(0, min(x, source_width - 1));
        y = max(0, min(y, source_height - 1));
        const int source_channel = source_format == PixelFormat::Rgb8 ? channel : 2 - channel;
        return static_cast<float>(source[static_cast<std::size_t>(y) * source_stride +
                                         static_cast<std::size_t>(x) * 3U +
                                         static_cast<std::size_t>(source_channel)]);
    }

    __device__ float bilinear_channel(const std::uint8_t* source, std::size_t source_stride,
                                      const LetterboxTransform& transform, int destination_x,
                                      int destination_y, int channel, PixelFormat source_format,
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
        source_x = fminf(fmaxf(source_x, 0.0f),
                         static_cast<float>(transform.source_width - 1));
        source_y = fminf(fmaxf(source_y, 0.0f),
                         static_cast<float>(transform.source_height - 1));
        const int x0 = static_cast<int>(floorf(source_x));
        const int y0 = static_cast<int>(floorf(source_y));
        const int x1 = min(x0 + 1, transform.source_width - 1);
        const int y1 = min(y0 + 1, transform.source_height - 1);
        const float x_weight = source_x - static_cast<float>(x0);
        const float y_weight = source_y - static_cast<float>(y0);

        const float top_left = pixel_channel(source, source_stride, transform.source_width,
                                             transform.source_height, x0, y0, channel,
                                             source_format);
        const float top_right = pixel_channel(source, source_stride, transform.source_width,
                                              transform.source_height, x1, y0, channel,
                                              source_format);
        const float bottom_left = pixel_channel(source, source_stride, transform.source_width,
                                                transform.source_height, x0, y1, channel,
                                                source_format);
        const float bottom_right = pixel_channel(source, source_stride, transform.source_width,
                                                 transform.source_height, x1, y1, channel,
                                                 source_format);
        const float top = top_left + (top_right - top_left) * x_weight;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_weight;
        return top + (bottom - top) * y_weight;
    }

    template <typename Destination>
    __global__ void letterbox_kernel(const std::uint8_t* source, std::size_t source_stride,
                                     PixelFormat source_format, Destination* destination,
                                     std::int32_t destination_width,
                                     std::size_t total_pixels,
                                     LetterboxTransform transform, float mean0, float mean1,
                                     float mean2, float stddev0, float stddev1, float stddev2,
                                     float border_value)
    {
        const float means[3] = { mean0, mean1, mean2 };
        const float standard_deviations[3] = { stddev0, stddev1, stddev2 };
        const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                                  threadIdx.x;
        const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
        for (std::size_t pixel_index = start; pixel_index < total_pixels; pixel_index += step)
        {
            const int x = static_cast<int>(pixel_index %
                                           static_cast<std::size_t>(destination_width));
            const int y = static_cast<int>(pixel_index /
                                           static_cast<std::size_t>(destination_width));
            for (int channel = 0; channel < 3; ++channel)
            {
                const float pixel = bilinear_channel(source, source_stride, transform, x, y,
                                                     channel, source_format, border_value);
                const float normalized =
                    (pixel * kPixelScale - means[channel]) / standard_deviations[channel];
                destination[static_cast<std::size_t>(channel) * total_pixels + pixel_index] =
                    convert_destination<Destination>(normalized);
            }
        }
    }

    [[noreturn]] void throw_invalid(const char* detail)
    {
        throw YoloError(YoloErrorCode::InvalidArgument,
                        std::string("letterbox launch stage: ") + detail);
    }

} // namespace

void launch_letterbox(const std::uint8_t* source, std::size_t source_stride,
                      PixelFormat source_format, void* destination,
                      std::int32_t destination_width, std::int32_t destination_height,
                      TensorDataType destination_type, const LetterboxTransform& transform,
                      const std::array<float, 3>& mean, const std::array<float, 3>& stddev,
                      float border_value, void* stream)
{
    if (source == nullptr || destination == nullptr)
    {
        throw_invalid("source and destination must not be null");
    }
    if (destination_width <= 0 || destination_height <= 0 || transform.source_width <= 0 ||
        transform.source_height <= 0 || !std::isfinite(transform.scale) ||
        transform.scale <= 0.0f)
    {
        throw_invalid("dimensions and transform must be valid");
    }
    if (source_format != PixelFormat::Rgb8 && source_format != PixelFormat::Bgr8)
    {
        throw_invalid("source pixel format is unsupported");
    }

    const LetterboxLaunchPlan launch_plan =
        plan_letterbox_launch(static_cast<std::size_t>(destination_width),
                              static_cast<std::size_t>(destination_height));
    const dim3 block(kLetterboxThreadsPerBlock);
    const dim3 grid(launch_plan.block_count);
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (destination_type)
    {
    case TensorDataType::Float16:
        letterbox_kernel<<<grid, block, 0, cuda_stream>>>(
            source, source_stride, source_format, static_cast<__half*>(destination),
            destination_width, launch_plan.total_pixels, transform, mean[0], mean[1], mean[2],
            stddev[0], stddev[1], stddev[2], border_value);
        break;
    case TensorDataType::Float32:
        letterbox_kernel<<<grid, block, 0, cuda_stream>>>(
            source, source_stride, source_format, static_cast<float*>(destination),
            destination_width, launch_plan.total_pixels, transform, mean[0], mean[1], mean[2],
            stddev[0], stddev[1], stddev[2], border_value);
        break;
    case TensorDataType::Int32:
        throw_invalid("destination tensor type must be Float16 or Float32");
    default:
        throw_invalid("destination tensor type is unknown");
    }
    check_cuda(cudaGetLastError(), "letterbox_kernel", "CUDA preprocessing");
}

} // namespace kfcore::yolo::detail
