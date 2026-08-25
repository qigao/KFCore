#include "letterbox.hpp"
#include "tinytest.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

using namespace kfcore::yolo;
using namespace kfcore::yolo::detail;

namespace
{

class DeviceAllocation final
{
public:
    explicit DeviceAllocation(std::size_t bytes)
    {
        check(cudaMalloc(&data_, bytes) == cudaSuccess);
    }

    ~DeviceAllocation()
    {
        if (data_ != nullptr)
        {
            (void)cudaFree(data_);
        }
    }

    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    void* get() const noexcept
    {
        return data_;
    }

private:
    void* data_ = nullptr;
};

class ExplicitStream final
{
public:
    ExplicitStream()
    {
        check(cudaStreamCreate(&stream_) == cudaSuccess);
    }

    ~ExplicitStream()
    {
        if (stream_ != nullptr)
        {
            (void)cudaStreamDestroy(stream_);
        }
    }

    ExplicitStream(const ExplicitStream&) = delete;
    ExplicitStream& operator=(const ExplicitStream&) = delete;

    cudaStream_t get() const noexcept
    {
        return stream_;
    }

private:
    cudaStream_t stream_ = nullptr;
};

template <std::size_t Size>
void check_fp32(const std::array<float, Size>& actual,
                const std::array<float, Size>& expected, float tolerance)
{
    for (std::size_t index = 0; index < Size; ++index)
    {
        check(std::fabs(actual[index] - expected[index]) < tolerance);
    }
}

} // namespace

spec("CUDA letterbox")
{
    it("writes two padded non-packed images to consecutive FP32 NCHW slices")
    {
        constexpr std::size_t kSourceStride = 8;
        constexpr std::size_t kImageElements = 3 * 2 * 3;
        const std::array<std::uint8_t, kSourceStride> rgb = {
            255, 0, 0, 0, 128, 255, 17, 19,
        };
        const std::array<std::uint8_t, kSourceStride> bgr = {
            0, 255, 0, 64, 0, 255, 23, 29,
        };
        const std::array<float, kImageElements * 2> expected = {
            0.6941177f, 0.6941177f, 1.8f, -0.2f, 0.6941177f, 0.6941177f,
            0.9882353f, 0.9882353f, -0.8f, 1.2078432f, 0.9882353f, 0.9882353f,
            0.0735294f, 0.0735294f, -0.15f, 0.35f, 0.0735294f, 0.0735294f,
            0.6941177f, 0.6941177f, -0.2f, 1.8f, 0.6941177f, 0.6941177f,
            0.9882353f, 0.9882353f, 3.2f, -0.8f, 0.9882353f, 0.9882353f,
            0.0735294f, 0.0735294f, -0.15f, -0.0245098f, 0.0735294f, 0.0735294f,
        };
        DeviceAllocation first_source(rgb.size());
        DeviceAllocation second_source(bgr.size());
        DeviceAllocation destination(expected.size() * sizeof(float));
        ExplicitStream stream;
        std::array<float, kImageElements * 2> output {};

        check(cudaMemcpyAsync(first_source.get(), rgb.data(), rgb.size(),
                              cudaMemcpyHostToDevice, stream.get()) == cudaSuccess);
        check(cudaMemcpyAsync(second_source.get(), bgr.data(), bgr.size(),
                              cudaMemcpyHostToDevice, stream.get()) == cudaSuccess);

        const LetterboxTransform transform = compute_letterbox_transform(2, 1, 2, 3);
        const std::array<float, 3> mean = { 0.1f, 0.2f, 0.3f };
        const std::array<float, 3> stddev = { 0.5f, 0.25f, 2.0f };
        launch_letterbox(static_cast<const std::uint8_t*>(first_source.get()), kSourceStride,
                         PixelFormat::Rgb8, destination.get(), 2, 3,
                         TensorDataType::Float32, transform, mean, stddev, 114.0f,
                         stream.get());
        auto* second_destination = static_cast<std::byte*>(destination.get()) +
                                   kImageElements * sizeof(float);
        launch_letterbox(static_cast<const std::uint8_t*>(second_source.get()), kSourceStride,
                         PixelFormat::Bgr8, second_destination, 2, 3,
                         TensorDataType::Float32, transform, mean, stddev, 114.0f,
                         stream.get());

        check(cudaMemcpyAsync(output.data(), destination.get(), output.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream.get()) == cudaSuccess);
        check(cudaStreamSynchronize(stream.get()) == cudaSuccess);
        check_fp32(output, expected, 1.0e-5f);
    }

    it("bilinearly samples the midpoint during non-square FP32 scaling")
    {
        constexpr std::size_t kSourceStride = 8;
        const std::array<std::uint8_t, kSourceStride> rgb = {
            0, 64, 128, 200, 192, 0, 31, 37,
        };
        const std::array<float, 18> expected = {
            0.0f, 0.3921569f, 0.7843137f, 0.0f, 0.3921569f, 0.7843137f,
            0.2509804f, 0.5019608f, 0.7529412f,
            0.2509804f, 0.5019608f, 0.7529412f,
            0.5019608f, 0.2509804f, 0.0f, 0.5019608f, 0.2509804f, 0.0f,
        };
        DeviceAllocation source(rgb.size());
        DeviceAllocation destination(expected.size() * sizeof(float));
        ExplicitStream stream;
        std::array<float, expected.size()> output {};

        check(cudaMemcpyAsync(source.get(), rgb.data(), rgb.size(), cudaMemcpyHostToDevice,
                              stream.get()) == cudaSuccess);
        launch_letterbox(static_cast<const std::uint8_t*>(source.get()), kSourceStride,
                         PixelFormat::Rgb8, destination.get(), 3, 2,
                         TensorDataType::Float32, compute_letterbox_transform(2, 1, 3, 2),
                         { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, 114.0f,
                         stream.get());
        check(cudaMemcpyAsync(output.data(), destination.get(), output.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream.get()) == cudaSuccess);
        check(cudaStreamSynchronize(stream.get()) == cudaSuccess);
        check_fp32(output, expected, 1.0e-5f);
    }

    it("writes horizontal borders and image pixels to FP16 NCHW on an explicit stream")
    {
        constexpr std::size_t kSourceStride = 5;
        const std::array<std::uint8_t, kSourceStride * 2> rgb = {
            64, 128, 255, 41, 43,
            255, 0, 128, 47, 53,
        };
        const std::array<float, 18> expected = {
            0.4470588f, 0.2509804f, 0.4470588f,
            0.4470588f, 1.0f, 0.4470588f,
            0.4470588f, 0.5019608f, 0.4470588f,
            0.4470588f, 0.0f, 0.4470588f,
            0.4470588f, 1.0f, 0.4470588f,
            0.4470588f, 0.5019608f, 0.4470588f,
        };
        DeviceAllocation source(rgb.size());
        DeviceAllocation destination(expected.size() * sizeof(__half));
        ExplicitStream stream;
        std::array<__half, expected.size()> output {};

        check(cudaMemcpyAsync(source.get(), rgb.data(), rgb.size(), cudaMemcpyHostToDevice,
                              stream.get()) == cudaSuccess);
        launch_letterbox(static_cast<const std::uint8_t*>(source.get()), kSourceStride,
                         PixelFormat::Rgb8, destination.get(), 3, 2,
                         TensorDataType::Float16, compute_letterbox_transform(1, 2, 3, 2),
                         { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, 114.0f,
                         stream.get());
        check(cudaMemcpyAsync(output.data(), destination.get(), output.size() * sizeof(__half),
                              cudaMemcpyDeviceToHost, stream.get()) == cudaSuccess);
        check(cudaStreamSynchronize(stream.get()) == cudaSuccess);
        for (std::size_t index = 0; index < output.size(); ++index)
        {
            check(std::fabs(__half2float(output[index]) - expected[index]) < 8.0e-4f);
        }
    }
}
