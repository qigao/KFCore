#include "letterbox.hpp"
#include "tinytest.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

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

} // namespace

spec("CUDA letterbox")
{
    it("converts RGB8 and BGR8 host images to normalized FP32 NCHW")
    {
        const std::array<std::uint8_t, 6> rgb = { 255, 0, 0, 0, 128, 255 };
        const std::array<std::uint8_t, 6> bgr = { 0, 0, 255, 255, 128, 0 };
        DeviceAllocation source(rgb.size());
        DeviceAllocation destination(6 * sizeof(float));
        std::array<float, 6> output {};
        const LetterboxTransform transform = compute_letterbox_transform(2, 1, 2, 1);
        const std::array<float, 3> mean = { 0.0f, 0.0f, 0.0f };
        const std::array<float, 3> stddev = { 1.0f, 1.0f, 1.0f };

        for (const auto& input : { std::pair { &rgb, PixelFormat::Rgb8 },
                                  std::pair { &bgr, PixelFormat::Bgr8 } })
        {
            check(cudaMemcpy(source.get(), input.first->data(), input.first->size(),
                             cudaMemcpyHostToDevice) == cudaSuccess);
            launch_letterbox(static_cast<const std::uint8_t*>(source.get()), 6,
                             input.second, destination.get(), 2, 1,
                             TensorDataType::Float32, transform, mean, stddev, 114.0f, nullptr);
            check(cudaDeviceSynchronize() == cudaSuccess);
            check(cudaMemcpy(output.data(), destination.get(), 6 * sizeof(float),
                             cudaMemcpyDeviceToHost) == cudaSuccess);
            check(std::fabs(output[0] - 1.0f) < 1.0e-6f);
            check(std::fabs(output[1]) < 1.0e-6f);
            check(std::fabs(output[2]) < 1.0e-6f);
            check(std::fabs(output[3] - (128.0f / 255.0f)) < 1.0e-6f);
            check(std::fabs(output[4]) < 1.0e-6f);
            check(std::fabs(output[5] - 1.0f) < 1.0e-6f);
        }
    }

    it("writes normalized FP16 NCHW")
    {
        const std::array<std::uint8_t, 3> rgb = { 64, 128, 255 };
        DeviceAllocation source(rgb.size());
        DeviceAllocation destination(3 * sizeof(__half));
        std::array<__half, 3> output {};
        check(cudaMemcpy(source.get(), rgb.data(), rgb.size(), cudaMemcpyHostToDevice) ==
              cudaSuccess);

        launch_letterbox(static_cast<const std::uint8_t*>(source.get()), 3,
                         PixelFormat::Rgb8, destination.get(), 1, 1,
                         TensorDataType::Float16, compute_letterbox_transform(1, 1, 1, 1),
                         { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, 114.0f, nullptr);
        check(cudaDeviceSynchronize() == cudaSuccess);
        check(cudaMemcpy(output.data(), destination.get(), 3 * sizeof(__half),
                         cudaMemcpyDeviceToHost) == cudaSuccess);
        check(std::fabs(__half2float(output[0]) - (64.0f / 255.0f)) < 5.0e-4f);
        check(std::fabs(__half2float(output[1]) - (128.0f / 255.0f)) < 5.0e-4f);
        check(std::fabs(__half2float(output[2]) - 1.0f) < 5.0e-4f);
    }
}
