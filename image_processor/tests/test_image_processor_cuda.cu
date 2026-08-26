#include "kfcore/image_processor/image_processor.hpp"
#include "tinytest.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace kfcore::image;

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

    DeviceAllocation(const DeviceAllocation&)            = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    void* get() const noexcept
    {
        return data_;
    }

private:
    void* data_ = nullptr;
};

class PinnedAllocation final
{
public:
    explicit PinnedAllocation(std::size_t bytes)
    {
        check(cudaMallocHost(&data_, bytes) == cudaSuccess);
    }

    ~PinnedAllocation()
    {
        if (data_ != nullptr)
        {
            (void)cudaFreeHost(data_);
        }
    }

    PinnedAllocation(const PinnedAllocation&)            = delete;
    PinnedAllocation& operator=(const PinnedAllocation&) = delete;

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

    ExplicitStream(const ExplicitStream&)            = delete;
    ExplicitStream& operator=(const ExplicitStream&) = delete;

    cudaStream_t get() const noexcept
    {
        return stream_;
    }

private:
    cudaStream_t stream_ = nullptr;
};

TensorView fp32_tensor(void* data, std::size_t bytes, std::int32_t batch, std::int32_t height,
                       std::int32_t width)
{
    return { data,
             bytes,
             batch,
             3,
             height,
             width,
             TensorElementType::Float32,
             TensorLayout::Nchw,
             MemoryKind::CudaDevice };
}

template <std::size_t Size>
void check_fp32(const std::array<float, Size>& actual, const std::array<float, Size>& expected,
                float tolerance = 1.0e-5f)
{
    for (std::size_t index = 0; index < Size; ++index)
    {
        check(std::fabs(actual[index] - expected[index]) < tolerance);
    }
}

} // namespace

spec("ImageProcessor CUDA contract")
{
    it("letterboxes a CUDA RGB image into normalized FP32 NCHW on an explicit stream")
    {
        constexpr std::size_t                         kSourceStride = 8;
        const std::array<std::uint8_t, kSourceStride> rgb           = {
            255, 0, 0, 0, 128, 255, 17, 19,
        };
        const std::array<float, 18> expected = {
            0.6941177f, 0.6941177f, 1.8f,   -0.2f,      0.6941177f, 0.6941177f,
            0.9882353f, 0.9882353f, -0.8f,  1.2078432f, 0.9882353f, 0.9882353f,
            0.0735294f, 0.0735294f, -0.15f, 0.35f,      0.0735294f, 0.0735294f,
        };
        DeviceAllocation                   source(rgb.size());
        DeviceAllocation                   destination(expected.size() * sizeof(float));
        ExplicitStream                     stream;
        std::array<float, expected.size()> output {};
        check(cudaMemcpyAsync(source.get(), rgb.data(), rgb.size(), cudaMemcpyHostToDevice,
                              stream.get()) == cudaSuccess);

        const std::vector<ImageView> images = {
            { source.get(), rgb.size(), 2, 1, kSourceStride, PixelFormat::Rgb8,
              MemoryKind::CudaDevice },
        };
        const TensorView tensor =
            fp32_tensor(destination.get(), expected.size() * sizeof(float), 1, 3, 2);
        const BatchPlan   plan = ImageProcessor::plan(images, tensor, 1024, 1024);
        PreprocessOptions options;
        options.mean   = { 0.1f, 0.2f, 0.3f };
        options.stddev = { 0.5f, 0.25f, 2.0f };

        ImageProcessor::enqueue(images, tensor, plan, {}, {}, options, stream.get());
        check(cudaMemcpyAsync(output.data(), destination.get(), output.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream.get()) == cudaSuccess);
        check(cudaStreamSynchronize(stream.get()) == cudaSuccess);
        check_fp32(output, expected);
    }

    it("supports BGR tensor channel order independently of source format")
    {
        const std::array<std::uint8_t, 3> rgb      = { 10, 20, 30 };
        const std::array<float, 3>        expected = {
            30.0f / 255.0f,
            20.0f / 255.0f,
            10.0f / 255.0f,
        };
        DeviceAllocation     source(rgb.size());
        DeviceAllocation     destination(expected.size() * sizeof(float));
        ExplicitStream       stream;
        std::array<float, 3> output {};
        check(cudaMemcpyAsync(source.get(), rgb.data(), rgb.size(), cudaMemcpyHostToDevice,
                              stream.get()) == cudaSuccess);

        const std::vector<ImageView> images = {
            { source.get(), rgb.size(), 1, 1, 3, PixelFormat::Rgb8, MemoryKind::CudaDevice },
        };
        const TensorView  tensor = fp32_tensor(destination.get(), sizeof(output), 1, 1, 1);
        const BatchPlan   plan   = ImageProcessor::plan(images, tensor, 1024, 1024);
        PreprocessOptions options;
        options.output_format = PixelFormat::Bgr8;

        ImageProcessor::enqueue(images, tensor, plan, {}, {}, options, stream.get());
        check(cudaMemcpyAsync(output.data(), destination.get(), sizeof(output),
                              cudaMemcpyDeviceToHost, stream.get()) == cudaSuccess);
        check(cudaStreamSynchronize(stream.get()) == cudaSuccess);
        check_fp32(output, expected);
    }

    it("bilinearly samples the midpoint during non-square FP32 scaling")
    {
        constexpr std::size_t                         kSourceStride = 8;
        const std::array<std::uint8_t, kSourceStride> rgb           = {
            0, 64, 128, 200, 192, 0, 31, 37,
        };
        const std::array<float, 18> expected = {
            0.0f,       0.3921569f, 0.7843137f, 0.0f,       0.3921569f, 0.7843137f,
            0.2509804f, 0.5019608f, 0.7529412f, 0.2509804f, 0.5019608f, 0.7529412f,
            0.5019608f, 0.2509804f, 0.0f,       0.5019608f, 0.2509804f, 0.0f,
        };
        DeviceAllocation                   source(rgb.size());
        DeviceAllocation                   destination(expected.size() * sizeof(float));
        ExplicitStream                     stream;
        std::array<float, expected.size()> output {};
        check(cudaMemcpyAsync(source.get(), rgb.data(), rgb.size(), cudaMemcpyHostToDevice,
                              stream.get()) == cudaSuccess);

        const std::vector<ImageView> images = {
            { source.get(), rgb.size(), 2, 1, kSourceStride, PixelFormat::Rgb8,
              MemoryKind::CudaDevice },
        };
        const TensorView tensor =
            fp32_tensor(destination.get(), expected.size() * sizeof(float), 1, 2, 3);
        const BatchPlan plan = ImageProcessor::plan(images, tensor, 1024, 1024);

        ImageProcessor::enqueue(images, tensor, plan, {}, {}, {}, stream.get());
        check(cudaMemcpyAsync(output.data(), destination.get(), output.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream.get()) == cudaSuccess);
        check(cudaStreamSynchronize(stream.get()) == cudaSuccess);
        check_fp32(output, expected);
    }

    it("writes horizontal borders and image pixels to FP16 NCHW")
    {
        constexpr std::size_t                             kSourceStride = 5;
        const std::array<std::uint8_t, kSourceStride * 2> rgb           = {
            64, 128, 255, 41, 43, 255, 0, 128, 47, 53,
        };
        const std::array<float, 18> expected = {
            0.4470588f, 0.2509804f, 0.4470588f, 0.4470588f, 1.0f,       0.4470588f,
            0.4470588f, 0.5019608f, 0.4470588f, 0.4470588f, 0.0f,       0.4470588f,
            0.4470588f, 1.0f,       0.4470588f, 0.4470588f, 0.5019608f, 0.4470588f,
        };
        DeviceAllocation                    source(rgb.size());
        DeviceAllocation                    destination(expected.size() * sizeof(__half));
        ExplicitStream                      stream;
        std::array<__half, expected.size()> output {};
        check(cudaMemcpyAsync(source.get(), rgb.data(), rgb.size(), cudaMemcpyHostToDevice,
                              stream.get()) == cudaSuccess);

        const std::vector<ImageView> images = {
            { source.get(), rgb.size(), 1, 2, kSourceStride, PixelFormat::Rgb8,
              MemoryKind::CudaDevice },
        };
        const TensorView tensor = {
            destination.get(),
            expected.size() * sizeof(__half),
            1,
            3,
            2,
            3,
            TensorElementType::Float16,
            TensorLayout::Nchw,
            MemoryKind::CudaDevice,
        };
        const BatchPlan plan = ImageProcessor::plan(images, tensor, 1024, 1024);

        ImageProcessor::enqueue(images, tensor, plan, {}, {}, {}, stream.get());
        check(cudaMemcpyAsync(output.data(), destination.get(), output.size() * sizeof(__half),
                              cudaMemcpyDeviceToHost, stream.get()) == cudaSuccess);
        check(cudaStreamSynchronize(stream.get()) == cudaSuccess);
        for (std::size_t index = 0; index < output.size(); ++index)
        {
            check(std::fabs(__half2float(output[index]) - expected[index]) < 8.0e-4f);
        }
    }

    it("uploads padded host input and mixes it with a CUDA input in one FP16 batch")
    {
        const std::array<std::uint8_t, 5> host_rgb   = { 64, 128, 255, 41, 43 };
        const std::array<std::uint8_t, 3> device_bgr = { 255, 128, 64 };
        const std::array<float, 6>        expected   = {
            64.0f / 255.0f, 128.0f / 255.0f, 1.0f, 64.0f / 255.0f, 128.0f / 255.0f, 1.0f,
        };
        DeviceAllocation                    device_source(device_bgr.size());
        DeviceAllocation                    device_workspace(3);
        DeviceAllocation                    destination(expected.size() * sizeof(__half));
        PinnedAllocation                    pinned_workspace(3);
        ExplicitStream                      stream;
        std::array<__half, expected.size()> output {};
        check(cudaMemcpyAsync(device_source.get(), device_bgr.data(), device_bgr.size(),
                              cudaMemcpyHostToDevice, stream.get()) == cudaSuccess);

        const std::vector<ImageView> images = {
            { host_rgb.data(), host_rgb.size(), 1, 1, host_rgb.size(), PixelFormat::Rgb8,
              MemoryKind::Host },
            { device_source.get(), device_bgr.size(), 1, 1, 3, PixelFormat::Bgr8,
              MemoryKind::CudaDevice },
        };
        const TensorView tensor = {
            destination.get(),
            expected.size() * sizeof(__half),
            2,
            3,
            1,
            1,
            TensorElementType::Float16,
            TensorLayout::Nchw,
            MemoryKind::CudaDevice,
        };
        const BatchPlan plan = ImageProcessor::plan(images, tensor, 1024, 1024);
        ImageProcessor::stage_host_inputs(images, plan,
                                          { pinned_workspace.get(), plan.host_staging_bytes });

        ImageProcessor::enqueue(
            images, tensor, plan, { pinned_workspace.get(), plan.host_staging_bytes },
            { device_workspace.get(), plan.device_staging_bytes }, {}, stream.get());
        check(cudaMemcpyAsync(output.data(), destination.get(), output.size() * sizeof(__half),
                              cudaMemcpyDeviceToHost, stream.get()) == cudaSuccess);
        check(cudaStreamSynchronize(stream.get()) == cudaSuccess);
        for (std::size_t index = 0; index < output.size(); ++index)
        {
            check(std::fabs(__half2float(output[index]) - expected[index]) < 8.0e-4f);
        }
    }
}
