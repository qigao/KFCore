#include "kfcore/image_processor/cpu.hpp"
#include "kfcore/image_processor/image_processor.hpp"
#include "tinytest.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace kfcore::image;

namespace
{

constexpr std::int32_t kWidth = 640;
constexpr std::int32_t kHeight = 480;
constexpr std::size_t kFrameBytes =
    static_cast<std::size_t>(kWidth) * kHeight * 3U / 2U;
constexpr std::size_t kTensorElements =
    static_cast<std::size_t>(kWidth) * kHeight * 3U;
constexpr std::size_t kCpuSamples = 20U;
constexpr std::size_t kCudaSamples = 100U;
constexpr std::size_t kCopySamples = 1000U;

class DeviceAllocation final
{
public:
    explicit DeviceAllocation(std::size_t bytes)
    {
        if (cudaMalloc(&data_, bytes) != cudaSuccess)
        {
            throw std::runtime_error("benchmark cudaMalloc failed");
        }
    }
    ~DeviceAllocation() { (void)cudaFree(data_); }
    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;
    void* get() const noexcept { return data_; }

private:
    void* data_ = nullptr;
};

class PinnedAllocation final
{
public:
    explicit PinnedAllocation(std::size_t bytes)
    {
        if (cudaMallocHost(&data_, bytes) != cudaSuccess)
        {
            throw std::runtime_error("benchmark cudaMallocHost failed");
        }
    }
    ~PinnedAllocation() { (void)cudaFreeHost(data_); }
    PinnedAllocation(const PinnedAllocation&) = delete;
    PinnedAllocation& operator=(const PinnedAllocation&) = delete;
    void* get() const noexcept { return data_; }

private:
    void* data_ = nullptr;
};

class Stream final
{
public:
    Stream()
    {
        if (cudaStreamCreate(&stream_) != cudaSuccess)
        {
            throw std::runtime_error("benchmark cudaStreamCreate failed");
        }
    }
    ~Stream() { (void)cudaStreamDestroy(stream_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

ImageView nv12_view(const std::vector<std::uint8_t>& frame)
{
    return { frame.data(), frame.size(), kWidth, kHeight,
             static_cast<std::size_t>(kWidth), PixelFormat::Nv12,
             MemoryKind::Host };
}

} // namespace

spec("640x480 NV12 preprocessing benchmark")
{
    it("produces equivalent legacy and fused CPU tensors")
    {
        std::vector<std::uint8_t> frame(kFrameBytes, 128U);
        std::fill_n(frame.begin(), static_cast<std::size_t>(kWidth) * kHeight, 81U);
        PreprocessOptions options;
        LetterboxTransform transform;
        const BgrImage bgr = CpuImageProcessor::copy_bgr(nv12_view(frame), 4U * 1024U * 1024U);
        const std::vector<float> legacy = CpuImageProcessor::letterbox_nchw(
            bgr.view(), kWidth, kHeight, options, 4U * 1024U * 1024U,
            16U * 1024U * 1024U, &transform);
        const std::vector<float> fused = CpuImageProcessor::letterbox_nchw(
            nv12_view(frame), kWidth, kHeight, options, 4U * 1024U * 1024U,
            16U * 1024U * 1024U, &transform);
        check(legacy.size() == fused.size());
        check_eq_container(legacy, fused);
    }

    bench("bounded ownership copy and fused CPU/CUDA routes")
    {
        std::vector<std::uint8_t> frame(kFrameBytes, 128U);
        std::fill_n(frame.begin(), static_cast<std::size_t>(kWidth) * kHeight, 81U);
        std::vector<std::uint8_t> mailbox(kFrameBytes);
        const ImageView source = nv12_view(frame);
        PreprocessOptions options;
        LetterboxTransform transform;
        volatile float sink = 0.0F;

        benchmark_bytes("mailbox memcpy 460800 bytes", kCopySamples, kFrameBytes)
        {
            std::memcpy(mailbox.data(), frame.data(), frame.size());
        }

        benchmark_batch("CPU NV12 to BGR then BGR letterbox", kCpuSamples)
        {
            const BgrImage bgr = CpuImageProcessor::copy_bgr(
                source, 4U * 1024U * 1024U);
            const std::vector<float> tensor = CpuImageProcessor::letterbox_nchw(
                bgr.view(), kWidth, kHeight, options, 4U * 1024U * 1024U,
                16U * 1024U * 1024U, &transform);
            sink += tensor[0];
        }

        benchmark_batch("CPU fused NV12 letterbox", kCpuSamples)
        {
            const std::vector<float> tensor = CpuImageProcessor::letterbox_nchw(
                source, kWidth, kHeight, options, 4U * 1024U * 1024U,
                16U * 1024U * 1024U, &transform);
            sink += tensor[0];
        }

        DeviceAllocation destination(kTensorElements * sizeof(float));
        DeviceAllocation device_workspace(kFrameBytes);
        PinnedAllocation pinned_workspace(kFrameBytes);
        Stream stream;
        const TensorView tensor {
            destination.get(), kTensorElements * sizeof(float), 1, 3,
            kHeight, kWidth, TensorElementType::Float32, TensorLayout::Nchw,
            MemoryKind::CudaDevice,
        };
        const BatchPlan plan = ImageProcessor::plan(
            { source }, tensor, 4U * 1024U * 1024U, 16U * 1024U * 1024U);
        ImageProcessor::stage_host_inputs(
            { source }, plan, { pinned_workspace.get(), plan.host_staging_bytes });
        ImageProcessor::enqueue(
            { source }, tensor, plan,
            { pinned_workspace.get(), plan.host_staging_bytes },
            { device_workspace.get(), plan.device_staging_bytes }, options, stream.get());
        check(cudaStreamSynchronize(stream.get()) == cudaSuccess);

        auto shared_processor = CudaImageProcessor::create();
        (void)shared_processor->stage(source);
        benchmark_batch("CUDA one shared NV12 stage", kCudaSamples)
        {
            (void)shared_processor->stage(source);
        }
        benchmark_batch("CUDA three independent NV12 stages", kCudaSamples)
        {
            (void)shared_processor->stage(source);
            (void)shared_processor->stage(source);
            (void)shared_processor->stage(source);
        }

        benchmark_batch("CUDA fused NV12 upload and letterbox", kCudaSamples)
        {
            ImageProcessor::stage_host_inputs(
                { source }, plan, { pinned_workspace.get(), plan.host_staging_bytes });
            ImageProcessor::enqueue(
                { source }, tensor, plan,
                { pinned_workspace.get(), plan.host_staging_bytes },
                { device_workspace.get(), plan.device_staging_bytes }, options,
                stream.get());
            if (cudaStreamSynchronize(stream.get()) != cudaSuccess)
            {
                throw std::runtime_error("benchmark CUDA synchronization failed");
            }
        }
        check(sink >= 0.0F);
    }
}
