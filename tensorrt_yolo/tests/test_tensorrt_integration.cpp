#include "kfcore/yolo/tensorrt.hpp"
#include "tinytest.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

using namespace kfcore::yolo;

namespace
{

std::filesystem::path required_engine_path(const char* variable)
{
    const char* value = std::getenv(variable);
    if (value == nullptr || *value == '\0')
    {
        throw YoloError(YoloErrorCode::InvalidArgument,
                        std::string("integration test setup stage: ") + variable +
                            " must name a trusted TensorRT engine");
    }
    return std::filesystem::path(value);
}

std::filesystem::path test_engine_path()
{
    return required_engine_path("KFCORE_TENSORRT_TEST_ENGINE");
}

class OwnedRgbImage final
{
public:
    OwnedRgbImage(std::int32_t width, std::int32_t height)
        : width_(width)
        , height_(height)
        , pixels_(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U)
    {
        for (std::size_t index = 0; index < pixels_.size(); ++index)
        {
            pixels_[index] = static_cast<std::uint8_t>((index * 37U + 11U) % 251U);
        }
    }

    ImageView view() const noexcept
    {
        return { pixels_.data(), width_, height_, static_cast<std::size_t>(width_) * 3U,
                 PixelFormat::Rgb8, MemoryKind::Host };
    }

    const std::vector<std::uint8_t>& pixels() const noexcept
    {
        return pixels_;
    }

private:
    std::int32_t             width_;
    std::int32_t             height_;
    std::vector<std::uint8_t> pixels_;
};

[[noreturn]] void throw_cuda(cudaError_t error, const char* operation)
{
    throw YoloError(YoloErrorCode::CudaFailure,
                    std::string("integration test CUDA stage: ") + operation + " failed: " +
                        cudaGetErrorString(error));
}

class DeviceRgbImage final
{
public:
    explicit DeviceRgbImage(const OwnedRgbImage& source)
        : width_(source.view().width)
        , height_(source.view().height)
        , bytes_(source.pixels().size())
    {
        cudaError_t result = cudaMalloc(&data_, bytes_);
        if (result != cudaSuccess)
        {
            throw_cuda(result, "cudaMalloc");
        }
        result = cudaMemcpy(data_, source.pixels().data(), bytes_, cudaMemcpyHostToDevice);
        if (result != cudaSuccess)
        {
            (void)cudaFree(data_);
            data_ = nullptr;
            throw_cuda(result, "cudaMemcpy");
        }
    }

    ~DeviceRgbImage()
    {
        if (data_ != nullptr)
        {
            (void)cudaFree(data_);
        }
    }

    DeviceRgbImage(const DeviceRgbImage&) = delete;
    DeviceRgbImage& operator=(const DeviceRgbImage&) = delete;

    ImageView view() const noexcept
    {
        return { data_, width_, height_, static_cast<std::size_t>(width_) * 3U,
                 PixelFormat::Rgb8, MemoryKind::CudaDevice };
    }

private:
    void*        data_ = nullptr;
    std::int32_t width_;
    std::int32_t height_;
    std::size_t  bytes_;
};

void check_same_frame(const DetectionFrame& actual, const DetectionFrame& expected)
{
    check(actual.image_width == expected.image_width);
    check(actual.image_height == expected.image_height);
    check(actual.detections.size() == expected.detections.size());
    if (actual.detections.size() != expected.detections.size())
    {
        return;
    }
    for (std::size_t index = 0; index < actual.detections.size(); ++index)
    {
        const Detection& left = actual.detections[index];
        const Detection& right = expected.detections[index];
        check(left.class_id == right.class_id);
        check(std::fabs(left.score - right.score) < 1.0e-5f);
        check(std::fabs(left.box.left - right.box.left) < 1.0e-3f);
        check(std::fabs(left.box.top - right.box.top) < 1.0e-3f);
        check(std::fabs(left.box.right - right.box.right) < 1.0e-3f);
        check(std::fabs(left.box.bottom - right.box.bottom) < 1.0e-3f);
    }
}

} // namespace

spec("TensorRT YOLO integration")
{
#if defined(KFCORE_TENSORRT_TEST_FP16)
    it("supports an FP16 input engine when explicitly supplied")
    {
        auto engine = Engine::load(required_engine_path("KFCORE_TENSORRT_TEST_ENGINE_FP16"),
                                   EngineOptions {});
        auto detector = engine->create_detector(DetectorOptions {});
        OwnedRgbImage image(640, 384);
        const DetectionFrame result = detector->detect(image.view());
        check(result.image_width == 640);
        check(result.image_height == 384);
    }
#else
    it("rejects a missing engine file with a typed error")
    {
        EngineOptions options;
        check_throws_as(Engine::load(std::filesystem::path("Z:/kfcore/missing.engine"), options),
                        YoloError);
    }

    it("requires an explicit trusted engine path")
    {
        const char* value = std::getenv("KFCORE_TENSORRT_TEST_ENGINE");
        if (value == nullptr || *value == '\0')
        {
            check_throws_as(test_engine_path(), YoloError);
        }
        else
        {
            check(!test_engine_path().empty());
        }
    }

    it("keeps one result per non-square batch image")
    {
        auto engine = Engine::load(test_engine_path(), EngineOptions {});
        auto detector = engine->create_detector(DetectorOptions {});
        OwnedRgbImage wide(960, 320);
        OwnedRgbImage tall(320, 960);

        const auto results = detector->detect_batch({ wide.view(), tall.view() });

        check(results.size() == std::size_t { 2 });
        if (results.size() != std::size_t { 2 })
        {
            return;
        }
        check(results[0].image_width == 960);
        check(results[0].image_height == 320);
        check(results[1].image_width == 320);
        check(results[1].image_height == 960);
        for (const DetectionFrame& result : results)
        {
            for (const Detection& detection : result.detections)
            {
                check(detection.box.left >= 0.0f);
                check(detection.box.top >= 0.0f);
                check(detection.box.right <= static_cast<float>(result.image_width));
                check(detection.box.bottom <= static_cast<float>(result.image_height));
            }
        }
    }

    it("rejects empty and over-limit batches after loading the trusted engine")
    {
        auto engine = Engine::load(test_engine_path(), EngineOptions {});
        auto detector = engine->create_detector(DetectorOptions {});
        check_throws_as(detector->detect_batch({}), YoloError);

        OwnedRgbImage pixel(1, 1);
        std::vector<ImageView> excessive(17, pixel.view());
        check_throws_as(detector->detect_batch(excessive), YoloError);
    }

    it("produces equivalent results for Host and CUDA-device RGB8 inputs")
    {
        auto engine = Engine::load(test_engine_path(), EngineOptions {});
        auto detector = engine->create_detector(DetectorOptions {});
        OwnedRgbImage host(640, 384);
        DeviceRgbImage device(host);

        const DetectionFrame host_result = detector->detect(host.view());
        const DetectionFrame device_result = detector->detect(device.view());
        check_same_frame(device_result, host_result);
    }

    it("allows two detectors to share one immutable engine")
    {
        auto engine = Engine::load(test_engine_path(), EngineOptions {});
        auto first = engine->create_detector(DetectorOptions {});
        auto second = engine->create_detector(DetectorOptions {});
        OwnedRgbImage image(640, 384);

        check_same_frame(second->detect(image.view()), first->detect(image.view()));
    }

    it("rejects a host pointer labeled as CUDA-device memory")
    {
        auto engine = Engine::load(test_engine_path(), EngineOptions {});
        auto detector = engine->create_detector(DetectorOptions {});
        OwnedRgbImage image(16, 16);
        ImageView invalid = image.view();
        invalid.memory_kind = MemoryKind::CudaDevice;

        check_throws_as(detector->detect(invalid), YoloError);
    }
#endif
}
