#include "kfcore/yolo/tensorrt.hpp"
#include "tinytest.hpp"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
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

class SilentTensorRtLogger final : public nvinfer1::ILogger
{
public:
    void log(Severity, const char*) noexcept override
    {
    }
};

struct ProfileInputSizes
{
    std::size_t minimum_batch;
    std::array<std::array<std::int32_t, 2>, 3> spatial_sizes;
};

std::array<std::int32_t, 2> checked_spatial_size(const nvinfer1::Dims& shape,
                                                 const char* selector)
{
    if (shape.nbDims != 4 || shape.d[0] <= 0 || shape.d[2] <= 0 || shape.d[3] <= 0 ||
        static_cast<std::uintmax_t>(shape.d[0]) >
            static_cast<std::uintmax_t>((std::numeric_limits<std::size_t>::max)()) ||
        shape.d[2] > (std::numeric_limits<std::int32_t>::max)() ||
        shape.d[3] > (std::numeric_limits<std::int32_t>::max)())
    {
        throw YoloError(YoloErrorCode::EngineContractMismatch,
                        std::string("integration profile inspection stage: invalid ") +
                            selector + " images shape");
    }
    return { static_cast<std::int32_t>(shape.d[2]),
             static_cast<std::int32_t>(shape.d[3]) };
}

ProfileInputSizes inspect_profile_input_sizes(const std::filesystem::path& engine_path)
{
    std::ifstream stream(engine_path, std::ios::binary | std::ios::ate);
    if (!stream.is_open() || stream.tellg() <= std::ifstream::pos_type { 0 })
    {
        throw YoloError(YoloErrorCode::FileIo,
                        "integration profile inspection stage: cannot read engine");
    }
    const std::size_t byte_count = static_cast<std::size_t>(stream.tellg());
    std::vector<char> bytes(byte_count);
    stream.seekg(0, std::ios::beg);
    if (!stream.read(bytes.data(), static_cast<std::streamsize>(byte_count)))
    {
        throw YoloError(YoloErrorCode::FileIo,
                        "integration profile inspection stage: truncated engine");
    }

    SilentTensorRtLogger logger;
    if (!initLibNvInferPlugins(&logger, ""))
    {
        throw YoloError(YoloErrorCode::TensorRtFailure,
                        "integration profile inspection stage: plugin initialization failed");
    }
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    if (!runtime)
    {
        throw YoloError(YoloErrorCode::TensorRtFailure,
                        "integration profile inspection stage: runtime creation failed");
    }
    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!engine || engine->getNbOptimizationProfiles() != 1)
    {
        throw YoloError(YoloErrorCode::EngineContractMismatch,
                        "integration profile inspection stage: exactly profile 0 is required");
    }

    const nvinfer1::Dims minimum = engine->getProfileShape(
        "images", 0, nvinfer1::OptProfileSelector::kMIN);
    const nvinfer1::Dims optimum = engine->getProfileShape(
        "images", 0, nvinfer1::OptProfileSelector::kOPT);
    const nvinfer1::Dims maximum = engine->getProfileShape(
        "images", 0, nvinfer1::OptProfileSelector::kMAX);
    const std::array<std::int32_t, 2> minimum_size =
        checked_spatial_size(minimum, "minimum");
    return {
        static_cast<std::size_t>(minimum.d[0]),
        {{ minimum_size,
           checked_spatial_size(optimum, "optimum"),
           checked_spatial_size(maximum, "maximum") }},
    };
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

    it("runs profile-zero minimum optimum and maximum spatial input sizes")
    {
        const std::filesystem::path engine_path = test_engine_path();
        const ProfileInputSizes profile = inspect_profile_input_sizes(engine_path);
        auto engine = Engine::load(engine_path, EngineOptions {});
        OwnedRgbImage image(960, 320);
        const std::vector<ImageView> batch(profile.minimum_batch, image.view());

        for (const auto& spatial_size : profile.spatial_sizes)
        {
            DetectorOptions options;
            options.input_size = spatial_size;
            options.mirror_horizontal = true;
            auto detector = engine->create_detector(options);
            const std::vector<DetectionFrame> results = detector->detect_batch(batch);
            check(results.size() == profile.minimum_batch);
            for (const DetectionFrame& result : results)
            {
                check(result.image_width == image.view().width);
                check(result.image_height == image.view().height);
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
