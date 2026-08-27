#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "kfcore/vision_models/tensorrt.hpp"
#include "tinytest.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

using namespace kfcore::vision_models;

namespace
{

class DeviceImage final
{
public:
    explicit DeviceImage(const kfcore::image::BgrImage& source)
        : byte_size_(source.pixels.size())
        , width_(source.width)
        , height_(source.height)
    {
        if (cudaMalloc(&data_, byte_size_) != cudaSuccess ||
            cudaMemcpy(data_, source.pixels.data(), byte_size_,
                       cudaMemcpyHostToDevice) != cudaSuccess)
        {
            if (data_ != nullptr)
            {
                (void)cudaFree(data_);
            }
            throw std::runtime_error("failed to upload integration image to CUDA");
        }
    }

    ~DeviceImage()
    {
        if (data_ != nullptr)
        {
            (void)cudaFree(data_);
        }
    }

    DeviceImage(const DeviceImage&)            = delete;
    DeviceImage& operator=(const DeviceImage&) = delete;

    kfcore::image::ImageView view() const noexcept
    {
        return { data_, byte_size_, width_, height_,
                 static_cast<std::size_t>(width_) * 3U,
                 kfcore::image::PixelFormat::Bgr8,
                 kfcore::image::MemoryKind::CudaDevice };
    }

private:
    void*        data_      = nullptr;
    std::size_t  byte_size_ = 0;
    std::int32_t width_     = 0;
    std::int32_t height_    = 0;
};

kfcore::image::BgrImage read_bgr(const std::filesystem::path& path)
{
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* rgb = stbi_load(path.string().c_str(), &width, &height, &channels, 3);
    if (rgb == nullptr || width <= 0 || height <= 0)
    {
        throw std::runtime_error("failed to decode TensorRT integration image");
    }
    kfcore::image::BgrImage result;
    result.width  = width;
    result.height = height;
    result.pixels.resize(static_cast<std::size_t>(width) * height * 3U);
    for (std::size_t pixel = 0; pixel < result.pixels.size(); pixel += 3U)
    {
        result.pixels[pixel]      = rgb[pixel + 2U];
        result.pixels[pixel + 1U] = rgb[pixel + 1U];
        result.pixels[pixel + 2U] = rgb[pixel];
    }
    stbi_image_free(rgb);
    return result;
}

HandTensorRtEnginePaths hand_paths()
{
    return { KFCORE_VISION_TRT_TEST_PALM, KFCORE_VISION_TRT_TEST_HAND,
             KFCORE_VISION_TRT_TEST_CLASSIFIER };
}

void check_hand_frame(const HandFrame& frame, std::size_t max_hands)
{
    check_true(!frame.hands.empty());
    check(frame.hands.size() <= max_hands);
    check_true(std::isfinite(frame.timings.total_ms));
    check_true(frame.timings.preprocess_ms > 0.0);
    check_true(frame.timings.palm_inference_ms > 0.0);
    check_true(frame.timings.landmark_inference_ms > 0.0);
    check_true(frame.timings.classifier_inference_ms > 0.0);
    for (const HandResult& hand : frame.hands)
    {
        check_true(std::isfinite(hand.palm.confidence));
        check_true(std::isfinite(hand.landmark_confidence));
        check(hand.gesture != Gesture::Unknown);
        for (const HandLandmark& point : hand.landmarks)
        {
            check_true(std::isfinite(point.x));
            check_true(std::isfinite(point.y));
            check_true(std::isfinite(point.z));
        }
    }
}

} // namespace

spec("TensorRT CUDA vision real-engine integration")
{
    it("runs Palm, hand landmark, INT64 gesture and ByteTrack from Host and CUDA images")
    {
        TensorRtVisionOptions options;
        auto backend = TensorRtHandBackend::load(hand_paths(), options);
        const kfcore::image::BgrImage image = read_bgr(
            KFCORE_VISION_TRT_TEST_HAND_IMAGE);

        const HandFrame host_frame = backend->infer(image.view());
        check_hand_frame(host_frame, options.max_hands);

        const DeviceImage device_image(image);
        const HandFrame device_frame = backend->infer(device_image.view());
        check_hand_frame(device_frame, options.max_hands);
        check(device_frame.hands.size() == host_frame.hands.size());

        HandPipelineOptions pipeline_options;
        pipeline_options.max_hands = options.max_hands;
        pipeline_options.tracker.minimum_consecutive_frames = 1;
        auto pipeline = HandPipeline::create(
            TensorRtHandBackend::load(hand_paths(), options), pipeline_options);
        (void)pipeline->process(device_image.view());
        const HandFrame tracked = pipeline->process(device_image.view());
        bool has_track = false;
        for (const HandResult& hand : tracked.hands)
        {
            has_track = has_track || hand.track_id >= 0;
        }
        check_true(has_track);
        check_true(tracked.timings.tracking_ms > 0.0);
    }

    it("runs the MediaPipe 468 face landmark engine on the CUDA preprocessing path")
    {
        TensorRtVisionOptions options;
        auto landmarker = TensorRtFaceLandmarker::load(
            KFCORE_VISION_TRT_TEST_FACE, options);
        const kfcore::image::BgrImage image = read_bgr(
            KFCORE_VISION_TRT_TEST_FACE_IMAGE);
        const FaceLandmarkResult result = landmarker->infer(
            image.view(), { 0.0F, 0.0F, static_cast<float>(image.width),
                            static_cast<float>(image.height) });
        check_true(std::isfinite(result.confidence));
        check_true(result.preprocess_ms > 0.0);
        check_true(result.inference_ms > 0.0);
        check_true(result.total_ms > 0.0);
        for (const Point3f& point : result.landmarks)
        {
            check_true(std::isfinite(point.x));
            check_true(std::isfinite(point.y));
            check_true(std::isfinite(point.z));
        }
    }
}
