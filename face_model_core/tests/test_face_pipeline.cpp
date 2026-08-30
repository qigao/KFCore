#include "kfcore/face_models/core.hpp"

#include "tinytest.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

using namespace kfcore::face_models;

namespace
{

struct ImageObservation
{
    const void*               data = nullptr;
    kfcore::image::MemoryKind memory_kind = kfcore::image::MemoryKind::Host;
    std::size_t               calls = 0U;
};

class FixtureDetector final : public FaceDetectorBackend
{
public:
    explicit FixtureDetector(std::optional<FaceDetection> face)
        : face_(std::move(face))
    {
    }

    FaceDetectionResult infer(const kfcore::image::ImageView&) override
    {
        FaceDetectionResult result;
        result.face          = face_;
        result.preprocess_ms = 1.25;
        result.inference_ms  = 2.50;
        result.total_ms      = 3.75;
        return result;
    }

private:
    std::optional<FaceDetection> face_;
};

class FixtureLandmarker final : public FaceLandmarkBackend
{
public:
    explicit FixtureLandmarker(float confidence)
        : confidence_(confidence)
    {
    }

    FaceLandmarkResult infer(const kfcore::image::ImageView&, const RectF& box) override
    {
        if (box.x != 10.0F || box.y != 20.0F || box.width != 30.0F ||
            box.height != 40.0F)
        {
            throw std::runtime_error("FaceMesh pipeline passed the wrong face box");
        }
        FaceLandmarkResult result;
        result.confidence       = confidence_;
        result.landmarks[0]     = { 11.0F, 22.0F, 0.5F };
        result.landmarks.back() = { 33.0F, 44.0F, -0.5F };
        result.preprocess_ms    = 0.75;
        result.inference_ms     = 1.50;
        result.total_ms         = 2.25;
        return result;
    }

private:
    float confidence_ = 0.0F;
};

class RejectingLandmarker final : public FaceLandmarkBackend
{
public:
    FaceLandmarkResult infer(const kfcore::image::ImageView&, const RectF&) override
    {
        throw std::runtime_error("landmarker must not run without a face");
    }
};

class RecordingDetector final : public FaceDetectorBackend
{
public:
    explicit RecordingDetector(std::shared_ptr<ImageObservation> observation)
        : observation_(std::move(observation))
    {
    }

    FaceDetectionResult infer(const kfcore::image::ImageView& image) override
    {
        observation_->data        = image.data;
        observation_->memory_kind = image.memory_kind;
        ++observation_->calls;
        FaceDetectionResult result;
        result.face = FaceDetection { { 10.0F, 20.0F, 30.0F, 40.0F }, 0.90F };
        return result;
    }

private:
    std::shared_ptr<ImageObservation> observation_;
};

class RecordingLandmarker final : public FaceLandmarkBackend
{
public:
    explicit RecordingLandmarker(std::shared_ptr<ImageObservation> observation)
        : observation_(std::move(observation))
    {
    }

    FaceLandmarkResult infer(const kfcore::image::ImageView& image,
                             const RectF&) override
    {
        observation_->data        = image.data;
        observation_->memory_kind = image.memory_kind;
        ++observation_->calls;
        FaceLandmarkResult result;
        result.confidence = 0.90F;
        return result;
    }

private:
    std::shared_ptr<ImageObservation> observation_;
};

kfcore::image::ImageView one_pixel_image()
{
    static const std::uint8_t pixel[3] = { 0, 0, 0 };
    return { pixel, sizeof(pixel), 1, 1, 3, kfcore::image::PixelFormat::Bgr8,
             kfcore::image::MemoryKind::Host };
}

void check_error(const std::function<void()>& operation, FaceModelErrorCode code,
                 const std::string& message)
{
    bool threw = false;
    try
    {
        operation();
    }
    catch (const FaceModelError& error)
    {
        threw = true;
        check(error.code() == code);
        check(std::string(error.what()).find(message) != std::string::npos);
    }
    check_true(threw);
}

} // namespace

spec("face mesh pipeline")
{
    it("shares one compute image view between detector and landmarker")
    {
        const auto source = one_pixel_image();
        static const std::uint8_t device_sentinel = 0U;
        auto compute = source;
        compute.data = &device_sentinel;
        compute.memory_kind = kfcore::image::MemoryKind::CudaDevice;
        const auto detector_observation = std::make_shared<ImageObservation>();
        const auto landmark_observation = std::make_shared<ImageObservation>();
        auto pipeline = FaceMeshPipeline::create(
            std::make_unique<RecordingDetector>(detector_observation),
            std::make_unique<RecordingLandmarker>(landmark_observation));

        const FaceMeshFrame result = pipeline->process(
            kfcore::image::FrameView { source, compute });

        check_true(result.detection.has_value());
        check_true(result.landmarks.has_value());
        check_true(detector_observation->calls == 1U);
        check_true(landmark_observation->calls == 1U);
        check_true(detector_observation->data == compute.data);
        check_true(landmark_observation->data == compute.data);
        check(detector_observation->memory_kind == kfcore::image::MemoryKind::CudaDevice);
        check(landmark_observation->memory_kind == kfcore::image::MemoryKind::CudaDevice);
    }

    it("composes detection and landmark results with stage timings")
    {
        auto pipeline = FaceMeshPipeline::create(
            std::make_unique<FixtureDetector>(
                FaceDetection { { 10.0F, 20.0F, 30.0F, 40.0F }, 0.90F }),
            std::make_unique<FixtureLandmarker>(0.80F),
            FaceMeshPipelineOptions { 0.50F });

        const FaceMeshFrame frame = pipeline->process(one_pixel_image());

        check_true(frame.detection.has_value());
        check_true(frame.landmarks.has_value());
        check(frame.detection->confidence == 0.90F);
        check(frame.landmarks->landmarks[0].x == 11.0F);
        check(frame.landmarks->landmarks.back().y == 44.0F);
        check(frame.timings.detection_preprocess_ms == 1.25);
        check(frame.timings.detection_inference_ms == 2.50);
        check(frame.timings.landmark_preprocess_ms == 0.75);
        check(frame.timings.landmark_inference_ms == 1.50);
    }

    it("skips or hides landmarks when detection or confidence is insufficient")
    {
        auto no_face = FaceMeshPipeline::create(
            std::make_unique<FixtureDetector>(std::nullopt),
            std::make_unique<RejectingLandmarker>());
        const FaceMeshFrame empty = no_face->process(one_pixel_image());
        check_false(empty.detection.has_value());
        check_false(empty.landmarks.has_value());

        auto low_confidence = FaceMeshPipeline::create(
            std::make_unique<FixtureDetector>(
                FaceDetection { { 10.0F, 20.0F, 30.0F, 40.0F }, 0.90F }),
            std::make_unique<FixtureLandmarker>(0.49F),
            FaceMeshPipelineOptions { 0.50F });
        const FaceMeshFrame filtered = low_confidence->process(one_pixel_image());
        check_true(filtered.detection.has_value());
        check_false(filtered.landmarks.has_value());
    }

    it("rejects invalid configuration and null dependencies")
    {
        check_error(
            [&] {
                (void)FaceMeshPipeline::create(
                    std::make_unique<FixtureDetector>(std::nullopt),
                    std::make_unique<RejectingLandmarker>(),
                    FaceMeshPipelineOptions { 1.01F });
            },
            FaceModelErrorCode::InvalidArgument, "landmark_score_threshold");
        check_error(
            [&] {
                (void)FaceMeshPipeline::create(
                    {}, std::make_unique<RejectingLandmarker>());
            },
            FaceModelErrorCode::InvalidArgument, "detector");
    }
}
