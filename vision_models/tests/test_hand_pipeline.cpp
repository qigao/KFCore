#include "kfcore/vision_models/core.hpp"
#include "tinytest.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>

using namespace kfcore::vision_models;

namespace
{

HandResult hand_at(float x)
{
    HandResult hand;
    hand.palm.confidence = 0.95F;
    hand.palm.box        = { x, 10.0F, 20.0F, 30.0F };
    hand.landmark_confidence = 0.9F;
    return hand;
}

class SequenceBackend final : public HandInferenceBackend
{
public:
    explicit SequenceBackend(std::vector<std::vector<HandResult>> frames)
        : frames_(std::move(frames))
    {
    }

    HandFrame infer(const kfcore::image::ImageView&) override
    {
        HandFrame frame;
        frame.hands = frames_.at(index_++);
        return frame;
    }

private:
    std::vector<std::vector<HandResult>> frames_;
    std::size_t index_ = 0;
};

class FixtureFaceDetector final : public FaceDetectorBackend
{
public:
    explicit FixtureFaceDetector(std::optional<FaceDetection> face)
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

class FixtureFaceLandmarker final : public FaceLandmarkBackend
{
public:
    explicit FixtureFaceLandmarker(float confidence)
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
        result.confidence              = confidence_;
        result.landmarks[0]            = { 11.0F, 22.0F, 0.5F };
        result.landmarks.back()        = { 33.0F, 44.0F, -0.5F };
        result.preprocess_ms           = 0.75;
        result.inference_ms            = 1.50;
        result.total_ms                = 2.25;
        return result;
    }

private:
    float confidence_ = 0.0F;
};

class RejectingFaceLandmarker final : public FaceLandmarkBackend
{
public:
    FaceLandmarkResult infer(const kfcore::image::ImageView&, const RectF&) override
    {
        throw std::runtime_error("landmarker must not run without a face");
    }
};

kfcore::image::ImageView one_pixel_image()
{
    static const std::uint8_t pixel[3] = { 0, 0, 0 };
    return { pixel, sizeof(pixel), 1, 1, 3, kfcore::image::PixelFormat::Bgr8,
             kfcore::image::MemoryKind::Host };
}

HandPipelineOptions immediate_tracking_options()
{
    HandPipelineOptions options;
    options.max_hands = 2;
    options.tracker.minimum_consecutive_frames = 1;
    return options;
}

void check_error(const std::function<void()>& operation, VisionModelErrorCode code,
                 const std::string& message)
{
    bool threw = false;
    try
    {
        operation();
    }
    catch (const VisionModelError& error)
    {
        threw = true;
        check(error.code() == code);
        check(std::string(error.what()).find(message) != std::string::npos);
    }
    check_true(threw);
}

} // namespace

spec("vision hand pipeline")
{
    it("associates ByteTrack ids through detection_index while preserving backend order")
    {
        auto backend = std::make_unique<SequenceBackend>(
            std::vector<std::vector<HandResult>> { { hand_at(0.0F), hand_at(100.0F) },
                                                   { hand_at(100.5F), hand_at(0.5F) } });
        auto pipeline = HandPipeline::create(std::move(backend), immediate_tracking_options());

        const HandFrame first = pipeline->process(one_pixel_image());
        const HandFrame second = pipeline->process(one_pixel_image());

        check(first.hands[0].track_id == -1);
        check(first.hands[1].track_id == -1);
        check(second.hands[0].track_id == 1);
        check(second.hands[1].track_id == 0);
        check(second.hands[0].palm.box.x == 100.5F);
        check(second.hands[1].palm.box.x == 0.5F);
    }

    it("keeps pipeline instances independent and reset restarts ids")
    {
        const auto frames = std::vector<std::vector<HandResult>> {
            { hand_at(0.0F) }, { hand_at(0.5F) }, { hand_at(0.0F) }, { hand_at(0.5F) }
        };
        auto first = HandPipeline::create(std::make_unique<SequenceBackend>(frames),
                                          immediate_tracking_options());
        auto second = HandPipeline::create(std::make_unique<SequenceBackend>(frames),
                                           immediate_tracking_options());

        (void)first->process(one_pixel_image());
        check(first->process(one_pixel_image()).hands[0].track_id == 0);
        (void)second->process(one_pixel_image());
        check(second->process(one_pixel_image()).hands[0].track_id == 0);

        first->reset();
        check(first->process(one_pixel_image()).hands[0].track_id == -1);
        check(first->process(one_pixel_image()).hands[0].track_id == 0);
    }

    it("rejects backend results beyond max_hands before advancing tracking")
    {
        auto options = immediate_tracking_options();
        options.max_hands = 1;
        auto pipeline = HandPipeline::create(
            std::make_unique<SequenceBackend>(
                std::vector<std::vector<HandResult>> { { hand_at(0.0F), hand_at(100.0F) },
                                                       { hand_at(0.0F) },
                                                       { hand_at(0.5F) } }),
            options);

        check_error([&] { (void)pipeline->process(one_pixel_image()); },
                    VisionModelErrorCode::ResourceLimitExceeded, "max_hands");
        check(pipeline->process(one_pixel_image()).hands[0].track_id == -1);
        check(pipeline->process(one_pixel_image()).hands[0].track_id == 0);
    }
}

spec("vision FaceMesh pipeline")
{
    it("composes detection and landmark results with stage timings")
    {
        auto pipeline = FaceMeshPipeline::create(
            std::make_unique<FixtureFaceDetector>(
                FaceDetection { { 10.0F, 20.0F, 30.0F, 40.0F }, 0.90F }),
            std::make_unique<FixtureFaceLandmarker>(0.80F),
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
        check(frame.timings.total_ms >= 0.0);
    }

    it("returns an empty frame without invoking landmarks when no face is detected")
    {
        auto pipeline = FaceMeshPipeline::create(
            std::make_unique<FixtureFaceDetector>(std::nullopt),
            std::make_unique<RejectingFaceLandmarker>());

        const FaceMeshFrame frame = pipeline->process(one_pixel_image());

        check_false(frame.detection.has_value());
        check_false(frame.landmarks.has_value());
        check(frame.timings.detection_inference_ms == 2.50);
    }

    it("retains detection but hides landmarks below the configured confidence")
    {
        auto pipeline = FaceMeshPipeline::create(
            std::make_unique<FixtureFaceDetector>(
                FaceDetection { { 10.0F, 20.0F, 30.0F, 40.0F }, 0.90F }),
            std::make_unique<FixtureFaceLandmarker>(0.49F),
            FaceMeshPipelineOptions { 0.50F });

        const FaceMeshFrame frame = pipeline->process(one_pixel_image());

        check_true(frame.detection.has_value());
        check_false(frame.landmarks.has_value());
        check(frame.timings.landmark_inference_ms == 1.50);
    }

    it("rejects invalid confidence configuration and null dependencies")
    {
        check_error(
            [&] {
                (void)FaceMeshPipeline::create(
                    std::make_unique<FixtureFaceDetector>(std::nullopt),
                    std::make_unique<RejectingFaceLandmarker>(),
                    FaceMeshPipelineOptions { 1.01F });
            },
            VisionModelErrorCode::InvalidArgument, "landmark_score_threshold");
        check_error(
            [&] {
                (void)FaceMeshPipeline::create(
                    {}, std::make_unique<RejectingFaceLandmarker>());
            },
            VisionModelErrorCode::InvalidArgument, "detector");
    }
}
