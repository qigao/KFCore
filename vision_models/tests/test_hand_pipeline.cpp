#include "kfcore/vision_models/core.hpp"
#include "tinytest.hpp"

#include <cstdint>
#include <cmath>
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

HandResult appearance_hand()
{
    HandResult hand = hand_at(72.0F);
    hand.palm.box = { 48.0F, 48.0F, 160.0F, 160.0F };
    constexpr std::array<HandLandmark, kHandLandmarkCount> kLandmarks = {{
        { 128.0F, 210.0F, 0.0F }, { 100.0F, 198.0F, 0.0F },
        { 82.0F, 178.0F, 0.0F },  { 68.0F, 158.0F, 0.0F },
        { 56.0F, 138.0F, 0.0F },  { 96.0F, 165.0F, 0.0F },
        { 92.0F, 132.0F, 0.0F },  { 88.0F, 100.0F, 0.0F },
        { 84.0F, 68.0F, 0.0F },   { 128.0F, 158.0F, 0.0F },
        { 128.0F, 121.0F, 0.0F }, { 128.0F, 88.0F, 0.0F },
        { 128.0F, 55.0F, 0.0F },  { 158.0F, 164.0F, 0.0F },
        { 163.0F, 130.0F, 0.0F }, { 168.0F, 100.0F, 0.0F },
        { 173.0F, 72.0F, 0.0F },  { 184.0F, 176.0F, 0.0F },
        { 193.0F, 149.0F, 0.0F }, { 200.0F, 125.0F, 0.0F },
        { 206.0F, 102.0F, 0.0F },
    }};
    hand.landmarks = kLandmarks;
    return hand;
}

struct TestImage
{
    std::vector<std::uint8_t> pixels;
    kfcore::image::PixelFormat format = kfcore::image::PixelFormat::Bgr8;

    [[nodiscard]] kfcore::image::ImageView view() const
    {
        return { pixels.data(), pixels.size(), 256, 256, 256U * 3U, format,
                 kfcore::image::MemoryKind::Host };
    }
};

TestImage patterned_image(float brightness, bool swap_red_blue,
                          kfcore::image::PixelFormat format)
{
    TestImage image;
    image.format = format;
    image.pixels.resize(256U * 256U * 3U);
    for (std::size_t y = 0; y < 256U; ++y)
    {
        for (std::size_t x = 0; x < 256U; ++x)
        {
            const auto scaled = [brightness](float value) {
                return static_cast<std::uint8_t>(std::lround(value * brightness));
            };
            std::uint8_t red   = scaled(90.0F + static_cast<float>((x + y) % 20U));
            const std::uint8_t green = scaled(55.0F + static_cast<float>(y % 16U));
            std::uint8_t blue  = scaled(25.0F + static_cast<float>(x % 12U));
            if (swap_red_blue)
            {
                std::swap(red, blue);
            }
            const std::size_t offset = (y * 256U + x) * 3U;
            if (format == kfcore::image::PixelFormat::Bgr8)
            {
                image.pixels[offset + 0U] = blue;
                image.pixels[offset + 1U] = green;
                image.pixels[offset + 2U] = red;
            }
            else
            {
                image.pixels[offset + 0U] = red;
                image.pixels[offset + 1U] = green;
                image.pixels[offset + 2U] = blue;
            }
        }
    }
    return image;
}

void paint_index_finger(TestImage& image)
{
    for (std::size_t y = 58U; y <= 140U; ++y)
    {
        for (std::size_t x = 74U; x <= 102U; ++x)
        {
            const std::size_t offset = (y * 256U + x) * 3U;
            const std::array<std::uint8_t, 3U> rgb = { 20U, 180U, 35U };
            if (image.format == kfcore::image::PixelFormat::Bgr8)
            {
                image.pixels[offset + 0U] = rgb[2];
                image.pixels[offset + 1U] = rgb[1];
                image.pixels[offset + 2U] = rgb[0];
            }
            else
            {
                image.pixels[offset + 0U] = rgb[0];
                image.pixels[offset + 1U] = rgb[1];
                image.pixels[offset + 2U] = rgb[2];
            }
        }
    }
}

double appearance_distance(const HandAppearanceDescriptor& first,
                           const HandAppearanceDescriptor& second)
{
    double distance = 0.0;
    for (std::size_t index = 0; index < first.values.size(); ++index)
    {
        distance += std::fabs(static_cast<double>(first.values[index]) -
                              static_cast<double>(second.values[index]));
    }
    return distance / static_cast<double>(first.values.size());
}

double appearance_part_distance(const HandAppearanceDescriptor& first,
                                const HandAppearanceDescriptor& second,
                                HandAppearancePart part)
{
    const std::size_t offset = hand_appearance_feature_offset(part);
    const std::size_t count = hand_appearance_feature_count(part);
    double distance = 0.0;
    for (std::size_t index = offset; index < offset + count; ++index)
    {
        distance += std::fabs(static_cast<double>(first.values[index]) -
                              static_cast<double>(second.values[index]));
    }
    return distance / static_cast<double>(count);
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
    it("extracts owned palm and five-finger appearance invariant to channel order and brightness")
    {
        HandPipelineOptions options = immediate_tracking_options();
        options.appearance.enabled = true;
        const TestImage bgr = patterned_image(1.0F, false,
                                              kfcore::image::PixelFormat::Bgr8);
        const TestImage rgb = patterned_image(1.0F, false,
                                              kfcore::image::PixelFormat::Rgb8);
        const TestImage brighter = patterned_image(1.5F, false,
                                                   kfcore::image::PixelFormat::Bgr8);
        const TestImage different = patterned_image(1.0F, true,
                                                    kfcore::image::PixelFormat::Bgr8);
        const auto infer = [&](const TestImage& image) {
            auto pipeline = HandPipeline::create(
                std::make_unique<SequenceBackend>(
                    std::vector<std::vector<HandResult>> { { appearance_hand() } }),
                options);
            return pipeline->process(image.view());
        };

        const HandFrame bgr_frame = infer(bgr);
        const HandFrame rgb_frame = infer(rgb);
        const HandFrame bright_frame = infer(brighter);
        const HandFrame different_frame = infer(different);

        check_true(bgr_frame.hands[0].appearance.has_value());
        check(bgr_frame.hands[0].appearance->valid_parts ==
              kAllHandAppearanceParts);
        for (float quality : bgr_frame.hands[0].appearance->quality)
        {
            check(quality > 0.0F);
            check(quality <= 1.0F);
        }
        for (float value : bgr_frame.hands[0].appearance->values)
        {
            check_true(std::isfinite(value));
        }
        check(appearance_distance(*bgr_frame.hands[0].appearance,
                                  *rgb_frame.hands[0].appearance) < 0.0001);
        check(appearance_distance(*bgr_frame.hands[0].appearance,
                                  *bright_frame.hands[0].appearance) < 0.03);
        check(appearance_distance(*bgr_frame.hands[0].appearance,
                                  *different_frame.hands[0].appearance) > 0.10);
        check(bgr_frame.timings.appearance_ms >= 0.0);
    }

    it("isolates index-finger texture from the palm descriptor")
    {
        HandPipelineOptions options = immediate_tracking_options();
        options.appearance.enabled = true;
        const TestImage base = patterned_image(
            1.0F, false, kfcore::image::PixelFormat::Bgr8);
        TestImage marked = base;
        paint_index_finger(marked);
        const auto infer = [&](const TestImage& image) {
            auto pipeline = HandPipeline::create(
                std::make_unique<SequenceBackend>(
                    std::vector<std::vector<HandResult>> { { appearance_hand() } }),
                options);
            return pipeline->process(image.view()).hands[0].appearance;
        };

        const auto plain = infer(base);
        const auto finger_marked = infer(marked);

        check_true(plain.has_value());
        check_true(finger_marked.has_value());
        check(appearance_part_distance(*plain, *finger_marked,
                                       HandAppearancePart::Palm) < 0.03);
        check(appearance_part_distance(*plain, *finger_marked,
                                       HandAppearancePart::Index) > 0.10);
    }

    it("rejects unsupported appearance sources before invoking the backend")
    {
        HandPipelineOptions options = immediate_tracking_options();
        options.appearance.enabled = true;
        const TestImage image = patterned_image(1.0F, false,
                                                kfcore::image::PixelFormat::Bgr8);
        kfcore::image::ImageView device = image.view();
        device.memory_kind = kfcore::image::MemoryKind::CudaDevice;
        auto pipeline = HandPipeline::create(
            std::make_unique<SequenceBackend>(
                std::vector<std::vector<HandResult>> { { appearance_hand() } }),
            options);
        check_error([&] { (void)pipeline->process(device); },
                    VisionModelErrorCode::InvalidArgument, "host BGR8/RGB8");

        kfcore::image::ImageView undersized = image.view();
        undersized.byte_size = 3U;
        check_error([&] { (void)pipeline->process(undersized); },
                    VisionModelErrorCode::InvalidArgument, "byte_size");
    }

    it("rejects invalid appearance extraction options at construction")
    {
        HandPipelineOptions options = immediate_tracking_options();
        options.appearance.minimum_palm_span_pixels = 0.0F;
        check_error(
            [&] {
                (void)HandPipeline::create(
                    std::make_unique<SequenceBackend>(
                        std::vector<std::vector<HandResult>> { { appearance_hand() } }),
                    options);
            },
            VisionModelErrorCode::InvalidArgument, "appearance options");

        options = immediate_tracking_options();
        options.appearance.minimum_part_in_frame_sample_ratio = 1.01F;
        check_error(
            [&] {
                (void)HandPipeline::create(
                    std::make_unique<SequenceBackend>(
                        std::vector<std::vector<HandResult>> { { appearance_hand() } }),
                    options);
            },
            VisionModelErrorCode::InvalidArgument, "appearance options");
    }

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
        check(first.timings.appearance_ms == 0.0);
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
