#include "raw_yolo.hpp"

#include "tinytest.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

using namespace kfcore::yolo;

spec("raw YOLO detection head decoding") {
    it("applies class-aware NMS and restores letterboxed boxes") {
        constexpr std::size_t kCandidates = 3U;
        constexpr std::size_t kClasses = 2U;
        std::vector<float> values((4U + kClasses) * kCandidates, 0.0F);
        const auto set = [&](std::size_t channel, std::size_t candidate, float value) {
            values[channel * kCandidates + candidate] = value;
        };

        set(0U, 0U, 50.0F); set(1U, 0U, 50.0F);
        set(2U, 0U, 20.0F); set(3U, 0U, 20.0F); set(4U, 0U, 0.90F);
        set(0U, 1U, 51.0F); set(1U, 1U, 51.0F);
        set(2U, 1U, 20.0F); set(3U, 1U, 20.0F); set(4U, 1U, 0.80F);
        set(0U, 2U, 50.0F); set(1U, 2U, 50.0F);
        set(2U, 2U, 20.0F); set(3U, 2U, 20.0F); set(5U, 2U, 0.70F);

        const std::vector<ImageView> images = {
            {values.data(), 100, 100, 300U, PixelFormat::Bgr8,
             MemoryKind::Host}};
        const std::vector<detail::LetterboxTransform> transforms = {
            {1.0F, 0.0F, 0.0F, 100, 100}};
        const detail::RawYoloOutputView output = {
            values.data(), values.size(), kClasses, kCandidates,
            TensorDataType::Float32, 0.25F, 0.45F, 10U};

        const auto frames = detail::decode_raw_yolo(images, transforms, output);

        check(frames.size() == std::size_t{1U});
        check(frames[0].detections.size() == std::size_t{2U});
        check(frames[0].detections[0].class_id == std::int32_t{0});
        check(frames[0].detections[1].class_id == std::int32_t{1});
    }

    it("decodes YOLOX anchor-major objectness and class confidence") {
        constexpr std::size_t kCandidates = 3U;
        constexpr std::size_t kClasses = 2U;
        constexpr std::size_t kChannels = 5U + kClasses;
        std::vector<float> values(kCandidates * kChannels, 0.0F);
        const auto set = [&](std::size_t candidate, std::size_t channel, float value) {
            values[candidate * kChannels + channel] = value;
        };

        set(0U, 0U, 50.0F); set(0U, 1U, 50.0F);
        set(0U, 2U, 20.0F); set(0U, 3U, 20.0F);
        set(0U, 4U, 0.90F); set(0U, 5U, 0.80F);

        set(1U, 0U, 51.0F); set(1U, 1U, 51.0F);
        set(1U, 2U, 20.0F); set(1U, 3U, 20.0F);
        set(1U, 4U, 0.95F); set(1U, 5U, 0.70F);

        set(2U, 0U, 50.0F); set(2U, 1U, 50.0F);
        set(2U, 2U, 20.0F); set(2U, 3U, 20.0F);
        set(2U, 4U, 0.80F); set(2U, 6U, 0.75F);

        const std::vector<ImageView> images = {
            {values.data(), 100, 100, 300U, PixelFormat::Bgr8,
             MemoryKind::Host}};
        const std::vector<detail::LetterboxTransform> transforms = {
            {1.0F, 0.0F, 0.0F, 100, 100}};
        const detail::RawYoloOutputView output = {
            values.data(), values.size(), kClasses, kCandidates,
            TensorDataType::Float32, 0.25F, 0.45F, 10U,
            detail::RawYoloOutputLayout::
                AnchorsFirstObjectnessClassScores};

        const auto frames =
            detail::decode_raw_yolo(images, transforms, output);

        check(frames.size() == std::size_t{1U});
        check(frames[0].detections.size() == std::size_t{2U});
        check(frames[0].detections[0].class_id == std::int32_t{0});
        check(frames[0].detections[1].class_id == std::int32_t{1});
        check_true(
            std::fabs(frames[0].detections[0].score - 0.72F) <
            1.0e-6F);
        check_true(
            std::fabs(frames[0].detections[1].score - 0.60F) <
            1.0e-6F);
    }
}
