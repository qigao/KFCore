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
}
