#include "yolo_domain_onnx.hpp"
#include "yolo_domain_profile.hpp"

#include "tinytest.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

using namespace kfcore::yolo;
using namespace kfcore::yolo::demo;

namespace
{

struct ModelCase
{
    const char* path;
    DomainKind kind;
    std::int32_t extent;
};

ImageView image_view(const std::vector<std::uint8_t>& pixels,
                     std::int32_t width, std::int32_t height)
{
    return { pixels.data(), width, height, static_cast<std::size_t>(width) * 3U,
             PixelFormat::Bgr8, MemoryKind::Host };
}
} // namespace

spec("YOLOv8 ONNX Runtime real-model integration")
{
    it("loads and executes drone football and parking compact-NMS models")
    {
        constexpr std::int32_t kSourceWidth = 640;
        constexpr std::int32_t kSourceHeight = 480;
        std::vector<std::uint8_t> pixels(
            static_cast<std::size_t>(kSourceWidth) * kSourceHeight * 3U);
        for (std::size_t index = 0U; index < pixels.size(); ++index)
        {
            pixels[index] = static_cast<std::uint8_t>((index * 17U) % 251U);
        }

        const std::array<ModelCase, 3> models {
            ModelCase { KFCORE_YOLO_TEST_DRONE, DomainKind::Drone, 640 },
            ModelCase { KFCORE_YOLO_TEST_FOOTBALL, DomainKind::Football, 960 },
            ModelCase { KFCORE_YOLO_TEST_PARKING, DomainKind::Parking, 640 },
        };
        for (const ModelCase& model : models)
        {
            OnnxDetectorOptions options;
            options.intra_op_threads = 1;
            options.inter_op_threads = 1;
            options.mirror_horizontal = true;
            auto detector = OnnxDomainDetector::load(model.path, options);
            check(detector->input_width() == model.extent);
            check(detector->input_height() == model.extent);
            check(detector->max_detections() == 300U);

            const DetectionFrame frame = detector->detect(
                image_view(pixels, kSourceWidth, kSourceHeight));
            check(frame.image_width == kSourceWidth);
            check(frame.image_height == kSourceHeight);
            check(frame.detections.size() <= detector->max_detections());
            const std::size_t class_count = domain_profile(model.kind).class_labels.size();
            for (const Detection& detection : frame.detections)
            {
                check_true(std::isfinite(detection.score));
                check_true(detection.score > 0.0F && detection.score <= 1.0F);
                check(detection.class_id >= 0);
                check(static_cast<std::size_t>(detection.class_id) < class_count);
                check_true(detection.box.left >= 0.0F);
                check_true(detection.box.top >= 0.0F);
                check_true(detection.box.right <= static_cast<float>(kSourceWidth));
                check_true(detection.box.bottom <= static_cast<float>(kSourceHeight));
                check_true(detection.box.left < detection.box.right);
                check_true(detection.box.top < detection.box.bottom);
            }

            std::vector<std::uint8_t> nv12(
                static_cast<std::size_t>(kSourceWidth) * kSourceHeight * 3U / 2U,
                128U);
            std::fill_n(nv12.begin(),
                        static_cast<std::size_t>(kSourceWidth) * kSourceHeight,
                        81U);
            const ImageView native_view {
                nv12.data(), kSourceWidth, kSourceHeight,
                static_cast<std::size_t>(kSourceWidth), PixelFormat::Nv12,
                MemoryKind::Host,
            };
            const DetectionFrame native_frame = detector->detect(native_view);
            check(native_frame.image_width == kSourceWidth);
            check(native_frame.image_height == kSourceHeight);
            check(native_frame.detections.size() <= detector->max_detections());
        }
    }
}
