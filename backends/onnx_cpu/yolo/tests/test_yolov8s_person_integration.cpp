#include "kfcore/yolo/onnx.hpp"

#include "tinytest.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

using namespace kfcore::yolo;

spec("YOLOv8s person-model ONNX integration") {
    it("loads the raw COCO detection head and executes a frame") {
        constexpr std::int32_t kWidth = 640;
        constexpr std::int32_t kHeight = 480;
        std::vector<std::uint8_t> pixels(
            static_cast<std::size_t>(kWidth) * kHeight * 3U, 114U);
        OnnxDetectorOptions options;
        options.output_name = "predictions";
        options.max_detections = 321;
        options.intra_op_threads = 1;
        options.inter_op_threads = 1;
        auto detector = OnnxDetector::load(KFCORE_YOLO_TEST_PERSON, options);

        const DetectionFrame frame = detector->detect(
            {pixels.data(), kWidth, kHeight,
             static_cast<std::size_t>(kWidth) * 3U,
             PixelFormat::Bgr8, MemoryKind::Host});

        check(frame.image_width == kWidth);
        check(frame.image_height == kHeight);
        check(detector->max_detections() == options.max_detections);
        check(frame.detections.size() <= detector->max_detections());
        for (const Detection& detection : frame.detections) {
            check(detection.class_id >= 0);
            check(detection.class_id < 80);
        }
    }
}
