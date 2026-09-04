#include "kfcore/yolo/tensorrt.hpp"

#include "tinytest.hpp"

#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

using namespace kfcore::yolo;

namespace
{

std::filesystem::path required_person_engine()
{
    const char* value = std::getenv("KFCORE_TENSORRT_TEST_PERSON_ENGINE");
    if (value == nullptr || *value == '\0')
    {
        throw YoloError(
            YoloErrorCode::InvalidArgument,
            "person integration setup stage: KFCORE_TENSORRT_TEST_PERSON_ENGINE is required");
    }
    return std::filesystem::path(value);
}

} // namespace

spec("YOLOv8s person-model TensorRT integration")
{
    it("loads the raw COCO detection head and executes a frame")
    {
        constexpr std::int32_t kWidth = 640;
        constexpr std::int32_t kHeight = 480;
        std::vector<std::uint8_t> pixels(
            static_cast<std::size_t>(kWidth) * kHeight * 3U, 114U);

        EngineOptions engine_options;
        engine_options.tensor_names.detections = "predictions";
        auto engine = Engine::load(required_person_engine(), engine_options);
        auto detector = engine->create_detector();
        const DetectionFrame frame = detector->detect(
            {pixels.data(), kWidth, kHeight,
             static_cast<std::size_t>(kWidth) * 3U,
             PixelFormat::Bgr8, MemoryKind::Host});

        check(frame.image_width == kWidth);
        check(frame.image_height == kHeight);
        check(frame.detections.size() <= engine_options.max_detections);
        for (const Detection& detection : frame.detections)
        {
            check(detection.class_id >= 0);
            check(detection.class_id < 80);
        }
    }
}
