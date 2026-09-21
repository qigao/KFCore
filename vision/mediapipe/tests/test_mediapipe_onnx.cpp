#include "kfcore/mediapipe/hand_landmarker.hpp"
#include "tinytest.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>

namespace
{
std::filesystem::path required_path(const char* name)
{
    const char* value = std::getenv(name);
    if (!value || !*value) throw std::runtime_error(std::string("required integration input: ") + name);
    return value;
}
}

spec("MediaPipe real ONNX inference")
{
    it("runs two real models on a hand image without a classifier")
    {
        using namespace kfcore;
        const auto root = required_path("KFCORE_MEDIAPIPE_TEST_ASSETS");
        runtime::Runtime runtime;
        const auto backend = runtime.load_backend(required_path("KFCORE_MEDIAPIPE_TEST_BACKEND"));
        const auto palm = runtime::ModelPackage::load(root / "palm_detection_full_inf_post_192x192.json");
        const auto landmark = runtime::ModelPackage::load(root / "hand_landmark_sparse_Nx3x224x224.json");
        const auto policy = runtime::ExecutionPolicy::exact("onnxruntime", "cpu");
        auto detector = mediapipe::HandLandmarker::load(runtime, palm, landmark, policy);
        constexpr std::int32_t kWidth = 640, kHeight = 480;
        constexpr std::size_t kChannels = 3;
        std::size_t size = 0;
        const std::unique_ptr<char, decltype(&std::free)> pixels(
            tt_read_file((root / "hand-640x480.bgr").string().c_str(), &size), &std::free);
        if (!pixels || size != kWidth * kHeight * kChannels)
            throw std::runtime_error("integration image must be packed 640x480 BGR8");
        const image::ImageView source{pixels.get(), size, kWidth, kHeight,
            kWidth * kChannels, image::PixelFormat::Bgr8, image::MemoryKind::Host};
        const auto frame = detector->infer(source);
        check_false(frame.hands.empty());
        check_true(frame.timings.classifier_inference_ms == 0.0);
        check_true(detector->palm_execution_route().backend_id == "onnxruntime");
        for (const auto& hand : frame.hands)
        {
            check_true(hand.gesture == hand_models::Gesture::Unknown);
            check_true(hand.track_id == -1);
            check_true(hand.handedness != hand_models::Handedness::Unknown);
            check_true(hand.landmarks.size() == hand_models::kHandLandmarkCount);
            for (const auto& point : hand.landmarks)
                check_true(std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z));
        }
        const auto repeated = detector->infer(source);
        check_true(repeated.hands.size() == frame.hands.size());
        check_throws_as(detector->infer(image::ImageView{}), hand_models::HandModelError);
        const auto recovered = detector->infer(source);
        check_true(recovered.hands.size() == frame.hands.size());
    }
}
