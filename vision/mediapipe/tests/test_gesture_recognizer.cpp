#include "kfcore/mediapipe/gesture_recognizer.hpp"
#include "tinytest.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <cstring>

using namespace kfcore;
spec("MediaPipe gesture features") {
    it("features require real world landmarks instead of fabricating them") {
        hand_models::HandResult hand;
        check_throws_as(mediapipe::gesture_features(hand, 640, 480), std::invalid_argument);
    }
    it("features normalize screen and world geometry independently at the wrist") {
        hand_models::HandResult hand;
        hand.world_landmarks.emplace(); hand.right_hand_probability = 0.8F;
        for (std::size_t i = 0; i < hand.landmarks.size(); ++i) {
            hand.landmarks[i] = {100.0F + float(i), 200.0F + float(i), 0};
            (*hand.world_landmarks)[i] = {float(i), float(i), 0};
        }
        const auto features = mediapipe::gesture_features(hand, 640, 480);
        check_true(features.hand[0] == 0 && features.world[0] == 0);
        check_true(std::abs(features.hand[60] - 1) < 0.001F);
        check_true(std::abs(features.world[60] - 1) < 0.001F);
        check_true(features.right_hand_probability == 0.8F);
    }
    it("features distinguish official None from no hand and reject invalid labels") {
        check_true(std::string(mediapipe::gesture_name(mediapipe::CannedGesture::None)) == "None");
        check_throws_as(mediapipe::gesture_name(mediapipe::CannedGesture(8)), std::invalid_argument);
    }
}

spec("MediaPipe official real inference") {
    it("runs world landmarks embedding and canned scores on the recorded hand") {
        const char* assets = std::getenv("KFCORE_GESTURE_TEST_ASSETS");
        const char* hands = std::getenv("KFCORE_MEDIAPIPE_TEST_ASSETS");
        const char* plugin = std::getenv("KFCORE_MEDIAPIPE_TEST_BACKEND");
        if (!assets || !hands || !plugin) throw std::runtime_error("real inference requires model and image assets");
        const std::filesystem::path root(assets), source_root(hands);
        runtime::Runtime runtime;
        const auto backend = runtime.load_backend(plugin);
        auto recognizer = mediapipe::GestureRecognizer::load(runtime,
            runtime::ModelPackage::load(source_root / "palm_detection_full_inf_post_192x192.json"),
            runtime::ModelPackage::load(root / "hand_world.json"),
            runtime::ModelPackage::load(root / "gesture_embedder.json"),
            runtime::ModelPackage::load(root / "gesture_classifier.json"),
            runtime::ExecutionPolicy::exact("onnxruntime", "cpu"));
        constexpr std::size_t kFeatureValues = hand_models::kHandLandmarkCount * 6 + 1;
        std::size_t golden_bytes = 0;
        const std::unique_ptr<char, decltype(&std::free)> golden(
            tt_read_file((root / "classification-golden.f32").string().c_str(), &golden_bytes), &std::free);
        std::array<float, kFeatureValues + mediapipe::kGestureCount> oracle{};
        if (!golden || golden_bytes != sizeof(oracle)) throw std::runtime_error("missing TFLite classification oracle");
        std::memcpy(oracle.data(), golden.get(), sizeof(oracle));
        mediapipe::GestureFeatures features;
        std::copy_n(oracle.begin(), features.hand.size(), features.hand.begin());
        std::copy_n(oracle.begin()+features.hand.size(), features.world.size(), features.world.begin());
        features.right_hand_probability = oracle[kFeatureValues-1];
        const auto classified = recognizer->classify(features);
        for (std::size_t i = 0; i < classified.scores.size(); ++i)
            check_true(std::abs(classified.scores[i]-oracle[kFeatureValues+i]) < 0.00001F);
        constexpr int kWidth = 640, kHeight = 480, kChannels = 3;
        std::size_t size = 0;
        const std::unique_ptr<char, decltype(&std::free)> data(
            tt_read_file((source_root / "hand-640x480.bgr").string().c_str(), &size), &std::free);
        if (!data || size != kWidth*kHeight*kChannels) throw std::runtime_error("invalid BGR test image");
        const image::ImageView image{data.get(), size, kWidth, kHeight, kWidth*kChannels,
            image::PixelFormat::Bgr8, image::MemoryKind::Host};
        const auto frame = recognizer->infer(image);
        check_false(frame.landmarks.hands.empty());
        check_true(frame.gestures.size() == frame.landmarks.hands.size());
        for (std::size_t i = 0; i < frame.gestures.size(); ++i) {
            check_true(frame.landmarks.hands[i].world_landmarks.has_value());
            float sum = 0;
            for (float score : frame.gestures[i].scores) { check_true(std::isfinite(score)); sum += score; }
            check_true(std::abs(sum-1) < 0.001F);
        }
        check_throws_as(recognizer->infer({}), hand_models::HandModelError);
        check_true(recognizer->infer(image).gestures.size() == frame.gestures.size());
    }
}
