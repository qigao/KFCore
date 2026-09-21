#include "kfcore/mediapipe/hand_landmarker.hpp"
#include "tinytest.hpp"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace
{
using namespace kfcore;
struct Packages
{
    char* directory = tt_make_temp_dir("mediapipe-contract");
    Packages() { if (!directory) throw std::runtime_error("cannot create test directory"); }
    ~Packages() { tt_remove_tree(directory); std::free(directory); }
    Packages(const Packages&) = delete;
    Packages& operator=(const Packages&) = delete;
    runtime::ModelPackage make(const char* type, const char* flavor)
    {
        const auto path = std::filesystem::path(directory) / (std::string(type) + ".json");
        const std::string json = std::string("{\"schema\":\"kfcore.model/1\",\"id\":\"test\",\"version\":\"1\",\"model_type\":\"") + type +
            "\",\"artifacts\":[{\"id\":\"onnx\",\"format\":\"onnx\",\"path\":\"absent.onnx\",\"backend\":\"onnxruntime\",\"device\":\"any\",\"flavor\":\"" + flavor +
            "\",\"sha256\":\"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\"}]}";
        if (tt_write_file(path.string().c_str(), json.data(), json.size()) != 0)
            throw std::runtime_error("cannot write test manifest");
        return runtime::ModelPackage::load(path);
    }
};
}

spec("MediaPipe ONNX boundary")
{
    it("owns sessions without copy semantics")
    {
        check_false(std::is_copy_constructible_v<mediapipe::HandLandmarker>);
        check_false(std::is_copy_assignable_v<mediapipe::HandLandmarker>);
    }
    it("rejects non-ONNX execution before loading models")
    {
        runtime::Runtime runtime;
        const runtime::ModelPackage empty;
        check_throws_with(mediapipe::HandLandmarker::load(runtime, empty, empty,
            runtime::ExecutionPolicy::exact("tensorrt", "cuda:0")), "exact onnxruntime");
    }
    it("rejects ordered fallback policies")
    {
        runtime::Runtime runtime;
        const runtime::ModelPackage empty;
        check_throws_with(mediapipe::HandLandmarker::load(runtime, empty, empty,
            runtime::ExecutionPolicy::ordered({{"onnxruntime", "cuda:0"}, {"onnxruntime", "cpu"}})),
            "exact onnxruntime");
    }
    it("rejects raw palm heads rather than guessing decoding")
    {
        Packages fixtures;
        runtime::Runtime runtime;
        const auto palm = fixtures.make("hand.palm-detector", "raw-palm");
        const auto landmark = fixtures.make("hand.landmarker", mediapipe::kLandmarkFlavor);
        check_throws_with(mediapipe::HandLandmarker::load(runtime, palm, landmark,
            runtime::ExecutionPolicy::exact("onnxruntime", "cpu")), mediapipe::kPalmFlavor);
    }
    it("rejects unqualified landmark exports")
    {
        Packages fixtures;
        runtime::Runtime runtime;
        const auto palm = fixtures.make("hand.palm-detector", mediapipe::kPalmFlavor);
        const auto landmark = fixtures.make("hand.landmarker", "unspecified");
        check_throws_with(mediapipe::HandLandmarker::load(runtime, palm, landmark,
            runtime::ExecutionPolicy::exact("onnxruntime", "cpu")), mediapipe::kLandmarkFlavor);
    }
    it("requires canonical model types")
    {
        runtime::Runtime runtime;
        const runtime::ModelPackage empty;
        check_throws_with(mediapipe::HandLandmarker::load(runtime, empty, empty,
            runtime::ExecutionPolicy::exact("onnxruntime", "cpu")), "hand.palm-detector");
    }
    it("preserves strict three-model loading")
    {
        Packages fixtures;
        runtime::Runtime runtime;
        const auto palm = fixtures.make("hand.palm-detector", mediapipe::kPalmFlavor);
        const auto landmark = fixtures.make("hand.landmarker", mediapipe::kLandmarkFlavor);
        const runtime::ModelPackage missing_classifier;
        const auto policy = runtime::ExecutionPolicy::exact("onnxruntime", "cpu");
        check_throws_with(hand_models::HandDetector::load(runtime, palm, policy, landmark,
            policy, missing_classifier, policy), "canonical hand.*");
    }
}
