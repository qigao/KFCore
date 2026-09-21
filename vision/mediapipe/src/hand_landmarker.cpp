#include "kfcore/mediapipe/hand_landmarker.hpp"

#include <string>
#include <utility>

namespace kfcore::mediapipe
{
namespace
{
void validate_package(const runtime::ModelPackage& package, const char* type,
                      const char* flavor)
{
    if (package.model_type() != type || package.artifacts().empty())
        throw hand_models::HandModelError(hand_models::HandModelErrorCode::ModelContractMismatch,
                                          std::string("MediaPipe package requires type ") + type);
    for (const auto& artifact : package.artifacts())
        if (artifact.format != "onnx" || artifact.backend != "onnxruntime" ||
            artifact.flavor != flavor)
            throw hand_models::HandModelError(hand_models::HandModelErrorCode::ModelContractMismatch,
                                              std::string("MediaPipe ONNX artifact requires flavor ") + flavor);
}
} // namespace

HandLandmarker::HandLandmarker(std::unique_ptr<hand_models::HandDetector> detector)
    : detector_(std::move(detector)) {}

std::unique_ptr<HandLandmarker> HandLandmarker::load(
    runtime::Runtime& runtime, const runtime::ModelPackage& palm,
    const runtime::ModelPackage& landmark, const runtime::ExecutionPolicy& policy,
    const hand_models::HandRuntimeOptions& options)
{
    const auto& preferences = policy.preferences();
    if (preferences.size() != 1U || preferences.front().backend_id != "onnxruntime")
        throw hand_models::HandModelError(hand_models::HandModelErrorCode::InvalidArgument,
                                          "MediaPipe ONNX requires an exact onnxruntime policy");
    validate_package(palm, "hand.palm-detector", kPalmFlavor);
    validate_package(landmark, "hand.landmarker", kLandmarkFlavor);
    return std::unique_ptr<HandLandmarker>(new HandLandmarker(
        hand_models::HandDetector::load_landmarks(runtime, palm, policy, landmark, policy, options)));
}

hand_models::HandFrame HandLandmarker::infer(const image::ImageView& image)
{
    return detector_->infer(image);
}

const runtime::ExecutionRoute& HandLandmarker::palm_execution_route() const noexcept
{
    return detector_->palm_execution_route();
}

const runtime::ExecutionRoute& HandLandmarker::landmark_execution_route() const noexcept
{
    return detector_->landmark_execution_route();
}
} // namespace kfcore::mediapipe
