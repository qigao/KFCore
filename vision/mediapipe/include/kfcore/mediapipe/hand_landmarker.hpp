#pragma once

#include "kfcore/hand_models/runtime.hpp"

namespace kfcore::mediapipe
{

inline constexpr const char* kPalmFlavor = "mediapipe-palm-postprocess-v1";
inline constexpr const char* kLandmarkFlavor = "mediapipe-hand-landmark-v1";

/** Adapter for the documented 192/224 MediaPipe-derived ONNX exports, not .task.
 * Owns model sessions and scratch buffers, not the borrowed input image.
 * infer() returns an owned HandFrame in source-pixel coordinates; z is relative,
 * not world-space meters. No tracking, gesture classification or ESN is run.
 * Concurrent infer() is rejected; callers must serialize destruction with use.
 */
class HandLandmarker final
{
public:
    /** Both packages must contain only ONNX artifacts of the corresponding
     * flavor. policy must select exactly one onnxruntime device (no fallback).
     * Throws HandModelError for invalid policy/package contracts or inference
     * failures. Package parsing and backend loading remain Runtime operations.
     */
    [[nodiscard]] static std::unique_ptr<HandLandmarker> load(
        runtime::Runtime& runtime,
        const runtime::ModelPackage& palm,
        const runtime::ModelPackage& landmark,
        const runtime::ExecutionPolicy& policy,
        const hand_models::HandRuntimeOptions& options = {});

    HandLandmarker(const HandLandmarker&) = delete;
    HandLandmarker& operator=(const HandLandmarker&) = delete;
    [[nodiscard]] hand_models::HandFrame infer(const image::ImageView& image);
    [[nodiscard]] const runtime::ExecutionRoute& palm_execution_route() const noexcept;
    [[nodiscard]] const runtime::ExecutionRoute& landmark_execution_route() const noexcept;

private:
    explicit HandLandmarker(std::unique_ptr<hand_models::HandDetector> detector);
    std::unique_ptr<hand_models::HandDetector> detector_;
};

} // namespace kfcore::mediapipe
