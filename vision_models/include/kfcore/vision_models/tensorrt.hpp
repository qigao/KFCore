#pragma once

#include "kfcore/vision_models/core.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>

namespace kfcore::vision_models
{

struct HandTensorRtEnginePaths
{
    std::filesystem::path palm;
    std::filesystem::path hand_landmark;
    std::filesystem::path keypoint_classifier;
};

struct TensorRtVisionOptions
{
    int         device_id                   = 0;
    std::size_t max_engine_bytes            = 256U * 1024U * 1024U;
    std::size_t max_source_bytes            = 64U * 1024U * 1024U;
    std::size_t max_tensor_bytes            = 64U * 1024U * 1024U;
    std::size_t max_output_bytes            = 16U * 1024U * 1024U;
    std::size_t max_palm_candidates         = 2016;
    std::size_t max_hands                   = 8;
    std::size_t max_face_detections         = 1000;
    float       palm_score_threshold        = 0.52F;
    float       hand_score_threshold        = 0.50F;
    float       face_detection_score_threshold = 0.50F;
    bool        face_coordinates_normalized = true;
};

class TensorRtVisionInput final
{
public:
    ~TensorRtVisionInput();

    TensorRtVisionInput(const TensorRtVisionInput&)            = delete;
    TensorRtVisionInput& operator=(const TensorRtVisionInput&) = delete;

    [[nodiscard]] static std::unique_ptr<TensorRtVisionInput>
    create(const TensorRtVisionOptions& options = {});

    // Host input is uploaded once into preparer-owned CUDA storage. CUDA input
    // is borrowed directly. The returned views remain borrowed: source follows
    // the caller's lifetime, and compute must not be used after the next
    // prepare call or this instance's destruction. Calls must not overlap.
    [[nodiscard]] VisionFrameView prepare(const image::ImageView& source);

private:
    struct Impl;
    explicit TensorRtVisionInput(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class TensorRtHandBackend final : public HandInferenceBackend
{
public:
    ~TensorRtHandBackend() override;

    TensorRtHandBackend(const TensorRtHandBackend&)            = delete;
    TensorRtHandBackend& operator=(const TensorRtHandBackend&) = delete;

    [[nodiscard]] static std::unique_ptr<TensorRtHandBackend>
    load(const HandTensorRtEnginePaths& paths,
         const TensorRtVisionOptions& options = {});

    HandFrame infer(const image::ImageView& image) override;

private:
    struct Impl;
    explicit TensorRtHandBackend(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class TensorRtFaceDetector final : public FaceDetectorBackend
{
public:
    ~TensorRtFaceDetector() override;

    TensorRtFaceDetector(const TensorRtFaceDetector&)            = delete;
    TensorRtFaceDetector& operator=(const TensorRtFaceDetector&) = delete;

    [[nodiscard]] static std::unique_ptr<TensorRtFaceDetector>
    load(const std::filesystem::path& engine_path,
         const TensorRtVisionOptions& options = {});

    FaceDetectionResult infer(const image::ImageView& image) override;

private:
    struct Impl;
    explicit TensorRtFaceDetector(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class TensorRtFaceLandmarker final : public FaceLandmarkBackend
{
public:
    ~TensorRtFaceLandmarker() override;

    TensorRtFaceLandmarker(const TensorRtFaceLandmarker&)            = delete;
    TensorRtFaceLandmarker& operator=(const TensorRtFaceLandmarker&) = delete;

    [[nodiscard]] static std::unique_ptr<TensorRtFaceLandmarker>
    load(const std::filesystem::path& engine_path,
         const TensorRtVisionOptions& options = {});

    FaceLandmarkResult infer(const image::ImageView& image,
                             const RectF& face_box) override;

private:
    struct Impl;
    explicit TensorRtFaceLandmarker(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::vision_models
