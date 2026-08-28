#pragma once

#include "kfcore/vision_models/error.hpp"
#include "kfcore/vision_models/types.hpp"

#include <memory>

namespace kfcore::vision_models
{

class HandInferenceBackend
{
public:
    virtual ~HandInferenceBackend() = default;

    virtual HandFrame infer(const image::ImageView& image) = 0;
};

class FaceDetectorBackend
{
public:
    virtual ~FaceDetectorBackend() = default;

    // The image is borrowed for this synchronous call. Implementations return
    // an owned best-face result or an empty result when no face passes policy.
    virtual FaceDetectionResult infer(const image::ImageView& image) = 0;
};

class FaceLandmarkBackend
{
public:
    virtual ~FaceLandmarkBackend() = default;

    // The image is borrowed for this synchronous call. face_box uses source
    // image coordinates and must have finite positive dimensions.
    virtual FaceLandmarkResult infer(const image::ImageView& image,
                                     const RectF& face_box) = 0;
};

class HandPipeline final
{
public:
    ~HandPipeline();

    HandPipeline(const HandPipeline&)            = delete;
    HandPipeline& operator=(const HandPipeline&) = delete;

    [[nodiscard]] static std::unique_ptr<HandPipeline>
    create(std::unique_ptr<HandInferenceBackend> backend,
           const HandPipelineOptions& options = {});

    HandFrame process(const image::ImageView& image);
    void      reset();

private:
    struct Impl;
    explicit HandPipeline(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class FaceMeshPipeline final
{
public:
    ~FaceMeshPipeline();

    FaceMeshPipeline(const FaceMeshPipeline&)            = delete;
    FaceMeshPipeline& operator=(const FaceMeshPipeline&) = delete;

    // Takes exclusive ownership of both synchronous backends. Null backends or
    // an invalid landmark threshold raise VisionModelError(InvalidArgument).
    [[nodiscard]] static std::unique_ptr<FaceMeshPipeline>
    create(std::unique_ptr<FaceDetectorBackend> detector,
           std::unique_ptr<FaceLandmarkBackend> landmarker,
           const FaceMeshPipelineOptions& options = {});

    // Returns owned detections, landmarks, and timings. Calls on one instance
    // must not overlap; invalid input and backend contract violations raise
    // VisionModelError with the corresponding error code.
    FaceMeshFrame process(const image::ImageView& image);

private:
    struct Impl;
    explicit FaceMeshPipeline(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::vision_models
