#pragma once

#include "kfcore/face_models/error.hpp"
#include "kfcore/face_models/types.hpp"
#include "kfcore/image_processor/types.hpp"

#include <memory>

namespace kfcore::face_models
{

class FaceDetectorBackend
{
public:
    virtual ~FaceDetectorBackend() = default;
    virtual FaceDetectionResult infer(const image::ImageView& image) = 0;
};

class FaceLandmarkBackend
{
public:
    virtual ~FaceLandmarkBackend() = default;
    virtual FaceLandmarkResult infer(const image::ImageView& image,
                                     const RectF& face_box) = 0;
};

class FaceMeshPipeline final
{
public:
    ~FaceMeshPipeline();

    FaceMeshPipeline(const FaceMeshPipeline&)            = delete;
    FaceMeshPipeline& operator=(const FaceMeshPipeline&) = delete;

    [[nodiscard]] static std::unique_ptr<FaceMeshPipeline>
    create(std::unique_ptr<FaceDetectorBackend> detector,
           std::unique_ptr<FaceLandmarkBackend> landmarker,
           const FaceMeshPipelineOptions& options = {});

    FaceMeshFrame process(const image::ImageView& image);
    FaceMeshFrame process(const image::FrameView& frame);

private:
    struct Impl;
    explicit FaceMeshPipeline(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::face_models
