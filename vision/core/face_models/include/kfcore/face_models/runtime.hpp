#pragma once

#include "kfcore/face_models/core.hpp"
#include "kfcore/runtime/model_package.hpp"
#include "kfcore/runtime/resolver.hpp"
#include "kfcore/runtime/runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace kfcore::face_models
{

struct FaceRuntimeOptions
{
    std::int32_t detector_input_extent = 640;
    std::int32_t landmarker_input_extent = kFaceMeshInputExtent;
    std::size_t max_detections = 300U;
    std::size_t max_source_bytes = 64U * 1024U * 1024U;
    std::size_t max_tensor_bytes = 64U * 1024U * 1024U;
    std::size_t max_output_bytes = 64U * 1024U * 1024U;
    float face_detection_score_threshold = 0.50F;
    std::int32_t face_class_id = 0;
    bool face_coordinates_normalized = true;
};

class FaceDetector final : public FaceDetectorBackend
{
public:
    ~FaceDetector() override;

    FaceDetector(const FaceDetector&) = delete;
    FaceDetector& operator=(const FaceDetector&) = delete;

    [[nodiscard]] static std::unique_ptr<FaceDetector>
    load(runtime::Runtime& runtime,
         const runtime::ModelPackage& package,
         const runtime::ExecutionPolicy& policy,
         const FaceRuntimeOptions& options = {});

    [[nodiscard]] FaceDetectionsResult infer_all(const image::ImageView& image);
    FaceDetectionResult infer(const image::ImageView& image) override;

    [[nodiscard]] const runtime::ExecutionRoute& execution_route() const noexcept;

private:
    struct Impl;
    explicit FaceDetector(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class FaceLandmarker final : public FaceLandmarkBackend
{
public:
    ~FaceLandmarker() override;

    FaceLandmarker(const FaceLandmarker&) = delete;
    FaceLandmarker& operator=(const FaceLandmarker&) = delete;

    [[nodiscard]] static std::unique_ptr<FaceLandmarker>
    load(runtime::Runtime& runtime,
         const runtime::ModelPackage& package,
         const runtime::ExecutionPolicy& policy,
         const FaceRuntimeOptions& options = {});

    FaceLandmarkResult infer(const image::ImageView& image,
                             const RectF& face_box) override;

    [[nodiscard]] const runtime::ExecutionRoute& execution_route() const noexcept;

private:
    struct Impl;
    explicit FaceLandmarker(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::face_models
