#pragma once

#include "kfcore/face_models/error.hpp"
#include "kfcore/face_models/types.hpp"
#include "kfcore/image_processor/types.hpp"
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

struct PreparedFaceModelOptions
{
    std::size_t max_tensor_bytes = 256U * 1024U * 1024U;
    std::size_t max_output_bytes = 256U * 1024U * 1024U;
};

struct PreparedTensorView
{
    const float* data = nullptr;
    std::size_t element_count = 0U;
};

class FaceDetector final
{
public:
    ~FaceDetector();
    FaceDetector(const FaceDetector&) = delete;
    FaceDetector& operator=(const FaceDetector&) = delete;

    [[nodiscard]] static std::unique_ptr<FaceDetector>
    load(runtime::Runtime& runtime,
         const runtime::ModelPackage& package,
         const runtime::ExecutionPolicy& policy,
         const FaceRuntimeOptions& options = {});

    [[nodiscard]] FaceDetectionsResult infer_all(const image::ImageView& image);
    [[nodiscard]] FaceDetectionResult infer(const image::ImageView& image);
    [[nodiscard]] const runtime::ExecutionRoute& execution_route() const noexcept;

private:
    struct Impl;
    explicit FaceDetector(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class FaceLandmarker final
{
public:
    ~FaceLandmarker();
    FaceLandmarker(const FaceLandmarker&) = delete;
    FaceLandmarker& operator=(const FaceLandmarker&) = delete;

    [[nodiscard]] static std::unique_ptr<FaceLandmarker>
    load(runtime::Runtime& runtime,
         const runtime::ModelPackage& package,
         const runtime::ExecutionPolicy& policy,
         const FaceRuntimeOptions& options = {});

    [[nodiscard]] FaceLandmarkResult infer(const image::ImageView& image,
                                           const RectF& face_box);
    [[nodiscard]] const runtime::ExecutionRoute& execution_route() const noexcept;

private:
    struct Impl;
    explicit FaceLandmarker(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class FaceMesh final
{
public:
    ~FaceMesh();
    FaceMesh(const FaceMesh&) = delete;
    FaceMesh& operator=(const FaceMesh&) = delete;

    [[nodiscard]] static std::unique_ptr<FaceMesh>
    load(runtime::Runtime& runtime,
         const runtime::ModelPackage& detector_package,
         const runtime::ExecutionPolicy& detector_policy,
         const runtime::ModelPackage& landmarker_package,
         const runtime::ExecutionPolicy& landmarker_policy,
         const FaceRuntimeOptions& runtime_options = {},
         const FaceMeshOptions& options = {});

    [[nodiscard]] FaceMeshFrame infer(const image::ImageView& image);
    [[nodiscard]] FaceMeshFrame infer(const image::FrameView& frame);
    [[nodiscard]] const runtime::ExecutionRoute& detector_execution_route() const noexcept;
    [[nodiscard]] const runtime::ExecutionRoute& landmarker_execution_route() const noexcept;

private:
    struct Impl;
    explicit FaceMesh(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class Face68 final
{
public:
    ~Face68();
    Face68(const Face68&) = delete;
    Face68& operator=(const Face68&) = delete;

    [[nodiscard]] static std::unique_ptr<Face68>
    load(runtime::Runtime& runtime,
         const runtime::ModelPackage& package,
         const runtime::ExecutionPolicy& policy,
         const PreparedFaceModelOptions& options = {});
    [[nodiscard]] Face68Result infer(const PreparedTensorView& prepared_input);
    [[nodiscard]] const runtime::ExecutionRoute& execution_route() const noexcept;

private:
    struct Impl;
    explicit Face68(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class ArcFace final
{
public:
    ~ArcFace();
    ArcFace(const ArcFace&) = delete;
    ArcFace& operator=(const ArcFace&) = delete;

    [[nodiscard]] static std::unique_ptr<ArcFace>
    load(runtime::Runtime& runtime,
         const runtime::ModelPackage& package,
         const runtime::ExecutionPolicy& policy,
         const PreparedFaceModelOptions& options = {});
    [[nodiscard]] ArcFaceResult infer(const PreparedTensorView& prepared_input);
    [[nodiscard]] const runtime::ExecutionRoute& execution_route() const noexcept;

private:
    struct Impl;
    explicit ArcFace(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class AgeGender final
{
public:
    ~AgeGender();
    AgeGender(const AgeGender&) = delete;
    AgeGender& operator=(const AgeGender&) = delete;

    [[nodiscard]] static std::unique_ptr<AgeGender>
    load(runtime::Runtime& runtime,
         const runtime::ModelPackage& package,
         const runtime::ExecutionPolicy& policy,
         const PreparedFaceModelOptions& options = {});
    [[nodiscard]] AgeGenderResult infer(const PreparedTensorView& prepared_input);
    [[nodiscard]] const runtime::ExecutionRoute& execution_route() const noexcept;

private:
    struct Impl;
    explicit AgeGender(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class InSwapper final
{
public:
    ~InSwapper();
    InSwapper(const InSwapper&) = delete;
    InSwapper& operator=(const InSwapper&) = delete;

    [[nodiscard]] static std::unique_ptr<InSwapper>
    load(runtime::Runtime& runtime,
         const runtime::ModelPackage& package,
         const runtime::ExecutionPolicy& policy,
         const PreparedFaceModelOptions& options = {});
    [[nodiscard]] InSwapperResult infer(const PreparedTensorView& prepared_target,
                                        const PreparedTensorView& projected_source);
    [[nodiscard]] const runtime::ExecutionRoute& execution_route() const noexcept;

private:
    struct Impl;
    explicit InSwapper(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class GfpGan final
{
public:
    ~GfpGan();
    GfpGan(const GfpGan&) = delete;
    GfpGan& operator=(const GfpGan&) = delete;

    [[nodiscard]] static std::unique_ptr<GfpGan>
    load(runtime::Runtime& runtime,
         const runtime::ModelPackage& package,
         const runtime::ExecutionPolicy& policy,
         const PreparedFaceModelOptions& options = {});
    [[nodiscard]] GfpGanResult infer(const PreparedTensorView& prepared_input);
    [[nodiscard]] const runtime::ExecutionRoute& execution_route() const noexcept;

private:
    struct Impl;
    explicit GfpGan(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::face_models
