#pragma once

#include "kfcore/face_models/error.hpp"
#include "kfcore/face_models/core.hpp"
#include "kfcore/face_models/types.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <vector>

namespace kfcore::face_models
{

struct CpuTensorView
{
    const float* data          = nullptr;
    std::size_t  element_count = 0U;
};

struct CpuFaceModelOptions
{
    int         intra_op_threads = 0;
    int         inter_op_threads = 0;
    std::size_t max_model_bytes  = kLargeFaceModelMaxAssetBytes;
    std::size_t max_output_bytes = 256U * 1024U * 1024U;
};

struct CpuFaceMeshOptions
{
    int          intra_op_threads                  = 0;
    int          inter_op_threads                  = 0;
    std::size_t  max_model_bytes                   = 256U * 1024U * 1024U;
    std::size_t  max_source_bytes                  = 64U * 1024U * 1024U;
    std::size_t  max_tensor_bytes                  = 64U * 1024U * 1024U;
    std::size_t  max_output_bytes                  = 16U * 1024U * 1024U;
    float        face_detection_score_threshold   = 0.50F;
    std::int32_t face_class_id                     = 0;
    bool         face_coordinates_normalized      = true;
};

class CpuFaceDetector final : public FaceDetectorBackend
{
public:
    ~CpuFaceDetector() override;
    CpuFaceDetector(const CpuFaceDetector&)            = delete;
    CpuFaceDetector& operator=(const CpuFaceDetector&) = delete;

    [[nodiscard]] static std::unique_ptr<CpuFaceDetector>
    load(const std::filesystem::path& model_path,
         const CpuFaceMeshOptions& options = {});
    FaceDetectionResult infer(const image::ImageView& image) override;

private:
    struct Impl;
    explicit CpuFaceDetector(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class CpuFaceLandmarker final : public FaceLandmarkBackend
{
public:
    ~CpuFaceLandmarker() override;
    CpuFaceLandmarker(const CpuFaceLandmarker&)            = delete;
    CpuFaceLandmarker& operator=(const CpuFaceLandmarker&) = delete;

    [[nodiscard]] static std::unique_ptr<CpuFaceLandmarker>
    load(const std::filesystem::path& model_path,
         const CpuFaceMeshOptions& options = {});
    FaceLandmarkResult infer(const image::ImageView& image,
                             const RectF& face_box) override;

private:
    struct Impl;
    explicit CpuFaceLandmarker(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class CpuFace68 final
{
public:
    ~CpuFace68();

    CpuFace68(const CpuFace68&)            = delete;
    CpuFace68& operator=(const CpuFace68&) = delete;

    [[nodiscard]] static std::unique_ptr<CpuFace68>
    load(const std::filesystem::path& model_path,
         const CpuFaceModelOptions& options = {});

    [[nodiscard]] Face68Result infer(const CpuTensorView& prepared_input);

private:
    struct Impl;
    explicit CpuFace68(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class CpuArcFace final
{
public:
    ~CpuArcFace();

    CpuArcFace(const CpuArcFace&)            = delete;
    CpuArcFace& operator=(const CpuArcFace&) = delete;

    [[nodiscard]] static std::unique_ptr<CpuArcFace>
    load(const std::filesystem::path& model_path,
         const CpuFaceModelOptions& options = {});

    [[nodiscard]] ArcFaceResult infer(const CpuTensorView& prepared_input);

private:
    struct Impl;
    explicit CpuArcFace(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class CpuAgeGender final
{
public:
    ~CpuAgeGender();

    CpuAgeGender(const CpuAgeGender&)            = delete;
    CpuAgeGender& operator=(const CpuAgeGender&) = delete;

    [[nodiscard]] static std::unique_ptr<CpuAgeGender>
    load(const std::filesystem::path& model_path,
         const CpuFaceModelOptions& options = {});

    [[nodiscard]] AgeGenderResult infer(const CpuTensorView& prepared_input);

private:
    struct Impl;
    explicit CpuAgeGender(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class CpuInSwapper final
{
public:
    ~CpuInSwapper();

    CpuInSwapper(const CpuInSwapper&)            = delete;
    CpuInSwapper& operator=(const CpuInSwapper&) = delete;

    [[nodiscard]] static std::unique_ptr<CpuInSwapper>
    load(const std::filesystem::path& model_path,
         const CpuFaceModelOptions& options = {});

    [[nodiscard]] InSwapperResult infer(const CpuTensorView& prepared_target,
                                        const CpuTensorView& projected_source);

private:
    struct Impl;
    explicit CpuInSwapper(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class CpuGfpGan final
{
public:
    ~CpuGfpGan();

    CpuGfpGan(const CpuGfpGan&)            = delete;
    CpuGfpGan& operator=(const CpuGfpGan&) = delete;

    [[nodiscard]] static std::unique_ptr<CpuGfpGan>
    load(const std::filesystem::path& model_path,
         const CpuFaceModelOptions& options = {});

    [[nodiscard]] GfpGanResult infer(const CpuTensorView& prepared_input);

private:
    struct Impl;
    explicit CpuGfpGan(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::face_models
