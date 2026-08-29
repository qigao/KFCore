#pragma once

#include "kfcore/image_processor/types.hpp"
#include "kfcore/face_models/types.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace kfcore::face_applications
{

enum class CpuFaceApplicationErrorCode
{
    InvalidArgument,
    NoFaceDetected,
    InvalidLandmarks,
    InvalidModelAsset,
    ModelContractMismatch,
    ResourceLimitExceeded,
    RuntimeFailure,
    ConcurrentExecution,
};

class CpuFaceApplicationError final : public std::runtime_error
{
public:
    CpuFaceApplicationError(CpuFaceApplicationErrorCode code, std::string message);
    [[nodiscard]] CpuFaceApplicationErrorCode code() const noexcept;

private:
    CpuFaceApplicationErrorCode code_;
};

struct CpuFaceApplicationModelPaths
{
    std::filesystem::path detector_model;
    std::filesystem::path face68_model;
    std::filesystem::path arcface_model;
    std::filesystem::path inswapper_model;
    std::filesystem::path inswapper_matrix;
    std::optional<std::filesystem::path> gfpgan_model;
    std::optional<std::filesystem::path> age_gender_model;
};

struct CpuFaceSwapOptions
{
    float       detector_score_threshold = 0.5F;
    int         face_class_id             = 0;
    float       enhancer_blend            = 0.8F;
    int         intra_op_threads           = 0;
    int         inter_op_threads           = 0;
    std::size_t max_image_bytes            = 64U * 1024U * 1024U;
    std::size_t max_tensor_bytes           = 32U * 1024U * 1024U;
    std::size_t max_model_bytes            = 1024U * 1024U * 1024U;
};

struct CpuPoint2f
{
    float x = 0.0F;
    float y = 0.0F;
};

struct CpuFaceBox
{
    float left   = 0.0F;
    float top    = 0.0F;
    float right  = 0.0F;
    float bottom = 0.0F;
};

using CpuFiveLandmarks = std::array<CpuPoint2f, 5>;

struct CpuFaceDetection
{
    CpuFaceBox box;
    float      score    = 0.0F;
    int        class_id = 0;
};

struct CpuFaceAnalysis
{
    CpuFaceDetection detection;
    kfcore::face_models::Face68Result landmarks68;
    CpuFiveLandmarks landmarks;
    kfcore::face_models::ArcFaceResult embedding;
    std::optional<kfcore::face_models::AgeGenderResult> age_gender_logits;
};

using CpuFaceSwapDuration = std::chrono::nanoseconds;

struct CpuFaceAnalysisTimingReport
{
    CpuFaceSwapDuration initial_staging {};
    CpuFaceSwapDuration detection {};
    CpuFaceSwapDuration face68_preprocess {};
    CpuFaceSwapDuration face68_inference_and_postprocess {};
    CpuFaceSwapDuration arcface_preprocess {};
    CpuFaceSwapDuration arcface_inference {};
    std::optional<CpuFaceSwapDuration> age_gender;
    CpuFaceSwapDuration total {};
};

struct CpuFaceSwapTimingReport
{
    CpuFaceAnalysisTimingReport source_analysis;
    CpuFaceAnalysisTimingReport target_analysis;
    CpuFaceSwapDuration embedding_projection {};
    CpuFaceSwapDuration inswapper_preprocess {};
    CpuFaceSwapDuration inswapper_inference_and_decode {};
    CpuFaceSwapDuration inswapper_composition {};
    std::optional<CpuFaceSwapDuration> gfpgan_preprocess;
    std::optional<CpuFaceSwapDuration> gfpgan_inference_and_decode;
    std::optional<CpuFaceSwapDuration> gfpgan_composition;
    CpuFaceSwapDuration total {};
};

struct ProfiledCpuFaceSwapResult
{
    kfcore::image::BgrImage image;
    CpuFaceSwapTimingReport timings;
};

class OnnxFaceSwapApplication final
{
public:
    ~OnnxFaceSwapApplication();

    OnnxFaceSwapApplication(const OnnxFaceSwapApplication&)            = delete;
    OnnxFaceSwapApplication& operator=(const OnnxFaceSwapApplication&) = delete;

    [[nodiscard]] static std::unique_ptr<OnnxFaceSwapApplication>
    load(const CpuFaceApplicationModelPaths& paths,
         const CpuFaceSwapOptions& options = {});

    // Calls are synchronous and non-reentrant. Inputs are borrowed and never mutated.
    [[nodiscard]] CpuFaceAnalysis analyze(const kfcore::image::BgrImage& image);
    [[nodiscard]] CpuFaceAnalysis analyze(const kfcore::image::ImageView& image);
    [[nodiscard]] kfcore::image::BgrImage swap(
        const kfcore::image::BgrImage& source,
        const kfcore::image::BgrImage& target);
    [[nodiscard]] kfcore::image::BgrImage swap(
        const kfcore::image::ImageView& source,
        const kfcore::image::ImageView& target);
    [[nodiscard]] ProfiledCpuFaceSwapResult swap_profiled(
        const kfcore::image::BgrImage& source,
        const kfcore::image::BgrImage& target);
    [[nodiscard]] ProfiledCpuFaceSwapResult swap_profiled(
        const kfcore::image::ImageView& source,
        const kfcore::image::ImageView& target);

private:
    struct Impl;
    explicit OnnxFaceSwapApplication(std::unique_ptr<Impl> impl);
    [[nodiscard]] kfcore::image::BgrImage swap_internal(
        const kfcore::image::BgrImage& source,
        const kfcore::image::BgrImage& target,
        CpuFaceSwapTimingReport* timings);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::face_applications
