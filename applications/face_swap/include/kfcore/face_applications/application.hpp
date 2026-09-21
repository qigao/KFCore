#pragma once

#include "kfcore/face_models/types.hpp"
#include "kfcore/image_processor/types.hpp"
#include "kfcore/runtime/resolver.hpp"
#include "kfcore/runtime/runtime.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace kfcore::face_applications
{

enum class FaceApplicationErrorCode
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

class FaceApplicationError final : public std::runtime_error
{
public:
    FaceApplicationError(FaceApplicationErrorCode code, std::string message);
    [[nodiscard]] FaceApplicationErrorCode code() const noexcept;

private:
    FaceApplicationErrorCode code_;
};

struct FaceApplicationModelPackages
{
    std::filesystem::path detector;
    std::filesystem::path face68;
    std::filesystem::path arcface;
    std::filesystem::path inswapper;
    std::filesystem::path inswapper_matrix;
    std::optional<std::filesystem::path> gfpgan;
    std::optional<std::filesystem::path> age_gender;
};

class FaceApplicationPolicies final
{
public:
    /** Construct fully explicit per-model policies. No fallback is added. */
    FaceApplicationPolicies(runtime::ExecutionPolicy detector_policy,
                            runtime::ExecutionPolicy face68_policy,
                            runtime::ExecutionPolicy arcface_policy,
                            runtime::ExecutionPolicy inswapper_policy,
                            runtime::ExecutionPolicy gfpgan_policy,
                            runtime::ExecutionPolicy age_gender_policy);

    /** Route every configured face model to exact `onnxruntime/cuda:N`. */
    [[nodiscard]] static FaceApplicationPolicies onnx_cuda(std::uint32_t device_ordinal);

    runtime::ExecutionPolicy detector;
    runtime::ExecutionPolicy face68;
    runtime::ExecutionPolicy arcface;
    runtime::ExecutionPolicy inswapper;
    runtime::ExecutionPolicy gfpgan;
    runtime::ExecutionPolicy age_gender;
};

struct FaceSwapOptions
{
    float detector_score_threshold = 0.5F;
    int face_class_id = 0;
    float enhancer_blend = 0.8F;
    std::size_t max_image_bytes = 64U * 1024U * 1024U;
    std::size_t max_tensor_bytes = 256U * 1024U * 1024U;
    std::size_t max_output_bytes = 256U * 1024U * 1024U;
};

struct Point2f
{
    float x = 0.0F;
    float y = 0.0F;
};

struct FaceBox
{
    float left = 0.0F;
    float top = 0.0F;
    float right = 0.0F;
    float bottom = 0.0F;
};

using FiveLandmarks = std::array<Point2f, 5>;

struct FaceDetection
{
    FaceBox box;
    float score = 0.0F;
    int class_id = 0;
};

struct FaceAnalysis
{
    FaceDetection detection;
    face_models::Face68Result landmarks68;
    FiveLandmarks landmarks;
    face_models::ArcFaceResult embedding;
    std::optional<face_models::AgeGenderResult> age_gender_logits;
};

using FaceSwapDuration = std::chrono::nanoseconds;

struct FaceAnalysisTimingReport
{
    FaceSwapDuration initial_staging{};
    FaceSwapDuration detection{};
    FaceSwapDuration face68_preprocess{};
    FaceSwapDuration face68_inference_and_postprocess{};
    FaceSwapDuration arcface_preprocess{};
    FaceSwapDuration arcface_inference{};
    std::optional<FaceSwapDuration> age_gender;
    FaceSwapDuration total{};
};

struct FaceSwapTimingReport
{
    FaceAnalysisTimingReport source_analysis;
    FaceAnalysisTimingReport target_analysis;
    FaceSwapDuration embedding_projection{};
    FaceSwapDuration inswapper_preprocess{};
    FaceSwapDuration inswapper_inference_and_decode{};
    FaceSwapDuration inswapper_composition{};
    std::optional<FaceSwapDuration> gfpgan_preprocess;
    std::optional<FaceSwapDuration> gfpgan_inference_and_decode;
    std::optional<FaceSwapDuration> gfpgan_composition;
    FaceSwapDuration total{};
};

struct ProfiledFaceSwapResult
{
    image::BgrImage image;
    FaceSwapTimingReport timings;
};

struct FaceApplicationExecutionRoutes
{
    runtime::ExecutionRoute detector;
    runtime::ExecutionRoute face68;
    runtime::ExecutionRoute arcface;
    runtime::ExecutionRoute inswapper;
    std::optional<runtime::ExecutionRoute> gfpgan;
    std::optional<runtime::ExecutionRoute> age_gender;
};

class FaceSwapApplication final
{
public:
    ~FaceSwapApplication();
    FaceSwapApplication(const FaceSwapApplication&) = delete;
    FaceSwapApplication& operator=(const FaceSwapApplication&) = delete;

    /**
     * Load all configured model packages transactionally.
     *
     * @param runtime Runtime containing every backend named by `policies`.
     *        It must remain valid only until this call returns.
     * @param packages Required package directories, optional package
     *        directories, and the InSwapper projection matrix.
     * @param policies Exact execution policy for every possible model stage.
     * @param options Image, tensor, output, threshold, and blend limits.
     * @return A synchronous, non-reentrant application owning all model
     *         sessions.
     * @throws FaceApplicationError on invalid configuration, package,
     *         execution route, model contract, or resource limit.
     */
    [[nodiscard]] static std::unique_ptr<FaceSwapApplication>
    load(runtime::Runtime& runtime,
         const FaceApplicationModelPackages& packages,
         const FaceApplicationPolicies& policies,
         const FaceSwapOptions& options = {});

    [[nodiscard]] FaceAnalysis analyze(const image::BgrImage& image);
    [[nodiscard]] FaceAnalysis analyze(const image::ImageView& image);
    [[nodiscard]] std::vector<FaceAnalysis> analyze_all(const image::BgrImage& image);
    [[nodiscard]] std::vector<FaceAnalysis> analyze_all(const image::ImageView& image);
    [[nodiscard]] image::BgrImage swap(const image::BgrImage& source,
                                       const image::BgrImage& target);
    [[nodiscard]] image::BgrImage swap(const image::ImageView& source,
                                       const image::ImageView& target);
    [[nodiscard]] ProfiledFaceSwapResult swap_profiled(const image::BgrImage& source,
                                                        const image::BgrImage& target);
    [[nodiscard]] ProfiledFaceSwapResult swap_profiled(const image::ImageView& source,
                                                        const image::ImageView& target);
    [[nodiscard]] image::BgrImage swap_prepared(
        const image::BgrImage& target,
        const face_models::ArcFaceResult& source_embedding,
        const FiveLandmarks& target_landmarks, bool enhance);
    [[nodiscard]] image::BgrImage swap_prepared(
        const image::BgrImage& target,
        const face_models::ArcFaceResult& source_embedding,
        const FiveLandmarks& target_landmarks, bool enhance,
        float enhancer_blend);
    [[nodiscard]] image::BgrImage swap_prepared(
        const image::ImageView& target,
        const face_models::ArcFaceResult& source_embedding,
        const FiveLandmarks& target_landmarks, bool enhance);
    [[nodiscard]] image::BgrImage swap_prepared(
        const image::ImageView& target,
        const face_models::ArcFaceResult& source_embedding,
        const FiveLandmarks& target_landmarks, bool enhance,
        float enhancer_blend);
    /** Return the routes actually selected during load for diagnostics/tests. */
    [[nodiscard]] FaceApplicationExecutionRoutes execution_routes() const;

private:
    struct Impl;
    explicit FaceSwapApplication(std::unique_ptr<Impl> impl);
    [[nodiscard]] image::BgrImage swap_internal(
        const image::BgrImage& source, const image::BgrImage& target,
        FaceSwapTimingReport* timings);
    [[nodiscard]] image::BgrImage swap_prepared_internal(
        const image::BgrImage& target,
        const face_models::ArcFaceResult& source_embedding,
        const FiveLandmarks& target_landmarks, bool enhance,
        float enhancer_blend, FaceSwapTimingReport* timings);

    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::face_applications
