#pragma once

#include "kfcore/face_applications/geometry.hpp"
#include "kfcore/face_models/inswapper_embedding.hpp"
#include "kfcore/face_models/tensorrt.hpp"
#include "kfcore/image_processor/types.hpp"
#include "kfcore/yolo/tensorrt.hpp"

#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace kfcore::face_applications
{

struct FaceApplicationModelPaths
{
    std::filesystem::path detector_engine;
    std::filesystem::path face68_engine;
    std::filesystem::path arcface_engine;
    std::filesystem::path inswapper_engine;
    std::filesystem::path inswapper_matrix;
    std::optional<std::filesystem::path> gfpgan_engine;
    std::optional<std::filesystem::path> age_gender_engine;
};

struct FaceSwapOptions
{
    float detector_score_threshold = 0.5F;
    int   face_class_id             = 0;
    float enhancer_blend            = 0.8F;

    kfcore::yolo::EngineOptions              detector_engine;
    kfcore::yolo::DetectorOptions            detector;
    kfcore::face_models::Face68Options       face68;
    kfcore::face_models::ArcFaceOptions      arcface;
    kfcore::face_models::InSwapperOptions    inswapper;
    kfcore::face_models::GfpGanOptions       gfpgan;
    kfcore::face_models::AgeGenderOptions    age_gender;
};

struct FaceAnalysis
{
    kfcore::yolo::Detection detection;
    kfcore::face_models::Face68Result landmarks68;
    FiveLandmarks landmarks;
    kfcore::face_models::ArcFaceResult embedding;
    std::optional<kfcore::face_models::AgeGenderResult> age_gender_logits;
};

using FaceSwapDuration = std::chrono::nanoseconds;

// Synchronous wall time for one analysis call. Inference fields include model
// adapter validation and postprocessing performed before the call returns.
struct FaceAnalysisTimingReport
{
    FaceSwapDuration initial_staging {};
    FaceSwapDuration detection {};
    FaceSwapDuration face68_preprocess {};
    FaceSwapDuration face68_inference_and_postprocess {};
    FaceSwapDuration arcface_preprocess {};
    FaceSwapDuration arcface_inference {};
    std::optional<FaceSwapDuration> age_gender;
    FaceSwapDuration total {};
};

// Timings are synchronous wall time and therefore include waiting for CUDA work
// to complete. Optional durations are absent when that model is not configured.
struct FaceSwapTimingReport
{
    FaceAnalysisTimingReport source_analysis;
    FaceAnalysisTimingReport target_analysis;
    FaceSwapDuration embedding_projection {};
    FaceSwapDuration inswapper_preprocess {};
    FaceSwapDuration inswapper_inference_and_decode {};
    FaceSwapDuration inswapper_composition {};
    std::optional<FaceSwapDuration> gfpgan_preprocess;
    std::optional<FaceSwapDuration> gfpgan_inference_and_decode;
    std::optional<FaceSwapDuration> gfpgan_composition;
    FaceSwapDuration total {};
};

struct ProfiledFaceSwapResult
{
    kfcore::image::BgrImage image;
    FaceSwapTimingReport timings;
};

[[nodiscard]] std::optional<kfcore::yolo::Detection> select_highest_score_face(
    const std::vector<kfcore::yolo::Detection>& detections, int face_class_id,
    float score_threshold);

class TensorRtFaceSwapApplication final
{
public:
    ~TensorRtFaceSwapApplication();

    TensorRtFaceSwapApplication(const TensorRtFaceSwapApplication&)            = delete;
    TensorRtFaceSwapApplication& operator=(const TensorRtFaceSwapApplication&) = delete;

    [[nodiscard]] static std::unique_ptr<TensorRtFaceSwapApplication>
    load(const FaceApplicationModelPaths& paths, const FaceSwapOptions& options = {});
    [[nodiscard]] static std::unique_ptr<TensorRtFaceSwapApplication>
    load(const FaceSwapOptions& options);

    // Calls are synchronous and non-reentrant. BGR8, RGB8, NV12, I420, NV21, YUY2 and UYVY inputs
    // are borrowed and never mutated. Host and same-device CUDA inputs are supported.
    [[nodiscard]] FaceAnalysis analyze(const kfcore::image::ImageView& image);
    [[nodiscard]] std::vector<FaceAnalysis> analyze_all(
        const kfcore::image::ImageView& image);
    [[nodiscard]] kfcore::image::BgrImage swap(
        const kfcore::image::ImageView& source,
        const kfcore::image::ImageView& target);
    [[nodiscard]] ProfiledFaceSwapResult swap_profiled(
        const kfcore::image::ImageView& source,
        const kfcore::image::ImageView& target);
    [[nodiscard]] kfcore::image::BgrImage swap_prepared(
        const kfcore::image::ImageView& target,
        const kfcore::face_models::ArcFaceResult& source_embedding,
        const FiveLandmarks& target_landmarks, bool enhance);
    [[nodiscard]] kfcore::image::BgrImage swap_prepared(
        const kfcore::image::ImageView& target,
        const kfcore::face_models::ArcFaceResult& source_embedding,
        const FiveLandmarks& target_landmarks, bool enhance,
        float enhancer_blend);

private:
    struct Impl;
    explicit TensorRtFaceSwapApplication(std::unique_ptr<Impl> impl);
    [[nodiscard]] kfcore::image::BgrImage swap_internal(
        const kfcore::image::ImageView& source,
        const kfcore::image::ImageView& target,
        FaceSwapTimingReport* timings);
    [[nodiscard]] kfcore::image::BgrImage swap_prepared_internal(
        const kfcore::image::ImageView& target,
        const kfcore::face_models::ArcFaceResult& source_embedding,
        const FiveLandmarks& target_landmarks, bool enhance, float enhancer_blend,
        FaceSwapTimingReport* timings);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::face_applications
