#pragma once

#include "kfcore/face_applications/geometry.hpp"
#include "kfcore/face_models/inswapper_embedding.hpp"
#include "kfcore/face_models/tensorrt.hpp"
#include "kfcore/yolo/tensorrt.hpp"

#include <opencv2/core.hpp>

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

    // Calls are synchronous and non-reentrant. Input images are borrowed and never mutated.
    [[nodiscard]] FaceAnalysis analyze(const cv::Mat& bgr_image);
    [[nodiscard]] cv::Mat swap(const cv::Mat& source_bgr, const cv::Mat& target_bgr);

private:
    struct Impl;
    explicit TensorRtFaceSwapApplication(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::face_applications
