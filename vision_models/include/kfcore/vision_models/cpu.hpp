#pragma once

#include "kfcore/vision_models/core.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>

namespace kfcore::vision_models
{

struct HandOnnxModelPaths
{
    std::filesystem::path palm;
    std::filesystem::path hand_landmark;
    std::filesystem::path keypoint_classifier;
};

struct CpuVisionOptions
{
    int         intra_op_threads      = 0;
    int         inter_op_threads      = 0;
    std::size_t max_model_bytes       = 256U * 1024U * 1024U;
    std::size_t max_source_bytes      = 64U * 1024U * 1024U;
    std::size_t max_tensor_bytes      = 64U * 1024U * 1024U;
    std::size_t max_output_bytes      = 16U * 1024U * 1024U;
    std::size_t max_palm_candidates   = 2016;
    std::size_t max_hands             = 8;
    float       palm_score_threshold  = 0.52F;
    float       hand_score_threshold  = 0.50F;
    bool        face_coordinates_normalized = true;
};

class CpuHandBackend final : public HandInferenceBackend
{
public:
    ~CpuHandBackend() override;

    CpuHandBackend(const CpuHandBackend&)            = delete;
    CpuHandBackend& operator=(const CpuHandBackend&) = delete;

    [[nodiscard]] static std::unique_ptr<CpuHandBackend>
    load(const HandOnnxModelPaths& paths, const CpuVisionOptions& options = {});

    HandFrame infer(const image::ImageView& image) override;

private:
    struct Impl;
    explicit CpuHandBackend(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class CpuFaceLandmarker final
{
public:
    ~CpuFaceLandmarker();

    CpuFaceLandmarker(const CpuFaceLandmarker&)            = delete;
    CpuFaceLandmarker& operator=(const CpuFaceLandmarker&) = delete;

    [[nodiscard]] static std::unique_ptr<CpuFaceLandmarker>
    load(const std::filesystem::path& model_path,
         const CpuVisionOptions& options = {});

    FaceLandmarkResult infer(const image::ImageView& image, const RectF& face_box);

private:
    struct Impl;
    explicit CpuFaceLandmarker(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::vision_models
