#pragma once

#include "kfcore/hand_models/core.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>

namespace kfcore::hand_models
{

struct HandTensorRtEnginePaths
{
    std::filesystem::path palm;
    std::filesystem::path hand_landmark;
    std::filesystem::path keypoint_classifier;
};

struct TensorRtHandOptions
{
    int         device_id                   = 0;
    std::size_t max_engine_bytes            = 256U * 1024U * 1024U;
    std::size_t max_source_bytes            = 64U * 1024U * 1024U;
    std::size_t max_tensor_bytes            = 64U * 1024U * 1024U;
    std::size_t max_output_bytes            = 16U * 1024U * 1024U;
    std::size_t max_palm_candidates         = 2016;
    std::size_t max_hands                   = 8;
    float       palm_score_threshold        = 0.52F;
    float       hand_score_threshold        = 0.50F;
};

class TensorRtHandInput final
{
public:
    ~TensorRtHandInput();

    TensorRtHandInput(const TensorRtHandInput&)            = delete;
    TensorRtHandInput& operator=(const TensorRtHandInput&) = delete;

    [[nodiscard]] static std::unique_ptr<TensorRtHandInput>
    create(const TensorRtHandOptions& options = {});

    // Host input is uploaded once into preparer-owned CUDA storage. CUDA input
    // is borrowed directly. The returned views remain borrowed: source follows
    // the caller's lifetime, and compute must not be used after the next
    // prepare call or this instance's destruction. Calls must not overlap.
    [[nodiscard]] image::FrameView prepare(const image::ImageView& source);

private:
    struct Impl;
    explicit TensorRtHandInput(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class TensorRtHandBackend final : public HandInferenceBackend
{
public:
    ~TensorRtHandBackend() override;

    TensorRtHandBackend(const TensorRtHandBackend&)            = delete;
    TensorRtHandBackend& operator=(const TensorRtHandBackend&) = delete;

    [[nodiscard]] static std::unique_ptr<TensorRtHandBackend>
    load(const HandTensorRtEnginePaths& paths,
         const TensorRtHandOptions& options = {});

    HandFrame infer(const image::ImageView& image) override;

private:
    struct Impl;
    explicit TensorRtHandBackend(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::hand_models
