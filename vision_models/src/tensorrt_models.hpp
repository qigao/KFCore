#pragma once

#include "kfcore/image_processor/types.hpp"
#include "kfcore/tensorrt/runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace kfcore::vision_models::detail
{

struct TensorRtModelOptions
{
    int         device_id;
    std::size_t max_engine_bytes;
    std::size_t max_tensor_bytes;
    std::size_t max_output_bytes;
    std::size_t max_palm_candidates;
    std::size_t max_hands;
};

struct HandTensorRtOutputs
{
    std::vector<float> landmarks;
    float              score      = 0.0F;
    float              handedness = 0.0F;
};

struct FaceTensorRtOutputs
{
    float              score = 0.0F;
    std::vector<float> landmarks;
};

class PalmTensorRtModel final
{
public:
    PalmTensorRtModel(const std::filesystem::path& path,
                      const TensorRtModelOptions& options);
    std::vector<float> run(const image::TensorView& input);

private:
    std::shared_ptr<const tensorrt::Engine> engine_;
    std::unique_ptr<tensorrt::Executor>     executor_;
    std::size_t max_output_bytes_ = 0;
};

class HandLandmarkTensorRtModel final
{
public:
    HandLandmarkTensorRtModel(const std::filesystem::path& path,
                              const TensorRtModelOptions& options);
    HandTensorRtOutputs run(const image::TensorView& input);

private:
    std::shared_ptr<const tensorrt::Engine> engine_;
    std::unique_ptr<tensorrt::Executor>     executor_;
};

class KeypointClassifierTensorRtModel final
{
public:
    KeypointClassifierTensorRtModel(const std::filesystem::path& path,
                                    const TensorRtModelOptions& options);
    std::vector<std::int64_t> run(const std::vector<float>& input,
                                  std::size_t batch);

private:
    std::shared_ptr<const tensorrt::Engine> engine_;
    std::unique_ptr<tensorrt::Executor>     executor_;
};

class FaceLandmarkTensorRtModel final
{
public:
    FaceLandmarkTensorRtModel(const std::filesystem::path& path,
                              const TensorRtModelOptions& options);
    FaceTensorRtOutputs run(const image::TensorView& input);

private:
    std::shared_ptr<const tensorrt::Engine> engine_;
    std::unique_ptr<tensorrt::Executor>     executor_;
};

} // namespace kfcore::vision_models::detail
