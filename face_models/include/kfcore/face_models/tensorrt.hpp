#pragma once

#include "kfcore/face_models/error.hpp"
#include "kfcore/face_models/types.hpp"
#include "kfcore/tensorrt/runtime.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace kfcore::face_models
{

struct Face68Options
{
    std::string                     input_name            = "input";
    std::string                     landmark_output_name  = "landmarks_xyscore";
    std::string                     heatmap_output_name   = "heatmaps";
    std::size_t                     max_batch             = 1;
    kfcore::tensorrt::EngineOptions engine;
};

struct ArcFaceOptions
{
    std::string                     input_name  = "input.1";
    std::string                     output_name = "683";
    std::size_t                     max_batch   = 1;
    kfcore::tensorrt::EngineOptions engine;
};

struct AgeGenderOptions
{
    std::string                     input_name  = "pixel_values";
    std::string                     output_name = "logits";
    std::size_t                     max_batch   = 1;
    kfcore::tensorrt::EngineOptions engine;
};

struct InSwapperOptions
{
    InSwapperOptions()
    {
        engine.max_serialized_engine_bytes = kLargeFaceModelMaxSerializedEngineBytes;
    }

    std::string                     target_input_name = "target";
    std::string                     source_input_name = "source";
    std::string                     output_name       = "output";
    std::size_t                     max_batch         = 1;
    kfcore::tensorrt::EngineOptions engine;
};

struct GfpGanOptions
{
    GfpGanOptions()
    {
        engine.max_serialized_engine_bytes = kLargeFaceModelMaxSerializedEngineBytes;
    }

    std::string                     input_name  = "input";
    std::string                     output_name = "output";
    std::size_t                     max_batch   = 1;
    kfcore::tensorrt::EngineOptions engine;
};

class TensorRtFace68 final
{
public:
    ~TensorRtFace68();

    TensorRtFace68(const TensorRtFace68&)            = delete;
    TensorRtFace68& operator=(const TensorRtFace68&) = delete;

    [[nodiscard]] static std::unique_ptr<TensorRtFace68>
    load(const std::filesystem::path& engine_path, const Face68Options& options = {});

    // The prepared NCHW FP32 view may be host or CUDA storage and is borrowed only for this call.
    // One adapter instance does not support overlapping calls. The returned vector owns its data.
    [[nodiscard]] std::vector<Face68Result>
    infer(const kfcore::tensorrt::TensorView& prepared_input);

private:
    struct Impl;
    explicit TensorRtFace68(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class TensorRtArcFace final
{
public:
    ~TensorRtArcFace();

    TensorRtArcFace(const TensorRtArcFace&)            = delete;
    TensorRtArcFace& operator=(const TensorRtArcFace&) = delete;

    [[nodiscard]] static std::unique_ptr<TensorRtArcFace>
    load(const std::filesystem::path& engine_path, const ArcFaceOptions& options = {});

    // Returns raw embeddings without implicit L2 normalization.
    [[nodiscard]] std::vector<ArcFaceResult>
    infer(const kfcore::tensorrt::TensorView& prepared_input);

private:
    struct Impl;
    explicit TensorRtArcFace(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class TensorRtAgeGender final
{
public:
    ~TensorRtAgeGender();

    TensorRtAgeGender(const TensorRtAgeGender&)            = delete;
    TensorRtAgeGender& operator=(const TensorRtAgeGender&) = delete;

    [[nodiscard]] static std::unique_ptr<TensorRtAgeGender>
    load(const std::filesystem::path& engine_path, const AgeGenderOptions& options = {});

    // Returns both logits unchanged; their semantic order is intentionally not named.
    [[nodiscard]] std::vector<AgeGenderResult>
    infer(const kfcore::tensorrt::TensorView& prepared_input);

private:
    struct Impl;
    explicit TensorRtAgeGender(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class TensorRtInSwapper final
{
public:
    ~TensorRtInSwapper();

    TensorRtInSwapper(const TensorRtInSwapper&)            = delete;
    TensorRtInSwapper& operator=(const TensorRtInSwapper&) = delete;

    [[nodiscard]] static std::unique_ptr<TensorRtInSwapper>
    load(const std::filesystem::path& engine_path, const InSwapperOptions& options = {});

    // Both FP32 views are borrowed only for this synchronous call. Batch is fixed at one.
    [[nodiscard]] InSwapperResult
    infer(const kfcore::tensorrt::TensorView& prepared_target,
          const kfcore::tensorrt::TensorView& projected_source);

private:
    struct Impl;
    explicit TensorRtInSwapper(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class TensorRtGfpGan final
{
public:
    ~TensorRtGfpGan();

    TensorRtGfpGan(const TensorRtGfpGan&)            = delete;
    TensorRtGfpGan& operator=(const TensorRtGfpGan&) = delete;

    [[nodiscard]] static std::unique_ptr<TensorRtGfpGan>
    load(const std::filesystem::path& engine_path, const GfpGanOptions& options = {});

    // The prepared FP32 view is borrowed only for this synchronous call. Batch is fixed at one.
    [[nodiscard]] GfpGanResult infer(const kfcore::tensorrt::TensorView& prepared_input);

private:
    struct Impl;
    explicit TensorRtGfpGan(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::face_models
