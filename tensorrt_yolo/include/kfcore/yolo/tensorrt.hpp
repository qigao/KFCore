#pragma once

#include "kfcore/yolo/error.hpp"
#include "kfcore/yolo/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace kfcore::yolo
{

struct TensorNames
{
    std::string images   = "images";
    std::string num_dets = "num_dets";
    std::string boxes    = "boxes";
    std::string scores   = "scores";
    std::string labels   = "labels";
};

struct EngineOptions
{
    int         device_id = 0;
    TensorNames tensor_names;
    std::size_t max_batch        = 16;
    std::size_t max_detections   = 1000;
    std::size_t max_input_bytes  = 256U * 1024U * 1024U;
    std::size_t max_output_bytes = 16U * 1024U * 1024U;
};

struct DetectorOptions
{
    std::optional<std::array<std::int32_t, 2>> input_size;
    std::array<float, 3>                       mean { 0.0f, 0.0f, 0.0f };
    std::array<float, 3>                       stddev { 1.0f, 1.0f, 1.0f };
    float                                      border_value = 114.0f;
};

class TensorRtDetector;

class Engine final
{
public:
    static std::shared_ptr<const Engine> load(const std::filesystem::path& engine_path,
                                              const EngineOptions&         options = {});

    std::unique_ptr<TensorRtDetector> create_detector(const DetectorOptions& options = {}) const;

private:
    friend class TensorRtDetector;
    struct State;

    explicit Engine(std::shared_ptr<const State> state);

    std::shared_ptr<const State> state_;
};

class TensorRtDetector final
{
public:
    ~TensorRtDetector();

    TensorRtDetector(const TensorRtDetector&)            = delete;
    TensorRtDetector& operator=(const TensorRtDetector&) = delete;

    DetectionFrame              detect(const ImageView& image);
    std::vector<DetectionFrame> detect_batch(const std::vector<ImageView>& images);

private:
    friend class Engine;
    struct Impl;

    explicit TensorRtDetector(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::yolo
