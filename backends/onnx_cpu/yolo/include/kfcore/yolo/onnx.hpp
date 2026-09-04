#pragma once

#include "kfcore/yolo/types.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace kfcore::yolo
{

struct OnnxDetectorOptions
{
    std::string input_name = "images";
    std::string output_name = "output0";
    int         intra_op_threads = 0;
    int         inter_op_threads = 0;
    std::size_t max_model_bytes  = 256U * 1024U * 1024U;
    std::size_t max_source_bytes = 64U * 1024U * 1024U;
    std::size_t max_tensor_bytes = 32U * 1024U * 1024U;
    std::size_t max_output_bytes = 16U * 1024U * 1024U;
    std::size_t max_detections   = 1000U;
    float       border_value     = 114.0F;
    float       score_threshold  = 0.25F;
    float       iou_threshold    = 0.45F;
    bool        mirror_horizontal = false;
};

class OnnxDetector final
{
public:
    [[nodiscard]] static std::unique_ptr<OnnxDetector>
    load(const std::filesystem::path& model_path,
         const OnnxDetectorOptions& options = {});
    [[nodiscard]] static std::unique_ptr<OnnxDetector>
    load_person_detector(const OnnxDetectorOptions& options = {});

    ~OnnxDetector();
    OnnxDetector(OnnxDetector&&) noexcept;
    OnnxDetector& operator=(OnnxDetector&&) noexcept;
    OnnxDetector(const OnnxDetector&)            = delete;
    OnnxDetector& operator=(const OnnxDetector&) = delete;

    [[nodiscard]] DetectionFrame detect(const ImageView& image);
    [[nodiscard]] std::int32_t input_width() const noexcept;
    [[nodiscard]] std::int32_t input_height() const noexcept;
    [[nodiscard]] std::size_t max_detections() const noexcept;

private:
    struct Impl;
    explicit OnnxDetector(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::yolo
