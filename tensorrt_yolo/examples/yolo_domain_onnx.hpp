#pragma once

#include "kfcore/yolo/types.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace kfcore::yolo::demo
{

struct OnnxDetectorOptions
{
    int         intra_op_threads = 0;
    int         inter_op_threads = 0;
    std::size_t max_model_bytes  = 256U * 1024U * 1024U;
    std::size_t max_source_bytes = 64U * 1024U * 1024U;
    std::size_t max_tensor_bytes = 32U * 1024U * 1024U;
    std::size_t max_output_bytes = 16U * 1024U * 1024U;
    std::size_t max_detections   = 1000U;
    float       border_value     = 114.0F;
};

class OnnxDomainDetector final
{
public:
    [[nodiscard]] static std::unique_ptr<OnnxDomainDetector>
    load(const std::filesystem::path& model_path,
         const OnnxDetectorOptions& options = {});

    ~OnnxDomainDetector();
    OnnxDomainDetector(OnnxDomainDetector&&) noexcept;
    OnnxDomainDetector& operator=(OnnxDomainDetector&&) noexcept;
    OnnxDomainDetector(const OnnxDomainDetector&)            = delete;
    OnnxDomainDetector& operator=(const OnnxDomainDetector&) = delete;

    [[nodiscard]] DetectionFrame detect(const ImageView& image);
    [[nodiscard]] std::int32_t input_width() const noexcept;
    [[nodiscard]] std::int32_t input_height() const noexcept;
    [[nodiscard]] std::size_t max_detections() const noexcept;

private:
    struct Impl;
    explicit OnnxDomainDetector(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::yolo::demo
