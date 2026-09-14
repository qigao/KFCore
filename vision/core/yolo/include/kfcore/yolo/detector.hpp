#pragma once

#include "kfcore/runtime/model_package.hpp"
#include "kfcore/runtime/resolver.hpp"
#include "kfcore/yolo/types.hpp"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace kfcore::yolo
{

struct YoloDetectorOptions
{
    std::int32_t input_width = 0;
    std::int32_t input_height = 0;
    float border_value = 114.0F;
    bool mirror_horizontal = false;
    float score_threshold = 0.25F;
    float iou_threshold = 0.45F;
    std::size_t max_detections = 300U;
    std::size_t max_source_bytes = 64U * 1024U * 1024U;
    std::size_t max_tensor_bytes = 64U * 1024U * 1024U;
    std::size_t max_output_bytes = 64U * 1024U * 1024U;
};

class YoloDetector final
{
public:
    ~YoloDetector();
    YoloDetector(YoloDetector&&) noexcept;
    YoloDetector& operator=(YoloDetector&&) noexcept;
    YoloDetector(const YoloDetector&) = delete;
    YoloDetector& operator=(const YoloDetector&) = delete;

    [[nodiscard]] static std::unique_ptr<YoloDetector> load(
        const runtime::ModelPackage& package,
        const runtime::BackendRegistry& backends,
        const runtime::ExecutionPolicy& policy,
        const YoloDetectorOptions& options = {});

    [[nodiscard]] DetectionFrame detect(const ImageView& image);

    [[nodiscard]] std::int32_t input_width() const noexcept;
    [[nodiscard]] std::int32_t input_height() const noexcept;
    [[nodiscard]] const runtime::ExecutionRoute& execution_route() const noexcept;

private:
    struct Impl;
    explicit YoloDetector(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::yolo
