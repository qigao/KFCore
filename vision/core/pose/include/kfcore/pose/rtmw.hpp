#pragma once

#include "kfcore/image_processor/types.hpp"
#include "kfcore/pose/error.hpp"
#include "kfcore/pose/types.hpp"
#include "kfcore/runtime/model_package.hpp"
#include "kfcore/runtime/resolver.hpp"
#include "kfcore/runtime/runtime.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace kfcore::pose
{

struct RtmwOptions
{
    std::int32_t input_width = 0;
    std::int32_t input_height = 0;
    float bbox_padding = 1.25F;
    float simcc_split_ratio = 2.0F;
    float border_value = 0.0F;
    std::array<float, 3> mean {0.485F, 0.456F, 0.406F};
    std::array<float, 3> stddev {0.229F, 0.224F, 0.225F};
    std::string input_name;
    std::string simcc_x_name = "simcc_x";
    std::string simcc_y_name = "simcc_y";
    std::size_t max_source_bytes = 64U * 1024U * 1024U;
    std::size_t max_tensor_bytes = 32U * 1024U * 1024U;
    std::size_t max_output_bytes = 32U * 1024U * 1024U;
};

class Rtmw final
{
public:
    ~Rtmw();

    Rtmw(const Rtmw&) = delete;
    Rtmw& operator=(const Rtmw&) = delete;

    [[nodiscard]] static std::unique_ptr<Rtmw>
    load(runtime::Runtime& runtime,
         const runtime::ModelPackage& package,
         const runtime::ExecutionPolicy& policy,
         const RtmwOptions& options = {});

    [[nodiscard]] WholeBodyPose infer(const image::ImageView& image,
                                      const RectF& person_box);

    [[nodiscard]] std::vector<WholeBodyPose>
    infer(const image::ImageView& image,
          const std::vector<RectF>& person_boxes);

    [[nodiscard]] std::int32_t input_width() const noexcept;
    [[nodiscard]] std::int32_t input_height() const noexcept;
    [[nodiscard]] const runtime::ExecutionRoute& execution_route() const noexcept;

private:
    struct Impl;
    explicit Rtmw(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::pose
