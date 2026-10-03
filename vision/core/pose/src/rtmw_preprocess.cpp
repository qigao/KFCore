#include "rtmw_preprocess.hpp"

#include "kfcore/image_processor/cpu.hpp"
#include "kfcore/pose/error.hpp"

#include <cmath>
#include <string>

namespace kfcore::pose::detail
{
namespace
{

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw PoseError(PoseErrorCode::InvalidArgument,
                    "RTMW preprocess: " + detail);
}

RtmwCropGeometry crop_geometry(const RectF& box,
                               const RtmwOptions& options,
                               std::int32_t input_width,
                               std::int32_t input_height)
{
    if (!std::isfinite(box.x) || !std::isfinite(box.y) ||
        !std::isfinite(box.width) || !std::isfinite(box.height) ||
        box.width <= 0.0F || box.height <= 0.0F)
    {
        throw_invalid("person bbox must be finite with positive width and height");
    }
    if (input_width <= 0 || input_height <= 0)
    {
        throw_invalid("model input dimensions must be positive");
    }

    RtmwCropGeometry geometry;
    geometry.center_x = box.x + box.width * 0.5F;
    geometry.center_y = box.y + box.height * 0.5F;
    geometry.scale_width = box.width * options.bbox_padding;
    geometry.scale_height = box.height * options.bbox_padding;

    const float aspect = static_cast<float>(input_width) /
                         static_cast<float>(input_height);
    if (geometry.scale_width > geometry.scale_height * aspect)
    {
        geometry.scale_height = geometry.scale_width / aspect;
    }
    else
    {
        geometry.scale_width = geometry.scale_height * aspect;
    }
    return geometry;
}

image::AffineTransform destination_to_source(
    const RtmwCropGeometry& geometry,
    std::int32_t input_width,
    std::int32_t input_height)
{
    const float scale_x =
        geometry.scale_width / static_cast<float>(input_width);
    const float scale_y =
        geometry.scale_height / static_cast<float>(input_height);
    return {{
        scale_x,
        0.0F,
        geometry.center_x - geometry.scale_width * 0.5F,
        0.0F,
        scale_y,
        geometry.center_y - geometry.scale_height * 0.5F,
    }};
}

} // namespace

RtmwPreprocessResult
preprocess_rtmw(const image::BgrImage& source,
                const RectF& box,
                const RtmwOptions& options,
                std::int32_t input_width,
                std::int32_t input_height)
{
    RtmwPreprocessResult result;
    result.geometry = crop_geometry(
        box, options, input_width, input_height);

    const image::BgrImage crop =
        image::CpuImageProcessor::warp_affine_bgr(
            source,
            input_width,
            input_height,
            destination_to_source(
                result.geometry, input_width, input_height),
            options.border_value,
            options.max_source_bytes);

    image::PreprocessOptions preprocess;
    preprocess.output_format = image::PixelFormat::Rgb8;
    preprocess.mean = options.mean;
    preprocess.stddev = options.stddev;
    preprocess.border_value = options.border_value;
    result.nchw = image::CpuImageProcessor::to_nchw(
        crop, preprocess, options.max_tensor_bytes);

    return result;
}

std::pair<float, float>
rtmw_model_to_source(const RtmwCropGeometry& geometry,
                     float model_x,
                     float model_y,
                     std::int32_t input_width,
                     std::int32_t input_height)
{
    if (input_width <= 0 || input_height <= 0)
    {
        throw_invalid("model input dimensions must be positive");
    }
    return {
        model_x / static_cast<float>(input_width) *
                geometry.scale_width +
            geometry.center_x - geometry.scale_width * 0.5F,
        model_y / static_cast<float>(input_height) *
                geometry.scale_height +
            geometry.center_y - geometry.scale_height * 0.5F,
    };
}

} // namespace kfcore::pose::detail
