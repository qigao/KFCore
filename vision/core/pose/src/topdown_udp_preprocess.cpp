#include "topdown_udp_preprocess.hpp"

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
    throw PoseError(
        PoseErrorCode::InvalidArgument,
        "Top-down UDP preprocess: " + detail);
}

void validate_desc(const TopdownUdpPreprocessDesc& desc)
{
    if (!std::isfinite(desc.bbox_padding) ||
        desc.bbox_padding <= 0.0F)
    {
        throw_invalid("bbox padding must be finite and positive");
    }
    if (!std::isfinite(desc.border_value))
    {
        throw_invalid("border value must be finite");
    }
    for (std::size_t channel = 0U;
         channel < desc.mean.size();
         ++channel)
    {
        if (!std::isfinite(desc.mean[channel]) ||
            !std::isfinite(desc.stddev[channel]) ||
            desc.stddev[channel] <= 0.0F)
        {
            throw_invalid(
                "normalization values must be finite and stddev positive");
        }
    }
}

} // namespace

HeatmapSourceGeometry
topdown_udp_geometry(
    const RectF& box,
    float bbox_padding,
    std::int32_t input_width,
    std::int32_t input_height)
{
    if (!std::isfinite(box.x) ||
        !std::isfinite(box.y) ||
        !std::isfinite(box.width) ||
        !std::isfinite(box.height) ||
        box.width <= 0.0F ||
        box.height <= 0.0F)
    {
        throw_invalid(
            "person bbox must be finite with positive width and height");
    }
    if (!std::isfinite(bbox_padding) ||
        bbox_padding <= 0.0F)
    {
        throw_invalid("bbox padding must be finite and positive");
    }
    if (input_width <= 1 || input_height <= 1)
    {
        throw_invalid(
            "UDP model input dimensions must be greater than one");
    }

    HeatmapSourceGeometry geometry;
    geometry.center_x =
        box.x + box.width * 0.5F;
    geometry.center_y =
        box.y + box.height * 0.5F;

    float width = box.width;
    float height = box.height;
    const float aspect =
        static_cast<float>(input_width) /
        static_cast<float>(input_height);

    if (width > aspect * height)
    {
        height = width / aspect;
    }
    else if (width < aspect * height)
    {
        width = height * aspect;
    }

    geometry.scale_width = width * bbox_padding;
    geometry.scale_height = height * bbox_padding;
    return geometry;
}

image::AffineTransform
topdown_udp_destination_to_source(
    const HeatmapSourceGeometry& geometry,
    std::int32_t input_width,
    std::int32_t input_height)
{
    if (!std::isfinite(geometry.center_x) ||
        !std::isfinite(geometry.center_y) ||
        !std::isfinite(geometry.scale_width) ||
        !std::isfinite(geometry.scale_height) ||
        geometry.scale_width <= 0.0F ||
        geometry.scale_height <= 0.0F)
    {
        throw_invalid(
            "source geometry must be finite with positive scale");
    }
    if (input_width <= 1 || input_height <= 1)
    {
        throw_invalid(
            "UDP model input dimensions must be greater than one");
    }

    const float scale_x =
        geometry.scale_width /
        static_cast<float>(input_width - 1);
    const float scale_y =
        geometry.scale_height /
        static_cast<float>(input_height - 1);

    return {{
        scale_x,
        0.0F,
        geometry.center_x -
            geometry.scale_width * 0.5F,
        0.0F,
        scale_y,
        geometry.center_y -
            geometry.scale_height * 0.5F,
    }};
}

TopdownUdpPreprocessResult
preprocess_topdown_udp(
    const image::BgrImage& source,
    const RectF& box,
    const TopdownUdpPreprocessDesc& desc,
    std::int32_t input_width,
    std::int32_t input_height)
{
    validate_desc(desc);

    TopdownUdpPreprocessResult result;
    result.geometry = topdown_udp_geometry(
        box,
        desc.bbox_padding,
        input_width,
        input_height);
    result.destination_to_source =
        topdown_udp_destination_to_source(
            result.geometry,
            input_width,
            input_height);

    const image::BgrImage crop =
        image::CpuImageProcessor::warp_affine_bgr(
            source,
            input_width,
            input_height,
            result.destination_to_source,
            desc.border_value,
            desc.max_source_bytes);

    image::PreprocessOptions preprocess;
    preprocess.output_format =
        image::PixelFormat::Rgb8;
    preprocess.mean = desc.mean;
    preprocess.stddev = desc.stddev;
    preprocess.border_value = desc.border_value;

    result.nchw = image::CpuImageProcessor::to_nchw(
        crop,
        preprocess,
        desc.max_tensor_bytes);
    return result;
}

} // namespace kfcore::pose::detail
