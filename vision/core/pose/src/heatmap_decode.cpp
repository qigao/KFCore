#include "heatmap_decode.hpp"

#include "kfcore/pose/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace kfcore::pose::detail
{
namespace
{

constexpr std::size_t kMaxHeatmapExtent = 16384U;
constexpr std::size_t kMaxHeatmapPlaneElements =
    16U * 1024U * 1024U;
constexpr std::size_t kMaxKeypoints = 4096U;
constexpr std::size_t kMaxDarkKernel = 255U;

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw PoseError(
        PoseErrorCode::InvalidArgument,
        "Heatmap decode: " + detail);
}

[[noreturn]] void throw_contract(const std::string& detail)
{
    throw PoseError(
        PoseErrorCode::ModelContractMismatch,
        "Heatmap decode: " + detail);
}

float half_to_float(std::uint16_t bits) noexcept
{
    const bool negative = (bits & UINT16_C(0x8000)) != 0U;
    const std::uint16_t exponent =
        static_cast<std::uint16_t>((bits >> 10U) & 0x1fU);
    const std::uint16_t fraction =
        static_cast<std::uint16_t>(bits & 0x03ffU);

    float value = 0.0F;
    if (exponent == 0U)
    {
        value = std::ldexp(static_cast<float>(fraction), -24);
    }
    else if (exponent == UINT16_C(0x001f))
    {
        value = fraction == 0U
                    ? (std::numeric_limits<float>::infinity)()
                    : (std::numeric_limits<float>::quiet_NaN)();
    }
    else
    {
        value = std::ldexp(
            static_cast<float>(UINT16_C(0x0400) + fraction),
            static_cast<int>(exponent) - 25);
    }
    return negative ? -value : value;
}

float heatmap_value(const HeatmapView& view, std::size_t index)
{
    if (view.data_type == HeatmapElementType::Float32)
    {
        return static_cast<const float*>(view.data)[index];
    }
    return half_to_float(
        static_cast<const std::uint16_t*>(view.data)[index]);
}

void validate_view(const HeatmapView& view)
{
    if (view.data == nullptr)
    {
        throw_invalid("data must not be null");
    }
    if (view.keypoints == 0U ||
        view.keypoints > kMaxKeypoints)
    {
        throw_invalid("keypoint count is outside the supported bound");
    }
    if (view.width == 0U || view.height == 0U ||
        view.width > kMaxHeatmapExtent ||
        view.height > kMaxHeatmapExtent)
    {
        throw_invalid("heatmap extent is outside the supported bound");
    }

    const std::size_t plane = view.width * view.height;
    if (plane / view.width != view.height ||
        plane > kMaxHeatmapPlaneElements ||
        view.keypoints >
            (std::numeric_limits<std::size_t>::max)() / plane)
    {
        throw_invalid(
            "heatmap plane is too large or element count overflows size_t");
    }
}

std::size_t reflect101(std::int64_t index, std::size_t extent)
{
    if (extent == 1U)
    {
        return 0U;
    }

    const std::int64_t n =
        static_cast<std::int64_t>(extent);
    while (index < 0 || index >= n)
    {
        if (index < 0)
        {
            index = -index;
        }
        else
        {
            index = 2 * n - index - 2;
        }
    }
    return static_cast<std::size_t>(index);
}

std::vector<double> gaussian_kernel(std::size_t kernel)
{
    if (kernel == 0U ||
        (kernel & 1U) == 0U ||
        kernel > kMaxDarkKernel)
    {
        throw_invalid(
            "DARK kernel must be odd, positive, and <= 255");
    }

    if (kernel == 1U)
    {
        return {1.0};
    }

    const double radius =
        static_cast<double>(kernel - 1U) * 0.5;
    const double sigma =
        0.3 * (radius - 1.0) + 0.8;
    const std::int64_t half =
        static_cast<std::int64_t>(kernel / 2U);

    std::vector<double> weights(kernel);
    double sum = 0.0;
    for (std::int64_t offset = -half;
         offset <= half;
         ++offset)
    {
        const double value = std::exp(
            -static_cast<double>(offset * offset) /
            (2.0 * sigma * sigma));
        weights[static_cast<std::size_t>(offset + half)] = value;
        sum += value;
    }
    if (!std::isfinite(sum) || sum <= 0.0)
    {
        throw_contract("Gaussian kernel normalization failed");
    }
    for (double& value : weights)
    {
        value /= sum;
    }
    return weights;
}

void gaussian_blur_reflect101(
    const std::vector<float>& source,
    std::size_t width,
    std::size_t height,
    const std::vector<double>& kernel,
    std::vector<float>& output)
{
    const std::int64_t half =
        static_cast<std::int64_t>(kernel.size() / 2U);
    std::vector<float> horizontal(source.size());
    output.resize(source.size());

    for (std::size_t y = 0U; y < height; ++y)
    {
        for (std::size_t x = 0U; x < width; ++x)
        {
            double sum = 0.0;
            for (std::int64_t offset = -half;
                 offset <= half;
                 ++offset)
            {
                const std::size_t sx = reflect101(
                    static_cast<std::int64_t>(x) + offset,
                    width);
                sum +=
                    static_cast<double>(
                        source[y * width + sx]) *
                    kernel[
                        static_cast<std::size_t>(
                            offset + half)];
            }
            horizontal[y * width + x] =
                static_cast<float>(sum);
        }
    }

    for (std::size_t y = 0U; y < height; ++y)
    {
        for (std::size_t x = 0U; x < width; ++x)
        {
            double sum = 0.0;
            for (std::int64_t offset = -half;
                 offset <= half;
                 ++offset)
            {
                const std::size_t sy = reflect101(
                    static_cast<std::int64_t>(y) + offset,
                    height);
                sum +=
                    static_cast<double>(
                        horizontal[sy * width + x]) *
                    kernel[
                        static_cast<std::size_t>(
                            offset + half)];
            }
            output[y * width + x] =
                static_cast<float>(sum);
        }
    }
}

float sample_edge(
    const std::vector<float>& values,
    std::size_t width,
    std::size_t height,
    std::int64_t x,
    std::int64_t y)
{
    x = (std::max)(
        std::int64_t{0},
        (std::min)(
            x,
            static_cast<std::int64_t>(width) - 1));
    y = (std::max)(
        std::int64_t{0},
        (std::min)(
            y,
            static_cast<std::int64_t>(height) - 1));
    return values[
        static_cast<std::size_t>(y) * width +
        static_cast<std::size_t>(x)];
}

void refine_dark_udp(
    std::vector<float>& heatmap,
    std::size_t width,
    std::size_t height,
    const std::vector<double>& kernel,
    float& x,
    float& y)
{
    std::vector<float> blurred;
    gaussian_blur_reflect101(
        heatmap, width, height, kernel, blurred);

    for (float& value : blurred)
    {
        if (!std::isfinite(value))
        {
            throw_contract(
                "DARK input produced a non-finite blurred response");
        }
        value = std::log(
            (std::max)(
                0.001F,
                (std::min)(50.0F, value)));
    }

    const std::int64_t px =
        static_cast<std::int64_t>(x);
    const std::int64_t py =
        static_cast<std::int64_t>(y);

    const float i =
        sample_edge(blurred, width, height, px, py);
    const float ix1 =
        sample_edge(blurred, width, height, px + 1, py);
    const float iy1 =
        sample_edge(blurred, width, height, px, py + 1);
    const float ix1y1 =
        sample_edge(
            blurred, width, height, px + 1, py + 1);
    const float ix1_y1_ =
        sample_edge(
            blurred, width, height, px - 1, py - 1);
    const float ix1_ =
        sample_edge(blurred, width, height, px - 1, py);
    const float iy1_ =
        sample_edge(blurred, width, height, px, py - 1);

    const double dx =
        0.5 * static_cast<double>(ix1 - ix1_);
    const double dy =
        0.5 * static_cast<double>(iy1 - iy1_);
    const double dxx =
        static_cast<double>(ix1 - 2.0F * i + ix1_);
    const double dyy =
        static_cast<double>(iy1 - 2.0F * i + iy1_);
    const double dxy =
        0.5 * static_cast<double>(
            ix1y1 - ix1 - iy1 +
            i + i - ix1_ - iy1_ + ix1_y1_);

    const double eps =
        static_cast<double>(
            (std::numeric_limits<float>::epsilon)());
    const double a = dxx + eps;
    const double b = dxy;
    const double c = dxy;
    const double d = dyy + eps;
    const double determinant = a * d - b * c;
    if (!std::isfinite(determinant) ||
        std::fabs(determinant) <=
            (std::numeric_limits<double>::epsilon)())
    {
        throw_contract("DARK Hessian is singular");
    }

    const double inv00 = d / determinant;
    const double inv01 = -b / determinant;
    const double inv10 = -c / determinant;
    const double inv11 = a / determinant;

    const double shift_x = inv00 * dx + inv01 * dy;
    const double shift_y = inv10 * dx + inv11 * dy;
    if (!std::isfinite(shift_x) ||
        !std::isfinite(shift_y))
    {
        throw_contract("DARK coordinate refinement is non-finite");
    }

    x -= static_cast<float>(shift_x);
    y -= static_cast<float>(shift_y);
}

} // namespace

void decode_gaussian_heatmap_udp(
    const HeatmapView& heatmap,
    const HeatmapDecodeDesc& desc,
    DecodedHeatmapKeypoint* output,
    std::size_t output_count)
{
    validate_view(heatmap);
    if (output == nullptr)
    {
        throw_invalid("output storage must not be null");
    }
    if (output_count < heatmap.keypoints)
    {
        throw_invalid(
            "output storage is smaller than the keypoint count");
    }

    const std::vector<double> kernel =
        gaussian_kernel(desc.dark_kernel);
    const std::size_t plane =
        heatmap.width * heatmap.height;

    std::vector<float> values(plane);
    for (std::size_t keypoint = 0U;
         keypoint < heatmap.keypoints;
         ++keypoint)
    {
        const std::size_t base = keypoint * plane;
        float maximum =
            -(std::numeric_limits<float>::infinity)();
        std::size_t maximum_index = 0U;

        for (std::size_t index = 0U;
             index < plane;
             ++index)
        {
            const float value =
                heatmap_value(heatmap, base + index);
            if (!std::isfinite(value))
            {
                throw_contract(
                    "heatmap contains a non-finite response");
            }
            values[index] = value;
            if (value > maximum)
            {
                maximum = value;
                maximum_index = index;
            }
        }

        DecodedHeatmapKeypoint decoded;
        decoded.confidence = maximum;
        if (maximum > 0.0F)
        {
            decoded.x = static_cast<float>(
                maximum_index % heatmap.width);
            decoded.y = static_cast<float>(
                maximum_index / heatmap.width);
            refine_dark_udp(
                values,
                heatmap.width,
                heatmap.height,
                kernel,
                decoded.x,
                decoded.y);
        }
        output[keypoint] = decoded;
    }
}

std::pair<float, float>
heatmap_udp_to_source(
    const HeatmapSourceGeometry& geometry,
    float heatmap_x,
    float heatmap_y,
    std::size_t heatmap_width,
    std::size_t heatmap_height)
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
    if (!std::isfinite(heatmap_x) ||
        !std::isfinite(heatmap_y))
    {
        throw_invalid("heatmap coordinates must be finite");
    }
    if (heatmap_width <= 1U ||
        heatmap_height <= 1U)
    {
        throw_invalid(
            "UDP source transform requires heatmap extent > 1");
    }

    const float scale_x =
        geometry.scale_width /
        static_cast<float>(heatmap_width - 1U);
    const float scale_y =
        geometry.scale_height /
        static_cast<float>(heatmap_height - 1U);

    return {
        heatmap_x * scale_x +
            geometry.center_x -
            geometry.scale_width * 0.5F,
        heatmap_y * scale_y +
            geometry.center_y -
            geometry.scale_height * 0.5F,
    };
}

} // namespace kfcore::pose::detail
