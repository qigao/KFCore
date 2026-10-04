#include "heatmap_decode.hpp"

#include "kfcore/pose/error.hpp"
#include "tinytest.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using namespace kfcore::pose;
using namespace kfcore::pose::detail;

namespace
{

std::vector<float> sparse_heatmap()
{
    constexpr std::size_t kWidth = 12U;
    constexpr std::size_t kHeight = 14U;
    std::vector<float> values(kWidth * kHeight, 0.0F);

    const auto set = [&](std::size_t x, std::size_t y, float value)
    {
        values[y * kWidth + x] = value;
    };

    set(5U, 7U, 1.0F);
    set(4U, 7U, 0.4F);
    set(6U, 7U, 0.8F);
    set(5U, 6U, 0.3F);
    set(5U, 8U, 0.7F);
    set(4U, 6U, 0.15F);
    set(6U, 6U, 0.25F);
    set(4U, 8U, 0.2F);
    set(6U, 8U, 0.55F);
    return values;
}

std::vector<float> boundary_heatmap()
{
    constexpr std::size_t kWidth = 12U;
    constexpr std::size_t kHeight = 14U;
    std::vector<float> values(kWidth * kHeight, 0.0F);

    const auto set = [&](std::size_t x, std::size_t y, float value)
    {
        values[y * kWidth + x] = value;
    };

    set(1U, 1U, 1.0F);
    set(0U, 1U, 0.5F);
    set(2U, 1U, 0.7F);
    set(1U, 0U, 0.3F);
    set(1U, 2U, 0.6F);
    set(0U, 0U, 0.1F);
    set(2U, 0U, 0.2F);
    set(0U, 2U, 0.15F);
    set(2U, 2U, 0.4F);
    return values;
}

} // namespace

spec("Gaussian heatmap UDP decoder")
{
    it("matches upstream DARK UDP refinement for an asymmetric interior peak")
    {
        const auto values = sparse_heatmap();
        DecodedHeatmapKeypoint output;

        decode_gaussian_heatmap_udp(
            {values.data(), HeatmapElementType::Float32, 1U, 14U, 12U},
            {11U},
            &output,
            1U);

        check(std::fabs(output.x - 5.20810681F) < 2.0e-5F);
        check(std::fabs(output.y - 7.18355461F) < 2.0e-5F);
        check(output.confidence == 1.0F);
    }

    it("matches reflect-101 DARK behavior near the heatmap boundary")
    {
        const auto values = boundary_heatmap();
        DecodedHeatmapKeypoint output;

        decode_gaussian_heatmap_udp(
            {values.data(), HeatmapElementType::Float32, 1U, 14U, 12U},
            {11U},
            &output,
            1U);

        check(std::fabs(output.x - 0.01955329F) < 3.0e-5F);
        check(std::fabs(output.y - 0.01808651F) < 3.0e-5F);
        check(output.confidence == 1.0F);
    }

    it("keeps invalid coordinates when the raw heatmap peak is non-positive")
    {
        std::array<float, 16> values {};
        for (float& value : values)
        {
            value = -1.0F;
        }

        DecodedHeatmapKeypoint output;
        decode_gaussian_heatmap_udp(
            {values.data(), HeatmapElementType::Float32, 1U, 4U, 4U},
            {3U},
            &output,
            1U);

        check(output.x == -1.0F);
        check(output.y == -1.0F);
        check(output.confidence == -1.0F);
    }

    it("projects refined UDP heatmap coordinates back to source geometry")
    {
        const auto source = heatmap_udp_to_source(
            {100.0F, 200.0F, 240.0F, 320.0F},
            5.20810681F,
            7.18355461F,
            12U,
            14U);

        check(std::fabs(source.first - 93.6314213F) < 2.0e-5F);
        check(std::fabs(source.second - 216.8259596F) < 3.0e-5F);
    }

    it("rejects malformed heatmaps, kernels, and non-finite responses")
    {
        const std::array<float, 4> values {{0.0F, 1.0F, 0.0F, 0.0F}};
        DecodedHeatmapKeypoint output;

        check_throws_as(
            decode_gaussian_heatmap_udp(
                {nullptr, HeatmapElementType::Float32, 1U, 2U, 2U},
                {3U},
                &output,
                1U),
            PoseError);

        check_throws_as(
            decode_gaussian_heatmap_udp(
                {values.data(), HeatmapElementType::Float32, 1U, 2U, 2U},
                {4U},
                &output,
                1U),
            PoseError);

        const float nan = (std::numeric_limits<float>::quiet_NaN)();
        const std::array<float, 4> invalid {{0.0F, nan, 1.0F, 0.0F}};
        check_throws_as(
            decode_gaussian_heatmap_udp(
                {invalid.data(), HeatmapElementType::Float32, 1U, 2U, 2U},
                {3U},
                &output,
                1U),
            PoseError);
    }
}
