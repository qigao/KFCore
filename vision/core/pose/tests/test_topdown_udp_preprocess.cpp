#include "topdown_udp_preprocess.hpp"

#include "kfcore/pose/error.hpp"
#include "tinytest.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

using namespace kfcore::pose;
using namespace kfcore::pose::detail;

spec("top-down UDP preprocessing")
{
    it("matches the canonical ViTPose bbox-to-center-scale contract")
    {
        const HeatmapSourceGeometry geometry =
            topdown_udp_geometry(
                {10.0F, 20.0F, 30.0F, 50.0F},
                1.25F,
                192,
                256);

        check(std::fabs(geometry.center_x - 25.0F) < 1.0e-6F);
        check(std::fabs(geometry.center_y - 45.0F) < 1.0e-6F);
        check(std::fabs(geometry.scale_width - 46.875F) < 1.0e-6F);
        check(std::fabs(geometry.scale_height - 62.5F) < 1.0e-6F);
    }

    it("uses unbiased input-size-minus-one affine denominators")
    {
        const HeatmapSourceGeometry geometry =
            topdown_udp_geometry(
                {10.0F, 20.0F, 30.0F, 50.0F},
                1.25F,
                192,
                256);
        const auto affine =
            topdown_udp_destination_to_source(
                geometry,
                192,
                256);

        check(std::fabs(
                  affine.values[0] -
                  (46.875F / 191.0F)) < 1.0e-7F);
        check(affine.values[1] == 0.0F);
        check(std::fabs(
                  affine.values[2] - 1.5625F) < 1.0e-6F);
        check(affine.values[3] == 0.0F);
        check(std::fabs(
                  affine.values[4] -
                  (62.5F / 255.0F)) < 1.0e-7F);
        check(std::fabs(
                  affine.values[5] - 13.75F) < 1.0e-6F);
    }

    it("shares exact source bounds with the UDP heatmap projection")
    {
        const HeatmapSourceGeometry geometry =
            topdown_udp_geometry(
                {10.0F, 20.0F, 30.0F, 50.0F},
                1.25F,
                192,
                256);
        const auto affine =
            topdown_udp_destination_to_source(
                geometry,
                192,
                256);

        const float preprocess_left = affine.values[2];
        const float preprocess_top = affine.values[5];
        const float preprocess_right =
            affine.values[0] * 191.0F + affine.values[2];
        const float preprocess_bottom =
            affine.values[4] * 255.0F + affine.values[5];

        const auto heatmap_top_left =
            heatmap_udp_to_source(
                geometry,
                0.0F,
                0.0F,
                48U,
                64U);
        const auto heatmap_bottom_right =
            heatmap_udp_to_source(
                geometry,
                47.0F,
                63.0F,
                48U,
                64U);

        check(std::fabs(
                  preprocess_left -
                  heatmap_top_left.first) < 1.0e-6F);
        check(std::fabs(
                  preprocess_top -
                  heatmap_top_left.second) < 1.0e-6F);
        check(std::fabs(
                  preprocess_right -
                  heatmap_bottom_right.first) < 1.0e-5F);
        check(std::fabs(
                  preprocess_bottom -
                  heatmap_bottom_right.second) < 1.0e-5F);
    }

    it("produces RGB ImageNet-normalized NCHW for a constant crop")
    {
        kfcore::image::BgrImage image;
        image.width = 100;
        image.height = 100;
        image.pixels.resize(100U * 100U * 3U);

        for (std::size_t pixel = 0U; pixel < 100U * 100U; ++pixel)
        {
            image.pixels[pixel * 3U + 0U] = UINT8_C(10);
            image.pixels[pixel * 3U + 1U] = UINT8_C(20);
            image.pixels[pixel * 3U + 2U] = UINT8_C(30);
        }

        TopdownUdpPreprocessDesc desc;
        const auto result = preprocess_topdown_udp(
            image,
            {25.0F, 20.0F, 30.0F, 40.0F},
            desc,
            6,
            8);

        check(result.nchw.size() == std::size_t{6U * 8U * 3U});

        const float expected_r =
            (30.0F / 255.0F - desc.mean[0]) / desc.stddev[0];
        const float expected_g =
            (20.0F / 255.0F - desc.mean[1]) / desc.stddev[1];
        const float expected_b =
            (10.0F / 255.0F - desc.mean[2]) / desc.stddev[2];

        const std::size_t plane = 6U * 8U;
        for (std::size_t index = 0U; index < plane; ++index)
        {
            check(std::fabs(
                      result.nchw[index] -
                      expected_r) < 1.0e-6F);
            check(std::fabs(
                      result.nchw[plane + index] -
                      expected_g) < 1.0e-6F);
            check(std::fabs(
                      result.nchw[2U * plane + index] -
                      expected_b) < 1.0e-6F);
        }
    }

    it("rejects malformed bbox, normalization, and one-pixel UDP extents")
    {
        TopdownUdpPreprocessDesc desc;

        check_throws_as(
            topdown_udp_geometry(
                {0.0F, 0.0F, 0.0F, 10.0F},
                1.25F,
                192,
                256),
            PoseError);

        check_throws_as(
            topdown_udp_geometry(
                {0.0F, 0.0F, 10.0F, 10.0F},
                1.25F,
                1,
                256),
            PoseError);

        kfcore::image::BgrImage image;
        image.width = 2;
        image.height = 2;
        image.pixels.resize(12U);

        desc.stddev[1] = 0.0F;
        check_throws_as(
            preprocess_topdown_udp(
                image,
                {0.0F, 0.0F, 2.0F, 2.0F},
                desc,
                192,
                256),
            PoseError);
    }
}
