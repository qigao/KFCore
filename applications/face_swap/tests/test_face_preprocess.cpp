#include "preprocess.hpp"
#include "tinytest.hpp"

#include <cmath>

using namespace kfcore::face_applications::detail;

namespace
{

bool approximately_equal(float actual, float expected, float tolerance = 1.0e-6F)
{
    return std::fabs(actual - expected) <= tolerance;
}

kfcore::image::BgrImage constant_bgr(int extent, std::uint8_t blue,
                                     std::uint8_t green, std::uint8_t red)
{
    kfcore::image::BgrImage image;
    image.width = extent;
    image.height = extent;
    image.pixels.resize(static_cast<std::size_t>(extent) * extent * 3U);
    for (std::size_t pixel = 0; pixel < image.pixels.size() / 3U; ++pixel)
    {
        image.pixels[pixel * 3U] = blue;
        image.pixels[pixel * 3U + 1U] = green;
        image.pixels[pixel * 3U + 2U] = red;
    }
    return image;
}

} // namespace

spec("OpenCV-free face preprocessing")
{
    it("keeps BGR planar order for Face68 and divides by 255")
    {
        const std::vector<float> tensor =
            preprocess_face68(constant_bgr(256, 10, 20, 30), 4U * 1024U * 1024U);
        const std::size_t plane = 256U * 256U;
        check_true(approximately_equal(tensor[0], 10.0F / 255.0F));
        check_true(approximately_equal(tensor[plane], 20.0F / 255.0F));
        check_true(approximately_equal(tensor[2U * plane], 30.0F / 255.0F));
    }

    it("uses the established RGB normalization for ArcFace InSwapper and GFPGAN")
    {
        const std::vector<float> arcface =
            preprocess_arcface(constant_bgr(112, 0, 127, 255), 4U * 1024U * 1024U);
        const std::size_t arc_plane = 112U * 112U;
        check_true(approximately_equal(arcface[0], 1.0F));
        check_true(approximately_equal(arcface[arc_plane], 127.0F / 127.5F - 1.0F));
        check_true(approximately_equal(arcface[2U * arc_plane], -1.0F));

        const std::vector<float> swap =
            preprocess_inswapper(constant_bgr(128, 0, 127, 255), 4U * 1024U * 1024U);
        const std::size_t swap_plane = 128U * 128U;
        check_true(approximately_equal(swap[0], 1.0F));
        check_true(approximately_equal(swap[swap_plane], 127.0F / 255.0F));
        check_true(approximately_equal(swap[2U * swap_plane], 0.0F));

        const std::vector<float> gfpgan =
            preprocess_gfpgan(constant_bgr(512, 0, 127, 255), 8U * 1024U * 1024U);
        const std::size_t gfp_plane = 512U * 512U;
        check_true(approximately_equal(gfpgan[0], 1.0F));
        check_true(approximately_equal(gfpgan[gfp_plane], 127.0F / 127.5F - 1.0F));
        check_true(approximately_equal(gfpgan[2U * gfp_plane], -1.0F));
    }

    it("resizes and applies ImageNet normalization for age gender")
    {
        const std::vector<float> tensor =
            preprocess_age_gender(constant_bgr(10, 0, 0, 255), 4U * 1024U * 1024U,
                                  4U * 1024U * 1024U);
        const std::size_t plane = 224U * 224U;
        check_true(approximately_equal(tensor[0], (1.0F - 0.485F) / 0.229F, 1.0e-5F));
        check_true(approximately_equal(tensor[plane], (0.0F - 0.456F) / 0.224F, 1.0e-5F));
        check_true(approximately_equal(tensor[2U * plane], (0.0F - 0.406F) / 0.225F,
                                       1.0e-5F));
    }
}

