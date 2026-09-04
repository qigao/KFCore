#include "composer.hpp"
#include "tinytest.hpp"

#include <cstdint>
#include <limits>

using namespace kfcore::face_applications::detail;

namespace
{

kfcore::image::BgrImage constant_bgr(int extent, std::uint8_t value)
{
    kfcore::image::BgrImage image;
    image.width = extent;
    image.height = extent;
    image.pixels.assign(static_cast<std::size_t>(extent) * extent * 3U, value);
    return image;
}

} // namespace

spec("OpenCV-free face composition")
{
    it("creates a bounded blurred static box mask")
    {
        const FloatMask mask = create_static_box_mask(128, 128);
        check(mask.width == 128);
        check(mask.height == 128);
        check(mask.values[0] <= 0.01F);
        check(mask.values[64U * 128U + 64U] >= 0.99F);
        check(mask.values[9U * 128U + 64U] > 0.0F);
        check(mask.values[9U * 128U + 64U] < 1.0F);
    }

    it("decodes RGB CHW unit output into BGR pixels")
    {
        std::vector<float> values(3U * 128U * 128U, 0.0F);
        const std::size_t plane = 128U * 128U;
        values[0] = 1.2F;
        values[plane] = 0.5F;
        values[2U * plane] = -0.2F;

        const kfcore::image::BgrImage image = decode_rgb_chw(values, 128, false);

        check(image.pixels[0] == std::uint8_t { 0 });
        check(image.pixels[1] == std::uint8_t { 127 });
        check(image.pixels[2] == std::uint8_t { 255 });
    }

    it("pastes identity masks without mutating the target")
    {
        const kfcore::image::BgrImage target = constant_bgr(8, 20);
        const kfcore::image::BgrImage aligned = constant_bgr(8, 220);
        FloatMask mask;
        mask.width = 8;
        mask.height = 8;
        mask.values.assign(64U, 1.0F);
        const AffineMatrix identity { { 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F } };

        const kfcore::image::BgrImage pasted =
            paste_back(target, aligned, mask, identity, 4096);

        check_eq_container(pasted.pixels, aligned.pixels);
        check(target.pixels[0] == std::uint8_t { 20 });
    }

    it("blends images using an explicit bounded enhancer strength")
    {
        const kfcore::image::BgrImage base = constant_bgr(2, 20);
        const kfcore::image::BgrImage enhanced = constant_bgr(2, 220);
        const kfcore::image::BgrImage quarter = blend_images(base, enhanced, 0.25F, 1024);
        check(quarter.pixels[0] == std::uint8_t { 70 });
        check_throws_as(blend_images(base, enhanced, 1.1F, 1024),
                        kfcore::face_applications::CpuFaceApplicationError);
    }
}
