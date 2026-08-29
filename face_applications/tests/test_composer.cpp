#include "kfcore/face_applications/composer.hpp"
#include "kfcore/face_applications/error.hpp"
#include "tinytest.hpp"

#include <limits>

using namespace kfcore::face_applications;

spec("face result composition")
{
    it("creates the reference bounded blurred static box mask")
    {
        const FloatMask mask = create_static_box_mask(128, 128);
        check(mask.width == 128);
        check(mask.height == 128);
        check(mask.values.size() == 128U * 128U);
        check(mask.values[0] <= 0.01F);
        check(mask.values[64U * 128U + 64U] >= 0.99F);
        check(mask.values[9U * 128U + 64U] > 0.0F);
        check(mask.values[9U * 128U + 64U] < 1.0F);
    }

    it("honors explicit padding and rejects malformed mask options")
    {
        FaceMaskOptions options;
        options.blur_fraction = 0.0F;
        options.padding_percent = { 25, 0, 0, 0 };
        const FloatMask mask = create_static_box_mask(8, 8, options);
        check(mask.values[0U * 8U + 4U] == 0.0F);
        check(mask.values[3U * 8U + 4U] == 1.0F);

        options.blur_fraction = (std::numeric_limits<float>::quiet_NaN)();
        check_throws_as(create_static_box_mask(8, 8, options), FaceApplicationError);
        check_throws_as(create_static_box_mask(0, 8), FaceApplicationError);
    }
}
