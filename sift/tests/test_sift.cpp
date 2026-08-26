#include "kfcore/sift/error.hpp"
#include "kfcore/sift/sift_extractor.hpp"
#include "tinytest.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>

using namespace kfcore::sift;

namespace
{

class FakeExtractor final : public SiftExtractor
{
public:
    FeatureSet extract(const kfcore::image::ImageView& image) override
    {
        check(image.width == 1);
        check(image.height == 1);
        Feature feature;
        feature.x           = 2.0f;
        feature.y           = 3.0f;
        feature.scale       = 4.0f;
        feature.orientation_radians = 0.5f;
        feature.octave      = 1;
        feature.descriptor[0] = 0.25f;
        return { { feature } };
    }
};

} // namespace

spec("SIFT public contract")
{
    it("returns owned descriptors through the backend-independent strategy")
    {
        const std::array<std::uint8_t, 3> pixel = { 1, 2, 3 };
        const kfcore::image::ImageView image = {
            pixel.data(), pixel.size(), 1, 1, pixel.size(),
            kfcore::image::PixelFormat::Rgb8, kfcore::image::MemoryKind::Host,
        };
        std::unique_ptr<SiftExtractor> extractor = std::make_unique<FakeExtractor>();

        FeatureSet result = extractor->extract(image);
        check(result.features.size() == 1);
        check(result.features[0].x == 2.0f);
        check(result.features[0].descriptor.size() == kSiftDescriptorLength);
        check(result.features[0].descriptor[0] == 0.25f);
    }

    it("reports stable typed errors")
    {
        const SiftError error(SiftErrorCode::ResourceLimitExceeded,
                              "feature capacity exceeded");
        check(error.code() == SiftErrorCode::ResourceLimitExceeded);
        check(std::string(error.what()) == "feature capacity exceeded");
    }
}
