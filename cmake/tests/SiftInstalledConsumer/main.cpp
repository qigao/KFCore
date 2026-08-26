#include <kfcore/sift/sift_extractor.hpp>

#include <array>
#include <cstdint>

namespace
{

class ConsumerExtractor final : public kfcore::sift::SiftExtractor
{
public:
    kfcore::sift::FeatureSet extract(const kfcore::image::ImageView&) override
    {
        kfcore::sift::Feature feature;
        feature.descriptor[0] = 1.0f;
        return { { feature } };
    }
};

} // namespace

int main()
{
    const std::array<std::uint8_t, 1> pixel = { 0 };
    const kfcore::image::ImageView image = {
        pixel.data(), pixel.size(), 1, 1, 1, kfcore::image::PixelFormat::Gray8,
        kfcore::image::MemoryKind::Host,
    };
    ConsumerExtractor extractor;
    const kfcore::sift::FeatureSet features = extractor.extract(image);
    return features.features.size() == 1 && features.features[0].descriptor[0] == 1.0f ? 0 : 1;
}
