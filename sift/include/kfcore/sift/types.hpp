#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kfcore::sift
{

inline constexpr std::size_t kSiftDescriptorLength = 128;

struct Feature
{
    // Pixel coordinates and Gaussian sigma in the original input image coordinate system.
    float x           = 0.0f;
    float y           = 0.0f;
    float scale       = 0.0f;
    float orientation_radians = 0.0f;
    std::int32_t octave       = 0;
    std::array<float, kSiftDescriptorLength> descriptor {};
};

struct FeatureSet
{
    std::vector<Feature> features;
};

} // namespace kfcore::sift
