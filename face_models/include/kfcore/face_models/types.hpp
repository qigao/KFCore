#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace kfcore::face_models
{

inline constexpr std::size_t kFace68LandmarkCount    = 68;
inline constexpr std::size_t kArcFaceEmbeddingLength = 512;
inline constexpr std::size_t kAgeGenderLogitCount    = 2;
inline constexpr std::int64_t kFaceModelInputChannels = 3;
inline constexpr std::int64_t kFace68InputExtent      = 256;
inline constexpr std::int64_t kFace68LandmarkWidth    = 3;
inline constexpr std::int64_t kFace68HeatmapExtent    = 64;
inline constexpr std::int64_t kArcFaceInputExtent     = 112;
inline constexpr std::int64_t kAgeGenderInputExtent   = 224;

struct Face68Landmark
{
    // x and y use the model's 256-pixel input coordinate system. score is returned unchanged.
    float x     = 0.0f;
    float y     = 0.0f;
    float score = 0.0f;
};

using Face68Result    = std::array<Face68Landmark, kFace68LandmarkCount>;
using ArcFaceResult   = std::array<float, kArcFaceEmbeddingLength>;
using AgeGenderResult = std::array<float, kAgeGenderLogitCount>;

} // namespace kfcore::face_models
