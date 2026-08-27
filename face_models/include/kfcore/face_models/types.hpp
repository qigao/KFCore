#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace kfcore::face_models
{

inline constexpr std::size_t kFace68LandmarkCount    = 68;
inline constexpr std::size_t kArcFaceEmbeddingLength = 512;
inline constexpr std::size_t kAgeGenderLogitCount    = 2;
inline constexpr std::size_t kInSwapperEmbeddingLength = 512;
inline constexpr std::int64_t kFaceModelInputChannels = 3;
inline constexpr std::int64_t kFace68InputExtent      = 256;
inline constexpr std::int64_t kFace68LandmarkWidth    = 3;
inline constexpr std::int64_t kFace68HeatmapExtent    = 64;
inline constexpr std::int64_t kArcFaceInputExtent     = 112;
inline constexpr std::int64_t kAgeGenderInputExtent   = 224;
inline constexpr std::int64_t kInSwapperInputExtent   = 128;
inline constexpr std::int64_t kGfpGanInputExtent      = 512;
inline constexpr std::size_t  kInSwapperOutputElementCount =
    static_cast<std::size_t>(kFaceModelInputChannels * kInSwapperInputExtent *
                             kInSwapperInputExtent);
inline constexpr std::size_t kGfpGanOutputElementCount =
    static_cast<std::size_t>(kFaceModelInputChannels * kGfpGanInputExtent *
                             kGfpGanInputExtent);

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
using InSwapperResult = std::array<float, kInSwapperOutputElementCount>;
using GfpGanResult    = std::array<float, kGfpGanOutputElementCount>;

} // namespace kfcore::face_models
