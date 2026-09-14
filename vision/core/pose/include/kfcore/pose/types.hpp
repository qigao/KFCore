#pragma once

#include <array>
#include <cstddef>

namespace kfcore::pose
{

inline constexpr std::size_t kWholeBodyKeypointCount = 133U;

struct RectF
{
    float x = 0.0F;
    float y = 0.0F;
    float width = 0.0F;
    float height = 0.0F;
};

struct Keypoint
{
    float x = -1.0F;
    float y = -1.0F;
    float score = 0.0F;
};

struct WholeBodyPose
{
    RectF source_box;
    std::array<Keypoint, kWholeBodyKeypointCount> keypoints {};
};

} // namespace kfcore::pose
