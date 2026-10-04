#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace kfcore::pose
{

inline constexpr std::size_t kWholeBodyKeypointCount = 133U;
inline constexpr std::uint32_t kInvalidPoseKeypointId =
    (std::numeric_limits<std::uint32_t>::max)();

inline constexpr std::uint64_t kPoseCapabilityConfidence = UINT64_C(1) << 0U;
inline constexpr std::uint64_t kPoseCapabilityPresence   = UINT64_C(1) << 1U;
inline constexpr std::uint64_t kPoseCapabilityVisibility = UINT64_C(1) << 2U;
inline constexpr std::uint64_t kPoseCapabilityEmbedding  = UINT64_C(1) << 3U;

inline constexpr std::uint32_t kPoseKeypointFlagValid =
    UINT32_C(1) << 0U;
inline constexpr std::uint32_t kPoseKeypointFlagVisible =
    UINT32_C(1) << 1U;
inline constexpr std::uint32_t kPoseKeypointFlagOccluded =
    UINT32_C(1) << 2U;
inline constexpr std::uint32_t kPoseKeypointFlagOutsideImage =
    UINT32_C(1) << 3U;
inline constexpr std::uint32_t kPoseKeypointFlagTruncated =
    UINT32_C(1) << 4U;

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

struct PoseKeypoint
{
    float x = -1.0F;
    float y = -1.0F;
    float confidence = 0.0F;
    float presence = (std::numeric_limits<float>::quiet_NaN)();
    float visibility = (std::numeric_limits<float>::quiet_NaN)();
    std::uint32_t id = kInvalidPoseKeypointId;
    std::uint32_t flags = 0U;
};

struct PoseResult
{
    RectF source_box;
    std::uint64_t schema_id = 0U;
    std::uint64_t capabilities = 0U;

    // Results follow model output-channel order. Semantic identity is carried
    // by PoseKeypoint::id; consumers must not treat the vector index as a
    // stable keypoint id.
    std::vector<PoseKeypoint> keypoints;
};

} // namespace kfcore::pose
