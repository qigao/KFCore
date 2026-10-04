#pragma once

#include "kfcore/pose/types.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace kfcore::pose
{

inline constexpr std::uint64_t kCocoWholeBody133SchemaId =
    UINT64_C(0x434f434f57423133);
inline constexpr std::string_view kCocoWholeBody133SemanticContract =
    "pose.coco-wholebody-133";
inline constexpr std::string_view kCocoWholeBody133SemanticVersion = "1";

struct PoseKeypointDescriptor
{
    std::uint32_t id = kInvalidPoseKeypointId;
    std::string name;
    std::uint32_t mirror_id = kInvalidPoseKeypointId;
};

struct PoseEdge
{
    std::uint32_t from_id = kInvalidPoseKeypointId;
    std::uint32_t to_id = kInvalidPoseKeypointId;
};

struct PoseGroup
{
    std::string name;
    std::vector<std::uint32_t> keypoint_ids;
};

struct PoseSchema
{
    std::uint64_t id = 0U;
    std::uint32_t version = 0U;
    std::string name;
    std::vector<PoseKeypointDescriptor> keypoints;
    std::vector<PoseEdge> edges;
    std::vector<PoseGroup> groups;

    // Model output channel -> stable PoseKeypointDescriptor::id.
    std::vector<std::uint32_t> output_map;
};

void validate_pose_schema(const PoseSchema& schema);

[[nodiscard]] const PoseSchema& coco_wholebody_133_schema();
[[nodiscard]] const PoseSchema&
pose_schema_for_semantic_contract(std::string_view semantic_contract,
                                  std::string_view semantic_version);

} // namespace kfcore::pose
