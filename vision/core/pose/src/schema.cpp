#include "kfcore/pose/schema.hpp"

#include "kfcore/pose/error.hpp"

#include <array>
#include <cstdint>
#include <iterator>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace kfcore::pose
{
namespace
{

[[noreturn]] void throw_schema(const std::string& detail)
{
    throw PoseError(PoseErrorCode::InvalidArgument, "Pose schema: " + detail);
}

[[noreturn]] void throw_contract(const std::string& detail)
{
    throw PoseError(PoseErrorCode::ModelContractMismatch,
                    "Pose semantic contract: " + detail);
}

PoseSchema make_coco_wholebody_133_schema()
{
    PoseSchema schema;
    schema.id = kCocoWholeBody133SchemaId;
    schema.version = 1U;
    schema.name = "coco-wholebody-133";
    schema.keypoints.reserve(kWholeBodyKeypointCount);
    schema.output_map.reserve(kWholeBodyKeypointCount);

    static constexpr std::array<const char*, 23> kBodyNames {{
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_big_toe",
        "left_small_toe",
        "left_heel",
        "right_big_toe",
        "right_small_toe",
        "right_heel",
    }};

    for (std::uint32_t id = 0U; id < kBodyNames.size(); ++id)
    {
        schema.keypoints.push_back({id, kBodyNames[id], kInvalidPoseKeypointId});
    }

    for (std::uint32_t face = 0U; face < 68U; ++face)
    {
        const std::uint32_t id = 23U + face;
        schema.keypoints.push_back(
            {id, "face-" + std::to_string(face), kInvalidPoseKeypointId});
    }

    static constexpr std::array<const char*, 21> kHandSuffixes {{
        "hand_root",
        "thumb1",
        "thumb2",
        "thumb3",
        "thumb4",
        "forefinger1",
        "forefinger2",
        "forefinger3",
        "forefinger4",
        "middle_finger1",
        "middle_finger2",
        "middle_finger3",
        "middle_finger4",
        "ring_finger1",
        "ring_finger2",
        "ring_finger3",
        "ring_finger4",
        "pinky_finger1",
        "pinky_finger2",
        "pinky_finger3",
        "pinky_finger4",
    }};

    const auto append_hand = [&](const char* side, std::uint32_t first_id)
    {
        for (std::uint32_t offset = 0U; offset < kHandSuffixes.size(); ++offset)
        {
            const std::uint32_t id = first_id + offset;
            schema.keypoints.push_back(
                {id,
                 std::string(side) + "_" + kHandSuffixes[offset],
                 kInvalidPoseKeypointId});
        }
    };
    append_hand("left", 91U);
    append_hand("right", 112U);

    const auto mirror_pair = [&](std::uint32_t left, std::uint32_t right)
    {
        schema.keypoints[left].mirror_id = right;
        schema.keypoints[right].mirror_id = left;
    };

    mirror_pair(1U, 2U);
    mirror_pair(3U, 4U);
    mirror_pair(5U, 6U);
    mirror_pair(7U, 8U);
    mirror_pair(9U, 10U);
    mirror_pair(11U, 12U);
    mirror_pair(13U, 14U);
    mirror_pair(15U, 16U);
    mirror_pair(17U, 20U);
    mirror_pair(18U, 21U);
    mirror_pair(19U, 22U);

    static constexpr std::pair<std::uint32_t, std::uint32_t> kFaceMirrorPairs[] {
        {0U, 16U},
        {1U, 15U},
        {2U, 14U},
        {3U, 13U},
        {4U, 12U},
        {5U, 11U},
        {6U, 10U},
        {7U, 9U},
        {17U, 26U},
        {18U, 25U},
        {19U, 24U},
        {20U, 23U},
        {21U, 22U},
        {31U, 35U},
        {32U, 34U},
        {36U, 45U},
        {37U, 44U},
        {38U, 43U},
        {39U, 42U},
        {40U, 47U},
        {41U, 46U},
        {48U, 54U},
        {49U, 53U},
        {50U, 52U},
        {55U, 59U},
        {56U, 58U},
        {60U, 64U},
        {61U, 63U},
        {65U, 67U},
    };
    for (const auto& pair : kFaceMirrorPairs)
    {
        mirror_pair(23U + pair.first, 23U + pair.second);
    }

    for (std::uint32_t offset = 0U; offset < 21U; ++offset)
    {
        mirror_pair(91U + offset, 112U + offset);
    }

    static constexpr PoseEdge kBodyEdges[] {
        {15U, 13U},
        {13U, 11U},
        {16U, 14U},
        {14U, 12U},
        {11U, 12U},
        {5U, 11U},
        {6U, 12U},
        {5U, 6U},
        {5U, 7U},
        {6U, 8U},
        {7U, 9U},
        {8U, 10U},
        {1U, 2U},
        {0U, 1U},
        {0U, 2U},
        {1U, 3U},
        {2U, 4U},
        {3U, 5U},
        {4U, 6U},
        {15U, 17U},
        {15U, 18U},
        {15U, 19U},
        {16U, 20U},
        {16U, 21U},
        {16U, 22U},
    };
    schema.edges.insert(schema.edges.end(), std::begin(kBodyEdges), std::end(kBodyEdges));

    const auto append_hand_edges = [&](std::uint32_t root)
    {
        static constexpr std::array<std::uint32_t, 5> kFingerOffsets {{
            1U, 5U, 9U, 13U, 17U
        }};
        for (const std::uint32_t first_offset : kFingerOffsets)
        {
            const std::uint32_t first = root + first_offset;
            schema.edges.push_back({root, first});
            schema.edges.push_back({first, first + 1U});
            schema.edges.push_back({first + 1U, first + 2U});
            schema.edges.push_back({first + 2U, first + 3U});
        }
    };
    append_hand_edges(91U);
    append_hand_edges(112U);

    const auto append_group = [&](const char* name,
                                  std::uint32_t first,
                                  std::uint32_t last_inclusive)
    {
        PoseGroup group;
        group.name = name;
        group.keypoint_ids.reserve(last_inclusive - first + 1U);
        for (std::uint32_t id = first; id <= last_inclusive; ++id)
        {
            group.keypoint_ids.push_back(id);
        }
        schema.groups.push_back(std::move(group));
    };
    append_group("body", 0U, 22U);
    append_group("face", 23U, 90U);
    append_group("left_hand", 91U, 111U);
    append_group("right_hand", 112U, 132U);

    for (std::uint32_t id = 0U; id < kWholeBodyKeypointCount; ++id)
    {
        schema.output_map.push_back(id);
    }

    validate_pose_schema(schema);
    return schema;
}

} // namespace

void validate_pose_schema(const PoseSchema& schema)
{
    if (schema.id == 0U)
    {
        throw_schema("id must be non-zero");
    }
    if (schema.version == 0U)
    {
        throw_schema("version must be non-zero");
    }
    if (schema.name.empty())
    {
        throw_schema("name must not be empty");
    }
    if (schema.keypoints.empty())
    {
        throw_schema("keypoints must not be empty");
    }

    std::unordered_map<std::uint32_t, std::size_t> keypoint_index;
    keypoint_index.reserve(schema.keypoints.size());
    for (std::size_t index = 0U; index < schema.keypoints.size(); ++index)
    {
        const PoseKeypointDescriptor& keypoint = schema.keypoints[index];
        if (keypoint.id == kInvalidPoseKeypointId)
        {
            throw_schema("keypoint id must not use the invalid sentinel");
        }
        if (keypoint.name.empty())
        {
            throw_schema("keypoint name must not be empty");
        }
        if (!keypoint_index.emplace(keypoint.id, index).second)
        {
            throw_schema("duplicate keypoint id " + std::to_string(keypoint.id));
        }
    }

    for (const PoseKeypointDescriptor& keypoint : schema.keypoints)
    {
        if (keypoint.mirror_id == kInvalidPoseKeypointId)
        {
            continue;
        }
        if (keypoint.mirror_id == keypoint.id)
        {
            throw_schema("keypoint mirror must not reference itself");
        }
        const auto mirror = keypoint_index.find(keypoint.mirror_id);
        if (mirror == keypoint_index.end())
        {
            throw_schema("keypoint mirror references an unknown id");
        }
        if (schema.keypoints[mirror->second].mirror_id != keypoint.id)
        {
            throw_schema("keypoint mirror mapping must be symmetric");
        }
    }

    for (const PoseEdge& edge : schema.edges)
    {
        if (edge.from_id == edge.to_id)
        {
            throw_schema("edge endpoints must be distinct");
        }
        if (keypoint_index.find(edge.from_id) == keypoint_index.end() ||
            keypoint_index.find(edge.to_id) == keypoint_index.end())
        {
            throw_schema("edge references an unknown keypoint id");
        }
    }

    std::unordered_set<std::string> group_names;
    group_names.reserve(schema.groups.size());
    for (const PoseGroup& group : schema.groups)
    {
        if (group.name.empty())
        {
            throw_schema("group name must not be empty");
        }
        if (!group_names.emplace(group.name).second)
        {
            throw_schema("duplicate group name " + group.name);
        }
        if (group.keypoint_ids.empty())
        {
            throw_schema("group must contain at least one keypoint");
        }
        std::unordered_set<std::uint32_t> members;
        members.reserve(group.keypoint_ids.size());
        for (const std::uint32_t id : group.keypoint_ids)
        {
            if (keypoint_index.find(id) == keypoint_index.end())
            {
                throw_schema("group references an unknown keypoint id");
            }
            if (!members.emplace(id).second)
            {
                throw_schema("group contains a duplicate keypoint id");
            }
        }
    }

    if (schema.output_map.empty())
    {
        throw_schema("output map must not be empty");
    }
    std::unordered_set<std::uint32_t> mapped_ids;
    mapped_ids.reserve(schema.output_map.size());
    for (const std::uint32_t id : schema.output_map)
    {
        if (keypoint_index.find(id) == keypoint_index.end())
        {
            throw_schema("output map references an unknown keypoint id");
        }
        if (!mapped_ids.emplace(id).second)
        {
            throw_schema("output map contains a duplicate keypoint id");
        }
    }
}

const PoseSchema& coco_wholebody_133_schema()
{
    static const PoseSchema schema = make_coco_wholebody_133_schema();
    return schema;
}

const PoseSchema&
pose_schema_for_semantic_contract(std::string_view semantic_contract,
                                  std::string_view semantic_version)
{
    if (semantic_contract == kCocoWholeBody133SemanticContract &&
        semantic_version == kCocoWholeBody133SemanticVersion)
    {
        return coco_wholebody_133_schema();
    }
    throw_contract(
        "unsupported semantic identity '" + std::string(semantic_contract) +
        "' version '" + std::string(semantic_version) + "'");
}

} // namespace kfcore::pose
