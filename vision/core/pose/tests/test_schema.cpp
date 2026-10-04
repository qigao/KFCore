#include "kfcore/pose/error.hpp"
#include "kfcore/pose/schema.hpp"
#include "tinytest.hpp"

#include <cstdint>
#include <string>

using namespace kfcore::pose;

namespace
{

PoseSchema minimal_schema()
{
    PoseSchema schema;
    schema.id = UINT64_C(1);
    schema.version = 1U;
    schema.name = "test";
    schema.keypoints = {
        {10U, "left", 20U},
        {20U, "right", 10U},
    };
    schema.edges = {{10U, 20U}};
    schema.groups = {{"pair", {10U, 20U}}};
    schema.output_map = {10U, 20U};
    return schema;
}

} // namespace

spec("pose schema")
{
    it("exposes the canonical COCO WholeBody 133 ordering")
    {
        const PoseSchema& schema = coco_wholebody_133_schema();

        check(schema.id == kCocoWholeBody133SchemaId);
        check(schema.version == std::uint32_t{1U});
        check(schema.name == std::string{"coco-wholebody-133"});
        check(schema.keypoints.size() == kWholeBodyKeypointCount);
        check(schema.output_map.size() == kWholeBodyKeypointCount);
        check(schema.edges.size() == std::size_t{65U});
        check(schema.groups.size() == std::size_t{4U});

        check(schema.keypoints[0].name == std::string{"nose"});
        check(schema.keypoints[17].name == std::string{"left_big_toe"});
        check(schema.keypoints[23].name == std::string{"face-0"});
        check(schema.keypoints[91].name == std::string{"left_hand_root"});
        check(schema.keypoints[112].name == std::string{"right_hand_root"});
        check(schema.keypoints[132].name == std::string{"right_pinky_finger4"});

        check(schema.keypoints[1].mirror_id == std::uint32_t{2U});
        check(schema.keypoints[2].mirror_id == std::uint32_t{1U});
        check(schema.keypoints[91].mirror_id == std::uint32_t{112U});
        check(schema.keypoints[112].mirror_id == std::uint32_t{91U});

        for (std::uint32_t id = 0U; id < kWholeBodyKeypointCount; ++id)
        {
            check(schema.keypoints[id].id == id);
            check(schema.output_map[id] == id);
        }
    }

    it("resolves the built-in schema from explicit model semantics")
    {
        const PoseSchema& schema = pose_schema_for_semantic_contract(
            kCocoWholeBody133SemanticContract,
            kCocoWholeBody133SemanticVersion);

        check(&schema == &coco_wholebody_133_schema());
        check_throws_as(
            pose_schema_for_semantic_contract(
                "pose.unknown",
                "1"),
            PoseError);
        check_throws_as(
            pose_schema_for_semantic_contract(
                kCocoWholeBody133SemanticContract,
                "2"),
            PoseError);
    }

    it("accepts a valid custom schema")
    {
        PoseSchema schema = minimal_schema();
        validate_pose_schema(schema);
    }

    it("rejects duplicate keypoint ids")
    {
        PoseSchema schema = minimal_schema();
        schema.keypoints[1].id = 10U;

        check_throws_as(validate_pose_schema(schema), PoseError);
    }

    it("rejects invalid or asymmetric mirror mappings")
    {
        PoseSchema missing = minimal_schema();
        missing.keypoints[0].mirror_id = 30U;
        check_throws_as(validate_pose_schema(missing), PoseError);

        PoseSchema asymmetric = minimal_schema();
        asymmetric.keypoints[1].mirror_id = kInvalidPoseKeypointId;
        check_throws_as(validate_pose_schema(asymmetric), PoseError);
    }

    it("rejects edges and groups that reference unknown keypoints")
    {
        PoseSchema bad_edge = minimal_schema();
        bad_edge.edges[0].to_id = 30U;
        check_throws_as(validate_pose_schema(bad_edge), PoseError);

        PoseSchema bad_group = minimal_schema();
        bad_group.groups[0].keypoint_ids.push_back(30U);
        check_throws_as(validate_pose_schema(bad_group), PoseError);
    }

    it("rejects duplicate or unknown output mappings")
    {
        PoseSchema duplicate = minimal_schema();
        duplicate.output_map = {10U, 10U};
        check_throws_as(validate_pose_schema(duplicate), PoseError);

        PoseSchema unknown = minimal_schema();
        unknown.output_map = {10U, 30U};
        check_throws_as(validate_pose_schema(unknown), PoseError);
    }

    it("keeps unsupported generic keypoint semantics explicitly unavailable")
    {
        const PoseKeypoint keypoint;

        check(keypoint.id == kInvalidPoseKeypointId);
        check(keypoint.confidence == 0.0F);
        check(keypoint.presence != keypoint.presence);
        check(keypoint.visibility != keypoint.visibility);
    }
}
