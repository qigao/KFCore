#include "scene_graph_detail.hpp"

#include "tinytest.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

using namespace kfcore;

namespace
{

yolo::TrackFrame tracked_frame()
{
    yolo::TrackFrame frame;
    frame.image_width = 640;
    frame.image_height = 480;
    frame.detections = {
        {{{10.0F, 20.0F, 110.0F, 220.0F}, 0.90F, 0}, std::uint64_t{101}},
        {{{200.0F, 100.0F, 340.0F, 300.0F}, 0.80F, 2}, std::nullopt},
    };
    return frame;
}

relation::RelationFrame relation_frame()
{
    relation::RelationFrame frame;
    frame.image_width = 640;
    frame.image_height = 480;
    frame.vocabulary_version = 7U;
    frame.edges = {
        {0U, 1U, 3U, 0.91F, std::uint64_t{101}, std::nullopt},
    };
    return frame;
}

} // namespace

spec("scene graph pipeline dynamic vocabulary API")
{
    it("exposes typed dynamic-vocabulary control without changing frame ABI")
    {
        using Pipeline = pipelines::SceneGraphPipeline;
        static_assert(
            std::is_same_v<
                decltype(std::declval<const Pipeline&>()
                             .supports_dynamic_vocabulary()),
                bool>);
        static_assert(
            std::is_same_v<
                decltype(std::declval<const Pipeline&>()
                             .vocabulary_version()),
                std::uint64_t>);
        static_assert(
            std::is_same_v<
                decltype(std::declval<Pipeline&>().set_vocabulary(
                    std::declval<relation::PredicateVocabulary>())),
                void>);
        check(true);
    }
}

spec("scene graph pipeline composition")
{
    it("maps tracked detections to relation regions without losing identity")
    {
        const auto regions =
            pipelines::detail::regions_from_tracks(tracked_frame());

        check(regions.size() == std::size_t{2U});
        check(regions[0].left == 10.0F);
        check(regions[0].bottom == 220.0F);
        check(regions[0].detector_score == 0.90F);
        check(regions[0].track_id == std::uint64_t{101});
        check_false(regions[1].track_id.has_value());
    }

    it("keeps object table and relation indices in one frame contract")
    {
        auto scene = pipelines::detail::assemble_scene_graph(
            tracked_frame(), relation_frame());

        check(scene.objects.detections.size() == std::size_t{2U});
        check(scene.relations.edges.size() == std::size_t{1U});
        check(scene.relations.edges[0].subject_index == std::size_t{0U});
        check(scene.relations.edges[0].object_index == std::size_t{1U});
        check(scene.relations.edges[0].subject_track_id ==
              scene.objects.detections[0].track_id);
        check(scene.relations.edges[0].object_track_id ==
              scene.objects.detections[1].track_id);
        check(scene.relations.vocabulary_version == std::uint64_t{7});
    }

    it("rejects relation frames from a different image geometry")
    {
        auto relations = relation_frame();
        relations.image_width = 320;

        check_throws_as(
            pipelines::detail::assemble_scene_graph(
                tracked_frame(), std::move(relations)),
            std::runtime_error);
    }

    it("rejects relation edges outside the object table")
    {
        auto relations = relation_frame();
        relations.edges[0].object_index = 4U;

        check_throws_as(
            pipelines::detail::assemble_scene_graph(
                tracked_frame(), std::move(relations)),
            std::runtime_error);
    }

    it("rejects relation track identities that drift from the object table")
    {
        auto relations = relation_frame();
        relations.edges[0].subject_track_id = std::uint64_t{999};

        check_throws_as(
            pipelines::detail::assemble_scene_graph(
                tracked_frame(), std::move(relations)),
            std::runtime_error);
    }
}
