#include "scene_graph_detail.hpp"

#include "tinytest.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace kfcore;

namespace
{

class FakeDetector final : public pipelines::detail::DetectionRunner
{
public:
    yolo::DetectionFrame frame;
    std::size_t calls = 0U;

    yolo::DetectionFrame
    detect(const image::ImageView&) override
    {
        ++calls;
        return frame;
    }
};

class FakeRelation final : public pipelines::detail::RelationRunner
{
public:
    std::size_t max_boxes_value = 8U;
    bool dynamic = true;
    bool live = true;
    std::uint64_t version = 7U;
    std::vector<std::string> names {"interacting"};
    std::vector<relation::Region> last_regions;
    std::size_t infer_calls = 0U;

    std::size_t max_boxes() const noexcept override
    {
        return max_boxes_value;
    }

    bool supports_dynamic_vocabulary() const noexcept override
    {
        return dynamic;
    }

    bool supports_live_predicates() const noexcept override
    {
        return live;
    }

    void set_vocabulary(
        relation::PredicateVocabulary vocabulary) override
    {
        if (!dynamic)
        {
            throw std::logic_error("fixed vocabulary");
        }
        names = std::move(vocabulary.predicates);
        ++version;
    }

    void set_predicates(
        const std::vector<std::string>& predicates) override
    {
        if (!live)
        {
            throw std::logic_error("live predicates unavailable");
        }
        names = predicates;
        ++version;
    }

    std::uint64_t vocabulary_version() const noexcept override
    {
        return version;
    }

    const std::vector<std::string>&
    predicates() const noexcept override
    {
        return names;
    }

    relation::RelationFrame infer(
        const image::ImageView&,
        const std::vector<relation::Region>& regions) override
    {
        ++infer_calls;
        last_regions = regions;

        relation::RelationFrame frame;
        frame.image_width = 640;
        frame.image_height = 480;
        frame.vocabulary_version = version;
        if (regions.size() >= 2U)
        {
            frame.edges.push_back({
                0U,
                1U,
                0U,
                0.9F,
                regions[0].track_id,
                regions[1].track_id,
            });
        }
        return frame;
    }
};

yolo::DetectionFrame detector_frame()
{
    yolo::DetectionFrame frame;
    frame.image_width = 640;
    frame.image_height = 480;
    frame.detections = {
        {{10.0F, 20.0F, 110.0F, 220.0F}, 0.95F, 0},
        {{200.0F, 100.0F, 340.0F, 300.0F}, 0.92F, 1},
    };
    return frame;
}

pipelines::SceneGraphPipelineOptions engine_options()
{
    pipelines::SceneGraphPipelineOptions options;
    options.tracking.minimum_consecutive_frames = 1;
    options.tracking.track_activation_threshold = 0.5F;
    options.tracking.high_conf_det_threshold = 0.5F;
    return options;
}

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

spec("scene graph production engine")
{
    it("runs detector through real ByteTrack into relation regions")
    {
        auto detector = std::make_unique<FakeDetector>();
        auto relation_model = std::make_unique<FakeRelation>();
        FakeDetector* detector_view = detector.get();
        FakeRelation* relation_view = relation_model.get();
        detector_view->frame = detector_frame();

        pipelines::detail::SceneGraphEngine engine(
            std::move(detector),
            std::move(relation_model),
            engine_options());

        image::ImageView image;

        // ByteTrack exposes the first observation as tentative even when
        // minimum_consecutive_frames=1. The second consistent observation
        // receives the persistent track identity.
        const auto tentative = engine.process(image);
        check(detector_view->calls == std::size_t{1U});
        check(relation_view->infer_calls == std::size_t{1U});
        check(tentative.objects.detections.size() == std::size_t{2U});
        check_false(
            tentative.objects.detections[0].track_id.has_value());
        check_false(
            tentative.objects.detections[1].track_id.has_value());

        const auto scene = engine.process(image);

        check(detector_view->calls == std::size_t{2U});
        check(relation_view->infer_calls == std::size_t{2U});
        check(relation_view->last_regions.size() == std::size_t{2U});
        check(relation_view->last_regions[0].track_id.has_value());
        check(relation_view->last_regions[1].track_id.has_value());
        check(scene.objects.detections.size() == std::size_t{2U});
        check(scene.relations.edges.size() == std::size_t{1U});
        check(scene.relations.edges[0].subject_track_id ==
              scene.objects.detections[0].track_id);
        check(scene.relations.edges[0].object_track_id ==
              scene.objects.detections[1].track_id);
        check(scene.relations.vocabulary_version == std::uint64_t{7U});
    }

    it("reports production stage timings and cardinalities")
    {
        auto detector = std::make_unique<FakeDetector>();
        auto relation_model = std::make_unique<FakeRelation>();
        detector->frame = detector_frame();

        pipelines::detail::SceneGraphEngine engine(
            std::move(detector),
            std::move(relation_model),
            engine_options());

        const auto timed =
            engine.process_timed(image::ImageView{});

        check(
            timed.timing.detection_count ==
            std::size_t{2U});
        check(
            timed.timing.tracked_object_count ==
            std::size_t{2U});
        check(
            timed.timing.relation_edge_count ==
            std::size_t{1U});
        check(
            timed.frame.objects.detections.size() ==
            std::size_t{2U});
        check(
            timed.frame.relations.edges.size() ==
            std::size_t{1U});

        const double stages[] = {
            timed.timing.detector_ms,
            timed.timing.tracker_ms,
            timed.timing.region_prepare_ms,
            timed.timing.relation_ms,
            timed.timing.assembly_ms,
            timed.timing.total_ms,
        };
        for (const double value : stages)
        {
            check(std::isfinite(value));
            check(value >= 0.0);
            check(timed.timing.total_ms >= value);
        }
    }

    it("forwards dynamic and live vocabulary controls")
    {
        auto detector = std::make_unique<FakeDetector>();
        detector->frame = detector_frame();
        auto relation_model = std::make_unique<FakeRelation>();

        pipelines::detail::SceneGraphEngine engine(
            std::move(detector),
            std::move(relation_model),
            engine_options());

        check(engine.supports_dynamic_vocabulary());
        check(engine.supports_live_predicates());
        check(engine.vocabulary_version() == std::uint64_t{7U});
        check(engine.predicates()[0] == "interacting");

        relation::PredicateVocabulary vocabulary;
        vocabulary.predicates = {"holding"};
        engine.set_vocabulary(std::move(vocabulary));
        check(engine.vocabulary_version() == std::uint64_t{8U});
        check(engine.predicates()[0] == "holding");

        engine.set_predicates({"riding"});
        check(engine.vocabulary_version() == std::uint64_t{9U});
        check(engine.predicates()[0] == "riding");

        check(engine.tracking_epoch() == std::uint64_t{1U});
        check(engine.reset_tracking() == std::uint64_t{2U});
        check(engine.tracking_epoch() == std::uint64_t{2U});
        const auto scene = engine.process(image::ImageView{});
        check(scene.objects.tracking_epoch == std::uint64_t{2U});
        check(scene.objects.detections.size() == std::size_t{2U});
    }

    it("rejects tracked object counts above the relation ceiling")
    {
        auto detector = std::make_unique<FakeDetector>();
        detector->frame = detector_frame();
        auto relation_model = std::make_unique<FakeRelation>();
        relation_model->max_boxes_value = 1U;

        pipelines::detail::SceneGraphEngine engine(
            std::move(detector),
            std::move(relation_model),
            engine_options());

        check_throws_as(
            engine.process(image::ImageView{}),
            std::length_error);
    }
}

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
                decltype(std::declval<const Pipeline&>()
                             .supports_live_predicates()),
                bool>);
        static_assert(
            std::is_same_v<
                decltype(std::declval<Pipeline&>().set_vocabulary(
                    std::declval<relation::PredicateVocabulary>())),
                void>);
        static_assert(
            std::is_same_v<
                decltype(std::declval<Pipeline&>().set_predicates(
                    std::declval<const std::vector<std::string>&>())),
                void>);
        static_assert(
            std::is_same_v<
                decltype(std::declval<const Pipeline&>().predicates()),
                const std::vector<std::string>&>);
        static_assert(
            std::is_same_v<
                decltype(std::declval<Pipeline&>().process_timed(
                    std::declval<const image::ImageView&>())),
                pipelines::TimedSceneGraphFrame>);
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
