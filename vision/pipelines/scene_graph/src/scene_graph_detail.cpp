#include "scene_graph_detail.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kfcore::pipelines::detail
{
namespace
{

using SteadyClock = std::chrono::steady_clock;

double elapsed_ms(
    SteadyClock::time_point begin,
    SteadyClock::time_point end)
{
    return std::chrono::duration<double, std::milli>(
        end - begin).count();
}

} // namespace

SceneGraphEngine::SceneGraphEngine(
    std::unique_ptr<DetectionRunner> detector,
    std::unique_ptr<RelationRunner> relation_model,
    const SceneGraphPipelineOptions& options)
    : detector_(std::move(detector))
    , relation_model_(std::move(relation_model))
    , tracking_(options.tracking)
{
    if (!detector_ || !relation_model_)
    {
        throw std::invalid_argument(
            "SceneGraphEngine requires detector and relation runners");
    }
}

SceneGraphFrame SceneGraphEngine::process(
    const image::ImageView& image)
{
    TimedSceneGraphFrame timed =
        process_timed(image);
    return std::move(timed.frame);
}

TimedSceneGraphFrame SceneGraphEngine::process_timed(
    const image::ImageView& image)
{
    const auto total_begin = SteadyClock::now();

    const auto detector_begin = total_begin;
    yolo::DetectionFrame detections =
        detector_->detect(image);
    const auto detector_end = SteadyClock::now();

    const std::size_t detection_count =
        detections.detections.size();

    const auto tracker_begin = detector_end;
    yolo::TrackFrame tracks =
        tracking_.update(detections);
    const auto tracker_end = SteadyClock::now();

    if (tracks.detections.size() >
        relation_model_->max_boxes())
    {
        throw std::length_error(
            "SceneGraphPipeline tracked object count exceeds relation max_boxes");
    }
    const std::size_t tracked_object_count =
        tracks.detections.size();

    const auto region_begin = tracker_end;
    const std::vector<relation::Region> regions =
        regions_from_tracks(tracks);
    const auto region_end = SteadyClock::now();

    const auto relation_begin = region_end;
    relation::RelationFrame relations =
        relation_model_->infer(
            image,
            regions);
    const auto relation_end = SteadyClock::now();

    const std::size_t relation_edge_count =
        relations.edges.size();

    const auto assembly_begin = relation_end;
    SceneGraphFrame frame =
        assemble_scene_graph(
            std::move(tracks),
            std::move(relations));
    const auto assembly_end = SteadyClock::now();

    SceneGraphTiming timing;
    timing.detector_ms = elapsed_ms(
        detector_begin,
        detector_end);
    timing.tracker_ms = elapsed_ms(
        tracker_begin,
        tracker_end);
    timing.region_prepare_ms = elapsed_ms(
        region_begin,
        region_end);
    timing.relation_ms = elapsed_ms(
        relation_begin,
        relation_end);
    timing.assembly_ms = elapsed_ms(
        assembly_begin,
        assembly_end);
    timing.total_ms = elapsed_ms(
        total_begin,
        assembly_end);
    timing.detection_count = detection_count;
    timing.tracked_object_count = tracked_object_count;
    timing.relation_edge_count = relation_edge_count;

    return {
        std::move(frame),
        timing,
    };
}

bool SceneGraphEngine::supports_dynamic_vocabulary() const noexcept
{
    return relation_model_ &&
        relation_model_->supports_dynamic_vocabulary();
}

bool SceneGraphEngine::supports_live_predicates() const noexcept
{
    return relation_model_ &&
        relation_model_->supports_live_predicates();
}

void SceneGraphEngine::set_vocabulary(
    relation::PredicateVocabulary vocabulary)
{
    if (!relation_model_)
    {
        throw std::logic_error(
            "SceneGraphEngine relation state is unavailable");
    }
    relation_model_->set_vocabulary(
        std::move(vocabulary));
}

void SceneGraphEngine::set_predicates(
    const std::vector<std::string>& predicates)
{
    if (!relation_model_)
    {
        throw std::logic_error(
            "SceneGraphEngine relation state is unavailable");
    }
    relation_model_->set_predicates(predicates);
}

std::uint64_t
SceneGraphEngine::vocabulary_version() const noexcept
{
    return relation_model_ ?
        relation_model_->vocabulary_version() :
        0U;
}

const std::vector<std::string>&
SceneGraphEngine::predicates() const noexcept
{
    static const std::vector<std::string> empty;
    return relation_model_ ?
        relation_model_->predicates() :
        empty;
}

std::uint64_t
SceneGraphEngine::tracking_epoch() const noexcept
{
    return tracking_.tracking_epoch();
}

std::uint64_t SceneGraphEngine::reset_tracking()
{
    return tracking_.reset();
}

std::vector<relation::Region>
regions_from_tracks(const yolo::TrackFrame& tracks)
{
    std::vector<relation::Region> regions;
    regions.reserve(tracks.detections.size());
    for (const yolo::TrackedDetection& tracked : tracks.detections)
    {
        const yolo::Detection& detection = tracked.detection;
        regions.push_back({
            detection.box.left,
            detection.box.top,
            detection.box.right,
            detection.box.bottom,
            detection.score,
            tracked.track_id,
        });
    }
    return regions;
}

SceneGraphFrame assemble_scene_graph(yolo::TrackFrame tracks,
                                     relation::RelationFrame relations)
{
    if (tracks.image_width != relations.image_width ||
        tracks.image_height != relations.image_height)
    {
        throw std::runtime_error(
            "SceneGraphPipeline relation frame dimensions do not match tracked objects");
    }

    for (const relation::RelationEdge& edge : relations.edges)
    {
        if (edge.subject_index >= tracks.detections.size() ||
            edge.object_index >= tracks.detections.size())
        {
            throw std::runtime_error(
                "SceneGraphPipeline relation edge references an unavailable object");
        }

        const auto& subject = tracks.detections[edge.subject_index].track_id;
        const auto& object = tracks.detections[edge.object_index].track_id;
        if (edge.subject_track_id != subject ||
            edge.object_track_id != object)
        {
            throw std::runtime_error(
                "SceneGraphPipeline relation edge track identity does not match object table");
        }
    }

    return {std::move(tracks), std::move(relations)};
}

} // namespace kfcore::pipelines::detail
