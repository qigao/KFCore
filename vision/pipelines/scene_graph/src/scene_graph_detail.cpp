#include "scene_graph_detail.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

namespace kfcore::pipelines::detail
{

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
    const yolo::DetectionFrame detections =
        detector_->detect(image);
    yolo::TrackFrame tracks =
        tracking_.update(detections);

    if (tracks.detections.size() >
        relation_model_->max_boxes())
    {
        throw std::length_error(
            "SceneGraphPipeline tracked object count exceeds relation max_boxes");
    }

    const std::vector<relation::Region> regions =
        regions_from_tracks(tracks);
    relation::RelationFrame relations =
        relation_model_->infer(
            image,
            regions);

    return assemble_scene_graph(
        std::move(tracks),
        std::move(relations));
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

void SceneGraphEngine::reset_tracking() noexcept
{
    tracking_.reset();
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
