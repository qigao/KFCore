#include "scene_graph_detail.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

namespace kfcore::pipelines::detail
{

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
