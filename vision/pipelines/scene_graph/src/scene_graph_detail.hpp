#pragma once

#include "kfcore/pipelines/scene_graph.hpp"

#include <vector>

namespace kfcore::pipelines::detail
{

[[nodiscard]] std::vector<relation::Region>
regions_from_tracks(const yolo::TrackFrame& tracks);

[[nodiscard]] SceneGraphFrame
assemble_scene_graph(yolo::TrackFrame tracks,
                     relation::RelationFrame relations);

} // namespace kfcore::pipelines::detail
