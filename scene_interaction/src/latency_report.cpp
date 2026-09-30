#include "kfcore/scene_interaction/latency_report.hpp"

#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace kfcore::scene_interaction
{
namespace
{

void require_timing(double value, const char* name)
{
    if (!std::isfinite(value) || value < 0.0)
    {
        throw std::invalid_argument(
            std::string("SceneBehaviorTiming ") +
            name +
            " must be finite and non-negative");
    }
}

} // namespace

std::string scene_behavior_timing_json(
    const SceneBehaviorTiming& timing)
{
    require_timing(
        timing.scene_graph.detector_ms,
        "detector_ms");
    require_timing(
        timing.scene_graph.tracker_ms,
        "tracker_ms");
    require_timing(
        timing.scene_graph.region_prepare_ms,
        "region_prepare_ms");
    require_timing(
        timing.scene_graph.relation_ms,
        "relation_ms");
    require_timing(
        timing.scene_graph.assembly_ms,
        "assembly_ms");
    require_timing(
        timing.scene_graph.total_ms,
        "scene_graph.total_ms");
    require_timing(
        timing.temporal_ms,
        "temporal_ms");
    require_timing(
        timing.total_ms,
        "total_ms");

    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision(17)
           << "{"
           << "\"schema\":\""
           << kSceneBehaviorTimingSampleSchema
           << "\","
           << "\"detector_ms\":"
           << timing.scene_graph.detector_ms
           << ",\"tracker_ms\":"
           << timing.scene_graph.tracker_ms
           << ",\"region_prepare_ms\":"
           << timing.scene_graph.region_prepare_ms
           << ",\"relation_ms\":"
           << timing.scene_graph.relation_ms
           << ",\"assembly_ms\":"
           << timing.scene_graph.assembly_ms
           << ",\"scene_graph_total_ms\":"
           << timing.scene_graph.total_ms
           << ",\"temporal_ms\":"
           << timing.temporal_ms
           << ",\"total_ms\":"
           << timing.total_ms
           << ",\"detection_count\":"
           << timing.scene_graph.detection_count
           << ",\"tracked_object_count\":"
           << timing.scene_graph.tracked_object_count
           << ",\"relation_edge_count\":"
           << timing.scene_graph.relation_edge_count
           << ",\"event_count\":"
           << timing.event_count
           << ",\"pair_state_count\":"
           << timing.pair_state_count
           << "}";
    return stream.str();
}

} // namespace kfcore::scene_interaction
