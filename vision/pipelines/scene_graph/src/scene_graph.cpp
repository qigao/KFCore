#include "kfcore/pipelines/scene_graph.hpp"

#include "scene_graph_detail.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

namespace kfcore::pipelines
{
struct SceneGraphPipeline::Impl final
{
    Impl(std::unique_ptr<yolo::YoloDetector> detector_value,
         std::unique_ptr<relation::RelateAnything> relation_value,
         const SceneGraphPipelineOptions& options)
        : detector(std::move(detector_value))
        , relation_model(std::move(relation_value))
        , tracking(options.tracking)
    {
    }

    std::unique_ptr<yolo::YoloDetector> detector;
    std::unique_ptr<relation::RelateAnything> relation_model;
    yolo::ByteTrackSession tracking;
};

SceneGraphPipeline::SceneGraphPipeline(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

SceneGraphPipeline::~SceneGraphPipeline() = default;

std::unique_ptr<SceneGraphPipeline>
SceneGraphPipeline::create(
    std::unique_ptr<yolo::YoloDetector> detector,
    std::unique_ptr<relation::RelateAnything> relation_model,
    const SceneGraphPipelineOptions& options)
{
    if (!detector || !relation_model)
    {
        throw std::invalid_argument(
            "SceneGraphPipeline requires detector and relation model instances");
    }

    auto impl = std::make_unique<Impl>(
        std::move(detector), std::move(relation_model), options);
    return std::unique_ptr<SceneGraphPipeline>(
        new SceneGraphPipeline(std::move(impl)));
}

SceneGraphFrame SceneGraphPipeline::process(const image::ImageView& image)
{
    if (!impl_ || !impl_->detector || !impl_->relation_model)
    {
        throw std::logic_error("SceneGraphPipeline state is unavailable");
    }

    const yolo::DetectionFrame detections = impl_->detector->detect(image);
    yolo::TrackFrame tracks = impl_->tracking.update(detections);

    if (tracks.detections.size() > impl_->relation_model->max_boxes())
    {
        throw std::length_error(
            "SceneGraphPipeline tracked object count exceeds relation max_boxes");
    }

    const std::vector<relation::Region> regions =
        detail::regions_from_tracks(tracks);
    relation::RelationFrame relations =
        impl_->relation_model->infer(image, regions);

    return detail::assemble_scene_graph(
        std::move(tracks), std::move(relations));
}

void SceneGraphPipeline::reset_tracking() noexcept
{
    if (impl_)
    {
        impl_->tracking.reset();
    }
}

} // namespace kfcore::pipelines
