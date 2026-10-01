#include "kfcore/pipelines/scene_graph.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

namespace kfcore::yolo
{

struct YoloDetector::Impl final
{
};

YoloDetector::~YoloDetector() = default;

} // namespace kfcore::yolo

namespace kfcore::relation
{

struct OpenVocabularyRelation::Impl final
{
};

OpenVocabularyRelation::~OpenVocabularyRelation() = default;

} // namespace kfcore::relation

namespace kfcore::pipelines
{

struct SceneGraphPipeline::Impl final
{
    std::uint64_t vocabulary_version = 7U;
    std::uint64_t tracking_epoch = 1U;
    std::vector<std::string> predicates {"interacting"};
};

SceneGraphPipeline::SceneGraphPipeline(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

SceneGraphPipeline::~SceneGraphPipeline() = default;

std::unique_ptr<SceneGraphPipeline>
SceneGraphPipeline::create(
    std::unique_ptr<yolo::YoloDetector>,
    std::unique_ptr<relation::OpenVocabularyRelation>,
    const SceneGraphPipelineOptions&)
{
    return std::unique_ptr<SceneGraphPipeline>(
        new SceneGraphPipeline(std::make_unique<Impl>()));
}

SceneGraphFrame SceneGraphPipeline::process(const image::ImageView&)
{
    if (!impl_)
    {
        throw std::logic_error("stub scene graph state unavailable");
    }

    SceneGraphFrame frame;
    frame.objects.image_width = 100;
    frame.objects.image_height = 100;
    frame.objects.tracking_epoch = impl_->tracking_epoch;
    frame.objects.detections = {
        {{{0.0F, 0.0F, 20.0F, 20.0F}, 0.95F, 0}, std::uint64_t{101}},
        {{{20.0F, 0.0F, 40.0F, 20.0F}, 0.90F, 1}, std::uint64_t{202}},
    };

    frame.relations.image_width = 100;
    frame.relations.image_height = 100;
    frame.relations.vocabulary_version = impl_->vocabulary_version;
    frame.relations.edges = {
        {0U, 1U, 0U, 1.0F, std::uint64_t{101}, std::uint64_t{202}},
    };
    return frame;
}

TimedSceneGraphFrame
SceneGraphPipeline::process_timed(const image::ImageView& image)
{
    SceneGraphFrame frame = process(image);
    SceneGraphTiming timing;
    timing.detection_count = frame.objects.detections.size();
    timing.tracked_object_count = frame.objects.detections.size();
    timing.relation_edge_count = frame.relations.edges.size();
    return {
        std::move(frame),
        timing,
    };
}

bool SceneGraphPipeline::supports_dynamic_vocabulary() const noexcept
{
    return impl_ != nullptr;
}

bool SceneGraphPipeline::supports_live_predicates() const noexcept
{
    return impl_ != nullptr;
}

void SceneGraphPipeline::set_vocabulary(
    relation::PredicateVocabulary vocabulary)
{
    if (!impl_)
    {
        throw std::logic_error("stub scene graph state unavailable");
    }
    impl_->predicates = std::move(vocabulary.predicates);
    ++impl_->vocabulary_version;
}

void SceneGraphPipeline::set_predicates(
    const std::vector<std::string>& predicates)
{
    if (!impl_)
    {
        throw std::logic_error("stub scene graph state unavailable");
    }
    impl_->predicates = predicates;
    ++impl_->vocabulary_version;
}

std::uint64_t SceneGraphPipeline::vocabulary_version() const noexcept
{
    return impl_ ? impl_->vocabulary_version : 0U;
}

const std::vector<std::string>&
SceneGraphPipeline::predicates() const noexcept
{
    static const std::vector<std::string> empty;
    return impl_ ? impl_->predicates : empty;
}

std::uint64_t SceneGraphPipeline::tracking_epoch() const noexcept
{
    return impl_ ? impl_->tracking_epoch : 0U;
}

std::uint64_t SceneGraphPipeline::reset_tracking()
{
    if (!impl_)
    {
        throw std::logic_error("stub scene graph state unavailable");
    }
    ++impl_->tracking_epoch;
    return impl_->tracking_epoch;
}

} // namespace kfcore::pipelines
