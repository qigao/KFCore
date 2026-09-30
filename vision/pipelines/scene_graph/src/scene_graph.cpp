#include "kfcore/pipelines/scene_graph.hpp"

#include "scene_graph_detail.hpp"

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace kfcore::pipelines
{
namespace
{

class RelationRunner
{
public:
    virtual ~RelationRunner() = default;

    [[nodiscard]] virtual std::size_t max_boxes() const noexcept = 0;
    [[nodiscard]] virtual bool supports_dynamic_vocabulary() const noexcept = 0;
    virtual void set_vocabulary(relation::PredicateVocabulary vocabulary) = 0;
    [[nodiscard]] virtual std::uint64_t vocabulary_version() const noexcept = 0;
    [[nodiscard]] virtual relation::RelationFrame
    infer(const image::ImageView& image,
          const std::vector<relation::Region>& regions) = 0;
};

template <typename Model>
class TypedRelationRunner final : public RelationRunner
{
public:
    explicit TypedRelationRunner(std::unique_ptr<Model> model)
        : model_(std::move(model))
    {
        if (!model_)
        {
            throw std::invalid_argument(
                "SceneGraphPipeline relation model must not be null");
        }
    }

    std::size_t max_boxes() const noexcept override
    {
        return model_->max_boxes();
    }

    bool supports_dynamic_vocabulary() const noexcept override
    {
        return std::is_same_v<
            Model,
            relation::OpenVocabularyRelation>;
    }

    void set_vocabulary(
        relation::PredicateVocabulary vocabulary) override
    {
        if constexpr (
            std::is_same_v<
                Model,
                relation::OpenVocabularyRelation>)
        {
            model_->set_vocabulary(
                std::move(vocabulary));
        }
        else
        {
            throw std::logic_error(
                "SceneGraphPipeline relation model has a fixed vocabulary");
        }
    }

    std::uint64_t vocabulary_version() const noexcept override
    {
        if constexpr (
            std::is_same_v<
                Model,
                relation::OpenVocabularyRelation>)
        {
            return model_->vocabulary_version();
        }
        return 0U;
    }

    relation::RelationFrame
    infer(const image::ImageView& image,
          const std::vector<relation::Region>& regions) override
    {
        return model_->infer(image, regions);
    }

private:
    std::unique_ptr<Model> model_;
};

template <typename Model>
std::unique_ptr<RelationRunner>
make_relation_runner(std::unique_ptr<Model> model)
{
    return std::make_unique<TypedRelationRunner<Model>>(
        std::move(model));
}

} // namespace
struct SceneGraphPipeline::Impl final
{
    Impl(std::unique_ptr<yolo::YoloDetector> detector_value,
         std::unique_ptr<RelationRunner> relation_value,
         const SceneGraphPipelineOptions& options)
        : detector(std::move(detector_value))
        , relation_model(std::move(relation_value))
        , tracking(options.tracking)
    {
    }

    std::unique_ptr<yolo::YoloDetector> detector;
    std::unique_ptr<RelationRunner> relation_model;
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
        std::move(detector),
        make_relation_runner(std::move(relation_model)),
        options);
    return std::unique_ptr<SceneGraphPipeline>(
        new SceneGraphPipeline(std::move(impl)));
}

std::unique_ptr<SceneGraphPipeline>
SceneGraphPipeline::create(
    std::unique_ptr<yolo::YoloDetector> detector,
    std::unique_ptr<relation::OpenVocabularyRelation> relation_model,
    const SceneGraphPipelineOptions& options)
{
    if (!detector || !relation_model)
    {
        throw std::invalid_argument(
            "SceneGraphPipeline requires detector and relation model instances");
    }

    auto impl = std::make_unique<Impl>(
        std::move(detector),
        make_relation_runner(std::move(relation_model)),
        options);
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

bool SceneGraphPipeline::supports_dynamic_vocabulary() const noexcept
{
    return impl_ &&
        impl_->relation_model &&
        impl_->relation_model->supports_dynamic_vocabulary();
}

void SceneGraphPipeline::set_vocabulary(
    relation::PredicateVocabulary vocabulary)
{
    if (!impl_ || !impl_->relation_model)
    {
        throw std::logic_error(
            "SceneGraphPipeline state is unavailable");
    }
    impl_->relation_model->set_vocabulary(
        std::move(vocabulary));
}

std::uint64_t
SceneGraphPipeline::vocabulary_version() const noexcept
{
    if (!impl_ || !impl_->relation_model)
    {
        return 0U;
    }
    return impl_->relation_model->vocabulary_version();
}

void SceneGraphPipeline::reset_tracking() noexcept
{
    if (impl_)
    {
        impl_->tracking.reset();
    }
}

} // namespace kfcore::pipelines
