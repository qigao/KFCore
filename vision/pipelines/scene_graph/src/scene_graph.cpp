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

class YoloDetectionRunner final : public detail::DetectionRunner
{
public:
    explicit YoloDetectionRunner(
        std::unique_ptr<yolo::YoloDetector> detector)
        : detector_(std::move(detector))
    {
        if (!detector_)
        {
            throw std::invalid_argument(
                "SceneGraphPipeline detector must not be null");
        }
    }

    yolo::DetectionFrame
    detect(const image::ImageView& image) override
    {
        return detector_->detect(image);
    }

private:
    std::unique_ptr<yolo::YoloDetector> detector_;
};

template <typename Model>
class TypedRelationRunner final : public detail::RelationRunner
{
public:
    explicit TypedRelationRunner(
        std::unique_ptr<Model> model,
        std::unique_ptr<relation::PredicateTextEncoder> text_encoder = nullptr,
        std::unique_ptr<relation::PredicateRoutingGate> routing_gate = nullptr)
        : model_(std::move(model))
        , text_encoder_(std::move(text_encoder))
        , routing_gate_(std::move(routing_gate))
    {
        if (!model_)
        {
            throw std::invalid_argument(
                "SceneGraphPipeline relation model must not be null");
        }
        if constexpr (
            std::is_same_v<
                Model,
                relation::OpenVocabularyRelation>)
        {
            const bool has_text =
                static_cast<bool>(text_encoder_);
            const bool has_gate =
                static_cast<bool>(routing_gate_);
            if (has_text != has_gate)
            {
                throw std::invalid_argument(
                    "SceneGraphPipeline live predicate support requires "
                    "both text encoder and routing gate");
            }
            if (has_text &&
                (text_encoder_->embedding_dim() != model_->query_dim() ||
                 routing_gate_->embedding_dim() != model_->query_dim()))
            {
                throw std::invalid_argument(
                    "SceneGraphPipeline live vocabulary model dimensions "
                    "do not match");
            }
        }
        else if (text_encoder_ || routing_gate_)
        {
            throw std::invalid_argument(
                "fixed-vocabulary relation model cannot use live predicates");
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

    bool supports_live_predicates() const noexcept override
    {
        if constexpr (
            std::is_same_v<
                Model,
                relation::OpenVocabularyRelation>)
        {
            return static_cast<bool>(text_encoder_) &&
                static_cast<bool>(routing_gate_);
        }
        return false;
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

    void set_predicates(
        const std::vector<std::string>& predicates) override
    {
        if constexpr (
            std::is_same_v<
                Model,
                relation::OpenVocabularyRelation>)
        {
            if (!text_encoder_ || !routing_gate_)
            {
                throw std::logic_error(
                    "SceneGraphPipeline was not created with live predicate support");
            }
            relation::PredicateVocabulary candidate =
                text_encoder_->encode(predicates);
            candidate = routing_gate_->apply(
                std::move(candidate));
            model_->set_vocabulary(
                std::move(candidate));
        }
        else
        {
            (void)predicates;
            throw std::logic_error(
                "SceneGraphPipeline relation model has a fixed vocabulary");
        }
    }

    std::uint64_t
    vocabulary_version() const noexcept override
    {
        return model_->vocabulary_version();
    }

    const std::vector<std::string>&
    predicates() const noexcept override
    {
        return model_->predicates();
    }

    relation::RelationFrame
    infer(const image::ImageView& image,
          const std::vector<relation::Region>& regions) override
    {
        return model_->infer(image, regions);
    }

private:
    std::unique_ptr<Model> model_;
    std::unique_ptr<relation::PredicateTextEncoder> text_encoder_;
    std::unique_ptr<relation::PredicateRoutingGate> routing_gate_;
};

std::unique_ptr<detail::DetectionRunner>
make_detector_runner(
    std::unique_ptr<yolo::YoloDetector> detector)
{
    return std::make_unique<YoloDetectionRunner>(
        std::move(detector));
}

template <typename Model>
std::unique_ptr<detail::RelationRunner>
make_relation_runner(
    std::unique_ptr<Model> model,
    std::unique_ptr<relation::PredicateTextEncoder> text_encoder = nullptr,
    std::unique_ptr<relation::PredicateRoutingGate> routing_gate = nullptr)
{
    return std::make_unique<TypedRelationRunner<Model>>(
        std::move(model),
        std::move(text_encoder),
        std::move(routing_gate));
}

} // namespace

struct SceneGraphPipeline::Impl final
{
    Impl(
        std::unique_ptr<detail::DetectionRunner> detector,
        std::unique_ptr<detail::RelationRunner> relation_model,
        const SceneGraphPipelineOptions& options)
        : engine(
              std::move(detector),
              std::move(relation_model),
              options)
    {
    }

    detail::SceneGraphEngine engine;
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
        make_detector_runner(std::move(detector)),
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
        make_detector_runner(std::move(detector)),
        make_relation_runner(std::move(relation_model)),
        options);
    return std::unique_ptr<SceneGraphPipeline>(
        new SceneGraphPipeline(std::move(impl)));
}

std::unique_ptr<SceneGraphPipeline>
SceneGraphPipeline::create(
    std::unique_ptr<yolo::YoloDetector> detector,
    std::unique_ptr<relation::OpenVocabularyRelation> relation_model,
    std::unique_ptr<relation::PredicateTextEncoder> text_encoder,
    std::unique_ptr<relation::PredicateRoutingGate> routing_gate,
    const SceneGraphPipelineOptions& options)
{
    if (!detector || !relation_model ||
        !text_encoder || !routing_gate)
    {
        throw std::invalid_argument(
            "SceneGraphPipeline live vocabulary requires detector, relation, "
            "text encoder, and routing gate");
    }

    auto impl = std::make_unique<Impl>(
        make_detector_runner(std::move(detector)),
        make_relation_runner(
            std::move(relation_model),
            std::move(text_encoder),
            std::move(routing_gate)),
        options);
    return std::unique_ptr<SceneGraphPipeline>(
        new SceneGraphPipeline(std::move(impl)));
}

SceneGraphFrame SceneGraphPipeline::process(
    const image::ImageView& image)
{
    if (!impl_)
    {
        throw std::logic_error(
            "SceneGraphPipeline state is unavailable");
    }
    return impl_->engine.process(image);
}

TimedSceneGraphFrame SceneGraphPipeline::process_timed(
    const image::ImageView& image)
{
    if (!impl_)
    {
        throw std::logic_error(
            "SceneGraphPipeline state is unavailable");
    }
    return impl_->engine.process_timed(image);
}

bool SceneGraphPipeline::supports_dynamic_vocabulary() const noexcept
{
    return impl_ &&
        impl_->engine.supports_dynamic_vocabulary();
}

bool SceneGraphPipeline::supports_live_predicates() const noexcept
{
    return impl_ &&
        impl_->engine.supports_live_predicates();
}

void SceneGraphPipeline::set_vocabulary(
    relation::PredicateVocabulary vocabulary)
{
    if (!impl_)
    {
        throw std::logic_error(
            "SceneGraphPipeline state is unavailable");
    }
    impl_->engine.set_vocabulary(
        std::move(vocabulary));
}

void SceneGraphPipeline::set_predicates(
    const std::vector<std::string>& predicates)
{
    if (!impl_)
    {
        throw std::logic_error(
            "SceneGraphPipeline state is unavailable");
    }
    impl_->engine.set_predicates(predicates);
}

std::uint64_t
SceneGraphPipeline::vocabulary_version() const noexcept
{
    return impl_ ?
        impl_->engine.vocabulary_version() :
        0U;
}

const std::vector<std::string>&
SceneGraphPipeline::predicates() const noexcept
{
    static const std::vector<std::string> empty;
    return impl_ ?
        impl_->engine.predicates() :
        empty;
}

std::uint64_t
SceneGraphPipeline::tracking_epoch() const noexcept
{
    return impl_ ? impl_->engine.tracking_epoch() : 0U;
}

std::uint64_t SceneGraphPipeline::reset_tracking()
{
    if (!impl_)
    {
        throw std::logic_error(
            "SceneGraphPipeline state is unavailable");
    }
    return impl_->engine.reset_tracking();
}

} // namespace kfcore::pipelines
