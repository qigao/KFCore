#include "kfcore/scene_interaction/pipeline.hpp"

#include <stdexcept>
#include <utility>

namespace kfcore::scene_interaction
{

struct SceneBehaviorPipeline::Impl final
{
    Impl(std::unique_ptr<pipelines::SceneGraphPipeline> scene_graph_value,
         const SceneInteractionOptions& interaction_options)
        : scene_graph(std::move(scene_graph_value))
        , interaction(interaction_options)
    {
        if (!scene_graph)
        {
            throw std::invalid_argument(
                "SceneBehaviorPipeline requires a SceneGraphPipeline");
        }
    }

    std::unique_ptr<pipelines::SceneGraphPipeline> scene_graph;
    SceneInteraction interaction;
};

SceneBehaviorPipeline::SceneBehaviorPipeline(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

SceneBehaviorPipeline::~SceneBehaviorPipeline() = default;

std::unique_ptr<SceneBehaviorPipeline>
SceneBehaviorPipeline::create(
    std::unique_ptr<pipelines::SceneGraphPipeline> scene_graph,
    const SceneInteractionOptions& interaction_options)
{
    return std::unique_ptr<SceneBehaviorPipeline>(
        new SceneBehaviorPipeline(
            std::make_unique<Impl>(
                std::move(scene_graph),
                interaction_options)));
}

std::vector<SceneBehaviorEvent>
SceneBehaviorPipeline::configure_model(
    const SceneBehaviorModel& model,
    double seconds)
{
    if (!impl_)
    {
        throw std::logic_error(
            "SceneBehaviorPipeline state is unavailable");
    }
    return impl_->interaction.configure_model(
        model,
        seconds);
}

SceneBehaviorFrame SceneBehaviorPipeline::process(
    const image::ImageView& image,
    double seconds)
{
    if (!impl_ || !impl_->scene_graph)
    {
        throw std::logic_error(
            "SceneBehaviorPipeline state is unavailable");
    }

    pipelines::SceneGraphFrame scene =
        impl_->scene_graph->process(image);
    std::vector<SceneBehaviorEvent> events =
        impl_->interaction.process(
            scene,
            seconds);

    return {
        std::move(scene),
        std::move(events),
    };
}

std::vector<SceneBehaviorEvent>
SceneBehaviorPipeline::advance(double seconds)
{
    if (!impl_)
    {
        throw std::logic_error(
            "SceneBehaviorPipeline state is unavailable");
    }
    return impl_->interaction.advance(seconds);
}

std::vector<SceneBehaviorEvent>
SceneBehaviorPipeline::reset_temporal(double seconds)
{
    if (!impl_)
    {
        throw std::logic_error(
            "SceneBehaviorPipeline state is unavailable");
    }
    return impl_->interaction.reset(seconds);
}

void SceneBehaviorPipeline::reset_tracking() noexcept
{
    if (impl_ && impl_->scene_graph)
    {
        impl_->scene_graph->reset_tracking();
    }
}

bool SceneBehaviorPipeline::supports_dynamic_vocabulary() const noexcept
{
    return impl_ &&
        impl_->scene_graph &&
        impl_->scene_graph->supports_dynamic_vocabulary();
}

bool SceneBehaviorPipeline::supports_live_predicates() const noexcept
{
    return impl_ &&
        impl_->scene_graph &&
        impl_->scene_graph->supports_live_predicates();
}

void SceneBehaviorPipeline::set_vocabulary(
    relation::PredicateVocabulary vocabulary)
{
    if (!impl_ || !impl_->scene_graph)
    {
        throw std::logic_error(
            "SceneBehaviorPipeline state is unavailable");
    }
    impl_->scene_graph->set_vocabulary(
        std::move(vocabulary));
}

void SceneBehaviorPipeline::set_predicates(
    const std::vector<std::string>& predicates)
{
    if (!impl_ || !impl_->scene_graph)
    {
        throw std::logic_error(
            "SceneBehaviorPipeline state is unavailable");
    }
    impl_->scene_graph->set_predicates(predicates);
}

std::uint64_t
SceneBehaviorPipeline::vocabulary_version() const noexcept
{
    return (
        impl_ && impl_->scene_graph
    ) ? impl_->scene_graph->vocabulary_version() : 0U;
}

const std::vector<std::string>&
SceneBehaviorPipeline::predicates() const noexcept
{
    static const std::vector<std::string> empty;
    return (
        impl_ && impl_->scene_graph
    ) ? impl_->scene_graph->predicates() : empty;
}

bool SceneBehaviorPipeline::temporal_configured() const noexcept
{
    return impl_ && impl_->interaction.configured();
}

std::size_t
SceneBehaviorPipeline::pair_state_count() const noexcept
{
    return impl_ ?
        impl_->interaction.pair_state_count() :
        0U;
}

} // namespace kfcore::scene_interaction
