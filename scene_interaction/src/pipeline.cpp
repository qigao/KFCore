#include "kfcore/scene_interaction/pipeline.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace kfcore::scene_interaction
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
    TimedSceneBehaviorFrame timed =
        process_timed(image, seconds);
    return std::move(timed.frame);
}

TimedSceneBehaviorFrame SceneBehaviorPipeline::process_timed(
    const image::ImageView& image,
    double seconds)
{
    if (!impl_ || !impl_->scene_graph)
    {
        throw std::logic_error(
            "SceneBehaviorPipeline state is unavailable");
    }

    const auto total_begin = SteadyClock::now();
    pipelines::TimedSceneGraphFrame timed_scene =
        impl_->scene_graph->process_timed(image);

    const auto temporal_begin = SteadyClock::now();
    std::vector<SceneBehaviorEvent> events =
        impl_->interaction.process(
            timed_scene.frame,
            seconds);
    const auto temporal_end = SteadyClock::now();

    SceneBehaviorFrame frame {
        std::move(timed_scene.frame),
        std::move(events),
    };

    SceneBehaviorTiming timing;
    timing.scene_graph = timed_scene.timing;
    timing.temporal_ms = elapsed_ms(
        temporal_begin,
        temporal_end);
    timing.total_ms = elapsed_ms(
        total_begin,
        temporal_end);
    timing.event_count = frame.events.size();
    timing.pair_state_count =
        impl_->interaction.pair_state_count();

    return {
        std::move(frame),
        timing,
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

std::vector<SceneBehaviorEvent>
SceneBehaviorPipeline::reset_tracking(double seconds)
{
    if (!impl_ || !impl_->scene_graph)
    {
        throw std::logic_error(
            "SceneBehaviorPipeline state is unavailable");
    }
    const std::uint64_t epoch =
        impl_->scene_graph->reset_tracking();
    return impl_->interaction.reset_tracking_epoch(
        epoch,
        seconds);
}

std::uint64_t
SceneBehaviorPipeline::tracking_epoch() const noexcept
{
    return (
        impl_ && impl_->scene_graph
    ) ? impl_->scene_graph->tracking_epoch() : 0U;
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
