#include "scene_features.hpp"

#include "kfcore/scene_interaction/interaction.hpp"
#include "tinytest.hpp"

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

using namespace kfcore;

namespace
{

pipelines::SceneGraphFrame make_frame(
    float relation_score = 1.0F,
    std::optional<std::uint64_t> subject_track = std::uint64_t{101},
    std::optional<std::uint64_t> object_track = std::uint64_t{202})
{
    pipelines::SceneGraphFrame frame;
    frame.objects.image_width = 100;
    frame.objects.image_height = 100;
    frame.objects.detections = {
        {{{0.0F, 0.0F, 20.0F, 20.0F}, 0.95F, 0}, subject_track},
        {{{20.0F, 0.0F, 40.0F, 20.0F}, 0.90F, 1}, object_track},
    };

    frame.relations.image_width = 100;
    frame.relations.image_height = 100;
    frame.relations.edges = {
        {0U, 1U, 0U, relation_score, subject_track, object_track},
    };
    return frame;
}

scene_interaction::SceneBehaviorModel make_model()
{
    scene_interaction::SceneBehaviorModel model;
    model.predicate_count = 1U;
    model.reservoir_size = 1;
    model.leak_rate = 1.0F;
    model.neutral_index = 0U;
    model.labels = {"none", "interacting"};

    model.input_weights.assign(
        model.predicate_count +
            scene_interaction::kSceneGeometryFeatureCount,
        0.0F);
    model.input_weights[0] = 1.0F;
    model.recurrent_weights = {0.0F};
    model.reservoir_bias = {0.0F};
    model.output_weights = {0.0F, 1.0F};
    model.output_bias = {0.0F, 0.0F};
    return model;
}

scene_interaction::SceneInteractionOptions make_options()
{
    scene_interaction::SceneInteractionOptions options;
    options.minimum_score = 0.5F;
    options.minimum_margin = 0.2F;
    options.confirmation_seconds = 0.10;
    options.end_seconds = 0.10;
    options.maximum_gap_seconds = 0.50;
    options.max_pair_states = 4U;
    return options;
}

} // namespace

spec("scene graph temporal feature encoding")
{
    it("encodes relation confidence and normalized pair geometry")
    {
        const auto observations =
            scene_interaction::detail::encode_pair_observations(
                make_frame(0.8F), 1U);

        check(observations.size() == std::size_t{1U});
        const auto& observation = observations.front();
        check(observation.pair.subject_track_id == std::uint64_t{101});
        check(observation.pair.object_track_id == std::uint64_t{202});
        check(observation.subject_class_id == std::int32_t{0});
        check(observation.object_class_id == std::int32_t{1});
        check(observation.values.size() == std::size_t{7U});
        check(std::fabs(observation.values[0] - 0.8F) < 1.0e-6F);
        check(std::fabs(observation.values[1] - 0.2F) < 1.0e-6F);
        check(std::fabs(observation.values[2]) < 1.0e-6F);
        check(std::fabs(observation.values[3] -
                        (0.2F / std::sqrt(2.0F))) < 1.0e-6F);
        check(std::fabs(observation.values[4] - 0.04F) < 1.0e-6F);
        check(std::fabs(observation.values[5] - 0.04F) < 1.0e-6F);
        check(std::fabs(observation.values[6]) < 1.0e-6F);
    }

    it("keeps untracked graph edges out of persistent temporal state")
    {
        const auto frame =
            make_frame(0.8F, std::nullopt, std::uint64_t{202});
        const auto observations =
            scene_interaction::detail::encode_pair_observations(frame, 1U);
        check(observations.empty());
    }

    it("rejects relation identity drift from the object table")
    {
        auto frame = make_frame();
        frame.relations.edges[0].subject_track_id = std::uint64_t{999};

        check_throws_as(
            scene_interaction::detail::encode_pair_observations(frame, 1U),
            std::invalid_argument);
    }
}

spec("pair-centric scene ESN behavior lifecycle")
{
    it("starts and ends a behavior from repeated tracked-pair evidence")
    {
        scene_interaction::SceneInteraction interaction(make_options());
        const auto configured =
            interaction.configure_model(make_model(), 0.0);
        check(configured.empty());

        const auto first = interaction.process(make_frame(1.0F), 0.0);
        check(first.empty());
        check(interaction.pair_state_count() == std::size_t{1U});

        const auto second = interaction.process(make_frame(1.0F), 0.20);
        check(second.size() == std::size_t{1U});
        check(second[0].kind ==
              scene_interaction::SceneBehaviorEventKind::BehaviorStarted);
        check(second[0].reason ==
              scene_interaction::SceneBehaviorEventReason::Recognized);
        check(second[0].behavior_index == std::size_t{1U});
        check(second[0].pair.subject_track_id == std::uint64_t{101});
        check(second[0].pair.object_track_id == std::uint64_t{202});

        const auto ended = interaction.process(make_frame(0.0F), 0.40);
        check(ended.size() == std::size_t{1U});
        check(ended[0].kind ==
              scene_interaction::SceneBehaviorEventKind::BehaviorEnded);
        check(ended[0].reason ==
              scene_interaction::SceneBehaviorEventReason::Unrecognized);
    }

    it("cancels active behavior when pair evidence disappears beyond the gap")
    {
        scene_interaction::SceneInteraction interaction(make_options());
        (void)interaction.configure_model(make_model(), 0.0);
        (void)interaction.process(make_frame(1.0F), 0.0);
        const auto started = interaction.process(make_frame(1.0F), 0.20);
        check(started.size() == std::size_t{1U});

        const auto cancelled = interaction.advance(0.80);
        check(cancelled.size() == std::size_t{1U});
        check(cancelled[0].kind ==
              scene_interaction::SceneBehaviorEventKind::BehaviorCancelled);
        check(cancelled[0].reason ==
              scene_interaction::SceneBehaviorEventReason::PairLost);
        check(interaction.pair_state_count() == std::size_t{0U});
    }

    it("treats a global frame gap as continuity break before new evidence")
    {
        scene_interaction::SceneInteraction interaction(make_options());
        (void)interaction.configure_model(make_model(), 0.0);
        (void)interaction.process(make_frame(1.0F), 0.0);
        (void)interaction.process(make_frame(1.0F), 0.20);

        const auto events = interaction.process(make_frame(1.0F), 0.80);
        check(events.size() == std::size_t{1U});
        check(events[0].kind ==
              scene_interaction::SceneBehaviorEventKind::BehaviorCancelled);
        check(events[0].reason ==
              scene_interaction::SceneBehaviorEventReason::FrameGap);
        check(interaction.pair_state_count() == std::size_t{1U});
    }

    it("rejects pair-state capacity transactionally")
    {
        auto options = make_options();
        options.max_pair_states = 1U;
        scene_interaction::SceneInteraction interaction(options);
        (void)interaction.configure_model(make_model(), 0.0);

        auto frame = make_frame();
        frame.objects.detections.push_back(
            {{{50.0F, 0.0F, 70.0F, 20.0F}, 0.85F, 2},
             std::uint64_t{303}});
        frame.relations.edges.push_back(
            {0U, 2U, 0U, 1.0F, std::uint64_t{101},
             std::uint64_t{303}});

        check_throws_as(interaction.process(frame, 0.0),
                        std::length_error);
        check(interaction.pair_state_count() == std::size_t{0U});
    }

    it("cancels active behavior when the model is replaced")
    {
        scene_interaction::SceneInteraction interaction(make_options());
        (void)interaction.configure_model(make_model(), 0.0);
        (void)interaction.process(make_frame(1.0F), 0.0);
        (void)interaction.process(make_frame(1.0F), 0.20);

        const auto events =
            interaction.configure_model(make_model(), 0.20);
        check(events.size() == std::size_t{1U});
        check(events[0].kind ==
              scene_interaction::SceneBehaviorEventKind::BehaviorCancelled);
        check(events[0].reason ==
              scene_interaction::SceneBehaviorEventReason::ModelChanged);
        check(interaction.pair_state_count() == std::size_t{0U});
    }
}
