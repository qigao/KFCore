#include "kfcore/scene_interaction/interaction.hpp"

#include "scene_features.hpp"

#include <esn.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kfcore::scene_interaction
{
namespace
{

std::size_t checked_product(std::size_t left, std::size_t right,
                            const char* subject)
{
    if (left != 0U &&
        right > (std::numeric_limits<std::size_t>::max)() / left)
    {
        throw std::length_error(std::string(subject) + " size overflow");
    }
    return left * right;
}

bool positive(double value) noexcept
{
    return std::isfinite(value) && value > 0.0;
}

void validate_options(const SceneInteractionOptions& options)
{
    constexpr std::size_t kMaximumPairStates = 4096U;
    if (!std::isfinite(options.minimum_score) ||
        !std::isfinite(options.minimum_margin) ||
        options.minimum_margin < 0.0F ||
        !positive(options.confirmation_seconds) ||
        !positive(options.end_seconds) ||
        !positive(options.maximum_gap_seconds) ||
        options.max_pair_states == 0U ||
        options.max_pair_states > kMaximumPairStates)
    {
        throw std::invalid_argument("invalid scene interaction options");
    }
}

void validate_finite(const std::vector<float>& values, const char* subject)
{
    for (float value : values)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument(
                std::string(subject) + " contains non-finite values");
        }
    }
}

void validate_model(const SceneBehaviorModel& model)
{
    constexpr std::size_t kMaximumPredicates = 4096U;
    constexpr int kMaximumReservoirSize = 4096;
    constexpr std::size_t kMaximumLabels = 256U;
    const std::size_t input_size =
        detail::scene_input_size(model.predicate_count);
    if (model.predicate_count > kMaximumPredicates ||
        input_size >
            static_cast<std::size_t>((std::numeric_limits<int>::max)()) ||
        model.reservoir_size <= 0 ||
        model.reservoir_size > kMaximumReservoirSize ||
        !std::isfinite(model.leak_rate) ||
        model.leak_rate <= 0.0F || model.leak_rate > 1.0F ||
        model.labels.size() < 2U ||
        model.labels.size() > kMaximumLabels ||
        model.labels.size() >
            static_cast<std::size_t>((std::numeric_limits<int>::max)()) ||
        model.neutral_index >= model.labels.size())
    {
        throw std::invalid_argument("invalid scene behavior model metadata");
    }

    std::set<std::string> labels;
    for (const std::string& label : model.labels)
    {
        if (label.empty() || !labels.insert(label).second)
        {
            throw std::invalid_argument(
                "scene behavior labels must be non-empty and unique");
        }
    }

    const std::size_t reservoir_size =
        static_cast<std::size_t>(model.reservoir_size);
    const std::size_t input_weights =
        checked_product(reservoir_size, input_size, "input weights");
    const std::size_t recurrent_weights =
        checked_product(reservoir_size, reservoir_size,
                        "recurrent weights");
    const std::size_t output_weights =
        checked_product(model.labels.size(), reservoir_size,
                        "output weights");

    if (model.input_weights.size() != input_weights ||
        model.recurrent_weights.size() != recurrent_weights ||
        model.reservoir_bias.size() != reservoir_size ||
        model.output_weights.size() != output_weights ||
        model.output_bias.size() != model.labels.size())
    {
        throw std::invalid_argument(
            "scene behavior model weight shapes do not match metadata");
    }

    validate_finite(model.input_weights, "input weights");
    validate_finite(model.recurrent_weights, "recurrent weights");
    validate_finite(model.reservoir_bias, "reservoir bias");
    validate_finite(model.output_weights, "output weights");
    validate_finite(model.output_bias, "output bias");
}

kfcore_esn_model model_view(const SceneBehaviorModel& model)
{
    return {
        static_cast<int>(detail::scene_input_size(model.predicate_count)),
        model.reservoir_size,
        static_cast<int>(model.labels.size()),
        model.leak_rate,
        model.input_weights.data(),
        model.recurrent_weights.data(),
        model.reservoir_bias.data(),
        model.output_weights.data(),
        model.output_bias.data(),
    };
}

void require_esn(kfcore_esn_status status)
{
    if (status != KFCORE_ESN_OK)
    {
        throw std::runtime_error(
            "scene behavior ESN operation failed: " +
            std::to_string(static_cast<int>(status)));
    }
}

struct PairState
{
    explicit PairState(const SceneBehaviorModel& model)
        : reservoir(static_cast<std::size_t>(model.reservoir_size), 0.0F)
        , workspace(static_cast<std::size_t>(model.reservoir_size), 0.0F)
        , scores(model.labels.size(), 0.0F)
    {
    }

    std::vector<float> reservoir;
    std::vector<float> workspace;
    std::vector<float> scores;
    std::optional<std::size_t> candidate;
    std::optional<std::size_t> active;
    double candidate_since = 0.0;
    double supported_at = 0.0;
    double last_seen = 0.0;
    float active_score = 0.0F;
    std::int32_t subject_class_id = -1;
    std::int32_t object_class_id = -1;
};

struct Recognition
{
    std::optional<std::size_t> label;
    float score = 0.0F;
};

Recognition recognize(const SceneBehaviorModel& model,
                      const SceneInteractionOptions& options,
                      const std::vector<float>& scores)
{
    if (scores.size() != model.labels.size())
    {
        throw std::logic_error("scene behavior output size drifted from model");
    }

    std::size_t best = 0U;
    float best_score = scores[0];
    float second_score = -(std::numeric_limits<float>::infinity)();
    for (std::size_t index = 1U; index < scores.size(); ++index)
    {
        if (scores[index] > best_score)
        {
            second_score = best_score;
            best_score = scores[index];
            best = index;
        }
        else if (scores[index] > second_score)
        {
            second_score = scores[index];
        }
    }

    if (best == model.neutral_index ||
        best_score < options.minimum_score ||
        best_score - second_score < options.minimum_margin)
    {
        return {};
    }
    return {best, best_score};
}

} // namespace

struct SceneInteraction::Impl
{
    explicit Impl(SceneInteractionOptions value)
        : options(value)
    {
        validate_options(options);
    }

    SceneInteractionOptions options;
    std::shared_ptr<const SceneBehaviorModel> model;
    std::map<PairKey, PairState> pairs;
    std::optional<double> clock;
    std::optional<double> last_frame;

    void validate_time(double seconds) const
    {
        if (!std::isfinite(seconds) || seconds < 0.0 ||
            (clock && seconds < *clock))
        {
            throw std::invalid_argument(
                "scene interaction timestamp must be finite and monotonic");
        }
    }

    void emit(std::vector<SceneBehaviorEvent>& events,
              SceneBehaviorEventKind kind,
              SceneBehaviorEventReason reason,
              double seconds,
              const PairKey& pair,
              const PairState& state,
              std::size_t behavior_index,
              float score) const
    {
        events.push_back({
            kind,
            reason,
            seconds,
            pair,
            state.subject_class_id,
            state.object_class_id,
            behavior_index,
            score,
        });
    }

    void cancel_pair(std::vector<SceneBehaviorEvent>& events,
                     SceneBehaviorEventReason reason,
                     double seconds,
                     const PairKey& pair,
                     const PairState& state) const
    {
        if (state.active)
        {
            emit(events,
                 SceneBehaviorEventKind::BehaviorCancelled,
                 reason,
                 seconds,
                 pair,
                 state,
                 *state.active,
                 state.active_score);
        }
    }

    void cancel_all(std::vector<SceneBehaviorEvent>& events,
                    SceneBehaviorEventReason reason,
                    double seconds)
    {
        for (const auto& [pair, state] : pairs)
        {
            cancel_pair(events, reason, seconds, pair, state);
        }
        pairs.clear();
    }

    void expire(std::vector<SceneBehaviorEvent>& events, double seconds)
    {
        for (auto it = pairs.begin(); it != pairs.end();)
        {
            if (seconds - it->second.last_seen >
                options.maximum_gap_seconds)
            {
                cancel_pair(events,
                            SceneBehaviorEventReason::PairLost,
                            seconds,
                            it->first,
                            it->second);
                it = pairs.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    void update_pair(const detail::PairObservation& observation,
                     double seconds,
                     std::vector<SceneBehaviorEvent>& events)
    {
        if (!model)
        {
            throw std::logic_error(
                "scene interaction model is not configured");
        }

        auto [it, inserted] =
            pairs.try_emplace(observation.pair, *model);
        PairState& state = it->second;
        if (inserted)
        {
            state.subject_class_id = observation.subject_class_id;
            state.object_class_id = observation.object_class_id;
        }
        else if (state.subject_class_id != observation.subject_class_id ||
                 state.object_class_id != observation.object_class_id)
        {
            throw std::invalid_argument(
                "tracked pair class identity changed without a new track ID");
        }

        const kfcore_esn_model view = model_view(*model);
        require_esn(kfcore_esn_step_predict(
            &view,
            observation.values.data(),
            state.reservoir.data(),
            state.workspace.data(),
            state.scores.data()));

        const Recognition current =
            recognize(*model, options, state.scores);
        state.last_seen = seconds;

        if (current.label && state.active == current.label)
        {
            state.supported_at = seconds;
            state.active_score = current.score;
            state.candidate.reset();
            return;
        }

        if (state.candidate != current.label)
        {
            state.candidate = current.label;
            state.candidate_since = seconds;
        }

        if (current.label &&
            seconds - state.candidate_since >=
                options.confirmation_seconds)
        {
            if (state.active)
            {
                emit(events,
                     SceneBehaviorEventKind::BehaviorEnded,
                     SceneBehaviorEventReason::Recognized,
                     seconds,
                     observation.pair,
                     state,
                     *state.active,
                     state.active_score);
            }

            state.active = current.label;
            state.supported_at = seconds;
            state.active_score = current.score;
            state.candidate.reset();
            emit(events,
                 SceneBehaviorEventKind::BehaviorStarted,
                 SceneBehaviorEventReason::Recognized,
                 seconds,
                 observation.pair,
                 state,
                 *state.active,
                 state.active_score);
            return;
        }

        if (state.active &&
            seconds - state.supported_at >= options.end_seconds)
        {
            emit(events,
                 SceneBehaviorEventKind::BehaviorEnded,
                 SceneBehaviorEventReason::Unrecognized,
                 seconds,
                 observation.pair,
                 state,
                 *state.active,
                 state.active_score);
            state.active.reset();
            state.active_score = 0.0F;
        }
    }

    void process(const pipelines::SceneGraphFrame& frame,
                 double seconds,
                 std::vector<SceneBehaviorEvent>& events)
    {
        if (!model)
        {
            throw std::logic_error(
                "scene interaction model is not configured");
        }

        if (last_frame &&
            seconds - *last_frame > options.maximum_gap_seconds)
        {
            cancel_all(events,
                       SceneBehaviorEventReason::FrameGap,
                       seconds);
        }
        expire(events, seconds);

        const std::vector<detail::PairObservation> observations =
            detail::encode_pair_observations(
                frame, model->predicate_count);

        std::size_t new_pairs = 0U;
        for (const detail::PairObservation& observation : observations)
        {
            if (pairs.find(observation.pair) == pairs.end())
            {
                ++new_pairs;
            }
        }
        if (new_pairs >
                (std::numeric_limits<std::size_t>::max)() -
                    pairs.size() ||
            pairs.size() + new_pairs > options.max_pair_states)
        {
            throw std::length_error(
                "scene interaction pair-state capacity exceeded");
        }

        for (const detail::PairObservation& observation : observations)
        {
            update_pair(observation, seconds, events);
        }

        last_frame = seconds;
    }
};

SceneInteraction::SceneInteraction(SceneInteractionOptions options)
    : impl_(std::make_unique<Impl>(options))
{
}

SceneInteraction::~SceneInteraction() = default;

std::vector<SceneBehaviorEvent>
SceneInteraction::configure_model(const SceneBehaviorModel& model,
                                  double seconds)
{
    impl_->validate_time(seconds);
    validate_model(model);
    auto snapshot =
        std::make_shared<const SceneBehaviorModel>(model);

    auto next = std::make_unique<Impl>(*impl_);
    std::vector<SceneBehaviorEvent> events;
    events.reserve(next->pairs.size());
    next->cancel_all(events,
                     SceneBehaviorEventReason::ModelChanged,
                     seconds);
    next->model = std::move(snapshot);
    next->last_frame.reset();
    next->clock = seconds;
    impl_.swap(next);
    return events;
}

std::vector<SceneBehaviorEvent>
SceneInteraction::process(const pipelines::SceneGraphFrame& frame,
                          double seconds)
{
    impl_->validate_time(seconds);
    if (impl_->last_frame && seconds <= *impl_->last_frame)
    {
        throw std::invalid_argument(
            "scene interaction frame timestamps must strictly increase");
    }

    auto next = std::make_unique<Impl>(*impl_);
    std::vector<SceneBehaviorEvent> events;
    events.reserve(next->options.max_pair_states);
    next->process(frame, seconds, events);
    next->clock = seconds;
    impl_.swap(next);
    return events;
}

std::vector<SceneBehaviorEvent>
SceneInteraction::advance(double seconds)
{
    impl_->validate_time(seconds);
    auto next = std::make_unique<Impl>(*impl_);
    std::vector<SceneBehaviorEvent> events;
    events.reserve(next->pairs.size());
    next->expire(events, seconds);
    next->clock = seconds;
    impl_.swap(next);
    return events;
}

std::vector<SceneBehaviorEvent>
SceneInteraction::reset(double seconds)
{
    impl_->validate_time(seconds);
    auto next = std::make_unique<Impl>(*impl_);
    std::vector<SceneBehaviorEvent> events;
    events.reserve(next->pairs.size());
    next->cancel_all(events,
                     SceneBehaviorEventReason::Reset,
                     seconds);
    next->last_frame.reset();
    next->clock = seconds;
    impl_.swap(next);
    return events;
}

bool SceneInteraction::configured() const noexcept
{
    return impl_ && static_cast<bool>(impl_->model);
}

std::size_t SceneInteraction::pair_state_count() const noexcept
{
    return impl_ ? impl_->pairs.size() : 0U;
}

const char* event_kind_name(SceneBehaviorEventKind kind)
{
    switch (kind)
    {
    case SceneBehaviorEventKind::BehaviorStarted:
        return "BehaviorStarted";
    case SceneBehaviorEventKind::BehaviorEnded:
        return "BehaviorEnded";
    case SceneBehaviorEventKind::BehaviorCancelled:
        return "BehaviorCancelled";
    }
    throw std::invalid_argument("invalid scene behavior event kind");
}

const char* event_reason_name(SceneBehaviorEventReason reason)
{
    switch (reason)
    {
    case SceneBehaviorEventReason::Recognized:
        return "recognized";
    case SceneBehaviorEventReason::Unrecognized:
        return "unrecognized";
    case SceneBehaviorEventReason::PairLost:
        return "pair lost";
    case SceneBehaviorEventReason::FrameGap:
        return "frame gap";
    case SceneBehaviorEventReason::Reset:
        return "reset";
    case SceneBehaviorEventReason::ModelChanged:
        return "model changed";
    }
    throw std::invalid_argument("invalid scene behavior event reason");
}

} // namespace kfcore::scene_interaction
