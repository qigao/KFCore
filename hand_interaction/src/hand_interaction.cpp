#include "kfcore/hand_interaction/hand_interaction.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace kfcore::hand_interaction
{
namespace
{

constexpr char kHandEntityKind[] = "hand";
constexpr char kRegionPrefix[] = "Region ";

bool is_declared_region(const std::string& relation) noexcept
{
    return relation == "Region Center" || relation == "Region Left" ||
           relation == "Region Right" || relation == "Region Top" ||
           relation == "Region Bottom" || relation == "Region Unclassified";
}

void validate_external_observations(const std::vector<thig::Observation>& observations,
                                    const GestureFrameContext& context)
{
    std::unordered_set<std::int64_t> region_sources;
    region_sources.reserve(observations.size());
    for (const thig::Observation& observation : observations)
    {
        if (observation.serial != context.serial)
        {
            throw std::invalid_argument(
                "external observation serial must match the frame context");
        }
        if (observation.source.kind != kHandEntityKind || observation.source.id <= 0 ||
            observation.target.has_value())
        {
            throw std::invalid_argument(
                "external Region observations require one canonical hand source");
        }
        if (!is_declared_region(observation.relation))
        {
            throw std::invalid_argument(
                "external observations must use a declared Region relation");
        }
        if (!std::isfinite(observation.confidence) || observation.confidence < 0.0F ||
            observation.confidence > 1.0F)
        {
            throw std::invalid_argument(
                "external observation confidence must be within [0,1]");
        }
        if (!region_sources.insert(observation.source.id).second)
        {
            throw std::invalid_argument(
                "external observations contain multiple Regions for one hand");
        }
    }
}

std::size_t maximum_primitive_observations(std::size_t hand_count)
{
    constexpr std::size_t kObservationsPerHand = 8U;
    const std::size_t pair_count =
        hand_count > 1U ? hand_count * (hand_count - 1U) / 2U : 0U;
    return hand_count * kObservationsPerHand + pair_count;
}

std::uint64_t context_time_ms(const GestureFrameContext& context)
{
    const auto count = std::chrono::duration_cast<std::chrono::milliseconds>(
        context.observed_at.time_since_epoch()).count();
    if (count < 0)
    {
        throw std::invalid_argument("frame context time must not precede steady-clock epoch");
    }
    return static_cast<std::uint64_t>(count);
}

thig::ActionEvent make_action(const std::string& name,
                              int canonical_id,
                              const GestureFrameContext& context,
                              float confidence)
{
    thig::ActionEvent action;
    action.action = name;
    action.source = {kHandEntityKind, canonical_id};
    const std::uint64_t now_ms = context_time_ms(context);
    action.startMs = now_ms;
    action.confirmedMs = now_ms;
    action.endMs = now_ms;
    action.confidence = confidence;
    action.support = confidence;
    action.specVersion = kHandInteractionSpecVersion;
    return action;
}

std::unordered_map<int, int> raw_to_canonical(const PrimitiveFrame& frame)
{
    std::unordered_map<int, int> result;
    result.reserve(frame.hands.size());
    for (const CanonicalHand& hand : frame.hands)
    {
        if (hand.raw_track_id < 0 || hand.canonical_id <= 0)
        {
            continue;
        }
        result.emplace(hand.raw_track_id, hand.canonical_id);
    }
    return result;
}

std::unordered_map<int, std::string> region_by_canonical(
    const std::vector<thig::Observation>& external_observations,
    const std::unordered_set<int>& canonical_ids)
{
    std::unordered_map<int, std::string> result;
    result.reserve(external_observations.size());
    for (const auto& observation : external_observations)
    {
        const int canonical_id = static_cast<int>(observation.source.id);
        if (canonical_ids.find(canonical_id) == canonical_ids.end())
        {
            throw std::invalid_argument(
                "external Region observation refers to a hand absent from this frame");
        }
        result.emplace(canonical_id, observation.relation);
    }
    return result;
}

std::string click_action_name(int canonical_id,
                              const std::unordered_map<int, std::string>& regions)
{
    const auto found = regions.find(canonical_id);
    if (found == regions.end() || found->second == "Region Unclassified")
    {
        return "Click";
    }
    return "Click " + found->second.substr(sizeof(kRegionPrefix) - 1U);
}

void append_learned_action(
    std::vector<thig::ActionEvent>& actions,
    const hand_gesture::GestureEvent& gesture,
    int canonical_id,
    const GestureFrameContext& context,
    const std::unordered_map<int, std::string>& regions,
    std::optional<int>& grabbed_hand,
    bool& dragging)
{
    if (!std::isfinite(gesture.confidence) || gesture.confidence < 0.0F ||
        gesture.confidence > 1.0F)
    {
        throw std::invalid_argument("learned gesture confidence must be within [0,1]");
    }

    using hand_gesture::GestureClass;
    using hand_gesture::GesturePhase;

    if (gesture.gesture == GestureClass::Wave && gesture.phase == GesturePhase::End)
    {
        actions.push_back(make_action("Wave", canonical_id, context, gesture.confidence));
        return;
    }

    if ((gesture.gesture == GestureClass::SwipeLeft ||
         gesture.gesture == GestureClass::SwipeRight) &&
        gesture.phase == GesturePhase::End)
    {
        const bool left = gesture.gesture == GestureClass::SwipeLeft;
        if (grabbed_hand && *grabbed_hand == canonical_id)
        {
            const std::string name = dragging
                ? (left ? "Drag Left" : "Drag Right")
                : (left ? "Drag Start Left" : "Drag Start Right");
            dragging = true;
            actions.push_back(make_action(name, canonical_id, context, gesture.confidence));
        }
        else
        {
            actions.push_back(make_action(
                left ? "Swipe Left" : "Swipe Right", canonical_id, context,
                gesture.confidence));
        }
        return;
    }

    if (gesture.gesture == GestureClass::Grab && gesture.phase == GesturePhase::Start)
    {
        if (!grabbed_hand)
        {
            grabbed_hand = canonical_id;
            dragging = false;
            actions.push_back(make_action("Grasp", canonical_id, context, gesture.confidence));
        }
        return;
    }

    if (gesture.gesture == GestureClass::Release && gesture.phase == GesturePhase::Start)
    {
        if (grabbed_hand && *grabbed_hand == canonical_id)
        {
            actions.push_back(make_action(
                dragging ? "Drag End" : "Release", canonical_id, context,
                gesture.confidence));
            grabbed_hand.reset();
            dragging = false;
        }
        else
        {
            actions.push_back(make_action("Release", canonical_id, context, gesture.confidence));
        }
        return;
    }

    if (gesture.gesture == GestureClass::Click && gesture.phase == GesturePhase::End)
    {
        actions.push_back(make_action(
            click_action_name(canonical_id, regions), canonical_id, context,
            gesture.confidence));
    }
}

} // namespace

struct HandInteractionPipeline::Impl
{
    explicit Impl(const HandInteractionOptions& options)
        : primitives(options.primitives)
        , engine(build_hand_interaction_graph(options.semantic))
        , max_hands(options.primitives.max_hands)
    {
    }

    HandPrimitiveExtractor primitives;
    thig::TemporalGraphEngine engine;
    std::size_t max_hands;
    std::optional<int> grabbed_hand;
    bool dragging = false;
};

HandInteractionPipeline::HandInteractionPipeline(const HandInteractionOptions& options)
    : impl_(std::make_unique<Impl>(options))
{
}

HandInteractionPipeline::~HandInteractionPipeline() = default;
HandInteractionPipeline::HandInteractionPipeline(HandInteractionPipeline&&) noexcept = default;
HandInteractionPipeline&
HandInteractionPipeline::operator=(HandInteractionPipeline&&) noexcept = default;

HandInteractionFrame HandInteractionPipeline::process(
    const hand_models::HandFrame& frame,
    const GestureFrameContext& context,
    const std::vector<hand_gesture::GestureEvent>& gestures,
    const std::vector<thig::Observation>& external_observations)
{
    validate_external_observations(external_observations, context);
    if (frame.hands.size() > impl_->max_hands)
    {
        throw std::length_error("hand interaction frame exceeds max_hands");
    }
    const std::size_t maximum_observations = maximum_primitive_observations(frame.hands.size());
    if (maximum_observations > impl_->engine.Spec().maxObservationsPerFrame)
    {
        throw std::length_error("hand interaction frame exceeds THIG primitive capacity");
    }

    HandPrimitiveExtractor staged_primitives = impl_->primitives.clone();
    HandInteractionFrame result;
    result.primitives = staged_primitives.process(frame, context);

    const auto canonical_map = raw_to_canonical(result.primitives);
    std::unordered_set<int> canonical_ids;
    canonical_ids.reserve(canonical_map.size());
    for (const auto& entry : canonical_map)
    {
        canonical_ids.insert(entry.second);
    }
    const auto regions = region_by_canonical(external_observations, canonical_ids);

    std::unordered_set<int> gesture_tracks;
    gesture_tracks.reserve(gestures.size());
    std::optional<int> staged_grabbed = impl_->grabbed_hand;
    bool staged_dragging = impl_->dragging;
    std::vector<thig::ActionEvent> learned_actions;
    learned_actions.reserve(gestures.size());

    for (const auto& gesture : gestures)
    {
        if (gesture.track_id < 0 || !gesture_tracks.insert(gesture.track_id).second)
        {
            throw std::invalid_argument(
                "learned gesture events require one non-negative unique track_id per frame");
        }
        const auto canonical = canonical_map.find(gesture.track_id);
        if (canonical == canonical_map.end())
        {
            throw std::invalid_argument(
                "learned gesture event refers to a hand absent from this frame");
        }
        append_learned_action(
            learned_actions, gesture, canonical->second, context, regions,
            staged_grabbed, staged_dragging);
    }

    result.actions = impl_->engine.ProcessFrame(
        result.primitives.observations, context.observed_at);
    result.actions.insert(result.actions.end(),
                          learned_actions.begin(), learned_actions.end());
    result.primitives.observations.insert(
        result.primitives.observations.end(),
        external_observations.begin(), external_observations.end());

    impl_->primitives = std::move(staged_primitives);
    impl_->grabbed_hand = staged_grabbed;
    impl_->dragging = staged_dragging;
    return result;
}

void HandInteractionPipeline::reset()
{
    impl_->primitives.reset();
    impl_->engine.Reset();
    impl_->grabbed_hand.reset();
    impl_->dragging = false;
}

const thig::EngineSpec& HandInteractionPipeline::graph_spec() const
{
    return impl_->engine.Spec();
}

} // namespace kfcore::hand_interaction
