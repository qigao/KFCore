#include <kfcore/gesture_interaction/interaction.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kfcore::gesture_interaction {
namespace {
using Gesture = mediapipe::CannedGesture;
constexpr std::size_t kEventCapacity = 8;
bool positive(double value) { return std::isfinite(value) && value > 0; }
bool ratio(float value) { return std::isfinite(value) && value >= 0 && value <= 1; }
struct Branch {
    std::shared_ptr<const CompositionEsn> model;
    ActionGate gate;
    CompositionClip history;
    void clear() { history.clear(); gate.reset(); }
    std::optional<Composition> update(const CompositionSample& sample) {
        if (!model) return std::nullopt;
        const auto& options = model->options();
        if (!history.empty() && sample.seconds-history.back().seconds > options.maximum_gap_seconds) clear();
        if (history.size() >= options.maximum_samples) throw std::length_error("sequence history capacity exceeded");
        history.push_back(sample);
        if (!sample.values.back()) gate.interrupt();
        if (sample.seconds-history.front().seconds < options.duration_seconds) return std::nullopt;
        const auto prediction = model->predict(history);
        const auto event = sample.values.back() ? gate.update(prediction, sample.seconds) : std::nullopt;
        const double cutoff = history.front().seconds + options.live_stride_seconds;
        history.erase(history.begin(), std::lower_bound(history.begin(), history.end(), cutoff,
            [](const CompositionSample& value, double time) { return value.seconds < time; }));
        return event;
    }
};
Branch make_branch(const CompositionEsn* model, CompositionTask task, ActionGateOptions options) {
    Branch branch; branch.gate = ActionGate(options);
    if (model) {
        if (!model->trained() || model->options().task != task)
            throw std::invalid_argument("sequence model must be trained and task-matched");
        branch.model = std::make_shared<const CompositionEsn>(*model);
        branch.history.reserve(model->options().maximum_samples);
    }
    return branch;
}
}
struct GestureInteraction::Impl {
    explicit Impl(InteractionOptions value) : options(value) {
        if (!ratio(options.basic.minimum_score) || !ratio(options.basic.minimum_margin) ||
            !positive(options.basic.confirmation_seconds) || !positive(options.basic.end_seconds) ||
            !positive(options.maximum_gap_seconds) || !positive(options.side_change_seconds) ||
            !ratio(options.side_confidence) || options.side_confidence <= 0.5F)
            throw std::invalid_argument("invalid gesture interaction options");
    }
    InteractionOptions options;
    Branch interaction, motion;
    std::optional<double> clock, last_frame, last_seen;
    std::optional<std::uint64_t> source;
    std::uint64_t epoch = 0;
    bool connected = false, held = false;
    std::optional<bool> side, pending_side;
    double side_since = 0;
    std::optional<Gesture> active, candidate;
    double candidate_since = 0, supported_at = 0;

    double maximum_gap() const {
        double gap = options.maximum_gap_seconds;
        if (interaction.model) gap = std::min(gap, interaction.model->options().maximum_gap_seconds);
        if (motion.model) gap = std::min(gap, motion.model->options().maximum_gap_seconds);
        return gap;
    }
    void validate_time(double seconds) const {
        if (!std::isfinite(seconds) || seconds < 0 || (clock && seconds < *clock))
            throw std::invalid_argument("interaction timestamp must be finite and monotonic");
    }
    void emit(std::vector<Event>& output, EventKind kind, double seconds,
              EventReason reason = EventReason::Recognized, Gesture gesture = Gesture::None) const {
        output.push_back({kind, gesture, reason, seconds, epoch, source});
    }
    void cancel(std::vector<Event>& output, double seconds, EventReason reason) {
        if (active) emit(output, EventKind::GestureCancelled, seconds, reason, *active);
        if (held) emit(output, EventKind::GraspCancelled, seconds, reason);
        active.reset(); candidate.reset(); held = false; connected = false;
        pending_side.reset(); side.reset(); source.reset(); last_seen.reset();
        interaction.clear(); motion.clear();
    }
    void expire(std::vector<Event>& output, double seconds) {
        if (connected && last_seen && seconds-*last_seen > maximum_gap())
            cancel(output, seconds, EventReason::HandLost);
    }
    void basic(const mediapipe::GesturePrediction& prediction, double seconds, std::vector<Event>& output) {
        auto scores = prediction.scores;
        const auto index = std::max_element(scores.begin(), scores.end())-scores.begin();
        std::sort(scores.begin(), scores.end(), std::greater<float>());
        std::optional<Gesture> recognized;
        if (index != std::size_t(Gesture::None) && scores[0] >= options.basic.minimum_score &&
            scores[0]-scores[1] >= options.basic.minimum_margin) recognized = Gesture(index);
        if (recognized && recognized == active) supported_at = seconds;
        if (recognized != candidate) { candidate = recognized; candidate_since = seconds; }
        if (recognized && seconds-candidate_since >= options.basic.confirmation_seconds && recognized != active) {
            if (active) emit(output, EventKind::GestureEnded, seconds, EventReason::Recognized, *active);
            active = recognized; supported_at = seconds;
            emit(output, EventKind::GestureStarted, seconds, EventReason::Recognized, *active);
        } else if (active && seconds-supported_at >= options.basic.end_seconds) {
            emit(output, EventKind::GestureEnded, seconds, EventReason::Unrecognized, *active);
            active.reset();
        }
    }
    void sequence_event(std::optional<Composition> event, double seconds, std::vector<Event>& output) {
        if (event == Composition::Grasp && !held) { held = true; emit(output, EventKind::Grasp, seconds); }
        else if (event == Composition::Release && held) { held = false; emit(output, EventKind::Release, seconds); }
        else if (event == Composition::Wave) emit(output, EventKind::Wave, seconds);
    }
    void process(const mediapipe::GestureFrame& frame, const FrameContext& context, std::vector<Event>& output) {
        const auto sample = composition_sample(frame, context.seconds, context.width, context.height);
        if (last_frame && context.seconds-*last_frame > maximum_gap()) cancel(output, context.seconds, EventReason::FrameGap);
        last_frame = context.seconds;
        expire(output, context.seconds);
        if (!sample) { cancel(output, context.seconds, EventReason::MultipleHands); return; }
        if ((interaction.model && interaction.history.size() >= interaction.model->options().maximum_samples) ||
            (motion.model && motion.history.size() >= motion.model->options().maximum_samples)) {
            cancel(output, context.seconds, EventReason::Capacity); return;
        }
        if (frame.gestures.empty()) {
            candidate.reset(); pending_side.reset();
            if (connected) {
                sequence_event(interaction.update(*sample), context.seconds, output);
                sequence_event(motion.update(*sample), context.seconds, output);
            }
            return;
        }
        if (connected && source != context.source_id) cancel(output, context.seconds, EventReason::SourceChanged);
        const float probability = sample->values[kCompositionInputs-2];
        std::optional<bool> observed_side;
        if (probability >= options.side_confidence) observed_side = true;
        else if (probability <= 1-options.side_confidence) observed_side = false;
        if (connected && side && observed_side && side != observed_side) {
            candidate.reset(); interaction.clear(); motion.clear();
            if (pending_side != observed_side) { pending_side = observed_side; side_since = context.seconds; }
            if (context.seconds-side_since >= options.side_change_seconds)
                cancel(output, context.seconds, EventReason::SourceChanged);
            else return; // Do not extend known-source liveness with contradictory identity evidence.
        } else pending_side.reset();
        if (!connected) {
            if (epoch == (std::numeric_limits<std::uint64_t>::max)()) throw std::overflow_error("continuity ID exhausted");
            ++epoch; connected = true; source = context.source_id; side = observed_side;
        }
        if (!side) side = observed_side;
        last_seen = context.seconds;
        basic(frame.gestures.front(), context.seconds, output);
        sequence_event(interaction.update(*sample), context.seconds, output);
        sequence_event(motion.update(*sample), context.seconds, output);
    }
};
GestureInteraction::GestureInteraction(InteractionOptions options) : impl_(std::make_unique<Impl>(options)) {}
GestureInteraction::~GestureInteraction() = default;
std::vector<Event> GestureInteraction::configure_sequences(const SequenceModels& models, double seconds) {
    impl_->validate_time(seconds);
    auto interaction = make_branch(models.interaction, CompositionTask::Interaction, models.interaction_gate);
    auto motion = make_branch(models.motion, CompositionTask::Motion, models.motion_gate);
    std::vector<Event> events; events.reserve(kEventCapacity);
    impl_->cancel(events, seconds, EventReason::ModelChanged);
    impl_->interaction = std::move(interaction); impl_->motion = std::move(motion); impl_->clock = seconds;
    return events;
}
std::vector<Event> GestureInteraction::process(const mediapipe::GestureFrame& frame, const FrameContext& context) {
    impl_->validate_time(context.seconds);
    if (context.width <= 0 || context.height <= 0 || (context.source_id && !*context.source_id) ||
        (impl_->last_frame && context.seconds <= *impl_->last_frame) ||
        frame.landmarks.hands.size() != frame.gestures.size())
        throw std::invalid_argument("invalid gesture frame context");
    for (std::size_t i = 0; i < frame.gestures.size(); ++i) {
        if (std::size_t(frame.gestures[i].label) >= mediapipe::kGestureCount)
            throw std::invalid_argument("invalid basic gesture label");
        for (float score : frame.gestures[i].scores) if (!ratio(score))
            throw std::invalid_argument("gesture scores must be finite in [0,1]");
        const auto& hand = frame.landmarks.hands[i];
        if (!hand.right_hand_probability || !ratio(*hand.right_hand_probability) ||
            !std::isfinite(hand.landmarks[0].x) || !std::isfinite(hand.landmarks[0].y))
            throw std::invalid_argument("invalid hand observation");
    }
    // Only bounded temporal state is copied per frame; immutable model snapshots are shared.
    // Commit after all branches succeed, so an exception cannot consume a partial frame.
    auto next = std::make_unique<Impl>(*impl_);
    std::vector<Event> events; events.reserve(kEventCapacity);
    next->process(frame, context, events);
    next->clock = context.seconds; impl_.swap(next);
    return events;
}
std::vector<Event> GestureInteraction::advance(double seconds) {
    impl_->validate_time(seconds);
    std::vector<Event> events; events.reserve(kEventCapacity);
    impl_->expire(events, seconds); impl_->clock = seconds;
    return events;
}
std::vector<Event> GestureInteraction::reset(double seconds) {
    impl_->validate_time(seconds);
    std::vector<Event> events; events.reserve(kEventCapacity);
    impl_->cancel(events, seconds, EventReason::Reset); impl_->clock = seconds;
    return events;
}
bool GestureInteraction::grasping() const noexcept { return impl_->held; }
const char* event_name(EventKind kind) {
    switch (kind) {
    case EventKind::GestureStarted: return "GestureStarted";
    case EventKind::GestureEnded: return "GestureEnded";
    case EventKind::GestureCancelled: return "GestureCancelled";
    case EventKind::Grasp: return "Grasp";
    case EventKind::Release: return "Release";
    case EventKind::Wave: return "Wave";
    case EventKind::GraspCancelled: return "GraspCancelled";
    }
    throw std::invalid_argument("invalid event kind");
}
const char* event_reason_name(EventReason reason) {
    switch (reason) {
    case EventReason::Recognized: return "recognized";
    case EventReason::Unrecognized: return "unrecognized";
    case EventReason::HandLost: return "hand lost";
    case EventReason::FrameGap: return "frame gap";
    case EventReason::MultipleHands: return "multiple hands";
    case EventReason::SourceChanged: return "source changed";
    case EventReason::Reset: return "reset";
    case EventReason::ModelChanged: return "model changed";
    case EventReason::Capacity: return "history capacity";
    }
    throw std::invalid_argument("invalid event reason");
}
} // namespace kfcore::gesture_interaction
