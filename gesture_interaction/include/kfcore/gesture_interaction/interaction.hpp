#pragma once
#include "action_gate.hpp"
#include <memory>

namespace kfcore::gesture_interaction {
enum class EventKind { GestureStarted, GestureEnded, GestureCancelled, Grasp, Release, Wave, GraspCancelled };
enum class EventReason { Recognized, Unrecognized, HandLost, FrameGap, MultipleHands, SourceChanged, Reset, ModelChanged, Capacity };
struct Event {
    EventKind kind;
    mediapipe::CannedGesture gesture = mediapipe::CannedGesture::None;
    EventReason reason = EventReason::Recognized;
    double seconds = 0;
    std::uint64_t continuity_id = 0;
    std::optional<std::uint64_t> source_id;
};
struct FrameContext {
    double seconds = 0; // Monotonic acquisition clock, also used by advance/reset.
    int width = 0, height = 0;
    std::optional<std::uint64_t> source_id; // Optional stable identity supplied by caller; never handedness.
};
struct StaticGestureOptions {
    float minimum_score = 0.7F;
    float minimum_margin = 0.15F;
    double confirmation_seconds = 0.15;
    double end_seconds = 0.2;
};
struct InteractionOptions {
    StaticGestureOptions basic;
    double maximum_gap_seconds = 0.25;
    double side_change_seconds = 0.15;
    float side_confidence = 0.85F;
};
// Borrowed only for configure_sequences(); trained classifiers are copied into owned snapshots.
struct SequenceModels {
    const CompositionEsn* interaction = nullptr;
    const CompositionEsn* motion = nullptr;
    ActionGateOptions interaction_gate, motion_gate;
};
// Single-owner synchronous API. Returned events own their data. No UI, I/O or callbacks.
// Static gestures work with no ESN. Sequence classifiers must be trained and task-matched.
class GestureInteraction final {
public:
    explicit GestureInteraction(InteractionOptions options = {});
    ~GestureInteraction();
    GestureInteraction(const GestureInteraction&) = delete;
    GestureInteraction& operator=(const GestureInteraction&) = delete;
    // Validate first, then atomically replace models and return cancellation events.
    [[nodiscard]] std::vector<Event> configure_sequences(const SequenceModels& models, double seconds);
    // Exactly one visible hand is supported; multi-hand input cancels active lifecycles.
    // Invalid observations/timestamps throw without committing state; caller may reset on a fatal stream error.
    [[nodiscard]] std::vector<Event> process(const mediapipe::GestureFrame& frame, const FrameContext& context);
    // Call periodically when no frames arrive. Same clock as process; does not create observations.
    [[nodiscard]] std::vector<Event> advance(double seconds);
    // Call before pause, shutdown or changing input. Consume cancellations before destroying this object.
    [[nodiscard]] std::vector<Event> reset(double seconds);
    [[nodiscard]] bool grasping() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
const char* event_name(EventKind kind);
const char* event_reason_name(EventReason reason);
} // namespace kfcore::gesture_interaction
