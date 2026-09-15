# Hand interaction

`KFCore::hand_interaction` is the deterministic semantic layer above tracked hands and learned temporal gesture events.
It no longer recognizes dynamic physical gestures such as Wave, Swipe, Grab, Release, or Click from THIG timing graphs.
Those gestures are produced by `KFCore::hand_gesture::TemporalGestureRecognizer`.

## Architecture

```text
HandDetector
  -> HandTracker
  -> TemporalGestureRecognizer (GRU)
  -> GestureEvent
  -> HandInteractionPipeline
       |-> learned gesture -> semantic application action
       `-> primitive/static/spatial THIG semantics
```

The remaining THIG graph is deliberately limited to static or spatial semantics such as:

- `OK`;
- `Single Hand V` / `Two Hand V`;
- `Zoom In` / `Zoom Out`;
- `Rotate Clockwise` / `Rotate CounterClockwise`.

There is no THIG Wave graph, click state graph, grasp/release timing graph, direction dwell requirement, or gesture-specific reversal/duration rule.

## Learned gesture mapping

The interaction layer consumes `hand_gesture::GestureEvent` values and applies only deterministic application semantics:

- `Wave/End` -> `Wave`;
- `SwipeLeft/End` -> `Swipe Left`, or `Drag Start Left` / `Drag Left` while a hand is grabbed;
- `SwipeRight/End` -> `Swipe Right`, or `Drag Start Right` / `Drag Right` while a hand is grabbed;
- `Grab/Start` -> `Grasp` and binds the canonical hand;
- `Release/Start` -> `Release` or `Drag End` and clears the grab binding;
- `Click/End` -> `Click` or `Click Center/Left/Right/Top/Bottom` when the caller supplies a current Region observation.

These rules do not decide whether the physical gesture occurred. They only map a learned event to application semantics.

## Usage

```cpp
#include <kfcore/hand_interaction/hand_interaction.hpp>

kfcore::hand_interaction::HandInteractionOptions options;
options.primitives.max_hands = 8;
options.semantic.max_observations_per_frame = 128;

kfcore::hand_interaction::HandInteractionPipeline interaction(options);

const auto result = interaction.process(
    tracked_hand_frame,
    {serial, std::chrono::steady_clock::now(), image_width, image_height},
    learned_gesture_events,
    region_observations);

for (const auto& action : result.actions) {
    dispatch_application_command(action.action);
}
```

`learned_gesture_events` use raw `HandResult::track_id`; the pipeline maps each current raw track to the canonical hand identity produced by `HandPrimitiveExtractor`.
An event referring to a hand absent from the current frame is rejected.

External Region observations still use canonical hand IDs. Accepted relations are exactly:

```text
Region Center
Region Left
Region Right
Region Top
Region Bottom
Region Unclassified
```

Each canonical hand may have at most one Region observation per frame. Region observations are used directly for learned Click semantics and are not fed into THIG temporal recognition.

## Canonical hand identity

Canonical identity remains owned by `HandPrimitiveExtractor`. It combines bounded raw-track continuity, geometry, shape, handedness, motion, scale, and optional appearance evidence.
The identity registry is independent of the GRU hidden state: `TemporalGestureRecognizer` is keyed by the current raw track ID, while the interaction layer maps the emitted event to the current canonical ID before creating an action.

Long-gap recurrent-state re-identification is intentionally outside Temporal Gesture V1.

## Settings

`HandInteractionSettings` now contains only deterministic static/spatial THIG settings and capacities:

- OK / V / dual-hand / spatial dwell values;
- shape observation-window stabilization;
- THIG history and bounded capacities;
- rotation cooldown.

The following old dynamic-gesture recognition settings no longer exist and have no compatibility aliases:

```text
direction_dwell_ms
direction_window_ms
wave_reversal_max_ms
wave_total_max_ms
wave_require_horizontal_palm_axis
grab_select_stable_ms
grab_release_stable_ms
grab_transition_max_ms
click_ready_dwell_ms
click_press_dwell_ms
click_release_dwell_ms
click_transition_max_ms
neutral_rearm_ms
```

Dynamic gesture timing is learned by `gesture.temporal-gru` instead of configured here.

## State and reset

`HandInteractionPipeline` owns:

- canonical hand primitive/identity state;
- the remaining static/spatial THIG engine state;
- one deterministic grab/drag binding.

`reset()` clears all three. The pipeline is a single-owner stateful object; callers serialize access to one instance.

## Build target

`KFCore::hand_interaction` is a static SDK target and publicly depends on:

- `KFCore::hand_model_core`;
- `KFCore::hand_gesture`;
- `KFCore::thig` for the remaining static/spatial semantic graph.

Learned temporal model execution remains in `KFCore::hand_gesture` and uses the normal backend-neutral Runtime/Model Package path.
