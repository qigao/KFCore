# Hand interaction

`KFCore::hand_interaction` converts tracked hands into bounded primitive
observations and recognizes production gesture actions with THIG. The causal GRU
implementation remains an independent experiment and is not a dependency of this
module.

## Architecture

```text
HandDetector
  -> HandTracker
  -> HandPrimitiveExtractor
  -> TemporalGraphEngine (THIG)
  -> ActionEvent
```

The V5 graph recognizes:

- `Swipe Left` / `Swipe Right` from open-hand sustained direction observations;
- `Grasp` / `Release` from stationary Open/Fist transitions;
- `Drag Start Left/Right`, `Drag Left/Right`, `Drag End`, and `Drag Cancelled`;
- `OK`, `Single Hand V`, stationary `Two Hand V`, stationary `V Fist`;
- `Zoom In`, `Zoom Out`, `Rotate Clockwise`, `Rotate CounterClockwise`.

Wave, Point, Click, and Pinch are intentionally excluded. No GRU-to-THIG fallback
or compatibility adapter is present.

## Usage

```cpp
#include <kfcore/hand_interaction/hand_interaction.hpp>

kfcore::hand_interaction::HandInteractionOptions options;
options.primitives.max_hands = 8;
options.temporal.max_observations_per_frame = 128;
options.temporal.max_relation_events = 512;
options.temporal.max_action_states = 5120;

kfcore::hand_interaction::HandInteractionPipeline interaction(options);
const auto result = interaction.process(
    tracked_hand_frame,
    {serial, std::chrono::steady_clock::now(), image_width, image_height},
    region_observations);
```

`process()` synchronously borrows its inputs and returns owned observations and
actions. Frame serials must strictly increase, timestamps must not decrease, and
image dimensions must be positive. Invalid configuration, identity, observation,
or capacity input fails immediately.

## Temporal settings

`HandInteractionSettings` owns the configurable direction stabilization, Open/Fist
transition timing, neutral rearm timing, static/spatial dwell values, shape
stabilization, history, cooldown, and bounded THIG capacities. The default ten
stateless actions and 512 relation states require `max_action_states >= 5120`;
custom lower capacities are rejected by THIG configuration validation.

Default Swipe confirmation retains up to five direction samples in 167 ms and
requires at least two supporting samples plus 67 ms of winning evidence. The
independent duration gate prevents high-frame-rate streams from confirming
earlier merely because they provide more samples, while the two-sample floor
lets sustained low-frame-rate input confirm without waiting for a third frame.
All thresholds remain configurable.

The direction window is shared by Swipe, Drag Start/Drag, direction switching,
and post-drag `Direction Neutral` rearm. The two-sample default therefore applies
to each consumer; Swipe and Drag still independently require 67 ms of direction
duration, while the interaction state graph continues to enforce binding and
phase order.

One competing direction frame immediately ends the active direction relation
but does not switch the stabilizer unless window support and the switch margin
are satisfied. Returning to the retained stable direction can therefore recover
without rebuilding its complete window.

`Two Hand V` and `V Fist` require overlapping shape and `Motion Stationary`
evidence observed in the current frame from both distinct hands. Same-source
THIG patterns retain bounded `UNKNOWN` dropout behavior, but a disappeared hand
cannot combine with a current second hand to create a new two-hand action.

Gesture thresholds are production configuration until a trained GRU artifact and
quality gate justify another explicit migration. They are not mirrored in the
experimental GRU runtime.

## Canonical identity

ByteTrack raw IDs are not the identity fact source for temporal gestures.
`HandPrimitiveExtractor` resolves each reliable hand to a canonical ID using raw
continuity, geometry, motion, scale, hand shape, handedness, and optional
appearance. THIG relations and state graphs use that canonical ID, so a raw-ID
replacement does not by itself reset an unambiguous gesture. Ambiguous matches
remain `canonical_id == 0` and emit no primitive observations instead of guessing.

ByteTrack should be replaced only if recorded crossing, occlusion, fast-motion,
and detector-dropout sequences establish a ByteTrack baseline and explicit
acceptance thresholds for ID switches, fragmentation, canonical reacquisition,
gesture cancellation, and false activation. That recorded corpus and numerical
gate are not yet present in the repository. Appearance remains opt-in and must be
evaluated with its extraction cost.

## External Region context

Optional external observations accept exactly one declared Region per current
canonical hand and frame:

```text
Region Center
Region Left
Region Right
Region Top
Region Bottom
Region Unclassified
```

Region observations are validated and copied into the returned primitive frame.
They are not fed into THIG and do not produce Click actions.

## State and reset

Each `HandInteractionPipeline` is single-owner and not reentrant. It owns canonical
identity/primitive history and one `TemporalGraphEngine`; callers must serialize
access to one instance. `reset()` clears both owners. Primitive state is staged and
committed only after THIG accepts the complete frame.

The single interaction state graph supports one active Grasp/Drag lifecycle. An
idle graph does not reserve a visible hand: the source becomes bound only after
that source completes the stationary Open-to-Fist Grasp transition, and remains
bound through Release or timeout.

## Build target

`KFCore::hand_interaction` publicly depends only on:

- `KFCore::hand_model_core`;
- `KFCore::thig`.

`KFCore::hand_gesture` remains independently buildable for contract/runtime
experiments when `KFCORE_ENABLE_TEMPORAL_GESTURE_GRU=ON`, but it is disabled by
default and production interaction callers neither include nor link it.
