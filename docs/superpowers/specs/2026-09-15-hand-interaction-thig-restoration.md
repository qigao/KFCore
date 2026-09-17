# Hand Interaction THIG Restoration

## Status

Restoration is complete. The action scope and THIG-only dependency decision remain
current; default timing, source binding, and two-hand composition semantics are
superseded by
`docs/superpowers/specs/2026-09-15-hand-interaction-thig-robustness.md` (V4),
then by `docs/superpowers/specs/2026-09-17-hand-interaction-thig-v5-resilience.md`
for distinct-source freshness and frame-rate-tolerant Swipe confirmation.

## Decision

Production `KFCore::hand_interaction` continues to recognize dynamic gestures with
the deterministic THIG temporal graph because there is currently no trained GRU
artifact suitable for deployment. The GRU runtime and training tools remain
available as experimental infrastructure but are not consumed by
`KFCore::hand_interaction`.

## Gesture scope

The production dynamic gesture set is limited to:

```text
Swipe Left
Swipe Right
Grasp
Release
```

Grasped left/right motion also produces the existing `Drag Start Left/Right`,
`Drag Left/Right`, `Drag End`, and timeout `Drag Cancelled` actions. Wave, Point,
Click, and Pinch are not restored.

## Ownership and interface

`HandInteractionPipeline` owns primitive extraction, canonical identity, and one
`thig::TemporalGraphEngine`. Its public `process()` consumes only the tracked hand
frame, frame context, and optional Region observations. It does not consume
`hand_gesture::GestureEvent`, and the `hand_interaction` target does not publicly
link `KFCore::hand_gesture`.

The graph owns all swipe/grab/release/drag state. A failed validation or THIG frame
must not commit staged primitive state. `reset()` clears both primitive and graph
state.

## Compatibility

This reverses the GRU-based `HandInteractionPipeline::process()` signature and is
therefore a source/ABI change for consumers built against the experimental branch.
The semantic graph version becomes `kfcore-hand-interaction-v3`; no adapter accepts
learned events and no implicit GRU fallback is provided.

Static/spatial actions include `OK`, single-hand V, two-hand V, distinct-hand
`V Fist`, Zoom, and Rotate. External Region observations remain validated and
copied into the returned primitive frame, but do not produce Click actions and are
not fed into the THIG graph.

## Verification

- A left/right open-hand direction produces the corresponding Swipe action.
- Open-to-fist and fist-to-open stationary transitions produce Grasp and Release.
- Fist-direction sequences produce bounded drag actions.
- Concurrent `Shape V` and `Shape Fist` observations from two distinct hands
  produce `V Fist` within the configured dual-hand onset window.
- `HandInteractionPipeline::process(frame, context)` exercises the graph without a
  learned-event argument.
- Wave and Click are absent from the graph.
- `KFCore::hand_interaction` has no `KFCore::hand_gesture` link dependency.
- Existing static/spatial, Region validation, reset, and capacity tests pass.
