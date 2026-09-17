# Hand Tracking Resilience

## Decision

Keep ByteTrack as the bounded short-term palm-box association layer. Do not
replace it before recorded-sequence measurements establish a reproducible
baseline and explicit acceptance thresholds for canonical identity and gesture
continuity.

`HandTracker` continues to produce raw `track_id` values. `HandPrimitiveExtractor`
continues to own the canonical hand identity consumed by THIG. This preserves one
semantic identity fact source and avoids coupling THIG state to a replaceable
tracking backend.

## Contract

### Raw tracking

- A new ByteTrack track is unconfirmed and reports `track_id == -1`.
- A track reports a non-negative raw ID after it reaches the configured
  `minimum_consecutive_frames` successful-update threshold.
- A detection gap shorter than the configured lost-track buffer preserves the
  confirmed raw ID when the palm is reacquired spatially.
- A gap at or beyond the buffer does not revive the expired raw identity; the
  next detection starts unconfirmed.
- No appearance, landmark, or hand-shape semantics are added to ByteTrack itself.

### Canonical identity and THIG

- THIG consumes canonical IDs, never raw ByteTrack IDs directly.
- A reliable hand whose raw ID changes may retain its canonical ID through the
  existing shape, geometry, motion, handedness, and optional appearance evidence.
- An Open-to-Fist Grasp may complete across a raw-ID replacement when canonical
  evidence remains unambiguous.
- Ambiguous identity remains `canonical_id == 0` and emits no gesture primitives;
  the system must not guess and attach temporal state to the wrong hand.

### Appearance

Appearance remains opt-in. Enabling it requires the source-image overload of
`HandTracker::update`; changing the default would break source-free callers and
is outside this change.

## Compatibility

- No public API, default option, ABI, action name, or dependency changes.
- Existing ByteTrack configuration remains the raw-tracker fact source.
- Existing canonical identity thresholds remain configurable and unchanged.

## Replacement evaluation status

This change establishes the measurements, not their acceptance thresholds. No
recorded tracking corpus or reproducible ByteTrack baseline is present in the
repository, so an alternative tracker decision remains blocked until the corpus,
metric calculations, baseline results, and thresholds are reviewed together.

## Verification

- Raw tracker tests cover initial confirmation, short dropout recovery, buffer
  expiry, and reset.
- A hand-interaction integration test changes raw ID during Open-to-Fist and
  verifies one canonical identity and a Grasp action from that identity.
- Existing crossing, appearance reacquisition, ambiguity, malformed-input,
  primitive, THIG, and interaction tests remain green.
- A future alternative-tracker evaluation defines recorded-data scope, metric
  calculations, ByteTrack baseline results, and acceptance thresholds for ID
  switches, track fragmentation, canonical reacquisition rate, ambiguous-frame
  duration, false activations, and gesture cancellation.
