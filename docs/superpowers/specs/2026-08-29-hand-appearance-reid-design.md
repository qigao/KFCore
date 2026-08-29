# Hand Appearance Re-ID Design

## Context

The canonical hand registry combines ByteTrack continuity, palm motion, scale,
handedness, and a normalized 20-edge landmark-shape descriptor. Two hands with
similar proportions can still become indistinguishable during a crossing. The
image contains additional appearance evidence in both the palm and fingers, but
the repository has no validated hand-specific learned Re-ID model.

## Decision

`HandPipeline` optionally produces one fixed full-hand descriptor partitioned into
`Palm`, `Thumb`, `Index`, `Middle`, `Ring`, and `Pinky`. This contract replaces the
earlier palm-only layout; there is no V1 kind, conversion, or compatibility path.
All downstream C++ binaries must be rebuilt.

The built-in extractor accepts borrowed Host BGR8/RGB8 input. It uses landmarks to
sample a palm grid and five finger-aligned strips after inference and before the
transactional ByteTrack commit. It retains no image pointer. CUDA-device input
fails before inference when built-in extraction is enabled; no implicit download
or CPU fallback is performed.

`HandTrackIdentityRegistry` remains the sole owner of canonical identity state.
It stores a bounded per-part EMA prototype. Appearance is supporting Re-ID
evidence, not biometric identity and not a cross-session identity guarantee.

## Data flow and ownership

```text
borrowed Host BGR/RGB ImageView + owned landmarks
  -> six aligned fixed-size appearance parts
  -> HandResult::appearance (owned 256-float value + mask/quality)
  -> one-call HandIdentityObservation
  -> retention-limited per-part EMA prototype
  -> canonical ID and association reason
  -> THIG only for confirmed canonical IDs
```

The synchronous `HandPipeline::process()` call is the only borrower of source
pixels. Descriptor output, resolver input, and retained identity state are owned
values. There is no background queue or shared mutable descriptor state.

## Descriptor contract

- Total size is 256 floats.
- Palm contributes 96 values: 48 normalized luminance samples from a 6x8 grid,
  followed by 24 red and 24 green chromaticity values.
- Each finger contributes 32 values: 16 normalized luminance samples from an 8x2
  landmark-aligned strip, followed by 8 red and 8 green chromaticity values.
- `valid_parts` identifies the available regions; `quality[6]` records each valid
  region's in-frame sample ratio. Invalid regions have zero quality.
- Luminance values are finite in `[-1, 1]`; chromaticity values are finite in
  `[0, 1]`.
- A part is absent when its coverage is below
  `minimum_part_in_frame_sample_ratio`. A degenerate palm produces no descriptor.
- Extraction is deterministic, respects row stride, and supports BGR8/RGB8 order.

## Matching and update rules

1. Confidence, finite geometry/shape, retention horizon, shape gate, and known
   handedness compatibility remain mandatory.
2. Only common valid parts are compared. At least
   `minimum_comparable_appearance_parts` are required before appearance can affect
   candidate assignment or produce `AppearanceReacquired`.
3. Each part uses mean absolute component distance. The quality-weighted overall
   distance (`maximum_appearance_distance`) and worst common part
   (`maximum_appearance_part_distance`) classify appearance as compatible. A pose
   change can move or occlude finger texture, so incompatibility is bounded
   negative cost rather than a candidate veto; shape and motion can still reacquire
   the same physical hand after ByteTrack changes its raw ID.
4. Compatible appearance contributes its overall distance to assignment cost.
   Incompatible appearance contributes the larger of overall and worst-part
   distance. This preserves distinctive-finger separation without turning a
   gesture change into a new canonical identity.
5. Accepted observations update only their valid parts by
   `appearance_update_weight`. Missing/occluded parts retain their prototype.
   Ambiguous and rejected observations never update state.
6. Exact equal-cost occlusion remains canonical ID `0`; the registry does not
   guess a physical identity without discriminating evidence.

## Public configuration and compatibility

`vision_models` exposes the six-part descriptor, `HandAppearanceOptions`, optional
`HandResult::appearance`, and `StageTimings::appearance_ms`. Extraction defaults
off. The demo enables it because Turbo Capture provides Host BGR input for both
CPU and TensorRT inference backends.

`hand_interaction` exposes overall/part gates, cost/update weights, minimum common
parts, and `AppearanceReacquired`. Public C++ object layouts and descriptor source
contracts changed. No V1 descriptor is accepted or converted; downstream code
must compile against the new header. No serialized or persisted identity format
exists, so no data migration is required.

## Complexity and resources

Extraction samples a fixed 48 palm pixels plus 16 pixels per finger: `O(H * 128)`
for `H` hands. Matching is `O(H * I * 256)`. With the defaults `H = 8` and
`I = 32`, the upper bound is 65,536 scalar differences per frame.

The value array occupies 1,024 bytes per descriptor. Eight outputs plus 32 retained
states use about 40 KiB for value arrays, plus masks, quality, optional-object
alignment, and existing state. Capacity remains bounded by `max_hands` and
`maximum_identities`. `appearance_ms` reports extraction cost separately; no SIMD,
new thread, CUDA kernel, OpenCV, or learned-model dependency is introduced.

## Failure, rollback, and verification

Unsupported memory/format/stride/size fails before backend inference. Invalid
descriptor masks, qualities, ranges, or non-finite values fail before identity
state mutation. A partially out-of-frame hand simply omits insufficient parts;
this is explicit missing evidence, not a hidden alternate algorithm.

Verification covers RGB/BGR equivalence, brightness stability, full six-part
extraction, finger/palm isolation, invalid image/config contracts, distinctive
finger separation during crossings, raw-ID changes combined with gesture-driven
appearance changes, occluded-part prototype retention, exact
overlap ambiguity, association text/timing display, and complete CPU/TensorRT
preset suites. Runtime validation must rebuild and restart the demo because the
previous running executable contains the obsolete descriptor layout.
