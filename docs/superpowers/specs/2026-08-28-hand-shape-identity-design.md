# Hand-Shape Canonical Identity Design

## Context

`HandTrackIdentityRegistry` currently derives canonical hand identity from a
short-lived ByteTrack ID, palm center, palm scale, velocity, and frame age.
Although these signals are useful for nearby consecutive frames, none describes
the hand itself. Normal detector or landmark jitter can therefore reject the
only previous identity, while an absence longer than `reacquire_frames` deletes
the identity completely. THIG actions consume the resulting canonical ID, so an
identity change also resets the semantic history associated with that hand.

The hand landmark model already returns 21 labelled three-dimensional points
and handedness for every accepted hand. Those values are the authoritative
feature source for this change.

## Decision

Canonical identity uses a normalized hand-shape descriptor for continuity within
a bounded `reacquire_frames` retention horizon. Palm position, palm size,
motion, handedness, and raw ByteTrack ID remain supporting evidence rather than
a person-recognition key.

The descriptor is the 20 MediaPipe skeleton-edge lengths in `(x, y, z)` space,
normalized by their sum. It has the same useful invariance goal as Hu moments:
translation, uniform scale, and in-plane rotation do not change it. Labelled
bone lengths retain more anatomical information than moments of an unordered
point cloud and do not require a segmented hand silhouette.

Exact OpenCV Hu moments are not used because the core pipeline has no binary
hand mask. Adding contour segmentation would introduce a new and unstable fact
source, and linking OpenCV into `KFCore::hand_interaction` would violate the
existing dependency boundary where OpenCV Lite is demo-only.

## Data flow and ownership

```text
HandResult landmarks + handedness
  -> make_hand_shape_descriptor (fixed value, no ownership transfer)
  -> HandIdentityObservation (owned value for one resolver call)
  -> HandTrackIdentityRegistry::IdentityState (retention-limited prototype)
  -> canonical Hand ID
```

`HandTrackIdentityRegistry` remains the sole owner of canonical identity state.
Each `IdentityState` owns one fixed descriptor prototype and the latest spatial
evidence. There is no shared mutable state and one registry instance remains
single-threaded, as before.

## Descriptor contract

- The 20 edges are the four consecutive bones of each thumb/finger chain,
  beginning at the wrist for every chain.
- Every landmark component and every derived length must be finite.
- The total bone length must be positive; otherwise the observation is
  unconfirmable and receives canonical ID `0`.
- Descriptor values are non-negative and sum to one within floating-point
  tolerance.
- Descriptor distance is the L1 distance, in `[0, 2]`.
- Prototype updates use an exponential moving average followed by L1
  renormalization. Invalid values fail the observation instead of falling back
  to position-only matching.

## Matching rules

1. Reject observations below `minimum_confidence` or without valid palm geometry
   or a valid shape descriptor.
2. A stored identity is a candidate only when shape distance is at most
   `maximum_shape_distance`.
3. A known left/right handedness disagreement rejects a candidate; `Unknown`
   remains compatible. Shape distance contributes `shape_cost_weight * distance`
   to the candidate cost.
4. Before matching, identities older than `reacquire_frames` are removed. Within
   that horizon, predicted palm distance is divided by
   `maximum_distance_scale_ratio` and clamped to `[0, 1]`; logarithmic scale
   difference is divided by `log(maximum_linear_scale_ratio)` and clamped to
   `[0, 1]`, then weighted by `scale_cost_weight`.
5. If two observations share a top canonical candidate and their top costs differ
   by less than `ambiguity_cost_margin`, neither receives it. This is evaluated
   across the frame, so input order cannot decide the result.
6. A reliable observation with no shape-compatible candidate allocates a new
   identity. Exceeding `maximum_identities` remains a fail-fast capacity error.

## Public configuration

The following fields are appended to `HandIdentityOptions`:

| Field | Default | Meaning |
|---|---:|---|
| `maximum_shape_distance` | `0.20` | Conservative L1 candidate gate; calibrate for landmark noise |
| `shape_cost_weight` | `2.0` | Shape contribution to assignment cost |
| `shape_update_weight` | `0.20` | EMA weight of the newest accepted descriptor |
| `handedness_mismatch_penalty` | `0.35` | Validated retained layout field; known mismatch is a hard gate |

`maximum_identities` changes from `8` to `32`, bounding retention-window state
while allowing transient false detections and multiple people without immediate
capacity exhaustion. All new thresholds are validated at construction; invalid
configuration throws `std::invalid_argument` before any frame is consumed.

## Complexity and resource effects

Descriptor construction is `O(20)` time and `O(20)` inline storage per detected
hand. Matching plus candidate sorting is `O(H * (20I + I log I))` time for `H`
observations and `I` retained identities; staged state copy and expiry pruning
are `O(I)`. With the public limits `H <= 8` and default `I <= 32`, the descriptor
term has at most 5,120 scalar differences per processed frame. Identity storage
is bounded by `maximum_identities`; the descriptor adds 80 bytes per identity
before alignment and existing state.

The descriptor itself performs no heap allocation, logging, I/O, locking, or
virtual dispatch. Model inference remains the dominant frame cost; a SIMD path
is not justified for 20-element arrays without profiling evidence.

## Compatibility, migration, and rollback

Appending fields preserves ordinary source use of default construction and
short aggregate initializers, but changes the public structure layout. Binary
consumers of `KFCore::hand_interaction` must be rebuilt. The visible behavior
also changes: identities expire after the bounded `reacquire_frames` retention
horizon, so a feature-compatible hand after that horizon receives a new ID.

Rollback consists of reverting the descriptor source, the four option fields,
the increased capacity default, and the resolver matching changes. No stored
data or external format requires migration.

## Validation

- A hand retains its ID across translation, uniform scale, in-plane rotation,
  raw Track ID recreation inside `reacquire_frames` empty frames.
- A shape-compatible hand after the retention horizon receives a new ID.
- Small independent landmark jitter retains the ID.
- A hand with materially different bone proportions at the same position gets a
  different ID.
- Two differently shaped hands retain distinct IDs when their positions swap
  inside the retention horizon.
- Shape-identical candidates remain unconfirmed when supporting evidence cannot
  disambiguate them.
- Non-finite or degenerate landmarks yield ID `0`; invalid thresholds fail at
  construction.
- CPU and TensorRT presets pass their focused and full CTest suites.
