# THIG hand interaction migration

## Decision

Migrate Retro's perception-independent temporal graph into KFCore as
`KFCore::thig`, then add a separate `KFCore::hand_interaction` adapter that
turns `vision_models::HandFrame` values into bounded primitive observations.
CPU ONNX Runtime and TensorRT CUDA remain inference backends only; both feed
the same tracker, primitive extractor, and THIG graph.

```text
CpuHandBackend ---------+
                        +-> HandPipeline -> HandPrimitiveExtractor
TensorRtHandBackend ----+                         |
                                                  v
                                           TemporalGraphEngine
                                                  |
                                                  v
                                  semantic ActionEvent values
```

## Background and alternatives

The existing KFCore hand path exposes Palm boxes, 21 image-space landmarks,
model shape classification, handedness, confidence, and ByteTrack IDs. Retro
adds three independent responsibilities: landmark-derived primitive evidence,
a generic temporal graph, and a Camera/Gallery command adapter.

Three options were considered:

1. Put temporal decisions inside each CPU/GPU backend. Rejected because it
   duplicates state and makes backend output differ.
2. Put all Retro Camera code into `vision_models`. Rejected because layout,
   Gallery commands, FaceMesh references, and UI state are not model concerns.
3. Add a generic THIG library plus a hand adapter. Selected because inference,
   tracking, temporal meaning, and application commands retain distinct owners.

## Public boundaries

`KFCore::thig` owns relation definitions, bounded observation stabilization,
relation history, pattern matching, state graphs, cooldowns, and copied action
evidence. It has no dependency on images, models, trackers, UI, file I/O, or
callbacks.

`KFCore::hand_interaction` consumes a borrowed `HandFrame` plus explicit frame
serial, monotonic timestamp, and image dimensions. It owns canonical hand
identity, per-hand geometry history, pair-distance history, and its THIG engine.
It emits owned primitive observations and action events. It does not retain the
input frame or landmark pointers.

Canonical identity is backend-neutral. CPU ONNX Runtime and TensorRT CUDA only
produce `HandFrame`; the common extractor performs identity resolution, so the
same input frame has the same identity semantics regardless of inference
backend. The installed core target has no OpenCV, CUDA, TensorRT, or ONNX
Runtime dependency; those dependencies stay at the optional demo/backend edge.

Application code remains responsible for mapping semantic actions to commands.
Optional screen-region or face-relative evidence may be appended as external
THIG observations; KFCore does not infer application layout.

## State and bounded-memory protocol

| Item | Contract |
|---|---|
| Data unit | Owned `Observation`, `RelationEvent`, history samples, and `ActionEvent` values |
| Fact source | Current `HandFrame` for raw perception; extractor histories for primitives; THIG event store for temporal facts |
| Ownership | `process()` borrows the frame for the call; extractor and THIG copy all retained fields |
| Topology | One producer and one consumer executor; every mutable instance is single-owner and non-reentrant |
| Ordering | Frame serial and timestamp must both be monotonic |
| Capacity | `max_hands`, samples per hand, canonical identities, pair histories, observations per frame, relation events, observation-window states, and action states have explicit hard limits |
| Full behavior | Invalid configuration and capacity exhaustion throw before partially committing a frame |
| Reset | `reset()` clears identities, primitive histories, THIG relations, cooldowns, and graph state together |
| Shutdown | RAII destruction after the owning executor stops calling the instance; there are no background threads or callbacks |
| Observability | Result exposes primitives/actions; configuration and exceptions identify the failed boundary |

Histories and action state are pruned by monotonic age and bounded by count. No unbounded map,
queue, or vector is allowed on the frame path. THIG action events copy evidence,
so later pruning cannot invalidate returned events.

## Persistent shape-based canonical identity

The identity registry owns one persistent prototype for every canonical hand.
The prototype is Hu-like invariant shape evidence: 20 normalized 3D skeleton-edge
lengths from the five labelled wrist-to-finger chains of the 21-point hand
skeleton. It is a fixed inline value, normalized by the total edge length, and
is compared with L1 distance. It is not OpenCV Hu moments—there is no hand mask
or contour in this core path—and it is not a biometric guarantee.

All landmark components and derived lengths must be finite and the total length
must be positive. A failed descriptor is unconfirmable and produces canonical ID
`0`; it does not fall back to a position-only match. A descriptor farther than
`maximum_shape_distance` is not a candidate. Handedness, raw ByteTrack ID,
capped age, and, while fresh, palm position/scale/velocity only rank compatible
candidates. Ambiguous matches remain ID `0` rather than being resolved by input
order.

Identity prototypes remain until the owning pipeline receives explicit
`reset()`. `reacquire_frames` is therefore the spatial-evidence horizon, not an
identity deletion TTL: palm motion and scale evidence are used only through that
age, while shape-compatible re-acquisition continues after it. A reliable hand
with no compatible prototype allocates a new identity. State remains bounded by
`maximum_identities`; exhaustion throws `std::length_error` before a partial
frame commit.

`maximum_identities` defaults to `32`. `HandIdentityOptions` additionally has
these validated defaults:

| Field | Default | Contract |
|---|---:|---|
| `maximum_shape_distance` | `0.35F` | Candidate L1 gate; must be in `(0, 2]` |
| `shape_cost_weight` | `2.0F` | Non-negative shape-distance ranking weight |
| `shape_update_weight` | `0.20F` | EMA prototype update weight in `(0, 1]` |
| `handedness_mismatch_penalty` | `0.35F` | Non-negative cost for differing known handedness |

The new fields are appended to the public `HandIdentityOptions` structure.
Default construction and short aggregate initializers retain their source
behavior, but the public layout changes; binary downstream consumers must
rebuild.

## Compatibility and migration

Existing `HandInferenceBackend`, `HandPipeline`, `HandFrame`, CPU, and TensorRT
APIs remain unchanged. New targets are additive. KFCore ByteTrack ID zero is
valid; the adapter maps every non-negative raw ID to a positive canonical ID and
never applies Retro's `track_id <= 0` rejection rule.

Model `Gesture::Closed` maps to primitive `Shape Fist`. V, OK, index state,
direction, stationarity, scale, rotation, and two-hand distance are derived from
raw landmarks without modifying them. Handedness is display metadata and never
an entity key. It is secondary categorical evidence for canonical identity, not
a hard key: a known mismatch adds the configured ranking penalty.

The source implementation is derived from the first-party Retro repository at
`C:/projects/project-cpp-template/Retro/interaction/thig`; migration retains the
complete behavior tests and records intentional namespace/build adaptations.

Rollback is additive: disable `KFCORE_BUILD_HAND_INTERACTION` or revert the new
targets. Existing vision inference and tracking behavior is unaffected.

## Verification

- Port the complete Retro THIG behavior suite under TinyTest.
- Test ID zero, timestamp/serial rejection, capacity limits, reset, ambiguity,
  and history pruning.
- Test persistent shape re-acquisition across a frame gap, translation, scale,
  in-plane rotation, and raw Track ID recreation; test shape rejection, swapped
  hands, invalid/degenerate descriptors, and capacity exhaustion.
- Test literal landmark fixtures for V, OK, index state, direction,
  stationarity, scale, rotation, and pair distance.
- Test graph sequences for Wave, Grasp, Release, OK, dual-hand V, Zoom, and
  Rotate without model or GPU mocks.
- Run existing vision core, CPU integration, and TensorRT integration tests to
  prove the common semantic layer does not alter backend contracts.
