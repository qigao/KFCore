# Hand Interaction THIG Robustness

> Superseded for distinct-source freshness and default Swipe sample count by
> `docs/superpowers/specs/2026-09-17-hand-interaction-thig-v5-resilience.md`.

## Decision

Production gesture recognition remains deterministic and THIG-based. This change
hardens the existing graph without changing the generic THIG ABI or introducing a
trained-model dependency.

The semantic graph version becomes `kfcore-hand-interaction-v4` because default
recognition timing and two-hand acceptance rules change observably.

## Problems and invariants

### Interaction source ownership

An idle interaction state graph must not bind merely because an open or fist hand
is visible. A source is claimed only when that same source completes the
open-to-fist stationary Grasp transition. Release and drag then remain bound to
that source until the state graph returns to idle or times out.

The pipeline continues to own one interaction state graph, so it supports one
active grasp/drag lifecycle at a time. Static and spatial actions remain
multi-source.

### Swipe evidence

A Swipe requires an open hand plus sustained left/right direction evidence. The
default direction stabilizer requires at least three supporting observations and
67 ms of winning evidence inside a 167 ms window, retaining at most five samples
per source. This is a deterministic 30 fps-oriented guardrail, not an
empirically-trained threshold.

The primitive extractor remains responsible for deriving direction from its
motion history. THIG adds confirmation and conflict handling; it does not invent
motion observations or silently fall back to a classifier.

### Two-hand static actions

`Two Hand V` and `V Fist` require both participating hands to be stationary while
their shapes overlap. The sources must remain distinct and satisfy the configured
dual-hand dwell and onset window. `V Fist` keeps semantic roles in its evidence:
one source supplies `Shape V`, the other supplies `Shape Fist`.

### Tracker contract

The hand tracking wrapper preserves the underlying ByteTrack confirmation
contract: a newly created track may expose `track_id == -1`; a subsequent matched
frame confirms and assigns the stable non-negative ID. Reset starts this process
again. Tests and documentation must describe this behavior rather than changing
the tracker to manufacture a first-frame identity.

## Compatibility and risks

- `MED`: V4 intentionally rejects one- or two-frame direction spikes that V3
  could report as Swipe. Sustained swipes remain supported with additional
  confirmation latency.
- `MED`: Moving `V Fist` and `Two Hand V` poses no longer activate until both
  hands are stationary. This favors interaction precision over activation while
  hands are repositioning.
- `LOW`: Grasp can now be initiated by a second hand while another open hand is
  visible. Existing single-hand behavior and action names are unchanged.
- No public struct fields, action names, tensor contracts, or generic THIG APIs
  are added or removed.

## Verification

- An unrelated open hand cannot reserve the interaction state graph before a
  second hand performs Grasp.
- Default Swipe settings reject early direction evidence and accept sustained
  evidence at the documented time boundary.
- `V Fist` rejects a moving participant and accepts two stationary distinct
  sources.
- Existing Grasp, Release, drag, OK, Zoom, Rotate, V, Region, reset, and capacity
  behavior remains covered.
- Hand tracking tests cover unconfirmed first-frame IDs, confirmation, stability,
  and reset.
