# Hand Interaction THIG V5 Resilience

## Decision

Keep production gesture recognition deterministic and THIG-based. V5 tightens
freshness only for concurrent evidence joined from distinct sources and makes
direction confirmation primarily time-based across common camera frame rates.

No public field, action name, dependency, tensor contract, or configuration
format changes. The semantic graph version becomes
`kfcore-hand-interaction-v5` because the shared direction sampling rule and
distinct-source acceptance behavior change observably.

## Distinct-source freshness

Same-source `Both` retains its existing short-dropout behavior: one active
operand may be temporarily `UNKNOWN` while another operand is observed in the
current frame. This is required by existing THIG dropout semantics.

For `PatternSourceJoin::Distinct`, both participating source matches must contain
evidence observed in the current frame. Historical overlap from a disappeared
source must not combine with a currently observed second source to create a new
two-hand action. Interval overlap, onset-window, dwell, and distinct-identity
requirements remain unchanged.

## Direction frame-rate tolerance

The default direction window remains 167 ms, direction dwell remains 67 ms,
minimum support remains 0.60, switch margin remains 0.15, and at most five
samples are retained. The minimum supporting observation count changes from
three to two.

The two-sample floor rejects a single direction spike. The independent 67 ms
dwell still prevents high-frame-rate streams from confirming earlier merely
because they provide samples more frequently. At low frame rates, two samples
separated by at least 67 ms can confirm without requiring a third sample that
may fall outside the 167 ms window.

The direction observation window is shared by Swipe, Drag Start/Drag, direction
switching, and `Direction Neutral` rearm. All of those consumers therefore use
the two-sample floor. Their relation-specific dwell, graph state, source binding,
support ratio, and switch margin remain unchanged.

The existing conflict behavior remains: a raw competing direction immediately
ends the active relation for safety, but the stable candidate is retained. A
single conflict therefore cannot activate the competing direction, and a return
to the prior direction can recover without rebuilding the entire window.

## Compatibility and risks

- `MED`: Two-hand actions no longer activate when either participant lacks
  current-frame evidence. This intentionally reduces tolerance to a complete
  one-hand detector dropout at the exact activation frame while preventing
  ghost activations.
- `MED`: At low frame rates, a sustained Swipe, Drag direction, or Neutral rearm
  may confirm after two samples instead of three. The 67 ms direction duration
  still applies to Swipe and Drag; relation-specific graph state, support ratio,
  conflict rejection, and shape/source constraints remain in force.
- `LOW`: Generic same-source THIG `Both` behavior is unchanged.
- `LOW`: Direction support ratio, switch margin, and state ownership are
  unchanged; the minimum sample threshold intentionally changes.

## Verification

- A historical first source plus a current second source cannot trigger a
  distinct-source `Both` action.
- Current containers cannot revive ended `During` intervals inside a distinct
  concurrent match.
- Same-source `Both` still survives the existing bounded observed dropout.
- `V Fist` and `Two Hand V` reject an absent historical participant.
- Default Swipe confirms at or after 67 ms for jittered 12, 30, 60, and 120 FPS
  timestamp sequences and rejects all earlier frames.
- The 67 ms dwell and 167 ms window boundaries are inclusive; 168 ms falls
  outside the window.
- Drag direction and post-drag Neutral rearm use the same two-sample direction
  stabilizer.
- A single opposite-direction frame suppresses the prior direction for that
  frame, cannot activate the opposite action, and permits the prior direction
  to recover.
