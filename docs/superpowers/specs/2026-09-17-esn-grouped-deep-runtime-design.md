# Grouped-Deep ESN Runtime Design

Date: 2026-09-17
Issue: #40
Base: `master` at `1497e34c3ed228d4838044cb8a693277c6e1cfdc`

## Goal

Add grouped-deep Echo State Network (gdESN) topology to KFCore as a pure composition layer over the existing dense deep ESN runtime.

A grouped-deep ESN is an ordered collection of independent deep reservoir chains. Every group consumes the same external input sample at timestep `t`. Inside each group, the existing dESN semantics remain unchanged: layer 0 consumes the external input and each later layer consumes the freshly computed state of the immediately preceding layer from that same timestep. The grouped-deep feature vector is the concatenation of each complete deep-group state in group declaration order.

This slice does not change `kfcore_esn_model`, `kfcore_esn_step`, `kfcore_esn_grouped_model`, `kfcore_esn_grouped_step`, `kfcore_esn_deep_model`, `kfcore_esn_deep_step`, sparse ESN, ridge, or RLS semantics.

## Why this design

Three approaches were considered.

1. **Dedicated grouped composition over the public deep API — selected.** Each group remains an ordinary `kfcore_esn_deep_model`. The grouped-deep runtime validates each group through `kfcore_esn_deep_layout`, stages the complete caller state once at the outer level, invokes `kfcore_esn_deep_step` for each group against its candidate-state slice, and commits caller state only after every group succeeds. This preserves public boundaries and reuses the already verified dESN same-timestep semantics.
2. **Extract a private non-staging deep helper and let gdESN call that helper.** This can reduce duplicate validation and nested state staging, but it couples gdESN to dESN internals and reopens a recently stabilized implementation boundary. That optimization is deferred until profiling shows nested staging is material.
3. **Replace gESN/dESN/gdESN with one generic topology graph/backend descriptor.** This could express more architectures, but it is significantly broader than the required grouped-deep topology and would force redesign of already stable public APIs.

The selected design favors explicit composition and correctness over minimum workspace or minimum memory copies.

## Public model

Add a new public module `esn_group_deep.h` / `esn_group_deep.c`.

The public grouped-deep view is non-owning:

```c
typedef struct kfcore_esn_grouped_deep_model
{
    int group_count;
    const kfcore_esn_deep_model* groups;
} kfcore_esn_grouped_deep_model;
```

`groups` points to an array of existing `kfcore_esn_deep_model` values. Each deep group owns no memory; all layer arrays and all reservoir buffers continue to be caller-owned through the existing dESN contract.

Every group must be a valid dense deep chain according to `kfcore_esn_deep_layout`.

All groups must expose the same positive external input size, defined by each group's first layer:

```text
groups[i].layers[0].input_size == groups[0].layers[0].input_size
```

Groups may have different:

- layer counts;
- layer reservoir sizes;
- leak rates;
- feed-forward/recurrent weights;
- reservoir biases;
- total deep state sizes;
- deep workspace sizes.

Readout fields remain irrelevant because grouped-deep stepping performs no prediction.

The first implementation is dense-only. It never inspects missing dense weights to select sparse recurrence and never changes backend automatically.

## Group and layer execution semantics

All groups consume the same external input vector for timestep `t`.

For two groups, execution is semantically:

```text
group0_state(t) = DEEP_STEP(group0, external_input(t), group0_state(t-1))
group1_state(t) = DEEP_STEP(group1, external_input(t), group1_state(t-1))
```

Groups are independent: no group consumes another group's state.

Within each group, `kfcore_esn_deep_step` preserves the existing ordered same-timestep propagation:

```text
x_g,0(t) = ESN_STEP(layer_g,0, external_input(t), x_g,0(t-1))
x_g,1(t) = ESN_STEP(layer_g,1, x_g,0(t),          x_g,1(t-1))
...
```

Grouped-deep therefore combines parallel group independence with serial same-timestep propagation inside each group.

The implementation may execute groups sequentially in declaration order. That is a deterministic implementation order, not a semantic dependency between groups. Parallel scheduling is outside this slice.

## State and feature layout

Caller state is one contiguous float array using **group-major, depth-minor** ordering.

Each group's state uses the existing dESN depth-order layout internally, then complete group states are concatenated in group declaration order:

```text
[group 0 layer 0][group 0 layer 1]...[group 0 layer N]
[group 1 layer 0][group 1 layer 1]...[group 1 layer M]
...
```

If group 0 has layer reservoir sizes `2, 1` and group 1 has layer reservoir sizes `3, 2, 1`, the public state layout is:

```text
state[0..1] -> group 0 layer 0
state[2]    -> group 0 layer 1
state[3..5] -> group 1 layer 0
state[6..7] -> group 1 layer 1
state[8]    -> group 1 layer 2
```

The total state size is the sum of every group's deep state size. This same contiguous state is the grouped-deep feature vector; no separate feature copy is introduced.

## Layout API

Expose:

```c
kfcore_esn_status kfcore_esn_grouped_deep_layout(
    const kfcore_esn_grouped_deep_model* grouped_deep,
    int* state_size,
    int* workspace_size);
```

For each group, call the public deep layout API:

```c
kfcore_esn_deep_layout(&grouped_deep->groups[i],
                       &group_state_size,
                       &group_workspace_size);
```

The grouped-deep layout validates all groups before producing outputs.

On success:

```text
state_size = sum(group_state_size)
workspace_size = state_size + max(group_workspace_size)
```

The `max(group_workspace_size)` region is reusable because groups execute one at a time.

The layout query also validates that all groups' first layers expose the same external input size. This check occurs only after the individual deep group has passed `kfcore_esn_deep_layout`, so the implementation never dereferences an invalid empty/null layer array merely to inspect its first layer.

`state_size` and `workspace_size` are required outputs. NULL output pointers are invalid. Neither output may be modified on failure.

All arithmetic is checked against `INT_MAX` before narrowing or addition. A failure from an underlying `kfcore_esn_deep_layout` is returned unchanged.

## Grouped-deep step API

Expose:

```c
kfcore_esn_status kfcore_esn_grouped_deep_step(
    const kfcore_esn_grouped_deep_model* grouped_deep,
    const float* input,
    float* state,
    float* workspace);
```

`input` contains the common external input size reported implicitly by the first layer of every group.

`state` contains all deep-group states in the documented group-major/depth-minor order on entry and is replaced only after the complete grouped-deep step succeeds.

`workspace` contains at least the count returned by `kfcore_esn_grouped_deep_layout`.

`input`, `state`, `workspace`, all group layer arrays, and every layer weight/bias buffer follow the existing non-owning contracts. In particular, the writable `state` and `workspace` regions must not overlap each other, the external input, or any model weight/bias storage. The group/deep descriptor structs themselves are metadata views and are not mutated.

## Nested staging and workspace policy

The first grouped-deep implementation intentionally uses two levels of staging.

Outer grouped-deep staging protects the complete caller-visible state across groups. Inner dESN staging protects each candidate deep-group state across layers.

For example, if:

```text
group 0: deep state = 5, deep workspace = 8
group 1: deep state = 9, deep workspace = 13
group 2: deep state = 4, deep workspace = 6
```

then:

```text
grouped-deep state = 5 + 9 + 4 = 18
grouped-deep workspace = 18 + max(8, 13, 6) = 31 floats
```

The first `18` workspace floats are the outer candidate state. The remaining `13` floats are reusable inner dESN workspace.

This is not the mathematically minimum workspace. The design deliberately accepts the additional staging copy so gdESN can depend only on the public dESN API rather than a private backend helper.

There is no hidden allocation to compensate for insufficient caller workspace and no alternate low-memory path.

## Execution algorithm

Grouped-deep stepping composes `kfcore_esn_deep_step` rather than duplicating deep-chain logic.

1. Validate required public pointers.
2. Validate the complete grouped-deep model and compute checked layout before any workspace staging. Validation calls `kfcore_esn_deep_layout` for every group, verifies the common external input size, sums all group state sizes, and finds the largest deep workspace requirement.
3. Copy caller `state` into the first `state_size` floats of `workspace`; this becomes the outer candidate state.
4. Set `inner_workspace = workspace + state_size`.
5. Iterate groups in declaration order. For each group:
   - identify the group's candidate-state slice using its previously validated deep state size;
   - call `kfcore_esn_deep_step(group, input, candidate_slice, inner_workspace)`;
   - if the delegated deep step returns non-OK, return that status immediately without modifying caller state;
   - advance the candidate offset by that group's deep state size.
6. After every group succeeds, copy the complete outer candidate state back into caller `state` once.

Earlier groups may have advanced inside the outer candidate if a later group fails. That is allowed: workspace is scratch. Caller state remains unchanged until the final commit.

## Validation strategy

The implementation may use a private measurement helper that loops over groups and calls `kfcore_esn_deep_layout` for each group.

The helper records only scalar measurements required by the current operation:

- common external input size;
- total grouped-deep state size;
- maximum group deep workspace size.

It does not allocate or persist a per-group layout table.

Because the execution loop later needs each group's state offset, it calls `kfcore_esn_deep_layout` again for the current group before its `kfcore_esn_deep_step`, or otherwise recomputes that scalar through the same public API. This duplicate validation is acceptable in v1 and keeps the implementation allocation-free without storing variable-length metadata.

No private dESN implementation helper is exposed or shared in #40.

## Failure semantics

Grouped-deep execution is failure-atomic with respect to caller state.

The following are rejected with `KFCORE_ESN_INVALID_ARGUMENT` before caller state is staged:

- NULL grouped-deep model, group array, input, state, or workspace;
- non-positive `group_count`;
- any invalid deep group rejected by `kfcore_esn_deep_layout`;
- different external input sizes across groups;
- grouped state-size overflow;
- grouped workspace-size overflow.

If an underlying `kfcore_esn_deep_layout` or `kfcore_esn_deep_step` returns another non-OK status, grouped-deep returns that status unchanged.

After staging begins, workspace may be modified on failure. Caller state remains unchanged unless all groups succeed.

There is no retry, group skip, rollback heuristic, sparse fallback, alternate topology, or partial caller-state commit.

## Readout and training boundary

This slice does not create grouped-deep-specific readout or training objects.

The complete grouped-deep state is an ordinary dense feature vector. Callers may collect it and use existing ridge/RLS APIs with feature size equal to total grouped-deep state size where those APIs' contracts otherwise apply.

No per-group or per-layer predictions are produced by `kfcore_esn_grouped_deep_step`.

## Relationship to existing gESN

The existing `kfcore_esn_grouped_model` remains a composition of independent single-reservoir ESNs sharing one external input.

Grouped-deep does not retrofit gESN so that a group can dynamically be either a single reservoir or a deep chain. Instead, gdESN is a separate explicit topology over deep groups.

This avoids a tagged backend union or generic graph abstraction in the current slice and preserves existing gESN ABI/API behavior.

A one-layer deep group is mathematically capable of representing the same reservoir behavior as one gESN group, but the public types remain distinct.

## Sparse boundary

The existing sparse runtime remains independent.

Grouped-deep v1 accepts only dense `kfcore_esn_deep_model` groups and delegates to `kfcore_esn_deep_step`. It does not inspect missing dense weights and switch to sparse stepping.

A later backend-descriptor design may explicitly allow sparse deep layers, but there is no automatic dense/sparse fallback here.

## Files

Planned implementation files:

```text
esn/esn_group_deep.h
esn/esn_group_deep.c
esn/tests/test_esn_group_deep.c
```

Existing integration points to extend:

```text
esn/CMakeLists.txt
esn/tests/CMakeLists.txt
.github/workflows/esn-miniblas.yml
```

The following production files should not require mathematical or behavioral changes:

```text
esn/esn.c
esn/esn_group.c
esn/esn_deep.c
esn/esn_sparse.c
esn/esn_reservoir.c
esn/esn_rls.c
```

## Verification contract

The implementation must preserve a clean RED -> GREEN sequence.

1. **Layout contract:** heterogeneous deep groups return exact total state size and `total + max(group deep workspace)` workspace size.
2. **One-group equivalence:** one-group grouped-deep execution matches ordinary `kfcore_esn_deep_step` for the same deep model, input, initial state, and sufficient workspace.
3. **Common external input:** two groups with different first-layer input sizes are rejected by layout and step before caller-state commit.
4. **Group-major/depth-minor ordering:** heterogeneous groups with different depths and reservoir sizes occupy exactly the documented offsets.
5. **Nested same-timestep propagation:** an analytical multi-group case proves each group's deeper layer consumes that group's freshly computed upstream state from the same timestep.
6. **Later-group invalidity atomicity:** a valid earlier group followed by an invalid later deep group leaves complete caller state byte-for-byte unchanged.
7. **Delegated failure atomicity:** any non-OK delegated group step leaves caller state unchanged even if earlier candidate groups advanced.
8. **NULL handling:** all required public pointers are rejected; supplied caller state remains unchanged.
9. **Checked state overflow:** valid individual group layouts whose summed state sizes exceed `INT_MAX` are rejected without partial outputs.
10. **Checked workspace overflow:** a representable grouped total state plus maximum group workspace that exceeds `INT_MAX` is rejected without partial outputs.
11. **No readout requirement:** all layer output fields may be NULL and do not prevent grouped-deep layout/step execution.
12. **Existing regression gate:** the complete focused C11 ASan+UBSan ESN suite remains GREEN.

The first layout RED must fail only because `kfcore_esn_grouped_deep_layout` is absent. Compiler, TinyTest harness, or unrelated linker failures do not count.

After layout GREEN, the step RED must fail only because `kfcore_esn_grouped_deep_step` is absent.

## Scope exclusions

The following are explicitly outside #40:

- sparse grouped-deep backend dispatch;
- extracting/reusing private non-staging dESN internals;
- reducing nested staging copies/workspace;
- generic graph/topology descriptors;
- modifying existing gESN/dESN public semantics;
- parallel group scheduling;
- grouped-deep-specific readout/training APIs;
- external-input skip connections to deeper layers;
- residual or concatenated layer inputs;
- inter-group recurrent/feed-forward connections;
- automatic topology/reservoir construction;
- random initialization policy changes;
- serialization/persistence;
- bias adaptation or backpropagation;
- application-specific behavior/gesture integration.

These exclusions keep #40 focused on explicit composition of already verified dESN groups, deterministic feature layout, and complete caller-state failure atomicity.