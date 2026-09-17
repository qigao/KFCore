# Grouped ESN Runtime Design

Date: 2026-09-17
Issue: #36
Base: `master` at `61ebd0793ce0cc55018c6d4af7fb07131e898d90`

## Goal

Add the first composite Echo State Network topology to KFCore: grouped ESN (gESN). A grouped ESN contains multiple independent dense reservoirs. Every group consumes the same input sample, each group advances its own state, and the grouped feature vector is the concatenation of group states in declaration order.

This design is intentionally a composition layer over the existing single-reservoir ESN runtime. It does not replace or generalize `kfcore_esn_step`, and it does not introduce deep or grouped-deep topology in the same change.

## Why this design

Three approaches were considered.

1. **Dedicated composition layer over existing `kfcore_esn_model` — selected.** Each group is an existing ESN model. Grouped execution owns no weights or state and only coordinates layout, validation, atomic stepping, and state concatenation. This preserves the current single-reservoir API and gives later deep/gdESN work an explicit composition boundary.
2. **Generalize the existing model to a dense/sparse/composite backend union.** This would make topology dispatch more uniform, but it would force a broad redesign of the now-stable single-reservoir and sparse APIs before grouped behavior is proven useful.
3. **Flatten all groups into one block-diagonal reservoir.** This can reproduce grouped recurrence mathematically, but it erases the group boundary, makes heterogeneous group metadata awkward, and works against later grouped/deep composition.

The selected approach is the smallest design that preserves group identity and failure semantics without destabilizing existing APIs.

## Public model

Add a new public module `esn_group.h` / `esn_group.c`.

The public grouped view is non-owning:

```c
typedef struct kfcore_esn_grouped_model
{
    int group_count;
    const kfcore_esn_model* groups;
} kfcore_esn_grouped_model;
```

`groups` points to an array of existing `kfcore_esn_model` values. Only reservoir-side fields are required by grouped stepping: `input_size`, `reservoir_size`, `leak_rate`, `input_weights`, `reservoir_weights`, and `reservoir_bias`. Readout fields may be NULL because grouped execution does not perform prediction.

All groups must have the same positive `input_size`. Groups may have different reservoir sizes, leak rates, recurrent weights, input weights, and biases.

The first grouped implementation is dense-only. Sparse grouped execution is a later slice rather than an implicit backend fallback.

## State and feature layout

Caller state is one contiguous float array. Group state is laid out by group array order:

```text
[group 0 state][group 1 state]...[group N-1 state]
```

The total state size is the sum of every `reservoir_size`. This concatenated state is also the grouped feature vector. No separate feature-output buffer or copy is introduced.

For groups with reservoir sizes 2, 3, and 1, the state layout is:

```text
s[0..1]   -> group 0
s[2..4]   -> group 1
s[5]      -> group 2
```

This ordering is part of the public contract and is deterministic.

## Layout API

Expose one layout query:

```c
kfcore_esn_status kfcore_esn_grouped_layout(
    const kfcore_esn_grouped_model* grouped,
    int* state_size,
    int* workspace_size);
```

On success:

- `state_size = sum(group.reservoir_size)`;
- `workspace_size = state_size + max(group.reservoir_size)`.

The workspace formula supports atomic execution with one staged candidate-state vector plus one reusable per-group scratch vector.

The function validates the full grouped reservoir contract, including common input size, and rejects integer overflow. It does not dereference or require readout fields.

`state_size` and `workspace_size` are required outputs. NULL output pointers are invalid.

## Grouped step API

Expose:

```c
kfcore_esn_status kfcore_esn_grouped_step(
    const kfcore_esn_grouped_model* grouped,
    const float* input,
    float* state,
    float* workspace);
```

`input` contains the common group input vector. `state` contains the concatenated group state and is updated only after all groups successfully compute their candidate next state. `workspace` contains at least the amount reported by `kfcore_esn_grouped_layout`.

`state`, `workspace`, and `input` must not overlap each other or group weight/bias storage.

## Execution algorithm

Grouped stepping uses the existing dense single-reservoir implementation instead of duplicating its recurrence math.

1. Validate the grouped model and all group reservoir contracts before mutation.
2. Compute total state size and maximum group reservoir size with checked integer arithmetic.
3. Copy the caller state into the first `state_size` floats of workspace. This is the candidate concatenated state.
4. Use the remaining `max_reservoir_size` floats as reusable scratch.
5. For each group in order:
   - locate that group's candidate state slice;
   - invoke `kfcore_esn_step(group, input, candidate_slice, scratch)`;
   - if it returns any non-OK status, propagate that status and leave caller state untouched.
6. Only after all groups succeed, copy the complete candidate state back to caller state.

The caller-visible grouped state therefore has transaction-like step semantics even though individual `kfcore_esn_step` calls mutate staged candidate state in place.

## Failure semantics

Grouped execution is failure-atomic with respect to caller state.

The following return `KFCORE_ESN_INVALID_ARGUMENT` before caller-state mutation:

- NULL grouped model, group array, input, state, or workspace;
- non-positive group count;
- invalid single-group reservoir configuration;
- inconsistent group `input_size`;
- invalid or overflowing state/workspace layout;
- required output pointers missing from the layout query.

Any non-OK status returned by a staged `kfcore_esn_step` is propagated and caller state remains unchanged. The current dense step implementation has no stable public input that intentionally produces `KFCORE_ESN_NUMERICAL_FAILURE`, so #36 does not modify that implementation merely to manufacture such a test case.

There is no retry, group skip, state rollback heuristic, alternate solver, sparse fallback, or partial commit. Workspace is scratch and may be modified on any call, including failures.

## Readout and training boundary

This slice does not create a grouped-specific readout. The concatenated grouped state is an ordinary dense feature vector and can be collected by callers for the existing ridge or RLS training paths using the total grouped state size.

No group-local predictions are produced by grouped stepping. This avoids coupling topology composition to a readout policy and leaves later deep/gdESN work free to reuse the same concatenated-state concept.

## Sparse boundary

The existing sparse reservoir runtime remains independent. Grouped ESN v1 accepts only dense `kfcore_esn_model` groups and calls `kfcore_esn_step`.

A future grouped sparse slice may introduce explicit per-group backend descriptors. The current API must not silently inspect `reservoir_weights == NULL` and switch to sparse recurrence because that would introduce fallback/implicit dispatch semantics.

## Files

Planned implementation files:

```text
esn/esn_group.h
esn/esn_group.c
esn/tests/test_esn_group.c
```

`esn/CMakeLists.txt`, `esn/tests/CMakeLists.txt`, and the focused ESN sanitizer workflow will be extended to build/test the module and install its public header.

No existing ESN source file needs a mathematical rewrite.

## Verification contract

The RED test slice must establish these requirements before production symbols are added:

1. **Single-group equivalence:** one grouped reservoir produces the same next state as `kfcore_esn_step` for the same model/input/state.
2. **Same-input parallel composition:** two independent groups both consume one input and produce analytically expected results.
3. **Heterogeneous sizes:** groups with different reservoir sizes occupy the documented concatenated offsets in declaration order.
4. **Layout query:** total state and workspace sizes match `sum + max`, including one-group behavior.
5. **Input agreement:** mismatched group input dimensions fail before state mutation.
6. **Later-group invalidity is atomic:** a valid first group followed by malformed second-group metadata is rejected before any caller-state commit.
7. **Overflow/NULL handling:** invalid counts, required NULL output pointers, and checked-size overflow are rejected.
8. **Existing regression gate:** the full focused ESN C11 ASan+UBSan suite remains GREEN.

The clean RED must fail only because grouped production symbols are absent. Test harness or unrelated compile failures do not count as the algorithm RED.

## Scope exclusions

The following are explicitly outside #36:

- sparse grouped backend dispatch;
- deep ESN and grouped-deep ESN;
- grouped-specific readout objects;
- automatic group/reservoir construction;
- random initialization policy changes;
- serialization/persistence;
- bias adaptation or backpropagation;
- application-specific gesture/behavior integration;
- parallel threads/SIMD scheduling across groups.

These exclusions keep #36 focused on composition semantics, deterministic layout, and failure-atomic runtime behavior.