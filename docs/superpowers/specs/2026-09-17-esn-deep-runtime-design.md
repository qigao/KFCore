# Deep ESN Runtime Design

Date: 2026-09-17
Issue: #38
Base: `master` at `289651cc3a8f21b2dc400dc46b781da84f936dfa`

## Goal

Add the dense deep Echo State Network (dESN) topology to KFCore as a pure composition layer over the existing single-reservoir ESN runtime.

A deep ESN is an ordered chain of independent reservoirs. Layer 0 consumes the external input sample at timestep `t`. Every later layer consumes the **newly computed state of the immediately previous layer from that same timestep `t`**. Each layer keeps its own recurrent state across timesteps through the existing `kfcore_esn_step` recurrence. The deep feature vector is the concatenation of every layer state in declaration/depth order.

This slice does not change `kfcore_esn_model`, `kfcore_esn_step`, grouped ESN, sparse ESN, ridge, or RLS semantics.

## Why this design

Three approaches were considered.

1. **Dedicated ordered composition over existing `kfcore_esn_model` — selected.** Each layer remains an ordinary dense ESN model. The deep runtime validates layer-to-layer dimensions, stages candidate state, feeds each freshly computed layer state to the next layer, and commits the complete caller state only after the full chain succeeds. This reuses existing recurrence math and keeps deep topology explicit.
2. **Concatenate external input with the previous layer state at every depth.** This can represent skip-input/deep variants, but it changes the layer input contract, needs extra concatenation workspace, and introduces a policy that is not required for the first dESN slice.
3. **Flatten the chain into one larger recurrent matrix.** This hides layer identity and same-timestep feed-forward ordering inside a custom matrix construction. It would also make later grouped-deep composition and heterogeneous layer metadata harder to express.

The selected design is the smallest topology that preserves explicit depth and directly supports later gdESN composition.

## Public model

Add a new public module `esn_deep.h` / `esn_deep.c`.

The public deep view is non-owning:

```c
typedef struct kfcore_esn_deep_model
{
    int layer_count;
    const kfcore_esn_model* layers;
} kfcore_esn_deep_model;
```

`layers` points to an array of existing `kfcore_esn_model` values. Only reservoir-side fields are required by deep stepping:

- `input_size`;
- `reservoir_size`;
- `leak_rate`;
- `input_weights`;
- `reservoir_weights`;
- `reservoir_bias`.

Readout fields may be NULL because deep execution does not perform prediction.

Layer 0 must have a positive `input_size`; that value is the external input dimension for the complete deep model.

For every layer `i > 0`:

```text
layers[i].input_size == layers[i - 1].reservoir_size
```

Layers may use different reservoir sizes, leak rates, recurrent weights, input weights, and biases.

The first deep implementation is dense-only. It never inspects NULL dense recurrent weights to select another backend and does not dispatch to sparse recurrence implicitly.

## Same-timestep propagation semantics

The core semantic requirement is that deeper layers consume freshly computed upstream state from the same timestep.

For a two-layer network:

```text
x0(t) = ESN_STEP(layer0, external_input(t), x0(t-1))
x1(t) = ESN_STEP(layer1, x0(t),            x1(t-1))
```

Layer 1 therefore consumes `x0(t)`, not `x0(t-1)`.

For `N` layers, execution is strictly ordered from layer 0 through layer `N-1` within each timestep. This is a semantic order, not a scheduling suggestion: later layers cannot be stepped before earlier layers for the same sample.

Each individual layer's recurrent term still uses that layer's own old state as defined by `kfcore_esn_step`; only the inter-layer feed-forward input uses newly computed state from the preceding layer.

There are no external-input skip connections, residual connections, or concatenated `[external_input; previous_layer_state]` inputs in this slice.

## State and feature layout

Caller state is one contiguous float array containing each layer state in declaration order:

```text
[layer 0 state][layer 1 state]...[layer N-1 state]
```

The total state size is the sum of all `reservoir_size` values. This concatenated state is also the deep feature vector; no separate feature copy is introduced.

For layer reservoir sizes 2, 3, and 1:

```text
state[0..1] -> layer 0
state[2..4] -> layer 1
state[5]    -> layer 2
```

This ordering is public and deterministic.

## Layout API

Expose:

```c
kfcore_esn_status kfcore_esn_deep_layout(
    const kfcore_esn_deep_model* deep,
    int* state_size,
    int* workspace_size);
```

On success:

```text
state_size = sum(layer.reservoir_size)
workspace_size = state_size + max(layer.reservoir_size)
```

The layout query validates the complete dense layer chain, including the layer-to-layer input-size rule, and rejects checked integer overflow.

`state_size` and `workspace_size` are required outputs. NULL output pointers are invalid. Neither output may be partially modified on failure.

Readout fields are not required or dereferenced by layout validation.

## Deep step API

Expose:

```c
kfcore_esn_status kfcore_esn_deep_step(
    const kfcore_esn_deep_model* deep,
    const float* input,
    float* state,
    float* workspace);
```

`input` contains `deep->layers[0].input_size` floats.

`state` contains the concatenated old layer states on entry and is replaced by the concatenated new layer states only after the complete chain succeeds.

`workspace` contains at least the count returned by `kfcore_esn_deep_layout`.

`input`, `state`, `workspace`, and all layer weight/bias buffers must obey explicit non-overlap requirements. Workspace is scratch and may be modified after staging begins.

## Execution algorithm

Deep stepping reuses `kfcore_esn_step` rather than duplicating reservoir recurrence math.

1. Validate the complete deep model and checked layout before caller-state mutation or workspace staging.
2. Copy caller `state` into the first `state_size` floats of `workspace`; this is the staged candidate state.
3. Use the remaining `max_reservoir_size` floats as one reusable per-layer scratch buffer.
4. Initialize `layer_input = input`.
5. Iterate layers in declaration order:
   - locate the layer's candidate-state slice;
   - call `kfcore_esn_step(layer, layer_input, candidate_slice, scratch)`;
   - if the delegated step returns non-OK, immediately return the same status without modifying caller state;
   - set `layer_input = candidate_slice`, so the next layer consumes the just-computed state from the same timestep.
6. After all layers succeed, copy the complete candidate concatenated state back into caller `state` once.

The staged state therefore provides transaction-like caller-state semantics while preserving ordered same-timestep layer propagation.

## Failure semantics

Deep execution is failure-atomic with respect to caller state.

The following are rejected with `KFCORE_ESN_INVALID_ARGUMENT` before workspace staging and before caller-state mutation:

- NULL deep model, layer array, input, state, or workspace;
- non-positive `layer_count`;
- invalid reservoir-side metadata in any layer;
- non-positive external input size;
- any layer-chain dimension mismatch;
- invalid or overflowing total state/workspace layout.

Any delegated non-OK status from `kfcore_esn_step` is returned unchanged and caller state remains untouched because only candidate workspace has been modified.

There is no retry, layer skip, rollback heuristic, alternate recurrence, sparse fallback, or partial caller-state commit.

Validation failures do not require workspace mutation because validation completes before staging. Once staging begins, workspace is scratch and may be modified even if a delegated runtime step later fails.

## Readout and training boundary

This slice does not create a deep-specific readout object.

The concatenated deep state is an ordinary dense feature vector. Callers may collect it and use existing ridge or RLS APIs with `reservoir_size` equal to the total deep state size where those APIs' contracts otherwise apply.

No per-layer predictions are produced by `kfcore_esn_deep_step`.

Keeping topology composition independent from readout policy allows later grouped-deep work to reuse the same deep feature representation.

## Grouped-deep boundary

Grouped-deep ESN (gdESN) is explicitly deferred.

The intended future composition is multiple independent deep chains consuming the same external input, with their deep feature vectors concatenated in group order. This future topology should compose the deep and grouped concepts explicitly rather than changing the semantics defined here.

No gdESN API or backend abstraction is added in #38.

## Sparse boundary

The existing sparse runtime remains independent.

Deep ESN v1 accepts only dense `kfcore_esn_model` layers and calls `kfcore_esn_step`. It does not inspect missing dense recurrent weights and switch to `kfcore_esn_step_sparse`.

A later sparse-deep slice may add an explicit backend descriptor, but there is no automatic dense/sparse fallback in this design.

## Files

Planned implementation files:

```text
esn/esn_deep.h
esn/esn_deep.c
esn/tests/test_esn_deep.c
```

The following existing integration points will be extended:

```text
esn/CMakeLists.txt
esn/tests/CMakeLists.txt
.github/workflows/esn-miniblas.yml
```

`esn.h`, `esn.c`, grouped ESN, sparse ESN, reservoir initialization, ridge, and RLS do not require mathematical rewrites.

## Verification contract

The implementation must preserve a clean RED -> GREEN sequence.

1. **Layout contract:** known heterogeneous layer sizes return exact concatenated state and `sum + max` workspace sizes.
2. **Single-layer equivalence:** one-layer deep execution matches ordinary `kfcore_esn_step` for the same model/input/initial state.
3. **Fresh-state propagation:** a two-layer analytical test must distinguish `layer1(input = layer0_state(t))` from an incorrect implementation using `layer0_state(t-1)`. The expected layer-1 state is computed from the freshly generated layer-0 state.
4. **Heterogeneous layer sizes:** layers with different reservoir sizes occupy the documented state offsets in declaration order.
5. **Chain validation:** if `layers[i].input_size != layers[i-1].reservoir_size`, layout and step reject the model before caller-state mutation.
6. **Later-layer invalidity atomicity:** a valid first layer followed by malformed later-layer metadata leaves the complete caller state unchanged.
7. **NULL handling:** required public pointers are rejected; caller state is unchanged whenever state is present.
8. **Checked overflow:** invalid counts and state/workspace size overflow are rejected without dereferencing impossible large matrices.
9. **No readout requirement:** layer output fields may be NULL and do not prevent deep layout/step execution.
10. **Existing regression gate:** the full focused C11 ASan+UBSan ESN suite remains GREEN.

The first layout RED must fail only because `kfcore_esn_deep_layout` is absent. Test-harness/compiler/unrelated-link failures do not count.

After layout GREEN, the step RED must fail only because `kfcore_esn_deep_step` is absent.

## Scope exclusions

The following are explicitly outside #38:

- sparse deep backend dispatch;
- grouped-deep ESN;
- external-input skip connections to deeper layers;
- residual connections;
- concatenated layer inputs;
- bidirectional/recurrent inter-layer connections;
- deep-specific readout/training APIs;
- automatic topology/reservoir construction;
- random initialization policy changes;
- serialization/persistence;
- bias adaptation or backpropagation;
- application-specific behavior/gesture integration;
- parallel threads/SIMD scheduling across dependent layers.

These exclusions keep #38 focused on ordered same-timestep propagation, deterministic state layout, and failure-atomic dense composition.