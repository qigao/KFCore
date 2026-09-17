# Deep ESN Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an allocation-free dense deep ESN runtime that advances an ordered reservoir chain within one timestep, feeds freshly computed upstream state into the next layer, and commits caller state atomically.

**Architecture:** Add `esn_deep.h` / `esn_deep.c` as a composition layer over `kfcore_esn_step`. Caller-owned state stores all layer states in depth order; caller-owned workspace stores a staged complete state plus one reusable per-layer scratch vector. Existing single-reservoir, grouped, sparse, ridge, and RLS semantics remain unchanged.

**Tech Stack:** C11, KFCore ESN API, miniblas-backed dense runtime, TinyTest, CMake, GitHub Actions ASan+UBSan focused ESN gate.

**Spec:** `docs/superpowers/specs/2026-09-17-esn-deep-runtime-design.md`

## Global Constraints

- Base lineage: `master` at `289651cc3a8f21b2dc400dc46b781da84f936dfa`.
- Layer 0 consumes the external input at timestep `t`.
- For every `i > 0`, `layers[i].input_size == layers[i - 1].reservoir_size`.
- Layer `i > 0` consumes the freshly computed preceding-layer state from the same timestep, never that layer's previous-timestep state.
- Layers may have different reservoir sizes, leak rates, weights, and biases.
- State/feature layout is `[layer 0][layer 1]...[layer N-1]`.
- No hidden allocation/global state.
- Workspace size is `sum(reservoir_size) + max(reservoir_size)` floats.
- Validate the complete chain before staging caller state.
- Caller-visible state is failure-atomic.
- Dense-only: no sparse dispatch/fallback, retry, skip connections, residual inputs, concatenated inputs, deep-specific readout/training, gdESN, serialization, or application integration.
- Public APIs document non-overlap requirements explicitly.

---

## File Structure

- Create `esn/esn_deep.h` — public non-owning deep view and layout/step declarations.
- Create `esn/esn_deep.c` — checked chain measurement, layout, ordered same-timestep step, atomic commit.
- Create `esn/tests/test_esn_deep.c` — layout, overflow, equivalence, same-timestep propagation, ordering, validation, NULL, atomicity.
- Modify `esn/CMakeLists.txt` — build/install deep module.
- Modify `esn/tests/CMakeLists.txt` — add deep test source.
- Modify `.github/workflows/esn-miniblas.yml` — compile deep source/test in sanitizer gate.

---

### Task 1: Public contract + clean layout RED

**Files:** create `esn/esn_deep.h`, `esn/tests/test_esn_deep.c`; modify test CMake and focused workflow.

**Produces:**

```c
typedef struct kfcore_esn_deep_model
{
    int layer_count;
    const kfcore_esn_model* layers;
} kfcore_esn_deep_model;

kfcore_esn_status kfcore_esn_deep_layout(const kfcore_esn_deep_model* deep,
                                         int* state_size, int* workspace_size);

kfcore_esn_status kfcore_esn_deep_step(const kfcore_esn_deep_model* deep,
                                       const float* input, float* state,
                                       float* workspace);
```

- [ ] **Step 1: Add `esn/esn_deep.h`**

Use include guards, include `esn.h`, preserve C++ `extern "C"`, and document:

- layer 0 takes external input;
- later layers take the newly computed preceding-layer state from the same timestep;
- state is concatenated in depth order;
- layout outputs are required;
- readout fields are not required;
- input/state/workspace/weight/bias buffers obey explicit non-overlap rules;
- no sparse fallback or hidden allocation.

- [ ] **Step 2: Add layout-only tests**

Create `test_esn_deep.c` with `<limits.h>`, `esn.h`, `esn_deep.h`, then `#define TINYTEST_NO_MAIN` before `tinytest.h`.

Valid heterogeneous chain: reservoir sizes `2, 3, 1`, input sizes `2, 2, 3`:

```c
int state_size = -1;
int workspace_size = -1;
check_equal(kfcore_esn_deep_layout(&deep, &state_size, &workspace_size), KFCORE_ESN_OK);
check_equal(state_size, 6);
check_equal(workspace_size, 9);
```

Dimension mismatch: layer 0 `reservoir_size = 2`, layer 1 `input_size = 3` must return `KFCORE_ESN_INVALID_ARGUMENT`.

Required NULL outputs, zero layer count, and NULL layer array must return `KFCORE_ESN_INVALID_ARGUMENT`.

Cover both arithmetic boundaries without dereferencing impossible matrices:

1. **State-sum overflow:** layer 0 has `input_size = 1`, `reservoir_size = INT_MAX`; layer 1 has `input_size = INT_MAX`, `reservoir_size = 1`. All required weight/bias pointers use non-NULL one-element placeholders. The chain dimensions are valid, so rejection must occur because the state-size sum exceeds `INT_MAX`.
2. **Workspace overflow:** a single layer has `input_size = 1`, `reservoir_size = INT_MAX` with non-NULL placeholder buffers. State size itself is representable as `INT_MAX`, but `state_size + max_reservoir_size` is not, so layout must reject the workspace size.

- [ ] **Step 3: Register only the test source**

Append `test_esn_deep.c` to `esn/tests/CMakeLists.txt` and the focused compiler command. Do not add `esn_deep.c` yet.

- [ ] **Step 4: Open Draft PR and verify clean RED**

Expected: C11 compilation succeeds and link fails only on undefined `kfcore_esn_deep_layout`. Harness/compiler/unrelated linker failures do not count.

- [ ] **Step 5: Record exact RED SHA/run/job in #38 and PR metadata**

---

### Task 2: Checked layout GREEN

**Files:** create `esn/esn_deep.c`; modify `esn/CMakeLists.txt` and focused workflow.

- [ ] **Step 1: Add internal measurement helper**

```c
#include "esn_deep.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

static kfcore_esn_status kfcore_esn_deep_measure(
    const kfcore_esn_deep_model* deep,
    int* external_input_size,
    int* total_state_size,
    int* max_reservoir_size);
```

Reject NULL deep/helper outputs, `layer_count <= 0`, or NULL layer array. For every layer require positive `input_size`/`reservoir_size`, finite leak in `(0, 1]`, and non-NULL input weights/recurrent weights/bias. For each `i > 0`, require `layers[i].input_size == layers[i - 1].reservoir_size`.

Accumulate reservoir sizes in `size_t`; before addition require `reservoir_size <= INT_MAX - total`. Track maximum reservoir size. Write helper outputs only after the complete chain validates.

- [ ] **Step 2: Implement layout without partial outputs**

```c
kfcore_esn_status kfcore_esn_deep_layout(const kfcore_esn_deep_model* deep,
                                         int* state_size, int* workspace_size)
{
    if (!state_size || !workspace_size)
        return KFCORE_ESN_INVALID_ARGUMENT;

    int input_size = 0;
    int total = 0;
    int max_layer = 0;
    const kfcore_esn_status status =
        kfcore_esn_deep_measure(deep, &input_size, &total, &max_layer);
    if (status != KFCORE_ESN_OK)
        return status;

    if (total > INT_MAX - max_layer)
        return KFCORE_ESN_INVALID_ARGUMENT;

    *state_size = total;
    *workspace_size = total + max_layer;
    return KFCORE_ESN_OK;
}
```

- [ ] **Step 3: Wire production source/header**

Add `esn_deep.c` / `esn_deep.h` to `kfcore_esn`, install `esn_deep.h`, and add `esn/esn_deep.c` to the focused compile command.

- [ ] **Step 4: Run exact-head sanitizer gate**

Expected: all layout and existing ESN tests pass; step is still unreferenced.

- [ ] **Step 5: Record exact layout-GREEN SHA/run**

---

### Task 3: Deep-step RED

**Files:** modify `esn/tests/test_esn_deep.c`.

- [ ] **Step 1: Add single-layer equivalence**

```c
float ordinary_state[2] = {0.25f, -0.5f};
float deep_state[2] = {0.25f, -0.5f};
float ordinary_workspace[2] = {0};
float deep_workspace[4] = {0};

check_equal(kfcore_esn_step(&layer, input, ordinary_state, ordinary_workspace), KFCORE_ESN_OK);
check_equal(kfcore_esn_deep_step(&deep, input, deep_state, deep_workspace), KFCORE_ESN_OK);
check_within(deep_state[0], ordinary_state[0], 1.0e-5f);
check_within(deep_state[1], ordinary_state[1], 1.0e-5f);
```

- [ ] **Step 2: Add analytical same-timestep propagation test**

Two one-state layers: input weight `1`, recurrent weight `0`, bias `0`, leak `1`. Initial state `{-0.5f, 0.0f}`, external input `{1.0f}`.

```c
const float expected_layer0 = tanhf(1.0f);
const float expected_layer1 = tanhf(expected_layer0);
```

Assert both expected values within `1e-5f`; also verify layer 1 is materially different from `tanhf(-0.5f)`, which is the value an old-state feed bug would produce.

- [ ] **Step 3: Add heterogeneous state-order test**

Layer 0 reservoir size 2, layer 1 reservoir size 1 with `layer1.input_size = 2`; zero recurrence and deterministic feed-forward weights. Assert final state order exactly `[layer0[0], layer0[1], layer1[0]]`.

- [ ] **Step 4: Add chain-mismatch and later-layer-invalid atomicity**

For each case preserve `before` with `memcpy`, expect `KFCORE_ESN_INVALID_ARGUMENT`, and require `memcmp(state, before, sizeof(state)) == 0`.

- [ ] **Step 5: Add NULL public argument tests**

NULL deep/input/state/workspace each returns `KFCORE_ESN_INVALID_ARGUMENT`; supplied caller state remains unchanged.

- [ ] **Step 6: Verify clean step RED**

Expected: linker fails only on undefined `kfcore_esn_deep_step`.

- [ ] **Step 7: Record exact step-RED SHA/run/job**

---

### Task 4: Ordered deep-step GREEN

**Files:** modify `esn/esn_deep.c`.

- [ ] **Step 1: Validate before staging**

Reject NULL input/state/workspace, call `kfcore_esn_deep_measure`, and reject `total > INT_MAX - max_layer` before any workspace write.

- [ ] **Step 2: Stage complete caller state**

```c
float* candidate = workspace;
float* scratch = workspace + total_state_size;
memcpy(candidate, state, sizeof(float) * (size_t)total_state_size);
```

- [ ] **Step 3: Execute depth order with fresh-state feed-forward**

```c
const float* layer_input = input;
int offset = 0;
for (int i = 0; i < deep->layer_count; ++i)
{
    const kfcore_esn_model* layer = &deep->layers[i];
    float* layer_state = candidate + offset;
    const kfcore_esn_status status =
        kfcore_esn_step(layer, layer_input, layer_state, scratch);
    if (status != KFCORE_ESN_OK)
        return status;

    layer_input = layer_state;
    offset += layer->reservoir_size;
}
```

- [ ] **Step 4: Commit once**

```c
memcpy(state, candidate, sizeof(float) * (size_t)total_state_size);
return KFCORE_ESN_OK;
```

No retry, rollback loop, sparse fallback, or partial caller-state copy.

- [ ] **Step 5: Run exact-head focused sanitizer gate**

Require build and runtime success, including the fresh-state propagation regression.

- [ ] **Step 6: Review full diff against #38 scope**

No changes to single-reservoir math, grouped runtime, sparse runtime, ridge, RLS, or application code except integration lists/workflow.

- [ ] **Step 7: Update PR/issue evidence without moving code head**

Record layout RED, layout GREEN, step RED, final exact-head GREEN, test/assertion counts, source cleanliness.

---

### Task 5: Finish only after exact-head verification

- [ ] Verify PR mergeable and no unresolved review thread.
- [ ] Mark Ready and merge with expected-head SHA guard.
- [ ] Verify the exact `master` merge commit's push run through build, ASan+UBSan runtime, tracked-source cleanliness, and artifact upload.
- [ ] Record merge SHA/master run in PR and #38, then close #38 as completed.
