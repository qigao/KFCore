# Deep ESN Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an allocation-free dense deep ESN runtime that advances an ordered chain of reservoirs within one timestep, feeds each freshly computed upstream state into the next layer, and commits caller state atomically.

**Architecture:** Introduce `esn_deep.h` / `esn_deep.c` as a composition layer over the existing dense `kfcore_esn_step` API. A caller-owned contiguous state stores all layer states in depth order; a caller-owned workspace stages the complete candidate state plus one reusable per-layer scratch vector. Existing single-reservoir, grouped, sparse, ridge, and RLS semantics remain unchanged.

**Tech Stack:** C11, existing KFCore ESN API, miniblas-backed dense runtime, TinyTest, CMake, GitHub Actions ASan+UBSan focused ESN gate.

**Spec:** `docs/superpowers/specs/2026-09-17-esn-deep-runtime-design.md`

## Global Constraints

- Base implementation work on `master` lineage starting from `289651cc3a8f21b2dc400dc46b781da84f936dfa`.
- Layer 0 consumes the external input for timestep `t`.
- For every layer `i > 0`, `layers[i].input_size == layers[i - 1].reservoir_size`.
- Layer `i > 0` consumes the freshly computed `layers[i - 1]` state from the same timestep, never the previous timestep state.
- Layers may use different reservoir sizes, leak rates, weights, and biases.
- Deep state layout is `[layer 0][layer 1]...[layer N-1]` in declaration order and is also the deep feature vector.
- Runtime owns no model/state/workspace memory and performs no hidden allocation or global mutation.
- Workspace size is `sum(reservoir_size) + max(reservoir_size)` floats.
- The complete deep model is validated before workspace staging begins.
- Caller-visible state is failure-atomic: validation failure or any delegated non-OK layer step does not partially advance caller state.
- Dense-only composition. No sparse dispatch, fallback, retry, layer skip, external-input skip connections, residual connections, concatenated inputs, deep-specific readout/training, gdESN, serialization, or application integration.
- Public APIs must document buffer non-overlap requirements explicitly.

---

## File Structure

- Create `esn/esn_deep.h` — public non-owning deep model view and layout/step declarations.
- Create `esn/esn_deep.c` — chain validation, checked layout measurement, ordered same-timestep stepping, and final atomic state commit.
- Create `esn/tests/test_esn_deep.c` — layout, single-layer equivalence, fresh-state propagation, heterogeneous layout, chain validation, NULL, overflow, and failure-atomic tests.
- Modify `esn/CMakeLists.txt` — compile `esn_deep.c` and install `esn_deep.h`.
- Modify `esn/tests/CMakeLists.txt` — include the deep tests in the existing TinyTest target.
- Modify `.github/workflows/esn-miniblas.yml` — compile the new source/test in the focused sanitizer gate.

---

### Task 1: Define the public deep contract and record a clean layout RED

**Files:**
- Create: `esn/esn_deep.h`
- Create: `esn/tests/test_esn_deep.c`
- Modify: `esn/tests/CMakeLists.txt`
- Modify: `.github/workflows/esn-miniblas.yml`

**Interfaces:**
- Consumes: `kfcore_esn_model` and `kfcore_esn_status` from `esn/esn.h`.
- Produces:
  - `typedef struct kfcore_esn_deep_model { int layer_count; const kfcore_esn_model* layers; } kfcore_esn_deep_model;`
  - `kfcore_esn_status kfcore_esn_deep_layout(const kfcore_esn_deep_model*, int* state_size, int* workspace_size);`
  - `kfcore_esn_status kfcore_esn_deep_step(const kfcore_esn_deep_model*, const float* input, float* state, float* workspace);`

- [ ] **Step 1: Add `esn/esn_deep.h`**

Use this public shape:

```c
#ifndef KFCORE_ESN_DEEP_H
#define KFCORE_ESN_DEEP_H

#include "esn.h"

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif

#endif
```

Document that layer 0 consumes the external input, later layers consume the newly computed preceding-layer state from the same timestep, concatenated state follows layer order, readout fields are not required, and `input`, `state`, `workspace`, and all layer weight/bias buffers must obey the explicit non-overlap contract.

- [ ] **Step 2: Add layout-only RED tests**

Create `esn/tests/test_esn_deep.c` with `<limits.h>`, `esn.h`, `esn_deep.h`, then `#define TINYTEST_NO_MAIN` before `tinytest.h`.

Cover a valid heterogeneous chain with reservoir sizes `2, 3, 1` and input sizes `2, 2, 3`:

```c
int state_size = -1;
int workspace_size = -1;
check_equal(kfcore_esn_deep_layout(&deep, &state_size, &workspace_size), KFCORE_ESN_OK);
check_equal(state_size, 6);
check_equal(workspace_size, 9);
```

Cover a dimension mismatch where layer 0 has `reservoir_size = 2` but layer 1 has `input_size = 3`:

```c
check_equal(kfcore_esn_deep_layout(&mismatched, &state_size, &workspace_size),
            KFCORE_ESN_INVALID_ARGUMENT);
```

Cover required NULL outputs, zero layer count, NULL layer array, and checked overflow using two layers with `reservoir_size = INT_MAX` and non-NULL one-element placeholder buffers. The overflow test must fail from metadata arithmetic and must not attempt to dereference matrices sized by `INT_MAX`.

- [ ] **Step 3: Register only the deep test source**

Append `test_esn_deep.c` to `esn/tests/CMakeLists.txt` and to the focused compiler command in `.github/workflows/esn-miniblas.yml`. Do **not** add `esn_deep.c` yet.

- [ ] **Step 4: Open/update a Draft PR and verify clean layout RED**

Expected exact-head gate result: C11 compilation succeeds and link fails only with undefined reference to `kfcore_esn_deep_layout`. A TinyTest duplicate-main error, compiler error, or unrelated linker error is not an accepted RED.

- [ ] **Step 5: Record the RED checkpoint**

Record exact RED SHA and Actions run/job ID in #38 / the Draft PR before adding production symbols.

---

### Task 2: Implement checked deep layout validation

**Files:**
- Create: `esn/esn_deep.c`
- Modify: `esn/CMakeLists.txt`
- Modify: `.github/workflows/esn-miniblas.yml`

**Interfaces:**
- Consumes: `kfcore_esn_deep_model` and reservoir-side fields of `kfcore_esn_model`.
- Produces: working `kfcore_esn_deep_layout` and an internal validation/measurement helper reused by Task 4.

- [ ] **Step 1: Add the internal measurement helper**

Start `esn/esn_deep.c` with:

```c
#include "esn_deep.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <string.h>
```

Define:

```c
static kfcore_esn_status kfcore_esn_deep_measure(
    const kfcore_esn_deep_model* deep,
    int* external_input_size,
    int* total_state_size,
    int* max_reservoir_size);
```

Reject NULL helper outputs, NULL deep model, `layer_count <= 0`, or NULL layer array. For every layer require:

```c
layer->input_size > 0
layer->reservoir_size > 0
isfinite(layer->leak_rate)
layer->leak_rate > 0.0f && layer->leak_rate <= 1.0f
layer->input_weights != NULL
layer->reservoir_weights != NULL
layer->reservoir_bias != NULL
```

For each `i > 0`, additionally require:

```c
deep->layers[i].input_size == deep->layers[i - 1].reservoir_size
```

Accumulate reservoir sizes in `size_t`; reject any sum above `INT_MAX`; track the maximum reservoir size. Write helper outputs only after the complete chain validates.

- [ ] **Step 2: Implement `kfcore_esn_deep_layout` without partial outputs**

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

Do not write either output on failure.

- [ ] **Step 3: Wire production source/header into build/install**

Add `esn_deep.c` / `esn_deep.h` to `kfcore_esn` in `esn/CMakeLists.txt`, install `esn_deep.h` beside the existing ESN public headers, and add `esn/esn_deep.c` to the focused workflow compile command.

- [ ] **Step 4: Run the exact-head focused sanitizer gate**

Expected: all layout tests and all existing ESN tests pass under ASan+UBSan. `kfcore_esn_deep_step` is declared but is not yet referenced by tests.

- [ ] **Step 5: Record layout GREEN**

Record exact SHA and run evidence before adding deep-step tests.

---

### Task 3: Record the deep-step RED, including fresh same-timestep propagation

**Files:**
- Modify: `esn/tests/test_esn_deep.c`

**Interfaces:**
- Consumes: GREEN `kfcore_esn_deep_layout`.
- Produces: RED behavioral contract for `kfcore_esn_deep_step`.

- [ ] **Step 1: Add one-layer equivalence**

Use identical model/input/initial state for ordinary and deep execution. Compare every state element within `1.0e-5f`:

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

- [ ] **Step 2: Add an analytical fresh-state propagation test**

Use two 1-state layers, both with zero recurrent weight, zero bias, leak `1.0f`, and scalar input weight `1.0f`. Initialize caller state to `{-0.5f, 0.0f}` and external input to `{1.0f}`.

Correct same-timestep expectations are:

```c
const float expected_layer0 = tanhf(1.0f);
const float expected_layer1 = tanhf(expected_layer0);
```

Assert both values within `1.0e-5f`. Also assert layer 1 is not approximately `tanhf(-0.5f)`, which would expose an incorrect implementation feeding the old layer-0 state.

- [ ] **Step 3: Add heterogeneous state-order coverage**

Use layer 0 with reservoir size 2 and layer 1 with reservoir size 1 (`layer1.input_size == 2`), zero recurrent weights, deterministic feed-forward weights, and leak `1.0f`. Assert final caller state is `[layer0_state0, layer0_state1, layer1_state]` in that exact order.

- [ ] **Step 4: Add chain mismatch and later-layer invalidity atomicity tests**

For chain mismatch, preserve caller state bytes, call deep step, expect `KFCORE_ESN_INVALID_ARGUMENT`, and assert state is unchanged.

For later-layer invalidity, make layer 0 valid and layer 1 `reservoir_bias = NULL`; again assert the complete caller state remains byte-for-byte unchanged.

- [ ] **Step 5: Add NULL runtime argument tests**

Verify NULL deep model, input, state, and workspace return `KFCORE_ESN_INVALID_ARGUMENT`; whenever caller state is supplied, verify no mutation.

- [ ] **Step 6: Verify clean deep-step RED**

Expected exact-head gate result: existing layout implementation compiles, and link fails only because `kfcore_esn_deep_step` is undefined.

- [ ] **Step 7: Record the step RED checkpoint**

Record exact SHA and Actions run/job evidence before implementing the symbol.

---

### Task 4: Implement ordered same-timestep deep stepping with atomic caller-state commit

**Files:**
- Modify: `esn/esn_deep.c`

**Interfaces:**
- Consumes: `kfcore_esn_deep_measure`, `kfcore_esn_step`, state/workspace formula from layout.
- Produces: working `kfcore_esn_deep_step`.

- [ ] **Step 1: Validate public pointers and the complete chain before staging**

At function entry reject NULL `input`, `state`, or `workspace`, then call `kfcore_esn_deep_measure`. Reject `total > INT_MAX - max_layer` before any `memcpy` into workspace.

- [ ] **Step 2: Stage the complete old caller state**

```c
float* candidate = workspace;
float* scratch = workspace + total_state_size;
memcpy(candidate, state, sizeof(float) * (size_t)total_state_size);
```

Because the public contract requires non-overlap, this preserves the caller state until final commit.

- [ ] **Step 3: Execute layers strictly in depth order**

Use this data flow:

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

The assignment `layer_input = layer_state` must occur only after that layer's step succeeds, ensuring layer `i+1` consumes the freshly computed same-timestep state.

- [ ] **Step 4: Commit once after every layer succeeds**

```c
memcpy(state, candidate, sizeof(float) * (size_t)total_state_size);
return KFCORE_ESN_OK;
```

There is no retry, rollback loop, sparse fallback, or partial caller-state copy.

- [ ] **Step 5: Run exact-head focused sanitizer verification**

Expected: the complete ESN suite passes under ASan+UBSan, including the analytical fresh-state propagation test.

- [ ] **Step 6: Review the PR diff against #38 scope**

Confirm the changed production surface is limited to the deep module plus CMake/workflow wiring; no changes to single-reservoir math, grouped runtime, sparse runtime, ridge, RLS, or application code.

- [ ] **Step 7: Update PR/issue evidence without moving code head**

Record layout RED, layout GREEN, step RED, final exact-head GREEN, test/assertion counts, and source-cleanliness evidence in PR # / issue #38 metadata.

---

### Task 5: Finish the branch only after exact-head verification

**Files:**
- No production changes expected.

**Interfaces:**
- Consumes: final GREEN PR head.
- Produces: merge-ready PR and verified `master` result.

- [ ] **Step 1: Verify no unresolved review thread and PR is mergeable**

Do not mark Ready if a review thread or mergeability blocker exists.

- [ ] **Step 2: Mark Ready and merge using the exact verified head SHA**

Use GitHub's expected-head guard so a moved head cannot be merged using stale CI evidence.

- [ ] **Step 3: Verify the resulting `master` merge commit**

Wait for the `ESN miniblas contract` push run on the exact merge commit. Require build, ASan+UBSan runtime, tracked-source cleanliness, and artifact upload all to succeed.

- [ ] **Step 4: Close #38 only after the merge-commit push gate is GREEN**

Record the merge commit SHA and final master run ID in both PR and issue metadata, then close #38 as completed.
