# Grouped ESN Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an allocation-free grouped ESN runtime that advances multiple independent dense reservoirs from one shared input and exposes their concatenated state as the grouped feature vector with failure-atomic caller state.

**Architecture:** Introduce a dedicated `esn_group` composition layer over the existing dense `kfcore_esn_step` API. Group metadata remains non-owning; layout is deterministic; grouped execution stages a full candidate concatenated state in caller workspace and commits it only after every group succeeds.

**Tech Stack:** C11, existing KFCore ESN API, existing miniblas-backed dense runtime, TinyTest, CMake, GitHub Actions ASan+UBSan focused ESN gate.

**Spec:** `docs/superpowers/specs/2026-09-17-esn-grouped-runtime-design.md`

## Global Constraints

- Base implementation work on `master` lineage starting from `61ebd0793ce0cc55018c6d4af7fb07131e898d90`.
- Every group consumes the same input and must expose the same positive `input_size`.
- Groups may have different `reservoir_size`, leak rates, weights, and biases.
- Group state layout is `[group 0][group 1]...[group N-1]` in declaration order.
- Grouped runtime owns no model/state/workspace memory and performs no hidden allocation or global mutation.
- Workspace size is `sum(reservoir_size) + max(reservoir_size)` floats.
- Caller-visible state must be failure-atomic: validation failure or any delegated non-OK group step must not partially advance caller state.
- Dense grouped execution only. No sparse backend dispatch or automatic fallback.
- No grouped-specific readout, training, deep ESN, grouped-deep ESN, serialization, topology construction, bias adaptation, application integration, or parallel scheduling in this slice.
- Public APIs and caller buffers must document non-overlap requirements explicitly.

---

## File Structure

- Create `esn/esn_group.h` — public non-owning grouped model view and layout/step declarations.
- Create `esn/esn_group.c` — grouped validation, checked layout calculation, staged grouped step, and final commit.
- Create `esn/tests/test_esn_group.c` — grouped layout, equivalence, ordering, heterogeneous-size, and failure-atomic tests.
- Modify `esn/CMakeLists.txt` — build `esn_group.c`, expose/install `esn_group.h`.
- Modify `esn/tests/CMakeLists.txt` — add grouped tests to the existing ESN TinyTest target.
- Modify `.github/workflows/esn-miniblas.yml` — compile `esn_group.c` and `test_esn_group.c` in the focused C11 sanitizer contract.

---

### Task 1: Define the grouped public contract and record a clean layout RED

**Files:**
- Create: `esn/esn_group.h`
- Create: `esn/tests/test_esn_group.c`
- Modify: `esn/tests/CMakeLists.txt`
- Modify: `.github/workflows/esn-miniblas.yml`

**Interfaces:**
- Consumes: existing `kfcore_esn_model` and `kfcore_esn_status` from `esn/esn.h`.
- Produces:
  - `typedef struct kfcore_esn_grouped_model { int group_count; const kfcore_esn_model* groups; } kfcore_esn_grouped_model;`
  - `kfcore_esn_status kfcore_esn_grouped_layout(const kfcore_esn_grouped_model*, int* state_size, int* workspace_size);`
  - `kfcore_esn_status kfcore_esn_grouped_step(const kfcore_esn_grouped_model*, const float* input, float* state, float* workspace);`

- [ ] **Step 1: Add the public grouped header**

Create `esn/esn_group.h` with this contract:

```c
#ifndef KFCORE_ESN_GROUP_H
#define KFCORE_ESN_GROUP_H

#include "esn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kfcore_esn_grouped_model
{
    int group_count;
    const kfcore_esn_model* groups;
} kfcore_esn_grouped_model;

kfcore_esn_status kfcore_esn_grouped_layout(
    const kfcore_esn_grouped_model* grouped,
    int* state_size,
    int* workspace_size);

kfcore_esn_status kfcore_esn_grouped_step(
    const kfcore_esn_grouped_model* grouped,
    const float* input,
    float* state,
    float* workspace);

#ifdef __cplusplus
}
#endif

#endif
```

Document in the header that all groups must share `input_size`, grouped state is concatenated in declaration order, layout outputs are required, and grouped step requires `input`, `state`, and `workspace` to be mutually non-overlapping and not overlap group weight/bias storage.

- [ ] **Step 2: Add layout-only RED tests**

Create `esn/tests/test_esn_group.c` with `#define TINYTEST_NO_MAIN` before including TinyTest. Start with tests that only call `kfcore_esn_grouped_layout`:

```c
#include "esn.h"
#include "esn_group.h"
#define TINYTEST_NO_MAIN
#include "tinytest.h"

spec("kfcore esn grouped")
{
    it("reports concatenated state and staged workspace sizes")
    {
        const kfcore_esn_model groups[3] = {
            { .input_size = 2, .reservoir_size = 2, .leak_rate = 1.0f,
              .input_weights = (const float[]){1,0,0,1},
              .reservoir_weights = (const float[]){0,0,0,0},
              .reservoir_bias = (const float[]){0,0} },
            { .input_size = 2, .reservoir_size = 3, .leak_rate = 1.0f,
              .input_weights = (const float[]){1,0,0,1,1,1},
              .reservoir_weights = (const float[]){0,0,0,0,0,0,0,0,0},
              .reservoir_bias = (const float[]){0,0,0} },
            { .input_size = 2, .reservoir_size = 1, .leak_rate = 1.0f,
              .input_weights = (const float[]){1,1},
              .reservoir_weights = (const float[]){0},
              .reservoir_bias = (const float[]){0} }
        };
        const kfcore_esn_grouped_model grouped = { 3, groups };
        int state_size = -1;
        int workspace_size = -1;

        check_equal(kfcore_esn_grouped_layout(&grouped, &state_size, &workspace_size),
                    KFCORE_ESN_OK);
        check_equal(state_size, 6);
        check_equal(workspace_size, 9);
    }

    it("rejects mismatched group input sizes")
    {
        const float w1[1] = {0};
        const float win1[1] = {1};
        const float b1[1] = {0};
        const float win2[2] = {1, 1};
        const kfcore_esn_model groups[2] = {
            {1, 1, 0, 1.0f, win1, w1, b1, NULL, NULL},
            {2, 1, 0, 1.0f, win2, w1, b1, NULL, NULL}
        };
        const kfcore_esn_grouped_model grouped = {2, groups};
        int state_size = -1;
        int workspace_size = -1;

        check_equal(kfcore_esn_grouped_layout(&grouped, &state_size, &workspace_size),
                    KFCORE_ESN_INVALID_ARGUMENT);
    }

    it("rejects required null layout outputs")
    {
        const float w[1] = {0};
        const float win[1] = {1};
        const float b[1] = {0};
        const kfcore_esn_model group = {1, 1, 0, 1.0f, win, w, b, NULL, NULL};
        const kfcore_esn_grouped_model grouped = {1, &group};
        int size = 0;

        check_equal(kfcore_esn_grouped_layout(&grouped, NULL, &size),
                    KFCORE_ESN_INVALID_ARGUMENT);
        check_equal(kfcore_esn_grouped_layout(&grouped, &size, NULL),
                    KFCORE_ESN_INVALID_ARGUMENT);
    }
}
```

- [ ] **Step 3: Register grouped tests in CMake and focused CI without production source**

Append `test_esn_group.c` to `esn/tests/CMakeLists.txt` and append it to the focused compiler command in `.github/workflows/esn-miniblas.yml`. Do not add `esn_group.c` yet.

- [ ] **Step 4: Push and verify a clean RED**

Run the existing GitHub Actions `ESN miniblas contract` on the exact head.

Expected: C11 compilation succeeds and link fails only with undefined reference to `kfcore_esn_grouped_layout`. A TinyTest duplicate-main error, compiler error, or unrelated linker error does not count as the RED.

- [ ] **Step 5: Commit/checkpoint the clean RED**

Record the exact RED SHA and Actions run ID in PR/issue notes before adding production symbols.

---

### Task 2: Implement checked grouped layout validation

**Files:**
- Create: `esn/esn_group.c`
- Modify: `esn/CMakeLists.txt`
- Modify: `.github/workflows/esn-miniblas.yml`

**Interfaces:**
- Consumes: `kfcore_esn_grouped_model`, existing dense reservoir field semantics from `kfcore_esn_model`.
- Produces: working `kfcore_esn_grouped_layout` and an internal reusable grouped validation/layout helper for Task 3.

- [ ] **Step 1: Add checked grouped validation helper**

In `esn/esn_group.c`, validate all reservoir-side fields without requiring readout fields:

```c
static kfcore_esn_status kfcore_esn_grouped_measure(
    const kfcore_esn_grouped_model* grouped,
    int* common_input_size,
    int* total_state_size,
    int* max_reservoir_size)
```

Requirements:

```c
if (!grouped || grouped->group_count <= 0 || !grouped->groups ||
    !common_input_size || !total_state_size || !max_reservoir_size)
    return KFCORE_ESN_INVALID_ARGUMENT;
```

For every group require:

```c
group->input_size > 0
group->reservoir_size > 0
isfinite(group->leak_rate)
group->leak_rate > 0.0f && group->leak_rate <= 1.0f
group->input_weights != NULL
group->reservoir_weights != NULL
group->reservoir_bias != NULL
group->input_size == first_group_input_size
```

Accumulate reservoir sizes in `size_t`, reject any sum above `INT_MAX`, and return the largest group reservoir size.

- [ ] **Step 2: Implement `kfcore_esn_grouped_layout`**

```c
kfcore_esn_status kfcore_esn_grouped_layout(
    const kfcore_esn_grouped_model* grouped,
    int* state_size,
    int* workspace_size)
{
    if (!state_size || !workspace_size)
        return KFCORE_ESN_INVALID_ARGUMENT;

    int input_size = 0;
    int total = 0;
    int max_group = 0;
    kfcore_esn_status status =
        kfcore_esn_grouped_measure(grouped, &input_size, &total, &max_group);
    if (status != KFCORE_ESN_OK)
        return status;

    if (total > INT_MAX - max_group)
        return KFCORE_ESN_INVALID_ARGUMENT;

    *state_size = total;
    *workspace_size = total + max_group;
    return KFCORE_ESN_OK;
}
```

Do not partially write either output on failure; assign both only after all validation and overflow checks succeed.

- [ ] **Step 3: Wire the production file into build/install inputs**

Add `esn_group.c` and `esn_group.h` to `kfcore_esn` in `esn/CMakeLists.txt`. Install `esn_group.h` beside `esn.h` and `esn_sparse.h`. Add `esn/esn_group.c` to the focused GitHub Actions compile command.

- [ ] **Step 4: Run the focused gate**

Expected: grouped layout tests pass and all pre-existing ESN tests remain GREEN under ASan+UBSan. At this checkpoint `kfcore_esn_grouped_step` may still be declared but must not be referenced by tests yet.

- [ ] **Step 5: Commit**

Commit message:

```text
feat: add grouped ESN layout contract
```

---

### Task 3: Record grouped-step RED and failure-atomic behavior requirements

**Files:**
- Modify: `esn/tests/test_esn_group.c`

**Interfaces:**
- Consumes: working `kfcore_esn_grouped_layout` and public `kfcore_esn_grouped_step` declaration.
- Produces: RED runtime expectations for state equivalence, concatenation order, heterogeneous sizes, and validation atomicity.

- [ ] **Step 1: Add one-group equivalence test**

Use one 2-state model, clone the same initial state into `dense_state` and `grouped_state`, call ordinary `kfcore_esn_step` on the first and grouped step on the second, then compare both state elements within `1e-5f`.

```c
float dense_state[2] = {0.25f, -0.5f};
float grouped_state[2] = {0.25f, -0.5f};
float dense_workspace[2] = {0};
float grouped_workspace[4] = {0};

check_equal(kfcore_esn_step(&model, input, dense_state, dense_workspace), KFCORE_ESN_OK);
check_equal(kfcore_esn_grouped_step(&grouped, input, grouped_state, grouped_workspace),
            KFCORE_ESN_OK);
check_close(grouped_state[0], dense_state[0], 1.0e-5f);
check_close(grouped_state[1], dense_state[1], 1.0e-5f);
```

- [ ] **Step 2: Add same-input concatenation and heterogeneous-size test**

Create two zero-recurrence groups with sizes 1 and 2 and leak 1.0 so expected states are direct `tanh` results from the same scalar input. Assert output state layout is `[group0_state, group1_state0, group1_state1]` in that exact order.

- [ ] **Step 3: Add later-group invalidity atomicity test**

Create a valid first group and a second group with `reservoir_bias = NULL`. Initialize caller state with sentinel values, copy them to `before`, call grouped step, expect `KFCORE_ESN_INVALID_ARGUMENT`, then assert `memcmp(state, before, sizeof(state)) == 0`.

- [ ] **Step 4: Add null runtime-argument tests**

Verify NULL `grouped`, `input`, `state`, and `workspace` each return `KFCORE_ESN_INVALID_ARGUMENT` without mutation when a caller state exists.

- [ ] **Step 5: Push and verify grouped-step RED**

Expected: build reaches the linker and fails only because `kfcore_esn_grouped_step` is undefined. Existing layout tests must still compile.

- [ ] **Step 6: Commit/checkpoint the step RED**

Record exact head and workflow evidence before implementing grouped stepping.

---

### Task 4: Implement failure-atomic grouped stepping

**Files:**
- Modify: `esn/esn_group.c`

**Interfaces:**
- Consumes: `kfcore_esn_grouped_measure`, `kfcore_esn_step`, workspace formula from grouped layout.
- Produces: `kfcore_esn_grouped_step` with atomic caller-state commit.

- [ ] **Step 1: Validate complete grouped runtime before mutation**

At function entry reject NULL input/state/workspace, then call the grouped measurement helper. Compute `workspace_size = total + max_group` with the same checked arithmetic used by layout.

- [ ] **Step 2: Stage the complete candidate state**

Use:

```c
float* candidate = workspace;
float* scratch = workspace + total_state_size;
memcpy(candidate, state, sizeof(float) * (size_t)total_state_size);
```

The public header must state that workspace contains at least the size returned by `kfcore_esn_grouped_layout` and does not overlap input/state/model buffers.

- [ ] **Step 3: Advance each candidate group in declaration order**

Maintain `int offset = 0;`. For every group:

```c
kfcore_esn_status status =
    kfcore_esn_step(&grouped->groups[i], input, candidate + offset, scratch);
if (status != KFCORE_ESN_OK)
    return status;
offset += grouped->groups[i].reservoir_size;
```

Do not write caller state inside the loop.

- [ ] **Step 4: Commit caller state once**

After every group succeeds:

```c
memcpy(state, candidate, sizeof(float) * (size_t)total_state_size);
return KFCORE_ESN_OK;
```

No retry, skip, alternate backend, or partial commit.

- [ ] **Step 5: Run the focused sanitizer gate**

Expected: all grouped tests and all existing ESN tests pass with ASan+UBSan; source cleanliness and artifact upload succeed.

- [ ] **Step 6: Commit**

Commit message:

```text
feat: add failure-atomic grouped ESN step
```

---

### Task 5: Final public-contract and integration review

**Files:**
- Review: `esn/esn_group.h`
- Review: `esn/esn_group.c`
- Review: `esn/tests/test_esn_group.c`
- Review: `esn/CMakeLists.txt`
- Review: `.github/workflows/esn-miniblas.yml`

**Interfaces:**
- Consumes: complete grouped runtime.
- Produces: merge-ready #36 evidence without changing grouped math unless a real review regression is found.

- [ ] **Step 1: Verify public contract against spec**

Confirm the header explicitly states common input size, declaration-order concatenation, required layout outputs, caller-owned buffers, workspace formula, dense-only execution, and non-overlap requirements.

- [ ] **Step 2: Review failure atomicity**

Confirm no write to caller `state` occurs before every delegated group step has returned `KFCORE_ESN_OK`.

- [ ] **Step 3: Review scope containment**

Confirm the diff does not introduce sparse backend dispatch, grouped readout/training, deep/gdESN, serialization, topology construction, or changes to existing single-reservoir math.

- [ ] **Step 4: Run final exact-head GitHub Actions gate**

Require a fresh `ESN miniblas contract` result on the final exact PR head after all documentation/build changes. Record run ID, head SHA, test count, assertion count, ASan+UBSan result, source-cleanliness result, and artifact upload result.

- [ ] **Step 5: Update PR/issue checkpoint**

Document the clean layout RED, grouped-step RED, final exact-head GREEN, and any review-discovered regressions with their exact SHAs/run IDs.

- [ ] **Step 6: Mark Ready only after exact-head GREEN**

Do not merge or claim completion from an older head. After Ready/merge, verify the resulting `master` push gate before closing #36 as completed.
