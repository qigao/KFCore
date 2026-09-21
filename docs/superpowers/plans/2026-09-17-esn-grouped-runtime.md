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
- Create `esn/tests/test_esn_group.c` — grouped layout, equivalence, ordering, heterogeneous-size, overflow, and failure-atomic tests.
- Modify `esn/CMakeLists.txt` — build `esn_group.c`, expose/install `esn_group.h`.
- Modify `esn/tests/CMakeLists.txt` — add grouped tests to the existing ESN TinyTest target.
- Modify `.github/workflows/esn-miniblas.yml` — compile `esn_group.c` and `test_esn_group.c` in the focused C11 sanitizer contract.

---

### Task 1: Define the public contract and record a clean layout RED

**Files:**
- Create: `esn/esn_group.h`
- Create: `esn/tests/test_esn_group.c`
- Modify: `esn/tests/CMakeLists.txt`
- Modify: `.github/workflows/esn-miniblas.yml`

**Interfaces:**
- Consumes: `kfcore_esn_model` and `kfcore_esn_status` from `esn/esn.h`.
- Produces:
  - `typedef struct kfcore_esn_grouped_model { int group_count; const kfcore_esn_model* groups; } kfcore_esn_grouped_model;`
  - `kfcore_esn_status kfcore_esn_grouped_layout(const kfcore_esn_grouped_model*, int* state_size, int* workspace_size);`
  - `kfcore_esn_status kfcore_esn_grouped_step(const kfcore_esn_grouped_model*, const float* input, float* state, float* workspace);`

- [ ] **Step 1: Add `esn/esn_group.h`**

Use this public shape:

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

Document that all groups share one positive `input_size`; concatenated state follows declaration order; `state_size` and `workspace_size` are required outputs; grouped step workspace contains at least the layout-reported count; and `input`, `state`, `workspace`, and group weight/bias buffers must obey the documented non-overlap contract.

- [ ] **Step 2: Add layout-only RED tests**

Create `esn/tests/test_esn_group.c` with `#define TINYTEST_NO_MAIN` before TinyTest. Include `<limits.h>` for overflow coverage.

Cover exact layout for reservoir sizes `2, 3, 1`:

```c
int state_size = -1;
int workspace_size = -1;
check_equal(kfcore_esn_grouped_layout(&grouped, &state_size, &workspace_size),
            KFCORE_ESN_OK);
check_equal(state_size, 6);
check_equal(workspace_size, 9);
```

Cover mismatched input sizes:

```c
check_equal(kfcore_esn_grouped_layout(&mismatched, &state_size, &workspace_size),
            KFCORE_ESN_INVALID_ARGUMENT);
```

Cover required NULL outputs:

```c
check_equal(kfcore_esn_grouped_layout(&one_group, NULL, &workspace_size),
            KFCORE_ESN_INVALID_ARGUMENT);
check_equal(kfcore_esn_grouped_layout(&one_group, &state_size, NULL),
            KFCORE_ESN_INVALID_ARGUMENT);
```

Cover invalid group count / group array:

```c
const kfcore_esn_grouped_model zero_groups = {0, &valid_group};
const kfcore_esn_grouped_model null_groups = {1, NULL};
check_equal(kfcore_esn_grouped_layout(&zero_groups, &state_size, &workspace_size),
            KFCORE_ESN_INVALID_ARGUMENT);
check_equal(kfcore_esn_grouped_layout(&null_groups, &state_size, &workspace_size),
            KFCORE_ESN_INVALID_ARGUMENT);
```

Cover checked-size overflow without dereferencing large matrices. Use non-NULL one-element buffers for required pointers and two groups with `reservoir_size = INT_MAX`; layout validation must reject the sum before any runtime stepping:

```c
const float scalar = 0.0f;
const kfcore_esn_model huge_groups[2] = {
    { .input_size = 1, .reservoir_size = INT_MAX, .leak_rate = 1.0f,
      .input_weights = &scalar, .reservoir_weights = &scalar, .reservoir_bias = &scalar },
    { .input_size = 1, .reservoir_size = INT_MAX, .leak_rate = 1.0f,
      .input_weights = &scalar, .reservoir_weights = &scalar, .reservoir_bias = &scalar }
};
const kfcore_esn_grouped_model huge = {2, huge_groups};
check_equal(kfcore_esn_grouped_layout(&huge, &state_size, &workspace_size),
            KFCORE_ESN_INVALID_ARGUMENT);
```

- [ ] **Step 3: Register the test source without production grouped code**

Append `test_esn_group.c` to `esn/tests/CMakeLists.txt` and the focused compile command in `.github/workflows/esn-miniblas.yml`. Do not add `esn_group.c` yet.

- [ ] **Step 4: Verify a clean layout RED**

Push the exact head and inspect `ESN miniblas contract`.

Expected: C11 compilation succeeds and the first real failure is undefined reference to `kfcore_esn_grouped_layout`. TinyTest duplicate-main, source compile errors, or unrelated link failures do not count.

- [ ] **Step 5: Record checkpoint**

Record exact RED SHA and Actions run/job evidence in #36 / the Draft PR before production symbols are added.

---

### Task 2: Implement checked grouped layout validation

**Files:**
- Create: `esn/esn_group.c`
- Modify: `esn/CMakeLists.txt`
- Modify: `.github/workflows/esn-miniblas.yml`

**Interfaces:**
- Consumes: `kfcore_esn_grouped_model` and existing dense reservoir-side `kfcore_esn_model` fields.
- Produces: working `kfcore_esn_grouped_layout` and an internal validation/measurement helper reused by grouped stepping.

- [ ] **Step 1: Add headers and internal measurement helper**

Start `esn/esn_group.c` with:

```c
#include "esn_group.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <string.h>
```

Define:

```c
static kfcore_esn_status kfcore_esn_grouped_measure(
    const kfcore_esn_grouped_model* grouped,
    int* common_input_size,
    int* total_state_size,
    int* max_reservoir_size);
```

Reject NULL helper outputs, NULL grouped view, `group_count <= 0`, or NULL group array. For every group require:

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

Accumulate reservoir sizes in `size_t`; reject the sum if it exceeds `INT_MAX`; track the maximum group reservoir size. Write helper outputs only after the complete model validates.

- [ ] **Step 2: Implement layout with no partial outputs**

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
    const kfcore_esn_status status =
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

Do not modify either caller output on failure.

- [ ] **Step 3: Wire production source/header into build and install**

In `esn/CMakeLists.txt`, add `esn_group.c` and `esn_group.h` to `kfcore_esn`, and install `esn_group.h` beside `esn.h` / `esn_sparse.h`. Add `esn/esn_group.c` to the focused workflow compile command.

- [ ] **Step 4: Run focused sanitizer gate**

Expected: all layout tests and all pre-existing ESN tests pass under ASan+UBSan. `kfcore_esn_grouped_step` is declared but is not referenced by tests yet.

- [ ] **Step 5: Commit**

```text
feat: add grouped ESN layout contract
```

---

### Task 3: Record grouped-step RED

**Files:**
- Modify: `esn/tests/test_esn_group.c`

**Interfaces:**
- Consumes: GREEN `kfcore_esn_grouped_layout`.
- Produces: RED requirements for `kfcore_esn_grouped_step`.

- [ ] **Step 1: Add single-group equivalence test**

Use identical initial state for ordinary and grouped execution:

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

- [ ] **Step 2: Add same-input heterogeneous concatenation test**

Create group 0 with one state and group 1 with two states; use zero recurrent matrices and leak `1.0f`. Both consume one scalar input. Assert the resulting caller state is exactly ordered as group 0 state, then group 1 state 0, then group 1 state 1, comparing against direct `tanhf(input_weight * input + bias)` expectations.

- [ ] **Step 3: Add validation failure atomicity test**

Use a valid first group and a later group with `reservoir_bias = NULL`. Preserve a byte-for-byte copy of caller state:

```c
float before[3];
memcpy(before, state, sizeof(state));
check_equal(kfcore_esn_grouped_step(&grouped, input, state, workspace),
            KFCORE_ESN_INVALID_ARGUMENT);
check_equal(memcmp(state, before, sizeof(state)), 0);
```

- [ ] **Step 4: Add NULL runtime argument tests**

Verify NULL grouped model, input, state, and workspace return `KFCORE_ESN_INVALID_ARGUMENT`; where caller state is present, verify it remains unchanged.

- [ ] **Step 5: Verify clean step RED**

Push and inspect the exact-head workflow.

Expected: existing layout implementation builds; linker fails only because `kfcore_esn_grouped_step` is undefined.

- [ ] **Step 6: Record checkpoint**

Record exact step-RED head and run/job evidence before production implementation.

---

### Task 4: Implement failure-atomic grouped stepping and final integration

**Files:**
- Modify: `esn/esn_group.c`
- Review: `esn/esn_group.h`
- Review: `esn/CMakeLists.txt`
- Review: `esn/tests/CMakeLists.txt`
- Review: `.github/workflows/esn-miniblas.yml`

**Interfaces:**
- Consumes: `kfcore_esn_grouped_measure`, `kfcore_esn_step`, and the layout contract.
- Produces: `kfcore_esn_grouped_step` and merge-ready grouped runtime.

- [ ] **Step 1: Validate runtime arguments and layout bounds**

Reject NULL `input`, `state`, or `workspace`. Call `kfcore_esn_grouped_measure`. Then reject if:

```c
if (total_state_size > INT_MAX - max_reservoir_size)
    return KFCORE_ESN_INVALID_ARGUMENT;
```

This mirrors layout overflow validation without introducing an unused local variable.

- [ ] **Step 2: Stage caller state**

```c
float* candidate = workspace;
float* scratch = workspace + total_state_size;
memcpy(candidate, state, sizeof(float) * (size_t)total_state_size);
```

No caller-state write occurs before all groups succeed.

- [ ] **Step 3: Advance staged groups in declaration order**

```c
int offset = 0;
for (int i = 0; i < grouped->group_count; ++i)
{
    const kfcore_esn_model* group = &grouped->groups[i];
    const kfcore_esn_status status =
        kfcore_esn_step(group, input, candidate + offset, scratch);
    if (status != KFCORE_ESN_OK)
        return status;
    offset += group->reservoir_size;
}
```

Do not retry, skip, dispatch to sparse runtime, or partially commit.

- [ ] **Step 4: Commit staged state once**

```c
memcpy(state, candidate, sizeof(float) * (size_t)total_state_size);
return KFCORE_ESN_OK;
```

- [ ] **Step 5: Run full focused ASan+UBSan gate**

Require grouped tests plus every existing ESN test to pass. Also require exact-source verification, tracked-source cleanliness, and artifact upload.

- [ ] **Step 6: Review public contract and scope**

Verify `esn_group.h` explicitly documents common input size, declaration-order state concatenation, workspace formula, caller ownership, dense-only behavior, and non-overlap requirements. Confirm the diff contains no sparse dispatch, grouped readout/training, deep/gdESN, serialization, topology generation, or changes to existing single-reservoir recurrence math.

- [ ] **Step 7: Record final exact-head evidence**

Record final head SHA, workflow run/job ID, test count, assertion count, ASan+UBSan result, source-cleanliness result, and artifact result in #36 / PR body. Do not reuse a GREEN result from an older head after documentation/build changes.

- [ ] **Step 8: Ready, merge, and verify master**

Mark Ready only after final exact-head GREEN. Merge with expected head SHA. Then require the resulting `master` push gate to pass before closing #36 as completed.
