# Grouped-Deep ESN Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an allocation-free grouped-deep ESN runtime that composes multiple existing dense deep chains over one shared external input, preserves group-major/depth-minor feature layout, and commits caller state atomically.

**Architecture:** Add `esn_group_deep.h` / `esn_group_deep.c` as a composition layer over the public `kfcore_esn_deep_layout` and `kfcore_esn_deep_step` APIs. The outer runtime stages the complete grouped state, reuses one inner workspace region sized to the largest deep-group workspace, and commits once after every group succeeds. Existing single-reservoir, gESN, dESN, sparse, ridge, and RLS behavior remains unchanged.

**Tech Stack:** C11, existing KFCore dESN API, TinyTest, CMake, GitHub Actions focused C11 ASan+UBSan ESN gate.

**Spec:** `docs/superpowers/specs/2026-09-17-esn-grouped-deep-runtime-design.md`

## Global Constraints

- Base lineage: `master` at `1497e34c3ed228d4838044cb8a693277c6e1cfdc`.
- Every group is an existing valid dense `kfcore_esn_deep_model`.
- Every group consumes the same external input sample at timestep `t`.
- All groups' first layers must expose the same positive `input_size`.
- dESN same-timestep propagation remains delegated to `kfcore_esn_deep_step`; do not reimplement layer recurrence.
- Public state/feature layout is group-major, depth-minor: concatenate each complete deep-group state in group declaration order.
- Grouped state size is `sum(group_deep_state_size)`.
- Grouped workspace size is `total_state_size + max(group_deep_workspace_size)` floats.
- Complete grouped-deep validation occurs before outer candidate staging.
- Runtime owns no model/state/workspace memory and performs no hidden allocation or global mutation.
- Caller state is failure-atomic: validation failure or any delegated non-OK deep-step status must not partially commit caller state.
- Nested outer/inner staging is intentional in v1; do not extract private non-staging dESN helpers.
- Dense-only composition. No sparse dispatch/fallback, retry, group skip, topology graph abstraction, parallel group scheduling, skip/residual connections, grouped-deep-specific readout/training, serialization, or application integration.
- Public APIs must document non-overlap requirements explicitly.

---

## File Structure

- Create `esn/esn_group_deep.h` — public non-owning grouped-deep model view and layout/step declarations.
- Create `esn/esn_group_deep.c` — public-dESN-based validation/layout, nested staging, group iteration, and final caller-state commit.
- Create `esn/tests/test_esn_group_deep.c` — layout, overflow, equivalence, shared-input, ordering, same-timestep propagation, invalid-later-group, NULL, and atomicity coverage.
- Modify `esn/CMakeLists.txt` — compile `esn_group_deep.c` and install `esn_group_deep.h`.
- Modify `esn/tests/CMakeLists.txt` — add grouped-deep tests to the existing TinyTest target.
- Modify `.github/workflows/esn-miniblas.yml` — compile grouped-deep source/test in the focused sanitizer gate.

---

### Task 1: Public grouped-deep contract and clean layout RED

**Files:**
- Create: `esn/esn_group_deep.h`
- Create: `esn/tests/test_esn_group_deep.c`
- Modify: `esn/tests/CMakeLists.txt`
- Modify: `.github/workflows/esn-miniblas.yml`

**Interfaces:**
- Consumes: `kfcore_esn_deep_model`, `kfcore_esn_deep_layout`, and `kfcore_esn_status` from the existing dESN/ESN public headers.
- Produces:

```c
typedef struct kfcore_esn_grouped_deep_model
{
    int group_count;
    const kfcore_esn_deep_model* groups;
} kfcore_esn_grouped_deep_model;

kfcore_esn_status kfcore_esn_grouped_deep_layout(
    const kfcore_esn_grouped_deep_model* grouped_deep,
    int* state_size,
    int* workspace_size);

kfcore_esn_status kfcore_esn_grouped_deep_step(
    const kfcore_esn_grouped_deep_model* grouped_deep,
    const float* input,
    float* state,
    float* workspace);
```

- [ ] **Step 1: Add `esn/esn_group_deep.h`**

Use include guards, include `esn_deep.h`, preserve C++ `extern "C"`, and document:

- all groups consume the same external input;
- state/feature ordering is group-major then each group's existing depth order;
- layout outputs are required;
- workspace is `total state + max deep workspace`;
- complete validation occurs before staging;
- `input`, `state`, `workspace`, and all deep-layer weight/bias buffers obey explicit non-overlap requirements;
- no hidden allocation or sparse fallback.

- [ ] **Step 2: Add layout-only RED tests**

Create `esn/tests/test_esn_group_deep.c` with `<limits.h>`, `esn.h`, `esn_deep.h`, `esn_group_deep.h`, then `#define TINYTEST_NO_MAIN` before `tinytest.h`.

Use a heterogeneous valid layout:

```text
group 0 layer reservoirs: 2, 1  -> deep state 3, deep workspace 5
group 1 layer reservoirs: 3, 2  -> deep state 5, deep workspace 8
gd state = 8
gd workspace = 8 + 8 = 16
```

Both groups' first layers use the same external `input_size = 2`; each later layer input size equals the preceding reservoir size.

Assert:

```c
check_equal(kfcore_esn_grouped_deep_layout(&model, &state_size, &workspace_size),
            KFCORE_ESN_OK);
check_equal(state_size, 8);
check_equal(workspace_size, 16);
```

Also cover:

1. required NULL layout outputs;
2. NULL model, zero `group_count`, NULL group array;
3. different first-layer external input sizes across otherwise-valid groups;
4. invalid later deep group metadata rejected through public `kfcore_esn_deep_layout` without modifying output sentinels;
5. checked grouped-state overflow using **three individually valid one-layer deep groups**, each with `input_size = 1`, `reservoir_size = INT_MAX / 2`, finite leak, and non-NULL one-element placeholder weight/bias pointers. Each group's deep workspace is `2 * (INT_MAX / 2) = INT_MAX - 1`, so the individual group is layout-valid while the three-group state sum exceeds `INT_MAX`;
6. checked grouped-workspace overflow using **two individually valid one-layer deep groups**, each with `input_size = 1`, `reservoir_size = INT_MAX / 3`. Each deep workspace is `2 * (INT_MAX / 3)` and fits in `int`; grouped total state also fits, but `total_state + max_group_workspace` exceeds `INT_MAX`.

Do not allocate matrices matching the huge metadata dimensions; layout validation only requires non-NULL reservoir-side pointers and does not dereference those matrices.

- [ ] **Step 3: Register only the grouped-deep test source**

Append `test_esn_group_deep.c` to `esn/tests/CMakeLists.txt` and to the focused compiler command in `.github/workflows/esn-miniblas.yml`. Do **not** add `esn_group_deep.c` yet.

- [ ] **Step 4: Open/update a Draft PR and verify clean layout RED**

Expected exact-head gate result: C11 compilation succeeds and link fails only with undefined references to `kfcore_esn_grouped_deep_layout`. TinyTest duplicate-main, compiler, or unrelated linker failures do not count.

- [ ] **Step 5: Record exact layout RED SHA/run/job evidence**

Record the exact RED head and GitHub Actions run/job IDs in #40 and the Draft PR before adding production symbols.

---

### Task 2: Checked grouped-deep layout GREEN

**Files:**
- Create: `esn/esn_group_deep.c`
- Modify: `esn/CMakeLists.txt`
- Modify: `.github/workflows/esn-miniblas.yml`

**Interfaces:**
- Consumes: public `kfcore_esn_deep_layout`.
- Produces: working `kfcore_esn_grouped_deep_layout` and a private scalar measurement helper reused by Task 4.

- [ ] **Step 1: Add private grouped-deep measurement helper**

Start `esn/esn_group_deep.c` with:

```c
#include "esn_group_deep.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>
```

Define:

```c
static kfcore_esn_status kfcore_esn_grouped_deep_measure(
    const kfcore_esn_grouped_deep_model* grouped_deep,
    int* common_input_size,
    int* total_state_size,
    int* max_group_workspace_size);
```

Reject NULL helper outputs, NULL model, `group_count <= 0`, or NULL group array.

For each group:

```c
int group_state_size = 0;
int group_workspace_size = 0;
const kfcore_esn_status status =
    kfcore_esn_deep_layout(&grouped_deep->groups[i],
                           &group_state_size,
                           &group_workspace_size);
if (status != KFCORE_ESN_OK)
    return status;
```

Only after that call succeeds, read `grouped_deep->groups[i].layers[0].input_size`. The first valid group establishes the common external input size; every later group must match it.

Accumulate `group_state_size` in `size_t`. Before addition require:

```c
(size_t)group_state_size <= (size_t)INT_MAX - total
```

Track the largest `group_workspace_size`. Write helper outputs only after every group validates.

- [ ] **Step 2: Implement layout without partial outputs**

```c
kfcore_esn_status kfcore_esn_grouped_deep_layout(
    const kfcore_esn_grouped_deep_model* grouped_deep,
    int* state_size,
    int* workspace_size)
{
    if (!state_size || !workspace_size)
        return KFCORE_ESN_INVALID_ARGUMENT;

    int input_size = 0;
    int total = 0;
    int max_group_workspace = 0;
    const kfcore_esn_status status =
        kfcore_esn_grouped_deep_measure(grouped_deep, &input_size,
                                        &total, &max_group_workspace);
    if (status != KFCORE_ESN_OK)
        return status;

    if (total > INT_MAX - max_group_workspace)
        return KFCORE_ESN_INVALID_ARGUMENT;

    *state_size = total;
    *workspace_size = total + max_group_workspace;
    return KFCORE_ESN_OK;
}
```

Do not modify either output on failure.

- [ ] **Step 3: Wire production source/header into build/install**

Add `esn_group_deep.c` / `esn_group_deep.h` to the `kfcore_esn` target, install `esn_group_deep.h` beside the other ESN public headers, and add `esn/esn_group_deep.c` to the focused workflow compile command.

- [ ] **Step 4: Run the exact-head focused sanitizer gate**

Expected: grouped-deep layout tests and all existing ESN tests pass under ASan+UBSan. `kfcore_esn_grouped_deep_step` is declared but remains unreferenced by tests.

- [ ] **Step 5: Record exact layout GREEN evidence**

Record exact head, run/job ID, test/assertion counts, source cleanliness, and artifact result before adding step tests.

---

### Task 3: Grouped-deep step clean RED

**Files:**
- Modify: `esn/tests/test_esn_group_deep.c`

**Interfaces:**
- Consumes: GREEN `kfcore_esn_grouped_deep_layout` and existing `kfcore_esn_deep_step`.
- Produces: behavioral RED contract for `kfcore_esn_grouped_deep_step`.

- [ ] **Step 1: Add one-group equivalence**

Build one valid two-layer deep group. Clone the same initial deep state into `ordinary_state` and `grouped_state`. Call ordinary `kfcore_esn_deep_step` with its own deep workspace, then call grouped-deep step with `grouped_state` and the grouped workspace returned by layout. Compare every state element within `1.0e-5f`.

- [ ] **Step 2: Add group-major/depth-minor ordering**

Use two groups with different depths/sizes, zero recurrence, deterministic feed-forward weights, and leak `1.0f`:

```text
group 0: layer sizes 2, 1
group 1: layer sizes 1, 2
```

Assert the final flat caller state uses exactly:

```text
[group0.layer0(2)][group0.layer1(1)][group1.layer0(1)][group1.layer1(2)]
```

- [ ] **Step 3: Add nested same-timestep propagation regression**

For each of two groups, use a two-layer scalar chain with zero recurrence/bias, leak `1`, scalar input weights. Use different group input weights so the groups produce different values from the same external input. For each group independently assert:

```c
expected_l0 = tanhf(group_input_weight * input[0]);
expected_l1 = tanhf(expected_l0);
```

Also verify `expected_l1` differs from the value obtained by feeding that group's old layer-0 state, so the test detects stale-state propagation inside either group.

- [ ] **Step 4: Add pre-validation atomicity tests**

Preserve caller state with `memcpy`, then verify `KFCORE_ESN_INVALID_ARGUMENT` and byte-for-byte unchanged state for:

1. external input-size mismatch between two otherwise-valid groups;
2. valid group 0 followed by malformed group 1 metadata such as `reservoir_bias = NULL` in one layer.

These failures must occur during complete grouped-deep pre-validation, before outer workspace staging.

- [ ] **Step 5: Add NULL public argument tests**

Verify NULL grouped-deep model, input, state, and workspace return `KFCORE_ESN_INVALID_ARGUMENT`; whenever caller state is supplied, verify it remains unchanged.

Do not manufacture a fake post-validation dESN runtime failure solely to exercise delegated status forwarding. The implementation must still return any delegated non-OK status unchanged and avoid caller commit.

- [ ] **Step 6: Verify clean step RED**

Expected exact-head gate result: existing layout implementation compiles, and link fails only because `kfcore_esn_grouped_deep_step` is undefined.

- [ ] **Step 7: Record exact step RED evidence**

Record exact SHA and run/job evidence before implementing the step symbol.

---

### Task 4: Nested-staging grouped-deep step GREEN

**Files:**
- Modify: `esn/esn_group_deep.c`

**Interfaces:**
- Consumes: `kfcore_esn_grouped_deep_measure`, public `kfcore_esn_deep_layout`, public `kfcore_esn_deep_step`.
- Produces: working `kfcore_esn_grouped_deep_step`.

- [ ] **Step 1: Validate public pointers and complete grouped model before staging**

Reject NULL input/state/workspace, call `kfcore_esn_grouped_deep_measure`, and reject:

```c
total_state_size > INT_MAX - max_group_workspace_size
```

before any workspace write.

- [ ] **Step 2: Stage complete caller state once**

```c
float* candidate = workspace;
float* inner_workspace = workspace + total_state_size;
memcpy(candidate, state, sizeof(float) * (size_t)total_state_size);
```

The public non-overlap contract makes this outer candidate independent from caller state.

- [ ] **Step 3: Execute each deep group using the public dESN API**

Use declaration order and recompute each group's scalar state size through public layout:

```c
int offset = 0;
for (int i = 0; i < grouped_deep->group_count; ++i)
{
    int group_state_size = 0;
    int group_workspace_size = 0;
    const kfcore_esn_status layout_status =
        kfcore_esn_deep_layout(&grouped_deep->groups[i],
                               &group_state_size,
                               &group_workspace_size);
    if (layout_status != KFCORE_ESN_OK)
        return layout_status;

    const kfcore_esn_status step_status =
        kfcore_esn_deep_step(&grouped_deep->groups[i], input,
                             candidate + offset, inner_workspace);
    if (step_status != KFCORE_ESN_OK)
        return step_status;

    offset += group_state_size;
}
```

Do not expose or call a private dESN non-staging helper. The repeated per-group layout call is intentional v1 validation and scalar-offset recovery.

- [ ] **Step 4: Commit caller state once after every group succeeds**

```c
memcpy(state, candidate, sizeof(float) * (size_t)total_state_size);
return KFCORE_ESN_OK;
```

There is no retry, rollback loop, sparse fallback, group skip, or partial caller-state copy.

- [ ] **Step 5: Run exact-head focused sanitizer verification**

Require focused C11 build and full ESN runtime success under ASan+UBSan, including one-group equivalence, group-major/depth-minor ordering, same-timestep propagation in both groups, and pre-validation atomicity.

- [ ] **Step 6: Review the full PR diff against #40 scope**

Expected production changes are limited to the new grouped-deep module and build/install/workflow integration. No changes to `esn.c`, `esn_group.c`, `esn_deep.c`, sparse runtime, reservoir initialization, RLS, or application code.

- [ ] **Step 7: Update PR/#40 evidence without moving code head**

Record layout RED, layout GREEN, step RED, final exact-head GREEN, test/assertion counts, tracked-source cleanliness, and artifact evidence.

---

### Task 5: Finish only after exact-head verification

- [ ] Verify the PR is still on the recorded exact head, mergeable, and has no unresolved review thread.
- [ ] Mark Ready and merge with an expected-head SHA guard.
- [ ] Verify the exact `master` merge commit's push-triggered `ESN miniblas contract` through exact-source verification, focused C11 build, ASan+UBSan runtime, tracked-source cleanliness, and artifact upload.
- [ ] Read the final master runtime summary and record exact test/assertion counts.
- [ ] Update the merged PR and #40 with final master evidence, then close #40 as `completed`.
