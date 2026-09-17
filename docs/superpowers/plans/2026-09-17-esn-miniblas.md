# ESN miniblas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure-C ESN runtime and ridge readout fitter backed by KFCore miniblas.

**Architecture:** Keep miniblas as the numerical primitive layer and add an independent `KFCore::esn` target. The ESN model is a non-owning view over caller-provided column-major weights; state/workspace are caller-owned and the runtime performs no hidden allocation.

**Tech Stack:** C11, CMake, KFCore miniblas, Salts TinyTest.

**Spec:** `docs/superpowers/specs/2026-09-17-esn-miniblas-design.md`

## Global Constraints

- Use float32 and column-major matrices.
- No explicit matrix inverse.
- No hidden allocation or global mutable state.
- No fallback solver after Cholesky failure.
- Keep random initialization, spectral-radius scaling, sparse reservoirs, RLS, deep/grouped ESN, and serialization out of this slice.

---

### Task 1: Runtime contract and deterministic tests

**Files:**
- Create: `esn/esn.h`
- Create: `esn/esn.c`
- Create: `esn/tests/test_esn.c`

**Interfaces:**
- Produces: `kfcore_esn_model`, `kfcore_esn_step`, `kfcore_esn_predict`, `kfcore_esn_step_predict`.
- Consumes: `matvec` from `linalg.h`.

- [ ] **Step 1: Write failing tests** for a two-neuron leaky state update, linear prediction, and invalid leak/pointer rejection.
- [ ] **Step 2: Build the test target** and confirm the symbols are initially absent.
- [ ] **Step 3: Implement the minimal runtime** using `matvec`, `tanhf`, caller-provided state, and a reservoir-sized workspace.
- [ ] **Step 4: Rebuild and run** the focused ESN test.

### Task 2: Ridge readout fitting

**Files:**
- Modify: `esn/esn.h`
- Modify: `esn/esn.c`
- Modify: `esn/tests/test_esn.c`

**Interfaces:**
- Produces: `kfcore_esn_fit_ridge(const float *states, const float *targets, int reservoir_size, int output_size, int sample_count, float lambda, float *output_weights, float *gram_workspace)`.
- Consumes: `matmul`, `cholesky`, and `trisolveright` from `linalg.h`.

- [ ] **Step 1: Add failing tests** that recover a slope near 2 and reject non-positive lambda.
- [ ] **Step 2: Implement** `Wout = Y X^T (X X^T + lambda I)^-1` without forming an inverse.
- [ ] **Step 3: Add numerical-failure coverage** for non-finite/overflowing Gram matrices; do not add a fallback solver.
- [ ] **Step 4: Run** the focused ESN test again.

### Task 3: Build/export integration

**Files:**
- Create: `esn/CMakeLists.txt`
- Create: `esn/tests/CMakeLists.txt`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: build/install target `KFCore::esn` and CTest target `kfcore_esn_tests`.
- Consumes: `KFCore::miniblas_internal` and `Salts::TinyTest`.

- [ ] **Step 1: Register** `esn` from the top-level CMake file.
- [ ] **Step 2: Export/install** the ESN target and `esn.h`.
- [ ] **Step 3: Register tests** only when `BUILD_TESTS` is enabled.
- [ ] **Step 4: Run the repository CI** and record the exact head/check results in the PR.
