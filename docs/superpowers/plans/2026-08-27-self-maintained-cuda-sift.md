# Self-Maintained CUDA SIFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the machine-local PopSift dependency with a reproducible KFCore-maintained CUDA SIFT backend whose queues, failures, ownership, and shutdown are bounded and testable.

**Architecture:** Preserve `KFCore::sift` and `KFCore::sift_popsift`, but compile a private static CUDA target from a traceable PopSift 0.10.1 source fork in `sift/vendor/popsift`. Keep one native backend per CUDA device because upstream algorithm state is device-global; make its two-stage pipeline bounded and propagate every worker failure through the job future.

**Tech Stack:** C++17, CUDA 12.x/CUDAToolkit, CMake presets and package exports, TurboUtils TinyTest, PopSift 0.10.1 algorithm sources under MPL-2.0.

**Spec:** `docs/superpowers/specs/2026-08-26-sift-extractor-design.md`

## Global Constraints

- Preserve the public target names `KFCore::sift` and `KFCore::sift_popsift` and all existing feature/result/error types.
- Import only PopSift commit `36d704d39b4cc065839d84f3706b3fa88eff2518` library sources; preserve MPL-2.0 headers, `COPYING.md`, and exact provenance.
- Remove `POPSIFT_ROOT`, the external `add_subdirectory`, and the separately installed `popsift.dll`.
- The pipeline is multi-producer, one upload worker, one CUDA extraction worker; stage-1 is bounded by `max_pending_jobs`, stage-2 and the image pool have capacity 2.
- Queue construction may allocate once; push, pull, close, and wake-up paths must not allocate while holding the queue mutex.
- Worker exceptions must complete the affected promise or close/drain the pipeline; no submitted job may wait forever.
- Same-device native state remains singleton because algorithm constants and pyramid pointers are device-global.
- Run configure/build/tests from the VS 2022 x64 developer environment because the Windows presets use `architecture.strategy=external`.

---

### Task 1: Record the reproducible backend boundary

**Files:**
- Modify: `docs/superpowers/specs/2026-08-26-sift-extractor-design.md`
- Create: `docs/superpowers/plans/2026-08-27-self-maintained-cuda-sift.md`
- Create: `sift/vendor/popsift/UPSTREAM.md`
- Create: `sift/vendor/popsift/COPYING.md`
- Create: `sift/vendor/popsift/cmake/sift_config.h.in`
- Create: all 63 files below `sift/vendor/popsift/src/popsift/`

**Interfaces:**
- Consumes: upstream PopSift source commit `36d704d39b4cc065839d84f3706b3fa88eff2518`.
- Produces: an exact source baseline at `sift/vendor/popsift/src/popsift` and a human-auditable modification ledger.

- [x] **Step 1: Verify the local research snapshot against upstream**

Run `git -C build/research/popsift-upstream rev-parse HEAD` and compare the local snapshot with `git diff --no-index`.
Expected: commit is exact and the source diff has no content changes.

- [x] **Step 2: Import only the library source and metadata**

Copy `src/popsift`, `cmake/sift_config.h.in`, and `COPYING.md` without transformations. Create `UPSTREAM.md` with URL,
commit, version, import scope, exclusions, verification commands, and a local-modification ledger.

- [x] **Step 3: Verify the mechanical import before local patches**

Run `git diff --no-index -- build/research/popsift-upstream/src/popsift sift/vendor/popsift/src/popsift`.
Expected: no content differences.

- [x] **Step 4: Commit the source baseline separately**

```powershell
git add sift/vendor/popsift docs/superpowers
git commit -m "vendor: import PopSift CUDA SIFT baseline"
```

### Task 2: Make pipeline capacity an explicit KFCore contract

**Files:**
- Modify: `sift/include/kfcore/sift/popsift_options.hpp`
- Modify: `sift/src/sift_extractor.cpp`
- Modify: `sift/src/popsift_extractor.cpp`
- Modify: `sift/tests/test_popsift_options.cpp`

**Interfaces:**
- Consumes: existing `PopSiftOptions::validate()` and backend registry.
- Produces: `std::size_t PopSiftOptions::max_pending_jobs`, default 8, accepted range `[1, 1024]`.

- [x] **Step 1: Write the failing options tests**

```cpp
check(defaults.max_pending_jobs == PopSiftOptions::kDefaultMaxPendingJobs);
options.max_pending_jobs = 0;
expect_sift_error([&] { options.validate(); },
                  SiftErrorCode::ResourceLimitExceeded, "pending job limit");
```

- [x] **Step 2: Run the focused test and observe RED**

Run the `test_popsift_options` target and test. Expected: compile failure because the new members do not exist.

- [x] **Step 3: Implement validation and backend compatibility**

```cpp
static constexpr std::size_t kDefaultMaxPendingJobs = 8U;
static constexpr std::size_t kMaximumMaxPendingJobs = 1024U;
std::size_t max_pending_jobs = kDefaultMaxPendingJobs;
```

Validate the closed interval and include the value in `BackendState`, `compatible_backend`, and internal constructor call.

- [x] **Step 4: Re-run the focused test and commit**

Expected: `test_popsift_options` passes; commit as `feat(sift): bound pending CUDA jobs`.

### Task 3: Replace the unbounded queue and broken future errors

**Files:**
- Modify: `sift/vendor/popsift/src/popsift/common/sync_queue.h`
- Modify: `sift/vendor/popsift/src/popsift/popsift.h`
- Modify: `sift/vendor/popsift/src/popsift/popsift.cu`
- Create: `sift/tests/test_popsift_internal.cpp`
- Modify: `sift/CMakeLists.txt`

**Interfaces:**
- Consumes: `max_pending_jobs` from Task 2.
- Produces: `SyncQueue<T>(std::size_t)`, `bool push(const T&)`, `bool pull(T&)`, `void close()`, and promise-based error propagation.

- [x] **Step 1: Write deterministic RED tests**

Cover FIFO, capacity-1 producer blocking/wake-up, consumer close wake-up, drain-after-close, rejected push, and host/device getter
exception propagation. Synchronize with promises/futures; `wait_for` is only a bounded failure guard.

- [x] **Step 2: Build the internal test and observe RED**

Expected: compile failure because bounded/close APIs and direct future exception propagation do not exist.

- [x] **Step 3: Implement the preallocated bounded queue**

Use one constructor-sized `std::vector<T>`, head/tail/count, one mutex, and `not_empty`/`not_full` condition variables.
`push` and `pull` wait on capacity/data or closure; `close` changes state under lock then wakes all. Reject zero capacity.

- [x] **Step 4: Implement exception completion and orderly close**

Replace `_err` plus `set_value(nullptr)` with `promise.set_exception`. Convert sentinel shutdown to close/drain. Catch per-job
upload/extraction errors, return an acquired image exactly once, delete partial results, and complete the job. Worker initialization
failure closes input and completes queued jobs with the same exception.

- [x] **Step 5: Run CPU-internal and CUDA adapter tests**

Run `test_popsift_internal` and `test_popsift_adapter`; repeat the adapter test 20 times. Expected: no failure or hang.

- [x] **Step 6: Commit the runtime hardening**

Commit as `fix(sift): bound PopSift pipeline failures`.

### Task 4: Build the maintained fork without POPSIFT_ROOT

**Files:**
- Modify: `sift/CMakeLists.txt`
- Modify: `CMakeUserPresets.json`
- Modify: `cmake/tests/test_sift_installed_consumer.cmake`
- Modify: `sift/README.md`
- Modify: `sift/vendor/popsift/UPSTREAM.md`

**Interfaces:**
- Consumes: vendored sources and internal runtime APIs.
- Produces: private static `kfcore_popsift_internal`; installed public target stays `KFCore::sift_popsift`.

- [x] **Step 1: Establish configure RED**

Remove `POPSIFT_ROOT` from the process and preset, then run a fresh configure. Expected before implementation: `requires POPSIFT_ROOT`.

- [x] **Step 2: Add the internal CUDA target**

List 63 files explicitly, generate private `popsift/sift_config.h`, link CUDA/Threads, set C++ and CUDA 17, separable compilation,
PIC, and the existing MSVC CUDA preprocessor option. Do not install or export this target.

- [x] **Step 3: Remove external build/runtime assumptions**

Remove external subdirectory/version checks, target-file workaround, DLL copy/install, and preset environment variable. Install only
`COPYING.md` and `UPSTREAM.md` under `share/licenses/KFCore/popsift`.

- [x] **Step 4: Configure, build, and run installed consumer**

Expected: no `POPSIFT_ROOT` access and installed consumer runs without `popsift.dll`.

- [x] **Step 5: Commit the reproducible build**

Commit as `build(sift): internalize CUDA SIFT backend`.

### Task 5: Verify behavior, performance, package contents, and PR metadata

**Files:**
- Modify if evidence requires: `sift/tests/test_popsift_adapter.cpp`
- Modify if evidence requires: `sift/README.md`
- Modify: PR #12 and issue #11 descriptions/comments

**Interfaces:**
- Consumes: all earlier tasks.
- Produces: repeatable verification evidence and an updated reviewable PR.

- [x] **Step 1: Run focused and repeated tests**

Run all `test_(sift|popsift)` tests and repeat `test_popsift_adapter` 20 times. Expected: zero failures.

- [x] **Step 2: Run the full preset**

Run `ctest --preset win-release-user --output-on-failure`. Expected: all configured tests pass.

- [x] **Step 3: Inspect installed artifacts**

Use `fd.exe` under the isolated consumer prefix. Expected: KFCore SIFT DLLs, license, and provenance exist; `popsift.dll` and
public PopSift headers do not.

- [x] **Step 4: Compare persistent-extractor timing**

Use identical image, options, warm-up, and iteration count before/after. Report measured median/per-image time; do not infer GPU
saturation from functional tests.

- [x] **Step 5: Inspect, push, and update tracking**

Run `git diff --check origin/master...HEAD`, inspect status/log, push `feature/sift-extractor`, then update PR #12 and issue #11
with upstream commit, one-native-backend/device limit, test evidence, artifact change, and remaining performance risk.
