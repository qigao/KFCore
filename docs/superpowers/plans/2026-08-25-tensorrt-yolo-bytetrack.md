# TensorRT YOLO ByteTrack Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a clean-room TensorRT YOLO detector and a class-isolated tracking pipeline that uses KFCore's native ByteTrack/Kalman implementation.

**Architecture:** The feature is split into a TensorRT-free `yolo_tracking` library, a TensorRT/CUDA `tensorrt_yolo` library, and an optional `yolo_opencv` adapter. Immutable TensorRT engine state may be shared, while each detector owns its execution context, CUDA stream, and bounded buffers; each video stream owns one sequential `ByteTrackSession`.

**Tech Stack:** C11, C++17, CUDA17, TensorRT 10.x/11.x runtime API, KFCore ByteTrack/Kalman, TurboUtils TinyTest, CMake Presets, optional OpenCV 4.13 lite.

**Spec:** `docs/superpowers/specs/2026-08-25-tensorrt-yolo-bytetrack-design.md`

## Global Constraints

- Do not copy, compile, include, or link source from `TensorRT-YOLO/`; it is GPL-3.0 reference material only.
- `TensorRT-YOLO11/` is method reference only; do not adopt its Eigen Kalman or ByteTrack implementation.
- Existing `bytetrack_update()` behavior and signature remain compatible; all new tracker failures use the additive `_ex` API.
- `KFCORE_BUILD_YOLO_TRACKING`, `KFCORE_BUILD_TENSORRT_YOLO`, `KFCORE_BUILD_YOLO_OPENCV`, and `KFCORE_BUILD_TENSORRT_INTEGRATION_TESTS` default to `OFF`.
- TensorRT and OpenCV options require tracking to be explicitly enabled; CMake must not rewrite the user's option values.
- TensorRT code uses named tensors, `setTensorAddress()`, `enqueueV3()`, and 64-bit dimensions; old binding-array APIs are forbidden.
- Only trusted EfficientNMS engines with the documented five-tensor contract are accepted.
- All byte-size arithmetic is checked before allocation; every growing resource has a configured hard limit.
- `ImageView` is borrowed only for the duration of `detect()`/`detect_batch()`; detector-owned mutable buffers never escape.
- One detector instance and one tracking session are each single-threaded; parallelism uses distinct instances.
- No runtime fallback to ONNX Runtime, OpenCV DNN, CPU NMS, smaller batches, lower resolution, or dropped detections.
- CMake configure, build, test, and install verification uses version-controlled user presets.
- TinyTest supplies `main()`; each `it(...)` verifies one named behavior.

---

## File Structure

### Existing files to modify

- `trackers/include/trackers/tracker.h`: additive status and indexed ByteTrack update API.
- `trackers/src/tracker.c`: transactional `_ex` update and compatibility wrapper.
- `trackers/tests/test_trackers.c`: legacy compatibility and indexed-output behavior tests.
- `trackers/CMakeLists.txt`: deterministic allocation-failure test target.
- `CMakeOptions.cmake`: four feature options and dependency checks.
- `CMakeLists.txt`: conditionally enable C++/CUDA and add the new subdirectory.
- `CMakeUserPresets.json`: public tracking-only and full TensorRT YOLO presets.
- `cmake/KFCoreConfig.cmake.in`: installed optional-target dependency discovery.
- `README.md`: build, ownership, engine trust, and minimal usage documentation.

### New files

- `trackers/tests/test_bytetrack_status.c`: `_ex` error atomicity and OOM injection tests.
- `trackers/tests/tracker_test_alloc.h`: test-only deterministic allocator control.
- `tensorrt_yolo/CMakeLists.txt`: three library targets, tests, examples, and install rules.
- `tensorrt_yolo/include/kfcore/yolo/types.hpp`: image, detection, and track value types.
- `tensorrt_yolo/include/kfcore/yolo/error.hpp`: typed public error code and exception.
- `tensorrt_yolo/include/kfcore/yolo/tracking.hpp`: `ByteTrackOptions` and `ByteTrackSession`.
- `tensorrt_yolo/include/kfcore/yolo/tensorrt.hpp`: `Engine`, options, and `TensorRtDetector`.
- `tensorrt_yolo/include/kfcore/yolo/opencv.hpp`: optional `cv::Mat` adapter and renderer.
- `tensorrt_yolo/src/checked_size.hpp`: overflow-safe size helpers.
- `tensorrt_yolo/src/tracking.cpp`: class grouping, KFCore adapter, ordering, and global IDs.
- `tensorrt_yolo/src/engine_contract.hpp`: TensorRT-independent metadata contract types.
- `tensorrt_yolo/src/engine_contract.cpp`: strict tensor validation.
- `tensorrt_yolo/src/tensorrt_raii.hpp`: TensorRT/CUDA RAII handles and deleters.
- `tensorrt_yolo/src/cuda_buffer.hpp`: bounded device and pinned-host buffer owners.
- `tensorrt_yolo/src/cuda_buffer.cpp`: checked allocation and release.
- `tensorrt_yolo/src/engine.cpp`: trusted engine loading and shared immutable engine state.
- `tensorrt_yolo/src/detector.cpp`: context creation, batching, enqueue, and postprocessing.
- `tensorrt_yolo/src/letterbox.hpp`: preprocessing interface and per-image transform.
- `tensorrt_yolo/src/letterbox.cu`: FP16/FP32 BGR/RGB letterbox kernel.
- `tensorrt_yolo/src/opencv.cpp`: `cv::Mat` view validation and drawing.
- `tensorrt_yolo/tests/test_yolo_tracking.cpp`: no-GPU tracking tests.
- `tensorrt_yolo/tests/test_tensorrt_contract.cpp`: no-GPU tensor contract tests.
- `tensorrt_yolo/tests/test_yolo_opencv.cpp`: OpenCV ROI/type/drawing tests.
- `tensorrt_yolo/tests/test_tensorrt_integration.cpp`: opt-in trusted-engine GPU tests.
- `tensorrt_yolo/examples/track_image_sequence.cpp`: image-sequence detect/track example.
- `cmake/FindTensorRT.cmake`: exact `TENSORRT_ROOT` discovery and imported targets.
- `cmake/FindOpenCVLite.cmake`: exact `OPENCV_LITE_ROOT` discovery and imported targets.

---

### Task 1: Add a transactional, indexed ByteTrack C API

**Files:**
- Modify: `trackers/include/trackers/tracker.h:19-105`
- Modify: `trackers/src/tracker.c:248-325,493-503,670-881`
- Modify: `trackers/tests/test_trackers.c:121-235`
- Modify: `trackers/CMakeLists.txt:34-48`
- Create: `trackers/tests/test_bytetrack_status.c`
- Create: `trackers/tests/tracker_test_alloc.h`

**Interfaces:**
- Consumes: existing `bytetrack_t`, `detection_t`, and `tracked_detection_t`.
- Produces: `tracker_status_t`, `tracked_detection_ex_t`, and `bytetrack_update_ex()` for Task 2.

- [ ] **Step 1: Write failing public-contract tests**

Add these public declarations to the test's expected usage before implementing them:

```c
it("reports indexed detections without changing legacy output") {
    bytetrack_config_t config = bytetrack_default_config();
    config.minimum_consecutive_frames = 1;
    bytetrack_t* tracker = bytetrack_create(&config);
    detection_t detections[2] = {
        make_detection(100.0f, 0.0f, 110.0f, 10.0f, 0.95f),
        make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.95f),
    };
    tracked_detection_ex_t output[2];
    size_t written = 99;

    check_equal(bytetrack_update_ex(tracker, detections, 2, output, 2, &written),
                TRACKER_STATUS_OK);
    check_equal(written, (size_t)2);
    check_less(output[0].detection_index, (size_t)2);
    check_less(output[1].detection_index, (size_t)2);
    check_not_equal(output[0].detection_index, output[1].detection_index);
    bytetrack_destroy(tracker);
}

it("rejects short output capacity without advancing tracker state") {
    bytetrack_config_t config = bytetrack_default_config();
    config.minimum_consecutive_frames = 1;
    bytetrack_t* tracker = bytetrack_create(&config);
    detection_t detections[1] = {
        make_detection(0.0f, 0.0f, 10.0f, 10.0f, 0.95f),
    };
    size_t written = 99;

    check_equal(bytetrack_update_ex(tracker, detections, 1, NULL, 0, &written),
                TRACKER_STATUS_CAPACITY);
    check_equal(written, (size_t)0);

    tracked_detection_ex_t output[1];
    check_equal(bytetrack_update_ex(tracker, detections, 1, output, 1, &written),
                TRACKER_STATUS_OK);
    check_equal(output[0].tracked.tracker_id, -1);
    check_equal(bytetrack_update_ex(tracker, detections, 1, output, 1, &written),
                TRACKER_STATUS_OK);
    check_equal(output[0].tracked.tracker_id, 0);
    bytetrack_destroy(tracker);
}
```

- [ ] **Step 2: Run the existing tracker target and verify compilation fails**

From an x64 Visual Studio 2022 Developer Command Prompt:

```powershell
cmake --fresh --preset win-dev-user
cmake --build --preset win-dev-user --target test_tracker
```

Expected: compilation fails because `tracked_detection_ex_t`, `tracker_status_t`, and `bytetrack_update_ex` are not defined.

- [ ] **Step 3: Define the additive API and transactional update**

Add to `tracker.h`:

```c
typedef enum tracker_status {
    TRACKER_STATUS_OK = 0,
    TRACKER_STATUS_INVALID_ARGUMENT = 1,
    TRACKER_STATUS_CAPACITY = 2,
    TRACKER_STATUS_OVERFLOW = 3,
    TRACKER_STATUS_ALLOCATION_FAILED = 4
} tracker_status_t;

typedef struct tracked_detection_ex {
    tracked_detection_t tracked;
    size_t detection_index;
} tracked_detection_ex_t;

tracker_status_t bytetrack_update_ex(
    bytetrack_t* tracker,
    const detection_t* detections,
    size_t detection_count,
    tracked_detection_ex_t* output,
    size_t output_capacity,
    size_t* output_count
);
```

Refactor the ByteTrack frame path into prepare and commit phases:

```c
typedef struct byte_frame_scratch {
    int* high_indices;
    int* low_indices;
    box_t* track_boxes;
    box_t* high_boxes;
    box_t* remaining_boxes;
    box_t* low_boxes;
    float* high_iou;
    float* low_iou;
} byte_frame_scratch_t;
```

Before the first `kf_xyxy_predict()` call:

- set `*output_count = 0`;
- validate null/count/capacity combinations;
- reject `detection_count > SIZE_MAX / sizeof(...)`;
- reserve `tracker->track_count + detection_count` with checked addition;
- allocate all index, box, IoU, and assignment storage;
- return the exact status after releasing scratch if any preparation fails.

Replace the allocating `assign_greedy()` calls on this path with an `assign_greedy_into()` variant whose match/unmatched arrays are supplied from prepared scratch. During commit, write `detection_index = det_idx` with every tracked result. No allocation is allowed after prediction begins. Keep legacy `bytetrack_update()` by allocating an `_ex` result array, copying at most `output_capacity` legacy records, and returning the full count as before.

- [ ] **Step 4: Add deterministic allocation-failure coverage**

Compile a test-only copy of `tracker.c` with `KFCORE_TRACKERS_TEST_ALLOCATOR`. Under that definition, route `malloc/calloc/realloc` through counters declared in `tracker_test_alloc.h`:

```c
trackers_test_alloc_reset();
trackers_test_alloc_fail_after(0);
check_equal(bytetrack_update_ex(tracker, detections, 1, output, 1, &written),
            TRACKER_STATUS_ALLOCATION_FAILED);
trackers_test_alloc_reset();

check_equal(bytetrack_update_ex(tracker, detections, 1, output, 1, &written),
            TRACKER_STATUS_OK);
check_equal(output[0].tracked.tracker_id, 0);
```

The second update proves the failed first call did not consume an ID or mutate Kalman age/state.

- [ ] **Step 5: Run focused and adjacent tracker tests**

```powershell
cmake --build --preset win-dev-user --target test_tracker test_bytetrack_status
ctest --preset win-dev-user -R "test_tracker|test_bytetrack_status"
```

Expected: both tests pass under the configured AddressSanitizer profile.

- [ ] **Step 6: Commit Task 1**

```powershell
git add trackers/include/trackers/tracker.h trackers/src/tracker.c trackers/tests/test_trackers.c trackers/tests/test_bytetrack_status.c trackers/tests/tracker_test_alloc.h trackers/CMakeLists.txt
git commit -m "feat(trackers): add transactional ByteTrack update API"
```

---

### Task 2: Add public YOLO types and class-isolated tracking sessions

**Files:**
- Modify: `CMakeOptions.cmake`
- Modify: `CMakeLists.txt`
- Modify: `CMakeUserPresets.json`
- Create: `tensorrt_yolo/CMakeLists.txt`
- Create: `tensorrt_yolo/include/kfcore/yolo/types.hpp`
- Create: `tensorrt_yolo/include/kfcore/yolo/error.hpp`
- Create: `tensorrt_yolo/include/kfcore/yolo/tracking.hpp`
- Create: `tensorrt_yolo/src/checked_size.hpp`
- Create: `tensorrt_yolo/src/tracking.cpp`
- Create: `tensorrt_yolo/tests/test_yolo_tracking.cpp`

**Interfaces:**
- Consumes: `bytetrack_update_ex()` from Task 1.
- Produces: `ImageView`, `DetectionFrame`, `TrackFrame`, `YoloError`, and `ByteTrackSession` for Tasks 5 and 6.

- [ ] **Step 1: Write failing TinyTest cases for public semantics**

Create `test_yolo_tracking.cpp` with TinyTest-provided `main()`:

```cpp
#include "kfcore/yolo/tracking.hpp"
#include "tinytest.h"

#include <limits>

using namespace kfcore::yolo;

spec("YOLO ByteTrack session") {
    it("isolates overlapping detections by class and preserves input order") {
        ByteTrackOptions options;
        options.minimum_consecutive_frames = 1;
        ByteTrackSession session(options);
        DetectionFrame frame{640, 480, {
            {{10, 10, 50, 50}, 0.95f, 7},
            {{10, 10, 50, 50}, 0.95f, 2},
        }};

        (void)session.update(frame);
        TrackFrame result = session.update(frame);

        check_equal(result.detections.size(), (size_t)2);
        check_equal(result.detections[0].detection.class_id, 7);
        check_equal(result.detections[1].detection.class_id, 2);
        check(result.detections[0].track_id.has_value());
        check(result.detections[1].track_id.has_value());
        check_not_equal(*result.detections[0].track_id,
                        *result.detections[1].track_id);
    }

    it("advances absent classes on empty frames") {
        ByteTrackOptions options;
        options.minimum_consecutive_frames = 2;
        options.lost_track_buffer = 1;
        ByteTrackSession session(options);
        (void)session.update({640, 480, {{{0, 0, 10, 10}, 0.95f, 0}}});
        TrackFrame mature = session.update(
            {640, 480, {{{0, 0, 10, 10}, 0.95f, 0}}});
        check_equal(*mature.detections[0].track_id, (std::uint64_t)0);
        (void)session.update({640, 480, {}});
        TrackFrame tentative = session.update(
            {640, 480, {{{0, 0, 10, 10}, 0.95f, 0}}});
        check(!tentative.detections[0].track_id.has_value());
        TrackFrame result = session.update(
            {640, 480, {{{0, 0, 10, 10}, 0.95f, 0}}});
        check(result.detections[0].track_id.has_value());
        check_equal(*result.detections[0].track_id, (std::uint64_t)1);
    }

    it("rejects non-finite boxes without mutating the session") {
        ByteTrackSession session;
        DetectionFrame invalid{640, 480, {
            {{0, 0, std::numeric_limits<float>::quiet_NaN(), 10}, 0.9f, 0},
        }};
        check_throws_as(session.update(invalid), YoloError);
        check_nothrow(session.update({640, 480, {}}));
    }
}
```

- [ ] **Step 2: Add tracking-only options and verify the test initially fails**

Add `KFCORE_BUILD_YOLO_TRACKING` defaulting to `OFF`. Add public `win-yolo-tracking-dev-user` configure/build/test presets that inherit the matching `win-dev-user` entries and set only this option to `ON`; add `install-win-yolo-tracking-dev-user` targeting `install`.

```powershell
cmake --list-presets
cmake --fresh --preset win-yolo-tracking-dev-user
cmake --build --preset win-yolo-tracking-dev-user --target test_yolo_tracking
```

Expected: configure reaches the new module, then compilation fails because the public types/session are not implemented.

- [ ] **Step 3: Implement bounded public types and typed errors**

Define the value types and error model exactly:

```cpp
enum class PixelFormat { Bgr8, Rgb8 };
enum class MemoryKind { Host, CudaDevice };

struct ImageView {
    const void* data = nullptr;
    std::int32_t width = 0;
    std::int32_t height = 0;
    std::size_t row_stride = 0;
    PixelFormat pixel_format = PixelFormat::Bgr8;
    MemoryKind memory_kind = MemoryKind::Host;
};

struct BoxF { float left, top, right, bottom; };
struct Detection { BoxF box; float score; std::int32_t class_id; };
struct DetectionFrame {
    std::int32_t image_width;
    std::int32_t image_height;
    std::vector<Detection> detections;
};
struct TrackedDetection {
    Detection detection;
    std::optional<std::uint64_t> track_id;
};
struct TrackFrame {
    std::int32_t image_width;
    std::int32_t image_height;
    std::vector<TrackedDetection> detections;
};

enum class YoloErrorCode {
    InvalidArgument,
    FileIo,
    EngineDeserialize,
    EngineContractMismatch,
    TensorRtFailure,
    CudaFailure,
    TrackerAllocationFailure,
    ResourceLimitExceeded,
};

class YoloError final : public std::runtime_error {
public:
    YoloError(YoloErrorCode code, std::string message);
    YoloErrorCode code() const noexcept;
};

struct ByteTrackOptions {
    int lost_track_buffer = 30;
    float frame_rate = 30.0f;
    float track_activation_threshold = 0.7f;
    int minimum_consecutive_frames = 2;
    float minimum_iou_threshold = 0.1f;
    float high_conf_det_threshold = 0.6f;
    std::size_t max_detections_per_frame = 300;
    std::size_t max_class_trackers = 128;
};
```

Expose the session as:

```cpp
class ByteTrackSession final {
public:
    explicit ByteTrackSession(ByteTrackOptions options = {});
    ~ByteTrackSession();
    ByteTrackSession(ByteTrackSession&&) noexcept;
    ByteTrackSession& operator=(ByteTrackSession&&) noexcept;
    ByteTrackSession(const ByteTrackSession&) = delete;
    ByteTrackSession& operator=(const ByteTrackSession&) = delete;

    TrackFrame update(const DetectionFrame& frame);
    void reset() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
```

Validate all thresholds and limits in the constructor. Use PImpl so `tracker.h` does not leak through the C++ ABI.

- [ ] **Step 4: Implement class grouping, borrowed/copy boundaries, and IDs**

`ByteTrackSession::update()` copies public detections into per-class `detection_t` scratch arrays, so no caller pointer survives the call. Maintain `std::map<std::int32_t, TrackerOwner>` for deterministic class traversal and stable ownership. For each returned local ID:

```cpp
std::uint64_t global_track_id(std::int32_t class_id, int local_id) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(class_id)) << 32U)
         | static_cast<std::uint32_t>(local_id);
}
```

Pre-size `TrackFrame::detections` to the input count and fill by the `_ex` detection index. Call empty updates for existing classes absent from the current frame. Convert every non-OK C status to `YoloError` without returning a partial frame.

- [ ] **Step 5: Run focused no-GPU validation**

```powershell
cmake --build --preset win-yolo-tracking-dev-user --target test_yolo_tracking
ctest --preset win-yolo-tracking-dev-user -R test_yolo_tracking
```

Expected: all session tests pass; no TensorRT, CUDA, or OpenCV package is searched.

- [ ] **Step 6: Commit Task 2**

```powershell
git add CMakeOptions.cmake CMakeLists.txt CMakeUserPresets.json tensorrt_yolo
git commit -m "feat(yolo): add class-isolated ByteTrack sessions"
```

---

### Task 3: Add a TensorRT-independent engine contract validator

**Files:**
- Create: `tensorrt_yolo/src/engine_contract.hpp`
- Create: `tensorrt_yolo/src/engine_contract.cpp`
- Create: `tensorrt_yolo/tests/test_tensorrt_contract.cpp`
- Modify: `tensorrt_yolo/CMakeLists.txt`

**Interfaces:**
- Consumes: `YoloError`, `YoloErrorCode`, and checked-size helpers from Task 2.
- Produces: `EngineMetadata`, `TensorDesc`, `ValidatedContract`, and `validate_engine_contract()` for Task 4.

- [ ] **Step 1: Write failing contract tests**

```cpp
spec("TensorRT YOLO engine contract") {
    it("accepts named NCHW EfficientNMS tensors") {
        EngineMetadata metadata = make_valid_fp32_metadata();
        ValidatedContract contract = validate_engine_contract(metadata, default_names(), limits());
        check_equal(contract.max_batch, (std::int64_t)4);
        check_equal(contract.max_detections, (std::int64_t)300);
    }

    it("rejects binding-order lookalikes with wrong names") {
        EngineMetadata metadata = make_valid_fp32_metadata();
        metadata.tensors[1].name = "output0";
        check_throws_with(validate_engine_contract(metadata, default_names(), limits()),
                          "num_dets");
    }

    it("rejects overflowing maximum buffer bytes") {
        EngineMetadata metadata = make_valid_fp32_metadata();
        metadata.tensors[0].max_shape = {INT64_MAX, 3, INT64_MAX, 2};
        check_throws_as(validate_engine_contract(metadata, default_names(), limits()),
                        YoloError);
    }
}
```

- [ ] **Step 2: Build and verify undefined contract symbols fail**

```powershell
cmake --build --preset win-yolo-tracking-dev-user --target test_tensorrt_contract
```

Expected: compilation or linkage fails because the contract API is missing.

- [ ] **Step 3: Implement exact metadata validation**

Use TensorRT-independent enums and exact validation inputs:

```cpp
enum class TensorDataType { Float32, Float16, Int32 };
enum class TensorIoMode { Input, Output };

struct TensorDesc {
    std::string name;
    TensorIoMode mode;
    TensorDataType data_type;
    std::vector<std::int64_t> min_shape;
    std::vector<std::int64_t> opt_shape;
    std::vector<std::int64_t> max_shape;
};

struct ContractNames {
    std::string images = "images";
    std::string num_dets = "num_dets";
    std::string boxes = "boxes";
    std::string scores = "scores";
    std::string labels = "labels";
};

struct ContractLimits {
    std::size_t max_batch = 16;
    std::size_t max_detections = 1000;
    std::size_t max_input_bytes = 256U * 1024U * 1024U;
    std::size_t max_output_bytes = 16U * 1024U * 1024U;
};

ValidatedContract validate_engine_contract(
    const EngineMetadata& metadata,
    const ContractNames& names,
    const ContractLimits& limits
);
```

Require one rank-4 NCHW input with channel 3, INT32 `num_dets`/`labels`, same-type FP16-or-FP32 `boxes`/`scores`, box width 4, and consistent batch/detection dimensions. Use `checked_mul_size()` for every element and byte count. Reject missing, duplicate, extra required names, negative fixed dimensions, profile-order violations, and configured limits.

- [ ] **Step 4: Run contract tests and tracking regression**

```powershell
cmake --build --preset win-yolo-tracking-dev-user --target test_tensorrt_contract test_yolo_tracking
ctest --preset win-yolo-tracking-dev-user -R "test_tensorrt_contract|test_yolo_tracking"
```

Expected: both targets pass without finding TensorRT.

- [ ] **Step 5: Commit Task 3**

```powershell
git add tensorrt_yolo/src/engine_contract.hpp tensorrt_yolo/src/engine_contract.cpp tensorrt_yolo/tests/test_tensorrt_contract.cpp tensorrt_yolo/CMakeLists.txt
git commit -m "feat(yolo): validate TensorRT engine contracts"
```

---

### Task 4: Add exact SDK discovery and TensorRT/CUDA resource ownership

**Files:**
- Modify: `CMakeOptions.cmake`
- Modify: `CMakeLists.txt`
- Modify: `CMakeUserPresets.json`
- Create: `cmake/FindTensorRT.cmake`
- Create: `tensorrt_yolo/include/kfcore/yolo/tensorrt.hpp`
- Create: `tensorrt_yolo/src/tensorrt_raii.hpp`
- Create: `tensorrt_yolo/src/cuda_buffer.hpp`
- Create: `tensorrt_yolo/src/cuda_buffer.cpp`
- Create: `tensorrt_yolo/src/engine.cpp`
- Modify: `tensorrt_yolo/CMakeLists.txt`

**Interfaces:**
- Consumes: contract validator from Task 3.
- Produces: `Engine::load()`, `Engine::create_detector()`, immutable engine state, and bounded buffer owners for Task 5.

- [ ] **Step 1: Add CMake dependency-boundary tests**

Add `KFCORE_BUILD_TENSORRT_YOLO` default `OFF` and an explicit configure check:

```cmake
if(KFCORE_BUILD_TENSORRT_YOLO AND NOT KFCORE_BUILD_YOLO_TRACKING)
  message(FATAL_ERROR
    "KFCORE_BUILD_TENSORRT_YOLO requires KFCORE_BUILD_YOLO_TRACKING=ON")
endif()
```

Create `win-yolo-release-user` configure/build/test presets plus `install-win-yolo-release-user`. The configure preset defines:

```json
"environment": {
  "TENSORRT_ROOT": "$env{PKG_ROOT}/tensorrt",
  "OPENCV_LITE_ROOT": "$env{PKG_ROOT}/opencv-lite"
},
"cacheVariables": {
  "KFCORE_BUILD_YOLO_TRACKING": "ON",
  "KFCORE_BUILD_TENSORRT_YOLO": "ON",
  "KFCORE_BUILD_YOLO_OPENCV": "ON"
}
```

Run `cmake --fresh --preset win-yolo-release-user`. In the current machine state, expected result is a clear configure failure naming the missing `TENSORRT_ROOT` directory, not selection of a system SDK.

- [ ] **Step 2: Implement exact `FindTensorRT.cmake`**

Require environment `TENSORRT_ROOT`, normalize it, and use `find_path`/`find_library` with `NO_DEFAULT_PATH` for `NvInfer.h`, `nvinfer`, and `nvinfer_plugin`. Parse `NV_TENSORRT_MAJOR/MINOR/PATCH` from `NvInferVersion.h`; accept major 10 or 11 only. Create imported targets `TensorRT::nvinfer` and `TensorRT::nvinfer_plugin`. Verify every resolved path remains under the normalized root before returning success.

- [ ] **Step 3: Write a compile-time ownership smoke test**

Add a test source under `tests/test_tensorrt_integration.cpp` guarded by the integration option:

```cpp
it("rejects a missing engine file with a typed error") {
    EngineOptions options;
    check_throws_as(Engine::load("Z:/kfcore/missing.engine", options), YoloError);
}
```

Expected before implementation: compile failure because `Engine` is undefined.

- [ ] **Step 4: Implement RAII engine and buffer owners**

Define the public API exactly:

```cpp
struct TensorNames {
    std::string images = "images";
    std::string num_dets = "num_dets";
    std::string boxes = "boxes";
    std::string scores = "scores";
    std::string labels = "labels";
};

struct EngineOptions {
    int device_id = 0;
    TensorNames tensor_names;
    std::size_t max_batch = 16;
    std::size_t max_detections = 1000;
    std::size_t max_input_bytes = 256U * 1024U * 1024U;
    std::size_t max_output_bytes = 16U * 1024U * 1024U;
};

struct DetectorOptions {
    std::optional<std::array<std::int32_t, 2>> input_size;
    std::array<float, 3> mean{0.0f, 0.0f, 0.0f};
    std::array<float, 3> stddev{1.0f, 1.0f, 1.0f};
    float border_value = 114.0f;
};

class TensorRtDetector;

class Engine final {
public:
    static std::shared_ptr<const Engine> load(
        const std::filesystem::path& engine_path,
        const EngineOptions& options = {}
    );
    std::unique_ptr<TensorRtDetector> create_detector(
        const DetectorOptions& options = {}
    ) const;
};

class TensorRtDetector final {
public:
    ~TensorRtDetector();
    DetectionFrame detect(const ImageView& image);
    std::vector<DetectionFrame> detect_batch(
        const std::vector<ImageView>& images
    );
};
```

Implement `Engine::load()` so construction order is logger -> runtime -> engine -> metadata validation. Store runtime before engine in the owning struct so reverse member destruction releases engine first. Use `std::unique_ptr<T, TensorRtDeleter<T>>`; do not call removed `destroy()` APIs.

`CudaBuffer` and `PinnedHostBuffer` each own exactly one allocation, are move-only, and expose a temporary pointer view valid until resize/destruction. `reserve(bytes, hard_limit)` checks the limit and allocates replacement storage before releasing the old buffer, so failed growth preserves the old valid allocation.

Call `cudaSetDevice(options.device_id)` at detector-creation and inference boundaries. Convert CUDA/TensorRT failures to typed errors with operation and stage.

- [ ] **Step 5: Verify available paths or record the external block**

If `$env:PKG_ROOT/tensorrt` exists:

```powershell
cmake --fresh --preset win-yolo-release-user
cmake --build --preset win-yolo-release-user --target tensorrt_yolo
```

Expected: target compiles against TensorRT 10.x or 11.x. If the SDK remains absent, attach the configure error to the GitHub issue and do not mark the SDK compile acceptance criterion complete.

- [ ] **Step 6: Commit Task 4**

```powershell
git add CMakeOptions.cmake CMakeLists.txt CMakeUserPresets.json cmake/FindTensorRT.cmake tensorrt_yolo
git commit -m "feat(yolo): own TensorRT engine and CUDA resources"
```

---

### Task 5: Implement CUDA preprocessing, inference, and detection postprocessing

**Files:**
- Create: `tensorrt_yolo/src/letterbox.hpp`
- Create: `tensorrt_yolo/src/letterbox.cu`
- Create: `tensorrt_yolo/src/detector.cpp`
- Modify: `tensorrt_yolo/src/engine.cpp`
- Modify: `tensorrt_yolo/CMakeLists.txt`
- Modify: `tensorrt_yolo/tests/test_tensorrt_integration.cpp`

**Interfaces:**
- Consumes: public `ImageView`, validated engine contract, shared `Engine`, and bounded buffers.
- Produces: `TensorRtDetector::detect()` and `detect_batch()` for Task 6 and the example.

- [ ] **Step 1: Write GPU integration behaviors before implementation**

Use only the trusted path from `KFCORE_TENSORRT_TEST_ENGINE`:

```cpp
it("keeps one result per non-square batch image") {
    auto engine = Engine::load(test_engine_path(), EngineOptions{});
    auto detector = engine->create_detector(DetectorOptions{});
    OwnedRgbImage wide(960, 320);
    OwnedRgbImage tall(320, 960);

    auto results = detector->detect_batch({wide.view(), tall.view()});

    check_equal(results.size(), (size_t)2);
    check_equal(results[0].image_width, 960);
    check_equal(results[0].image_height, 320);
    check_equal(results[1].image_width, 320);
    check_equal(results[1].image_height, 960);
    for (const auto& result : results) {
        for (const auto& detection : result.detections) {
            check(detection.box.left >= 0.0f);
            check(detection.box.top >= 0.0f);
            check(detection.box.right <= result.image_width);
            check(detection.box.bottom <= result.image_height);
        }
    }
}
```

Define `OwnedRgbImage` in the test as a `std::vector<std::uint8_t>` owner whose `view()` returns a Host/Rgb8 `ImageView` with `row_stride = width * 3`. Define `test_engine_path()` to read `KFCORE_TENSORRT_TEST_ENGINE`, reject a missing/empty value with `YoloError`, and return `std::filesystem::path`. Also add cases for empty batch rejection, batch above limit, Host/CudaDevice input equivalence, FP16 input engine if supplied, two detectors sharing one engine, and invalid CUDA-device pointers.

- [ ] **Step 2: Implement per-image transform and typed letterbox kernels**

Define:

```cpp
struct LetterboxTransform {
    float scale;
    float pad_x;
    float pad_y;
    std::int32_t source_width;
    std::int32_t source_height;
};
```

Launch one bounded kernel region per image on the detector stream. Support BGR8/RGB8 source and FP16/FP32 NCHW destination. Compute source offsets with checked host-side stride validation; use border value for out-of-image coordinates. Keep one transform per batch element.

- [ ] **Step 3: Implement the detector execution state machine**

The call sequence is fixed:

```text
validate views -> set device -> reserve bounded buffers -> copy/letterbox
-> set input shape -> set all tensor addresses -> enqueueV3
-> copy four outputs to pinned host -> synchronize -> validate counts
-> inverse-transform each image -> return owned results
```

Before enqueue, validate every borrowed view and use `cudaPointerGetAttributes` for CudaDevice input. After enqueue, reject negative or over-limit `num_dets`, non-finite scores/boxes, invalid labels, or vector-size inconsistencies. Clamp only finite coordinates to the source image boundary; do not repair inverted boxes.

- [ ] **Step 4: Run integration tests when the SDK and engine are present**

Set the version-matched trusted engine at the documented fixed test-data path:

```powershell
$env:KFCORE_TENSORRT_TEST_ENGINE = "C:/projects/cpp/external/test-data/yolo11n-efficientnms.engine"
cmake --fresh --preset win-yolo-release-user -DKFCORE_BUILD_TENSORRT_INTEGRATION_TESTS=ON
cmake --build --preset win-yolo-release-user --target test_tensorrt_integration
ctest --preset win-yolo-release-user -R test_tensorrt_integration
```

Expected: all GPU cases pass. If either SDK or engine is missing, the explicit configure failure remains an open GitHub blocker and no GPU success claim is made.

- [ ] **Step 5: Run no-GPU regression tests**

```powershell
cmake --build --preset win-yolo-tracking-dev-user --target test_yolo_tracking test_tensorrt_contract
ctest --preset win-yolo-tracking-dev-user -R "test_yolo_tracking|test_tensorrt_contract"
```

Expected: tracking and contract tests remain independent of CUDA/TensorRT.

- [ ] **Step 6: Commit Task 5**

```powershell
git add tensorrt_yolo/src/letterbox.hpp tensorrt_yolo/src/letterbox.cu tensorrt_yolo/src/detector.cpp tensorrt_yolo/src/engine.cpp tensorrt_yolo/tests/test_tensorrt_integration.cpp tensorrt_yolo/CMakeLists.txt
git commit -m "feat(yolo): run EfficientNMS TensorRT detection"
```

---

### Task 6: Add the optional opencv-lite adapter and image-sequence example

**Files:**
- Create: `cmake/FindOpenCVLite.cmake`
- Create: `tensorrt_yolo/include/kfcore/yolo/opencv.hpp`
- Create: `tensorrt_yolo/src/opencv.cpp`
- Create: `tensorrt_yolo/tests/test_yolo_opencv.cpp`
- Create: `tensorrt_yolo/examples/track_image_sequence.cpp`
- Modify: `tensorrt_yolo/CMakeLists.txt`

**Interfaces:**
- Consumes: `ImageView`, `TrackFrame`, detector, and tracking session.
- Produces: exact OpenCV adapter functions and a runnable image-sequence example.

- [ ] **Step 1: Write failing OpenCV adapter tests**

```cpp
spec("YOLO OpenCV adapter") {
    it("borrows a strided CV_8UC3 ROI without flattening it") {
        cv::Mat parent(20, 30, CV_8UC3, cv::Scalar::all(0));
        cv::Mat roi = parent(cv::Rect(3, 4, 10, 8));
        ImageView view = image_view(roi, PixelFormat::Bgr8);
        check_equal(view.width, 10);
        check_equal(view.height, 8);
        check_equal(view.row_stride, roi.step[0]);
        check_equal(view.data, static_cast<const void*>(roi.data));
    }

    it("rejects unsupported Mat element types") {
        cv::Mat gray(10, 10, CV_8UC1);
        check_throws_as(image_view(gray, PixelFormat::Bgr8), YoloError);
    }
}
```

Add a drawing test that snapshots pixels outside an ROI, draws inside it, and uses `check_equal` on the outside memory to prove no out-of-ROI modification.

- [ ] **Step 2: Implement exact opencv-lite discovery**

Require `OPENCV_LITE_ROOT`; use `NO_DEFAULT_PATH` to find headers and versioned-or-generic `opencv_core`, `opencv_imgproc`, and `opencv_imgcodecs` libraries under that root. Create imported targets `OpenCVLite::core`, `OpenCVLite::imgproc`, and `OpenCVLite::imgcodecs`. Do not search or link dnn, highgui, videoio, or ONNX Runtime.

- [ ] **Step 3: Implement the borrowed view and pure renderer**

Expose:

```cpp
struct DrawOptions {
    bool draw_unconfirmed = false;
    int line_thickness = 2;
    double font_scale = 0.5;
};

ImageView image_view(
    const cv::Mat& image,
    PixelFormat pixel_format = PixelFormat::Bgr8
);

void draw_tracks(
    cv::Mat& image,
    const TrackFrame& tracks,
    const DrawOptions& options = {}
);
```

`image_view()` accepts only non-empty `CV_8UC3`, preserves `step[0]`, and returns `MemoryKind::Host`. Its documentation states the view expires on Mat release, reallocation, or mutation that changes storage.

`draw_tracks()` validates image dimensions against `TrackFrame`, clips finite drawing coordinates, uses deterministic colors derived from `track_id`, and draws only confirmed IDs unless an explicit `draw_unconfirmed` option is true. It owns no tracker state.

- [ ] **Step 4: Implement the image-sequence example**

The example command contract is:

```text
track_image_sequence --engine C:/projects/cpp/external/test-data/yolo11n-efficientnms.engine --images C:/projects/cpp/external/test-data/tracking-images --output C:/projects/cpp/external/test-output/tracking-images
```

It sorts supported image paths, creates one engine/detector/session, processes every image in sequence, writes annotated images, and calls `reset()` only at startup. Invalid directories, empty sequences, decode failures, and output-write failures terminate with a nonzero exit and a concrete message. Do not add video decoding.

- [ ] **Step 5: Build and run OpenCV tests**

```powershell
cmake --fresh --preset win-yolo-release-user
cmake --build --preset win-yolo-release-user --target test_yolo_opencv track_image_sequence
ctest --preset win-yolo-release-user -R test_yolo_opencv
```

Expected: OpenCV tests pass using `C:/projects/cpp/external/pkgs/opencv-lite`; the targets do not link `opencv_dnn4130.lib`, `opencv_highgui4130.lib`, or ONNX Runtime.

- [ ] **Step 6: Commit Task 6**

```powershell
git add cmake/FindOpenCVLite.cmake tensorrt_yolo/include/kfcore/yolo/opencv.hpp tensorrt_yolo/src/opencv.cpp tensorrt_yolo/tests/test_yolo_opencv.cpp tensorrt_yolo/examples/track_image_sequence.cpp tensorrt_yolo/CMakeLists.txt
git commit -m "feat(yolo): add opencv-lite tracking adapter"
```

---

### Task 7: Export packages, document deployment, and run final verification

**Files:**
- Modify: `cmake/KFCoreConfig.cmake.in`
- Modify: `CMakeLists.txt`
- Modify: `CMakeUserPresets.json`
- Modify: `tensorrt_yolo/CMakeLists.txt`
- Modify: `README.md`
- Create: `tensorrt_yolo/README.md`

**Interfaces:**
- Consumes: all Task 1-6 targets.
- Produces: installable `KFCore::yolo_tracking`, `KFCore::tensorrt_yolo`, and `KFCore::yolo_opencv` targets plus reproducible preset commands.

- [ ] **Step 1: Add an installed-consumer smoke project**

Under the build tree, generate a minimal consumer source during CTest configuration:

```cpp
#include <kfcore/yolo/tracking.hpp>

int main() {
    kfcore::yolo::ByteTrackSession session;
    auto result = session.update({640, 480, {}});
    return result.detections.empty() ? 0 : 1;
}
```

Its CMake uses `find_package(KFCore CONFIG REQUIRED)` and links `KFCore::yolo_tracking`. The test must consume the installed prefix, not build-tree include paths.

- [ ] **Step 2: Implement conditional package dependencies and exports**

Export only targets actually built. Pass `EXPORT_SET KFCoreTargets` explicitly to the repository CMake helper for every new installed target. Bake feature booleans into `KFCoreConfig.cmake`; conditionally call `find_dependency(trackers CONFIG)` for tracking, include installed `FindTensorRT.cmake` and call `find_dependency(CUDAToolkit)`/`find_package(TensorRT)` for TensorRT, and include `FindOpenCVLite.cmake` for the adapter. Install public headers under `include/kfcore/yolo`.

Do not copy vcpkg runtime DLLs in CMake. Keep TensorRT/CUDA runtime deployment external. For the opencv-lite example/test runtime, use the preset `PATH` entry `$env{OPENCV_LITE_ROOT}/bin` instead of a post-build copy rule.

- [ ] **Step 3: Document the public contract and operational boundaries**

Document:

- trusted-engine requirement and five tensor names/types;
- input memory ownership and invalidation points;
- one detector per worker and one session per stream;
- class-isolated/global-ID formula and reset behavior;
- all capacity options and rejection behavior;
- tracking-only, full TensorRT, OpenCV, integration-test, and install presets;
- absence of videoio, raw-head decoding, ReID, and runtime fallbacks;
- exact external blocker when TensorRT SDK/test engine is not installed.

- [ ] **Step 4: Run preset and default-build compatibility checks**

From an x64 Visual Studio 2022 Developer Command Prompt:

```powershell
cmake --list-presets
cmake --build --list-presets
ctest --list-presets
cmake --fresh --preset win-dev-user
cmake --build --preset win-dev-user
ctest --preset win-dev-user
```

Expected: the default C-only-compatible profile configures without TensorRT/OpenCV discovery and all existing tests pass.

- [ ] **Step 5: Run tracking-only, install, and optional full checks**

```powershell
cmake --fresh --preset win-yolo-tracking-dev-user
cmake --build --preset win-yolo-tracking-dev-user
ctest --preset win-yolo-tracking-dev-user
cmake --build --preset install-win-yolo-tracking-dev-user
```

Expected: tracker/session/contract and installed-consumer tests pass without TensorRT.

When the external SDK and trusted engine exist:

```powershell
$env:KFCORE_TENSORRT_TEST_ENGINE = "C:/projects/cpp/external/test-data/yolo11n-efficientnms.engine"
cmake --fresh --preset win-yolo-release-user -DKFCORE_BUILD_TENSORRT_INTEGRATION_TESTS=ON
cmake --build --preset win-yolo-release-user
ctest --preset win-yolo-release-user
cmake --build --preset install-win-yolo-release-user
```

Expected: full TensorRT/OpenCV/integration/install validation passes. If prerequisites are absent, record those exact unverified rows; do not close the integration issue.

- [ ] **Step 6: Inspect linkage and source provenance**

Use `dumpbin /dependents` on `test_yolo_opencv.exe` and `track_image_sequence.exe`. Confirm only required OpenCV core/imgproc/imgcodecs DLLs appear. Use:

```powershell
rg.exe -n "TensorRT-YOLO[/\\]|TensorRT-YOLO11[/\\]|opencv_dnn|onnxruntime|enqueueV2|getBinding" CMakeLists.txt CMakeOptions.cmake cmake tensorrt_yolo
```

Expected: no source/build dependency on either reference directory, no forbidden fallback library, and no deprecated TensorRT binding API.

- [ ] **Step 7: Commit Task 7**

```powershell
git add CMakeLists.txt CMakeUserPresets.json cmake/KFCoreConfig.cmake.in tensorrt_yolo/CMakeLists.txt README.md tensorrt_yolo/README.md
git commit -m "docs(yolo): export and verify TensorRT tracking module"
```

---

## Completion Gate

Before claiming the feature complete:

- Task 1-3 and tracking-only install verification must pass locally.
- Default KFCore presets must remain green.
- The two reference directories must remain absent from all build inputs.
- GitHub child issues close only after their listed test commands pass.
- If TensorRT SDK or trusted engine remains unavailable, Tasks 4-5 may be implemented but the epic stays open with GPU compilation/inference explicitly unverified.
- Final review reports every finding or remaining risk with `HIGH`/`MED`/`LOW` and labels evidence as `事实`, `计算`, `推论`, or `常用做法`.
