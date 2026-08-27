# TensorRT Compact NMS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让既有 `TensorRtDetector` 在保持 EfficientNMS 行为的同时支持 `[B,N,6]` Compact NMS engine。

**Architecture:** 将 engine 的输出事实建模为有限的内部契约类型，加载时严格选择 EfficientNMS 或 Compact NMS。预处理和 TensorRT enqueue 骨架保持共享，输出缓冲绑定与纯 CPU 解码按已验证契约分派。

**Tech Stack:** C++17、CUDA 12.8、TensorRT 11.2、KFCore ImageProcessor、TurboUtils TinyTest、CMake Presets。

**Spec:** `docs/superpowers/specs/2026-08-27-tensorrt-compact-nms-design.md`

## Global Constraints

- 不支持 raw YOLO head，不增加 CPU/GPU NMS 或推理 fallback。
- Compact NMS 必须精确为 `[B,N,6]`，默认输出名为 `output0`。
- 现有 EfficientNMS 五输出行为和公开检测调用保持不变。
- 所有 shape/元素/字节计算必须检查溢出并服从现有资源上限。
- 每个生产行为先写 TinyTest 并观察预期失败，再写最小实现。

---

### Task 1: Compact NMS engine contract

**Files:**
- Modify: `tensorrt_yolo/include/kfcore/yolo/tensorrt.hpp`
- Modify: `tensorrt_yolo/src/engine_contract.hpp`
- Modify: `tensorrt_yolo/src/engine_contract.cpp`
- Modify: `tensorrt_yolo/src/engine.cpp`
- Test: `tensorrt_yolo/tests/test_tensorrt_contract.cpp`

**Interfaces:**
- Consumes: `EngineMetadata`, `ContractLimits`, existing EfficientNMS validation.
- Produces: `TensorNames::detections`, `DetectionOutputLayout`, validated Compact NMS tensor metadata.

- [x] **Step 1: Write failing contract tests**

Add a literal `images + output0` fixture with output shapes `{1,300,6}`, `{2,300,6}`, `{4,300,6}` and assert the returned layout, capacity, dtype, batch and detection count. Add focused rejection cases for wrong name, rank, final width, batch profile, dtype, layout and limits.

- [x] **Step 2: Run the focused test and verify RED**

Run:

```powershell
cmake --build --preset win-yolo-release-user --target test_tensorrt_contract
build/Msvc-Release/bin/test_tensorrt_contract.exe --filter "Compact NMS"
```

Expected: compilation or assertion failure because the Compact NMS contract types and behavior do not exist.

- [x] **Step 3: Implement the minimal validated contract**

Add the default `detections = "output0"` public name. Represent the two internal output forms with a finite type-safe contract, validate an exact two- or five-tensor set, and remove the metadata extractor's unconditional five-tensor rejection.

- [x] **Step 4: Run contract tests and verify GREEN**

Run the focused filter, then the complete `test_tensorrt_contract` executable. Both must exit zero.

### Task 2: Compact NMS decoder

**Files:**
- Modify: `tensorrt_yolo/src/detector_helpers.hpp`
- Modify: `tensorrt_yolo/src/detector_helpers.cpp`
- Test: `tensorrt_yolo/tests/test_tensorrt_detection_helpers.cpp`

**Interfaces:**
- Consumes: `ImageView`, `LetterboxTransform`, `TensorDataType`.
- Produces: `CompactNmsOutputView`, bounded buffer layout, `decode_compact_nms(...)`.

- [x] **Step 1: Write failing decoder tests**

Use hand-derived rows such as `{0, 160, 640, 480, 0.75, 3}` plus a zero padding row. Assert original-coordinate restoration, padding removal and order. Add separate FP16 and malformed-input cases for non-finite values, score outside `[0,1]`, fractional/negative class, inverted/collapsed box and wrong element count.

- [x] **Step 2: Run the focused test and verify RED**

Run:

```powershell
cmake --build --preset win-yolo-release-user --target test_tensorrt_detection_helpers
build/Msvc-Release/bin/test_tensorrt_detection_helpers.exe --filter "compact NMS"
```

Expected: compilation failure because `CompactNmsOutputView` and `decode_compact_nms` do not exist.

- [x] **Step 3: Implement bounded layout and decoder**

Reuse the existing checked multiplication, FP16 conversion and letterbox validation primitives. Skip only exact zero-score padding; reject every malformed positive row rather than silently dropping it.

- [x] **Step 4: Run helper tests and verify GREEN**

Run the focused filter, then the complete detection-helper executable. Both must exit zero.

### Task 3: TensorRT buffer binding and real engine verification

**Files:**
- Modify: `tensorrt_yolo/src/tensorrt_raii.hpp`
- Modify: `tensorrt_yolo/src/engine.cpp`
- Modify: `tensorrt_yolo/src/detector.cpp`
- Modify: `tensorrt_yolo/README.md`
- Modify: `docs/superpowers/specs/2026-08-25-tensorrt-yolo-bytetrack-design.md`

**Interfaces:**
- Consumes: validated output contract and decoder from Tasks 1-2.
- Produces: end-to-end `TensorRtDetector::detect[_batch]()` support for Compact NMS with no public call-site changes.

- [x] **Step 1: Configure a real-engine RED check**

Run the current `track_image_sequence` with local `yolov12n-face.engine` and the repository image directory. Preserve the observed `EngineContractMismatch` as the pre-implementation failure evidence.

- [x] **Step 2: Implement contract-specific allocation, binding and download**

Allocate only the buffers belonging to the validated output form. Bind `output0` for Compact NMS, download its exact actual-batch byte count, synchronize once, and call `decode_compact_nms`. Leave the EfficientNMS branch semantically unchanged.

- [x] **Step 3: Run focused unit and real-engine GREEN checks**

Build the affected targets, run both TinyTest executables, then execute `track_image_sequence` against the local engine and repository images. Expected: exit zero and one output image per input image.

- [x] **Step 4: Update contract documentation**

Document both accepted output forms, tensor names, padding semantics, non-goals, compatibility and the exact local verification boundary. Remove statements that claim only five-output engines are accepted.

- [x] **Step 5: Run complete verification**

Run:

```powershell
cmake --build --preset win-yolo-release-user
ctest --preset win-yolo-release-user
```

Then rerun the real Compact NMS image sequence and inspect `git diff --check` plus `git status --short`.
