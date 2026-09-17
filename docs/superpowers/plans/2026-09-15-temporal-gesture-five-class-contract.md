# Five-Class Temporal Gesture Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 收缩 Temporal Gesture GRU V1 为 `None / SwipeLeft / SwipeRight / Grab / Release` 五类，并用真实 ORT 三帧冒烟验证固定 tensor contract。

**Architecture:** 78 维 reference encoder、2-layer GRU、64 维 hidden state 和四阶段 phase head 保持不变；只收缩 gesture head 与其跨 Python/C++/ModelPackage 的固定契约。静态 `pointer` 特征仍属于输入事实，不再对应时序输出类别。

**Tech Stack:** Python 3、PyTorch、ONNX、ONNX Runtime、C++17、CMake Presets、TinyTest

**Spec:** `docs/superpowers/specs/2026-09-15-temporal-gesture-gru-v1-design.md`

## Global Constraints

- [x] 标签必须紧凑编号为 `0..4`，不保留旧八类 checkpoint/package 的兼容或 fallback。
- [x] `features [1,78]`、`hidden [2,1,64]`、`phase_logits [1,4]` 保持不变。
- [x] 旧八类 checkpoint、ONNX 或数据标签 `5..7` 必须 fail fast。
- [x] `--contract-smoke` 产物必须继续明确标记为未训练，不得用于质量声明。

---

## Task 1: 用测试冻结五类公共契约

**Files:**
- Modify: `vision/core/hand_gesture/tests/test_runtime_contract.cpp`
- Create: `tools/temporal_gesture/test_contract.py`

- [x] 将 C++ 契约测试改为 class count 5，并断言五个紧凑 numeric identity。
- [x] 新增 Python 测试，断言标签表、sequence head `[B,T,5]`、streaming head `[1,5]`，以及标签 5 被拒绝。
- [x] 要求每条 JSONL 记录声明 `kfcore-temporal-gesture-classes/1`，拒绝缺失或旧 label contract，避免旧 `0..4` 子集被静默重解释。
- [x] 运行最小测试，确认旧实现因八类契约而失败。

## Task 2: 收缩训练、评估与导出契约

**Files:**
- Modify: `tools/temporal_gesture/model.py`
- Modify: `tools/temporal_gesture/dataset.py`
- Modify: `tools/temporal_gesture/train.py`
- Modify: `tools/temporal_gesture/evaluate.py`

- [x] 将唯一 Python 类别事实源改为五类，并让数据校验、checkpoint metadata、模型加载和 5x5 confusion matrix 从该事实源派生。
- [x] 运行 Python 契约测试，确认通过。

## Task 3: 同步 C++ runtime 与 interaction 行为

**Files:**
- Modify: `vision/core/hand_gesture/include/kfcore/hand_gesture/types.hpp`
- Modify: `vision/core/hand_gesture/src/runtime.cpp`
- Modify: `hand_interaction/src/hand_interaction.cpp`
- Modify: `hand_interaction/src/gesture_graph.cpp`
- Modify: `hand_interaction/tests/test_hand_interaction.cpp`

- [x] 删除 Wave/Point/Click enum，紧凑重编号剩余类别。
- [x] 让 runtime 的 gesture output shape 从 `kTemporalGestureClassCount` 派生并拒绝旧 `[1,8]`。
- [x] 用 typed load-boundary 单元测试构造旧 `[1,8]` descriptor，验证抛出 `HandGestureError`。
- [x] 删除 Wave/Click learned action 映射及对应行为测试，保留 Swipe/Grab/Release drag 状态与 external Region 输入校验。
- [x] 构建并运行 hand_gesture、hand_interaction 最小相关测试。

## Task 4: 同步活跃文档和训练数据定义

**Files:**
- Modify: `tools/temporal_gesture/README.md`
- Modify: `vision/core/hand_gesture/README.md`
- Modify: `hand_interaction/README.md`
- Modify: `docs/runtime/model-package-v1.md`
- Modify: `docs/superpowers/specs/2026-09-15-temporal-gesture-gru-v1-design.md`
- Modify: `docs/superpowers/plans/2026-09-15-temporal-gesture-gru-v1.md`

- [x] 将标签、tensor shape、confusion matrix、数据采集动作与 interaction 动作更新为五类契约。
- [x] 明确旧八类数据/checkpoint/package 不兼容，训练集只采 SwipeLeft/SwipeRight/Grab/Release 与充分的 None hard negatives。

## Task 5: 生成并执行真实三帧 contract smoke

**Files:**
- Generated locally: `build/gesture/contract-smoke/temporal_gesture.onnx`
- Generated locally: `build/gesture/contract-smoke/model.json`

- [x] 用 `export_onnx.py --contract-smoke` 重新生成未训练 package。
- [x] 独立检查 ONNX 输入输出均为固定 shape，gesture output 为 `[1,5]`。
- [x] 执行 `kfmodel validate`。
- [x] 执行 `kfgesture-smoke`，确认 Runtime -> ORT plugin -> ModelPackage -> recognizer 连续三帧及 hidden state 成功。
- [x] 运行相邻 CTest、`git diff --check` 与残留八类契约检索，记录命令和结果。

## Verification Record

- Python contract tests: `6/6` passed.
- Related CTest: `5/5` passed.
- Full configured CTest: `37/38` passed; pre-existing `test_hand_tracking`
  failed two track-id assertions in the unmodified `vision/core/hand_models` module.
- Exported ONNX opset: `17`; outputs: `[1,5]`, `[1,4]`, `[2,1,64]`.
- Model Package SHA-256: `658a25ac6375a19fc47381a4f694638203eeaf0ff2b5a5acfab315414327bced`.
- `kfmodel validate`: passed.
- `kfgesture-smoke`: ORT CPU route, frames `0..2` passed.
- Stale eight-class contract matches: `0`.
