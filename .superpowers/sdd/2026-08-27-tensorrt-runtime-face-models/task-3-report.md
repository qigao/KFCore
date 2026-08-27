# Task 3 报告：严格 TensorRT face model adapters

## 状态

- 新增 `KFCore::face_models`，提供 `TensorRtFace68`、`TensorRtArcFace`、
  `TensorRtAgeGender` 三个严格 adapter。
- adapter 只接受调用方已经准备好的 NCHW FP32 `TensorView`（Host 或 CUDA），不保留借用
  指针，不实现 crop、affine、alignment 或图像归一化。
- `KFCORE_BUILD_FACE_MODELS` 默认 `OFF`；启用时要求
  `KFCORE_BUILD_TENSORRT_RUNTIME=ON`，并公开链接 `KFCore::tensorrt_runtime`。
- 本 task 不运行真实 serialized engine integration；该范围按 brief 明确留给后续 task。

## 合同证据

只读检查 `C:\projects\cpp\KFCore\yolo-models` 中三个 ONNX graph 得到：

- `2dfan4.onnx`：`input` FP32 `[1,3,256,256]`，`landmarks_xyscore`
  FP32 `[1,68,3]`，`heatmaps` FP32 `[1,68,64,64]`。
- `arcface_w600k_r50.onnx`：`input.1` FP32 `[N,3,112,112]`，`683`
  FP32 `[1,512]`。
- `age-gender.onnx`：`pixel_values` FP32 `[N,3,224,224]`，`logits`
  FP32 `[N,2]`。

只读参考 faceswap 的 `Face68Landmarks`、`FaceEmbedding`、`AgeGenderPredictor` 后，只采用
模型 I/O 与 Face68 64→256 坐标尺度合同；未复制该仓源码。Age/gender 输出保持两个 raw
logits，不采用其派生语义或 fallback。

## 文件

- 修改 `CMakeOptions.cmake`
- 修改根 `CMakeLists.txt`
- 新增 `face_models/CMakeLists.txt`
- 新增公共头：
  - `face_models/include/kfcore/face_models/error.hpp`
  - `face_models/include/kfcore/face_models/types.hpp`
  - `face_models/include/kfcore/face_models/tensorrt.hpp`
- 新增实现：
  - `face_models/src/error.cpp`
  - `face_models/src/contracts.{hpp,cpp}`
  - `face_models/src/face68.cpp`
  - `face_models/src/arcface.cpp`
  - `face_models/src/age_gender.cpp`
- 新增测试：
  - `face_models/tests/test_contracts.cpp`
  - `face_models/tests/test_results.cpp`

## 实现事实

- construction-time pure contract validation 按 exact name 绑定，不猜别名；拒绝 missing、
  duplicate、extra、错误 mode/type/rank/fixed dimension。Face68 只允许 landmarks 加可选 exact
  heatmaps，其余额外 I/O 全部拒绝。
- batch 只允许 declared positive fixed value 或 `-1`。adapter 可用 batch 是 input profile 0、
  output fixed declarations 与 `options.max_batch` 的交集；固定声明互相矛盾或落在 profile 外时
  fail fast。
- checked element/byte arithmetic 在构造期计算最大 output storage；landmarks、raw embedding、
  raw logits 及可选 heatmaps host buffer一次性分配，并受 runtime `max_output_bytes` hard limit
  约束。
- adapter-owned input/output view vectors、names、shapes 与 output pointers 也在构造期建立；
  infer steady-state 仅更新 batch、借用 data/byte size/memory kind。返回结果 vector 仍按实际
  batch 构造并直接从复用 host buffer 复制，这是公开 owned-return API 的唯一 adapter 侧每次
  调用分配。
- 每个 adapter 持有 shared immutable `Engine`、unique mutable `Executor` 与 adapter 级
  atomic overlap guard；同一实例重叠调用以 `InvalidArgument` 拒绝，不会在进入 executor 前
  产生 view data race。
- `Face68Result` 是 68 个 `(x,y,score)` fixed array，x/y 乘以 `256/64` 后处于 256-pixel
  模型坐标系；ArcFace 512 值与 age/gender 两值均原样复制，不做 L2 normalization、sigmoid、
  标签命名或排序猜测。
- runtime `TensorRtError` 在 model boundary 转为带 model/stage 上下文的
  `FaceModelError`；model contract、invalid caller view、resource limit 与 runtime failure
  保持独立错误码。
- option names/max batch/output limit 在 engine 文件读取和 GPU/runtime 构造之前验证，避免
  外部失败掩盖 adapter 配置错误。

## TDD RED

### 初始 API/模块 RED

命令：

```text
cmake --fresh --preset win-release-user
  -DKFCORE_BUILD_TENSORRT_RUNTIME=ON
  -DKFCORE_BUILD_FACE_MODELS=ON
```

关键预期失败：

```text
Cannot find source file: src/age_gender.cpp
No SOURCES given to target: kfcore_face_models
CMake Generate step failed.
```

### caller-view 错误域 RED

命令：

```text
cmake --build --preset win-release-user --target test_face_model_contracts
ctest --preset win-release-user -R "test_face_model_contracts" --output-on-failure
```

关键预期失败：

```text
classifies a non-positive prepared batch as an invalid caller view
Check failed: error.code() == code
8 passed, 1 failed
```

该 RED 证明旧路径把 batch=0 错分为 model contract；修复后由 input-view boundary 报
`InvalidTensorView`。

### pre-load option validation RED

命令：

```text
cmake --build --preset win-release-user --target test_face_model_contracts
```

关键预期失败：

```text
LNK2019: unresolved external symbol validate_face68_options
LNK2019: unresolved external symbol validate_arcface_options
LNK2019: unresolved external symbol validate_age_gender_options
LNK1120: 3 unresolved externals
```

## TDD GREEN

命令：

```text
cmake --build --preset win-release-user --target
  test_face_model_contracts test_face_model_results
ctest --preset win-release-user -R "test_face_model_" --output-on-failure
```

最终关键输出：

```text
2/2 tests passed, 0 tests failed
Total Test time (real) = 0.13 sec
```

focused coverage 包含 exact/default/overridden names、missing/duplicate/extra I/O、FP32、rank、
固定维度、profile 与 dynamic/fixed batch、checked/bounded output capacity、prepared view、
fixed-array result sizes、Face68 坐标尺度、raw embedding/logits 和 decode size mismatch。

## 构建依赖 fail-fast 验证

命令：

```text
cmake --preset win-release-user
  -DKFCORE_BUILD_TENSORRT_RUNTIME=OFF
  -DKFCORE_BUILD_FACE_MODELS=ON
```

预期输出与 exit code：

```text
KFCORE_BUILD_FACE_MODELS requires KFCORE_BUILD_TENSORRT_RUNTIME=ON
exit code 1
```

随后以两个选项均 `ON` 的 fresh configure 恢复并完成最终验证。

## 最终验证

环境：VS 2022 Professional DevCmd，`TENSORRT_ROOT=C:\projects\TensorRT-11.2.1.2`，
`win-release-user` preset。

命令：

```text
cmake --fresh --preset win-release-user
  -DKFCORE_BUILD_TENSORRT_RUNTIME=ON
  -DKFCORE_BUILD_FACE_MODELS=ON
cmake --build --preset win-release-user
ctest --preset win-release-user -R "test_face_model_" --output-on-failure
ctest --preset win-release-user -R "test_tensorrt_runtime" --output-on-failure
ctest --preset win-release-user --output-on-failure
```

关键输出：

```text
face models: 2/2 passed, 0 failed (0.13 sec)
TensorRT runtime adjacent: 4/4 passed, 0 failed (0.14 sec)
full configured CTest: 14/14 passed, 0 failed (最终复跑 9.89 sec；fresh configure 轮为 10.10 sec)
```

另执行 `git diff --check` 与禁止占位符检索；未发现 whitespace error、TODO/FIXME/HACK、
focused-test marker 或占位实现。

## Commit

本报告与实现位于同一个 Task 3 聚焦提交；最终 commit hash 由 Task 3 最终回复提供。

## 自审

- `HIGH`：未发现。事实：所有 tensor 名称、mode、type、rank、fixed dimension、batch/profile、
  capacity 与额外 I/O 均在执行前验证；invalid view 不会进入 runtime；无 fallback 或隐式
  semantic decoding。
- `MED`：已修复两项。事实：non-positive prepared batch 现归入 `InvalidTensorView`；adapter
  options 现于 engine I/O/GPU 之前 fail fast。事实：adapter 自身的 output/view storage 在构造
  期有界分配，重叠调用由共享 guard 在任何 mutable view 更新前拒绝。
- `LOW`：Windows 新测试需加载公开依赖链中的 TensorRT DLL；测试属性复用 runtime 的
  `$ENV{TENSORRT_ROOT}/bin` PATH 注入，不复制 DLL、不硬编码 SDK 内部版本目录。
- 残余风险（事实）：本 task 没有 trusted serialized face engine integration 输入，因此未执行
  真实 Host/CUDA inference，也未验证 TensorRT builder 对原始 ONNX symbolic/fixed output batch
  的最终 engine declaration。真实 engine/golden output 属于后续 opt-in integration task。
- 残余成本（计算）：每次 infer 返回 `batch * sizeof(fixed-result)` 的 owned vector；Face68、
  ArcFace、AgeGender 分别为每 item `68*3*4=816` bytes、`512*4=2048` bytes、`2*4=8`
  bytes（不含 vector allocator 元数据）。这是公开按值返回合同允许的成本，中间 decoded
  storage 未另行分配。
