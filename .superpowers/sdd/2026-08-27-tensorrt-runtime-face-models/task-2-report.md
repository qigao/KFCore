# Task 2 报告：TensorRT engine/executor 后端

## 状态

- 完成通用 `KFCore::tensorrt_runtime` engine 加载、profile 0 metadata 提取与同步 executor。
- 保持现有 `tensorrt_yolo` 源码和行为不变；新实现不依赖 `YoloError`。
- 本任务不运行真实 engine integration；该项按 task brief 明确不要求。

## 文件

- 修改 `tensorrt_runtime/CMakeLists.txt`
- 修改 `tensorrt_runtime/include/kfcore/tensorrt/runtime.hpp`
- 新增 `tensorrt_runtime/src/cuda_buffer.{hpp,cpp}`
- 新增 `tensorrt_runtime/src/engine.{hpp,cpp}`
- 新增 `tensorrt_runtime/src/engine_file.{hpp,cpp}`
- 新增 `tensorrt_runtime/src/executor.{hpp,cpp}`
- 新增 `tensorrt_runtime/src/tensorrt_raii.hpp`
- 新增 `tensorrt_runtime/tests/test_cuda_buffer.cpp`
- 新增 `tensorrt_runtime/tests/test_engine_file.cpp`

未修改 root `CMakeLists.txt`：`tensorrt_runtime/CMakeLists.txt` 可在模块边界完成
`CUDAToolkit`/`TensorRT` 查找与私有链接，不需要扩大根构建逻辑。

## 实现事实

- `Engine::Impl` 持有 TensorRT logger/runtime/engine、不可变 metadata 与 options；
  `Executor::Impl` 通过 `shared_ptr<const Engine::Impl>` 延长 engine 生命周期，并独占
  context、non-blocking CUDA stream、device/pinned staging buffers 与 atomic overlap guard。
- engine 文件读取以初始文件大小为边界，拒绝截短、追加、空文件和超过
  `max_serialized_engine_bytes` 的输入。
- metadata 仅使用 profile 0；拒绝非 DEVICE tensor、shape-inference I/O、TensorRT I/O
  alias、非 linear/packed/vectorized format 和不支持的数据类型。
- 输出 profile 通过临时 context 分别代入 profile 0 min/opt/max 输入解析；仍含非正维度的
  data-dependent/unresolved 输出由既有 validation 拒绝。
- `Executor::run()` 先调用 Task 1 view validation，再检查 caller storage alias；Host view
  经 executor-owned pinned/device staging，CUDA view 直接 bind。输出形状必须与 context
  解析结果完全一致。
- run 返回前同步 stream 并清空 TensorRT tensor address。提交后异常会尽力同步并传播原异常；
  若 TensorRT 拒绝清空地址，则销毁 context，避免保留 borrowed pointer。
- device/pinned buffer 的增长先分配 replacement；分配或 hard-limit 失败保留旧 allocation。

## TDD RED

命令：

```text
cmake --build --preset win-release-user --target
  test_tensorrt_runtime_engine_file test_tensorrt_runtime_cuda_buffer
```

关键输出（预期失败）：

```text
test_engine_file.cpp(1): fatal error C1083: Cannot open include file: 'engine_file.hpp'
test_cuda_buffer.cpp(1): fatal error C1083: Cannot open include file: 'cuda_buffer.hpp'
ninja: build stopped: subcommand failed.
```

## TDD GREEN

命令：

```text
cmake --build --preset win-release-user --target
  kfcore_tensorrt_runtime test_tensorrt_runtime_contract
  test_tensorrt_runtime_engine_file test_tensorrt_runtime_cuda_buffer
ctest --preset win-release-user -R "test_tensorrt_runtime" --output-on-failure
```

关键输出：

```text
3/3 tests passed, 0 tests failed
Total Test time (real) = 0.10 sec
```

## 完整验证

环境：VS 2022 Professional DevCmd，`TENSORRT_ROOT=C:\projects\TensorRT-11.2.1.2`，
`win-release-user` preset。

命令：

```text
cmake --build --preset win-release-user
ctest --preset win-release-user --output-on-failure
```

关键输出：

```text
ninja: no work to do.
11/11 tests passed, 0 tests failed
Total Test time (real) = 8.48 sec
```

另执行 `git diff --check`，未发现 whitespace error。

## Commit

本报告与实现位于同一个 Task 2 聚焦提交；最终 commit hash 由 Task 2 最终回复提供。

## 自审

- `HIGH`：未发现。所有 CUDA/TensorRT work-submission 异常路径均在释放/解绑 borrowed
  storage 前尽力同步；checked shape/byte/aggregate limits 继续以 Task 1 validation 为事实源。
- `MED`：未发现。现有 YOLO 文件无修改；TensorRT/CUDA headers 仅出现在私有实现。
- `LOW`：Windows CTest 需要 TensorRT DLL 搜索路径。测试属性从已经由 `FindTensorRT`
  校验的 `TENSORRT_ROOT/bin` 注入 PATH，避免复制 DLL 或硬编码 SDK 版本目录。
- 残余风险：本 task 未以真实 serialized engine/GPU 执行 host/device inference；executor
  数据通路已通过 TensorRT 11.2.1.2 编译链接，但真实 engine 行为留给后续 opt-in integration
  tests 验证。
