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

## Fix round 1（review NEEDS CHANGES）

本节取代上文初版自审中“未发现 HIGH/MED”的结论。review 指出的边界均已修复：

- `HIGH`：metadata 将 network declared shape 与仅属于 input 的 optional profile 分离。
  output 不再把 min/opt/max 输入采样结果伪装成逐维 bounds；`run()` 设置实际输入后只接受
  context 解析出的全正 output shape，并以该实际 shape 重新检查 caller shape、bytes、
  `max_input_bytes`/`max_output_bytes` 与 aggregate `max_output_bytes`。
- `HIGH`：CUDA view 在 bind 前经 `cudaPointerGetAttributes` 校验；仅接受
  `cudaMemoryTypeDevice` 且 device id 与 `EngineOptions::device_id` 相等的 pointer。
  host、managed、unregistered、other-device 和查询失败均以 `InvalidTensorView` fail fast。
- `HIGH`：alias query 版本门改为 TensorRT major > 10，或 major == 10 且 minor >= 11；
  只有缺少该 API 的旧版本跳过查询。
- `MED`：新增 checked device-scope guard，保存并在正常/异常路径恢复调用线程原 device。
  `Executor::Impl` 在目标 device scope 内显式销毁 context、staging buffers 和 stream，避免
  成员自动析构越过 device scope。
- `MED`：plugin 初始化改为进程级 `std::call_once`，共享 logger 的输出序列化；初始化抛出
  时遵循 `call_once` 的 retry 语义。每个 Engine 自身 runtime logger 生命周期保持不变。
- `MED`：buffer strong guarantee 明确仅覆盖 allocation/limit 等 pre-commit 失败。replacement
  commit 后若释放旧 allocation 失败，旧 pointer 有效性未知，owner 保持 replacement，并报告
  `committed cleanup` 错误；不回滚到可能悬空的旧 pointer。
- `MED`：新增纯/injectable control seam 单测，不伪造 TensorRT inference；真实 Host/CUDA run
  integration 仍按 ruling 属于 Task 4。

### Fix 文件

- 修改 `tensorrt_runtime/include/kfcore/tensorrt/types.hpp`
- 修改 `tensorrt_runtime/src/{engine,executor,cuda_buffer,tensor_validation}.{hpp,cpp}` 中适用文件
- 修改 `tensorrt_runtime/src/tensorrt_raii.hpp`
- 新增 `tensorrt_runtime/src/cuda_device.{hpp,cpp}`
- 新增 `tensorrt_runtime/src/tensorrt_version.hpp`
- 修改 `tensorrt_runtime/tests/test_{tensor_validation,cuda_buffer}.cpp`
- 新增 `tensorrt_runtime/tests/test_runtime_control.cpp`
- 修改 `tensorrt_runtime/CMakeLists.txt`

### Fix TDD RED

命令：

```text
cmake --build --preset win-release-user --target
  test_tensorrt_runtime_contract test_tensorrt_runtime_cuda_buffer
  test_tensorrt_runtime_control
```

关键输出（预期失败）：

```text
fatal error C1083: Cannot open include file: 'cuda_device.hpp'
test_tensor_validation.cpp: aggregate initialization cannot convert to the old TensorDescriptor
ninja: build stopped: subcommand failed.
```

### Fix TDD GREEN

命令：

```text
cmake --build --preset win-release-user --target
  test_tensorrt_runtime_contract test_tensorrt_runtime_cuda_buffer
  test_tensorrt_runtime_control
ctest --preset win-release-user -R
  "test_tensorrt_runtime_(contract|cuda_buffer|control)$" --output-on-failure
```

关键输出：

```text
3/3 tests passed, 0 tests failed
Total Test time (real) = 0.19 sec
```

### Fix 最终验证

环境：VS 2022 Professional DevCmd，`TENSORRT_ROOT=C:\projects\TensorRT-11.2.1.2`，
`win-release-user` preset。

命令：

```text
cmake --build --preset win-release-user
ctest --preset win-release-user --output-on-failure
```

关键输出：

```text
100% tests passed, 0 tests failed out of 12
Total Test time (real) = 9.58 sec
```

runtime 相关测试为 4/4：contract、engine file、CUDA buffer、runtime control。新增 focused
coverage 包含 device-scope 正常恢复和异常展开、CUDA pointer 类型/device id 判定、TensorRT
alias 版本门，以及 output declaration/实际 shape 和 post-commit cleanup 语义。

### Fix 自审与残余风险

- `HIGH`：未发现未处理项。output runtime shape、CUDA pointer provenance、版本门均有明确
  fail-fast 边界与 focused test。
- `MED`：未发现未处理项。device 状态恢复、显式资源销毁、plugin once 和 buffer commit
  语义均已覆盖实现及测试/结构性检查。
- 残余风险：真实 serialized engine 上的 Host/CUDA inference integration 不在 Task 2 范围，
  依 ruling 留给 Task 4；TensorRT 10.11 之前因 SDK 本身不存在 alias API，无法执行 alias query。

## Fix round 2（scoped re-review）

### 修复

- `Engine::load()` 先构造 `CudaDeviceScope`，再构造 `Engine::Impl`。runtime creation、engine
  deserialization 或 metadata extraction 抛出时，逆序析构保证 TensorRT owners 先在目标
  device 释放，随后才恢复 caller device。
- 新增 `cleanup_on_cuda_device_or_abandon()` noexcept 控制边界。只有目标 device scope 成功
  建立才执行 cleanup；scope 建立失败时绝不调用 CUDA/TensorRT cleanup，而是执行显式
  abandon。
- `Executor::Impl` 使用预分配的 heap shared-lifetime anchor。不可恢复的 device-selection
  failure 下，context、device/pinned staging allocations、stream 和 engine lifetime anchor 均
  relinquish；成员析构因此不会在未确认 device 上再次 cleanup。该路径是刻意的安全泄漏。
- `Engine` 的 impl shared lifetime 同样使用 heap anchor，避免其 noexcept 析构在 device scope
  建立失败时重复出现错误-device cleanup。
- 报告中的不存在选项 `max_tensor_bytes` 已改为真实的
  `max_input_bytes`/`max_output_bytes`。

涉及文件：

- `tensorrt_runtime/include/kfcore/tensorrt/runtime.hpp`
- `tensorrt_runtime/src/cuda_buffer.{hpp,cpp}`
- `tensorrt_runtime/src/cuda_device.{hpp,cpp}`
- `tensorrt_runtime/src/engine.cpp`
- `tensorrt_runtime/src/executor.{hpp,cpp}`
- `tensorrt_runtime/src/tensorrt_raii.hpp`
- `tensorrt_runtime/tests/test_runtime_control.cpp`

### TDD RED

focused test 模拟 `cudaGetDevice` 拒绝 scope establishment，并要求 cleanup 调用数为 0、
abandon 调用数为 1。

命令：

```text
cmake --build --preset win-release-user --target test_tensorrt_runtime_control
```

关键输出（预期失败）：

```text
error C4430/C2065: DeviceCleanupActions/actions 未定义
error C3861: cleanup_on_cuda_device_or_abandon: identifier not found
ninja: build stopped: subcommand failed.
```

### TDD GREEN 与 covering runtime tests

命令：

```text
cmake --build --preset win-release-user --target
  kfcore_tensorrt_runtime test_tensorrt_runtime_contract
  test_tensorrt_runtime_engine_file test_tensorrt_runtime_cuda_buffer
  test_tensorrt_runtime_control
ctest --preset win-release-user -R "test_tensorrt_runtime" --output-on-failure
```

关键输出：

```text
4/4 tests passed, 0 tests failed
Total Test time (real) = 0.17 sec
```

新增 control test 单独 GREEN：

```text
1/1 test passed, 0 tests failed
Total Test time (real) = 0.06 sec
```

### 完整相邻验证

环境：VS 2022 Professional DevCmd，`TENSORRT_ROOT=C:\projects\TensorRT-11.2.1.2`，
`win-release-user` preset。

命令：

```text
cmake --build --preset win-release-user
ctest --preset win-release-user --output-on-failure
```

关键输出：

```text
100% tests passed, 0 tests failed out of 12
Total Test time (real) = 10.43 sec
```

### 自审与残余风险

- `MED`：目标 device scope 成功时显式 cleanup 后恢复；scope 建立失败时所有 destructor-owned
  CUDA/TensorRT handles 均已 relinquish，未留自动 cleanup 路径。
- `LOW`：报告 resource-limit 名称已与实际 `EngineOptions` 对齐。
- 残余风险：device-selection failure 的 abandon 路径按 review ruling 有意泄漏资源，以避免在
  错误 device 上释放；进程恢复资源需要退出。真实 inference integration 仍属于 Task 4。

## Fix round 3（Engine copy API re-review）

### 修复与所有权协议

- 在公开 `Engine` API 显式恢复 copy constructor 与 copy assignment；没有声明新的 move
  overload，rvalue 继续由既有 copy 语义处理。
- 每个 Engine copy 拥有独立 heap lifetime anchor，但 anchor 中的
  `shared_ptr<const Engine::Impl>` retain 同一个 immutable Impl/control block。
- copy construction 先分配新 anchor，再复制 shared owner；分配失败不会改变 source。
- copy assignment 使用 replacement/copy-and-swap：replacement 完整构造后，以 noexcept
  `unique_ptr::swap` 作为唯一提交点。旧 state 转移给 replacement，并由 replacement 的
  `Engine::~Engine()` 进入既有目标 device scope/abandon 协议；不会由裸 unique_ptr
  replacement 在 caller 当前 device 自动 cleanup。
- 新增内部 `shared_lifetime.hpp`，统一 Engine/Executor 的 anchor create/copy/abandon 操作。
  无 engine 单测确认两个 anchor retain 同一个 immutable object，释放一个副本不会提前销毁。

涉及文件：

- `tensorrt_runtime/include/kfcore/tensorrt/runtime.hpp`
- `tensorrt_runtime/src/engine.cpp`
- `tensorrt_runtime/src/executor.cpp`
- 新增 `tensorrt_runtime/src/shared_lifetime.hpp`
- `tensorrt_runtime/tests/test_runtime_control.cpp`

### TDD RED

第一轮先锁定公开 API compile-time contract：

```text
cmake --build --preset win-release-user --target test_tensorrt_runtime_control
```

预期失败：

```text
error C2338: static_assert failed: 'Engine must remain copy constructible'
error C2338: static_assert failed: 'Engine must remain copy assignable'
ninja: build stopped: subcommand failed.
```

第二轮加入无真实 engine 的 ownership helper test，预期失败：

```text
fatal error C1083: Cannot open include file: 'shared_lifetime.hpp'
ninja: build stopped: subcommand failed.
```

### TDD GREEN 与 covering runtime tests

focused 命令：

```text
cmake --build --preset win-release-user --target
  test_tensorrt_runtime_control kfcore_tensorrt_runtime
ctest --preset win-release-user -R "test_tensorrt_runtime_control$" --output-on-failure
```

关键输出：

```text
1/1 test passed, 0 tests failed
Total Test time (real) = 0.05 sec
```

covering runtime 命令：

```text
cmake --build --preset win-release-user --target
  kfcore_tensorrt_runtime test_tensorrt_runtime_contract
  test_tensorrt_runtime_engine_file test_tensorrt_runtime_cuda_buffer
  test_tensorrt_runtime_control
ctest --preset win-release-user -R "test_tensorrt_runtime" --output-on-failure
```

关键输出：

```text
4/4 tests passed, 0 tests failed
Total Test time (real) = 0.27 sec
```

### 完整相邻验证

环境：VS 2022 Professional DevCmd，`TENSORRT_ROOT=C:\projects\TensorRT-11.2.1.2`，
`win-release-user` preset。

命令：

```text
cmake --build --preset win-release-user
ctest --preset win-release-user --output-on-failure
```

关键输出：

```text
100% tests passed, 0 tests failed out of 12
Total Test time (real) = 12.31 sec
```

### 自审与残余风险

- `MED`：`std::is_copy_constructible_v<Engine>` 与
  `std::is_copy_assignable_v<Engine>` 均由 static_assert 锁定；copy assignment 的 old-state
  last-owner release 只发生在 replacement Engine 的 device-safe 析构中。
- `LOW`：未新增不同的 move overload 或 move-only state transition。
- 残余风险：Engine copy 需要为 heap lifetime anchor 分配内存；allocation failure 向调用方
  传播且不改变 source/assignment target。真实 inference integration 仍属于 Task 4。
