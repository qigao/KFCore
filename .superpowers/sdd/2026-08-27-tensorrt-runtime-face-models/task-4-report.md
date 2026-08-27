# Task 4 报告：真实 TensorRT engine integration tests

## 状态

- 新增独立 `KFCORE_BUILD_TENSORRT_RUNTIME_INTEGRATION_TESTS`，默认 `OFF`；启用时要求
  `KFCORE_BUILD_TENSORRT_RUNTIME=ON`。
- 新增 ArcFace、AgeGender、Face68 三条 `CACHE FILEPATH`，首次 configure 默认读取同名环境
  变量；仓库默认值不包含任何本机路径。
- 非空路径在 configure 阶段规范化并验证为存在、非目录且 size > 0 的普通 engine 文件；空路径
  逐模型输出 `STATUS`，且不注册假 passing/skipped test。
- ArcFace 路径注册 generic Host/CUDA runtime CTest；启用 `face_models` 时，各个非空模型路径
  分别注册 adapter CTest。CTest 进程通过 PATH 使用 TensorRT/CUDA runtime，不复制 DLL。
- 未修改 YOLO API、实现、测试命令或既有 engine helper 行为；新 helper 仅追加独立 runtime
  integration 路径验证入口。

## 文件

- 修改 `CMakeOptions.cmake`
- 修改根 `CMakeLists.txt`
- 修改 `cmake/ValidateTensorRtIntegrationEngine.cmake`
- 修改 `tensorrt_runtime/CMakeLists.txt`
- 新增 `tensorrt_runtime/tests/test_tensorrt_runtime_integration.cpp`
- 修改 `face_models/CMakeLists.txt`
- 新增 `face_models/tests/test_face_models_integration.cpp`

## TDD RED

测试源码先于 CMake 注册新增。执行：

```text
cmake --build --preset win-release-user --target
  test_tensorrt_runtime_integration test_face_models_integration
```

关键预期失败：

```text
ninja: error: unknown target 'test_tensorrt_runtime_integration'
exit code 1
```

首次接入 target 后，TinyTest C++ 安装头的 `check_equal` overload selector 对这些调用展开为未
定义 sentinel，两个源文件均稳定编译失败：

```text
error C3861: 'TTEST_EQUAL_SENTINEL__': identifier not found
ninja: build stopped: subcommand failed
```

根因证据：错误只位于新增断言调用；仓库现有 C++ TinyTest 使用 `check(...)`，安装的
`tinytest.hpp` 在 `check_equal(...)` 宏中通过 overload sentinel 分派。最小修复仅将新增测试的
标量/enum equality 改为仓库已有 `check(actual == expected)`，未改 production runtime。

## 配置边界验证

使用 `win-release-user` fresh configure，启用 runtime、face models 与本 option，并将三个路径
全部置空。configure 对 ArcFace、AgeGender、Face68 分别输出：

```text
<CACHE_VARIABLE> is empty; <Model> real-engine integration tests will not be registered
```

随后 CTest 按两个 integration target 前缀查询结果为：

```text
Total Tests: 0
```

使用 build 目录内临时零字节 `task-4-empty.engine` 作为 ArcFace 路径时，configure 以 exit code
1 失败：

```text
KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE must name an existing non-empty regular file
```

该临时 artifact 已验证路径位于 workspace 后删除。关闭 runtime 但启用本 option 时，configure
同样以 exit code 1 失败：

```text
KFCORE_BUILD_TENSORRT_RUNTIME_INTEGRATION_TESTS requires
KFCORE_BUILD_TENSORRT_RUNTIME=ON
```

## GREEN 与真实输出

最终 fresh configure 在命令行提供任务指定的三个本地 trusted engine；报告不记录绝对路径，
避免将机器配置写入提交。engine basename 与 configure 前只读 size 证据：

| 模型 | engine | bytes |
|---|---|---:|
| ArcFace | `arcface_w600k_r50.engine` | 227,372,900 |
| AgeGender | `age-gender.engine` | 345,610,460 |
| Face68 | `2dfan4.engine` | 149,115,764 |

真实 integration 输出摘要：

| 路径 | 固定输出合同 | finite | 实际摘要 |
|---|---|---|---|
| generic ArcFace Host | `[1,512]` Host | 512/512 | min `-1.22298`, max `1.23128`, sum `-4.13207` |
| generic ArcFace direct CUDA | `[1,512]` Host | 512/512 | 同 Host；max abs diff `0` |
| ArcFace adapter | batch 1 × 512 | 512/512 | min `-1.22298`, max `1.23128`, sum `-4.13207` |
| AgeGender adapter | batch 1 × 2 | 2/2 | min `0.815101`, max `56.7196`, sum `57.5347` |
| Face68 adapter | batch 1 × 68 triples | 204/204 | heatmap 已绑定；min `0.0339477`, max `201.759`, sum `18099.7` |

这些值只作为同一次真实执行的 contract/smoke 证据；零输入没有 preprocessing 语义，不能作为
semantic 或 golden accuracy 结论。Face68 测试先验证真实 engine 暴露 FP32
`[N,68,64,64]` heatmap，再通过 adapter 执行；runtime 的 exact-output validation 使成功执行
必须实际绑定并消费该 heatmap output。

generic 测试使用 `cudaMalloc`，再由测试自有 non-blocking CUDA stream 执行
`cudaMemsetAsync`；显式 `cudaStreamSynchronize` 成功后才把 direct `CudaDevice` input 交给
executor。stream 与 device buffer 均由 RAII 管理；在 engine load、executor create、zero-init
同步、Host/CUDA run、device buffer cleanup 以及 executor/engine teardown 后均复验 caller 原
CUDA device 未改变。

## 命令与 CTest

Windows 命令均从 VS 2022 `VsDevCmd.bat` 环境运行，`TENSORRT_ROOT` 与三个 engine path 只在
调用环境/命令行提供：

```text
cmake --fresh --preset win-release-user
  -DKFCORE_BUILD_TENSORRT_RUNTIME=ON
  -DKFCORE_BUILD_FACE_MODELS=ON
  -DKFCORE_BUILD_TENSORRT_RUNTIME_INTEGRATION_TESTS=ON
  -DKFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE=<trusted ArcFace engine>
  -DKFCORE_TENSORRT_RUNTIME_TEST_ENGINE_AGE_GENDER=<trusted AgeGender engine>
  -DKFCORE_TENSORRT_RUNTIME_TEST_ENGINE_FACE68=<trusted Face68 engine>

cmake --build --preset win-release-user --target
  test_tensorrt_runtime_integration test_face_models_integration

ctest --preset win-release-user -V -R
  "^(test_tensorrt_runtime_integration_arcface|test_face_models_integration_(arcface|age_gender|face68))$"

ctest --preset win-release-user -R
  "^(test_tensorrt_runtime_(contract|engine_file|cuda_buffer|control)|test_face_model_(contracts|results))$"

cmake --build --preset win-release-user
ctest --preset win-release-user
```

最终关键结果：

```text
real-engine integration: 4/4 passed, 0 failed (2.25 sec)
adjacent runtime/face tests: 6/6 passed, 0 failed (0.30 sec)
full configured CTest: 18/18 passed, 0 failed (final fresh configure run: 12.70 sec)
```

## Commit

本报告与实现位于同一个 Task 4 聚焦提交，subject 为
`test(tensorrt): add real-engine runtime integration`；最终 hash 由 Task 4 handoff 提供。

## 自审

- `HIGH`：未发现。事实：所有注册测试都使用 configure 已验证的真实 engine，路径缺失不会
  生成 passing/skipped executable；generic 和三个 adapter CTest 均真实执行 TensorRT。
- `MED`：未发现。事实：generic 测试覆盖 Host staging、direct CUDA binding、Host output、
  finite/exact count、数值一致性和 CUDA device restore；Face68 成功路径要求 exact heatmap
  output 被绑定。
- `LOW`：TinyTest 当前 C++ `check_equal` overload macro 无法编译本测试的标量/enum 调用；按
  仓库现有测试模式使用 `check(a == b)`，保留同等失败语义。未修改 TinyTest 依赖。
- 兼容性（事实）：新 option 默认 `OFF`，三个 cache path 默认空；YOLO target/CTest 和既有
  `KFCORE_BUILD_TENSORRT_INTEGRATION_TESTS` 均未改。CodeGraph affected 只识别两份新增测试。
- 残余风险（事实）：测试只验证 prepared zero tensor 的执行合同与 finite 输出，不验证模型
  preprocessing、身份 embedding 质量、age/gender 语义顺序或 landmark accuracy。
- 生产修复（事实）：真实 engine 未暴露 runtime/adapter bug，本 Task 未修改 production C++
  执行逻辑。

## Fix round 1（review NEEDS CHANGES）

### Review 发现与根因

- `MED` direct CUDA input 初始化：旧测试使用 `cudaMemset` 后立即调用拥有独立 non-blocking
  stream 的 executor，没有在两个 stream 之间建立显式完成边界。即使本机输出一致，这个测试
  前置条件也依赖隐式 CUDA 同步语义，不能证明 executor 读取前 zero-init 已完成。
- `MED` face CTest 选择：旧 CMake 用 TinyTest substring `--filter` 选择模型。事实：不存在的
  filter 输出 `0 passed, 0 failed, 3 filtered, 0 assertions`，进程仍 exit 0；因此测试名称改动可使
  CTest 假绿。

### Selector RED

修改前直接执行不存在的 filter：

```text
test_face_models_integration --filter __no_model_matches__
0 passed, 0 failed, 3 filtered, 0 assertions
exit code 0
```

在三个真实 engine 环境变量均提供时，分别令 `KFCORE_FACE_MODEL_TEST_KIND` 缺失或为
`unknown`，旧程序都忽略该 selector、执行全部三模型并成功：

```text
3 passed, 0 failed, 0 filtered, 730 assertions
exit code 0
```

这两个同命令 RED 证明旧程序没有 strict selector boundary。首次实现中还发现当前 TinyTest
C++/MSVC 的多参数 `check(false, format, ...)` 宏分派会记录 passing assertion；改用框架现有
`info(...)` + 单参数 `check(false)` 后，missing/unknown 都成为真实失败。

### 修复

- `ZeroedDeviceBuffer` 内新增测试专用 `InitializationStream`。成功路径依次 checked
  `cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)`、`cudaMalloc`、
  `cudaMemsetAsync`、`cudaStreamSynchronize`、`cudaStreamDestroy`；只有同步和销毁成功后构造
  才返回并允许调用 executor。stream 析构在异常路径做 RAII fallback cleanup。
- device input 构造完成、executor 返回、device buffer 析构及 executor/engine teardown 后继续
  检查 caller CUDA device，新增同步不会污染调用方 device 状态。
- face integration executable 现在只有一个 TinyTest `it`。它精确解析
  `KFCORE_FACE_MODEL_TEST_KIND=arcface|age_gender|face68` 并只 dispatch 一个 adapter；missing
  或 unknown 通过唯一 test 内的 `check(false)` fail fast。
- 三条 face CTest 不再传 `--filter`，而是分别注入 exact selector 与唯一对应 engine path。
  selector 合法但 engine 环境缺失仍由 `required_engine_path` 真实失败。

### Negative selector GREEN

最新 executable 对 unknown selector：

```text
with info: KFCORE_FACE_MODEL_TEST_KIND has unsupported value: unknown
0 passed, 1 failed, 0 filtered, 1 assertion
exit code 1
```

最新 executable 对 missing selector：

```text
with info: KFCORE_FACE_MODEL_TEST_KIND must be arcface, age_gender, or face68
0 passed, 1 failed, 0 filtered, 1 assertion
exit code 1
```

负向证据直接使用 executable exit code；未使用 CTest `PASS_REGULAR_EXPRESSION` 掩盖零执行。

### Fix GREEN 与回归

命令：

```text
cmake --fresh --preset win-release-user
  -DKFCORE_BUILD_TENSORRT_RUNTIME=ON
  -DKFCORE_BUILD_FACE_MODELS=ON
  -DKFCORE_BUILD_TENSORRT_RUNTIME_INTEGRATION_TESTS=ON
  -DKFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE=<trusted ArcFace engine>
  -DKFCORE_TENSORRT_RUNTIME_TEST_ENGINE_AGE_GENDER=<trusted AgeGender engine>
  -DKFCORE_TENSORRT_RUNTIME_TEST_ENGINE_FACE68=<trusted Face68 engine>
cmake --build --preset win-release-user
ctest --preset win-release-user -V -R
  "^(test_tensorrt_runtime_integration_arcface|test_face_models_integration_(arcface|age_gender|face68))$"
ctest --preset win-release-user -R
  "^(test_tensorrt_runtime_(contract|engine_file|cuda_buffer|control)|test_face_model_(contracts|results))$"
ctest --preset win-release-user
```

最终结果：

```text
real-engine integration: 4/4 passed, 0 failed (final rebuild run: 2.11 sec)
each face command: 1 passed, 0 failed, 0 filtered
generic Host/CUDA output max abs diff: 0; assertions: 1036 passed
adjacent runtime/face: 6/6 passed, 0 failed (final run: 0.27 sec)
full configured CTest: 18/18 passed, 0 failed (final run: 10.65 sec)
```

真实三模型输出摘要与 fix 前一致；本轮只建立确定的 CUDA happens-before 与 strict test
selection，不改变 runtime、adapter、模型输入或输出解释。

### Fix commit 与自审

- 聚焦 commit subject：`fix(tests): harden TensorRT integration execution`；最终 hash 由 fix
  handoff 提供。
- `HIGH`：未发现。事实：每条 face CTest 必须进入唯一 test 并精确选择一个真实 adapter。
- `MED`：review 两项均有 RED/覆盖证据；generic zero-init 在 executor 前显式同步，selector
  missing/unknown 均 exit 1，三个合法 selector 均为 `1 passed / 0 filtered`。
- `LOW`：CMake helper fixture ledger 未扩展，按 review 明确留给 final review。
- 生产影响：无；本轮只修改 opt-in integration 测试源码、CTest 注册与 Task 4 报告。
