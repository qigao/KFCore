# Task 5 Report: Package export and documentation

## 状态

- 完成 `KFCore::tensorrt_runtime` 与 `KFCore::face_models` 的 installed package
  capability、依赖发现、target/header 消费验证。
- 完成 Windows TensorRT model Release configure/build/test/install preset。
- 完成模型能力矩阵与 prepared-tensor/preprocessing/raw-output 边界说明。
- 完成 generic runtime integration-engine helper 的纯 CMake fixture 覆盖。

## RED / GREEN

- RED：先注册 `test_tensorrt_runtime_installed_consumer`，在未修改 package config 时运行；
  consumer configure 按预期失败，错误为 installed package 未报告
  `KFCore_HAS_TENSORRT_RUNTIME` 和 `KFCore_HAS_FACE_MODELS`。
- GREEN：package config 增加 flags、CUDA/TensorRT 发现与 module path 后，同一测试通过安装、
  隔离 configure、build、link 和 run。
- 测试脚本首次在嵌套 Visual Studio generator 路径中触发 Windows 路径长度错误；consumer
  改用仓库已有的 Ninja 后通过。这是测试构建目录问题，不是 package 行为 fallback。

## Package 路径与隔离

- 测试安装前缀：
  `build/Msvc-TensorRT-Models-Release/tests/tensorrt_runtime_installed_consumer/install`
- 测试 package 目录：上述前缀的 `lib/cmake/KFCore`
- consumer 显式传入 `TENSORRT_ROOT`、`CUDAToolkit_ROOT`，关闭 CMake user/system package
  registry，并以 `NO_DEFAULT_PATH` 只查找测试安装的 KFCore package。
- Windows run PATH 只组合测试 install `bin`、TensorRT `bin`、CUDA `bin` 与系统目录；
  检查并拒绝 KFCore build `bin` 混入。
- `KFCORE_BUILD_FACE_MODELS=OFF` 的 `win-release-user` configure 下，CTest 查询结果为
  `Total Tests: 0`，未注册该 consumer。

## Preset / JSON

- 新入口：`win-tensorrt-models-release-user` 与
  `install-win-tensorrt-models-release-user`。
- `TENSORRT_ROOT` 及三个 engine cache path 都来自同名 `$penv{...}`；未提供 engine 时
  configure 仅报告真实 engine 测试未注册。
- `ConvertFrom-Json`、`cmake --list-presets`、`cmake --build --list-presets`、
  `ctest --list-presets` 均成功。

## 验证证据

- 模型 preset fresh configure/build：成功；TensorRT 11.2.1、CUDA 12.8 被显式根解析。
- 模型 preset full CTest：15/15 通过，包括 installed consumer 与 runtime helper fixture。
- 模型 install preset：成功，安装两个 DLL/import library、公开 headers、FindTensorRT 与
  KFCore package files。
- 相邻 installed consumers：ImageProcessor 1/1、SIFT 1/1、YOLO/tracker 4/4 通过。
- 真实 ArcFace/AgeGender/Face68 engine 环境变量本次为空，因此对应 zero-input integration
  tests 按设计未注册；此结果不代表 accuracy 验证。

## Commit

- 本报告与 Task 5 改动使用同一聚焦提交；最终 hash 记录在 handoff（提交不能可靠地包含自身 hash）。

## 自审

- `事实`：runtime 构建时 installed config 查找 CUDAToolkit，并从 installed module path
  查找 TensorRT；Face 没有额外 dependency 分支，只通过 runtime target 形成依赖。
- `事实`：FindTensorRT 安装条件为 runtime 或 TensorRT YOLO，原 YOLO package tests 全部通过。
- `事实`：tracked diff 未加入本机 model/engine 绝对路径；preset 只保留父环境引用。
- `LOW`：构建仍输出既有 vendor/CUDA 编译警告，未发现由本任务引入的新编译错误。
- `LOW`：本机没有可用的三个真实 face engine path，真实 zero-input smoke 保持未注册；
  已由 cache fixture 验证空值报告、目录/零字节拒绝、相对有效路径与环境 cache 路径。
