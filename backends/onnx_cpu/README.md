# ONNX Runtime CPU / CUDA 构建

本目录的模型封装同时用于 CPU 和 ONNX Runtime CUDA 两种独立构建。
CUDA 版本使用 CUDA Execution Provider，不注册 TensorRT EP，不需要 TensorRT engine。

## 构建与运行

在 Visual Studio x64 开发者终端、仓库根目录执行：

```powershell
cmake --preset win-onnx-cuda-release-user
cmake --build --preset win-onnx-cuda-release-user
ctest --preset win-onnx-cuda-release-user -R test_runtime_onnx --output-on-failure
```

产物目录为 `build/onnx-cuda/Msvc-Release/bin`。
安装入口为 `cmake --build --preset install-win-onnx-cuda-release-user`，
安装到独立的 `external/pkgs/kfcore/release/onnx-cuda` 前缀。
原 `win-cpu-release-user` 仍生成 CPU 推理版本。

新 preset 使用 `C:/projects/cpp/external/pkgs/onnxruntime-gpu`、CUDA 12.8
以及本机 cuDNN 9 的 CUDA 12 运行库目录；机器路径统一配置在
`CMakeUserPresets.json`。SDK 必须包含 CUDA 与 shared provider 动态库。
CUDA/cuDNN 大版本必须匹配所用 ORT 版本，参见
[官方 CUDA EP 依赖说明](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。

直接运行程序时，也必须设置绝对路径 `ONNXRUNTIME_ROOT`，并将 CUDA、cuDNN、
项目依赖的运行库目录加入进程 `PATH`；CMake preset 的环境不会修改父终端。
Windows 适配层按该根目录显式加载编译时选定布局中的 ORT DLL，避免
`System32/onnxruntime.dll` 抢先加载。错误根目录、缺失 DLL 或 API 版本不匹配
会返回 `RuntimeFailure`；不搜索其他 ORT 安装。

## 行为与兼容性

- `KFCORE_ENABLE_ONNX_CUDA=ON` 时，必须关闭 `KFCORE_ENABLE_ONNX_CPU`
  和旧的 `KFCORE_ENABLE_CUDA`（后者控制 TensorRT 栈）。两种 ORT 版本分目录构建，
  不应在同一进程中混装其同名 DLL。
- 模型封装、导出目标、类名以及示例的 `--backend cpu` 名称沿用既有接口。
  在此 CUDA 构建中，它们统一调用 CUDA EP；示例也可省略 `--backend` 自动选择
  唯一编入的 ONNX 后端。名称中的 `cpu` 不代表此构建的推理设备。
- 保留主机张量输入输出、CPU 图像前后处理和同步调用语义；这不是全流程 GPU
  常驻或零拷贝实现。CUDA EP 管理推理内存与数据传输。
- `KFCORE_ONNX_CUDA_ALLOW_CPU_NODES` 默认为 `OFF`（严格模式）；本机 CUDA Release
  preset 经用户授权设为 `ON`，允许 ORT 将 CUDA 不支持或更适合 CPU 的节点分配给
  CPU EP。CUDA provider 创建始终是必需步骤，CUDA 初始化失败仍返回错误，
  不改为仅 CPU 的会话。不注册 TensorRT。此选项不限定 CPU 节点数量，性能需实测；
  若模型与 ORT 后续支持全 CUDA，可关闭该选项并重跑模型测试。
- `KFCORE_ONNX_CUDA_DEVICE_ID` 默认 `0`；`KFCORE_ONNX_CUDA_MEMORY_LIMIT`
  默认 `2147483648` 字节，限制每个会话的 CUDA arena，不是进程显存总上限。
  在 user preset 的 `cacheVariables` 中调整后重新构建。TF32 关闭以减少
  与原 FP32 CPU 路径的数值差异；仍需按模型容差比较结果。

## 结构决策与验证

统一运行时拥有 ORT 库、环境、会话与 provider 配置；模型层保持张量契约与领域结果，
应用层保持原有处理流程。相比复制 YOLO、人脸、手部封装，新构建复用同一实现，
避免两份校验、后处理与错误语义逐渐分叉；代价是保留历史 CPU 命名，且不能在同一
构建中混用 CPU/CUDA ONNX 会话。若以后需要同进程多 provider，应单独设计显式
会话选项和模型入口，不在当前补丁中改变公共结构体布局。

回滚只需选择原 CPU preset 和其独立产物；不涉及模型数据或状态迁移。
加载阶段失败不会创建半有效会话，provider 临时配置通过 RAII 释放。
运行库生命周期覆盖所有使用其 API 的环境与张量。

`test_runtime_onnx_execution` 使用仓库内微型模型，验证运算、重复调用与主机输出，
并确认 CPU-only Binarizer 按 CPU 节点选项被允许或拒绝，CPU 构建正常执行。
Windows 下还检查 CUDA provider 已加载且 TensorRT provider 未加载；
`test_runtime_onnx_load_failure` 验证错误 SDK 根目录不会转用系统 ORT。
完整回归通过 `ctest --preset win-onnx-cuda-release-user --output-on-failure` 运行，
真实模型测试需要先提供测试中指定的 `yolo-models/` 资产；缺失资产会明确失败。

测试工作目录由 `KFCORE_MODEL_TEST_WORKING_DIRECTORY` 指定，其下应有 `yolo-models/`。
本机 CPU/CUDA Release preset 现指向
`C:/projects/project-cpp-template/build/Msvc-Release/bin/Retro`，直接使用 Retro 的模型资产。

2026-09-10 本机实测：授权 CPU 节点后的 CUDA 配置全套 54/54 通过，
其中真实模型集成测试 9/9 通过；CPU 模型基线也为 9/9。测试覆盖 YOLO 域模型、
YOLOv8s 人体、FaceMesh、Face68、ArcFace、年龄性别、InSwapper、GFPGAN 和手部。
这些是合成输入的模型推理与输出校验，不代表相机画面识别精度或端到端性能已经验证。

严格模式曾为 5/9：YOLOv8s 的 `/model.10/Resize` 与 `/model.13/Resize` 没有 CUDA
内核，年龄性别部分 Gather/Slice/Concat 节点被 ORT 选择在 CPU 执行。允许 CPU 节点
解决这些会话初始化失败，不修改原始模型文件、业务状态或对外结果格式。
