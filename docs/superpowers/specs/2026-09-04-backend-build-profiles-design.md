# CUDA 默认与 CPU 可选构建设计

## 背景与证据

当前顶层构建无条件启用 CUDA，并同时查找 TensorRT 与 ONNX Runtime；所有 CPU/ONNX 和
TensorRT/CUDA 子目录也都会进入构建。生成的 `KFCoreConfig.cmake` 同样无条件要求两套
推理依赖。因此，即使消费方只使用默认 CUDA 路线，也会得到 CPU/ONNX DLL，并在配置消费
工程时被要求提供 `onnxruntime.dll`。

本阶段只收敛“构建与安装选择”这一变化轴。模型 DLL 合并、统一运行时工厂和模型资产
manifest 属于后续独立阶段，避免一次提交同时改变二进制边界、公开 C++ API 与资产格式。

## 决策

新增两个正交的 CMake 选项：

- `KFCORE_ENABLE_CUDA=ON`：构建 CUDA 图像处理、PopSift、TensorRT runtime/model/application。
- `KFCORE_ENABLE_ONNX_CPU=OFF`：构建 ONNX Runtime CPU runtime/model/application。

`win-release-user` 和 `win-dev-user` 明确采用上述 CUDA 默认值。新增隔离 build tree 的
`win-cpu-release-user`，其值为 CUDA OFF、ONNX CPU ON。CUDA 产物使用
`build/cuda/<toolchain-profile>` 和 `.../kfcore/<configuration>/cuda`，CPU 产物使用
`build/cpu/<toolchain-profile>` 和 `.../kfcore/<configuration>/cpu`。两者不得复用生成或安装
目录，避免切换选项后旧 DLL 留在 `bin` 中造成“仍然构建了两套后端”的假象。未来如需同时
编译两套后端，必须使用独立 `hybrid` 根。

顶层始终构建后端无关 core、CPU 图像 reference、THIG、hand interaction、Kalman、AprilTag
与 trackers。CUDA 选项控制 CUDA language、CUDAToolkit、TensorRT、CUDA image processor、
PopSift 和 TensorRT 子目录；ONNX CPU 选项只控制 ONNX Runtime 与 ONNX 子目录。

## 依赖与导出

生成的 package config 记录构建时启用的后端，并只查找对应依赖：CUDA 包要求
CUDAToolkit/TensorRT，CPU 包要求 ONNX Runtime。Salts 与 trackers 保持既有要求。安装时也只
安装启用后端需要的 `FindTensorRT.cmake` 或 `FindONNXRuntime.cmake`。

现有 `KFCore::*` target、公开头文件路径和 C++ 类型名称保持不变；被关闭后端的 target 不存在，
消费方若请求它会在 CMake 生成阶段明确失败，不做自动 fallback。

## 示例与用户可见行为

YOLO 和 hand interaction 示例根据实际编译后端声明可用性。只有一个后端时 CLI 自动选择它；
显式请求未编译后端时沿用现有 fail-fast 错误。双后端构建仍要求显式 `--backend`，不引入静默
降级。CUDA face swap 示例只随 CUDA 构建，CPU face application 只随 ONNX CPU 构建。

## 状态、错误与性能

本阶段不新增运行时共享状态，不改变 session/executor 所有权和线程模型。后端选择仍是 CMake
事实源；示例通过 target compile definition 读取同一选项，不维护第二份配置。关闭一个后端会
缩短配置和链接过程并减少安装 DLL，但不改变已启用后端的热路径、错误类型或模型执行结果。

## 兼容性风险

- **HIGH**：默认构建不再导出 ONNX CPU target。依赖这些 target 的消费方必须使用
  `win-cpu-release-user`，或显式设置 `KFCORE_ENABLE_ONNX_CPU=ON`。
- **MED**：旧 `build/Msvc*` 与 `.../kfcore/<configuration>` 不再是新 preset 的输出位置，仍可能
  保留历史 DLL。验证产物数量必须使用新的 `cpu`/`cuda` 隔离根，不能用旧目录枚举替代当前
  target graph；本次不自动删除历史目录。
- **MED**：CUDA-only 与 CPU-only 包的 target 集不同。消费方必须链接自己所需的显式 target，
  缺失 target 立即报错。
- **LOW**：单后端示例省略 `--backend` 时会自动选择唯一编译后端；这延续现有 CLI 契约。

## 迁移与回滚

CUDA 客户端继续使用 `win-release-user`，从 `.../kfcore/release/cuda` 消费且无需携带 ORT。
CPU 客户端改用 `win-cpu-release-user`，从 `.../kfcore/release/cpu` 消费。需要同时编译两套后端
以支持后续运行时切换时，显式同时开启两个选项，并使用独立 `hybrid` build/install 目录。

回滚只需恢复无条件依赖与子目录遍历，并移除新增 presets；不涉及模型转换、数据迁移或持久化
状态回滚。

## 验证范围

- 默认 CUDA preset configure/build/CTest，确认不生成 ONNX target、规则或 DLL。
- CPU preset configure/build/CTest，确认不启用 CUDA language、不生成 TensorRT/CUDA target。
- 两种安装包分别配置最小消费工程，确认 package config 只要求相应后端依赖。
- 检查生成 `KFCoreTargets.cmake` 的 target 集、`git diff --check` 和 `.codegraph/` 提交状态。
