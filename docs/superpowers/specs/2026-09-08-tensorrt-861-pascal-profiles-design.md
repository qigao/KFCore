# TensorRT 8.6.1 Pascal 部署 Profiles 设计

## 背景与证据

KFCore 当前 Windows CUDA presets 使用 TensorRT 11.2.1，并把
`rtx4060-sm89-trt11.2.1-default` 编译为默认 engine profile。TensorRT 11 不支持 Pascal，
而用户需要面向 GTX 1060 与 GTX 1070（compute capability 6.1）的独立部署产物。

本机只有 RTX 4060 Laptop（SM 8.9），因此可以验证 TensorRT 8.6.1/CUDA 12 的配置、编译和
ONNX 解析，但不能生成或验证可部署的 SM 6.1 engine。TensorRT 8.6.1 的硬件兼容模式只覆盖
Ampere 及更新架构，不能从 SM 8.9 反向生成 Pascal plan。

## 决策

新增两个 Windows Release presets：

- `win-gtx1060-release-user` 使用 engine profile
  `gtx1060-sm61-trt8.6.1-default`；
- `win-gtx1070-release-user` 使用 engine profile
  `gtx1070-sm61-trt8.6.1-default`。

两者共享 TensorRT 8.6.1、CUDA 12.8、cuDNN 8.9 与 `CMAKE_CUDA_ARCHITECTURES=61`，但使用
独立 build/install 目录。默认 `win-release-user` 的 TensorRT 11.2.1/RTX 4060 行为不变。

GTX 1060 与 GTX 1070 虽同为 SM 6.1，默认 TensorRT engine 仍只保证与构建时 GPU 型号及
软件栈匹配。两者不共享 engine 目录，避免把可反序列化与已针对目标设备选取最优 tactics
混为同一契约。

## TensorRT 版本边界

`FindTensorRT.cmake` 接受 TensorRT 8.6.x、10.x 与 11.x；TensorRT 8.0 至 8.5、9.x 和未来
未验证 major 继续 fail fast。TensorRT runtime 的 alias 检查保持现状：8.6 不具备 alias query，
10.3 至 10.10 继续因无法完整验证 alias contract 而拒绝，10.11+ 与 11.x 使用 query。

现有公开 C++ target、类型、错误码、tensor contract、执行所有权和线程模型不变。新增支持只
扩展构建依赖范围，不增加运行时 fallback，也不允许一个二进制在 TensorRT 8/11 之间动态切换。

## 依赖与路径

Pascal presets 从以下单一根解析依赖：

- TensorRT：`C:/projects/TensorRT-8.6.1`；
- CUDA：现有 `CUDA_PATH_V12_8`；
- cuDNN：`$env{PKG_ROOT}/cudnn-8.9.7-cuda12`。

TensorRT 8.6 的 `nvinfer_plugin.dll` 直接依赖 `cudnn64_8.dll`，因此 Pascal preset 的运行时
`PATH` 显式加入 cuDNN `bin`。缺失任一根或 DLL 时配置/执行必须失败，不回退到系统安装。

## Engine 生成与状态

ONNX 是 engine 的唯一事实源。Engine 只写入：

- `yolo-models/tensorrt/<gpu-profile>/`；
- `yolo-models/hand_gesture_model/tensorrt/<gpu-profile>/`。

每个 engine 在对应 GTX 主机使用 TensorRT 8.6.1 构建并立即通过 `trtexec --loadEngine` 做
最小推理验证。构建失败时不发布该 engine；已经成功生成的其他 engine 保持有效，无跨文件事务。
`model_matrix.bin` 仍是 ONNX initializer 导出的共享 sidecar，不随 GPU profile 复制或重建。

## 兼容性风险

- **HIGH**：SM 8.9 上生成的 plan 不能作为 SM 6.1 artifact；最终 engine 验证只能在 Pascal
  主机完成。
- **HIGH**：TensorRT plan 不能由 TensorRT 11 构建的 KFCore 二进制直接替代加载；每个 Pascal
  package 必须链接并部署 TensorRT 8.6.1 runtime DLL。
- **MED**：GTX 1060 存在 3 GiB/6 GiB 等显存配置，大模型（尤其 GFPGAN/InSwapper）可能在
  engine 构建或执行时超出显存；失败必须保留为明确结果，不降低精度或静默跳过。
- **MED**：TensorRT 8.6 ONNX parser 对较新 opset/model graph 的支持可能小于 TensorRT 11；
  每个模型独立记录构建结果。
- **LOW**：CUDA 12.8 仍可编译 SM 6.1，但最终驱动版本和部署 DLL 集须在目标主机验证。

## 迁移与回滚

Pascal 用户选择匹配显卡的 configure/build/test/install preset，并部署相同 profile 名称下的
engine。现有 RTX 4060 用户继续使用 `win-release-user`，无迁移。

回滚时删除新增 Pascal presets、TensorRT 8.6 版本许可和对应文档即可；不涉及公开 API、模型
格式或用户数据迁移。外部生成的 engine 位于忽略目录，由部署者按 GPU profile 单独管理。

## 验证范围

- preset JSON discovery 与各 profile 的隔离路径；
- TensorRT 8.5/9 拒绝、8.6 接受、现有 10/11 alias policy 回归测试；
- 使用真实 `C:/projects/TensorRT-8.6.1` 的 Windows configure 与编译；
- RTX 4060 上只做工具链/解析预检，不宣称 SM 6.1 engine 验证；
- GTX 1060 和 GTX 1070 主机上的逐 engine build/load smoke test 作为最终部署门槛。
