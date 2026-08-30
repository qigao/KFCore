# CPU/CUDA 推理后端与 YOLO 分层设计

## 背景

旧结构把后端无关契约、ONNX Runtime CPU 实现、TensorRT/CUDA 实现和 OpenCV 适配混在
`face_models`、`tensorrt_runtime` 与 `tensorrt_yolo` 等旧目标中。目标名称不能准确表达执行后端，
CPU 模型实现也无法在应用层之外复用；YOLO 的生产接口还暴露了 `cv::Mat`。

## 决策

所有推理与图像处理模块按职责和执行后端显式拆分：

- runtime：`KFCore::runtime_onnx`、`KFCore::runtime_tensorrt`；
- image processor：`KFCore::image_processor_core`、`KFCore::image_processor_cpu`、
  `KFCore::image_processor_cuda`；
- YOLO：`KFCore::yolo_core`、`KFCore::yolo_onnx`、`KFCore::yolo_tensorrt`；
- face models：`KFCore::face_model_core`、`KFCore::face_models_cpu`、
  `KFCore::face_models_cuda`；
- hand models：`KFCore::hand_model_core`、`KFCore::hand_models_cpu`、
  `KFCore::hand_models_cuda`；
- face applications：`KFCore::face_applications_cpu`、`KFCore::face_applications_cuda`。

后端在链接时选择，不提供无后端含义的兼容 target，不做运行时自动 fallback。
ONNX 模型由 `runtime_onnx` 执行 CPU 路线；TensorRT engine 由 `runtime_tensorrt` 执行
CUDA 路线。

## YOLO 边界

`yolo_core` 只保存检测、跟踪、解码和共享数据契约；`yolo_onnx` 提供公开的
`kfcore::yolo::OnnxDetector`；`yolo_tensorrt` 提供 TensorRT engine 执行器。

生产 YOLO API 不依赖 OpenCV，不接受或返回 `cv::Mat`。调用方使用 `ImageView`、连续 tensor
或后端明确的数据视图传递图像。OpenCV 可以继续用于仓库中的 demo/UI 捕获、显示与绘制，但不
进入上述生产 target 的公开接口或链接依赖。

## 状态、所有权与错误语义

- 每个 runtime/session/executor 拥有自己的可变执行状态，同一实例不保证可重入；
- 同步推理期间借用输入，返回值拥有输出数据；
- 模型路径、资产大小、tensor 名称、dtype、rank、shape 和元素数量在后端边界严格验证；
- 后端异常在 adapter 边界转换为对应领域错误，不记录后返回成功；
- CPU/CUDA 共享结果类型和解码契约，避免应用层维护第二份模型事实源。

## 兼容性与迁移

这是公开 CMake target 的显式迁移。外部消费方必须选择 `_cpu`、`_cuda`、`_onnx` 或
`_tensorrt` target；旧的 `tensorrt_runtime`、`tensorrt_yolo`、`yolo_tracking`、
`yolo_opencv`、`face_models` 和 `face_applications` 不再导出。

模型资产格式不变：CPU 路线加载 `.onnx`，CUDA 路线加载由 TensorRT 构建的 `.engine`。
仓库不隐式转换模型，也不因某一后端失败而切换到另一后端。

## 验证范围

- 标准 `win-release-user` configure 与全量 build；
- runtime、image processor、YOLO、face、vision 和 application 的最小相关测试；
- 真实 ONNX 与 TensorRT engine 集成测试；
- 完整 CTest；
- 生成的 CMake export target 名和旧 target 残留检查；
- `git diff --check` 与 `.codegraph/` 状态检查。
