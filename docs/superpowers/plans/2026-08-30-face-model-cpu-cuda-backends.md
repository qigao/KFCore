# CPU/CUDA 推理后端与 YOLO 分层实施记录

**目标：** 将运行时、图像处理、模型和应用拆成明确的 CPU/ONNX 与 CUDA/TensorRT 路线，
并从生产 YOLO 模块移除 OpenCV。

**约束：** 使用标准 `win-release-user` preset；无兼容 target、无自动 fallback；保持模型数据
契约与用户可见推理语义；OpenCV 只允许留在 demo/UI 边界。

## 已完成

- [x] 拆分 `runtime_onnx` 与 `runtime_tensorrt`。
- [x] 拆分 `image_processor_core`、`image_processor_cpu` 与 `image_processor_cuda`。
- [x] 拆分 `face_model_core`、`face_models_cpu` 与 `face_models_cuda`。
- [x] 拆分 `vision_model_core`、`vision_models_cpu` 与 `vision_models_cuda`。
- [x] 拆分 `face_applications_cpu` 与 `face_applications_cuda`。
- [x] 拆分 `yolo_core`、`yolo_onnx` 与 `yolo_tensorrt`。
- [x] 将 ONNX YOLO detector 提升为公开 API，并复用 `runtime_onnx`。
- [x] 删除生产 YOLO 的 OpenCV adapter、`cv::Mat` API 和 OpenCV image-sequence example。
- [x] 更新仓库内 CMake 消费方和主要架构文档。
- [x] 使用标准 preset 完成 configure、全量 build 与 69 项 CTest。

## 验证命令

在 Visual Studio x64 开发环境中执行：

```powershell
cmake --preset win-release-user
cmake --build --preset win-release-user
ctest --preset win-release-user
```

另外检查生成的 `KFCoreTargets*.cmake`、旧 target 名残留、`git diff --check` 和
`.codegraph/` 是否意外进入版本控制。
