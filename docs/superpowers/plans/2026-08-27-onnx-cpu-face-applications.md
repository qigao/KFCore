# ONNX Runtime CPU 人脸应用实施计划

## 目标

新增一条不依赖 OpenCV、TensorRT 或 CUDA 的 CPU 换脸路线，复用现有模型契约与
InSwapper 矩阵语义；现有 TensorRT/OpenCV API 和默认构建行为保持不变。后端由调用方
显式选择，不增加自动 fallback。

## Task 1：拆出可独立使用的图像与人脸模型核心

**文件：**

- 修改：`image_processor/CMakeLists.txt`
- 修改：`image_processor/include/kfcore/image_processor/types.hpp`
- 新增：`image_processor/include/kfcore/image_processor/cpu.hpp`
- 新增：`image_processor/src/cpu.cpp`
- 新增：`image_processor/tests/test_cpu_image_processor.cpp`
- 修改：`face_models/CMakeLists.txt`
- 修改：`CMakeLists.txt`

**行为：**

- `KFCore::image_core` 只依赖 C++17，提供有所有权的 BGR 图像、仿射 warp、resize、
  letterbox 与 FP32 NCHW 转换。
- 现有 `KFCore::image_processor` 继续提供 CUDA API，并公开依赖 `image_core`。
- `KFCore::face_model_core` 持有共享结果类型和 InSwapper embedding projector；
  `KFCore::face_models` 继续作为 TensorRT adapter target。
- 每项图像行为先写 TinyTest 并观察缺少 API 的失败，再实现最小代码使其通过。

## Task 2：加入严格的 ONNX Runtime CPU 应用边界

**文件：**

- 新增：`cmake/FindONNXRuntime.cmake`
- 新增：`face_applications_cpu/CMakeLists.txt`
- 新增：`face_applications_cpu/include/kfcore/face_applications/cpu.hpp`
- 新增：`face_applications_cpu/include/kfcore/face_applications/error.hpp`（若与现有错误
  类型可直接共享则复用现有文件）
- 新增：`face_applications_cpu/src/onnx_session.*`
- 新增：`face_applications_cpu/tests/test_onnx_contract.cpp`
- 修改：`CMakeOptions.cmake`
- 修改：`CMakeLists.txt`
- 修改：`cmake/KFCoreConfig.cmake.in`

**行为：**

- 新增 `KFCORE_BUILD_FACE_APPLICATIONS_CPU`，缺失或越界的 `ONNXRUNTIME_ROOT` 在
  configure 阶段失败。
- 一个应用对象共享一个 `Ort::Env`，每个模型拥有独立 `Ort::Session`；Ort 类型不进入
  公开 API。
- 加载时严格验证名称、FP32 dtype、rank 与固定维度；动态 batch 只接受实际调用的 batch 1。
- CPU 调用同步、非重入，输入只在调用期间借用。

## Task 3：实现无 OpenCV 的模型预处理、几何与结果合成

**文件：**

- 新增：`face_applications_cpu/src/geometry.*`
- 新增：`face_applications_cpu/src/preprocess.*`
- 新增：`face_applications_cpu/src/composer.*`
- 新增：`face_applications_cpu/tests/test_cpu_face_geometry.cpp`
- 新增：`face_applications_cpu/tests/test_cpu_face_preprocess.cpp`
- 新增：`face_applications_cpu/tests/test_cpu_face_composer.cpp`

**行为：**

- 复现现有 Face68 crop、68-to-5、ArcFace/InSwapper/GFPGAN canonical alignment。
- 复现各模型的 BGR/RGB、CHW、mean/stddev 语义。
- InSwapper/GFPGAN 输出有限值检查、clamp、mask、inverse warp、paste 与 blend 全部
  使用 `image_core`，不引入 OpenCV 类型。
- 单元测试以手工推导像素和变换为期望值，覆盖错误输入和输入不可变。

## Task 4：串联 12face 与完整 CPU 换脸流程

**文件：**

- 新增：`face_applications_cpu/src/cpu.cpp`
- 新增：`face_applications_cpu/tests/test_cpu_pipeline_validation.cpp`
- 新增：`face_applications_cpu/tests/test_cpu_pipeline_integration.cpp`
- 新增：`face_applications_cpu/tests/benchmark_cpu_pipeline.cpp`
- 修改：`CMakeUserPresets.json`
- 修改：`face_applications/README.md`

**行为：**

- `OnnxFaceSwapApplication::load/analyze/swap/swap_profiled` 提供单人脸同步 CPU 流程。
- 12face 使用居中 letterbox、RGB `/255`，严格解析 `[1,300,6]` compact 输出并选择
  最高分的指定类别。
- Face68 输出按 64->256 比例解码；ArcFace、InSwapper、可选 GFPGAN 和 Age/Gender
  按本地 ONNX 合约运行。
- 集成测试显式接收本地 ONNX、矩阵和图片资产；不自动下载、不静默跳过配置错误。
- benchmark 记录加载、各模型、图像处理、合成和总耗时，便于与 GPU 路线同口径比较。

## Task 5：验证、安装与兼容性

**验证顺序：**

1. 新增 CPU 图像单元测试。
2. CPU 人脸几何、预处理、合成与契约测试。
3. 本地真实 ONNX 端到端测试与 benchmark。
4. 现有 `win-release-user` 回归。
5. CPU preset 的完整构建、CTest 与安装消费测试。

**兼容性约束：**

- 不删除或改签现有 `TensorRtFaceSwapApplication` 方法。
- OpenCV 仅保留在现有 façade、示例和 UI target。
- CPU target 的导出依赖只有 ONNX Runtime、TurboUtils 和无 GPU 的 KFCore core targets。
- 任一模型契约不匹配立即失败，不切换其他 detector/backend。
