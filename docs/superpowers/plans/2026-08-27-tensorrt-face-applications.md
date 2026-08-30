# TensorRT Face Applications Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 以 YOLOv12-face 为唯一人脸检测器，加入 Face68、ArcFace、InSwapper 与可选 GFPGAN/AgeGender 的单脸 TensorRT 应用流程。

**Architecture:** `face_models` 保持 prepared-tensor adapter 边界并新增 InSwapper/GFPGAN 与 embedding projector；独立 `face_applications` 模块通过 OpenCV Lite 实现几何、预处理、合成和同步 facade。每个 adapter 独占 executor，调用失败不修改输入图像，也不返回部分结果。

**Tech Stack:** C++17、CUDA/TensorRT 11.2、OpenCV Lite core/imgproc/imgcodecs、CMake Presets、TurboUtils TinyTest。

**Design:** `docs/superpowers/specs/2026-08-27-tensorrt-face-applications-design.md`

---

### Task 1: 增加 face model 类型与失败合约

**Files:**
- Modify: `face_models/include/kfcore/face_models/types.hpp`
- Modify: `face_models/include/kfcore/face_models/tensorrt.hpp`
- Modify: `face_models/src/contracts.hpp`
- Modify: `face_models/src/contracts.cpp`
- Test: `face_models/tests/test_contracts.cpp`

**Step 1: 写失败测试**

为 InSwapper `[1,3,128,128] + [1,512] -> [1,3,128,128]` 和 GFPGAN `[1,3,512,512] -> [1,3,512,512]` 增加严格 contract 测试。覆盖错误名称、缺 input、非 FP32、错误 rank/dimension、batch 非 1、动态未解析维度和 byte 上限。

**Step 2: 运行测试并确认 RED**

```powershell
cmake --build --preset win-release-user --target test_face_model_contracts
ctest --preset win-release-user -R '^test_face_model_contracts$' --output-on-failure
```

预期：编译失败或新增测试失败，因为类型与 contract 尚不存在。

**Step 3: 最小实现**

增加具名 extent/element 常量、拥有结果缓冲的 `InSwapperResult`/`GfpGanResult`，以及复用现有 tensor descriptor 检查器的两个 contract。保持 batch 1 和完整输出绑定，不添加 fallback tensor 名。

**Step 4: 运行测试并确认 GREEN**

重复 Step 2，预期通过。

**Step 5: 提交**

```powershell
git add face_models/include/kfcore/face_models/types.hpp face_models/include/kfcore/face_models/tensorrt.hpp face_models/src/contracts.hpp face_models/src/contracts.cpp face_models/tests/test_contracts.cpp
git commit -m "feat(face-models): add swap and enhancer contracts"
```

### Task 2: 实现 InSwapper 与 GFPGAN TensorRT adapters

**Files:**
- Create: `face_models/src/inswapper.cpp`
- Create: `face_models/src/gfpgan.cpp`
- Modify: `face_models/CMakeLists.txt`
- Create: `face_models/tests/test_swap_adapter_api.cpp`
- Create: `face_models/tests/test_inswapper_integration.cpp`
- Create: `face_models/tests/test_gfpgan_integration.cpp`
- Modify: `CMakeOptions.cmake`
- Modify: `CMakeLists.txt`

**Step 1: 写 API/contract 失败测试**

测试 move-only/Pimpl API、空 engine path、零 resource limit、同步 infer 的非法 host/CUDA view，以及 InSwapper 必须同时提供 target/source。integration test 只在显式 engine cache path 有效时注册。

**Step 2: 确认 RED**

```powershell
cmake --build --preset win-release-user --target test_face_swap_adapter_api
ctest --preset win-release-user -R '^test_face_swap_adapter_api$' --output-on-failure
```

**Step 3: 最小实现**

沿用 `arcface.cpp` 的 Pimpl、`Engine::load`、`create_executor`、有界 output buffer 和异常转换结构。InSwapper 建立两个命名 input binding；GFPGAN 建立一个 input binding。输出复制到拥有结果，函数返回后不泄露 runtime buffer view。

**Step 4: 单元验证**

运行 adapter API 与全部 face contract tests，预期通过。

**Step 5: 真 engine 验证**

用 `C:\projects\TensorRT-11.2.1.2\bin\trtexec.exe` 从本地 ONNX 生成临时 engine，显式配置：

```text
KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_INSWAPPER=<absolute engine>
KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_GFPGAN=<absolute engine>
```

运行对应 integration tests；检查输出数量、finite 和重复执行。engine 不加入 Git。

**Step 6: 提交**

```powershell
git add CMakeLists.txt CMakeOptions.cmake face_models
git commit -m "feat(face-models): add TensorRT swap and enhancer adapters"
```

### Task 3: 增加 InSwapper embedding projector

**Files:**
- Create: `face_models/include/kfcore/face_models/inswapper_embedding.hpp`
- Create: `face_models/src/inswapper_embedding.cpp`
- Modify: `face_models/CMakeLists.txt`
- Create: `face_models/tests/test_inswapper_embedding.cpp`

**Step 1: 写矩阵加载与投影失败测试**

用 TinyTest temporary-file helper 构造小于、等于和大于 1,048,576 bytes 的文件；加入 NaN 元素、全零矩阵、非有限 input embedding。使用参考工程已记录的 matrix spot values 和固定 embedding 验证 row-major 乘法与 L2 normalization。

**Step 2: 确认 RED**

```powershell
cmake --build --preset win-release-user --target test_inswapper_embedding
ctest --preset win-release-user -R '^test_inswapper_embedding$' --output-on-failure
```

**Step 3: 最小实现**

`load()` 一次读取精确大小文件，验证每个 float finite；`project()` 计算 `embedding * matrix` 并 L2 normalize，拒绝非有限/零范数。矩阵由 projector 不可变拥有，不返回可写 view。

**Step 4: 确认 GREEN 并提交**

```powershell
ctest --preset win-release-user -R '^test_inswapper_embedding$' --output-on-failure
git add face_models
git commit -m "feat(face-models): add InSwapper embedding projection"
```

### Task 4: 建立 OpenCV 人脸几何与 tensor 预处理层

**Files:**
- Create: `face_applications/CMakeLists.txt`
- Create: `face_applications/include/kfcore/face_applications/error.hpp`
- Create: `face_applications/include/kfcore/face_applications/geometry.hpp`
- Create: `face_applications/include/kfcore/face_applications/preprocess.hpp`
- Create: `face_applications/src/error.cpp`
- Create: `face_applications/src/geometry.cpp`
- Create: `face_applications/src/preprocess.cpp`
- Create: `face_applications/tests/test_geometry.cpp`
- Create: `face_applications/tests/test_preprocess.cpp`
- Modify: `CMakeOptions.cmake`
- Modify: `CMakeLists.txt`

**Step 1: 写几何 RED tests**

覆盖：

- Face68 bbox center/extent affine 和 inverse round trip；
- 68 点到 5 点的 eye average 与 `30/48/54` 索引；
- 112/128/512 canonical points；
- similarity transform 将已知 source 点映射到 canonical 点，误差在明确 epsilon 内；
- 空图、非 `CV_8UC3`、退化 bbox、NaN landmark fail fast。

**Step 2: 写预处理 RED tests**

用 2x2/常量 BGR 图验证 Face68、ArcFace、InSwapper、GFPGAN 的 BGR/RGB、planar CHW、scale/mean/std。断言输入 `cv::Mat` 内容不变，输出 shape 和连续性正确。

**Step 3: 配置并确认 RED**

开启 `KFCORE_BUILD_FACE_APPLICATIONS=ON`，设置 `OPENCV_LITE_ROOT=C:\projects\cpp\external\pkgs\opencv-lite`，构建两个测试，预期因实现缺失失败。

**Step 4: 最小实现**

使用 OpenCV Lite 的 `warpAffine`、`invertAffineTransform` 和受测的 similarity 求解路径。模型常量只定义一处；所有像素数量乘法先做溢出检查。几何函数不持有 `cv::Mat` data pointer。

**Step 5: 确认 GREEN 并提交**

```powershell
ctest --preset win-release-user -R '^test_face_(geometry|preprocess)$' --output-on-failure
git add CMakeLists.txt CMakeOptions.cmake face_applications
git commit -m "feat(face-applications): add face geometry and preprocessing"
```

### Task 5: 实现 mask、paste-back 与 GFPGAN blend

**Files:**
- Create: `face_applications/include/kfcore/face_applications/composer.hpp`
- Create: `face_applications/src/composer.cpp`
- Create: `face_applications/tests/test_composer.cpp`
- Modify: `face_applications/CMakeLists.txt`

**Step 1: 写 RED tests**

验证静态 box mask 的边缘/中心、Gaussian blur、identity inverse warp、目标图外裁剪、alpha 0/1、中间 blend、错误 mask type/size、输入不可变。使用逐像素期望值，不只断言“非空”。

**Step 2: 确认 RED**

```powershell
cmake --build --preset win-release-user --target test_face_composer
ctest --preset win-release-user -R '^test_face_composer$' --output-on-failure
```

**Step 3: 最小实现**

输出先 clone target；inverse-warp face 与 mask 后按 mask 合成。GFPGAN 结果先按 enhancer blend 与当前 aligned face 混合，再走同一 paste-back 事实源。所有 model output 在转换前检查 finite 并 clamp。

**Step 4: GREEN、回归与提交**

```powershell
ctest --preset win-release-user -R '^test_face_(geometry|preprocess|composer)$' --output-on-failure
git add face_applications
git commit -m "feat(face-applications): add bounded face composition"
```

### Task 6: 实现 12face 单脸分析与换脸 facade

**Files:**
- Create: `face_applications/include/kfcore/face_applications/tensorrt.hpp`
- Create: `face_applications/src/tensorrt.cpp`
- Create: `face_applications/tests/test_application_validation.cpp`
- Modify: `face_applications/CMakeLists.txt`

**Step 1: 写 facade validation RED tests**

覆盖 model path 缺失、threshold/blend 越界、空图/错误图像类型、没有 class 0 人脸、重叠调用拒绝、可选 engine 配置不一致。把“最高 score，平分保留原序”提取为纯选择函数并用真实 Detection 值测试，不构造假 engine。

**Step 2: 确认 RED**

构建并运行 `test_face_application_validation`，预期失败。

**Step 3: 最小实现 analyzer**

调用现有 `TensorRtDetector` 和 `yolo_opencv::image_view`；选择 12face 结果，运行 bbox crop、Face68、5 点、ArcFace；AgeGender 仅在显式加载时运行并返回 raw logits。

**Step 4: 最小实现 swap**

分别分析 source/target；project source embedding；对 target aligned face 运行 InSwapper；paste-back 到 target clone；若配置 GFPGAN，再用同一 target landmarks 对当前结果增强。任何阶段失败都丢弃局部 output 并带阶段转换错误。

**Step 5: GREEN 与提交**

```powershell
ctest --preset win-release-user -R '^test_face_application_' --output-on-failure
git add face_applications
git commit -m "feat(face-applications): add YOLOv12 face swap facade"
```

### Task 7: 增加命令行示例和端到端 opt-in 验证

**Files:**
- Create: `face_applications/examples/face_swap_image.cpp`
- Create: `face_applications/tests/test_face_swap_integration.cpp`
- Modify: `face_applications/CMakeLists.txt`
- Modify: `CMakeOptions.cmake`
- Modify: `CMakeUserPresets.json`

**Step 1: 写 CLI 参数失败测试/最小解析测试**

CLI 要求 source、target、output、12face/Face68/ArcFace/InSwapper engine 和 matrix；GFPGAN/AgeGender 为显式可选参数。缺参数、文件不存在、重复参数和未知参数立即失败，不搜索默认 weights 目录。

**Step 2: 实现示例**

用 OpenCVLite imgcodecs 读写 BGR 图；只在成功得到完整输出后写文件。日志只报告阶段与最终路径，不输出 embedding 或大 tensor。

**Step 3: 本机 end-to-end**

使用 `yolo-models`、已生成 engine、参考 matrix sidecar 和可合法使用的本地测试图片运行：

```powershell
face_swap_image.exe --source <image> --target <image> --output <image> `
  --detector <yolov12-face.engine> --face68 <2dfan4.engine> `
  --arcface <arcface.engine> --inswapper <inswapper.engine> `
  --matrix <model_matrix.bin> --gfpgan <gfpgan.engine>
```

验收：进程成功、输出可重新读取为 `CV_8UC3`、尺寸等于 target、source/target 文件不变、启用和禁用 GFPGAN 都成功。

**Step 4: 提交**

```powershell
git add CMakeOptions.cmake CMakeUserPresets.json face_applications
git commit -m "feat(face-applications): add face swap image application"
```

### Task 8: 安装导出、文档与全量回归

**Files:**
- Modify: `cmake/KFCoreConfig.cmake.in`
- Modify: `CMakeLists.txt`
- Modify: `README.md`
- Create: `face_applications/README.md`
- Create or Modify: package-consumer test under existing CMake test structure

**Step 1: 写安装消费 RED test**

验证启用模块的 install tree 导出 `KFCore::face_applications`，并且 consumer 能通过 `find_package(KFCore CONFIG REQUIRED)` 链接；禁用模块时不要求 OpenCVLite。

**Step 2: 实现 CMake 导出与文档**

增加 `KFCore_HAS_FACE_APPLICATIONS`，按启用状态查找 OpenCVLite core/imgproc。README 记录唯一 12face 检测入口、完整数据流、模型 I/O、matrix sidecar 规则、非重入/所有权约束和可复验命令。

**Step 3: 最小测试、相邻回归、全量回归**

```powershell
ctest --preset win-release-user -R 'face|tensorrt|opencv' --output-on-failure
ctest --preset win-release-user --output-on-failure
```

预期：新增测试与原有 27 项基线全部通过。记录实际测试数、耗时和任何 opt-in 未运行项的原因。

**Step 4: 安装消费验证**

构建 install target，再运行 consumer configure/build/test。确认 OpenCVLite、TensorRT 和 CUDA 依赖只在启用组件时传播。

**Step 5: 最终检查与提交**

```powershell
git diff --check
git status --short
git add CMakeLists.txt cmake/KFCoreConfig.cmake.in README.md face_applications/README.md
git commit -m "docs(face-applications): document TensorRT face workflow"
```

完成前使用 `superpowers:verification-before-completion` 复核 fresh configure、目标构建、CTest、端到端命令与 Git diff；未实际运行的验证不得写成通过。

