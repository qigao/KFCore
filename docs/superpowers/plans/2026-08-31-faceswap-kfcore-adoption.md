# Faceswap 统一迁移到 KFCore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让 Retro `faceswap`/`image_mcp` 的 CPU 与 CUDA 人脸分析、换脸、增强全部由 KFCore 执行，同时保持多脸目录和缓存行为。

**Architecture:** KFCore CPU/CUDA 应用增加语义对等的 `analyze_all` 与 `swap_prepared`；Retro 的 `NetworkManager` 变为薄适配器和唯一模型 owner，业务层继续拥有缓存、目录搜索与多脸配对。旧单脸 API 保持兼容，CPU/CUDA 不互相 fallback。

**Tech Stack:** C++17/20、CMake Presets、TinyTest/Catch2、ONNX Runtime（仅 KFCore CPU runtime）、TensorRT 11.2/CUDA（KFCore CUDA runtime）。

**Spec:** `docs/superpowers/specs/2026-08-31-faceswap-kfcore-adoption-design.md`

## Global Constraints

- 保留工作区现有脏改动，不覆盖与本任务无关的文件；当前模型名称和 raw YOLO 改动是前置依赖。
- 所有生产修改前先添加并运行对应红测；每次只补足当前测试所需实现。
- 仅使用标准 configure/build/test presets；Windows 命令在 `VsDevCmd.bat` 环境中执行。
- 不提交 `.codegraph/`、build tree、模型资产或生成的 engine。
- 不改变 image_mcp 协议、目录数据库或缓存格式。

---

### Task 1: CPU detector 返回全部人脸且保持单脸兼容

**Files:**

- Modify: `face_model_core/include/kfcore/face_models/types.hpp`
- Modify: `face_model_core/src/facemesh_decode.hpp`
- Modify: `face_model_core/src/facemesh_decode.cpp`
- Modify: `face_model_core/tests/test_facemesh_decode.cpp`
- Modify: `face_models_cpu/include/kfcore/face_models/cpu.hpp`
- Modify: `face_models_cpu/src/facemesh.cpp`
- Modify: `face_models_cpu/tests/test_cpu_face_models_api.cpp`

**Steps:**

1. 添加解码测试：两条合格人脸全部返回、保持模型行顺序、旧解码仍选最高分、畸形任意行整体失败。
2. 运行 `test_facemesh_decode`，确认因 `decode_yolo12_faces` 不存在而失败。
3. 添加 `FaceDetectionsResult`、`decode_yolo12_faces` 与 `CpuFaceDetector::infer_all`；旧接口从全部结果稳定选最高分。
4. 运行 face_model_core 与 face_models_cpu 的相关 API/解码测试。

### Task 2: CPU face application 增加全脸分析与 prepared swap

**Files:**

- Modify: `face_applications_cpu/include/kfcore/face_applications/cpu.hpp`
- Modify: `face_applications_cpu/src/cpu.cpp`
- Modify: `face_applications_cpu/tests/test_cpu_face_application_validation.cpp`
- Modify: `face_applications_cpu/tests/test_cpu_face_pipeline_integration.cpp`
- Modify: `face_applications_cpu/README.md`

**Steps:**

1. 添加 API 类型测试，要求 `analyze_all` 和 `swap_prepared(..., bool enhance)` 存在。
2. 添加纯验证测试：非有限 embedding/landmarks、请求未配置增强模型必须抛明确错误。
3. 运行 CPU application 测试，确认红测。
4. 将单检测分析拆为“检测一次 + 对每个框分析”，`analyze` 稳定选择最高分。
5. 抽取 prepared swap 核心；现有 `swap` 分析源/目标后调用它并保持默认增强行为。
6. 用真实 ONNX 资产运行 CPU 单脸、双脸、prepared swap 集成测试。

### Task 3: CUDA face application 提供对等接口并复用设备 staging

**Files:**

- Modify: `face_applications_cuda/include/kfcore/face_applications/tensorrt.hpp`
- Modify: `face_applications_cuda/src/tensorrt.cpp`
- Modify: `face_applications_cuda/tests/test_application_validation.cpp`
- Modify: `face_applications_cuda/tests/test_tensorrt_face_application_integration.cpp`
- Modify: `face_applications_cuda/README.md`

**Steps:**

1. 添加与 CPU 同语义的 API/验证红测。
2. 运行 CUDA validation 测试，确认红测。
3. 检测一次后过滤全部人脸；在一次 frame generation 中为各脸运行 Face68/ArcFace/age-gender。
4. 抽取 `swap_prepared`，复用 target device staging；按 `enhance` 决定 GFPGAN 阶段。
5. 现有 `analyze`/`swap` 复用新核心并保持异常与默认增强行为。
6. 使用 `rtx4060-sm89-trt11.2.1-default` 中的 2dfan4、ArcFace、InSwapper、GFPGAN engine 跑 CUDA 集成测试。

### Task 4: 安装并验证 KFCore 包导出

**Files:**

- Verify: `CMakeLists.txt`
- Verify: `face_applications_cpu/CMakeLists.txt`
- Verify: `face_applications_cuda/CMakeLists.txt`
- Verify: installed `lib/cmake/KFCore/KFCoreTargets*.cmake`

**Steps:**

1. `cmake --preset win-release-user`。
2. 构建相关测试和 CPU/CUDA application targets。
3. 运行相关 CTest filters。
4. 用标准 install preset 安装。
5. 用 `rg.exe` 验证安装包导出 `KFCore::face_applications_cpu` 与 `KFCore::face_applications_cuda`，且 DLL/header 已更新。

### Task 5: Retro 增加 KFCore faceswap 薄适配器

**Files:**

- Create: `faceswap/include/kfcore_face_application.h`
- Create: `faceswap/src/kfcore_face_application.cpp`
- Modify: `faceswap/include/network_manager.h`
- Modify: `faceswap/src/network_manager.cpp`
- Modify: `faceswap/include/faceswap_config.h`
- Modify: `faceswap/src/faceswap_config.cpp`
- Modify: `faceswap/tests/test_model_config.cpp`
- Create: `faceswap/tests/test_kfcore_face_application.cpp`

**Steps:**

1. 添加 provider/model-path 映射测试：CPU 解析 ONNX 根，CUDA 解析 engine profile；DirectML fail fast。
2. 添加领域转换测试：bbox、68/5 点、512 维 embedding、age/gender 结果保持。
3. 运行 faceswap 测试，确认红测。
4. 实现一个 move-only 适配器，以 `std::variant`/Pimpl 持有 CPU 或 CUDA KFCore application；不暴露 KFCore 后端类型给业务层。
5. `NetworkManager` 只拥有该适配器，提供 `AnalyzeAll`、`SwapPrepared` 与 readiness/error 接口。
6. 运行新增测试并确认无 fallback。

### Task 6: 迁移 faceswap 仓库内调用方

**Files:**

- Modify: `faceswap/src/source_processor.cpp`
- Modify: `faceswap/src/target_processor.cpp`
- Modify: `faceswap/src/batch_processor.cpp`
- Modify: `faceswap/src/face_detection_facade.cpp`
- Modify: `Retro/manager/src/face_service.cpp`
- Modify: `Retro/snap/src/template_preprocessor.cpp`
- Modify: `Retro/snap/src/snap_controller.cpp`
- Modify: related headers/tests under `faceswap/tests`, `Retro/manager/tests`, `Retro/snap/tests`

**Steps:**

1. 为 source analyze、目标 prepared swap、多脸顺序和无脸结果添加/更新行为测试。
2. 运行相关 tests，确认它们仍依赖旧单模型 accessors。
3. 逐调用点迁移到 `AnalyzeAll`/`SwapPrepared`，保持排序、选择和 cache 数据结构。
4. 删除 pool/accessor 调用；并行批次按 worker 使用独立 `NetworkManager`，不得共享非重入实例。
5. 运行 faceswap、manager、snap 相关测试。

### Task 7: 迁移 image_mcp 的目录、多脸和缓存流程

**Files:**

- Modify: `image_mcp/src/service.cpp`
- Modify: `image_mcp/include/image_mcp/service.hpp`
- Modify: `image_mcp/tests/*` 中分析、缓存、搜索、换脸相关测试

**Steps:**

1. 添加/更新测试：多脸 `Analyze` 数量与顺序不变、缓存 embedding 可直接换脸、连续多脸换脸、enhance 开关。
2. 运行 image_mcp 最小相关测试，确认红测。
3. `Analyze` 改用 `NetworkManager::AnalyzeAll` 并转换到现有 `ImageAnalysis`。
4. `ExtractEmbeddingAt` 使用分析结果选择指定 bbox 对应人脸，不直接调用模型类。
5. 单脸和多脸换脸改用 `SwapPrepared`；保留缓存、目录搜索、匹配和事件语义。
6. 运行 image_mcp library/server tests。

### Task 8: 移除 Retro 直接模型实现与依赖

**Files:**

- Modify: `faceswap/CMakeLists.txt`
- Modify: `faceswap/tests/CMakeLists.txt`
- Modify: `image_mcp/CMakeLists.txt`
- Modify: project runtime staging CMake only where faceswap/image_mcp currently require ONNX Runtime
- Remove from build/source ownership: direct ONNX model implementations under `faceswap/src` and their obsolete headers/tests

**Steps:**

1. 添加 CMake/依赖检查测试或 configure 断言，要求 faceswap/image_mcp 链接 KFCore applications 且不链接 ONNX Runtime/Eigen。
2. 运行 configure/检查，确认当前失败。
3. 将 source glob 改为明确源清单，链接 `KFCore::face_applications_cpu`、`KFCore::face_applications_cuda` 与必要 image processor/core targets。
4. 移除 `onnxruntime`、`Eigen3::Eigen`、ORT include/link directories 和 faceswap 目标上的 ORT runtime selection。
5. 删除已无调用者的单模型、pool 和 ONNX helper 编译入口；用 `rg.exe` 确认生产调用为零。
6. 重新 configure 并构建 faceswap/image_mcp。

### Task 9: 端到端验证与残余依赖审计

**Files:**

- Verify only: both repositories and installed KFCore package

**Steps:**

1. KFCore：运行 face model core、CPU/CUDA face model/application tests。
2. Retro：运行 faceswap、image_mcp、manager/snap 相关 tests。
3. 用真实单脸/双脸输入分别执行 CPU 与 CUDA analyze/swap；记录模型加载、分析、prepared swap、GFPGAN 和总耗时。
4. `rg.exe` 审计 faceswap/image_mcp CMake 与生产源，不得残留直接 Ort/Eigen/model class 依赖。
5. 检查 `git diff --check`、工作区状态与 `.codegraph/` 未跟踪情况，只报告本任务文件和已有脏改动的边界。
