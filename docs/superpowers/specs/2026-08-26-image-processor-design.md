# ImageProcessor 独立模块设计

日期：2026-08-26
状态：已实现；engine 级复验待重新提供本地可信 `.engine`

## 1. 背景与目标

现有 TensorRT YOLO 预处理由 `tensorrt_yolo/src/detector_helpers.cpp`、
`tensorrt_yolo/src/detector.cpp` 和 `tensorrt_yolo/src/letterbox.cu` 共同完成。
其中图像校验、host 行打包、CUDA 上传、双线性 letterbox、通道转换、归一化以及
HWC 到 NCHW 转换均不依赖 TensorRT，却被编译进 TensorRT 专用目标，其他推理后端无法复用。

本次新增独立 `KFCore::image_processor` 模块，并让 TensorRT YOLO 成为其首个调用方。
首期目标是无行为变化地迁移现有 BGR8/RGB8 到 CUDA FP16/FP32 NCHW Tensor 的处理路径。

首期不实现视频解码、摄像头采集、颜色空间全覆盖、CPU resize、AprilTag GPU 检测或
推理 runtime 抽象。Gray8、NV12、YUY2 和图像到图像输出属于后续按设备需求扩展的格式能力；
未知格式必须 fail fast，不做隐式 OpenCV fallback。

## 2. 架构与依赖

```text
USB/文件/网络解码器                   GPU producer
       | Host ImageView                    | CudaDevice ImageView
       +------------------+----------------+
                          v
                 KFCore::image_processor
                 |- validate/plan (CPU)
                 |- pack host rows (CPU)
                 `- letterbox/normalize/NCHW (CUDA)
                          |
                          v
                   caller-owned TensorView
                    /          |          \
              TensorRT      ONNX RT      other runtime
```

依赖方向为 `tensorrt_yolo -> image_processor -> CUDA Runtime`。`image_processor` 不包含、
链接或公开任何 TensorRT/OpenCV 类型。OpenCV 仍只是 host `ImageView` 的可选适配器。

构建开关 `KFCORE_BUILD_IMAGE_PROCESSOR` 允许单独构建模块；
`KFCORE_BUILD_TENSORRT_YOLO=ON` 隐式要求并构建该模块，保持既有 TensorRT preset 可用。

## 3. 公开契约

命名空间为 `kfcore::image`，安装目标为 `KFCore::image_processor`。

- `PixelFormat`：首期 `Bgr8`、`Rgb8`。
- `MemoryKind`：`Host`、`CudaDevice`。
- `TensorElementType`：`Float16`、`Float32`。
- `TensorLayout`：首期仅 `Nchw`。
- `ImageView`：借用单平面图像指针、实际容量、宽高、行跨度、格式和内存位置。
- `TensorView`：借用连续输出指针、容量、N/C/H/W、元素类型、布局和内存位置。
- `PreprocessOptions`：输出通道顺序、逐通道 mean/stddev、0..255 border value。
- `ImageProcessor::plan()`：纯 CPU 校验和有界字节规划，返回逐图 transform 与工作区需求。
- `ImageProcessor::stage_host_inputs()`：将 host 输入逐行打包到调用方提供的 pinned host 工作区。
- `ImageProcessor::enqueue()`：上传已打包 host 数据，并将 host/device 输入转换到调用方 Tensor。

Tensor 输出公式为 `(pixel / 255 - mean[channel]) / stddev[channel]`。
resize 使用现有像素中心映射与双线性采样，letterbox 居中填充。

## 4. 内存与并发协议

数据单位是一次 batch：输入 `ImageView[]`、一个连续 NCHW `TensorView`、一块 pinned host
工作区和一块 CUDA device 工作区。输入和 Tensor 是调用方的事实源，模块不缓存或保留任何指针。

- Host 输入必须保持有效至 `stage_host_inputs()` 返回；之后异步处理只依赖 pinned 工作区。
- pinned host 工作区、device 工作区、device 输入和输出 Tensor 必须保持有效，且不得复用，
  直到传入 CUDA stream 上的工作完成。
- `plan()` 返回的字节数是容量下界；Image、工作区和 Tensor 视图都携带实际容量并在 CPU
  读取或入队前逐项校验。所需 source span 按 `(height - 1) * stride + width * 3` 计算。
- API 无内部可变状态。不同调用可并发执行，但调用方必须为重叠任务提供互不冲突的工作区、
  输出和 stream；同一 stream 上也不得提前覆盖 pinned host 工作区。
- CPU 规划和打包阶段失败时没有外部副作用；CUDA enqueue 中途失败时调用方应同步该 stream
  后再释放或复用相关内存。

增长上限由 `plan(..., max_source_bytes, max_tensor_bytes)` 显式提供；加法和乘法均检查溢出。
算法复杂度为规划/打包 `O(source bytes)`，CUDA 转换 `O(batch * output width * output height)`；
内核外额外空间等于 host 输入的紧凑字节总数。

## 5. 错误语义与兼容性

`ImageProcessorError` 携带 `InvalidArgument`、`ResourceLimitExceeded` 或 `CudaFailure`。
不支持的枚举、空指针、非法尺寸、跨度不足、容量不足、非 CUDA 输出、设备不匹配和 CUDA
调用失败均立即报错。模块不记录热路径日志。

TensorRT YOLO 的公开 `ImageView`、`PixelFormat`、`MemoryKind` 和 `YoloError` 保持不变。
适配层把 YOLO 视图转换为 image_processor 视图，并把处理器错误映射回原有 `YoloErrorCode`，
因此现有调用代码、异常类型和检测结果不变。旧 YOLO 公开视图没有容量字段，适配层只声明按其
尺寸/stride 算出的最小 source span；其既有“调用方保证 backing storage 足够”契约不变。现有内部
letterbox 实现删除，避免双事实源。

## 6. 验证范围

- CPU：transform、batch/枚举/跨度/溢出/容量校验、混合 Host/CUDA 规划、带 padding 行打包。
- CUDA：BGR/RGB、输出 RGB/BGR 顺序、FP32/FP16、非整数双线性缩放、显式 stream、混合输入。
- TensorRT：helper 单测、GPU 集成测试以及 YOLO11-face engine 路径。
- 构建/安装：独立 image_processor preset 路径、TensorRT preset、安装导出 consumer。

回滚方式是恢复 TensorRT 目标中的原 `letterbox.cu` 和 detector staging 调用；不涉及数据迁移或
持久化格式变化。
