# Face Applications

`KFCore::face_applications_cuda` 提供 TensorRT/CUDA 单脸分析和换脸应用层。检测器固定为
[YOLOv12-face](https://github.com/akanametov/yolo-face)；本模块没有 MediaPipe、YOLO11-face
或其他 detector fallback，也不会扫描默认 weights 目录。

`KFCore::face_applications_cpu` 使用相同模型语义和 `model_matrix.bin`：12face 检测来自
`KFCore::hand_models_cpu`，其余五个 ONNX 模型来自 `KFCore::face_models_cpu`。图像处理、
仿射对齐、mask、paste 与 blend 均由 `KFCore::image_processor_cpu` 完成，不依赖 OpenCV、TensorRT
或 CUDA。两个后端由调用方在链接时显式选择，不会在运行失败后自动切换。

CPU 入口接收拥有内存的 `kfcore::image::BgrImage`，也接收 borrowed Host
`kfcore::image::ImageView`（BGR8/RGB8/NV12/I420/NV21/YUY2/UYVY）。`ImageView` 路线在入口做一次有界的
YUV/RGB→BGR 转换，然后复用既有 CPU pipeline；CUDA memory 会明确失败。应用同步且单实例
不可重入，不同任务可以各自创建实例并行运行。`analyze()` 返回检测框、68/5 点、ArcFace
embedding 和可选 Age/Gender logits；`swap()` 返回拥有像素内存的 BGR 图，
`swap_profiled()` 额外返回逐阶段 wall time。

```cpp
auto application = kfcore::face_applications::OnnxFaceSwapApplication::load(paths);
const auto result = application->swap_profiled(source_bgr, target_bgr);
```

## 数据流与状态边界

```text
ImageView source/target (borrowed BGR8/RGB8/NV12/I420/NV21/YUY2/UYVY, Host/CUDA)
  -> YOLOv12-face: highest-score class 0 face
  -> Face68: bbox affine crop -> 68 landmarks -> source coordinates
  -> five landmarks: eye averages + nose[30] + mouth[48,54]
  -> ArcFace: 112x112 alignment -> source embedding
  -> model_matrix.bin: embedding * 512x512 matrix -> L2 normalize
  -> InSwapper: target 128x128 alignment + projected embedding
  -> bounded mask and inverse-affine paste-back
  -> optional GFPGAN 512x512 alignment, paste-back, configured blend
  -> owned BgrImage output
```

`TensorRtFaceSwapApplication` 通过 `KFCore::image_processor_cuda` 为每个 CUDA device、每个分析帧
最多 stage 一次。YOLO 检测复用该 device image；Face68、ArcFace、Age/Gender、InSwapper 和
GFPGAN 从 BGR/RGB/NV12/I420/NV21/YUY2/UYVY device image 直接生成各自的 FP32 NCHW tensor。
模型本身不接收 raw YUV。InSwapper 与可选 GFPGAN 的输出、仿射 mask 合成和两模型之间的 image 保持在同一
CUDA device；第一次合成可直接采样 YUV target，并始终产生 packed BGR8。只有最终 BGR 图像
回传 host；两模型配置为不同 device 时加载即失败。

`TensorRtFaceSwapApplication` 同步且单实例不可重入。输入 `ImageView` 只在调用期间借用且
不被修改，成功结果 `BgrImage` 拥有自己的像素内存；任一阶段失败时抛出带阶段上下文的
`FaceApplicationError`，不会返回部分结果。若要并行处理，应为每个任务加载独立实例。
核心公开头文件与动态库不依赖 OpenCV；OpenCV Lite 只用于示例和真实模型测试中的图片
解码、显示与编码适配。

## 分阶段耗时

需要诊断时使用 `swap_profiled(source_view, target_view)`；返回的
`ProfiledFaceSwapResult` 同时包含拥有像素内存的 `image` 和 `FaceSwapTimingReport`。
参数、线程约束、错误条件与 `swap()` 相同：

```cpp
const kfcore::face_applications::ProfiledFaceSwapResult result =
    application->swap_profiled(source_view, target_view);
const double total_ms =
    std::chrono::duration<double, std::milli>(result.timings.total).count();
```

报告使用 `std::chrono::nanoseconds` 保存以下同步 wall time：source/target 分析中的 staging、
YOLO 检测、Face68 预处理与执行、ArcFace 预处理与执行，以及 embedding projection、
InSwapper 预处理/执行/合成。启用 Age/Gender 或 GFPGAN 时，相应
`std::optional` 字段有值；未配置模型时字段为空，而不是伪造 0 ms。

TensorRT executor 与 CUDA image processor 都在阶段调用返回前同步，因此这些数值包含等待
GPU 完成的时间，也包含 adapter 校验；为兼容已有 API，`*_inference_and_decode` 字段名保留，
但 InSwapper/GFPGAN 已不执行 host decode。终端模型的 composition 时间包含最终 BGR 下载。它们不是 CUDA kernel
独占时间，不能换算为 FLOPS、GPU 利用率或功耗；这些指标应由 Nsight Systems/Compute 或
设备遥测独立采集。`swap()` 不创建报告且不读取时钟，常规调用不会承担逐阶段计时开销。

本机 CPU/CUDA 预处理、完整 pipeline P50/P95、CPU 占用、GPU 利用率/显存/功耗的可复验记录
见 [`docs/performance/2026-08-27-face-swap-cpu-gpu.md`](../docs/performance/2026-08-27-face-swap-cpu-gpu.md)。

## 模型契约

应用使用语义名称而不是上游文件名识别模型角色。CPU ONNX 与 CUDA engine 的固定物理名称分别
定义在 `kfcore/face_models/cpu_model_names.hpp` 和
`kfcore/face_models/cuda_model_names.hpp`；模型根固定为进程启动工作目录下的
`./yolo-models`，CUDA 路径插入 CMake cache variable
`KFCORE_TENSORRT_ENGINE_PROFILE`。例如 `face_mesh_landmarker` 在 CPU 映射到
`MediaPipeFaceLandmarkDetector.onnx`，在 CUDA 映射到 `face_landmark.engine`，因此不能仅替换扩展名。

| 模型 | 默认 binding | 固定 shape / 结果 |
|---|---|---|
| YOLOv12-face | `images` / compact `output0` | 现有 `TensorRtDetector` compact-NMS contract；只选 class 0 |
| Face68 (`2dfan4`) | `input`, `landmarks_xyscore`, `heatmaps` | input `[1,3,256,256]`; landmarks `[1,68,3]` |
| ArcFace | `input.1`, `683` | `[1,3,112,112]` -> `[1,512]` |
| InSwapper | `target`, `source`, `output` | `[1,3,128,128]` + `[1,512]` -> `[1,3,128,128]` |
| GFPGAN（可选） | `input`, `output` | `[1,3,512,512]` -> `[1,3,512,512]` |
| Age/Gender（可选） | `pixel_values`, `logits` | `[1,3,224,224]` -> raw `[1,2]` logits |

`model_matrix.bin` 不是 TensorRT binding，而是 InSwapper 必需 sidecar。文件必须恰为
1,048,576 bytes（512×512 个 little-endian host `float`），所有元素必须有限；加载后由
projector 不可变拥有。模型实际 binding 与上表不一致时应通过 `FaceSwapOptions` 显式配置，
不会猜测别名。

## 从 ONNX 生成当前 GPU profile

模型根固定为仓库的 `yolo-models`；`KFCORE_TENSORRT_ENGINE_PROFILE` 在 CMake 配置阶段定义。
TensorRT 11.2 默认使用 strongly typed network；以下命令保持 adapter 要求的 FP32 I/O，并允许默认 TF32
tactics。ArcFace 与 Age/Gender 的 ONNX batch 维为动态，但应用合约固定为 batch 1：

```powershell
$models = (Resolve-Path "$PWD/yolo-models").Path
$profile = 'rtx4060-sm89-trt11.2.1-default'
$engines = Join-Path $models "tensorrt/$profile"
$trtexec = Join-Path $env:TENSORRT_ROOT 'bin/trtexec.exe'

& $trtexec --onnx="$models/2dfan4.onnx" `
  --saveEngine="$engines/2dfan4.engine" --skipInference --builderOptimizationLevel=3
& $trtexec --onnx="$models/arcface_w600k_r50.onnx" `
  --saveEngine="$engines/arcface_w600k_r50.engine" `
  --minShapes=input.1:1x3x112x112 --optShapes=input.1:1x3x112x112 `
  --maxShapes=input.1:1x3x112x112 --skipInference --builderOptimizationLevel=3
& $trtexec --onnx="$models/inswapper_128.onnx" `
  --saveEngine="$engines/inswapper_128.engine" --skipInference --builderOptimizationLevel=3
& $trtexec --onnx="$models/gfpgan_1.4.onnx" `
  --saveEngine="$engines/gfpgan_1.4.engine" --skipInference --builderOptimizationLevel=3
& $trtexec --onnx="$models/age-gender.onnx" `
  --saveEngine="$engines/age-gender.engine" `
  --minShapes=pixel_values:1x3x224x224 --optShapes=pixel_values:1x3x224x224 `
  --maxShapes=pixel_values:1x3x224x224 --skipInference --builderOptimizationLevel=3
```

**事实（2026-08-30，RTX 4060 Laptop / SM89 / TensorRT 11.2.1）**：

| engine | 构建时间 | engine 大小 | 构建峰值 GPU allocator |
|---|---:|---:|---:|
| `2dfan4.engine` | 40.6 s | 142.2 MiB | 141 MiB |
| `arcface_w600k_r50.engine` | 24.6 s | 245.3 MiB | 245 MiB |
| `inswapper_128.engine` | 41.5 s | 618.4 MiB | 617 MiB |
| `gfpgan_1.4.engine` | 121.5 s | 761.9 MiB | 891 MiB |
| `age-gender.engine` | 11.2 s | 329.7 MiB | 329 MiB |

`model_matrix.bin` 从 `inswapper_128.onnx` 的 `buff2fs [512,512]` FP32 initializer 导出，
不是通过 TensorRT 生成。本次导出结果为 1,048,576 bytes，SHA-256 为
`370af5bf707dafdbea8a40448d697d9697610bd223ecf92887af9c9cc7055ac8`。

## 构建

Windows 核心库使用调用方提供的 TensorRT；构建图片 CLI/桌面 demo 时额外使用 OpenCV Lite：

```powershell
$env:TENSORRT_ROOT = 'C:\projects\TensorRT-11.2.1.2'
$env:OPENCV_LITE_ROOT = 'C:\projects\cpp\external\pkgs\opencv-lite'
cmake --fresh --preset win-release-user
cmake --build --preset win-release-user
ctest --preset win-release-user -R '^test_face_' --output-on-failure
```

TensorRT runtime、face models、TensorRT YOLO 和 face applications 均随 KFCore 构建；
`BUILD_EXAMPLES=ON` 只控制 CLI/demo。OpenCV Lite 只链接到示例和相应测试目标，
不会出现在 `KFCore::face_applications_cuda` 的公开依赖中。

无 GPU 的 CPU 路线使用同一标准 preset；`ONNXRUNTIME_ROOT` 必须包含匹配版本的 `include/`、
import library 和 runtime DLL：

```powershell
$env:ONNXRUNTIME_ROOT = 'C:\projects\cpp\external\pkgs\onnxruntime'
cmake --fresh --preset win-release-user
cmake --build --preset win-release-user
ctest --preset win-release-user --output-on-failure
```

Windows 部署时应把 finder 选中的 `onnxruntime.dll` 放在最终可执行文件目录。仅修改 `PATH`
不足以覆盖 System32 中可能存在的同名旧版本 DLL；真实模型集成测试会为其测试目标做 app-local
staging。真实模型集成测试还需显式启用对应选项，并提供六个 ONNX、矩阵和两张测试图；配置错误会立即失败。

## 命令行应用

detector、Face68、ArcFace、InSwapper、matrix、GFPGAN 和 Age/Gender 都从固定模型根/profile
的固定相对路径解析；命令行只接收输入与输出图片：

```powershell
face_swap_image.exe `
  --source source.jpg --target target.jpg --output swapped.png
```

缺少参数、重复参数、未知参数和不可读输入/模型文件会在加载 engine 前失败。应用只在完整换脸
成功后调用 `imwrite`；`--output` 不得与 source/target 使用同一路径。

## 桌面 Demo

`face_swap_demo` 使用相同参数启动 OpenCV Lite 窗口，并显示 source、target 和 TensorRT
输出三联视图：

```powershell
face_swap_demo.exe `
  --source source.jpg --target target.jpg --output swapped.png
```

- `R`：使用已加载的 engine 重新运行
- `S`：将当前结果写入 `--output`
- `Q` / `Esc` / 关闭窗口：退出

窗口会在 engine 加载前显示状态；推理保持同步。每次首次运行或按 `R` 重跑后，底部显示各
阶段的本次耗时以及最近 120 次样本的 nearest-rank P50/P95；未启用的可选阶段显示 `--`。
应用使用固定 GFPGAN engine，不会自动搜索模型、转换 ONNX 或降级到 CPU。

## 真实模型 opt-in 测试

配置下列任一相对路径时，必须同时提供全部必需项；所有值都从
固定模型根解析，路径完整后集成测试自动注册：

- `KFCORE_FACE_APPLICATION_TEST_MATRIX_RELATIVE`
- `KFCORE_FACE_APPLICATION_TEST_SOURCE_IMAGE_RELATIVE`
- `KFCORE_FACE_APPLICATION_TEST_TARGET_IMAGE_RELATIVE`

detector、Face68、ArcFace、InSwapper、GFPGAN 与 Age/Gender 都使用当前 engine profile 下的
固定文件名；其中 Face68 使用由 `2dfan4.onnx` 生成且满足 Face68 binding 合约的
`2dfan4.engine`。验收内容
包括真实 I420 输入、TensorRT 执行成功、owned BGR 输出可重新编码/解码、输出尺寸等于 target，
source/target 内存不变，以及必需计时为正、可选计时与模型配置一致；这不是模型 accuracy 或
身份相似度的 golden test。
