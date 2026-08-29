# Hand 与 MediaPipe Landmark

`vision_models/` 提供同一套手部几何、结果类型和 ByteTrack/Kalman 编排，以及显式分离的
ONNX Runtime CPU 与 TensorRT CUDA 后端。它不依赖 OpenCV，也不引入 MediaPipe face
detector：现有 YOLOv12-face 仍负责产生人脸框，468 点模型只消费调用方给出的 `RectF`。

## 模块与数据流

| CMake target | 输入 | 推理设备 | 图像处理 | 状态 |
|---|---|---|---|---|
| `KFCore::vision_model_core` | backend 返回的手部结果 | 无 | 共享几何/解码 | 每个 `HandPipeline` 独占一份 KFCore ByteTrack/Kalman |
| `KFCore::vision_models_cpu` | Host BGR/RGB/Gray `ImageView` | CPU（ONNX Runtime） | `CpuImageProcessor` | 每个 backend 独占 sessions |
| `KFCore::vision_models_tensorrt` | Host 或 CUDA BGR/RGB/Gray `ImageView` | GPU（TensorRT） | `CudaImageProcessor` | 每个 backend 独占 processor、engines 与 executors |
| `KFCore::hand_interaction` | 已跟踪的 `HandFrame` | 无额外推理 | 21 点几何 primitive + THIG | 每个实例独占身份、运动历史和时序图状态 |

手部路径是：

```text
image -> Palm [N,8] -> rotated hand ROI -> 21 landmarks
      -> 42-value keypoint feature -> gesture class -> ByteTrack/Kalman
      -> HandPrimitiveExtractor -> THIG -> semantic ActionEvent
```

GPU backend 每次 `infer()` 只调用一次 `stage()`。Host 图像在此处上传；CUDA 图像在同一
device 上复制到 processor-owned storage。Palm letterbox 和每个 hand ROI 都从这份 staged
image 生成 CUDA FP32 NCHW tensor，并在下一次复用 tensor buffer 之前同步完成推理。这里只有
小型输出、解码、关键点特征和跟踪回到 CPU；不存在 CPU inference fallback。

人脸路径是：

```text
YOLOv12-face/caller RectF -> expanded square ROI -> MediaPipe 468 landmarks
```

## 严格模型契约

| 模型 | 输入 | 输出 |
|---|---|---|
| Palm | FP32 `input [1,3,192,192]` | FP32 `pdscore_boxx_boxy_boxsize_kp0x_kp0y_kp2x_kp2y [N,8]` |
| Hand landmark | FP32 `input [N,3,224,224]` | FP32 `xyz_x21 [N,63]`, `hand_score [N,1]`, `lefthand_0_or_righthand_1 [N,1]` |
| Keypoint classifier | FP32 `input [N,42]` | INT64 `class_ids [N]` |
| MediaPipe face landmark | FP32 `image [1,3,192,192]` | FP32 `scores [1]`, `landmarks [1,468,3]` |

加载时会精确验证 tensor 名称、标量类型、声明形状和动态 batch profile。Palm 的 NMS
输出是 data-dependent `[N,8]`，由 `max_palm_candidates` 同时约束 engine output allocator
与 host 下载；`max_hands` 再独立限制 ROI 数量。engine/model、源图、tensor 和输出都有可配置
字节上限。契约不符、容量越界或并发重入会直接报错，不会换后端或返回部分结果。

默认手势 id 为 `0=Open`、`1=Closed`、`2=Pointer`；其他 id 返回 `Unknown`。默认 face
landmark 模型输出归一化坐标；若部署的是像素坐标模型，必须显式设置
`face_coordinates_normalized=false`。

## 构建

CPU：

```powershell
$env:ONNXRUNTIME_ROOT = "C:\path\to\onnxruntime"
cmake --preset win-vision-models-cpu-release-user
cmake --build --preset win-vision-models-cpu-release-user
ctest --preset win-vision-models-cpu-release-user --output-on-failure
```

GPU：

```powershell
$env:TENSORRT_ROOT = "C:\projects\TensorRT-11.2.1.2"
cmake --preset win-vision-models-tensorrt-release-user
cmake --build --preset win-vision-models-tensorrt-release-user
ctest --preset win-vision-models-tensorrt-release-user --output-on-failure
```

真实模型 integration tests 默认只在对应 preset 中启用，并要求 cache 中的 ONNX/engine/图片
都是存在的绝对路径。模型和 engine 是部署资产，不会安装或提交到 KFCore。

## 从 ONNX 生成 TensorRT engine

以下命令对应本模块验证过的 TensorRT 11.2 合约。engine 与 TensorRT 版本、GPU 架构和构建
参数相关；必须在部署环境重新生成并保留 ONNX 来源、hash 和转换命令。

```powershell
$models = "C:\path\to\yolo-models"
$engines = "C:\path\to\engines"
$trtexec = Join-Path $env:TENSORRT_ROOT "bin\trtexec.exe"

& $trtexec --onnx="$models\hand_gesture_model\palm_detection\palm_detection_full_inf_post_192x192.onnx" `
  --saveEngine="$engines\palm.engine" --fp16 --skipInference

& $trtexec --onnx="$models\hand_gesture_model\hand_landmark\hand_landmark_sparse_Nx3x224x224.onnx" `
  --minShapes=input:1x3x224x224 --optShapes=input:2x3x224x224 `
  --maxShapes=input:8x3x224x224 --saveEngine="$engines\hand_landmark.engine" `
  --fp16 --skipInference

& $trtexec --onnx="$models\hand_gesture_model\keypoint_classifier\keypoint_classifier.onnx" `
  --minShapes=input:1x42 --optShapes=input:2x42 --maxShapes=input:8x42 `
  --saveEngine="$engines\keypoint_classifier.engine" --fp16 --skipInference

& $trtexec --onnx="$models\MediaPipeFaceLandmarkDetector.onnx" `
  --saveEngine="$engines\face_landmark.engine" --fp16 --skipInference
```

若配置的 `max_hands` 大于 engine 的 classifier dynamic batch maximum，backend 会在加载时
拒绝该 engine。Palm engine 必须保留 ONNX NMS 的 data-dependent 输出，不能把它伪装成固定
候选数。

## API 示例与所有权

```cpp
#include <kfcore/vision_models/cpu.hpp>

kfcore::vision_models::HandOnnxModelPaths paths{
    "palm.onnx", "hand_landmark.onnx", "keypoint_classifier.onnx"};
auto backend = kfcore::vision_models::CpuHandBackend::load(paths);

kfcore::vision_models::HandPipelineOptions pipeline_options;
pipeline_options.tracker.minimum_consecutive_frames = 1;
// Host BGR8/RGB8 only. The returned HandResult owns the fixed descriptor.
pipeline_options.appearance.enabled = true;
auto pipeline = kfcore::vision_models::HandPipeline::create(
    std::move(backend), pipeline_options);
const auto frame = pipeline->process(image_view);
```

TensorRT 用法只需把 paths/options 类型换为 `HandTensorRtEnginePaths` 和
`TensorRtVisionOptions`。输入 bytes 只借用到同步调用结束；返回值拥有全部结果。一个实例不允许
重入，不同实例可以并行并拥有独立 tracker/session/executor 状态。

CPU 和 TensorRT backend 都产生相同的 `HandFrame`，复杂手势统一进入
`KFCore::hand_interaction`，不会在两个推理后端各维护一套时序规则。完整动作、容量、外部
区域观察与 reset 契约见 [hand_interaction/README.md](../hand_interaction/README.md)。

`appearance.enabled` 默认关闭，以保持 CUDA-device `ImageView` 调用和成本不变。启用后，内置提取器从
同步调用期间借用的 Host BGR8/RGB8 原图与 landmark 生成 256 维六分区描述子：掌心 96 维，拇指、食指、
中指、无名指和小指各 32 维。`valid_parts` 和 `quality` 分别表示分区有效掩码与采样覆盖率。若传入
CUDA-device、Gray8、非法 stride/byte size，则在模型推理前返回 `InvalidArgument`，不会静默下载或
切换路径。USB capture 的 CPU/TensorRT demo 输入都是 Host BGR，因此显式启用同一 CPU Re-ID 提取器。
对纯 device 流，调用方应保持该选项关闭，直到提供经过验证的 CUDA/learned descriptor producer。

旧的掌心专用 96 维契约不再兼容：没有 kind 标记、转换器或双读路径。所有使用
`HandAppearanceDescriptor`、`HandResult` 或 `HandPipelineOptions` 的 C++ 下游必须用当前头文件重新编译。

## 耗时记录

`HandFrame::timings` 分别记录 `preprocess_ms`、Palm、landmark、classifier、`appearance_ms`、tracking 和
`total_ms`；`FaceLandmarkResult` 记录 preprocess、inference 和 total。GPU 计时包含同步的
Host 上传（若有）、CUDA 图像处理、TensorRT 执行与小输出下载；CPU 计时包含 CPU 图像处理和
ONNX Runtime。模型加载不在单帧 timing 内，应在构造前后由应用另行计时。比较 CPU/GPU 时应
固定图片、阈值、warm-up 次数、设备功耗状态和 batch/profile，并分别报告 P50/P95/P99，不能把
一次 integration smoke test 当成性能基准。

## 限制

- 目前 hand landmark ROI 逐个同步推理；classifier 会对有效手批处理。这样保证复用一个有界
  CUDA affine tensor buffer，后续只有 profiling 证明 landmark batching 是瓶颈时才扩展。
- `HandPipeline` 用 Palm box 作为 ByteTrack observation；它不预测或平滑 21 个关键点。
- 内置六分区描述子是轻量图像外观证据，不等价于经过手部数据训练的 OSNet/FastReID；当前仓库没有
  可验证的手部 Re-ID 权重，不能宣称跨会话或人员级身份准确率。
- 本模块不验证模型精度、训练标签来源或许可证；部署方必须对有 provenance 的 golden samples
  做 accuracy 验收。
