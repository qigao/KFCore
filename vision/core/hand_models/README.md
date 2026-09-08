# Hand Landmark

`hand_model_core/` 提供手部几何、结果类型和 ByteTrack/Kalman 编排；`hand_models_cpu/` 与
`hand_models_cuda/` 提供显式分离的 ONNX Runtime CPU 与 TensorRT CUDA 后端。生产目标不
依赖 OpenCV，也不包含人脸检测或 FaceMesh。人脸模型属于独立的 `face_model_*` target。

## 模块与数据流

| CMake target | 输入 | 推理设备 | 图像处理 | 状态 |
|---|---|---|---|---|
| `KFCore::hand_model_core` | backend 返回的手部结果 | 无 | 共享几何/解码 | 每个 `HandPipeline` 独占一份 KFCore ByteTrack/Kalman |
| `KFCore::hand_models_cpu` | Host BGR/RGB/NV12/I420/NV21/YUY2/UYVY `ImageView` | CPU（ONNX Runtime） | `CpuImageProcessor` | 每个 backend 独占 sessions |
| `KFCore::hand_models_cuda` | Host 或 CUDA BGR/RGB/NV12/I420/NV21/YUY2/UYVY `ImageView` | CUDA（TensorRT） | `CudaImageProcessor` | 每个 backend 独占 processor、engines 与 executors |
| `KFCore::hand_interaction` | 已跟踪的 `HandFrame` | 无额外推理 | 21 点几何 primitive + THIG | 每个实例独占身份、运动历史和时序图状态 |

手部路径是：

```text
image -> Palm [N,8] -> rotated hand ROI -> 21 landmarks
      -> 42-value keypoint feature -> gesture class -> ByteTrack/Kalman
      -> HandPrimitiveExtractor -> THIG -> semantic ActionEvent
```

单 backend 直接接收 Host 图像时，每次 `infer()` 只调用一次 `stage()`。组合计算应先用
`TensorRtHandInput::prepare()` 把该帧上传一次，再把同一个 `image::FrameView` 交给一个或多个
模型 pipeline；hand backend 收到 CUDA `compute` 后直接借用，不再做 device-to-device stage。
Palm letterbox 和每个 hand ROI 都从这份共享图像生成 CUDA FP32 NCHW tensor，并在下一次复用
tensor buffer 之前同步完成推理。这里只有小型输出、解码、关键点特征和跟踪回到 CPU；不存在
CPU inference fallback。

## 严格模型契约

| 模型 | 输入 | 输出 |
|---|---|---|
| Palm | FP32 `input [1,3,192,192]` | FP32 `pdscore_boxx_boxy_boxsize_kp0x_kp0y_kp2x_kp2y [N,8]` |
| Hand landmark | FP32 `input [N,3,224,224]` | FP32 `xyz_x21 [N,63]`, `hand_score [N,1]`, `lefthand_0_or_righthand_1 [N,1]` |
| Keypoint classifier | FP32 `input [N,42]` | INT64 `class_ids [N]` |

加载时会精确验证 tensor 名称、标量类型、声明形状和动态 batch profile。Palm 的 NMS
输出是 data-dependent `[N,8]`，由 `max_palm_candidates` 同时约束 engine output allocator
与 host 下载；`max_hands` 再独立限制 ROI 数量。engine/model、源图、tensor 和输出都有可配置
字节上限。契约不符、容量越界或并发重入会直接报错，不会换后端或返回部分结果。

默认手势 id 为 `0=Open`、`1=Closed`、`2=Pointer`；其他 id 返回 `Unknown`。

## 构建

CPU：

```powershell
$env:ONNXRUNTIME_ROOT = "C:\path\to\onnxruntime"
cmake --fresh --preset win-release-user
cmake --build --preset win-release-user
ctest --preset win-release-user --output-on-failure
```

CPU + GPU（同一安装包，可由应用运行时选择）：

```powershell
$env:TENSORRT_ROOT = "C:\projects\TensorRT-11.2.1.2"
$env:ONNXRUNTIME_ROOT = "C:\projects\cpp\external\pkgs\onnxruntime-gpu"
cmake --fresh --preset win-release-user
cmake --build --preset win-release-user
ctest --preset win-release-user --output-on-failure
```

ONNX 与 engine 从进程启动工作目录下唯一的 `./yolo-models` 加模块相对路径解析。
模型和 engine 是应用部署资产，不随 KFCore 安装。
CPU ONNX 与 CUDA engine 的固定名称分别公开在
`kfcore/hand_models/cpu_model_names.hpp` 和 `kfcore/hand_models/cuda_model_names.hpp`。其中 CPU
默认加载接口使用固定相对模型根；CUDA 再拼接 `engine_profile_directory`、CMake cache variable
`KFCORE_TENSORRT_ENGINE_PROFILE` 和对应 engine 名。底层显式路径 overload 保留给测试
与库内验证；应用运行时不读取模型环境变量，也不自动切换后端。

## 从 ONNX 生成 TensorRT engine

以下命令对应本模块验证过的 TensorRT 11.2 合约。engine 与 TensorRT 版本、GPU 架构和构建
参数相关；必须在部署环境重新生成并保留 ONNX 来源、hash 和转换命令。

```powershell
$models = (Resolve-Path "$PWD/yolo-models").Path
$profile = 'rtx4060-sm89-trt11.2.1-default'
$handEngines = Join-Path $models "hand_gesture_model/tensorrt/$profile"
$trtexec = Join-Path $env:TENSORRT_ROOT "bin\trtexec.exe"

& $trtexec --onnx="$models\hand_gesture_model\palm_detection\palm_detection_full_inf_post_192x192.onnx" `
  --saveEngine="$handEngines\palm_detection.engine" --fp16 --skipInference

& $trtexec --onnx="$models\hand_gesture_model\hand_landmark\hand_landmark_sparse_Nx3x224x224.onnx" `
  --minShapes=input:1x3x224x224 --optShapes=input:2x3x224x224 `
  --maxShapes=input:8x3x224x224 --saveEngine="$handEngines\hand_landmark.engine" `
  --fp16 --skipInference

& $trtexec --onnx="$models\hand_gesture_model\keypoint_classifier\keypoint_classifier.onnx" `
  --minShapes=input:1x42 --optShapes=input:2x42 --maxShapes=input:8x42 `
  --saveEngine="$handEngines\keypoint_classifier.engine" --fp16 --skipInference

```

若配置的 `max_hands` 大于 engine 的 classifier dynamic batch maximum，backend 会在加载时
拒绝该 engine。Palm engine 必须保留 ONNX NMS 的 data-dependent 输出，不能把它伪装成固定
候选数。

TensorRT 8.6 Pascal 使用相同 shape contract，但必须在匹配的 GTX 主机选择独立 profile，并把
TensorRT `lib` 与 cuDNN 8 加入运行时路径：

```powershell
$env:TENSORRT_ROOT = 'C:\projects\TensorRT-8.6.1'
$env:CUDNN_ROOT = 'C:\projects\cpp\external\pkgs\cudnn-8.9.7-cuda12'
$env:PATH = "$env:TENSORRT_ROOT\lib;$env:TENSORRT_ROOT\bin;$env:CUDNN_ROOT\bin;$env:CUDA_PATH_V12_8\bin;$env:PATH"
$models = (Resolve-Path "$PWD/build/Msvc-Release/bin/yolo-models").Path
$device = 0
$profile = 'gtx1060-sm61-trt8.6.1-default' # GTX 1070 使用 gtx1070-sm61-trt8.6.1-default
$expectedGpu = 'GTX 1060' # GTX 1070 profile 同时改为 GTX 1070
$handEngines = Join-Path $models "hand_gesture_model/tensorrt/$profile"
$trtexec = Join-Path $env:TENSORRT_ROOT 'bin/trtexec.exe'

$gpu = nvidia-smi --id=$device --query-gpu=name,compute_cap --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0 -or $gpu -notmatch "$([regex]::Escape($expectedGpu)).*,\s*6\.1$") {
    throw "Expected $expectedGpu compute capability 6.1 on device $device; got: $gpu"
}
if (Test-Path -LiteralPath $handEngines) { throw "Refusing to overwrite profile: $handEngines" }
New-Item -ItemType Directory -Path $handEngines | Out-Null

& $trtexec --onnx="$models/hand_gesture_model/palm_detection/palm_detection_full_inf_post_192x192.onnx" `
  --saveEngine="$handEngines/palm_detection.engine" --device=$device --fp16 --skipInference
if ($LASTEXITCODE -ne 0) { throw 'Palm engine build failed' }
& $trtexec --onnx="$models/hand_gesture_model/hand_landmark/hand_landmark_sparse_Nx3x224x224.onnx" `
  --minShapes=input:1x3x224x224 --optShapes=input:2x3x224x224 `
  --maxShapes=input:8x3x224x224 --saveEngine="$handEngines/hand_landmark.engine" `
  --device=$device --fp16 --skipInference
if ($LASTEXITCODE -ne 0) { throw 'Hand landmark engine build failed' }
& $trtexec --onnx="$models/hand_gesture_model/keypoint_classifier/keypoint_classifier.onnx" `
  --minShapes=input:1x42 --optShapes=input:2x42 --maxShapes=input:8x42 `
  --saveEngine="$handEngines/keypoint_classifier.engine" --device=$device --fp16 --skipInference
if ($LASTEXITCODE -ne 0) { throw 'Gesture classifier engine build failed' }

& $trtexec --loadEngine="$handEngines/palm_detection.engine" `
  --device=$device --iterations=1 --warmUp=0 --duration=0
if ($LASTEXITCODE -ne 0) { throw 'Palm engine smoke failed' }
& $trtexec --loadEngine="$handEngines/hand_landmark.engine" --shapes=input:2x3x224x224 `
  --device=$device --iterations=1 --warmUp=0 --duration=0
if ($LASTEXITCODE -ne 0) { throw 'Hand landmark engine smoke failed' }
& $trtexec --loadEngine="$handEngines/keypoint_classifier.engine" --shapes=input:2x42 `
  --device=$device --iterations=1 --warmUp=0 --duration=0
if ($LASTEXITCODE -ne 0) { throw 'Gesture classifier engine smoke failed' }
```

`nvidia-smi` 必须报告目标型号和 compute capability `6.1`。FP16 是既有 hand engine 构建
契约；Pascal 没有 Tensor Core，最终延迟必须以对应 GTX 主机的 smoke/benchmark 为准。

## API 示例与所有权

```cpp
#include <kfcore/hand_models/cpu.hpp>

kfcore::hand_models::HandOnnxModelPaths paths{
    "palm.onnx", "hand_landmark.onnx", "keypoint_classifier.onnx"};
auto backend = kfcore::hand_models::CpuHandBackend::load(paths);

kfcore::hand_models::HandPipelineOptions pipeline_options;
pipeline_options.tracker.minimum_consecutive_frames = 1;
// Host BGR8/RGB8/NV12/I420/NV21/YUY2/UYVY. The returned HandResult owns the fixed descriptor.
pipeline_options.appearance.enabled = true;
auto pipeline = kfcore::hand_models::HandPipeline::create(
    std::move(backend), pipeline_options);
const auto frame = pipeline->process(image_view);
```

TensorRT 用法只需把 paths/options 类型换为 `HandTensorRtEnginePaths` 和
`TensorRtHandOptions`。多模型处理同一帧时，调用方创建一个 `TensorRtHandInput`，每帧调用一次
`prepare(host_view)`，再把返回的 `image::FrameView` 依次交给各 pipeline。`source` 借用调用方原图；`compute` 借用 preparer 的 CUDA storage，并在下一次
`prepare()` 或 preparer 析构时失效。所有 pipeline 调用必须在失效前同步完成。返回值拥有全部
结果。一个实例不允许重入，不同实例可以并行并拥有独立 tracker/session/executor 状态。

CPU 和 TensorRT backend 都产生相同的 `HandFrame`，复杂手势统一进入
`KFCore::hand_interaction`，不会在两个推理后端各维护一套时序规则。完整动作、容量、外部
区域观察与 reset 契约见 [hand_interaction/README.md](../hand_interaction/README.md)。

`appearance.enabled` 默认关闭。启用后，内置提取器从 `image::FrameView::source` 的 Host
BGR8/RGB8/NV12/I420/NV21/YUY2/UYVY 原图与 landmark 生成 256 维六分区描述子：掌心 96 维，拇指、食指、
中指、无名指和小指各 32 维。`valid_parts` 和 `quality` 分别表示分区有效掩码与采样覆盖率。若传入
CUDA-device source、Gray8、非法 stride/byte size，则在模型推理前返回 `InvalidArgument`，不会静默
下载或切换路径。`compute` 可以是同帧的 CUDA view，因此 USB capture 的 TensorRT demo 能在共享一次
上传的同时继续使用同一 CPU Re-ID 提取器。对没有 Host source 的纯 device 流，调用方应保持该选项
关闭，直到提供经过验证的 CUDA/learned descriptor producer。

旧的掌心专用 96 维契约不再兼容：没有 kind 标记、转换器或双读路径。所有使用
`HandAppearanceDescriptor`、`HandResult` 或 `HandPipelineOptions` 的 C++ 下游必须用当前头文件重新编译。

## 耗时记录

`HandFrame::timings` 分别记录 `preprocess_ms`、Palm、landmark、classifier、`appearance_ms`、tracking 和
`total_ms`。GPU 计时包含同步的
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
