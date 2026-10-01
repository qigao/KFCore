# YOLO CPU/CUDA 后端

YOLO 按职责拆成三个可安装 target：

- `KFCore::yolo_core`：`ImageView`、检测结果、共享 compact-NMS 解码和 ByteTrack；
- `KFCore::yolo_onnx`：ONNX Runtime CPU detector；
- `KFCore::yolo_tensorrt`：TensorRT/CUDA detector。

生产 target 不依赖 OpenCV，也不公开 `cv::Mat` adapter。`yolov8_domain_demo` 使用 stb
处理图片、Turbo Capture 采集摄像头，并在 Windows 使用 Win32 GDI 显示；其他模块自己的
demo/UI 可以独立选择 OpenCV，但 OpenCV 不进入 YOLO 导出依赖。

## 选择后端

CPU 路线使用正式 API `kfcore/yolo/onnx.hpp`：

```cpp
#include <kfcore/yolo/onnx.hpp>

auto detector = kfcore::yolo::OnnxDetector::load(model_path);
kfcore::yolo::DetectionFrame detections = detector->detect(image);
```

```cmake
target_link_libraries(my_app PRIVATE KFCore::yolo_onnx)
```

CUDA 路线使用 `kfcore/yolo/tensorrt.hpp`：

```cpp
#include <kfcore/yolo/tensorrt.hpp>

auto engine = kfcore::yolo::Engine::load(engine_path);
auto detector = engine->create_detector();
kfcore::yolo::DetectionFrame detections = detector->detect(image);
```

```cmake
target_link_libraries(my_app PRIVATE KFCore::yolo_tensorrt)
```

两条路线均返回相同的 `DetectionFrame`，之后可交给 `ByteTrackSession`。后端在链接和对象
构造时显式选择；模型加载或推理失败不会切换后端。

## 输入与所有权

`ImageView` 是同步借用视图。CPU detector 只接受 Host 内存；TensorRT detector 接受 Host
或匹配 CUDA device 的内存。调用返回后 detector 不再保留输入指针。支持 BGR8、RGB8、
NV12、I420、NV21、YUY2 和 UYVY；预处理分别由 `image_processor_cpu` 与
`image_processor_cuda` 完成。

ModelPackage artifact `flavor` 决定 detector tensor contract：

- `compact-nms`：`[1,N,6]` compact NMS 输出；
- `efficient-nms`：TensorRT EfficientNMS 四输出；
- `raw-yolo`：`[1,4+C,A]`、class-score-only 的 channel-major raw head；
- `raw-yolox`：解码后的 YOLOX `[1,A,5+C]`，score 使用
  `objectness * class_confidence`。

`raw-yolox` 不是 `raw-yolo` 的别名。它还固定 YOLOX inference preprocessing：
BGR channel order、0..255 FP32 值域、114 border，以及左上对齐 letterbox。现有其他
flavor 继续使用 RGB、0..1、居中 letterbox，因此不会因 YOLOX 支持改变既有模型语义。

用于 `raw-yolox` 的 ONNX 必须已经解码 grid/stride，使前四列为输入图像坐标系中的
`cx,cy,w,h`。使用官方 Megvii YOLOX exporter 时应显式启用
`--decode_in_inference`；未解码的官方 ONNX 输出需要先按 YOLOX 官方 postprocess
处理，不能仅通过改 artifact flavor 强行加载。

YOLOX 官方代码与模型发布链采用 Apache-2.0，可作为 #236 fixed-detector
qualification 的候选；detector 模型、配置、源 commit 和 license 仍必须作为独立
provenance 记录，不能归入 KFCore relation checkpoint 的许可证元数据。

## 构建与验证

仅使用标准 preset：

```powershell
cmake --preset win-release-user
cmake --build --preset win-release-user
ctest --preset win-release-user -R "test_yolo_" --output-on-failure
```

模型根固定为进程启动工作目录下的 `./yolo-models`；TensorRT engine 使用
`tensorrt/${KFCORE_TENSORRT_ENGINE_PROFILE}/<model>.engine`，profile 只在 CMake 配置阶段
设置并编译进库。ONNX detector 通过
`KFCore::runtime_onnx` 执行，TensorRT detector 使用 CUDA 图像处理与 TensorRT SDK；两条路线
没有自动 fallback，模型根不接受 CMake、Preset JSON、环境变量或 API 配置。

## TensorRT 8.6 Pascal engine

以下命令必须在匹配的 GTX 1060 或 GTX 1070 主机执行。先选择与 KFCore preset 完全相同的
profile；不得在其他 GPU 上生成后改名。示例从仓库根使用现有部署模型目录：

```powershell
$env:TENSORRT_ROOT = 'C:\projects\TensorRT-8.6.1'
$env:CUDNN_ROOT = 'C:\projects\cpp\external\pkgs\cudnn-8.9.7-cuda12'
$env:PATH = "$env:TENSORRT_ROOT\lib;$env:TENSORRT_ROOT\bin;$env:CUDNN_ROOT\bin;$env:CUDA_PATH_V12_8\bin;$env:PATH"
$models = (Resolve-Path "$PWD/build/Msvc-Release/bin/yolo-models").Path
$device = 0
$profile = 'gtx1060-sm61-trt8.6.1-default' # GTX 1070 使用 gtx1070-sm61-trt8.6.1-default
$expectedGpu = 'GTX 1060' # GTX 1070 profile 同时改为 GTX 1070
$engines = Join-Path $models "tensorrt/$profile"
$trtexec = Join-Path $env:TENSORRT_ROOT 'bin/trtexec.exe'

$gpu = nvidia-smi --id=$device --query-gpu=name,compute_cap --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0 -or $gpu -notmatch "$([regex]::Escape($expectedGpu)).*,\s*6\.1$") {
    throw "Expected $expectedGpu compute capability 6.1 on device $device; got: $gpu"
}
if (Test-Path -LiteralPath $engines) { throw "Refusing to overwrite profile: $engines" }
New-Item -ItemType Directory -Path $engines | Out-Null

$modelsToBuild = @(
    @{ Name = 'yolov8s'; Source = 'yolov8s.onnx' },
    @{ Name = 'yolov8n-drone'; Source = 'yolov8n-drone.onnx' },
    @{ Name = 'yolov8n-football'; Source = 'yolov8n-football.onnx' },
    @{ Name = 'yolov8n-parking'; Source = 'yolov8n-parking.onnx' }
)
foreach ($model in $modelsToBuild) {
    & $trtexec "--onnx=$(Join-Path $models $model.Source)" `
      "--saveEngine=$(Join-Path $engines ($model.Name + '.engine'))" `
      --device=$device --builderOptimizationLevel=3 --skipInference
    if ($LASTEXITCODE -ne 0) { throw "TensorRT build failed: $($model.Name)" }
    & $trtexec "--loadEngine=$(Join-Path $engines ($model.Name + '.engine'))" `
      --device=$device --iterations=1 --warmUp=0 --duration=0
    if ($LASTEXITCODE -ne 0) { throw "TensorRT smoke failed: $($model.Name)" }
}
```

`nvidia-smi` 必须报告目标型号和 compute capability `6.1`。这些模型保持 FP32 I/O；不使用
`--hardwareCompatibilityLevel=ampere+`，因为该模式不覆盖 Pascal。任一命令失败时保留失败日志，
不要把不完整 profile 发布给应用。
