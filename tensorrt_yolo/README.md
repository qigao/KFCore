# TensorRT YOLO 与 ByteTrack

本模块把可信的 TensorRT YOLO engine 接到 KFCore 自有 ByteTrack。它使用独立的
`KFCore::image_processor` 完成通用 CUDA 图像到 Tensor 处理，并提供三个 YOLO target：
`KFCore::yolo_tracking`、`KFCore::tensorrt_yolo` 与
`KFCore::yolo_opencv`。默认 KFCore C 构建不会发现 CUDA、TensorRT 或 OpenCV。

## 运行契约

只加载部署方生成或经可信渠道认证的序列化 engine。engine 必须有一个 FP16 或 FP32、NCHW
三通道 `images` 输入，并精确匹配以下一种最终检测输出契约：

- EfficientNMS：`num_dets`（INT32）、`boxes`、`scores` 和 `labels`（INT32）。`boxes` 与
  `scores` 为 FP16 或 FP32 且必须同型；输入与输出浮点类型可以不同。`num_dets` 可使用
  `[batch]` 或 `[batch, 1]`，后一种形式的尾维必须固定为 1。
- Compact NMS：单个 FP16 或 FP32 `output0`，shape 必须为 `[batch, max_detections, 6]`，列语义
  固定为 `left, top, right, bottom, score, class_id`。模型图内必须已完成 NMS；`score == 0`
  的全零行作为 padding 跳过。

名称可在 `TensorNames` 中显式覆盖，其中 Compact NMS 名称字段为 `detections`。不合约的 engine
会失败，绝不改走 raw-head 解码、CPU NMS、ONNX Runtime 或 OpenCV DNN。

全部输入输出张量都必须使用线性、非向量化的标量物理布局：TensorRT format 为 `kLINEAR`、
vectorized dimension 为 `-1`、components per element 为 `1`，且 bytes per component 与逻辑
dtype 一致。engine 必须只有 profile 0；`images` 可在 profile 0 中动态改变 batch、height 和
width，其余轴与所有输出的非 batch 轴必须固定。`DetectorOptions::input_size` 按
`{height, width}` 指定尺寸并必须落在 profile 0 的 min/max 范围内；未指定时使用 opt H/W。
缓冲区按 profile max shape 分配，每次推理按实际 batch 与选定 H/W 设置输入 shape。

`ImageView` 是借用视图：host 或同 CUDA device 的输入内存必须在 `detect()` 或
`detect_batch()` 返回前持续有效；返回后 detector 不再保留该视图。一个 `Engine` 可被多个
worker 共享，但每个 worker 必须拥有自己的 `TensorRtDetector`；同一 detector 不可并发调用。
由 `cv::Mat` 创建视图时，`Mat` 的释放、重分配或 backing storage 改变同样会使该视图失效。
检测器把 YOLO 视图适配到 `KFCore::image_processor`，处理器不包含 TensorRT 类型，也不保留输入、
工作区、输出 Tensor 或 stream。支持范围和独立使用方式见 `image_processor/README.md`。

每条摄像头/图片序列拥有一个顺序调用的 `ByteTrackSession`。跟踪按类别隔离，公开 ID 为
`(uint64_t(class_id) << 32) | uint32_t(local_tracker_id)`；`reset()` 清空全部类别状态，之后
允许重新使用 ID。每帧上限由 `EngineOptions::{max_batch,max_detections,max_input_bytes,max_output_bytes}`
与 `ByteTrackOptions::{max_detections_per_frame,max_class_trackers}` 控制。超过任一上限直接报错，
不会缩小 batch、丢弃检测或降低分辨率。

没有视频文件解码（opencv-lite 不含 videoio）、raw-head 解码、ReID 或运行时 fallback。
`track_image_sequence` 只处理目录内的图片并要求 `--engine`、`--images` 和 `--output`。

## 构建和部署

在 x64 VS 2022 Developer Command Prompt 中执行。预设本身不记录 TensorRT SDK 的本机路径；
full profile 从继承环境读取 `TENSORRT_ROOT`。opencv-lite 使用 `$env{PKG_ROOT}/opencv-lite`，
full profile 的运行时 `PATH` 已加入其 `bin`，不会复制 DLL。
缺少 `TENSORRT_ROOT`、目录不存在，或缺少所需 TensorRT headers/libraries 时，full configure
会立即失败，绝不改用系统 SDK 或其他推理后端。

```powershell
cmake --fresh --preset win-dev-user
cmake --build --preset win-dev-user
ctest --preset win-dev-user

cmake --fresh --preset win-yolo-tracking-dev-user
cmake --build --preset win-yolo-tracking-dev-user
ctest --preset win-yolo-tracking-dev-user
cmake --build --preset install-win-yolo-tracking-dev-user

$env:OPENCV_LITE_ROOT = 'C:/path/to/opencv-lite'
cmake --fresh --preset win-yolo-tracking-dev-user -DKFCORE_BUILD_YOLO_OPENCV=ON
cmake --build --preset win-yolo-tracking-dev-user --target test_yolo_opencv

$env:TENSORRT_ROOT = 'C:/path/to/TensorRT'
cmake --fresh --preset win-yolo-release-user
cmake --build --preset win-yolo-release-user
ctest --preset win-yolo-release-user
cmake --build --preset install-win-yolo-release-user
```

GPU 集成测试还必须显式启用并提供可信 engine：

```powershell
$env:TENSORRT_ROOT = 'C:/path/to/TensorRT'
$env:OPENCV_LITE_ROOT = 'C:/path/to/opencv-lite'
$env:KFCORE_TENSORRT_TEST_ENGINE = 'C:/path/to/yolo11n-efficientnms.engine'
$env:KFCORE_TENSORRT_TEST_ENGINE_YOLO11_FACE = 'C:/path/to/yolov11n-face-efficientnms.engine'
cmake --fresh --preset win-yolo-release-user -DKFCORE_BUILD_TENSORRT_INTEGRATION_TESTS=ON
cmake --build --preset win-yolo-release-user
ctest --preset win-yolo-release-user -R '^test_tensorrt_integration(_yolo11_face)?$'
```

`KFCORE_TENSORRT_TEST_ENGINE` 是 configure-time `FILEPATH` cache 变量（同名环境变量仅用于初始化
它）。启用集成测试时，空路径、不存在的路径或目录都会在 configure 阶段失败；验证后的规范路径
会显式写入 `test_tensorrt_integration` 的 CTest environment。

`KFCORE_TENSORRT_TEST_ENGINE_YOLO11_FACE` 也是 configure-time `FILEPATH` cache 变量，同名环境
变量仅用于初始化。它是可选缓存：空值明确表示不参加 face 验证，且不会注册额外 CTest；非空值若
不是现有普通文件，则在 configure 阶段失败。其值有效时，会登记
`test_tensorrt_integration_yolo11_face`；该测试复用既有的 `test_tensorrt_integration` 二进制，
只把已验证的 face engine 路径作为该测试的 `KFCORE_TENSORRT_TEST_ENGINE` 环境变量注入。因此两个
CTest 名称使用相同测试逻辑，但分别加载各自已验证的 engine。

只验证缓存注册和环境注入时，可复现地运行：

```powershell
ctest --preset win-yolo-tracking-dev-user -R '^test_tensorrt_integration_engine_config$' --output-on-failure
```

TensorRT/CUDA DLL 由部署环境提供；安装包不复制它们。若没有与目标 GPU/TensorRT 版本匹配的
可信 engine，GPU 推理测试是明确阻塞项，不能用任意 engine 或其他推理后端代替。

## YOLOv8 场景应用

`yolov8_domain_demo` 为以下本地模型提供统一的图片目录和 USB 摄像头应用：

| `--application` | ONNX 输入 | 类别 |
|---|---:|---|
| `drone` | `1x3x640x640` | `class-0`、`class-1`（模型没有提供更具体的类别名） |
| `football` | `1x3x960x960` | `ball`、`goalkeeper`、`player`、`referee` |
| `parking` | `1x3x640x640` | `space-empty`、`space-occupied` |

CPU 路线使用 ONNX Runtime 和 CPU ImageProcessor；GPU 路线使用 TensorRT 和 CUDA
ImageProcessor。两条路线产生同一个 `DetectionFrame`，之后共用 KFCore 自有
`ByteTrackSession`、类别校验、统计和 UI。`--backend` 是强制选择项，加载或推理失败不会切换
到另一条路线。

```powershell
$env:TENSORRT_ROOT = 'C:/projects/TensorRT-11.2.1.2'
cmake --fresh --preset win-yolov8-applications-release-user
cmake --build --preset win-yolov8-applications-release-user --target yolov8_domain_demo
ctest --preset win-yolov8-applications-release-user -R '^test_yolo_domain_' --output-on-failure

# preset 的环境只作用于 CMake/CTest 子进程；直接启动 exe 前显式设置 DLL 搜索路径
$pkgRoot = 'C:/projects/cpp/external/pkgs'
$appBin = "$PWD/build/Msvc-YOLOv8-Applications/bin"
$env:PATH = "$pkgRoot/onnxruntime/lib;$env:TENSORRT_ROOT/bin;$env:CUDA_PATH_V12_8/bin;" +
            "$pkgRoot/opencv-lite/bin;$appBin;$pkgRoot/turboparser/release/bin;" +
            "$pkgRoot/turboutils/release/bin;$pkgRoot/turbonet/release/bin;$env:PATH"

# CPU：有界图片目录处理
build/Msvc-YOLOv8-Applications/bin/yolov8_domain_demo.exe `
  --application football --backend cpu `
  --model C:/projects/cpp/KFCore/yolo-models/yolov8n-football.onnx `
  --images C:/absolute/input --output C:/absolute/output --max-frames 100

# 查询摄像头及 mode id；此命令不加载模型
build/Msvc-YOLOv8-Applications/bin/yolov8_domain_demo.exe --list-cameras

# GPU：默认严格选择 640x480@30 NV12；Q/Esc 退出，R 清空跟踪状态
build/Msvc-YOLOv8-Applications/bin/yolov8_domain_demo.exe `
  --application parking --backend tensorrt `
  --model C:/absolute/yolov8n-parking.engine --camera 0 --mirror

# 无窗口采样必须给定边界
build/Msvc-YOLOv8-Applications/bin/yolov8_domain_demo.exe `
  --application drone --backend cpu `
  --model C:/projects/cpp/KFCore/yolo-models/yolov8n-drone.onnx `
  --camera 0 --headless --max-frames 300
```

没有精确 NV12 mode 时程序直接报错；可以先用 `--list-cameras` 查出 mode，再通过
`--mode <id>` 显式使用 I420、RGB24 或 BGRA。MJPEG 会列出但不解码。摄像头回调使用有界的
latest-frame mailbox，推理跟不上采集时覆盖尚未消费的旧帧，并在 `coalesced` 中明确计数。

TensorRT engine 必须在部署机器上由可信 ONNX 生成。TensorRT 11.2 的 strongly typed 构建示例：

```powershell
C:/projects/TensorRT-11.2.1.2/bin/trtexec.exe `
  --onnx=C:/projects/cpp/KFCore/yolo-models/yolov8n-parking.onnx `
  --saveEngine=C:/absolute/yolov8n-parking.engine --stronglyTyped
```

最终控制台输出和窗口叠加层包含 model load、capture wait、pixel conversion、detect、track、
render/output 与 total 的 current/mean/P50/P95。`detect` 包含 letterbox/归一化、后端推理和
Compact NMS 解码；图片模式的 `render` 还包含编码写盘，摄像头模式包含绘制与窗口提交。
这些阶段用于同一输入、同一模型下比较 CPU/GPU；不能把不同输入尺寸的 football 与其他模型的
数字直接当作后端差异。

## YOLO11 / YOLO11-face 验证记录

**事实（2026-08-26 ImageProcessor 重构前基线）**：通用 `yolo11n` engine 与
`yolov11n-face` engine 均在真实的
TensorRT 11.2 / CUDA 12.8 环境执行。face engine 直接运行既有集成测试为 8/8 cases、25 assertions；
独立 CTest `test_tensorrt_integration_yolo11_face` 为 1/1；相邻 TensorRT、tracking、OpenCV、
CUDA buffer、engine-file 与 CMake 配置测试合计 66 cases、2087 assertions；合并的 focused 范围为
74 cases、2112 assertions。

**事实（2026-08-26 ImageProcessor 重构验证）**：CUDA 12.8 standalone preset 的 CPU/CUDA 与
安装消费端 ImageProcessor CTests 为 3/3；TensorRT 11.2 release preset 的非 opt-in 全套 CTests 为 19/19，包含
安装消费端、YOLO helper、tracking、OpenCV 和真实 GPU kernel 测试。此前本地 engine 随参考目录
清理，故本次重构后没有重跑需要 `.engine` 的 opt-in TensorRT runtime/YOLO11-face 测试；上段数据
只作为重构前行为基线，不作为本次 engine 级复验结果。

**事实（2026-08-27 Compact NMS 扩展验证）**：本地 `yolov12n-face.onnx` 与
`yolov8n-drone.onnx` 的 TensorRT I/O 均为 `images + output0`，输出 shape 为 `[1,300,6]`。
其中 `yolov12n-face.engine` 已在 TensorRT 11.2 / CUDA 12.8 / RTX 4060 上通过
`track_image_sequence` 执行仓库内 3 张图片并写出 3/3 张结果。测试图片不含可确认的人脸，因此
该结果只证明 GPU 预处理、Tensor 绑定、推理、Compact NMS 解码与写出链路可执行，不外推为模型
精度验证。

**边界**：`TensorRtDetector` 是最终目标检测适配器，不是任意 TensorRT 图执行器。ArcFace
`[batch,512]` embedding、年龄/性别 `[batch,2]` 分类、关键点 `[batch,42]` 等模型可使用 TensorRT
Tensor，但需要各自的输入/输出契约和后处理适配器，不能作为 YOLO 检测结果加载。

**事实（engine 契约）**：face engine 是单类别 EfficientNMS engine，张量为 `images`、`num_dets`、
`boxes`、`scores`、`labels`；`num_dets` 与 `labels` 为 INT32，`images`、`boxes`、`scores` 为 FP32，
全部为 `kLINEAR`。其动态 profile 为 min `1x3x320x320`、opt `2x3x640x640`、max
`4x3x960x960`。动态性由这些张量 shape/profile 建立；不以 raw ONNX metadata 的 `dynamic=True`
作为依据。该 engine 是为 TensorRT 11.2 和当前 RTX 4060 生成的 strongly typed FP32 本地产物，
不是可移植 fixture，且未提交。

**事实（图片序列）**：对 6 帧 `zidane` 序列，两个确认的轨迹从第 2 帧起稳定为 `id=0` 和 `id=1`。
**限制（事实）**：该序列复用静态图片，不能覆盖运动、遮挡或重新关联；这些场景仍需独立的时序素材
验证。

### YOLO11-face 来源与本地产物边界

**事实**：验证所用源模型由用户在本地 `yolo-models/yolov11n-face.pt` 提供。上游项目为
[`akanametov/yolo-face`](https://github.com/akanametov/yolo-face)，发布资产的规范 URL 为
[`yolov11n-face.pt`](https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.pt)，
源码许可证见 [upstream LICENSE](https://github.com/akanametov/yolo-face/blob/dev/LICENSE)。本仓库不
复制、编译或链接任何该上游源码。

**边界**：`.pt`、raw/转换后的 `.onnx`、TensorRT `.engine`、下载或生成的图片及渲染输出均为本地
验证产物，不提交。此次验证未单独确认发布资产的权重再分发授权；部署或再分发前必须自行核实适用
条款，本文不声明任何权利。

OpenCV-only 入口仅启用 `KFCore::yolo_opencv` 和 `KFCore::yolo_tracking`；它仍要求显式开启
tracking 与有效 `OPENCV_LITE_ROOT`，但不会启用 CUDA/TensorRT detector，也不会提供
`track_image_sequence`。adapter 与已安装的 KFCore package 仅要求 OpenCV Lite 的 `core`、
`imgproc`；只有 `track_image_sequence` 额外要求 `imgcodecs`。不发现或链接 `dnn`、`highgui`
或 `videoio`。
