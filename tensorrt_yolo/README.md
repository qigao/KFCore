# TensorRT YOLO 与 ByteTrack

本模块把可信的 TensorRT EfficientNMS engine 接到 KFCore 自有 ByteTrack。它由三个可选
CMake target 组成：`KFCore::yolo_tracking`、`KFCore::tensorrt_yolo` 与
`KFCore::yolo_opencv`。默认 KFCore C 构建不会发现 CUDA、TensorRT 或 OpenCV。

## 运行契约

只加载部署方生成或经可信渠道认证的序列化 engine。首期 engine 必须有一个 NCHW 三通道
`images` 输入，以及 `num_dets`（INT32）、`boxes`、`scores` 和 `labels`（INT32）四个输出。
`images` 可独立为 FP16 或 FP32；`boxes` 和 `scores` 也可为 FP16 或 FP32，但二者必须同型。
输入与 EfficientNMS 浮点输出不要求同型。名称可在 `TensorNames` 中显式覆盖。不合约的
engine 会失败，绝不改走 raw-head 解码、CPU NMS、ONNX Runtime 或 OpenCV DNN。
`num_dets` 可使用 `[batch]`，也可使用 TensorRT EfficientNMS 的 `[batch, 1]`；后一种形式的尾维必须固定为 1。

五个张量都必须使用线性、非向量化的标量物理布局：TensorRT format 为 `kLINEAR`、
vectorized dimension 为 `-1`、components per element 为 `1`，且 bytes per component 与逻辑
dtype 一致。engine 必须只有 profile 0；`images` 可在 profile 0 中动态改变 batch、height 和
width，其余轴与所有输出的非 batch 轴必须固定。`DetectorOptions::input_size` 按
`{height, width}` 指定尺寸并必须落在 profile 0 的 min/max 范围内；未指定时使用 opt H/W。
缓冲区按 profile max shape 分配，每次推理按实际 batch 与选定 H/W 设置输入 shape。

`ImageView` 是借用视图：host 或同 CUDA device 的输入内存必须在 `detect()` 或
`detect_batch()` 返回前持续有效；返回后 detector 不再保留该视图。一个 `Engine` 可被多个
worker 共享，但每个 worker 必须拥有自己的 `TensorRtDetector`；同一 detector 不可并发调用。
由 `cv::Mat` 创建视图时，`Mat` 的释放、重分配或 backing storage 改变同样会使该视图失效。

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
$env:KFCORE_TENSORRT_TEST_ENGINE = 'C:/trusted/yolo-efficientnms.engine'
cmake --fresh --preset win-yolo-release-user -DKFCORE_BUILD_TENSORRT_INTEGRATION_TESTS=ON
cmake --build --preset win-yolo-release-user
ctest --preset win-yolo-release-user
```

`KFCORE_TENSORRT_TEST_ENGINE` 是 configure-time `FILEPATH` cache 变量（同名环境变量仅用于初始化
它）。启用集成测试时，空路径、不存在的路径或目录都会在 configure 阶段失败；验证后的规范路径
会显式写入 `test_tensorrt_integration` 的 CTest environment。

TensorRT/CUDA DLL 由部署环境提供；安装包不复制它们。若没有与目标 GPU/TensorRT 版本匹配的
可信 engine，GPU 推理测试是明确阻塞项，不能用任意 engine 或其他推理后端代替。

OpenCV-only 入口仅启用 `KFCore::yolo_opencv` 和 `KFCore::yolo_tracking`；它仍要求显式开启
tracking 与有效 `OPENCV_LITE_ROOT`，但不会启用 CUDA/TensorRT detector，也不会提供
`track_image_sequence`。adapter 与已安装的 KFCore package 仅要求 OpenCV Lite 的 `core`、
`imgproc`；只有 `track_image_sequence` 额外要求 `imgcodecs`。不发现或链接 `dnn`、`highgui`
或 `videoio`。
