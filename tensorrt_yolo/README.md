# TensorRT YOLO 与 ByteTrack

本模块把可信的 TensorRT EfficientNMS engine 接到 KFCore 自有 ByteTrack。它由三个可选
CMake target 组成：`KFCore::yolo_tracking`、`KFCore::tensorrt_yolo` 与
`KFCore::yolo_opencv`。默认 KFCore C 构建不会发现 CUDA、TensorRT 或 OpenCV。

## 运行契约

只加载部署方生成或经可信渠道认证的序列化 engine。首期 engine 必须有一个 NCHW 三通道
`images` 输入，以及 `num_dets`（INT32）、`boxes`（FP32/FP16）、`scores`（与 boxes
同类型）和 `labels`（INT32）四个输出；名称可在 `TensorNames` 中显式覆盖。不合约的
engine 会失败，绝不改走 raw-head 解码、CPU NMS、ONNX Runtime 或 OpenCV DNN。

`ImageView` 是借用视图：host 或同 CUDA device 的输入内存必须在 `detect()` 或
`detect_batch()` 返回前持续有效；返回后 detector 不再保留该视图。一个 `Engine` 可被多个
worker 共享，但每个 worker 必须拥有自己的 `TensorRtDetector`；同一 detector 不可并发调用。

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

```powershell
cmake --fresh --preset win-dev-user
cmake --build --preset win-dev-user
ctest --preset win-dev-user

cmake --fresh --preset win-yolo-tracking-dev-user
cmake --build --preset win-yolo-tracking-dev-user
ctest --preset win-yolo-tracking-dev-user
cmake --build --preset install-win-yolo-tracking-dev-user

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

TensorRT/CUDA DLL 由部署环境提供；安装包不复制它们。若没有与目标 GPU/TensorRT 版本匹配的
可信 engine，GPU 推理测试是明确阻塞项，不能用任意 engine 或其他推理后端代替。
