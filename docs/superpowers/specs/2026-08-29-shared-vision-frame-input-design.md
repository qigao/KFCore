# KFCore 单次视觉输入与共享图像设计

## 背景

现有 `HandPipeline` 与 `FaceMeshPipeline` 都接受借用的 `ImageView`。CPU 组合调用会共享同一
host 指针，但 TensorRT hand、face detector 与 face landmarker 各自决定是否上传，导致同一帧
在多模型组合中可能发生多次 host-to-device 或 device-to-device 复制。

## 决策

新增只含借用视图的 `image::FrameView`：

- `source` 是调用方拥有的权威原图；手部 appearance 等必须读取原始 host 像素的阶段使用它。
  它可以是 BGR8、RGB8、NV12 或 I420，不为推理预先物化 BGR 副本。
- `compute` 是与 `source` 内容、尺寸和像素格式一致的计算视图；CPU 指向同一 host 图像，
  TensorRT 指向一次上传得到的 CUDA 图像。
- `TensorRtHandInput::prepare()` 每帧只执行一次 `CudaImageProcessor::stage()`，返回上述两个视图。
  NV12/I420 保留原像素格式与紧密 plane 布局，后续 detector/landmarker 的 affine
  预处理在 CUDA kernel 内按需采样 YUV。
- hand、face detector 与 face landmarker 都只借用 `compute` 完成同步推理；TensorRT backend 收到
  CUDA `compute` 后不得再次 `stage()`。
- 既有 `process(ImageView)` 接口保留，并等价于 `image::FrameView::borrow(image)`，避免破坏现有调用方。

## 所有权与生命周期协议

- 帧像素只有调用方一个 owner；`image::FrameView` 不拥有、retain 或释放像素。
- CPU：`source` 与 `compute` 是同一个 borrowed host view，在同步 `process()` 返回时借用结束。
- GPU：`source` 借用调用方 host 图像；`compute` 借用 `TensorRtHandInput` 的 CUDA storage。
- GPU `compute` 在下一次 `prepare()` 或 `TensorRtHandInput` 析构时失效。调用方必须完成该帧所有
  同步 pipeline 调用后才能准备下一帧，不得跨线程并发复用同一 preparer。
- backend 返回值完全拥有其结果，不保存任何输入指针。

## 错误语义

- `source`/`compute` 指针、尺寸或像素格式不一致时，在 pipeline 边界 fail fast，抛出
  `HandModelError(InvalidArgument)`。
- CUDA device、容量或上传失败沿用现有 `HandModelError` 映射，不做 CPU fallback。
- appearance 启用时 `source` 必须是合法 host BGR8/RGB8/NV12/I420；NV12/I420
  只对 descriptor 采样点执行 YUV-to-RGB，不生成整帧 BGR 副本。

## 采集与显示边界

- `LatestFrameMailbox` 拥有容量有界的唯一 host 帧副本；消费线程从
  `CapturedFrame::pixels` 构造只读 `ImageView`，不再复制。
- RGB24、NV12 和 I420 可直接映射为推理视图。BGRA 不在 ImageProcessor 公开格式内，
  demo 在应用边界显式转换为 BGR，不作静默 fallback。
- UI 的 BGR 物化发生在 hand、FaceMesh 与 THIG 同步处理完成之后；该副本只供
  overlay 和 `imshow` 使用，不反向成为模型输入。

## 并发与状态归属

每个 pipeline 与 `TensorRtHandInput` 都由一个 orchestration thread 独占。共享仅发生在同一帧的
顺序同步计算中，不跨线程、不跨帧缓存，也不把不同裁剪或不同 serial 的图像视为同一帧。

## 兼容性与验证

- 公开 API 只有新增类型与重载，原接口、结果格式、backend 选择和错误码保持兼容。
- 单元测试验证 compute/source 路由、形状不一致拒绝、CUDA compute 不再 stage、
  NV12/I420 CUDA stage/affine 与 CPU 参考结果一致，以及 appearance 对等价 RGB/YUV 的容差。
- TensorRT 集成测试继续验证 host/device 输入结果等价；组合 demo 与 Retro Release 构建验证消费端。
