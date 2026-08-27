# TensorRT YOLO 与 KFCore ByteTrack 集成设计

日期：2026-08-25
状态：已实现并验证

## 1. 背景与目标

KFCore 需要新增一个基于 TensorRT 的 YOLO 检测模块，并将检测结果交给仓库自有的 ByteTrack/Kalman 实现完成逐帧跟踪。

设计参考来源：

- `TensorRT-YOLO/`：仅参考其 engine/context 分层、GPU letterbox、命名张量推理和批处理方法。该目录是 GPL-3.0 代码，不复制其源码。
- `TensorRT-YOLO11/`：仅参考其检测结果到 ByteTrack 的逐帧数据流。该目录是 MIT 代码，不采用其 Eigen Kalman 或 C++ ByteTrack 实现。
- `trackers/` 与 `kalman/`：跟踪和状态估计的唯一实现来源。
- NVIDIA TensorRT 官方 C++ API：新推理后端的接口事实源。

首期只覆盖：

- TensorRT engine 加载和执行基础层；
- 带 EfficientNMS 输出的 YOLO detect engine；
- KFCore ByteTrack 多类别跟踪会话；
- 可选 opencv-lite 图像适配与绘制；
- 单元测试、可选 GPU 集成测试和图片序列示例。

首期不覆盖分类、OBB、分割、姿态、raw YOLO head 解码、跨摄像头 ReID、视频文件解码或 Kalman-only 预测框输出。

## 2. 许可证决策

KFCore README 声明 modified BSD-3-Clause；本地 `TensorRT-YOLO` 使用 GPL-3.0，`TensorRT-YOLO11` 使用 MIT。

本设计采用 clean-room 路线：

- 新实现放入 KFCore 自有 `tensorrt_yolo/` 模块；
- 不复制、改写或编译 `TensorRT-YOLO/` 中的 GPL 源码；
- 两个上游目录不进入新目标的 include path、source list 或链接依赖；
- 算法和 API 行为以官方 TensorRT 文档、engine 契约及 KFCore 测试为准。

## 3. 架构与状态所有权

数据流如下：

```text
ImageView
   |
   v
ImageProcessor ---> TensorRtDetector ---> DetectionFrame
                           |
                           v
                    ByteTrackSession
                    |- class 0 tracker
                    |- class 1 tracker
                    `- class N tracker
                           |
                           v
                       TrackFrame
                           |
                           `-- optional OpenCV adapter/renderer
```

### 3.1 `KFCore::tensorrt_yolo`

职责：

- 读取受信来源的序列化 TensorRT engine；
- 校验 engine 输入输出契约；
- 通过独立 `KFCore::image_processor` 管理 CUDA 预处理，并管理 TensorRT I/O、推理和后处理；
- 返回原图坐标系中的 `DetectionFrame`。

所有权：

- `Engine` 唯一拥有 TensorRT runtime 和反序列化 engine；
- `Engine` 是只读共享对象；
- 每个 `TensorRtDetector` 独占 execution context、CUDA stream、I/O 缓冲和预处理工作区；
- 一个 detector 实例不允许并发调用；调用方通过多个 detector 共享同一 `Engine` 实现多线程推理。

这避免共享可变 execution context，同时保证模型权重只保留一份。

公开生命周期接口采用显式工厂，避免构造完成前暴露半初始化对象：

```cpp
class Engine {
public:
    static std::shared_ptr<const Engine> load(
        const std::filesystem::path& engine_path,
        const EngineOptions& options
    );

    std::unique_ptr<TensorRtDetector> create_detector(
        const DetectorOptions& options
    ) const;
};

class TensorRtDetector {
public:
    DetectionFrame detect(const ImageView& image);
    std::vector<DetectionFrame> detect_batch(
        const std::vector<ImageView>& images
    );
};
```

`Engine::load()` 成功后对象才可见；`create_detector()` 每次创建独立 context/stream/buffer。`detect_batch()` 返回结果数量和输入数量严格一致，空 batch 是非法参数。

### 3.2 `KFCore::yolo_tracking`

职责：

- 将 `DetectionFrame` 转换为 KFCore `detection_t`；
- 按 `class_id` 隔离 ByteTrack 状态；
- 推进空类别帧；
- 恢复检测输入顺序；
- 返回跨类别唯一轨迹 ID。

所有权：

- 每个摄像头或视频流拥有一个 `ByteTrackSession`；
- session 独占其所有 `bytetrack_t`；
- session 是顺序调用、非线程安全对象；
- `reset()` 是唯一清空跟踪状态和重新开始 ID 周期的入口。

### 3.3 `KFCore::yolo_opencv`

职责仅限：

- 从 `cv::Mat` 创建零拷贝 `ImageView`；
- 绘制框、类别、置信度和轨迹 ID；
- 为示例读取图片或图片序列。

核心类型不暴露 `cv::Mat`、`cv::Rect` 或其他 OpenCV 类型。

## 4. 公开数据契约

### 4.1 图像

```cpp
enum class PixelFormat {
    Bgr8,
    Rgb8,
};

enum class MemoryKind {
    Host,
    CudaDevice,
};

struct ImageView {
    const void* data;
    std::int32_t width;
    std::int32_t height;
    std::size_t row_stride;
    PixelFormat pixel_format;
    MemoryKind memory_kind;
};
```

约束：

- `data` 非空；
- 宽高为正；
- `row_stride >= width * 3`，且乘法先做溢出检查；
- 首期只接受 8 位三通道 BGR/RGB；
- `CudaDevice` 指针必须属于 detector 配置的 CUDA device；
- 输入内存在 `detect()` 返回前必须保持有效。

### 4.2 检测与跟踪结果

```cpp
struct BoxF {
    float left;
    float top;
    float right;
    float bottom;
};

struct Detection {
    BoxF box;
    float score;
    std::int32_t class_id;
};

struct DetectionFrame {
    std::int32_t image_width;
    std::int32_t image_height;
    std::vector<Detection> detections;
};

struct TrackedDetection {
    Detection detection;
    std::optional<std::uint64_t> track_id;
};

struct TrackFrame {
    std::int32_t image_width;
    std::int32_t image_height;
    std::vector<TrackedDetection> detections;
};
```

结果框使用原图像素坐标。未成熟或未匹配检测用 `std::nullopt` 表达，不使用看似有效的占位 ID。

## 5. TensorRT engine 契约

首期只支持已经集成 EfficientNMS、可直接返回最终检测结果的序列化 engine。2026-08-27 的
Compact NMS 扩展在不改变公开检测调用的前提下，另行接受图内已完成 NMS 的 `[B,N,6]` 最终检测
输出；完整约束见 `2026-08-27-tensorrt-compact-nms-design.md`。

默认张量名：

- 输入：`images`；
- EfficientNMS 输出：`num_dets`、`boxes`、`scores`、`labels`；或
- Compact NMS 输出：`output0`，列为 `left, top, right, bottom, score, class_id`。

张量名称允许通过 `EngineOptions::tensor_names` 显式覆盖。初始化时校验：

- 恰好一个图像输入，并精确匹配 EfficientNMS 四输出或 Compact NMS 单输出；
- 图像输入是 rank-4 NCHW，channel 固定为 3，dtype 为 FP32 或 FP16；
- EfficientNMS 的 `num_dets` 和 `labels` 为 INT32，`boxes` 和 `scores` 为 FP32 或 FP16 且同型；
- Compact NMS 输出为 FP32 或 FP16，shape 严格为 `[batch, max_detections, 6]`；
- I/O mode、名称、dtype 和 rank；
- batch、动态维度和 optimization profile；
- EfficientNMS 的 `boxes` 最后一维为 4，四个输出的 batch 和最大检测数一致；
- Compact NMS 的 batch 与输入 profile 一致，最大检测数固定且受资源上限约束；
- 所有维度到字节数的计算无整数溢出；
- 配置的输入尺寸位于 engine profile 范围内。

任何不匹配都抛出 `EngineContractMismatch`，不回退到 raw head 解码、CPU NMS、ONNX Runtime 或 OpenCV DNN。

运行时使用 TensorRT 10.x/11.x 共有的命名张量 API、`setTensorAddress()`、`enqueueV3()` 和 64 位维度，不使用旧 bindings-array API：

- <https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-8x-to-10x-c-api.html>
- <https://docs.nvidia.com/deeplearning/tensorrt/latest/api/c-api.html>

序列化 engine 是可执行产物，只允许加载由部署方构建或通过可信、已认证渠道获得的文件：

- <https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-c-api.html>

## 6. 预处理、推理与坐标恢复

预处理流程固定为：

1. 校验 `ImageView`；
2. 由 `KFCore::image_processor` 将 host 输入打包并异步复制到 detector 工作区，或直接读取同
   device 的 CUDA 输入；
3. 由同一处理器在 detector stream 上执行 letterbox、BGR/RGB 排列、归一化和 NCHW 输出；
4. 设置动态输入 shape 和全部命名张量地址；
5. `enqueueV3()`；
6. 仅将已验证契约对应的四个 EfficientNMS 输出或一个 Compact NMS 输出复制回 host；
7. 同步 detector stream；
8. 按该图自己的 letterbox transform 恢复原图坐标。

每张图保存独立 transform，批处理不得共享首张图的缩放或 padding 参数。非正方形输入是强制测试用例。

默认归一化为 `value / 255.0f`，边界填充值、均值和标准差均由配置给出。标准差必须为有限非零值。

## 7. ByteTrack 适配与 C API 增强

现有 `bytetrack_update()` 的返回值 `0` 同时可能表示无输出、非法输入或分配失败，无法满足 fail-fast 错误语义；同时结果没有原始 detection 索引。

新增兼容 API：

```c
typedef struct tracked_detection_ex {
    tracked_detection_t tracked;
    size_t detection_index;
} tracked_detection_ex_t;

int bytetrack_update_ex(
    bytetrack_t* tracker,
    const detection_t* detections,
    size_t detection_count,
    tracked_detection_ex_t* output,
    size_t output_capacity,
    size_t* output_count
);
```

语义：

- 成功返回 `0`，并写入 `output_count`；
- 非法参数和分配失败返回明确的非零 tracker 状态码；
- `detection_index` 是输入数组索引；
- 输出容量不足是错误，不截断；
- 所有内部新轨迹分配失败必须传播，不能忽略；
- 失败时 tracker 状态保持不变，不允许只推进 Kalman 预测或 age 后返回错误；
- 旧 `bytetrack_update()` 保持签名和既有输出顺序，内部委托 `_ex`。

为满足失败原子性，`_ex` 分为两阶段：

1. 校验所有计数乘法、参数与输出容量，预留最坏情况下的 track 容量，并完成本帧全部 scratch 分配；此阶段不改变 tracker；
2. 所有资源就绪后才执行 predict、两阶段关联、update、增删 track 和 age 推进；提交阶段不再发生可失败分配。

若第一阶段失败，局部资源统一释放，原 tracker 的 ID、Kalman 状态、age 和 track 集合均不改变。

`ByteTrackSession::update()`：

1. 校验 detection 有限值、坐标顺序、分数范围和非负类别；
2. 按类别稳定分组并保留原始索引；
3. 对本帧缺席但已存在的类别执行空更新；
4. 对每个类别调用 `bytetrack_update_ex()`；
5. 按原始索引恢复整个 frame 的顺序；
6. 将局部 ID 编码为全局 ID。

全局 ID 计算：

```text
global_id = (uint64(class_id) << 32) | uint32(local_tracker_id)
```

输入约束为 `class_id >= 0`、`local_tracker_id >= 0`，两者均不超过 32 位。该 ID 在同一 session、同一次 reset 周期内跨类别唯一。`reset()` 后允许复用。

## 8. 错误模型与资源上限

新 C++ API 使用带错误码的 `YoloError` 异常：

- `InvalidArgument`；
- `FileIo`；
- `EngineDeserialize`；
- `EngineContractMismatch`；
- `TensorRtFailure`；
- `CudaFailure`；
- `TrackerAllocationFailure`；
- `ResourceLimitExceeded`。

错误信息包含操作、失败阶段、相关张量或输入摘要以及底层错误码。错误只在适配边界记录一次；库本身不在每层重复写日志。

空检测是成功结果。推理或 tracker 失败不得返回空列表伪装成功。

所有可增长结构设置上限：

- 每帧最大检测数；
- 每个 session 最大类别 tracker 数；
- 最大 batch；
- engine profile 允许的最大输入尺寸；
- 单次 I/O 缓冲字节数。

超过上限直接失败，不自动缩小 batch、丢弃检测或降低输入分辨率。

## 9. 构建与依赖

新增选项：

- `KFCORE_BUILD_YOLO_TRACKING=OFF`；
- `KFCORE_BUILD_IMAGE_PROCESSOR=OFF`；
- `KFCORE_BUILD_TENSORRT_YOLO=OFF`；
- `KFCORE_BUILD_YOLO_OPENCV=OFF`；
- `KFCORE_BUILD_TENSORRT_INTEGRATION_TESTS=OFF`。

`KFCORE_BUILD_YOLO_TRACKING` 只启用公共类型和跟踪会话，依赖 C++17 与 `KFCore::trackers`，不查找 CUDA、TensorRT 或 OpenCV。`KFCORE_BUILD_IMAGE_PROCESSOR` 可独立启用 CUDA 图像处理而不查找 TensorRT。`KFCORE_BUILD_TENSORRT_YOLO` 自动构建 ImageProcessor，且它与 `KFCORE_BUILD_YOLO_OPENCV` 均要求 tracking 已启用。

默认 C-only KFCore 构建不变。启用 TensorRT YOLO 后：

- `enable_language(CXX)` 和 `enable_language(CUDA)`；
- C++17、CUDA17；
- `find_package(CUDAToolkit REQUIRED)`；
- 从 `TENSORRT_ROOT` 查找官方 headers、`nvinfer` 和 `nvinfer_plugin`；
- 要求 TensorRT 10.x 或 11.x；
- 缺失或版本不支持时在 configure 阶段失败。

opencv-lite 通过 `OPENCV_LITE_ROOT` 配置，不在通用 CMake 中写死开发机绝对路径。目标依赖：

- 适配/绘制：`opencv_core`、`opencv_imgproc`；
- 图片示例：额外 `opencv_imgcodecs`；
- 不链接 `opencv_dnn`、ONNX Runtime 或 `opencv_highgui`；
- 当前包没有 `opencv_videoio`，因此不提供视频文件解码示例。

TensorRT 与 CUDA 运行时依赖由部署环境提供；opencv-lite DLL 只复制到示例和测试输出目录，不隐式安装为 KFCore 自身文件。

## 10. 测试与验证

### 10.1 无 GPU 单元测试

`test_yolo_tracking`：

- 同类别目标 ID 连续；
- 重叠但类别不同的目标不交叉关联；
- 跨类别 ID 唯一；
- tracker 内部重排后恢复原检测顺序；
- 高低置信度两阶段关联；
- 空帧推进和 lost buffer 到期；
- reset 清空状态；
- NaN、Inf、反向框、非法类别和容量上限；
- `_ex` 的成功、非法参数、容量不足和分配失败传播。

`test_tensorrt_contract` 将张量元数据校验拆为不依赖 GPU 的纯逻辑测试，覆盖名称、dtype、rank、动态维度、profile 范围和尺寸乘法溢出。

### 10.2 OpenCV 测试

`test_yolo_opencv`：

- `CV_8UC3` 连续 Mat；
- 带 stride 的 ROI；
- BGR/RGB 配置；
- 空 Mat 和不支持类型失败；
- 绘制不会越界修改 ROI 外像素。

### 10.3 GPU 集成测试

启用 `KFCORE_BUILD_TENSORRT_INTEGRATION_TESTS` 时必须设置 `KFCORE_TENSORRT_TEST_ENGINE`。缺失时 configure 失败，不跳过。

覆盖：

- 单张和 batch；
- 非正方形原图及输入尺寸；
- dynamic shape 的最小、典型和最大 profile；
- 共享 engine、独立 context；
- host 与 CUDA device 输入；
- 非法输出契约拒绝；
- 构造/销毁和 clone 生命周期。

TensorRT engine 一般不跨平台、TensorRT 版本或 GPU 任意移植，因此不把二进制 engine fixture 提交到仓库：

- <https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html>

`KFCORE_TENSORRT_TEST_ENGINE_YOLO11_FACE` 是第二个 configure-time `FILEPATH` cache：空值为明确
opt-out，不注册额外测试；非空但无效的值在 configure 阶段失败。值有效时，
`test_tensorrt_integration_yolo11_face` 复用 `test_tensorrt_integration` 二进制，并仅为该 CTest
注入已验证 face 路径的 `KFCORE_TENSORRT_TEST_ENGINE` 环境变量。这样不新增运行时路径、API 或
engine 契约。

**事实（2026-08-26 本地验证）**：通用 `yolo11n` 与单类别 `yolov11n-face` engine 均在真实
TensorRT 11.2 / CUDA 12.8 上运行。face 直接集成测试为 8/8 cases、25 assertions；单独 CTest
为 1/1；相邻范围为 66 cases、2087 assertions；合并 focused 范围为 74 cases、2112 assertions。
face engine 使用 `images`、`num_dets`、`boxes`、`scores`、`labels`，其中 `num_dets`/`labels` 为
INT32，其他为 FP32，全部 `kLINEAR`；profile 为 min `1x3x320x320`、opt `2x3x640x640`、max
`4x3x960x960`。其动态性以 tensor shapes/profile 为证据，不宣称 raw ONNX metadata 含
`dynamic=True`。该 strongly typed FP32 engine 为 TensorRT 11.2 与当前 RTX 4060 本地生成，不是
可移植 fixture，且未提交。

**事实（图片序列）**：6 帧 `zidane` 序列中两个确认轨迹自第 2 帧起保持 `id=0`、`id=1`。静态图片
重复不能覆盖运动、遮挡或重新关联，因此不把此结果外推为这些时序行为已验证。

可复现命令（本地路径须由部署者替换）：

```powershell
$env:TENSORRT_ROOT = 'C:/path/to/TensorRT'
$env:OPENCV_LITE_ROOT = 'C:/path/to/opencv-lite'
$env:KFCORE_TENSORRT_TEST_ENGINE = 'C:/path/to/yolo11n-efficientnms.engine'
$env:KFCORE_TENSORRT_TEST_ENGINE_YOLO11_FACE = 'C:/path/to/yolov11n-face-efficientnms.engine'
cmake --fresh --preset win-yolo-release-user -DKFCORE_BUILD_TENSORRT_INTEGRATION_TESTS=ON
cmake --build --preset win-yolo-release-user
ctest --preset win-yolo-release-user -R '^test_tensorrt_integration(_yolo11_face)?$'
ctest --preset win-yolo-tracking-dev-user -R '^test_tensorrt_integration_engine_config$' --output-on-failure
```

### 10.4 YOLO11-face 来源与许可证边界

**事实**：源模型由用户在本地 `yolo-models/yolov11n-face.pt` 提供；上游项目为
[`akanametov/yolo-face`](https://github.com/akanametov/yolo-face)，规范发布资产为
[`yolov11n-face.pt`](https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.pt)，
源码许可证见 [upstream LICENSE](https://github.com/akanametov/yolo-face/blob/dev/LICENSE)。本验证不复制、
编译或链接上游源码。

`.pt`、ONNX、TensorRT engine、图片和输出均为本地验证产物，不提交。本验证未建立发布资产的单独
权重再分发授权；任何部署或再分发必须先自行核实适用条款，本文不声明再分发权。

## 11. 兼容性、迁移与回滚

兼容性：

- 现有 KFCore C API 和默认构建行为不变；
- `bytetrack_update()` 保持公开签名和既有输出顺序；
- 新 C++/CUDA 目标仅在显式选项开启时出现；
- 无 TensorRT SDK 的环境可只开启 `KFCORE_BUILD_YOLO_TRACKING` 完成公共类型、tracker session 和契约纯逻辑测试；
- 不修改现有 `TensorRT-YOLO/` 和 `TensorRT-YOLO11/` 参考目录。

迁移路径：

1. 先实现 tracker `_ex` 和无 GPU 测试；
2. 实现公共类型、session 和契约验证；
3. 接入 TensorRT runtime 与 CUDA 预处理；
4. 接入 opencv-lite；
5. 使用部署环境生成的可信 engine 运行 GPU 集成测试。

回滚方式：

- 关闭 `KFCORE_BUILD_TENSORRT_YOLO` 即恢复原 C-only 构建；
- 新目标和新头文件可独立移除；
- `_ex` 是增量 API，移除新模块时可保留，不影响旧调用方。

## 12. 已知实施约束

**事实（2026-08-26）**：此前“当前开发环境尚未发现 TensorRT SDK 或可用于集成测试的 engine”的
限制已不适用；已在 TensorRT 11.2 / CUDA 12.8 与当前 RTX 4060 上完成本地 `yolo11n` 和
`yolov11n-face` 验证，具体范围见 10.3。该结论不改变部署边界：序列化 engine 与目标 GPU、
TensorRT/CUDA 版本绑定，TensorRT/CUDA DLL 仍由部署环境提供；缺少匹配且可信 engine 时，GPU
推理测试仍是明确阻塞项，不能以其他后端替代。

## 13. GitHub issue 跟踪

实施计划批准后创建一个总跟踪 issue，并按可独立验收的交付物创建子 issue：

1. ByteTrack `_ex` 状态 API、失败原子性和兼容测试；
2. 公共检测类型、多类别 `ByteTrackSession` 和无 GPU 测试；
3. TensorRT engine 契约验证、runtime/context/buffer 生命周期；
4. CUDA letterbox、批推理和坐标恢复；
5. opencv-lite 适配、绘制与图片序列示例；
6. CMake 安装导出、GPU 集成测试和部署验证。

每个 issue 必须包含：

- 关联本设计文档；
- 明确范围和不在范围内的内容；
- 前置依赖 issue；
- 验收测试及可重复命令；
- TensorRT SDK/可信 engine 等外部阻塞条件；
- 未实际通过验收前不得关闭。

不假设仓库已有特定 label；创建 issue 前先读取可用 label，只使用现有 label，避免擅自改变仓库分类体系。
