# TensorRT 人脸应用设计

## 背景与范围

KFCore 已有以下可复用能力：

- `KFCore::tensorrt_runtime`：加载并同步执行 TensorRT engine；
- `KFCore::face_models`：Face68、ArcFace、Age/Gender 的严格 tensor 合约；
- `KFCore::tensorrt_yolo`：支持 YOLOv12-face 使用的 compact `[B,N,6]` 输出；
- `KFCore::image_processor`：CUDA letterbox 与通用 NCHW 归一化；
- `KFCore::yolo_opencv`：`cv::Mat` 到 YOLO `ImageView` 的薄适配。

本次加入参考本地 `faceswap` 工程的模型应用，但不复制其 ONNX Runtime、Eigen 或图像框架。人脸检测只使用已经验证可用的 YOLOv12-face（下称 12face），不接入或研究其他 face detector。

第一阶段交付单源脸、单目标脸的完整同步流程：

```text
source BGR image
  -> 12face -> highest-score face -> Face68 -> five landmarks -> ArcFace
  -> source embedding -> InSwapper model matrix projection

target BGR image
  -> 12face -> highest-score face -> Face68 -> five landmarks
  -> InSwapper -> inverse warp + mask composition
  -> optional GFPGAN -> inverse warp + mask composition
  -> owned output BGR image
```

Age/Gender 不参与换脸决策；它作为可选的人脸分析应用返回两个 raw logits，不猜测模型未声明的标签顺序。

## 目标

- 提供 InSwapper 128 与 GFPGAN 1.4 的严格 TensorRT adapter。
- 提供 12face、Face68、ArcFace、InSwapper、可选 GFPGAN 的可复用应用外观。
- 保持现有 runtime、YOLO、face model API 与默认构建行为不变。
- 使用 `C:\projects\cpp\external\pkgs\opencv-lite` 所提供的 core/imgproc；命令行示例另用 imgcodecs。
- 对 tensor 名称、shape、dtype、输入图像、模型矩阵和资源上限 fail fast。
- 用纯几何/像素测试、adapter 合约测试和 opt-in 真 engine 测试覆盖关键行为。

## 非目标

- 不接入 MediaPipe、RetinaFace、SCRFD 或其他 face detector。
- 不在第一阶段实现多脸配对、视频时序跟踪、身份图库或 UI。
- 不自动下载模型，不在仓库提交 engine、ONNX、PT 或来源不清的二进制矩阵。
- 不把 OpenCV 加入 `tensorrt_runtime`、`face_models` 或 CUDA `image_processor` 的公开依赖。
- 不隐式选择 CPU/其他模型作为 fallback。

## 候选方案

### 方案 A：全部加入 `face_models`

优点是目标较少。缺点是 prepared-tensor adapter、图像几何、模型编排和 OpenCV 生命周期混在同一模块，令纯 tensor 用户被迫承担 OpenCV 依赖，也使单元测试难以隔离。

### 方案 B：全部加入 `image_processor`

优点是几何能力集中。缺点是现有 `image_processor` 是 CUDA、第三方中立的通用预处理层；人脸 landmark 索引、canonical points、mask 与模型特定归一化不是通用图像处理契约。

### 方案 C：严格 adapter + 独立应用层（采用）

- `face_models` 新增 InSwapper/GFPGAN prepared-tensor adapter 与模型矩阵投影器；
- `face_applications` 依赖 OpenCV Lite、12face detector 与 `face_models`，负责图像级编排；
- 公共应用外观接受 `cv::Mat`，明确表明这是 OpenCV adapter/application 边界；
- 内部的几何、预处理和合成函数可独立测试，不依赖 engine。

该方案保持依赖单向：

```text
TensorRT runtime <- face_models <- face_applications
image_processor  <- TensorRT YOLO <- face_applications
OpenCV Lite      <- yolo_opencv   <- face_applications
```

## 模块与公开接口

### `KFCore::face_models`

新增常量和结果类型：

```cpp
inline constexpr std::size_t kInSwapperEmbeddingLength = 512;
inline constexpr std::int64_t kInSwapperInputExtent = 128;
inline constexpr std::int64_t kGfpGanInputExtent = 512;

using InSwapperResult = std::array<float, 3 * 128 * 128>;
using GfpGanResult = std::array<float, 3 * 512 * 512>;
```

为避免将大结果复制到栈上，实际 adapter 返回拥有连续存储的结果对象，其 `data()`/`size()` 与生命周期由结果对象自身负责；具体类型在测试先行后确定为固定上限的 `std::vector<float>` 包装，而不是公开裸指针。

新增：

```cpp
struct InSwapperOptions {
    std::string target_tensor_name = "target";
    std::string source_tensor_name = "source";
    std::string output_tensor_name = "output";
    std::size_t max_tensor_bytes = /* named bounded default */;
};

struct GfpGanOptions {
    std::string input_tensor_name = "input";
    std::string output_tensor_name = "output";
    std::size_t max_tensor_bytes = /* named bounded default */;
};

class TensorRtInSwapper final {
public:
    static TensorRtInSwapper load(const std::filesystem::path&, InSwapperOptions = {});
    InSwapperResult infer(const TensorView& prepared_target,
                          const TensorView& projected_source);
};

class TensorRtGfpGan final {
public:
    static TensorRtGfpGan load(const std::filesystem::path&, GfpGanOptions = {});
    GfpGanResult infer(const TensorView& prepared_input);
};

class InSwapperEmbeddingProjector final {
public:
    static InSwapperEmbeddingProjector load(const std::filesystem::path& matrix_path);
    ArcFaceResult project(const ArcFaceResult& embedding) const;
};
```

adapter 延续现有 Pimpl、严格 tensor 名、FP32、固定 batch 1 和同步执行约束。InSwapper 的两个输入都必须绑定；GFPGAN 只接受 `[1,3,512,512]`。

### `KFCore::face_applications`

模块公开 OpenCV 应用边界，内部保持可拆分职责：

- `FaceGeometry`：bbox crop affine、Face68 反变换、68 点到 5 点、similarity transform；
- `FaceTensorPreprocessor`：Face68、ArcFace、InSwapper、GFPGAN 的 NCHW 变换；
- `FaceComposer`：mask、inverse warp、paste-back 与 enhancer blend；
- `TensorRtFaceAnalyzer`：12face -> Face68 -> ArcFace，可选 Age/Gender；
- `TensorRtFaceSwapApplication`：仅协调 analyzer、projector、swapper、可选 enhancer。

应用级核心接口：

```cpp
struct FaceApplicationModelPaths {
    std::filesystem::path face_detector_engine;
    std::filesystem::path face68_engine;
    std::filesystem::path arcface_engine;
    std::filesystem::path inswapper_engine;
    std::filesystem::path inswapper_matrix;
    std::optional<std::filesystem::path> gfpgan_engine;
    std::optional<std::filesystem::path> age_gender_engine;
};

struct FaceSwapOptions {
    float detector_score_threshold = 0.5f;
    int face_class_id = 0;
    float enhancer_blend = 0.8f;
};

struct FaceAnalysis {
    kfcore::yolo::Detection detection;
    std::array<cv::Point2f, 5> landmarks;
    kfcore::face_models::ArcFaceResult embedding;
    std::optional<kfcore::face_models::AgeGenderResult> age_gender_logits;
};

class TensorRtFaceSwapApplication final {
public:
    static TensorRtFaceSwapApplication load(
        const FaceApplicationModelPaths&, FaceSwapOptions = {});

    FaceAnalysis analyze(const cv::Mat& bgr_image);
    cv::Mat swap(const cv::Mat& source_bgr, const cv::Mat& target_bgr);
};
```

实现时可根据现有 YOLO 公开类型微调字段名称，但不得暴露第三方 TensorRT 对象、裸 CUDA stream 或无生命周期约束的 pointer。

## 模型契约

### 12face

- 输入和 letterbox 继续由现有 `TensorRtDetector` 负责；
- 只接受已有 compact `[B,N,6]` 输出路径；
- 过滤 `class_id == face_class_id` 后选择最高 score；同 score 时按原检测顺序保持确定性；
- 没有合格人脸时返回明确 `NoFaceDetected`，不换用其他 detector。

### Face68

- 用 `max(width, height)` 构造正方形尺度；
- `scale = 195 / extent`，把 bbox 中心映射到 256 图像中心；
- BGR planar CHW，`value / 255`；
- adapter 返回的 x/y 已在 256 坐标系，用逆 affine 映射回原图；
- 5 点索引为：左眼 `36..41` 均值、右眼 `42..47` 均值、鼻尖 `30`、嘴角 `48/54`。

### ArcFace

- canonical 112 点：
  `(38.29459984,51.69630032)`、`(73.53180016,51.50140016)`、
  `(56.0252,71.73660032)`、`(41.54929968,92.36549952)`、
  `(70.72989952,92.20409968)`；
- similarity align 至 112；
- BGR 转 RGB planar CHW，`value / 127.5 - 1`；
- 保留 adapter 的 raw 512 embedding，不在 ArcFace adapter 内隐式归一化。

### InSwapper

- target canonical 128 点：
  `(46.29459968,51.69629952)`、`(81.53180032,51.50140032)`、
  `(64.02519936,71.73660032)`、`(49.54930048,92.36550016)`、
  `(78.72989952,92.20409984)`；
- target：BGR 转 RGB planar CHW，`value / 255`；
- source：ArcFace 512 embedding 乘以 row-major `512x512` 模型矩阵，再 L2 normalize；
- 输出：RGB CHW `[1,3,128,128]`，有限值校验并 clamp `[0,1]` 后转 BGR；
- 使用静态 box mask 的模糊版本 inverse warp 并合成。

### GFPGAN

- canonical 512 点：
  `(192.98138112,239.94707968)`、`(318.90276864,240.19360256)`、
  `(256.63415808,314.01934848)`、`(201.26116864,371.410432)`、
  `(313.0890496,371.1511808)`；
- BGR 转 RGB planar CHW，`value / 127.5 - 1`；
- 输出有限值校验，clamp `[-1,1]` 后转 BGR；
- inverse warp 后按 `enhancer_blend` 与 swap 结果混合，范围严格为 `[0,1]`。

## 模型矩阵资产

InSwapper 的 `512x512` float32 矩阵是模型部署资产，不是通用常量。应用要求调用方显式提供 sidecar 文件：

- 精确大小 `512 * 512 * sizeof(float) = 1,048,576` bytes；
- row-major、host little-endian float32；
- 拒绝短文件、尾随数据、非有限元素和零范数投影结果；
- 只在 `load()` 时读取一次，之后由 projector 独占不可变存储；
- 不自动从 ONNX 提取，不搜索工作目录，不自动下载。

本地参考矩阵的 SHA-256 为 `370af5bf707dafdbea8a40448d697d9697610bd223ecf92887af9c9cc7055ac8`，仅用于本机验证和溯源，不作为运行时硬编码白名单。

## 所有权、线程和资源协议

- 输入 `cv::Mat` 在同步调用期间借用；不保存其 data pointer。
- `swap()` 先 clone target，在成功前不修改调用方 source/target；返回值拥有像素。
- prepared tensors 由一次调用的局部有界缓冲拥有；runtime 仅在同步 infer 期间借用。
- engine 是不可变模型事实源；每个 adapter 拥有自己的 mutable executor 和输出缓冲。
- 单个 facade 实例非重入；检测到重叠调用立即报错。并发 worker 必须各自拥有 facade/executor。
- 第一阶段固定 batch 1、每张图只选一个最高分人脸，因此内存上界由固定 tensor、两张输入图和一个输出图决定，不引入队列或背压状态。
- 尺寸乘法在分配前检查溢出，并复用 runtime 的 `max_tensor_bytes` 限制。
- 失败不提交部分合成结果；临时 CUDA/host/OpenCV 资源依 RAII 回收。

## 错误语义

应用层新增 `FaceApplicationErrorCode`，至少覆盖：

- `InvalidArgument`
- `NoFaceDetected`
- `InvalidLandmarks`
- `InvalidModelMatrix`
- `ModelContractMismatch`
- `ResourceLimitExceeded`
- `RuntimeFailure`
- `ImageProcessingFailure`

模型/runtime 异常只在应用边界转换一次并附带阶段（detect/landmark/embed/swap/enhance）；中间层不记录后继续，也不返回半可信图像。

## 构建、安装与兼容性

新增默认关闭的 `KFCORE_BUILD_FACE_APPLICATIONS`。开启时要求：

- `KFCORE_BUILD_TENSORRT_RUNTIME=ON`
- `KFCORE_BUILD_FACE_MODELS=ON`
- `KFCORE_BUILD_TENSORRT_YOLO=ON`
- `KFCORE_BUILD_YOLO_OPENCV=ON`

缺少任一项在 configure 阶段 fail fast。安装导出 `KFCore::face_applications`，package config 仅在该模块启用时查找 OpenCVLite core/imgproc。示例需要 imgcodecs，但不把 imgcodecs 传播给库消费者。

这是纯新增能力；现有公开 API、默认 OFF 构建和已安装组件保持兼容。新增依赖只影响显式开启该模块的构建。

## 测试与验收

1. 纯单元测试：bbox affine、inverse mapping、68-to-5、canonical similarity、各模型像素归一化、CHW 通道顺序、mask 与 blend、输入不可变。
2. 矩阵测试：精确字节数、短/长文件、NaN、投影 spot values、零范数。
3. adapter contract：错误 tensor 名、shape、dtype、batch、缺失第二输入、资源限制。
4. opt-in engine integration：Face68、ArcFace、InSwapper、GFPGAN 各自加载并执行，输出 shape、finite 和重复执行稳定。
5. 端到端本地验收：12face 检测 source/target，成功输出非空 BGR 图；输入像素不变；启用/禁用 GFPGAN 两条路径都执行。
6. 回归：现有 27 项基线测试和新增测试全部通过，安装消费测试可找到新增 target。

真实 engine/图片路径通过 CMake cache 或 CLI 显式传入。缺失时只跳过 opt-in integration，不让纯单元测试失去覆盖。

