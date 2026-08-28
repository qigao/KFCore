# Hand Interaction 与 THIG

`KFCore::hand_interaction` 把 `vision_models::HandFrame` 中的 Palm box、21 点、模型手势和
ByteTrack ID 转成有界的 primitive observations，再交给 `KFCore::thig` 解释时序动作。
它不执行模型推理，因此 ONNX Runtime CPU 与 TensorRT CUDA 共用完全相同的身份、几何和
手势图实现。

```text
CPU Hand backend ---------+
                          +-> HandPipeline -> HandInteractionPipeline
TensorRT Hand backend ----+                         |
                                                    +-> primitives + actions
```

## 支持的语义动作

- `Wave`：张开手掌的 Left/Right/Left 或 Right/Left/Right 三段反向运动；可通过
  `wave_require_horizontal_palm_axis` 要求整个动作期间掌根轴保持水平。
- `Grasp`、`Release`、`Drag Start *`、`Drag *`、`Drag End`、`Drag Cancelled`。
- `OK`、`Single Hand V`、`Two Hand V`。
- `Zoom In`、`Zoom Out`、`Rotate Clockwise`、`Rotate CounterClockwise`；Rotate 默认使用
  `rotation_cooldown_ms=600` 抑制短时间关系抖动造成的重复事件。
- `Click Center/Left/Right/Top/Bottom`：需要应用提供 `Region *` 外部观察。

动作只表达语义，不直接操作 Camera、Gallery、UI 或窗口。应用负责把动作映射为命令；人脸
相对位置、屏幕布局和控件 hit-test 也不进入本模块。

## 使用

```cpp
#include <kfcore/hand_interaction/hand_interaction.hpp>

kfcore::hand_interaction::HandInteractionOptions options;
options.primitives.max_hands = 8;
options.temporal.max_observations_per_frame = 128;
options.temporal.max_relation_events = 512;
options.temporal.max_action_states = 4096;
kfcore::hand_interaction::HandInteractionPipeline gestures(options);

std::uint64_t serial = 1;
const auto result = gestures.process(
    tracked_hand_frame,
    {serial++, std::chrono::steady_clock::now(), image_width, image_height});
for (const auto& action : result.actions) {
    dispatch_application_command(action.action);
}
```

`rotation_cooldown_ms` 追加在公开 aggregate `HandInteractionSettings` 的末尾，旧的位置初始化代码仍按
原字段顺序解释；由于结构体尺寸发生变化，升级 KFCore 后必须重新编译 C++ 消费端。

`process()` 同步借用输入 `HandFrame`，返回值拥有 observations、evidence 和 actions；不会保留
landmark 指针。frame serial 必须严格递增，timestamp 必须单调不减，图像尺寸必须为正。
违反配置、顺序或容量约束会抛出明确异常，不会自动切换后端。

`HandSpatialOptions::palm_axis_horizontal_max_degrees` 默认 `30°`，小于等于该值归为
`Palm Axis Horizontal`；`palm_axis_vertical_min_degrees` 默认 `60°`，大于等于该值归为
`Palm Axis Vertical`，两者之间归为 `Palm Axis Diagonal`。两个阈值必须满足
`0 < horizontal_max < vertical_min < 90`，非法配置在构造 extractor 时抛出异常。

外部观察目前只接受图中已经声明的 `Region Center/Left/Right/Top/Bottom/Unclassified`，每帧
每个 canonical hand 最多一个同名观察，不允许 target。`source.id` 使用前一帧
`HandInteractionFrame::primitives.hands[].canonical_id`；首帧尚无 canonical ID 时应省略区域
观察。输入 observation 会复制进结果和 THIG，不得使用 raw ByteTrack ID 代替 canonical ID。

## Canonical hand identity

canonical hand ID 由 registry 单独拥有；它在有限重获时间窗内以 20 条已标注手部骨架边的归一化三维
长度作为形状连续性证据，再把 palm 几何、handedness 和 raw ByteTrack ID 用作排序证据。该描述符对
平移、统一缩放和图像平面内旋转保持不变，采用 L1 距离比较；它是从 21 个 landmark 的五条
wrist-to-finger 链构成的固定值，不分配堆内存。它不是 OpenCV Hu moments，也不构成生物识别、人员
识别或跨会话身份保证。

所有 landmark 分量和边长必须有限，20 条边的总长度必须为正；否则形状证据无效，该手的 canonical
ID 为 `0`，不会退回为仅按位置匹配。候选还必须满足 `maximum_shape_distance`；形状不兼容的可靠手会
尝试分配新 canonical ID。若容量已耗尽，处理会以 `std::length_error` fail fast，而不是把资源耗尽
伪装成 ID `0`。

`reacquire_frames` 是 canonical identity 的保留 horizon：在每帧匹配前，年龄超过该值的原型会被删除。
因此只在该时间窗内可因形状、空间、尺度与 raw-ID 连续性重获同一 ID；超出时间窗的同形状观测会分配
新 ID，避免另一用户继承旧的 THIG 状态。双方 handedness 已知且冲突时不是候选；`Unknown` 与任意值
兼容。若两个观测对同一最佳 ID 的成本差小于 `ambiguity_cost_margin`，两者均为 `0`，不按输入顺序取胜。

`maximum_distance_scale_ratio` 与 `maximum_linear_scale_ratio` 是有限的软证据饱和值：预测距离分别按前者
线性归一化到 `[0, 1]`，尺度比的对数差按 `log(后者)` 归一化到 `[0, 1]`。它们不会放宽形状或
handedness gate；部署应以实际镜头、手势和 landmark 噪声校准这些阈值。

所有公开 cost 权重仍为有限 `float` 配置，但候选 cost、排序和歧义差值均以 `double` 累加与比较；因此极大
但有效的权重不会把同成本候选变成 `+inf` 并绕过 `ambiguity_cost_margin`。

`HandIdentityOptions` 的默认容量为 `maximum_identities = 32`，并追加下列已在构造时验证的配置：

| Field | Default | Meaning |
|---|---:|---|
| `maximum_shape_distance` | `0.20F` | 保守的形状候选 L1 gate（范围 `(0, 2]`）；应按实际 landmark 噪声校准 |
| `shape_cost_weight` | `2.0F` | 形状距离在候选成本中的权重 |
| `shape_update_weight` | `0.20F` | 已接受形状写入原型的 EMA 权重（范围 `(0, 1]`） |

这三个字段被追加到公开 `HandIdentityOptions` 末尾。源码默认构造和短 aggregate 初始化保持可用，但其
对象布局已变化；所有二进制下游消费者必须重新构建。

## 状态、并发与容量

每个 `HandInteractionPipeline` 是单 owner、不可重入对象。不同实例可由不同任务并行运行，
但同一实例的 `process()`、`reset()` 和状态查询必须由调用方串行化。`reset()` 同时清空 canonical
身份、运动/尺度/旋转/双手距离历史、THIG relation、cooldown 和状态图绑定。

所有保留数据同时受时间窗与数量上限约束：`max_hands`、每手样本数、pair histories、
`history_ms`、`max_observations_per_frame`、`max_relation_events` 和
`max_observation_window_states`、`max_action_states`。action-state 容量必须至少覆盖
`max_relation_events * action_count`，配置时会先验证且 action 状态按 `history_ms` 过期。
默认 8 手的 primitive 上界为
`8 * 8 + 8 * 7 / 2 = 92` 个 observations/帧，低于默认 THIG 上限 128；缩小任一容量时应按
部署的最大手数重新计算。

facade 每帧在 bounded extractor state 的副本上计算；只有 THIG 接受整帧后才提交 identity 和
geometry histories。relation/window/action 容量拒绝不会消费 serial，调用方可用同一 frame
重试。canonical identity 容量耗尽同样显式抛错，不会用 ID 0 静默表示资源不足。

## 构建和安装

源码树构建需要同时启用：

```cmake
-DKFCORE_BUILD_VISION_MODELS=ON
-DKFCORE_BUILD_THIG=ON
-DKFCORE_BUILD_HAND_INTERACTION=ON
```

Windows 用户 preset 已在 CPU/TensorRT vision profile 中启用这三个选项。安装后：

```cmake
find_package(KFCore CONFIG REQUIRED)
target_link_libraries(app PRIVATE KFCore::hand_interaction)
```

包配置提供 `KFCore_HAS_THIG` 和 `KFCore_HAS_HAND_INTERACTION`，便于可选功能在 configure 阶段
fail fast。模型与 TensorRT engine 仍由对应 backend 管理，既不复制也不安装到本模块。

identity 与 primitive core 不依赖 OpenCV、CUDA、TensorRT 或 ONNX Runtime；CPU ONNX Runtime 和
TensorRT CUDA backend 只提供同一种 `HandFrame`，因此使用同一套 canonical identity 行为。实时 demo
是可选边界，才会链接 Capture、OpenCV Lite 和所选推理 backend。

## Turbo Capture 实时 Demo

`hand_interaction_demo` 是可选的桌面示例，不改变已安装库的接口或依赖。数据流为：

```text
USB camera -> Turbo Capture -> bounded latest-frame mailbox -> owning BGR image
           +-> CPU ONNX Runtime 或 TensorRT/ImageProcessor -> HandPipeline -> THIG --+
           +-> 可选 YOLOv12-face -> MediaPipe FaceMesh 468 点 -----------------------+-> HighGUI
```

Capture 回调中的像素指针只在回调期间有效，因此示例在回调返回前复制一次。邮箱固定保留两个
受 `--max-frame-bytes` 限制的 vector；推理落后时以最新帧替换未消费帧并增加 `coalesced`，不会
让采集线程等待。BGR 图像由示例拥有，并同时借给显示和所选推理后端；TensorRT 后端再由现有
ImageProcessor 上传和预处理。手部与人脸流程共享同一个只读 host BGR `ImageView`；当前
TensorRT detector 与 landmarker 各自拥有 CUDA stream、staging 和 tensor buffer，因此这不是
GPU tensor 级零拷贝共享。OpenCV Lite 与 Turbo Capture 只链接到示例，不成为
`KFCore::hand_interaction` 的传递依赖。

### 构建

CPU 路径：

```powershell
cmake --preset win-hand-interaction-demo-cpu-release-user
cmake --build --preset win-hand-interaction-demo-cpu-release-user --target test_hand_primitive_extractor
ctest --preset win-hand-interaction-demo-cpu-release-user --output-on-failure
```

TensorRT 路径需要先指向本机 SDK；preset 只做编译验证，不要求集成测试 engine：

```powershell
$env:TENSORRT_ROOT = 'C:\projects\TensorRT-11.2.1.2'
cmake --preset win-hand-interaction-demo-tensorrt-release-user
cmake --build --preset win-hand-interaction-demo-tensorrt-release-user
ctest --preset win-hand-interaction-demo-tensorrt-release-user --output-on-failure
```

本机 TurboParser release package 若未导出 `TurboParser::Capture`，只能使用
`-DKFCORE_BUILD_HAND_INTERACTION_EXAMPLES=OFF` 验证 core 和非 demo 测试；这不代表 identity core
测试失败。需要构建实时 demo 时，应安装包含 Capture component 的 TurboParser SDK，再使用上述
preset 的默认 `ON` 配置重新 configure。

两个 preset 都从 `CMakeUserPresets.json` 设置 `OPENCV_LITE_ROOT`，并为 configure、build 和
CTest 子进程加入 OpenCV Lite、Turbo Capture 及对应推理 runtime 的 DLL 目录。直接从当前
PowerShell 启动 exe 时，父 shell 仍需把这些目录加入 `PATH`。

### 运行

先列出摄像头及原生 mode；该操作不会加载模型：

```powershell
build\HandCPU\bin\hand_interaction_demo.exe --list-cameras
```

CPU 使用仓库已有的三份 ONNX hand 模型：

```powershell
build\HandCPU\bin\hand_interaction_demo.exe `
  --backend cpu --camera 0 --width 640 --height 480 --fps 30 `
  --model-dir C:\projects\cpp\KFCore\yolo-models
```

在同一窗口启用 FaceMesh 时，额外显式提供 YOLOv12-face detector 与 MediaPipe 468 点模型；
两项必须同时出现，省略两项仍保持原有 hand-only 行为：

```powershell
build\HandCPU\bin\hand_interaction_demo.exe `
  --backend cpu --camera 1 --mode 0 --max-frames 100 `
  --model-dir C:\projects\cpp\KFCore\yolo-models `
  --face-detector C:\projects\cpp\KFCore\yolo-models\yolov12n-face.onnx `
  --facemesh C:\projects\cpp\KFCore\yolo-models\MediaPipeFaceLandmarkDetector.onnx
```

TensorRT 要求显式提供三份可信 engine，不会自动生成 engine 或回退到 CPU：

```powershell
build\HandTRT\bin\hand_interaction_demo.exe `
  --backend tensorrt --camera 0 --mode 385 `
  --palm C:\models\palm.engine `
  --hand C:\models\hand_landmark.engine `
  --classifier C:\models\keypoint_classifier.engine
```

TensorRT FaceMesh 使用同模型生成的 strongly typed engine：

```powershell
build\HandTRT\bin\hand_interaction_demo.exe `
  --backend tensorrt --camera 1 --mode 0 --max-frames 100 `
  --palm C:\models\palm_detection.engine `
  --hand C:\models\hand_landmark.engine `
  --classifier C:\models\keypoint_classifier.engine `
  --face-detector C:\models\yolov12n-face.engine `
  --facemesh C:\models\face_landmark.engine
```

`--face-score` 和 `--facemesh-score` 分别设置 detector 与 landmarks 的 `[0,1]` 置信度阈值，
默认均为 `0.5`。当前只消费 class 0 中得分最高的一张脸，并绘制 468 个点；没有使用未经本地
模型验证的 mesh 连线表。TensorRT-YOLO 当前不拆分 detector 内部的上传、预处理与 enqueue
计时，因此 TensorRT 窗口中的 `face-pre` 为 `0`，`face-det` 是 detector 整次调用的墙钟耗时；
CPU 路径会分别报告两项。

`--mode` 与 `--width/--height/--fps` 二选一，且只接受精确匹配。省略时请求
1280x720@30；同一规格按 NV12、I420、BGRA、RGB24 的顺序选择。MJPEG 会被明确拒绝，因为
示例没有隐式 JPEG 解码路径。`--max-frames N` 可用于可重复的有界 smoke test，默认持续运行；
`R` 同时重置 tracker 与 THIG 状态，`Q`、Escape 或关闭窗口正常退出。

启动时输出全部已启用模型的总加载耗时；窗口逐帧显示 Capture-to-BGR 转换、preprocess、Palm、landmark、
classifier、tracking、model 总计、THIG 和整条处理 pipeline 的耗时，以及
FaceMesh 启用时的 face preprocess、detector、mesh preprocess、mesh inference 与 face 总计，
以及 `captured/consumed/coalesced/rejected` 计数。这些是当前帧和当前运行的诊断数据，不等同于
稳定的 P50/P95 性能结论，也不包含 HighGUI 的显示刷新时间。

窗口只在显示层镜像画面及手部/FaceMesh 坐标，模型仍消费原始相机帧。状态栏的 `FPS` 是按实际
消费帧间隔计算的指数平滑值；每只手明确显示 THIG `Hand ID`、tracker `Track ID`、原始分类与
派生 primitive。

手框中的 `Raw` 只表示 keypoint classifier 的 `Open/Closed/Pointer` 原始三分类；`Derived`、
`Motion`、`Direction`、`Rotation`、`Axis` 和可选的 `Pose:OK` 来自当前帧 21 点几何 primitive。
`Axis` 将掌根 5→17 的无向轴按可配置阈值归类为 `Horizontal/Diagonal/Vertical`。demo 默认允许
任意掌轴完成 `Wave`；需要严格限制时显式传入 `--wave-require-horizontal-axis`，要求全程为
`Horizontal`。状态栏同时显示 hand、wave、click
三张 THIG 状态图。`Single Hand V`、`Grasp` 等一次性 `ActionEvent` 不改变核心事件语义，但在
demo 中最多保留四条、每条显示 1500ms，便于人工观察；按 `R` 会同时清除这段显示历史。

若 configure 报告缺少 `TurboParser::Capture`，说明 TurboParser SDK 没有安装 Capture 导出目标；
在 TurboParser 源码树依次运行 `win-capture-release-user` 的 configure、build、test 和 install
preset 后重新配置。若启动时 Windows 在进入 `main` 前退出，应先检查当前 shell 的 `PATH`
是否包含 `opencv-lite/bin`、`turboparser/release/bin` 和相应推理 runtime 目录。
