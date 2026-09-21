# MediaPipe 衍生 ONNX 手部模块

目标 `KFCore::mediapipe`，入口 `kfcore::mediapipe::HandLandmarker`。
使用既有 ONNX Runtime 插件执行手掌检测、21 点关键点和左右手判断。
此两模型入口不加载手势分类器，不执行跟踪或复杂动作识别。
另有 `GestureRecognizer` 官方基础手势入口，见下文。

## 边界与架构决策

本模块是已约定导出格式的薄适配层，不是 MediaPipe 原生 SDK，不能加载
`.task` / TFLite，也不支持任意命名为 MediaPipe 的 ONNX 文件。
本地权重的上游版本和转换过程尚未核实；通过推理测试不代表与官方
MediaPipe Tasks 的精度、跟踪或坐标语义完全等价。

选择复用 `hand_model_runtime` 的预处理、ROI、解码和推理资源管理，避免
复制第二套算法。原生 SDK 方案会新增依赖和部署要求；直接复制现有管线
会产生两套几何逻辑。代价是本模块的模型契约受现有 192/224 管线约束。
依赖方向为 `mediapipe -> hand_model_runtime -> runtime/image/core`，不依赖 UI。

共享运行时新增显式 `load_landmarks` 入口，旧三模型 `load` 的调用方式和
分类行为保留。新旧入口共用推理主体，分类部分仅在加载了分类器时执行。
迁移只需改用新入口及合格模型包；回滚可继续使用旧三模型入口。
MED 风险：共享推理主体的改动可能影响旧分类路径，需执行相邻回归测试；
本次没有新增第三方依赖，也没有修改已有模型包、标注或视频格式。

## 模型契约

两包的所有 artifact 必须为 `format=onnx`、`backend=onnxruntime`。
调用策略必须为单个 ONNX Runtime 设备，不允许有序回退。
CPU 已有本地真实模型测试；CUDA 尚未验证。

| 模型 | model_type | flavor | 张量约束 |
| --- | --- | --- | --- |
| 手掌 | `hand.palm-detector` | `mediapipe-palm-postprocess-v1` | float NCHW RGB `[1,3,192,192]`，归一化到 `[0,1]`；已后处理输出 `[N,8]` |
| 关键点 | `hand.landmarker` | `mediapipe-hand-landmark-v1` | float NCHW RGB `[B,3,224,224]`，执行时 B=1；xyz `[B,63]`、score `[B,1]`、handedness `[B,1]` |

手掌输出每行是归一化画布坐标
`score, box_x, box_y, box_size, keypoint0_x, keypoint0_y, keypoint2_x, keypoint2_y`。
原始 anchor/score 双输出不支持。关键点输出名称须符合现有运行时的
xyz/landmark、score、left/right/handed 识别规则。
flavor 是显式格式声明，不是模型来源认证；不能仅改标签来兼容不同模型。
加载还会检查实际张量布局，模型包会校验 artifact 摘要。

## API 与状态

`load(runtime, palm, landmark, policy, options)` 返回独占实例。
参数为已解析的两个 ModelPackage、精确设备策略和可调整的
`HandRuntimeOptions`（阈值、最大手数及资源限额）。
无效策略抛 `HandModelError::InvalidArgument`；格式不匹配抛
`ModelContractMismatch`；后端失败转换为 `RuntimeFailure`。
模型包解析和后端加载错误仍由 Runtime 边界报告，不静默降级。

`infer(ImageView)` 借用 Host 图像到调用结束，返回拥有自身数据的 `HandFrame`。
实例拥有模型会话和工作缓存，同实例并发推理被拒绝，销毁不得与使用并发。
`x/y` 是源图像像素，`z` 是相对深度而非世界坐标米；
`gesture=Unknown`、`track_id=-1`，分类器计时为零。
无手返回空列表；非法图像抛 `HandModelError`。
两个 `*_execution_route()` 返回实例拥有的只读实际路由引用。

完整的加载、图像输入和结果访问示例见可编译执行的
[`tests/test_mediapipe_onnx.cpp`](tests/test_mediapipe_onnx.cpp)。

## 验证与复现

在已初始化 MSVC 的终端，使用仓库本地 user preset：

```powershell
cmake --preset win-cpu-release-user
cmake --build --preset win-cpu-release-user --target test_mediapipe test_mediapipe_onnx kfcore_backend_onnxruntime kfmodel test_hand_model_api test_hand_model_geometry test_hand_model_decode test_hand_tracking
ctest --preset win-cpu-release-user -R '^(test_mediapipe|test_hand_model_api|test_hand_model_geometry|test_hand_model_decode|test_hand_tracking)$' --output-on-failure
```

真实推理测试单独启用，不下载权重，不以 mock 代替推理。测试资产目录包含：

- `palm_detection_full_inf_post_192x192.onnx` 及同名 `.json` 模型包。
- `hand_landmark_sparse_Nx3x224x224.onnx` 及同名 `.json` 模型包。
- `hand-640x480.bgr`：包含可检测手部的 640×480、紧密排列 BGR8 原始帧。

可用既有 `kfmodel create-onnx <source.onnx> <existing-output-dir> <id> <model-type> <flavor>`
生成独立模型包；不要覆盖原模型包。运行 CLI 时 PATH 需包含 user preset
配置的运行时依赖目录。设置资产环境变量后重新 configure，CTest 才注册真实测试：

```powershell
$env:KFCORE_MEDIAPIPE_TEST_ASSETS = 'C:/projects/cpp/KFCore/build/mediapipe-onnx-smoke'
cmake --preset win-cpu-release-user
ctest --preset win-cpu-release-user -R '^test_mediapipe_onnx$' --output-on-failure
```

本地 smoke 输入取自 `1CM1_1_R_#217.mp4` 的第 15 秒，使用 FFmpeg
`-ss 15 -frames:v 1 -pix_fmt bgr24 -f rawvideo` 导出，不改变原视频。
测试要求真实检测非空、21 点有限值、左右手有效、不执行分类，并验证
重复调用及非法输入后的恢复。这是功能 smoke，不是准确率、性能或完整视频评测。
权重、视频帧及构建产物不提交到仓库。

## 官方基础手势入口

`GestureRecognizer::load(runtime, palm, landmark, embedder, classifier, policy, options)`
返回独占实例。palm 使用上述格式；其余三个包由
[离线转换工具](../../tools/mediapipe_gesture/README.md)产生，flavor 分别为
`mediapipe-hand-world-v1`、`mediapipe-gesture-embedder-v1`、`mediapipe-canned-gesture-v1`。
策略必须只有 ONNX Runtime；缺失真实 world landmarks 或模型契约不符立即失败。

`infer(ImageView)` 借用输入到返回，结果拥有自身数据：`landmarks` 保存检测结果，
`gestures` 与手列表一一对应。每项包含八类分数和 argmax 标签。
类别依次为 None、Closed_Fist、Open_Palm、Pointing_Up、Thumb_Down、Thumb_Up、
Victory、ILoveYou；无手是空列表，不是 None。不会映射到旧 Retro Gesture 枚举。
分类计时包含归一化、embedding 和分类，累计入总耗时。

`gesture_features(hand,width,height)` 需要正数源图像尺寸、真实 world landmarks
和右手概率；返回腕部中心化并按 xy 最大跨度归一化的屏幕/世界坐标。
仅支持完整源图像、调用方旋转为零；z 使用官方 0.4 归一化比例。
`classify(features)` 接收已归一化的两组 63 维坐标和 [0,1] 右手概率，返回八类预测。
无效输入/契约抛 invalid_argument，非法模型分数抛 runtime_error；底层加载和
HandDetector 错误继续向调用方传播，不回退。实例单线程使用，销毁不得与调用并发。

可执行示例为 `tests/test_gesture_recognizer.cpp`。这是官方模型在既有几何管线上的
ONNX 适配，不声称与完整 Tasks 图等价；没有时间跟踪、置信阈值过滤或自定义手势模型。
ESN 实验属于 tools/hand_preview，基础分类器没有依赖 ESN。
