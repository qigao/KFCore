# 官方 Gesture Recognizer → ONNX Runtime

使用 Google 固定版本 [float16/1 task bundle](https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task)。
SHA256：`97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482`。
脚本校验摘要，不接受其他 bundle。权重与生成产物放 build，不提交。

## 离线转换

在仓库根目录使用独立 Python 3.11 环境；需要已安装 uv。
以下命令仅用于新环境、新下载文件及新输出目录，已有 bundle 可直接验证并转换：

```powershell
uv venv --python 3.11 build/mediapipe-convert-env
uv pip install --python build/mediapipe-convert-env/Scripts/python.exe -r tools/mediapipe_gesture/requirements.txt
New-Item -ItemType Directory -Force build/mediapipe-gesture-official
Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task' -OutFile build/mediapipe-gesture-official/gesture_recognizer.task
build/mediapipe-convert-env/Scripts/python.exe tools/mediapipe_gesture/export_models.py --bundle build/mediapipe-gesture-official/gesture_recognizer.task --output build/mediapipe-gesture-official/onnx-v3
```

输出目录存在则立即失败；失败留下部分产物供诊断，重试请指定新的输出目录。
只有全部转换、检查、包装和参考数据生成成功才写入 `parity.json`，缺少报告不可部署。
TensorFlow/tf2onnx 仅供离线转换，C++ app 不依赖 Python/TFLite。

四个原始模型转换为 opset 13。每个用三个确定性输入比较 TFLite 与 ORT，
绝对/相对容差均为 0.002。生成 NCHW 关键点包装模型和三个既有格式模型包：
`hand_world.json`、`gesture_embedder.json`、`gesture_classifier.json`。
`classification-golden.f32` 为独立 TFLite 分类链的参考输入/输出，小端 float32，
仅供原生测试，不是业务存储格式。

## 架构决策、风险与迁移

选择离线转换，运行时保留 ONNX 后端、既有图像模块和 Salts 采集。
原生 Tasks SDK 会新增运行时依赖和部署要求；旧关键点模型缺少真实 world landmarks，
不能正确驱动官方 embedder。没有用伪世界坐标、规则分类或 Retro 分类器替代。

依赖方向 `preview → mediapipe → hand_models → runtime`；ESN 只读取识别历史，
不依赖文件格式和 UI。识别实例拥有上下文，UI session 拥有录制状态与样本，
ESN 拥有训练特征和读出层；新训练数据使读出层失效。配置或张量不匹配立即失败。

MED：仍复用旧的已后处理 palm ONNX、ROI 和几何投影。官方 raw palm 虽已转换并
比较数值，尚未接入其 anchor 解码。此实现不是完整 Tasks 图复刻，不包含跟踪。
单模型转换对齐不能证明端到端与 Tasks 完全等价。CPU 已验证，其他 provider 未验证。
HandResult 新增可选字段要求二进制调用方重新编译；旧三输出模型保持原行为。
迁移仅替换关键点包并增加两个 gesture 包；回滚使用旧两模型入口且不启用 ESN。
没有持久化数据迁移，不自动修改现有标注或视频。

## 验证与运行

旧资产目录须包含 palm 包及 `hand-640x480.bgr`，见 vision/mediapipe/README.md。
在 MSVC 开发终端执行：

```powershell
$env:KFCORE_MEDIAPIPE_TEST_ASSETS='C:/projects/cpp/KFCore/build/mediapipe-onnx-smoke'
$env:KFCORE_GESTURE_TEST_ASSETS='C:/projects/cpp/KFCore/build/mediapipe-gesture-official/onnx-v3'
cmake --preset win-hand-preview-user
cmake --build --preset win-hand-preview-user --target test_gesture_recognizer test_composition_esn kfhand_preview
ctest --preset win-hand-preview-user -R '^(test_gesture_.*|test_composition_esn)$' -V
./build/hand-preview/Release/bin/run-hand-preview.cmd --list-modes 0
```

核对实际用例数：TinyTest filter 匹配测试名称而非 spec 名。
真实图像测试检查检测非空、world 数据、八类分数与失败后恢复，并将 C++ 分类结果
和 TFLite oracle 按绝对误差 1e-5 比较。这不是实际手势准确率基准。

按枚举结果修改摄像头模式；以下模式 120 仅适用于本机此前的枚举结果：

```powershell
./build/hand-preview/Release/bin/run-hand-preview.cmd --backend build/hand-preview/Release/bin/kfcore_backend_onnxruntime.dll --palm build/mediapipe-onnx-smoke/palm_detection_full_inf_post_192x192.json --landmark build/mediapipe-gesture-official/onnx-v3/hand_world.json --gesture-embedder build/mediapipe-gesture-official/onnx-v3/gesture_embedder.json --gesture-classifier build/mediapipe-gesture-official/onnx-v3/gesture_classifier.json --camera 0 --mode 120 --esn
```

来源：[官方说明](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer)、
[识别图](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/gesture_recognizer/hand_gesture_recognizer_graph.cc)、
[归一化](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/gesture_recognizer/calculators/landmarks_to_matrix_calculator.cc)、
[tf2onnx](https://github.com/onnx/tensorflow-onnx)。
