# TensorRT Face Applications

`KFCore::face_applications` 提供单脸分析和换脸应用层。检测器固定为
[YOLOv12-face](https://github.com/akanametov/yolo-face)；本模块没有 MediaPipe、YOLO11-face
或其他 detector fallback，也不会扫描默认 weights 目录。

## 数据流与状态边界

```text
BGR source/target (borrowed CV_8UC3)
  -> YOLOv12-face: highest-score class 0 face
  -> Face68: bbox affine crop -> 68 landmarks -> source coordinates
  -> five landmarks: eye averages + nose[30] + mouth[48,54]
  -> ArcFace: 112x112 alignment -> source embedding
  -> model_matrix.bin: embedding * 512x512 matrix -> L2 normalize
  -> InSwapper: target 128x128 alignment + projected embedding
  -> bounded mask and inverse-affine paste-back
  -> optional GFPGAN 512x512 alignment, paste-back, configured blend
  -> owned CV_8UC3 output
```

`TensorRtFaceSwapApplication` 同步且单实例不可重入。输入 `cv::Mat` 只在调用期间借用且不被
修改，成功结果拥有自己的像素内存；任一阶段失败时抛出带阶段上下文的
`FaceApplicationError`，不会返回部分结果。若要并行处理，应为每个任务加载独立实例。

## 模型契约

| 模型 | 默认 binding | 固定 shape / 结果 |
|---|---|---|
| YOLOv12-face | `images` / compact `output0` | 现有 `TensorRtDetector` compact-NMS contract；只选 class 0 |
| Face68 (`2dfan4`) | `input`, `landmarks_xyscore`, `heatmaps` | input `[1,3,256,256]`; landmarks `[1,68,3]` |
| ArcFace | `input.1`, `683` | `[1,3,112,112]` -> `[1,512]` |
| InSwapper | `target`, `source`, `output` | `[1,3,128,128]` + `[1,512]` -> `[1,3,128,128]` |
| GFPGAN（可选） | `input`, `output` | `[1,3,512,512]` -> `[1,3,512,512]` |
| Age/Gender（可选） | `pixel_values`, `logits` | `[1,3,224,224]` -> raw `[1,2]` logits |

`model_matrix.bin` 不是 TensorRT binding，而是 InSwapper 必需 sidecar。文件必须恰为
1,048,576 bytes（512×512 个 little-endian host `float`），所有元素必须有限；加载后由
projector 不可变拥有。模型实际 binding 与上表不一致时应通过 `FaceSwapOptions` 显式配置，
不会猜测别名。

## 构建

Windows 使用调用方提供的 TensorRT 和 OpenCV Lite：

```powershell
$env:TENSORRT_ROOT = 'C:\projects\TensorRT-11.2.1.2'
$env:OPENCV_LITE_ROOT = 'C:\projects\cpp\external\pkgs\opencv-lite'
cmake --preset win-face-applications-release-user
cmake --build --preset win-face-applications-release-user
ctest --preset win-face-applications-release-user -R '^test_face_' --output-on-failure
```

对应配置开启 `KFCORE_BUILD_TENSORRT_RUNTIME`、`KFCORE_BUILD_FACE_MODELS`、
`KFCORE_BUILD_TENSORRT_YOLO`、`KFCORE_BUILD_YOLO_OPENCV`、
`KFCORE_BUILD_FACE_APPLICATIONS` 和 `KFCORE_BUILD_FACE_APPLICATION_EXAMPLES`。

## 命令行应用

所有必需路径都须显式提供；GFPGAN 和 Age/Gender 仅在给出路径时加载：

```powershell
face_swap_image.exe `
  --source source.jpg --target target.jpg --output swapped.png `
  --detector yolov12n-face.engine --face68 2dfan4.engine `
  --arcface arcface_w600k_r50.engine --inswapper inswapper_128.engine `
  --matrix model_matrix.bin --gfpgan gfpgan_1.4.engine
```

缺少参数、重复参数、未知参数和不可读输入/模型文件会在加载 engine 前失败。应用只在完整换脸
成功后调用 `imwrite`；`--output` 不得与 source/target 使用同一路径。

## 真实模型 opt-in 测试

`KFCORE_BUILD_FACE_APPLICATION_INTEGRATION_TESTS=ON` 时，必须同时提供下列绝对路径：

- `KFCORE_FACE_APPLICATION_TEST_ENGINE_12FACE`
- `KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_FACE68`
- `KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE`
- `KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_INSWAPPER`
- `KFCORE_FACE_APPLICATION_TEST_MATRIX`
- `KFCORE_FACE_APPLICATION_TEST_SOURCE_IMAGE`
- `KFCORE_FACE_APPLICATION_TEST_TARGET_IMAGE`

若 `KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_GFPGAN` 也存在，同一测试额外验证增强路径。验收内容是
TensorRT 执行成功、输出可重新解码为 `CV_8UC3`、输出尺寸等于 target，且 source/target
内存不变；这不是模型 accuracy 或身份相似度的 golden test。
