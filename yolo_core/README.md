# YOLO CPU/CUDA 后端

YOLO 按职责拆成三个可安装 target：

- `KFCore::yolo_core`：`ImageView`、检测结果、共享 compact-NMS 解码和 ByteTrack；
- `KFCore::yolo_onnx`：ONNX Runtime CPU detector；
- `KFCore::yolo_tensorrt`：TensorRT/CUDA detector。

生产 target 不依赖 OpenCV，也不公开 `cv::Mat` adapter。`yolov8_domain_demo` 使用 stb
处理图片、Turbo Capture 采集摄像头，并在 Windows 使用 Win32 GDI 显示；其他模块自己的
demo/UI 可以独立选择 OpenCV，但 OpenCV 不进入 YOLO 导出依赖。

## 选择后端

CPU 路线使用正式 API `kfcore/yolo/onnx.hpp`：

```cpp
#include <kfcore/yolo/onnx.hpp>

auto detector = kfcore::yolo::OnnxDetector::load(model_path);
kfcore::yolo::DetectionFrame detections = detector->detect(image);
```

```cmake
target_link_libraries(my_app PRIVATE KFCore::yolo_onnx)
```

CUDA 路线使用 `kfcore/yolo/tensorrt.hpp`：

```cpp
#include <kfcore/yolo/tensorrt.hpp>

auto engine = kfcore::yolo::Engine::load(engine_path);
auto detector = engine->create_detector();
kfcore::yolo::DetectionFrame detections = detector->detect(image);
```

```cmake
target_link_libraries(my_app PRIVATE KFCore::yolo_tensorrt)
```

两条路线均返回相同的 `DetectionFrame`，之后可交给 `ByteTrackSession`。后端在链接和对象
构造时显式选择；模型加载或推理失败不会切换后端。

## 输入与所有权

`ImageView` 是同步借用视图。CPU detector 只接受 Host 内存；TensorRT detector 接受 Host
或匹配 CUDA device 的内存。调用返回后 detector 不再保留输入指针。支持 BGR8、RGB8、
NV12、I420、NV21、YUY2 和 UYVY；预处理分别由 `image_processor_cpu` 与
`image_processor_cuda` 完成。

ONNX detector 要求一个名为 `images` 的静态 FP32 `[1,3,H,W]` 输入，以及一个名为
`output0` 的静态 FP32 `[1,N,6]` 输出。TensorRT detector 支持同语义 compact-NMS 输出，
以及经过严格 tensor/profile 校验的 EfficientNMS 输出。两者都要求模型已经完成 NMS，不处理
raw YOLO head。

## 构建与验证

仅使用标准 preset：

```powershell
cmake --preset win-release-user
cmake --build --preset win-release-user
ctest --preset win-release-user -R "test_yolo_" --output-on-failure
```

模型根固定为进程启动工作目录下的 `./yolo-models`；TensorRT engine 使用
`tensorrt/${KFCORE_TENSORRT_ENGINE_PROFILE}/<model>.engine`，profile 只在 CMake 配置阶段
设置并编译进库。ONNX detector 通过
`KFCore::runtime_onnx` 执行，TensorRT detector 使用 CUDA 图像处理与 TensorRT SDK；两条路线
没有自动 fallback，模型根不接受 CMake、Preset JSON、环境变量或 API 配置。
