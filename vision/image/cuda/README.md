# KFCore ImageProcessor

`KFCore::image_processor_cuda` 是独立于推理 runtime 的 CUDA 图像处理模块。当前支持 Host
Gray8/BGR8/RGB8 到紧密 Gray8，并支持 BGR8/RGB8/NV12/I420/NV21/YUY2/UYVY 的 Host 或同设备
CUDA 输入输出 CUDA
FP16/FP32 NCHW Tensor；一次 fused kernel
完成可选水平镜像、BT.601 YUV 转换、双线性 letterbox、RGB/BGR 通道排列、`pixel / 255`、
mean/stddev 归一化和 HWC→NCHW。
同步 facade 还可提供有界、可复用的 CUDA inference-output tensor，并把 RGB NCHW tensor 通过
FP32 alpha mask 和仿射变换直接合成到 CUDA RGB/BGR 图像，最后按调用方明确的边界下载 packed
BGR。它适用于 TensorRT 或其他能读写 CUDA pointer 的后端，不依赖特定推理库。
模块不依赖 TensorRT、ONNX Runtime 或 OpenCV。

## 构建

Windows 使用标准 Release preset；ImageProcessor 会随 KFCore 一起构建：

```powershell
cmake --fresh --preset win-release-user
cmake --build --preset win-release-user
ctest --preset win-release-user -R '^test_image_processor(_cuda)?$'
```

本仓库的 `CMakeUserPresets.json` 从 Windows 安装器提供的 `CUDA_PATH_V12_8` 读取
`CUDA_TOOLKIT_ROOT`，并同时设置 CUDA compiler、CUDAToolkit 查找根和运行时 `PATH`，避免机器上
并存的更新 toolkit 被 CMake 静默选中。若本机使用其他已验证版本，应在用户 preset 中同步修改该
环境来源；路径无效时 configure 必须失败。

安装后使用 `find_package(KFCore CONFIG REQUIRED)` 并链接 `KFCore::image_processor_cuda`。

## API 与生命周期

```cpp
#include <kfcore/image_processor/image_processor.hpp>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

using namespace kfcore::image;

void enqueue_rgb(const std::uint8_t* rgb_bytes, std::size_t rgb_byte_size,
                 std::int32_t width, std::int32_t height, std::size_t row_stride,
                 void* device_tensor,
                 std::size_t device_tensor_bytes, std::int32_t output_width,
                 std::int32_t output_height, void* pinned_workspace,
                 std::size_t pinned_workspace_bytes, void* device_workspace,
                 std::size_t device_workspace_bytes, std::size_t max_source_bytes,
                 std::size_t max_tensor_bytes, cudaStream_t cuda_stream)
{
    const std::vector<ImageView> images = {
        {rgb_bytes, rgb_byte_size, width, height, row_stride, PixelFormat::Rgb8,
         MemoryKind::Host},
    };
    const TensorView tensor = {
        device_tensor, device_tensor_bytes, 1, 3, output_height, output_width,
        TensorElementType::Float32, TensorLayout::Nchw, MemoryKind::CudaDevice,
    };
    const BatchPlan plan = ImageProcessor::plan(
        images, tensor, max_source_bytes, max_tensor_bytes);

    // pinned_workspace 由 cudaMallocHost 分配；device_workspace 由 cudaMalloc 分配。
    ImageProcessor::stage_host_inputs(
        images, plan, {pinned_workspace, pinned_workspace_bytes});
    ImageProcessor::enqueue(
        images, tensor, plan,
        {pinned_workspace, pinned_workspace_bytes},
        {device_workspace, device_workspace_bytes},
        PreprocessOptions{}, cuda_stream);
}
```

每个 `ImageView::byte_size` 都是从 `data` 起可访问的实际容量；`plan()` 会用宽高和 stride
计算最小 source span，并在任何 CPU 读取或 CUDA 提交前拒绝容量不足的视图。`plan()` 返回
`host_staging_bytes`、`device_staging_bytes` 和 `tensor_bytes` 的精确容量下界。
没有 Host 输入时两个工作区可为空。`stage_host_inputs()` 返回后原 Host 图像可释放；pinned
工作区、device 工作区、CUDA 输入和输出 Tensor 必须保持有效且不得覆盖，直到传入 stream 上的
任务完成。模块不保留任何视图或 stream，也不隐式同步。

`plan()` 的四个参数依次是输入 batch、输出 Tensor、可接受的源图总跨度上限和 Tensor 字节上限；
成功时返回 transform 与工作区需求。`stage_host_inputs()` 要求输入与 plan 未被修改，并校验 host
工作区容量。`enqueue()` 要求 plan、输入、Tensor 和两块工作区一致；成功只表示任务已提交，CUDA
执行错误仍由调用方在 stream 同步边界处理。

Host 灰度输出不需要 CUDA stream 或工作区。先用
`packed_grayscale_bytes(image, max_source_bytes)` 验证视图并取得精确输出字节数，再分配输出并调用
`stage_host_grayscale(image, destination, max_source_bytes)`。Gray8 输入逐行去除 padding；RGB/BGR
使用固定 Q16 BT.601 权重转换，因此结果不依赖浮点舍入。两次调用都只在返回前借用输入，不保留
指针；source 与 destination 不得重叠。CUDA-device 输入会明确失败，不会隐式执行 device-to-host
复制。

NV12/NV21/I420 要求偶数宽高；`row_stride` 是偶数 Y stride。NV12/NV21 的交错 UV/VU plane
紧随完整 Y plane并使用同一 stride；I420 的 U/V plane 依次紧随 Y plane，chroma stride 为 Y
stride 的一半。YUY2/UYVY 要求偶数宽度，每两个像素分别按 `Y0 U Y1 V` / `U Y0 V Y1`
占用四字节，`row_stride >= width * 2`。Host staging 会去除 padding。`mirror_horizontal` 在
letterbox 源坐标中融合执行，不会先分配或写出一张镜像图。

参数错误、容量不足、溢出、不支持格式、CUDA 指针设备不匹配和 CUDA 调用失败分别通过
`ImageProcessorError::{code(),what()}` 报告。Tensor 路径只接受连续 NCHW Tensor 和上述连续
单指针图像；Gray8 仅用于 Host 灰度输出。独立指针/stride 的多 plane、P010 及 CUDA array
尚不属于该契约，当前不会隐式 fallback。

`CudaImageProcessor` 是同步、单实例不可重入的便利 facade：

- `stage()` 返回 processor-owned CUDA image；下次 `stage()` 使该 view 失效。
- `process_affine()` 返回 processor-owned preprocess tensor；下次同名调用使该 view 失效。
- `acquire_tensor()` 返回可由任意 CUDA inference backend 写入的独立 tensor storage；下次
  `acquire_tensor()` 使该 view 失效。
- `composite_affine()` 接收 CUDA RGB NCHW、Host/CUDA FP32 alpha、
  BGR/RGB/NV12/I420/NV21/YUY2/UYVY CUDA
  base 和 destination-to-aligned transform，始终返回独立 packed CUDA BGR8；允许把上一次合成
  结果原位作为 base。下次合成会更新同一 owned storage。
- `download_bgr()` 是显式同步 device→host 边界，目标 buffer 由调用方拥有。

这几类 storage 彼此独立，因此可按“预处理 → 外部推理写入 → 合成 → 再预处理”的顺序复用，
无需中间图像回传。所有容量由 `CudaImageProcessorOptions::{max_source_bytes,max_tensor_bytes}`
约束；跨 device pointer、重叠 tensor/output、非有限 tensor 值和容量不足都会立即失败。

## AprilTag 边界

vendored AprilTag 当前读取 CPU `image_u8_t` 灰度图。Host 相机的 Gray8/BGR8/RGB8 帧可通过
`stage_host_grayscale()` 生成紧密灰度 buffer，再以调用者拥有的生命周期交给 AprilTag。CUDA-device
帧仍需调用者显式决定 device→host 边界；本模块不会隐藏该同步和复制成本。
