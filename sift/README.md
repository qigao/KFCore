# KFCore SIFT

`KFCore::sift` 定义不依赖具体算法库的同步 SIFT 提取接口。输入复用
`kfcore::image::ImageView`，输出 `FeatureSet` 完全拥有关键点和 128 维 descriptor；第三方类型、指针和
释放规则不会越过模块边界。

```cpp
#include <kfcore/sift/sift_extractor.hpp>

kfcore::sift::FeatureSet extract(
    kfcore::sift::SiftExtractor& extractor,
    const std::uint8_t* pixels,
    std::size_t bytes,
    std::int32_t width,
    std::int32_t height,
    std::size_t stride)
{
    const kfcore::image::ImageView image = {
        pixels, bytes, width, height, stride,
        kfcore::image::PixelFormat::Rgb8,
        kfcore::image::MemoryKind::Host,
    };
    return extractor.extract(image);
}
```

`image` 只在调用期间借用；调用返回后即可释放或复用输入。成功结果由调用者按普通值对象管理。
失败抛出 `SiftError`：`InvalidArgument` 表示输入契约错误，`ResourceLimitExceeded` 表示配置上限或
字节计算溢出，`BackendFailure` 表示具体实现失败。

公共契约和 CUDA 后端会随 KFCore 一起构建；安装后分别链接 `KFCore::sift` 和
`KFCore::sift_popsift`。

## CUDA SIFT 后端

Windows 开发环境直接构建仓库内维护的 CUDA SIFT 源码：

```powershell
cmake --fresh --preset win-release-user -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build --preset win-release-user
ctest --preset win-release-user -R '^test_(sift|popsift)' --output-on-failure
cmake --build --preset win-release-user --target benchmark_popsift
build/Msvc-Release/bin/benchmark_popsift.exe
```

KFCore 从 `sift/vendor/popsift` 构建私有静态 CUDA target；不读取 `POPSIFT_ROOT`，也不要求本机安装 PopSift。
安装后链接 `KFCore::sift_popsift`。内部头文件和独立 `popsift.dll` 不会安装；安装包附带
`COPYING.md` 和 `UPSTREAM.md`，分别记录 MPL-2.0 许可证、上游 commit 和 KFCore 修改。

```cpp
#include <kfcore/sift/popsift_extractor.hpp>

kfcore::sift::PopSiftOptions options;
options.device = 0;
options.max_image_bytes = 64U * 1024U * 1024U;
options.max_features = 100000U;
options.max_pending_jobs = 8U;
options.normalization = kfcore::sift::PopSiftDescriptorNormalization::RootSift;
kfcore::sift::PopSiftExtractor extractor(options);
```

适配器只接受 Host Gray8/BGR8/RGB8，先复用 ImageProcessor 生成紧密 Gray8，再同步等待一个 PopSift
job。每个 extremum 的每个 orientation 展开成一个 KFCore feature。`max_features` 是返回 descriptor
数的硬上限，同时用于配置 extrema 过滤；`max_image_bytes` 同时限制源跨度和灰度工作区。
`max_pending_jobs` 限制同一原生后端的完整在途 job 数，范围是 1 到 1024，默认 8；permit 在灰度
staging 前取得并持有到结果复制完成，满载时 `extract()` 产生背压，不会在内部队列之外继续复制图像。
返回坐标和 scale 使用原输入图像的像素坐标系，方向单位为弧度。`normalization` 可显式选择经典
L2 SIFT 或 RootSift，默认保持 PopSift 0.10.1 的 RootSift；调用者不应混合不同归一化模式的
descriptor。

PopSift 0.10.1 使用 device-global 常量和缓冲指针，直接创建同 device 的多个原生对象会发生状态
串扰。适配器因此让同进程、同 device、同 `max_features`/`normalization` 的多个逻辑实例共享一个
原生后端；它们可并发投递到有界的两阶段 pipeline。不同进程各有独立 CUDA context，可各自创建后端。
同 device 同时请求不同算法配置会明确报 `ResourceLimitExceeded`，旧后端全部释放后即可换配置。
`max_pending_jobs` 也属于共享后端配置；每个 extractor 的 `max_image_bytes` 仍独立生效。worker 的 CUDA、
上传和提取异常通过 job future 返回并由适配器转换为 `BackendFailure`；关闭会排空已接受任务并唤醒
被背压阻塞的线程。析构前必须停止对该逻辑实例的并发调用。

同一 device 上多个逻辑实例不等于多个 CUDA kernel pipeline：device-global 状态要求它们共享一个原生
后端，算法按 pipeline 顺序执行。若需要同 device 多 stream 并行，必须先移除这些全局状态并通过
benchmark 验证收益。
