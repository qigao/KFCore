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

构建公共契约时设置 `KFCORE_BUILD_SIFT=ON`，安装后链接 `KFCore::sift`。

## PopSift 适配器

Windows 开发环境可直接使用本地 PopSift 源码树：

```powershell
cmake --fresh --preset win-sift-release-user
cmake --build --preset win-sift-release-user
ctest --preset win-sift-release-user -R '^test_(sift|popsift)' --output-on-failure
cmake --build --preset install-win-sift-release-user
```

该 preset 将 `POPSIFT_ROOT` 设为 `C:/projects/cpp/KFCore/popsift-develop`。其他布局可在环境中设置
`POPSIFT_ROOT` 后启用 `KFCORE_BUILD_SIFT=ON` 和 `KFCORE_BUILD_SIFT_POPSIFT=ON`。KFCore 在隔离的
binary dir 中通过 `add_subdirectory(... EXCLUDE_FROM_ALL)` 构建上游源码，不复制或修改它。安装后
链接 `KFCore::sift_popsift`。启用适配器的安装会把此次源码构建得到的 PopSift 动态库和
`COPYING.md` 一并安装，因此开发、测试和消费都不要求系统预装 PopSift；部署时仍须满足其
MPL-2.0 许可证及 CUDA 运行时要求。

```cpp
#include <kfcore/sift/popsift_extractor.hpp>

kfcore::sift::PopSiftOptions options;
options.device = 0;
options.max_image_bytes = 64U * 1024U * 1024U;
options.max_features = 100000U;
options.normalization = kfcore::sift::PopSiftDescriptorNormalization::RootSift;
kfcore::sift::PopSiftExtractor extractor(options);
```

适配器只接受 Host Gray8/BGR8/RGB8，先复用 ImageProcessor 生成紧密 Gray8，再同步等待一个 PopSift
job。每个 extremum 的每个 orientation 展开成一个 KFCore feature。`max_features` 是返回 descriptor
数的硬上限，同时用于配置上游 extrema 过滤；`max_image_bytes` 同时限制源跨度和灰度工作区。
返回坐标和 scale 使用原输入图像的像素坐标系，方向单位为弧度。`normalization` 可显式选择经典
L2 SIFT 或 RootSift，默认保持 PopSift 0.10.1 的 RootSift；调用者不应混合不同归一化模式的
descriptor。

PopSift 0.10.1 使用 device-global 常量和缓冲指针，直接创建同 device 的多个原生对象会发生状态
串扰。适配器因此让同进程、同 device、同 `max_features`/`normalization` 的多个逻辑实例共享一个
原生后端；它们可并发投递到上游的线程安全队列。不同进程各有独立 CUDA context，可各自创建后端。
同 device 同时请求不同算法配置会明确报 `ResourceLimitExceeded`，旧后端全部释放后即可换配置。
每个 extractor 的 `max_image_bytes` 仍独立生效；析构前必须停止对该逻辑实例的并发调用。
PopSift 0.10.1 的 extracting worker 本身没有异常捕获，
`SiftJob::getHost()` 也不重抛 worker 错误；KFCore 能转换同步异常、null job/result 和畸形计数，但
无法从上游工作线程的进程级终止恢复。需要更强故障隔离时，应把 PopSift 放进独立进程。
