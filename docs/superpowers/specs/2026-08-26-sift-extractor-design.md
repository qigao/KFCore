# KFCore SIFT 提取模块设计

## 背景与目标

本地 `sift-cpp-master` 缺少可验证的构建、测试和许可证信息，且审查发现 Hessian 偏移计算错误；
`popsift-develop` 提供 CUDA SIFT，但公开 API 使用原始指针、内部异步队列，并只接受紧密排列的
Host 灰度图。KFCore 不复制这两份源码。本次新增自有 SIFT 契约和可选 PopSift 薄适配层，复用
`KFCore::image_processor` 的图像视图与灰度转换。

## 候选方案

1. 直接暴露 PopSift 类型：改动少，但把第三方 ABI、所有权和错误模型扩散给调用者，拒绝。
2. 将 PopSift vendor 进仓库：可直接补丁，但引入 MPL-2.0 源码、CUDA 构建和长期维护成本，拒绝。
3. 自有策略接口加可选薄适配器：公共契约稳定，依赖可选，后续可增加其他实现，采用。

## 结构与依赖

- `KFCore::sift`：公共 `SiftExtractor` 策略接口、自有 Feature/错误类型，公开依赖
  `KFCore::image_processor` 的 `ImageView`。
- `KFCore::sift_popsift`：可选 PopSift 实现，使用 Pimpl 隔离第三方头文件；仅在
  `KFCORE_BUILD_SIFT_POPSIFT=ON` 且 `POPSIFT_ROOT` 指向 PopSift 源码树时，以隔离 binary dir
  参与开发构建。
- `ImageProcessor::stage_host_grayscale`：将 Host Gray8/BGR8/RGB8 视图转换为紧密 Gray8；不接受
  CUDA-device 输入，也不做隐式 device-to-host fallback。

依赖保持单向：调用者 -> SIFT 契约 -> ImageProcessor 类型；PopSift 细节只存在于适配器实现。

## 数据、所有权与错误

`ImageView` 在 `extract()` 调用期间借用，适配器先验证尺寸、stride、容量和上限，再生成自持有的
紧密灰度缓冲。返回的 `FeatureSet` 完全拥有特征和 128 维 descriptor，不保留 PopSift 指针。
每个 PopSift extremum 的每个 orientation 展开为一个 KFCore feature。
方向以弧度表达；descriptor 归一化模式由显式选项控制，默认 RootSift。

`max_image_bytes` 限制可访问源跨度和灰度工作区，`max_features` 同时传给 PopSift 的 extrema
过滤器并作为返回 descriptor 的硬上限。非法契约报 `InvalidArgument`，容量/溢出报
`ResourceLimitExceeded`，第三方异常、空结果或畸形结果报 `BackendFailure`。错误不被静默吞掉。

## 并发、背压与关闭

一个适配器实例内部用 mutex 串行化 `extract()`；KFCore 不创建任务线程或队列。一次调用立即等待
该任务结果，因此每实例最多一个 PopSift 在途任务，调用线程的同步阻塞就是背压。不同实例可由
调用者并行使用。析构前调用者必须确保没有并发调用；这是常规 C++ 对象生命周期约束。

PopSift 自己仍拥有工作线程和队列，其 worker 异常模型属于上游实现；适配器会处理同步可观察到的
异常、null job 和 null result，但无法恢复第三方进程级终止。这一残余风险必须在 README 中披露。

## 兼容性、迁移与回滚

`PixelFormat::Gray8` 是枚举的向后兼容扩展；现有 Tensor 预处理仍只接受 BGR8/RGB8，Gray8 只用于
新的 Host 灰度 API。默认选项关闭，所以现有构建与部署不增加 PopSift 依赖。适配器部署仍需要
PopSift 动态库，KFCore 不复制其源码或接管第三方许可证文件。使用者通过链接 `KFCore::sift` 或
`KFCore::sift_popsift` 渐进迁移。回滚可删除新模块和枚举/API 扩展，不改变已有
YOLO、AprilTag 或 Kalman 数据格式。

## 验证范围

- TinyTest：灰度精确值、padding、Gray8 复制、输入/容量/溢出/内存类型拒绝。
- TinyTest：SIFT 错误契约和公共类型基本行为。
- 条件构建：PopSift 适配器编译与真实 API 对齐。
- 安装消费测试：`KFCore::sift` 导出和运行；启用 PopSift 时检查依赖导出。
- 相邻回归：ImageProcessor CPU/CUDA 测试和完整 CTest。
