# KFCore SIFT 提取模块设计

## 背景与目标

KFCore 需要可复用、可多实例调用的 CUDA SIFT，同时不能把第三方 ABI、原始指针或生命周期规则扩散给调用者。
原先 PR 只对本机 `popsift-develop` 做薄适配，构建依赖 `POPSIFT_ROOT`，且继承了上游无界队列、worker
异常无法可靠传回调用线程、未检查 worker 设备选择等问题。这不满足“开发我们自己的版本”的目标。

本设计保留已建立的 `KFCore::sift` 和 `KFCore::sift_popsift` 公共契约，把 PopSift 0.10.1 的算法源码
作为可追溯的内部 fork 纳入 `sift/vendor/popsift`，由 KFCore 直接构建、测试和维护。外部模型或应用代码
不进入该目录。

## 上游来源与许可证

- 上游：`https://github.com/alicevision/popsift.git`
- 基线分支/提交：`develop` / `36d704d39b4cc065839d84f3706b3fa88eff2518`
- 上游项目版本：`0.10.1`
- 许可证：MPL-2.0
- 导入范围：`src/popsift` 的 63 个算法/运行时文件、`cmake/sift_config.h.in` 和 `COPYING.md`
- 不导入：application、sample、文档站点、独立包导出文件和上游测试命令

`sift/vendor/popsift/UPSTREAM.md` 记录来源、commit、导入范围、本地修改和复验命令。所有上游文件保留
原 MPL 头；修改过的 MPL 文件仍作为源码随仓库提供。安装产物附带许可证和来源说明。

## 候选方案

1. 继续依赖外部 `POPSIFT_ROOT`：改动最小，但构建不可复现，修复与 KFCore 版本脱节，拒绝。
2. 重新实现完整 CUDA SIFT：控制力最高，但算法验证、性能回归和迁移成本过高，拒绝。
3. 维护精简 PopSift fork，并以 KFCore 契约隔离：复用已验证算法，同时拥有构建、并发和错误语义，采用。

## 结构与依赖

- `KFCore::sift`：公共同步策略接口、自有 Feature/错误类型，依赖 `KFCore::image_processor` 的 `ImageView`。
- `KFCore::sift_popsift`：公开的 CUDA 实现，继续用 Pimpl 隐藏内部 fork 类型。
- `kfcore_popsift_internal`：仅构建树可见的静态 CUDA 目标，不安装头文件、不导出 CMake target。
- `sift/vendor/popsift`：MPL-2.0 算法 fork、内部有界队列和来源记录。

依赖单向为：调用者 -> KFCore SIFT 契约 -> PopSift 适配器 -> 内部 CUDA 算法。删除 `POPSIFT_ROOT`，
启用 `KFCORE_BUILD_SIFT_POPSIFT=ON` 时只额外要求项目已有的 CUDAToolkit。

## 数据、所有权与错误

`ImageView` 仅在 `extract()` 调用期间借用。适配器先验证尺寸、stride、源跨度和容量，再生成自持有的紧密
Gray8 缓冲。`SiftJob` 拥有该次提交的图像副本；队列只持有非 owning 指针处理租约，调用线程在 future
完成后销毁 job。返回的 `FeatureSet` 完全拥有特征和 128 维 descriptor，不保留内部指针。

`max_image_bytes` 限制源跨度与灰度工作区，`max_features` 同时限制算法过滤与结果，`max_pending_jobs`
限制 stage-1 待处理队列。所有乘法在分配前检查溢出。非法契约报 `InvalidArgument`，容量/溢出报
`ResourceLimitExceeded`，CUDA、worker 或畸形结果报 `BackendFailure`，不返回部分成功结果。

worker 失败通过 `std::promise::set_exception` 传回 `getHost()`/`getDev()`。单 job 的上传或提取异常必须
归还已取得的 staging image、释放局部 result 并完成该 job；不得导致调用线程永久等待。worker 启动失败
会关闭输入队列、唤醒 producer，并使未完成 job 以同一错误结束。

## 并发、容量、背压与关闭

拓扑是多 producer -> 共享 admission gate -> 单 upload worker -> 单 CUDA extract worker。gate 和
stage-1 容量均为 `max_pending_jobs`，stage-2 与 staging image pool 容量均为 2。permit 在灰度 staging
前取得并持有到结果复制完成；队列在构造期预分配固定存储，push/pull 期间不分配。满载 producer 阻塞，
队列关闭后立即失败并被唤醒。

关闭顺序为：禁止新提交并关闭 stage-1 -> upload worker 排空已有任务并关闭 stage-2 -> extract worker
排空并完成 promise -> join 两个 worker -> 释放 staging image 与 pyramid。调用者仍必须保证析构时没有线程
继续使用同一个逻辑 extractor，这是 C++ 对象生命周期边界。

PopSift 的高斯常量和 pyramid 指针是 device-global 状态。因此同进程同 device 只保留一个原生后端；
相同 `max_features`、normalization 和 `max_pending_jobs` 的多个逻辑实例共享它，不同配置在旧后端释放前
fail fast。多个调用线程可并发提交，但单 device 的 CUDA 算法按内部 pipeline 顺序执行。真正的同 device
多 stream 并行必须先移除 device-global 状态并以 benchmark 证明收益，不在本次迁移中伪称支持。

## 兼容性、迁移与回滚

现有 `SiftExtractor`、`PopSiftExtractor`、Feature 和错误类型保持不变。`PopSiftOptions` 只新增有默认值的
`max_pending_jobs`，源兼容已有聚合默认构造；不同值参与后端兼容性判断。构建不再读取
`POPSIFT_ROOT`，安装也不再部署独立 `popsift.dll`，因此部署产物减少一个运行时 DLL。

默认选项仍关闭，未启用 SIFT 的 YOLO、AprilTag 和 Kalman 构建不增加依赖。回滚可以恢复外部适配提交，
不会改变这些模块的数据格式。MPL 源码与 KFCore 适配器保持文件边界，避免许可证语义扩散到公共头文件。

## 验证范围

- TinyTest：`max_pending_jobs` 默认值、零值和上限校验。
- TinyTest：有界队列 FIFO、满队列背压、close 唤醒、关闭后 push 失败与排空语义。
- TinyTest：`SiftJob::getHost()` 和 `getDev()` 都能收到 worker exception。
- CUDA 实测：真实 checkerboard 提取、多逻辑实例、并发提交、冲突配置拒绝及重复构造销毁。
- CMake：清除 `POPSIFT_ROOT` 后 preset 配置、构建、安装和独立 consumer 运行。
- 回归：SIFT 最小测试后运行完整 `ctest --preset win-sift-release-user`。
- 性能：同一持久 extractor 上记录内部化前后多次提取耗时；若出现显著回退，先定位再优化。
