# Task 4：TensorRT/CUDA 精确发现与资源所有权

提交信息：`feat(yolo): own TensorRT engine and CUDA resources`

## 实现

- 新增 `KFCORE_BUILD_TENSORRT_YOLO`（默认 OFF）和 `KFCORE_BUILD_TENSORRT_INTEGRATION_TESTS`（默认 OFF）。TensorRT ON、tracking OFF 会在 configure 阶段 fail fast；选项不互相 `FORCE`。
- root project 只在 TensorRT ON 时查找 TensorRT、启用 CUDA language 并查找 CUDAToolkit。tracking-only profile 保持纯 C/C++，其 cache 不产生 `CMAKE_CUDA_COMPILER`。
- 新增 `win-yolo-release-user` configure/build/test preset 和 `install-win-yolo-release-user`；`TENSORRT_ROOT`、`OPENCV_LITE_ROOT` 均从唯一的 `PKG_ROOT` 派生。
- `FindTensorRT.cmake` 只接受显式 `TENSORRT_ROOT`：先 canonicalize root，再用 `NO_DEFAULT_PATH` 查找 `NvInfer.h`、`nvinfer`、`nvinfer_plugin`；解析 `NV_TENSORRT_MAJOR/MINOR/PATCH`，仅接受 10/11；所有 header/library 再经 real-path containment 检查；最终创建 `TensorRT::nvinfer` 和 `TensorRT::nvinfer_plugin` imported targets。
- 新增公开 `Engine` / `TensorRtDetector` API。`Engine::load()` 在访问 CUDA 前完成 checked/bounded engine file read，随后按 plugin init、logger、runtime、engine、named metadata extraction、Task 3 contract validation 的边界返回 typed errors。
- TensorRT engine metadata 仅使用 10/11 named tensor API：`getNbIOTensors()`、`getIOTensorName()`、`getTensorIOMode()`、`getTensorDataType()`、`getTensorShape()`、`getProfileShape()`；未使用 binding-index API。
- TensorRT owner 使用 `std::unique_ptr<T, TensorRtDeleter<T>>` 和 `delete`。`Engine::State` 声明顺序为 logger → runtime → engine → validated contract/options，逆序析构保证 engine 早于 runtime/logger 释放。
- `Engine` 持有 `std::shared_ptr<const State>`；每个 detector 独占 execution context、non-blocking CUDA stream、五组 device buffers 与 input/output pinned host buffers。创建 detector 时调用 `cudaSetDevice(device_id)`；Task 5 的 inference boundary 将再次设置 device。
- `CudaBuffer` / `PinnedHostBuffer` 单 owner、move-only。`data()` 是 borrow view，在 reserve/move/destruction 后失效。`reserve(bytes, hard_limit)` 先拒绝超限，再分配 replacement，旧 allocation 释放成功后才提交；分配失败保持旧 pointer/capacity。
- Task 4 不实现 letterbox、inference、`setTensorAddress()`、`enqueueV3()` 或 postprocess。`detect()` / `detect_batch()` 仅为完整 PImpl 公开声明；Task 4 target 不引用它们，Task 5 增加执行 source 后再提供定义。

所有权协议：engine state 是共享只读事实源；detector 是 context/stream/buffer 的唯一可变 owner；无跨 detector 可变共享。所有 buffer 以 bytes 为容量单位，hard limit 来源为 `EngineOptions`；满额或算术/分配失败均抛 typed `YoloError`，不降级、不静默扩容。对象不承诺并发调用同一 detector。

TensorRT API 依据：NVIDIA [ICudaEngine named tensor API](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/classnvinfer1_1_1_i_cuda_engine.html)、[IRuntime lifetime/deserialization](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/classnvinfer1_1_1_i_runtime.html) 和 [initLibNvInferPlugins](https://docs.nvidia.com/deeplearning/tensorrt/10.15.1/_static/c-api/_nv_infer_plugin_8h.html)。CUDA owner 依据 NVIDIA [memory management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html) 与 [stream management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)。

## RED → GREEN

- **RED（事实）**：基线以 TensorRT ON、tracking OFF configure，exit 0 且只报告 `KFCORE_BUILD_TENSORRT_YOLO` 未使用。新增 gate 后相同命令 exit 1，明确报告 `KFCORE_BUILD_TENSORRT_YOLO requires KFCORE_BUILD_YOLO_TRACKING=ON`。
- **RED（事实）**：先新增 `test_tensorrt_integration.cpp`；direct MSVC compile 因 `kfcore/yolo/tensorrt.hpp` 不存在而 C1083。新增公开 API 后同一 test source compile exit 0。
- **RED（事实）**：隔离 FindTensorRT configure test 在 module 不存在时只能报告找不到 `FindTensorRT.cmake`。实现后，无 root 明确报告 `TENSORRT_ROOT is required`；fake 10.15.1/11.2.1 roots configure 成功，fake 9.x 明确拒绝；外部 cached library 注入被清除后仍选择 root 内 library；root 内 junction 指向 root 外时 containment 明确拒绝。

## 验证

在 VS DevCmd 下执行并观察：

```powershell
cmake --fresh --preset win-yolo-release-user -DVCPKG_MANIFEST_MODE=ON -DBUILD_TESTING=OFF
cmake --fresh --preset win-yolo-tracking-dev-user -DVCPKG_MANIFEST_MODE=ON -DBUILD_TESTING=OFF
cmake --build --preset win-yolo-tracking-dev-user --target test_tensorrt_contract test_yolo_tracking kfcore_yolo_tracking
build\Msvc\bin\test_tensorrt_contract.exe
build\Msvc\bin\test_yolo_tracking.exe
git diff --check
```

结果：

- 真实 YOLO preset 在启用 CUDA 前 exit 1，精确报告 `C:/projects/cpp/external/pkgs/tensorrt` 不是目录；未回退系统 SDK。
- tracking-only configure/build exit 0，cache 中无 `CMAKE_CUDA_COMPILER`。
- `test_tensorrt_contract`：10/10 tests、86 assertions 通过。
- `test_yolo_tracking`：11/11 tests、57 assertions 通过。
- 以 NVIDIA 10/11 named API 签名构造的临时、ignored API stub 配合本机 CUDA headers，`engine.cpp` 与 `cuda_buffer.cpp` 在 MSVC `/W4 /permissive-` 下编译无诊断。补充 smoke 验证 missing engine 返回 `FileIo`，move-only/hard-limit成立，CUDA allocation failure 后旧 buffer pointer/capacity 保持。
- forbidden API 检索不含 `.destroy()`、binding-index、`enqueueV2()`、`executeV2()`；Task 4 也未提前实现 Task 5 的 enqueue。

`BUILD_TESTING=OFF` 不登记新的 contract/integration CTest，因此 tracking-only 两个精确 test target 直接执行；运行时使用 preset 等价 PATH 提供依赖 DLL。

## 未完成的外部验收与风险

- **HIGH（事实）**：`C:\projects\cpp\external\pkgs\tensorrt` 不存在，且本机检索未找到 `NvInfer.h` / `NvInferVersion.h`。因此 `tensorrt_yolo` **未使用真实 TensorRT 10/11 SDK 编译或链接**；SDK ABI、实际 distribution library naming 和编译器兼容性仍需在安装 SDK 后验证。
- **HIGH（事实）**：没有可信 serialized engine，因此未运行 plugin-backed deserialization、named metadata extraction、Task 3 contract handoff或 `createExecutionContext()`；未声称这些 GPU acceptance criteria 通过。
- **MED（事实）**：本机 CUDA 13.0 runtime 的 `cudaSetDevice(0)` 返回 801 (`cudaErrorNotSupported`)；相同补充 smoke 链接本机 CUDA 12.8 后 exit 0。真实 TensorRT SDK 到位后需使用其支持的 CUDA toolkit/profile 重跑 target 和 integration test。
- **MED（推论）**：当前 FindTensorRT 已同时接受 Windows version-suffixed import libraries 与 generic Unix names，但没有真实 SDK package 可验证其最终目录布局；containment 和缺件错误已由隔离 fixtures 验证。

待 SDK 到位后复验：

```powershell
cmake --fresh --preset win-yolo-release-user
cmake --build --preset win-yolo-release-user --target tensorrt_yolo
```

若再提供可信 engine，可显式启用 `KFCORE_BUILD_TENSORRT_INTEGRATION_TESTS=ON` 构建 Task 4 missing-file smoke；真实 inference acceptance 属于 Task 5。
