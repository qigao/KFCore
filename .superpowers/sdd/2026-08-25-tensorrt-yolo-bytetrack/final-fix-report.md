# TensorRT YOLO / ByteTrack 最终集中修复报告

日期：2026-08-25
修复前 HEAD：47106788b72b46b585a55ee2e2a47bfb2585963a
审查基线：0a7f094bc0a7080a2a2ad09ad1f4774110408156
工作树：C:/projects/cpp/KFCore/.worktrees/tensorrt-yolo-bytetrack

## 结论

final-review-findings.md 的 6 项 finding（2 HIGH、2 MED、2 LOW）均已按 RED→GREEN
修复。真实 TensorRT 11.2.1、CUDA Toolkit 12.8.61 与 OpenCV Lite 4.13.0 下，
生产 target、track_image_sequence 和 integration test binary 均已编译/链接；所有可在
无可信 engine 条件下执行的相关单元、fixture、CUDA 和安装消费测试均通过。

C:/projects 下没有可用于本任务的可信兼容 engine。7 字节的 fixture 文件只用于验证
configure-time 路径校验和 integration target 编译，从未传给
test_tensorrt_integration.exe。因此真实 engine 反序列化、min/opt/max 三种 H/W 推理仍明确记为
UNVERIFIED，本文不声称推理通过。

## 逐 finding 的 TDD 证据

### 1. HIGH — 物理 tensor format seam

事实：

- RED：先扩展 test_tensorrt_contract，使五个张量逐一改变 format、
  vectorized dimension、components per element、bytes per component。实现尚未存在时，
  MSVC 编译失败，输出明确包含缺少 TensorPhysicalLayout、TensorPhysicalFormat 和
  TensorDesc::physical_layout，Ninja 以 1 退出。
- GREEN：engine_contract.hpp/cpp 新增完全独立于 TensorRT header 的
  TensorPhysicalFormat/TensorPhysicalLayout，并对五个张量共同要求 Linear、-1、1、与 dtype
  一致的 scalar bytes。
- GREEN：engine.cpp 从真实 ICudaEngine 的 getTensorFormat、
  getTensorVectorizedDim、getTensorComponentsPerElement、
  getTensorBytesPerComponent 填充 metadata。仅在线性非向量化时，把 TensorRT 文档规定的
  -1 scalar sentinel 规范化为 1 与 dtype bytes；没有 format fallback。
- GREEN：test_tensorrt_contract 的四组逐 tensor 拒绝用例全部通过。

涉及文件：tensorrt_yolo/src/engine_contract.hpp、engine_contract.cpp、engine.cpp、
tests/test_tensorrt_contract.cpp。

### 2. HIGH — 动态 H/W 全链路

事实：

- RED：先把 contract fixture 改成 min=[1,3,320,480]、
  opt=[2,3,640,640]、max=[4,3,960,1280]，并加入 default/显式 min、内部值、max 和越界选择。
  实现尚未存在时编译失败，输出明确包含缺少 min/opt/max input H/W 字段、
  SelectedInputSize 和 select_input_size，Ninja 以 1 退出。
- GREEN：ValidatedContract 保留 profile 0 的 min/opt/max H/W；仍要求 exactly one profile。
  只有 input 的 batch/H/W 可动态，channel 与所有 output 非 batch 维仍固定。
- GREEN：未设置 DetectorOptions::input_size 时选择 opt H/W；显式
  {height,width} 必须为正并分别落入 profile min/max。
- GREEN：detector 的 selected size 同时驱动 prepare_batch、letterbox、输入 byte 计算、
  destination slice 与 setInputShape([batch,3,H,W])。创建 detector 时仍按 profile max shape
  分配；FP32 最大输入精确锁定为
  4 × 3 × 960 × 1280 × 4 = 58,982,400 bytes。
- GREEN：integration test binary 直接检查真实 engine profile 0，并为 min/opt/max 分别建立
  detector 与推理；本轮仅证明该测试编译/链接，未运行。

涉及文件：tensorrt.hpp、engine_contract.hpp/cpp、tensorrt_raii.hpp、engine.cpp、
detector.cpp、test_tensorrt_contract.cpp、test_tensorrt_integration.cpp 与 README.md。

### 3. MED — 负 class 在状态变化前拒绝

事实：

- RED：先用 typed failure、无 class slot 预留、首个合法 track ID 仍为 0，以及混合非法帧与
  control session 等价的测试替换旧的负数 ID 编码测试；两个新用例均失败，因为旧实现没有抛错。
- GREEN：validate_frame 在 detection count limit、scratch、grouping、clone、tracker 构造/update、
  swap 之前拒绝 class_id < 0，错误为 YoloErrorCode::InvalidArgument 且消息含 class_id。
- GREEN：完整 tracking 套件 12/12、65 assertions。

涉及文件：tensorrt_yolo/src/tracking.cpp、tests/test_yolo_tracking.cpp。

### 4. MED — finder dependency-first / repeated 的精确根验证

事实：

- RED：先加入 TensorRT/OpenCV dependency_first、repeated、partial 与 root_mismatch fixture。
  原 finder 发现已有 target 后以 “requires ownership ... a target already exists” 失败。
- GREEN：已有 target 必须为 imported；所有 include、generic/per-config location/implib 必须存在、
  不含 generator expression、位于环境变量指定的 exact real root 下，并匹配预期 header 与
  library filename/version format。完整 target 集复用；partial、未验证或错根立即失败。
- GREEN：真实安装包四种模式 KFCore-first、trackers-first、dependency-first、
  repeated-KFCore 均 configure/build/run 成功。

涉及文件：FindTensorRT.cmake、FindOpenCVLite.cmake、两组 finder fixture、两组 finder 驱动脚本、
YoloTrackingInstalledConsumer 与 tensorrt_yolo/CMakeLists.txt。

### 5. LOW — OpenCV COMPONENTS 边界

事实：

- RED：adapter-only SDK 只含 core/imgproc，imgcodecs-only SDK 只含 imgcodecs。旧 finder
  无论请求内容都要求三组件，两个配置均失败。
- GREEN：FindOpenCVLite 只接受 core/imgproc/imgcodecs，拒绝重复/未知 component，只发现并导出
  requested targets；无 COMPONENTS 时兼容性默认值为 core,imgproc。
- GREEN：adapter 与安装 KFCore config 只请求 core,imgproc；
  track_image_sequence 单独请求 imgcodecs。fixture 证明未请求 target 不会被导出。
- GREEN：dumpbin 证明 kfcore_yolo_opencv.dll 只直接依赖 OpenCV core/imgproc；
  track_image_sequence.exe 才直接依赖 imgcodecs。禁用检索未发现 dnn/highgui/videoio。

涉及文件：FindOpenCVLite.cmake、KFCoreConfig.cmake.in、OpenCV fixture、
tensorrt_yolo/CMakeLists.txt 与 README.md。

### 6. LOW — integration engine 配置期 fail-fast 与 CTest 显式传递

事实：

- RED：先加入 empty/nonexistent/directory/valid configure fixture；没有 validation module/helper
  时 valid 场景不能配置，非法路径也没有目标错误语义。
- GREEN：KFCORE_TENSORRT_TEST_ENGINE 是 FILEPATH cache；integration tests 为 ON 时在任何 SDK
  discovery 之前验证非空、存在且为文件，并输出规范 real path。
- GREEN：专用 helper 把规范路径写入 test_tensorrt_integration 的 CTest ENVIRONMENT。
  fixture 同时读取 test property，并真实运行 CTest 子测试核对进程收到的环境变量。
- GREEN：补强 CTest runtime 证据时首次 RED 为 multi-config fixture 缺 -C：
  “Test not available without configuration. (Missing -C <config>?)”；加入 -C Debug 后 GREEN。

涉及文件：CMakeOptions.cmake、根 CMakeLists.txt、ValidateTensorRtIntegrationEngine.cmake、
TensorRtIntegrationEngineFixture、test_tensorrt_integration_engine_config.cmake、
tensorrt_yolo/CMakeLists.txt 与 README.md。

## 最终可复验命令与输出

所有 Windows 编译命令均先调用：

~~~text
C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat -arch=x64 -host_arch=x64
~~~

真实 SDK 环境：

~~~text
TENSORRT_ROOT=C:\projects\TensorRT-11.2.1.2
OPENCV_LITE_ROOT=C:\projects\cpp\external\pkgs\opencv-lite
CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
PATH 前置 CUDA\v12.8\bin、TensorRT\bin、opencv-lite\bin 和当前 build\bin
~~~

### Fresh real-SDK configure

命令：

~~~powershell
cmake --fresh --preset win-yolo-release-user -DVCPKG_MANIFEST_MODE=ON -DBUILD_TESTING=OFF -DKFCORE_BUILD_TENSORRT_INTEGRATION_TESTS=ON -DKFCORE_TENSORRT_TEST_ENGINE=C:/projects/cpp/KFCore/.worktrees/tensorrt-yolo-bytetrack/build/final-fixtures/tensorrt_integration_engine_config/trusted.engine
~~~

输出：

~~~text
-- The C compiler identification is MSVC 19.44.35217.0
-- Check for working C compiler: .../MSVC/14.44.35207/bin/Hostx64/x64/cl.exe - skipped
-- Found Threads: TRUE
-- The CXX compiler identification is MSVC 19.44.35217.0
-- Check for working CXX compiler: .../MSVC/14.44.35207/bin/Hostx64/x64/cl.exe - skipped
-- Found TensorRT: C:/projects/TensorRT-11.2.1.2/include (found version "11.2.1")
-- The CUDA compiler identification is NVIDIA 12.8.61 with host compiler MSVC 19.44.35217.0
-- Check for working CUDA compiler: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe - skipped
-- Found CUDAToolkit: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include (found version "12.8.61")
-- Found OpenCVLite: C:/projects/cpp/external/pkgs/opencv-lite/include (found version "4.13.0") found components: core imgproc
-- Found OpenCVLite: C:/projects/cpp/external/pkgs/opencv-lite/include (found version "4.13.0") found components: imgcodecs
-- Configuring done
-- Generating done
-- Build files have been written to: C:/projects/cpp/KFCore/.worktrees/tensorrt-yolo-bytetrack/build/Msvc-Release
exit=0
~~~

vcpkg 同次输出列出的所有 manifest package 均为 already installed；唯一 CMake warning 是 preset
已有的 ENABLE_CLANG_TIDY_DEFAULT、ENABLE_CPPCHECK_DEFAULT、PKG_CONFIG_EXECUTABLE 未被项目使用。

### 真实 SDK compile/link

命令：

~~~powershell
cmake --build --preset win-yolo-release-user --target tensorrt_yolo kfcore_yolo_opencv track_image_sequence test_yolo_opencv test_tensorrt_contract test_tensorrt_detection_helpers test_yolo_tracking test_cuda_buffer test_engine_file test_letterbox_cuda test_tensorrt_integration
~~~

首次包含本轮 production 改动的输出：

~~~text
[1/8] Building CXX object tensorrt_yolo\CMakeFiles\tensorrt_yolo.dir\src\detector.cpp.obj
[2/8] Building CXX object tensorrt_yolo\CMakeFiles\tensorrt_yolo.dir\src\engine.cpp.obj
[3/8] Building CXX object tensorrt_yolo\CMakeFiles\track_image_sequence.dir\examples\track_image_sequence.cpp.obj
[4/8] Linking CXX shared library bin\tensorrt_yolo.dll
[5/7] Linking CXX executable bin\track_image_sequence.exe
[6/7] Building CXX object tensorrt_yolo\CMakeFiles\test_tensorrt_integration.dir\tests\test_tensorrt_integration.cpp.obj
[7/7] Linking CXX executable bin\test_tensorrt_integration.exe
exit=0
~~~

最终 fresh configure 后复验：

~~~text
ninja: no work to do.
exit=0
~~~

### Debug focused suite

命令：

~~~powershell
build/Msvc/bin/test_tensorrt_contract.exe
build/Msvc/bin/test_tensorrt_detection_helpers.exe
build/Msvc/bin/test_yolo_tracking.exe
~~~

输出：

~~~text
TensorRT YOLO engine contract
Total tests [PASSED]: 17
Total tests [FAILED]: 0
Assertions: 202 passed, 0 failed
All tests passed.

TensorRT YOLO detection helpers
Total tests [PASSED]: 9
Total tests [FAILED]: 0
Assertions: 84 passed, 0 failed
All tests passed.

YOLO ByteTrack session
Total tests [PASSED]: 12
Total tests [FAILED]: 0
Assertions: 65 passed, 0 failed
All tests passed.
exit=0
~~~

### Release real-SDK non-engine runtime suite

命令按最小相关顺序：

~~~powershell
build/Msvc-Release/bin/test_tensorrt_contract.exe
build/Msvc-Release/bin/test_tensorrt_detection_helpers.exe
build/Msvc-Release/bin/test_yolo_tracking.exe
build/Msvc-Release/bin/test_yolo_opencv.exe
build/Msvc-Release/bin/test_cuda_buffer.exe
build/Msvc-Release/bin/test_engine_file.exe
build/Msvc-Release/bin/test_letterbox_cuda.exe
~~~

原样汇总输出：

~~~text
TensorRT YOLO engine contract: 17 passed, 0 failed, 202 assertions.
TensorRT YOLO detection helpers: 9 passed, 0 failed, 84 assertions.
YOLO ByteTrack session: 12 passed, 0 failed, 65 assertions.
YOLO OpenCV adapter: 14 passed, 0 failed, 1596 assertions.
TensorRT YOLO CUDA buffers: 4 passed, 0 failed, 24 assertions.
TensorRT serialized engine file reads: 2 passed, 0 failed, 6 assertions.
CUDA letterbox: 3 passed, 0 failed, 92 assertions.
exit=0
~~~

### CMake finder/config fixture

命令：

~~~powershell
cmake -DKFCORE_SOURCE_DIR=<worktree> -DKFCORE_TEST_BINARY_DIR=<worktree>/build/final-fixtures -P cmake/tests/test_find_tensorrt.cmake
cmake -DKFCORE_SOURCE_DIR=<worktree> -DKFCORE_TEST_BINARY_DIR=<worktree>/build/final-fixtures -P cmake/tests/test_find_opencv_lite.cmake
cmake -DKFCORE_SOURCE_DIR=<worktree> -DKFCORE_TEST_BINARY_DIR=<worktree>/build/final-fixtures -P cmake/tests/test_tensorrt_integration_engine_config.cmake
~~~

输出：

~~~text
test_find_tensorrt.cmake: no stdout/stderr, exit=0
test_find_opencv_lite.cmake: no stdout/stderr, exit=0
test_tensorrt_integration_engine_config.cmake: no stdout/stderr, exit=0
~~~

显式 CTest environment 的 verbose 输出：

~~~text
Test project .../build/final-fixtures/tensorrt_integration_engine_config/valid
    Start 1: fixture_tensorrt_integration
1: Test command: cmake.exe
   "-DEXPECTED_ENGINE=.../tensorrt_integration_engine_config/trusted.engine"
   "-P" ".../TensorRtIntegrationEngineFixture/check_engine_environment.cmake"
1: Environment variables:
1:  KFCORE_TENSORRT_TEST_ENGINE=.../tensorrt_integration_engine_config/trusted.engine
1/1 Test #1: fixture_tensorrt_integration ..... Passed
100% tests passed, 0 tests failed out of 1
exit=0
~~~

### 四种安装消费

四次均以 BUILD_CONFIG=Release、TensorRT/OpenCV feature=ON 执行
cmake/tests/test_yolo_tracking_installed_consumer.cmake，且使用互不共享的 build/install prefix。

~~~text
KFCore-first:       no stdout/stderr, exit=0
trackers-first:     no stdout/stderr, exit=0
dependency-first:  no stdout/stderr, exit=0
repeated-KFCore:    no stdout/stderr, exit=0
~~~

脚本的 exit=0 已覆盖 install、consumer configure、consumer build、只使用安装 bin 的运行时启动，
以及 exact KFCore_DIR/trackers_DIR 检查。

### dumpbin /dependents

~~~text
Dump of file build\Msvc-Release\bin\kfcore_yolo_opencv.dll
  Image has the following dependencies:
    kfcore_yolo_tracking.dll
    opencv_core4130.dll
    opencv_imgproc4130.dll
    [MSVC/UCRT system runtimes]

Dump of file build\Msvc-Release\bin\test_yolo_opencv.exe
  Image has the following dependencies:
    kfcore_yolo_opencv.dll
    opencv_core4130.dll
    [MSVC/UCRT system runtimes]

Dump of file build\Msvc-Release\bin\track_image_sequence.exe
  Image has the following dependencies:
    tensorrt_yolo.dll
    kfcore_yolo_opencv.dll
    opencv_imgcodecs4130.dll
    kfcore_yolo_tracking.dll
    opencv_core4130.dll
    [MSVC/UCRT system runtimes]
exit=0
~~~

### 禁止 provenance/API 检索

命令：

~~~powershell
rg.exe -n "TensorRT-YOLO[/\\]|TensorRT-YOLO11[/\\]|opencv_dnn|onnxruntime|enqueueV2|getBinding" CMakeLists.txt CMakeOptions.cmake cmake tensorrt_yolo
rg.exe -n "OpenCVLite::(dnn|highgui|videoio)|opencv_(dnn|highgui|videoio)" CMakeLists.txt CMakeOptions.cmake cmake tensorrt_yolo
rg.exe -n "copy.*(nvinfer|cudart|opencv).*\.dll|install\(FILES.*\.dll" CMakeLists.txt CMakeOptions.cmake cmake tensorrt_yolo
~~~

输出：

~~~text
[三条命令均无匹配输出]
exit=1（rg 的“无匹配”预期状态）
~~~

### GPU/driver 诊断

~~~text
NVIDIA-SMI 571.96
Driver Version: 571.96
CUDA Version: 12.8
GPU 0: NVIDIA GeForce RTX 4060 Laptop GPU
exit=0
~~~

验证过程中曾由系统 PATH 选到 CUDA 13，letterbox 的第一次 cudaMalloc 因驱动只支持 CUDA 12.8
而失败。按系统化诊断固定 CUDA_PATH/PATH 到 v12.8 后，fresh configure 显示 nvcc 12.8.61，
letterbox 3/3、92 assertions 通过。这是验证环境修正，不是代码 fallback。

## 未验证项与既有阻塞

1. UNVERIFIED：没有可信兼容 .engine，因此没有运行 test_tensorrt_integration.exe，也没有声称
   TensorRT 反序列化或 min/opt/max inference 通过。
2. 事实：尝试 BUILD_TESTING=ON 时，生成阶段在既有
   vendor/uuid/CMakeLists.txt 的 uuid_test 失败，因为它链接不存在的 TurboNet::TinyTest。
   该问题已在 ledger 标为任务外 baseline；按约束未修改 vendor/uuid。相关目标与测试改用
   BUILD_TESTING=OFF 配置后直接执行。
3. 事实：构建仅见外部 warning（NVCC 对低于 sm_75 offline architecture 的弃用提示，以及外部
   TinyTest header 的 MSVC C4819 code-page warning）；本轮代码没有新增编译 warning。

## 自审

- 状态归属：Engine::State 的 ValidatedContract 是唯一 engine profile/容量事实源；
  TensorRtDetector::Impl 只持有一次校验后选定的 H/W，推理不会另行推导或回退。
- 失败原子性：negative class 在任何 tracker/scratch/group 状态变化前拒绝；新增测试同时比较
  control session 并验证首个 ID 未消耗。
- 依赖边界：finder 只接受 exact-root、真实存在且格式可验证的 imported target；重复调用复用
  已验证 target，不访问系统/default fallback。
- OpenCV 边界：adapter/package 为 core+imgproc；example 单独增量请求 imgcodecs；
  没有 dnn/highgui/videoio。
- 安全/容量：动态输入实际 byte 数必须不超过由 profile max shape 验证的容量和用户 limit；
  所有维度为正并使用 checked multiplication。
- API：保持既有 public type/function；只落实已文档化的 invalid negative class 和动态 H/W。
- 未触碰 vendor/uuid baseline，也未提交 .codegraph 或 build 产物。

## 最终清洁度

最终提交前执行：

~~~text
codegraph sync .                         exit=0
git diff --check                         exit=0（仅 Git LF→CRLF 提示，无 whitespace error）
~~~
