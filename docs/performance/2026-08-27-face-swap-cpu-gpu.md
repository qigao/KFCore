# Face Swap CPU/GPU 性能记录（2026-08-27）

## 范围

本记录区分两类路径：

1. ImageProcessor 的 CPU/OpenCV reference 与 CUDA production 预处理；
2. 完整 face swap production pipeline。完整 pipeline 只有 TensorRT GPU 推理，没有 CPU
   inference backend；其中 embedding projection、mask/paste/blend 和部分 decode 是 CPU 工作。

所有结果都是当前机器的 Release 实测，不是跨设备延迟保证。`swap_profiled()` 记录同步 wall
time：CUDA/TensorRT 调用返回前已同步，因此包含 GPU 完成等待和相邻 CPU adapter 工作，不是
CUDA kernel 独占时间。

production GPU 路线按“每帧、每个 CUDA device”缓存一份 staged image：source 的 YOLO、
Face68、ArcFace 共享一次 upload；target 的 YOLO、Face68、ArcFace、InSwapper 共享另一次
upload。各模型从同一 staged image 生成自己的固定尺寸 tensor，并在同步消费完成后复用
ImageProcessor tensor buffer。GFPGAN 输入是换脸后的新图，因此需要第三次 stage。不同 device
各自 stage；缓存不跨帧、不跨 application 实例，也不支持并发调用共享。

## 测试环境（事实）

| 项目 | 值 |
| --- | --- |
| CPU | AMD Ryzen 9 7940HX，16C/32T |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB |
| NVIDIA driver | 571.96 |
| CUDA / TensorRT | CUDA 12.8 / TensorRT 11.2.1.2 |
| OS | Windows 11 10.0.26200 |
| 输入 | source/target 测试图；预处理 benchmark 使用合成 1920×1080 BGR |
| 构建 | MSVC Release，`build/Msvc-Face` |

## CPU 与 CUDA 预处理（事实）

本机运行时 DLL 路径与命令：

```powershell
$env:Path = 'C:\projects\TensorRT-11.2.1.2\bin;' +
  'C:\projects\cpp\external\pkgs\opencv-lite\bin;' +
  'C:\projects\cpp\external\pkgs\turboutils\release\bin;' +
  'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;' + $env:Path
build\Msvc-Face\bin\benchmark_face_preprocess.exe
```

每项 10 次 warmup、100 个测量样本；CPU 项包含 OpenCV affine 和 CPU HWC→FP32 NCHW，
CUDA 项消费已 staged 的 frame。二者在 benchmark 中先做数值误差校验。

| 操作 | 输出工作量 | CPU avg | CUDA avg | CUDA 相对加速 |
| --- | ---: | ---: | ---: | ---: |
| Face68 256×256 | 65,536 pixel / 0.750 MiB FP32 | 568.583 us | 25.122 us | 22.63× |
| ArcFace 112×112 | 12,544 pixel / 0.144 MiB FP32 | 81.325 us | 12.210 us | 6.66× |
| InSwapper 128×128 | 16,384 pixel / 0.188 MiB FP32 | 103.722 us | 16.229 us | 6.39× |
| GFPGAN 512×512 | 262,144 pixel / 3.000 MiB FP32 | 1791.520 us | 32.652 us | 54.87× |
| 1920×1080 BGR host→device stage | 2,073,600 pixel / 5.933 MiB | 不适用 | 656.697 us | 不适用 |

原始观测范围：CPU GFPGAN 的 min/max 为 1228.5/2929.6 us；CUDA stage 的 min/max 为
594.5/1622.1 us。其余 min/max 可由上述命令重新输出，因此比较稳定值时应优先看多次运行的
P50/P95，而不是单个 min。

## 预处理汇总（计算）

输入取上表平均值：

- 四个 affine/normalize 操作：CPU `568.583 + 81.325 + 103.722 + 1791.520 =
  2545.150 us`；CUDA（frame 已在 device）`25.122 + 12.210 + 16.229 + 32.652 =
  86.213 us`，即 `2545.150 / 86.213 = 29.52×`。
- 加一次 1080p upload 后，CUDA 为 `656.697 + 86.213 = 742.910 us`，相对 CPU 四项为
  `2545.150 / 742.910 = 3.43×`。
- 无 GFPGAN 的实际拓扑是两次 frame stage、两次 Face68、两次 ArcFace、一次 InSwapper。
  CUDA 估算为 `2×656.697 + 2×25.122 + 2×12.210 + 16.229 = 1404.287 us`。
  若 detector 所需的两次 upload 不变、只把三个模型预处理换回 CPU，则下界为
  `2×656.697 + 2×568.583 + 2×81.325 + 103.722 = 2716.932 us`，CUDA 至少快
  `1.94×`；CPU tensor 再上传到 TensorRT 的成本尚未计入，所以这是保守下界。

这里用 pixel 与输出 tensor bytes 表达工作量；不能从这些数值推出 FLOPS。CPU reference 与
CUDA kernel 虽语义一致，但 OpenCV affine、planar conversion 与 CUDA fused kernel 的实际
指令数和内存访问不同。

## 完整 TensorRT pipeline（事实）

命令：

```powershell
build\Msvc-Face\bin\benchmark_face_swap_pipeline.exe
```

目标仅在 `KFCORE_BUILD_FACE_APPLICATION_INTEGRATION_TESTS=ON` 且真实模型路径有效时生成。
每种配置先 warmup 5 次；兼容 `swap()` 路径测 50 次平均/min/max，随后
`swap_profiled()` 再测 50 次 stage P50/P95。engine load 不进入计时。

| 配置 | avg | min | max | P50 | P95 | 吞吐（计算） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 无 GFPGAN | 112.116 ms | 103.957 ms | 121.730 ms | 115.031 ms | 121.758 ms | 8.92 frame/s |
| 有 GFPGAN | 212.234 ms | 197.518 ms | 339.316 ms | 211.822 ms | 241.525 ms | 4.71 frame/s |

三次独立运行的 warmed average 范围为：无 GFPGAN `111.038–128.149 ms`，有 GFPGAN
`212.234–240.703 ms`。这说明笔记本 GPU 的频率/电源状态会显著影响单轮平均值；容量规划宜用
P95 并在目标功耗模式下重新采样。

主要 P50 stage：

| stage | 无 GFPGAN | 有 GFPGAN |
| --- | ---: | ---: |
| source analysis total | 21.482 ms | 21.265 ms |
| target analysis total | 21.628 ms | 21.567 ms |
| InSwapper inference + decode | 35.156 ms | 36.081 ms |
| InSwapper CPU composition | 35.795 ms | 30.334 ms |
| GFPGAN preprocess | 未运行 | 1.496 ms |
| GFPGAN inference + decode | 未运行 | 55.077 ms |
| GFPGAN CPU composition | 未运行 | 42.137 ms |

`GFPGAN` 平均增加 `212.234 - 112.116 = 100.118 ms`，即相对无增强路径增加约
`89.30%` 延迟。无 GFPGAN 时，InSwapper inference/decode 与 CPU composition 合计
`70.951 ms`，占 P50 total 的约 `61.68%`；这两项比进一步压缩几十微秒的预处理更值得优化。

## 冷启动分解（事实）

同一 benchmark 单次记录如下。`load` 包含 TensorRT engine、InSwapper matrix、execution
context、CUDA buffer 与 ImageProcessor 创建；`first swap` 包含首帧可能发生的 lazy 初始化。

| 阶段 | 无 GFPGAN | 有 GFPGAN |
| --- | ---: | ---: |
| source + target `imread`（两配置共享一次） | 31.852 ms | 31.852 ms |
| application/model load | 2053.116 ms | 2680.389 ms |
| first complete swap | 281.311 ms | 251.273 ms |
| load + first swap | 2334.427 ms | 2931.662 ms |
| image decode + load + first swap（计算） | 2366.279 ms | 2963.514 ms |

无 GFPGAN 是新进程的首次配置；GFPGAN 配置在同一进程随后加载，因此 CUDA runtime 已初始化。
另一次独立 UI 冷启动观测约为 2.23 s（无 GFPGAN）和 3.05 s（有 GFPGAN），与此量级一致，
但 UI 观测混合窗口创建，不能替代上表阶段分解。

## CPU/GPU 资源遥测（事实）

第二次完整 benchmark 运行期间，每约 200 ms 读取一次进程计数器与 `nvidia-smi`；该次运行
用于资源测量，不作为上面的延迟事实源。143 个 GPU 样本覆盖 engine load、无 GFPGAN 和
GFPGAN 两段持续负载。

| 指标 | 观测值 |
| --- | ---: |
| wall / process CPU time | 39.572 s / 69.266 CPU-s |
| CPU 占用 | 1.750 个逻辑核等效；32T 整机口径 5.47% |
| process peak working set | 1470.7 MiB |
| process peak private bytes | 3401.7 MiB |
| GPU utilization | baseline 23%；运行平均 52.90%，最大 66% |
| GPU memory | baseline 1693 MiB；运行平均 3672.1 MiB，最大 4142 MiB |
| GPU memory 最大增量 | 2449 MiB（global used delta） |
| GPU power | baseline 5.23 W；运行平均 56.34 W，最大 113.43 W |
| SM clock | 运行平均 2315.1 MHz，最大 2610 MHz |

另一次分配置 UI 采样得到 global memory plateau：无 GFPGAN 约增加 1447 MiB；启用 GFPGAN
约增加 2164 MiB，即 GFPGAN 额外约 717 MiB。Windows WDDM 下这里是整卡 global 指标，可能
混入桌面和其他进程；CPU private bytes 也可能包含驱动映射，不能当作纯 host heap。

## 结论边界

- **事实：** production inference 是 TensorRT GPU 路线；仓库没有完整 CPU inference 路线，
  所以不能声称“CPU 完整换脸需要多少 ms”。
- **事实：** CPU/OpenCV 路线是算法 reference 和数值校验路径；production CUDA preprocess
  复用 detector 已需要的 staged frame，避免重复 CPU affine/planar 和 host tensor upload。
- **推论：** 当前优化优先级应是 InSwapper/GFPGAN inference 与 CPU composition；预处理已
  低至每项约 12–33 us，对完整 128–241 ms pipeline 的占比很小。
- **限制：** `nvidia-smi` utilization 是整卡时间占比，不是已执行 FLOP 数；精确 kernel
  时间、occupancy、memory throughput 和 Tensor Core 利用率需要 Nsight Systems/Compute。
