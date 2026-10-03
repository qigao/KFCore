# RTX 4060 关系模型 GPU 验证（2026-10-03）

## 范围与环境

本次在 Windows 11、NVIDIA GeForce RTX 4060 Laptop GPU（8 GiB、SM 8.9）、驱动 571.96、CUDA 12.8、TensorRT 11.2.1.2 上执行。动态 V 对照使用 PyTorch 2.11.0+cu128 和 ONNX Runtime 1.26.0 CPU EP；C++ 生产路线使用本机 ONNX Runtime GPU SDK 1.21.0。GitHub 自托管 Linux GPU workflow 尚未运行，因此本报告只证明上述本机环境。

固定发布源为 [RelateAnything ONNX](https://huggingface.co/maelic/relsgg-vits16plus/tree/2db90096be5217bdc7a9003c042950f45723d105)，SHA-256 `b8b6a047c5e0771a897a5015c2ffb09d8fe5e3ffa0651e1e61af0e8436a3617a`；predicate bank 的 SHA-256 为 `708f812d6579eab1be85b3378b79e376453c2d4f65b4fad9bcebb9f5bf05da06`。YOLOX-Tiny 来源为 [0.1.1rc0 发布资产](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx)，SHA-256 `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7`。

## #228：单一 TensorRT engine 的动态 V 执行

[完整机器报告](2026-10-03-rtx4060-dynamic-v.json)记录 `hardware_executed=true`、`passed=true`、同一 engine/context 各加载一次，以及运行前后相同的 engine SHA-256 `98251889a5c529b1b05e0130ac2f7a473ffdf0415fd12cf08ed4be0d752c2601`。构建配置为 FP32、TF32 关闭、2 GiB workspace、V 范围 1/64/243。四个输入 V=1、3、64、243 的有向 pair key 集合均与 ORT CPU 一致，pair/predicate logits 在 `1e-4` 绝对及相对容差内。每个 V 预热 2 次、计时 5 次；TensorRT 中位延迟分别为 17.89、18.60、25.24、51.35 ms。第 64 和 243 项的 5 次样本波动较大，应以报告中的 min/max 同看。

原测试夹具使用完全规则的框网格。实际运行发现 V=243 时第一段 TopK 的第 400 名附近只有约 `2.4e-7` 的分数差，ORT CPU 与 TensorRT 选入不同候选，最终 pair 集合不同。夹具现使用固定种子的轻微不对称框；严格 pair key 校验和原有 `1e-4` 数值容差未放宽。这次通过结果不代表任意输入的 TopK 都稳定。

## #247：完整 C++ SceneBehavior 与内存采样

使用 `win-relsgg-gpu-user`（`CMakeUserPresets.json`）构建，分别执行 ORT CUDA 关系和 TensorRT 关系；两条路线的探测器均为 ORT CUDA。发布关系模型的 C++ 资格程序也分别通过，报告见 [ORT CUDA](2026-10-03-rtx4060-relation-ort-cuda.json) 与 [TensorRT](2026-10-03-rtx4060-relation-tensorrt.json)。完整流水线为 YOLOX-Tiny → ByteTrack → relation → SceneGraph → SceneBehavior。

ORT CUDA 必须允许模型中的部分节点由 CPU EP 执行，否则 C++ session 初始化失败。另用 Python ORT 1.26.0 对同一 ONNX 做 V=3 节点 profiling，[摘要](2026-10-03-rtx4060-ort-cuda-node-assignment.json)显示 2103 次 CUDA 节点执行、798 次 CPU 节点执行，173 次 MatMul 均在 CUDA；这份节点分配证据不等同于 C++ SDK 1.21.0 的逐节点 profiling。

[300 帧测量报告](2026-10-03-rtx4060-scene-memory-latency.json)使用同一张 [YOLOX demo 图](https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/e1052df71842031413f6030723c3607b839c80ce/assets/demo.png)重复 300 次；程序将每一行视为独立静态图，在测量前做一次跟踪预热。两条路线各运行一个进程，每次产生 300 条有效样本。P95 取排序后第 285 个值。

原始延迟样本分别保存在 [ORT CUDA JSONL](2026-10-03-rtx4060-scene-ort-cuda.jsonl) 和 [TensorRT JSONL](2026-10-03-rtx4060-scene-tensorrt.jsonl)，可直接重算中位数与 P95。

| 关系后端 | 总延迟中位 / P95 | 探测器中位 | 关系中位 | 进程 RSS 峰值 | 稳定阶段 RSS 中位（前→后） | 设备显存峰值相对启动前增量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ORT CUDA | 42.24 / 46.44 ms | 7.01 ms | 35.16 ms | 908.62 MiB | 908.04→908.62 MiB | 792.62 MiB |
| TensorRT | 32.62 / 64.66 ms | 8.47 ms | 23.98 ms | 991.54 MiB | 991.54→989.24 MiB | 546.74 MiB |

内存每 50 ms 采样一次。RSS 为目标进程的 Windows working set；显存为 NVML 设备总用量，启动前基线约 2009 MiB，包含桌面和其他进程，不能解释为 KFCore 独占显存。TensorRT 的设备显存稳定阶段中位数后半段比前半段高约 27 MiB；目前无法区分本进程分配、缓存和其他进程波动。本次 300 帧窗口没有观察到进程 RSS 持续增长，但不足以证明长期无泄漏。重复静态图也不代表真实视频负载。

## 复验入口

在 VS x64 开发命令行中，使用本仓库的 `CMakeUserPresets.json`：

```text
cmake --preset win-relsgg-gpu-user
cmake --build --preset win-relsgg-gpu-user --target kfcore_backend_onnxruntime kfcore_backend_tensorrt kfcore_released_relsgg_qualification kfcore_released_scene_behavior_latency
```

模型准备与 TensorRT 构建、资格命令分别见 `tools/relation_training/prepare_released_tensorrt_source.py`、`build_dynamic_vocab_tensorrt.py`、`tensorrt_dynamic_vocab_qualification.py` 及 `.github/workflows/tensorrt-dynamic-v-hardware.yml`。C++ 路线、demo 帧和词表的生成命令见 `.github/workflows/released-relsgg-gpu-production.yml`。本机复验使用 `--opt-vocab 64 --max-vocab 243 --workspace-gib 2`；工作流默认 workspace 为 4 GiB，可按 GPU 容量设置。两份 JSON 报告保留原始计时、内存统计、引擎及源文件哈希。
