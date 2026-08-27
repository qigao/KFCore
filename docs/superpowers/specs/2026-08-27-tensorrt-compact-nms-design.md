# TensorRT YOLO Compact NMS 输出设计

日期：2026-08-27
状态：已实现并验证

## 1. 背景与目标

本地 `yolo-models` 中的 YOLOv8/YOLOv12 ONNX 均已在 TensorRT 11.2 上验证为一个
`images` 输入和一个 `output0` 输出。输出 shape 为 `[batch, max_detections, 6]`，六列依次为
`left`、`top`、`right`、`bottom`、`score`、`class_id`，模型图内已执行 NMS，并以全零行补齐固定容量。

本设计让既有 `TensorRtDetector` 同时接受原 EfficientNMS 五输出契约和该 Compact NMS
单输出契约。公开的 `Engine`、`TensorRtDetector`、`DetectionFrame` 与调用流程保持不变。

## 2. 范围与非目标

范围：

- 严格识别 `images + output0` 两张量 engine；
- 校验输入 NCHW、输出 `[B,N,6]`、dtype、物理布局、profile 和资源上限；
- 下载紧凑输出并恢复到原图坐标；
- FP32 与 FP16 紧凑输出；
- 保持 EfficientNMS 五输出路径的既有行为。

非目标：

- raw YOLO head 解码；
- CPU/GPU NMS；
- 分类、ArcFace embedding、关键点或分割输出；
- 自动猜测未命名或列语义不同的输出；
- 为无 `num_dets` 输出引入置信度阈值。NMS 阈值仍由模型导出时确定。

## 3. 契约与错误语义

`TensorNames` 增加默认值为 `output0` 的 `detections` 名称。契约必须精确匹配以下一种形式：

1. EfficientNMS：`images`、`num_dets`、`boxes`、`scores`、`labels`；
2. Compact NMS：`images`、`detections`。

Compact NMS 输出约束：

- rank 为 3；
- shape 为 `[batch, max_detections, 6]`；
- batch profile 与 `images` 一致；
- `max_detections` 固定、为正且不超过 `EngineOptions::max_detections`；
- FP16 或 FP32、`kLINEAR`、非向量化标量布局；
- 最大输出字节数不超过 `EngineOptions::max_output_bytes`。

任何缺失、额外、重名、错误 mode/dtype/rank/shape/layout 或资源越界均抛出既有的
`EngineContractMismatch` 或 `ResourceLimitExceeded`，不尝试其他解码方式。

## 4. 运行时与状态所有权

`ValidatedContract` 保存公共输入事实和一个有限输出契约：EfficientNMS 或 Compact NMS。
该类型是 engine 的不可变事实源。每个 detector 仍独占 context、stream 与缓冲；Compact NMS
只增加 detector 私有的一对 device/host 输出缓冲，不引入共享可变状态。

每个 Compact NMS 行按以下规则处理：

- 六个标量必须有限；
- `score` 必须位于 `[0,1]`；
- `score == 0` 是 padding，跳过该行；
- 正分数行的 `class_id` 必须是可精确表示的非负 `int32` 整数；
- box 不得反向，经 letterbox 逆变换与图像边界裁剪后必须保持正面积。

解码复杂度为 `O(batch * max_detections)`，额外结果空间为 `O(valid_detections)`。

## 5. 兼容性、迁移与回滚

- 现有 EfficientNMS engine、默认张量名和公开检测调用不变；
- `TensorNames` 末尾新增字段，已有按成员赋值的调用不受影响；少数依赖精确 struct 二进制布局的
  C++ 调用方需要重新编译；
- 以前被拒绝的精确 Compact NMS engine 现在可加载，这是有意的公开行为扩展；
- 关闭 `KFCORE_BUILD_TENSORRT_YOLO` 或回退本次提交即可恢复旧行为，不涉及数据迁移。

## 6. 验证范围

- 纯契约 TinyTest：两种有效契约及 Compact NMS 的名称、rank、列数、batch、dtype、物理布局和上限；
- 纯解码 TinyTest：FP32/FP16、padding、坐标恢复、类别整数性、非法分数/坐标/尺寸；
- 既有 TensorRT/YOLO/tracking/ImageProcessor CTest 全量回归；
- TensorRT 11.2 + RTX 4060 使用本地 `yolov12n-face.engine` 完成真实图片序列执行。

**事实（2026-08-27）**：契约 TinyTest 为 26/26、275 assertions，检测 helper TinyTest 为
16/16、130 assertions，`win-yolo-release-user` CTest 为 19/19；本地 Compact NMS engine 对仓库
3 张输入图片写出 3/3 张结果。该本地 engine 的 batch 固定为 1，因此真实 TensorRT 执行只覆盖
逐张 `detect()`；`B > 1` 的行偏移、独立 letterbox transform、padding 与结果顺序由双图片纯解码
测试覆盖，不宣称真实动态 batch engine 已复验。
