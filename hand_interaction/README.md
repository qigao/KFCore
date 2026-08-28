# Hand Interaction 与 THIG

`KFCore::hand_interaction` 把 `vision_models::HandFrame` 中的 Palm box、21 点、模型手势和
ByteTrack ID 转成有界的 primitive observations，再交给 `KFCore::thig` 解释时序动作。
它不执行模型推理，因此 ONNX Runtime CPU 与 TensorRT CUDA 共用完全相同的身份、几何和
手势图实现。

```text
CPU Hand backend ---------+
                          +-> HandPipeline -> HandInteractionPipeline
TensorRT Hand backend ----+                         |
                                                    +-> primitives + actions
```

## 支持的语义动作

- `Wave`：张开手掌的 Left/Right/Left 或 Right/Left/Right 三段反向运动。
- `Grasp`、`Release`、`Drag Start *`、`Drag *`、`Drag End`、`Drag Cancelled`。
- `OK`、`Single Hand V`、`Two Hand V`。
- `Zoom In`、`Zoom Out`、`Rotate Clockwise`、`Rotate CounterClockwise`。
- `Click Center/Left/Right/Top/Bottom`：需要应用提供 `Region *` 外部观察。

动作只表达语义，不直接操作 Camera、Gallery、UI 或窗口。应用负责把动作映射为命令；人脸
相对位置、屏幕布局和控件 hit-test 也不进入本模块。

## 使用

```cpp
#include <kfcore/hand_interaction/hand_interaction.hpp>

kfcore::hand_interaction::HandInteractionOptions options;
options.primitives.max_hands = 8;
options.temporal.max_observations_per_frame = 128;
options.temporal.max_relation_events = 512;
kfcore::hand_interaction::HandInteractionPipeline gestures(options);

std::uint64_t serial = 1;
const auto result = gestures.process(
    tracked_hand_frame,
    {serial++, std::chrono::steady_clock::now(), image_width, image_height});
for (const auto& action : result.actions) {
    dispatch_application_command(action.action);
}
```

`process()` 同步借用输入 `HandFrame`，返回值拥有 observations、evidence 和 actions；不会保留
landmark 指针。frame serial 必须严格递增，timestamp 必须单调不减，图像尺寸必须为正。
违反配置、顺序或容量约束会抛出明确异常，不会自动切换后端。

外部观察目前只接受图中已经声明的 `Region Center/Left/Right/Top/Bottom/Unclassified`，每帧
每个 canonical hand 最多一个同名观察，不允许 target。`source.id` 使用前一帧
`HandInteractionFrame::primitives.hands[].canonical_id`；首帧尚无 canonical ID 时应省略区域
观察。输入 observation 会复制进结果和 THIG，不得使用 raw ByteTrack ID 代替 canonical ID。

## 状态、并发与容量

每个 `HandInteractionPipeline` 是单 owner、不可重入对象。不同实例可由不同任务并行运行，
但同一实例的 `process()`、`reset()` 和状态查询必须由调用方串行化。`reset()` 同时清空 canonical
身份、运动/尺度/旋转/双手距离历史、THIG relation、cooldown 和状态图绑定。

所有保留数据同时受时间窗与数量上限约束：`max_hands`、每手样本数、pair histories、
`history_ms`、`max_observations_per_frame`、`max_relation_events` 和
`max_observation_window_states`。默认 8 手的 primitive 上界为
`8 * 7 + 8 * 7 / 2 = 84` 个 observations/帧，低于默认 THIG 上限 128；缩小任一容量时应按
部署的最大手数重新计算。

## 构建和安装

源码树构建需要同时启用：

```cmake
-DKFCORE_BUILD_VISION_MODELS=ON
-DKFCORE_BUILD_THIG=ON
-DKFCORE_BUILD_HAND_INTERACTION=ON
```

Windows 用户 preset 已在 CPU/TensorRT vision profile 中启用这三个选项。安装后：

```cmake
find_package(KFCore CONFIG REQUIRED)
target_link_libraries(app PRIVATE KFCore::hand_interaction)
```

包配置提供 `KFCore_HAS_THIG` 和 `KFCore_HAS_HAND_INTERACTION`，便于可选功能在 configure 阶段
fail fast。模型与 TensorRT engine 仍由对应 backend 管理，既不复制也不安装到本模块。
