# Gesture Interaction

`KFCore::gesture_interaction` / `kfcore::gesture_interaction` 将 MediaPipe 基础观察和衍生动作转换为类型化事件。
静态手势不需要训练；抓取/放开与 Wave 使用分别训练的 ESN 分支。库不操作鼠标、UI 或物体。

## 输入与事件契约

`process(GestureFrame, FrameContext)` 借用一帧 MediaPipe 数据及源图尺寸、采集时间，返回拥有自身数据的事件列表。
完整分数由调用方保留为只读基础观察；关键点采用现有源图坐标，运动分支归一化手腕 x/y。
`None` 表示有手但未识别，不等于没有手，也不会产生基础手势开始事件。

| 事件 | 含义 |
|---|---|
| `GestureStarted` | 一个基础手势持续满足分数和领先幅度条件 |
| `GestureEnded` | 基础手势被另一确认手势替换，或持续未识别 |
| `GestureCancelled` | 丢手、输入中断、来源改变、暂停/重置、模型替换等中断基础手势 |
| `Grasp` | 抓放 ESN 确认整段序列，末端确认握拳并持续保持 |
| `Release` | 抓放 ESN 确认整段序列，末端由确认握拳转为张掌 |
| `Wave` | ESN 确认挥手，门控负责去重 |
| `GraspCancelled` | 活动抓取被中断，不伪装成自然放开 |

基础手势包括 ClosedFist、OpenPalm、PointingUp、ThumbDown、ThumbUp、Victory、ILoveYou。
抓放模型按“非动作/抓取/放开”三类整段训练：抓取为任意手型→握拳→持续握拳，放开为任意手型→握拳→张掌。起始手型可为握拳；因此持续握拳属于抓取训练样本。静止张掌或其他没有完整动作的片段可作为非动作样本。
`GestureStarted(ClosedFist)` 只确认基础姿态；抓取事件还需模型确认及默认 300ms 的握拳保持。
确认握拳后默认 1 秒内确认张掌，将放开候选保留到整窗模型确认；即使此前没有 `Grasp` 事件也可发 `Release`。
没有抓放模型不产生 Grasp/Release。单独张掌不产生 `Release`；短暂丢手清空动作候选。活动抓取失去末端证据且未及时确认张掌时发 `GraspCancelled`。
基础事件与复杂事件可同时产生。保持同一基础姿态不连续发送 Started；切换时先 Ended，再 Started。
`Event.reason` 描述正常确认、未识别、丢手、帧间隔、多手、来源变化、重置、模型替换或历史容量耗尽。

默认基础阈值：分数 >= 0.7、领先第二名 >= 0.15、持续 150ms 确认、持续 200ms 不支持后结束。
两条 ESN 使用各自存档内的门控参数，线性分数不是概率。`InteractionOptions` 可调整基础阈值、丢手边界、握拳保持时间、放开转换时间和左右手变化确认。
基础参数目前由调用方配置，**不写入 V1 `.kfesn`**；该文件仍保存原有训练参数、分支门控、样本及权重。

## 身份、时间和错误

- 当前只支持一只可见手；多手立即取消，不从多手中猜选目标。旧双手组合、拖拽、缩放、旋转、Swipe 不在新模块范围内。
- `source_id` 是调用方可选提供的稳定、非零 64 位身份，不窄化为 int，也不接收旧 Region 观察。
  有无 ID 或值发生改变均重置连续性；左右手不是身份 ID。
- 没有稳定 ID 时，使用可见性和左右手高置信变化划分连续段。默认右手概率 >= 0.85 或 <= 0.15 才作左右判断，
  相反判断持续 150ms 才切换；短暂翻转不会立即取消，但会中断待确认候选和时序窗口。
  `continuity_id` 只表示本对象内的连续段，不证明物理身份；同侧手无缝替换仍不能可靠识别。
- 默认丢手/最大帧间隔 250ms；配置 ESN 后取运行参数与两个模型间隔上限的最小值。
  短暂缺失保留已有生命周期、停止事件确认；超过上限取消并清空窗口。基础 None 不触发丢手取消。
- `process` 的采集时间须严格递增；`advance`、`reset`、`configure_sequences` 允许与上一操作相同时间，但不能倒退。
  所有时间使用同一单调时钟（秒），不能混入 Unix 时间。调用 `advance` 后不得提交更早的缓存帧。
- 无新帧时必须周期调用 `advance(now)`，否则库没有后台线程能自行发出取消。暂停、关闭、切换输入前必须消费 `reset(now)` 返回值。
  析构不会执行回调或自动投递事件；忽略返回值会丢失取消通知。
- 非法时间、分数、图像尺寸、缺少左右手概率及非法模型配置抛出明确异常，不改变已提交状态。
  调用方决定是否修正同一帧重试，或将流错误视为终止并调用 reset。历史容量耗尽会明确取消，不使用半个窗口继续判定。

## 使用示例

下例不加载模型文件，演示已得到的 MediaPipe 观察如何产生基础手势事件；构建链接 `KFCore::gesture_interaction`。

```cpp
#include <kfcore/gesture_interaction/interaction.hpp>
#include <iostream>

int main() {
    namespace gi = kfcore::gesture_interaction;
    kfcore::mediapipe::GestureFrame frame;
    frame.landmarks.hands.resize(1);
    frame.landmarks.hands[0].right_hand_probability = 0.95F;
    frame.gestures.resize(1);
    frame.gestures[0].label = kfcore::mediapipe::CannedGesture::Victory;
    frame.gestures[0].scores[std::size_t(frame.gestures[0].label)] = 1;
    gi::GestureInteraction interaction;
    for (double seconds : {0.0, 0.1, 0.2}) {
        for (const auto& event : interaction.process(frame, {seconds, 640, 480}))
            std::cout << gi::event_name(event.kind) << '\n';
    }
    for (const auto& event : interaction.reset(0.2))
        std::cout << gi::event_name(event.kind) << '\n';
}
```

输出为 `GestureStarted`、`GestureCancelled`。真正推理时以 `GestureRecognizer::infer()` 的输出替换示例帧。

分别用 `CompositionEsn` 训练 `CompositionTask::Interaction` 与 `CompositionTask::Motion`，或通过 `experiment.hpp` 中的 `load_experiment(path, expected_pipeline)` 读取 V1 文件（Windows 文件适配器），再用 `restore_model(archive.sessions[0/1])` 恢复两条分支，调用
`configure_sequences({&grasp_release_model, &wave_model, grasp_gate, wave_gate}, now)`。未训练分支可传空指针；其动作事件不会产生。
模型在调用期间复制为库拥有的不可变快照，后续修改训练对象不影响运行模型。成功替换先返回原生命周期的取消事件，
再从新历史开始；失败保留原模型及状态。该操作不重新训练。

`encode_experiment` / `decode_experiment` 是跨平台内存编码接口；`save_experiment` / `load_experiment` 仅在 Windows 提供文件适配，
其余平台由调用方提供文件字节。必须以当前管线指纹作为 expected_pipeline，不能直接信任文件里的自报指纹。
当前 demo 指纹算法按后端 DLL、palm/landmark/embedder/classifier 清单及各包 artifact 的顺序拼接 SHA-256；`--actions` 另附动作标签规则标识，以拒绝旧语义实验；
它不包含新库 DLL、UI 参数或源图尺寸。文件上限、损坏校验、不覆盖语义见 [demo 存档说明](../tools/hand_preview/README.md)。

## 架构决策与迁移

背景：旧 hand_interaction 将身份、基础原语和 THIG 图耦合，公开接口泄漏 THIG 类型；demo 则另外维护 ESN 事件。
候选方案是保留图并添加 ESN 适配器，或共享特征/分类核心并以统一事件生命周期替换旧链路。
采用后者：MediaPipe → 基础确认 / 抓放 ESN / Wave ESN → 事件；ESN 编码、门控及存档适配从 demo 下沉，录制交互和文件对话框留在工具层。
采用组合与私有实现，不引入事件总线、后台线程、订阅者生命周期或新的第三方库。业务同步消费返回值，外部区域映射属于业务。

状态主事实源是 `GestureInteraction` 内的活动基础手势、抓取生命周期与连续段。两条模型不可变；每帧先在有界候选状态中计算，再提交。
单分支复杂度为 O(历史采样数 + 60 × 48²)，最多两个 ESN 分支；每窗采样受模型容量约束，
每帧事务复制历史的成本需要后续实测，本实现不声称已完成性能基准或低延迟优化。

HIGH：这是旧 SDK 的破坏性迁移：`KFCore::hand_interaction`、`KFCore::thig`、旧公开头及图接口已移除，不提供伪兼容别名。
外部调用方需改为 MediaPipe GestureFrame 和新事件枚举。新模块不承诺旧多手身份重关联能力。
旧基础方向分支及 Region ID 窄化路径随旧模块移除；抓取取消语义由新模块显式覆盖。
回滚须从 Git 历史恢复旧模块、构建入口和调用方一起回滚；不是在运行时遇错退回 THIG。

demo 的 S/L 和 V1 文件编码、标签数值、ESN 初始化/特征维度不变，已有文件不迁移、不覆盖。
旧抓放样本标签与新定义不兼容；demo 使用新的动作规则指纹拒绝旧 `--actions` 实验，文件本身不迁移、不覆盖。
模型配置变化会清空运行历史并取消活动事件；新增任一分支训练样本使该分支已训练模型失效，运行配置重新绑定会重启两条历史。
MED：合成测试和单元测试不能代替真实摄像头验证；尚无跨用户、跨天误触发/漏检率和事件延迟基准。

```powershell
cmake --build --preset win-hand-preview-user --target test_gesture_interaction test_composition_esn test_hand_experiment kfhand_preview
ctest --preset win-hand-preview-user -R '^(test_gesture_interaction|test_composition_esn|test_hand_experiment|hand_preview_.*)$' --output-on-failure
```
