# MediaPipe 手部预览 app

参考 Retro `tools/camera_feed_test` 的采集与预览职责，独立于 Retro 业务。
当前仅 Windows Release、CPU ONNX；可选官方基础手势分类，没有跟踪。
可用 `--esn` 显式启用组合手势实验，默认不启用。

- 摄像头：已有 `Salts::Capture`，精确选取枚举所得原生模式。
- 显示：本地 `OPENCV_LITE_ROOT` 的 core/imgproc/highgui/imgcodecs。
- 推理：`KFCore::mediapipe` 两模型入口。
- 图片：用于重复验证检测与绘制；**不支持 MP4 文件播放**，本地 opencv-lite
  没有 videoio，已安装 Salts Capture 只提供设备采集接口。

## 构建与启动

在 MSVC 开发终端中，从 KFCore 根目录执行：

```powershell
cmake --preset win-hand-preview-user
cmake --build --preset win-hand-preview-user --target kfhand_preview kfcore_backend_onnxruntime
ctest --preset win-hand-preview-user -R '^hand_preview_' --output-on-failure
./build/hand-preview/Release/bin/run-hand-preview.cmd --list-cameras
./build/hand-preview/Release/bin/run-hand-preview.cmd --list-modes 0
```

生成的 `run-hand-preview.cmd` 从 configure 环境继承运行库路径，不复制 DLL。
仅供本机 build tree 使用；移动 SDK 后应重新 configure。原有库 preset 不启用此 app。
用户指定复用本地 opencv-lite，因此不新增 vcpkg OpenCV 依赖。

以下命令使用之前准备的两模型测试资产，模式索引须以本机枚举结果为准：

```powershell
./build/hand-preview/Release/bin/run-hand-preview.cmd --backend build/hand-preview/Release/bin/kfcore_backend_onnxruntime.dll --palm build/mediapipe-onnx-smoke/palm_detection_full_inf_post_192x192.json --landmark build/mediapipe-onnx-smoke/hand_landmark_sparse_Nx3x224x224.json --camera 0 --mode 120
```

本机此次枚举的模式 120 是 640×480、30 fps、NV12。不能假定其他摄像头
或重连后的索引相同；不支持指定模式时明确失败，不自动更换模式。
模型包契约与制作方式见 [MediaPipe 模块说明](../../vision/mediapipe/README.md)。

| 输入参数 | 用途 |
| --- | --- |
| `--backend/--palm/--landmark` | 必填，插件与两个模型包路径 |
| `--camera INDEX` / `--image PATH` | 必须二选一 |
| `--mode INDEX` | 原生模式索引，默认 0，可先枚举 |
| `--frames N` | 推理 N 帧后退出，0 表示不限制；图片最多 1 帧 |
| `--headless` | 不开窗口，必须给正数 `--frames` |
| `--snapshot NEW.png` | 退出时保存带骨架的预览，不覆盖已有文件；默认不保存 |
| `--timeout SECONDS` | 等待新摄像头帧超时，默认 10 秒 |
| `--max-frame-bytes N` | 采集帧字节限制，默认 64 MiB |

空格暂停/恢复预览，N 消费下一帧，M 镜像显示，O 切换骨架，Esc 退出。
**暂停只冻结预览和推理，摄像头仍采集**；退出程序才释放设备。
镜像只影响绘制，不改变模型输入或左右手标签。
高频画面不增加动画；状态用文本表示，不只依赖颜色。
OpenCV 画布没有屏幕阅读器语义，本 demo 不声称达到完整无障碍支持。
加载/采集错误输出到启动终端并以非零退出码结束，不切换后端或模型。

## 状态与资源协议

采集回调借用原始数据，仅在回调期间有效；转换后的 BGR Mat 拥有数据。
回调为生产者，界面线程为消费者，使用互斥锁保护一个待消费帧。
满时用新帧替换旧帧并计数，退出汇总 `replaced`；这是实时预览，不是逐帧录像。
转换和旧帧释放在锁外执行。回调异常保存为 exception_ptr，由界面消费并终止；
异常不穿过 C 回调边界。采集 stop/destroy 完成后才销毁回调状态。
推理、显示和 UI 操作由同一线程执行，耗时推理期间 UI 响应会延迟。

MED：本例依赖安装版本 Capture 的 stop/destroy 回调排空语义；没有验证设备
热拔插和驱动永久挂起。图片/MJPEG 解码由 OpenCV 执行，只用于可信本地输入；
不能把此 demo 当作不可信图片解码沙箱。预览图像有固定画布尺寸。

## 本机验证

已完成 Release 编译、CLI 帮助与非法参数测试、真实图片两模型推理。
摄像头 0 模式 120 的一次 30 帧运行输出：
`frames=30 frames_with_hands=6 replaced=8`。这是一次功能验证，不是精度或性能基准。
窗口实测看到双手手框、21 点骨架、左右手标签，并验证空格暂停。
没有保存摄像头画面；测试图片来自既有数据集视频抽帧。

## ESN 组合手势实验

先按[官方模型转换说明](../mediapipe_gesture/README.md)生成模型。使用新
`hand_world.json` 替换关键点包，增加 `--gesture-embedder` 和
`--gesture-classifier` 两个模型包参数，再加 `--esn`。两个 gesture 参数必须成对提供。
不传 ESN 可单独验证基础手势；原两模型入口保持可用。

MediaPipe 提供 None、Closed_Fist、Open_Palm、Pointing_Up、Thumb_Down、
Thumb_Up、Victory、ILoveYou 八类分数。None 表示有手但未匹配基础手势，不等于无手。
旧 `--esn` 实验接收八类分数、右手概率、有手标记的历史；下述 `--actions`
模式的运动分支另外读取手腕轨迹。

自然动作可以是 `Open_Palm → None → Closed_Fist → None → Open_Palm`。
其中 None 可能来自弯指、侧转等过渡姿态，但它本身不是“过渡”专用类别。
界面将三种含义分开：`Unrecognized (None)` 表示有手但基础手势未匹配，
`No hand detected` 表示未检测到手，`No-combo` 表示 ESN 的组合负例类别。
这些只是显示名称，官方类别索引、八类分数和有手标记不变。
只要手仍被检测到，就保留完整八类分数与时间戳，不删除 None 帧、不强制改成上一手势，
也不因其持续超过丢手阈值而取消录制。基础 None 不会自动成为组合负例标签：
整段属于哪个组合由录制按键指定，需录入包含自然过渡的完整样本。

1. 点击窗口获得键盘焦点。按 `0` 录负例，`1` 录「张开→握拳→张开」，
   `2` 录「握拳→张开→握拳」。一次按键录一段，默认 3 秒，每类至少 3 段。
   连续挥动不会自动产生多个片段。负例应包含静态手势、不完整组合及无关手势。
2. 按 `T` 拟合 ESN 读出层；基础手势模型与 reservoir 不训练。
   新增训练片段使读出层失效。反馈保留失败原因、重试动作及训练状态。
3. 训练后按不重叠的 3 秒窗口预测。线性分数不是概率，尚无拒识校准。
   这是片段分类，不是动作起止检测，也不触发业务行为。
4. 按 `3/4/5` 采集三类独立测试片段，每类至少 3 段，按 `V` 输出混淆矩阵和正确数。
   类别顺序为负例、张开握拳张开、握拳张开握拳。T/V 不区分大小写。
   测试片段不参加训练；调参后需要另采最终测试集。

`CompositionEsn::encode` 按真实时间戳重采样至 60 步，每窗口重置 48 单元 reservoir，
取早/中/末三份状态形成 144 维历史，通过既有 KFCore ridge 算子拟合三类读出层。
统一存储另留 96 维相邻状态差平方；旧模式填零，新动作模式启用，以区分静态姿态
和方向相反的变化。原 reservoir 初始化及旧模式输入保持不变。
seed=42、谱半径 0.8、leak=0.5、ridge=0.1 是未根据测试集优化的基线参数。
`CompositionOptions` 集中管理配置；CLI 支持 `--esn-window-ms`、`--esn-gap-ms`、
`--esn-min-clips` 调整采集协议。

单片默认最多 256 采样，训练与测试各默认最多 120 片。按 S 手动保存实验；未保存的更改退出即丢失。
没有改动视频或标注格式，也不会自动保存到数据集目录。
短暂丢手编码为无手；超过默认 250ms、多手、时间戳异常、暂停或单步重置窗口，
正在录制时保留失败原因并提示重试。左右手概率抖动本身不取消。
尚无跨帧身份跟踪，不能保证同侧手无缝替换时的连续性。

```powershell
cmake --build --preset win-hand-preview-user --target test_composition_esn test_gesture_recognizer kfcore_esn_tests kfcore_esn_sequence_tests
ctest --preset win-hand-preview-user -R '^(test_composition_esn|test_gesture_.*|kfcore_esn.*tests|hand_preview_.*)$' --output-on-failure
```

真实模型测试的资产环境变量见转换说明。合成用例覆盖训练门槛、历史区别、
短暂错误手势、无手/None 区分、时间戳与容量、失败提示、独立测试和读出层失效。
另覆盖 None 完整分数保留、超过丢手阈值的有手过渡连续录制，以及未用于训练的
过渡时长下的合成组合分类；不把这些合成序列当作真实视频精度证据。
MED：尚无真实标注组合识别准确率、跨被试泛化、误触发率或事件延迟基准；
窗口边界未对齐动作可能误判，合成测试不能替代真实效果验证。

## 抓取/放开与 Wave（`--actions`）

在上述官方四模型启动参数后使用 `--actions`，无需再传 `--esn`。
衍生动作只处理抓取、放开和 Wave，不包含 Swipe、点击、OK 或捏合。基础七类手势同时直接确认，无需训练。
所有权保持在 UI 线程：两套 Session 持有训练/测试样本与分类模型；
`KFCore::gesture_interaction` 是业务事件唯一来源，负责静态手势、ESN 门控和抓取生命周期。
旧交互 SDK 接口已替换，见[新库契约](../../gesture_interaction/README.md)；原模型包和 V1 实验文件格式不变。

| 分支 | 中性负例 | 动作 1 | 动作 2 |
|---|---|---|---|
| Grasp/Release | 静态张开、静态握拳、其他姿态变化 | 张开→握拳，保持结束姿态 | 握拳→张开，保持结束姿态 |
| Wave | 静止、单向划动、竖直移动、轻微晃动、普通姿态变化 | 水平来回至少两轮 | 无（二分类） |

### UI 操作

1. 点击 `G: Grasp/Release` 或 `W: Wave` 选择**训练分支**；两条训练完成的
   分支始终并行推理，切换选择不停止另一条分支。
2. 抓取分支点击 `Train Neutral`、`Train Grasp`、`Train Release`；
   Wave 分支只有 `Train Neutral` 和 `Train Wave`，不显示第三类按钮。
   检测到手后先倒计时 2 秒，保持起始姿态；出现 `REC` 后完成一次动作，默认录制 3 秒。
   结束后保持姿态直到保存提示。重复点击不会重置进行中的录制；`C: Cancel` 可取消。
3. 每类至少录 3 段；中性类必须覆盖多种静态姿态与无关运动，不只是张开手。
   动作应覆盖速度、幅度和片段内发生时刻的变化；最低数量是训练门槛，不是精度保证。
4. 点击 `T: Train` 训练当前分支。新增训练片段只使当前分支失效，另一分支保持可用。
5. 用 `Test ...` 按钮录制独立测试片段，每类至少 3 段，再点 `V: Evaluate`。
   窗口显示正确数；完整混淆矩阵输出到启动终端。测试数据不进入训练。
6. 停止录制后自由做动作。`Window candidate` 是窗口候选类别，不是已触发事件；
   `Events` 显示最近的 Grasp、Release、Wave 或 GraspCancelled；上方另外显示最近的基础手势事件。
   保持动作不重复发送；孤立 Release 不作为业务事件发出。窗口候选仍可能显示 Release。

抓取分支数字键 `0/1/2` 对应三类训练，`3/4/5` 对应测试。
Wave 分支 `0/1` 对应 Neutral/Wave 训练，`3/4` 对应测试；`2/5` 无效且提示正确按键。
Wave 只需两类各满足最少样本数即可训练和评估，输出 2×2 混淆矩阵。
G/W、T/V、C 保留键盘操作。S 保存全部分支，L 加载全部分支。
OpenCV 自绘按钮没有屏幕阅读器语义，不宣称完整无障碍。

### 算法与边界

- 运动分支输入为原 10 维观察，加相对片段起点的手腕 x/y 位移及 x/y 速度。
  坐标先除源图尺寸，速度使用真实时间差，之后以可配置比例经 tanh 约束；镜像仅影响预览。
  无手期间不插值出运动速度。没有轨迹不能录制/推理运动任务，明确报错。
- 在线使用默认 3 秒滑动窗口，约 100ms 更新一次候选，每窗重算有界 60 步 ESN。
  复杂度为 O(采样数 + 步数×储备池单元数平方)，每分支最多 256 个历史样本。
  这仍是窗口式识别，不是精确动作起止分割；实际延迟必须实测。
- 每分支独立门控：默认线性最大分数至少 0.5、领先第二名至少 0.15，并持续 150ms 才触发。
  分数不是概率。相同动作需先获得持续 300ms 的中性结果才能再次触发；不同动作可切换。
  不确定结果不能当作中性来重新解锁。阈值位于 `ActionGateOptions`，尚未经真人校准。
- `CompositionOptions` 管理窗口、更新步长、倒计时、运动缩放及样本容量。
  录制和测试期间当前分支不触发实时事件。模型重绑定取消活动事件并重启两条运行历史，
  未修改的分支无需重训。事件层容忍短暂左右手概率翻转；采样 Session 的左右手标签切换仍会取消录制。
  丢手超时、超长帧间隔、多手、暂停、取消、文件对话框和退出会清理运行历史，并取消活动抓取；
  训练样本不被删除。None 仍保留为有手的原始分数。
- 当前仅支持一只可见手，没有稳定身份跟踪；同侧手无缝替换无法可靠发现。
  抓取表示姿态变化，不证明抓住真实物体。事件只显示在 demo，不触发鼠标或外部业务。

设计取舍：复用既有片段编码与 ridge 算子，独立训练三分类抓取分支和二分类 Wave 分支，避免将抓取和运动
强制放进一个互斥类别。代价是两份有界历史与读出层。旧入口可直接回退验证；旧样本不
自动迁移为新标签。实验持久化使用下述独立格式，不修改原有视频与标注文件。

验证覆盖：旧组合回归、静态握拳负例、两种方向的合成 Wave 和单向划动负例、归一化、平移不变性、
非法轨迹、门控确认与去重、录制倒计时、分支样本隔离及连续握拳不重复抓取。
MED：这些是功能与合成回归，不代表真实摄像头准确率；仍需真人独立测试误触发、漏检与延迟。

移除 Swipe 后旧进程不会热更新，也不会自动迁移其内存样本。关闭旧 demo 前应明确确认
样本丢失可接受；新版 Wave 分支需要重新录制 Neutral/Wave。移除类别并不保证消除所有
误触发，单向划动与不完整挥手仍应加入 Neutral 训练和独立验证。

## 保存与加载实验（`.kfesn`）

1. 完成录制或按 C 取消后，点击 **S: Save as...**，选择新的 `.kfesn` 文件名。
   一次保存所有分支，包括尚未训练的样本；不是只保存当前选择的分支。
2. 重启时使用相同模式和相同模型文件，点击 **L: Load...** 选择实验。
   已训练模型直接恢复权重，无需重新按 T；未训练实验继续录制后按 T。
3. 当前实验有未保存更改时，加载前询问是否替换；选择 No 可先另存。
   加载失败保留当前样本与模型。新增训练样本后仍需重新训练相应分支。

这是手动保存，不是自动保存；退出前请按 S。已有文件永不覆盖，每次保存使用新名字。
保存的数据是时间序列特征，不是摄像头视频，也不是完整 21 点骨架：包括原始采样时间、
八类基础手势分数、右手概率、有手标记、可用的归一化手腕坐标、片段标签、录制批次及时间，
并保留训练/独立测试归属。同时保存各分支 ESN 参数、完整储备池和读出权重、训练状态及事件门控参数。
录制批次以会话创建时的 Unix 毫秒时间标记，用于来源追溯，不保证跨设备全局唯一。

加载会重建派生训练状态，但不会重新拟合权重；实时窗口和待确认事件全部重置，
避免将上一次运行的历史拼接进当前动作。保存的测试片段不参加拟合。

### 格式、兼容与失败语义

- 二进制格式版本 1、特征版本 1，不使用 JSON。整数显式小端，浮点以 IEEE 754 double 编码。
  文件按头部版本/维度、模型指纹、分支配置/标签顺序/权重/训练与测试片段、SHA-256 尾校验排列。
- 最大文件 32 MiB；每分支训练和测试数量、每片采样量均受配置及 4096 硬上限约束。
  不支持未知版本、未知标签、非有限数值、异常时间戳、超限容量或尾随数据；不静默修复。
- 指纹覆盖 ONNX 后端 DLL、四个模型清单及其模型文件。模型文件、清单或后端变化会拒绝加载，
  即便模型名称相同。旧 `--esn` 和 `--actions` 实验不能互相加载，也不自动迁移格式。
- SHA-256 用于发现文件损坏，不提供来源认证；只加载可信来源的实验。
- 保存先校验和编码，再独占创建同目录 `.partial`，写入并刷盘后以不覆盖方式发布。
  同名目标或临时文件已存在时明确失败；不会删除他人的临时文件。崩溃可能留下 `.partial`，
  应先检查来源再人工处理；也可选择新名字。磁盘/权限失败不把实验标记为已保存。

### 状态归属与验证

Session 持有原始片段这一主事实源，ESN 编码状态是派生缓存；文件只是保存时的快照。
独立编码层负责格式校验，Windows 文件适配层负责发布，UI 先构造完整候选实验再一次性替换两条分支。
复用已有 Salts Crypto 计算校验；已安装 Salts FS 不提供独占创建及禁止替换的 rename，
因此仅在 Windows 适配层使用对应系统 API，文件对话框链接系统 comdlg32，不新增第三方依赖。
这一方案保留现有训练与分类算法，代价是保留原始片段及加载时重建缓存；可继续使用未持久化的工作流。

```powershell
cmake --build --preset win-hand-preview-user --target test_hand_experiment test_composition_esn kfhand_preview
ctest --preset win-hand-preview-user -R '^(test_hand_experiment|test_composition_esn|hand_preview_.*)$' --output-on-failure
```

持久化测试覆盖原始片段与权重精确往返、已训练/未训练恢复、继续训练、校验损坏、版本与模型不匹配、
非法数据、禁止覆盖及临时文件冲突。MED：实现迁移不等于精度已达标；仍需真实独立测试集验证
抓取、放开和 Wave 的漏检率、误触发率、跨录制批次表现和响应延迟。
