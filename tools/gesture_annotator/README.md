# MediaPipe 动作标注台

本地 Web 工具：导入视频，检测手部关键点，人工标注单手 / 双手动作区间，导出版本化 JSON。视频和标注不上传服务器。此工具独立于 KFCore C++ 构建，不修改 ESN、THIG 或现有训练数据接口。

## 运行

需要 Node.js 22+，以及支持 Web Workers、OffscreenCanvas、WebAssembly、IndexedDB 的现代浏览器。建议先用桌面 Chrome / Edge 验证目标视频。浏览器编解码器决定可打开的视频类型。

```powershell
cd C:\projects\cpp\KFCore\tools\gesture_annotator
npm start
```

打开 <http://127.0.0.1:5178>。无需 `npm install`，无前端打包步骤；`dist/` 内为直接维护的源码。可用 `PORT` 环境变量更改端口。服务仅绑定回环地址，且只允许读取固定静态资源，无上传接口。不要使用 `file://` 打开页面。

1. 打开视频。工具计算完整文件 SHA-256，并恢复此浏览器中对应视频的项目。
2. 可直接标注，也可“检测当前画面”或按 5 / 10 / 15 / 30 Hz “采样整段视频”。首次检测需要联网下载模型。
3. 用 I / O 设置区间起止；输入动作名称、阶段、Hand A / B 或双手，以及遮挡、未完成、负样本标记。添加区间后可编辑、删除、撤销和重做。
4. 在“当前采样 · 身份校对”中人工分配稳定身份。只修改当前采样，不推测后续帧身份。
5. 导出 JSON，并保留原视频。导入前须打开对应视频；哈希和元数据不匹配时拒绝导入。

空格播放 / 暂停，左右箭头按采样间隔移动。输入框、按钮等原生控件保留自己的键盘行为。可以保存重叠区间，例如同一动作的完整区间和阶段子区间；工具不会擅自消解重叠标签。

## 数据契约与归属

根对象 `schema` 为 `kfcore-gesture-annotation/1`：

| 字段 | 语义 |
| --- | --- |
| `video` | 文件名、字节数、SHA-256、原始尺寸与秒单位时长；不含视频字节 |
| `subject`, `session` | 人工填写的被试、会话标识；用于后续按被试划分训练 / 测试集 |
| `annotations` | 人工事实源：ID、动作名称、`start/end` 秒、阶段、主体、标记、备注 |
| `frames` | 派生检测结果，按 `timestampUs` 严格递增；无手的采样保留空数组 |
| `frames[].hands[]` | 原始 `landmarks`、`worldLandmarks`、左右手分类及分类分数；人工 `entity` 默认为 `null` |
| `inference` | 固定包版本、模型 URL / 版本、CPU、IMAGE 模式和置信度阈值 |
| `sampling` | 目标采样率，时间来源固定为 `html-video-currentTime` |

`landmarks` 为 21 个 `[x,y,z]`，使用 MediaPipe 归一化图像坐标；`worldLandmarks` 为米单位的手部局部三维坐标。保留模型浮点输出，不裁剪坐标，不伪造关键点置信度。`handednessScore` 仅是左右手分类分数。详见 [MediaPipe 官方 Web 文档](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/web_js)。

`entity` 只能为 `null`、`hand-A` 或 `hand-B`。检测数组顺序和左右手分类都不是稳定跟踪 ID。同一采样不得将两只手绑定为同一身份。区间的主体与逐采样身份分别由人工指定，不隐式传播。

阶段为 `whole/start/active/end/idle`，标记为 `occluded/incomplete/negative`，动作名称是自由文本。区间使用 `[start, end)` 语义。JSON 导入拒绝未知版本、模型配置不匹配、非有限关键点、重复 ID、非法时间、身份冲突和超限数据。

内存 `project` 是当前工作事实源；DOM 和时间轴由它派生，IndexedDB 是快照。整段推理先写临时数组，全部成功并验证后一次替换；取消 / 失败不提交部分结果。重新采样需要确认，会清除被覆盖检测的人工身份，不改变动作区间。撤销 / 重做仅涵盖动作区间，最多 30 步，不包括采样、身份或项目导入。

## 范围与兼容性风险

- **MED — 时间精度：**通过 HTMLVideoElement 定位并采样，保存 `currentTime` 的微秒整数表示，**不是**解码 PTS 或精确帧号。可变帧率视频可能多次采到同一解码画面。微秒字段不代表微秒测量精度。导出用于精确训练前需另行做解码时间对齐。
- **MED — 训练分布：**当前导出不兼容 `tools/temporal_gesture` 的生产检测器 JSONL，也未接 ESN 的训练/评估入口。不生成既有 78 维特征、不伪造 `track_id`、不自动声称可跨检测器泛化。后续转换需明确坐标、身份、标签映射和按被试划分，重新验证特征与效果。
- **MED — 模型运行：**Worker 采用 IMAGE 模式以支持任意顺序定位，不使用 VIDEO 跟踪加速。经典 Worker 保留 Emscripten `importScripts` 兼容性；不支持相关浏览器能力时明确报错，不静默切换模型或线程。
- **MED — 本地存储：**自动保存使用当前浏览器、当前 origin 下的 IndexedDB；换端口 / 浏览器不会共享。浏览器清理、配额不足或隐私模式均可能影响持久化；失败会显示错误，JSON 备份仍是必要的。未保存或采样中关闭页面会请求浏览器提示。
- **LOW — 资源：**视频上限 256 MiB、1 小时、单边 8192 像素；最多 3000 个采样、1000 个区间、48 MiB JSON。完整 SHA-256 需要读入视频字节，大视频会增加内存占用。限制集中在 `domain.mjs`，推理超时集中在 `app.mjs`。

架构为纯领域数据校验 + 页面动作 / 派生视图 + Worker 推理适配 + IndexedDB 存储适配。没有服务端业务状态，不引入 C++ 依赖。此首版不包含自动身份跟踪、关键点坐标拖拽修正、摄像头录制、精确帧解码或模型训练。

## 外部依赖与隐私

用户请求的手部检测能力由 `@mediapipe/tasks-vision@0.10.21` 提供，通过 jsDelivr 加载；模型固定为 Google `hand_landmarker/float16/1`。下载约 9.6 MB SIMD WASM 和 7.8 MB 模型，另有脚本资源。未使用 `latest`，不打包第三方二进制。许可证信息见仓库根 `THIRD_PARTY_NOTICES.md`。

推理在本机运行，但依赖下载会联系 jsDelivr / Google，二者可看到网络请求信息；该工具没有上传视频或标注的代码。**不承诺完全离线运行**：浏览器缓存不构成可靠的离线资源分发方案。私有部署需另行审查依赖本地化和许可，不在此版自动更改部署方式。

可选 WebMCP 仅在 `document.modelContext.registerTool` 存在时注册只读摘要工具；其输出包含人工文本，应视为不可信数据。未支持 WebMCP 的浏览器正常使用页面功能。

## 验证

```powershell
npm test
npm run check
```

Node 测试覆盖数据往返、边界校验、身份冲突、采样限制、最近采样查找、原地替换隔离、历史有界性及静态入口引用。另可启动服务后检查：

```powershell
(Invoke-WebRequest http://127.0.0.1:5178 -Method Head).StatusCode
```

这些检查不等同于浏览器推理验收。发布或用于采集前，应在目标浏览器用真实单手 / 双手 / 遮挡视频验证：模型加载、叠加对齐、取消后数据不变、导出后再导入、刷新恢复、拒绝不匹配视频、键盘和窄屏操作。当前未完成真实视频浏览器端和 WebMCP 运行验证。
