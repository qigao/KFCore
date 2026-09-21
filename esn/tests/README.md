# ESN 时序验证

`kfcore_esn_tests` 覆盖算子、布局与失败语义；`kfcore_esn_sequence_tests`
覆盖从预热、状态收集、ridge 拟合到独立测试段预测的完整路径。
后者已注册 CTest，也加入现有 Linux ASan/UBSan 工作流。

测试方法借鉴 [AutoESN](https://github.com/Ro6ertWcislo/AutoESN#grid-search-configuration)
的时间序列预测、验证集选参和多种子架构比较。这里使用自行生成的数据和 KFCore
的 tanh 实现，不复制上游代码，也不声称复现论文的 SNA、数据集或精度结果。

## 固定实验协议

- 所有架构总计 32 个神经元：ESN 为 32；gESN 为两组各 16；dESN 为两层各 16；
  gdESN 为两组、每组两层、每层 8。
- 输入缩放 0.5、leak 0.8、连接密度 0.3；谱半径使用当前实现的 200 步估计缩放至 0.8。
  这是估计值，不是精确特征值分解结果。
- 按时间顺序：预热 100 步、训练 800 步、验证 300 步、测试 400 步，不打乱。
- 单步预测：输入 `u[t]`，目标为 `u[t+1]`。状态在各段间连续传递；测试时仍输入真实
  观测，属于单步预测，不是闭环多步生成。
- 仅用训练段拟合权重；从 `{0.01, 0.1, 1}` 选择验证 NRMSE 最小的 ridge lambda。
  不使用测试段选参，也不在测试段更新权重。更小正则可能受 FP32 Gram 矩阵条件数影响，
  因而不列入本实验候选；任何候选求解失败都会使测试失败，不跳过候选。
- `NRMSE = sqrt(sum((prediction-target)^2) / sum((target-mean(target))^2))`，
  分母取当前评估段目标的中心平方和，仅用于评分，不反馈模型。零方差或非有限结果视为失败。

## 两类入口

日常回归采用种子 42 和 `(sin(0.17*t) + 0.5*sin(0.31*t))/1.5`。
四种架构均要求测试 NRMSE 小于 0.15，且小于持续值基线误差的 80%。
基线预测为 `u[t]`。这些是本地宽松回归门槛，不是论文指标。

独立质量目标 `kfcore_esn_quality` 默认不构建、不注册 CTest；显式构建后运行。
使用种子 42–46，对每种任务、每种架构输出每次的 lambda、验证/测试 NRMSE、基线 NRMSE，
以及跨种子的均值、总体标准差、最小值和最大值。质量评估要求运行成功且结果有限，
不把架构排名或特定预测精度作为通过条件。

三个任务为：

1. 上述叠加正弦单步预测。
2. Mackey–Glass 型离散序列：
   `x[t+1] = x[t] + 0.2*x[t-17]/(1+x[t-17]^10) - 0.1*x[t]`。
   使用 Euler 步长 1、常量历史 1.2、丢弃前 1000 步，再减去固定常量 1。
   这是明确指定的离散测试序列，不等同于连续系统高精度积分数据。基线为持续值预测。
3. 五步延迟记忆：独立固定种子的均匀伪随机输入，目标为 `u[t-5]`；
   基线为训练段目标均值。该项只测一个延迟，不代表完整 memory capacity。

## Windows 复验

先进入 VS `VsDevCmd.bat -arch=x64 -host_arch=x64` 环境，再在仓库根目录执行：

```bat
cmake --preset win-cpu-release-user
cmake --build --preset win-cpu-release-user --target kfcore_esn_tests kfcore_esn_sequence_tests kfcore_esn_quality
ctest --preset win-cpu-release-user -R "^kfcore_esn.*tests$" --output-on-failure
build\cpu\Msvc-Release\bin\kfcore_esn_quality.exe
```

测试数据、模型缓冲区和生成逻辑全部位于 `test_esn_sequence.c`，不进入生产库。
规模和候选参数集中在该文件顶部；静态固定容量约束内存，无网络下载或新增依赖。
如调整任务规模、信号或模型参数，应重新解释回归门槛，不能根据测试集结果调参。
