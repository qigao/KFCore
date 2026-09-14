# ONNX 执行测试数据

两个模型均为本仓库生成的无权重小图：IR 8、ONNX opset 13，
输入 `input` 与输出 `output` 都是 FP32 `[1, 4]`。

- `relu.onnx`：单个 `Relu` 节点，输入 `[-2, 0, 3, -1]` 得到 `[0, 0, 3, 0]`。
- `binarizer.onnx`：单个 `ai.onnx.ml::Binarizer` 节点（该域 opset 1，默认阈值 0），
  同样输入得到 `[0, 0, 1, 0]`。在 ORT 1.21 中需要 CPU EP，用于验证 CPU 节点允许/拒绝策略。

模型已通过 ONNX checker 校验；运行 C++ 测试不依赖 Python 或 ONNX Python 包。
