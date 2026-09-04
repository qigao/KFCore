#pragma once

#include "kfcore/runtime_onnx/runtime.hpp"

#include <filesystem>
#include <vector>

namespace kfcore::hand_models::detail
{

using OnnxElementType     = runtime_onnx::ElementType;
using OnnxTensorContract  = runtime_onnx::TensorContract;
using OnnxModelContract   = runtime_onnx::ModelContract;
using OnnxFloatTensorView = runtime_onnx::FloatTensorView;
using OnnxHostTensor      = runtime_onnx::HostTensor;
using OnnxSessionOptions  = runtime_onnx::SessionOptions;

runtime_onnx::Session make_session(
    runtime_onnx::Environment& environment,
    const std::filesystem::path& model_path,
    OnnxModelContract contract, const OnnxSessionOptions& options);

std::vector<OnnxHostTensor> run(
    const runtime_onnx::Session& session,
    const std::vector<OnnxFloatTensorView>& inputs);

} // namespace kfcore::hand_models::detail
