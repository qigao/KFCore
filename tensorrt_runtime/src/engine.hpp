#pragma once

#include "kfcore/tensorrt/runtime.hpp"
#include "tensorrt_raii.hpp"

#include <NvInfer.h>

#include <vector>

namespace kfcore::tensorrt
{

struct Engine::Impl final
{
    detail::TensorRtLogger                       logger;
    detail::TensorRtOwner<nvinfer1::IRuntime>    runtime;
    detail::TensorRtOwner<nvinfer1::ICudaEngine> engine;
    std::vector<TensorDescriptor>                tensors;
    EngineOptions                                options;
};

} // namespace kfcore::tensorrt
