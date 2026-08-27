#pragma once

#include "engine.hpp"

#include <atomic>
#include <memory>
#include <vector>

namespace kfcore::tensorrt
{

namespace detail
{

struct ExecutorStagingBuffers
{
    CudaBuffer       device;
    PinnedHostBuffer host;
};

} // namespace detail

struct Executor::Impl final
{
    Impl(std::shared_ptr<const Engine::Impl> engine_state,
         detail::TensorRtOwner<nvinfer1::IExecutionContext> execution_context);
    ~Impl() noexcept;

    std::shared_ptr<const Engine::Impl>              engine;
    detail::TensorRtOwner<nvinfer1::IExecutionContext> context;
    detail::CudaStream                               stream;
    std::vector<detail::ExecutorStagingBuffers>      staging;
    std::atomic_flag                                 in_use = ATOMIC_FLAG_INIT;
};

} // namespace kfcore::tensorrt
