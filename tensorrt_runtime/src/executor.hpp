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

    // Keeping the shared_ptr in a heap anchor permits a deliberate safe leak when the target CUDA
    // device cannot be established during noexcept destruction.
    std::unique_ptr<std::shared_ptr<const Engine::Impl>> engine_owner;
    const Engine::Impl*                                 engine = nullptr;
    detail::TensorRtOwner<nvinfer1::IExecutionContext> context;
    std::unique_ptr<detail::CudaStream>              stream;
    std::vector<detail::ExecutorStagingBuffers>      staging;
    std::atomic_flag                                 in_use = ATOMIC_FLAG_INIT;
};

} // namespace kfcore::tensorrt
