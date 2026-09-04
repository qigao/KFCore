#pragma once

#include "kfcore/face_models/cpu.hpp"
#include "kfcore/face_models/error.hpp"
#include "kfcore/runtime_onnx/runtime.hpp"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace kfcore::face_models::detail
{

using CpuModelContract = runtime_onnx::ModelContract;
using CpuSessionOutput = runtime_onnx::HostTensor;

runtime_onnx::TensorContract fp32_contract(
    std::string name, std::vector<std::int64_t> dimensions);

runtime_onnx::OwnedSession make_session(
    const char* model_name, const std::filesystem::path& model_path,
    CpuModelContract contract, const CpuFaceModelOptions& options);

std::vector<CpuSessionOutput> run(
    const runtime_onnx::OwnedSession& session,
    const std::vector<runtime_onnx::FloatTensorView>& inputs);

class CpuCallGuard final
{
public:
    CpuCallGuard(std::atomic_flag& in_use, const char* model_name)
        : in_use_(in_use)
    {
        if (in_use_.test_and_set(std::memory_order_acquire))
        {
            throw FaceModelError(FaceModelErrorCode::InvalidArgument,
                                 std::string(model_name) +
                                     " inference stage: concurrent calls are not supported");
        }
    }

    ~CpuCallGuard() noexcept { in_use_.clear(std::memory_order_release); }

    CpuCallGuard(const CpuCallGuard&)            = delete;
    CpuCallGuard& operator=(const CpuCallGuard&) = delete;

private:
    std::atomic_flag& in_use_;
};

[[noreturn]] inline void throw_allocation(const char* model_name, const char* stage)
{
    throw FaceModelError(FaceModelErrorCode::ResourceLimitExceeded,
                         std::string(model_name) + " " + stage +
                             " stage: allocation failed");
}

inline runtime_onnx::FloatTensorView image_input(const CpuTensorView& view,
                                                  std::int64_t extent)
{
    return { view.data, view.element_count,
             { 1, kFaceModelInputChannels, extent, extent } };
}

inline runtime_onnx::FloatTensorView embedding_input(const CpuTensorView& view)
{
    return { view.data, view.element_count,
             { 1, static_cast<std::int64_t>(kInSwapperEmbeddingLength) } };
}

} // namespace kfcore::face_models::detail
