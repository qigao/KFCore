#pragma once

#include "kfcore/face_applications/cpu.hpp"

#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace kfcore::face_applications::detail
{

struct TensorContract
{
    std::string               name;
    std::vector<std::int64_t> dimensions;
};

struct ModelContract
{
    std::string                 model_name;
    std::vector<TensorContract> inputs;
    std::vector<TensorContract> outputs;
};

struct HostTensorView
{
    const float*               data = nullptr;
    std::size_t                element_count = 0U;
    std::vector<std::int64_t> dimensions;
};

struct HostTensor
{
    std::vector<std::int64_t> dimensions;
    std::vector<float>        values;
};

class OnnxSession final
{
public:
    OnnxSession(Ort::Env& environment, const std::filesystem::path& model_path,
                const ModelContract& contract, const CpuFaceSwapOptions& options);
    ~OnnxSession();

    OnnxSession(OnnxSession&&) noexcept;
    OnnxSession& operator=(OnnxSession&&) noexcept;
    OnnxSession(const OnnxSession&)            = delete;
    OnnxSession& operator=(const OnnxSession&) = delete;

    [[nodiscard]] std::vector<HostTensor>
    run(const std::vector<HostTensorView>& inputs) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::face_applications::detail
