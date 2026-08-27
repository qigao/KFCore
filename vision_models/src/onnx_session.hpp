#pragma once

#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace kfcore::vision_models::detail
{

enum class OnnxElementType
{
    Float32,
    Int64,
};

struct OnnxTensorContract
{
    std::string               name;
    OnnxElementType           element_type = OnnxElementType::Float32;
    std::vector<std::int64_t> dimensions;
};

struct OnnxModelContract
{
    std::string                     model_name;
    std::vector<OnnxTensorContract> inputs;
    std::vector<OnnxTensorContract> outputs;
};

struct OnnxFloatTensorView
{
    const float*               data = nullptr;
    std::size_t                element_count = 0;
    std::vector<std::int64_t> dimensions;
};

struct OnnxHostTensor
{
    OnnxElementType           element_type = OnnxElementType::Float32;
    std::vector<std::int64_t> dimensions;
    std::vector<float>        float_values;
    std::vector<std::int64_t> int64_values;
};

struct OnnxSessionOptions
{
    int         intra_op_threads = 0;
    int         inter_op_threads = 0;
    std::size_t max_model_bytes  = 0;
    std::size_t max_output_bytes = 0;
};

class OnnxSession final
{
public:
    OnnxSession(Ort::Env& environment, const std::filesystem::path& model_path,
                OnnxModelContract contract, const OnnxSessionOptions& options);
    ~OnnxSession();

    OnnxSession(OnnxSession&&) noexcept;
    OnnxSession& operator=(OnnxSession&&) noexcept;
    OnnxSession(const OnnxSession&)            = delete;
    OnnxSession& operator=(const OnnxSession&) = delete;

    [[nodiscard]] std::vector<OnnxHostTensor>
    run(const std::vector<OnnxFloatTensorView>& inputs) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::vision_models::detail
