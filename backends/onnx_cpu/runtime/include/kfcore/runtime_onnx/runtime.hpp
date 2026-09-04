#pragma once

#include "kfcore/runtime_onnx/error.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace kfcore::runtime_onnx
{

inline constexpr std::size_t kDefaultMaxModelBytes  = 1024U * 1024U * 1024U;
inline constexpr std::size_t kDefaultMaxOutputBytes = 256U * 1024U * 1024U;

enum class ElementType
{
    Float32,
    Int64
};

struct TensorContract
{
    std::string               name;
    ElementType               element_type = ElementType::Float32;
    std::vector<std::int64_t> dimensions;
};

struct ModelContract
{
    std::string                 model_name;
    std::vector<TensorContract> inputs;
    std::vector<TensorContract> outputs;
};

struct FloatTensorView
{
    const float*              data          = nullptr;
    std::size_t               element_count = 0U;
    std::vector<std::int64_t> dimensions;
};

struct HostTensor
{
    ElementType               element_type = ElementType::Float32;
    std::vector<std::int64_t> dimensions;
    std::vector<float>        float_values;
    std::vector<std::int64_t> int64_values;
};

struct SessionOptions
{
    int         intra_op_threads = 0;
    int         inter_op_threads = 0;
    std::size_t max_model_bytes  = kDefaultMaxModelBytes;
    std::size_t max_output_bytes = kDefaultMaxOutputBytes;
};

class Environment final
{
public:
    explicit Environment(const char* log_id);
    ~Environment();

    Environment(Environment&&)                 = delete;
    Environment& operator=(Environment&&)      = delete;
    Environment(const Environment&)            = delete;
    Environment& operator=(const Environment&) = delete;

private:
    friend class Session;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class Session final
{
public:
    Session(Environment& environment, const std::filesystem::path& model_path,
            ModelContract contract, const SessionOptions& options = {});
    ~Session();

    Session(Session&&) noexcept;
    Session& operator=(Session&&) noexcept;
    Session(const Session&)            = delete;
    Session& operator=(const Session&) = delete;

    [[nodiscard]] std::vector<HostTensor>
    run(const std::vector<FloatTensorView>& inputs) const;

    [[nodiscard]] const std::vector<std::int64_t>&
    declared_input_dimensions(std::size_t index) const;
    [[nodiscard]] const std::vector<std::int64_t>&
    declared_output_dimensions(std::size_t index) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class OwnedSession final
{
public:
    OwnedSession(std::string log_id, const std::filesystem::path& model_path,
                 ModelContract contract, const SessionOptions& options = {});
    ~OwnedSession();

    OwnedSession(OwnedSession&&) noexcept;
    OwnedSession& operator=(OwnedSession&&) noexcept;
    OwnedSession(const OwnedSession&)            = delete;
    OwnedSession& operator=(const OwnedSession&) = delete;

    [[nodiscard]] Session&       session() noexcept;
    [[nodiscard]] const Session& session() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::runtime_onnx
