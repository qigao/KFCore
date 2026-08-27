#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace kfcore::tensorrt
{

enum class DataType
{
    Float32,
    Float16,
    Int8,
    Int32,
    Bool,
    UInt8,
    BFloat16,
    Int64,
};

enum class MemoryKind
{
    Host,
    CudaDevice
};
enum class TensorIoMode
{
    Input,
    Output
};

using TensorShape = std::vector<std::int64_t>;

struct TensorProfile
{
    TensorShape minimum;
    TensorShape optimum;
    TensorShape maximum;
};

struct TensorDescriptor
{
    std::string                  name;
    TensorIoMode                 mode      = TensorIoMode::Input;
    DataType                     data_type = DataType::Float32;
    // The network declaration uses -1 for runtime dimensions. It is not an output bound.
    TensorShape                  declared_shape;
    // Only inputs have profile 0 bounds. Output sizes are resolved from IExecutionContext.
    std::optional<TensorProfile> profile;
};

// TensorView borrows read-only input storage for the duration of Executor::run().
struct TensorView
{
    std::string name;
    DataType    data_type = DataType::Float32;
    TensorShape shape;
    const void* data        = nullptr;
    std::size_t byte_size   = 0;
    MemoryKind  memory_kind = MemoryKind::Host;
};

// MutableTensorView borrows caller-owned output storage for the duration of Executor::run().
struct MutableTensorView
{
    std::string name;
    DataType    data_type = DataType::Float32;
    TensorShape shape;
    void*       data        = nullptr;
    std::size_t byte_size   = 0;
    MemoryKind  memory_kind = MemoryKind::Host;
};

struct EngineOptions
{
    int         device_id                   = 0;
    std::size_t max_serialized_engine_bytes = 512U * 1024U * 1024U;
    std::size_t max_tensor_count            = 64;
    std::size_t max_input_bytes             = 256U * 1024U * 1024U;
    std::size_t max_output_bytes            = 256U * 1024U * 1024U;
};

} // namespace kfcore::tensorrt
