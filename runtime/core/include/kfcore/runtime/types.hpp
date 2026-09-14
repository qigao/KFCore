#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace kfcore::runtime
{

enum class DataType : std::uint32_t
{
    Float32,
    Float16,
    Int8,
    Int32,
    Int64,
    UInt8,
    Bool,
    BFloat16,
};

enum class MemoryKind : std::uint32_t
{
    Host,
    PinnedHost,
    Device,
};

using TensorShape = std::vector<std::int64_t>;

struct RuntimeVersion
{
    std::uint32_t major = 0U;
    std::uint32_t minor = 0U;
    std::uint32_t patch = 0U;
    std::uint32_t build = 0U;
};

struct TensorDescriptor
{
    std::string name;
    DataType    data_type = DataType::Float32;
    TensorShape shape;
    bool        is_input = true;
};

struct TensorView
{
    std::string_view name;
    DataType         data_type = DataType::Float32;
    TensorShape      shape;
    const void*      data = nullptr;
    std::size_t      byte_size = 0U;
    MemoryKind       memory_kind = MemoryKind::Host;
    std::string_view device_id;
};

struct MutableTensorView
{
    std::string_view name;
    DataType         data_type = DataType::Float32;
    TensorShape      shape;
    void*            data = nullptr;
    std::size_t      byte_size = 0U;
    MemoryKind       memory_kind = MemoryKind::Host;
    std::string_view device_id;
};

struct DynamicMutableTensorView
{
    std::string_view name;
    DataType         data_type = DataType::Float32;
    void*            data = nullptr;
    std::size_t      capacity_bytes = 0U;
    MemoryKind       memory_kind = MemoryKind::Host;
    std::string_view device_id;
    TensorShape      shape;
    std::size_t      byte_size = 0U;
};

struct BackendDevice
{
    std::string   id;
    std::string   name;
    std::uint64_t capabilities = 0U;
    std::uint32_t compute_capability_major = 0U;
    std::uint32_t compute_capability_minor = 0U;
};

} // namespace kfcore::runtime
