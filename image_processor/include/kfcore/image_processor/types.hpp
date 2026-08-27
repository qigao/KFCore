#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kfcore::image
{

enum class PixelFormat
{
    Bgr8,
    Rgb8,
    Gray8,
};

enum class MemoryKind
{
    Host,
    CudaDevice,
};

enum class TensorElementType
{
    Float16,
    Float32,
};

enum class TensorLayout
{
    Nchw,
};

struct ImageView
{
    const void*  data         = nullptr;
    std::size_t  byte_size    = 0;
    std::int32_t width        = 0;
    std::int32_t height       = 0;
    std::size_t  row_stride   = 0;
    PixelFormat  pixel_format = PixelFormat::Bgr8;
    MemoryKind   memory_kind  = MemoryKind::Host;
};

struct TensorView
{
    void*             data         = nullptr;
    std::size_t       byte_size    = 0;
    std::int32_t      batch        = 0;
    std::int32_t      channels     = 0;
    std::int32_t      height       = 0;
    std::int32_t      width        = 0;
    TensorElementType element_type = TensorElementType::Float32;
    TensorLayout      layout       = TensorLayout::Nchw;
    MemoryKind        memory_kind  = MemoryKind::CudaDevice;
};

struct MutableBufferView
{
    void*       data      = nullptr;
    std::size_t byte_size = 0;
};

struct LetterboxTransform
{
    float        scale         = 0.0f;
    float        pad_x         = 0.0f;
    float        pad_y         = 0.0f;
    std::int32_t source_width  = 0;
    std::int32_t source_height = 0;
};

struct ImagePlan
{
    LetterboxTransform transform;
    std::size_t        source_span_bytes = 0;
    std::size_t        packed_bytes      = 0;
    std::size_t        staging_offset    = 0;
    bool               requires_staging  = false;
};

struct BatchPlan
{
    std::vector<ImagePlan> images;
    std::size_t            host_staging_bytes   = 0;
    std::size_t            device_staging_bytes = 0;
    std::size_t            tensor_bytes         = 0;
};

struct PreprocessOptions
{
    PixelFormat          output_format = PixelFormat::Rgb8;
    std::array<float, 3> mean { 0.0f, 0.0f, 0.0f };
    std::array<float, 3> stddev { 1.0f, 1.0f, 1.0f };
    float                border_value = 114.0f;
};

struct AffineTransform
{
    // Maps an integer destination pixel coordinate to the corresponding source coordinate.
    std::array<float, 6> destination_to_source { 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F };
};

struct CudaImageProcessorOptions
{
    int         device_id        = 0;
    std::size_t max_source_bytes = 64U * 1024U * 1024U;
    std::size_t max_tensor_bytes = 16U * 1024U * 1024U;
};

} // namespace kfcore::image
