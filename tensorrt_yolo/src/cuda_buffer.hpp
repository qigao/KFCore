#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <string_view>

namespace kfcore::yolo::detail
{

void check_cuda(cudaError_t result, std::string_view operation, std::string_view stage);

class CudaBuffer final
{
public:
    CudaBuffer() noexcept = default;
    ~CudaBuffer() noexcept;

    CudaBuffer(const CudaBuffer&)            = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&& other) noexcept;
    CudaBuffer& operator=(CudaBuffer&& other) noexcept;

    void        reserve(std::size_t bytes, std::size_t hard_limit);
    void*       data() noexcept;
    const void* data() const noexcept;
    std::size_t capacity() const noexcept;

private:
    void*       data_     = nullptr;
    std::size_t capacity_ = 0;
};

class PinnedHostBuffer final
{
public:
    PinnedHostBuffer() noexcept = default;
    ~PinnedHostBuffer() noexcept;

    PinnedHostBuffer(const PinnedHostBuffer&)            = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;

    void        reserve(std::size_t bytes, std::size_t hard_limit);
    void*       data() noexcept;
    const void* data() const noexcept;
    std::size_t capacity() const noexcept;

private:
    void*       data_     = nullptr;
    std::size_t capacity_ = 0;
};

} // namespace kfcore::yolo::detail
