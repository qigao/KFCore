#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <string_view>

namespace kfcore::yolo::detail
{

void check_cuda(cudaError_t result, std::string_view operation, std::string_view stage);

struct CudaMemoryApi
{
    cudaError_t (*malloc_device)(void**, std::size_t);
    cudaError_t (*free_device)(void*);
    cudaError_t (*malloc_pinned)(void**, std::size_t);
    cudaError_t (*free_pinned)(void*);
};

class CudaBuffer final
{
public:
    CudaBuffer() noexcept;
    explicit CudaBuffer(const CudaMemoryApi& api) noexcept;
    ~CudaBuffer() noexcept;

    CudaBuffer(const CudaBuffer&)            = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&& other) noexcept;
    CudaBuffer& operator=(CudaBuffer&& other) noexcept;

    // Allocation and limit failures preserve the previous allocation. Once replacement begins
    // releasing the old allocation, a cleanup error provides only the basic guarantee: this
    // owner retains the valid replacement and never accesses or releases the old pointer again.
    void        reserve(std::size_t bytes, std::size_t hard_limit);
    void*       data() noexcept;
    const void* data() const noexcept;
    std::size_t capacity() const noexcept;

private:
    CudaMemoryApi api_;
    void*         data_     = nullptr;
    std::size_t   capacity_ = 0;
};

class PinnedHostBuffer final
{
public:
    PinnedHostBuffer() noexcept;
    explicit PinnedHostBuffer(const CudaMemoryApi& api) noexcept;
    ~PinnedHostBuffer() noexcept;

    PinnedHostBuffer(const PinnedHostBuffer&)            = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;

    // The same replacement and cleanup guarantees as CudaBuffer::reserve apply.
    void        reserve(std::size_t bytes, std::size_t hard_limit);
    void*       data() noexcept;
    const void* data() const noexcept;
    std::size_t capacity() const noexcept;

private:
    CudaMemoryApi api_;
    void*         data_     = nullptr;
    std::size_t   capacity_ = 0;
};

} // namespace kfcore::yolo::detail
