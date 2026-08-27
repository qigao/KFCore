#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <string_view>

namespace kfcore::tensorrt::detail
{

void check_cuda(cudaError_t result, std::string_view operation, std::string_view stage);

#if defined(KFCORE_TENSORRT_RUNTIME_CUDA_BUFFER_TESTING)
struct CudaMemoryApi
{
    cudaError_t (*malloc_device)(void**, std::size_t);
    cudaError_t (*free_device)(void*);
    cudaError_t (*malloc_pinned)(void**, std::size_t);
    cudaError_t (*free_pinned)(void*);
};
#endif

class CudaBuffer final
{
public:
    CudaBuffer() noexcept;
#if defined(KFCORE_TENSORRT_RUNTIME_CUDA_BUFFER_TESTING)
    explicit CudaBuffer(const CudaMemoryApi& api) noexcept;
#endif
    ~CudaBuffer() noexcept;

    CudaBuffer(const CudaBuffer&)            = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&& other) noexcept;
    CudaBuffer& operator=(CudaBuffer&& other) noexcept;

    // Allocation and limit failures preserve the old allocation. If freeing the old allocation
    // reports an error after replacement, this owner retains only the valid replacement.
    void        reserve(std::size_t bytes, std::size_t hard_limit);
    void*       data() noexcept;
    const void* data() const noexcept;
    std::size_t capacity() const noexcept;

private:
#if defined(KFCORE_TENSORRT_RUNTIME_CUDA_BUFFER_TESTING)
    CudaMemoryApi api_;
#endif
    void*       data_     = nullptr;
    std::size_t capacity_ = 0;
};

class PinnedHostBuffer final
{
public:
    PinnedHostBuffer() noexcept;
#if defined(KFCORE_TENSORRT_RUNTIME_CUDA_BUFFER_TESTING)
    explicit PinnedHostBuffer(const CudaMemoryApi& api) noexcept;
#endif
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
#if defined(KFCORE_TENSORRT_RUNTIME_CUDA_BUFFER_TESTING)
    CudaMemoryApi api_;
#endif
    void*       data_     = nullptr;
    std::size_t capacity_ = 0;
};

} // namespace kfcore::tensorrt::detail
