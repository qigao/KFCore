#include "cuda_buffer.hpp"

#include "kfcore/yolo/error.hpp"

#include <string>
#include <utility>

namespace kfcore::yolo::detail
{
namespace
{

    [[noreturn]] void throw_limit(std::string_view allocation_kind, std::size_t bytes,
                                  std::size_t hard_limit)
    {
        throw YoloError(YoloErrorCode::ResourceLimitExceeded,
                        std::string(allocation_kind) + " reserve stage: requested " +
                            std::to_string(bytes) + " bytes exceeds hard limit " +
                            std::to_string(hard_limit));
    }

    const CudaMemoryApi& default_memory_api() noexcept
    {
        static const CudaMemoryApi api {
            cudaMalloc,
            cudaFree,
            cudaMallocHost,
            cudaFreeHost,
        };
        return api;
    }

    void release_cuda(const CudaMemoryApi& api, void* pointer) noexcept
    {
        if (pointer != nullptr)
        {
            (void)api.free_device(pointer);
        }
    }

    void release_pinned(const CudaMemoryApi& api, void* pointer) noexcept
    {
        if (pointer != nullptr)
        {
            (void)api.free_pinned(pointer);
        }
    }

} // namespace

void check_cuda(cudaError_t result, std::string_view operation, std::string_view stage)
{
    if (result == cudaSuccess)
    {
        return;
    }
    const char* error_name = cudaGetErrorName(result);
    const char* error_text = cudaGetErrorString(result);
    throw YoloError(YoloErrorCode::CudaFailure,
                    std::string(operation) + " failed during " + std::string(stage) + " stage: " +
                        (error_name != nullptr ? error_name : "unknown CUDA error") + " (" +
                        (error_text != nullptr ? error_text : "no CUDA error description") + ")");
}

CudaBuffer::CudaBuffer() noexcept
    : CudaBuffer(default_memory_api())
{
}

CudaBuffer::CudaBuffer(const CudaMemoryApi& api) noexcept
    : api_(api)
{
}

CudaBuffer::~CudaBuffer() noexcept
{
    release_cuda(api_, data_);
}

CudaBuffer::CudaBuffer(CudaBuffer&& other) noexcept
    : api_(other.api_)
    , data_(std::exchange(other.data_, nullptr))
    , capacity_(std::exchange(other.capacity_, 0))
{
}

CudaBuffer& CudaBuffer::operator=(CudaBuffer&& other) noexcept
{
    if (this != &other)
    {
        release_cuda(api_, data_);
        api_      = other.api_;
        data_     = std::exchange(other.data_, nullptr);
        capacity_ = std::exchange(other.capacity_, 0);
    }
    return *this;
}

void CudaBuffer::reserve(std::size_t bytes, std::size_t hard_limit)
{
    if (bytes > hard_limit)
    {
        throw_limit("CUDA device buffer", bytes, hard_limit);
    }
    if (bytes <= capacity_)
    {
        return;
    }

    void* replacement = nullptr;
    check_cuda(api_.malloc_device(&replacement, bytes), "cudaMalloc", "CUDA device buffer reserve");

    void* old = data_;
    data_     = replacement;
    capacity_ = bytes;
    if (old != nullptr)
    {
        const cudaError_t release_result = api_.free_device(old);
        check_cuda(release_result, "cudaFree", "CUDA device buffer replacement cleanup");
    }
}

void* CudaBuffer::data() noexcept
{
    return data_;
}

const void* CudaBuffer::data() const noexcept
{
    return data_;
}

std::size_t CudaBuffer::capacity() const noexcept
{
    return capacity_;
}

PinnedHostBuffer::PinnedHostBuffer() noexcept
    : PinnedHostBuffer(default_memory_api())
{
}

PinnedHostBuffer::PinnedHostBuffer(const CudaMemoryApi& api) noexcept
    : api_(api)
{
}

PinnedHostBuffer::~PinnedHostBuffer() noexcept
{
    release_pinned(api_, data_);
}

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
    : api_(other.api_)
    , data_(std::exchange(other.data_, nullptr))
    , capacity_(std::exchange(other.capacity_, 0))
{
}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept
{
    if (this != &other)
    {
        release_pinned(api_, data_);
        api_      = other.api_;
        data_     = std::exchange(other.data_, nullptr);
        capacity_ = std::exchange(other.capacity_, 0);
    }
    return *this;
}

void PinnedHostBuffer::reserve(std::size_t bytes, std::size_t hard_limit)
{
    if (bytes > hard_limit)
    {
        throw_limit("CUDA pinned host buffer", bytes, hard_limit);
    }
    if (bytes <= capacity_)
    {
        return;
    }

    void* replacement = nullptr;
    check_cuda(api_.malloc_pinned(&replacement, bytes), "cudaMallocHost",
               "CUDA pinned host buffer reserve");

    void* old = data_;
    data_     = replacement;
    capacity_ = bytes;
    if (old != nullptr)
    {
        const cudaError_t release_result = api_.free_pinned(old);
        check_cuda(release_result, "cudaFreeHost", "CUDA pinned host buffer replacement cleanup");
    }
}

void* PinnedHostBuffer::data() noexcept
{
    return data_;
}

const void* PinnedHostBuffer::data() const noexcept
{
    return data_;
}

std::size_t PinnedHostBuffer::capacity() const noexcept
{
    return capacity_;
}

} // namespace kfcore::yolo::detail
