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

    void release_cuda(void* pointer) noexcept
    {
        if (pointer != nullptr)
        {
            (void)cudaFree(pointer);
        }
    }

    void release_pinned(void* pointer) noexcept
    {
        if (pointer != nullptr)
        {
            (void)cudaFreeHost(pointer);
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

CudaBuffer::~CudaBuffer() noexcept
{
    release_cuda(data_);
}

CudaBuffer::CudaBuffer(CudaBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr))
    , capacity_(std::exchange(other.capacity_, 0))
{
}

CudaBuffer& CudaBuffer::operator=(CudaBuffer&& other) noexcept
{
    if (this != &other)
    {
        release_cuda(data_);
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
    check_cuda(cudaMalloc(&replacement, bytes), "cudaMalloc", "CUDA device buffer reserve");
    if (data_ != nullptr)
    {
        const cudaError_t release_result = cudaFree(data_);
        if (release_result != cudaSuccess)
        {
            release_cuda(replacement);
            check_cuda(release_result, "cudaFree", "CUDA device buffer replacement");
        }
    }
    data_     = replacement;
    capacity_ = bytes;
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

PinnedHostBuffer::~PinnedHostBuffer() noexcept
{
    release_pinned(data_);
}

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr))
    , capacity_(std::exchange(other.capacity_, 0))
{
}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept
{
    if (this != &other)
    {
        release_pinned(data_);
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
    check_cuda(cudaMallocHost(&replacement, bytes), "cudaMallocHost",
               "CUDA pinned host buffer reserve");
    if (data_ != nullptr)
    {
        const cudaError_t release_result = cudaFreeHost(data_);
        if (release_result != cudaSuccess)
        {
            release_pinned(replacement);
            check_cuda(release_result, "cudaFreeHost", "CUDA pinned host buffer replacement");
        }
    }
    data_     = replacement;
    capacity_ = bytes;
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
