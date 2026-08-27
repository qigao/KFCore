#pragma once

#include "cuda_buffer.hpp"

#include <NvInfer.h>

#include <cstdio>
#include <memory>

namespace kfcore::tensorrt::detail
{

template <typename T> struct TensorRtDeleter
{
    void operator()(T* pointer) const noexcept
    {
        delete pointer;
    }
};

template <typename T> using TensorRtOwner = std::unique_ptr<T, TensorRtDeleter<T>>;

class TensorRtLogger final : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* message) noexcept override
    {
        if (severity <= Severity::kWARNING && message != nullptr)
        {
            std::fputs("TensorRT: ", stderr);
            std::fputs(message, stderr);
            std::fputc('\n', stderr);
        }
    }
};

class CudaStream final
{
public:
    CudaStream()
    {
        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags", "executor creation");
    }

    ~CudaStream() noexcept
    {
        if (stream_ != nullptr)
        {
            (void)cudaStreamDestroy(stream_);
        }
    }

    CudaStream(const CudaStream&)            = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const noexcept
    {
        return stream_;
    }

private:
    cudaStream_t stream_ = nullptr;
};

} // namespace kfcore::tensorrt::detail
