#pragma once

#include "cuda_buffer.hpp"
#include "engine_contract.hpp"
#include "kfcore/yolo/tensorrt.hpp"

#include <NvInfer.h>

#include <cstdio>
#include <memory>
#include <utility>

namespace kfcore::yolo
{
namespace detail
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
                       "cudaStreamCreateWithFlags", "detector creation");
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

} // namespace detail

struct Engine::State final
{
    detail::TensorRtLogger                       logger;
    detail::TensorRtOwner<nvinfer1::IRuntime>    runtime;
    detail::TensorRtOwner<nvinfer1::ICudaEngine> engine;
    ValidatedContract                            contract;
    EngineOptions                                options;
};

struct TensorRtDetector::Impl final
{
    Impl(std::shared_ptr<const Engine::State>               state_in,
         detail::TensorRtOwner<nvinfer1::IExecutionContext> context_in, DetectorOptions options_in)
        : state(std::move(state_in))
        , context(std::move(context_in))
        , options(std::move(options_in))
    {
    }

    std::shared_ptr<const Engine::State>               state;
    detail::TensorRtOwner<nvinfer1::IExecutionContext> context;
    detail::CudaStream                                 stream;
    DetectorOptions                                    options;
    detail::CudaBuffer                                 input_device;
    detail::CudaBuffer                                 num_dets_device;
    detail::CudaBuffer                                 boxes_device;
    detail::CudaBuffer                                 scores_device;
    detail::CudaBuffer                                 labels_device;
    detail::PinnedHostBuffer                           input_host;
    detail::PinnedHostBuffer                           num_dets_host;
    detail::PinnedHostBuffer                           boxes_host;
    detail::PinnedHostBuffer                           scores_host;
    detail::PinnedHostBuffer                           labels_host;
};

} // namespace kfcore::yolo
