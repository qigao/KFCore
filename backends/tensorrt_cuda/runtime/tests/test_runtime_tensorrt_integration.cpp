#include "kfcore/tensorrt/runtime.hpp"
#include "tinytest.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kfcore::tensorrt;

namespace
{

constexpr std::int64_t kBatch             = 1;
constexpr std::int64_t kInputChannels     = 3;
constexpr std::int64_t kArcFaceInputSize  = 112;
constexpr std::int64_t kEmbeddingLength   = 512;
constexpr float        kAbsoluteTolerance = 1.0e-5F;
constexpr float        kRelativeTolerance = 1.0e-5F;

std::filesystem::path required_engine_path()
{
    constexpr char kVariable[] = "KFCORE_RUNTIME_TENSORRT_TEST_ENGINE_ARCFACE";
    const char* value = std::getenv(kVariable);
    if (value == nullptr || *value == '\0')
    {
        throw std::runtime_error(std::string(kVariable) +
                                 " must name a trusted ArcFace TensorRT engine");
    }
    return std::filesystem::path(value);
}

[[noreturn]] void throw_cuda(cudaError_t error, const char* operation)
{
    throw std::runtime_error(std::string("integration CUDA stage: ") + operation +
                             " failed: " + cudaGetErrorString(error));
}

void check_cuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess)
    {
        throw_cuda(error, operation);
    }
}

int current_cuda_device()
{
    int device = -1;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    return device;
}

class ZeroedDeviceBuffer final
{
public:
    explicit ZeroedDeviceBuffer(std::size_t byte_size)
        : byte_size_(byte_size)
        , device_(current_cuda_device())
    {
        check_cuda(cudaMalloc(&data_, byte_size_), "cudaMalloc");
        try
        {
            check_cuda(cudaMemsetAsync(data_, 0, byte_size_, nullptr),
                       "cudaMemsetAsync");
            check_cuda(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize");
        }
        catch (...)
        {
            (void)cudaFree(data_);
            data_ = nullptr;
            throw;
        }
    }

    ~ZeroedDeviceBuffer()
    {
        if (data_ == nullptr)
        {
            return;
        }

        int caller_device = device_;
        if (cudaGetDevice(&caller_device) == cudaSuccess &&
            cudaSetDevice(device_) == cudaSuccess)
        {
            (void)cudaFree(data_);
            (void)cudaSetDevice(caller_device);
        }
    }

    ZeroedDeviceBuffer(const ZeroedDeviceBuffer&)            = delete;
    ZeroedDeviceBuffer& operator=(const ZeroedDeviceBuffer&) = delete;

    void* data() const noexcept
    {
        return data_;
    }

    std::size_t byte_size() const noexcept
    {
        return byte_size_;
    }

private:
    void*       data_      = nullptr;
    std::size_t byte_size_ = 0;
    int         device_    = 0;
};

const TensorDescriptor& exactly_one_tensor(const std::vector<TensorDescriptor>& tensors,
                                           TensorIoMode mode)
{
    const TensorDescriptor* found = nullptr;
    std::size_t count = 0;
    for (const TensorDescriptor& tensor : tensors)
    {
        if (tensor.mode == mode)
        {
            found = &tensor;
            ++count;
        }
    }
    if (count != 1 || found == nullptr)
    {
        throw std::runtime_error("trusted ArcFace engine must expose exactly one input and output");
    }
    return *found;
}

void check_finite(const std::vector<float>& values)
{
    check(values.size() == static_cast<std::size_t>(kEmbeddingLength));
    for (float value : values)
    {
        check_true(std::isfinite(value));
    }
}

void print_summary(const char* source, const std::vector<float>& values)
{
    const auto range = std::minmax_element(values.begin(), values.end());
    double sum = 0.0;
    for (float value : values)
    {
        sum += static_cast<double>(value);
    }
    std::cout << "[integration] ArcFace generic " << source << " output_count="
              << values.size() << " min=" << *range.first << " max=" << *range.second
              << " sum=" << sum << '\n';
}

} // namespace

spec("TensorRT runtime real-engine integration")
{
    it("executes a trusted ArcFace engine from Host and direct CUDA-device zero inputs")
    {
        const int caller_device = current_cuda_device();
        auto engine = Engine::load(required_engine_path());
        check(current_cuda_device() == caller_device);

        const TensorDescriptor& input =
            exactly_one_tensor(engine->tensors(), TensorIoMode::Input);
        const TensorDescriptor& output =
            exactly_one_tensor(engine->tensors(), TensorIoMode::Output);
        check(input.data_type == DataType::Float32);
        check(output.data_type == DataType::Float32);

        auto executor = engine->create_executor();
        check(current_cuda_device() == caller_device);

        const TensorShape input_shape =
            { kBatch, kInputChannels, kArcFaceInputSize, kArcFaceInputSize };
        const TensorShape output_shape = { kBatch, kEmbeddingLength };
        std::vector<float> host_input(static_cast<std::size_t>(kBatch * kInputChannels *
                                                               kArcFaceInputSize *
                                                               kArcFaceInputSize),
                                      0.0F);
        std::vector<float> host_output(static_cast<std::size_t>(kEmbeddingLength),
                                       std::numeric_limits<float>::quiet_NaN());
        executor->run({ { input.name, DataType::Float32, input_shape, host_input.data(),
                          host_input.size() * sizeof(float), MemoryKind::Host } },
                      { { output.name, DataType::Float32, output_shape, host_output.data(),
                          host_output.size() * sizeof(float), MemoryKind::Host } });
        check(current_cuda_device() == caller_device);
        check_finite(host_output);
        print_summary("Host", host_output);

        std::vector<float> device_output(static_cast<std::size_t>(kEmbeddingLength),
                                         std::numeric_limits<float>::quiet_NaN());
        float max_absolute_difference = 0.0F;
        float max_reference_magnitude = 0.0F;
        {
            ZeroedDeviceBuffer device_input(host_input.size() * sizeof(float));
            check(current_cuda_device() == caller_device);
            executor->run({ { input.name, DataType::Float32, input_shape, device_input.data(),
                              device_input.byte_size(), MemoryKind::CudaDevice } },
                          { { output.name, DataType::Float32, output_shape,
                              device_output.data(), device_output.size() * sizeof(float),
                              MemoryKind::Host } });
            check(current_cuda_device() == caller_device);
            check_finite(device_output);

            for (std::size_t index = 0; index < host_output.size(); ++index)
            {
                max_absolute_difference =
                    (std::max)(max_absolute_difference,
                               std::fabs(device_output[index] - host_output[index]));
                max_reference_magnitude =
                    (std::max)(max_reference_magnitude, std::fabs(host_output[index]));
            }
        }
        check(current_cuda_device() == caller_device);
        check_true(max_absolute_difference <=
                   kAbsoluteTolerance + kRelativeTolerance * max_reference_magnitude);
        print_summary("CudaDevice", device_output);
        std::cout << "[integration] ArcFace generic max_abs_difference="
                  << max_absolute_difference << '\n';

        executor.reset();
        engine.reset();
        check(current_cuda_device() == caller_device);
    }
}
