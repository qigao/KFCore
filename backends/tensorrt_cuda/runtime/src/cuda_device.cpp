#include "cuda_device.hpp"

#include "cuda_buffer.hpp"
#include "kfcore/tensorrt/error.hpp"

#include <string>
#include <utility>

namespace kfcore::tensorrt::detail
{
namespace
{

    [[noreturn]] void throw_view(std::string message)
    {
        throw TensorRtError(TensorRtErrorCode::InvalidTensorView, std::move(message));
    }

} // namespace

const CudaRuntimeApi& default_cuda_runtime_api() noexcept
{
    static const CudaRuntimeApi api {
        cudaGetDevice,
        cudaSetDevice,
        cudaPointerGetAttributes,
    };
    return api;
}

CudaDeviceScope::CudaDeviceScope(int requested_device)
    : CudaDeviceScope(requested_device, default_cuda_runtime_api())
{
}

CudaDeviceScope::CudaDeviceScope(int requested_device, const CudaRuntimeApi& api)
    : api_(&api)
{
    check_cuda(api_->get_device(&previous_device_), "cudaGetDevice", "CUDA device scope");
    if (previous_device_ != requested_device)
    {
        check_cuda(api_->set_device(requested_device), "cudaSetDevice", "CUDA device scope");
        restore_required_ = true;
    }
}

CudaDeviceScope::~CudaDeviceScope() noexcept
{
    if (restore_required_)
    {
        (void)api_->set_device(previous_device_);
    }
}

void CudaDeviceScope::restore()
{
    if (!restore_required_)
    {
        return;
    }
    check_cuda(api_->set_device(previous_device_), "cudaSetDevice", "CUDA device restore");
    restore_required_ = false;
}

void validate_cuda_device_pointer(const void* pointer, int expected_device,
                                  std::string_view tensor_name)
{
    validate_cuda_device_pointer(pointer, expected_device, tensor_name,
                                 default_cuda_runtime_api());
}

void validate_cuda_device_pointer(const void* pointer, int expected_device,
                                  std::string_view tensor_name, const CudaRuntimeApi& api)
{
    cudaPointerAttributes attributes {};
    const cudaError_t result = api.get_pointer_attributes(&attributes, pointer);
    if (result != cudaSuccess)
    {
        const char* error_name = cudaGetErrorName(result);
        throw_view(std::string("tensor view validation stage: cudaPointerGetAttributes rejected ") +
                   std::string(tensor_name) + ": " +
                   (error_name != nullptr ? error_name : "unknown CUDA error"));
    }
    if (attributes.type != cudaMemoryTypeDevice)
    {
        throw_view("tensor view validation stage: CUDA view must reference a device allocation: " +
                   std::string(tensor_name));
    }
    if (attributes.device != expected_device)
    {
        throw_view("tensor view validation stage: CUDA view for " + std::string(tensor_name) +
                   " belongs to device " + std::to_string(attributes.device) +
                   ", expected device " + std::to_string(expected_device));
    }
}

void cleanup_on_cuda_device_or_abandon(int requested_device,
                                       const DeviceCleanupActions& actions) noexcept
{
    cleanup_on_cuda_device_or_abandon(requested_device, actions,
                                      default_cuda_runtime_api());
}

void cleanup_on_cuda_device_or_abandon(int requested_device,
                                       const DeviceCleanupActions& actions,
                                       const CudaRuntimeApi& api) noexcept
{
    try
    {
        CudaDeviceScope device_scope(requested_device, api);
        actions.cleanup(actions.state);
        device_scope.restore();
    }
    catch (...)
    {
        // Cleanup on an unconfirmed device can corrupt CUDA/TensorRT ownership. The owners must
        // instead relinquish their handles; leaking is the only noexcept-safe outcome after an
        // unrecoverable device-selection failure.
        actions.abandon(actions.state);
    }
}

} // namespace kfcore::tensorrt::detail
