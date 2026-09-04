#pragma once

#include <cuda_runtime_api.h>

#include <string_view>

namespace kfcore::tensorrt::detail
{

struct CudaRuntimeApi
{
    cudaError_t (*get_device)(int*);
    cudaError_t (*set_device)(int);
    cudaError_t (*get_pointer_attributes)(cudaPointerAttributes*, const void*);
};

struct DeviceCleanupActions
{
    void* state;
    void (*cleanup)(void*) noexcept;
    void (*abandon)(void*) noexcept;
};

const CudaRuntimeApi& default_cuda_runtime_api() noexcept;

class CudaDeviceScope final
{
public:
    explicit CudaDeviceScope(int requested_device);
    CudaDeviceScope(int requested_device, const CudaRuntimeApi& api);
    ~CudaDeviceScope() noexcept;

    CudaDeviceScope(const CudaDeviceScope&)            = delete;
    CudaDeviceScope& operator=(const CudaDeviceScope&) = delete;

    void restore();

private:
    const CudaRuntimeApi* api_              = nullptr;
    int                   previous_device_  = 0;
    bool                  restore_required_ = false;
};

void validate_cuda_device_pointer(const void* pointer, int expected_device,
                                  std::string_view tensor_name);
void validate_cuda_device_pointer(const void* pointer, int expected_device,
                                  std::string_view tensor_name, const CudaRuntimeApi& api);

void cleanup_on_cuda_device_or_abandon(int requested_device,
                                       const DeviceCleanupActions& actions) noexcept;
void cleanup_on_cuda_device_or_abandon(int requested_device,
                                       const DeviceCleanupActions& actions,
                                       const CudaRuntimeApi& api) noexcept;

} // namespace kfcore::tensorrt::detail
