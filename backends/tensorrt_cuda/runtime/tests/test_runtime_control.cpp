#include "cuda_device.hpp"
#include "executor_control.hpp"
#include "shared_lifetime.hpp"
#include "tensorrt_version.hpp"

#include "kfcore/tensorrt/error.hpp"
#include "kfcore/tensorrt/runtime.hpp"
#include "tinytest.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>

using namespace kfcore::tensorrt;
using namespace kfcore::tensorrt::detail;

static_assert(std::is_copy_constructible_v<Engine>, "Engine must remain copy constructible");
static_assert(std::is_copy_assignable_v<Engine>, "Engine must remain copy assignable");

namespace
{

struct FakeCudaState
{
    int            current_device = 1;
    int            set_calls      = 0;
    int            cleanup_calls  = 0;
    int            abandon_calls  = 0;
    cudaMemoryType pointer_type   = cudaMemoryTypeDevice;
    int            pointer_device = 2;
    cudaError_t    get_result     = cudaSuccess;
    cudaError_t    query_result   = cudaSuccess;
};

struct LifetimeProbe
{
    explicit LifetimeProbe(int& destructions)
        : destructions(&destructions)
    {
    }

    ~LifetimeProbe()
    {
        ++*destructions;
    }

    int* destructions;
};

FakeCudaState state;

cudaError_t fake_get_device(int* device)
{
    if (state.get_result != cudaSuccess)
    {
        return state.get_result;
    }
    *device = state.current_device;
    return cudaSuccess;
}

cudaError_t fake_set_device(int device)
{
    state.current_device = device;
    ++state.set_calls;
    return cudaSuccess;
}

cudaError_t fake_get_pointer_attributes(cudaPointerAttributes* attributes, const void*)
{
    attributes->type   = state.pointer_type;
    attributes->device = state.pointer_device;
    return state.query_result;
}

const CudaRuntimeApi fake_api {
    fake_get_device,
    fake_set_device,
    fake_get_pointer_attributes,
};

void fake_cleanup(void* opaque) noexcept
{
    auto& fake = *static_cast<FakeCudaState*>(opaque);
    ++fake.cleanup_calls;
}

void fake_abandon(void* opaque) noexcept
{
    auto& fake = *static_cast<FakeCudaState*>(opaque);
    ++fake.abandon_calls;
}

void check_view_error(const std::function<void()>& operation, const char* message)
{
    bool threw = false;
    try
    {
        operation();
    }
    catch (const TensorRtError& error)
    {
        threw = true;
        check(error.code() == TensorRtErrorCode::InvalidTensorView);
        check(std::string(error.what()).find(message) != std::string::npos);
    }
    check_true(threw);
}

} // namespace

spec("TensorRT runtime CUDA control")
{
    before_each()
    {
        state = {};
    }

    it("restores the caller CUDA device after normal completion")
    {
        CudaDeviceScope scope(2, fake_api);
        check(state.current_device == 2);

        scope.restore();

        check(state.current_device == 1);
        check(state.set_calls == 2);
    }

    it("restores the caller CUDA device while unwinding an exception")
    {
        try
        {
            CudaDeviceScope scope(2, fake_api);
            throw std::runtime_error("stop");
        }
        catch (const std::runtime_error&)
        {
        }

        check(state.current_device == 1);
        check(state.set_calls == 2);
    }

    it("abandons resources without cleanup when target device scope cannot be established")
    {
        state.get_result = cudaErrorInvalidDevice;
        const DeviceCleanupActions actions { &state, fake_cleanup, fake_abandon };

        cleanup_on_cuda_device_or_abandon(2, actions, fake_api);

        check(state.cleanup_calls == 0);
        check(state.abandon_calls == 1);
        check(state.set_calls == 0);
    }

    it("copies lifetime anchors by retaining one immutable state")
    {
        int destructions = 0;
        std::shared_ptr<const LifetimeProbe> state_owner =
            std::make_shared<const LifetimeProbe>(destructions);
        auto first = make_shared_lifetime_anchor(state_owner);
        auto copy  = copy_shared_lifetime_anchor(first);

        check(first->get() == copy->get());
        state_owner.reset();
        first.reset();
        check(destructions == 0);

        copy.reset();
        check(destructions == 1);
    }

    it("accepts only a device allocation on the configured device")
    {
        std::byte storage {};
        validate_cuda_device_pointer(&storage, 2, "input", fake_api);

        state.pointer_device = 3;
        check_view_error([&] { validate_cuda_device_pointer(&storage, 2, "input", fake_api); },
                         "device 3");
    }

    it("rejects host and managed pointers for CUDA device views")
    {
        std::byte storage {};
        state.pointer_type = cudaMemoryTypeHost;
        check_view_error([&] { validate_cuda_device_pointer(&storage, 2, "input", fake_api); },
                         "device allocation");

        state.pointer_type = cudaMemoryTypeManaged;
        check_view_error([&] { validate_cuda_device_pointer(&storage, 2, "input", fake_api); },
                         "device allocation");
    }

    it("classifies a rejected CUDA pointer query as an invalid view")
    {
        std::byte storage {};
        state.query_result = cudaErrorInvalidValue;
        check_view_error([&] { validate_cuda_device_pointer(&storage, 2, "input", fake_api); },
                         "cudaPointerGetAttributes");
    }

    it("guards mutable executor state before invoking its state access")
    {
        std::atomic_flag in_use = ATOMIC_FLAG_INIT;
        int              state_accesses = 0;
        bool             rejected = false;

        {
            ExecutorStateGuard first(in_use, [&] { ++state_accesses; });
            try
            {
                ExecutorStateGuard overlapping(in_use, [&] { ++state_accesses; });
            }
            catch (const TensorRtError& error)
            {
                rejected = true;
                check(error.code() == TensorRtErrorCode::ConcurrentExecution);
            }
            check(state_accesses == 1);
        }

        ExecutorStateGuard after_completion(in_use, [&] { ++state_accesses; });
        check_true(rejected);
        check(state_accesses == 2);
    }

    it("classifies an address-cleanup-invalidated executor as a TensorRT failure")
    {
        ExecutorInvalidationReason reason = ExecutorInvalidationReason::None;
        record_tensor_address_cleanup_failure(reason);

        bool threw = false;
        try
        {
            validate_executor_run_state(true, false, reason);
        }
        catch (const TensorRtError& error)
        {
            threw = true;
            const std::string message(error.what());
            check(error.code() == TensorRtErrorCode::TensorRtFailure);
            check(message.find("invalidated") != std::string::npos);
            check(message.find("tensor address cleanup stage") != std::string::npos);
            check(message.find("address reset") != std::string::npos);
        }
        check_true(threw);
    }

    it("retains invalid argument for an unavailable executor that was not poisoned")
    {
        bool threw = false;
        try
        {
            validate_executor_run_state(true, false, ExecutorInvalidationReason::None);
        }
        catch (const TensorRtError& error)
        {
            threw = true;
            check(error.code() == TensorRtErrorCode::InvalidArgument);
        }
        check_true(threw);
    }

    it("selects a fail-closed alias policy across supported TensorRT versions")
    {
        struct VersionCase
        {
            int                     major;
            int                     minor;
            TensorRtAliasPolicy     expected;
        };
        const std::array<VersionCase, 6> cases = {
            VersionCase { 10, 0, TensorRtAliasPolicy::NoAliasFeature },
            VersionCase { 10, 2, TensorRtAliasPolicy::NoAliasFeature },
            VersionCase { 10, 3, TensorRtAliasPolicy::RejectUnavailableQuery },
            VersionCase { 10, 10, TensorRtAliasPolicy::RejectUnavailableQuery },
            VersionCase { 10, 11, TensorRtAliasPolicy::QueryEngine },
            VersionCase { 11, 0, TensorRtAliasPolicy::QueryEngine },
        };

        for (const VersionCase& version : cases)
        {
            check(tensorrt_alias_policy(version.major, version.minor) == version.expected);
        }
    }

    it("accepts only the explicitly supported TensorRT release families")
    {
        struct VersionCase
        {
            int  major;
            int  minor;
            bool supported;
        };
        const std::array<VersionCase, 6> cases = {
            VersionCase { 8, 5, false }, VersionCase { 8, 6, true },
            VersionCase { 9, 0, false }, VersionCase { 10, 0, true },
            VersionCase { 11, 0, true }, VersionCase { 12, 0, false },
        };

        for (const VersionCase& version : cases)
        {
            check(tensorrt_version_supported(version.major, version.minor) == version.supported);
        }
    }

    it("rejects an alias-capable TensorRT version before engine loading when query is unavailable")
    {
        validate_runtime_tensorrt_version(8, 6);
        validate_runtime_tensorrt_version(10, 0);
        validate_runtime_tensorrt_version(10, 2);
        validate_runtime_tensorrt_version(10, 11);
        validate_runtime_tensorrt_version(11, 4);

        bool threw = false;
        try
        {
            validate_runtime_tensorrt_version(10, 3);
        }
        catch (const TensorRtError& error)
        {
            threw = true;
            const std::string message(error.what());
            check(error.code() == TensorRtErrorCode::EngineContractMismatch);
            check(message.find("version gate stage") != std::string::npos);
            check(message.find("10.3 through 10.10") != std::string::npos);
        }
        check_true(threw);
    }

    it("rejects an unsupported TensorRT version before engine loading")
    {
        bool threw = false;
        try
        {
            validate_runtime_tensorrt_version(8, 5);
        }
        catch (const TensorRtError& error)
        {
            threw = true;
            const std::string message(error.what());
            check(error.code() == TensorRtErrorCode::EngineContractMismatch);
            check(message.find("version gate stage") != std::string::npos);
            check(message.find("8.6, 10.x, or 11.x") != std::string::npos);
        }
        check_true(threw);
    }
}
