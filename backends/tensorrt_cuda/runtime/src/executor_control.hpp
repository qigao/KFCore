#pragma once

#include "kfcore/tensorrt/error.hpp"

#include <atomic>
#include <utility>

namespace kfcore::tensorrt::detail
{

class ExecutorCallGuard final
{
public:
    explicit ExecutorCallGuard(std::atomic_flag& in_use)
        : in_use_(in_use)
    {
        if (in_use_.test_and_set(std::memory_order_acquire))
        {
            throw TensorRtError(
                TensorRtErrorCode::ConcurrentExecution,
                "execution stage: concurrent calls on one executor are unsupported");
        }
    }

    ~ExecutorCallGuard()
    {
        in_use_.clear(std::memory_order_release);
    }

    ExecutorCallGuard(const ExecutorCallGuard&)            = delete;
    ExecutorCallGuard& operator=(const ExecutorCallGuard&) = delete;

private:
    std::atomic_flag& in_use_;
};

class ExecutorStateGuard final
{
public:
    template <typename StateAccess>
    ExecutorStateGuard(std::atomic_flag& in_use, StateAccess&& state_access)
        : call_guard_(in_use)
    {
        std::forward<StateAccess>(state_access)();
    }

    ExecutorStateGuard(const ExecutorStateGuard&)            = delete;
    ExecutorStateGuard& operator=(const ExecutorStateGuard&) = delete;

private:
    ExecutorCallGuard call_guard_;
};

enum class ExecutorInvalidationReason
{
    None,
    TensorAddressCleanupFailure,
};

inline void record_tensor_address_cleanup_failure(
    ExecutorInvalidationReason& reason) noexcept
{
    reason = ExecutorInvalidationReason::TensorAddressCleanupFailure;
}

inline void validate_executor_run_state(bool engine_available, bool context_available,
                                        ExecutorInvalidationReason invalidation_reason)
{
    if (invalidation_reason == ExecutorInvalidationReason::TensorAddressCleanupFailure)
    {
        throw TensorRtError(
            TensorRtErrorCode::TensorRtFailure,
            "execution stage: executor is invalidated after tensor address cleanup stage: "
            "TensorRT rejected address reset");
    }
    if (!engine_available || !context_available)
    {
        throw TensorRtError(TensorRtErrorCode::InvalidArgument,
                            "execution stage: executor state is unavailable");
    }
}

} // namespace kfcore::tensorrt::detail
