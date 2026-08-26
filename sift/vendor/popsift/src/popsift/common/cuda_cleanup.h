/*
 * Copyright 2026 KFCore contributors
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>

namespace popsift::cuda
{

#if defined(KFCORE_POPSIFT_TESTING)
inline std::atomic<int> cleanup_failures_to_inject{0};
inline std::atomic<int> injected_cleanup_failures_observed{0};

inline void injectCleanupFailures(int count) noexcept
{
    cleanup_failures_to_inject.store(count, std::memory_order_release);
    injected_cleanup_failures_observed.store(0, std::memory_order_release);
}

inline int injectedCleanupFailuresObserved() noexcept
{
    return injected_cleanup_failures_observed.load(std::memory_order_acquire);
}
#endif

class CleanupStatus
{
public:
    void record(cudaError_t error, const char* operation) noexcept
    {
#if defined(KFCORE_POPSIFT_TESTING)
        int remaining = cleanup_failures_to_inject.load(std::memory_order_acquire);
        while (remaining > 0 &&
               !cleanup_failures_to_inject.compare_exchange_weak(
                   remaining, remaining - 1, std::memory_order_acq_rel))
        {
        }
        if (remaining > 0)
        {
            error = cudaErrorUnknown;
            injected_cleanup_failures_observed.fetch_add(1, std::memory_order_acq_rel);
        }
#endif

        if (_error == cudaSuccess && error != cudaSuccess)
        {
            _error = error;
            _operation = operation;
        }
    }

    void merge(const CleanupStatus& other) noexcept
    {
        if (_error == cudaSuccess && other._error != cudaSuccess)
        {
            _error = other._error;
            _operation = other._operation;
        }
    }

    bool ok() const noexcept { return _error == cudaSuccess; }

    void report() const noexcept
    {
        if (!ok())
        {
            std::fprintf(stderr, "PopSift CUDA cleanup failed during %s: %s\n",
                         _operation, cudaGetErrorString(_error));
        }
    }

private:
    cudaError_t _error{cudaSuccess};
    const char* _operation{"unknown cleanup operation"};
};

} // namespace popsift::cuda
