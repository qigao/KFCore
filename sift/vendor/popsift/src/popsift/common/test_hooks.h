/*
 * Copyright 2026 KFCore contributors
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#if defined(KFCORE_POPSIFT_TESTING)

#include <atomic>
#include <stdexcept>

namespace popsift::testing
{

enum class WorkerFailureStage
{
    None,
    UploadStartup,
    ExtractStartup,
    UploadJob,
    ExtractJob,
};

inline std::atomic<WorkerFailureStage> worker_failure_stage{WorkerFailureStage::None};

inline void injectWorkerFailure(WorkerFailureStage stage) noexcept
{
    worker_failure_stage.store(stage, std::memory_order_release);
}

inline bool workerFailurePending() noexcept
{
    return worker_failure_stage.load(std::memory_order_acquire) != WorkerFailureStage::None;
}

inline void throwIfWorkerFailureInjected(WorkerFailureStage expected)
{
    WorkerFailureStage actual = expected;
    if (worker_failure_stage.compare_exchange_strong(
            actual, WorkerFailureStage::None, std::memory_order_acq_rel))
    {
        throw std::runtime_error("injected PopSift worker failure");
    }
}

} // namespace popsift::testing

#endif
