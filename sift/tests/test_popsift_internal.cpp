#include <popsift/common/cuda_cleanup.h>
#include <popsift/common/sync_queue.h>
#include <popsift/common/test_hooks.h>
#include <popsift/features.h>
#include <popsift/popsift.h>
#include <popsift/scale_geometry.h>
#include <popsift/s_image.h>

#include <cuda_runtime.h>

#include <chrono>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>

#include "tinytest.hpp"

using namespace std::chrono_literals;

spec("PopSift internal bounded queue")
{
    it("preserves FIFO order and drains values after close")
    {
        popsift::SyncQueue<int> queue(2);
        check(queue.push(11));
        check(queue.push(22));

        queue.close();
        check(!queue.push(33));

        int value = 0;
        check(queue.pull(value));
        check(value == 11);
        check(queue.pull(value));
        check(value == 22);
        check(!queue.pull(value));
        check(queue.empty());
    }

    it("rejects zero capacity")
    {
        bool threw = false;
        try
        {
            popsift::SyncQueue<int> queue(0);
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }
        check(threw);
    }

    it("blocks a producer at capacity and wakes it after a pull")
    {
        popsift::SyncQueue<int> queue(1);
        check(queue.push(1));

        std::promise<void> entered;
        std::future<void> entered_future = entered.get_future();
        std::promise<bool> pushed;
        std::future<bool> pushed_future = pushed.get_future();
        std::thread producer([&] {
            entered.set_value();
            pushed.set_value(queue.push(2));
        });

        entered_future.get();
        const bool observed_backpressure = pushed_future.wait_for(50ms) ==
                                           std::future_status::timeout;

        int first = 0;
        const bool first_pulled = queue.pull(first);
        const bool producer_woke = pushed_future.wait_for(2s) ==
                                   std::future_status::ready;
        if (!producer_woke)
        {
            queue.close();
        }
        const bool second_pushed = producer_woke && pushed_future.get();
        producer.join();

        int second = 0;
        const bool second_pulled = queue.pull(second);
        queue.close();

        check(observed_backpressure);
        check(first_pulled);
        check(first == 1);
        check(producer_woke);
        check(second_pushed);
        check(second_pulled);
        check(second == 2);
    }

    it("wakes a blocked consumer when closed")
    {
        popsift::SyncQueue<int> queue(1);
        std::promise<void> entered;
        std::future<void> entered_future = entered.get_future();
        std::promise<bool> pulled;
        std::future<bool> pulled_future = pulled.get_future();
        std::thread consumer([&] {
            entered.set_value();
            int value = 0;
            pulled.set_value(queue.pull(value));
        });

        entered_future.get();
        queue.close();
        const bool consumer_woke = pulled_future.wait_for(2s) ==
                                   std::future_status::ready;
        const bool pull_succeeded = consumer_woke && pulled_future.get();
        consumer.join();

        check(consumer_woke);
        check(!pull_succeeded);
    }

    it("wakes a blocked producer when closed")
    {
        popsift::SyncQueue<int> queue(1);
        check(queue.push(1));

        std::promise<void> entered;
        std::future<void> entered_future = entered.get_future();
        std::promise<bool> pushed;
        std::future<bool> pushed_future = pushed.get_future();
        std::thread producer([&] {
            entered.set_value();
            pushed.set_value(queue.push(2));
        });

        entered_future.get();
        const bool observed_backpressure = pushed_future.wait_for(50ms) ==
                                           std::future_status::timeout;
        queue.close();
        const bool producer_woke = pushed_future.wait_for(2s) ==
                                   std::future_status::ready;
        const bool push_succeeded = producer_woke && pushed_future.get();
        producer.join();

        int accepted_value = 0;
        check(queue.pull(accepted_value));
        check(!queue.pull(accepted_value));
        check(observed_backpressure);
        check(producer_woke);
        check(!push_succeeded);
        check(accepted_value == 1);
    }

    it("reserves capacity before a producer allocates its payload")
    {
        popsift::SyncQueue<int> queue(1);
        auto reservation = queue.reserve();
        check(static_cast<bool>(reservation));

        std::promise<bool> pushed;
        std::future<bool> pushed_future = pushed.get_future();
        std::thread producer([&] { pushed.set_value(queue.push(2)); });

        const bool blocked_by_reservation = pushed_future.wait_for(50ms) ==
                                            std::future_status::timeout;
        const bool committed = reservation.commit(1);
        const bool still_blocked_while_full = pushed_future.wait_for(50ms) ==
                                              std::future_status::timeout;

        int first = 0;
        const bool first_pulled = queue.pull(first);
        const bool producer_woke = pushed_future.wait_for(2s) ==
                                   std::future_status::ready;
        if (!producer_woke)
        {
            queue.close();
        }
        const bool second_pushed = producer_woke && pushed_future.get();
        producer.join();

        int second = 0;
        const bool second_pulled = queue.pull(second);
        queue.close();

        check(blocked_by_reservation);
        check(committed);
        check(still_blocked_while_full);
        check(first_pulled);
        check(first == 1);
        check(producer_woke);
        check(second_pushed);
        check(second_pulled);
        check(second == 2);
    }

    it("rejects commit after close and releases the reserved slot")
    {
        popsift::SyncQueue<int> queue(1);
        auto reservation = queue.reserve();
        check(static_cast<bool>(reservation));

        queue.close();
        check(!reservation.commit(7));

        int value = 0;
        check(!queue.pull(value));
        check(queue.empty());
    }

    it("wakes a blocked producer when a reservation is cancelled")
    {
        popsift::SyncQueue<int> queue(1);
        std::promise<bool> pushed;
        std::future<bool> pushed_future = pushed.get_future();
        std::thread producer;

        {
            auto reservation = queue.reserve();
            check(static_cast<bool>(reservation));
            producer = std::thread([&] { pushed.set_value(queue.push(9)); });
            check(pushed_future.wait_for(50ms) == std::future_status::timeout);
        }

        const bool producer_woke = pushed_future.wait_for(2s) ==
                                   std::future_status::ready;
        if (!producer_woke)
        {
            queue.close();
        }
        const bool push_succeeded = producer_woke && pushed_future.get();
        producer.join();

        int value = 0;
        check(producer_woke);
        check(push_succeeded);
        check(queue.pull(value));
        check(value == 9);
        queue.close();
    }
}

spec("PopSift image geometry")
{
    it("computes automatic octaves without mutating shared configuration")
    {
        popsift::Config config;
        const int original_octaves = config.octaves;

        const popsift::ScaledImageGeometry small =
            popsift::scaleImageGeometry(config, 64, 48);
        const popsift::ScaledImageGeometry large =
            popsift::scaleImageGeometry(config, 640, 480);

        check(config.octaves == original_octaves);
        check(small.octaves > 0);
        check(large.octaves > small.octaves);
        check(small.width == 128);
        check(small.height == 96);
    }
}

spec("PopSift CUDA cleanup")
{
    it("keeps image destruction noexcept and attempts every cleanup after an error")
    {
        static_assert(std::is_nothrow_destructible_v<popsift::Image>);
        static_assert(std::is_nothrow_destructible_v<popsift::ImageFloat>);

        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0)
        {
            popsift::cuda::injectCleanupFailures(3);
            {
                popsift::Image image(8, 8);
            }
            check(popsift::cuda::injectedCleanupFailuresObserved() == 3);
            popsift::cuda::injectCleanupFailures(0);
        }
        else
        {
            check(true);
        }
    }

    it("keeps worker shutdown ordered when CUDA cleanup reports failures")
    {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0)
        {
            constexpr int kWidth = 64;
            constexpr int kHeight = 64;
            constexpr int kInjectedFailures = 5;
            std::vector<unsigned char> pixels(
                static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(kHeight), 127);

            {
                popsift::Config config;
                PopSift backend(config, popsift::Config::ExtractingMode,
                                PopSift::ByteImages, 0, 1);
                std::unique_ptr<SiftJob> job(backend.enqueue(kWidth, kHeight, pixels.data()));
                std::unique_ptr<popsift::FeaturesHost> features(job->getHost());
                check(features != nullptr);
                popsift::cuda::injectCleanupFailures(kInjectedFailures);
            }

            check(popsift::cuda::injectedCleanupFailuresObserved() == kInjectedFailures);
            popsift::cuda::injectCleanupFailures(0);
        }
        else
        {
            check(true);
        }
    }
}

spec("PopSift worker failure protocol")
{
    it("completes accepted jobs and joins after injected worker failures")
    {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0)
        {
            constexpr int kWidth = 64;
            constexpr int kHeight = 64;
            std::vector<unsigned char> pixels(
                static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(kHeight), 127);
            const auto run_failure = [&](popsift::testing::WorkerFailureStage stage,
                                         bool worker_should_continue) {
                popsift::testing::injectWorkerFailure(stage);
                popsift::Config config;
                PopSift backend(config, popsift::Config::ExtractingMode,
                                PopSift::ByteImages, 0, 2);

                std::unique_ptr<SiftJob> job;
                bool rejected = false;
                try
                {
                    job.reset(backend.enqueue(kWidth, kHeight, pixels.data()));
                }
                catch (const std::runtime_error&)
                {
                    rejected = true;
                }

                bool ready = false;
                bool failed = false;
                if (job != nullptr)
                {
                    std::future<bool> result = std::async(std::launch::async, [&] {
                        try
                        {
                            std::unique_ptr<popsift::FeaturesHost> features(job->getHost());
                            return false;
                        }
                        catch (const std::runtime_error&)
                        {
                            return true;
                        }
                    });
                    ready = result.wait_for(2s) == std::future_status::ready;
                    if (!ready)
                    {
                        backend.uninit();
                    }
                    failed = ready && result.get();
                }

                check(!popsift::testing::workerFailurePending());
                check(rejected || (ready && failed));

                if (worker_should_continue && ready)
                {
                    std::unique_ptr<SiftJob> recovery(
                        backend.enqueue(kWidth, kHeight, pixels.data()));
                    std::unique_ptr<popsift::FeaturesHost> features(recovery->getHost());
                    check(features != nullptr);
                }
            };

            run_failure(popsift::testing::WorkerFailureStage::UploadStartup, false);
            run_failure(popsift::testing::WorkerFailureStage::ExtractStartup, false);
            run_failure(popsift::testing::WorkerFailureStage::UploadJob, true);
            run_failure(popsift::testing::WorkerFailureStage::ExtractJob, true);
        }
        else
        {
            check(true);
        }
    }
}

spec("PopSift internal job completion")
{
    it("propagates worker errors through the host result future")
    {
        const unsigned char pixel = 0;
        SiftJob job(1, 1, &pixel);
        job.setError(std::make_exception_ptr(std::runtime_error("host worker failed")));

        bool threw = false;
        try
        {
            (void)job.getHost();
        }
        catch (const std::runtime_error& error)
        {
            threw = std::string(error.what()) == "host worker failed";
        }
        check(threw);
    }

    it("propagates worker errors through the device result future")
    {
        const unsigned char pixel = 0;
        SiftJob job(1, 1, &pixel);
        job.setError(std::make_exception_ptr(std::runtime_error("device worker failed")));

        bool threw = false;
        try
        {
            (void)job.getDev();
        }
        catch (const std::runtime_error& error)
        {
            threw = std::string(error.what()) == "device worker failed";
        }
        check(threw);
    }
}
