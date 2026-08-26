#include <popsift/common/sync_queue.h>
#include <popsift/popsift.h>

#include <chrono>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>

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
        const bool second_pushed = pushed_future.get();
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
        const bool pull_succeeded = pulled_future.get();
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
        const bool push_succeeded = pushed_future.get();
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
        const bool second_pushed = pushed_future.get();
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
