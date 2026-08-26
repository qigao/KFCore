/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace popsift {

/**
 * A bounded blocking queue for trivially copyable pipeline handles.
 *
 * Storage is allocated once by the constructor. A producer blocks while the
 * queue is full. Closing wakes all waiters, rejects new values, and still lets
 * consumers drain values that were accepted before close.
 */
template<typename T>
class SyncQueue {
  static_assert(std::is_trivially_copyable<T>::value,
                "PopSift SyncQueue stores only trivially copyable handles");

public:
  class Reservation {
  public:
    Reservation() noexcept = default;
    Reservation(const Reservation&) = delete;
    Reservation& operator=(const Reservation&) = delete;

    Reservation(Reservation&& other) noexcept
        : queue_(other.queue_) {
      other.queue_ = nullptr;
    }

    Reservation& operator=(Reservation&& other) noexcept {
      if (this != &other) {
        reset();
        queue_ = other.queue_;
        other.queue_ = nullptr;
      }
      return *this;
    }

    ~Reservation() {
      reset();
    }

    explicit operator bool() const noexcept {
      return queue_ != nullptr;
    }

    bool commit(const T& value) {
      if (queue_ == nullptr) {
        return false;
      }
      SyncQueue* queue = queue_;
      queue_ = nullptr;
      return queue->commitReserved(value);
    }

  private:
    friend class SyncQueue;

    explicit Reservation(SyncQueue* queue) noexcept
        : queue_(queue) {
    }

    void reset() noexcept {
      if (queue_ != nullptr) {
        SyncQueue* queue = queue_;
        queue_ = nullptr;
        queue->cancelReserved();
      }
    }

    SyncQueue* queue_ = nullptr;
  };

  explicit SyncQueue(std::size_t capacity)
      : items_(capacity) {
    if (capacity == 0) {
      throw std::invalid_argument("PopSift SyncQueue capacity must be positive");
    }
  }

  SyncQueue(const SyncQueue&) = delete;
  SyncQueue& operator=(const SyncQueue&) = delete;

  /** Reserve capacity before a producer allocates its payload. */
  Reservation reserve() {
    std::unique_lock<std::mutex> lock(mtx_);
    not_full_.wait(lock, [this] {
      return count_ + reservations_ < items_.size() || closed_;
    });
    if (closed_) {
      return Reservation();
    }
    ++reservations_;
    return Reservation(this);
  }

  /** Block until space is available or the queue is closed. */
  bool push(const T& value) {
    Reservation reservation = reserve();
    return reservation && reservation.commit(value);
  }

  /** Pull one value, or return false after a closed queue has been drained. */
  bool pull(T& value) {
    std::unique_lock<std::mutex> lock(mtx_);
    not_empty_.wait(lock, [this] { return count_ != 0 || closed_; });
    if (count_ == 0) {
      return false;
    }

    value = items_[head_];
    head_ = (head_ + 1) % items_.size();
    --count_;
    lock.unlock();
    not_full_.notify_one();
    return true;
  }

  /** Stop accepting values and wake every blocked producer and consumer. */
  void close() {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      closed_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  bool empty() {
    std::lock_guard<std::mutex> lock(mtx_);
    return count_ == 0;
  }

private:
  bool commitReserved(const T& value) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (reservations_ == 0) {
      std::terminate();
    }
    --reservations_;
    if (closed_) {
      lock.unlock();
      not_full_.notify_one();
      return false;
    }

    items_[tail_] = value;
    tail_ = (tail_ + 1) % items_.size();
    ++count_;
    lock.unlock();
    not_empty_.notify_one();
    return true;
  }

  void cancelReserved() noexcept {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      if (reservations_ == 0) {
        std::terminate();
      }
      --reservations_;
    }
    not_full_.notify_one();
  }

  std::mutex mtx_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  std::vector<T> items_;
  std::size_t head_ = 0;
  std::size_t tail_ = 0;
  std::size_t count_ = 0;
  std::size_t reservations_ = 0;
  bool closed_ = false;
};

}  // namespace popsift
