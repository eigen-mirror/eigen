// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Dmitry Vyukov <dvyukov@google.com>
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_USE_THREADS
#include "main.h"
#include <Eigen/ThreadPool>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

// Visual studio doesn't implement a rand_r() function since its
// implementation of rand() is already thread safe
int rand_reentrant(unsigned int* s) {
#if EIGEN_COMP_MSVC_STRICT
  EIGEN_UNUSED_VARIABLE(s);
  return rand();
#else
  return rand_r(s);
#endif
}

static void test_basic_eventcount() {
  MaxSizeVector<EventCount::Waiter> waiters(1);
  waiters.resize(1);
  EventCount ec(waiters);
  EventCount::Waiter& w = waiters[0];
  ec.Notify(false);
  ec.Prewait();
  ec.Notify(true);
  ec.CommitWait(&w);
  ec.Prewait();
  ec.CancelWait();
}

static void test_cancel_wait_signal_forwarding() {
  MaxSizeVector<EventCount::Waiter> waiters(2);
  waiters.resize(2);
  EventCount ec(waiters);

  std::atomic<bool> t0_started(false);
  std::atomic<bool> t0_done(false);

  std::unique_ptr<std::thread> t0(new std::thread([&]() {
    ec.Prewait();
    t0_started.store(true);
    ec.CommitWait(&waiters[0]);
    t0_done.store(true);
  }));

  while (!t0_started.load()) {
    std::this_thread::yield();
  }
  // Brief delay to allow t0 to enter CommitWait and park on stack.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Thread 1 calls Prewait, so waiters count becomes 1 (with t0 on the waiter stack).
  ec.Prewait();
  // Main thread signals a waiter.
  // Because waiters count is 1 and signals count is 0 (signals < waiters),
  // Notify(false) increments signals count rather than unparking t0 from the stack.
  ec.Notify(false);
  // Thread 1 cancels its wait.
  // Without the fix, CancelWait decrements signals count to maintain signals <= waiters,
  // discarding the notification and leaving t0 parked on the stack forever.
  // With the fix, CancelWait calls Notify(false) to forward the discarded signal to t0.
  ec.CancelWait();

  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!t0_done.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  const bool done = t0_done.load();
  if (!done) {
    // Unblock t0 if test failed so the process doesn't hang.
    ec.Notify(false);
  }
  t0->join();
  VERIFY(done);
}

// Fake bounded counter-based queue.
struct TestQueue {
  std::atomic<int> val_;
  static const int kQueueSize = 10;

  TestQueue() : val_() {}

  ~TestQueue() { VERIFY_IS_EQUAL(val_.load(), 0); }

  bool Push() {
    int val = val_.load(std::memory_order_relaxed);
    for (;;) {
      VERIFY_GE(val, 0);
      VERIFY_LE(val, kQueueSize);
      if (val == kQueueSize) return false;
      if (val_.compare_exchange_weak(val, val + 1, std::memory_order_relaxed)) return true;
    }
  }

  bool Pop() {
    int val = val_.load(std::memory_order_relaxed);
    for (;;) {
      VERIFY_GE(val, 0);
      VERIFY_LE(val, kQueueSize);
      if (val == 0) return false;
      if (val_.compare_exchange_weak(val, val - 1, std::memory_order_relaxed)) return true;
    }
  }

  bool Empty() { return val_.load(std::memory_order_relaxed) == 0; }
};

const int TestQueue::kQueueSize;

// A number of producers send messages to a set of consumers using a set of
// fake queues. Ensure that it does not crash, consumers don't deadlock and
// number of blocked and unblocked threads match.
static void test_stress_eventcount() {
  const int kThreads = (std::min)(static_cast<int>(std::thread::hardware_concurrency()), 16);
  static const int kEvents = 1 << 16;
  static const int kQueues = 10;

  MaxSizeVector<EventCount::Waiter> waiters(kThreads);
  waiters.resize(kThreads);
  EventCount ec(waiters);
  TestQueue queues[kQueues];

  std::vector<std::unique_ptr<std::thread>> producers;
  for (int i = 0; i < kThreads; i++) {
    producers.emplace_back(new std::thread([&ec, &queues]() {
      unsigned int rnd = static_cast<unsigned int>(std::hash<std::thread::id>()(std::this_thread::get_id()));
      for (int j = 0; j < kEvents; j++) {
        unsigned idx = rand_reentrant(&rnd) % kQueues;
        if (queues[idx].Push()) {
          ec.Notify(false);
          continue;
        }
        EIGEN_THREAD_YIELD();
        j--;
      }
    }));
  }

  std::vector<std::unique_ptr<std::thread>> consumers;
  for (int i = 0; i < kThreads; i++) {
    consumers.emplace_back(new std::thread([&ec, &queues, &waiters, i]() {
      EventCount::Waiter& w = waiters[i];
      unsigned int rnd = static_cast<unsigned int>(std::hash<std::thread::id>()(std::this_thread::get_id()));
      for (int j = 0; j < kEvents; j++) {
        unsigned idx = rand_reentrant(&rnd) % kQueues;
        if (queues[idx].Pop()) continue;
        j--;
        ec.Prewait();
        bool empty = true;
        for (int q = 0; q < kQueues; q++) {
          if (!queues[q].Empty()) {
            empty = false;
            break;
          }
        }
        if (!empty) {
          ec.CancelWait();
          continue;
        }
        ec.CommitWait(&w);
      }
    }));
  }

  for (int i = 0; i < kThreads; i++) {
    producers[i]->join();
    consumers[i]->join();
  }
}

EIGEN_DECLARE_TEST(threads_eventcount) {
  CALL_SUBTEST(test_basic_eventcount());
  CALL_SUBTEST(test_cancel_wait_signal_forwarding());
  CALL_SUBTEST(test_stress_eventcount());
}
