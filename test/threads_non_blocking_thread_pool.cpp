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
#include "Eigen/ThreadPool"

// Spin-wait with yielding until the condition is met, or fail after a timeout.
// Returns true if the condition was met before the deadline.
template <typename Cond>
static bool spin_wait(Cond cond, int timeout_seconds = 60) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
  while (!cond()) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::yield();
  }
  return true;
}

static void test_create_destroy_empty_pool() {
  // Just create and destroy the pool. This will wind up and tear down worker
  // threads. Ensure there are no issues in that logic.
  for (int i = 0; i < 16; ++i) {
    ThreadPool tp(i);
  }
}

static thread_local bool delaying_environment_worker = false;

struct DelayingThreadEnvironment {
  using EnvThread = StlThreadEnvironment::EnvThread;

  struct Task {
    std::function<void()> f;
    Task() = default;
    Task(Task&&) = default;
    Task& operator=(Task&& other) {
      f = std::move(other.f);
      // Widen the interval between taking a task and retiring the worker's spinner.
      if (f && delaying_environment_worker) std::this_thread::sleep_for(std::chrono::microseconds(100));
      return *this;
    }
  };

  EnvThread* CreateThread(std::function<void()> f) {
    return new EnvThread([f = std::move(f)]() {
      delaying_environment_worker = true;
      f();
    });
  }
  Task CreateTask(std::function<void()> f) {
    Task task;
    task.f = std::move(f);
    return task;
  }
  void ExecuteTask(const Task& task) { task.f(); }
};

template <typename Environment = StlThreadEnvironment>
static void test_parallelism(bool allow_spinning, int kThreads = 16) {
  // Test we never-ever fail to match available tasks with idle threads.
  // The phases below require a multiple of four threads.
  ThreadPoolTempl<Environment> tp(kThreads, allow_spinning);
  VERIFY_IS_EQUAL(tp.NumThreads(), kThreads);
  VERIFY_IS_EQUAL(tp.CurrentThreadId(), -1);
  for (int iter = 0; iter < 100; ++iter) {
    std::atomic<int> running(0);
    std::atomic<int> done(0);
    std::atomic<int> phase(0);
    // Schedule kThreads tasks and ensure that they all are running.
    for (int i = 0; i < kThreads; ++i) {
      tp.Schedule([&]() {
        const int thread_id = tp.CurrentThreadId();
        VERIFY_GE(thread_id, 0);
        VERIFY_LE(thread_id, kThreads - 1);
        running++;
        while (phase < 1) {
          std::this_thread::yield();
        }
        done++;
      });
    }
    VERIFY(spin_wait([&] { return running == kThreads; }));
    running = 0;
    phase = 1;
    // Now, while the previous tasks exit, schedule another kThreads tasks and
    // ensure that they are running.
    for (int i = 0; i < kThreads; ++i) {
      tp.Schedule([&, i]() {
        running++;
        while (phase < 2) {
          std::this_thread::yield();
        }
        // When all tasks are running, half of tasks exit, quarter of tasks
        // continue running and quarter of tasks schedule another 2 tasks each.
        // Concurrently main thread schedules another quarter of tasks.
        // This gives us another kThreads tasks and we ensure that they all
        // are running.
        if (i < kThreads / 2) {
        } else if (i < 3 * kThreads / 4) {
          running++;
          while (phase < 3) {
            std::this_thread::yield();
          }
          done++;
        } else {
          for (int j = 0; j < 2; ++j) {
            tp.Schedule([&]() {
              running++;
              while (phase < 3) {
                std::this_thread::yield();
              }
              done++;
            });
          }
        }
        done++;
      });
    }
    VERIFY(spin_wait([&] { return running == kThreads; }));
    running = 0;
    phase = 2;
    for (int i = 0; i < kThreads / 4; ++i) {
      tp.Schedule([&]() {
        running++;
        while (phase < 3) {
          std::this_thread::yield();
        }
        done++;
      });
    }
    VERIFY(spin_wait([&] { return running == kThreads; }));
    phase = 3;
    VERIFY(spin_wait([&] { return done == 3 * kThreads; }));
  }
}

static void test_cancel() {
  ThreadPool tp(2);

  // Schedule a large number of closure that each sleeps for one second. This
  // will keep the thread pool busy for much longer than the default test timeout.
  for (int i = 0; i < 1000; ++i) {
    tp.Schedule([]() { std::this_thread::sleep_for(std::chrono::milliseconds(2000)); });
  }

  // Cancel the processing of all the closures that are still pending.
  tp.Cancel();
}

// A closure that is still running when Cancel() is called, and then schedules another one, leaves that closure in
// its worker's queue. The destructor must discard it only after the workers have exited: ~RunQueue asserts that the
// queue is empty, and the queues' owner-only operations must not run beside the workers.
static void test_cancel_while_task_schedules() {
  for (int threads : {1, 2, 4}) {
    std::atomic<bool> started(false), cancelled(false);
    {
      ThreadPool tp(threads);
      tp.Schedule([&] {
        started = true;
        while (!cancelled) std::this_thread::yield();
        // By now the destructor is waiting to join this worker; the closures scheduled below must still be discarded.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        for (int i = 0; i < 4; ++i) tp.Schedule([] {});
      });
      VERIFY(spin_wait([&] { return started.load(); }));
      tp.Cancel();
      cancelled = true;
    }
  }
}

// A closure that is running at Cancel() may wait for something that a queued closure owns. Cancel() discards queued
// closures without running them, so destroying them must not wait for the running one to return: it is destroyed
// before the workers are joined. The pool is destroyed in a thread, so that a hang fails the test instead of
// stopping it.
static void test_cancel_destroys_queued_closures_before_join() {
  struct State {
    std::atomic<bool> finished{false}, released{false};
  };
  for (int threads : {1, 2, 4}) {
    auto state = std::make_shared<State>();
    std::thread driver([state, threads] {
      {
        ThreadPool tp(threads);
        // The owner of a queued closure: its destruction releases the waiting closure.
        std::shared_ptr<void> owner(nullptr, [state](void*) { state->released = true; });
        std::atomic<int> running(0);
        // Occupy every worker, so that the closure scheduled below stays queued.
        for (int i = 0; i < threads; ++i) {
          tp.Schedule([state, &running] {
            ++running;
            while (!state->released) std::this_thread::yield();
          });
        }
        while (running < threads) std::this_thread::yield();
        tp.Schedule([owner] {});
        owner.reset();
        tp.Cancel();
      }
      state->finished = true;
    });
    const bool done = spin_wait([&] { return state->finished.load(); }, 30);
    if (!done) std::cerr << "~ThreadPool did not return after Cancel(), " << threads << " threads\n";
    VERIFY(done);
    if (!done) {
      driver.detach();
      return;
    }
    driver.join();
  }
}

// A worker that passed the cancellation check of its loop may reach WaitForWork() after Cancel() has notified the
// waiters. It must see the cancellation after Prewait(): the other workers leave without blocking, so nothing would
// wake it again, and the destructor would wait for it forever. The window is narrow, so run many short lifetimes with
// a Cancel() at a random point, in a thread that a hang cannot take the test down with: the test fails when the
// lifetimes stop completing.
static void test_cancel_races_with_idle_workers() {
  std::atomic<bool> finished(false);
  std::atomic<int> completed(0);
  std::thread driver([&] {
    for (int iter = 0; iter < 3000; ++iter) {
      {
        ThreadPool tp(2 + iter % 3);
        const int tasks = internal::random<int>(0, 4);
        for (int i = 0; i < tasks; ++i) {
          const int micros = internal::random<int>(0, 50);
          tp.Schedule([micros] {
            const auto end = std::chrono::steady_clock::now() + std::chrono::microseconds(micros);
            while (std::chrono::steady_clock::now() < end) {
            }
          });
        }
        for (int i = internal::random<int>(0, 100); i > 0; --i) std::this_thread::yield();
        tp.Cancel();
      }  // The destructor returns only when every worker has exited.
      completed = iter + 1;
    }
    finished = true;
  });
  int seen = 0;
  while (!finished) {
    const bool progress = spin_wait([&] { return finished.load() || completed.load() > seen; }, 30);
    if (!progress) std::cerr << "no pool was destroyed after iteration " << seen << "\n";
    VERIFY(progress);
    seen = completed.load();
  }
  driver.join();
}

static void test_pool_partitions() {
  const int kThreads = 2;

  std::atomic<int> running(0);
  std::atomic<int> done(0);
  std::atomic<int> phase(0);

  {
    ThreadPool tp(kThreads);

    // Assign each thread to its own partition, so that stealing other work only
    // occurs globally when a thread is idle.
    std::vector<std::pair<unsigned, unsigned>> steal_partitions(kThreads);
    for (int i = 0; i < kThreads; ++i) {
      steal_partitions[i] = std::make_pair(i, i + 1);
    }
    tp.SetStealPartitions(steal_partitions);

    // Schedule kThreads tasks and ensure that they all are running.
    for (int i = 0; i < kThreads; ++i) {
      tp.Schedule([&]() {
        const int thread_id = tp.CurrentThreadId();
        VERIFY_GE(thread_id, 0);
        VERIFY_LE(thread_id, kThreads - 1);
        ++running;
        while (phase < 1) {
          std::this_thread::yield();
        }
        ++done;
      });
    }
    VERIFY(spin_wait([&] { return running == kThreads; }));
    // Schedule each closure to only run on thread 'i' and verify that it does.
    for (int i = 0; i < kThreads; ++i) {
      tp.ScheduleWithHint(
          [&, i]() {
            ++running;
            const int thread_id = tp.CurrentThreadId();
            VERIFY_IS_EQUAL(thread_id, i);
            while (phase < 2) {
              std::this_thread::yield();
            }
            ++done;
          },
          i, i + 1);
    }
    running = 0;
    phase = 1;
    VERIFY(spin_wait([&] { return running == kThreads; }));
    running = 0;
    phase = 2;
  }
}

// A worker of another pool gets no work from this pool's queues.
static void test_maybe_get_task_from_other_pool() {
  std::atomic<int> taken(0);
  Barrier busy(1), release(1), done(4);
  // Declared last, so both pools are joined before the barriers their tasks use are destroyed.
  ThreadPool a(4), b(1);
  b.Schedule([&]() {
    busy.Notify();
    release.Wait();
  });
  busy.Wait();
  b.Schedule([]() {});
  for (int i = 0; i < 4; ++i) {
    a.Schedule([&]() {
      ThreadPool::Task t;
      b.MaybeGetTask(&t);
      if (t.f) ++taken;
      done.Notify();
    });
  }
  done.Wait();
  release.Notify();
  VERIFY_IS_EQUAL(taken.load(), 0);
}

EIGEN_DECLARE_TEST(threads_non_blocking_thread_pool) {
  CALL_SUBTEST(test_parallelism<DelayingThreadEnvironment>(true, 4));
  CALL_SUBTEST(test_create_destroy_empty_pool());
  CALL_SUBTEST(test_parallelism(true));
  CALL_SUBTEST(test_parallelism(false));
  CALL_SUBTEST(test_cancel());
  CALL_SUBTEST(test_cancel_while_task_schedules());
  CALL_SUBTEST(test_cancel_destroys_queued_closures_before_join());
  CALL_SUBTEST(test_cancel_races_with_idle_workers());
  CALL_SUBTEST(test_pool_partitions());
  CALL_SUBTEST(test_maybe_get_task_from_other_pool());
}
