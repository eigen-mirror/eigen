// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// The thread pool without thread-local storage, where all threads that are not workers share one per-thread record.
#define EIGEN_AVOID_THREAD_LOCAL
#define EIGEN_USE_THREADS
#include "main.h"
#include "Eigen/ThreadPool"

// Closures scheduled from several free-standing threads at once all run. Under ThreadSanitizer this also checks that
// the choice of a queue does not write state shared by those threads.
static void test_concurrent_schedule() {
  constexpr int kCallers = 3, kPerCaller = 2000;
  std::atomic<int> ran(0);
  {
    ThreadPool tp(3);
    std::vector<std::thread> callers;
    for (int c = 0; c < kCallers; ++c)
      callers.emplace_back([&] {
        for (int i = 0; i < kPerCaller; ++i) tp.Schedule([&] { ++ran; });
      });
    for (std::thread& t : callers) t.join();
  }  // The destructor runs the closures that are still queued.
  VERIFY_IS_EQUAL(ran.load(), kCallers * kPerCaller);
}

EIGEN_DECLARE_TEST(threads_non_blocking_thread_pool_avoid_tls) {
#ifdef EIGEN_THREAD_LOCAL
  VERIFY(false);  // EIGEN_AVOID_THREAD_LOCAL did not take effect.
#endif
  CALL_SUBTEST(test_concurrent_schedule());
}
