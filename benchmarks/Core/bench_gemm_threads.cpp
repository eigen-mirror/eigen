// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// GEMM on the thread pool over the pool size. On a machine whose GEMM kernel
// runs on shared units (the ARM SME backend: one unit per cluster), the rate
// levels off at the unit count and falls beyond it unless the thread count is
// capped; the "uncapped" variant removes the cap (setNbSmeUnits(0)) for the
// comparison and is a no-op on other backends.
#define EIGEN_GEMM_THREADPOOL
#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <Eigen/ThreadPool>
#include <initializer_list>
#include <thread>

using namespace Eigen;

#ifndef SCALAR
#define SCALAR float
#endif
using Scalar = SCALAR;
using Mat = Matrix<Scalar, Dynamic, Dynamic>;

template <typename A, typename B, typename C>
EIGEN_DONT_INLINE void gemm(const A& a, const B& b, C& c) {
  c.noalias() += a * b;
}

template <bool Uncapped>
static void BM_GemmThreads(benchmark::State& state) {
  const int n = static_cast<int>(state.range(0));
  const int threads = static_cast<int>(state.range(1));
  ThreadPool pool(threads);
  setGemmThreadPool(&pool);
  const int units = nbSmeUnits();
  if (Uncapped) setNbSmeUnits(0);
  Mat a = Mat::Random(n, n), b = Mat::Random(n, n), c = Mat::Zero(n, n);
  for (auto _ : state) {
    gemm(a, b, c);
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] =
      benchmark::Counter(2.0 * n * n * n, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
  setNbSmeUnits(units);
}

// Pool sizes up to the core count (a larger pool only oversubscribes), keeping the smallest of each list.
static void ThreadArgs(::benchmark::Benchmark* b, std::initializer_list<int> threads) {
  const int cores = static_cast<int>(std::thread::hardware_concurrency());
  for (int n : {1024, 2048})
    for (int t : threads)
      if (t <= cores || t == *threads.begin()) b->Args({n, t});
}

BENCHMARK(BM_GemmThreads<false>)
    ->Name("BM_GemmThreads")
    ->Apply([](::benchmark::Benchmark* b) {
      ThreadArgs(b, {1, 2, 4, 8, 12});
    })
    ->UseRealTime();
BENCHMARK(BM_GemmThreads<true>)
    ->Name("BM_GemmThreads_uncapped")
    ->Apply([](::benchmark::Benchmark* b) {
      ThreadArgs(b, {4, 8, 12});
    })
    ->UseRealTime();
