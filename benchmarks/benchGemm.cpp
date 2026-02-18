#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

#ifndef SCALAR
#define SCALAR float
#endif

typedef SCALAR Scalar;
typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

template <typename A, typename B, typename C>
EIGEN_DONT_INLINE void gemm(const A& a, const B& b, C& c) {
  c.noalias() += a * b;
}

static void BM_EigenGemm(benchmark::State& state) {
  int m = state.range(0);
  int n = state.range(1);
  int p = state.range(2);
  Mat a(m, p);
  a.setRandom();
  Mat b(p, n);
  b.setRandom();
  Mat c = Mat::Zero(m, n);
  for (auto _ : state) {
    c.setZero();
    gemm(a, b, c);
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] =
      benchmark::Counter(2.0 * m * n * p, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

static void GemmSizes(::benchmark::Benchmark* b) {
  for (int size : {8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512, 768, 1024, 1536, 2048}) {
    b->Args({size, size, size});
  }
  // Non-square sizes
  b->Args({64, 64, 1024});
  b->Args({1024, 64, 64});
  b->Args({64, 1024, 64});
  b->Args({256, 256, 1024});
  b->Args({1024, 256, 256});
}

BENCHMARK(BM_EigenGemm)->Apply(GemmSizes);

#ifdef HAVE_BLAS
extern "C" {
#include <Eigen/src/misc/blas.h>
}

static void BM_BlasGemm(benchmark::State& state) {
  int m = state.range(0);
  int n = state.range(1);
  int p = state.range(2);
  Mat a(m, p);
  a.setRandom();
  Mat b(p, n);
  b.setRandom();
  Mat c = Mat::Zero(m, n);
  char notrans = 'N';
  Scalar one = 1, zero = 0;
  for (auto _ : state) {
    c.setZero();
    if constexpr (std::is_same_v<Scalar, float>) {
      sgemm_(&notrans, &notrans, &m, &n, &p, &one, a.data(), &m, b.data(), &p, &one, c.data(), &m);
    } else {
      dgemm_(&notrans, &notrans, &m, &n, &p, &one, a.data(), &m, b.data(), &p, &one, c.data(), &m);
    }
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] =
      benchmark::Counter(2.0 * m * n * p, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}
BENCHMARK(BM_BlasGemm)->Apply(GemmSizes);
#endif
