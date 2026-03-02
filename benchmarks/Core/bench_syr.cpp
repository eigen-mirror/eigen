// Benchmarks for symmetric rank-1 update (SYR).
//
// Tests C.selfadjointView<Lower>().rankUpdate(v, alpha) which computes
// C += alpha * v * v^T, updating only the lower (or upper) triangle.
// Exercises SelfadjointProduct.h / selfadjoint_rank1_update.

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

template <typename Scalar>
double syrFlops(Index n) {
  // SYR: n*(n+1)/2 multiply-adds ~ n^2
  return (NumTraits<Scalar>::IsComplex ? 8.0 : 2.0) * n * (n + 1) / 2;
}

template <typename Scalar>
static void BM_SYR_Lower(benchmark::State& state) {
  const Index n = state.range(0);
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  Vec v = Vec::Random(n);
  Mat C = Mat::Zero(n, n);
  Scalar alpha(1);
  for (auto _ : state) {
    C.template selfadjointView<Lower>().rankUpdate(v, alpha);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] = benchmark::Counter(syrFlops<Scalar>(n), benchmark::Counter::kIsIterationInvariantRate,
                                                benchmark::Counter::kIs1000);
}

template <typename Scalar>
static void BM_SYR_Upper(benchmark::State& state) {
  const Index n = state.range(0);
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  Vec v = Vec::Random(n);
  Mat C = Mat::Zero(n, n);
  Scalar alpha(1);
  for (auto _ : state) {
    C.template selfadjointView<Upper>().rankUpdate(v, alpha);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] = benchmark::Counter(syrFlops<Scalar>(n), benchmark::Counter::kIsIterationInvariantRate,
                                                benchmark::Counter::kIs1000);
}

static void SyrSizes(::benchmark::Benchmark* b) {
  for (int n : {8, 16, 32, 64, 128, 256, 512, 1024, 2048}) b->Arg(n);
}

BENCHMARK(BM_SYR_Lower<float>)->Apply(SyrSizes)->Name("SYR_Lower_float");
BENCHMARK(BM_SYR_Lower<double>)->Apply(SyrSizes)->Name("SYR_Lower_double");
BENCHMARK(BM_SYR_Upper<float>)->Apply(SyrSizes)->Name("SYR_Upper_float");
BENCHMARK(BM_SYR_Upper<double>)->Apply(SyrSizes)->Name("SYR_Upper_double");
