// Benchmarks for symmetric rank-2 update (SYR2).
//
// Tests C.selfadjointView<Lower>().rankUpdate(u, v, alpha) which computes
// C += alpha * u * v^T + conj(alpha) * v * u^T.
// Exercises SelfadjointRank2Update.h.

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

template <typename Scalar>
double syr2Flops(Index n) {
  // SYR2: 2 * n*(n+1)/2 multiply-adds ~ 2*n^2
  return (NumTraits<Scalar>::IsComplex ? 8.0 : 2.0) * 2 * n * (n + 1) / 2;
}

template <typename Scalar>
static void BM_SYR2_Lower(benchmark::State& state) {
  const Index n = state.range(0);
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  Vec u = Vec::Random(n);
  Vec v = Vec::Random(n);
  Mat C = Mat::Zero(n, n);
  Scalar alpha(1);
  for (auto _ : state) {
    C.template selfadjointView<Lower>().rankUpdate(u, v, alpha);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] = benchmark::Counter(syr2Flops<Scalar>(n), benchmark::Counter::kIsIterationInvariantRate,
                                                benchmark::Counter::kIs1000);
}

template <typename Scalar>
static void BM_SYR2_Upper(benchmark::State& state) {
  const Index n = state.range(0);
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  Vec u = Vec::Random(n);
  Vec v = Vec::Random(n);
  Mat C = Mat::Zero(n, n);
  Scalar alpha(1);
  for (auto _ : state) {
    C.template selfadjointView<Upper>().rankUpdate(u, v, alpha);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] = benchmark::Counter(syr2Flops<Scalar>(n), benchmark::Counter::kIsIterationInvariantRate,
                                                benchmark::Counter::kIs1000);
}

static void Syr2Sizes(::benchmark::Benchmark* b) {
  for (int n : {8, 16, 32, 64, 128, 256, 512, 1024, 2048}) b->Arg(n);
}

BENCHMARK(BM_SYR2_Lower<float>)->Apply(Syr2Sizes)->Name("SYR2_Lower_float");
BENCHMARK(BM_SYR2_Lower<double>)->Apply(Syr2Sizes)->Name("SYR2_Lower_double");
BENCHMARK(BM_SYR2_Upper<float>)->Apply(Syr2Sizes)->Name("SYR2_Upper_float");
BENCHMARK(BM_SYR2_Upper<double>)->Apply(Syr2Sizes)->Name("SYR2_Upper_double");
