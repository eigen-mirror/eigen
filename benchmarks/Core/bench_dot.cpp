// Benchmarks for dot product (BLAS-1 critical path).
//
// Flop count: 2n for real, 8n for complex.

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

template <typename Scalar>
double dotFlops(Index n) {
  return (NumTraits<Scalar>::IsComplex ? 8.0 : 2.0) * n;
}

template <typename Scalar>
static void BM_Dot(benchmark::State& state) {
  const Index n = state.range(0);
  using Vec = Matrix<Scalar, Dynamic, 1>;
  Vec a = Vec::Random(n);
  Vec b = Vec::Random(n);
  for (auto _ : state) {
    Scalar d = a.dot(b);
    benchmark::DoNotOptimize(d);
  }
  state.counters["GFLOPS"] = benchmark::Counter(dotFlops<Scalar>(n), benchmark::Counter::kIsIterationInvariantRate,
                                                benchmark::Counter::kIs1000);
}

template <typename Scalar>
static void BM_SquaredNorm(benchmark::State& state) {
  const Index n = state.range(0);
  using Vec = Matrix<Scalar, Dynamic, 1>;
  Vec a = Vec::Random(n);
  for (auto _ : state) {
    auto d = a.squaredNorm();
    benchmark::DoNotOptimize(d);
  }
  state.counters["GFLOPS"] = benchmark::Counter(dotFlops<Scalar>(n), benchmark::Counter::kIsIterationInvariantRate,
                                                benchmark::Counter::kIs1000);
}

static void DotSizes(::benchmark::Benchmark* b) {
  for (int n : {64, 256, 1024, 4096, 16384, 65536, 262144, 1048576}) b->Arg(n);
}

BENCHMARK(BM_Dot<float>)->Apply(DotSizes)->Name("Dot_float");
BENCHMARK(BM_Dot<double>)->Apply(DotSizes)->Name("Dot_double");
BENCHMARK(BM_Dot<std::complex<float>>)->Apply(DotSizes)->Name("Dot_cfloat");
BENCHMARK(BM_Dot<std::complex<double>>)->Apply(DotSizes)->Name("Dot_cdouble");
BENCHMARK(BM_SquaredNorm<float>)->Apply(DotSizes)->Name("SquaredNorm_float");
BENCHMARK(BM_SquaredNorm<double>)->Apply(DotSizes)->Name("SquaredNorm_double");
