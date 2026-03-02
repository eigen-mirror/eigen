// Benchmarks for selfadjoint matrix-vector product (SYMV/HEMV).
//
// Tests y += selfadjointView(A) * x for various sizes and scalar types.
// Exercises SelfadjointMatrixVector.h kernel.

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

template <typename Scalar>
double symvFlops(Index n) {
  // SYMV uses n^2 multiply-adds (exploiting symmetry)
  return (NumTraits<Scalar>::IsComplex ? 8.0 : 2.0) * n * n;
}

// y += selfadjointView<Lower>(A) * x
template <typename Scalar>
static void BM_SYMV_Lower(benchmark::State& state) {
  const Index n = state.range(0);
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  Mat A = Mat::Random(n, n);
  A = (A + A.transpose().eval()) / Scalar(2);
  Vec x = Vec::Random(n);
  Vec y = Vec::Random(n);
  for (auto _ : state) {
    y.noalias() += A.template selfadjointView<Lower>() * x;
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] = benchmark::Counter(symvFlops<Scalar>(n), benchmark::Counter::kIsIterationInvariantRate,
                                                benchmark::Counter::kIs1000);
}

// y += selfadjointView<Upper>(A) * x
template <typename Scalar>
static void BM_SYMV_Upper(benchmark::State& state) {
  const Index n = state.range(0);
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  Mat A = Mat::Random(n, n);
  A = (A + A.transpose().eval()) / Scalar(2);
  Vec x = Vec::Random(n);
  Vec y = Vec::Random(n);
  for (auto _ : state) {
    y.noalias() += A.template selfadjointView<Upper>() * x;
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] = benchmark::Counter(symvFlops<Scalar>(n), benchmark::Counter::kIsIterationInvariantRate,
                                                benchmark::Counter::kIs1000);
}

static void SymvSizes(::benchmark::Benchmark* b) {
  for (int n : {8, 16, 32, 64, 128, 256, 512, 1024, 2048}) b->Arg(n);
}

BENCHMARK(BM_SYMV_Lower<float>)->Apply(SymvSizes)->Name("SYMV_Lower_float");
BENCHMARK(BM_SYMV_Lower<double>)->Apply(SymvSizes)->Name("SYMV_Lower_double");
BENCHMARK(BM_SYMV_Upper<float>)->Apply(SymvSizes)->Name("SYMV_Upper_float");
BENCHMARK(BM_SYMV_Upper<double>)->Apply(SymvSizes)->Name("SYMV_Upper_double");
