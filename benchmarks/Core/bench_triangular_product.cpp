// Benchmarks for triangular-dense matrix products (TRMM).
//
// Tests C = triangular(A) * B for various modes (Lower/Upper) and sides (Left/Right).

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

// C = triangularView<Mode>(A) * B
template <typename Scalar, unsigned int Mode>
static void BM_TRMM_Left(benchmark::State& state) {
  const Index n = state.range(0);
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  Mat A = Mat::Random(n, n);
  Mat B = Mat::Random(n, n);
  Mat C(n, n);
  for (auto _ : state) {
    C.noalias() = A.template triangularView<Mode>() * B;
    benchmark::DoNotOptimize(C.data());
  }
  state.counters["GFLOPS"] =
      benchmark::Counter(1.0 * n * n * n, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// C = B * triangularView<Mode>(A)
template <typename Scalar, unsigned int Mode>
static void BM_TRMM_Right(benchmark::State& state) {
  const Index n = state.range(0);
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  Mat A = Mat::Random(n, n);
  Mat B = Mat::Random(n, n);
  Mat C(n, n);
  for (auto _ : state) {
    C.noalias() = B * A.template triangularView<Mode>();
    benchmark::DoNotOptimize(C.data());
  }
  state.counters["GFLOPS"] =
      benchmark::Counter(1.0 * n * n * n, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

static void TrmmSizes(::benchmark::Benchmark* b) {
  for (int n : {64, 128, 256, 512, 1024}) b->Arg(n);
}

// Left product
BENCHMARK(BM_TRMM_Left<float, Lower>)->Apply(TrmmSizes)->Name("TRMM_Left_float_Lower");
BENCHMARK(BM_TRMM_Left<float, Upper>)->Apply(TrmmSizes)->Name("TRMM_Left_float_Upper");
BENCHMARK(BM_TRMM_Left<double, Lower>)->Apply(TrmmSizes)->Name("TRMM_Left_double_Lower");
BENCHMARK(BM_TRMM_Left<double, Upper>)->Apply(TrmmSizes)->Name("TRMM_Left_double_Upper");

// Right product
BENCHMARK(BM_TRMM_Right<float, Lower>)->Apply(TrmmSizes)->Name("TRMM_Right_float_Lower");
BENCHMARK(BM_TRMM_Right<float, Upper>)->Apply(TrmmSizes)->Name("TRMM_Right_float_Upper");
BENCHMARK(BM_TRMM_Right<double, Lower>)->Apply(TrmmSizes)->Name("TRMM_Right_double_Lower");
BENCHMARK(BM_TRMM_Right<double, Upper>)->Apply(TrmmSizes)->Name("TRMM_Right_double_Upper");
