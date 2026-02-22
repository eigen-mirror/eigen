// Benchmarks for self-adjoint (symmetric/hermitian) matrix operations.
//
// Tests SYMM (selfadjointView * dense) and rank-k updates.

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

// C = selfadjointView<Lower>(A) * B  (SYMM)
template <typename Scalar>
static void BM_SYMM_Left(benchmark::State& state) {
  const Index n = state.range(0);
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  Mat A = Mat::Random(n, n);
  A = (A + A.transpose()).eval() / Scalar(2);
  Mat B = Mat::Random(n, n);
  Mat C(n, n);
  for (auto _ : state) {
    C.noalias() = A.template selfadjointView<Lower>() * B;
    benchmark::DoNotOptimize(C.data());
  }
  state.counters["GFLOPS"] =
      benchmark::Counter(2.0 * n * n * n, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// C = B * selfadjointView<Lower>(A)
template <typename Scalar>
static void BM_SYMM_Right(benchmark::State& state) {
  const Index n = state.range(0);
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  Mat A = Mat::Random(n, n);
  A = (A + A.transpose()).eval() / Scalar(2);
  Mat B = Mat::Random(n, n);
  Mat C(n, n);
  for (auto _ : state) {
    C.noalias() = B * A.template selfadjointView<Lower>();
    benchmark::DoNotOptimize(C.data());
  }
  state.counters["GFLOPS"] =
      benchmark::Counter(2.0 * n * n * n, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// Rank-k update: C.selfadjointView<Lower>().rankUpdate(A)
// Computes C += A * A^T
template <typename Scalar>
static void BM_RankUpdate(benchmark::State& state) {
  const Index n = state.range(0);
  const Index k = state.range(1);
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  Mat A = Mat::Random(n, k);
  Mat C = Mat::Zero(n, n);
  for (auto _ : state) {
    C.template selfadjointView<Lower>().rankUpdate(A);
    benchmark::DoNotOptimize(C.data());
  }
  state.counters["GFLOPS"] =
      benchmark::Counter(1.0 * n * n * k, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

static void SymmSizes(::benchmark::Benchmark* b) {
  for (int n : {64, 128, 256, 512, 1024}) b->Arg(n);
}

static void RankUpdateSizes(::benchmark::Benchmark* b) {
  for (int n : {64, 128, 256, 512}) {
    for (int k : {16, 64, 256}) {
      b->Args({n, k});
    }
  }
}

BENCHMARK(BM_SYMM_Left<float>)->Apply(SymmSizes)->Name("SYMM_Left_float");
BENCHMARK(BM_SYMM_Left<double>)->Apply(SymmSizes)->Name("SYMM_Left_double");
BENCHMARK(BM_SYMM_Right<float>)->Apply(SymmSizes)->Name("SYMM_Right_float");
BENCHMARK(BM_SYMM_Right<double>)->Apply(SymmSizes)->Name("SYMM_Right_double");
BENCHMARK(BM_RankUpdate<float>)->Apply(RankUpdateSizes)->Name("RankUpdate_float");
BENCHMARK(BM_RankUpdate<double>)->Apply(RankUpdateSizes)->Name("RankUpdate_double");
