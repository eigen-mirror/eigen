// Benchmarks for triangular-dense matrix products (TRMM).
//
// Tests C = triangular(A) * B for various modes (Lower/Upper) and sides (Left/Right).
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

// C and reference are two evaluations of n-term inner products; || |T|*|B| ||_F <= ||T||_F * ||B||_F.
template <typename Mat>
static bool MatchesDenseProduct(benchmark::State& state, const Mat& C, const Mat& reference, const Mat& T,
                                const Mat& B) {
  using Scalar = typename Mat::Scalar;
  const Scalar bound = Scalar(2) * NumTraits<Scalar>::epsilon() * Scalar(T.rows()) * T.norm() * B.norm();
  if (!numext::isfinite(bound) || !((C - reference).norm() <= bound)) {
    state.SkipWithError("TRMM result differs from dense product");
    return false;
  }
  return true;
}

// C = triangularView<Mode>(A) * B
template <typename Scalar, unsigned int Mode, int Order = ColMajor>
static void BM_TRMM_Left(benchmark::State& state) {
  const Index n = state.range(0);
  const Index cols = state.range(1);
  using Mat = Matrix<Scalar, Dynamic, Dynamic, Order>;
  Mat A = Mat::Random(n, n);
  Mat B = Mat::Random(n, cols);
  Mat C(n, cols);
  const Mat dense = A.template triangularView<Mode>();
  const Mat reference = dense * B;
  C.noalias() = A.template triangularView<Mode>() * B;
  if (!MatchesDenseProduct(state, C, reference, dense, B)) return;
  for (auto _ : state) {
    C.noalias() = A.template triangularView<Mode>() * B;
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] = benchmark::Counter(
      double(n) * (n + 1) * cols, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// C = B * triangularView<Mode>(A)
template <typename Scalar, unsigned int Mode, int Order = ColMajor>
static void BM_TRMM_Right(benchmark::State& state) {
  const Index n = state.range(0);
  const Index rows = state.range(1);
  using Mat = Matrix<Scalar, Dynamic, Dynamic, Order>;
  Mat A = Mat::Random(n, n);
  Mat B = Mat::Random(rows, n);
  Mat C(rows, n);
  const Mat dense = A.template triangularView<Mode>();
  const Mat reference = B * dense;
  C.noalias() = B * A.template triangularView<Mode>();
  if (!MatchesDenseProduct(state, C, reference, dense, B)) return;
  for (auto _ : state) {
    C.noalias() = B * A.template triangularView<Mode>();
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] = benchmark::Counter(
      double(n) * (n + 1) * rows, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// clang-format off
#define TRMM_SIZES \
  ->Args({64, 64})->Args({128, 128})->Args({256, 256})->Args({512, 512}) \
  ->Args({1001, 1001})->Args({1024, 1024})->Args({2048, 2048})->Args({4096, 4096}) \
  ->Args({4096, 16})->Args({4096, 128})->Args({128, 4096}) \
  ->Args({4096, 1024})->Args({1024, 4096})
// Left product
BENCHMARK(BM_TRMM_Left<float, Lower>) TRMM_SIZES ->Name("TRMM_Left_float_Lower");
BENCHMARK(BM_TRMM_Left<float, Upper>) TRMM_SIZES ->Name("TRMM_Left_float_Upper");
BENCHMARK(BM_TRMM_Left<double, Lower>) TRMM_SIZES ->Name("TRMM_Left_double_Lower");
BENCHMARK(BM_TRMM_Left<double, Upper>) TRMM_SIZES ->Name("TRMM_Left_double_Upper");
// Right product
BENCHMARK(BM_TRMM_Right<float, Lower>) TRMM_SIZES ->Name("TRMM_Right_float_Lower");
BENCHMARK(BM_TRMM_Right<float, Upper>) TRMM_SIZES ->Name("TRMM_Right_float_Upper");
BENCHMARK(BM_TRMM_Right<double, Lower>) TRMM_SIZES ->Name("TRMM_Right_double_Lower");
BENCHMARK(BM_TRMM_Right<double, Upper>) TRMM_SIZES ->Name("TRMM_Right_double_Upper");
BENCHMARK(BM_TRMM_Left<float, Lower, RowMajor>) TRMM_SIZES ->Name("TRMM_Left_float_Lower_RowMajor");
BENCHMARK(BM_TRMM_Left<float, Upper, RowMajor>) TRMM_SIZES ->Name("TRMM_Left_float_Upper_RowMajor");
BENCHMARK(BM_TRMM_Left<double, Lower, RowMajor>) TRMM_SIZES ->Name("TRMM_Left_double_Lower_RowMajor");
BENCHMARK(BM_TRMM_Left<double, Upper, RowMajor>) TRMM_SIZES ->Name("TRMM_Left_double_Upper_RowMajor");
BENCHMARK(BM_TRMM_Right<float, Lower, RowMajor>) TRMM_SIZES ->Name("TRMM_Right_float_Lower_RowMajor");
BENCHMARK(BM_TRMM_Right<float, Upper, RowMajor>) TRMM_SIZES ->Name("TRMM_Right_float_Upper_RowMajor");
BENCHMARK(BM_TRMM_Right<double, Lower, RowMajor>) TRMM_SIZES ->Name("TRMM_Right_double_Lower_RowMajor");
BENCHMARK(BM_TRMM_Right<double, Upper, RowMajor>) TRMM_SIZES ->Name("TRMM_Right_double_Upper_RowMajor");
#undef TRMM_SIZES
// clang-format on
