// Benchmarks for triangular matrix-vector product (TRMV).
//
// Tests y += triangularView(A) * x for various modes and sizes, with real, complex and mixed (real matrix, complex
// vector) scalars, for column-major and row-major A.
// Exercises TriangularMatrixVector.h kernel.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

template <typename LhsScalar, typename RhsScalar>
double trmvFlops(Index n) {
  // TRMV: ~n^2 multiply-adds, of 2 flops (real), 4 (real times complex) or 8 (complex) each
  const int complex_operands = int(NumTraits<LhsScalar>::IsComplex) + int(NumTraits<RhsScalar>::IsComplex);
  return double(2 << complex_operands) * n * n;
}

// y = triangularView<Mode>(A) * x
template <typename LhsScalar, typename RhsScalar, unsigned int Mode, int StorageOrder = ColMajor>
static void BM_TRMV(benchmark::State& state) {
  const Index n = state.range(0);
  using Mat = Matrix<LhsScalar, Dynamic, Dynamic, StorageOrder>;
  using Vec = Matrix<RhsScalar, Dynamic, 1>;
  using ResVec = Matrix<typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType, Dynamic, 1>;
  Mat A = Mat::Random(n, n);
  Vec x = Vec::Random(n);
  ResVec y(n);
  for (auto _ : state) {
    y.noalias() = A.template triangularView<Mode>() * x;
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] = benchmark::Counter(
      trmvFlops<LhsScalar, RhsScalar>(n), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

using cfloat = std::complex<float>;
using cdouble = std::complex<double>;

// clang-format off
#define TRMV_SIZES ->Arg(8)->Arg(16)->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(512)->Arg(1024)->Arg(2048)
BENCHMARK(BM_TRMV<float, float, Lower>) TRMV_SIZES ->Name("TRMV_float_Lower");
BENCHMARK(BM_TRMV<float, float, Upper>) TRMV_SIZES ->Name("TRMV_float_Upper");
BENCHMARK(BM_TRMV<float, float, UnitLower>) TRMV_SIZES ->Name("TRMV_float_UnitLower");
BENCHMARK(BM_TRMV<float, float, UnitUpper>) TRMV_SIZES ->Name("TRMV_float_UnitUpper");
BENCHMARK(BM_TRMV<double, double, Lower>) TRMV_SIZES ->Name("TRMV_double_Lower");
BENCHMARK(BM_TRMV<double, double, Upper>) TRMV_SIZES ->Name("TRMV_double_Upper");
BENCHMARK(BM_TRMV<double, double, UnitLower>) TRMV_SIZES ->Name("TRMV_double_UnitLower");
BENCHMARK(BM_TRMV<double, double, UnitUpper>) TRMV_SIZES ->Name("TRMV_double_UnitUpper");
BENCHMARK(BM_TRMV<cfloat, cfloat, Lower>) TRMV_SIZES ->Name("TRMV_cfloat_Lower");
BENCHMARK(BM_TRMV<cfloat, cfloat, Upper>) TRMV_SIZES ->Name("TRMV_cfloat_Upper");
BENCHMARK(BM_TRMV<cdouble, cdouble, Lower>) TRMV_SIZES ->Name("TRMV_cdouble_Lower");
BENCHMARK(BM_TRMV<cdouble, cdouble, Upper>) TRMV_SIZES ->Name("TRMV_cdouble_Upper");
BENCHMARK(BM_TRMV<float, cfloat, Lower>) TRMV_SIZES ->Name("TRMV_float_cfloat_Lower");
BENCHMARK(BM_TRMV<float, cfloat, Upper>) TRMV_SIZES ->Name("TRMV_float_cfloat_Upper");
BENCHMARK(BM_TRMV<double, cdouble, Lower>) TRMV_SIZES ->Name("TRMV_double_cdouble_Lower");
BENCHMARK(BM_TRMV<double, cdouble, Upper>) TRMV_SIZES ->Name("TRMV_double_cdouble_Upper");
BENCHMARK(BM_TRMV<double, double, Lower, RowMajor>) TRMV_SIZES ->Name("TRMV_double_Lower_RowMajor");
BENCHMARK(BM_TRMV<double, double, Upper, RowMajor>) TRMV_SIZES ->Name("TRMV_double_Upper_RowMajor");
BENCHMARK(BM_TRMV<cdouble, cdouble, Lower, RowMajor>) TRMV_SIZES ->Name("TRMV_cdouble_Lower_RowMajor");
BENCHMARK(BM_TRMV<cdouble, cdouble, Upper, RowMajor>) TRMV_SIZES ->Name("TRMV_cdouble_Upper_RowMajor");
BENCHMARK(BM_TRMV<double, cdouble, Lower, RowMajor>) TRMV_SIZES ->Name("TRMV_double_cdouble_Lower_RowMajor");
BENCHMARK(BM_TRMV<double, cdouble, Upper, RowMajor>) TRMV_SIZES ->Name("TRMV_double_cdouble_Upper_RowMajor");
#undef TRMV_SIZES
// clang-format on
