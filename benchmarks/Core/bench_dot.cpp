// Benchmarks for dot product (BLAS-1 critical path).
//
// Flop count: 2n for real, 8n for complex.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

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

// clang-format off
#define DOT_SIZES ->Arg(64)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384)->Arg(65536)->Arg(262144)->Arg(1048576)
BENCHMARK(BM_Dot<float>) DOT_SIZES ->Name("Dot_float");
BENCHMARK(BM_Dot<double>) DOT_SIZES ->Name("Dot_double");
BENCHMARK(BM_Dot<std::complex<float>>) DOT_SIZES ->Name("Dot_cfloat");
BENCHMARK(BM_Dot<std::complex<double>>) DOT_SIZES ->Name("Dot_cdouble");
#undef DOT_SIZES
// clang-format on

// Runtime strides exercise the contiguous-map dispatch and its scalar fallback.
template <typename Scalar, bool Product>
static void BM_StridedInnerProduct(benchmark::State& state) {
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using StridedMap = Map<const Vec, Unaligned, InnerStride<Dynamic>>;
  const Index n = state.range(0), lhsStride = state.range(1), rhsStride = state.range(2);
  const Vec a = Vec::Random(n * lhsStride + 1), b = Vec::Random(n * rhsStride + 1);
  const StridedMap lhs(a.data() + 1, n, InnerStride<Dynamic>(lhsStride));
  const StridedMap rhs(b.data() + 1, n, InnerStride<Dynamic>(rhsStride));
  Scalar expected(0);
  double magnitude = 0;
  for (Index i = 0; i < n; ++i) {
    expected += (Product ? lhs(i) : numext::conj(lhs(i))) * rhs(i);
    magnitude += numext::abs(lhs(i)) * numext::abs(rhs(i));
  }
  const Scalar result = Product ? (lhs.transpose() * rhs).value() : lhs.dot(rhs);
  const double bound = 8 * n * NumTraits<typename NumTraits<Scalar>::Real>::epsilon() * magnitude;
  if (!(numext::abs(result - expected) <= bound)) {
    state.SkipWithError("Inner product disagrees with scalar reference");
    return;
  }
  for (auto _ : state) {
    benchmark::ClobberMemory();
    Scalar value = Product ? (lhs.transpose() * rhs).value() : lhs.dot(rhs);
    benchmark::DoNotOptimize(value);
  }
}

// clang-format off
#define STRIDED_INNER_PRODUCT_SIZES ->ArgsProduct({{1, 2, 3, 4, 5, 8, 17, 64, 1024, 16384}, {1, 2}, {1, 2}})
BENCHMARK_TEMPLATE(BM_StridedInnerProduct, float, false) STRIDED_INNER_PRODUCT_SIZES ->Name("StridedDot_float");
BENCHMARK_TEMPLATE(BM_StridedInnerProduct, double, false) STRIDED_INNER_PRODUCT_SIZES ->Name("StridedDot_double");
BENCHMARK_TEMPLATE(BM_StridedInnerProduct, std::complex<float>, false) STRIDED_INNER_PRODUCT_SIZES ->Name("StridedDot_cfloat");
BENCHMARK_TEMPLATE(BM_StridedInnerProduct, std::complex<double>, false) STRIDED_INNER_PRODUCT_SIZES ->Name("StridedDot_cdouble");
BENCHMARK_TEMPLATE(BM_StridedInnerProduct, float, true) STRIDED_INNER_PRODUCT_SIZES ->Name("StridedProduct_float");
BENCHMARK_TEMPLATE(BM_StridedInnerProduct, double, true) STRIDED_INNER_PRODUCT_SIZES ->Name("StridedProduct_double");
BENCHMARK_TEMPLATE(BM_StridedInnerProduct, std::complex<float>, true) STRIDED_INNER_PRODUCT_SIZES ->Name("StridedProduct_cfloat");
BENCHMARK_TEMPLATE(BM_StridedInnerProduct, std::complex<double>, true) STRIDED_INNER_PRODUCT_SIZES ->Name("StridedProduct_cdouble");
#undef STRIDED_INNER_PRODUCT_SIZES
// clang-format on
