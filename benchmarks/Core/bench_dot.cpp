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
  using WideScalar = std::conditional_t<NumTraits<Scalar>::IsComplex, std::complex<long double>, long double>;
  WideScalar expected(0);
  long double magnitude = 0;
  for (Index i = 0; i < n; ++i) {
    const WideScalar term = numext::conj(WideScalar(a[i])) * WideScalar(b[i]);
    expected += term;
    magnitude += numext::abs(term);
  }
  // exp(k*eps)-1 bounds (1+eps)^k-1 even when k*eps >= 1 in a large scalar-only run.
  // Four packet accumulators in the baseline; also account for the scalar reference's rounding.
  const long double depth = n / (4 * internal::packet_traits<Scalar>::size) + 32;
  const long double rounding =
      4 * depth * NumTraits<typename NumTraits<Scalar>::Real>::epsilon() + 4 * n * NumTraits<long double>::epsilon();
  const long double bound = numext::expm1(rounding) * magnitude;
  if (!((numext::isfinite)(bound) && numext::abs(WideScalar(a.dot(b)) - expected) <= bound)) {
    state.SkipWithError("DOT differs from the scalar reference");
    return;
  }
  for (auto _ : state) {
    benchmark::ClobberMemory();
    Scalar d = a.dot(b);
    benchmark::DoNotOptimize(d);
  }
  state.counters["GFLOPS"] = benchmark::Counter(dotFlops<Scalar>(n), benchmark::Counter::kIsIterationInvariantRate,
                                                benchmark::Counter::kIs1000);
}

// clang-format off
#define DOT_SIZES ->Arg(1)->Arg(4)->Arg(16)->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->Arg(8192)->Arg(16384)->Arg(32768)->Arg(65536)->Arg(262144)->Arg(524288)->Arg(1048576)->Arg(2097152)
BENCHMARK(BM_Dot<float>) DOT_SIZES ->Arg(16777216) ->Name("Dot_float");
BENCHMARK(BM_Dot<double>) DOT_SIZES ->Arg(16777216) ->Name("Dot_double");
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

template <typename Scalar, bool Produce>
static void BM_DotLayout(benchmark::State& state) {
  using Vec = Vector<Scalar, Dynamic>;
  const Index n = state.range(0), padding = 128 / sizeof(Scalar);
  Vec a_storage(n + padding), b_storage(n + padding), source = Vec::Random(n);
  Map<Vec> a(a_storage.data() + internal::first_aligned<64>(a_storage.data(), a_storage.size()) +
                 state.range(1) / sizeof(Scalar),
             n);
  Map<Vec> b(b_storage.data() + internal::first_aligned<64>(b_storage.data(), b_storage.size()) +
                 state.range(2) / sizeof(Scalar),
             n);
  a = source.array() + Scalar(0.25);
  b.setRandom();
  long double expected = 0, magnitude = 0;
  for (Index i = 0; i < n; ++i) {
    const long double term = static_cast<long double>(a[i]) * static_cast<long double>(b[i]);
    expected += term;
    magnitude += numext::abs(term);
  }
  const long double bound = 8 * n * NumTraits<Scalar>::epsilon() * magnitude;
  if (!(numext::abs(static_cast<long double>(a.dot(b)) - expected) <= bound)) {
    state.SkipWithError("DOT layout differs from the scalar reference");
    return;
  }
  for (auto _ : state) {
    EIGEN_IF_CONSTEXPR (Produce) a = source.array() + Scalar(0.25);
    benchmark::ClobberMemory();
    Scalar result = a.dot(b);
    benchmark::DoNotOptimize(result);
  }
}

// clang-format off
#define DOT_LAYOUT_SIZES ->ArgsProduct({{64, 256, 1024, 4096, 8192, 16384, 32768, 65536}, {0, 16, 32, 48}, {0, 16}})
BENCHMARK_TEMPLATE(BM_DotLayout, float, false) DOT_LAYOUT_SIZES ->Name("DotLayout_float");
BENCHMARK_TEMPLATE(BM_DotLayout, double, false) DOT_LAYOUT_SIZES ->Name("DotLayout_double");
BENCHMARK_TEMPLATE(BM_DotLayout, float, true) DOT_LAYOUT_SIZES ->Name("DotProduced_float");
BENCHMARK_TEMPLATE(BM_DotLayout, double, true) DOT_LAYOUT_SIZES ->Name("DotProduced_double");
#undef DOT_LAYOUT_SIZES
// clang-format on

// Disjoint supports give an exactly zero dot product.
template <typename Scalar>
static void BM_DotOrthogonal(benchmark::State& state) {
  using Vec = Vector<Scalar, Dynamic>;
  const Index n = state.range(0);
  Vec a = Vec::Zero(n), b = Vec::Zero(n);
  for (Index i = 0; i + 1 < n; i += 2) {
    a[i] = Scalar(1);
    b[i + 1] = Scalar(1);
  }
  if (a.dot(b) != Scalar(0)) {
    state.SkipWithError("DOT of disjoint supports is not zero");
    return;
  }
  for (auto _ : state) {
    benchmark::ClobberMemory();
    Scalar result = a.dot(b);
    benchmark::DoNotOptimize(result);
  }
}

// clang-format off
BENCHMARK_TEMPLATE(BM_DotOrthogonal, float) ->Arg(4096)->Arg(16384)->Arg(65536) ->Name("DotOrthogonal_float");
BENCHMARK_TEMPLATE(BM_DotOrthogonal, double) ->Arg(4096)->Arg(16384)->Arg(65536) ->Name("DotOrthogonal_double");
// clang-format on
