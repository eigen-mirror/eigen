// Benchmarks for fixed-size products: batches (4x4 transform of 4xN points, arrays of 3x3 products), critical for
// PCL, ROS, Sophus, Drake which use small matrices extensively, and the M x K times K x N shape sweep behind
// EIGEN_FIXED_SIZE_GEMM_TO_COEFFBASED_THRESHOLD (#3119).
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// The *_gemm targets zero every coeff-based bound, so BM_FixedProduct there runs the GEMM path at every shape.
#ifdef EIGEN_BENCH_FORCE_GEMM
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 2
#define EIGEN_GEMM_TO_COEFFBASED_THRESHOLD 0
#define EIGEN_FIXED_SIZE_GEMM_TO_COEFFBASED_THRESHOLD 0
#define EIGEN_SME_FIXED_SIZE_GEMM_TO_COEFFBASED_THRESHOLD 0
#define EIGEN_SME_GEMM_TO_COEFFBASED_OUTPUT_AREA_THRESHOLD(Scalar) 0
#endif

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

#ifndef SCALAR
#define SCALAR float
#endif

typedef SCALAR Scalar;
#ifdef EIGEN_BENCH_FORCE_GEMM
static_assert(internal::product_type<Matrix<Scalar, 16, 16>, Matrix<Scalar, 16, 16>>::value == GemmProduct,
              "the *_gemm targets must not leave any fixed-size product on the coeff-based path");
#endif
using RealScalar = NumTraits<Scalar>::Real;

// A complex multiply-add is four real multiplies and four real adds.
static constexpr double kFlopsPerMulAdd = NumTraits<Scalar>::IsComplex ? 8.0 : 2.0;

#ifndef EIGEN_BENCH_FORCE_GEMM
// --- Batch transform: Matrix4 * Matrix<4,N> ---
static void BM_BatchTransform4xN(benchmark::State& state) {
  int N = state.range(0);
  typedef Matrix<Scalar, 4, 4> Mat4;
  typedef Matrix<Scalar, 4, Dynamic> MatXN;

  Mat4 transform = Mat4::Random();
  MatXN points = MatXN::Random(4, N);
  MatXN result(4, N);

  for (auto _ : state) {
    result.noalias() = transform * points;
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] =
      benchmark::Counter(2.0 * 4 * 4 * N, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// --- Fixed 3x3 batch operations (common in point cloud processing) ---
static void BM_Batch3x3Gemm(benchmark::State& state) {
  int count = state.range(0);
  typedef Matrix<Scalar, 3, 3> Mat3;

  std::vector<Mat3> a(count), b(count), c(count);
  for (int i = 0; i < count; ++i) {
    a[i] = Mat3::Random();
    b[i] = Mat3::Random();
  }

  for (auto _ : state) {
    for (int i = 0; i < count; ++i) {
      c[i].noalias() = a[i] * b[i];
    }
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] =
      benchmark::Counter(2.0 * 27 * count, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// Batch 4xN transform
BENCHMARK(BM_BatchTransform4xN)->Arg(1)->Arg(4)->Arg(8)->Arg(16)->Arg(64);

// Batch 3x3 GEMM
BENCHMARK(BM_Batch3x3Gemm)->Arg(100)->Arg(1000)->Arg(10000);
#endif

// --- Fixed-size product dispatch: Matrix<Scalar, M, K> * Matrix<Scalar, K, N> ---
template <int M, int N, int K, bool Lazy>
static void fixed_product(benchmark::State& state) {
  using Lhs = Matrix<Scalar, M, K>;
  using Rhs = Matrix<Scalar, K, N>;
  using Res = Matrix<Scalar, M, N>;
  Lhs a = Lhs::Random();
  Rhs b = Rhs::Random();
  Res c, ref;

  ref.noalias() = a.lazyProduct(b);
  if (Lazy)
    c.noalias() = a.lazyProduct(b);
  else
    c.noalias() = a * b;
  const RealScalar err = (c - ref).norm();
  const RealScalar bound = RealScalar(8 * K) * NumTraits<RealScalar>::epsilon() * a.norm() * b.norm();
  if (!(err <= bound)) {
    state.SkipWithError("product differs from lazyProduct");
    return;
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(a.data());
    benchmark::DoNotOptimize(b.data());
    if (Lazy)
      c.noalias() = a.lazyProduct(b);
    else
      c.noalias() = a * b;
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  // Compile-time selection only: in an SME build the output-area bound can still reroute a GEMM product at run time.
  state.counters["gemm"] = !Lazy && internal::product_type<Lhs, Rhs>::value == GemmProduct;
  state.counters["GFLOPS"] = benchmark::Counter(
      kFlopsPerMulAdd * M * N * K, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// Takes whichever path this translation unit's thresholds select.
template <int M, int N, int K>
static void BM_FixedProduct(benchmark::State& state) {
  fixed_product<M, N, K, false>(state);
}

template <int M, int N, int K>
static void BM_FixedLazyProduct(benchmark::State& state) {
  fixed_product<M, N, K, true>(state);
}

// X(M, N, K): square n x n x n, then M = N against K, then rectangular outputs. Every non-square shape has product
// type GemmProduct under the default EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD, so the fixed-size threshold decides it.
// clang-format off
#define EIGEN_BENCH_FIXED_PRODUCT_SHAPES(X)                                                                         \
  X(4, 4, 4) X(5, 5, 5) X(6, 6, 6) X(7, 7, 7) X(8, 8, 8) X(9, 9, 9) X(10, 10, 10) X(11, 11, 11) X(12, 12, 12)       \
  X(13, 13, 13) X(14, 14, 14) X(15, 15, 15) X(16, 16, 16) X(17, 17, 17) X(18, 18, 18) X(19, 19, 19) X(20, 20, 20)   \
  X(21, 21, 21) X(22, 22, 22) X(23, 23, 23) X(24, 24, 24) X(25, 25, 25) X(26, 26, 26) X(27, 27, 27) X(28, 28, 28)   \
  X(29, 29, 29) X(30, 30, 30) X(31, 31, 31) X(32, 32, 32)                                                           \
  X(2, 2, 8) X(2, 2, 16) X(2, 2, 32) X(2, 2, 64) X(2, 2, 128)                                                       \
  X(4, 4, 8) X(4, 4, 16) X(4, 4, 32) X(4, 4, 64) X(4, 4, 128)                                                       \
  X(8, 8, 2) X(8, 8, 4) X(8, 8, 16) X(8, 8, 32) X(8, 8, 64) X(8, 8, 128)                                            \
  X(12, 12, 2) X(12, 12, 4) X(12, 12, 8) X(12, 12, 16) X(12, 12, 32) X(12, 12, 64) X(12, 12, 128)                   \
  X(16, 16, 2) X(16, 16, 4) X(16, 16, 8) X(16, 16, 32) X(16, 16, 64) X(16, 16, 128)                                 \
  X(24, 24, 2) X(24, 24, 4) X(24, 24, 8) X(24, 24, 16) X(24, 24, 32) X(24, 24, 64) X(24, 24, 128)                   \
  X(32, 32, 2) X(32, 32, 4) X(32, 32, 8) X(32, 32, 16) X(32, 32, 64) X(32, 32, 128)                                 \
  X(4, 32, 8) X(4, 32, 16) X(4, 32, 32) X(32, 4, 8) X(32, 4, 16) X(32, 4, 32)                                       \
  X(8, 32, 8) X(8, 32, 16) X(8, 32, 32) X(32, 8, 8) X(32, 8, 16) X(32, 8, 32)                                       \
  X(2, 64, 16) X(64, 2, 16) X(16, 32, 16) X(32, 16, 16)                                                             \
  X(2, 2, 256) X(2, 2, 512) X(4, 4, 256) X(4, 4, 512)                                                               \
  X(8, 8, 20) X(8, 8, 24) X(8, 8, 28) X(16, 16, 20) X(16, 16, 24) X(16, 16, 28)                                     \
  X(32, 32, 20) X(32, 32, 24) X(32, 32, 28)                                                                         \
  X(48, 48, 2) X(48, 48, 4) X(48, 48, 8) X(48, 48, 16) X(48, 48, 32)                                                \
  X(64, 64, 2) X(64, 64, 4) X(64, 64, 8) X(64, 64, 16) X(64, 64, 32)
// clang-format on

#define EIGEN_BENCH_REGISTER_FIXED_PRODUCT(M, N, K) \
  BENCHMARK_TEMPLATE(BM_FixedProduct, M, N, K);     \
  BENCHMARK_TEMPLATE(BM_FixedLazyProduct, M, N, K);
EIGEN_BENCH_FIXED_PRODUCT_SHAPES(EIGEN_BENCH_REGISTER_FIXED_PRODUCT)
