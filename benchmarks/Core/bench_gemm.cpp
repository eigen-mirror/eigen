// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

#ifndef SCALAR
#define SCALAR float
#endif

typedef SCALAR Scalar;
typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

// A complex multiply-add is four real multiplies and four real adds.
static constexpr double kFlopsPerMulAdd = NumTraits<Scalar>::IsComplex ? 8.0 : 2.0;

template <typename A, typename B, typename C>
EIGEN_DONT_INLINE void gemm(const A& a, const B& b, C& c) {
  c.noalias() += a * b;
}

static void add_gflops_counter(benchmark::State& state) {
  const double m = state.range(0), n = state.range(1), p = state.range(2);
  state.counters["GFLOPS"] = benchmark::Counter(
      kFlopsPerMulAdd * m * n * p, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// Sampled coefficients of c = a * b: |c(i,j) - sum_q a(i,q) b(q,j)| <= 8 p eps sum_q |a(i,q)| |b(q,j)|, p the depth.
template <typename Lhs, typename Rhs>
static bool gemm_result_ok(const Lhs& a, const Rhs& b, const Mat& c) {
  for (Index i : {Index(0), a.rows() / 2, a.rows() - 1}) {
    for (Index j : {Index(0), b.cols() / 2, b.cols() - 1}) {
      Scalar expected(0);
      typename NumTraits<Scalar>::Real scale(0);
      for (Index p = 0; p < a.cols(); ++p) {
        expected += a(i, p) * b(p, j);
        scale += numext::abs(a(i, p)) * numext::abs(b(p, j));
      }
      if (!(numext::abs(c(i, j) - expected) <= 8 * a.cols() * NumTraits<Scalar>::epsilon() * scale)) return false;
    }
  }
  return true;
}

// c += a * b with a stored in LhsOrder and b in RhsOrder; c is column-major.
template <int LhsOrder, int RhsOrder = ColMajor>
static void eigen_gemm(benchmark::State& state) {
  int m = state.range(0);
  int n = state.range(1);
  int p = state.range(2);
  Matrix<Scalar, Dynamic, Dynamic, LhsOrder> a(m, p);
  a.setRandom();
  Matrix<Scalar, Dynamic, Dynamic, RhsOrder> b(p, n);
  b.setRandom();
  Mat c = Mat::Zero(m, n);
  gemm(a, b, c);
  if (!gemm_result_ok(a, b, c)) {
    state.SkipWithError("GEMM result check failed");
    return;
  }
  for (auto _ : state) {
    c.setZero();
    gemm(a, b, c);
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  add_gflops_counter(state);
}

static void BM_EigenGemm(benchmark::State& state) { eigen_gemm<ColMajor>(state); }
static void BM_EigenGemmRowMajorLhs(benchmark::State& state) { eigen_gemm<RowMajor>(state); }
static void BM_EigenGemmRowMajorRhs(benchmark::State& state) { eigen_gemm<ColMajor, RowMajor>(state); }

// clang-format off
#define GEMM_SIZES \
    ->Args({8, 8, 8})->Args({16, 16, 16})->Args({32, 32, 32}) \
    ->Args({64, 64, 64})->Args({96, 96, 96})->Args({128, 128, 128}) \
    ->Args({160, 160, 160})->Args({192, 192, 192})->Args({224, 224, 224}) \
    ->Args({256, 256, 256})->Args({288, 288, 288})->Args({320, 320, 320}) \
    ->Args({384, 384, 384})->Args({448, 448, 448})->Args({512, 512, 512}) \
    ->Args({768, 768, 768})->Args({1024, 1024, 1024})->Args({1536, 1536, 1536}) \
    ->Args({2048, 2048, 2048})->Args({4096, 4096, 4096}) \
    /* Non-square sizes */ \
    ->Args({64, 64, 1024})->Args({1024, 64, 64})->Args({64, 1024, 64}) \
    ->Args({256, 256, 1024})->Args({1024, 256, 256}) \
    /* Tall-skinny sizes that straddle the computeProductBlockingSizes cache */ \
    /* threshold (k*n*sizeof(Lhs) near L1). mc blocking only takes effect when */ \
    /* m is large; hitting the threshold requires sweeping small n=k with large m. */ \
    ->Args({4096, 96, 96})->Args({4096, 128, 128})->Args({4096, 144, 144}) \
    ->Args({4096, 160, 160})->Args({4096, 176, 176})->Args({8192, 128, 128}) \
    /* Small and thin products on both sides of the SME backend's NEON small-product bound. */ \
    ->Args({17, 17, 17})->Args({20, 20, 20})->Args({25, 25, 25})->Args({28, 28, 28})->Args({31, 31, 31}) \
    ->Args({4, 32, 32})->Args({8, 32, 32})->Args({32, 4, 32})->Args({32, 8, 32})->Args({16, 16, 32}) \
    ->Args({24, 24, 48})->Args({12, 12, 64})->Args({2, 64, 64})->Args({8, 64, 128})->Args({64, 8, 128})

// Depth sweeps through the packers' transposer, which packs b and a row-major a; their leading dimension is k, so
// these also sweep the source stride. At SVL 512, k = 41 (complex<double>) and 81 (double) end in a paired step and a
// tail after the four-chunk fills, and 120 and 136 straddle the float transposer entry depth.
#define DEPTH_SWEEP_SIZES \
    ->ArgsProduct({{64}, {64}, {41, 81, 120, 136, 256, 512, 2048, 4096}}) \
    ->ArgsProduct({{2048}, {64}, {64, 128, 256, 512, 1024}}) \
    ->Args({128, 128, 2048})->Args({256, 256, 2048})->Args({4096, 64, 512})->Args({2048, 128, 512})

// Tail panels: at SVL 512, m = 60, 120 and 252 leave a's last panel between one and two vectors wide (28, 24 and 28
// floats; 12 doubles for 60 and 252), and n = 60 leaves b's. A column-major b and a row-major a pack through the ZA
// transposer, a row-major b through the copy; a column-major a is mostly read in place.
#define TAIL_PANEL_SIZES \
    ->Args({60, 60, 1024})->Args({60, 64, 1024})->Args({64, 60, 1024})->Args({120, 120, 1024})->Args({252, 252, 252})
// clang-format on

BENCHMARK(BM_EigenGemm) GEMM_SIZES DEPTH_SWEEP_SIZES TAIL_PANEL_SIZES;
BENCHMARK(BM_EigenGemmRowMajorLhs)->Args({256, 256, 256})->Args({1024, 1024, 1024}) DEPTH_SWEEP_SIZES TAIL_PANEL_SIZES;
BENCHMARK(BM_EigenGemmRowMajorRhs)
    ->Args({64, 64, 1024})
    ->Args({256, 256, 256})
    ->Args({1024, 1024, 1024}) TAIL_PANEL_SIZES;

// The reference below calls sgemm_/dgemm_ directly, so it is real-only.
#if defined(HAVE_BLAS) && !defined(EIGEN_BENCH_COMPLEX_SCALAR)
extern "C" {
#include <Eigen/src/Core/util/blas.h>
}

// c += a * b, matching gemm() above.
static void blas_gemm(EIGEN_BLAS_INT m, EIGEN_BLAS_INT n, EIGEN_BLAS_INT p, const float* a, const float* b, float* c) {
  const char notrans = 'N';
  const float one = 1;
  EIGEN_BLAS_SYM(sgemm)(&notrans, &notrans, &m, &n, &p, &one, a, &m, b, &p, &one, c, &m);
}

static void blas_gemm(EIGEN_BLAS_INT m, EIGEN_BLAS_INT n, EIGEN_BLAS_INT p, const double* a, const double* b,
                      double* c) {
  const char notrans = 'N';
  const double one = 1;
  EIGEN_BLAS_SYM(dgemm)(&notrans, &notrans, &m, &n, &p, &one, a, &m, b, &p, &one, c, &m);
}

static void BM_BlasGemm(benchmark::State& state) {
  int m = state.range(0);
  int n = state.range(1);
  int p = state.range(2);
  Mat a(m, p);
  a.setRandom();
  Mat b(p, n);
  b.setRandom();
  Mat c = Mat::Zero(m, n);
  for (auto _ : state) {
    c.setZero();
    blas_gemm(m, n, p, a.data(), b.data(), c.data());
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  add_gflops_counter(state);
}
BENCHMARK(BM_BlasGemm) GEMM_SIZES;
#endif
