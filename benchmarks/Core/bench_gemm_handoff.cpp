// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

#include <chrono>

using namespace Eigen;

#ifndef SCALAR
#define SCALAR float
#endif

using Scalar = SCALAR;
using Clock = std::chrono::steady_clock;

// Handoff rewrites the operands on the core, as a factorization does, then times product + read; ReadOnly is its base.
enum HandoffMode { Repeated, Handoff, ReadOnly };

// Sampled coefficients of c = a * b: |c(i,j) - sum_q a(i,q) b(q,j)| <= 8 k eps sum_q |a(i,q)| |b(q,j)|.
template <typename Lhs, typename Rhs, typename Res>
static bool product_ok(const Lhs& a, const Rhs& b, const Res& c) {
  for (Index i : {Index(0), a.rows() / 2, a.rows() - 1}) {
    for (Index j : {Index(0), b.cols() / 2, b.cols() - 1}) {
      Scalar expected(0);
      typename NumTraits<Scalar>::Real scale(0);
      for (Index q = 0; q < a.cols(); ++q) {
        expected += a(i, q) * b(q, j);
        scale += numext::abs(a(i, q)) * numext::abs(b(q, j));
      }
      if (!(numext::abs(c(i, j) - expected) <= 8 * a.cols() * NumTraits<Scalar>::epsilon() * scale)) return false;
    }
  }
  return true;
}

template <int Order, int ResOrder, HandoffMode Mode>
static void BM_GemmHandoff(benchmark::State& state) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using Res = Matrix<Scalar, Dynamic, Dynamic, ResOrder>;
  const Index m = state.range(0), n = state.range(1), k = state.range(2);
  const Mat a0 = Mat::Random(m, k), b0 = Mat::Random(k, n);
  const Res c0 = Res::Random(m, n);
  Mat a = a0, b = b0;
  Res c = c0;
  c.noalias() = a * b;
  if (!product_ok(a, b, c)) {
    state.SkipWithError("wrong product");
    return;
  }
  for (auto _ : state) {
    if (Mode == Repeated) {
      c.noalias() = a * b;
      benchmark::DoNotOptimize(c.data());
      benchmark::ClobberMemory();
      continue;
    }
    a = a0;
    b = b0;
    c = c0;
    benchmark::ClobberMemory();
    const auto t0 = Clock::now();
    if (Mode == Handoff) c.noalias() = a * b;
    Scalar s = c.sum();
    benchmark::DoNotOptimize(s);
    const auto t1 = Clock::now();
    state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
  }
}

// The NEON small-product bound on its domain m, n <= 128, k <= 256, two-row results, and shallow products past it.
// clang-format off
#define HANDOFF_SIZES \
  ->ArgsProduct({{12, 24, 48, 96, 128}, {3, 6, 12, 24, 48, 96, 128}, {8, 32, 128, 256}}) \
  ->ArgsProduct({{2}, {16, 64, 128}, {128, 256, 512, 2048, 16384}}) \
  ->ArgsProduct({{200, 256, 340}, {3, 4, 8}, {8, 16}}) \
  ->ArgsProduct({{6, 8}, {425}, {2, 3}})
// clang-format on

#define HANDOFF_LAYOUT(Order, ResOrder)                                                        \
  BENCHMARK_TEMPLATE(BM_GemmHandoff, Order, ResOrder, Repeated) HANDOFF_SIZES;                 \
  BENCHMARK_TEMPLATE(BM_GemmHandoff, Order, ResOrder, Handoff) HANDOFF_SIZES->UseManualTime(); \
  BENCHMARK_TEMPLATE(BM_GemmHandoff, Order, ResOrder, ReadOnly) HANDOFF_SIZES->UseManualTime()

HANDOFF_LAYOUT(ColMajor, ColMajor);
HANDOFF_LAYOUT(RowMajor, ColMajor);
HANDOFF_LAYOUT(ColMajor, RowMajor);
HANDOFF_LAYOUT(RowMajor, RowMajor);
