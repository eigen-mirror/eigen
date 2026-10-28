// Small fixed-size products for every storage order of the operands and the destination. An assignment evaluates the
// coefficient-based product along the destination's storage order, which vectorizes when the operand that runs along
// it is contiguous: the lhs for a column-major destination, the rhs for a row-major one.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

#ifndef SCALAR
#define SCALAR float
#endif

using Scalar = SCALAR;
using RealScalar = NumTraits<Scalar>::Real;

template <typename Dst, typename Lhs, typename Rhs>
static bool matches_reference(const Dst& dst, const Dst& before, const Lhs& lhs, const Rhs& rhs) {
  for (Index i = 0; i < dst.rows(); ++i) {
    for (Index j = 0; j < dst.cols(); ++j) {
      Scalar sum = before(i, j);
      RealScalar scale = numext::abs(before(i, j));
      for (Index k = 0; k < lhs.cols(); ++k) {
        sum += lhs(i, k) * rhs(k, j);
        scale += numext::abs(lhs(i, k)) * numext::abs(rhs(k, j));
      }
      if (!(numext::abs(dst(i, j) - sum) <= RealScalar(8) * NumTraits<RealScalar>::epsilon() * scale)) return false;
    }
  }
  return true;
}

// dst = lhs * rhs (Accumulate == false) or dst += lhs * rhs.
template <int N, int LhsOrder, int RhsOrder, int DstOrder, bool Accumulate>
static void BM_ProductOrder(benchmark::State& state) {
  const Matrix<Scalar, N, N, LhsOrder> lhs = Matrix<Scalar, N, N, LhsOrder>::Random();
  const Matrix<Scalar, N, N, RhsOrder> rhs = Matrix<Scalar, N, N, RhsOrder>::Random();
  using Dst = Matrix<Scalar, N, N, DstOrder>;
  Dst dst = Accumulate ? Dst(Dst::Random()) : Dst(Dst::Zero());
  const Dst before = dst;
  if (Accumulate)
    dst.noalias() += lhs * rhs;
  else
    dst.noalias() = lhs * rhs;
  if (!matches_reference(dst, before, lhs, rhs)) {
    state.SkipWithError("Product disagrees with the scalar reference");
    return;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(lhs.data());
    benchmark::DoNotOptimize(rhs.data());
    if (Accumulate)
      dst.noalias() += lhs * rhs;
    else
      dst.noalias() = lhs * rhs;
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * N * N * N);
}

// dst -= src.col(k) * src.row(k): a rank-1 update with a strided column and a contiguous row when Order is RowMajor.
template <int N, int Order>
static void BM_RankOneUpdate(benchmark::State& state) {
  using Mat = Matrix<Scalar, N, N, Order>;
  const Mat src = Mat::Random();
  Mat dst = Mat::Random();
  const Mat before = dst;
  dst.noalias() -= src.col(1) * src.row(2);
  if (!matches_reference(dst, before, -src.col(1), src.row(2))) {
    state.SkipWithError("Rank-1 update disagrees with the scalar reference");
    return;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(src.data());
    dst.noalias() -= src.col(1) * src.row(2);
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * N * N);
}

#define PRODUCT_ORDERS(N, ACC)                                               \
  BENCHMARK_TEMPLATE(BM_ProductOrder, N, ColMajor, ColMajor, ColMajor, ACC); \
  BENCHMARK_TEMPLATE(BM_ProductOrder, N, ColMajor, ColMajor, RowMajor, ACC); \
  BENCHMARK_TEMPLATE(BM_ProductOrder, N, ColMajor, RowMajor, ColMajor, ACC); \
  BENCHMARK_TEMPLATE(BM_ProductOrder, N, ColMajor, RowMajor, RowMajor, ACC); \
  BENCHMARK_TEMPLATE(BM_ProductOrder, N, RowMajor, ColMajor, ColMajor, ACC); \
  BENCHMARK_TEMPLATE(BM_ProductOrder, N, RowMajor, ColMajor, RowMajor, ACC); \
  BENCHMARK_TEMPLATE(BM_ProductOrder, N, RowMajor, RowMajor, ColMajor, ACC); \
  BENCHMARK_TEMPLATE(BM_ProductOrder, N, RowMajor, RowMajor, RowMajor, ACC)

PRODUCT_ORDERS(3, false);
PRODUCT_ORDERS(4, false);
PRODUCT_ORDERS(8, false);
PRODUCT_ORDERS(3, true);
PRODUCT_ORDERS(4, true);
PRODUCT_ORDERS(8, true);

BENCHMARK_TEMPLATE(BM_RankOneUpdate, 4, ColMajor);
BENCHMARK_TEMPLATE(BM_RankOneUpdate, 4, RowMajor);
BENCHMARK_TEMPLATE(BM_RankOneUpdate, 8, ColMajor);
BENCHMARK_TEMPLATE(BM_RankOneUpdate, 8, RowMajor);
BENCHMARK_TEMPLATE(BM_RankOneUpdate, 16, ColMajor);
BENCHMARK_TEMPLATE(BM_RankOneUpdate, 16, RowMajor);
