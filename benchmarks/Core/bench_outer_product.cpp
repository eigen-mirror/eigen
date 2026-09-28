// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

template <int Operation, typename Dst, typename Lhs, typename Rhs>
static void outer_product(Dst& dst, const Lhs& lhs, const Rhs& rhs) {
  if (Operation == 0) dst.noalias() = lhs * rhs;
  if (Operation == 1) dst.noalias() += lhs * rhs;
  if (Operation == 2) dst.noalias() -= lhs * rhs;
}

template <typename LhsScalar, typename RhsScalar, int Order, int Operation>
static void BM_OuterProduct(benchmark::State& state) {
  using Scalar = typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType;
  using Real = typename NumTraits<Scalar>::Real;
  using Lhs = Matrix<LhsScalar, Dynamic, 1>;
  using Rhs = Matrix<RhsScalar, 1, Dynamic>;
  static_assert(internal::product_type<Lhs, Rhs>::value == OuterProduct, "Exercise the outer-product selector");
  const Index rows = state.range(0), cols = state.range(1);
  const Lhs lhs = Lhs::Random(rows);
  const Rhs rhs = Rhs::Random(cols);
  Matrix<Scalar, Dynamic, Dynamic, Order> dst =
      Matrix<Scalar, Dynamic, Dynamic, Order>::Constant(rows, cols, Scalar(1));
  outer_product<Operation>(dst, lhs, rhs);
  for (Index j = 0; j < cols; ++j) {
    for (Index i = 0; i < rows; ++i) {
      const Scalar value = lhs(i) * rhs(j);
      const Scalar expected = Operation == 0 ? value : Operation == 1 ? Scalar(1) + value : Scalar(1) - value;
      const Real bound = Real(8) * NumTraits<Real>::epsilon() * (Real(1) + numext::abs(value));
      if (!(numext::abs(dst(i, j) - expected) <= bound)) {
        state.SkipWithError("Outer product disagrees with the scalar reference");
        return;
      }
    }
  }
  for (auto _ : state) {
    outer_product<Operation>(dst, lhs, rhs);
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * rows * cols);
}

// Include both sides of the small-assignment threshold and rectangular tails.
#define OUTER_SIZES      \
  ->Args({3, 3})         \
      ->Args({4, 4})     \
      ->Args({8, 8})     \
      ->Args({15, 15})   \
      ->Args({16, 16})   \
      ->Args({17, 17})   \
      ->Args({32, 32})   \
      ->Args({128, 128}) \
      ->Args({3, 16})    \
      ->Args({16, 3})    \
      ->Args({3, 17})    \
      ->Args({17, 3})
#define OUTER_OPERATIONS(LHS, RHS, ORDER)                              \
  BENCHMARK_TEMPLATE(BM_OuterProduct, LHS, RHS, ORDER, 0) OUTER_SIZES; \
  BENCHMARK_TEMPLATE(BM_OuterProduct, LHS, RHS, ORDER, 1) OUTER_SIZES; \
  BENCHMARK_TEMPLATE(BM_OuterProduct, LHS, RHS, ORDER, 2) OUTER_SIZES
#define OUTER_LAYOUTS(LHS, RHS)         \
  OUTER_OPERATIONS(LHS, RHS, ColMajor); \
  OUTER_OPERATIONS(LHS, RHS, RowMajor)

OUTER_LAYOUTS(float, float);
OUTER_LAYOUTS(double, double);
OUTER_LAYOUTS(std::complex<float>, std::complex<float>);
OUTER_LAYOUTS(std::complex<double>, std::complex<double>);
OUTER_LAYOUTS(float, std::complex<float>);
OUTER_LAYOUTS(std::complex<float>, float);
OUTER_LAYOUTS(double, std::complex<double>);
OUTER_LAYOUTS(std::complex<double>, double);

#undef OUTER_LAYOUTS
#undef OUTER_OPERATIONS
#undef OUTER_SIZES
