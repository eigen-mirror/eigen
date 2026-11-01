// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/SparseCore>
#include <contrib/Eigen/AutoDiff>

template <int Order, bool SparseOnLeft>
static void BM_AutoDiffSparseProduct(benchmark::State& state) {
  using AD = Eigen::AutoDiffScalar<Eigen::Vector4d>;
  using Dense = Eigen::Matrix<AD, Eigen::Dynamic, Eigen::Dynamic>;
  const Eigen::Index size = state.range(0), width = state.range(1);
  Eigen::SparseMatrix<AD, Order> sparse(size, size);
  for (Eigen::Index col = 0; col < size; ++col)
    for (Eigen::Index k = 0; k < 8; ++k) {
      AD value(double((col + k) % 5 - 2));
      value.derivatives() << double(k + 1), 0, 1, -1;
      sparse.insert((col + 17 * k) % size, col) = value;
    }
  sparse.makeCompressed();
  Dense dense(SparseOnLeft ? size : width, SparseOnLeft ? width : size);
  for (Eigen::Index col = 0; col < dense.cols(); ++col)
    for (Eigen::Index row = 0; row < dense.rows(); ++row) {
      AD value(double((row + 2 * col) % 7 - 3));
      value.derivatives() = Eigen::Vector4d(1, 2, 3, 4) * double((row - col) % 3);
      dense(row, col) = value;
    }
  Dense result(SparseOnLeft ? size : width, SparseOnLeft ? width : size);
  const auto evaluate = [&] {
    EIGEN_IF_CONSTEXPR (SparseOnLeft)
      result.noalias() = sparse * dense;
    else
      result.noalias() = dense * sparse;
  };
  evaluate();
  AD expected(0);
  expected.derivatives().setZero();
  for (Eigen::Index k = 0; k < size; ++k) {
    EIGEN_IF_CONSTEXPR (SparseOnLeft)
      expected += sparse.coeff(0, k) * dense(k, 0);
    else
      expected += dense(0, k) * sparse.coeff(k, 0);
  }
  if (result(0, 0).value() != expected.value() || !(result(0, 0).derivatives() - expected.derivatives()).isZero(0)) {
    state.SkipWithError("incorrect AutoDiff sparse product");
    return;
  }
  for (auto _ : state) {
    evaluate();
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK_TEMPLATE(BM_AutoDiffSparseProduct, Eigen::ColMajor, true)->Args({128, 16})->Args({1000, 50});
BENCHMARK_TEMPLATE(BM_AutoDiffSparseProduct, Eigen::RowMajor, true)->Args({128, 16})->Args({1000, 50});
BENCHMARK_TEMPLATE(BM_AutoDiffSparseProduct, Eigen::ColMajor, false)->Args({128, 16})->Args({1000, 50});
BENCHMARK_TEMPLATE(BM_AutoDiffSparseProduct, Eigen::RowMajor, false)->Args({128, 16})->Args({1000, 50});
