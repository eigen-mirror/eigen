// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <cstdlib>

template <typename Scalar, int Order, int Operation, bool Unit = true>
static void BM_ScaledUnitTriangular(benchmark::State& state) {
  using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Order>;
  using Real = typename Eigen::NumTraits<Scalar>::Real;
  const Eigen::Index n = state.range(0);
  std::srand(1);
  const MatrixType a = MatrixType::Random(n, n), b = MatrixType::Random(n, n);
  const Scalar alpha = Scalar(3.14159265358979323846) + Eigen::internal::random<Scalar>();
  constexpr int Mode = Unit ? Eigen::UnitLower : Eigen::Lower;
  const auto triangle = a.template triangularView<Mode>();
  MatrixType expected = MatrixType::Zero(n, n), dst = MatrixType::Zero(n, n);
  Real budget = 0;
  for (Eigen::Index j = 0; j < n; ++j)
    for (Eigen::Index i = 0; i < n; ++i) {
      Real sumAbs = 0;
      for (Eigen::Index k = 0; k <= i; ++k) {
        const Scalar left = Unit && k == i ? Scalar(1) : a(i, k);
        expected(i, j) += left * b(k, j);
        sumAbs += Eigen::numext::abs(left) * Eigen::numext::abs(b(k, j));
      }
      expected(i, j) = alpha * expected(i, j);
      budget = (std::max)(budget, Eigen::numext::abs(alpha) * sumAbs);
    }
  EIGEN_IF_CONSTEXPR (Operation == 0)
    dst.noalias() = alpha * (triangle * b);
  else EIGEN_IF_CONSTEXPR (Operation == 1)
    dst.noalias() += alpha * (triangle * b);
  else
    dst.noalias() -= alpha * (triangle * b);
  EIGEN_IF_CONSTEXPR (Operation == 2) expected = -expected;
  const Real bound = Real(64 * (n + 1)) * Eigen::NumTraits<Real>::epsilon() * (Real(1) + budget);
  if (!((dst - expected).cwiseAbs().template maxCoeff<Eigen::PropagateNaN>() <= bound)) {
    state.SkipWithError("Scaled triangular product differs from scalar reference");
    return;
  }
  for (auto _ : state) {
    EIGEN_IF_CONSTEXPR (Operation == 0)
      dst.noalias() = alpha * (triangle * b);
    else EIGEN_IF_CONSTEXPR (Operation == 1)
      dst.noalias() += alpha * (triangle * b);
    else
      dst.noalias() -= alpha * (triangle * b);
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }
}

#define SCALED_UNIT_BENCH(Scalar, Order, Operation) \
  BENCHMARK(BM_ScaledUnitTriangular<Scalar, Order, Operation>)->Arg(4)->Arg(24)->Arg(64)->Arg(256)->Arg(512)

SCALED_UNIT_BENCH(double, Eigen::ColMajor, 0);
SCALED_UNIT_BENCH(double, Eigen::ColMajor, 1);
SCALED_UNIT_BENCH(double, Eigen::ColMajor, 2);
SCALED_UNIT_BENCH(double, Eigen::RowMajor, 0);
SCALED_UNIT_BENCH(double, Eigen::RowMajor, 1);
SCALED_UNIT_BENCH(double, Eigen::RowMajor, 2);
SCALED_UNIT_BENCH(std::complex<double>, Eigen::ColMajor, 0);
SCALED_UNIT_BENCH(std::complex<double>, Eigen::ColMajor, 1);
SCALED_UNIT_BENCH(std::complex<double>, Eigen::ColMajor, 2);
SCALED_UNIT_BENCH(std::complex<double>, Eigen::RowMajor, 0);
SCALED_UNIT_BENCH(std::complex<double>, Eigen::RowMajor, 1);
SCALED_UNIT_BENCH(std::complex<double>, Eigen::RowMajor, 2);
BENCHMARK(BM_ScaledUnitTriangular<double, Eigen::ColMajor, 0, false>)->Arg(4)->Arg(64)->Arg(256);

#undef SCALED_UNIT_BENCH
