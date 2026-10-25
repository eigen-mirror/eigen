// Benchmarks for DiscreteAlgebraicRiccatiSolver::compute on well-conditioned random problems.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <contrib/Eigen/Riccati>

#include <complex>
#include <cstdlib>

using namespace Eigen;

namespace {

template <typename Scalar>
struct DareProblem {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  MatrixType A, B, Q, R, E, S;
};

// A = U T U^* with U unitary and T upper triangular with eigenvalue moduli in [0.1, 0.8], except for n/2 well
// controllable ones in [1.25, 2]; Q, R positive definite; E = I + O(1/2) and ||S||_F <= 1/2 for generalized problems.
template <typename Scalar>
DareProblem<Scalar> makeProblem(Index m, Index n, bool generalized) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using MatrixType = typename DareProblem<Scalar>::MatrixType;
  std::srand(1);
  MatrixType T = MatrixType::Random(m, m).template triangularView<StrictlyUpper>();
  T *= RealScalar(0.5) / numext::sqrt(RealScalar(m));
  const Index unstable = n / 2;
  for (Index i = 0; i < m; ++i) {
    const RealScalar modulus =
        i < unstable ? internal::random<RealScalar>(1.25, 2) : internal::random<RealScalar>(0.1, 0.8);
    T(i, i) = (i % 2 ? Scalar(modulus) : Scalar(-modulus));
  }
  const MatrixType U = HouseholderQR<MatrixType>(MatrixType::Random(m, m)).householderQ();
  MatrixType Bu = MatrixType::Random(m, n);
  for (Index i = 0; i < unstable; ++i) Bu(i, i) += Scalar(2);
  const MatrixType C = MatrixType::Random(m, m), D = MatrixType::Random(n, n);
  DareProblem<Scalar> p;
  p.A = U * T * U.adjoint();
  p.B = U * Bu;
  p.Q = C.adjoint() * C + MatrixType::Identity(m, m);
  p.R = D.adjoint() * D + MatrixType::Identity(n, n);
  p.E = MatrixType::Identity(m, m);
  p.S = MatrixType::Zero(m, n);
  if (generalized) {
    p.E += MatrixType::Random(m, m) * (RealScalar(0.5) / numext::sqrt(RealScalar(m)));
    p.S = MatrixType::Random(m, n) * (RealScalar(0.5) / numext::sqrt(RealScalar(m * n)));
    p.A = p.E * p.A;
    p.B = p.E * p.B;
  }
  return p;
}

// ||res||_F / (m eps (||A^* X A||_F + ||E^* X E||_F + ||Q||_F)), the backward error measure of the unit test.
template <typename Scalar>
double normalizedResidual(const DareProblem<Scalar>& p, const typename DareProblem<Scalar>::MatrixType& X) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using MatrixType = typename DareProblem<Scalar>::MatrixType;
  const MatrixType AXB = p.A.adjoint() * X * p.B + p.S;
  const MatrixType res = p.A.adjoint() * X * p.A - p.E.adjoint() * X * p.E -
                         AXB * (p.R + p.B.adjoint() * X * p.B).fullPivLu().solve(AXB.adjoint()) + p.Q;
  const RealScalar scale = (p.A.adjoint() * X * p.A).norm() + (p.E.adjoint() * X * p.E).norm() + p.Q.norm();
  return double(res.norm() / (RealScalar(p.A.rows()) * NumTraits<RealScalar>::epsilon() * scale));
}

// state.range(0) = m, n = max(1, m / 4); Generalized selects the (E, S) form.
template <typename Scalar, bool Generalized>
static void BM_DiscreteAlgebraicRiccatiSolver(benchmark::State& state) {
  const Index m = state.range(0);
  const Index n = numext::maxi(Index(1), m / 4);
  const DareProblem<Scalar> p = makeProblem<Scalar>(m, n, Generalized);
  DiscreteAlgebraicRiccatiSolver<typename DareProblem<Scalar>::MatrixType> dare;
  const auto solve = [&] {
    if (Generalized)
      dare.compute(p.A, p.B, p.Q, p.R, p.E, p.S);
    else
      dare.compute(p.A, p.B, p.Q, p.R);
  };
  solve();
  // The unit test bounds this measure by 1000; well-conditioned problems stay near 1.
  if (dare.info() != Success || !(normalizedResidual(p, dare.solution()) <= 1000.0)) {
    state.SkipWithError("DARE solution failed the backward error check");
    return;
  }
  for (auto _ : state) {
    solve();
    benchmark::DoNotOptimize(dare.solution().data());
  }
  state.counters["m"] = double(m);
  state.counters["n"] = double(n);
}

}  // namespace

BENCHMARK_TEMPLATE(BM_DiscreteAlgebraicRiccatiSolver, double, false)->RangeMultiplier(2)->Range(4, 256);
BENCHMARK_TEMPLATE(BM_DiscreteAlgebraicRiccatiSolver, double, true)->RangeMultiplier(4)->Range(4, 64);
BENCHMARK_TEMPLATE(BM_DiscreteAlgebraicRiccatiSolver, float, false)->RangeMultiplier(4)->Range(4, 64);
BENCHMARK_TEMPLATE(BM_DiscreteAlgebraicRiccatiSolver, std::complex<double>, false)->RangeMultiplier(4)->Range(4, 64);
