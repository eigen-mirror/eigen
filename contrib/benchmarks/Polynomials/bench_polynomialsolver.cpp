// Benchmarks for PolynomialSolver::compute on batches of random polynomials.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <contrib/Eigen/Polynomials>

#include <complex>
#include <vector>

using namespace Eigen;

namespace {

constexpr int kBatch = 64;

// Largest componentwise backward error |p(z)| / sum_k |a_k| |z|^k over the roots, in units of eps, evaluated in
// long double so that the check does not share the solver's rounding.
template <typename Scalar, int Deg, typename Polynomial>
double maxBackwardError(const PolynomialSolver<Scalar, Deg>& solver, const Polynomial& poly) {
  using Real = typename NumTraits<Scalar>::Real;
  double worst = 0;
  for (Index i = 0; i < solver.roots().size(); ++i) {
    const std::complex<long double> z(solver.roots()[i].real(), solver.roots()[i].imag());
    std::complex<long double> value(0);
    long double magnitude = 0;
    for (Index k = poly.size() - 1; k >= 0; --k) {
      value = value * z + std::complex<long double>(poly[k]);
      magnitude = magnitude * std::abs(z) + std::abs(std::complex<long double>(poly[k]));
    }
    worst = std::max(worst, double(std::abs(value) / magnitude / NumTraits<Real>::epsilon()));
  }
  return worst;
}

template <typename Scalar, int Deg>
static void BM_PolynomialSolver(benchmark::State& state) {
  const Index degree = Deg == Dynamic ? Index(state.range(0)) : Index(Deg);
  using Polynomial = Matrix<Scalar, Deg == Dynamic ? Dynamic : Deg + 1, 1>;
  std::srand(1);
  std::vector<Polynomial, aligned_allocator<Polynomial>> polys;
  for (int b = 0; b < kBatch; ++b) polys.push_back(Polynomial::Random(degree + 1));
  PolynomialSolver<Scalar, Deg> solver;
  for (const Polynomial& poly : polys) {
    solver.compute(poly);
    // A backward-stable root finder stays within a small multiple of degree * eps.
    if (!(maxBackwardError(solver, poly) <= 1024.0 * double(degree))) {
      state.SkipWithError("root backward error too large");
      return;
    }
  }
  for (auto _ : state) {
    for (const Polynomial& poly : polys) {
      solver.compute(poly);
      benchmark::DoNotOptimize(solver.roots().data());
    }
  }
  state.counters["polys/s"] = benchmark::Counter(kBatch, benchmark::Counter::kIsIterationInvariantRate);
}

}  // namespace

BENCHMARK_TEMPLATE(BM_PolynomialSolver, float, Dynamic)->Arg(5)->Arg(10)->Arg(20)->Arg(50);
BENCHMARK_TEMPLATE(BM_PolynomialSolver, double, Dynamic)->Arg(5)->Arg(10)->Arg(20)->Arg(50)->Arg(100);
BENCHMARK_TEMPLATE(BM_PolynomialSolver, std::complex<double>, Dynamic)->Arg(5)->Arg(20);
BENCHMARK_TEMPLATE(BM_PolynomialSolver, double, 5);
BENCHMARK_TEMPLATE(BM_PolynomialSolver, std::complex<double>, 5);
