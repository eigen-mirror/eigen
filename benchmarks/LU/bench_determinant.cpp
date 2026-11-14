// Benchmarks for the determinant family of a computed factorization: the O(n) product of the pivots and its log.
//
// The pivots come in three regimes, since the product separates its exponent only when a partial product leaves the
// normal range: never (InRange), every few hundred pivots (Growing), and at every other pivot (Extreme).
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/LU>

#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <utility>

using namespace Eigen;

enum PivotRegime { InRange = 0, Growing = 1, Extreme = 2 };

// I + E with |E_ij| <= 1 / (2 n) has pivots within a few percent of one, so the regime is set by the scale alone.
template <typename Scalar>
Matrix<Scalar, Dynamic, Dynamic> makeMatrix(Index n, int regime) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  Mat A = Mat::Identity(n, n) + (RealScalar(0.5) / RealScalar(n)) * Mat::Random(n, n);
  if (regime == Growing) A *= RealScalar(8);
  if (regime == Extreme) A *= std::ldexp(RealScalar(1), std::numeric_limits<RealScalar>::max_exponent / 2 + 2);
  return A;
}

// One factorization per scalar, size and regime: the benchmark library calls each benchmark several times.
template <typename Scalar>
const PartialPivLU<Matrix<Scalar, Dynamic, Dynamic>>& factorization(Index n, int regime) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  static std::map<std::pair<Index, int>, PartialPivLU<Mat>> cache;
  auto entry = cache.find({n, regime});
  if (entry == cache.end()) entry = cache.emplace(std::make_pair(n, regime), makeMatrix<Scalar>(n, regime)).first;
  return entry->second;
}

// log|det| from the pivots one at a time, as the reference for every regime.
template <typename Mat>
typename Mat::RealScalar referenceLogAbsDeterminant(const PartialPivLU<Mat>& lu) {
  typename Mat::RealScalar result(0);
  for (Index i = 0; i < lu.rows(); ++i) result += std::log(std::abs(lu.matrixLU().coeff(i, i)));
  return result;
}

template <typename Mat>
bool logMatches(const typename Mat::RealScalar& value, const PartialPivLU<Mat>& lu) {
  using RealScalar = typename Mat::RealScalar;
  const RealScalar reference = referenceLogAbsDeterminant(lu);
  return std::abs(value - reference) <= RealScalar(64) * RealScalar(lu.rows()) * NumTraits<RealScalar>::epsilon() *
                                            (std::max)(RealScalar(1), std::abs(reference));
}

template <typename Mat>
EIGEN_DONT_INLINE typename Mat::Scalar doDeterminant(const PartialPivLU<Mat>& lu) {
  return lu.determinant();
}

template <typename Mat>
EIGEN_DONT_INLINE typename Mat::RealScalar doAbsDeterminant(const PartialPivLU<Mat>& lu) {
  return lu.absDeterminant();
}

template <typename Mat>
EIGEN_DONT_INLINE typename Mat::RealScalar doLogAbsDeterminant(const PartialPivLU<Mat>& lu) {
  return lu.logAbsDeterminant();
}

template <typename Scalar>
static void BM_PartialPivLU_Determinant(benchmark::State& state) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  const PartialPivLU<Mat>& lu = factorization<Scalar>(state.range(0), int(state.range(1)));
  // Only an in-range product has a determinant that the plain product also represents.
  if (state.range(1) == InRange && !logMatches<Mat>(std::log(std::abs(doDeterminant(lu))), lu)) {
    state.SkipWithError("determinant() disagrees with the pivots");
    return;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(&lu);
    Scalar det = doDeterminant(lu);
    benchmark::DoNotOptimize(&det);
  }
  state.SetItemsProcessed(state.iterations() * state.range(0));
}

template <typename Scalar>
static void BM_PartialPivLU_AbsDeterminant(benchmark::State& state) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename Mat::RealScalar;
  const PartialPivLU<Mat>& lu = factorization<Scalar>(state.range(0), int(state.range(1)));
  if (state.range(1) == InRange && !logMatches<Mat>(std::log(doAbsDeterminant(lu)), lu)) {
    state.SkipWithError("absDeterminant() disagrees with the pivots");
    return;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(&lu);
    RealScalar det = doAbsDeterminant(lu);
    benchmark::DoNotOptimize(&det);
  }
  state.SetItemsProcessed(state.iterations() * state.range(0));
}

template <typename Scalar>
static void BM_PartialPivLU_LogAbsDeterminant(benchmark::State& state) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename Mat::RealScalar;
  const PartialPivLU<Mat>& lu = factorization<Scalar>(state.range(0), int(state.range(1)));
  if (!logMatches<Mat>(doLogAbsDeterminant(lu), lu)) {
    state.SkipWithError("logAbsDeterminant() disagrees with the pivots");
    return;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(&lu);
    RealScalar logDet = doLogAbsDeterminant(lu);
    benchmark::DoNotOptimize(&logDet);
  }
  state.SetItemsProcessed(state.iterations() * state.range(0));
}

// The determinant of a small fixed-size triangular view, where a per-factor cost shows undiluted.
template <typename Mat>
EIGEN_DONT_INLINE typename Mat::Scalar doTriangularDeterminant(const Mat& A) {
  return A.template triangularView<Upper>().determinant();
}

template <typename Mat>
static void BM_Triangular_Determinant(benchmark::State& state) {
  using Scalar = typename Mat::Scalar;
  Mat A = Mat::Random();
  A.diagonal().array() += Scalar(2);
  const Scalar expected = A.diagonal().prod();
  if (std::abs(doTriangularDeterminant(A) - expected) > Scalar(8) * NumTraits<Scalar>::epsilon() * std::abs(expected)) {
    state.SkipWithError("determinant() disagrees with the diagonal");
    return;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(A.data());
    Scalar det = doTriangularDeterminant(A);
    benchmark::DoNotOptimize(&det);
  }
  state.SetItemsProcessed(state.iterations() * A.rows());
}

#define EIGEN_BENCH_DETERMINANT_GRID ArgsProduct({{8, 64, 512, 2048}, {InRange, Growing, Extreme}})

BENCHMARK(BM_PartialPivLU_Determinant<float>)->EIGEN_BENCH_DETERMINANT_GRID;
BENCHMARK(BM_PartialPivLU_Determinant<double>)->EIGEN_BENCH_DETERMINANT_GRID;
BENCHMARK(BM_PartialPivLU_Determinant<std::complex<double>>)->EIGEN_BENCH_DETERMINANT_GRID;
BENCHMARK(BM_PartialPivLU_AbsDeterminant<double>)->EIGEN_BENCH_DETERMINANT_GRID;
BENCHMARK(BM_PartialPivLU_AbsDeterminant<std::complex<double>>)->EIGEN_BENCH_DETERMINANT_GRID;
BENCHMARK(BM_PartialPivLU_LogAbsDeterminant<float>)->EIGEN_BENCH_DETERMINANT_GRID;
BENCHMARK(BM_PartialPivLU_LogAbsDeterminant<double>)->EIGEN_BENCH_DETERMINANT_GRID;
BENCHMARK(BM_PartialPivLU_LogAbsDeterminant<std::complex<double>>)->EIGEN_BENCH_DETERMINANT_GRID;
BENCHMARK(BM_Triangular_Determinant<Matrix3d>);
BENCHMARK(BM_Triangular_Determinant<Matrix4d>);
BENCHMARK(BM_Triangular_Determinant<Matrix<double, 8, 8>>);
