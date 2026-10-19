// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include "../../test/MovableScalar.h"
#include <complex>

using Complex = std::complex<double>;
using NonTrivialScalar = Eigen::MovableScalar<Complex>;

template <typename Scalar, int Capacity>
static void BM_ConservativeResizeAppend(benchmark::State& state) {
  using Vector = Eigen::Array<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor, Capacity, 1>;
  const Eigen::Index size = state.range(0);
  const Scalar value(Complex(1, 2));
  Vector v;
  for (auto _ : state) {
    v.resize(0);
    for (Eigen::Index i = 0; i < size; ++i) {
      v.conservativeResize(i + 1);
      v(i) = value;
      benchmark::DoNotOptimize(v.data());
      benchmark::ClobberMemory();
    }
  }
  for (Eigen::Index i = 0; i < size; ++i) {
    if (Complex(v(i)) != Complex(1, 2)) state.SkipWithError("Appended coefficient was not preserved");
  }
  state.SetItemsProcessed(state.iterations() * size);
}

template <int Storage>
static void BM_ConservativeResizeOuter(benchmark::State& state) {
  using BoundedMatrix = Eigen::Matrix<NonTrivialScalar, Eigen::Dynamic, Eigen::Dynamic, Storage,
                                      Storage == Eigen::RowMajor ? 256 : 3, Storage == Eigen::RowMajor ? 3 : 256>;
  const Eigen::Index size = state.range(0);
  const NonTrivialScalar value(Complex(1, 2));
  BoundedMatrix m;
  for (auto _ : state) {
    m.resize(Storage == Eigen::RowMajor ? 0 : 3, Storage == Eigen::RowMajor ? 3 : 0);
    for (Eigen::Index i = 0; i < size; ++i) {
      m.conservativeResize(Storage == Eigen::RowMajor ? i + 1 : 3, Storage == Eigen::RowMajor ? 3 : i + 1);
      for (Eigen::Index j = 0; j < 3; ++j) m.data()[3 * i + j] = value;
      benchmark::DoNotOptimize(m.data());
      benchmark::ClobberMemory();
    }
  }
  for (Eigen::Index i = 0; i < m.size(); ++i) {
    if (Complex(m.data()[i]) != Complex(1, 2)) state.SkipWithError("Appended coefficient was not preserved");
  }
  state.SetItemsProcessed(state.iterations() * size * 3);
}

BENCHMARK_TEMPLATE(BM_ConservativeResizeAppend, Complex, 256)->Arg(16)->Arg(64)->Arg(200)->Arg(256);
BENCHMARK_TEMPLATE(BM_ConservativeResizeAppend, NonTrivialScalar, 256)->Arg(16)->Arg(64)->Arg(200)->Arg(256);
BENCHMARK_TEMPLATE(BM_ConservativeResizeAppend, NonTrivialScalar, Eigen::Dynamic)->Arg(16)->Arg(64)->Arg(200)->Arg(256);
BENCHMARK_TEMPLATE(BM_ConservativeResizeOuter, Eigen::RowMajor)->Arg(16)->Arg(64)->Arg(200)->Arg(256);
BENCHMARK_TEMPLATE(BM_ConservativeResizeOuter, Eigen::ColMajor)->Arg(16)->Arg(64)->Arg(200)->Arg(256);
