// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include <benchmark/benchmark.h>
#include <contrib/Eigen/StructuredMatrices>

using namespace Eigen;

template <int Options, typename Op>
void materialize(benchmark::State& state, const Op& op) {
  using Scalar = typename Op::Scalar;
  Matrix<Scalar, Dynamic, Dynamic, Options> dst(op.rows(), op.cols());
  dst = op;
  for (Index j = 0; j < op.cols(); ++j)
    for (Index i = 0; i < op.rows(); ++i)
      if (dst(i, j) != op.coeff(i, j)) {
        state.SkipWithError("incorrect materialization");
        return;
      }
  if (state.range(2) == 0) {
    for (auto _ : state) {
      dst = op;
      benchmark::DoNotOptimize(dst.data());
      benchmark::ClobberMemory();
    }
  } else {
    for (auto _ : state) {
      dst += op;
      dst -= op;
      benchmark::DoNotOptimize(dst.data());
      benchmark::ClobberMemory();
    }
  }
}

template <typename Scalar, int Options>
void BM_CirculantMaterialize(benchmark::State& state) {
  Matrix<Scalar, Dynamic, 1> c = Matrix<Scalar, Dynamic, 1>::Random(state.range(0));
  materialize<Options>(state, makeCirculant(c));
}

template <typename Scalar, int Options>
void BM_ToeplitzMaterialize(benchmark::State& state) {
  Matrix<Scalar, Dynamic, 1> c = Matrix<Scalar, Dynamic, 1>::Random(state.range(0));
  Matrix<Scalar, Dynamic, 1> r = Matrix<Scalar, Dynamic, 1>::Random(state.range(1));
  materialize<Options>(state, makeToeplitz(c, r));
}

template <typename Scalar, int Options>
void BM_HankelMaterialize(benchmark::State& state) {
  Matrix<Scalar, Dynamic, 1> c = Matrix<Scalar, Dynamic, 1>::Random(state.range(0));
  Matrix<Scalar, Dynamic, 1> r = Matrix<Scalar, Dynamic, 1>::Random(state.range(1));
  materialize<Options>(state, makeHankel(c, r));
}

#define STRUCTURED_MATERIALIZE_CASES(Name, Scalar, Options, Wide) \
  BENCHMARK_TEMPLATE(Name, Scalar, Options)                       \
      ->Args({32, 32, 0})                                         \
      ->Args({256, 256, 0})                                       \
      ->Args({127, Wide, 0})                                      \
      ->Args({256, 256, 1})
#define STRUCTURED_MATERIALIZE_TYPES(Name, Wide)                            \
  STRUCTURED_MATERIALIZE_CASES(Name, double, ColMajor, Wide);               \
  STRUCTURED_MATERIALIZE_CASES(Name, double, RowMajor, Wide);               \
  STRUCTURED_MATERIALIZE_CASES(Name, std::complex<double>, ColMajor, Wide); \
  STRUCTURED_MATERIALIZE_CASES(Name, std::complex<double>, RowMajor, Wide)

STRUCTURED_MATERIALIZE_TYPES(BM_CirculantMaterialize, 127);
STRUCTURED_MATERIALIZE_TYPES(BM_ToeplitzMaterialize, 257);
STRUCTURED_MATERIALIZE_TYPES(BM_HankelMaterialize, 257);
