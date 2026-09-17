// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <type_traits>

template <typename DiagonalType, bool Move>
static void BM_DiagonalConstruct(benchmark::State& state) {
  using Argument = std::conditional_t<Move, DiagonalType&&, const DiagonalType&>;
  DiagonalType source(state.range(0));
  source.diagonal().setRandom();
  const DiagonalType expected(source);
  for (auto _ : state) {
    DiagonalType temporary(static_cast<Argument>(source));
    benchmark::DoNotOptimize(temporary);
    benchmark::ClobberMemory();
    source = static_cast<Argument>(temporary);
    benchmark::DoNotOptimize(source);
    benchmark::ClobberMemory();
  }
  if (source.rows() != expected.rows() || source.diagonal() != expected.diagonal()) {
    state.SkipWithError("Diagonal construction changed coefficients");
  }
  state.SetItemsProcessed(state.iterations());
}

template <typename DiagonalType, bool Move>
static void BM_DiagonalAssign(benchmark::State& state) {
  using Argument = std::conditional_t<Move, DiagonalType&&, const DiagonalType&>;
  DiagonalType source(state.range(0)), destination(state.range(0));
  source.diagonal().setRandom();
  destination.diagonal().setZero();
  const DiagonalType expected(source);
  for (auto _ : state) {
    destination = static_cast<Argument>(source);
    benchmark::DoNotOptimize(destination);
    benchmark::ClobberMemory();
    source = static_cast<Argument>(destination);
    benchmark::DoNotOptimize(source);
    benchmark::ClobberMemory();
  }
  if (source.rows() != expected.rows() || source.diagonal() != expected.diagonal()) {
    state.SkipWithError("Diagonal assignment changed coefficients");
  }
  state.SetItemsProcessed(2 * state.iterations());
}

using DynamicFloat = Eigen::DiagonalMatrix<float, Eigen::Dynamic>;
using DynamicDouble = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;
using DynamicComplex = Eigen::DiagonalMatrix<std::complex<double>, Eigen::Dynamic>;
using Fixed4 = Eigen::DiagonalMatrix<double, 4>;
using Fixed16 = Eigen::DiagonalMatrix<double, 16>;
using Bounded16 = Eigen::DiagonalMatrix<double, Eigen::Dynamic, 16>;

// Construct measures construction plus assignment back; Assign measures two assignments.
#define DIAGONAL_MOVE_BENCHMARKS(Type, ...)                                                                \
  BENCHMARK_TEMPLATE(BM_DiagonalConstruct, Type, true)->ArgNames({"size"})->ArgsProduct({{__VA_ARGS__}});  \
  BENCHMARK_TEMPLATE(BM_DiagonalConstruct, Type, false)->ArgNames({"size"})->ArgsProduct({{__VA_ARGS__}}); \
  BENCHMARK_TEMPLATE(BM_DiagonalAssign, Type, true)->ArgNames({"size"})->ArgsProduct({{__VA_ARGS__}});     \
  BENCHMARK_TEMPLATE(BM_DiagonalAssign, Type, false)->ArgNames({"size"})->ArgsProduct({{__VA_ARGS__}})

DIAGONAL_MOVE_BENCHMARKS(DynamicFloat, 0, 1, 4, 16, 256, 4096, 65536);
DIAGONAL_MOVE_BENCHMARKS(DynamicDouble, 0, 1, 4, 16, 256, 4096, 65536);
DIAGONAL_MOVE_BENCHMARKS(DynamicComplex, 0, 1, 4, 16, 256, 4096, 65536);
DIAGONAL_MOVE_BENCHMARKS(Fixed4, 4);
DIAGONAL_MOVE_BENCHMARKS(Fixed16, 16);
DIAGONAL_MOVE_BENCHMARKS(Bounded16, 0, 1, 4, 16);

#undef DIAGONAL_MOVE_BENCHMARKS
