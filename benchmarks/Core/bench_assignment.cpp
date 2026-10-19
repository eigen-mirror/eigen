// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <cstdint>

#ifndef EIGEN_BENCH_ASSIGN_SCALAR
#define EIGEN_BENCH_ASSIGN_SCALAR double
#endif
#ifndef EIGEN_BENCH_ASSIGN_OP
// 0: abs, 1: square, 2: add, 3: copy, 4: exp.
#define EIGEN_BENCH_ASSIGN_OP 0
#endif

using namespace Eigen;

static void BM_Assignment(benchmark::State& state) {
  using Scalar = EIGEN_BENCH_ASSIGN_SCALAR;
  using Vector = Array<Scalar, Dynamic, 1>;
  // Control the low address bits independently: matching page offsets can trigger false store/load dependencies.
  constexpr Index pageSize = 4096;
  const Index size = state.range(0);
  Vector storage(3 * size + 3 * pageSize / sizeof(Scalar));
  Scalar* source = storage.data() + internal::first_aligned<pageSize>(storage.data(), storage.size());
  const Index separation = (((size + 1) * sizeof(Scalar) + pageSize - 1) / pageSize) * pageSize / sizeof(Scalar);
  Map<Vector> input(source + state.range(1), size);
  Map<Vector> output(source + separation + state.range(2), size);
  Vector other(size);
  for (Index i = 0; i < size; ++i) {
    input(i) = Scalar(i % 127 - 63) / Scalar(32);
    other(i) = Scalar(i % 61 - 30) / Scalar(16);
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(input.data());
    benchmark::DoNotOptimize(other.data());
#if EIGEN_BENCH_ASSIGN_OP == 0
    output = input.abs();
#elif EIGEN_BENCH_ASSIGN_OP == 1
    output = input.square();
#elif EIGEN_BENCH_ASSIGN_OP == 2
    output = input + other;
#elif EIGEN_BENCH_ASSIGN_OP == 3
    output = input;
#else
    output = input.exp();
#endif
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }
  for (Index i = 0; i < size; ++i) {
#if EIGEN_BENCH_ASSIGN_OP == 0
    const Scalar expected = numext::abs(input(i));
#elif EIGEN_BENCH_ASSIGN_OP == 1
    const Scalar expected = input(i) * input(i);
#elif EIGEN_BENCH_ASSIGN_OP == 2
    const Scalar expected = input(i) + other(i);
#elif EIGEN_BENCH_ASSIGN_OP == 3
    const Scalar expected = input(i);
#else
    const Scalar expected = numext::exp(input(i));
#endif
    if (!(numext::abs(output(i) - expected) <= Scalar(8) * NumTraits<Scalar>::epsilon() * numext::abs(expected))) {
      state.SkipWithError("assignment differs from the scalar reference");
      return;
    }
  }
  state.SetBytesProcessed(state.iterations() * size * sizeof(Scalar) * (EIGEN_BENCH_ASSIGN_OP == 2 ? 3 : 2));
}

BENCHMARK(BM_Assignment)
    ->ArgsProduct({{31, 32, 33, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 4095, 4096, 4097, 16384, 65536,
                    1048576},
                   {0, 1},
                   {0, 16}});
