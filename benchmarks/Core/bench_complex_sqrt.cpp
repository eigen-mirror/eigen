// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <complex>
#include <limits>

using namespace Eigen;

// Inputs (+-3, +-4) * 2^e whose roots are exact multiples of (2, 1) or (1, 2): ordinary, large and subnormal e, and
// ordinary e with every 8th input, at pseudo-random positions, replaced by zero.
template <typename T, bool Reciprocal>
static void MakeComplexRootInputs(Index size, int mode, Array<std::complex<T>, Dynamic, 1>& input,
                                  Array<std::complex<T>, Dynamic, 1>& reference) {
  using C = std::complex<T>;
  input.resize(size);
  reference.resize(size);
  for (Index i = 0; i < size; ++i) {
    int exponent = int(i % 9) - 4;
    if (mode == 1) exponent = std::numeric_limits<T>::max_exponent - 4 + int(i % 2);
    if (mode == 2) exponent = std::numeric_limits<T>::min_exponent - std::numeric_limits<T>::digits + int(i % 9);
    const T scale = std::ldexp(T(1), exponent);
    const T root_scale = std::sqrt(scale);
    const T x_sign = i % 2 ? T(-1) : T(1);
    const T y_sign = i % 3 ? T(-1) : T(1);
    const T u = x_sign > 0 ? T(2) : T(1);
    const T v = x_sign > 0 ? T(1) : T(2);
    input(i) = C(x_sign * T(3) * scale, y_sign * T(4) * scale);
    if (mode == 3 && ((std::uint32_t(i) * 2654435761u) >> 20) % 8 == 0) {
      input(i) = reference(i) = C(0);
      continue;
    }
    reference(i) = Reciprocal ? C((u / T(5)) / root_scale, (-y_sign * v / T(5)) / root_scale)
                              : C(u * root_scale, y_sign * v * root_scale);
  }
}

template <typename T>
static bool MatchesComplexRootReference(const Array<std::complex<T>, Dynamic, 1>& output,
                                        const Array<std::complex<T>, Dynamic, 1>& reference) {
  for (Index i = 0; i < output.size(); ++i) {
    if (reference(i) == std::complex<T>(0)) {
      if (output(i) != std::complex<T>(0)) return false;
      continue;
    }
    const std::complex<T> error = output(i) / reference(i) - std::complex<T>(T(1));
    if (!(std::abs(error) <= T(8) * NumTraits<T>::epsilon())) return false;
  }
  return true;
}

static const char* ComplexRootLabel(int mode) {
  return mode == 0 ? "ordinary" : mode == 1 ? "large" : mode == 2 ? "subnormal" : "1/8 zero";
}

// Scalar helpers also serve strided expressions and the tails of vectorized expressions.
template <typename T, bool Reciprocal>
static void BM_ComplexRoot(benchmark::State& state) {
  using C = std::complex<T>;
  const Index size = state.range(0);
  const int mode = int(state.range(1));
  Array<C, Dynamic, 1> input, output(size), reference;
  MakeComplexRootInputs<T, Reciprocal>(size, mode, input, reference);
  const auto operation = [](const C& z) { return Reciprocal ? numext::rsqrt(z) : numext::sqrt(z); };
  output = input.unaryExpr(operation);
  if (!MatchesComplexRootReference(output, reference)) {
    state.SkipWithError("complex root failed reference validation");
    return;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(input.data());
    output = input.unaryExpr(operation);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * size);
  state.SetLabel(ComplexRootLabel(mode));
}

// The packet path, psqrt_complex, for whole packets.
template <typename T>
static void BM_ComplexSqrtPacket(benchmark::State& state) {
  using C = std::complex<T>;
  const Index size = state.range(0);
  const int mode = int(state.range(1));
  Array<C, Dynamic, 1> input, output(size), reference;
  MakeComplexRootInputs<T, false>(size, mode, input, reference);
  output = input.sqrt();
  if (!MatchesComplexRootReference(output, reference)) {
    state.SkipWithError("complex sqrt failed reference validation");
    return;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(input.data());
    output = input.sqrt();
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * size);
  state.SetLabel(ComplexRootLabel(mode));
}

BENCHMARK_TEMPLATE(BM_ComplexRoot, float, false)->ArgsProduct({{1, 16, 4097}, {0, 1, 2}});
BENCHMARK_TEMPLATE(BM_ComplexRoot, double, false)->ArgsProduct({{1, 16, 4097}, {0, 1, 2}});
BENCHMARK_TEMPLATE(BM_ComplexRoot, float, true)->ArgsProduct({{1, 16, 4097}, {0, 1, 2}});
BENCHMARK_TEMPLATE(BM_ComplexRoot, double, true)->ArgsProduct({{1, 16, 4097}, {0, 1, 2}});
BENCHMARK_TEMPLATE(BM_ComplexSqrtPacket, float)->ArgsProduct({{16, 4096}, {0, 1, 2, 3}});
BENCHMARK_TEMPLATE(BM_ComplexSqrtPacket, double)->ArgsProduct({{16, 4096}, {0, 1, 2, 3}});
