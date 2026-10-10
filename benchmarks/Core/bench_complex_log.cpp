// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

#include <cmath>
#include <complex>
#include <limits>

using namespace Eigen;

// The packet path, plog_complex, for whole packets. Inputs (+-3, +-4) * 2^e with ordinary, large and subnormal e;
// log|z| = log(5) + e log(2) and arg z = atan2(+-4, +-3).
template <typename T>
static void BM_ComplexLogPacket(benchmark::State& state) {
  using C = std::complex<T>;
  const Index size = state.range(0);
  const int mode = int(state.range(1));
  Array<C, Dynamic, 1> input(size), output(size);
  for (Index i = 0; i < size; ++i) {
    int exponent = int(i % 9) - 4;
    if (mode == 1) exponent = std::numeric_limits<T>::max_exponent - 4 + int(i % 2);
    if (mode == 2) exponent = std::numeric_limits<T>::min_exponent - std::numeric_limits<T>::digits + int(i % 9);
    const T scale = std::ldexp(T(1), exponent);
    input(i) = C((i % 2 ? T(-3) : T(3)) * scale, (i % 3 ? T(-4) : T(4)) * scale);
  }
  output = input.log();
  for (Index i = 0; i < size; ++i) {
    const T x = std::real(input(i)) / std::ldexp(T(1), std::ilogb(std::real(input(i))) - 1);
    const T y = std::imag(input(i)) / std::ldexp(T(1), std::ilogb(std::real(input(i))) - 1);
    const T log_abs = std::log(T(5)) + T(std::ilogb(std::real(input(i))) - 1) * std::log(T(2));
    const T tol = T(16) * NumTraits<T>::epsilon();
    if (!(std::abs(std::real(output(i)) - log_abs) <= tol * std::abs(log_abs)) ||
        !(std::abs(std::imag(output(i)) - std::atan2(y, x)) <= tol)) {
      state.SkipWithError("complex log failed reference validation");
      return;
    }
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(input.data());
    output = input.log();
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * size);
  state.SetLabel(mode == 0 ? "ordinary" : mode == 1 ? "large" : "subnormal");
}

BENCHMARK_TEMPLATE(BM_ComplexLogPacket, float)->ArgsProduct({{16, 4096}, {0, 1, 2}});
BENCHMARK_TEMPLATE(BM_ComplexLogPacket, double)->ArgsProduct({{16, 4096}, {0, 1, 2}});
