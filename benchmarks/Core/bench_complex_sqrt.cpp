// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

#include <cmath>
#include <complex>
#include <limits>

using namespace Eigen;

// Scalar helpers also serve strided expressions and the tails of vectorized expressions.
template <typename T, bool Reciprocal>
static void BM_ComplexRoot(benchmark::State& state) {
  using C = std::complex<T>;
  const Index size = state.range(0);
  const int mode = int(state.range(1));
  Array<C, Dynamic, 1> input(size), output(size), reference(size);
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
    reference(i) = Reciprocal ? C((u / T(5)) / root_scale, (-y_sign * v / T(5)) / root_scale)
                              : C(u * root_scale, y_sign * v * root_scale);
  }
  const auto operation = [](const C& z) { return Reciprocal ? numext::rsqrt(z) : numext::sqrt(z); };
  output = input.unaryExpr(operation);
  for (Index i = 0; i < size; ++i) {
    const C error = output(i) / reference(i) - C(T(1));
    if (!(std::abs(error) <= T(8) * NumTraits<T>::epsilon())) {
      state.SkipWithError("complex root failed reference validation");
      return;
    }
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(input.data());
    output = input.unaryExpr(operation);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * size);
  state.SetLabel(mode == 0 ? "ordinary" : mode == 1 ? "large" : "subnormal");
}

BENCHMARK_TEMPLATE(BM_ComplexRoot, float, false)->ArgsProduct({{1, 16, 4097}, {0, 1, 2}});
BENCHMARK_TEMPLATE(BM_ComplexRoot, double, false)->ArgsProduct({{1, 16, 4097}, {0, 1, 2}});
BENCHMARK_TEMPLATE(BM_ComplexRoot, float, true)->ArgsProduct({{1, 16, 4097}, {0, 1, 2}});
BENCHMARK_TEMPLATE(BM_ComplexRoot, double, true)->ArgsProduct({{1, 16, 4097}, {0, 1, 2}});
