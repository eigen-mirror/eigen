// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

#include <cmath>
#include <complex>
#include <limits>
#include <vector>

using namespace Eigen;

// Integer powers by double-word repeated squaring, against the log/exp path of generic_pow that they replace for
// large exponents. The "tiny" label puts the results about 2^-100 (float) or 2^-960 (double): normal numbers whose
// double-word residuals would be subnormal without the scaling of the running power.

template <typename T>
Array<T, Dynamic, 1> real_bases(Index size, int exponent, bool tiny) {
  Array<T, Dynamic, 1> x = Array<T, Dynamic, 1>::Random(size) * T(0.5) + T(1.25);  // [0.75, 1.75]
  if (tiny) {
    int target = -(std::numeric_limits<T>::max_exponent * 3) / 4;
    x *= T(std::ldexp(1.0, target / std::abs(exponent)));
    if (exponent < 0) x = x.inverse();
  }
  return x;
}

template <typename T>
static void BM_UnaryPowInt(benchmark::State& state) {
  Index size = 4097;
  int exponent = int(state.range(0));
  bool tiny = state.range(1) != 0;
  Array<T, Dynamic, 1> x = real_bases<T>(size, exponent, tiny), y(size);
  for (auto _ : state) {
    benchmark::DoNotOptimize(x.data());
    y = x.pow(exponent);
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * size);
  state.SetLabel(tiny ? "tiny" : "ordinary");
}

template <typename T>
static void BM_UnaryPowReal(benchmark::State& state) {
  Index size = 4097;
  int exponent = int(state.range(0));
  bool tiny = state.range(1) != 0;
  Array<T, Dynamic, 1> x = real_bases<T>(size, exponent, tiny), y(size);
  for (auto _ : state) {
    benchmark::DoNotOptimize(x.data());
    y = x.pow(T(exponent));
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * size);
  state.SetLabel(tiny ? "tiny" : "ordinary");
}

// The vectorized log/exp power on its own.
template <typename T>
static void BM_GenericPow(benchmark::State& state) {
  using Packet = typename internal::packet_traits<T>::type;
  constexpr Index kPacketSize = internal::unpacket_traits<Packet>::size;
  Index size = 4096;
  int exponent = int(state.range(0));
  bool tiny = state.range(1) != 0;
  Array<T, Dynamic, 1> x = real_bases<T>(size, exponent, tiny), y(size);
  Packet packet_exponent = internal::pset1<Packet>(T(exponent));
  for (auto _ : state) {
    benchmark::DoNotOptimize(x.data());
    for (Index i = 0; i < size; i += kPacketSize)
      internal::pstoreu(y.data() + i, internal::generic_pow(internal::ploadu<Packet>(x.data() + i), packet_exponent));
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * size);
  state.SetLabel(tiny ? "tiny" : "ordinary");
}

// Fixed sizes with a literal exponent: the assignment unrolls and the exponent can fold into straight-line code.
template <typename T, int Size, int Exponent>
static void BM_UnaryPowFixed(benchmark::State& state) {
  Array<T, Size, 1> x = Array<T, Size, 1>::Random() * T(0.5) + T(1.25), y;
  for (auto _ : state) {
    benchmark::DoNotOptimize(x.data());
    y = x.pow(Exponent);
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * Size);
}

template <typename T>
Array<std::complex<T>, Dynamic, 1> complex_bases(Index size) {
  using C = std::complex<T>;
  Array<C, Dynamic, 1> z = Array<C, Dynamic, 1>::Random(size);
  return z.unaryExpr([](const C& v) { return v / std::abs(v) * T(1.05); });
}

template <typename T>
static void BM_ComplexPowInt(benchmark::State& state) {
  Index size = 4097;
  int exponent = int(state.range(0));
  Array<std::complex<T>, Dynamic, 1> z = complex_bases<T>(size), y(size);
  for (auto _ : state) {
    benchmark::DoNotOptimize(z.data());
    y = z.pow(exponent);
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * size);
}

// Complex bases with a real exponent take the scalar path.
template <typename T>
static void BM_ComplexPowReal(benchmark::State& state) {
  Index size = 4097;
  int exponent = int(state.range(0));
  Array<std::complex<T>, Dynamic, 1> z = complex_bases<T>(size), y(size);
  for (auto _ : state) {
    benchmark::DoNotOptimize(z.data());
    y = z.pow(T(exponent));
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * size);
}

template <typename T>
static void BM_StdComplexPow(benchmark::State& state) {
  Index size = 4097;
  int exponent = int(state.range(0));
  Array<std::complex<T>, Dynamic, 1> z = complex_bases<T>(size), y(size);
  for (auto _ : state) {
    benchmark::DoNotOptimize(z.data());
    for (Index i = 0; i < size; ++i) y(i) = std::pow(z(i), exponent);
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * size);
}

const std::vector<int64_t> kExponents = {2, 3, 8, 100, 1000, 1 << 20, -2, -3, -8, -100};
BENCHMARK_TEMPLATE(BM_UnaryPowInt, float)->ArgsProduct({kExponents, {0, 1}});
BENCHMARK_TEMPLATE(BM_UnaryPowInt, double)->ArgsProduct({kExponents, {0, 1}});
BENCHMARK_TEMPLATE(BM_UnaryPowReal, float)->ArgsProduct({kExponents, {0, 1}});
BENCHMARK_TEMPLATE(BM_UnaryPowReal, double)->ArgsProduct({kExponents, {0, 1}});
BENCHMARK_TEMPLATE(BM_GenericPow, float)->ArgsProduct({kExponents, {0, 1}});
BENCHMARK_TEMPLATE(BM_GenericPow, double)->ArgsProduct({kExponents, {0, 1}});
BENCHMARK_TEMPLATE(BM_UnaryPowFixed, float, 8, 3);
BENCHMARK_TEMPLATE(BM_UnaryPowFixed, float, 8, 8);
BENCHMARK_TEMPLATE(BM_UnaryPowFixed, float, 64, 3);
BENCHMARK_TEMPLATE(BM_UnaryPowFixed, float, 64, 8);
BENCHMARK_TEMPLATE(BM_UnaryPowFixed, float, 64, 100);
BENCHMARK_TEMPLATE(BM_UnaryPowFixed, double, 4, 3);
BENCHMARK_TEMPLATE(BM_UnaryPowFixed, double, 4, 8);
BENCHMARK_TEMPLATE(BM_UnaryPowFixed, double, 64, 3);
BENCHMARK_TEMPLATE(BM_UnaryPowFixed, double, 64, 8);
BENCHMARK_TEMPLATE(BM_UnaryPowFixed, double, 64, 100);
BENCHMARK_TEMPLATE(BM_ComplexPowInt, float)->Args({2})->Args({3})->Args({8})->Args({100})->Args({-2})->Args({-8});
BENCHMARK_TEMPLATE(BM_ComplexPowInt, double)->Args({2})->Args({3})->Args({8})->Args({100})->Args({-2})->Args({-8});
BENCHMARK_TEMPLATE(BM_ComplexPowReal, float)->Args({2})->Args({3})->Args({8})->Args({100})->Args({-2})->Args({-8});
BENCHMARK_TEMPLATE(BM_ComplexPowReal, double)->Args({2})->Args({3})->Args({8})->Args({100})->Args({-2})->Args({-8});
BENCHMARK_TEMPLATE(BM_StdComplexPow, float)->Args({2})->Args({3})->Args({8})->Args({100})->Args({-2})->Args({-8});
BENCHMARK_TEMPLATE(BM_StdComplexPow, double)->Args({2})->Args({3})->Args({8})->Args({100})->Args({-2})->Args({-8});
