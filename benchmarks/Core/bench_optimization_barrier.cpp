// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

template <typename Scalar, typename Packet, bool TwoSum>
EIGEN_DONT_INLINE void barrier_kernel(const Scalar* x, const Scalar* y, Scalar* hi, Scalar* lo, Index size) {
  constexpr int width = internal::unpacket_traits<Packet>::size;
  for (Index i = 0; i < size; i += width) {
    const Packet value = internal::ploadu<Packet>(x + i);
    Packet high = value;
    Packet low = internal::pset1<Packet>(Scalar(0));
    if (TwoSum) {
      internal::twosum(value, internal::ploadu<Packet>(y + i), high, low);
    } else {
      EIGEN_OPTIMIZATION_BARRIER(high);
    }
    internal::pstoreu(hi + i, high);
    internal::pstoreu(lo + i, low);
  }
}

template <typename Scalar, typename Packet, bool TwoSum>
static void BM_OptimizationBarrier(benchmark::State& state) {
  const Index size = state.range(0);
  const Scalar tiny = numext::ldexp(Scalar(1), -NumTraits<Scalar>::digits() - 1);
  Array<Scalar, Dynamic, 1> x(size), y(size), hi(size), lo(size);
  for (Index i = 0; i < size; ++i) {
    x[i] = Scalar(1) + Scalar(i % 16) / Scalar(16);
    y[i] = tiny;
  }
  barrier_kernel<Scalar, Packet, TwoSum>(x.data(), y.data(), hi.data(), lo.data(), size);
  for (Index i = 0; i < size; ++i) {
    if (hi[i] != x[i] || lo[i] != (TwoSum ? tiny : Scalar(0))) {
      state.SkipWithError("optimization barrier changed the result");
      return;
    }
  }
  for (auto _ : state) {
    barrier_kernel<Scalar, Packet, TwoSum>(x.data(), y.data(), hi.data(), lo.data(), size);
    benchmark::DoNotOptimize(hi.data());
    benchmark::DoNotOptimize(lo.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * size);
}

using FloatPacket = internal::packet_traits<float>::type;
using DoublePacket = internal::packet_traits<double>::type;
BENCHMARK_TEMPLATE(BM_OptimizationBarrier, float, float, false)->Name("MacroFloatScalar")->Arg(4096);
BENCHMARK_TEMPLATE(BM_OptimizationBarrier, float, FloatPacket, false)->Name("MacroFloatPacket")->Arg(4096);
BENCHMARK_TEMPLATE(BM_OptimizationBarrier, double, double, false)->Name("MacroDoubleScalar")->Arg(4096);
BENCHMARK_TEMPLATE(BM_OptimizationBarrier, double, DoublePacket, false)->Name("MacroDoublePacket")->Arg(4096);
BENCHMARK_TEMPLATE(BM_OptimizationBarrier, float, float, true)->Name("TwoSumFloatScalar")->Arg(4096);
BENCHMARK_TEMPLATE(BM_OptimizationBarrier, float, FloatPacket, true)->Name("TwoSumFloatPacket")->Arg(4096);
BENCHMARK_TEMPLATE(BM_OptimizationBarrier, double, double, true)->Name("TwoSumDoubleScalar")->Arg(4096);
BENCHMARK_TEMPLATE(BM_OptimizationBarrier, double, DoublePacket, true)->Name("TwoSumDoublePacket")->Arg(4096);
