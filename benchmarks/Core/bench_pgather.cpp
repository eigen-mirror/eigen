// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

template <typename Scalar, int Stride>
EIGEN_DONT_INLINE void gather_packets(const Scalar* input, Scalar* output, Index size, Index runtime_stride) {
  using Packet = typename internal::packet_traits<Scalar>::type;
  constexpr Index PacketSize = internal::packet_traits<Scalar>::size;
  const Index stride = Stride == Dynamic ? runtime_stride : Stride;
  for (Index i = 0; i < size; i += PacketSize) {
    internal::pstoreu(output + i, internal::pgather<Scalar, Packet>(input + i * stride, stride));
  }
}

template <typename Scalar, int Stride>
static void BM_Pgather(benchmark::State& state) {
  const Index size = state.range(0);
  const Index stride = Stride == Dynamic ? state.range(1) : Stride;
  Array<Scalar, Dynamic, 1> input((size - 1) * stride + 1), output(size);
  for (Index i = 0; i < input.size(); ++i) input(i) = Scalar(i % 127);
  gather_packets<Scalar, Stride>(input.data(), output.data(), size, stride);
  for (Index i = 0; i < size; ++i) {
    if (output(i) != input(i * stride)) {
      state.SkipWithError("Incorrect gathered coefficient");
      return;
    }
  }
  for (auto _ : state) {
    benchmark::ClobberMemory();
    gather_packets<Scalar, Stride>(input.data(), output.data(), size, stride);
    benchmark::DoNotOptimize(output.data());
  }
  state.SetItemsProcessed(state.iterations() * size);
}

BENCHMARK_TEMPLATE(BM_Pgather, float, 2)->Arg(64)->Arg(4096);
BENCHMARK_TEMPLATE(BM_Pgather, double, 2)->Arg(64)->Arg(4096);
BENCHMARK_TEMPLATE(BM_Pgather, float, Dynamic)->ArgsProduct({{64, 4096}, {1, 2, 3, 7}});
BENCHMARK_TEMPLATE(BM_Pgather, double, Dynamic)->ArgsProduct({{64, 4096}, {1, 2, 3, 7}});
