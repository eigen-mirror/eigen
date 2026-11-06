// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <vector>

template <bool Subtract, typename Lhs, typename Rhs, typename Packet>
EIGEN_STRONG_INLINE Packet mixed_conj_madd(const Lhs& x, const Rhs& y, const Packet& c) {
  EIGEN_IF_CONSTEXPR (Subtract) return Eigen::internal::conj_helper<Lhs, Rhs, false, false>().pmsub(x, y, c);
  return Eigen::internal::conj_helper<Lhs, Rhs, false, false>().pmadd(x, y, c);
}

template <typename Real, bool ComplexFirst, bool Subtract = false>
static void BM_MixedConjHelper(benchmark::State& state) {
  using Scalar = std::complex<Real>;
  using Packet = typename Eigen::internal::packet_traits<Scalar>::type;
  using RealPacket = typename Eigen::internal::packet_traits<Real>::type;
  constexpr int Size = Eigen::internal::unpacket_traits<Packet>::size;
  const Eigen::Index count = state.range(0) * Size;
  std::vector<Real> real(count);
  std::vector<Scalar> complex(count), accumulator(count), result(count);
  for (Eigen::Index i = 0; i < count; ++i) {
    real[i] = Real(i % 7 + 1) / Real(8);
    complex[i] = Scalar(Real(0.25), Real(-0.5));
    accumulator[i] = Scalar(Real(0.5), Real(0.25));
  }
  const auto evaluate = [&] {
    for (Eigen::Index i = 0; i < count; i += Size) {
      const RealPacket x = Eigen::internal::ploaddup<RealPacket>(real.data() + i);
      const Packet y = Eigen::internal::ploadu<Packet>(complex.data() + i);
      const Packet c = Eigen::internal::ploadu<Packet>(accumulator.data() + i);
      Eigen::internal::pstoreu(result.data() + i,
                               ComplexFirst ? mixed_conj_madd<Subtract>(y, x, c) : mixed_conj_madd<Subtract>(x, y, c));
    }
  };
  evaluate();
  for (Eigen::Index i = 0; i < count; ++i) {
    if (result[i] != real[i] * complex[i] + (Subtract ? -accumulator[i] : accumulator[i])) {
      state.SkipWithError("incorrect mixed multiply-add/subtract");
      return;
    }
  }
  for (auto _ : state) {
    evaluate();
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * count);
}

BENCHMARK_TEMPLATE(BM_MixedConjHelper, float, false)->Arg(4)->Arg(64)->Arg(4096);
BENCHMARK_TEMPLATE(BM_MixedConjHelper, float, true)->Arg(4)->Arg(64)->Arg(4096);
BENCHMARK_TEMPLATE(BM_MixedConjHelper, double, false)->Arg(4)->Arg(64)->Arg(4096);
BENCHMARK_TEMPLATE(BM_MixedConjHelper, double, true)->Arg(4)->Arg(64)->Arg(4096);
BENCHMARK_TEMPLATE(BM_MixedConjHelper, float, false, true)->Arg(4)->Arg(64)->Arg(4096);
BENCHMARK_TEMPLATE(BM_MixedConjHelper, float, true, true)->Arg(4)->Arg(64)->Arg(4096);
BENCHMARK_TEMPLATE(BM_MixedConjHelper, double, false, true)->Arg(4)->Arg(64)->Arg(4096);
BENCHMARK_TEMPLATE(BM_MixedConjHelper, double, true, true)->Arg(4)->Arg(64)->Arg(4096);
