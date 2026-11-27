// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// pmadd, pmsub, pnmadd and pnmsub on the default packet of float, double and their complex types, over arrays that
// stay in L1.
// Throughput: out[i] = op(a[i], b[i], c[i]), independent iterations.
// Chain: acc = op(a[i], b[i], acc), each result feeds the next operation, so the latency of op bounds it.

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <complex>
#include <vector>

namespace {

using namespace Eigen::internal;

enum Op { kMadd, kMsub, kNmadd, kNmsub };

template <int Variant, typename Packet>
EIGEN_STRONG_INLINE Packet apply(const Packet& a, const Packet& b, const Packet& c) {
  EIGEN_IF_CONSTEXPR (Variant == kMadd) return pmadd(a, b, c);
  EIGEN_IF_CONSTEXPR (Variant == kMsub) return pmsub(a, b, c);
  EIGEN_IF_CONSTEXPR (Variant == kNmadd) return pnmadd(a, b, c);
  return pnmsub(a, b, c);
}

template <int Variant, typename Scalar>
Scalar reference(Scalar a, Scalar b, Scalar c) {
  const Scalar ab = (Variant == kNmadd || Variant == kNmsub) ? -(a * b) : a * b;
  return (Variant == kMsub || Variant == kNmsub) ? ab - c : ab + c;
}

// re / 8 for a real Scalar, (re + i * im) / 8 for a complex one.
template <typename Scalar, bool IsComplex = Eigen::NumTraits<Scalar>::IsComplex>
struct Eighths {
  static Scalar value(Eigen::Index re, Eigen::Index /*im*/) { return Scalar(re) / Scalar(8); }
};
template <typename Scalar>
struct Eighths<Scalar, true> {
  static Scalar value(Eigen::Index re, Eigen::Index im) {
    using Real = typename Eigen::NumTraits<Scalar>::Real;
    return Scalar(Real(re) / Real(8), Real(im) / Real(8));
  }
};

template <typename Scalar>
struct Operands {
  // 4 KiB per array.
  static constexpr Eigen::Index kCount = 4096 / sizeof(Scalar);
  std::vector<Scalar> a, b, c, out;
  Operands() : a(kCount), b(kCount), c(kCount), out(kCount) {
    for (Eigen::Index i = 0; i < kCount; ++i) {
      a[i] = Eighths<Scalar>::value(i % 7 + 1, i % 3 - 1);
      b[i] = Eighths<Scalar>::value(2 * (i % 5) - 4, i % 4 - 2);
      c[i] = Eighths<Scalar>::value(4 * (i % 3) - 2, i % 5 - 2);
    }
  }
};

template <typename Scalar, int Variant>
void BM_Throughput(benchmark::State& state) {
  using Packet = typename packet_traits<Scalar>::type;
  constexpr int Size = unpacket_traits<Packet>::size;
  Operands<Scalar> x;
  const Eigen::Index count = x.kCount;
  const auto evaluate = [&] {
    for (Eigen::Index i = 0; i + Size <= count; i += Size)
      pstoreu(x.out.data() + i, apply<Variant>(ploadu<Packet>(x.a.data() + i), ploadu<Packet>(x.b.data() + i),
                                               ploadu<Packet>(x.c.data() + i)));
  };
  evaluate();
  // The operands are multiples of 1/8, so the reference is exact with or without fusion.
  for (Eigen::Index i = 0; i < count; ++i) {
    if (x.out[i] != reference<Variant>(x.a[i], x.b[i], x.c[i])) {
      state.SkipWithError("incorrect multiply-add variant");
      return;
    }
  }
  for (auto _ : state) {
    evaluate();
    benchmark::DoNotOptimize(x.out.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * count);
}

template <typename Scalar, int Variant>
void BM_Chain(benchmark::State& state) {
  using Packet = typename packet_traits<Scalar>::type;
  constexpr int Size = unpacket_traits<Packet>::size;
  Operands<Scalar> x;
  const Eigen::Index count = x.kCount;
  Packet acc = pset1<Packet>(Scalar(0));
  for (auto _ : state) {
    for (Eigen::Index i = 0; i + Size <= count; i += Size)
      acc = apply<Variant>(ploadu<Packet>(x.a.data() + i), ploadu<Packet>(x.b.data() + i), acc);
    benchmark::DoNotOptimize(acc);
  }
  state.SetItemsProcessed(state.iterations() * (count / Size));
}

}  // namespace

#define BENCH_MADD_VARIANTS(SCALAR)                  \
  BENCHMARK_TEMPLATE(BM_Throughput, SCALAR, kMadd);  \
  BENCHMARK_TEMPLATE(BM_Throughput, SCALAR, kMsub);  \
  BENCHMARK_TEMPLATE(BM_Throughput, SCALAR, kNmadd); \
  BENCHMARK_TEMPLATE(BM_Throughput, SCALAR, kNmsub); \
  BENCHMARK_TEMPLATE(BM_Chain, SCALAR, kMadd);       \
  BENCHMARK_TEMPLATE(BM_Chain, SCALAR, kMsub);       \
  BENCHMARK_TEMPLATE(BM_Chain, SCALAR, kNmadd);      \
  BENCHMARK_TEMPLATE(BM_Chain, SCALAR, kNmsub)

BENCH_MADD_VARIANTS(float);
BENCH_MADD_VARIANTS(double);
BENCH_MADD_VARIANTS(std::complex<float>);
BENCH_MADD_VARIANTS(std::complex<double>);
