// Benchmarks for the PacketMath implementations of `plset`, `ploaddup`,
// `ploadquad`, `psign`, `predux_mul`, `ptranspose`, and `pldexp`, at the packet-op
// level and shared across whichever architecture backend the build targets. To
// compare against a prior implementation, build and run this same file
// against the Eigen checkout in question -- it only calls the public
// `Eigen::internal` packet API, so it is source-compatible with whatever
// PacketMath.h happens to provide.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace Eigen {
namespace {

using internal::packet_traits;
using internal::PacketBlock;
using internal::pfirst;
using internal::ploadu;
using internal::pstoreu;

template <typename Packet, int N>
EIGEN_DONT_INLINE void call_ptranspose(PacketBlock<Packet, N>& kernel) {
  internal::ptranspose(kernel);
}

// ---- plset ----

template <typename Scalar>
void BM_Plset(benchmark::State& state) {
  using Packet = typename packet_traits<Scalar>::type;
  constexpr int N = packet_traits<Scalar>::size;
  Scalar a = Scalar(7);
  Scalar out[N];

  pstoreu(out, internal::plset<Packet>(a));
  for (int i = 0; i < N; ++i) {
    if (out[i] != static_cast<Scalar>(a + i)) {
      state.SkipWithError("Plset: materialized result does not match scalar reference");
      return;
    }
  }

  benchmark::DoNotOptimize(a);
  for (auto _ : state) {
    pstoreu(out, internal::plset<Packet>(a));
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_Plset<numext::int32_t>)->Name("Plset_int32");
BENCHMARK(BM_Plset<float>)->Name("Plset_float");
BENCHMARK(BM_Plset<double>)->Name("Plset_double");

// ---- ploaddup ----

template <typename Scalar>
void BM_Ploaddup(benchmark::State& state) {
  using Packet = typename packet_traits<Scalar>::type;
  constexpr int N = packet_traits<Scalar>::size;
  Scalar in[N];
  for (int i = 0; i < N; ++i) in[i] = static_cast<Scalar>(i);
  Scalar out[N];

  pstoreu(out, internal::ploaddup<Packet>(in));
  for (int i = 0; i < N; ++i) {
    if (out[i] != in[i / 2]) {
      state.SkipWithError("Ploaddup: materialized result does not match scalar reference");
      return;
    }
  }

  benchmark::DoNotOptimize(in);
  for (auto _ : state) {
    pstoreu(out, internal::ploaddup<Packet>(in));
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_Ploaddup<numext::int32_t>)->Name("Ploaddup_int32");
BENCHMARK(BM_Ploaddup<float>)->Name("Ploaddup_float");
BENCHMARK(BM_Ploaddup<double>)->Name("Ploaddup_double");

// ---- psign ----

#if defined(EIGEN_VECTORIZE_AVX) || defined(EIGEN_VECTORIZE_AVX512) || defined(EIGEN_VECTORIZE_NEON) || \
    defined(EIGEN_VECTORIZE_ALTIVEC) || defined(EIGEN_VECTORIZE_VSX)
void BM_PsignBfloat16(benchmark::State& state) {
  using Packet = typename packet_traits<bfloat16>::type;
  constexpr int N = packet_traits<bfloat16>::size;
  const int mode = static_cast<int>(state.range(0));
  bfloat16 input[N], output[N];
  for (int i = 0; i < N; ++i) {
    numext::uint16_t bits = i % 2 ? 0xbf80 : 0x3f80;
    if (mode == 1 || (mode == 2 && i == 0)) bits = i % 2 ? 0x8001 : 0x0001;
    input[i] = numext::bit_cast<bfloat16>(bits);
  }

  Packet packet = ploadu<Packet>(input);
  pstoreu(output, internal::psign(packet));
  for (int i = 0; i < N; ++i) {
    const numext::uint16_t bits = numext::bit_cast<numext::uint16_t>(input[i]);
    const numext::uint16_t expected = bits & 0x8000 ? 0xbf80 : 0x3f80;
    if (numext::bit_cast<numext::uint16_t>(output[i]) != expected) {
      state.SkipWithError("Psign_bfloat16: output does not match the bitwise reference");
      return;
    }
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(packet);
    Packet result = internal::psign(packet);
    benchmark::DoNotOptimize(result);
  }
  state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_PsignBfloat16)->Arg(0)->Arg(1)->Arg(2)->Name("Psign_bfloat16");
#endif

// ---- ploadquad ----

template <typename Scalar>
void BM_Ploadquad(benchmark::State& state) {
  using Packet = typename packet_traits<Scalar>::type;
  constexpr int N = packet_traits<Scalar>::size;
  Scalar in[N];
  for (int i = 0; i < N; ++i) in[i] = static_cast<Scalar>(i);
  Scalar out[N];

  pstoreu(out, internal::ploadquad<Packet>(in));
  for (int i = 0; i < N; ++i) {
    if (out[i] != in[i / 4]) {
      state.SkipWithError("Ploadquad: materialized result does not match scalar reference");
      return;
    }
  }

  benchmark::DoNotOptimize(in);
  for (auto _ : state) {
    pstoreu(out, internal::ploadquad<Packet>(in));
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_Ploadquad<numext::int32_t>)->Name("Ploadquad_int32");
BENCHMARK(BM_Ploadquad<float>)->Name("Ploadquad_float");
BENCHMARK(BM_Ploadquad<double>)->Name("Ploadquad_double");

// ---- predux_mul ----
// Inputs are chosen so the true product is exactly representable regardless
// of the order the reduction folds lanes together, at every vector length:
// powers of two multiply without rounding, and centering the exponents around
// zero keeps both the inputs and the product in range for float/double even
// at N == 64; multiplying by 1 is exact and overflow-free for int32.

template <typename Scalar>
void fill_redux_mul_input(Scalar (&in)[packet_traits<Scalar>::size], Scalar& expected) {
  constexpr int N = packet_traits<Scalar>::size;
  if constexpr (std::is_integral<Scalar>::value) {
    for (int i = 0; i < N; ++i) in[i] = Scalar(1);
    in[0] = Scalar(-3);
    expected = Scalar(-3);
    if (N > 1) {
      in[N - 1] = Scalar(2);
      expected = Scalar(-6);
    }
  } else {
    for (int i = 0; i < N; ++i) in[i] = std::ldexp(Scalar(1), i - N / 2);
    expected = std::ldexp(Scalar(1), -N / 2);
  }
}

template <typename Scalar>
void BM_ReduxMul(benchmark::State& state) {
  using Packet = typename packet_traits<Scalar>::type;
  constexpr int N = packet_traits<Scalar>::size;
  Scalar in[N];
  Scalar expected;
  fill_redux_mul_input<Scalar>(in, expected);
  Packet a = ploadu<Packet>(in);

  if (internal::predux_mul<Packet>(a) != expected) {
    state.SkipWithError("ReduxMul: materialized result does not match scalar reference");
    return;
  }

  benchmark::DoNotOptimize(a);
  for (auto _ : state) benchmark::DoNotOptimize(internal::predux_mul<Packet>(a));
}
BENCHMARK(BM_ReduxMul<numext::int32_t>)->Name("ReduxMul_int32");
BENCHMARK(BM_ReduxMul<float>)->Name("ReduxMul_float");
BENCHMARK(BM_ReduxMul<double>)->Name("ReduxMul_double");

// ---- ptranspose ----
// Benchmarked at N == the type's packet width, i.e. a full square transpose,
// so the expected result is simply new_packet[i][k] == old_packet[k][i].
// Correctness is checked once on a scratch kernel -- ptranspose applied
// repeatedly toggles between the original and transposed state, so checking
// after the timed loop would depend on the (unpredictable) iteration count.
//
// call_ptranspose is deliberately EIGEN_DONT_INLINE: real callers inline
// ptranspose on register-resident packets, but this wrapper forces the block
// through memory, so this benchmark understates the win; transposeInPlace is
// the macro benchmark for the inlined case.

template <typename Scalar>
void BM_Ptranspose(benchmark::State& state) {
  using Packet = typename packet_traits<Scalar>::type;
  constexpr int N = packet_traits<Scalar>::size;
  Scalar in[N * N];
  for (int i = 0; i < N * N; ++i) in[i] = static_cast<Scalar>(i);

  PacketBlock<Packet, N> check;
  for (int i = 0; i < N; ++i) check.packet[i] = ploadu<Packet>(in + i * N);
  call_ptranspose<Packet, N>(check);
  for (int i = 0; i < N; ++i) {
    Scalar row[N];
    pstoreu(row, check.packet[i]);
    for (int k = 0; k < N; ++k) {
      if (row[k] != in[k * N + i]) {
        state.SkipWithError("Ptranspose: materialized result does not match scalar reference");
        return;
      }
    }
  }

  PacketBlock<Packet, N> kernel;
  for (int i = 0; i < N; ++i) kernel.packet[i] = ploadu<Packet>(in + i * N);
  for (auto _ : state) {
    call_ptranspose<Packet, N>(kernel);
    benchmark::DoNotOptimize(pfirst<Packet>(kernel.packet[0]));
  }
}
BENCHMARK(BM_Ptranspose<numext::int32_t>)->Name("Ptranspose_int32");
BENCHMARK(BM_Ptranspose<float>)->Name("Ptranspose_float");
BENCHMARK(BM_Ptranspose<double>)->Name("Ptranspose_double");

// ---- pldexp ----

// Throughput over 256 packets of bases in [1, 2) and integer exponents in [-range, range]: a small range stays
// normal, the full range (278 for float, 2099 for double) mostly under- or overflows.
template <typename Scalar>
void BM_Pldexp(benchmark::State& state) {
  using Packet = typename packet_traits<Scalar>::type;
  constexpr int N = packet_traits<Scalar>::size;
  constexpr int kCount = 256 * N;
  const int range = static_cast<int>(state.range(0));
  Scalar a[kCount], e[kCount], out[kCount];
  std::uint32_t seed = 1;
  for (int i = 0; i < kCount; ++i) {
    seed = seed * 1664525u + 1013904223u;
    a[i] = Scalar(1) + Scalar(seed >> 8) / Scalar(1 << 24);
    e[i] = Scalar(static_cast<int>(seed % std::uint32_t(2 * range + 1)) - range);
  }
  for (int i = 0; i < kCount; i += N) pstoreu(out + i, internal::pldexp(ploadu<Packet>(a + i), ploadu<Packet>(e + i)));
  for (int i = 0; i < kCount; ++i) {
    if (out[i] != std::ldexp(a[i], static_cast<int>(e[i]))) {
      state.SkipWithError("Pldexp: materialized result does not match std::ldexp");
      return;
    }
  }

  for (auto _ : state) {
    for (int i = 0; i < kCount; i += N)
      pstoreu(out + i, internal::pldexp(ploadu<Packet>(a + i), ploadu<Packet>(e + i)));
    benchmark::DoNotOptimize(out);
  }
  state.SetItemsProcessed(state.iterations() * (kCount / N));
}
BENCHMARK(BM_Pldexp<float>)->Name("Pldexp_float")->Arg(20)->Arg(278);
BENCHMARK(BM_Pldexp<double>)->Name("Pldexp_double")->Arg(20)->Arg(2099);

// Latency: a dependent chain scaling by 2^37 and back.
template <typename Scalar>
void BM_PldexpLatency(benchmark::State& state) {
  using Packet = typename packet_traits<Scalar>::type;
  const Packet up = internal::pset1<Packet>(Scalar(37));
  const Packet down = internal::pset1<Packet>(Scalar(-37));
  Packet x = internal::pset1<Packet>(Scalar(1.5));
  for (auto _ : state) {
    x = internal::pldexp(internal::pldexp(x, up), down);
    benchmark::DoNotOptimize(x);
  }
  if (pfirst(x) != Scalar(1.5)) state.SkipWithError("PldexpLatency: the chain changed its value");
  state.SetItemsProcessed(state.iterations() * 2);
}
BENCHMARK(BM_PldexpLatency<float>)->Name("PldexpLatency_float");
BENCHMARK(BM_PldexpLatency<double>)->Name("PldexpLatency_double");

}  // namespace
}  // namespace Eigen
