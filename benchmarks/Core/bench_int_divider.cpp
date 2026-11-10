// Benchmarks for IntDivider: coefficient-wise integer division by a divisor known only at run time.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <cstdint>

namespace Eigen {
namespace {

template <typename Scalar>
ArrayX<Scalar> Numerators(Index n) {
  return ArrayX<Scalar>::Random(n);
}

// state.range() hides the divisor from the compiler. benchmark::DoNotOptimize(d) would too, but GCC 15.2 at -O3
// drops the initial value of its "+m,r" asm operand here, leaving d uninitialized.
template <typename Scalar>
Scalar RuntimeDivisor(const benchmark::State& state) {
  return static_cast<Scalar>(state.range(1));
}

template <typename Scalar>
void Check(benchmark::State& state, const ArrayX<Scalar>& a, const ArrayX<Scalar>& q, Scalar d) {
  for (Index i = 0; i < a.size(); ++i) {
    if (q[i] != a[i] / d) {
      state.SkipWithError("quotient does not match the built-in operator");
      return;
    }
  }
}

// a / d through Eigen's quotient functor: a hardware division per coefficient, or for int32 the vectorized
// conversion to double.
template <typename Scalar>
void BM_Quotient(benchmark::State& state) {
  Index n = state.range(0);
  Scalar d = RuntimeDivisor<Scalar>(state);
  ArrayX<Scalar> a = Numerators<Scalar>(n);
  ArrayX<Scalar> q(n);
  for (auto _ : state) {
    q = a / d;
    benchmark::DoNotOptimize(q.data());
  }
  Check(state, a, q, d);
  state.SetItemsProcessed(state.iterations() * n);
}

template <typename Scalar>
void BM_IntDivider(benchmark::State& state) {
  Index n = state.range(0);
  Scalar d = RuntimeDivisor<Scalar>(state);
  ArrayX<Scalar> a = Numerators<Scalar>(n);
  ArrayX<Scalar> q(n);
  IntDivider<Scalar> by(d);
  for (auto _ : state) {
    q = a / by;
    benchmark::DoNotOptimize(q.data());
  }
  Check(state, a, q, d);
  state.SetItemsProcessed(state.iterations() * n);
}

// The scalar path of IntDivider, as a loop the compiler may vectorize on its own.
template <typename Scalar>
void BM_IntDividerLoop(benchmark::State& state) {
  Index n = state.range(0);
  Scalar d = RuntimeDivisor<Scalar>(state);
  ArrayX<Scalar> a = Numerators<Scalar>(n);
  ArrayX<Scalar> q(n);
  IntDivider<Scalar> by(d);
  for (auto _ : state) {
    for (Index i = 0; i < n; ++i) q[i] = a[i] / by;
    benchmark::DoNotOptimize(q.data());
  }
  Check(state, a, q, d);
  state.SetItemsProcessed(state.iterations() * n);
}

// Division by the compile-time constant 7, which the compiler turns into a multiplication: the target for
// IntDivider with the divisor 7.
template <typename Scalar>
void BM_ConstantLoop(benchmark::State& state) {
  Index n = state.range(0);
  ArrayX<Scalar> a = Numerators<Scalar>(n);
  ArrayX<Scalar> q(n);
  for (auto _ : state) {
    for (Index i = 0; i < n; ++i) q[i] = a[i] / Scalar(7);
    benchmark::DoNotOptimize(q.data());
  }
  Check(state, a, q, Scalar(7));
  state.SetItemsProcessed(state.iterations() * n);
}

#define EIGEN_INT_DIVIDER_BENCHMARKS(SCALAR)                               \
  BENCHMARK(BM_Quotient<SCALAR>)->ArgsProduct({{1024, 65536}, {7}});       \
  BENCHMARK(BM_IntDivider<SCALAR>)->ArgsProduct({{1024, 65536}, {7}});     \
  BENCHMARK(BM_IntDividerLoop<SCALAR>)->ArgsProduct({{1024, 65536}, {7}}); \
  BENCHMARK(BM_ConstantLoop<SCALAR>)->Arg(1024)->Arg(65536)

EIGEN_INT_DIVIDER_BENCHMARKS(int32_t);
EIGEN_INT_DIVIDER_BENCHMARKS(uint32_t);
EIGEN_INT_DIVIDER_BENCHMARKS(int64_t);
EIGEN_INT_DIVIDER_BENCHMARKS(uint64_t);

}  // namespace
}  // namespace Eigen
