// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// all(), any(), allFinite(), isZero() and isApproxToConstant(). The plain cases are full scans: no coefficient decides
// the result early. The "/first" cases are decided by coefficient 0, which measures the reads past the deciding one.
// Dynamic sizes hide the size and data pointer from the compiler on every iteration, as a caller whose sizes vary
// would; fixed sizes and a block of a matrix cover the other traversals.
//
// Each binary measures one family, selected by EIGEN_BENCH_PREDICATE_SCALAR and EIGEN_BENCH_PREDICATE_OP (dynamic),
// EIGEN_BENCH_PREDICATE_XPR and EIGEN_BENCH_PREDICATE_OP (fixed), or neither (block).

#include <benchmark/benchmark.h>

#include <Eigen/Core>

#include <cstdint>
#include <limits>
#include <string>

namespace Eigen {
namespace {

using std::int16_t;

enum Op { All, Any, AllFinite, IsZero, ApproxConstant };

template <typename T, Op op>
T fill() {
  return op == Any || op == IsZero ? T(0) : op == ApproxConstant ? T(2) : T(1);
}

// A coefficient that decides the result on its own.
template <typename T, Op op>
T decider() {
  return op == All ? T(0) : op == AllFinite ? std::numeric_limits<T>::infinity() : op == ApproxConstant ? T(3) : T(1);
}

template <Op op, typename Xpr>
bool apply(const Xpr& x) {
  using Scalar = typename Xpr::Scalar;
  switch (op) {
    case All:
      return x.all();
    case Any:
      return x.any();
    case AllFinite:
      return x.allFinite();
    case IsZero:
      return x.isZero();
    default:
      return x.isApproxToConstant(Scalar(2));
  }
}

template <typename T, Op op, bool First>
void BM_Dynamic(benchmark::State& state) {
  const Index n = state.range(0);
  ArrayX<T> a = ArrayX<T>::Constant(n, fill<T, op>());
  if (First) a(0) = decider<T, op>();
  for (auto _ : state) {
    Index m = n;
    benchmark::DoNotOptimize(m);
    const T* p = a.data();
    benchmark::DoNotOptimize(p);
    bool r = apply<op>(Map<const ArrayX<T>>(p, m));
    benchmark::DoNotOptimize(r);
  }
  state.SetItemsProcessed(state.iterations() * n);
}

template <typename Xpr, Op op>
void BM_Fixed(benchmark::State& state) {
  Xpr a = Xpr::Constant(fill<typename Xpr::Scalar, op>());
  for (auto _ : state) {
    benchmark::DoNotOptimize(a);
    bool r = apply<op>(a);
    benchmark::DoNotOptimize(r);
  }
}

// isZero() on the top rows of a 64-column matrix: one inner vector per column, no linear access.
void BM_BlockIsZero(benchmark::State& state) {
  const Index n = state.range(0);
  MatrixXf m = MatrixXf::Zero(n + 3, 64);
  for (auto _ : state) {
    Index k = n;
    benchmark::DoNotOptimize(k);
    benchmark::DoNotOptimize(m.data());
    bool r = m.topRows(k).isZero();
    benchmark::DoNotOptimize(r);
  }
  state.SetItemsProcessed(state.iterations() * n * 64);
}

using Array16f = Array<float, 16, 1>;
using Array33f = Array<float, 33, 1>;

const char* const opNames[] = {"all", "any", "allFinite", "isZero", "isApproxToConstant"};

#if defined(EIGEN_BENCH_PREDICATE_SCALAR)
const bool registered = [] {
  using T = EIGEN_BENCH_PREDICATE_SCALAR;
  constexpr Op op = EIGEN_BENCH_PREDICATE_OP;
  const std::string name = std::string(opNames[op]) + "<" EIGEN_MAKESTRING(EIGEN_BENCH_PREDICATE_SCALAR) ">";
  benchmark::RegisterBenchmark(name.c_str(), BM_Dynamic<T, op, false>)
      ->DenseRange(1, 16)
      ->Arg(32)
      ->Arg(64)
      ->Arg(1024)
      ->Arg(16384);
  benchmark::RegisterBenchmark((name + "/first").c_str(), BM_Dynamic<T, op, true>)
      ->Arg(16)
      ->Arg(64)
      ->Arg(256)
      ->Arg(1024)
      ->Arg(16384);
  return true;
}();
#elif defined(EIGEN_BENCH_PREDICATE_XPR)
const bool registered = [] {
  constexpr Op op = EIGEN_BENCH_PREDICATE_OP;
  const std::string name = std::string(opNames[op]) + "<" EIGEN_MAKESTRING(EIGEN_BENCH_PREDICATE_XPR) ">";
  benchmark::RegisterBenchmark(name.c_str(), BM_Fixed<EIGEN_BENCH_PREDICATE_XPR, op>);
  return true;
}();
#else
const bool registered = [] {
  benchmark::RegisterBenchmark("blockIsZero<float>", BM_BlockIsZero)->Arg(4)->Arg(16)->Arg(64)->Arg(256);
  return true;
}();
#endif

}  // namespace
}  // namespace Eigen
