// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// GEMMs of 16-bit operands: cast to float (lazily converted while packing), mixed with a float operand, and the
// native product, which accumulates in float.

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

#ifndef SCALAR
#define SCALAR Eigen::half
#endif

using Scalar = SCALAR;
using Mat = Matrix<Scalar, Dynamic, Dynamic>;

template <typename Dst, typename Expected>
bool close_enough(const Dst& dst, const Expected& expected, float tol) {
  return (dst.template cast<float>() - expected).norm() <= tol * expected.norm();
}

struct Operands {
  MatrixXf af, bf, expected;
  Mat a, b;
  Operands(Index m, Index n, Index k) {
    a = MatrixXf::Random(m, k).cast<Scalar>();
    b = MatrixXf::Random(k, n).cast<Scalar>();
    af = a.cast<float>();
    bf = b.cast<float>();
    expected = af * bf;
  }
};

template <typename Dst, typename Xpr>
EIGEN_DONT_INLINE void assign(Dst& dst, const Xpr& xpr) {
  dst.noalias() = xpr;
}

template <typename Dst, typename Xpr>
EIGEN_DONT_INLINE void add(Dst& dst, const Xpr& xpr) {
  dst.noalias() += xpr;
}

static void set_counters(benchmark::State& state) {
  state.counters["GFLOPS"] =
      benchmark::Counter(2.0 * state.range(0) * state.range(1) * state.range(2),
                         benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// Kernel speed of light: the float product of float operands.
static void BM_Float(benchmark::State& state) {
  Operands op(state.range(0), state.range(1), state.range(2));
  MatrixXf c(op.af.rows(), op.bf.cols());
  for (auto _ : state) {
    assign(c, op.af * op.bf);
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  set_counters(state);
}

// C (float) = A.cast<float>() * B.cast<float>().
static void BM_CastCast(benchmark::State& state) {
  Operands op(state.range(0), state.range(1), state.range(2));
  MatrixXf c(op.af.rows(), op.bf.cols());
  assign(c, op.a.cast<float>() * op.b.cast<float>());
  if (!close_enough(c, op.expected, 1e-5f)) return state.SkipWithError("validation failed");
  for (auto _ : state) {
    assign(c, op.a.cast<float>() * op.b.cast<float>());
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  set_counters(state);
}

// C (float) = A (float) * B.cast<float>().
static void BM_FloatCast(benchmark::State& state) {
  Operands op(state.range(0), state.range(1), state.range(2));
  MatrixXf c(op.af.rows(), op.bf.cols());
  assign(c, op.af * op.b.cast<float>());
  if (!close_enough(c, op.expected, 1e-5f)) return state.SkipWithError("validation failed");
  for (auto _ : state) {
    assign(c, op.af * op.b.cast<float>());
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  set_counters(state);
}

// C (float) = A.cast<float>() * B (float).
static void BM_CastFloat(benchmark::State& state) {
  Operands op(state.range(0), state.range(1), state.range(2));
  MatrixXf c(op.af.rows(), op.bf.cols());
  assign(c, op.a.cast<float>() * op.bf);
  if (!close_enough(c, op.expected, 1e-5f)) return state.SkipWithError("validation failed");
  for (auto _ : state) {
    assign(c, op.a.cast<float>() * op.bf);
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  set_counters(state);
}

// C (16-bit) = (A.cast<float>() * B.cast<float>()).cast<Scalar>(): widening spelled out by the user.
static void BM_CastCastNarrow(benchmark::State& state) {
  Operands op(state.range(0), state.range(1), state.range(2));
  Mat c(op.a.rows(), op.b.cols());
  assign(c, (op.a.cast<float>() * op.b.cast<float>()).cast<Scalar>());
  if (!close_enough(c, op.expected, 1e-2f)) return state.SkipWithError("validation failed");
  for (auto _ : state) {
    assign(c, (op.a.cast<float>() * op.b.cast<float>()).cast<Scalar>());
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  set_counters(state);
}

// C (16-bit) = A * B, the native product. The loose bound admits a product summed in the 16-bit type.
static void BM_Native(benchmark::State& state) {
  Operands op(state.range(0), state.range(1), state.range(2));
  Mat c(op.a.rows(), op.b.cols());
  assign(c, op.a * op.b);
  if (!close_enough(c, op.expected, 1e-1f)) return state.SkipWithError("validation failed");
  for (auto _ : state) {
    assign(c, op.a * op.b);
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  set_counters(state);
}

// C (16-bit) += A * B.
static void BM_NativeAdd(benchmark::State& state) {
  Operands op(state.range(0), state.range(1), state.range(2));
  Mat c = Mat::Zero(op.a.rows(), op.b.cols());
  add(c, op.a * op.b);
  if (!close_enough(c, op.expected, 1e-1f)) return state.SkipWithError("validation failed");
  for (auto _ : state) {
    add(c, op.a * op.b);
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  set_counters(state);
}

// clang-format off
#define EIGEN_GEMM_CAST_SHAPES \
    ->Args({32, 32, 32})->Args({64, 64, 64})->Args({128, 128, 128})->Args({256, 256, 256}) \
    ->Args({512, 512, 512})->Args({1024, 1024, 1024}) \
    ->Args({4096, 128, 128})->Args({128, 4096, 128})->Args({128, 128, 4096})
BENCHMARK(BM_Float) EIGEN_GEMM_CAST_SHAPES;
BENCHMARK(BM_CastCast) EIGEN_GEMM_CAST_SHAPES;
BENCHMARK(BM_FloatCast) EIGEN_GEMM_CAST_SHAPES;
BENCHMARK(BM_CastFloat) EIGEN_GEMM_CAST_SHAPES;
BENCHMARK(BM_CastCastNarrow) EIGEN_GEMM_CAST_SHAPES;
BENCHMARK(BM_Native) EIGEN_GEMM_CAST_SHAPES;
BENCHMARK(BM_NativeAdd) EIGEN_GEMM_CAST_SHAPES;
// clang-format on
