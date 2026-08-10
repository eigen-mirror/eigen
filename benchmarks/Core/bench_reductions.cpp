// Benchmarks for full reductions: sum, prod, minCoeff, maxCoeff, mean, norm,
// squaredNorm, lpNorm<1>, lpNorm<Infinity>, plus the scalar reduction paths and
// the partial-reduction packet-segment tail.
//
// These are memory-bandwidth-bound for large vectors, so we report
// bytes processed rather than FLOPS.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

// --- Vector reductions (1-D) ---

template <typename Scalar>
static void BM_VectorSum(benchmark::State& state) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, 1> v = Matrix<Scalar, Dynamic, 1>::Random(n);
  for (auto _ : state) {
    Scalar s = v.sum();
    benchmark::DoNotOptimize(s);
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar));
}

template <typename Scalar>
static void BM_VectorProd(benchmark::State& state) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, 1> v = Matrix<Scalar, Dynamic, 1>::Constant(n, Scalar(1));
  // Use values near 1 to avoid overflow/underflow.
  v += Scalar(0.001) * Matrix<Scalar, Dynamic, 1>::Random(n);
  for (auto _ : state) {
    Scalar p = v.prod();
    benchmark::DoNotOptimize(p);
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar));
}

template <typename Scalar>
static void BM_VectorMinCoeff(benchmark::State& state) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, 1> v = Matrix<Scalar, Dynamic, 1>::Random(n);
  for (auto _ : state) {
    Scalar m = v.minCoeff();
    benchmark::DoNotOptimize(m);
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar));
}

template <typename Scalar>
static void BM_VectorMaxCoeff(benchmark::State& state) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, 1> v = Matrix<Scalar, Dynamic, 1>::Random(n);
  for (auto _ : state) {
    Scalar m = v.maxCoeff();
    benchmark::DoNotOptimize(m);
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar));
}

template <typename Scalar>
static void BM_VectorMean(benchmark::State& state) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, 1> v = Matrix<Scalar, Dynamic, 1>::Random(n);
  for (auto _ : state) {
    Scalar m = v.mean();
    benchmark::DoNotOptimize(m);
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar));
}

template <typename Scalar>
static void BM_VectorSquaredNorm(benchmark::State& state) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, 1> v = Matrix<Scalar, Dynamic, 1>::Random(n);
  for (auto _ : state) {
    Scalar s = v.squaredNorm();
    benchmark::DoNotOptimize(s);
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar));
}

template <typename Scalar>
static void BM_VectorNorm(benchmark::State& state) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, 1> v = Matrix<Scalar, Dynamic, 1>::Random(n);
  for (auto _ : state) {
    Scalar s = v.norm();
    benchmark::DoNotOptimize(s);
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar));
}

template <typename Scalar>
static void BM_VectorLpNorm1(benchmark::State& state) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, 1> v = Matrix<Scalar, Dynamic, 1>::Random(n);
  for (auto _ : state) {
    Scalar s = v.template lpNorm<1>();
    benchmark::DoNotOptimize(s);
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar));
}

template <typename Scalar>
static void BM_VectorLpNormInf(benchmark::State& state) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, 1> v = Matrix<Scalar, Dynamic, 1>::Random(n);
  for (auto _ : state) {
    Scalar s = v.template lpNorm<Infinity>();
    benchmark::DoNotOptimize(s);
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar));
}

// --- Matrix reductions (2-D) ---

template <typename Scalar>
static void BM_MatrixSum(benchmark::State& state) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, Dynamic> m = Matrix<Scalar, Dynamic, Dynamic>::Random(n, n);
  for (auto _ : state) {
    Scalar s = m.sum();
    benchmark::DoNotOptimize(s);
  }
  state.SetBytesProcessed(state.iterations() * n * n * sizeof(Scalar));
}

template <typename Scalar>
static void BM_MatrixNorm(benchmark::State& state) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, Dynamic> m = Matrix<Scalar, Dynamic, Dynamic>::Random(n, n);
  for (auto _ : state) {
    Scalar s = m.norm();
    benchmark::DoNotOptimize(s);
  }
  state.SetBytesProcessed(state.iterations() * n * n * sizeof(Scalar));
}

// --- Reductions without a packet path ---

// A user functor has no packetOp, so redux takes a scalar traversal: LinearTraversal
// with LinearAccessBit, DefaultTraversal without it. Unmarked, so operand order is
// preserved (serial below the tree cutoff, ordered pairwise tree above it).
template <typename Scalar>
struct UserSumOp {
  EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return a + b; }
};

// The same functor marked commutative: redux may reorder operands into independent
// accumulators.
template <typename Scalar>
struct CommutativeUserSumOp {
  EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return a + b; }
};

namespace Eigen {
namespace internal {
template <typename Scalar>
struct functor_is_commutative<CommutativeUserSumOp<Scalar>> : std::true_type {};
}  // namespace internal
}  // namespace Eigen

template <typename Scalar, template <class> class Op>
static void BM_VectorReduxOp(benchmark::State& state) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, 1> v = Matrix<Scalar, Dynamic, 1>::Random(n);
  Op<Scalar> op;
  for (auto _ : state) {
    Scalar s = v.redux(op);
    benchmark::DoNotOptimize(s);
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar));
}

// A block without LinearAccessBit, reduced by outer/inner index.
template <typename Scalar, template <class> class Op>
static void BM_BlockReduxOp(benchmark::State& state) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, Dynamic> m = Matrix<Scalar, Dynamic, Dynamic>::Random(n + 1, n + 1);
  Op<Scalar> op;
  for (auto _ : state) {
    Scalar s = m.block(1, 1, n, n).redux(op);
    benchmark::DoNotOptimize(s);
  }
  state.SetBytesProcessed(state.iterations() * n * n * sizeof(Scalar));
}

// A row of a column-major matrix is strided, so sum() falls back to the scalar linear
// path; scalar_sum_op is marked commutative, so this reaches the reordering reduction.
template <typename Scalar>
static void BM_StridedRowSum(benchmark::State& state) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, Dynamic> m = Matrix<Scalar, Dynamic, Dynamic>::Random(64, n);
  for (auto _ : state) {
    Scalar s = m.row(32).sum();
    benchmark::DoNotOptimize(s);
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar));
}

// --- Partial reduction with a ragged packet tail ---

// An odd column count leaves a trailing partial packet, so the assignment ends with one
// packetSegment reduction over the whole outer dimension.
template <typename Scalar>
static void BM_ColwiseSumRaggedTail(benchmark::State& state) {
  const Index rows = state.range(0);
  const Index cols = state.range(1);
  Matrix<Scalar, Dynamic, Dynamic, RowMajor> m = Matrix<Scalar, Dynamic, Dynamic, RowMajor>::Random(rows, cols);
  Matrix<Scalar, 1, Dynamic> r(cols);
  for (auto _ : state) {
    r = m.colwise().sum();
    benchmark::DoNotOptimize(r.data());
  }
  state.SetBytesProcessed(state.iterations() * rows * cols * sizeof(Scalar));
}

// --- Size configurations ---

// clang-format off
#define VECTOR_SIZES ->Arg(64)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384)->Arg(65536)->Arg(262144)->Arg(1048576)
#define MATRIX_SIZES ->Arg(8)->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(512)->Arg(1024)
#define RAGGED_SIZES ->Args({4096, 5})->Args({4096, 9})->Args({4096, 17})->Args({65536, 5})->Args({65536, 9})->Args({65536, 17})
// Scalar redux paths change shape at the small-size cutoffs, so sample densely there.
#define REDUX_SIZES ->Arg(8)->Arg(16)->Arg(24)->Arg(32)->Arg(64)->Arg(128)->Arg(192)->Arg(256)->Arg(1024)->Arg(16384)->Arg(262144)

// --- Register: float ---
BENCHMARK(BM_VectorSum<float>) VECTOR_SIZES ->Name("VectorSum_float");
BENCHMARK(BM_VectorProd<float>) VECTOR_SIZES ->Name("VectorProd_float");
BENCHMARK(BM_VectorMinCoeff<float>) VECTOR_SIZES ->Name("VectorMinCoeff_float");
BENCHMARK(BM_VectorMaxCoeff<float>) VECTOR_SIZES ->Name("VectorMaxCoeff_float");
BENCHMARK(BM_VectorMean<float>) VECTOR_SIZES ->Name("VectorMean_float");
BENCHMARK(BM_VectorSquaredNorm<float>) VECTOR_SIZES ->Name("VectorSquaredNorm_float");
BENCHMARK(BM_VectorNorm<float>) VECTOR_SIZES ->Name("VectorNorm_float");
BENCHMARK(BM_VectorLpNorm1<float>) VECTOR_SIZES ->Name("VectorLpNorm1_float");
BENCHMARK(BM_VectorLpNormInf<float>) VECTOR_SIZES ->Name("VectorLpNormInf_float");
BENCHMARK(BM_MatrixSum<float>) MATRIX_SIZES ->Name("MatrixSum_float");
BENCHMARK(BM_MatrixNorm<float>) MATRIX_SIZES ->Name("MatrixNorm_float");
BENCHMARK(BM_VectorReduxOp<float, UserSumOp>) REDUX_SIZES ->Name("VectorReduxUserOp_float");
BENCHMARK(BM_VectorReduxOp<float, CommutativeUserSumOp>) REDUX_SIZES ->Name("VectorReduxCommutativeOp_float");
BENCHMARK(BM_BlockReduxOp<float, UserSumOp>) MATRIX_SIZES ->Name("BlockReduxUserOp_float");
BENCHMARK(BM_BlockReduxOp<float, CommutativeUserSumOp>) MATRIX_SIZES ->Name("BlockReduxCommutativeOp_float");
BENCHMARK(BM_StridedRowSum<float>) REDUX_SIZES ->Name("StridedRowSum_float");
BENCHMARK(BM_ColwiseSumRaggedTail<float>) RAGGED_SIZES ->Name("ColwiseSumRaggedTail_float");

// --- Register: double ---
BENCHMARK(BM_VectorSum<double>) VECTOR_SIZES ->Name("VectorSum_double");
BENCHMARK(BM_VectorProd<double>) VECTOR_SIZES ->Name("VectorProd_double");
BENCHMARK(BM_VectorMinCoeff<double>) VECTOR_SIZES ->Name("VectorMinCoeff_double");
BENCHMARK(BM_VectorMaxCoeff<double>) VECTOR_SIZES ->Name("VectorMaxCoeff_double");
BENCHMARK(BM_VectorMean<double>) VECTOR_SIZES ->Name("VectorMean_double");
BENCHMARK(BM_VectorSquaredNorm<double>) VECTOR_SIZES ->Name("VectorSquaredNorm_double");
BENCHMARK(BM_VectorNorm<double>) VECTOR_SIZES ->Name("VectorNorm_double");
BENCHMARK(BM_VectorLpNorm1<double>) VECTOR_SIZES ->Name("VectorLpNorm1_double");
BENCHMARK(BM_VectorLpNormInf<double>) VECTOR_SIZES ->Name("VectorLpNormInf_double");
BENCHMARK(BM_MatrixSum<double>) MATRIX_SIZES ->Name("MatrixSum_double");
BENCHMARK(BM_MatrixNorm<double>) MATRIX_SIZES ->Name("MatrixNorm_double");
BENCHMARK(BM_VectorReduxOp<double, UserSumOp>) REDUX_SIZES ->Name("VectorReduxUserOp_double");
BENCHMARK(BM_VectorReduxOp<double, CommutativeUserSumOp>) REDUX_SIZES ->Name("VectorReduxCommutativeOp_double");
BENCHMARK(BM_BlockReduxOp<double, UserSumOp>) MATRIX_SIZES ->Name("BlockReduxUserOp_double");
BENCHMARK(BM_BlockReduxOp<double, CommutativeUserSumOp>) MATRIX_SIZES ->Name("BlockReduxCommutativeOp_double");
BENCHMARK(BM_StridedRowSum<double>) REDUX_SIZES ->Name("StridedRowSum_double");
BENCHMARK(BM_ColwiseSumRaggedTail<double>) RAGGED_SIZES ->Name("ColwiseSumRaggedTail_double");

#undef VECTOR_SIZES
#undef MATRIX_SIZES
#undef RAGGED_SIZES
#undef REDUX_SIZES
// clang-format on
