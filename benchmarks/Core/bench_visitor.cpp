// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

namespace Eigen {
namespace {

enum Operation { All, Any, Count };

template <Operation Op, typename Derived>
Index reduce(const DenseBase<Derived>& input) {
  if (Op == All) return input.all();
  if (Op == Any) return input.any();
  return input.count();
}

template <Operation Op, typename Derived>
void measure(benchmark::State& state, const DenseBase<Derived>& input) {
  Index count = 0;
  for (Index col = 0; col < input.cols(); ++col)
    for (Index row = 0; row < input.rows(); ++row) count += input(row, col) != 0;
  const Index expected = Op == All ? Index(count == input.size()) : Op == Any ? Index(count != 0) : count;
  if (reduce<Op>(input) != expected) {
    state.SkipWithError("visitor result does not match the scalar reference");
    return;
  }
  for (auto _ : state) {
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(reduce<Op>(input));
  }
}

template <typename Scalar, int Options, Operation Op, bool Strided>
void BM_Visitor(benchmark::State& state) {
  const Index inner = state.range(0), outer = state.range(1);
  const Index rows = Options == RowMajor ? outer : inner;
  const Index cols = Options == RowMajor ? inner : outer;
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Options>;
  MatrixType input(rows, cols);
  for (Index i = 0; i < input.size(); ++i) input(i) = Scalar(Op == All || (Op == Count && i % 3 != 0));
  if (state.range(2) != 0) input(state.range(2) == 1 ? 0 : input.size() - 1) = Scalar(Op != All);
  if (Strided) {
    MatrixType storage = MatrixType::Zero(rows + 2, cols + 2);
    auto block = storage.block(1, 1, rows, cols);
    block = input;
    measure<Op>(state, block);
  } else {
    measure<Op>(state, input);
  }
}

#define EIGEN_BENCH_VISITOR(SCALAR, ORDER, OP, STRIDED)      \
  BENCHMARK_TEMPLATE(BM_Visitor, SCALAR, ORDER, OP, STRIDED) \
      ->Args({1, 4099, 0})                                   \
      ->Args({3, 1367, 0})                                   \
      ->Args({17, 241, 0})                                   \
      ->Args({64, 64, 0})                                    \
      ->Args({256, 256, 0})                                  \
      ->Args({3, 1367, 1})                                   \
      ->Args({3, 1367, 2})

#define EIGEN_BENCH_VISITORS(SCALAR, ORDER) \
  EIGEN_BENCH_VISITOR(SCALAR, ORDER, EIGEN_BENCH_VISITOR_OP, EIGEN_BENCH_VISITOR_STRIDED)

EIGEN_BENCH_VISITORS(float, ColMajor);
EIGEN_BENCH_VISITORS(float, RowMajor);
EIGEN_BENCH_VISITORS(bool, ColMajor);
EIGEN_BENCH_VISITORS(bool, RowMajor);

#undef EIGEN_BENCH_VISITORS
#undef EIGEN_BENCH_VISITOR

}  // namespace
}  // namespace Eigen
