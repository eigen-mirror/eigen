// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

template <typename Scalar, int Side = OnTheLeft, int TriOrder = ColMajor, int OtherOrder = ColMajor,
          unsigned int Mode = Lower>
static void BM_TrsmPacket(benchmark::State& state) {
  using Triangle = Matrix<Scalar, Dynamic, Dynamic, TriOrder>;
  using Operand = Matrix<Scalar, Dynamic, Dynamic, OtherOrder>;
  const Index n = state.range(0), nrhs = state.range(1);
  const Index rows = Side == OnTheLeft ? n : nrhs, cols = Side == OnTheLeft ? nrhs : n;
  Triangle a = Triangle::Random(n, n) / Scalar(n);
  a.diagonal().array() += Scalar(2);
  Operand b = Operand::Random(rows, cols), x = b;
  a.template triangularView<Mode>().template solveInPlace<Side>(x);
  Operand residual(rows, cols);
  if (Side == OnTheLeft)
    residual.noalias() = a.template triangularView<Mode>() * x;
  else
    residual.noalias() = x * a.template triangularView<Mode>();
  residual -= b;
  const Scalar bound = Scalar(8 * n) * NumTraits<Scalar>::epsilon() *
                       (a.template triangularView<Mode>().toDenseMatrix().norm() * x.norm() + b.norm());
  if (!(residual.norm() <= bound)) {
    state.SkipWithError("TRSM residual exceeds the rounding-error bound");
    return;
  }
  for (auto _ : state) {
    x = b;
    a.template triangularView<Mode>().template solveInPlace<Side>(x);
    benchmark::DoNotOptimize(x.data());
    benchmark::ClobberMemory();
  }
  state.counters["flops"] =
      benchmark::Counter(double(n) * (n + 1) * nrhs, benchmark::Counter::kIsIterationInvariantRate);
}

#define TRSM_SHAPES        \
  Args({8, 8})             \
      ->Args({12, 12})     \
      ->Args({16, 16})     \
      ->Args({24, 24})     \
      ->Args({31, 31})     \
      ->Args({32, 32})     \
      ->Args({33, 33})     \
      ->Args({48, 48})     \
      ->Args({63, 63})     \
      ->Args({64, 64})     \
      ->Args({65, 65})     \
      ->Args({95, 95})     \
      ->Args({96, 96})     \
      ->Args({97, 97})     \
      ->Args({100, 100})   \
      ->Args({127, 127})   \
      ->Args({128, 128})   \
      ->Args({129, 129})   \
      ->Args({192, 192})   \
      ->Args({256, 256})   \
      ->Args({257, 257})   \
      ->Args({512, 512})   \
      ->Args({1024, 1024}) \
      ->Args({2048, 2048}) \
      ->Args({4096, 4096}) \
      ->Args({64, 1})      \
      ->Args({64, 4})      \
      ->Args({64, 8})      \
      ->Args({64, 16})     \
      ->Args({64, 17})     \
      ->Args({64, 4096})   \
      ->Args({128, 4})     \
      ->Args({128, 8})     \
      ->Args({128, 17})    \
      ->Args({1024, 8})    \
      ->Args({1024, 32})   \
      ->Args({1024, 128})  \
      ->Args({4096, 32})

BENCHMARK(BM_TrsmPacket<float>)->TRSM_SHAPES->Name("TRSM_float");
BENCHMARK(BM_TrsmPacket<double>)->TRSM_SHAPES->Name("TRSM_double");
BENCHMARK(BM_TrsmPacket<float, OnTheLeft, RowMajor>)->TRSM_SHAPES->Name("TRSM_float_RowTri");
BENCHMARK(BM_TrsmPacket<double, OnTheLeft, RowMajor>)->TRSM_SHAPES->Name("TRSM_double_RowTri");
BENCHMARK(BM_TrsmPacket<float, OnTheRight>)->TRSM_SHAPES->Name("TRSM_float_Right");
BENCHMARK(BM_TrsmPacket<double, OnTheRight>)->TRSM_SHAPES->Name("TRSM_double_Right");
BENCHMARK(BM_TrsmPacket<float, OnTheLeft, ColMajor, ColMajor, Upper>)
    ->ArgsProduct({{24, 64, 128, 512}, {17, 64, 512}})
    ->Name("TRSM_float_Upper");
BENCHMARK(BM_TrsmPacket<double, OnTheLeft, ColMajor, ColMajor, Upper>)
    ->ArgsProduct({{24, 64, 128, 512}, {17, 64, 512}})
    ->Name("TRSM_double_Upper");
BENCHMARK(BM_TrsmPacket<float, OnTheRight, ColMajor, RowMajor>)
    ->ArgsProduct({{24, 64, 128, 512}, {17, 64, 512}})
    ->Name("TRSM_float_RightRow");
BENCHMARK(BM_TrsmPacket<double, OnTheRight, ColMajor, RowMajor>)
    ->ArgsProduct({{24, 64, 128, 512}, {17, 64, 512}})
    ->Name("TRSM_double_RightRow");

#undef TRSM_SHAPES
