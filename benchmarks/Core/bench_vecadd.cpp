// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

static void BM_VecAdd(benchmark::State& state) {
  int size = state.range(0);
  VectorXf a = VectorXf::Random(size);
  VectorXf b = VectorXf::Random(size);
  for (auto _ : state) {
    a = a + b;
    benchmark::DoNotOptimize(a.data());
  }
  state.SetBytesProcessed(state.iterations() * size * sizeof(float) * 3);
}
BENCHMARK(BM_VecAdd)->RangeMultiplier(4)->Range(64, 1 << 20);

static void BM_MatAdd(benchmark::State& state) {
  int n = state.range(0);
  MatrixXf a = MatrixXf::Random(n, n);
  MatrixXf b = MatrixXf::Random(n, n);
  for (auto _ : state) {
    a = a + b;
    benchmark::DoNotOptimize(a.data());
  }
  state.SetBytesProcessed(state.iterations() * n * n * sizeof(float) * 3);
}
BENCHMARK(BM_MatAdd)->RangeMultiplier(2)->Range(8, 512);

template <typename Scalar>
static void BM_Axpy(benchmark::State& state) {
  const Index size = state.range(0);
  using Vec = Matrix<Scalar, Dynamic, 1>;
  Vec x = Vec::Random(size), y = Vec::Random(size);
  Vec actual = y;
  actual += Scalar(0.75) * x;
  for (Index i = 0; i < size; ++i) {
    const long double expected = static_cast<long double>(y[i]) + Scalar(0.75) * static_cast<long double>(x[i]);
    const long double scale =
        numext::abs(static_cast<long double>(y[i])) + Scalar(0.75) * numext::abs(static_cast<long double>(x[i]));
    if (!(numext::abs(static_cast<long double>(actual[i]) - expected) <= 4 * NumTraits<Scalar>::epsilon() * scale)) {
      state.SkipWithError("AXPY differs from the scalar reference");
      return;
    }
  }
  for (auto _ : state) {
    benchmark::ClobberMemory();
    y += Scalar(0.75) * x;
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * size * sizeof(Scalar) * 3);
}
BENCHMARK(BM_Axpy<float>)
    ->RangeMultiplier(4)
    ->Range(1, 1 << 24)
    ->Arg(32)
    ->Arg(128)
    ->Arg(511)
    ->Arg(512)
    ->Arg(513)
    ->Arg(1023)
    ->Arg(1025)
    ->Arg(2048)
    ->Arg(8192)
    ->Arg(32768)
    ->Arg(524288)
    ->Arg(2097152)
    ->UseRealTime();
BENCHMARK(BM_Axpy<double>)
    ->RangeMultiplier(4)
    ->Range(1, 1 << 24)
    ->Arg(32)
    ->Arg(128)
    ->Arg(511)
    ->Arg(512)
    ->Arg(513)
    ->Arg(1023)
    ->Arg(1025)
    ->Arg(2048)
    ->Arg(8192)
    ->Arg(32768)
    ->Arg(524288)
    ->Arg(2097152)
    ->UseRealTime();

template <typename Scalar, bool Mixed>
static void BM_AxpyLayout(benchmark::State& state) {
  using Vec = Vector<Scalar, Dynamic>;
  const Index n = state.range(0), padding = 128 / sizeof(Scalar);
  Vec x_storage(n + padding), y_storage(n + padding), source = Vec::Random(n);
  Map<Vec> x(x_storage.data() + internal::first_aligned<64>(x_storage.data(), x_storage.size()) +
                 state.range(1) / sizeof(Scalar),
             n);
  Map<Vec> y(y_storage.data() + internal::first_aligned<64>(y_storage.data(), y_storage.size()) +
                 state.range(2) / sizeof(Scalar),
             n);
  x.setRandom();
  y = source.array() + Scalar(0.25);
  const Vec initial = y;
  y += Scalar(0.75) * x;
  for (Index i = 0; i < n; ++i) {
    const long double expected = static_cast<long double>(initial[i]) + 0.75L * static_cast<long double>(x[i]);
    const long double magnitude =
        numext::abs(static_cast<long double>(initial[i])) + 0.75L * numext::abs(static_cast<long double>(x[i]));
    if (!(numext::abs(static_cast<long double>(y[i]) - expected) <= 4 * NumTraits<Scalar>::epsilon() * magnitude)) {
      state.SkipWithError("AXPY layout differs from the scalar reference");
      return;
    }
  }
  for (auto _ : state) {
    EIGEN_IF_CONSTEXPR (Mixed) y = source.array() + Scalar(0.25);
    benchmark::ClobberMemory();
    y += Scalar(0.75) * x;
    benchmark::ClobberMemory();
    EIGEN_IF_CONSTEXPR (Mixed) {
      Scalar sum = y.sum();
      benchmark::DoNotOptimize(sum);
    }
    benchmark::DoNotOptimize(y.data());
  }
}

// clang-format off
#define AXPY_LAYOUT_SIZES ->ArgsProduct({{64, 256, 1024, 8192, 16384, 32768, 65536}, {0, 16}, {0, 16}}) ->UseRealTime()
BENCHMARK_TEMPLATE(BM_AxpyLayout, float, false) AXPY_LAYOUT_SIZES ->Name("AxpyLayout_float");
BENCHMARK_TEMPLATE(BM_AxpyLayout, double, false) AXPY_LAYOUT_SIZES ->Name("AxpyLayout_double");
BENCHMARK_TEMPLATE(BM_AxpyLayout, float, true) AXPY_LAYOUT_SIZES ->Name("AxpyMixed_float");
BENCHMARK_TEMPLATE(BM_AxpyLayout, double, true) AXPY_LAYOUT_SIZES ->Name("AxpyMixed_double");
#undef AXPY_LAYOUT_SIZES
// clang-format on
