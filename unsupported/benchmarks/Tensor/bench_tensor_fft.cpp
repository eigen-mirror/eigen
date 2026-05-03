// Benchmarks for Eigen Tensor FFT.

#include <benchmark/benchmark.h>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

#ifndef SCALAR
#define SCALAR float
#endif

typedef SCALAR Scalar;

// --- 1D FFT ---
static void BM_TensorFFT_1D(benchmark::State& state) {
  const int N = state.range(0);

  Tensor<Scalar, 1> input(N);
  input.setRandom();

  Eigen::array<int, 1> fft_dims = {0};

  for (auto _ : state) {
    Tensor<std::complex<Scalar>, 1> result = input.template fft<BothParts, FFT_FORWARD>(fft_dims);
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
  double mflops = 5.0 * N * std::log2(static_cast<double>(N)) / 2.0;  // real->complex
  state.counters["MFLOPS"] =
      benchmark::Counter(mflops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// --- 2D FFT ---
static void BM_TensorFFT_2D(benchmark::State& state) {
  const int N = state.range(0);

  Tensor<Scalar, 2> input(N, N);
  input.setRandom();

  Eigen::array<int, 2> fft_dims = {0, 1};

  for (auto _ : state) {
    Tensor<std::complex<Scalar>, 2> result = input.template fft<BothParts, FFT_FORWARD>(fft_dims);
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
  double total = N * N;
  double mflops = 5.0 * total * std::log2(static_cast<double>(N));
  state.counters["MFLOPS"] =
      benchmark::Counter(mflops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// --- 1D inverse FFT ---
static void BM_TensorIFFT_1D(benchmark::State& state) {
  const int N = state.range(0);

  Tensor<std::complex<Scalar>, 1> input(N);
  input.setRandom();

  Eigen::array<int, 1> fft_dims = {0};

  for (auto _ : state) {
    Tensor<std::complex<Scalar>, 1> result = input.template fft<BothParts, FFT_REVERSE>(fft_dims);
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
  double mflops = 5.0 * N * std::log2(static_cast<double>(N));
  state.counters["MFLOPS"] =
      benchmark::Counter(mflops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// clang-format off
#define FFT_SIZES ->Arg(64)->Arg(256)->Arg(1024)->Arg(4096)
// clang-format on

BENCHMARK(BM_TensorFFT_1D) FFT_SIZES;
BENCHMARK(BM_TensorFFT_2D) FFT_SIZES;
BENCHMARK(BM_TensorIFFT_1D) FFT_SIZES;
