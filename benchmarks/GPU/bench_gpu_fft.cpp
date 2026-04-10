// GPU FFT benchmarks: GpuFFT 1D and 2D throughput.
//
// Measures forward and inverse FFT performance across a range of sizes,
// including plan-amortized (reuse) and cold-start (new plan) scenarios.
//
// Usage:
//   cmake --build build-bench-gpu --target bench_gpu_fft
//   ./build-bench-gpu/bench_gpu_fft
//
// Profiling:
//   nsys profile --trace=cuda ./build-bench-gpu/bench_gpu_fft

#include <benchmark/benchmark.h>

#include <Eigen/GPU>

using namespace Eigen;

#ifndef SCALAR
#define SCALAR float
#endif

using Scalar = SCALAR;
using Complex = std::complex<Scalar>;
using CVec = Matrix<Complex, Dynamic, 1>;
using RVec = Matrix<Scalar, Dynamic, 1>;
using CMat = Matrix<Complex, Dynamic, Dynamic>;

// CUDA warm-up: ensure the GPU is initialized before timing.
static void cuda_warmup() {
  static bool done = false;
  if (!done) {
    void* p;
    cudaMalloc(&p, 1);
    cudaFree(p);
    done = true;
  }
}

// --------------------------------------------------------------------------
// 1D C2C Forward
// --------------------------------------------------------------------------

static void BM_GpuFFT_1D_C2C_Fwd(benchmark::State& state) {
  cuda_warmup();
  const Index n = state.range(0);
  CVec x = CVec::Random(n);
  GpuFFT<Scalar> fft;

  // Warm up plan.
  CVec tmp = fft.fwd(x);

  for (auto _ : state) {
    benchmark::DoNotOptimize(fft.fwd(x));
  }
  state.SetItemsProcessed(state.iterations() * n);
  state.SetBytesProcessed(state.iterations() * n * sizeof(Complex) * 2);  // read + write
}

BENCHMARK(BM_GpuFFT_1D_C2C_Fwd)->RangeMultiplier(4)->Range(1 << 10, 1 << 22);

// --------------------------------------------------------------------------
// 1D C2C Inverse
// --------------------------------------------------------------------------

static void BM_GpuFFT_1D_C2C_Inv(benchmark::State& state) {
  cuda_warmup();
  const Index n = state.range(0);
  CVec x = CVec::Random(n);
  GpuFFT<Scalar> fft;
  CVec X = fft.fwd(x);

  for (auto _ : state) {
    benchmark::DoNotOptimize(fft.inv(X));
  }
  state.SetItemsProcessed(state.iterations() * n);
  state.SetBytesProcessed(state.iterations() * n * sizeof(Complex) * 2);
}

BENCHMARK(BM_GpuFFT_1D_C2C_Inv)->RangeMultiplier(4)->Range(1 << 10, 1 << 22);

// --------------------------------------------------------------------------
// 1D R2C Forward
// --------------------------------------------------------------------------

static void BM_GpuFFT_1D_R2C_Fwd(benchmark::State& state) {
  cuda_warmup();
  const Index n = state.range(0);
  RVec r = RVec::Random(n);
  GpuFFT<Scalar> fft;

  // Warm up plan.
  CVec tmp = fft.fwd(r);

  for (auto _ : state) {
    benchmark::DoNotOptimize(fft.fwd(r));
  }
  state.SetItemsProcessed(state.iterations() * n);
  state.SetBytesProcessed(state.iterations() * (n * sizeof(Scalar) + (n / 2 + 1) * sizeof(Complex)));
}

BENCHMARK(BM_GpuFFT_1D_R2C_Fwd)->RangeMultiplier(4)->Range(1 << 10, 1 << 22);

// --------------------------------------------------------------------------
// 1D C2R Inverse
// --------------------------------------------------------------------------

static void BM_GpuFFT_1D_C2R_Inv(benchmark::State& state) {
  cuda_warmup();
  const Index n = state.range(0);
  RVec r = RVec::Random(n);
  GpuFFT<Scalar> fft;
  CVec R = fft.fwd(r);

  for (auto _ : state) {
    benchmark::DoNotOptimize(fft.invReal(R, n));
  }
  state.SetItemsProcessed(state.iterations() * n);
  state.SetBytesProcessed(state.iterations() * ((n / 2 + 1) * sizeof(Complex) + n * sizeof(Scalar)));
}

BENCHMARK(BM_GpuFFT_1D_C2R_Inv)->RangeMultiplier(4)->Range(1 << 10, 1 << 22);

// --------------------------------------------------------------------------
// 2D C2C Forward
// --------------------------------------------------------------------------

static void BM_GpuFFT_2D_C2C_Fwd(benchmark::State& state) {
  cuda_warmup();
  const Index n = state.range(0);  // square n x n
  CMat A = CMat::Random(n, n);
  GpuFFT<Scalar> fft;

  // Warm up plan.
  CMat tmp = fft.fwd2d(A);

  for (auto _ : state) {
    benchmark::DoNotOptimize(fft.fwd2d(A));
  }
  state.SetItemsProcessed(state.iterations() * n * n);
  state.SetBytesProcessed(state.iterations() * n * n * sizeof(Complex) * 2);
}

BENCHMARK(BM_GpuFFT_2D_C2C_Fwd)->RangeMultiplier(2)->Range(64, 4096);

// --------------------------------------------------------------------------
// 2D C2C Roundtrip (fwd + inv)
// --------------------------------------------------------------------------

static void BM_GpuFFT_2D_C2C_Roundtrip(benchmark::State& state) {
  cuda_warmup();
  const Index n = state.range(0);
  CMat A = CMat::Random(n, n);
  GpuFFT<Scalar> fft;

  // Warm up plans.
  CMat tmp = fft.inv2d(fft.fwd2d(A));

  for (auto _ : state) {
    CMat B = fft.fwd2d(A);
    benchmark::DoNotOptimize(fft.inv2d(B));
  }
  state.SetItemsProcessed(state.iterations() * n * n * 2);  // fwd + inv
  state.SetBytesProcessed(state.iterations() * n * n * sizeof(Complex) * 4);
}

BENCHMARK(BM_GpuFFT_2D_C2C_Roundtrip)->RangeMultiplier(2)->Range(64, 4096);

// --------------------------------------------------------------------------
// 1D Cold start (includes plan creation)
// --------------------------------------------------------------------------

static void BM_GpuFFT_1D_ColdStart(benchmark::State& state) {
  cuda_warmup();
  const Index n = state.range(0);
  CVec x = CVec::Random(n);

  for (auto _ : state) {
    GpuFFT<Scalar> fft;  // new object = new plans
    benchmark::DoNotOptimize(fft.fwd(x));
  }
  state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_GpuFFT_1D_ColdStart)->RangeMultiplier(4)->Range(1 << 10, 1 << 20);
