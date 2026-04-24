// GPU chaining benchmarks: measure async pipeline efficiency.
//
// Compares:
//   1. Host round-trip per solve (baseline)
//   2. DeviceMatrix chaining (no host round-trip between solves)
//   3. Varying chain lengths (1, 2, 4, 8 consecutive solves)
//
// For Nsight Systems: look for gaps between kernel launches in the timeline.
// Host round-trip creates visible idle gaps; chaining should show back-to-back kernels.
//
//   nsys profile --trace=cuda,nvtx ./bench_chaining

#include <benchmark/benchmark.h>

#include <Eigen/Cholesky>
#include <unsupported/Eigen/GPU>

using namespace Eigen;

#ifndef SCALAR
#define SCALAR double
#endif

using Scalar = SCALAR;
using Mat = Matrix<Scalar, Dynamic, Dynamic>;

static Mat make_spd(Index n) {
  Mat M = Mat::Random(n, n);
  return M.adjoint() * M + Mat::Identity(n, n) * static_cast<Scalar>(n);
}

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
// Baseline: host round-trip between every solve
// --------------------------------------------------------------------------

static void BM_Chain_HostRoundtrip(benchmark::State& state) {
  cuda_warmup();
  const Index n = state.range(0);
  const int chain_len = static_cast<int>(state.range(1));

  Mat A = make_spd(n);
  Mat B = Mat::Random(n, 1);
  GpuLLT<Scalar> llt(A);

  for (auto _ : state) {
    Mat X = B;
    for (int i = 0; i < chain_len; ++i) {
      X = llt.solve(X);  // host → device → host each time
    }
    benchmark::DoNotOptimize(X.data());
  }

  state.counters["n"] = n;
  state.counters["chain"] = chain_len;
  state.counters["solves/iter"] = chain_len;
}

// --------------------------------------------------------------------------
// DeviceMatrix chaining: no host round-trip between solves
// --------------------------------------------------------------------------

static void BM_Chain_Device(benchmark::State& state) {
  cuda_warmup();
  const Index n = state.range(0);
  const int chain_len = static_cast<int>(state.range(1));

  Mat A = make_spd(n);
  Mat B = Mat::Random(n, 1);
  GpuLLT<Scalar> llt(A);
  auto d_B = DeviceMatrix<Scalar>::fromHost(B);

  for (auto _ : state) {
    DeviceMatrix<Scalar> d_X = llt.solve(d_B);
    for (int i = 1; i < chain_len; ++i) {
      d_X = llt.solve(d_X);  // device → device, fully async
    }
    Mat X = d_X.toHost();  // single sync at end
    benchmark::DoNotOptimize(X.data());
  }

  state.counters["n"] = n;
  state.counters["chain"] = chain_len;
  state.counters["solves/iter"] = chain_len;
}

// --------------------------------------------------------------------------
// DeviceMatrix chaining with async download (overlap D2H with next iteration)
// --------------------------------------------------------------------------

static void BM_Chain_DeviceAsync(benchmark::State& state) {
  cuda_warmup();
  const Index n = state.range(0);
  const int chain_len = static_cast<int>(state.range(1));

  Mat A = make_spd(n);
  Mat B = Mat::Random(n, 1);
  GpuLLT<Scalar> llt(A);
  auto d_B = DeviceMatrix<Scalar>::fromHost(B);

  for (auto _ : state) {
    DeviceMatrix<Scalar> d_X = llt.solve(d_B);
    for (int i = 1; i < chain_len; ++i) {
      d_X = llt.solve(d_X);
    }
    auto transfer = d_X.toHostAsync();
    Mat X = transfer.get();
    benchmark::DoNotOptimize(X.data());
  }

  state.counters["n"] = n;
  state.counters["chain"] = chain_len;
  state.counters["solves/iter"] = chain_len;
}

// --------------------------------------------------------------------------
// Pure GPU chain (no download — measures kernel-only throughput)
// --------------------------------------------------------------------------

static void BM_Chain_DeviceNoDownload(benchmark::State& state) {
  cuda_warmup();
  const Index n = state.range(0);
  const int chain_len = static_cast<int>(state.range(1));

  Mat A = make_spd(n);
  Mat B = Mat::Random(n, 1);
  GpuLLT<Scalar> llt(A);
  auto d_B = DeviceMatrix<Scalar>::fromHost(B);

  for (auto _ : state) {
    DeviceMatrix<Scalar> d_X = llt.solve(d_B);
    for (int i = 1; i < chain_len; ++i) {
      d_X = llt.solve(d_X);
    }
    cudaStreamSynchronize(llt.stream());
    benchmark::DoNotOptimize(d_X.data());
  }

  state.counters["n"] = n;
  state.counters["chain"] = chain_len;
  state.counters["solves/iter"] = chain_len;
}

// --------------------------------------------------------------------------
// Compute + solve chain (full pipeline: factorize, then chain solves)
// --------------------------------------------------------------------------

static void BM_FullPipeline_Host(benchmark::State& state) {
  cuda_warmup();
  const Index n = state.range(0);
  const int chain_len = static_cast<int>(state.range(1));

  Mat A = make_spd(n);
  Mat B = Mat::Random(n, 1);

  for (auto _ : state) {
    GpuLLT<Scalar> llt(A);
    Mat X = B;
    for (int i = 0; i < chain_len; ++i) {
      X = llt.solve(X);
    }
    benchmark::DoNotOptimize(X.data());
  }

  state.counters["n"] = n;
  state.counters["chain"] = chain_len;
}

static void BM_FullPipeline_Device(benchmark::State& state) {
  cuda_warmup();
  const Index n = state.range(0);
  const int chain_len = static_cast<int>(state.range(1));

  Mat A = make_spd(n);
  Mat B = Mat::Random(n, 1);

  for (auto _ : state) {
    auto d_A = DeviceMatrix<Scalar>::fromHost(A);
    auto d_B = DeviceMatrix<Scalar>::fromHost(B);
    GpuLLT<Scalar> llt;
    llt.compute(d_A);
    DeviceMatrix<Scalar> d_X = llt.solve(d_B);
    for (int i = 1; i < chain_len; ++i) {
      d_X = llt.solve(d_X);
    }
    Mat X = d_X.toHost();
    benchmark::DoNotOptimize(X.data());
  }

  state.counters["n"] = n;
  state.counters["chain"] = chain_len;
}

// --------------------------------------------------------------------------
// Registration
// --------------------------------------------------------------------------

// clang-format off
// Args: {matrix_size, chain_length}
BENCHMARK(BM_Chain_HostRoundtrip)->ArgsProduct({{64, 256, 1024, 4096}, {1, 2, 4, 8}})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Chain_Device)->ArgsProduct({{64, 256, 1024, 4096}, {1, 2, 4, 8}})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Chain_DeviceAsync)->ArgsProduct({{64, 256, 1024, 4096}, {1, 2, 4, 8}})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Chain_DeviceNoDownload)->ArgsProduct({{64, 256, 1024, 4096}, {1, 2, 4, 8}})->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_FullPipeline_Host)->ArgsProduct({{256, 1024, 4096}, {1, 4}})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_FullPipeline_Device)->ArgsProduct({{256, 1024, 4096}, {1, 4}})->Unit(benchmark::kMicrosecond);
// clang-format on
