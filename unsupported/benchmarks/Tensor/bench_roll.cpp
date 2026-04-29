// Benchmarks for Eigen TensorRoll.

#include <benchmark/benchmark.h>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

typedef float Scalar;

// --- Roll only the inner-most (contiguous) dimension. ---
static void BM_Roll_Inner(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);
  const int shift = state.range(2);

  Tensor<Scalar, 2> A(M, N);
  A.setRandom();

  array<Index, 2> rolls = {shift, 0};

  for (auto _ : state) {
    Tensor<Scalar, 2> B = A.roll(rolls);
    benchmark::DoNotOptimize(B.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(M) * N * sizeof(Scalar));
}

// --- Roll only an outer dimension. Inner dim stays contiguous. ---
static void BM_Roll_Outer(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);
  const int shift = state.range(2);

  Tensor<Scalar, 2> A(M, N);
  A.setRandom();

  array<Index, 2> rolls = {0, shift};

  for (auto _ : state) {
    Tensor<Scalar, 2> B = A.roll(rolls);
    benchmark::DoNotOptimize(B.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(M) * N * sizeof(Scalar));
}

// --- Roll every dimension. ---
static void BM_Roll_All(benchmark::State& state) {
  const int M = state.range(0);
  const int N = state.range(1);
  const int shift = state.range(2);

  Tensor<Scalar, 2> A(M, N);
  A.setRandom();

  array<Index, 2> rolls = {shift, shift};

  for (auto _ : state) {
    Tensor<Scalar, 2> B = A.roll(rolls);
    benchmark::DoNotOptimize(B.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(M) * N * sizeof(Scalar));
}

// --- 3D roll with the inner dim shifted. ---
static void BM_Roll_3D_Inner(benchmark::State& state) {
  const int D0 = state.range(0);
  const int D1 = state.range(1);
  const int D2 = state.range(2);

  Tensor<Scalar, 3> A(D0, D1, D2);
  A.setRandom();

  array<Index, 3> rolls = {D0 / 4, 0, 0};

  for (auto _ : state) {
    Tensor<Scalar, 3> B = A.roll(rolls);
    benchmark::DoNotOptimize(B.data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(D0) * D1 * D2 * sizeof(Scalar));
}

static void RollSizes(::benchmark::Benchmark* b) {
  for (int size : {64, 256, 1024}) {
    for (int shift : {1, 13}) {
      b->Args({size, size, shift});
    }
  }
}

static void Roll3DSizes(::benchmark::Benchmark* b) {
  b->Args({32, 32, 32});
  b->Args({64, 64, 64});
  b->Args({128, 128, 128});
}

BENCHMARK(BM_Roll_Inner)->Apply(RollSizes);
BENCHMARK(BM_Roll_Outer)->Apply(RollSizes);
BENCHMARK(BM_Roll_All)->Apply(RollSizes);
BENCHMARK(BM_Roll_3D_Inner)->Apply(Roll3DSizes);
