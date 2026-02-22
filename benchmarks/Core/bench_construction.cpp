// Benchmarks for matrix initialization / construction.
//
// Tests setZero, setRandom, setIdentity, LinSpaced, Zero(), Constant()
// for both dynamic and small fixed-size matrices.

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

// --- Dynamic-size construction ---

template <typename Scalar>
static void BM_SetZero(benchmark::State& state) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, Dynamic> m(n, n);
  for (auto _ : state) {
    m.setZero();
    benchmark::DoNotOptimize(m.data());
  }
  state.SetBytesProcessed(state.iterations() * n * n * sizeof(Scalar));
}

template <typename Scalar>
static void BM_SetRandom(benchmark::State& state) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, Dynamic> m(n, n);
  for (auto _ : state) {
    m.setRandom();
    benchmark::DoNotOptimize(m.data());
  }
  state.SetBytesProcessed(state.iterations() * n * n * sizeof(Scalar));
}

template <typename Scalar>
static void BM_SetIdentity(benchmark::State& state) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, Dynamic> m(n, n);
  for (auto _ : state) {
    m.setIdentity();
    benchmark::DoNotOptimize(m.data());
  }
  state.SetBytesProcessed(state.iterations() * n * n * sizeof(Scalar));
}

template <typename Scalar>
static void BM_SetConstant(benchmark::State& state) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, Dynamic> m(n, n);
  for (auto _ : state) {
    m.setConstant(Scalar(42));
    benchmark::DoNotOptimize(m.data());
  }
  state.SetBytesProcessed(state.iterations() * n * n * sizeof(Scalar));
}

template <typename Scalar>
static void BM_LinSpaced(benchmark::State& state) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, 1> v(n);
  for (auto _ : state) {
    v = Matrix<Scalar, Dynamic, 1>::LinSpaced(n, Scalar(0), Scalar(1));
    benchmark::DoNotOptimize(v.data());
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar));
}

// --- Fixed-size construction ---

template <typename Scalar, int N>
static void BM_FixedSetZero(benchmark::State& state) {
  Matrix<Scalar, N, N> m;
  for (auto _ : state) {
    m.setZero();
    benchmark::DoNotOptimize(m.data());
  }
  state.SetBytesProcessed(state.iterations() * N * N * sizeof(Scalar));
}

template <typename Scalar, int N>
static void BM_FixedSetRandom(benchmark::State& state) {
  Matrix<Scalar, N, N> m;
  for (auto _ : state) {
    m.setRandom();
    benchmark::DoNotOptimize(m.data());
  }
  state.SetBytesProcessed(state.iterations() * N * N * sizeof(Scalar));
}

template <typename Scalar, int N>
static void BM_FixedSetIdentity(benchmark::State& state) {
  Matrix<Scalar, N, N> m;
  for (auto _ : state) {
    m.setIdentity();
    benchmark::DoNotOptimize(m.data());
  }
  state.SetBytesProcessed(state.iterations() * N * N * sizeof(Scalar));
}

// --- Size configurations ---

static void DynamicSizes(::benchmark::Benchmark* b) {
  for (int n : {4, 8, 16, 32, 64, 128, 256, 512, 1024}) b->Arg(n);
}

static void VectorSizes(::benchmark::Benchmark* b) {
  for (int n : {64, 256, 1024, 4096, 16384, 65536}) b->Arg(n);
}

// --- Register: dynamic float ---
BENCHMARK(BM_SetZero<float>)->Apply(DynamicSizes)->Name("SetZero_float");
BENCHMARK(BM_SetRandom<float>)->Apply(DynamicSizes)->Name("SetRandom_float");
BENCHMARK(BM_SetIdentity<float>)->Apply(DynamicSizes)->Name("SetIdentity_float");
BENCHMARK(BM_SetConstant<float>)->Apply(DynamicSizes)->Name("SetConstant_float");
BENCHMARK(BM_LinSpaced<float>)->Apply(VectorSizes)->Name("LinSpaced_float");

// --- Register: dynamic double ---
BENCHMARK(BM_SetZero<double>)->Apply(DynamicSizes)->Name("SetZero_double");
BENCHMARK(BM_SetRandom<double>)->Apply(DynamicSizes)->Name("SetRandom_double");
BENCHMARK(BM_SetIdentity<double>)->Apply(DynamicSizes)->Name("SetIdentity_double");
BENCHMARK(BM_SetConstant<double>)->Apply(DynamicSizes)->Name("SetConstant_double");
BENCHMARK(BM_LinSpaced<double>)->Apply(VectorSizes)->Name("LinSpaced_double");

// --- Register: fixed-size float ---
BENCHMARK(BM_FixedSetZero<float, 2>)->Name("FixedSetZero_float_2x2");
BENCHMARK(BM_FixedSetZero<float, 3>)->Name("FixedSetZero_float_3x3");
BENCHMARK(BM_FixedSetZero<float, 4>)->Name("FixedSetZero_float_4x4");
BENCHMARK(BM_FixedSetZero<float, 8>)->Name("FixedSetZero_float_8x8");
BENCHMARK(BM_FixedSetRandom<float, 4>)->Name("FixedSetRandom_float_4x4");
BENCHMARK(BM_FixedSetIdentity<float, 4>)->Name("FixedSetIdentity_float_4x4");

// --- Register: fixed-size double ---
BENCHMARK(BM_FixedSetZero<double, 2>)->Name("FixedSetZero_double_2x2");
BENCHMARK(BM_FixedSetZero<double, 3>)->Name("FixedSetZero_double_3x3");
BENCHMARK(BM_FixedSetZero<double, 4>)->Name("FixedSetZero_double_4x4");
BENCHMARK(BM_FixedSetZero<double, 8>)->Name("FixedSetZero_double_8x8");
BENCHMARK(BM_FixedSetRandom<double, 4>)->Name("FixedSetRandom_double_4x4");
BENCHMARK(BM_FixedSetIdentity<double, 4>)->Name("FixedSetIdentity_double_4x4");
