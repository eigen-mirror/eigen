// Benchmarks for Map and Ref with various strides.
//
// Compares contiguous Map vs strided Map vs owned matrix for basic
// operations (GEMV and vector sum).

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

// Sum a contiguous Map<VectorX>.
template <typename Scalar>
static void BM_MapContiguousSum(benchmark::State& state) {
  const Index n = state.range(0);
  std::vector<Scalar> buf(n);
  Map<Matrix<Scalar, Dynamic, 1>> v(buf.data(), n);
  v.setRandom();
  for (auto _ : state) {
    Scalar s = v.sum();
    benchmark::DoNotOptimize(s);
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar));
}

// Sum a strided Map (InnerStride).
template <typename Scalar>
static void BM_MapStridedSum(benchmark::State& state) {
  const Index n = state.range(0);
  const Index stride = 3;
  std::vector<Scalar> buf(n * stride);
  Map<Matrix<Scalar, Dynamic, 1>, 0, InnerStride<>> v(buf.data(), n, InnerStride<>(stride));
  v.setRandom();
  for (auto _ : state) {
    Scalar s = v.sum();
    benchmark::DoNotOptimize(s);
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar));
}

// Sum an owned VectorX (baseline).
template <typename Scalar>
static void BM_OwnedSum(benchmark::State& state) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, 1> v = Matrix<Scalar, Dynamic, 1>::Random(n);
  for (auto _ : state) {
    Scalar s = v.sum();
    benchmark::DoNotOptimize(s);
  }
  state.SetBytesProcessed(state.iterations() * n * sizeof(Scalar));
}

// GEMV through contiguous Map<MatrixX>.
template <typename Scalar>
static void BM_MapGemv(benchmark::State& state) {
  const Index n = state.range(0);
  std::vector<Scalar> buf(n * n);
  Map<Matrix<Scalar, Dynamic, Dynamic>> A(buf.data(), n, n);
  A.setRandom();
  Matrix<Scalar, Dynamic, 1> x = Matrix<Scalar, Dynamic, 1>::Random(n);
  Matrix<Scalar, Dynamic, 1> y = Matrix<Scalar, Dynamic, 1>::Random(n);
  for (auto _ : state) {
    y.noalias() += A * x;
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] =
      benchmark::Counter(2.0 * n * n, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// GEMV with owned matrix (baseline).
template <typename Scalar>
static void BM_OwnedGemv(benchmark::State& state) {
  const Index n = state.range(0);
  Matrix<Scalar, Dynamic, Dynamic> A = Matrix<Scalar, Dynamic, Dynamic>::Random(n, n);
  Matrix<Scalar, Dynamic, 1> x = Matrix<Scalar, Dynamic, 1>::Random(n);
  Matrix<Scalar, Dynamic, 1> y = Matrix<Scalar, Dynamic, 1>::Random(n);
  for (auto _ : state) {
    y.noalias() += A * x;
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] =
      benchmark::Counter(2.0 * n * n, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

static void SumSizes(::benchmark::Benchmark* b) {
  for (int n : {256, 1024, 4096, 16384, 65536, 262144, 1048576}) b->Arg(n);
}

static void GemvSizes(::benchmark::Benchmark* b) {
  for (int n : {32, 128, 512, 1024}) b->Arg(n);
}

BENCHMARK(BM_MapContiguousSum<float>)->Apply(SumSizes)->Name("MapContiguousSum_float");
BENCHMARK(BM_MapStridedSum<float>)->Apply(SumSizes)->Name("MapStridedSum_float");
BENCHMARK(BM_OwnedSum<float>)->Apply(SumSizes)->Name("OwnedSum_float");
BENCHMARK(BM_MapContiguousSum<double>)->Apply(SumSizes)->Name("MapContiguousSum_double");
BENCHMARK(BM_MapStridedSum<double>)->Apply(SumSizes)->Name("MapStridedSum_double");
BENCHMARK(BM_OwnedSum<double>)->Apply(SumSizes)->Name("OwnedSum_double");
BENCHMARK(BM_MapGemv<float>)->Apply(GemvSizes)->Name("MapGemv_float");
BENCHMARK(BM_OwnedGemv<float>)->Apply(GemvSizes)->Name("OwnedGemv_float");
BENCHMARK(BM_MapGemv<double>)->Apply(GemvSizes)->Name("MapGemv_double");
BENCHMARK(BM_OwnedGemv<double>)->Apply(GemvSizes)->Name("OwnedGemv_double");
