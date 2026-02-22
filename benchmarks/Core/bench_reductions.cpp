// Benchmarks for full reductions: sum, prod, minCoeff, maxCoeff, mean,
// norm, squaredNorm, lpNorm<1>, lpNorm<Infinity>.
//
// These are memory-bandwidth-bound for large vectors, so we report
// bytes processed rather than FLOPS.

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

// --- Size configurations ---

static void VectorSizes(::benchmark::Benchmark* b) {
  for (int n : {64, 256, 1024, 4096, 16384, 65536, 262144, 1048576}) b->Arg(n);
}

static void MatrixSizes(::benchmark::Benchmark* b) {
  for (int n : {8, 32, 64, 128, 256, 512, 1024}) b->Arg(n);
}

// --- Register: float ---
BENCHMARK(BM_VectorSum<float>)->Apply(VectorSizes)->Name("VectorSum_float");
BENCHMARK(BM_VectorProd<float>)->Apply(VectorSizes)->Name("VectorProd_float");
BENCHMARK(BM_VectorMinCoeff<float>)->Apply(VectorSizes)->Name("VectorMinCoeff_float");
BENCHMARK(BM_VectorMaxCoeff<float>)->Apply(VectorSizes)->Name("VectorMaxCoeff_float");
BENCHMARK(BM_VectorMean<float>)->Apply(VectorSizes)->Name("VectorMean_float");
BENCHMARK(BM_VectorSquaredNorm<float>)->Apply(VectorSizes)->Name("VectorSquaredNorm_float");
BENCHMARK(BM_VectorNorm<float>)->Apply(VectorSizes)->Name("VectorNorm_float");
BENCHMARK(BM_VectorLpNorm1<float>)->Apply(VectorSizes)->Name("VectorLpNorm1_float");
BENCHMARK(BM_VectorLpNormInf<float>)->Apply(VectorSizes)->Name("VectorLpNormInf_float");
BENCHMARK(BM_MatrixSum<float>)->Apply(MatrixSizes)->Name("MatrixSum_float");
BENCHMARK(BM_MatrixNorm<float>)->Apply(MatrixSizes)->Name("MatrixNorm_float");

// --- Register: double ---
BENCHMARK(BM_VectorSum<double>)->Apply(VectorSizes)->Name("VectorSum_double");
BENCHMARK(BM_VectorProd<double>)->Apply(VectorSizes)->Name("VectorProd_double");
BENCHMARK(BM_VectorMinCoeff<double>)->Apply(VectorSizes)->Name("VectorMinCoeff_double");
BENCHMARK(BM_VectorMaxCoeff<double>)->Apply(VectorSizes)->Name("VectorMaxCoeff_double");
BENCHMARK(BM_VectorMean<double>)->Apply(VectorSizes)->Name("VectorMean_double");
BENCHMARK(BM_VectorSquaredNorm<double>)->Apply(VectorSizes)->Name("VectorSquaredNorm_double");
BENCHMARK(BM_VectorNorm<double>)->Apply(VectorSizes)->Name("VectorNorm_double");
BENCHMARK(BM_VectorLpNorm1<double>)->Apply(VectorSizes)->Name("VectorLpNorm1_double");
BENCHMARK(BM_VectorLpNormInf<double>)->Apply(VectorSizes)->Name("VectorLpNormInf_double");
BENCHMARK(BM_MatrixSum<double>)->Apply(MatrixSizes)->Name("MatrixSum_double");
BENCHMARK(BM_MatrixNorm<double>)->Apply(MatrixSizes)->Name("MatrixNorm_double");
