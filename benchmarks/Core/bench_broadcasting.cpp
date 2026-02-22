// Benchmarks for colwise/rowwise reductions and broadcasting operations.
//
// Tests vectorwise reductions (sum, mean, norm, minCoeff, maxCoeff) and
// broadcasting arithmetic (rowwise += vec, colwise -= vec, rowwise *= vec).

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

// --- Colwise reductions (reduce each column to a scalar) ---

template <typename Scalar>
static void BM_ColwiseSum(benchmark::State& state) {
  const Index rows = state.range(0);
  const Index cols = state.range(1);
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  Mat m = Mat::Random(rows, cols);
  Matrix<Scalar, 1, Dynamic> result(cols);
  for (auto _ : state) {
    result = m.colwise().sum();
    benchmark::DoNotOptimize(result.data());
  }
  state.SetBytesProcessed(state.iterations() * rows * cols * sizeof(Scalar));
}

template <typename Scalar>
static void BM_ColwiseMean(benchmark::State& state) {
  const Index rows = state.range(0);
  const Index cols = state.range(1);
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  Mat m = Mat::Random(rows, cols);
  Matrix<Scalar, 1, Dynamic> result(cols);
  for (auto _ : state) {
    result = m.colwise().mean();
    benchmark::DoNotOptimize(result.data());
  }
  state.SetBytesProcessed(state.iterations() * rows * cols * sizeof(Scalar));
}

template <typename Scalar>
static void BM_ColwiseNorm(benchmark::State& state) {
  const Index rows = state.range(0);
  const Index cols = state.range(1);
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  Mat m = Mat::Random(rows, cols);
  Matrix<Scalar, 1, Dynamic> result(cols);
  for (auto _ : state) {
    result = m.colwise().norm();
    benchmark::DoNotOptimize(result.data());
  }
  state.SetBytesProcessed(state.iterations() * rows * cols * sizeof(Scalar));
}

template <typename Scalar>
static void BM_ColwiseMinCoeff(benchmark::State& state) {
  const Index rows = state.range(0);
  const Index cols = state.range(1);
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  Mat m = Mat::Random(rows, cols);
  Matrix<Scalar, 1, Dynamic> result(cols);
  for (auto _ : state) {
    result = m.colwise().minCoeff();
    benchmark::DoNotOptimize(result.data());
  }
  state.SetBytesProcessed(state.iterations() * rows * cols * sizeof(Scalar));
}

template <typename Scalar>
static void BM_ColwiseMaxCoeff(benchmark::State& state) {
  const Index rows = state.range(0);
  const Index cols = state.range(1);
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  Mat m = Mat::Random(rows, cols);
  Matrix<Scalar, 1, Dynamic> result(cols);
  for (auto _ : state) {
    result = m.colwise().maxCoeff();
    benchmark::DoNotOptimize(result.data());
  }
  state.SetBytesProcessed(state.iterations() * rows * cols * sizeof(Scalar));
}

// --- Rowwise reductions (reduce each row to a scalar) ---

template <typename Scalar>
static void BM_RowwiseSum(benchmark::State& state) {
  const Index rows = state.range(0);
  const Index cols = state.range(1);
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  Mat m = Mat::Random(rows, cols);
  Matrix<Scalar, Dynamic, 1> result(rows);
  for (auto _ : state) {
    result = m.rowwise().sum();
    benchmark::DoNotOptimize(result.data());
  }
  state.SetBytesProcessed(state.iterations() * rows * cols * sizeof(Scalar));
}

template <typename Scalar>
static void BM_RowwiseNorm(benchmark::State& state) {
  const Index rows = state.range(0);
  const Index cols = state.range(1);
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  Mat m = Mat::Random(rows, cols);
  Matrix<Scalar, Dynamic, 1> result(rows);
  for (auto _ : state) {
    result = m.rowwise().norm();
    benchmark::DoNotOptimize(result.data());
  }
  state.SetBytesProcessed(state.iterations() * rows * cols * sizeof(Scalar));
}

// --- Broadcasting operations ---

template <typename Scalar>
static void BM_RowwiseBroadcastAdd(benchmark::State& state) {
  const Index rows = state.range(0);
  const Index cols = state.range(1);
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, 1, Dynamic>;
  Mat m = Mat::Random(rows, cols);
  Vec v = Vec::Random(cols);
  for (auto _ : state) {
    m.noalias() = m.rowwise() + v;
    benchmark::DoNotOptimize(m.data());
  }
  state.SetBytesProcessed(state.iterations() * rows * cols * sizeof(Scalar) * 2);
}

template <typename Scalar>
static void BM_ColwiseBroadcastAdd(benchmark::State& state) {
  const Index rows = state.range(0);
  const Index cols = state.range(1);
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  Mat m = Mat::Random(rows, cols);
  Vec v = Vec::Random(rows);
  for (auto _ : state) {
    m.noalias() = m.colwise() + v;
    benchmark::DoNotOptimize(m.data());
  }
  state.SetBytesProcessed(state.iterations() * rows * cols * sizeof(Scalar) * 2);
}

template <typename Scalar>
static void BM_RowwiseBroadcastMul(benchmark::State& state) {
  const Index rows = state.range(0);
  const Index cols = state.range(1);
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  Mat m = Mat::Random(rows, cols);
  Array<Scalar, 1, Dynamic> v = Array<Scalar, 1, Dynamic>::Random(cols);
  for (auto _ : state) {
    m.array().rowwise() *= v;
    benchmark::DoNotOptimize(m.data());
  }
  state.SetBytesProcessed(state.iterations() * rows * cols * sizeof(Scalar) * 2);
}

// --- Size configurations ---

static void BroadcastSizes(::benchmark::Benchmark* b) {
  // Square matrices
  for (int n : {64, 128, 256, 512, 1024}) b->Args({n, n});
  // Tall-thin (many rows, few cols)
  b->Args({10000, 32});
  // Short-wide (few rows, many cols)
  b->Args({32, 10000});
}

// --- Register: float ---
BENCHMARK(BM_ColwiseSum<float>)->Apply(BroadcastSizes)->Name("ColwiseSum_float");
BENCHMARK(BM_ColwiseMean<float>)->Apply(BroadcastSizes)->Name("ColwiseMean_float");
BENCHMARK(BM_ColwiseNorm<float>)->Apply(BroadcastSizes)->Name("ColwiseNorm_float");
BENCHMARK(BM_ColwiseMinCoeff<float>)->Apply(BroadcastSizes)->Name("ColwiseMinCoeff_float");
BENCHMARK(BM_ColwiseMaxCoeff<float>)->Apply(BroadcastSizes)->Name("ColwiseMaxCoeff_float");
BENCHMARK(BM_RowwiseSum<float>)->Apply(BroadcastSizes)->Name("RowwiseSum_float");
BENCHMARK(BM_RowwiseNorm<float>)->Apply(BroadcastSizes)->Name("RowwiseNorm_float");
BENCHMARK(BM_RowwiseBroadcastAdd<float>)->Apply(BroadcastSizes)->Name("RowwiseBroadcastAdd_float");
BENCHMARK(BM_ColwiseBroadcastAdd<float>)->Apply(BroadcastSizes)->Name("ColwiseBroadcastAdd_float");
BENCHMARK(BM_RowwiseBroadcastMul<float>)->Apply(BroadcastSizes)->Name("RowwiseBroadcastMul_float");

// --- Register: double ---
BENCHMARK(BM_ColwiseSum<double>)->Apply(BroadcastSizes)->Name("ColwiseSum_double");
BENCHMARK(BM_ColwiseMean<double>)->Apply(BroadcastSizes)->Name("ColwiseMean_double");
BENCHMARK(BM_ColwiseNorm<double>)->Apply(BroadcastSizes)->Name("ColwiseNorm_double");
BENCHMARK(BM_ColwiseMinCoeff<double>)->Apply(BroadcastSizes)->Name("ColwiseMinCoeff_double");
BENCHMARK(BM_ColwiseMaxCoeff<double>)->Apply(BroadcastSizes)->Name("ColwiseMaxCoeff_double");
BENCHMARK(BM_RowwiseSum<double>)->Apply(BroadcastSizes)->Name("RowwiseSum_double");
BENCHMARK(BM_RowwiseNorm<double>)->Apply(BroadcastSizes)->Name("RowwiseNorm_double");
BENCHMARK(BM_RowwiseBroadcastAdd<double>)->Apply(BroadcastSizes)->Name("RowwiseBroadcastAdd_double");
BENCHMARK(BM_ColwiseBroadcastAdd<double>)->Apply(BroadcastSizes)->Name("ColwiseBroadcastAdd_double");
BENCHMARK(BM_RowwiseBroadcastMul<double>)->Apply(BroadcastSizes)->Name("RowwiseBroadcastMul_double");
