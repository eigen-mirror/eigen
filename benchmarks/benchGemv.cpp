// Benchmark for dense general matrix-vector multiplication (GEMV).
//
// Tests performance of y += op(A) * x for various matrix sizes, aspect ratios,
// scalar types, and operation variants (transpose, conjugate, adjoint).
//
// The Eigen GEMV kernel (Eigen/src/Core/products/GeneralMatrixVector.h) has
// two main specializations:
//   - ColMajor kernel: used for y += A * x with column-major A.
//     Processes vertical panels, vectorizes along rows.
//   - RowMajor kernel: used for y += A^T * x with column-major A.
//     Processes groups of rows, vectorizes the dot product along columns.
//
// For complex scalars, conjugation flags (ConjugateLhs, ConjugateRhs) select
// additional code paths within each kernel via conj_helper.
//
// Operation mapping (for column-major stored A):
//   Gemv       y += A * x           -> ColMajor kernel, no conjugation
//   GemvTrans  y += A^T * x         -> RowMajor kernel, no conjugation
//   GemvConj   y += conj(A) * x     -> ColMajor kernel, ConjugateLhs=true
//   GemvAdj    y += A^H * x         -> RowMajor kernel, ConjugateLhs=true

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

// ---------- Benchmark helpers ----------

// GEMV flop count: 2*m*n for real, 8*m*n for complex.
template <typename Scalar>
double gemvFlops(Index m, Index n) {
  return (NumTraits<Scalar>::IsComplex ? 8.0 : 2.0) * m * n;
}

// ---------- y += A * x  (ColMajor GEMV kernel, no conjugation) ----------

template <typename Scalar>
static void BM_Gemv(benchmark::State& state) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  const Index m = state.range(0);
  const Index n = state.range(1);
  Mat A = Mat::Random(m, n);
  Vec x = Vec::Random(n);
  Vec y = Vec::Random(m);
  for (auto _ : state) {
    y.noalias() += A * x;
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] = benchmark::Counter(gemvFlops<Scalar>(m, n), benchmark::Counter::kIsIterationInvariantRate,
                                                benchmark::Counter::kIs1000);
}

// ---------- y += A^T * x  (RowMajor GEMV kernel, no conjugation) ----------

template <typename Scalar>
static void BM_GemvTrans(benchmark::State& state) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  const Index m = state.range(0);
  const Index n = state.range(1);
  Mat A = Mat::Random(m, n);
  Vec x = Vec::Random(m);
  Vec y = Vec::Random(n);
  for (auto _ : state) {
    y.noalias() += A.transpose() * x;
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] = benchmark::Counter(gemvFlops<Scalar>(m, n), benchmark::Counter::kIsIterationInvariantRate,
                                                benchmark::Counter::kIs1000);
}

// ---------- y += conj(A) * x  (ColMajor kernel, ConjugateLhs=true) ----------

template <typename Scalar>
static void BM_GemvConj(benchmark::State& state) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  const Index m = state.range(0);
  const Index n = state.range(1);
  Mat A = Mat::Random(m, n);
  Vec x = Vec::Random(n);
  Vec y = Vec::Random(m);
  for (auto _ : state) {
    y.noalias() += A.conjugate() * x;
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] = benchmark::Counter(gemvFlops<Scalar>(m, n), benchmark::Counter::kIsIterationInvariantRate,
                                                benchmark::Counter::kIs1000);
}

// ---------- y += A^H * x  (RowMajor kernel, ConjugateLhs=true) ----------

template <typename Scalar>
static void BM_GemvAdj(benchmark::State& state) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  const Index m = state.range(0);
  const Index n = state.range(1);
  Mat A = Mat::Random(m, n);
  Vec x = Vec::Random(m);
  Vec y = Vec::Random(n);
  for (auto _ : state) {
    y.noalias() += A.adjoint() * x;
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOPS"] = benchmark::Counter(gemvFlops<Scalar>(m, n), benchmark::Counter::kIsIterationInvariantRate,
                                                benchmark::Counter::kIs1000);
}

// ---------- Size configurations ----------
// All sizes refer to the stored matrix A (m rows, n cols).

static void GemvSizes(::benchmark::Benchmark* b) {
  // Square matrices: exercises balanced kernel behavior.
  for (int size : {8, 16, 32, 64, 128, 256, 512, 1024, 4096}) {
    b->Args({size, size});
  }
  // Tall-thin (m >> n): in ColMajor kernel, the inner vectorized loop over rows
  // is long while the outer column loop is short. In RowMajor kernel (transpose),
  // there are many rows to process but short dot products.
  for (int n : {1, 4, 16, 64}) {
    for (int m : {256, 1024, 4096}) {
      if (m != n) b->Args({m, n});
    }
  }
  // Short-wide (m << n): in ColMajor kernel, the outer column loop is long but
  // the inner vectorized loop over rows is short. In RowMajor kernel (transpose),
  // there are few rows but long dot products.
  for (int m : {1, 4, 16, 64}) {
    for (int n : {256, 1024, 4096}) {
      if (m != n) b->Args({m, n});
    }
  }
}

// ---------- Register benchmarks ----------

// Real types: Gemv and GemvTrans exercise the two kernel specializations.
// Conjugation is a no-op for real scalars.

BENCHMARK(BM_Gemv<float>)->Apply(GemvSizes)->Name("Gemv_float");
BENCHMARK(BM_Gemv<double>)->Apply(GemvSizes)->Name("Gemv_double");
BENCHMARK(BM_GemvTrans<float>)->Apply(GemvSizes)->Name("GemvTrans_float");
BENCHMARK(BM_GemvTrans<double>)->Apply(GemvSizes)->Name("GemvTrans_double");

// Complex types: all four variants exercise distinct kernel code paths.

BENCHMARK(BM_Gemv<std::complex<float>>)->Apply(GemvSizes)->Name("Gemv_cfloat");
BENCHMARK(BM_Gemv<std::complex<double>>)->Apply(GemvSizes)->Name("Gemv_cdouble");
BENCHMARK(BM_GemvTrans<std::complex<float>>)->Apply(GemvSizes)->Name("GemvTrans_cfloat");
BENCHMARK(BM_GemvTrans<std::complex<double>>)->Apply(GemvSizes)->Name("GemvTrans_cdouble");
BENCHMARK(BM_GemvConj<std::complex<float>>)->Apply(GemvSizes)->Name("GemvConj_cfloat");
BENCHMARK(BM_GemvConj<std::complex<double>>)->Apply(GemvSizes)->Name("GemvConj_cdouble");
BENCHMARK(BM_GemvAdj<std::complex<float>>)->Apply(GemvSizes)->Name("GemvAdj_cfloat");
BENCHMARK(BM_GemvAdj<std::complex<double>>)->Apply(GemvSizes)->Name("GemvAdj_cdouble");
