#include <benchmark/benchmark.h>
#include <Eigen/Dense>

using namespace Eigen;

// ---------- TRSV: triangular solve with single RHS vector ----------

template <typename Scalar, unsigned int Mode>
static void BM_TRSV(benchmark::State& state) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  const Index n = state.range(0);
  Mat A = Mat::Random(n, n);
  // Make diagonally dominant to ensure well-conditioned triangular part.
  A.diagonal().array() += Scalar(n);
  Vec x = Vec::Random(n);
  Vec b = x;
  for (auto _ : state) {
    x = b;
    A.template triangularView<Mode>().solveInPlace(x);
    benchmark::DoNotOptimize(x.data());
  }
  state.SetItemsProcessed(state.iterations() * n * n);
}

// ---------- TRSM: triangular solve with multiple RHS (OnTheLeft) ----------

template <typename Scalar, unsigned int Mode>
static void BM_TRSM_Left(benchmark::State& state) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  const Index n = state.range(0);
  const Index nrhs = state.range(1);
  Mat A = Mat::Random(n, n);
  A.diagonal().array() += Scalar(n);
  Mat X = Mat::Random(n, nrhs);
  Mat B = X;
  for (auto _ : state) {
    X = B;
    A.template triangularView<Mode>().solveInPlace(X);
    benchmark::DoNotOptimize(X.data());
  }
  state.SetItemsProcessed(state.iterations() * n * n * nrhs);
}

// ---------- TRSM: triangular solve with multiple RHS (OnTheRight) ----------

template <typename Scalar, unsigned int Mode>
static void BM_TRSM_Right(benchmark::State& state) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  const Index n = state.range(0);
  const Index nrhs = state.range(1);
  Mat A = Mat::Random(n, n);
  A.diagonal().array() += Scalar(n);
  Mat X = Mat::Random(nrhs, n);
  Mat B = X;
  for (auto _ : state) {
    X = B;
    A.template triangularView<Mode>().template solveInPlace<OnTheRight>(X);
    benchmark::DoNotOptimize(X.data());
  }
  state.SetItemsProcessed(state.iterations() * n * n * nrhs);
}

// ---------- Size configurations ----------

static void TrsvSizes(::benchmark::Benchmark* b) {
  for (int n : {32, 64, 128, 256, 512, 1024}) {
    b->Args({n});
  }
}

static void TrsmSizes(::benchmark::Benchmark* b) {
  for (int n : {32, 64, 128, 256, 512, 1024}) {
    for (int nrhs : {1, 4, 16, 64, 256}) {
      b->Args({n, nrhs});
    }
  }
}

// ---------- TRSV benchmarks ----------

BENCHMARK(BM_TRSV<float, Lower>)->Apply(TrsvSizes)->Name("TRSV_float_Lower");
BENCHMARK(BM_TRSV<float, Upper>)->Apply(TrsvSizes)->Name("TRSV_float_Upper");
BENCHMARK(BM_TRSV<double, Lower>)->Apply(TrsvSizes)->Name("TRSV_double_Lower");
BENCHMARK(BM_TRSV<double, Upper>)->Apply(TrsvSizes)->Name("TRSV_double_Upper");

// ---------- TRSM Left benchmarks ----------

BENCHMARK(BM_TRSM_Left<float, Lower>)->Apply(TrsmSizes)->Name("TRSM_Left_float_Lower");
BENCHMARK(BM_TRSM_Left<float, Upper>)->Apply(TrsmSizes)->Name("TRSM_Left_float_Upper");
BENCHMARK(BM_TRSM_Left<double, Lower>)->Apply(TrsmSizes)->Name("TRSM_Left_double_Lower");
BENCHMARK(BM_TRSM_Left<double, Upper>)->Apply(TrsmSizes)->Name("TRSM_Left_double_Upper");

// ---------- TRSM Right benchmarks ----------

BENCHMARK(BM_TRSM_Right<float, Lower>)->Apply(TrsmSizes)->Name("TRSM_Right_float_Lower");
BENCHMARK(BM_TRSM_Right<float, Upper>)->Apply(TrsmSizes)->Name("TRSM_Right_float_Upper");
BENCHMARK(BM_TRSM_Right<double, Lower>)->Apply(TrsmSizes)->Name("TRSM_Right_double_Lower");
BENCHMARK(BM_TRSM_Right<double, Upper>)->Apply(TrsmSizes)->Name("TRSM_Right_double_Upper");
