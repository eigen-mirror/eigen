#include <benchmark/benchmark.h>
#include <Eigen/Dense>

using namespace Eigen;

// Benchmark to find the optimal BDCSVD -> JacobiSVD crossover threshold.
//
// Sweeps setSwitchSize() over a range of values for various matrix sizes
// to determine the best crossover point now that JacobiSVD has been optimized.

using Mat = Matrix<double, Dynamic, Dynamic>;

template <int Options>
EIGEN_DONT_INLINE void do_compute(BDCSVD<Mat, Options>& svd, const Mat& A) {
  svd.compute(A);
}

// Args: {matrix_size, switch_size}
template <int Options>
static void BM_BDCSVD_Crossover(benchmark::State& state) {
  const Index n = state.range(0);
  const int switchSize = static_cast<int>(state.range(1));
  Mat A = Mat::Random(n, n);
  BDCSVD<Mat, Options> svd(n, n);
  svd.setSwitchSize(switchSize);
  for (auto _ : state) {
    do_compute(svd, A);
    benchmark::DoNotOptimize(svd.singularValues().data());
  }
  state.SetItemsProcessed(state.iterations());
}

// Also benchmark JacobiSVD at the same sizes for direct comparison.
template <int Options>
static void BM_JacobiSVD_Ref(benchmark::State& state) {
  const Index n = state.range(0);
  Mat A = Mat::Random(n, n);
  JacobiSVD<Mat, Options> svd(n, n);
  for (auto _ : state) {
    svd.compute(A);
    benchmark::DoNotOptimize(svd.singularValues().data());
  }
  state.SetItemsProcessed(state.iterations());
}

// clang-format off

// Matrix sizes that exercise the crossover region and beyond.
// Switch sizes from 3 (minimum) to 64.
#define CROSSOVER_ARGS \
    ->Args({16, 3})->Args({16, 4})->Args({16, 6})->Args({16, 8})->Args({16, 12})->Args({16, 16}) \
    ->Args({32, 3})->Args({32, 4})->Args({32, 6})->Args({32, 8})->Args({32, 12})->Args({32, 16})->Args({32, 24})->Args({32, 32}) \
    ->Args({48, 3})->Args({48, 6})->Args({48, 8})->Args({48, 12})->Args({48, 16})->Args({48, 24})->Args({48, 32})->Args({48, 48}) \
    ->Args({64, 3})->Args({64, 6})->Args({64, 8})->Args({64, 12})->Args({64, 16})->Args({64, 24})->Args({64, 32})->Args({64, 48})->Args({64, 64}) \
    ->Args({96, 3})->Args({96, 8})->Args({96, 12})->Args({96, 16})->Args({96, 24})->Args({96, 32})->Args({96, 48})->Args({96, 64}) \
    ->Args({128, 3})->Args({128, 8})->Args({128, 12})->Args({128, 16})->Args({128, 24})->Args({128, 32})->Args({128, 48})->Args({128, 64}) \
    ->Args({256, 3})->Args({256, 8})->Args({256, 16})->Args({256, 24})->Args({256, 32})->Args({256, 48})->Args({256, 64}) \
    ->Args({512, 8})->Args({512, 16})->Args({512, 24})->Args({512, 32})->Args({512, 48})->Args({512, 64})

#define JACOBI_REF_SIZES \
    ->Args({16})->Args({32})->Args({48})->Args({64})->Args({96})->Args({128})

// With vectors (typical use case)
BENCHMARK(BM_BDCSVD_Crossover<ComputeThinU | ComputeThinV>) CROSSOVER_ARGS ->Name("BDCSVD_ThinUV");
BENCHMARK(BM_JacobiSVD_Ref<ComputeThinU | ComputeThinV>) JACOBI_REF_SIZES ->Name("JacobiSVD_ThinUV");

// Values only
BENCHMARK(BM_BDCSVD_Crossover<0>) CROSSOVER_ARGS ->Name("BDCSVD_ValuesOnly");
BENCHMARK(BM_JacobiSVD_Ref<0>) JACOBI_REF_SIZES ->Name("JacobiSVD_ValuesOnly");

#undef CROSSOVER_ARGS
#undef JACOBI_REF_SIZES
// clang-format on
