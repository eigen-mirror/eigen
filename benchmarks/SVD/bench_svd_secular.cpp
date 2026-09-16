// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/SVD>
#include <cstdlib>

#ifndef SCALAR
#define SCALAR double
#endif

template <bool Bidiagonal, int Options = Eigen::ComputeThinU | Eigen::ComputeThinV>
static void BM_SVDSecular(benchmark::State& state) {
  using Scalar = SCALAR;
  using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  const Eigen::Index rows = state.range(0), cols = state.range(1);
  std::srand(5489);
  Mat matrix = Mat::Random(rows, cols);
  Vec diagonal, superdiagonal;
  if (Bidiagonal) {
    diagonal = matrix.diagonal();
    superdiagonal = matrix.template diagonal<1>();
    matrix.setZero();
    matrix.diagonal() = diagonal;
    matrix.template diagonal<1>() = superdiagonal;
  }
  Eigen::BDCSVD<Mat, Eigen::ComputeThinU | Eigen::ComputeThinV> reference(matrix);
  const Scalar bound = Scalar(64 * rows) * Eigen::NumTraits<Scalar>::epsilon();
  const Mat reconstructed =
      reference.matrixU() * reference.singularValues().asDiagonal() * reference.matrixV().transpose();
  if (reference.info() != Eigen::Success || !((matrix - reconstructed).norm() <= bound * matrix.norm()) ||
      !((reference.matrixU().transpose() * reference.matrixU() - Mat::Identity(cols, cols)).norm() <= bound) ||
      !((reference.matrixV().transpose() * reference.matrixV() - Mat::Identity(cols, cols)).norm() <= bound)) {
    state.SkipWithError("SVD failed reconstruction or orthogonality check");
    return;
  }
  Eigen::BDCSVD<Mat, Options> svd(rows, cols);
  auto compute = [&] {
    if (Bidiagonal)
      svd.compute(diagonal, superdiagonal);
    else
      svd.compute(matrix);
  };
  compute();
  if (svd.info() != Eigen::Success ||
      !((svd.singularValues() - reference.singularValues()).norm() <= bound * matrix.norm())) {
    state.SkipWithError("SVD failed singular-value check");
    return;
  }
  if (Options != 0) {
    const Mat result = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
    if (!((matrix - result).norm() <= bound * matrix.norm()) ||
        !((svd.matrixU().transpose() * svd.matrixU() - Mat::Identity(cols, cols)).norm() <= bound) ||
        !((svd.matrixV().transpose() * svd.matrixV() - Mat::Identity(cols, cols)).norm() <= bound)) {
      state.SkipWithError("SVD failed reconstruction or orthogonality check");
      return;
    }
  }
  for (auto _ : state) {
    compute();
    benchmark::DoNotOptimize(svd.singularValues().data());
    if (Options != 0) {
      benchmark::DoNotOptimize(svd.matrixU().data());
      benchmark::DoNotOptimize(svd.matrixV().data());
    }
    benchmark::ClobberMemory();
  }
}

#define SVD_SECULAR_SIZES \
  ->Args({16, 16})        \
      ->Args({32, 32})    \
      ->Args({64, 64})    \
      ->Args({65, 65})    \
      ->Args({128, 128})  \
      ->Args({129, 129})  \
      ->Args({256, 256})  \
      ->Args({257, 257})  \
      ->Args({512, 512})

BENCHMARK_TEMPLATE(BM_SVDSecular, false)
SVD_SECULAR_SIZES->Args({1023, 256})
    ->Args({1024, 256})
    ->Args({1028, 256})
    ->Args({4096, 64})
    ->Args({4096, 1024})
    ->Args({10000, 8})
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_SVDSecular, true) SVD_SECULAR_SIZES->UseRealTime();

BENCHMARK_TEMPLATE(BM_SVDSecular, false, 0)->Args({128, 128})->Args({512, 512})->UseRealTime();
BENCHMARK_TEMPLATE(BM_SVDSecular, true, 0)->Args({128, 128})->Args({512, 512})->UseRealTime();

#undef SVD_SECULAR_SIZES
