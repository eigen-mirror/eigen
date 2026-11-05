// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

template <typename Scalar, int Mode, int VSize>
static void BM_TransformVec(benchmark::State& state) {
  using Trans = Transform<Scalar, 3, Mode>;
  using Vec = Matrix<Scalar, VSize, 1>;
  Trans t;
  t.setIdentity();
  Vec v;
  v.setRandom();
  for (auto _ : state) {
    v = t * v;
    benchmark::DoNotOptimize(v.data());
  }
}

template <typename Scalar, int Mode>
static void BM_TransformTransform(benchmark::State& state) {
  using Trans = Transform<Scalar, 3, Mode>;
  Trans t1, t2;
  t1.setIdentity();
  t2.setIdentity();
  for (auto _ : state) {
    t2 = Trans(t1 * t2);
    benchmark::DoNotOptimize(t2.data());
  }
}

BENCHMARK(BM_TransformVec<float, Isometry, 3>);
BENCHMARK(BM_TransformVec<float, Isometry, 4>);
BENCHMARK(BM_TransformVec<float, Projective, 4>);
BENCHMARK(BM_TransformVec<double, Isometry, 3>);
BENCHMARK(BM_TransformVec<double, Isometry, 4>);
BENCHMARK(BM_TransformVec<double, Projective, 4>);

BENCHMARK(BM_TransformTransform<float, Isometry>);
BENCHMARK(BM_TransformTransform<float, Projective>);
BENCHMARK(BM_TransformTransform<double, Isometry>);
BENCHMARK(BM_TransformTransform<double, Projective>);
