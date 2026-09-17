// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/SparseCore>

template <bool Optimized, typename Scalar>
EIGEN_DONT_INLINE void scatterUpdate(Scalar* dense, const int* indices,
                                     const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& values, const Scalar& scale) {
  Eigen::Index i = 0;
  if (Optimized)
    i = Eigen::internal::sparse_scatter_sub_packets<false>(dense, indices, values.data(), values.size(), scale);
  for (; i < values.size(); ++i) dense[indices[i]] -= values[i] * scale;
}

template <typename Scalar, bool Optimized>
void BM_SparseScatter(benchmark::State& state) {
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  const Eigen::Index size = state.range(0), stride = state.range(1);
  Vector dense = Vector::Ones(size * stride + 1);
  Vector values = Vector::Constant(size, Scalar(0.125));
  Eigen::VectorXi indices(size);
  for (Eigen::Index i = 0; i < size; ++i) indices[i] = int(i * stride);
  const Scalar scale(0.5);
  const Vector expected = [&]() {
    Vector result = dense;
    scatterUpdate<false>(result.data(), indices.data(), values, scale);
    return result;
  }();
  scatterUpdate<Optimized>(dense.data(), indices.data(), values, scale);
  if (dense != expected) {
    state.SkipWithError("scatter update differs from scalar reference");
    return;
  }
  for (auto _ : state) {
    scatterUpdate<Optimized>(dense.data(), indices.data(), values, scale);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * size);
}

BENCHMARK_TEMPLATE(BM_SparseScatter, float, false)->ArgsProduct({{3, 16, 31, 32, 33, 64, 256}, {1, 7}});
BENCHMARK_TEMPLATE(BM_SparseScatter, float, true)->ArgsProduct({{3, 16, 31, 32, 33, 64, 256}, {1, 7}});
BENCHMARK_TEMPLATE(BM_SparseScatter, double, false)->ArgsProduct({{3, 16, 31, 32, 33, 64, 256}, {1, 7}});
BENCHMARK_TEMPLATE(BM_SparseScatter, double, true)->ArgsProduct({{3, 16, 31, 32, 33, 64, 256}, {1, 7}});
BENCHMARK_TEMPLATE(BM_SparseScatter, std::complex<double>, false)->ArgsProduct({{3, 16, 64, 256}, {1, 7}});
BENCHMARK_TEMPLATE(BM_SparseScatter, std::complex<double>, true)->ArgsProduct({{3, 16, 64, 256}, {1, 7}});
