// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <contrib/Eigen/Tensor>

using Eigen::Index;
using Eigen::internal::TiledEvaluation;

template <typename Scalar, int Layout, bool Shuffle>
static void BM_BlockView(benchmark::State& state) {
  const Index n = state.range(0);
  const bool tiled = state.range(1) != 0;
  using Tensor = Eigen::Tensor<Scalar, 2, Layout>;
  Tensor input(n, n), bias(n, n), output(n, n);
  for (Index j = 0; j < n; ++j) {
    for (Index i = 0; i < n; ++i) {
      input(i, j) = Scalar((i * 17 + j * 31) % 1024) * Scalar(0.25);
      bias(i, j) = Scalar((i * 7 + j * 11) % 256) * Scalar(0.125);
    }
  }
  auto measure = [&](const auto& expression) {
    using Expression = Eigen::internal::remove_all_t<decltype(expression)>;
    using Assign = Eigen::TensorAssignOp<Tensor, const Expression>;
    using Device = Eigen::DefaultDevice;
    constexpr bool Vectorizable = Eigen::internal::IsVectorizable<Device, const Assign>::value;
    using Tiled = Eigen::internal::TensorExecutor<const Assign, Device, Vectorizable, TiledEvaluation::On>;
    using Untiled = Eigen::internal::TensorExecutor<const Assign, Device, Vectorizable, TiledEvaluation::Off>;
    const Assign assign(output, expression);
    if (tiled) {
      Tiled::run(assign);
    } else {
      Untiled::run(assign);
    }
    for (Index j = 0; j < n; ++j) {
      for (Index i = 0; i < n; ++i) {
        const Scalar expected = (Shuffle ? input(j, i) : input(i, j)) + bias(i, j);
        if (output(i, j) != expected) {
          state.SkipWithError("incorrect block result");
          return;
        }
      }
    }
    if (tiled) {
      for (auto unused : state) {
        Tiled::run(assign);
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
      }
    } else {
      for (auto unused : state) {
        Untiled::run(assign);
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
      }
    }
  };
  EIGEN_IF_CONSTEXPR (Shuffle) {
    const Eigen::array<int, 2> transpose{{1, 0}};
    measure(input.shuffle(transpose) + bias);
  } else {
    // Identity broadcasting selects contiguous blocks for the control case.
    const Eigen::array<Index, 2> broadcast{{1, 1}};
    measure(input.broadcast(broadcast) + bias);
  }
  state.SetItemsProcessed(state.iterations() * n * n);
}

#define REGISTER_BLOCK_VIEW(Scalar, Layout, Shuffle)        \
  BENCHMARK_TEMPLATE(BM_BlockView, Scalar, Layout, Shuffle) \
      ->ArgsProduct({{64, 96, 112, 128, 192, 256, 257, 384, 512, 1024}, {0, 1}})

REGISTER_BLOCK_VIEW(float, Eigen::ColMajor, true);
REGISTER_BLOCK_VIEW(float, Eigen::RowMajor, true);
REGISTER_BLOCK_VIEW(double, Eigen::ColMajor, true);
REGISTER_BLOCK_VIEW(double, Eigen::RowMajor, true);
REGISTER_BLOCK_VIEW(float, Eigen::ColMajor, false);
REGISTER_BLOCK_VIEW(float, Eigen::RowMajor, false);

#undef REGISTER_BLOCK_VIEW
