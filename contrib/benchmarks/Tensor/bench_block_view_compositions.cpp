// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <contrib/Eigen/Tensor>

using Eigen::Index;

enum BlockViewComposition {
  ShuffleAdd,
  ShuffleAffine,
  ShuffleSelect,
  SwapAffine,
  CropBroadcast,
  BroadcastAffine,
  BroadcastBelow
};

static const char* composition_name(int composition) {
  static const char* names[] = {"shuffle_add",    "shuffle_affine",   "shuffle_select", "swap_affine",
                                "crop_broadcast", "broadcast_affine", "broadcast_below"};
  return names[composition];
}

template <typename Scalar, int Layout, typename Measure>
static void block_view_composition(Index rows, Index cols, int composition, const Measure& measure) {
  using Tensor = Eigen::Tensor<Scalar, 2, Layout>;
  const bool shuffled = composition <= SwapAffine;
  Tensor input(shuffled ? cols : rows, shuffled ? rows : cols), b(rows, cols), c(rows, cols), d(rows, cols);
  Tensor tile(rows / 2, cols / 2), tile_b(rows / 2, cols / 2), tile_c(rows / 2, cols / 2);
  Eigen::Tensor<bool, 2, Layout> mask(rows, cols);
  const auto fill = [](auto& tensor, Index multiplier) {
    for (Index j = 0; j < tensor.dimension(1); ++j)
      for (Index i = 0; i < tensor.dimension(0); ++i) tensor(i, j) = Scalar((i * multiplier + j * 7) % 23) - Scalar(11);
  };
  fill(input, 3);
  fill(b, 5);
  fill(c, 7);
  fill(d, 11);
  fill(tile, 3);
  fill(tile_b, 5);
  fill(tile_c, 7);
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i) mask(i, j) = (i + j) % 3 != 0;

  const Eigen::array<int, 2> transpose{{1, 0}};
  const Eigen::array<Index, 2> repeats{{2, 2}}, offsets{{rows / 4, cols / 4}}, sizes{{rows / 2, cols / 2}};
  Tensor output(rows, cols);
  switch (composition) {
    case ShuffleAdd:
      measure(output, input.shuffle(transpose) + b, [&](Index i, Index j) { return input(j, i) + b(i, j); });
      break;
    case ShuffleAffine:
      measure(output, ((input.shuffle(transpose) + b) * c + d).cwiseMax(Scalar(0)), [&](Index i, Index j) {
        return Eigen::numext::maxi(Scalar(0), (input(j, i) + b(i, j)) * c(i, j) + d(i, j));
      });
      break;
    case ShuffleSelect:
      measure(output, mask.select(input.shuffle(transpose) + b * c, d),
              [&](Index i, Index j) { return mask(i, j) ? input(j, i) + b(i, j) * c(i, j) : d(i, j); });
      break;
    case SwapAffine: {
      Eigen::Tensor<Scalar, 2, Layout == Eigen::ColMajor ? Eigen::RowMajor : Eigen::ColMajor> swapped(cols, rows);
      measure(swapped, (input.shuffle(transpose) + b * c).swap_layout(),
              [&](Index j, Index i) { return input(j, i) + b(i, j) * c(i, j); });
      break;
    }
    case CropBroadcast:
      measure(output, (input + b * c).slice(offsets, sizes).broadcast(repeats), [&](Index i, Index j) {
        const Index row = offsets[0] + i % sizes[0], col = offsets[1] + j % sizes[1];
        return input(row, col) + b(row, col) * c(row, col);
      });
      break;
    case BroadcastAffine:
      measure(output, tile.broadcast(repeats) + b * c,
              [&](Index i, Index j) { return tile(i % sizes[0], j % sizes[1]) + b(i, j) * c(i, j); });
      break;
    case BroadcastBelow:
      measure(output, (tile + tile_b * tile_c).broadcast(repeats), [&](Index i, Index j) {
        const Index row = i % sizes[0], col = j % sizes[1];
        return tile(row, col) + tile_b(row, col) * tile_c(row, col);
      });
      break;
  }
}

template <typename Scalar, int Layout>
static void BM_BlockViewCompositions(benchmark::State& state) {
  const Index rows = state.range(0), cols = state.range(1);
  const int composition = static_cast<int>(state.range(2));
  const bool public_assignment = state.range(3) != 0;
  const auto measure = [&](auto& output, const auto& expression, const auto& expected) {
    using Output = Eigen::internal::remove_all_t<decltype(output)>;
    using Expression = Eigen::internal::remove_all_t<decltype(expression)>;
    using Assign = Eigen::TensorAssignOp<Output, const Expression>;
    using Device = Eigen::DefaultDevice;
    static_assert(Eigen::internal::IsTileable<Device, const Assign>::value == Eigen::internal::TiledEvaluation::On,
                  "Composite assignments must select block evaluation");
    constexpr bool Vectorizable = Eigen::internal::IsVectorizable<Device, const Assign>::value;
    using Untiled =
        Eigen::internal::TensorExecutor<const Assign, Device, Vectorizable, Eigen::internal::TiledEvaluation::Off>;
    const Assign assign(output, expression);
    const auto evaluate = [&]() {
      if (public_assignment)
        output = expression;
      else
        Untiled::run(assign);
    };
    evaluate();
    for (Index j = 0; j < output.dimension(1); ++j) {
      for (Index i = 0; i < output.dimension(0); ++i) {
        if (output(i, j) != expected(i, j)) {
          state.SkipWithError("incorrect composite result");
          return;
        }
      }
    }
    for (auto unused : state) {
      evaluate();
      benchmark::DoNotOptimize(output.data());
      benchmark::ClobberMemory();
    }
  };
  block_view_composition<Scalar, Layout>(rows, cols, composition, measure);
  state.SetLabel(composition_name(composition));
  state.SetItemsProcessed(state.iterations() * rows * cols);
}

#define COMPOSITION_SHAPE(Rows, Cols) ->ArgsProduct({{Rows}, {Cols}, {0, 1, 2, 3, 4, 5, 6}, {0, 1}})
#define REGISTER_COMPOSITIONS(Scalar, Layout)                  \
  BENCHMARK_TEMPLATE(BM_BlockViewCompositions, Scalar, Layout) \
  COMPOSITION_SHAPE(64, 64) COMPOSITION_SHAPE(256, 256) COMPOSITION_SHAPE(258, 194) COMPOSITION_SHAPE(1024, 1024)

REGISTER_COMPOSITIONS(float, Eigen::ColMajor);
REGISTER_COMPOSITIONS(float, Eigen::RowMajor);
REGISTER_COMPOSITIONS(double, Eigen::ColMajor);
REGISTER_COMPOSITIONS(double, Eigen::RowMajor);

#undef REGISTER_COMPOSITIONS
#undef COMPOSITION_SHAPE
