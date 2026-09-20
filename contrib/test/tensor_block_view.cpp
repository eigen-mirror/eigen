// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_USE_THREADS
#include "main.h"
#include <contrib/Eigen/Tensor>

template <typename Evaluator, typename TensorType, std::enable_if_t<Evaluator::PacketAccess, int> = 0>
static void check_view_packets(const Evaluator& evaluator, const TensorType& expected) {
  using Scalar = typename TensorType::Scalar;
  constexpr int PacketSize = PacketType<Scalar, DefaultDevice>::size;
  Scalar values[PacketSize];
  for (Index i = 0; i + PacketSize <= expected.size(); ++i) {
    internal::pstoreu(values, evaluator.template packet<Unaligned>(i));
    for (int k = 0; k < PacketSize; ++k) VERIFY_IS_EQUAL(values[k], expected.data()[i + k]);
  }
}

template <typename Evaluator, typename TensorType, std::enable_if_t<!Evaluator::PacketAccess, int> = 0>
static void check_view_packets(const Evaluator&, const TensorType&) {}

template <typename Scalar, int Layout>
static void test_strided_view() {
  using TensorType = Tensor<Scalar, 3, Layout>;
  using Dimensions = DSizes<Index, 3>;
  using Evaluator = TensorEvaluator<const TensorType, DefaultDevice>;
  DefaultDevice device;
  TensorType input(23, 19, 11);
  for (Index i = 0; i < input.size(); ++i) input.data()[i] = Scalar(i % 97);
  Evaluator evaluator(input, device);
  internal::TensorBlockScratchAllocator<DefaultDevice> scratch(device);
  const Index full_inner = Layout == ColMajor ? 23 : 11;
  for (Index inner : {Index(1), Index(3), Index(5), Index(9), full_inner}) {
    Dimensions offsets(2, 3, 1);
    if (inner == full_inner) offsets[Layout == ColMajor ? 0 : 2] = 0;
    const Index offset =
        Layout == ColMajor ? input.dimensions().IndexOfColMajor(offsets) : input.dimensions().IndexOfRowMajor(offsets);
    const Dimensions sizes = Layout == ColMajor ? Dimensions(inner, 7, 3) : Dimensions(3, 7, inner);
    internal::TensorBlockDescriptor<3> desc(offset, sizes);
    auto block = evaluator.block(desc, scratch);
    VERIFY_IS_EQUAL(block.kind(), internal::TensorBlockKind::kView);
    VERIFY(block.data() == nullptr);

    TensorType expected(sizes);
    for (Index k = 0; k < sizes[2]; ++k)
      for (Index j = 0; j < sizes[1]; ++j)
        for (Index i = 0; i < sizes[0]; ++i) expected(i, j, k) = input(i + offsets[0], j + offsets[1], k + offsets[2]);

    using BlockExpr = typename Evaluator::TensorBlock::XprType;
    TensorEvaluator<const BlockExpr, DefaultDevice> block_eval(block.expr(), device);
    for (Index i = 0; i < expected.size(); ++i) VERIFY_IS_EQUAL(block_eval.coeff(i), expected.data()[i]);
    check_view_packets(block_eval, expected);

    TensorType output(sizes);
    using Assignment = internal::TensorBlockAssignment<Scalar, 3, BlockExpr>;
    Assignment::Run(Assignment::target(sizes, internal::strides<Layout>(sizes), output.data()), block.expr());
    for (Index i = 0; i < output.size(); ++i) VERIFY_IS_EQUAL(output.data()[i], expected.data()[i]);

    const Scalar saved = input.data()[offset];
    input.data()[offset] = Scalar(101);
    VERIFY_IS_EQUAL(block_eval.coeff(0), Scalar(101));
    input.data()[offset] = saved;
    block.cleanup();
    scratch.reset();
  }

  const Dimensions sizes = Layout == ColMajor ? Dimensions(23, 4, 1) : Dimensions(1, 4, 11);
  internal::TensorBlockDescriptor<3> desc(0, sizes);
  auto block = evaluator.block(desc, scratch);
  VERIFY_IS_EQUAL(block.kind(), internal::TensorBlockKind::kView);
  VERIFY(block.data() == input.data());
  block.cleanup();
}

struct BlockViewCounter {
  mutable Index calls = 0;
  float operator()(float value) const { return value + float(calls++); }
};

template <bool Vectorized>
struct BlockViewRvalueFunctor {
  using Packet = typename internal::packet_traits<float>::type;
  float operator()(float&& value) const { return value + 1.0f; }
  Packet packetOp(Packet&& value) const { return internal::padd(value, internal::pset1<Packet>(1.0f)); }
};

template <bool Vectorized>
struct BlockViewOverloadedFunctor : BlockViewRvalueFunctor<Vectorized> {
  using Base = BlockViewRvalueFunctor<Vectorized>;
  using Packet = typename Base::Packet;
  using Base::operator();
  using Base::packetOp;
  float operator()(const float& value) const { return value + 2.0f; }
  Packet packetOp(const Packet& value) const { return internal::padd(value, internal::pset1<Packet>(2.0f)); }
};

namespace Eigen {
namespace internal {
template <bool Vectorized>
struct functor_traits<BlockViewRvalueFunctor<Vectorized>> {
  static constexpr int Cost = NumTraits<float>::AddCost;
  static constexpr bool PacketAccess = Vectorized && packet_traits<float>::Vectorizable;
};
template <bool Vectorized>
struct functor_traits<BlockViewOverloadedFunctor<Vectorized>> : functor_traits<BlockViewRvalueFunctor<Vectorized>> {};
}  // namespace internal
}  // namespace Eigen

template <int Layout, bool Vectorized, typename Device>
static void test_view_functor_forwarding(const Device& device) {
  using TensorType = Tensor<float, 2, Layout>;
  const Index rows = 129, cols = 193;
  TensorType input(cols, rows), bias(rows, cols), output(rows, cols);
  for (Index j = 0; j < cols; ++j) {
    for (Index i = 0; i < rows; ++i) {
      input(j, i) = float((i * 3 + j) % 17);
      bias(i, j) = float((i + j * 7) % 13);
    }
  }
  const array<int, 2> transpose{{1, 0}};
  const auto check = [&](const auto& functor) {
    auto expression = input.shuffle(transpose) + bias.unaryExpr(functor);
    // MSVC 19.29 treats decltype(output) as a reference inside this generic lambda.
    using Assign = TensorAssignOp<TensorType, const decltype(expression)>;
    static_assert(internal::IsTileable<Device, const Assign>::value == internal::TiledEvaluation::On,
                  "Functor forwarding must reach block evaluation");
    static_assert(internal::IsVectorizable<Device, const Assign>::value ==
                      (Vectorized && internal::packet_traits<float>::Vectorizable),
                  "Exercise both scalar and packet functor forwarding");
    output.device(device) = expression;
    for (Index j = 0; j < cols; ++j)
      for (Index i = 0; i < rows; ++i) VERIFY_IS_EQUAL(output(i, j), input(j, i) + bias(i, j) + 1.0f);
  };
  check(BlockViewRvalueFunctor<Vectorized>());
  check(BlockViewOverloadedFunctor<Vectorized>());
}

template <int Layout>
static void test_view_functor_state(bool dense_source) {
  Tensor<float, 2, Layout> input(dense_source ? 7 : 23, dense_source ? 9 : 19), output(23, 19);
  const DSizes<Index, 2> sizes(7, 9);
  input.setZero();
  DefaultDevice device;
  TensorEvaluator<const decltype(input), DefaultDevice> evaluator(input, device);
  internal::TensorBlockScratchAllocator<DefaultDevice> scratch(device);
  internal::TensorBlockDescriptor<2> desc(0, sizes);
  auto block = evaluator.block(desc, scratch);
  auto expression = block.expr().unaryExpr(BlockViewCounter());
  using Assignment = internal::TensorBlockAssignment<float, 2, decltype(expression)>;
  Assignment::Run(Assignment::target(sizes, internal::strides<Layout>(output.dimensions()), output.data()), expression);
  for (Index j = 0; j < sizes[1]; ++j)
    for (Index i = 0; i < sizes[0]; ++i)
      VERIFY_IS_EQUAL(output(i, j), float(Layout == ColMajor ? i + sizes[0] * j : j + sizes[1] * i));
  block.cleanup();
}

template <int Layout, typename Device>
static void test_view_compositions(const Device& device) {
  const Index rows = 129, cols = 193;
  Tensor<float, 2, Layout> input(cols, rows), bias(rows, cols), output(rows, cols);
  Tensor<bool, 2, Layout> condition(rows, cols);
  for (Index j = 0; j < cols; ++j) {
    for (Index i = 0; i < rows; ++i) {
      input(j, i) = float((i * 3 + j) % 17);
      bias(i, j) = float((i + j * 7) % 13);
      condition(i, j) = (i + j) % 2 == 0;
    }
  }
  const array<int, 2> transpose{{1, 0}};
  auto expression = input.shuffle(transpose) + bias;
  using Assign = TensorAssignOp<decltype(output), const decltype(expression)>;
  static_assert(internal::IsTileable<Device, const Assign>::value == internal::TiledEvaluation::On,
                "Shuffle and bias must reach block evaluation");
  output.device(device) = expression;
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i) VERIFY_IS_EQUAL(output(i, j), input(j, i) + bias(i, j));

  output.device(device) = condition.select(expression.square() + 2.0f, bias);
  for (Index j = 0; j < cols; ++j) {
    for (Index i = 0; i < rows; ++i) {
      const float sum = input(j, i) + bias(i, j);
      VERIFY_IS_EQUAL(output(i, j), condition(i, j) ? sum * sum + 2.0f : bias(i, j));
    }
  }

  Tensor<double, 2, Layout> converted(rows, cols);
  converted.device(device) = expression.template cast<double>();
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i) VERIFY_IS_EQUAL(converted(i, j), double(input(j, i) + bias(i, j)));

  output.device(device) = expression + bias.constant(3.0f);
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i) VERIFY_IS_EQUAL(output(i, j), input(j, i) + bias(i, j) + 3.0f);

  const auto add_three = [](float a, float b, float c) { return a + b + c; };
  const TensorCwiseTernaryOp<decltype(add_three), const decltype(expression), const decltype(bias),
                             const decltype(bias)>
      ternary(expression, bias, bias, add_three);
  output.device(device) = ternary;
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i) VERIFY_IS_EQUAL(output(i, j), input(j, i) + 3.0f * bias(i, j));

  Tensor<float, 2, Layout == ColMajor ? RowMajor : ColMajor> swapped(cols, rows);
  swapped.device(device) = expression.swap_layout();
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i) VERIFY_IS_EQUAL(swapped(j, i), input(j, i) + bias(i, j));

  bias.device(device) = expression;
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i) VERIFY_IS_EQUAL(bias(i, j), input(j, i) + float((i + j * 7) % 13));
}

static void test_scalar_and_empty_views() {
  DefaultDevice device;
  internal::TensorBlockScratchAllocator<DefaultDevice> scratch(device);
  float value = 3.0f;
  using ScalarBlock = internal::TensorMaterializedBlock<float, 0, ColMajor>;
  internal::TensorBlockDescriptor<0> scalar_desc(0, DSizes<Index, 0>());
  auto block = ScalarBlock::materialize(&value, DSizes<Index, 0>(), scalar_desc, scratch);
  Tensor<float, 0> scalar = block.expr();
  VERIFY_IS_EQUAL(scalar(), value);
  block.cleanup();

  Tensor<float, 2> input(0, 3), bias(3, 0), output(3, 0);
  output = input.shuffle(array<int, 2>{{1, 0}}) + bias;
  VERIFY_IS_EQUAL(output.size(), 0);
}

EIGEN_DECLARE_TEST(tensor_block_view) {
  CALL_SUBTEST((test_strided_view<float, ColMajor>()));
  CALL_SUBTEST((test_strided_view<float, RowMajor>()));
  CALL_SUBTEST((test_strided_view<double, ColMajor>()));
  CALL_SUBTEST((test_strided_view<double, RowMajor>()));
  CALL_SUBTEST((test_strided_view<int, ColMajor>()));
  CALL_SUBTEST((test_strided_view<int, RowMajor>()));
  CALL_SUBTEST((test_strided_view<std::complex<float>, ColMajor>()));
  CALL_SUBTEST((test_strided_view<std::complex<float>, RowMajor>()));
  for (bool dense_source : {false, true}) {
    CALL_SUBTEST((test_view_functor_state<ColMajor>(dense_source)));
    CALL_SUBTEST((test_view_functor_state<RowMajor>(dense_source)));
  }
  DefaultDevice device;
  CALL_SUBTEST((test_view_functor_forwarding<ColMajor, false>(device)));
  CALL_SUBTEST((test_view_functor_forwarding<RowMajor, false>(device)));
  CALL_SUBTEST((test_view_functor_forwarding<ColMajor, true>(device)));
  CALL_SUBTEST((test_view_functor_forwarding<RowMajor, true>(device)));
  CALL_SUBTEST((test_view_compositions<ColMajor>(device)));
  CALL_SUBTEST((test_view_compositions<RowMajor>(device)));
  ThreadPool pool(4);
  for (int threads : {1, 4}) {
    ThreadPoolDevice threaded_device(&pool, threads);
    CALL_SUBTEST((test_view_functor_forwarding<ColMajor, false>(threaded_device)));
    CALL_SUBTEST((test_view_functor_forwarding<RowMajor, false>(threaded_device)));
    CALL_SUBTEST((test_view_functor_forwarding<ColMajor, true>(threaded_device)));
    CALL_SUBTEST((test_view_functor_forwarding<RowMajor, true>(threaded_device)));
    CALL_SUBTEST((test_view_compositions<ColMajor>(threaded_device)));
    CALL_SUBTEST((test_view_compositions<RowMajor>(threaded_device)));
  }
  CALL_SUBTEST(test_scalar_and_empty_views());
}
