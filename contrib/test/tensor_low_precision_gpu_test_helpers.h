// SPDX-FileCopyrightText: 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
// SPDX-FileCopyrightText: 2021 Rohit Santhanam <rohit.santhanam@amd.com>
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TENSOR_LOW_PRECISION_GPU_TEST_HELPERS_H
#define EIGEN_TENSOR_LOW_PRECISION_GPU_TEST_HELPERS_H

#include <type_traits>

#include "tensor_gpu_test_helpers.h"

namespace tensor_test {

template <typename Scalar, typename Input>
void round_gpu_test_input(Input& input, const Eigen::GpuDevice& device) {
  // The bfloat16 unary/elementwise checks use inputs rounded to the working precision.
  EIGEN_IF_CONSTEXPR ((std::is_same<Scalar, Eigen::bfloat16>::value)) {
    input.device(device) = input.template cast<Scalar>().template cast<float>();
  }
}

template <typename Scalar>
void test_gpu_numext() {
  GpuTensorTestFixture fixture;
  auto input = fixture.tensor<float, 1>(101);
  auto reference = fixture.tensor<bool, 1>(101);
  auto result = fixture.tensor<bool, 1>(101);
  input.device(fixture.device()) = input.random() - input.constant(0.5f);
  reference.device(fixture.device()) = input.unaryExpr(Eigen::internal::scalar_isnan_op<float>());
  result.device(fixture.device()) = input.template cast<Scalar>().unaryExpr(Eigen::internal::scalar_isnan_op<Scalar>());
  const auto expected = fixture.toHost(reference);
  const auto actual = fixture.toHost(result);
  for (Index i = 0; i < input.size(); ++i) VERIFY_IS_EQUAL(expected(i), actual(i));
}

template <typename Scalar>
void test_gpu_conversion() {
  GpuTensorTestFixture fixture;
  auto input = fixture.tensor<float, 1>(101);
  auto converted = fixture.tensor<Scalar, 1>(101);
  auto result = fixture.tensor<float, 1>(101);
  input.device(fixture.device()) = input.random();
  converted.device(fixture.device()) = input.template cast<Scalar>();
  result.device(fixture.device()) = converted.template cast<float>();
  const auto expected = fixture.toHost(input);
  const auto actual = fixture.toHost(result);
  for (Index i = 0; i < input.size(); ++i) VERIFY_IS_APPROX(Scalar(expected(i)), Scalar(actual(i)));
}

template <typename Scalar>
void test_gpu_unary() {
  GpuTensorTestFixture fixture;
  auto input = fixture.tensor<float, 1>(101);
  auto reference = fixture.tensor<float, 1>(101);
  auto result = fixture.tensor<float, 1>(101);
  input.device(fixture.device()) = input.random() - input.constant(0.5f);
  round_gpu_test_input<Scalar>(input, fixture.device());
  reference.device(fixture.device()) = input.abs();
  result.device(fixture.device()) = input.template cast<Scalar>().abs().template cast<float>();
  const auto expected = fixture.toHost(reference);
  const auto actual = fixture.toHost(result);
  for (Index i = 0; i < input.size(); ++i) VERIFY_IS_APPROX(expected(i), actual(i));
}

template <typename Scalar>
void test_gpu_elementwise() {
  GpuTensorTestFixture fixture;
  auto left = fixture.tensor<float, 1>(101);
  auto right = fixture.tensor<float, 1>(101);
  auto reference = fixture.tensor<float, 1>(101);
  auto result = fixture.tensor<float, 1>(101);
  left.device(fixture.device()) = left.random();
  round_gpu_test_input<Scalar>(left, fixture.device());
  right.device(fixture.device()) = right.random();
  round_gpu_test_input<Scalar>(right, fixture.device());
  reference.device(fixture.device()) = (left + right) * left;
  result.device(fixture.device()) =
      ((left.template cast<Scalar>() + right.template cast<Scalar>()) * left.template cast<Scalar>())
          .template cast<float>();
  const auto expected = fixture.toHost(reference);
  const auto actual = fixture.toHost(result);
  for (Index i = 0; i < left.size(); ++i) VERIFY_IS_APPROX(Scalar(expected(i)), Scalar(actual(i)));
}

template <typename Scalar, typename Input, typename Operation>
void check_gpu_transcendental(GpuTensorTestFixture& fixture, const Input& input, const Operation& operation,
                              bool logarithm = false) {
  auto reference = fixture.tensor<Scalar, 1>(input.size());
  auto result = fixture.tensor<Scalar, 1>(input.size());
  reference.device(fixture.device()) = operation(input).template cast<Scalar>();
  result.device(fixture.device()) = input.template cast<Scalar>();
  result.device(fixture.device()) = operation(result);
  const auto host_input = fixture.toHost(input);
  const auto expected = fixture.toHost(reference);
  const auto actual = fixture.toHost(result);
  for (Index i = 0; i < input.size(); ++i) {
    if (logarithm && Eigen::numext::abs(host_input(i) - 1.f) < 0.05f) {
      // Preserve the existing absolute floor for log near 1.
      VERIFY_IS_APPROX(expected(i) + Scalar(0.1f), actual(i) + Scalar(0.1f));
    } else {
      VERIFY_IS_APPROX(expected(i), actual(i));
    }
  }
}

template <typename Scalar>
void test_gpu_transcendental() {
  GpuTensorTestFixture fixture;
  auto input1 = fixture.tensor<float, 1>(101);
  auto input2 = fixture.tensor<float, 1>(101);
  auto input3 = fixture.tensor<float, 1>(101);
  input1.device(fixture.device()) = input1.random() - input1.constant(0.5f);
  round_gpu_test_input<Scalar>(input1, fixture.device());
  input2.device(fixture.device()) = input2.random() + input1.constant(0.5f);
  round_gpu_test_input<Scalar>(input2, fixture.device());
  input3.device(fixture.device()) = input3.random();
  round_gpu_test_input<Scalar>(input3, fixture.device());
  check_gpu_transcendental<Scalar>(fixture, input1, [](const auto& input) { return input.exp(); });
  check_gpu_transcendental<Scalar>(
      fixture, input2, [](const auto& input) { return input.log(); }, true);
  check_gpu_transcendental<Scalar>(fixture, input3, [](const auto& input) { return input.log1p(); });
  check_gpu_transcendental<Scalar>(fixture, input3, [](const auto& input) { return input.expm1(); });
}

template <typename Scalar>
void test_gpu_contractions(float absolute_eps_factor) {
  GpuTensorTestFixture fixture;
  auto left = fixture.tensor<float, 2>(23, 23);
  auto right = fixture.tensor<float, 2>(23, 23);
  auto reference = fixture.tensor<Scalar, 2>(23, 23);
  auto result = fixture.tensor<Scalar, 2>(23, 23);
  left.device(fixture.device()) = left.random() - left.constant(0.5f);
  right.device(fixture.device()) = right.random() - right.constant(0.5f);
  using DimPair = typename Tensor<float, 2>::DimensionPair;
  const Eigen::array<DimPair, 1> dims{DimPair(1, 0)};
  reference.device(fixture.device()) = left.contract(right, dims).template cast<Scalar>();
  result.device(fixture.device()) = left.template cast<Scalar>().contract(right.template cast<Scalar>(), dims);
  const auto expected = fixture.toHost(reference);
  const auto actual = fixture.toHost(result);
  // Absolute slack for the 23-term dot product, then the existing relative comparison.
  const Scalar absolute_slack = Scalar(absolute_eps_factor) * Eigen::NumTraits<Scalar>::epsilon();
  for (Index i = 0; i < result.size(); ++i) {
    const Scalar error = Eigen::numext::abs(expected.data()[i] - actual.data()[i]);
    VERIFY((Eigen::numext::isfinite)(error));
    if (error > absolute_slack) VERIFY_IS_APPROX(expected.data()[i], actual.data()[i]);
  }
}

template <typename Scalar>
void test_gpu_reductions(Index rows, Index cols, int axis) {
  GpuTensorTestFixture fixture;
  auto input = fixture.tensor<float, 2>(rows, cols);
  const Index size = axis == 1 ? rows : cols;
  auto reference = fixture.tensor<Scalar, 1>(size);
  auto result = fixture.tensor<Scalar, 1>(size);
  input.device(fixture.device()) = input.random() * 2.0f;
  const Eigen::array<int, 1> dims{axis};
  reference.device(fixture.device()) = input.sum(dims).template cast<Scalar>();
  result.device(fixture.device()) = input.template cast<Scalar>().sum(dims);
  const auto expected = fixture.toHost(reference);
  const auto actual = fixture.toHost(result);
  for (Index i = 0; i < size; ++i) VERIFY_IS_APPROX(expected(i), actual(i));
}

template <typename Scalar>
void test_gpu_reductions() {
  test_gpu_reductions<Scalar>(13, 13, 0);
  test_gpu_reductions<Scalar>(13, 13, 1);
  test_gpu_reductions<Scalar>(35, 36, 0);
  test_gpu_reductions<Scalar>(35, 36, 1);
  test_gpu_reductions<Scalar>(36, 35, 0);
  test_gpu_reductions<Scalar>(36, 35, 1);
}

template <typename Scalar>
void test_gpu_full_reductions() {
  GpuTensorTestFixture fixture;
  auto input = fixture.tensor<float, 2>(13, 13);
  auto reference = fixture.tensor<Scalar, 0>();
  auto result = fixture.tensor<Scalar, 0>();
  input.device(fixture.device()) = input.random();
  reference.device(fixture.device()) = input.sum().template cast<Scalar>();
  result.device(fixture.device()) = input.template cast<Scalar>().sum();
  VERIFY_IS_APPROX(fixture.toHost(reference)(), fixture.toHost(result)());
  reference.device(fixture.device()) = input.maximum().template cast<Scalar>();
  result.device(fixture.device()) = input.template cast<Scalar>().maximum();
  VERIFY_IS_APPROX(fixture.toHost(reference)(), fixture.toHost(result)());
}

template <typename Scalar>
void test_gpu_forced_evals() {
  GpuTensorTestFixture fixture;
  auto input = fixture.tensor<float, 1>(101);
  auto reference = fixture.tensor<float, 1>(101);
  auto aligned_result = fixture.tensor<float, 1>(101);
  auto unaligned_result = fixture.tensor<float, 1, Eigen::Unaligned>(101);
  const Eigen::array<int, 1> no_broadcast{1};
  input.device(fixture.device()) = input.random() - input.constant(0.5f);
  round_gpu_test_input<Scalar>(input, fixture.device());
  reference.device(fixture.device()) = input.abs();
  aligned_result.device(fixture.device()) = input.template cast<Scalar>().abs().eval().template cast<float>();
  unaligned_result.device(fixture.device()) =
      input.template cast<Scalar>().abs().broadcast(no_broadcast).eval().template cast<float>();
  const auto expected = fixture.toHost(reference);
  const auto aligned = fixture.toHost(aligned_result);
  const auto unaligned = fixture.toHost(unaligned_result);
  for (Index i = 0; i < input.size(); ++i) {
    VERIFY_IS_APPROX(expected(i), aligned(i));
    VERIFY_IS_APPROX(expected(i), unaligned(i));
  }
}

}  // namespace tensor_test

#endif  // EIGEN_TENSOR_LOW_PRECISION_GPU_TEST_HELPERS_H
