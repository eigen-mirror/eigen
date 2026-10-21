// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// EIGEN_SUFFIXES;1;2

// Core use inside a SYCL kernel does not require EIGEN_USE_SYCL.
#if defined(EIGEN_TEST_PART_2)
#include <Eigen/Core>
#endif
#include <CL/sycl.hpp>
#include <Eigen/Core>
#include <contrib/Eigen/SpecialFunctions>

template <typename Scalar>
bool sycl_first_vector_addition() {
  using Vector = Eigen::Matrix<Scalar, 4, 1>;
  Vector a, b, result, expected;
  a << 1, 2, 3, 4;
  b << 5, 6, 7, 8;
  expected << 6, 8, 10, 12;
  cl::sycl::queue queue;
  {
    cl::sycl::buffer<Scalar, 1> input_a(a.data(), cl::sycl::range<1>(4));
    cl::sycl::buffer<Scalar, 1> input_b(b.data(), cl::sycl::range<1>(4));
    cl::sycl::buffer<Scalar, 1> output(result.data(), cl::sycl::range<1>(4));
    queue.submit([&](cl::sycl::handler& handler) {
      auto x = input_a.template get_access<cl::sycl::access::mode::read>(handler);
      auto y = input_b.template get_access<cl::sycl::access::mode::read>(handler);
      auto z = output.template get_access<cl::sycl::access::mode::discard_write>(handler);
      handler.single_task([=]() {
        Eigen::Map<const Vector> a_map(&x[0]), b_map(&y[0]);
        Eigen::Map<Vector> result_map(&z[0]);
        result_map = a_map + b_map;
      });
    });
    queue.wait_and_throw();
  }
  return result == expected;
}

template <typename Scalar>
bool sycl_scalar_math() {
  using Vector = Eigen::Matrix<Scalar, 4, 1>;
  Vector input, result_sin, result_lgamma;
  input << 1, 2, 3, 4;
  cl::sycl::queue queue;
  {
    cl::sycl::buffer<Scalar, 1> input_buffer(input.data(), cl::sycl::range<1>(4));
    cl::sycl::buffer<Scalar, 1> output_sin(result_sin.data(), cl::sycl::range<1>(4));
    cl::sycl::buffer<Scalar, 1> output_lgamma(result_lgamma.data(), cl::sycl::range<1>(4));
    queue.submit([&](cl::sycl::handler& handler) {
      auto x = input_buffer.template get_access<cl::sycl::access::mode::read>(handler);
      auto y = output_sin.template get_access<cl::sycl::access::mode::discard_write>(handler);
      auto z = output_lgamma.template get_access<cl::sycl::access::mode::discard_write>(handler);
      handler.single_task([=]() {
        Eigen::Map<const Vector> input_map(&x[0]);
        Eigen::Map<Vector> sin_map(&y[0]);
        sin_map = input_map.array().sin().matrix();
        for (Eigen::Index i = 0; i < 4; ++i) z[i] = Eigen::numext::lgamma(input_map[i]);
      });
    });
    queue.wait_and_throw();
  }
  const Scalar tolerance = Scalar(32) * Eigen::NumTraits<Scalar>::epsilon();
  for (Eigen::Index i = 0; i < 4; ++i) {
    const Scalar expected_sin = std::sin(input[i]);
    const Scalar expected_lgamma = std::lgamma(input[i]);
    if (!(std::abs(result_sin[i] - expected_sin) <= tolerance * (Scalar(1) + std::abs(expected_sin)))) return false;
    if (!(std::abs(result_lgamma[i] - expected_lgamma) <= tolerance * (Scalar(1) + std::abs(expected_lgamma))))
      return false;
  }
  return true;
}

template <typename Scalar>
bool sycl_low_precision_fma() {
  float input[] = {1.5f, 2.f, 0.5f}, result = 0;
  cl::sycl::queue queue;
  {
    cl::sycl::buffer<float, 1> inputs(input, cl::sycl::range<1>(3));
    cl::sycl::buffer<float, 1> output(&result, cl::sycl::range<1>(1));
    queue.submit([&](cl::sycl::handler& handler) {
      auto x = inputs.template get_access<cl::sycl::access::mode::read>(handler);
      auto y = output.template get_access<cl::sycl::access::mode::discard_write>(handler);
      handler.single_task([=]() { y[0] = float(Eigen::numext::fma(Scalar(x[0]), Scalar(x[1]), Scalar(x[2]))); });
    });
    queue.wait_and_throw();
  }
  const float expected =
      float(Scalar(std::fma(float(Scalar(input[0])), float(Scalar(input[1])), float(Scalar(input[2])))));
  return result == expected;
}

// The shared test driver explicitly enables CUDA when __CUDACC__ is defined;
// this test must leave backend selection to Eigen, as a user translation unit does.
int main() {
  if (!sycl_first_vector_addition<float>() || !sycl_scalar_math<float>() || !sycl_low_precision_fma<Eigen::half>() ||
      !sycl_low_precision_fma<Eigen::bfloat16>())
    return 1;
  cl::sycl::queue queue;
  if (queue.get_device().has(cl::sycl::aspect::fp64) &&
      (!sycl_first_vector_addition<double>() || !sycl_scalar_math<double>()))
    return 1;
  return 0;
}
