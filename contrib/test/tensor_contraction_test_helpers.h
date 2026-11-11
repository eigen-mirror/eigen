// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Output kernels and device-independent test bodies shared by tensor_contraction.cpp (DefaultDevice) and
// tensor_thread_pool.cpp (ThreadPoolDevice, synchronous and asynchronous). Each body takes `assign`, a callable
// `assign(dst, expr)` that evaluates `dst = expr` on the device under test.

#ifndef EIGEN_TENSOR_CONTRACTION_TEST_HELPERS_H
#define EIGEN_TENSOR_CONTRACTION_TEST_HELPERS_H

#include <cmath>

#include <contrib/Eigen/Tensor>

namespace tensor_test {

using Eigen::Index;
using Eigen::Tensor;
using DimPair = Tensor<float, 1>::DimensionPair;

// Apply Sqrt to all output elements.
struct SqrtOutputKernel {
  template <typename Index, typename Scalar>
  EIGEN_ALWAYS_INLINE void operator()(
      const Eigen::internal::blas_data_mapper<Scalar, Index, Eigen::ColMajor>& output_mapper,
      const Eigen::TensorContractionParams&, Index, Index, Index num_rows, Index num_cols) const {
    for (int i = 0; i < num_rows; ++i) {
      for (int j = 0; j < num_cols; ++j) {
        output_mapper(i, j) = std::sqrt(output_mapper(i, j));
      }
    }
  }
};

template <typename Scalar>
struct AddBiasOutputKernel {
  Scalar bias;
  explicit AddBiasOutputKernel(Scalar b) : bias(b) {}

  template <typename Index, typename ResScalar>
  EIGEN_ALWAYS_INLINE void operator()(
      const Eigen::internal::blas_data_mapper<ResScalar, Index, Eigen::ColMajor>& output_mapper,
      const Eigen::TensorContractionParams&, Index, Index, Index num_rows, Index num_cols) const {
    for (Index i = 0; i < num_rows; ++i) {
      for (Index j = 0; j < num_cols; ++j) {
        output_mapper(i, j) += bias;
      }
    }
  }
};

template <int DataLayout, typename Assign>
void test_large_contraction_with_output_kernel(const Assign& assign) {
  Tensor<float, 4, DataLayout> t_left(30, 50, 8, 31);
  Tensor<float, 5, DataLayout> t_right(8, 31, 7, 20, 10);
  Tensor<float, 5, DataLayout> t_result(30, 50, 7, 20, 10);

  t_left.setRandom();
  t_right.setRandom();
  // Put trash in mat4 to verify contraction clears output memory.
  t_result.setRandom();

  // Add a little offset so that the results won't be close to zero.
  t_left += t_left.constant(1.0f);
  t_right += t_right.constant(1.0f);

  using MapXf = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, DataLayout>>;
  MapXf m_left(t_left.data(), 1500, 248);
  MapXf m_right(t_right.data(), 248, 1400);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, DataLayout> m_result(1500, 1400);

  // this contraction should be equivalent to a single matrix multiplication
  Eigen::array<DimPair, 2> dims{{DimPair(2, 0), DimPair(3, 1)}};

  // compute results by separate methods
  assign(t_result, t_left.contract(t_right, dims, SqrtOutputKernel()));

  m_result = m_left * m_right;

  for (std::ptrdiff_t i = 0; i < t_result.dimensions().TotalSize(); i++) {
    VERIFY(&t_result.data()[i] != &m_result.data()[i]);
    VERIFY_IS_APPROX(t_result.data()[i], std::sqrt(m_result.data()[i]));
  }
}

// {10, 0} * {0, 20} -> {10, 20} and {64, 0} * {0, 64} -> {64, 64}. The second has max(m, n) >= 48, which previously
// triggered a SIGFPE (division by zero) in computeProductBlockingSizes.
template <int DataLayout, typename Scalar, typename Assign>
void test_zero_dim_contraction_matrices(const Assign& assign) {
  // Pre-fill the destination with non-zero values to verify that the contraction actively zeroes output memory.
  const Scalar non_zero = Scalar(123);
  Eigen::array<DimPair, 1> dims = {{DimPair(1, 0)}};
  const Index shapes[][2] = {{10, 20}, {64, 64}};
  for (const auto& shape : shapes) {
    const Index m = shape[0];
    const Index n = shape[1];
    Tensor<Scalar, 2, DataLayout> t_left(m, 0);
    Tensor<Scalar, 2, DataLayout> t_right(0, n);
    Tensor<Scalar, 2, DataLayout> t_result(m, n);
    t_result.setConstant(non_zero);

    assign(t_result, t_left.contract(t_right, dims));

    VERIFY_IS_EQUAL(t_result.dimension(0), m);
    VERIFY_IS_EQUAL(t_result.dimension(1), n);
    VERIFY_IS_EQUAL(t_result.size(), m * n);
    for (Index i = 0; i < t_result.size(); ++i) {
      VERIFY_IS_EQUAL(t_result.data()[i], Scalar(0));
    }
  }
}

template <int DataLayout, typename Scalar, typename Assign>
void test_zero_dim_contraction_multidims(const Assign& assign) {
  // Test {5, 0, 4} * {0, 3, 4} with contraction dims {1, 0} and {2, 2} -> {5, 3}
  Tensor<Scalar, 3, DataLayout> t_left(5, 0, 4);
  Tensor<Scalar, 3, DataLayout> t_right(0, 3, 4);

  const Scalar non_zero = Scalar(123);
  Tensor<Scalar, 2, DataLayout> t_result(5, 3);
  t_result.setConstant(non_zero);

  Eigen::array<DimPair, 2> dims = {{DimPair(1, 0), DimPair(2, 2)}};
  assign(t_result, t_left.contract(t_right, dims));

  VERIFY_IS_EQUAL(t_result.dimension(0), 5);
  VERIFY_IS_EQUAL(t_result.dimension(1), 3);
  VERIFY_IS_EQUAL(t_result.size(), 15);

  for (Index i = 0; i < t_result.dimension(0); ++i) {
    for (Index j = 0; j < t_result.dimension(1); ++j) {
      VERIFY_IS_EQUAL(t_result(i, j), Scalar(0));
    }
  }

  // 4D contractions: {2, 5, 0, 3} * {0, 4, 3, 2} with contraction dims {2, 0} and {3, 2} -> {2, 5, 4, 2}
  Tensor<Scalar, 4, DataLayout> t_left4(2, 5, 0, 3);
  Tensor<Scalar, 4, DataLayout> t_right4(0, 4, 3, 2);
  Tensor<Scalar, 4, DataLayout> t_result4(2, 5, 4, 2);
  t_result4.setConstant(non_zero);

  Eigen::array<DimPair, 2> dims4 = {{DimPair(2, 0), DimPair(3, 2)}};
  assign(t_result4, t_left4.contract(t_right4, dims4));

  VERIFY_IS_EQUAL(t_result4.dimension(0), 2);
  VERIFY_IS_EQUAL(t_result4.dimension(1), 5);
  VERIFY_IS_EQUAL(t_result4.dimension(2), 4);
  VERIFY_IS_EQUAL(t_result4.dimension(3), 2);
  VERIFY_IS_EQUAL(t_result4.size(), 80);

  for (Index i = 0; i < t_result4.size(); ++i) {
    VERIFY_IS_EQUAL(t_result4.data()[i], Scalar(0));
  }
}

template <int DataLayout, typename Scalar, typename Assign>
void test_zero_dim_contraction_outer_zeros(const Assign& assign) {
  Eigen::array<DimPair, 1> dims = {{DimPair(1, 0)}};

  // Outer zero on LHS: {0, 5} * {5, 10} -> {0, 10}
  {
    Tensor<Scalar, 2, DataLayout> t_left(0, 5);
    Tensor<Scalar, 2, DataLayout> t_right(5, 10);
    t_right.setRandom();
    Tensor<Scalar, 2, DataLayout> t_result(0, 10);
    assign(t_result, t_left.contract(t_right, dims));
    VERIFY_IS_EQUAL(t_result.dimension(0), 0);
    VERIFY_IS_EQUAL(t_result.dimension(1), 10);
    VERIFY_IS_EQUAL(t_result.size(), 0);
  }

  // Outer zero on RHS: {10, 5} * {5, 0} -> {10, 0}
  {
    Tensor<Scalar, 2, DataLayout> t_left(10, 5);
    Tensor<Scalar, 2, DataLayout> t_right(5, 0);
    t_left.setRandom();
    Tensor<Scalar, 2, DataLayout> t_result(10, 0);
    assign(t_result, t_left.contract(t_right, dims));
    VERIFY_IS_EQUAL(t_result.dimension(0), 10);
    VERIFY_IS_EQUAL(t_result.dimension(1), 0);
    VERIFY_IS_EQUAL(t_result.size(), 0);
  }

  // Outer zero on both LHS and RHS: {0, 5} * {5, 0} -> {0, 0}
  {
    Tensor<Scalar, 2, DataLayout> t_left(0, 5);
    Tensor<Scalar, 2, DataLayout> t_right(5, 0);
    Tensor<Scalar, 2, DataLayout> t_result(0, 0);
    assign(t_result, t_left.contract(t_right, dims));
    VERIFY_IS_EQUAL(t_result.dimension(0), 0);
    VERIFY_IS_EQUAL(t_result.dimension(1), 0);
    VERIFY_IS_EQUAL(t_result.size(), 0);
  }

  // Both outer and inner zero: {0, 0} * {0, 10} -> {0, 10}
  {
    Tensor<Scalar, 2, DataLayout> t_left(0, 0);
    Tensor<Scalar, 2, DataLayout> t_right(0, 10);
    Tensor<Scalar, 2, DataLayout> t_result(0, 10);
    assign(t_result, t_left.contract(t_right, dims));
    VERIFY_IS_EQUAL(t_result.dimension(0), 0);
    VERIFY_IS_EQUAL(t_result.dimension(1), 10);
    VERIFY_IS_EQUAL(t_result.size(), 0);
  }

  // Both outer and inner zero: {10, 0} * {0, 0} -> {10, 0}
  {
    Tensor<Scalar, 2, DataLayout> t_left(10, 0);
    Tensor<Scalar, 2, DataLayout> t_right(0, 0);
    Tensor<Scalar, 2, DataLayout> t_result(10, 0);
    assign(t_result, t_left.contract(t_right, dims));
    VERIFY_IS_EQUAL(t_result.dimension(0), 10);
    VERIFY_IS_EQUAL(t_result.dimension(1), 0);
    VERIFY_IS_EQUAL(t_result.size(), 0);
  }

  // Both outer and inner zero: {0, 0} * {0, 0} -> {0, 0}
  {
    Tensor<Scalar, 2, DataLayout> t_left(0, 0);
    Tensor<Scalar, 2, DataLayout> t_right(0, 0);
    Tensor<Scalar, 2, DataLayout> t_result(0, 0);
    assign(t_result, t_left.contract(t_right, dims));
    VERIFY_IS_EQUAL(t_result.dimension(0), 0);
    VERIFY_IS_EQUAL(t_result.dimension(1), 0);
    VERIFY_IS_EQUAL(t_result.size(), 0);
  }

  // Multidimensional with outer zero: {0, 3, 4} * {3, 2, 4} with contraction dims {1, 0} and {2, 2} -> {0, 2}
  {
    Tensor<Scalar, 3, DataLayout> t_left(0, 3, 4);
    Tensor<Scalar, 3, DataLayout> t_right(3, 2, 4);
    t_right.setRandom();
    Eigen::array<DimPair, 2> dims_md = {{DimPair(1, 0), DimPair(2, 2)}};
    Tensor<Scalar, 2, DataLayout> t_result(0, 2);
    assign(t_result, t_left.contract(t_right, dims_md));
    VERIFY_IS_EQUAL(t_result.dimension(0), 0);
    VERIFY_IS_EQUAL(t_result.dimension(1), 2);
    VERIFY_IS_EQUAL(t_result.size(), 0);
  }
}

template <int DataLayout, typename Scalar, typename Assign>
void test_zero_dim_contraction_output_kernel(const Assign& assign) {
  const Scalar non_zero = Scalar(123);
  const Scalar bias = Scalar(42);

  // 2D contraction with bias addition on zero-sized contraction dimension:
  // {10, 0} * {0, 20} -> {10, 20}
  // The contraction sum is 0, so output kernel adds bias to produce `bias` everywhere.
  {
    Tensor<Scalar, 2, DataLayout> t_left(10, 0);
    Tensor<Scalar, 2, DataLayout> t_right(0, 20);
    Tensor<Scalar, 2, DataLayout> t_result(10, 20);
    t_result.setConstant(non_zero);

    Eigen::array<DimPair, 1> dims = {{DimPair(1, 0)}};
    assign(t_result, t_left.contract(t_right, dims, AddBiasOutputKernel<Scalar>(bias)));

    VERIFY_IS_EQUAL(t_result.dimension(0), 10);
    VERIFY_IS_EQUAL(t_result.dimension(1), 20);
    for (Index i = 0; i < t_result.dimension(0); ++i) {
      for (Index j = 0; j < t_result.dimension(1); ++j) {
        VERIFY_IS_EQUAL(t_result(i, j), bias);
      }
    }
  }

  // 2D contraction with SqrtOutputKernel on zero-sized contraction dimension:
  // sqrt(0) == 0
  {
    Tensor<Scalar, 2, DataLayout> t_left(10, 0);
    Tensor<Scalar, 2, DataLayout> t_right(0, 20);
    Tensor<Scalar, 2, DataLayout> t_result(10, 20);
    t_result.setConstant(non_zero);

    Eigen::array<DimPair, 1> dims = {{DimPair(1, 0)}};
    assign(t_result, t_left.contract(t_right, dims, SqrtOutputKernel()));

    VERIFY_IS_EQUAL(t_result.dimension(0), 10);
    VERIFY_IS_EQUAL(t_result.dimension(1), 20);
    for (Index i = 0; i < t_result.size(); ++i) {
      VERIFY_IS_EQUAL(t_result.data()[i], Scalar(0));
    }
  }

  // Large contraction with bias addition on zero-sized contraction dimension:
  // {64, 0} * {0, 64} -> {64, 64}
  {
    Tensor<Scalar, 2, DataLayout> t_left_large(64, 0);
    Tensor<Scalar, 2, DataLayout> t_right_large(0, 64);
    Tensor<Scalar, 2, DataLayout> t_result_large(64, 64);
    t_result_large.setConstant(non_zero);

    Eigen::array<DimPair, 1> dims = {{DimPair(1, 0)}};
    assign(t_result_large, t_left_large.contract(t_right_large, dims, AddBiasOutputKernel<Scalar>(bias)));

    VERIFY_IS_EQUAL(t_result_large.dimension(0), 64);
    VERIFY_IS_EQUAL(t_result_large.dimension(1), 64);
    for (Index i = 0; i < t_result_large.size(); ++i) {
      VERIFY_IS_EQUAL(t_result_large.data()[i], bias);
    }
  }

  // Multidimensional contraction with bias: {5, 0, 4} * {0, 3, 4} -> {5, 3}
  {
    Tensor<Scalar, 3, DataLayout> t_left(5, 0, 4);
    Tensor<Scalar, 3, DataLayout> t_right(0, 3, 4);
    Tensor<Scalar, 2, DataLayout> t_result(5, 3);
    t_result.setConstant(non_zero);

    Eigen::array<DimPair, 2> dims = {{DimPair(1, 0), DimPair(2, 2)}};
    assign(t_result, t_left.contract(t_right, dims, AddBiasOutputKernel<Scalar>(bias)));

    VERIFY_IS_EQUAL(t_result.dimension(0), 5);
    VERIFY_IS_EQUAL(t_result.dimension(1), 3);
    for (Index i = 0; i < t_result.dimension(0); ++i) {
      for (Index j = 0; j < t_result.dimension(1); ++j) {
        VERIFY_IS_EQUAL(t_result(i, j), bias);
      }
    }
  }

  // GEMV-shape contraction with bias (e.g. 1 column in RHS): {10, 0} * {0, 1} -> {10, 1}
  {
    Tensor<Scalar, 2, DataLayout> t_left(10, 0);
    Tensor<Scalar, 2, DataLayout> t_right(0, 1);
    Tensor<Scalar, 2, DataLayout> t_result(10, 1);
    t_result.setConstant(non_zero);

    Eigen::array<DimPair, 1> dims = {{DimPair(1, 0)}};
    assign(t_result, t_left.contract(t_right, dims, AddBiasOutputKernel<Scalar>(bias)));

    VERIFY_IS_EQUAL(t_result.dimension(0), 10);
    VERIFY_IS_EQUAL(t_result.dimension(1), 1);
    for (Index i = 0; i < t_result.size(); ++i) {
      VERIFY_IS_EQUAL(t_result.data()[i], bias);
    }
  }
}

}  // namespace tensor_test

#endif  // EIGEN_TENSOR_CONTRACTION_TEST_HELPERS_H
