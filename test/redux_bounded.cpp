// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

using BoundedVector4d = Matrix<double, Dynamic, 1, ColMajor, 4, 1>;
using BoundedMatrix4d = Matrix<double, Dynamic, Dynamic, ColMajor, 4, 4>;

EIGEN_DONT_INLINE MatrixXd bounded_transform(const MatrixXd& vertices, const BoundedMatrix4d& transform) {
  MatrixXd transformed(vertices.rows(), vertices.cols());
  for (Index col = 0; col < vertices.cols(); ++col) {
    BoundedVector4d homogeneous = BoundedVector4d::Ones(vertices.rows() + 1);
    homogeneous.head(vertices.rows()) = vertices.col(col);
    transformed.col(col) = (transform * homogeneous).head(vertices.rows());
  }
  return transformed;
}

struct BoundedFirst {
  int operator()(int first, int) const { return first; }
};

template <int Order>
void mixed_bounded_products() {
  using BoundedVector = Matrix<double, Dynamic, 1, ColMajor, 3, 1>;
  using DynamicMatrix = Matrix<double, Dynamic, Dynamic, Order>;
  for (Index size = 1; size <= 3; ++size) {
    const DynamicMatrix matrix = DynamicMatrix::Random(size, size);
    const BoundedVector vector = BoundedVector::Random(size);
    const VectorXd dynamic = vector;
    const auto left = dynamic.cwiseProduct(vector);
    const auto right = vector.cwiseProduct(dynamic);
    STATIC_CHECK(decltype(left)::MaxSizeAtCompileTime == Dynamic);
    STATIC_CHECK(decltype(right)::MaxSizeAtCompileTime == 3);
    STATIC_CHECK(internal::redux_max_size<internal::remove_all_t<decltype(left)>>::Size == 3);
    STATIC_CHECK(internal::redux_max_size<internal::remove_all_t<decltype(right)>>::Size == 3);
    VERIFY_IS_EQUAL(left.sum(), right.sum());
    const BoundedVector result = matrix * vector;
    const auto reversed = (vector.transpose() * matrix.transpose()).eval();
    for (Index row = 0; row < size; ++row) {
      double expected = 0;
      for (Index col = 0; col < size; ++col) expected += matrix(row, col) * vector(col);
      VERIFY(numext::abs(result(row) - expected) <= 8 * NumTraits<double>::epsilon());
      VERIFY(numext::abs(reversed(row) - expected) <= 8 * NumTraits<double>::epsilon());
    }
  }
}

template <int Order>
void heap_backed_expression_bounds() {
  using RowsBounded = Matrix<double, Dynamic, Dynamic, Order, 200, Dynamic>;
  using ColsBounded = Matrix<double, Dynamic, Dynamic, Order, Dynamic, 200>;
  const RowsBounded a = RowsBounded::Ones(2, 2);
  const ColsBounded b = ColsBounded::Ones(2, 2);
  using Bounds = internal::redux_max_size<internal::remove_all_t<decltype(a + b)>>;
  STATIC_CHECK(Bounds::Rows == 200);
  STATIC_CHECK(Bounds::Cols == 200);
  STATIC_CHECK(Bounds::Size == 40000);
  const auto sum = (a + b).eval();
  const auto reversed = (b + a).eval();
  STATIC_CHECK((std::is_same<internal::remove_all_t<decltype(sum)>, RowsBounded>::value));
  STATIC_CHECK((std::is_same<internal::remove_all_t<decltype(reversed)>, ColsBounded>::value));
  VERIFY_IS_EQUAL(sum, RowsBounded::Constant(2, 2, 2));
  VERIFY_IS_EQUAL(reversed, ColsBounded::Constant(2, 2, 2));
  VERIFY_IS_EQUAL((a + b).sum(), 8);
}

template <int Order>
void complementary_reduction_bounds() {
  using RowsBounded = Matrix<int, Dynamic, Dynamic, Order, 4, Dynamic>;
  using ColsBounded = Matrix<int, Dynamic, Dynamic, Order, Dynamic, 4>;
  const RowsBounded a = RowsBounded::Constant(3, 4, 2);
  const ColsBounded b = ColsBounded::Constant(3, 4, 3);
  using Bounds = internal::redux_max_size<internal::remove_all_t<decltype(a + b)>>;
  STATIC_CHECK(Bounds::Size == 16);
  VERIFY_IS_EQUAL((a + b).sum(), 60);
  VERIFY_IS_EQUAL((a + b).redux(BoundedFirst()), 5);
  VERIFY_IS_EQUAL((b + a).redux(BoundedFirst()), 5);

  const auto wrapped_array = (a + b).array();
  const auto wrapped_matrix = (a.array() + b.array()).matrix();
  using ArrayBounds = internal::redux_max_size<internal::remove_all_t<decltype(wrapped_array)>>;
  using MatrixBounds = internal::redux_max_size<internal::remove_all_t<decltype(wrapped_matrix)>>;
  STATIC_CHECK(ArrayBounds::Rows == 4 && ArrayBounds::Cols == 4 && ArrayBounds::Size == 16);
  STATIC_CHECK(MatrixBounds::Rows == 4 && MatrixBounds::Cols == 4 && MatrixBounds::Size == 16);
  STATIC_CHECK(decltype(wrapped_array)::MaxSizeAtCompileTime == Dynamic);
  STATIC_CHECK(decltype(wrapped_matrix)::MaxSizeAtCompileTime == Dynamic);
  VERIFY_IS_EQUAL(wrapped_array.sum(), 60);
  VERIFY_IS_EQUAL(wrapped_matrix.sum(), 60);
  VERIFY_IS_EQUAL(wrapped_array.redux(BoundedFirst()), 5);
  VERIFY_IS_EQUAL(wrapped_matrix.redux(BoundedFirst()), 5);

  const Matrix<bool, Dynamic, Dynamic, Order> condition = Matrix<bool, Dynamic, Dynamic, Order>::Constant(3, 4, true);
  const auto selected = condition.select(a, b);
  const auto reversed = condition.select(b, a);
  using SelectedBounds = internal::redux_max_size<internal::remove_all_t<decltype(selected)>>;
  using ReversedBounds = internal::redux_max_size<internal::remove_all_t<decltype(reversed)>>;
  STATIC_CHECK(SelectedBounds::Rows == 4 && SelectedBounds::Cols == 4 && SelectedBounds::Size == 16);
  STATIC_CHECK(ReversedBounds::Size == 16);
  STATIC_CHECK(decltype(selected)::MaxSizeAtCompileTime == Dynamic);
  VERIFY_IS_EQUAL(selected.sum(), 24);
  VERIFY_IS_EQUAL(reversed.sum(), 36);
  VERIFY_IS_EQUAL(selected.redux(BoundedFirst()), 2);
  VERIFY_IS_EQUAL(reversed.redux(BoundedFirst()), 3);

  using Bounded = Matrix<int, Dynamic, Dynamic, Order, 4, 4>;
  const Bounded bounded = Bounded::Constant(3, 4, 7);
  const Matrix<int, Dynamic, Dynamic, Order> dynamic = bounded;
  const auto bounded_second = condition.select(bounded, dynamic);
  const auto bounded_third = condition.select(dynamic, bounded);
  const auto bounded_first = (bounded.array() > 0).select(dynamic.array(), dynamic.array());
  STATIC_CHECK(internal::redux_max_size<internal::remove_all_t<decltype(bounded_second)>>::Size == 16);
  STATIC_CHECK(internal::redux_max_size<internal::remove_all_t<decltype(bounded_third)>>::Size == 16);
  STATIC_CHECK(internal::redux_max_size<internal::remove_all_t<decltype(bounded_first)>>::Size == 16);
  VERIFY_IS_EQUAL(bounded_second.sum(), 84);
  VERIFY_IS_EQUAL(bounded_third.sum(), 84);
  VERIFY_IS_EQUAL(bounded_first.sum(), 84);

  using HugeRows = Matrix<int, Dynamic, Dynamic, Order, 100000, Dynamic>;
  using HugeCols = Matrix<int, Dynamic, Dynamic, Order, Dynamic, 100000>;
  using HugeExpression = decltype(std::declval<HugeRows>() + std::declval<HugeCols>());
  STATIC_CHECK(internal::redux_max_size<HugeExpression>::Size == Dynamic);
}

template <typename Scalar, int Capacity>
void mixed_packet_reductions() {
  using Bounded = Matrix<Scalar, Dynamic, 1, ColMajor, Capacity, 1>;
  using DynamicVector = Matrix<Scalar, Dynamic, 1>;
  for (Index size = 0; size <= Capacity; ++size) {
    Bounded bounded(size);
    DynamicVector dynamic(size);
    Scalar expected = 0;
    for (Index i = 0; i < size; ++i) {
      bounded(i) = Scalar(i % 5 - 2);
      dynamic(i) = Scalar(i % 3 + 1);
      expected += bounded(i) * dynamic(i);
    }
    const auto left = dynamic.cwiseProduct(bounded);
    const auto right = bounded.cwiseProduct(dynamic);
    STATIC_CHECK(decltype(left)::MaxSizeAtCompileTime == Dynamic);
    STATIC_CHECK(internal::redux_max_size<internal::remove_all_t<decltype(left)>>::Size == Capacity);
    STATIC_CHECK(internal::redux_max_size<internal::remove_all_t<decltype(right)>>::Size == Capacity);
    VERIFY_IS_EQUAL(left.sum(), expected);
    VERIFY_IS_EQUAL(right.sum(), expected);
    const auto squared = left.cwiseAbs2();
    const auto negated = -squared;
    STATIC_CHECK(decltype(squared)::MaxSizeAtCompileTime == Dynamic);
    STATIC_CHECK(internal::redux_max_size<internal::remove_all_t<decltype(squared)>>::Size == Capacity);
    STATIC_CHECK(internal::redux_max_size<internal::remove_all_t<decltype(negated)>>::Size == Capacity);
    Scalar expectedSquared = 0;
    for (Index i = 0; i < size; ++i) {
      const Scalar value = bounded(i) * dynamic(i);
      expectedSquared += value * value;
    }
    VERIFY_IS_EQUAL(squared.sum(), expectedSquared);
    VERIFY_IS_EQUAL(negated.sum(), -expectedSquared);
    VERIFY_IS_EQUAL(left.squaredNorm(), expectedSquared);

    const auto wrapped = left.array().square();
    auto array_product = dynamic.array() * bounded.array();
    const auto matrix = array_product.matrix();
    const auto nested = matrix.array().square().matrix();
    STATIC_CHECK(internal::redux_max_size<internal::remove_all_t<decltype(wrapped)>>::Size == Capacity);
    STATIC_CHECK(internal::redux_max_size<internal::remove_all_t<decltype(matrix)>>::Size == Capacity);
    STATIC_CHECK(internal::redux_max_size<internal::remove_all_t<decltype(nested)>>::Size == Capacity);
    VERIFY_IS_EQUAL(wrapped.sum(), expectedSquared);
    VERIFY_IS_EQUAL(matrix.sum(), expected);
    VERIFY_IS_EQUAL(matrix.squaredNorm(), expectedSquared);
    VERIFY_IS_EQUAL(nested.sum(), expectedSquared);
  }
}

template <int Order>
void bounded_reductions() {
  for (Index size : {1, 4, 15, 16, 31, 32, 33, 64}) {
    Matrix<int, Dynamic, Dynamic, Order, 64, 64> matrix(size, size);
    for (Index row = 0; row < size; ++row)
      for (Index col = 0; col < size; ++col) matrix(row, col) = int(row + col + 1);
    VERIFY_IS_EQUAL(matrix.sum(), size * size * size);
    VERIFY_IS_EQUAL(matrix.redux(BoundedFirst()), 1);
    VERIFY_IS_EQUAL(matrix.topLeftCorner(size, size).sum(), size * size * size);
    VERIFY_IS_EQUAL(matrix.topLeftCorner(size, size).redux(BoundedFirst()), 1);
  }
}

EIGEN_DECLARE_TEST(redux_bounded) {
  mixed_packet_reductions<float, 3>();
  mixed_packet_reductions<float, 7>();
  mixed_packet_reductions<float, 15>();
  mixed_packet_reductions<float, 17>();
  mixed_packet_reductions<double, 3>();
  mixed_packet_reductions<double, 7>();
  mixed_packet_reductions<double, 15>();
  mixed_packet_reductions<double, 17>();
  for (Index size = 0; size <= 3; ++size) {
    const MatrixXd vertices = MatrixXd::Random(size, 10);
    const BoundedMatrix4d transform = BoundedMatrix4d::Random(size + 1, size + 1);
    const MatrixXd result = bounded_transform(vertices, transform);
    for (Index row = 0; row < size; ++row) {
      for (Index col = 0; col < vertices.cols(); ++col) {
        double expected = transform(row, size);
        for (Index k = 0; k < size; ++k) expected += transform(row, k) * vertices(k, col);
        VERIFY(numext::abs(result(row, col) - expected) <= 32 * NumTraits<double>::epsilon());
      }
    }
  }
  bounded_reductions<ColMajor>();
  bounded_reductions<RowMajor>();
  mixed_bounded_products<ColMajor>();
  mixed_bounded_products<RowMajor>();
  heap_backed_expression_bounds<ColMajor>();
  heap_backed_expression_bounds<RowMajor>();
  complementary_reduction_bounds<ColMajor>();
  complementary_reduction_bounds<RowMajor>();
}
