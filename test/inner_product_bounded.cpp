// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/LU>

using BoundedVector3d = Matrix<double, Dynamic, 1, ColMajor, 3, 1>;
using BoundedMatrix3d = Matrix<double, Dynamic, Dynamic, ColMajor, 3, 3>;

// Keep the solve result visible to GCC's bounds analysis after inlining dot().
EIGEN_DONT_INLINE BoundedVector3d bounded_update(const BoundedMatrix3d& tangent, const BoundedVector3d& residual,
                                                 const BoundedMatrix3d& transformation) {
  const BoundedVector3d increment = tangent.fullPivLu().solve(residual);
  if (increment.dot(residual) > 0) return transformation * increment;
  return transformation * residual;
}

template <typename Scalar, int Capacity>
void bounded_inner_products(const Scalar& imaginary_unit = Scalar(0)) {
  using Bounded = Matrix<Scalar, Dynamic, 1, ColMajor, Capacity, 1>;
  using DynamicVector = Matrix<Scalar, Dynamic, 1>;
  using Op = internal::scalar_inner_product_op<Scalar, Scalar, true>;
  using LeftEvaluator = internal::inner_product_evaluator<Op, Bounded, DynamicVector>;
  using RightEvaluator = internal::inner_product_evaluator<Op, DynamicVector, Bounded>;
  STATIC_CHECK(LeftEvaluator::MaxSizeAtCompileTime == Capacity);
  STATIC_CHECK(RightEvaluator::MaxSizeAtCompileTime == Capacity);
  for (Index size = 0; size <= Capacity; ++size) {
    Bounded a(size), b(size);
    Scalar dot = 0, product = 0;
    for (Index i = 0; i < size; ++i) {
      a(i) = Scalar(i % 5 - 2) + imaginary_unit * Scalar(i % 3 - 1);
      b(i) = Scalar(i % 3 + 1) + imaginary_unit * Scalar(i % 2);
      dot += numext::conj(a(i)) * b(i);
      product += a(i) * b(i);
    }
    const DynamicVector dynamic = b;
    VERIFY_IS_EQUAL(a.dot(b), dot);
    VERIFY_IS_EQUAL(a.dot(dynamic), dot);
    VERIFY_IS_EQUAL(dynamic.dot(a), numext::conj(dot));
    VERIFY_IS_EQUAL(a.transpose().dot(b.transpose()), dot);
    VERIFY_IS_EQUAL((a.transpose() * b).value(), product);
  }
}

EIGEN_DECLARE_TEST(inner_product_bounded) {
  for (Index size = 1; size <= 3; ++size) {
    BoundedMatrix3d tangent = BoundedMatrix3d::Identity(size, size);
    const BoundedMatrix3d transformation = BoundedMatrix3d::Identity(size, size);
    BoundedVector3d residual(size);
    for (Index i = 0; i < size; ++i) tangent(i, i) = residual(i) = double(i + 2);
    VERIFY_IS_EQUAL(bounded_update(tangent, residual, transformation), BoundedVector3d::Ones(size));
    tangent = -tangent;
    VERIFY_IS_EQUAL(bounded_update(tangent, residual, transformation), residual);
  }
  bounded_inner_products<double, 3>();
  bounded_inner_products<double, 5>();
  bounded_inner_products<double, 7>();
  bounded_inner_products<double, 17>();
  constexpr int packet_size = internal::packet_traits<double>::size;
  bounded_inner_products<double, 5 * packet_size + 1>();
  bounded_inner_products<double, 6 * packet_size + 1>();
  bounded_inner_products<double, 7 * packet_size + 1>();
  bounded_inner_products<float, 3>();
  bounded_inner_products<float, 7>();
  bounded_inner_products<float, 15>();
  bounded_inner_products<float, 33>();
  bounded_inner_products<std::complex<double>, 9>(std::complex<double>(0, 1));
}
