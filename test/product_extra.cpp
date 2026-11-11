// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_RUNTIME_NO_MALLOC
#include "main.h"
#include "NonCommutativeScalar.h"
#include "product_test_helpers.h"

namespace noncommutative_scalar {
inline Quaternion operator*(double lhs, const Quaternion& rhs) { return Quaternion(lhs) * rhs; }
}  // namespace noncommutative_scalar

namespace Eigen {
template <>
struct ScalarBinaryOpTraits<noncommutative_scalar::Quaternion, double> {
  using ReturnType = noncommutative_scalar::Quaternion;
  static constexpr bool IsCommutative = true;
};
template <>
struct ScalarBinaryOpTraits<double, noncommutative_scalar::Quaternion>
    : ScalarBinaryOpTraits<noncommutative_scalar::Quaternion, double> {};
}  // namespace Eigen

template <int LhsOrder, int RhsOrder, int DstOrder>
void test_dense_quaternion_factor_order() {
  using Scalar = noncommutative_scalar::Quaternion;
  using Lhs = Matrix<Scalar, Dynamic, Dynamic, LhsOrder>;
  using Rhs = Matrix<Scalar, Dynamic, Dynamic, RhsOrder>;
  using Dst = Matrix<Scalar, Dynamic, Dynamic, DstOrder>;
  using Impl = internal::generic_product_impl<Lhs, Rhs, DenseShape, DenseShape, GemmProduct>;
  STATIC_CHECK((!internal::functor_is_commutative<internal::scalar_product_op<Scalar>>::value));
  const Scalar p(0, 1, 0, 0), q(0, 0, 1, 0), alpha(1, 0, 0, 1), initial(1, 2, 3, 4);
  const auto verify = [&](const auto& lhs, const auto& rhs) {
    const Lhs& left = lhs;
    const Rhs& right = rhs;
    Dst expected = Dst::Zero(lhs.rows(), rhs.cols());
    for (Index j = 0; j < expected.cols(); ++j)
      for (Index i = 0; i < expected.rows(); ++i)
        for (Index k = 0; k < left.cols(); ++k) expected(i, j) += left(i, k) * right(k, j);
    Dst actual = lhs * rhs;
    VERIFY(actual == expected);
    Array<Scalar, Dynamic, Dynamic, DstOrder> array = lhs * rhs;
    VERIFY(array.matrix() == expected);
    array = alpha * (lhs * rhs);
    for (Index k = 0; k < array.size(); ++k) VERIFY(array(k) == alpha * expected(k));
    actual.setConstant(initial);
    actual.noalias() += lhs * rhs;
    for (Index k = 0; k < actual.size(); ++k) VERIFY(actual(k) == initial + expected(k));
    actual.noalias() -= lhs * rhs;
    for (Index k = 0; k < actual.size(); ++k) VERIFY(actual(k) == initial);
    actual.noalias() = alpha * (lhs * rhs);
    for (Index k = 0; k < actual.size(); ++k) VERIFY(actual(k) == alpha * expected(k));
    actual.noalias() = (lhs * rhs) * alpha;
    for (Index k = 0; k < actual.size(); ++k) VERIFY(actual(k) == expected(k) * alpha);
    Lhs aliasLeft = left;
    aliasLeft = aliasLeft * rhs;
    VERIFY(aliasLeft == expected);
    Rhs aliasRight = right;
    aliasRight = lhs * aliasRight;
    VERIFY(aliasRight == expected);
    Dst storage = Dst::Constant(2 * expected.rows(), 2 * expected.cols(), initial);
    Map<Dst, Unaligned, Stride<Dynamic, 2>> strided(storage.data(), expected.rows(), expected.cols(),
                                                    Stride<Dynamic, 2>(storage.outerStride(), 2));
    strided.noalias() = lhs * rhs;
    VERIFY(strided == expected);
    VERIFY(strided_block_gaps_untouched(storage, Dst::Constant(storage.rows(), storage.cols(), initial), 0,
                                        DstOrder == ColMajor ? expected.rows() : expected.cols(),
                                        DstOrder == ColMajor ? expected.cols() : expected.rows(), 2,
                                        storage.outerStride()));
  };
  const Index shapes[][3] = {{3, 3, 3}, {12, 9, 11}, {0, 3, 2},  {2, 0, 3},
                             {1, 7, 5}, {7, 5, 1},   {30, 5, 1}, {1, 5, 30}};
  for (const auto& shape : shapes) {
    const Index rows = shape[0], depth = shape[1], cols = shape[2];
    Lhs a(rows, depth);
    Rhs b(depth, cols);
    for (Index k = 0; k < a.size(); ++k) a(k) = Scalar(double(k % 3), 1, double(k % 2), -1);
    for (Index k = 0; k < b.size(); ++k) b(k) = Scalar(1, double(k % 2), -1, double(k % 3));
    verify(a, b);
    verify(a, b * q);
    verify(a, q * b);
    verify(a * p, b);
    verify(p * a, b);
    verify(p * (a * q), (q * b) * p);
    verify(a.conjugate(), (b * q).conjugate());
    verify(a.conjugate(), b.conjugate());
    verify(b.adjoint(), a.adjoint());
    verify(a + a, b);
    verify(a, b + b);
    Lhs leftStorage(2 * rows, 2 * depth);
    Rhs rightStorage(2 * depth, 2 * cols);
    using Strides = Stride<Dynamic, Dynamic>;
    Map<Lhs, Unaligned, Strides> leftMap(leftStorage.data(), rows, depth, Strides(leftStorage.outerStride(), 2));
    Map<Rhs, Unaligned, Strides> rightMap(rightStorage.data(), depth, cols, Strides(rightStorage.outerStride(), 2));
    leftMap = a;
    rightMap = b;
    verify(leftMap, rightMap);
    if (cols == 1) {
      const Matrix<Scalar, Dynamic, 1> vector = b;
      verify(p * a, vector);
      verify(a * q, vector);
      verify(a + a, vector * q);
    }
    Dst expected = Dst::Zero(rows, cols);
    for (Index j = 0; j < cols; ++j)
      for (Index i = 0; i < rows; ++i)
        for (Index k = 0; k < depth; ++k) expected(i, j) += a(i, k) * b(k, j);
    Dst actual = Dst::Constant(rows, cols, initial);
    Impl::scaleAndAddTo(actual, a, b, alpha);
    for (Index k = 0; k < actual.size(); ++k) VERIFY(actual(k) == initial + alpha * expected(k));
    Impl::scaleAndEvalTo(actual, a, b, alpha);
    for (Index k = 0; k < actual.size(); ++k) VERIFY(actual(k) == alpha * expected(k));
    actual.setConstant(initial);
    Impl::scaleAndAddTo(actual, a, b, Scalar(0));
    for (Index k = 0; k < actual.size(); ++k) VERIFY(actual(k) == initial);
    Impl::scaleAndEvalTo(actual, a, b, Scalar(0));
    VERIFY(actual == Dst::Zero(rows, cols));
    a.setConstant(Scalar((std::numeric_limits<double>::quiet_NaN)()));
    actual.setConstant(initial);
    Impl::scaleAndAddTo(actual, a, b, Scalar(0));
    for (Index k = 0; k < actual.size(); ++k) VERIFY(actual(k) == initial);
    Impl::scaleAndEvalTo(actual, a, b, Scalar(0));
    VERIFY(actual == Dst::Zero(rows, cols));
    if (depth == 0) {
      const Scalar infinity((std::numeric_limits<double>::infinity)());
      actual.setConstant(initial);
      Impl::scaleAndAddTo(actual, a, b, infinity);
      VERIFY(actual == Dst::Constant(rows, cols, initial));
      Impl::scaleAndEvalTo(actual, a, b, infinity);
      VERIFY(actual == Dst::Zero(rows, cols));
    }
  }
  const Lhs nonfinite = Lhs::Constant(12, 12, Scalar((std::numeric_limits<double>::infinity)()));
  const Rhs ones = Rhs::Constant(12, 12, Scalar(1));
  const Scalar zero(0);
  Dst zeroResult(12, 12);
  const auto verifyZeroProduct = [&](const auto& product) {
    zeroResult.noalias() = product;
    VERIFY(zeroResult == Dst::Zero(12, 12));
    zeroResult.setConstant(initial);
    zeroResult.noalias() += product;
    VERIFY(zeroResult == Dst::Constant(12, 12, initial));
    zeroResult.noalias() -= product;
    VERIFY(zeroResult == Dst::Constant(12, 12, initial));
  };
  verifyZeroProduct(zero * (nonfinite * ones));
  verifyZeroProduct((zero * nonfinite) * ones);
  verifyZeroProduct((nonfinite * zero) * ones);
  verifyZeroProduct(nonfinite * (zero * ones));
  verifyZeroProduct(nonfinite * (ones * zero));
  verifyZeroProduct((q * (nonfinite * zero)).conjugate() * ones);
  verifyZeroProduct((zero * (nonfinite * q)).transpose() * ones);
  verifyZeroProduct(-(q * (nonfinite * zero)) * ones);
  const Scalar tiny(std::ldexp(1.0, -700)), huge(std::ldexp(1.0, 700));
  const Lhs largeLeft = Lhs::Constant(12, 12, huge);
  const Rhs largeRight = Rhs::Constant(12, 12, huge);
  VERIFY(tiny * tiny == zero);
  verify(tiny * (largeLeft * tiny), ones);
  verify(ones, tiny * (largeRight * tiny));
  const Matrix<Scalar, 9, 9, LhsOrder> fixedLeft = Matrix<Scalar, 9, 9, LhsOrder>::Constant(p);
  const Matrix<Scalar, 9, 9, RhsOrder> fixedRight = Matrix<Scalar, 9, 9, RhsOrder>::Constant(q);
  verify(fixedLeft, fixedRight * alpha);
  const Matrix<Scalar, 16, 16, LhsOrder> gemmLeft = Matrix<Scalar, 16, 16, LhsOrder>::Constant(p);
  const Matrix<Scalar, 16, 16, RhsOrder> gemmRight = Matrix<Scalar, 16, 16, RhsOrder>::Constant(q);
  STATIC_CHECK((internal::product_type<decltype(gemmLeft), decltype(gemmRight)>::value == GemmProduct));
  verify(gemmLeft, gemmRight * alpha);
  using BoundedLhs = Matrix<Scalar, Dynamic, Dynamic, LhsOrder, 9, 9>;
  using BoundedRhs = Matrix<Scalar, Dynamic, Dynamic, RhsOrder, 9, 9>;
  const BoundedLhs boundedLeft = fixedLeft.topLeftCorner(3, 3);
  const BoundedRhs boundedRight = fixedRight.topLeftCorner(3, 3);
  verify(boundedLeft, boundedRight * alpha);
  EIGEN_IF_CONSTEXPR (LhsOrder == ColMajor && RhsOrder == ColMajor && DstOrder == ColMajor) {
    using Fixed = Matrix<Scalar, 65, 65>;
    const Lhs left = Lhs::Constant(65, 65, p), right = Lhs::Constant(65, 65, q);
    const Map<const Fixed> leftMap(left.data()), rightMap(right.data());
    const Dst actual = leftMap * (rightMap * alpha);
    const Scalar expected = Scalar(65) * (p * (q * alpha));
    for (Index k = 0; k < actual.size(); ++k) VERIFY(actual(k) == expected);
    STATIC_CHECK((internal::functor_is_commutative<internal::scalar_product_op<Scalar, double>>::value));
    using Mixed = internal::generic_product_impl<Lhs, MatrixXd, DenseShape, DenseShape, GemmProduct>;
    STATIC_CHECK((std::is_base_of<internal::gemm_product_impl<Lhs, MatrixXd, false>, Mixed>::value));
    const Lhs mixedLeft = Lhs::Constant(12, 9, p), mixedRight = Lhs::Constant(9, 11, q);
    const MatrixXd realLeft = MatrixXd::Constant(12, 9, 2), realRight = MatrixXd::Constant(9, 11, 2);
    const Dst mixedProduct = mixedLeft * realRight;
    const Dst reversedProduct = realLeft * mixedRight;
    for (Index k = 0; k < mixedProduct.size(); ++k) {
      VERIFY(mixedProduct(k) == p * Scalar(18));
      VERIFY(reversedProduct(k) == Scalar(18) * q);
    }
  }
}

template <int Order>
void test_dense_quaternion_row_order() {
  using Scalar = noncommutative_scalar::Quaternion;
  const Scalar i(0, 1, 0, 0), j(0, 0, 1, 0);
  Matrix<Scalar, Dynamic, Dynamic, Order> matrix = Matrix<Scalar, Dynamic, Dynamic, Order>::Constant(3, 4, j);
  Matrix<Scalar, 1, Dynamic> row = Matrix<Scalar, 1, Dynamic>::Constant(3, i);
  const Matrix<Scalar, 1, Dynamic> rowProduct = row * matrix;
  const Matrix<Scalar, Dynamic, 1> column = row.transpose();
  const Matrix<Scalar, 1, Dynamic> transposedProduct = column.transpose() * matrix;
  Matrix<Scalar, 1, Dynamic> scaledRow = Matrix<Scalar, 1, Dynamic>::Zero(matrix.cols());
  scaledRow.noalias() += j * (row * matrix);
  for (Index col = 0; col < matrix.cols(); ++col) {
    Scalar expected;
    for (Index k = 0; k < row.size(); ++k) expected += row(k) * matrix(k, col);
    VERIFY(rowProduct(col) == expected);
    VERIFY(transposedProduct(col) == expected);
    VERIFY(scaledRow(col) == j * expected);
  }
}

template <int Order>
void test_dense_quaternion_outer_order() {
  using Scalar = noncommutative_scalar::Quaternion;
  using Vector = Matrix<Scalar, Dynamic, 1>;
  using Result = Matrix<Scalar, Dynamic, Dynamic, Order>;
  const Scalar i(0, 1, 0, 0), j(0, 0, 1, 0), scale(1, 0, 1, 0), initial(1, 2, 3, 4);
  // Exercise the scalar small-assignment path and the column/row expression path.
  for (Index n : {3, 40}) {
    const Vector u = Vector::Constant(n, i), v = Vector::Constant(n, j);
    const auto verify = [&](const auto& lhs, const auto& rhs) {
      Result expected(lhs.rows(), rhs.cols());
      for (Index col = 0; col < expected.cols(); ++col)
        for (Index row = 0; row < expected.rows(); ++row) expected(row, col) = lhs(row) * rhs(col);
      Result actual = lhs * rhs;
      VERIFY(actual == expected);
      actual.setConstant(initial);
      actual.noalias() += lhs * rhs;
      for (Index k = 0; k < actual.size(); ++k) VERIFY(actual(k) == initial + expected(k));
      actual.noalias() -= lhs * rhs;
      for (Index k = 0; k < actual.size(); ++k) VERIFY(actual(k) == initial);
      actual.setConstant(initial);
      actual.noalias() += scale * (lhs * rhs);
      for (Index k = 0; k < actual.size(); ++k) VERIFY(actual(k) == initial + scale * expected(k));
    };
    verify(u, v.transpose());
    verify((j * u).eval(), v.adjoint());
    verify(u.conjugate(), (v * i).transpose());
  }
}

template <int Order>
void test_dense_quaternion_constant_zero() {
  using Q = noncommutative_scalar::Quaternion;
  using M = Matrix<Q, Dynamic, Dynamic, Order>;
  using V = Matrix<Q, Dynamic, 1>;
  const V v = V::Constant(30, Q((std::numeric_limits<double>::quiet_NaN)()));
  const auto expression = Q(0) * M::Constant(30, 30, Q(2));
  using Traits = internal::blas_traits<internal::remove_all_t<decltype(expression)>>;
  STATIC_CHECK((Traits::HasScalarFactor));
  VERIFY(!Traits::hasZeroScalarFactor(expression));
  const V scaled = expression * v;
  const V rightScaled = (M::Constant(30, 30, Q(2)) * Q(0)) * v;
  const V unscaled = M::Constant(30, 30, Q(0)) * v;
  for (Index i = 0; i < v.size(); ++i) {
    VERIFY((numext::isnan)(scaled(i).r));
    VERIFY((numext::isnan)(rightScaled(i).r));
    VERIFY((numext::isnan)(unscaled(i).r));
  }
  const V infinite = M::Constant(30, 30, Q((std::numeric_limits<double>::infinity)())) * V::Constant(30, Q(1, 1, 1, 1));
  for (Index i = 0; i < infinite.size(); ++i) {
    VERIFY((numext::isinf)(infinite(i).r));
    VERIFY((numext::isinf)(infinite(i).i));
    VERIFY((numext::isinf)(infinite(i).j));
    VERIFY((numext::isinf)(infinite(i).k));
  }
}

template <int Order, bool Left>
void test_dense_quaternion_gemv_scales() {
  using Q = noncommutative_scalar::Quaternion;
  using MatrixType = Matrix<Q, Dynamic, Dynamic, Order>;
  using Vector = Matrix<Q, Dynamic, 1>;
  using Result = Matrix<Q, Left ? 1 : Dynamic, Left ? Dynamic : 1, Left ? RowMajor : ColMajor>;
  const Index size = 17, depth = 9;
  const Q p(0, 1, 0, 0), alpha(0, 0, 1, 0);
  MatrixType matrix = MatrixType::Constant(Left ? depth : size, Left ? size : depth, Q(1, 2, 3, 4));
  Vector vector = Vector::Constant(depth, Q(2, 1, 0, 1));
  Result result(Left ? 1 : size, Left ? size : 1);
  if (Left)
    result.noalias() = alpha * ((p * vector.transpose()) * matrix);
  else
    result.noalias() = alpha * ((p * matrix) * vector);
  Q sum(0);
  for (Index k = 0; k < depth; ++k) sum += Left ? vector(k) * matrix(k, 0) : matrix(0, k) * vector(k);
  for (Index i = 0; i < size; ++i) VERIFY(result(i) == alpha * (p * sum));
  for (bool large : {false, true}) {
    const double scale = large ? 1e200 : 1e-200;
    matrix.setConstant(Q(1 / scale));
    vector.setConstant(Q(1));
    if (Left)
      result.noalias() = Q(scale) * ((Q(scale) * vector.transpose()) * matrix);
    else
      result.noalias() = Q(scale) * ((Q(scale) * matrix) * vector);
    const double expected = scale * double(depth);
    for (Index i = 0; i < size; ++i) {
      VERIFY(std::abs(result(i).r - expected) <= 128 * NumTraits<double>::epsilon() * std::abs(expected));
      VERIFY_IS_EQUAL(result(i).i, 0);
      VERIFY_IS_EQUAL(result(i).j, 0);
      VERIFY_IS_EQUAL(result(i).k, 0);
    }
  }
}

template <typename MatrixType>
void product_extra(const MatrixType& m) {
  using Scalar = typename MatrixType::Scalar;
  using RowVectorType = Matrix<Scalar, 1, Dynamic>;
  using ColVectorType = Matrix<Scalar, Dynamic, 1>;
  using OtherMajorMatrixType = Matrix<Scalar, Dynamic, Dynamic, MatrixType::Flags & RowMajorBit>;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols), m2 = MatrixType::Random(rows, cols), m3(rows, cols),
             mzero = MatrixType::Zero(rows, cols), identity = MatrixType::Identity(rows, rows),
             square = MatrixType::Random(rows, rows), res = MatrixType::Random(rows, rows),
             square2 = MatrixType::Random(cols, cols), res2 = MatrixType::Random(cols, cols);
  RowVectorType v1 = RowVectorType::Random(rows), vrres(rows);
  ColVectorType vc2 = ColVectorType::Random(cols), vcres(cols);
  OtherMajorMatrixType tm1 = m1;

  Scalar s1 = internal::random<Scalar>(), s2 = internal::random<Scalar>(), s3 = internal::random<Scalar>();

  VERIFY_IS_APPROX(m3.noalias() = m1 * m2.adjoint(), m1 * m2.adjoint().eval());
  VERIFY_IS_APPROX(m3.noalias() = m1.adjoint() * square.adjoint(), m1.adjoint().eval() * square.adjoint().eval());
  VERIFY_IS_APPROX(m3.noalias() = m1.adjoint() * m2, m1.adjoint().eval() * m2);
  VERIFY_IS_APPROX(m3.noalias() = (s1 * m1.adjoint()) * m2, (s1 * m1.adjoint()).eval() * m2);
  VERIFY_IS_APPROX(m3.noalias() = ((s1 * m1).adjoint()) * m2, (numext::conj(s1) * m1.adjoint()).eval() * m2);
  VERIFY_IS_APPROX(m3.noalias() = (-m1.adjoint() * s1) * (s3 * m2), (-m1.adjoint() * s1).eval() * (s3 * m2).eval());
  VERIFY_IS_APPROX(m3.noalias() = (s2 * m1.adjoint() * s1) * m2, (s2 * m1.adjoint() * s1).eval() * m2);
  VERIFY_IS_APPROX(m3.noalias() = (-m1 * s2) * s1 * m2.adjoint(), (-m1 * s2).eval() * (s1 * m2.adjoint()).eval());

  // a very tricky case where a scale factor has to be automatically conjugated:
  VERIFY_IS_APPROX(m1.adjoint() * (s1 * m2).conjugate(), (m1.adjoint()).eval() * ((s1 * m2).conjugate()).eval());

  // test all possible conjugate combinations for the four matrix-vector product cases:

  VERIFY_IS_APPROX((-m1.conjugate() * s2) * (s1 * vc2), (-m1.conjugate() * s2).eval() * (s1 * vc2).eval());
  VERIFY_IS_APPROX((-m1 * s2) * (s1 * vc2.conjugate()), (-m1 * s2).eval() * (s1 * vc2.conjugate()).eval());
  VERIFY_IS_APPROX((-m1.conjugate() * s2) * (s1 * vc2.conjugate()),
                   (-m1.conjugate() * s2).eval() * (s1 * vc2.conjugate()).eval());

  VERIFY_IS_APPROX((s1 * vc2.transpose()) * (-m1.adjoint() * s2),
                   (s1 * vc2.transpose()).eval() * (-m1.adjoint() * s2).eval());
  VERIFY_IS_APPROX((s1 * vc2.adjoint()) * (-m1.transpose() * s2),
                   (s1 * vc2.adjoint()).eval() * (-m1.transpose() * s2).eval());
  VERIFY_IS_APPROX((s1 * vc2.adjoint()) * (-m1.adjoint() * s2),
                   (s1 * vc2.adjoint()).eval() * (-m1.adjoint() * s2).eval());

  VERIFY_IS_APPROX((-m1.adjoint() * s2) * (s1 * v1.transpose()),
                   (-m1.adjoint() * s2).eval() * (s1 * v1.transpose()).eval());
  VERIFY_IS_APPROX((-m1.transpose() * s2) * (s1 * v1.adjoint()),
                   (-m1.transpose() * s2).eval() * (s1 * v1.adjoint()).eval());
  VERIFY_IS_APPROX((-m1.adjoint() * s2) * (s1 * v1.adjoint()),
                   (-m1.adjoint() * s2).eval() * (s1 * v1.adjoint()).eval());

  VERIFY_IS_APPROX((s1 * v1) * (-m1.conjugate() * s2), (s1 * v1).eval() * (-m1.conjugate() * s2).eval());
  VERIFY_IS_APPROX((s1 * v1.conjugate()) * (-m1 * s2), (s1 * v1.conjugate()).eval() * (-m1 * s2).eval());
  VERIFY_IS_APPROX((s1 * v1.conjugate()) * (-m1.conjugate() * s2),
                   (s1 * v1.conjugate()).eval() * (-m1.conjugate() * s2).eval());

  // test the vector-matrix product with non aligned starts
  Index i = internal::random<Index>(0, m1.rows() - 2);
  Index j = internal::random<Index>(0, m1.cols() - 2);
  Index r = internal::random<Index>(1, m1.rows() - i);
  Index c = internal::random<Index>(1, m1.cols() - j);
  Index i2 = internal::random<Index>(0, m1.rows() - 1);
  Index j2 = internal::random<Index>(0, m1.cols() - 1);

  VERIFY_IS_APPROX(m1.col(j2).adjoint() * m1.block(0, j, m1.rows(), c),
                   m1.col(j2).adjoint().eval() * m1.block(0, j, m1.rows(), c).eval());
  VERIFY_IS_APPROX(m1.block(i, 0, r, m1.cols()) * m1.row(i2).adjoint(),
                   m1.block(i, 0, r, m1.cols()).eval() * m1.row(i2).adjoint().eval());

  // test negative strides
  {
    Map<MatrixType, Unaligned, Stride<Dynamic, Dynamic> > map1(&m1(rows - 1, cols - 1), rows, cols,
                                                               Stride<Dynamic, Dynamic>(-m1.outerStride(), -1));
    Map<MatrixType, Unaligned, Stride<Dynamic, Dynamic> > map2(&m2(rows - 1, cols - 1), rows, cols,
                                                               Stride<Dynamic, Dynamic>(-m2.outerStride(), -1));
    Map<RowVectorType, Unaligned, InnerStride<-1> > mapv1(&v1(v1.size() - 1), v1.size(), InnerStride<-1>(-1));
    Map<ColVectorType, Unaligned, InnerStride<-1> > mapvc2(&vc2(vc2.size() - 1), vc2.size(), InnerStride<-1>(-1));
    VERIFY_IS_APPROX(MatrixType(map1), m1.reverse());
    VERIFY_IS_APPROX(MatrixType(map2), m2.reverse());
    VERIFY_IS_APPROX(m3.noalias() = MatrixType(map1) * MatrixType(map2).adjoint(),
                     m1.reverse() * m2.reverse().adjoint());
    VERIFY_IS_APPROX(m3.noalias() = map1 * map2.adjoint(), m1.reverse() * m2.reverse().adjoint());
    VERIFY_IS_APPROX(map1 * vc2, m1.reverse() * vc2);
    VERIFY_IS_APPROX(m1 * mapvc2, m1 * vc2.reverse().eval());
    VERIFY_IS_APPROX(map1.adjoint() * v1.transpose(), m1.adjoint().reverse() * v1.transpose());
    VERIFY_IS_APPROX(m1.adjoint() * mapv1.transpose(), m1.adjoint() * v1.reverse().transpose());

    // a destination with a negative inner stride
    ColVectorType reversedRes(cols);
    Map<ColVectorType, Unaligned, InnerStride<-1> > mapvcres(&reversedRes(cols - 1), cols, InnerStride<-1>(-1));
    mapvcres.noalias() = m1.adjoint() * v1.transpose();
    VERIFY_IS_APPROX(reversedRes, (m1.adjoint() * v1.transpose()).eval().reverse());
  }

  // regression test
  MatrixType tmp = m1 * m1.adjoint() * s1;
  VERIFY_IS_APPROX(tmp, m1 * m1.adjoint() * s1);

  // regression test for bug 1343, assignment to arrays
  Array<Scalar, Dynamic, 1> a1 = m1 * vc2;
  VERIFY_IS_APPROX(a1.matrix(), m1 * vc2);
  Array<Scalar, Dynamic, 1> a2 = s1 * (m1 * vc2);
  VERIFY_IS_APPROX(a2.matrix(), s1 * m1 * vc2);
  Array<Scalar, 1, Dynamic> a3 = v1 * m1;
  VERIFY_IS_APPROX(a3.matrix(), v1 * m1);
  Array<Scalar, Dynamic, Dynamic> a4 = m1 * m2.adjoint();
  VERIFY_IS_APPROX(a4.matrix(), m1 * m2.adjoint());
}

// Regression test for bug reported at http://forum.kde.org/viewtopic.php?f=74&t=96947
void mat_mat_scalar_scalar_product() {
  Eigen::Matrix2Xd dNdxy(2, 3);
  dNdxy << -0.5, 0.5, 0, -0.3, 0, 0.3;
  double det = 6.0, wt = 0.5;
  VERIFY_IS_APPROX(dNdxy.transpose() * dNdxy * det * wt, det * wt * dNdxy.transpose() * dNdxy);
}

template <typename MatrixType>
void zero_sized_objects(const MatrixType& m) {
  using Scalar = typename MatrixType::Scalar;
  const int PacketSize = internal::packet_traits<Scalar>::size;
  constexpr int PacketSize1 = PacketSize > 1 ? PacketSize - 1 : 1;
  Index rows = m.rows();
  Index cols = m.cols();

  {
    MatrixType res, a(rows, 0), b(0, cols);
    VERIFY_IS_APPROX((res = a * b), MatrixType::Zero(rows, cols));
    VERIFY_IS_APPROX((res = a * a.transpose()), MatrixType::Zero(rows, rows));
    VERIFY_IS_APPROX((res = b.transpose() * b), MatrixType::Zero(cols, cols));
    VERIFY_IS_APPROX((res = b.transpose() * a.transpose()), MatrixType::Zero(cols, rows));
  }

  {
    MatrixType res, a(rows, cols), b(cols, 0);
    res = a * b;
    VERIFY(res.rows() == rows && res.cols() == 0);
    b.resize(0, rows);
    res = b * a;
    VERIFY(res.rows() == 0 && res.cols() == cols);
  }

  {
    Matrix<Scalar, PacketSize, 0> a;
    Matrix<Scalar, 0, 1> b;
    Matrix<Scalar, PacketSize, 1> res;
    VERIFY_IS_APPROX((res = a * b), MatrixType::Zero(PacketSize, 1));
    VERIFY_IS_APPROX((res = a.lazyProduct(b)), MatrixType::Zero(PacketSize, 1));
  }

  {
    Matrix<Scalar, PacketSize1, 0> a;
    Matrix<Scalar, 0, 1> b;
    Matrix<Scalar, PacketSize1, 1> res;
    VERIFY_IS_APPROX((res = a * b), MatrixType::Zero(PacketSize1, 1));
    VERIFY_IS_APPROX((res = a.lazyProduct(b)), MatrixType::Zero(PacketSize1, 1));
  }

  {
    Matrix<Scalar, PacketSize, Dynamic> a(PacketSize, 0);
    Matrix<Scalar, Dynamic, 1> b(0, 1);
    Matrix<Scalar, PacketSize, 1> res;
    VERIFY_IS_APPROX((res = a * b), MatrixType::Zero(PacketSize, 1));
    VERIFY_IS_APPROX((res = a.lazyProduct(b)), MatrixType::Zero(PacketSize, 1));
  }

  {
    Matrix<Scalar, PacketSize1, Dynamic> a(PacketSize1, 0);
    Matrix<Scalar, Dynamic, 1> b(0, 1);
    Matrix<Scalar, PacketSize1, 1> res;
    VERIFY_IS_APPROX((res = a * b), MatrixType::Zero(PacketSize1, 1));
    VERIFY_IS_APPROX((res = a.lazyProduct(b)), MatrixType::Zero(PacketSize1, 1));
  }
}

template <int>
void bug_127() {
  // Bug 127
  //
  // a product of the form lhs*rhs with
  //
  // lhs:
  // rows = 1, cols = 4
  // RowsAtCompileTime = 1, ColsAtCompileTime = -1
  // MaxRowsAtCompileTime = 1, MaxColsAtCompileTime = 5
  //
  // rhs:
  // rows = 4, cols = 0
  // RowsAtCompileTime = -1, ColsAtCompileTime = -1
  // MaxRowsAtCompileTime = 5, MaxColsAtCompileTime = 1
  //
  // was failing on a runtime assertion, because it had been mis-compiled as a dot product because Product.h was using
  // the max-sizes to detect size 1 indicating vectors, and that didn't account for 0-sized object with max-size 1.

  Matrix<float, 1, Dynamic, RowMajor, 1, 5> a(1, 4);
  Matrix<float, Dynamic, Dynamic, ColMajor, 5, 1> b(4, 0);
  a* b;
}

template <int>
void bug_817() {
  ArrayXXf B = ArrayXXf::Random(10, 10), C;
  VectorXf x = VectorXf::Random(10);
  C = (x.transpose() * B.matrix());
  B = (x.transpose() * B.matrix());
  VERIFY_IS_APPROX(B, C);
}

template <int>
void triangular_product_assignment_size_mismatch() {
  MatrixXd m1 = MatrixXd::Random(5, 5);
  MatrixXd m2(6, 6);
  VERIFY_RAISES_ASSERT(m2.triangularView<Lower>() = m1 * m1);
}

template <int>
void unaligned_objects() {
  // Regression test for the bug reported here:
  // http://forum.kde.org/viewtopic.php?f=74&t=107541
  // Recall the matrix*vector kernel avoid unaligned loads by loading two packets and then reassemble then.
  // There was a mistake in the computation of the valid range for fully unaligned objects: in some rare cases,
  // memory was read outside the allocated matrix memory. Though the values were not used, this might raise segfault.
  for (int m = 450; m < 460; ++m) {
    for (int n = 8; n < 12; ++n) {
      MatrixXf M(m, n);
      VectorXf v1(n), r1(500);
      RowVectorXf v2(m), r2(16);

      M.setRandom();
      v1.setRandom();
      v2.setRandom();
      for (int o = 0; o < 4; ++o) {
        r1.segment(o, m).noalias() = M * v1;
        VERIFY_IS_APPROX(r1.segment(o, m), M * MatrixXf(v1));
        r2.segment(o, n).noalias() = v2 * M;
        VERIFY_IS_APPROX(r2.segment(o, n), MatrixXf(v2) * M);
      }
    }
  }
}

template <typename T>
EIGEN_DONT_INLINE Index test_compute_block_size(Index m, Index n, Index k) {
  Index mc(m), nc(n), kc(k);
  internal::computeProductBlockingSizes<T, T>(kc, mc, nc);
  return kc + mc + nc;
}

template <typename T>
Index compute_block_size() {
  Index ret = 0;
  // Zero-sized inputs: verify they compile and don't crash.
  ret += test_compute_block_size<T>(0, 1, 1);
  ret += test_compute_block_size<T>(1, 0, 1);
  ret += test_compute_block_size<T>(1, 1, 0);
  ret += test_compute_block_size<T>(0, 0, 1);
  ret += test_compute_block_size<T>(0, 1, 0);
  ret += test_compute_block_size<T>(1, 0, 0);
  ret += test_compute_block_size<T>(0, 0, 0);

  // Sanity checks: blocking sizes must be positive and not exceed the original.
  {
    Index m = 200, n = 200, k = 200;
    Index mc = m, nc = n, kc = k;
    internal::computeProductBlockingSizes<T, T>(kc, mc, nc);
    VERIFY(kc > 0 && kc <= k);
    VERIFY(mc > 0 && mc <= m);
    VERIFY(nc > 0 && nc <= n);
  }
  // With EIGEN_DEBUG_SMALL_PRODUCT_BLOCKS (l1=9KB, l2=32KB, l3=512KB),
  // large sizes must be actually blocked (not returned as-is).
  {
    Index m = 500, n = 500, k = 500;
    Index mc = m, nc = n, kc = k;
    internal::computeProductBlockingSizes<T, T>(kc, mc, nc);
    VERIFY(kc < k);
  }

  return ret;
}

// Verify correctness of GEMM at sizes that require multiple blocking passes
// under EIGEN_DEBUG_SMALL_PRODUCT_BLOCKS (l1=9KB, l2=32KB, l3=512KB).
// The blocking early-return threshold is max(k,m,n) < 48, so sizes >= 48
// trigger actual multi-pass blocking with these tiny cache sizes.
// Verifies GEMM against column-by-column GEMV (a different code path).
template <int>
void test_small_block_correctness() {
  const int sizes[] = {48, 64, 96, 128, 200};
  for (int si = 0; si < 5; ++si) {
    int n = sizes[si];
    MatrixXd A = MatrixXd::Random(n, n);
    MatrixXd B = MatrixXd::Random(n, n);
    MatrixXd C(n, n);
    C.noalias() = A * B;
    MatrixXd Cref(n, n);
    for (int j = 0; j < n; ++j) Cref.col(j) = A * B.col(j);
    VERIFY_IS_APPROX(C, Cref);
  }
  // Non-square: exercise different blocking in m, n, k dimensions.
  {
    MatrixXd A = MatrixXd::Random(200, 64);
    MatrixXd B = MatrixXd::Random(64, 300);
    MatrixXd C(200, 300);
    C.noalias() = A * B;
    MatrixXd Cref(200, 300);
    for (int j = 0; j < 300; ++j) Cref.col(j) = A * B.col(j);
    VERIFY_IS_APPROX(C, Cref);
  }
}

template <typename>
void aliasing_with_resize() {
  Index m = internal::random<Index>(10, 50);
  Index n = internal::random<Index>(10, 50);
  MatrixXd A, B, C(m, n), D(m, m);
  VectorXd a, b, c(n);
  C.setRandom();
  D.setRandom();
  c.setRandom();
  double s = internal::random<double>(1, 10);

  A = C;
  B = A * A.transpose();
  A = A * A.transpose();
  VERIFY_IS_APPROX(A, B);

  A = C;
  B = (A * A.transpose()) / s;
  A = (A * A.transpose()) / s;
  VERIFY_IS_APPROX(A, B);

  A = C;
  B = (A * A.transpose()) + D;
  A = (A * A.transpose()) + D;
  VERIFY_IS_APPROX(A, B);

  A = C;
  B = D + (A * A.transpose());
  A = D + (A * A.transpose());
  VERIFY_IS_APPROX(A, B);

  A = C;
  B = s * (A * A.transpose());
  A = s * (A * A.transpose());
  VERIFY_IS_APPROX(A, B);

  A = C;
  a = c;
  b = (A * a) / s;
  a = (A * a) / s;
  VERIFY_IS_APPROX(a, b);
}

// Constructing an Array passes it to the product kernel itself, which must not add a Matrix expression to it (#3227).
template <int>
void array_destination_products() {
  const Index n = 30;  // above the coefficient-based product threshold
  const MatrixXd a = MatrixXd::Random(n, n), b = MatrixXd::Random(n, n);
  const VectorXd v = VectorXd::Random(n);
  const ArrayXXd gemm = (a + a) * b;
  VERIFY_IS_APPROX(gemm.matrix(), (2.0 * a) * b);
  // a + a has no direct access, which selects the column-by-column GEMV fallback.
  const ArrayXd gemv = (a + a) * v;
  VERIFY_IS_APPROX(gemv.matrix(), (2.0 * a) * v);
  // A factor inside a unit triangle takes the diagonal correction, which also accumulates into the Array.
  const MatrixXd upper = (2.0 * a).triangularView<UnitUpper>(), lower = (2.0 * a).triangularView<UnitLower>();
  const ArrayXXd trmm = (2.0 * a).triangularView<UnitUpper>() * b;
  VERIFY_IS_APPROX(trmm.matrix(), upper * b);
  const ArrayXXd trmmRight = b * (2.0 * a).triangularView<UnitLower>();
  VERIFY_IS_APPROX(trmmRight.matrix(), b * lower);
  const ArrayXd trmv = (2.0 * a).triangularView<UnitUpper>() * v;
  VERIFY_IS_APPROX(trmv.matrix(), upper * v);
}

template <int>
void bug_1308() {
  int n = 10;
  MatrixXd r(n, n);
  VectorXd v = VectorXd::Random(n);
  r = v * RowVectorXd::Ones(n);
  VERIFY_IS_APPROX(r, v.rowwise().replicate(n));
  r = VectorXd::Ones(n) * v.transpose();
  VERIFY_IS_APPROX(r, v.rowwise().replicate(n).transpose());

  Matrix4d ones44 = Matrix4d::Ones();
  Matrix4d m44 = Matrix4d::Ones() * Matrix4d::Ones();
  VERIFY_IS_APPROX(m44, Matrix4d::Constant(4));
  VERIFY_IS_APPROX(m44.noalias() = ones44 * Matrix4d::Ones(), Matrix4d::Constant(4));
  VERIFY_IS_APPROX(m44.noalias() = ones44.transpose() * Matrix4d::Ones(), Matrix4d::Constant(4));
  VERIFY_IS_APPROX(m44.noalias() = Matrix4d::Ones() * ones44, Matrix4d::Constant(4));
  VERIFY_IS_APPROX(m44.noalias() = Matrix4d::Ones() * ones44.transpose(), Matrix4d::Constant(4));

  using RMatrix4d = Matrix<double, 4, 4, RowMajor>;
  RMatrix4d r44 = Matrix4d::Ones() * Matrix4d::Ones();
  VERIFY_IS_APPROX(r44, Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias() = ones44 * Matrix4d::Ones(), Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias() = ones44.transpose() * Matrix4d::Ones(), Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias() = Matrix4d::Ones() * ones44, Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias() = Matrix4d::Ones() * ones44.transpose(), Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias() = ones44 * RMatrix4d::Ones(), Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias() = ones44.transpose() * RMatrix4d::Ones(), Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias() = RMatrix4d::Ones() * ones44, Matrix4d::Constant(4));
  VERIFY_IS_APPROX(r44.noalias() = RMatrix4d::Ones() * ones44.transpose(), Matrix4d::Constant(4));

  //   RowVector4d r4;
  m44.setOnes();
  r44.setZero();
  VERIFY_IS_APPROX(r44.noalias() += m44.row(0).transpose() * RowVector4d::Ones(), ones44);
  r44.setZero();
  VERIFY_IS_APPROX(r44.noalias() += m44.col(0) * RowVector4d::Ones(), ones44);
  r44.setZero();
  VERIFY_IS_APPROX(r44.noalias() += Vector4d::Ones() * m44.row(0), ones44);
  r44.setZero();
  VERIFY_IS_APPROX(r44.noalias() += Vector4d::Ones() * m44.col(0).transpose(), ones44);
}

// Regression test for issue #3059: GEBP asm register constraints fail
// for custom (non-vectorizable) scalar types. Type T has a non-trivial
// destructor (making sizeof(T) > sizeof(double)), while type U is a
// simple wrapper. Both must compile and produce correct products.
namespace issue_3059 {

class Ptr {
 public:
  ~Ptr() {}
  double* m_ptr = nullptr;
};

class T {
 public:
  T() = default;
  T(double v) : m_value(v) {}

  friend T operator*(const T& a, const T& b) { return T(a.m_value * b.m_value); }
  T& operator*=(const T& o) {
    m_value *= o.m_value;
    return *this;
  }
  friend T operator/(const T& a, const T& b) { return T(a.m_value / b.m_value); }
  T& operator/=(const T& o) {
    m_value /= o.m_value;
    return *this;
  }
  friend T operator+(const T& a, const T& b) { return T(a.m_value + b.m_value); }
  T& operator+=(const T& o) {
    m_value += o.m_value;
    return *this;
  }
  friend T operator-(const T& a, const T& b) { return T(a.m_value - b.m_value); }
  T& operator-=(const T& o) {
    m_value -= o.m_value;
    return *this;
  }
  friend T operator-(const T& a) { return T(-a.m_value); }

  bool operator==(const T& o) const { return m_value == o.m_value; }
  bool operator<(const T& o) const { return m_value < o.m_value; }
  bool operator<=(const T& o) const { return m_value <= o.m_value; }
  bool operator>(const T& o) const { return m_value > o.m_value; }
  bool operator>=(const T& o) const { return m_value >= o.m_value; }
  bool operator!=(const T& o) const { return m_value != o.m_value; }

  double value() const { return m_value; }

 private:
  double m_value = 0.0;
  Ptr m_ptr;  // Makes sizeof(T) > sizeof(double)
};

T sqrt(const T& x) { return T(std::sqrt(x.value())); }
T abs(const T& x) { return T(std::abs(x.value())); }
T abs2(const T& x) { return T(x.value() * x.value()); }

class U {
 public:
  U() = default;
  U(double v) : m_value(v) {}

  friend U operator*(const U& a, const U& b) { return U(a.m_value * b.m_value); }
  U& operator*=(const U& o) {
    m_value *= o.m_value;
    return *this;
  }
  friend U operator/(const U& a, const U& b) { return U(a.m_value / b.m_value); }
  U& operator/=(const U& o) {
    m_value /= o.m_value;
    return *this;
  }
  friend U operator+(const U& a, const U& b) { return U(a.m_value + b.m_value); }
  U& operator+=(const U& o) {
    m_value += o.m_value;
    return *this;
  }
  friend U operator-(const U& a, const U& b) { return U(a.m_value - b.m_value); }
  U& operator-=(const U& o) {
    m_value -= o.m_value;
    return *this;
  }
  friend U operator-(const U& a) { return U(-a.m_value); }

  bool operator==(const U& o) const { return m_value == o.m_value; }
  bool operator<(const U& o) const { return m_value < o.m_value; }
  bool operator<=(const U& o) const { return m_value <= o.m_value; }
  bool operator>(const U& o) const { return m_value > o.m_value; }
  bool operator>=(const U& o) const { return m_value >= o.m_value; }
  bool operator!=(const U& o) const { return m_value != o.m_value; }

  double value() const { return m_value; }

 private:
  double m_value = 0.0;
};

U sqrt(const U& x) { return U(std::sqrt(x.value())); }
U abs(const U& x) { return U(std::abs(x.value())); }
U abs2(const U& x) { return U(x.value() * x.value()); }

}  // namespace issue_3059

namespace Eigen {

template <>
struct NumTraits<issue_3059::T> : NumTraits<double> {
  using Real = issue_3059::T;
  using NonInteger = issue_3059::T;
  using Nested = issue_3059::T;
  enum { IsComplex = 0, RequireInitialization = 1 };
};

template <>
struct NumTraits<issue_3059::U> : NumTraits<double> {
  using Real = issue_3059::U;
  using NonInteger = issue_3059::U;
  using Nested = issue_3059::U;
  enum { IsComplex = 0, RequireInitialization = 0 };
};

}  // namespace Eigen

template <int>
void product_custom_scalar_types() {
  using namespace issue_3059;
  // Type T: has non-trivial destructor, sizeof(T) > sizeof(double)
  {
    Matrix<T, Dynamic, Dynamic> A(4, 4), B(4, 4), C(4, 4);
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) {
        A(i, j) = T(static_cast<double>(i + 1));
        B(i, j) = T(static_cast<double>(j + 1));
      }
    C.noalias() = A * B;
    // A*B: C(i,j) = sum_k (i+1)*(k+1) * ... no, A(i,k)=(i+1), B(k,j)=(j+1)
    // so C(i,j) = sum_k (i+1)*(j+1) = 4*(i+1)*(j+1)
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) VERIFY(C(i, j) == T(4.0 * (i + 1) * (j + 1)));
  }
  // Type U: simple wrapper, sizeof(U) == sizeof(double)
  {
    Matrix<U, Dynamic, Dynamic> A(4, 4), B(4, 4), C(4, 4);
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) {
        A(i, j) = U(static_cast<double>(i + 1));
        B(i, j) = U(static_cast<double>(j + 1));
      }
    C.noalias() = A * B;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) VERIFY(C(i, j) == U(4.0 * (i + 1) * (j + 1)));
  }
  // Larger matrices to exercise GEBP blocking.
  {
    const int n = 33;
    Matrix<U, Dynamic, Dynamic> A(n, n), B(n, n), C(n, n);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        A(i, j) = U(static_cast<double>((i * 7 + j * 3) % 13));
        B(i, j) = U(static_cast<double>((i * 5 + j * 11) % 17));
      }
    C.noalias() = A * B;
    // Verify against explicit triple loop.
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        double sum = 0;
        for (int k = 0; k < n; ++k) sum += A(i, k).value() * B(k, j).value();
        VERIFY(C(i, j) == U(sum));
      }
  }
}

// Test complex GEMV with all conjugation combinations at sizes that
// exercise full, half, and quarter packet code paths.
// The GEMV kernels in GeneralMatrixVector.h use conj_helper at three
// packet levels. The existing product_extra tests cover conjugation
// but only at random sizes, never systematically at packet boundaries.
template <int>
void gemv_complex_conjugate() {
  using Scf = std::complex<float>;
  using Scd = std::complex<double>;
  const Index PS_f = internal::packet_traits<Scf>::size;
  const Index PS_d = internal::packet_traits<Scd>::size;

  // Sizes chosen to exercise packet boundaries for both float and double.
  const Index sizes[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33};

  for (int si = 0; si < 14; ++si) {
    Index m = sizes[si];
    // Test complex<float> GEMV with all conjugation combos.
    {
      using Mat = Matrix<Scf, Dynamic, Dynamic>;
      using Vec = Matrix<Scf, Dynamic, 1>;
      Mat A = Mat::Random(m, m);
      Vec v = Vec::Random(m);
      Vec res(m);

      // A * v (no conjugation)
      res.noalias() = A * v;
      VERIFY_IS_APPROX(res, (A.eval() * v.eval()).eval());

      // A.conjugate() * v
      res.noalias() = A.conjugate() * v;
      VERIFY_IS_APPROX(res, (A.conjugate().eval() * v.eval()).eval());

      // A * v.conjugate()
      res.noalias() = A * v.conjugate();
      VERIFY_IS_APPROX(res, (A.eval() * v.conjugate().eval()).eval());

      // A.conjugate() * v.conjugate()
      res.noalias() = A.conjugate() * v.conjugate();
      VERIFY_IS_APPROX(res, (A.conjugate().eval() * v.conjugate().eval()).eval());

      // A.adjoint() * v (transpose + conjugate of lhs)
      Vec res2(m);
      res2.noalias() = A.adjoint() * v;
      VERIFY_IS_APPROX(res2, (A.adjoint().eval() * v.eval()).eval());

      // Row-major complex GEMV
      using RMat = Matrix<Scf, Dynamic, Dynamic, RowMajor>;
      RMat B = A;
      res.noalias() = B * v;
      VERIFY_IS_APPROX(res, (A.eval() * v.eval()).eval());

      res.noalias() = B.conjugate() * v;
      VERIFY_IS_APPROX(res, (A.conjugate().eval() * v.eval()).eval());
    }

    // Test complex<double> GEMV with conjugation.
    {
      using Mat = Matrix<Scd, Dynamic, Dynamic>;
      using Vec = Matrix<Scd, Dynamic, 1>;
      Mat A = Mat::Random(m, m);
      Vec v = Vec::Random(m);
      Vec res(m);

      res.noalias() = A.conjugate() * v;
      VERIFY_IS_APPROX(res, (A.conjugate().eval() * v.eval()).eval());

      res.noalias() = A * v.conjugate();
      VERIFY_IS_APPROX(res, (A.eval() * v.conjugate().eval()).eval());

      // Non-square: wide matrix × vector (exercises different cols path).
      Mat C = Mat::Random(m, m + 3);
      Vec w = Vec::Random(m + 3);
      Vec res3(m);
      res3.noalias() = C.conjugate() * w;
      VERIFY_IS_APPROX(res3, (C.conjugate().eval() * w.eval()).eval());
    }
  }
  (void)PS_f;
  (void)PS_d;
}

// y += alpha * A * x for every extent of the vectorized dimension from 1 to 12 packets + 2: the GEMV kernels split it
// into passes of up to 8 packets, a final pass of up to 9 packets, and a partial packet (masked where segments exist).
// A is a block of a NaN-padded matrix and x a segment of a NaN-padded vector, so a load past the extent poisons the
// result, and sentinels around y catch a store past it. The reference is an independent long-double loop.
template <typename Scalar, int Order>
void gemv_remainder_sweep() {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using WideScalar = std::conditional_t<NumTraits<Scalar>::IsComplex, std::complex<long double>, long double>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  const Index PS = internal::packet_traits<Scalar>::size;
  const Scalar nan = Scalar(NumTraits<RealScalar>::quiet_NaN());
  const Scalar sentinel = Scalar(RealScalar(12345));
  const Scalar alpha = internal::random<Scalar>();
  const Index pad = 2 * PS + 1;
  // Columns 130 > 128 make the col-major kernel block the columns.
  const Index others[] = {1, 2, 3, 5, 9, 17, 130};

  for (Index vec_extent = 1; vec_extent <= 12 * PS + 2; ++vec_extent) {
    for (Index other : others) {
      // The vectorized dimension is the rows for col-major and the columns for row-major.
      const Index rows = Order == ColMajor ? vec_extent : other;
      const Index cols = Order == ColMajor ? other : vec_extent;
      Mat big = Mat::Constant(rows + pad, cols + pad, nan);
      big.topLeftCorner(rows, cols).setRandom();
      Vec xbig = Vec::Constant(cols + pad, nan);
      xbig.head(cols).setRandom();
      Vec ybig = Vec::Constant(rows + 2 * pad, sentinel);
      ybig.segment(pad, rows).setRandom();
      const Vec y0 = ybig.segment(pad, rows);

      ybig.segment(pad, rows).noalias() += alpha * big.topLeftCorner(rows, cols) * xbig.head(cols);

      for (Index i = 0; i < rows; ++i) {
        WideScalar expected(0);
        long double magnitude = 0;
        for (Index j = 0; j < cols; ++j) {
          const WideScalar term = WideScalar(big(i, j)) * WideScalar(xbig(j));
          expected += term;
          magnitude += numext::abs(term);
        }
        expected = WideScalar(y0(i)) + WideScalar(alpha) * expected;
        magnitude = numext::abs(WideScalar(y0(i))) + numext::abs(WideScalar(alpha)) * magnitude;
        // Each product and sum rounds once in Scalar, and alpha is applied with one more product and sum.
        const long double bound = 4 * (cols + 2) * (long double)NumTraits<RealScalar>::epsilon() * magnitude;
        const long double error = numext::abs(WideScalar(ybig(pad + i)) - expected);
        VERIFY((numext::isfinite)(error) && error <= bound);
      }
      VERIFY_IS_CWISE_EQUAL(ybig.head(pad), Vec::Constant(pad, sentinel));
      VERIFY_IS_CWISE_EQUAL(ybig.tail(pad), Vec::Constant(pad, sentinel));
    }
  }
}

// Locks the BLAS contract that GEMM/GEMV leave the destination unchanged when
// alpha == 0, including under non-finite inputs in A/x/B that would otherwise
// taint the result via 0 * Inf = NaN.
// A zero scalar factor must leave the destination untouched for every
// assignment form: += and -= return early, while = zeroes the destination.
// Inf/NaN operands make a kernel that ran anyway visible as a NaN.
template <typename Scalar, int Order>
void alpha_zero_skips_gemm(Index m, Index k, Index n) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Mat = Matrix<Scalar, Dynamic, Dynamic, Order>;

  const Scalar pos_zero = Scalar(0);
  const Scalar neg_zero = Scalar(-RealScalar(0));

  Mat A = Mat::Random(m, k);
  Mat B = Mat::Random(k, n);
  A(0, 0) = Scalar(NumTraits<RealScalar>::infinity());
  B(1, 1) = Scalar(NumTraits<RealScalar>::quiet_NaN());

  const Mat C_ref = Mat::Random(m, n);
  const Mat zero = Mat::Zero(m, n);

  for (const Scalar& alpha : {pos_zero, neg_zero}) {
    Mat C = C_ref;
    C.noalias() += alpha * A * B;
    VERIFY_IS_CWISE_EQUAL(C, C_ref);

    C = C_ref;
    C.noalias() -= alpha * A * B;
    VERIFY_IS_CWISE_EQUAL(C, C_ref);

    C = C_ref;
    C.noalias() = alpha * A * B;
    VERIFY_IS_CWISE_EQUAL(C, zero);
  }
}

template <typename Scalar>
void alpha_zero_skips_kernel() {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using ColMat = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using RowMat = Matrix<Scalar, Dynamic, Dynamic, RowMajor>;
  using Vec = Matrix<Scalar, Dynamic, 1>;

  const Index m = 17, k = 13, n = 11;
  const Scalar inf = Scalar(NumTraits<RealScalar>::infinity());
  const Scalar nan = Scalar(NumTraits<RealScalar>::quiet_NaN());
  const Scalar pos_zero = Scalar(0);
  const Scalar neg_zero = Scalar(-RealScalar(0));

  // GEMM, both storage orders. 17x13x11 sums to 41 and so takes the GEMM path
  // on a default build; 5x5x5 sums to 15 and reaches the coeff-based path,
  // which has its own zero-factor exit. Without the small shape the latter is
  // only covered where the threshold is raised (SME), i.e. by one nightly job.
  alpha_zero_skips_gemm<Scalar, ColMajor>(m, k, n);
  alpha_zero_skips_gemm<Scalar, ColMajor>(5, 5, 5);
  alpha_zero_skips_gemm<Scalar, RowMajor>(m, k, n);
  alpha_zero_skips_gemm<Scalar, RowMajor>(5, 5, 5);

  // GEMV col-major.
  {
    ColMat A = ColMat::Random(m, k);
    Vec x = Vec::Random(k);
    A(0, 0) = inf;
    x(1) = nan;

    Vec y = Vec::Random(m);
    const Vec y_ref = y;

    y.noalias() += pos_zero * (A * x);
    VERIFY_IS_CWISE_EQUAL(y, y_ref);

    y.noalias() += neg_zero * (A * x);
    VERIFY_IS_CWISE_EQUAL(y, y_ref);
  }

  // GEMV row-major.
  {
    RowMat A = RowMat::Random(m, k);
    Vec x = Vec::Random(k);
    A(0, 0) = inf;
    x(1) = nan;

    Vec y = Vec::Random(m);
    const Vec y_ref = y;

    y.noalias() += pos_zero * (A * x);
    VERIFY_IS_CWISE_EQUAL(y, y_ref);

    y.noalias() += neg_zero * (A * x);
    VERIFY_IS_CWISE_EQUAL(y, y_ref);
  }
}

template <typename RealScalar, int Options>
void complex_gemm_scalar_accumulation() {
  using Scalar = std::complex<RealScalar>;
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Options>;
  for (Index n : {5, 8, 9}) {
    Matrix<Scalar, Dynamic, Dynamic> lhs(n, n);
    Matrix<Scalar, Dynamic, Dynamic, RowMajor> rhs(n, n);
    for (Index j = 0; j < n; ++j) {
      for (Index i = 0; i < n; ++i) {
        lhs(i, j) = Scalar(RealScalar((i + j) % 5 - 2), RealScalar((2 * i + j) % 3 - 1));
        rhs(i, j) = Scalar(RealScalar((i + 2 * j) % 3 - 1), RealScalar((i + j) % 5 - 2));
      }
    }
    const auto check = [&](const auto& a, const auto& b) {
      MatrixType expected = MatrixType::Zero(n, n);
      for (Index j = 0; j < n; ++j)
        for (Index i = 0; i < n; ++i)
          for (Index k = 0; k < n; ++k) expected(i, j) += a.coeff(i, k) * b.coeff(k, j);
      MatrixType actual(n, n);
      actual.noalias() = a * b;
      VERIFY_IS_EQUAL(actual, expected);
      // Integer components keep every operation exact, including a non-real alpha.
      const Scalar alpha(2, -1);
      actual.setOnes();
      actual.noalias() += alpha * (a * b);
      VERIFY_IS_EQUAL(actual, MatrixType(MatrixType::Ones(n, n) + alpha * expected));
      VERIFY_IS_EQUAL((expected - a * b).cwiseAbs().maxCoeff(), RealScalar(0));
    };
    check(lhs, rhs);
    check(lhs.conjugate(), rhs);
    check(lhs, rhs.conjugate());
    check(lhs.conjugate(), rhs.conjugate());
  }

  const MatrixType u = MatrixType::Identity(8, 8) * Scalar(0, 1);
  const MatrixType t = MatrixType::Constant(8, 8, Scalar(0, 1));
  VERIFY_IS_EQUAL((t - u * t * u.adjoint()).cwiseAbs().maxCoeff(), RealScalar(0));
}

#if defined(EIGEN_TEST_PART_14) || defined(EIGEN_TEST_PART_ALL)
template <int Order>
void complex_outer_product_overflow() {
  using Scalar = std::complex<double>;
  using Lhs = Matrix<Scalar, Dynamic, 1>;
  using Rhs = Matrix<Scalar, 1, Dynamic>;
  using Dst = Matrix<Scalar, Dynamic, Dynamic, Order>;
  STATIC_CHECK((internal::product_type<Lhs, Rhs>::value == OuterProduct));
  for (Index n : {4, 17}) {
    const Lhs lhs = Lhs::Constant(n, Scalar(1e308, 1));
    const Rhs rhs = Rhs::Constant(n, Scalar(10, 0));
    Dst dst(n, n);
    auto check = [&](double sign, double imaginary) {
      for (Index j = 0; j < n; ++j) {
        for (Index i = 0; i < n; ++i) {
          VERIFY((numext::isinf)(dst(i, j).real()));
          VERIFY(dst(i, j).real() * sign > 0);
          VERIFY_IS_EQUAL(dst(i, j).imag(), imaginary);
        }
      }
    };
    // A redundant multiplication by complex one would introduce 0*infinity into the imaginary part.
    dst.noalias() = lhs * rhs;
    check(1, 10);
    dst.setConstant(Scalar(2, 3));
    dst.noalias() += lhs * rhs;
    check(1, 13);
    dst.setConstant(Scalar(2, 3));
    dst.noalias() -= lhs * rhs;
    check(-1, -7);
  }
}

template <int Order>
void single_term_nullary_product() {
  const Vector4d lhs = Vector4d::LinSpaced(1, 4);
  const RowVector4d rhs = RowVector4d::LinSpaced(5, 8);
  auto check = [&](const auto& left, const auto& right) {
    const auto product = left.lazyProduct(right);
    Matrix<double, 4, 4, Order> dst = product;
    for (Index j = 0; j < 4; ++j) {
      for (Index i = 0; i < 4; ++i) {
        const double expected = left.coeff(i) * right.coeff(j);
        VERIFY_IS_EQUAL(product.coeff(i, j), expected);
        VERIFY_IS_EQUAL(dst(i, j), expected);
      }
    }
  };
  check(Vector4d::Ones(), rhs);
  check(lhs, RowVector4d::Ones());
  check(Vector4d::Ones(), RowVector4d::Ones());
}

namespace outer_product_count {
struct Scalar {
  double value;
  static int multiplications;
  Scalar(double v = 0) : value(v) {}
  Scalar operator*(const Scalar& other) const {
    ++multiplications;
    return Scalar(value * other.value);
  }
  Scalar operator+(const Scalar& other) const { return Scalar(value + other.value); }
  Scalar operator-(const Scalar& other) const { return Scalar(value - other.value); }
  Scalar& operator+=(const Scalar& other) { return *this = *this + other; }
  Scalar& operator-=(const Scalar& other) { return *this = *this - other; }
};
int Scalar::multiplications = 0;
}  // namespace outer_product_count

namespace Eigen {
template <>
struct NumTraits<outer_product_count::Scalar> : GenericNumTraits<outer_product_count::Scalar> {
  static constexpr bool RequireInitialization = true;
};
}  // namespace Eigen

template <int Order>
void small_outer_product_unscaled() {
  using Scalar = outer_product_count::Scalar;
  using Lhs = Matrix<Scalar, Dynamic, 1>;
  using Rhs = Matrix<Scalar, 1, Dynamic>;
  using Dst = Matrix<Scalar, Dynamic, Dynamic, Order>;
  STATIC_CHECK((internal::product_type<Lhs, Rhs>::value == OuterProduct));
  for (Index n : {1, 8, 16, 17}) {
    const Lhs lhs = Lhs::Constant(n, Scalar(2));
    const Rhs rhs = Rhs::Constant(n, Scalar(3));
    Dst dst(n, n);
    auto check = [&](double expected) {
      VERIFY_IS_EQUAL(Scalar::multiplications, n * n);
      for (Index j = 0; j < n; ++j)
        for (Index i = 0; i < n; ++i) VERIFY_IS_EQUAL(dst(i, j).value, expected);
    };
    Scalar::multiplications = 0;
    dst.noalias() = lhs * rhs;
    check(6);
    Scalar::multiplications = 0;
    dst.noalias() += lhs * rhs;
    check(12);
    Scalar::multiplications = 0;
    dst.noalias() -= lhs * rhs;
    check(6);
    internal::generic_product_impl<Lhs, Rhs>::scaleAndAddTo(dst, lhs, rhs, Scalar(2));
    for (Index j = 0; j < n; ++j)
      for (Index i = 0; i < n; ++i) VERIFY_IS_EQUAL(dst(i, j).value, 18.0);
  }
}

// s * M::Ones() is Constant * Constant; the GEMM and GEMV paths must unwrap its factor (#3226).
template <typename Scalar, int Order>
void scaled_constant_operands() {
  using M = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using V = Matrix<Scalar, Dynamic, 1>;
  using RV = Matrix<Scalar, 1, Dynamic>;
  const Scalar s = internal::random<Scalar>(), t = internal::random<Scalar>();
  // 3 takes the coefficient-based product, 30 the GEMM kernel.
  for (Index n : {3, 30}) {
    const M b = M::Random(n, n);
    const M sc = M::Constant(n, n, s), tc = M::Constant(n, n, t);
    M r(n, n);
    VERIFY_IS_APPROX(r = s * M::Ones(n, n) * b, sc * b);
    VERIFY_IS_APPROX(r = b * (M::Ones(n, n) * t), b * tc);
    VERIFY_IS_APPROX(r = (s * M::Ones(n, n)) * (M::Ones(n, n) * t), sc * tc);
    VERIFY_IS_APPROX(r.noalias() = (s * M::Ones(n, n)).transpose() * b, sc.transpose() * b);
    VERIFY_IS_APPROX(r.noalias() = b * (s * M::Ones(n, n)).adjoint(), b * sc.adjoint());
    r = b;
    r.noalias() += s * M::Ones(n, n) * b;
    VERIFY_IS_APPROX(r, b + sc * b);
    r.noalias() -= s * M::Ones(n, n) * b;
    VERIFY_IS_APPROX(r, b);
    V y(n);
    VERIFY_IS_APPROX(y = b * (s * V::Ones(n)), b * V::Constant(n, s));
    RV z(n);
    VERIFY_IS_APPROX(z = (s * RV::Ones(n)) * b, RV::Constant(n, s) * b);
  }
  using Fixed = Matrix<Scalar, 30, 30, Order>;
  const Fixed f = Fixed::Random();
  Fixed g = s * Fixed::Ones() * f;
  VERIFY_IS_APPROX(g, Fixed::Constant(s) * f);
}

#endif

// Products that must not allocate a workspace (fixed-size GEMM) or multiply more than needed (ordered GEMV).
namespace unannotated_product {
static Index multiplications = 0;
struct Scalar {
  double value;
  Scalar(double v = 0) : value(v) {}
  Scalar operator+(const Scalar& other) const { return value + other.value; }
  Scalar operator-(const Scalar& other) const { return value - other.value; }
  Scalar operator-() const { return -value; }
  Scalar operator*(const Scalar& other) const {
    ++multiplications;
    return value * other.value;
  }
  Scalar& operator+=(const Scalar& other) { return *this = *this + other; }
  Scalar& operator-=(const Scalar& other) { return *this = *this - other; }
  Scalar& operator*=(const Scalar& other) { return *this = *this * other; }
  bool operator==(const Scalar& other) const { return value == other.value; }
  bool operator!=(const Scalar& other) const { return !(*this == other); }
};
}  // namespace unannotated_product

namespace Eigen {
template <>
struct NumTraits<unannotated_product::Scalar> : GenericNumTraits<unannotated_product::Scalar> {
  static constexpr bool RequireInitialization = true;
};
}  // namespace Eigen

template <typename Scalar, int Order, bool Bounded = false>
void fixed_gemm_workspace() {
  using MatrixType = Matrix<Scalar, Bounded ? Dynamic : 16, Bounded ? Dynamic : 16, Order, 16, 16>;
  STATIC_CHECK((!internal::functor_is_commutative<internal::scalar_product_op<Scalar>>::value));
  STATIC_CHECK((internal::product_type<MatrixType, MatrixType>::value == GemmProduct));
  const MatrixType a = MatrixType::Constant(16, 16, Scalar(2)), b = MatrixType::Constant(16, 16, Scalar(3));
  MatrixType result(16, 16);
  const Scalar scale(2);
  const auto check = [&](double expected) {
    for (Index i = 0; i < result.size(); ++i) VERIFY(result(i) == Scalar(expected));
  };
  internal::set_is_malloc_allowed(false);
  result.noalias() = a * b;
  check(96);
  result.noalias() += a * b;
  check(192);
  result.noalias() -= a * b;
  check(96);
  result.noalias() = a * (b * scale);
  check(192);
  result.noalias() = (scale * a) * b;
  check(192);
  result.noalias() = scale * (a * b);
  check(192);
  using Vector = Matrix<Scalar, Bounded ? Dynamic : 16, 1, ColMajor, 16, 1>;
  const Vector x = Vector::Constant(16, Scalar(3));
  Vector y(16);
  y.noalias() = a * x;
  for (Index i = 0; i < y.size(); ++i) VERIFY(y(i) == Scalar(96));
  y.noalias() -= a * x;
  for (Index i = 0; i < y.size(); ++i) VERIFY(y(i) == Scalar(0));
  internal::set_is_malloc_allowed(true);
}

template <int Order>
void quaternion_unit_scale() {
  using Q = noncommutative_scalar::Quaternion;
  using MatrixType = Matrix<Q, Dynamic, Dynamic, Order>;
  MatrixType a = MatrixType::Zero(3, 3), b = MatrixType::Zero(3, 3);
  a(0, 0) = Q(NumTraits<double>::infinity());
  b(0, 0) = Q(1, 1, 1, 1);
  const MatrixType result = a * b;
  const Q value = result(0, 0);
  for (double component : {value.r, value.i, value.j, value.k}) {
    VERIFY((numext::isinf)(component));
    VERIFY(component > 0);
  }
}

template <int Order, bool Left>
void ordered_gemv_scale_cost() {
  using Scalar = unannotated_product::Scalar;
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using Vector = Matrix<Scalar, Dynamic, 1>;
  using Result = Matrix<Scalar, Left ? 1 : Dynamic, Left ? Dynamic : 1, Left ? RowMajor : ColMajor>;
  const Index size = 17, depth = 9;
  const MatrixType matrix = MatrixType::Constant(Left ? depth : size, Left ? size : depth, Scalar(2));
  const Vector vector = Vector::Constant(depth, Scalar(3));
  Result result(Left ? 1 : size, Left ? size : 1);
  internal::set_is_malloc_allowed(false);
  unannotated_product::multiplications = 0;
  if (Left)
    result.noalias() = Scalar(0.5) * (vector.transpose() * matrix);
  else
    result.noalias() = Scalar(0.5) * (matrix * vector);
  VERIFY_IS_EQUAL(unannotated_product::multiplications, size * depth + size);
  for (Index i = 0; i < size; ++i) VERIFY(result(i) == Scalar(27));
  internal::set_is_malloc_allowed(true);
}

EIGEN_DECLARE_TEST(product_extra) {
  CALL_SUBTEST_20((ordered_gemv_scale_cost<ColMajor, false>()));
  CALL_SUBTEST_20((ordered_gemv_scale_cost<RowMajor, false>()));
  CALL_SUBTEST_20((ordered_gemv_scale_cost<ColMajor, true>()));
  CALL_SUBTEST_20((ordered_gemv_scale_cost<RowMajor, true>()));
  CALL_SUBTEST_20((fixed_gemm_workspace<unannotated_product::Scalar, ColMajor>()));
  CALL_SUBTEST_20((fixed_gemm_workspace<unannotated_product::Scalar, RowMajor>()));
  CALL_SUBTEST_20((fixed_gemm_workspace<noncommutative_scalar::Quaternion, ColMajor>()));
  CALL_SUBTEST_20((fixed_gemm_workspace<noncommutative_scalar::Quaternion, RowMajor>()));
  CALL_SUBTEST_20((fixed_gemm_workspace<unannotated_product::Scalar, ColMajor, true>()));
  CALL_SUBTEST_20((fixed_gemm_workspace<unannotated_product::Scalar, RowMajor, true>()));
  CALL_SUBTEST_20((fixed_gemm_workspace<noncommutative_scalar::Quaternion, ColMajor, true>()));
  CALL_SUBTEST_20((fixed_gemm_workspace<noncommutative_scalar::Quaternion, RowMajor, true>()));
  CALL_SUBTEST_20(quaternion_unit_scale<ColMajor>());
  CALL_SUBTEST_20(quaternion_unit_scale<RowMajor>());
  CALL_SUBTEST_18((test_dense_quaternion_factor_order<ColMajor, ColMajor, ColMajor>()));
  CALL_SUBTEST_18((test_dense_quaternion_factor_order<ColMajor, RowMajor, ColMajor>()));
  CALL_SUBTEST_18((test_dense_quaternion_factor_order<RowMajor, ColMajor, ColMajor>()));
  CALL_SUBTEST_18((test_dense_quaternion_factor_order<RowMajor, RowMajor, ColMajor>()));
  CALL_SUBTEST_19((test_dense_quaternion_factor_order<ColMajor, ColMajor, RowMajor>()));
  CALL_SUBTEST_19((test_dense_quaternion_factor_order<ColMajor, RowMajor, RowMajor>()));
  CALL_SUBTEST_19((test_dense_quaternion_factor_order<RowMajor, ColMajor, RowMajor>()));
  CALL_SUBTEST_19((test_dense_quaternion_factor_order<RowMajor, RowMajor, RowMajor>()));
  CALL_SUBTEST_17((test_dense_quaternion_row_order<ColMajor>()));
  CALL_SUBTEST_17((test_dense_quaternion_row_order<RowMajor>()));
  CALL_SUBTEST_17((test_dense_quaternion_outer_order<ColMajor>()));
  CALL_SUBTEST_17((test_dense_quaternion_outer_order<RowMajor>()));
  CALL_SUBTEST_17((test_dense_quaternion_constant_zero<ColMajor>()));
  CALL_SUBTEST_17((test_dense_quaternion_constant_zero<RowMajor>()));
  CALL_SUBTEST_17((test_dense_quaternion_gemv_scales<ColMajor, false>()));
  CALL_SUBTEST_17((test_dense_quaternion_gemv_scales<RowMajor, false>()));
  CALL_SUBTEST_17((test_dense_quaternion_gemv_scales<ColMajor, true>()));
  CALL_SUBTEST_17((test_dense_quaternion_gemv_scales<RowMajor, true>()));
  CALL_SUBTEST_14(complex_outer_product_overflow<ColMajor>());
  CALL_SUBTEST_14(complex_outer_product_overflow<RowMajor>());
  CALL_SUBTEST_14(single_term_nullary_product<ColMajor>());
  CALL_SUBTEST_14(single_term_nullary_product<RowMajor>());
  CALL_SUBTEST_14(small_outer_product_unscaled<ColMajor>());
  CALL_SUBTEST_14(small_outer_product_unscaled<RowMajor>());
  CALL_SUBTEST_14((scaled_constant_operands<double, ColMajor>()));
  CALL_SUBTEST_14((scaled_constant_operands<double, RowMajor>()));
  CALL_SUBTEST_14((scaled_constant_operands<std::complex<double>, ColMajor>()));
  CALL_SUBTEST_13((complex_gemm_scalar_accumulation<float, ColMajor>()));
  CALL_SUBTEST_13((complex_gemm_scalar_accumulation<float, RowMajor>()));
  CALL_SUBTEST_13((complex_gemm_scalar_accumulation<double, ColMajor>()));
  CALL_SUBTEST_13((complex_gemm_scalar_accumulation<double, RowMajor>()));
#ifndef EIGEN_TEST_NO_LONGDOUBLE
  CALL_SUBTEST_13((complex_gemm_scalar_accumulation<long double, ColMajor>()));
  CALL_SUBTEST_13((complex_gemm_scalar_accumulation<long double, RowMajor>()));
#endif
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(product_extra(
        MatrixXf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_2(product_extra(
        MatrixXd(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_2(mat_mat_scalar_scalar_product());
    CALL_SUBTEST_3(product_extra(MatrixXcf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2),
                                           internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2))));
    CALL_SUBTEST_4(product_extra(MatrixXcd(internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2),
                                           internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2))));
    CALL_SUBTEST_1(zero_sized_objects(
        MatrixXf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
  }
  CALL_SUBTEST_5(bug_127<0>());
  CALL_SUBTEST_5(triangular_product_assignment_size_mismatch<0>());
  CALL_SUBTEST_5(bug_817<0>());
  CALL_SUBTEST_5(bug_1308<0>());
  CALL_SUBTEST_5(array_destination_products<0>());
  CALL_SUBTEST_6(unaligned_objects<0>());
  CALL_SUBTEST_7(compute_block_size<float>());
  CALL_SUBTEST_7(compute_block_size<double>());
  CALL_SUBTEST_7(compute_block_size<std::complex<double> >());
  CALL_SUBTEST_8(aliasing_with_resize<void>());
  CALL_SUBTEST_9(product_custom_scalar_types<0>());
  CALL_SUBTEST_10(test_small_block_correctness<0>());

  // Complex GEMV conjugation at varied sizes (deterministic, outside g_repeat).
  CALL_SUBTEST_11(gemv_complex_conjugate<0>());

  // alpha==0 fast path: GEMM/GEMV must leave the destination unchanged.
  CALL_SUBTEST_12(alpha_zero_skips_kernel<float>());
  CALL_SUBTEST_12(alpha_zero_skips_kernel<double>());
  CALL_SUBTEST_12(alpha_zero_skips_kernel<std::complex<float> >());
  CALL_SUBTEST_12(alpha_zero_skips_kernel<std::complex<double> >());

  CALL_SUBTEST_15((gemv_remainder_sweep<float, ColMajor>()));
  CALL_SUBTEST_15((gemv_remainder_sweep<float, RowMajor>()));
  CALL_SUBTEST_15((gemv_remainder_sweep<double, ColMajor>()));
  CALL_SUBTEST_15((gemv_remainder_sweep<double, RowMajor>()));
  CALL_SUBTEST_16((gemv_remainder_sweep<std::complex<float>, ColMajor>()));
  CALL_SUBTEST_16((gemv_remainder_sweep<std::complex<float>, RowMajor>()));
  CALL_SUBTEST_16((gemv_remainder_sweep<std::complex<double>, ColMajor>()));
  CALL_SUBTEST_16((gemv_remainder_sweep<std::complex<double>, RowMajor>()));
}
