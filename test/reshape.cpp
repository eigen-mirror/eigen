// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2014 yoco <peter.xiau@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include "random_for_arithmetic.h"

using Eigen::placeholders::all;
using Eigen::placeholders::last;

template <typename T1, typename T2>
std::enable_if_t<std::is_same<T1, T2>::value, bool> is_same_eq(const T1& a, const T2& b) {
  return (a.array() == b.array()).all();
}

template <int Order, typename MatType>
void check_auto_reshape4x4(const MatType& m) {
  internal::VariableAndFixedInt<MatType::SizeAtCompileTime == Dynamic ? -1 : 1> v1(1);
  internal::VariableAndFixedInt<MatType::SizeAtCompileTime == Dynamic ? -1 : 2> v2(2);
  internal::VariableAndFixedInt<MatType::SizeAtCompileTime == Dynamic ? -1 : 4> v4(4);
  internal::VariableAndFixedInt<MatType::SizeAtCompileTime == Dynamic ? -1 : 8> v8(8);
  internal::VariableAndFixedInt<MatType::SizeAtCompileTime == Dynamic ? -1 : 16> v16(16);

  VERIFY(is_same_eq(m.template reshaped<Order>(1, AutoSize), m.template reshaped<Order>(1, 16)));
  VERIFY(is_same_eq(m.template reshaped<Order>(AutoSize, 16), m.template reshaped<Order>(1, 16)));
  VERIFY(is_same_eq(m.template reshaped<Order>(2, AutoSize), m.template reshaped<Order>(2, 8)));
  VERIFY(is_same_eq(m.template reshaped<Order>(AutoSize, 8), m.template reshaped<Order>(2, 8)));
  VERIFY(is_same_eq(m.template reshaped<Order>(4, AutoSize), m.template reshaped<Order>(4, 4)));
  VERIFY(is_same_eq(m.template reshaped<Order>(AutoSize, 4), m.template reshaped<Order>(4, 4)));
  VERIFY(is_same_eq(m.template reshaped<Order>(8, AutoSize), m.template reshaped<Order>(8, 2)));
  VERIFY(is_same_eq(m.template reshaped<Order>(AutoSize, 2), m.template reshaped<Order>(8, 2)));
  VERIFY(is_same_eq(m.template reshaped<Order>(16, AutoSize), m.template reshaped<Order>(16, 1)));
  VERIFY(is_same_eq(m.template reshaped<Order>(AutoSize, 1), m.template reshaped<Order>(16, 1)));

  VERIFY(is_same_eq(m.template reshaped<Order>(fix<1>, AutoSize), m.template reshaped<Order>(fix<1>, v16)));
  VERIFY(is_same_eq(m.template reshaped<Order>(AutoSize, fix<16>), m.template reshaped<Order>(v1, fix<16>)));
  VERIFY(is_same_eq(m.template reshaped<Order>(fix<2>, AutoSize), m.template reshaped<Order>(fix<2>, v8)));
  VERIFY(is_same_eq(m.template reshaped<Order>(AutoSize, fix<8>), m.template reshaped<Order>(v2, fix<8>)));
  VERIFY(is_same_eq(m.template reshaped<Order>(fix<4>, AutoSize), m.template reshaped<Order>(fix<4>, v4)));
  VERIFY(is_same_eq(m.template reshaped<Order>(AutoSize, fix<4>), m.template reshaped<Order>(v4, fix<4>)));
  VERIFY(is_same_eq(m.template reshaped<Order>(fix<8>, AutoSize), m.template reshaped<Order>(fix<8>, v2)));
  VERIFY(is_same_eq(m.template reshaped<Order>(AutoSize, fix<2>), m.template reshaped<Order>(v8, fix<2>)));
  VERIFY(is_same_eq(m.template reshaped<Order>(fix<16>, AutoSize), m.template reshaped<Order>(fix<16>, v1)));
  VERIFY(is_same_eq(m.template reshaped<Order>(AutoSize, fix<1>), m.template reshaped<Order>(v16, fix<1>)));
}

template <typename MatType>
void check_direct_access_reshape4x4(const MatType&, internal::FixedInt<RowMajorBit>) {}

template <typename MatType>
void check_direct_access_reshape4x4(const MatType& m, internal::FixedInt<0>) {
  VERIFY_IS_EQUAL(m.reshaped(1, 16).data(), m.data());
  VERIFY_IS_EQUAL(m.reshaped(1, 16).innerStride(), 1);

  VERIFY_IS_EQUAL(m.reshaped(2, 8).data(), m.data());
  VERIFY_IS_EQUAL(m.reshaped(2, 8).innerStride(), 1);
  VERIFY_IS_EQUAL(m.reshaped(2, 8).outerStride(), 2);
}

// just test a 4x4 matrix, enumerate all combination manually
template <typename MatType>
void reshape4x4(const MatType& m0) {
  typedef typename MatType::Scalar Scalar;
  MatType m = m0;

  internal::VariableAndFixedInt<MatType::SizeAtCompileTime == Dynamic ? -1 : 1> v1(1);
  internal::VariableAndFixedInt<MatType::SizeAtCompileTime == Dynamic ? -1 : 2> v2(2);
  internal::VariableAndFixedInt<MatType::SizeAtCompileTime == Dynamic ? -1 : 4> v4(4);
  internal::VariableAndFixedInt<MatType::SizeAtCompileTime == Dynamic ? -1 : 8> v8(8);
  internal::VariableAndFixedInt<MatType::SizeAtCompileTime == Dynamic ? -1 : 16> v16(16);

  if ((MatType::Flags & RowMajorBit) == 0) {
    typedef Map<MatrixXi> MapMat;
    // dynamic
    VERIFY_IS_EQUAL((m.reshaped(1, 16)), MapMat(m.data(), 1, 16));
    VERIFY_IS_EQUAL((m.reshaped(2, 8)), MapMat(m.data(), 2, 8));
    VERIFY_IS_EQUAL((m.reshaped(4, 4)), MapMat(m.data(), 4, 4));
    VERIFY_IS_EQUAL((m.reshaped(8, 2)), MapMat(m.data(), 8, 2));
    VERIFY_IS_EQUAL((m.reshaped(16, 1)), MapMat(m.data(), 16, 1));

    // static
    VERIFY_IS_EQUAL(m.reshaped(fix<1>, fix<16>), MapMat(m.data(), 1, 16));
    VERIFY_IS_EQUAL(m.reshaped(fix<2>, fix<8>), MapMat(m.data(), 2, 8));
    VERIFY_IS_EQUAL(m.reshaped(fix<4>, fix<4>), MapMat(m.data(), 4, 4));
    VERIFY_IS_EQUAL(m.reshaped(fix<8>, fix<2>), MapMat(m.data(), 8, 2));
    VERIFY_IS_EQUAL(m.reshaped(fix<16>, fix<1>), MapMat(m.data(), 16, 1));

    // reshape chain
    VERIFY_IS_EQUAL((m.reshaped(1, 16)
                         .reshaped(fix<2>, fix<8>)
                         .reshaped(16, 1)
                         .reshaped(fix<8>, fix<2>)
                         .reshaped(2, 8)
                         .reshaped(fix<1>, fix<16>)
                         .reshaped(4, 4)
                         .reshaped(fix<16>, fix<1>)
                         .reshaped(8, 2)
                         .reshaped(fix<4>, fix<4>)),
                    MapMat(m.data(), 4, 4));
  }

  VERIFY(is_same_eq(m.reshaped(1, AutoSize), m.reshaped(1, 16)));
  VERIFY(is_same_eq(m.reshaped(AutoSize, 16), m.reshaped(1, 16)));
  VERIFY(is_same_eq(m.reshaped(2, AutoSize), m.reshaped(2, 8)));
  VERIFY(is_same_eq(m.reshaped(AutoSize, 8), m.reshaped(2, 8)));
  VERIFY(is_same_eq(m.reshaped(4, AutoSize), m.reshaped(4, 4)));
  VERIFY(is_same_eq(m.reshaped(AutoSize, 4), m.reshaped(4, 4)));
  VERIFY(is_same_eq(m.reshaped(8, AutoSize), m.reshaped(8, 2)));
  VERIFY(is_same_eq(m.reshaped(AutoSize, 2), m.reshaped(8, 2)));
  VERIFY(is_same_eq(m.reshaped(16, AutoSize), m.reshaped(16, 1)));
  VERIFY(is_same_eq(m.reshaped(AutoSize, 1), m.reshaped(16, 1)));

  VERIFY(is_same_eq(m.reshaped(fix<1>, AutoSize), m.reshaped(fix<1>, v16)));
  VERIFY(is_same_eq(m.reshaped(AutoSize, fix<16>), m.reshaped(v1, fix<16>)));
  VERIFY(is_same_eq(m.reshaped(fix<2>, AutoSize), m.reshaped(fix<2>, v8)));
  VERIFY(is_same_eq(m.reshaped(AutoSize, fix<8>), m.reshaped(v2, fix<8>)));
  VERIFY(is_same_eq(m.reshaped(fix<4>, AutoSize), m.reshaped(fix<4>, v4)));
  VERIFY(is_same_eq(m.reshaped(AutoSize, fix<4>), m.reshaped(v4, fix<4>)));
  VERIFY(is_same_eq(m.reshaped(fix<8>, AutoSize), m.reshaped(fix<8>, v2)));
  VERIFY(is_same_eq(m.reshaped(AutoSize, fix<2>), m.reshaped(v8, fix<2>)));
  VERIFY(is_same_eq(m.reshaped(fix<16>, AutoSize), m.reshaped(fix<16>, v1)));
  VERIFY(is_same_eq(m.reshaped(AutoSize, fix<1>), m.reshaped(v16, fix<1>)));

  check_auto_reshape4x4<ColMajor>(m);
  check_auto_reshape4x4<RowMajor>(m);
  check_auto_reshape4x4<AutoOrder>(m);
  check_auto_reshape4x4<ColMajor>(m.transpose());
  check_auto_reshape4x4<RowMajor>(m.transpose());
  check_auto_reshape4x4<AutoOrder>(m.transpose());

  check_direct_access_reshape4x4(m, fix<MatType::Flags & RowMajorBit>);

  if ((MatType::Flags & RowMajorBit) == 0) {
    VERIFY_IS_EQUAL(m.template reshaped<ColMajor>(2, 8), m.reshaped(2, 8));
    VERIFY_IS_EQUAL(m.template reshaped<ColMajor>(2, 8), m.template reshaped<AutoOrder>(2, 8));
    VERIFY_IS_EQUAL(m.transpose().template reshaped<RowMajor>(2, 8), m.transpose().template reshaped<AutoOrder>(2, 8));
  } else {
    VERIFY_IS_EQUAL(m.template reshaped<ColMajor>(2, 8), m.reshaped(2, 8));
    VERIFY_IS_EQUAL(m.template reshaped<RowMajor>(2, 8), m.template reshaped<AutoOrder>(2, 8));
    VERIFY_IS_EQUAL(m.transpose().template reshaped<ColMajor>(2, 8), m.transpose().template reshaped<AutoOrder>(2, 8));
    VERIFY_IS_EQUAL(m.transpose().reshaped(2, 8), m.transpose().template reshaped<AutoOrder>(2, 8));
  }

  MatrixXi m28r1 = m.template reshaped<RowMajor>(2, 8);
  MatrixXi m28r2 = m.transpose().template reshaped<ColMajor>(8, 2).transpose();
  VERIFY_IS_EQUAL(m28r1, m28r2);

  VERIFY(is_same_eq(m.reshaped(v16, fix<1>), m.reshaped()));
  VERIFY_IS_EQUAL(m.reshaped(16, 1).eval(), m.reshaped().eval());
  VERIFY_IS_EQUAL(m.reshaped(1, 16).eval(), m.reshaped().transpose().eval());
  VERIFY_IS_EQUAL(m.reshaped().reshaped(2, 8), m.reshaped(2, 8));
  VERIFY_IS_EQUAL(m.reshaped().reshaped(4, 4), m.reshaped(4, 4));
  VERIFY_IS_EQUAL(m.reshaped().reshaped(8, 2), m.reshaped(8, 2));

  VERIFY_IS_EQUAL(m.reshaped(), m.template reshaped<ColMajor>());
  VERIFY_IS_EQUAL(m.transpose().reshaped(), m.template reshaped<RowMajor>());
  VERIFY_IS_EQUAL(m.template reshaped<RowMajor>(AutoSize, fix<1>), m.template reshaped<RowMajor>());
  VERIFY_IS_EQUAL(m.template reshaped<AutoOrder>(AutoSize, fix<1>), m.template reshaped<AutoOrder>());

  VERIFY(is_same_eq(m.reshaped(AutoSize, fix<1>), m.reshaped()));
  VERIFY_IS_EQUAL(m.template reshaped<RowMajor>(fix<1>, AutoSize), m.transpose().reshaped().transpose());

  // check assignment
  {
    Matrix<Scalar, Dynamic, 1> m1x(m.size());
    m1x.setRandom();
    VERIFY_IS_APPROX(m.reshaped() = m1x, m1x);
    VERIFY_IS_APPROX(m, m1x.reshaped(4, 4));

    Matrix<Scalar, Dynamic, Dynamic> m28(2, 8);
    m28.setRandom();
    VERIFY_IS_APPROX(m.reshaped(2, 8) = m28, m28);
    VERIFY_IS_APPROX(m, m28.reshaped(4, 4));
    VERIFY_IS_APPROX(m.template reshaped<RowMajor>(2, 8) = m28, m28);

    Matrix<Scalar, Dynamic, Dynamic> m24(2, 4);
    m24.setRandom();
    VERIFY_IS_APPROX(m(seq(0, last, 2), all).reshaped(2, 4) = m24, m24);

    // check constness:
    m.reshaped(2, 8).nestedExpression() = m;
  }
}

// A direct-access reshape with unit inner stride is the nested expression's buffer with a new
// shape, and an expression-sourced reshape whose enumeration order matches the nested evaluator's
// forwards the nested linear accesses one-to-one. Both must preserve the nested evaluator's packet
// access and alignment; without them, copies through Reshaped silently fall back to scalar
// traversal.
template <typename Scalar>
void check_reshaped_evaluator_flags() {
  // Storage orders are pinned so the checks keep their meaning under EIGEN_DEFAULT_TO_ROW_MAJOR.
  typedef Matrix<Scalar, Dynamic, Dynamic, ColMajor> Mat;
  typedef Matrix<Scalar, Dynamic, Dynamic, RowMajor> RowMat;
  typedef Matrix<Scalar, Dynamic, 1, ColMajor> Vec;
  enum { BasePacket = int(internal::evaluator<Mat>::Flags) & PacketAccessBit };

  // Direct access with unit inner stride: packet access and alignment carry over.
  STATIC_CHECK((int(internal::evaluator<Reshaped<Mat, Dynamic, 1, ColMajor> >::Flags) & PacketAccessBit) ==
               int(BasePacket));
  STATIC_CHECK((int(internal::evaluator<Reshaped<Mat, Dynamic, Dynamic, ColMajor> >::Flags) & PacketAccessBit) ==
               int(BasePacket));
  STATIC_CHECK((int(internal::evaluator<Reshaped<Mat, 1, Dynamic, ColMajor> >::Flags) & PacketAccessBit) ==
               int(BasePacket));
  // Vector-shaped reshape of a row-major expression: the canonical storage order differs from the
  // nested one, but the view is still contiguous.
  STATIC_CHECK((int(internal::evaluator<Reshaped<RowMat, Dynamic, 1, RowMajor> >::Flags) & PacketAccessBit) ==
               (int(internal::evaluator<RowMat>::Flags) & PacketAccessBit));
  STATIC_CHECK(int(internal::evaluator<Reshaped<Mat, Dynamic, 1, ColMajor> >::Alignment) ==
               int(internal::evaluator<Mat>::Alignment));

  // Expression source with matching enumeration order: linear accesses and packets forward to the
  // nested evaluator, so matrix-shaped reshapes gain linear access as well.
  typedef CwiseBinaryOp<internal::scalar_sum_op<Scalar, Scalar>, const Mat, const Mat> Sum;
  STATIC_CHECK((int(internal::evaluator<Reshaped<Sum, Dynamic, Dynamic, ColMajor> >::Flags) & PacketAccessBit) ==
               int(BasePacket));
  STATIC_CHECK((int(internal::evaluator<Reshaped<Sum, Dynamic, Dynamic, ColMajor> >::Flags) & LinearAccessBit) ==
               LinearAccessBit);
  STATIC_CHECK(int(internal::evaluator<Reshaped<Sum, Dynamic, Dynamic, ColMajor> >::Alignment) ==
               int(internal::evaluator<Sum>::Alignment));
  // A vector source does not make a matrix-shaped result's storage order immaterial.
  typedef CwiseBinaryOp<internal::scalar_sum_op<Scalar, Scalar>, const Vec, const Vec> VecSum;
  STATIC_CHECK((int(internal::evaluator<Reshaped<VecSum, Dynamic, Dynamic, RowMajor>>::Flags) & PacketAccessBit) == 0);
  STATIC_CHECK((int(internal::evaluator<Reshaped<VecSum, Dynamic, Dynamic, RowMajor>>::Flags) & LinearAccessBit) == 0);
  STATIC_CHECK((int(internal::evaluator<Reshaped<VecSum, Dynamic, Dynamic, ColMajor>>::Flags) & PacketAccessBit) ==
               int(BasePacket));
  STATIC_CHECK((int(internal::evaluator<Reshaped<VecSum, 1, Dynamic, ColMajor>>::Flags) & PacketAccessBit) ==
               int(BasePacket));
  using RowVec = Matrix<Scalar, 1, Dynamic, RowMajor>;
  STATIC_CHECK((int(internal::evaluator<Reshaped<RowVec, Dynamic, Dynamic, ColMajor>>::Flags) &
                (LinearAccessBit | PacketAccessBit)) == 0);

  // No packet access: cross-order reshape (no direct access),
  STATIC_CHECK((int(internal::evaluator<Reshaped<Mat, Dynamic, Dynamic, RowMajor> >::Flags) & PacketAccessBit) == 0);
  // a cross-order expression source (a genuine element permutation),
  STATIC_CHECK((int(internal::evaluator<Reshaped<Sum, Dynamic, Dynamic, RowMajor> >::Flags) & PacketAccessBit) == 0);
  // and a non-unit inner stride.
  typedef Map<Vec, 0, InnerStride<2> > StridedVec;
  STATIC_CHECK((int(internal::evaluator<Reshaped<StridedVec, Dynamic, Dynamic, ColMajor> >::Flags) & PacketAccessBit) ==
               0);
}

// Exercise the (possibly vectorized) assignment kernels with sizes that have partial-packet tails.
template <typename Scalar>
void reshape_copies(Index rows, Index cols) {
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
  typedef Matrix<Scalar, Dynamic, Dynamic, RowMajor> RowMat;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  Mat m = random_for_arithmetic<Mat>(rows, cols);

  Vec v = m.reshaped();
  for (Index k = 0; k < m.size(); ++k) VERIFY_IS_EQUAL(v(k), m(k % rows, k / rows));

  Mat r = m.reshaped(cols, rows);
  for (Index k = 0; k < m.size(); ++k) VERIFY_IS_EQUAL(r(k % cols, k / cols), m(k % rows, k / rows));

  Mat d(rows, cols);
  d.reshaped() = v;
  VERIFY_IS_EQUAL(d, m);

  Vec w = random_for_arithmetic<Vec>(m.size());
  Vec expected = w + v;
  w += m.reshaped();
  VERIFY_IS_EQUAL(w, expected);

  RowMat rm = random_for_arithmetic<RowMat>(rows, cols);
  Vec rv = rm.template reshaped<AutoOrder>();
  for (Index k = 0; k < rm.size(); ++k) VERIFY_IS_EQUAL(rv(k), rm(k / cols, k % cols));

  // Expression-sourced reshapes forward the nested evaluator's linear accesses and packets.
  Mat a = random_for_arithmetic<Mat>(rows, cols), b = random_for_arithmetic<Mat>(rows, cols);
  Mat s = a + b;
  Vec ev = (a + b).reshaped();
  for (Index k = 0; k < s.size(); ++k) VERIFY_IS_EQUAL(ev(k), s(k % rows, k / rows));

  Mat er = (a + b).reshaped(cols, rows);
  for (Index k = 0; k < s.size(); ++k) VERIFY_IS_EQUAL(er(k % cols, k / cols), s(k % rows, k / rows));

  VERIFY_IS_APPROX((a + b).reshaped().sum(), s.sum());

  // Linear scalar accesses into a matrix-shaped row-major reshape must forward the index as-is.
  RowMat ra = random_for_arithmetic<RowMat>(rows, cols), rb = random_for_arithmetic<RowMat>(rows, cols);
  RowMat rer = (ra + rb).template reshaped<RowMajor>(cols, rows);
  for (Index k = 0; k < rer.size(); ++k) {
    VERIFY_IS_EQUAL(rer(k / rows, k % rows), ra(k / cols, k % cols) + rb(k / cols, k % cols));
  }

  // A reshaped lvalue expression without direct access exercises the forwarded packet writes.
  Vec vr(m.size());
  vr.reverse().reshaped(rows, cols) = m;
  for (Index k = 0; k < m.size(); ++k) VERIFY_IS_EQUAL(vr(m.size() - 1 - k), m(k % rows, k / rows));

  RowMat rvr(rows, cols);
  RowMat rw = random_for_arithmetic<RowMat>(cols, rows);
  rvr.reverse().template reshaped<RowMajor>(cols, rows) = rw;
  for (Index k = 0; k < rw.size(); ++k) {
    const Index reversed = rw.size() - 1 - k;
    VERIFY_IS_EQUAL(rvr(reversed / cols, reversed % cols), rw(k / rows, k % rows));
  }
}

template <int Order, typename VectorType, typename RowsType, typename ColsType>
void reshape_vector_order(RowsType rows, ColsType cols) {
  using Scalar = typename VectorType::Scalar;
  using ColMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using RowMatrix = Matrix<Scalar, Dynamic, Dynamic, RowMajor>;
  const Index nrows = internal::get_runtime_value(rows), ncols = internal::get_runtime_value(cols);
  VectorType vector(nrows * ncols);
  for (Index k = 0; k < vector.size(); ++k) vector(k) = Scalar(k + 1);
  ColMatrix expected(nrows, ncols);
  const bool rowOrder = Order == RowMajor || (Order == AutoOrder && VectorType::IsRowMajor);
  for (Index j = 0; j < ncols; ++j)
    for (Index i = 0; i < nrows; ++i) expected(i, j) = Scalar(1 + (rowOrder ? i * ncols + j : j * nrows + i));

  const VectorType& source = vector;
  const auto view = source.template reshaped<Order>(rows, cols);
  VERIFY_IS_EQUAL(ColMatrix(view), expected);
  VERIFY_IS_EQUAL(RowMatrix(view), expected);
  VERIFY_IS_EQUAL(ColMatrix(view.transpose()), expected.transpose());
  VERIFY_IS_EQUAL(RowMatrix(view.transpose()), expected.transpose());
  if (Order != AutoOrder) {
    VERIFY_IS_EQUAL(ColMatrix(source.transpose().template reshaped<Order>(rows, cols).transpose()),
                    expected.transpose());
    VERIFY_IS_EQUAL(RowMatrix(source.transpose().template reshaped<Order>(rows, cols).transpose()),
                    expected.transpose());
  }
  VERIFY_IS_EQUAL(ColMatrix((source + source).template reshaped<Order>(rows, cols)), 2 * expected);
  VERIFY_IS_EQUAL(RowMatrix((source + source).template reshaped<Order>(rows, cols)), 2 * expected);
  for (Index j = 0; j < ncols; ++j)
    for (Index i = 0; i < nrows; ++i) VERIFY_IS_EQUAL(view.coeff(i, j), expected(i, j));
  if (Order == ColMajor) VERIFY_IS_EQUAL(ColMatrix(source.reshaped(rows, cols).transpose()), expected.transpose());

  VectorType destination(vector.size());
  destination.template reshaped<Order>(rows, cols) = expected;
  VERIFY_IS_EQUAL(destination, vector);
  const RowMatrix rowExpected = expected;
  destination.setZero();
  destination.template reshaped<Order>(rows, cols) = rowExpected;
  VERIFY_IS_EQUAL(destination, vector);
  destination.template reshaped<Order>(rows, cols) += rowExpected;
  VERIFY_IS_EQUAL(destination, 2 * vector);
  auto reversed = destination.reverse().template reshaped<Order>(rows, cols);
  using AssignmentTraits =
      internal::copy_using_evaluator_traits<internal::evaluator<decltype(reversed)>, internal::evaluator<RowMatrix>,
                                            internal::assign_op<Scalar, Scalar>>;
  VERIFY(!AssignmentTraits::Vectorized);
  reversed = rowExpected;
  VERIFY_IS_EQUAL(destination, vector.reverse());
  reversed += rowExpected;
  VERIFY_IS_EQUAL(destination, 2 * vector.reverse());
  reversed -= rowExpected;
  VERIFY_IS_EQUAL(destination, vector.reverse());
}

template <typename Scalar, int Order>
void reshape_vector_orders() {
  reshape_vector_order<Order, Matrix<Scalar, Dynamic, 1>>(3, 2);
  reshape_vector_order<Order, Matrix<Scalar, 1, Dynamic>>(3, 2);
  reshape_vector_order<Order, Matrix<Scalar, Dynamic, 1>>(5, 7);
  reshape_vector_order<Order, Matrix<Scalar, 1, Dynamic>>(5, 7);
  reshape_vector_order<Order, Matrix<Scalar, 6, 1>>(fix<3>, fix<2>);
  reshape_vector_order<Order, Matrix<Scalar, 1, 6>>(fix<3>, fix<2>);
}

template <typename BlockType>
void reshape_block(const BlockType& M) {
  auto dense = M.eval();
  Index rows = M.size() / 2;
  Index cols = M.size() / rows;
  VERIFY_IS_EQUAL(dense.reshaped(rows, cols), M.reshaped(rows, cols));

  for (Index i = 0; i < rows; ++i) {
    VERIFY_IS_EQUAL(dense.reshaped(rows, cols).row(i), M.reshaped(rows, cols).row(i));
  }

  for (Index j = 0; j < cols; ++j) {
    VERIFY_IS_EQUAL(dense.reshaped(rows, cols).col(j), M.reshaped(rows, cols).col(j));
  }
}

void reshape_reversed_swap() {
  constexpr int packet_size = internal::packet_traits<double>::size;
  using VectorType = Matrix<double, 1, 3 * packet_size>;
  using MatrixType = Matrix<double, 3, packet_size, packet_size == 1 ? ColMajor : RowMajor>;
  VectorType destination;
  MatrixType source;
  for (Index i = 0; i < destination.size(); ++i) {
    destination(i) = double(i + 1);
    source.data()[i] = double(i + 101);
  }
  const VectorType original = destination;
  const MatrixType originalSource = source;
  auto view = destination.reverse().reshaped<RowMajor>(fix<3>, fix<packet_size>);
  using SwapTraits =
      internal::copy_using_evaluator_traits<internal::evaluator<decltype(view)>, internal::evaluator<MatrixType>,
                                            internal::swap_assign_op<double>>;
  using AssignTraits =
      internal::copy_using_evaluator_traits<internal::evaluator<decltype(view)>, internal::evaluator<MatrixType>,
                                            internal::assign_op<double, double>>;
  VERIFY(!AssignTraits::Vectorized);
  if (internal::packet_traits<double>::Vectorizable && EIGEN_UNALIGNED_VECTORIZE) VERIFY(SwapTraits::Vectorized);
  view.swap(source);
  for (Index i = 0; i < destination.size(); ++i) {
    VERIFY_IS_EQUAL(destination(i), originalSource.data()[destination.size() - 1 - i]);
    VERIFY_IS_EQUAL(source.data()[i], original(destination.size() - 1 - i));
  }
  source.swap(view);
  VERIFY_IS_EQUAL(destination, original);
  VERIFY_IS_EQUAL(source, originalSource);
}

EIGEN_DECLARE_TEST(reshape) {
  typedef Matrix<int, Dynamic, Dynamic, RowMajor> RowMatrixXi;
  typedef Matrix<int, 4, 4, RowMajor> RowMatrix4i;
  MatrixXi mx = MatrixXi::Random(4, 4);
  Matrix4i m4 = Matrix4i::Random(4, 4);
  RowMatrixXi rmx = RowMatrixXi::Random(4, 4);
  RowMatrix4i rm4 = RowMatrix4i::Random(4, 4);

  // test dynamic-size matrix
  CALL_SUBTEST_1(reshape4x4(mx));
  // test static-size matrix
  CALL_SUBTEST_2(reshape4x4(m4));

  CALL_SUBTEST_3(reshape4x4(rmx));
  CALL_SUBTEST_4(reshape4x4(rm4));
  CALL_SUBTEST_5(reshape_block(rm4.col(1)));

  CALL_SUBTEST_6(check_reshaped_evaluator_flags<float>());
  CALL_SUBTEST_6(check_reshaped_evaluator_flags<double>());
  CALL_SUBTEST_6(check_reshaped_evaluator_flags<int>());

  CALL_SUBTEST_8((reshape_vector_orders<float, ColMajor>()));
  CALL_SUBTEST_8((reshape_vector_orders<float, RowMajor>()));
  CALL_SUBTEST_8((reshape_vector_orders<double, ColMajor>()));
  CALL_SUBTEST_8((reshape_vector_orders<double, RowMajor>()));
  CALL_SUBTEST_8((reshape_vector_orders<double, AutoOrder>()));
  CALL_SUBTEST_8((reshape_vector_orders<int, ColMajor>()));
  CALL_SUBTEST_8((reshape_vector_orders<int, RowMajor>()));
  CALL_SUBTEST_8(reshape_reversed_swap());

  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_7(reshape_copies<float>(17, 13));
    CALL_SUBTEST_7(reshape_copies<float>(1, 19));
    CALL_SUBTEST_7(reshape_copies<double>(16, 8));
    CALL_SUBTEST_7(reshape_copies<double>(5, 7));
    CALL_SUBTEST_7(reshape_copies<int>(13, 11));
  }

  TEST_SET_BUT_UNUSED_VARIABLE(mx);
  TEST_SET_BUT_UNUSED_VARIABLE(m4);
  TEST_SET_BUT_UNUSED_VARIABLE(rmx);
  TEST_SET_BUT_UNUSED_VARIABLE(rm4);
}
