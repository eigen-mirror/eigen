// This file is triangularView of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include "product_test_helpers.h"

template <typename MatrixType>
void trmv(const MatrixType& m) {
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols), m3(rows, cols);
  VectorType v1 = VectorType::Random(rows);

  Scalar s1 = internal::random<Scalar>();

  m1 = MatrixType::Random(rows, cols);

  // check with a column-major matrix
  m3 = m1.template triangularView<Eigen::Lower>();
  VERIFY(verifyProduct(m3 * v1, m1.template triangularView<Eigen::Lower>() * v1, m3, v1));
  m3 = m1.template triangularView<Eigen::Upper>();
  VERIFY(verifyProduct(m3 * v1, m1.template triangularView<Eigen::Upper>() * v1, m3, v1));
  m3 = m1.template triangularView<Eigen::UnitLower>();
  VERIFY(verifyProduct(m3 * v1, m1.template triangularView<Eigen::UnitLower>() * v1, m3, v1));
  m3 = m1.template triangularView<Eigen::UnitUpper>();
  VERIFY(verifyProduct(m3 * v1, m1.template triangularView<Eigen::UnitUpper>() * v1, m3, v1));

  // check conjugated and scalar multiple expressions (col-major)
  m3 = m1.template triangularView<Eigen::Lower>();
  VERIFY(verifyProduct((s1 * m3).conjugate() * v1, (s1 * m1).conjugate().template triangularView<Eigen::Lower>() * v1,
                       (s1 * m3).conjugate(), v1));
  m3 = m1.template triangularView<Eigen::Upper>();
  VERIFY(verifyProduct(m3.conjugate() * v1.conjugate(),
                       m1.conjugate().template triangularView<Eigen::Upper>() * v1.conjugate(), m3.conjugate(),
                       v1.conjugate()));

  // check with a row-major matrix
  m3 = m1.template triangularView<Eigen::Upper>();
  VERIFY(verifyProduct(m3.transpose() * v1, m1.transpose().template triangularView<Eigen::Lower>() * v1, m3.transpose(),
                       v1));
  m3 = m1.template triangularView<Eigen::Lower>();
  VERIFY(verifyProduct(m3.transpose() * v1, m1.transpose().template triangularView<Eigen::Upper>() * v1, m3.transpose(),
                       v1));
  m3 = m1.template triangularView<Eigen::UnitUpper>();
  VERIFY(verifyProduct(m3.transpose() * v1, m1.transpose().template triangularView<Eigen::UnitLower>() * v1,
                       m3.transpose(), v1));
  m3 = m1.template triangularView<Eigen::UnitLower>();
  VERIFY(verifyProduct(m3.transpose() * v1, m1.transpose().template triangularView<Eigen::UnitUpper>() * v1,
                       m3.transpose(), v1));

  // check conjugated and scalar multiple expressions (row-major)
  m3 = m1.template triangularView<Eigen::Upper>();
  VERIFY(verifyProduct(m3.adjoint() * v1, m1.adjoint().template triangularView<Eigen::Lower>() * v1, m3.adjoint(), v1));
  m3 = m1.template triangularView<Eigen::Lower>();
  VERIFY(verifyProduct(m3.adjoint() * (s1 * v1.conjugate()),
                       m1.adjoint().template triangularView<Eigen::Upper>() * (s1 * v1.conjugate()), m3.adjoint(),
                       (s1 * v1.conjugate()).eval()));
  m3 = m1.template triangularView<Eigen::UnitUpper>();

  // check transposed cases:
  m3 = m1.template triangularView<Eigen::Lower>();
  VERIFY(verifyProduct(v1.transpose() * m3, v1.transpose() * m1.template triangularView<Eigen::Lower>(), v1.transpose(),
                       m3));
  VERIFY(verifyProduct(v1.adjoint() * m3, v1.adjoint() * m1.template triangularView<Eigen::Lower>(), v1.adjoint(), m3));
  VERIFY(verifyProduct(v1.adjoint() * m3.adjoint(), v1.adjoint() * m1.template triangularView<Eigen::Lower>().adjoint(),
                       v1.adjoint(), m3.adjoint()));

  // TODO check with sub-matrices
}

// A destination with a negative inner stride, which BLAS addresses from its lowest entry and Eigen from entry 0.
// Square and both non-square shapes reach the TRMV, AXPY and GEMV calls of the BLAS backend.
template <typename Scalar, int StorageOrder, int Mode>
void trmv_negative_stride_destination(Index rows, Index cols) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, StorageOrder>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  const Mat m = Mat::Random(rows, cols);
  const Vec v = Vec::Random(cols);
  const Mat dense = m.template triangularView<Mode>();
  const Vec ref = dense * v;

  // The reversed destination sits between two guard blocks of the same length.
  Vec buffer = Vec::Random(3 * rows);
  const Vec initial = buffer;
  Map<Vec, Unaligned, InnerStride<-1>> dst(buffer.data() + 2 * rows - 1, rows, InnerStride<-1>(-1));
  dst.noalias() = m.template triangularView<Mode>() * v;
  VERIFY_IS_APPROX(Vec(dst), ref);
  VERIFY_IS_EQUAL(buffer.head(rows), initial.head(rows));
  VERIFY_IS_EQUAL(buffer.tail(rows), initial.tail(rows));
}

template <typename Scalar, int StorageOrder>
void trmv_negative_stride_destination_shapes() {
  const Index shapes[][2] = {{7, 7}, {9, 5}, {5, 9}};
  for (const auto& shape : shapes) {
    trmv_negative_stride_destination<Scalar, StorageOrder, Lower>(shape[0], shape[1]);
    trmv_negative_stride_destination<Scalar, StorageOrder, Upper>(shape[0], shape[1]);
    trmv_negative_stride_destination<Scalar, StorageOrder, UnitLower>(shape[0], shape[1]);
    trmv_negative_stride_destination<Scalar, StorageOrder, UnitUpper>(shape[0], shape[1]);
  }
}

template <typename Scalar>
void trmv_negative_stride_destination_all() {
  trmv_negative_stride_destination_shapes<Scalar, ColMajor>();
  trmv_negative_stride_destination_shapes<Scalar, RowMajor>();
}

// A real triangular matrix times a complex vector at a fixed size, through the column-major kernel (directly and as the
// transpose of a row-vector product) and the row-major one. The unevaluated s * v.adjoint() reaches the kernel as a
// conjugated vector and a scale factor, as in #3214. Without the workaround, GCC 15.2 at -O3 with AVX2 gets sizes 3 and
// 4 wrong here; which sizes it miscompiles depends on every product in the translation unit.
template <int Size, int Mode>
void trmv_real_times_complex_fixed_size() {
  using CD = std::complex<double>;
  using RealMat = Matrix<double, Size, Size>;
  using ComplexMat = Matrix<CD, Size, Size>;
  using ComplexVec = Matrix<CD, Size, 1>;
  using ComplexRowVec = Matrix<CD, 1, Size>;
  const RealMat m = RealMat::Random();
  const ComplexVec v = ComplexVec::Random();
  const CD s = internal::random<CD>();
  const ComplexMat dense = m.template triangularView<Mode>().toDenseMatrix().template cast<CD>();
  const ComplexMat dense_of_transpose =
      m.transpose().template triangularView<Mode>().toDenseMatrix().template cast<CD>();

  const ComplexVec sv = s * v;
  VERIFY(verifyProduct(ComplexVec(m.template triangularView<Mode>() * sv), ComplexVec(dense * sv), dense, sv));
  VERIFY(verifyProduct(ComplexVec(m.transpose().template triangularView<Mode>() * sv),
                       ComplexVec(dense_of_transpose * sv), dense_of_transpose, sv));
  const ComplexRowVec svh = s * v.adjoint();
  VERIFY(verifyProduct(ComplexRowVec(svh * m.transpose().template triangularView<Mode>()),
                       ComplexRowVec(svh * dense_of_transpose), svh, dense_of_transpose));
  VERIFY(verifyProduct(ComplexRowVec(s * v.adjoint() * m.transpose().template triangularView<Mode>()),
                       ComplexRowVec(svh * dense_of_transpose), svh, dense_of_transpose));
}

template <int Size>
void trmv_real_times_complex_fixed_size_all_modes() {
  trmv_real_times_complex_fixed_size<Size, Lower>();
  trmv_real_times_complex_fixed_size<Size, Upper>();
  trmv_real_times_complex_fixed_size<Size, UnitLower>();
  trmv_real_times_complex_fixed_size<Size, UnitUpper>();
}

EIGEN_DECLARE_TEST(product_trmv) {
  int s = 0;
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(trmv(Matrix<float, 1, 1>()));
    CALL_SUBTEST_2(trmv(Matrix<float, 2, 2>()));
    CALL_SUBTEST_3(trmv(Matrix3d()));

    s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2);
    CALL_SUBTEST_4(trmv(MatrixXcf(s, s)));
    CALL_SUBTEST_5(trmv(MatrixXcd(s, s)));
    TEST_SET_BUT_UNUSED_VARIABLE(s);

    s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE);
    CALL_SUBTEST_6(trmv(Matrix<float, Dynamic, Dynamic, RowMajor>(s, s)));
    TEST_SET_BUT_UNUSED_VARIABLE(s);

    CALL_SUBTEST_8(trmv_real_times_complex_fixed_size_all_modes<2>());
    CALL_SUBTEST_8(trmv_real_times_complex_fixed_size_all_modes<3>());
    CALL_SUBTEST_8(trmv_real_times_complex_fixed_size_all_modes<4>());
    CALL_SUBTEST_8(trmv_real_times_complex_fixed_size_all_modes<5>());
  }
  CALL_SUBTEST_7(trmv_negative_stride_destination_all<float>());
  CALL_SUBTEST_7(trmv_negative_stride_destination_all<double>());
  CALL_SUBTEST_7(trmv_negative_stride_destination_all<std::complex<float>>());
  CALL_SUBTEST_7(trmv_negative_stride_destination_all<std::complex<double>>());
}
