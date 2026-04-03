// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#undef VERIFY_IS_APPROX
#define VERIFY_IS_APPROX(a, b) VERIFY((a) == (b));
#undef VERIFY_IS_NOT_APPROX
#define VERIFY_IS_NOT_APPROX(a, b) VERIFY((a) != (b));

template <typename MatrixType>
void signed_integer_type_tests(const MatrixType& m) {
  typedef typename MatrixType::Scalar Scalar;
  constexpr Scalar kMax = (Scalar(1) << ((8 * sizeof(Scalar) - 2) / 2)) - 1;

  enum { is_signed = (Scalar(-1) > Scalar(0)) ? 0 : 1 };
  VERIFY(is_signed == 1);

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = RandomMatrix<MatrixType>(rows, cols, Scalar(0), kMax);
  MatrixType m2 = RandomMatrix<MatrixType>(rows, cols, Scalar(0), kMax);

  // check linear structure

  Scalar s1 = internal::random<Scalar>(1, kMax);

  VERIFY_IS_EQUAL(-(-m1), m1);
  VERIFY_IS_EQUAL(-m2 + m1 + m2, m1);
  VERIFY_IS_EQUAL((-m1 + m2) * s1, -s1 * m1 + s1 * m2);
}

template <typename MatrixType>
void integer_type_tests(const MatrixType& m) {
  typedef typename MatrixType::Scalar Scalar;
  constexpr Scalar kMax = (Scalar(1) << ((8 * sizeof(Scalar) - 2) / 2)) - 1;

  VERIFY(NumTraits<Scalar>::IsInteger);
  enum { is_signed = (Scalar(-1) > Scalar(0)) ? 0 : 1 };
  VERIFY(int(NumTraits<Scalar>::IsSigned) == is_signed);

  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> SquareMatrixType;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = RandomMatrix<MatrixType>(rows, cols, Scalar(0), kMax);
  MatrixType m2 = RandomMatrix<MatrixType>(rows, cols, Scalar(0), kMax);
  MatrixType m3 = RandomMatrix<MatrixType>(rows, cols, Scalar(0), kMax);
  SquareMatrixType square = RandomMatrix<SquareMatrixType>(rows, rows, Scalar(0), kMax);
  VectorType v1 = RandomMatrix<VectorType, Scalar>(rows, Index(1), Scalar(0), NumTraits<Scalar>::highest() / Scalar(2));

  VERIFY_IS_APPROX(v1, v1);
  VERIFY_IS_NOT_APPROX(v1, 2 * v1);
  VERIFY_IS_APPROX(VectorType::Zero(rows), v1 - v1);

  VERIFY_IS_APPROX(m1, m1);
  VERIFY_IS_NOT_APPROX(m1, 2 * m1);
  VERIFY_IS_APPROX(MatrixType::Zero(rows, cols), m1 - m1);

  VERIFY_IS_APPROX(m3 = m1, m1);
  MatrixType m4;
  VERIFY_IS_APPROX(m4 = m1, m1);

  m3.real() = m1.real();
  VERIFY_IS_APPROX(static_cast<const MatrixType&>(m3).real(), static_cast<const MatrixType&>(m1).real());
  VERIFY_IS_APPROX(static_cast<const MatrixType&>(m3).real(), m1.real());

  // check == / != operators
  VERIFY(m1 == m1);
  VERIFY(m1 != m2);
  VERIFY(!(m1 == m2));
  VERIFY(!(m1 != m1));
  m1 = m2;
  VERIFY(m1 == m2);
  VERIFY(!(m1 != m2));

  // check linear structure

  Scalar s1 = internal::random<Scalar>(1, kMax);

  VERIFY_IS_EQUAL(m1 + m1, 2 * m1);
  VERIFY_IS_EQUAL(m1 + m2 - m1, m2);
  VERIFY_IS_EQUAL(m1 * s1, s1 * m1);
  VERIFY_IS_EQUAL((m1 + m2) * s1, s1 * m1 + s1 * m2);
  m3 = m2;
  m3 += m1;
  VERIFY_IS_EQUAL(m3, m1 + m2);
  m3 = m2;
  m3 -= m1;
  VERIFY_IS_EQUAL(m3, m2 - m1);
  m3 = m2;
  m3 *= s1;
  VERIFY_IS_EQUAL(m3, s1 * m2);

  // check matrix product.

  if (!NumTraits<Scalar>::IsSigned) {
    VERIFY_IS_APPROX(SquareMatrixType::Identity(rows, rows) * m1, m1);
    VERIFY_IS_APPROX(square * (m1 + m2), square * m1 + square * m2);
    VERIFY_IS_APPROX((m1 + m2).transpose() * square, m1.transpose() * square + m2.transpose() * square);
    VERIFY_IS_APPROX((m1 * m2.transpose()) * m1, m1 * (m2.transpose() * m1));
  }
}

template <int>
void integer_types_extra() {
  VERIFY_IS_EQUAL(int(internal::scalar_div_cost<int>::value), 8);
  VERIFY_IS_EQUAL(int(internal::scalar_div_cost<unsigned int>::value), 8);
  if (sizeof(long) > sizeof(int)) {
    VERIFY(int(internal::scalar_div_cost<long>::value) > int(internal::scalar_div_cost<int>::value));
    VERIFY(int(internal::scalar_div_cost<unsigned long>::value) > int(internal::scalar_div_cost<int>::value));
  }
}

EIGEN_DECLARE_TEST(integer_types) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(integer_type_tests(Matrix<unsigned int, 1, 1>()));
    CALL_SUBTEST_1(integer_type_tests(Matrix<unsigned long, 3, 4>()));

    CALL_SUBTEST_2(integer_type_tests(Matrix<long, 2, 2>()));
    CALL_SUBTEST_2(signed_integer_type_tests(Matrix<long, 2, 2>()));

    CALL_SUBTEST_3(integer_type_tests(Matrix<char, 2, Dynamic>(2, 10)));
    CALL_SUBTEST_3(signed_integer_type_tests(Matrix<signed char, 2, Dynamic>(2, 10)));

    CALL_SUBTEST_4(integer_type_tests(Matrix<unsigned char, 3, 3>()));
    CALL_SUBTEST_4(integer_type_tests(Matrix<unsigned char, Dynamic, Dynamic>(20, 20)));

    CALL_SUBTEST_5(integer_type_tests(Matrix<short, Dynamic, 4>(7, 4)));
    CALL_SUBTEST_5(signed_integer_type_tests(Matrix<short, Dynamic, 4>(7, 4)));

    CALL_SUBTEST_6(integer_type_tests(Matrix<unsigned short, 4, 4>()));

    CALL_SUBTEST_7(integer_type_tests(Matrix<long long, 11, 13>()));
    CALL_SUBTEST_7(signed_integer_type_tests(Matrix<long long, 11, 13>()));

    CALL_SUBTEST_8(integer_type_tests(Matrix<unsigned long long, Dynamic, 5>(1, 5)));
  }
  CALL_SUBTEST_9(integer_types_extra<0>());
}
