// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define TEST_ENABLE_TEMPORARY_TRACKING

#include "main.h"
#include "random_for_arithmetic.h"

struct AssignableScalar {
  double value = 0;
  AssignableScalar& operator=(double x) {
    value = x;
    return *this;
  }
};

struct ConvertingReturnValue;
namespace Eigen {
template <typename BinaryOp>
struct ScalarBinaryOpTraits<AssignableScalar, double, BinaryOp> {
  using ReturnType = AssignableScalar;
};
namespace internal {
template <>
struct traits<ConvertingReturnValue> {
  using ReturnType = Vector3d;
};
}  // namespace internal
}  // namespace Eigen

struct ConvertingReturnValue : ReturnByValue<ConvertingReturnValue> {
  Index rows() const { return 3; }
  Index cols() const { return 1; }
  template <typename Destination>
  void evalTo(Destination& dst) const {
    dst.setConstant(7);
  }
};

bool selects_matrix(const Matrix3d&) { return true; }
bool selects_matrix(const Array<double, 3, 3>&) { return false; }

void implicit_matrix_construction() {
  static_assert(!std::is_convertible<Vector3d, Vector3i>::value, "Mixed scalars require an explicit cast");
  static_assert(!std::is_convertible<Vector3cd, Vector3d>::value,
                "Complex coefficients cannot be assigned to real ones");
  const Vector3d source(1, 2, 3);
  const Vector3d expression = source + source;
  VERIFY_IS_EQUAL(expression, 2 * source);
  const Vector3i cast = source.cast<int>();
  VERIFY_IS_EQUAL(cast, Vector3i(1, 2, 3));
  const Vector3cd complex = source;
  VERIFY_IS_EQUAL(complex.real(), source);
  const Matrix<AssignableScalar, 3, 1> custom = source;
  for (Index i = 0; i < source.size(); ++i) VERIFY_IS_EQUAL(custom(i).value, source(i));
  const ConvertingReturnValue value;
  const Vector3i converted = value;
  VERIFY_IS_EQUAL(converted, Vector3i::Constant(7));
  const EigenBase<ReturnByValue<ConvertingReturnValue>>& base = value;
  const Vector3i converted_base = base;
  VERIFY_IS_EQUAL(converted_base, converted);
  PermutationMatrix<3> permutation;
  permutation.setIdentity();
  const Matrix3d identity = permutation;
  VERIFY_IS_EQUAL(identity, Matrix3d::Identity());
  static_assert(!std::is_convertible<PermutationMatrix<3>, Array<double, 3, 3>>::value,
                "Permutations must not become implicit Array conversion candidates");
  VERIFY(selects_matrix(permutation));

  static_assert(!std::is_convertible<Array3d, Array3i>::value, "Mixed scalars require an explicit cast");
  static_assert(!std::is_convertible<Array3cd, Array3d>::value, "Complex coefficients cannot be assigned to real ones");
  const Array3d array_source = source.array();
  const Array3i array_cast = array_source.cast<int>();
  VERIFY((array_cast == cast.array()).all());
  const Array3cd array_complex = array_source;
  VERIFY((array_complex.real() == array_source).all());
  const Array<AssignableScalar, 3, 1> array_custom = array_source;
  for (Index i = 0; i < source.size(); ++i) VERIFY_IS_EQUAL(array_custom(i).value, source(i));
  const Array3i array_converted = base;
  VERIFY((array_converted == converted.array()).all());
}

template <typename MatrixType>
struct Wrapper {
  MatrixType m_mat;
  inline Wrapper(const MatrixType& x) : m_mat(x) {}
  inline operator const MatrixType&() const { return m_mat; }
  inline operator MatrixType&() { return m_mat; }
};

enum my_sizes { M = 12, N = 7 };

template <typename MatrixType>
void ctor_init1(const MatrixType& m) {
  // Check logic in PlainObjectBase::_init1
  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m0 = random_for_arithmetic<MatrixType>(rows, cols);

  VERIFY_EVALUATION_COUNT(MatrixType m1(m0), 1);
  VERIFY_EVALUATION_COUNT(MatrixType m2(m0 + m0), 1);
  VERIFY_EVALUATION_COUNT(MatrixType m2(m0.block(0, 0, rows, cols)), 1);

  Wrapper<MatrixType> wrapper(m0);
  VERIFY_EVALUATION_COUNT(MatrixType m3(wrapper), 1);
}

EIGEN_DECLARE_TEST(constructor) {
  CALL_SUBTEST_1(implicit_matrix_construction());
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(ctor_init1(Matrix<float, 1, 1>()));
    CALL_SUBTEST_1(ctor_init1(Matrix4d()));
    CALL_SUBTEST_1(ctor_init1(
        MatrixXcf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_1(ctor_init1(
        MatrixXi(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
  }
  {
    Matrix<Index, 1, 1> a(123);
    VERIFY_IS_EQUAL(a[0], 123);
  }
  {
    Matrix<Index, 1, 1> a(123.0);
    VERIFY_IS_EQUAL(a[0], 123);
  }
  {
    Matrix<float, 1, 1> a(123);
    VERIFY_IS_EQUAL(a[0], 123.f);
  }
  {
    Array<Index, 1, 1> a(123);
    VERIFY_IS_EQUAL(a[0], 123);
  }
  {
    Array<Index, 1, 1> a(123.0);
    VERIFY_IS_EQUAL(a[0], 123);
  }
  {
    Array<float, 1, 1> a(123);
    VERIFY_IS_EQUAL(a[0], 123.f);
  }
  {
    Array<Index, 3, 3> a(123);
    VERIFY_IS_EQUAL(a(4), 123);
  }
  {
    Array<Index, 3, 3> a(123.0);
    VERIFY_IS_EQUAL(a(4), 123);
  }
  {
    Array<float, 3, 3> a(123);
    VERIFY_IS_EQUAL(a(4), 123.f);
  }
  {
    MatrixXi m1(M, N);
    VERIFY_IS_EQUAL(m1.rows(), M);
    VERIFY_IS_EQUAL(m1.cols(), N);
    ArrayXXi a1(M, N);
    VERIFY_IS_EQUAL(a1.rows(), M);
    VERIFY_IS_EQUAL(a1.cols(), N);
    VectorXi v1(M);
    VERIFY_IS_EQUAL(v1.size(), M);
    ArrayXi a2(M);
    VERIFY_IS_EQUAL(a2.size(), M);
  }
}
