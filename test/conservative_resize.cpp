// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Hauke Heibel <hauke.heibel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_RUNTIME_NO_MALLOC
#include "main.h"

#include <Eigen/Core>
#include "AnnoyingScalar.h"

using namespace Eigen;

template <typename Scalar, int Storage, int MaxSize = Dynamic>
void run_matrix_tests() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Storage, MaxSize, MaxSize>;

  MatrixType m, n;

  // boundary cases ...
  m = n = MatrixType::Random(50, 50);
  m.conservativeResize(1, 50);
  VERIFY_IS_APPROX(m, n.block(0, 0, 1, 50));

  m = n = MatrixType::Random(50, 50);
  m.conservativeResize(50, 1);
  VERIFY_IS_APPROX(m, n.block(0, 0, 50, 1));

  m = n = MatrixType::Random(50, 50);
  m.conservativeResize(50, 50);
  VERIFY_IS_APPROX(m, n.block(0, 0, 50, 50));

  // random shrinking ...
  for (int i = 0; i < 25; ++i) {
    const Index rows = internal::random<Index>(1, 50);
    const Index cols = internal::random<Index>(1, 50);
    m = n = MatrixType::Random(50, 50);
    m.conservativeResize(rows, cols);
    VERIFY_IS_APPROX(m, n.block(0, 0, rows, cols));
  }

  // random growing with zeroing ...
  for (int i = 0; i < 25; ++i) {
    const Index rows = internal::random<Index>(50, 75);
    const Index cols = internal::random<Index>(50, 75);
    m = n = MatrixType::Random(50, 50);
    m.conservativeResizeLike(MatrixType::Zero(rows, cols));
    VERIFY_IS_APPROX(m.block(0, 0, n.rows(), n.cols()), n);
    VERIFY(rows <= 50 || m.block(50, 0, rows - 50, cols).sum() == Scalar(0));
    VERIFY(cols <= 50 || m.block(0, 50, rows, cols - 50).sum() == Scalar(0));
  }
}

template <typename Scalar>
void run_vector_tests() {
  typedef Matrix<Scalar, 1, Eigen::Dynamic> VectorType;

  VectorType m, n;

  // boundary cases ...
  m = n = VectorType::Random(50);
  m.conservativeResize(1);
  VERIFY_IS_APPROX(m, n.segment(0, 1));

  m = n = VectorType::Random(50);
  m.conservativeResize(50);
  VERIFY_IS_APPROX(m, n.segment(0, 50));

  m = n = VectorType::Random(50);
  m.conservativeResize(m.rows(), 1);
  VERIFY_IS_APPROX(m, n.segment(0, 1));

  m = n = VectorType::Random(50);
  m.conservativeResize(m.rows(), 50);
  VERIFY_IS_APPROX(m, n.segment(0, 50));

  // random shrinking ...
  for (int i = 0; i < 50; ++i) {
    const int size = internal::random<int>(1, 50);
    m = n = VectorType::Random(50);
    m.conservativeResize(size);
    VERIFY_IS_APPROX(m, n.segment(0, size));

    m = n = VectorType::Random(50);
    m.conservativeResize(m.rows(), size);
    VERIFY_IS_APPROX(m, n.segment(0, size));
  }

  // random growing with zeroing ...
  for (int i = 0; i < 50; ++i) {
    const int size = internal::random<int>(50, 100);
    m = n = VectorType::Random(50);
    m.conservativeResizeLike(VectorType::Zero(size));
    VERIFY_IS_APPROX(m.segment(0, 50), n);
    VERIFY(size <= 50 || m.segment(50, size - 50).sum() == Scalar(0));

    m = n = VectorType::Random(50);
    m.conservativeResizeLike(Matrix<Scalar, Dynamic, Dynamic>::Zero(1, size));
    VERIFY_IS_APPROX(m.segment(0, 50), n);
    VERIFY(size <= 50 || m.segment(50, size - 50).sum() == Scalar(0));

    // A column vector of the same size: the row vector stays a row vector.
    m = n = VectorType::Random(50);
    m.conservativeResizeLike(Matrix<Scalar, Dynamic, 1>::Zero(size));
    VERIFY_IS_EQUAL(m.size(), Index(size));
    VERIFY_IS_APPROX(m.segment(0, 50), n);
    VERIFY(size <= 50 || m.segment(50, size - 50).sum() == Scalar(0));
  }
}

// Basic memory leak check with a non-copyable scalar type
template <int>
void noncopyable() {
  typedef Eigen::Matrix<AnnoyingScalar, Dynamic, 1> VectorType;
  typedef Eigen::Matrix<AnnoyingScalar, Dynamic, Dynamic> MatrixType;

  {
#ifndef EIGEN_TEST_ANNOYING_SCALAR_DONT_THROW
    AnnoyingScalar::dont_throw = true;
#endif
    int n = 50;
    VectorType v0(n), v1(n);
    MatrixType m0(n, n), m1(n, n), m2(n, n);
    v0.setOnes();
    v1.setOnes();
    m0.setOnes();
    m1.setOnes();
    m2.setOnes();
    VERIFY(m0 == m1);
    m0.conservativeResize(2 * n, 2 * n);
    VERIFY(m0.topLeftCorner(n, n) == m1);

    VERIFY(v0.head(n) == v1);
    v0.conservativeResize(2 * n);
    VERIFY(v0.head(n) == v1);
  }
  VERIFY(AnnoyingScalar::instances == 0 && "global memory leak detected in noncopyable");
}

// Growing past the compile-time maximum sizes is caught as resize() catches it.
template <int>
void max_size_checks() {
  Matrix<double, Dynamic, Dynamic, ColMajor, 4, 4> m(4, 2);
  VERIFY_RAISES_ASSERT(m.conservativeResize(4, 6));
  Matrix<double, Dynamic, 1, ColMajor, 4, 1> v(2);
  VERIFY_RAISES_ASSERT(v.conservativeResize(8));
  VERIFY_RAISES_ASSERT(m.conservativeResizeLike(MatrixXd::Zero(4, 6)));
  VERIFY_RAISES_ASSERT(v.conservativeResizeLike(VectorXd::Zero(6)));
  VERIFY_RAISES_ASSERT(v.conservativeResizeLike(RowVectorXd::Zero(6)));
  // Within the maximum size, a vector resized like a transposed vector is valid.
  v.conservativeResizeLike(RowVectorXd::Zero(4));
  VERIFY_IS_EQUAL(v.size(), 4);
}

struct ResizeScalar {
  int value = 0;
  static Index copies;
  static Index moves;

  ResizeScalar() = default;
  ResizeScalar(const ResizeScalar& other) : value(other.value) { ++copies; }
  ResizeScalar(ResizeScalar&& other) noexcept : value(other.value) { ++moves; }
  ResizeScalar& operator=(const ResizeScalar& other) {
    value = other.value;
    ++copies;
    return *this;
  }
  ResizeScalar& operator=(ResizeScalar&& other) noexcept {
    value = other.value;
    ++moves;
    return *this;
  }
  static void reset() { copies = moves = 0; }
};

Index ResizeScalar::copies = 0;
Index ResizeScalar::moves = 0;

template <typename VectorType>
void fixed_capacity_vector_resize() {
  STATIC_CHECK((!std::is_trivially_copyable<ResizeScalar>::value));
  STATIC_CHECK((internal::conservative_resize_like_impl<VectorType>::IsRelocatable));
  STATIC_CHECK((!internal::conservative_resize_like_impl<Matrix<ResizeScalar, Dynamic, 1>>::IsRelocatable));
  VectorType v;
  const ResizeScalar* data = v.data();
  ResizeScalar::reset();
  for (Index size = 1; size <= VectorType::MaxSizeAtCompileTime; ++size) {
    v.conservativeResize(size);
    v(size - 1).value = int(size);
  }
  VERIFY_IS_EQUAL(ResizeScalar::copies, 0);
  VERIFY_IS_EQUAL(ResizeScalar::moves, 0);
  VERIFY(v.data() == data);
  for (Index i = 0; i < v.size(); ++i) VERIFY_IS_EQUAL(v(i).value, i + 1);

  v.conservativeResize(2);
  v.conservativeResize(2);
  VERIFY_IS_EQUAL(ResizeScalar::copies, 0);
  VERIFY_IS_EQUAL(ResizeScalar::moves, 0);

  // A source with the opposite orientation exercises the vector resizeLike specialization.
  Matrix<ResizeScalar, VectorType::IsRowMajor ? Dynamic : 1, VectorType::IsRowMajor ? 1 : Dynamic> other(5);
  for (Index i = 0; i < other.size(); ++i) other(i).value = -int(i + 1);
  v.conservativeResizeLike(other);
  VERIFY_IS_EQUAL(ResizeScalar::copies, 3);
  VERIFY_IS_EQUAL(ResizeScalar::moves, 0);
  VERIFY_IS_EQUAL(v.rows(), VectorType::IsRowMajor ? 1 : 5);
  VERIFY_IS_EQUAL(v.cols(), VectorType::IsRowMajor ? 5 : 1);
  for (Index i = 0; i < v.size(); ++i) VERIFY_IS_EQUAL(v(i).value, i < 2 ? i + 1 : -(i + 1));

  ResizeScalar::reset();
  v.conservativeResizeLike(other.head(1));
  VERIFY_IS_EQUAL(v.size(), 1);
  VERIFY_IS_EQUAL(v(0).value, 1);
  v.conservativeResize(0);
  VERIFY_IS_EQUAL(v.size(), 0);
  VERIFY_IS_EQUAL(ResizeScalar::copies, 0);
  VERIFY_IS_EQUAL(ResizeScalar::moves, 0);
  VERIFY(v.data() == data);
}

template <typename MatrixType>
void fixed_capacity_matrix_resize() {
  STATIC_CHECK((internal::conservative_resize_like_impl<MatrixType>::IsRelocatable));
  STATIC_CHECK((!internal::conservative_resize_like_impl<Matrix<ResizeScalar, Dynamic, Dynamic>>::IsRelocatable));
  const Index inner = 3;
  MatrixType m(MatrixType::IsRowMajor ? 0 : inner, MatrixType::IsRowMajor ? inner : 0);
  const ResizeScalar* data = m.data();
  ResizeScalar::reset();
  for (Index outer = 1; outer <= 5; ++outer) {
    m.conservativeResize(MatrixType::IsRowMajor ? outer : inner, MatrixType::IsRowMajor ? inner : outer);
    for (Index i = (outer - 1) * inner; i < m.size(); ++i) m.data()[i].value = int(i + 1);
  }
  VERIFY_IS_EQUAL(ResizeScalar::copies, 0);
  VERIFY_IS_EQUAL(ResizeScalar::moves, 0);
  VERIFY(m.data() == data);
  for (Index i = 0; i < m.size(); ++i) VERIFY_IS_EQUAL(m.data()[i].value, i + 1);

  m.conservativeResize(MatrixType::IsRowMajor ? 2 : inner, MatrixType::IsRowMajor ? inner : 2);
  VERIFY_IS_EQUAL(ResizeScalar::copies, 0);
  VERIFY_IS_EQUAL(ResizeScalar::moves, 0);
  MatrixType other(MatrixType::IsRowMajor ? 4 : inner, MatrixType::IsRowMajor ? inner : 4);
  for (Index i = 0; i < other.size(); ++i) other.data()[i].value = -int(i + 1);
  m.conservativeResizeLike(other);
  VERIFY_IS_EQUAL(ResizeScalar::copies, 2 * inner);
  VERIFY_IS_EQUAL(ResizeScalar::moves, 0);
  VERIFY_IS_EQUAL(m.rows(), other.rows());
  VERIFY_IS_EQUAL(m.cols(), other.cols());
  for (Index i = 0; i < m.size(); ++i) VERIFY_IS_EQUAL(m.data()[i].value, i < 2 * inner ? i + 1 : -(i + 1));

  ResizeScalar::reset();
  other.resize(MatrixType::IsRowMajor ? 1 : inner, MatrixType::IsRowMajor ? inner : 1);
  m.conservativeResizeLike(other);
  VERIFY_IS_EQUAL(m.size(), inner);
  for (Index i = 0; i < m.size(); ++i) VERIFY_IS_EQUAL(m.data()[i].value, i + 1);
  m.conservativeResize(MatrixType::IsRowMajor ? 0 : inner, MatrixType::IsRowMajor ? inner : 0);
  VERIFY_IS_EQUAL(m.size(), 0);
  VERIFY_IS_EQUAL(ResizeScalar::copies, 0);
  VERIFY_IS_EQUAL(ResizeScalar::moves, 0);
  VERIFY(m.data() == data);
}

struct InitializedResizeScalar {
  int value;
  InitializedResizeScalar() : value(0) {}
};

template <typename Vector>
void same_shape_vector_resize() {
  using Scalar = typename Vector::Scalar;
  STATIC_CHECK(std::is_trivially_copyable<Scalar>::value);
  STATIC_CHECK(NumTraits<Scalar>::RequireInitialization);
  Vector v(8);
  for (Index i = 0; i < v.size(); ++i) v(i).value = int(i + 1);
  const auto* data = v.data();
  const bool malloc_allowed = internal::is_malloc_allowed();
  internal::set_is_malloc_allowed(false);
  // Every overload must leave unchanged storage alone, including scalars that require initialization.
  v.conservativeResize(v.size());
  v.conservativeResize(v.rows(), v.cols());
  v.conservativeResize(NoChange, v.cols());
  v.conservativeResize(v.rows(), NoChange);
  internal::set_is_malloc_allowed(malloc_allowed);
  VERIFY(v.data() == data);
  for (Index i = 0; i < v.size(); ++i) VERIFY_IS_EQUAL(v(i).value, i + 1);
}

template <typename Vector>
void two_dimensional_vector_resize() {
  STATIC_CHECK(!std::is_trivially_copyable<typename Vector::Scalar>::value);
  Vector v;
  const auto* data = v.data();
  ResizeScalar::reset();
  for (Index i = 0; i < 16; ++i) {
    if (i % 3 == 0) {
      v.conservativeResize(i + 1);
    } else if (i % 3 == 1) {
      v.conservativeResize(Vector::RowsAtCompileTime == 1 ? 1 : i + 1, Vector::RowsAtCompileTime == 1 ? i + 1 : 1);
    } else {
      if (Vector::RowsAtCompileTime == 1)
        v.conservativeResize(NoChange, i + 1);
      else
        v.conservativeResize(i + 1, NoChange);
    }
    v(i).value = int(i + 1);
  }
  VERIFY_IS_EQUAL(ResizeScalar::copies, 0);
  VERIFY_IS_EQUAL(ResizeScalar::moves, 0);
  VERIFY(v.data() == data);
  for (Index i = 0; i < v.size(); ++i) VERIFY_IS_EQUAL(v(i).value, i + 1);
  v.conservativeResize(Vector::RowsAtCompileTime == 1 ? 1 : 8, Vector::RowsAtCompileTime == 1 ? 8 : 1);
  VERIFY_IS_EQUAL(ResizeScalar::copies, 0);
  VERIFY_IS_EQUAL(ResizeScalar::moves, 0);
  Vector other(12);
  for (Index i = 0; i < other.size(); ++i) other(i).value = int(100 + i);
  v.conservativeResizeLike(other);
  VERIFY_IS_EQUAL(ResizeScalar::copies, 4);
  VERIFY_IS_EQUAL(ResizeScalar::moves, 0);
  for (Index i = 0; i < v.size(); ++i) VERIFY_IS_EQUAL(v(i).value, i < 8 ? i + 1 : 100 + i);
  other.resize(4);
  ResizeScalar::reset();
  v.conservativeResizeLike(other);
  VERIFY_IS_EQUAL(ResizeScalar::copies, 0);
  VERIFY_IS_EQUAL(ResizeScalar::moves, 0);
  for (Index i = 0; i < v.size(); ++i) VERIFY_IS_EQUAL(v(i).value, i + 1);
  VERIFY_RAISES_ASSERT(v.conservativeResize(2, 2));
  VERIFY_RAISES_ASSERT(
      v.conservativeResize(Vector::RowsAtCompileTime == 1 ? 1 : -1, Vector::RowsAtCompileTime == 1 ? -1 : 1));
  VERIFY_RAISES_ASSERT(
      v.conservativeResize(Vector::RowsAtCompileTime == 1 ? 1 : 17, Vector::RowsAtCompileTime == 1 ? 17 : 1));
  v.conservativeResize(Vector::RowsAtCompileTime == 1 ? 1 : 0, Vector::RowsAtCompileTime == 1 ? 0 : 1);
  VERIFY_IS_EQUAL(ResizeScalar::copies, 0);
  VERIFY_IS_EQUAL(ResizeScalar::moves, 0);
  VERIFY(v.data() == data);
}

template <typename Vector>
void heap_vector_resize() {
  for (bool no_change : {false, true}) {
    Vector v = Vector::LinSpaced(3, 1, 3);
    for (Index size : {5, 2, 0, 4}) {
      const Index common_size = (std::min)(size, v.size());
      if (!no_change) {
        v.conservativeResize(Vector::RowsAtCompileTime == 1 ? 1 : size, Vector::RowsAtCompileTime == 1 ? size : 1);
      } else if (Vector::RowsAtCompileTime == 1) {
        v.conservativeResize(NoChange, size);
      } else {
        v.conservativeResize(size, NoChange);
      }
      VERIFY_IS_EQUAL(v.rows(), Vector::RowsAtCompileTime == 1 ? 1 : size);
      VERIFY_IS_EQUAL(v.cols(), Vector::RowsAtCompileTime == 1 ? size : 1);
      for (Index i = 0; i < common_size; ++i) VERIFY_IS_EQUAL(v(i), i + 1);
      for (Index i = common_size; i < size; ++i) v(i) = typename Vector::Scalar(i + 1);
    }
    // A zero fixed dimension used to free the buffer without changing the vector's length.
    if (Vector::RowsAtCompileTime == 1) {
      VERIFY_RAISES_ASSERT(v.conservativeResize(0, v.cols()));
      VERIFY_RAISES_ASSERT(v.conservativeResize(0, NoChange));
    } else {
      VERIFY_RAISES_ASSERT(v.conservativeResize(v.rows(), 0));
      VERIFY_RAISES_ASSERT(v.conservativeResize(NoChange, 0));
    }
    VERIFY_IS_EQUAL(v.size(), 4);
    for (Index i = 0; i < v.size(); ++i) VERIFY_IS_EQUAL(v(i), i + 1);
  }
}

EIGEN_DECLARE_TEST(conservative_resize) {
  CALL_SUBTEST_7((same_shape_vector_resize<Matrix<InitializedResizeScalar, Dynamic, 1>>()));
  CALL_SUBTEST_7((same_shape_vector_resize<Matrix<InitializedResizeScalar, 1, Dynamic>>()));
  CALL_SUBTEST_7((same_shape_vector_resize<Array<InitializedResizeScalar, Dynamic, 1>>()));
  CALL_SUBTEST_7((same_shape_vector_resize<Array<InitializedResizeScalar, 1, Dynamic>>()));
  CALL_SUBTEST_7((two_dimensional_vector_resize<Matrix<ResizeScalar, Dynamic, 1, ColMajor, 16, 1>>()));
  CALL_SUBTEST_7((two_dimensional_vector_resize<Matrix<ResizeScalar, 1, Dynamic, RowMajor, 1, 16>>()));
  CALL_SUBTEST_7((two_dimensional_vector_resize<Array<ResizeScalar, Dynamic, 1, ColMajor, 16, 1>>()));
  CALL_SUBTEST_7((two_dimensional_vector_resize<Array<ResizeScalar, 1, Dynamic, RowMajor, 1, 16>>()));
  CALL_SUBTEST_7(heap_vector_resize<VectorXd>());
  CALL_SUBTEST_7(heap_vector_resize<RowVectorXd>());
  CALL_SUBTEST_7(heap_vector_resize<ArrayXd>());
  CALL_SUBTEST_7((heap_vector_resize<Array<double, 1, Dynamic>>()));
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1((run_matrix_tests<int, Eigen::RowMajor>()));
    CALL_SUBTEST_1((run_matrix_tests<int, Eigen::ColMajor>()));
    CALL_SUBTEST_2((run_matrix_tests<float, Eigen::RowMajor>()));
    CALL_SUBTEST_2((run_matrix_tests<float, Eigen::ColMajor>()));
    CALL_SUBTEST_3((run_matrix_tests<double, Eigen::RowMajor>()));
    CALL_SUBTEST_3((run_matrix_tests<double, Eigen::ColMajor>()));
    CALL_SUBTEST_4((run_matrix_tests<std::complex<float>, Eigen::RowMajor>()));
    CALL_SUBTEST_4((run_matrix_tests<std::complex<float>, Eigen::ColMajor>()));
    CALL_SUBTEST_5((run_matrix_tests<std::complex<double>, Eigen::RowMajor>()));
    CALL_SUBTEST_5((run_matrix_tests<std::complex<double>, Eigen::ColMajor>()));
    CALL_SUBTEST_1((run_matrix_tests<int, Eigen::RowMajor | Eigen::DontAlign>()));

    CALL_SUBTEST_1((run_vector_tests<int>()));
    CALL_SUBTEST_2((run_vector_tests<float>()));
    CALL_SUBTEST_3((run_vector_tests<double>()));
    CALL_SUBTEST_3((max_size_checks<0>()));
    CALL_SUBTEST_4((run_vector_tests<std::complex<float> >()));
    CALL_SUBTEST_5((run_vector_tests<std::complex<double> >()));

#ifndef EIGEN_TEST_ANNOYING_SCALAR_DONT_THROW
    AnnoyingScalar::dont_throw = true;
#endif
    CALL_SUBTEST_6((run_vector_tests<AnnoyingScalar>()));
    CALL_SUBTEST_6((noncopyable<0>()));

    CALL_SUBTEST_7((fixed_capacity_vector_resize<Array<ResizeScalar, Dynamic, 1, ColMajor, 8, 1>>()));
    CALL_SUBTEST_7((fixed_capacity_vector_resize<Matrix<ResizeScalar, 1, Dynamic, RowMajor, 1, 8>>()));
    CALL_SUBTEST_7((fixed_capacity_matrix_resize<Matrix<ResizeScalar, Dynamic, Dynamic, ColMajor, 5, 5>>()));
    CALL_SUBTEST_7((fixed_capacity_matrix_resize<Array<ResizeScalar, Dynamic, Dynamic, RowMajor, 5, 5>>()));
    CALL_SUBTEST_7((fixed_capacity_matrix_resize<Matrix<ResizeScalar, 3, Dynamic, ColMajor, 3, 5>>()));
    CALL_SUBTEST_7((fixed_capacity_matrix_resize<Array<ResizeScalar, Dynamic, 3, RowMajor, 5, 3>>()));
    CALL_SUBTEST_7((run_matrix_tests<AnnoyingScalar, RowMajor, 75>()));
    CALL_SUBTEST_7((run_matrix_tests<AnnoyingScalar, ColMajor, 75>()));
  }
}
