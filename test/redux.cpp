// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define TEST_ENABLE_TEMPORARY_TRACKING
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 8
// ^^ see bug 1449

#include "main.h"

    template <typename MatrixType>
    void matrixRedux(const MatrixType& m) {
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols);

  // The entries of m1 are uniformly distributed in [-1,1), so m1.prod() is very small. This may lead to test
  // failures if we underflow into denormals. Thus, we scale so that entries are close to 1.
  MatrixType m1_for_prod = MatrixType::Ones(rows, cols) + RealScalar(0.2) * m1;

  Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> m2(rows, rows);
  m2.setRandom();
  // Prevent overflows for integer types.
  if (Eigen::NumTraits<Scalar>::IsInteger) {
    Scalar kMaxVal = Scalar(10000);
    m1.array() = m1.array() - kMaxVal * (m1.array() / kMaxVal);
    m2.array() = m2.array() - kMaxVal * (m2.array() / kMaxVal);
  }

  VERIFY_IS_EQUAL(MatrixType::Zero(rows, cols).sum(), Scalar(0));
  Scalar sizeAsScalar = internal::cast<Index, Scalar>(rows * cols);
  VERIFY_IS_APPROX(MatrixType::Ones(rows, cols).sum(), sizeAsScalar);
  Scalar s(0), p(1), minc(numext::real(m1.coeff(0))), maxc(numext::real(m1.coeff(0)));
  for (int j = 0; j < cols; j++)
    for (int i = 0; i < rows; i++) {
      s += m1(i, j);
      p *= m1_for_prod(i, j);
      minc = (std::min)(numext::real(minc), numext::real(m1(i, j)));
      maxc = (std::max)(numext::real(maxc), numext::real(m1(i, j)));
    }
  const Scalar mean = s / Scalar(RealScalar(rows * cols));

  VERIFY_IS_APPROX(m1.sum(), s);
  VERIFY_IS_APPROX(m1.mean(), mean);
  VERIFY_IS_APPROX(m1_for_prod.prod(), p);
  VERIFY_IS_APPROX(m1.real().minCoeff(), numext::real(minc));
  VERIFY_IS_APPROX(m1.real().maxCoeff(), numext::real(maxc));

  // test that partial reduction works if nested expressions is forced to evaluate early
  VERIFY_IS_APPROX((m1.matrix() * m1.matrix().transpose()).cwiseProduct(m2.matrix()).rowwise().sum().sum(),
                   (m1.matrix() * m1.matrix().transpose()).eval().cwiseProduct(m2.matrix()).rowwise().sum().sum());

  // test slice vectorization assuming assign is ok
  Index r0 = internal::random<Index>(0, rows - 1);
  Index c0 = internal::random<Index>(0, cols - 1);
  Index r1 = internal::random<Index>(r0 + 1, rows) - r0;
  Index c1 = internal::random<Index>(c0 + 1, cols) - c0;
  VERIFY_IS_APPROX(m1.block(r0, c0, r1, c1).sum(), m1.block(r0, c0, r1, c1).eval().sum());
  VERIFY_IS_APPROX(m1.block(r0, c0, r1, c1).mean(), m1.block(r0, c0, r1, c1).eval().mean());
  VERIFY_IS_APPROX(m1_for_prod.block(r0, c0, r1, c1).prod(), m1_for_prod.block(r0, c0, r1, c1).eval().prod());
  VERIFY_IS_APPROX(m1.block(r0, c0, r1, c1).real().minCoeff(), m1.block(r0, c0, r1, c1).real().eval().minCoeff());
  VERIFY_IS_APPROX(m1.block(r0, c0, r1, c1).real().maxCoeff(), m1.block(r0, c0, r1, c1).real().eval().maxCoeff());

  // regression for bug 1090
  const int R1 = MatrixType::RowsAtCompileTime >= 2 ? MatrixType::RowsAtCompileTime / 2 : 6;
  const int C1 = MatrixType::ColsAtCompileTime >= 2 ? MatrixType::ColsAtCompileTime / 2 : 6;
  if (R1 <= rows - r0 && C1 <= cols - c0) {
    VERIFY_IS_APPROX((m1.template block<R1, C1>(r0, c0).sum()), m1.block(r0, c0, R1, C1).sum());
  }

  // test empty objects
  VERIFY_IS_APPROX(m1.block(r0, c0, 0, 0).sum(), Scalar(0));
  VERIFY_IS_APPROX(m1.block(r0, c0, 0, 0).prod(), Scalar(1));

  // test nesting complex expression
  VERIFY_EVALUATION_COUNT((m1.matrix() * m1.matrix().transpose()).sum(),
                          (MatrixType::IsVectorAtCompileTime && MatrixType::SizeAtCompileTime != 1 ? 0 : 1));
  VERIFY_EVALUATION_COUNT(((m1.matrix() * m1.matrix().transpose()) + m2).sum(),
                          (MatrixType::IsVectorAtCompileTime && MatrixType::SizeAtCompileTime != 1 ? 0 : 1));
}

template <typename VectorType>
void vectorRedux(const VectorType& w) {
  using std::abs;
  typedef typename VectorType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  Index size = w.size();

  VectorType v = VectorType::Random(size);
  VectorType v_for_prod = VectorType::Ones(size) + Scalar(0.2) * v;  // see comment above declaration of m1_for_prod

  for (int i = 1; i < size; i++) {
    Scalar s(0), p(1);
    RealScalar minc(numext::real(v.coeff(0))), maxc(numext::real(v.coeff(0)));
    for (int j = 0; j < i; j++) {
      s += v[j];
      p *= v_for_prod[j];
      minc = (std::min)(minc, numext::real(v[j]));
      maxc = (std::max)(maxc, numext::real(v[j]));
    }
    VERIFY_IS_MUCH_SMALLER_THAN(abs(s - v.head(i).sum()), Scalar(1));
    VERIFY_IS_APPROX(p, v_for_prod.head(i).prod());
    VERIFY_IS_APPROX(minc, v.real().head(i).minCoeff());
    VERIFY_IS_APPROX(maxc, v.real().head(i).maxCoeff());
  }

  for (int i = 0; i < size - 1; i++) {
    Scalar s(0), p(1);
    RealScalar minc(numext::real(v.coeff(i))), maxc(numext::real(v.coeff(i)));
    for (int j = i; j < size; j++) {
      s += v[j];
      p *= v_for_prod[j];
      minc = (std::min)(minc, numext::real(v[j]));
      maxc = (std::max)(maxc, numext::real(v[j]));
    }
    VERIFY_IS_MUCH_SMALLER_THAN(abs(s - v.tail(size - i).sum()), Scalar(1));
    VERIFY_IS_APPROX(p, v_for_prod.tail(size - i).prod());
    VERIFY_IS_APPROX(minc, v.real().tail(size - i).minCoeff());
    VERIFY_IS_APPROX(maxc, v.real().tail(size - i).maxCoeff());
  }

  for (int i = 0; i < size / 2; i++) {
    Scalar s(0), p(1);
    RealScalar minc(numext::real(v.coeff(i))), maxc(numext::real(v.coeff(i)));
    for (int j = i; j < size - i; j++) {
      s += v[j];
      p *= v_for_prod[j];
      minc = (std::min)(minc, numext::real(v[j]));
      maxc = (std::max)(maxc, numext::real(v[j]));
    }
    VERIFY_IS_MUCH_SMALLER_THAN(abs(s - v.segment(i, size - 2 * i).sum()), Scalar(1));
    VERIFY_IS_APPROX(p, v_for_prod.segment(i, size - 2 * i).prod());
    VERIFY_IS_APPROX(minc, v.real().segment(i, size - 2 * i).minCoeff());
    VERIFY_IS_APPROX(maxc, v.real().segment(i, size - 2 * i).maxCoeff());
  }

  // test empty objects
  VERIFY_IS_APPROX(v.head(0).sum(), Scalar(0));
  VERIFY_IS_APPROX(v.tail(0).prod(), Scalar(1));
  VERIFY_RAISES_ASSERT(v.head(0).mean());
  VERIFY_RAISES_ASSERT(v.head(0).minCoeff());
  VERIFY_RAISES_ASSERT(v.head(0).maxCoeff());
}

void boolRedux(Index rows, Index cols) {
  // Test boolean reductions: all(), any(), count()
  typedef Array<bool, Dynamic, Dynamic> BoolArray;

  // All-true
  BoolArray all_true = BoolArray::Constant(rows, cols, true);
  VERIFY(all_true.all());
  VERIFY(all_true.any());
  VERIFY_IS_EQUAL(all_true.count(), rows * cols);

  // All-false
  BoolArray all_false = BoolArray::Constant(rows, cols, false);
  if (rows > 0 && cols > 0) {
    VERIFY(!all_false.all());
    VERIFY(!all_false.any());
  }
  VERIFY_IS_EQUAL(all_false.count(), Index(0));

  // Mixed: set a checkerboard pattern
  BoolArray mixed(rows, cols);
  Index expected_count = 0;
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i) {
      mixed(i, j) = ((i + j) % 2 == 0);
      if (mixed(i, j)) expected_count++;
    }
  VERIFY_IS_EQUAL(mixed.count(), expected_count);
  if (rows > 0 && cols > 0) {
    VERIFY(mixed.any());
    VERIFY(mixed.all() == (expected_count == rows * cols));
  }

  // Partial reductions
  if (rows > 0 && cols > 0) {
    auto col_counts = mixed.colwise().count();
    for (Index k = 0; k < cols; ++k) VERIFY_IS_EQUAL(col_counts(k), mixed.col(k).count());
    auto row_counts = mixed.rowwise().count();
    for (Index k = 0; k < rows; ++k) VERIFY_IS_EQUAL(row_counts(k), mixed.row(k).count());
  }
}

// Test reductions at sizes that hit vectorization boundaries in Redux.h:
// LinearVectorizedTraversal with 2-way unrolled packet loop, scalar pre/post loops.
template <typename Scalar>
void redux_vec_boundary() {
  const Index PS = internal::packet_traits<Scalar>::size;
  // Critical sizes: around packet multiples and at 2-way unroll boundaries
  const Index sizes[] = {1,      PS - 1,     PS,         PS + 1, 2 * PS - 1, 2 * PS, 2 * PS + 1,
                         3 * PS, 3 * PS + 1, 4 * PS - 1, 4 * PS, 4 * PS + 1, 8 * PS, 8 * PS + 1};
  for (int si = 0; si < 14; ++si) {
    const Index n = sizes[si];
    if (n <= 0) continue;
    typedef Matrix<Scalar, Dynamic, 1> Vec;
    Vec v = Vec::Random(n);
    // For prod, use values near 1 to avoid underflow (float) or overflow (int).
    Vec v_for_prod = Vec::Ones(n) + Scalar(typename NumTraits<Scalar>::Real(0.2)) * v;
    // Reference: scalar loops
    Scalar ref_sum(0), ref_prod(1);
    typename NumTraits<Scalar>::Real ref_min = numext::real(v(0)), ref_max = numext::real(v(0));
    for (Index k = 0; k < n; ++k) {
      ref_sum += v(k);
      ref_prod *= v_for_prod(k);
      ref_min = (std::min)(ref_min, numext::real(v(k)));
      ref_max = (std::max)(ref_max, numext::real(v(k)));
    }
    VERIFY_IS_APPROX(v.sum(), ref_sum);
    VERIFY_IS_APPROX(v_for_prod.prod(), ref_prod);
    VERIFY_IS_APPROX(v.real().minCoeff(), ref_min);
    VERIFY_IS_APPROX(v.real().maxCoeff(), ref_max);
  }
}

// Test reductions on strided (non-contiguous) mapped data.
// This exercises SliceVectorizedTraversal or DefaultTraversal in Redux.h
// depending on stride and packet size.
template <typename Scalar>
void redux_strided() {
  const Index n = 64;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  Vec data = Vec::Random(2 * n);
  // Map with inner stride of 2 — every other element
  Map<Vec, 0, InnerStride<2>> strided(data.data(), n);
  Scalar ref_sum(0);
  typename NumTraits<Scalar>::Real ref_min = numext::real(strided(0)), ref_max = numext::real(strided(0));
  for (Index k = 0; k < n; ++k) {
    ref_sum += strided(k);
    ref_min = (std::min)(ref_min, numext::real(strided(k)));
    ref_max = (std::max)(ref_max, numext::real(strided(k)));
  }
  VERIFY_IS_APPROX(strided.sum(), ref_sum);
  VERIFY_IS_APPROX(strided.real().minCoeff(), ref_min);
  VERIFY_IS_APPROX(strided.real().maxCoeff(), ref_max);

  // Also test reduction on a non-contiguous matrix block (SliceVectorizedTraversal)
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
  Mat m = Mat::Random(16, 16);
  for (Index bsz = 1; bsz <= 8; bsz *= 2) {
    Scalar block_sum(0);
    for (Index j = 0; j < bsz; ++j)
      for (Index i = 0; i < bsz; ++i) block_sum += m(1 + i, 1 + j);
    VERIFY_IS_APPROX(m.block(1, 1, bsz, bsz).sum(), block_sum);
  }
}

EIGEN_DECLARE_TEST(redux) {
  // the max size cannot be too large, otherwise reduxion operations obviously generate large errors.
  int maxsize = (std::min)(100, EIGEN_TEST_MAX_SIZE);
  TEST_SET_BUT_UNUSED_VARIABLE(maxsize);
  for (int i = 0; i < g_repeat; i++) {
    int rows = internal::random<int>(1, maxsize);
    int cols = internal::random<int>(1, maxsize);
    EIGEN_UNUSED_VARIABLE(rows);
    EIGEN_UNUSED_VARIABLE(cols);
    CALL_SUBTEST_1(matrixRedux(Matrix<float, 1, 1>()));
    CALL_SUBTEST_1(matrixRedux(Array<float, 1, 1>()));
    CALL_SUBTEST_2(matrixRedux(Matrix2f()));
    CALL_SUBTEST_2(matrixRedux(Array2f()));
    CALL_SUBTEST_2(matrixRedux(Array22f()));
    CALL_SUBTEST_3(matrixRedux(Matrix4d()));
    CALL_SUBTEST_3(matrixRedux(Array4d()));
    CALL_SUBTEST_3(matrixRedux(Array44d()));
    CALL_SUBTEST_4(matrixRedux(MatrixXf(rows, cols)));
    CALL_SUBTEST_4(matrixRedux(ArrayXXf(rows, cols)));
    CALL_SUBTEST_4(matrixRedux(MatrixXd(rows, cols)));
    CALL_SUBTEST_4(matrixRedux(ArrayXXd(rows, cols)));
    /* TODO: fix test for boolean */
    /*CALL_SUBTEST_5(matrixRedux(MatrixX<bool>(rows, cols)));*/
    /*CALL_SUBTEST_5(matrixRedux(ArrayXX<bool>(rows, cols)));*/
    CALL_SUBTEST_5(matrixRedux(MatrixXi(rows, cols)));
    CALL_SUBTEST_5(matrixRedux(ArrayXXi(rows, cols)));
    CALL_SUBTEST_5(matrixRedux(MatrixX<int64_t>(rows, cols)));
    CALL_SUBTEST_5(matrixRedux(ArrayXX<int64_t>(rows, cols)));
    CALL_SUBTEST_6(matrixRedux(MatrixXcf(rows, cols)));
    CALL_SUBTEST_6(matrixRedux(ArrayXXcf(rows, cols)));
    CALL_SUBTEST_7(matrixRedux(MatrixXcd(rows, cols)));
    CALL_SUBTEST_7(matrixRedux(ArrayXXcd(rows, cols)));
  }
  for (int i = 0; i < g_repeat; i++) {
    int size = internal::random<int>(1, maxsize);
    EIGEN_UNUSED_VARIABLE(size);
    CALL_SUBTEST_8(vectorRedux(Vector4f()));
    CALL_SUBTEST_8(vectorRedux(Array4f()));
    CALL_SUBTEST_9(vectorRedux(VectorXf(size)));
    CALL_SUBTEST_9(vectorRedux(ArrayXf(size)));
    CALL_SUBTEST_10(vectorRedux(VectorXd(size)));
    CALL_SUBTEST_10(vectorRedux(ArrayXd(size)));
    /* TODO: fix test for boolean */
    /*CALL_SUBTEST_10(vectorRedux(VectorX<bool>(size)));*/
    /*CALL_SUBTEST_10(vectorRedux(ArrayX<bool>(size)));*/
    CALL_SUBTEST_10(vectorRedux(VectorXi(size)));
    CALL_SUBTEST_10(vectorRedux(ArrayXi(size)));
    CALL_SUBTEST_10(vectorRedux(VectorX<int64_t>(size)));
    CALL_SUBTEST_10(vectorRedux(ArrayX<int64_t>(size)));
  }
  // Bool reductions (deterministic, outside g_repeat)
  CALL_SUBTEST_11(boolRedux(1, 1));
  CALL_SUBTEST_11(boolRedux(4, 4));
  CALL_SUBTEST_11(boolRedux(7, 13));
  CALL_SUBTEST_11(boolRedux(63, 63));

  // Bool reductions at vectorization boundary sizes.
  // all()/any()/count() use packet-level visitors with remainder handling.
  {
    // bool packets are typically 16 bytes (SSE) or 32 bytes (AVX).
    // Test sizes around common packet sizes to catch off-by-one in remainder loops.
    const Index bsizes[] = {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129};
    EIGEN_UNUSED_VARIABLE(bsizes);
    for (int si = 0; si < 18; ++si) {
      CALL_SUBTEST_11(boolRedux(bsizes[si], 1));  // column vector
      CALL_SUBTEST_11(boolRedux(1, bsizes[si]));  // row vector
      CALL_SUBTEST_11(boolRedux(bsizes[si], 3));  // thin matrix
    }
  }

  // Vectorization boundary sizes — deterministic, run once.
  // Integer types are excluded: full-range random ints overflow in sum/prod (UB).
  // Integer reductions are already tested by matrixRedux/vectorRedux with clamped values.
  CALL_SUBTEST_12(redux_vec_boundary<float>());
  CALL_SUBTEST_12(redux_vec_boundary<double>());

  // Strided (non-contiguous) reductions.
  CALL_SUBTEST_13(redux_strided<float>());
  CALL_SUBTEST_13(redux_strided<double>());
  CALL_SUBTEST_13(redux_strided<std::complex<float>>());
}
