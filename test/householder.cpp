// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include "fp_control.h"
#include "NonCommutativeScalar.h"
#include <Eigen/QR>

template <int Tag>
struct HouseholderResultScalar {};

namespace Eigen {
template <>
struct ScalarBinaryOpTraits<HouseholderResultScalar<0>, HouseholderResultScalar<1>> {
  using ReturnType = HouseholderResultScalar<2>;
};
template <>
struct ScalarBinaryOpTraits<HouseholderResultScalar<1>, HouseholderResultScalar<0>> {
  using ReturnType = HouseholderResultScalar<3>;
};
}  // namespace Eigen

template <int Side>
void householder_ordered_result_types() {
  using Vectors = Matrix<HouseholderResultScalar<0>, 2, 2>;
  using Coefficients = Matrix<HouseholderResultScalar<0>, 2, 1>;
  using Sequence = HouseholderSequence<Vectors, Coefficients, Side>;
  using Dense = Matrix<HouseholderResultScalar<1>, 2, 2>;
  using Diagonal = DiagonalMatrix<HouseholderResultScalar<1>, 2>;
  using Triangular = TriangularView<Dense, Upper>;
  using Left = decltype(std::declval<const Sequence&>() * std::declval<const Dense&>());
  using Right = decltype(std::declval<const Dense&>() * std::declval<const Sequence&>());
  using DiagonalLeft = decltype(std::declval<const Sequence&>() * std::declval<const Diagonal&>());
  using DiagonalRight = decltype(std::declval<const Diagonal&>() * std::declval<const Sequence&>());
  using TriangularLeft = decltype(std::declval<const Sequence&>() * std::declval<const Triangular&>());
  using TriangularRight = decltype(std::declval<const Triangular&>() * std::declval<const Sequence&>());
  static_assert(std::is_same<typename Left::Scalar, HouseholderResultScalar<2>>::value, "sequence * dense");
  static_assert(std::is_same<typename Right::Scalar, HouseholderResultScalar<3>>::value, "dense * sequence");
  static_assert(std::is_same<typename DiagonalLeft::Scalar, HouseholderResultScalar<2>>::value, "sequence * diagonal");
  static_assert(std::is_same<typename DiagonalRight::Scalar, HouseholderResultScalar<3>>::value, "diagonal * sequence");
  static_assert(std::is_same<typename TriangularLeft::Scalar, HouseholderResultScalar<2>>::value,
                "sequence * triangular");
  static_assert(std::is_same<typename TriangularRight::Scalar, HouseholderResultScalar<3>>::value,
                "triangular * sequence");
}

namespace reordered_complex {

template <typename Real>
struct Complex {
  Complex() : imag_value(0), real_value(0) {}
  Complex(Real real) : imag_value(0), real_value(real) {}
  Complex(Real real, Real imag) : imag_value(imag), real_value(real) {}

  Complex operator+(const Complex& other) const {
    return Complex(real_value + other.real_value, imag_value + other.imag_value);
  }
  Complex operator-() const { return Complex(-real_value, -imag_value); }
  Complex operator-(const Complex& other) const {
    return Complex(real_value - other.real_value, imag_value - other.imag_value);
  }
  Complex operator*(const Complex& other) const {
    return Complex(real_value * other.real_value - imag_value * other.imag_value,
                   real_value * other.imag_value + imag_value * other.real_value);
  }
  Complex operator/(const Complex& other) const {
    const Real denominator = other.real_value * other.real_value + other.imag_value * other.imag_value;
    return Complex((real_value * other.real_value + imag_value * other.imag_value) / denominator,
                   (imag_value * other.real_value - real_value * other.imag_value) / denominator);
  }

  bool operator==(const Complex& other) const {
    return real_value == other.real_value && imag_value == other.imag_value;
  }
  bool operator!=(const Complex& other) const { return !(*this == other); }

  friend Complex operator+(const Real& lhs, const Complex& rhs) {
    return Complex(lhs + rhs.real_value, rhs.imag_value);
  }
  friend Complex operator-(const Real& lhs, const Complex& rhs) {
    return Complex(lhs - rhs.real_value, -rhs.imag_value);
  }
  friend Complex operator*(const Real& lhs, const Complex& rhs) {
    return Complex(lhs * rhs.real_value, lhs * rhs.imag_value);
  }
  friend Complex operator*(const Complex& lhs, const Real& rhs) {
    return Complex(lhs.real_value * rhs, lhs.imag_value * rhs);
  }
  friend Complex operator/(const Complex& lhs, const Real& rhs) {
    return Complex(lhs.real_value / rhs, lhs.imag_value / rhs);
  }

  // The reversed order is intentional: algorithms must not infer component access from IsComplex.
  Real imag_value;
  Real real_value;
};

template <typename Real>
Real real(const Complex<Real>& value) {
  return value.real_value;
}
template <typename Real>
Real imag(const Complex<Real>& value) {
  return value.imag_value;
}
template <typename Real>
Complex<Real> conj(const Complex<Real>& value) {
  return Complex<Real>(value.real_value, -value.imag_value);
}
template <typename Real>
Real abs(const Complex<Real>& value) {
  return numext::hypot(value.real_value, value.imag_value);
}

}  // namespace reordered_complex

namespace Eigen {
template <typename Real>
struct NumTraits<reordered_complex::Complex<Real>> : NumTraits<Real> {
  static constexpr bool IsComplex = true;
};
}  // namespace Eigen

template <typename MatrixType>
void householder(const MatrixType& m) {
  static bool even = true;
  even = !even;
  /* this test covers the following files:
     Householder.h
  */
  Index rows = m.rows();
  Index cols = m.cols();

  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using VectorType = Matrix<Scalar, MatrixType::RowsAtCompileTime, 1>;
  using EssentialVectorType = Matrix<Scalar, internal::decrement_size<MatrixType::RowsAtCompileTime>::value, 1>;
  using SquareMatrixType = Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime>;
  using HBlockMatrixType = Matrix<Scalar, Dynamic, MatrixType::ColsAtCompileTime>;
  using HCoeffsVectorType = Matrix<Scalar, Dynamic, 1>;

  using TMatrixType = Matrix<Scalar, MatrixType::ColsAtCompileTime, MatrixType::RowsAtCompileTime>;

  Matrix<Scalar, internal::max_size_prefer_dynamic(MatrixType::RowsAtCompileTime, MatrixType::ColsAtCompileTime), 1>
      _tmp((std::max)(rows, cols));
  Scalar* tmp = &_tmp.coeffRef(0, 0);

  Scalar beta;
  RealScalar alpha;
  EssentialVectorType essential;

  VectorType v1 = VectorType::Random(rows), v2;
  v2 = v1;
  v1.makeHouseholder(essential, beta, alpha);
  v1.applyHouseholderOnTheLeft(essential, beta, tmp);
  VERIFY_IS_APPROX(v1.norm(), v2.norm());
  if (rows >= 2) VERIFY_IS_MUCH_SMALLER_THAN(v1.tail(rows - 1).norm(), v1.norm());
  v1 = VectorType::Random(rows);
  v2 = v1;
  v1.applyHouseholderOnTheLeft(essential, beta, tmp);
  VERIFY_IS_APPROX(v1.norm(), v2.norm());

  // reconstruct householder matrix:
  SquareMatrixType id, H1, H2;
  id.setIdentity(rows, rows);
  H1 = H2 = id;
  VectorType vv(rows);
  vv << Scalar(1), essential;
  H1.applyHouseholderOnTheLeft(essential, beta, tmp);
  H2.applyHouseholderOnTheRight(essential, beta, tmp);
  VERIFY_IS_APPROX(H1, H2);
  VERIFY_IS_APPROX(H1, id - beta * vv * vv.adjoint());

  MatrixType m1(rows, cols), m2(rows, cols);

  v1 = VectorType::Random(rows);
  if (even) v1.tail(rows - 1).setZero();
  m1.colwise() = v1;
  m2 = m1;
  m1.col(0).makeHouseholder(essential, beta, alpha);
  m1.applyHouseholderOnTheLeft(essential, beta, tmp);
  VERIFY_IS_APPROX(m1.norm(), m2.norm());
  if (rows >= 2) VERIFY_IS_MUCH_SMALLER_THAN(m1.block(1, 0, rows - 1, cols).norm(), m1.norm());
  VERIFY_IS_MUCH_SMALLER_THAN(numext::imag(m1(0, 0)), numext::real(m1(0, 0)));
  VERIFY_IS_APPROX(numext::real(m1(0, 0)), alpha);

  v1 = VectorType::Random(rows);
  if (even) v1.tail(rows - 1).setZero();
  SquareMatrixType m3(rows, rows), m4(rows, rows);
  m3.rowwise() = v1.transpose();
  m4 = m3;
  m3.row(0).makeHouseholder(essential, beta, alpha);
  m3.applyHouseholderOnTheRight(essential.conjugate(), beta, tmp);
  VERIFY_IS_APPROX(m3.norm(), m4.norm());
  if (rows >= 2) VERIFY_IS_MUCH_SMALLER_THAN(m3.block(0, 1, rows, rows - 1).norm(), m3.norm());
  VERIFY_IS_MUCH_SMALLER_THAN(numext::imag(m3(0, 0)), numext::real(m3(0, 0)));
  VERIFY_IS_APPROX(numext::real(m3(0, 0)), alpha);

  // test householder sequence on the left with a shift

  Index shift = internal::random<Index>(0, std::max<Index>(rows - 2, 0));
  Index brows = rows - shift;
  m1.setRandom(rows, cols);
  HBlockMatrixType hbm = m1.block(shift, 0, brows, cols);
  HouseholderQR<HBlockMatrixType> qr(hbm);
  m2 = m1;
  m2.block(shift, 0, brows, cols) = qr.matrixQR();
  HCoeffsVectorType hc = qr.hCoeffs().conjugate();
  HouseholderSequence<MatrixType, HCoeffsVectorType> hseq(m2, hc);
  hseq.setLength(hc.size()).setShift(shift);
  VERIFY(hseq.length() == hc.size());
  VERIFY(hseq.shift() == shift);

  MatrixType m5 = m2;
  m5.block(shift, 0, brows, cols).template triangularView<StrictlyLower>().setZero();
  VERIFY_IS_APPROX(hseq * m5, m1);  // test applying hseq directly
  m3 = hseq;
  VERIFY_IS_APPROX(m3 * m5, m1);  // test evaluating hseq to a dense matrix, then applying

  SquareMatrixType hseq_mat = hseq;
  SquareMatrixType hseq_mat_conj = hseq.conjugate();
  SquareMatrixType hseq_mat_adj = hseq.adjoint();
  SquareMatrixType hseq_mat_trans = hseq.transpose();
  SquareMatrixType m6 = SquareMatrixType::Random(rows, rows);
  VERIFY_IS_APPROX(hseq_mat.adjoint(), hseq_mat_adj);
  VERIFY_IS_APPROX(hseq_mat.conjugate(), hseq_mat_conj);
  VERIFY_IS_APPROX(hseq_mat.transpose(), hseq_mat_trans);
  VERIFY_IS_APPROX(hseq * m6, hseq_mat * m6);
  VERIFY_IS_APPROX(hseq.adjoint() * m6, hseq_mat_adj * m6);
  VERIFY_IS_APPROX(hseq.conjugate() * m6, hseq_mat_conj * m6);
  VERIFY_IS_APPROX(hseq.transpose() * m6, hseq_mat_trans * m6);
  VERIFY_IS_APPROX(m6 * hseq, m6 * hseq_mat);
  VERIFY_IS_APPROX(m6 * hseq.adjoint(), m6 * hseq_mat_adj);
  VERIFY_IS_APPROX(m6 * hseq.conjugate(), m6 * hseq_mat_conj);
  VERIFY_IS_APPROX(m6 * hseq.transpose(), m6 * hseq_mat_trans);

  // test householder sequence on the right with a shift

  TMatrixType tm2 = m2.transpose();
  HouseholderSequence<TMatrixType, HCoeffsVectorType, OnTheRight> rhseq(tm2, hc);
  rhseq.setLength(hc.size()).setShift(shift);
  VERIFY_IS_APPROX(rhseq * m5, m1);  // test applying rhseq directly
  m3 = rhseq;
  VERIFY_IS_APPROX(m3 * m5, m1);  // test evaluating rhseq to a dense matrix, then applying
}

// A row-vector operand gives a row-major row-vector result; a column-major one fails to instantiate.
template <typename Scalar>
void householder_row_vector_products(Index size) {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  using RowVectorType = Matrix<Scalar, 1, Dynamic>;

  const HouseholderQR<MatrixType> qr(MatrixType::Random(size, size));
  const MatrixType q = qr.householderQ();
  const RowVectorType row = RowVectorType::Random(size);
  STATIC_CHECK((internal::is_same<decltype(row * qr.householderQ()), RowVectorType>::value));
  RowVectorType result = row * qr.householderQ();
  VERIFY_IS_APPROX(result, row * q);
  result = row * qr.householderQ().adjoint();
  VERIFY_IS_APPROX(result, row * q.adjoint());

  const HouseholderQR<MatrixType> qr1(MatrixType::Random(1, 1));
  const MatrixType q1 = qr1.householderQ();
  STATIC_CHECK((internal::is_same<decltype(qr1.householderQ() * row), RowVectorType>::value));
  result = qr1.householderQ() * row;
  VERIFY_IS_APPROX(result, q1 * row);

  using Matrix3 = Matrix<Scalar, 3, 3>;
  using RowVector3 = Matrix<Scalar, 1, 3>;
  const HouseholderQR<Matrix3> qr3(Matrix3::Random());
  const RowVector3 row3 = RowVector3::Random();
  STATIC_CHECK((internal::is_same<decltype(row3 * qr3.householderQ()), RowVector3>::value));
  VERIFY_IS_APPROX(RowVector3(row3 * qr3.householderQ()), RowVector3(row3 * Matrix3(qr3.householderQ())));

  // A single-column reflector set makes the blocked right-side tmp N x 1; it is instantiated although one
  // reflector never takes that path at run time.
  using VectorType = Matrix<Scalar, Dynamic, 1>;
  const HouseholderQR<VectorType> qrv(VectorType::Random(size));
  const MatrixType qv = qrv.householderQ();
  const MatrixType m = MatrixType::Random(3, size);
  VERIFY_IS_APPROX(MatrixType(m * qrv.householderQ()), m * qv);
}

template <typename MatrixType>
void householder_update(const MatrixType& m) {
  // This test is covering the internal::householder_qr_inplace_update function.
  // At time of writing, there is not public API that exposes this update behavior directly,
  // so we are testing the internal implementation.

  const Index rows = m.rows();
  const Index cols = m.cols();

  using Scalar = typename MatrixType::Scalar;
  using VectorType = Matrix<Scalar, MatrixType::RowsAtCompileTime, 1>;
  using HCoeffsVectorType = Matrix<Scalar, Dynamic, 1>;
  using MatrixX = Matrix<Scalar, Dynamic, Dynamic>;
  using VectorX = Matrix<Scalar, Dynamic, 1>;

  VectorX tmpOwner(cols);
  Scalar* tmp = tmpOwner.data();

  // The matrix to factorize.
  const MatrixType A = MatrixType::Random(rows, cols);

  // matQR and hCoeffs will hold the factorization of A,
  // built by a sequence of calls to `update`.
  MatrixType matQR(rows, cols);
  HCoeffsVectorType hCoeffs(cols);

  // householder_qr_inplace_update should be able to build a QR factorization one column at a time.
  // We verify this by starting with an empty factorization and 'updating' one column at a time.
  // After each call to update, we should have a QR factorization of the columns presented so far.

  const Index size = (std::min)(rows, cols);  // QR can only go up to 'size' b/c that's full rank.
  for (Index k = 0; k != size; ++k) {
    // Make a copy of the column to prevent any possibility of 'leaking' other parts of A.
    const VectorType newColumn = A.col(k);
    internal::householder_qr_inplace_update(matQR, hCoeffs, newColumn, k, tmp);

    // Verify Property:
    // matQR.leftCols(k+1) and hCoeffs.head(k+1) hold
    // a QR factorization of A.leftCols(k+1).
    // This is the fundamental guarantee of householder_qr_inplace_update.
    {
      const MatrixX matQR_k = matQR.leftCols(k + 1);
      const VectorX hCoeffs_k = hCoeffs.head(k + 1);
      MatrixX R = matQR_k.template triangularView<Upper>();
      MatrixX QxR = householderSequence(matQR_k, hCoeffs_k.conjugate()) * R;
      VERIFY_IS_APPROX(QxR, A.leftCols(k + 1));
    }

    // Verify Property:
    // A sequence of calls to 'householder_qr_inplace_update'
    // should produce the same result as 'householder_qr_inplace_unblocked'.
    // This is a property of the current implementation.
    // If these implementations diverge in the future,
    // then simply delete the test of this property.
    {
      MatrixX QR_at_once = A.leftCols(k + 1);
      VectorX hCoeffs_at_once(k + 1);
      internal::householder_qr_inplace_unblocked(QR_at_once, hCoeffs_at_once, tmp);
      VERIFY_IS_APPROX(QR_at_once, matQR.leftCols(k + 1));
      VERIFY_IS_APPROX(hCoeffs_at_once, hCoeffs.head(k + 1));
    }
  }

  // Verify Property:
  // We can go back and update any column to have a new value,
  // and get a QR factorization of the columns up to that one.
  {
    const Index k = internal::random<Index>(0, size - 1);
    VectorType newColumn = VectorType::Random(rows);
    internal::householder_qr_inplace_update(matQR, hCoeffs, newColumn, k, tmp);

    const MatrixX matQR_k = matQR.leftCols(k + 1);
    const VectorX hCoeffs_k = hCoeffs.head(k + 1);
    MatrixX R = matQR_k.template triangularView<Upper>();
    MatrixX QxR = householderSequence(matQR_k, hCoeffs_k.conjugate()) * R;
    VERIFY_IS_APPROX(QxR.leftCols(k), A.leftCols(k));
    VERIFY_IS_APPROX(QxR.col(k), newColumn);
  }
}

template <typename Scalar>
void householder_blocked_right_regression() {
  using MatrixX = Matrix<Scalar, Dynamic, Dynamic>;
  using VectorX = Matrix<Scalar, Dynamic, 1>;
  using LeftSequence = HouseholderSequence<MatrixX, VectorX>;
  using RightSequence = HouseholderSequence<MatrixX, VectorX, OnTheRight>;

  const Index rows = 256;
  const Index cols = 128;
  const Index shift = 17;

  // Force the blocked path added for right-side Householder application.
  VERIFY(cols >= 48 && rows - shift >= 4 * 48);

  MatrixX input = MatrixX::Random(rows, cols);
  MatrixX qr_input = input.block(shift, 0, rows - shift, cols);
  HouseholderQR<MatrixX> qr(qr_input);

  MatrixX packed = input;
  packed.block(shift, 0, rows - shift, cols) = qr.matrixQR();
  VectorX hcoeffs = qr.hCoeffs().conjugate();

  LeftSequence hseq(packed, hcoeffs);
  hseq.setLength(hcoeffs.size()).setShift(shift);
  MatrixX dense_left = MatrixX(hseq);

  MatrixX packed_transposed = packed.transpose();
  RightSequence rhseq(packed_transposed, hcoeffs);
  rhseq.setLength(hcoeffs.size()).setShift(shift);
  MatrixX dense_right = MatrixX(rhseq);

  MatrixX left_rhs = MatrixX::Random(rows, rows + 9);
  MatrixX right_lhs = MatrixX::Random(rows + 7, rows);

  VERIFY_IS_APPROX(hseq * left_rhs, dense_left * left_rhs);
  VERIFY_IS_APPROX(hseq.adjoint() * left_rhs, dense_left.adjoint() * left_rhs);
  VERIFY_IS_APPROX(right_lhs * hseq, right_lhs * dense_left);
  VERIFY_IS_APPROX(right_lhs * hseq.adjoint(), right_lhs * dense_left.adjoint());

  VERIFY_IS_APPROX(rhseq * left_rhs, dense_right * left_rhs);
  VERIFY_IS_APPROX(rhseq.adjoint() * left_rhs, dense_right.adjoint() * left_rhs);
  VERIFY_IS_APPROX(right_lhs * rhseq, right_lhs * dense_right);
  VERIFY_IS_APPROX(right_lhs * rhseq.adjoint(), right_lhs * dense_right.adjoint());
}

template <typename VectorType, typename EssentialType>
void verify_real_householder_result(const VectorType& vector, const EssentialType& essential,
                                    const typename VectorType::Scalar& tau,
                                    const typename VectorType::RealScalar& beta) {
  using RealScalar = typename VectorType::RealScalar;
  const long double epsilon = static_cast<long double>(NumTraits<RealScalar>::epsilon());
  long double input_squared_norm = 0;
  long double householder_squared_norm = 1;
  long double inner_product = static_cast<long double>(vector.coeff(0));
  for (Index i = 0; i < vector.size(); ++i) {
    const long double value = static_cast<long double>(vector.coeff(i));
    input_squared_norm += value * value;
    if (i > 0) {
      const long double essential_value = static_cast<long double>(essential.coeff(i - 1));
      householder_squared_norm += essential_value * essential_value;
      inner_product += essential_value * value;
    }
  }

  const long double input_norm = std::sqrt(input_squared_norm);
  const long double tau_wide = static_cast<long double>(tau);
  const long double bound = 64 * epsilon * input_norm;
  const long double transformed_head = static_cast<long double>(vector.coeff(0)) - tau_wide * inner_product;
  VERIFY(std::abs(transformed_head - static_cast<long double>(beta)) <= bound);

  long double transformed_tail_squared_norm = 0;
  for (Index i = 1; i < vector.size(); ++i) {
    const long double essential_value = static_cast<long double>(essential.coeff(i - 1));
    const long double transformed =
        static_cast<long double>(vector.coeff(i)) - tau_wide * essential_value * inner_product;
    transformed_tail_squared_norm += transformed * transformed;
  }
  VERIFY(std::sqrt(transformed_tail_squared_norm) <= bound);
  VERIFY(std::abs(2 * tau_wide - tau_wide * tau_wide * householder_squared_norm) <= 64 * epsilon);
}

template <typename RealScalar>
void verify_complex_zero_tail(RealScalar coefficient) {
  using Scalar = std::complex<RealScalar>;
  using WideScalar = std::complex<long double>;
  Matrix<Scalar, 2, 1> vector;
  vector << Scalar(0, coefficient), Scalar(0, 0);
  Matrix<Scalar, 1, 1> essential;
  Scalar tau;
  RealScalar beta;

  vector.makeHouseholder(essential, tau, beta);

  const WideScalar x0(0, static_cast<long double>(coefficient));
  const WideScalar x1(0, 0);
  const WideScalar essential_wide(static_cast<long double>(numext::real(essential[0])),
                                  static_cast<long double>(numext::imag(essential[0])));
  const WideScalar tau_wide(static_cast<long double>(numext::real(tau)), static_cast<long double>(numext::imag(tau)));
  const WideScalar inner_product = x0 + std::conj(essential_wide) * x1;
  const WideScalar transformed_head = x0 - tau_wide * inner_product;
  const WideScalar transformed_tail = x1 - tau_wide * essential_wide * inner_product;
  const long double epsilon = static_cast<long double>(NumTraits<RealScalar>::epsilon());
  const long double bound = 16 * epsilon * std::abs(x0);

  VERIFY(std::abs(transformed_head - WideScalar(static_cast<long double>(beta), 0)) <= bound);
  VERIFY(std::abs(transformed_tail) <= bound);
  VERIFY(std::abs(2 * numext::real(tau_wide) - std::norm(tau_wide) * (1 + std::norm(essential_wide))) <= 16 * epsilon);
}

template <typename RealScalar>
void verify_low_precision_complex_zero_tail() {
  using Scalar = std::complex<RealScalar>;
  const RealScalar zero = RealScalar(0);
  const RealScalar coefficient = RealScalar(0.25f);
  Matrix<Scalar, 2, 1> vector;
  vector << Scalar(zero, coefficient), Scalar(zero, zero);
  Matrix<Scalar, 1, 1> essential;
  Scalar tau;
  RealScalar beta;

  vector.makeHouseholder(essential, tau, beta);

  VERIFY_IS_EQUAL(numext::real(tau), RealScalar(1));
  VERIFY_IS_EQUAL(numext::imag(tau), RealScalar(-1));
  VERIFY_IS_EQUAL(beta, -coefficient);
  VERIFY_IS_EQUAL(numext::real(essential[0]), zero);
  VERIFY_IS_EQUAL(numext::imag(essential[0]), zero);
}

template <typename = void>
void verify_custom_complex_small_tail() {
  using Scalar = reordered_complex::Complex<float>;
  const float coefficient = 2e-16f;
  Matrix<Scalar, 2, 1> vector;
  vector << Scalar(0), Scalar(coefficient);
  Matrix<Scalar, 1, 1> essential;
  Scalar tau;
  float beta;

  vector.makeHouseholder(essential, tau, beta);

  VERIFY_IS_EQUAL(numext::real(tau), 1.0f);
  VERIFY_IS_EQUAL(numext::imag(tau), 0.0f);
  VERIFY_IS_EQUAL(beta, -coefficient);
  VERIFY_IS_EQUAL(numext::real(essential[0]), 1.0f);
  VERIFY_IS_EQUAL(numext::imag(essential[0]), 0.0f);
}

template <typename = void>
EIGEN_DONT_INLINE void verify_householder_flushed_tail() {
  volatile float normal_min_input = (std::numeric_limits<float>::min)();
  const float normal_min = normal_min_input;
  const float normal_root = numext::sqrt(normal_min);
  const float dominant = normal_root * numext::sqrt(1.1f);
  const float minor = normal_root * numext::sqrt(0.1f / 7.0f);
  VectorXf vector = VectorXf::Constant(9, minor);
  vector[0] = 0.0f;
  vector[1] = dominant;
  VectorXf essential(8);
  float tau;
  float beta;

  vector.makeHouseholder(essential, tau, beta);

  verify_real_householder_result(vector, essential, tau, beta);
}

template <typename = void>
EIGEN_DONT_INLINE void verify_householder_dimension_scaled_tail() {
  // The dominant square is just above the length-independent threshold, while the many minor squares are normal
  // inputs whose individually subnormal products become significant in aggregate.
  constexpr Index size = 262145;
  const double normal_min = static_cast<double>((std::numeric_limits<float>::min)());
  const double epsilon = static_cast<double>(NumTraits<float>::epsilon());
  const float dominant = static_cast<float>(std::sqrt(1.01 * normal_min / epsilon));
  const float minor = static_cast<float>(std::sqrt(0.5 * normal_min));
  VectorXf vector = VectorXf::Constant(size, minor);
  vector[0] = 0.0f;
  vector[1] = dominant;
  VectorXf essential(size - 1);
  float tau;
  float beta;

  vector.makeHouseholder(essential, tau, beta);

  verify_real_householder_result(vector, essential, tau, beta);
  const double expected_norm =
      std::sqrt(double(dominant) * double(dominant) + double(size - 2) * double(minor) * double(minor));
  VERIFY(std::abs(double(beta) + expected_norm) <= 8 * epsilon * expected_norm);
}

template <typename = void>
void householder_small_tail_layouts() {
  const float coefficient = 1e-20f;
  Vector4f column;
  column << 0.0f, coefficient, -2.0f * coefficient, 3.0f * coefficient;
  Vector3f column_essential;
  float tau;
  float beta;
  column.makeHouseholder(column_essential, tau, beta);
  verify_real_householder_result(column, column_essential, tau, beta);

  RowVector4f row = column.transpose();
  RowVector3f row_essential;
  row.makeHouseholder(row_essential, tau, beta);
  verify_real_householder_result(row, row_essential, tau, beta);

  Vector4f in_place = column;
  in_place.makeHouseholderInPlace(tau, beta);
  verify_real_householder_result(column, in_place.tail<3>(), tau, beta);

  Vector4f aliased = column;
  const Vector4f original = aliased;
  auto aliased_essential = aliased.tail<3>();
  aliased.makeHouseholder(aliased_essential, tau, beta);
  verify_real_householder_result(original, aliased_essential, tau, beta);

  float input_storage[8] = {0.0f, 11.0f, coefficient, 12.0f, -2.0f * coefficient, 13.0f, 3.0f * coefficient, 14.0f};
  float essential_storage[6] = {0.0f, 21.0f, 0.0f, 22.0f, 0.0f, 23.0f};
  using StridedVector = Map<VectorXf, Unaligned, InnerStride<2>>;
  StridedVector strided_input(input_storage, 4, InnerStride<2>());
  StridedVector strided_essential(essential_storage, 3, InnerStride<2>());
  strided_input.makeHouseholder(strided_essential, tau, beta);
  verify_real_householder_result(strided_input, strided_essential, tau, beta);
  VERIFY_IS_EQUAL(input_storage[1], 11.0f);
  VERIFY_IS_EQUAL(input_storage[3], 12.0f);
  VERIFY_IS_EQUAL(input_storage[5], 13.0f);
  VERIFY_IS_EQUAL(input_storage[7], 14.0f);
  VERIFY_IS_EQUAL(essential_storage[1], 21.0f);
  VERIFY_IS_EQUAL(essential_storage[3], 22.0f);
  VERIFY_IS_EQUAL(essential_storage[5], 23.0f);
}

template <typename = void>
void householder_small_tail() {
  {
    const float coefficient = 1e-20f;
    Vector2f vector(0.0f, coefficient);
    VectorXf essential(1);
    float tau;
    float beta;

    vector.makeHouseholder(essential, tau, beta);

    VERIFY_IS_APPROX(tau, 1.0f);
    VERIFY_IS_APPROX(beta, -coefficient);
    VERIFY_IS_APPROX(essential[0], 1.0f);
  }

  {
    volatile float denormInput = (std::numeric_limits<float>::denorm_min)();
    const float denorm = denormInput;
    // Subnormal operands are indistinguishable from zero when the target flushes them.
    if (denorm > 0.0f && denorm + denorm > 0.0f) {
      const Vector2f vector = Vector2f::Constant(denorm);
      VectorXf essential(1);
      float tau;
      float beta;

      vector.makeHouseholder(essential, tau, beta);

      Vector2f householder;
      householder << 1.0f, essential;
      const Matrix2f transform = Matrix2f::Identity() - tau * householder * householder.transpose();
      VERIFY_IS_APPROX(transform.transpose() * transform, Matrix2f::Identity());
      const Vector2d transformed = transform.cast<double>() * vector.cast<double>();
      const Vector2d expected(-numext::sqrt(2.0) * double(denorm), 0.0);
      VERIFY((transformed - expected).norm() <= 4.0 * double(NumTraits<float>::epsilon()) * expected.norm());

      const std::complex<float> complexDenorm(denorm, denorm);
      const Vector2cf complexVector = Vector2cf::Constant(complexDenorm);
      VectorXcf complexEssential(1);
      std::complex<float> complexTau;
      float complexBeta;

      complexVector.makeHouseholder(complexEssential, complexTau, complexBeta);

      Vector2cf complexHouseholder;
      complexHouseholder[0] = std::complex<float>(1.0f, 0.0f);
      complexHouseholder[1] = complexEssential[0];
      const Matrix2cf complexTransform =
          Matrix2cf::Identity() - complexTau * complexHouseholder * complexHouseholder.adjoint();
      VERIFY_IS_APPROX(complexTransform.adjoint() * complexTransform, Matrix2cf::Identity());
      const Vector2cd complexTransformed =
          complexTransform.cast<std::complex<double>>() * complexVector.cast<std::complex<double>>();
      const Vector2cd complexExpected(std::complex<double>(-2.0 * double(denorm), 0.0), std::complex<double>(0.0, 0.0));
      VERIFY((complexTransformed - complexExpected).norm() <=
             8.0 * double(NumTraits<float>::epsilon()) * complexExpected.norm());
    }
  }

  {
    Vector2d vector(0.0, 1e-160);
    VectorXd essential(1);
    double tau;
    double beta;

    vector.makeHouseholder(essential, tau, beta);

    verify_real_householder_result(vector, essential, tau, beta);
  }

  verify_complex_zero_tail<float>(1e-20f);
  verify_complex_zero_tail<double>(1e-160);
  verify_low_precision_complex_zero_tail<half>();
  verify_low_precision_complex_zero_tail<bfloat16>();
  verify_custom_complex_small_tail<>();

  {
    const float coefficient = 1e-20f;
    const Matrix2cf matrix = std::complex<float>(0.0f, coefficient) * Matrix2cf::Identity();
    const HouseholderQR<Matrix2cf> qr(matrix);
    const Matrix2cf q = qr.householderQ();
    const Matrix2cf r = qr.matrixQR().template triangularView<Upper>();
    const Matrix2cd matrix_wide = matrix.cast<std::complex<double>>();
    const Matrix2cd reconstructed = (q * r).cast<std::complex<double>>();
    const double relative_residual = (reconstructed - matrix_wide).norm() / matrix_wide.norm();
    VERIFY(relative_residual <= 16 * double(NumTraits<float>::epsilon()));
  }

  verify_householder_flushed_tail<>();
  verify_householder_dimension_scaled_tail<>();
  {
    Eigen::ScopedFlushToZero flush_to_zero;
    if (flush_to_zero.isSupported()) {
      verify_householder_flushed_tail<>();
      verify_householder_dimension_scaled_tail<>();
    }
  }

  householder_small_tail_layouts<>();

  constexpr Index size = 65;
  const float coefficient = 1e-22f;
  VectorXf vector = VectorXf::Constant(size, coefficient);
  VectorXf essential(size - 1);
  float tau;
  float beta;

  vector.makeHouseholder(essential, tau, beta);

  VERIFY(!numext::is_exactly_zero(tau));
  VERIFY_IS_APPROX(beta / coefficient, -numext::sqrt(float(size)));
  VERIFY_IS_APPROX(2.0f * tau, tau * tau * (1.0f + essential.squaredNorm()));
}

// tau and the essential vector are scale invariant and beta is homogeneous, so rescaling by a power of two is exact.
// It keeps the long double reference below the squaring overflow that this path exists to avoid, which matters where
// long double is only as wide as double.
template <typename VectorType, typename EssentialType>
void verify_large_householder_result(const VectorType& vector, const EssentialType& essential,
                                     const typename VectorType::Scalar& tau,
                                     const typename VectorType::RealScalar& beta) {
  using RealScalar = typename VectorType::RealScalar;
  int exponent = 0;
  (void)std::frexp(vector.cwiseAbs().maxCoeff(), &exponent);
  const RealScalar scale = std::ldexp(RealScalar(1), exponent);
  verify_real_householder_result((vector / scale).eval(), essential, tau, RealScalar(beta / scale));
}

template <typename = void>
void householder_large_components() {
  {
    // The head coefficient squares out of the float range; the tail coefficient does not. The direct
    // construction forms their sum regardless, so it produced beta = -inf and tau = NaN.
    Vector2f vector(4e19f, -5.2e18f);
    Matrix<float, 1, 1> essential;
    float tau;
    float beta;

    vector.makeHouseholder(essential, tau, beta);

    VERIFY((numext::isfinite)(tau));
    VERIFY((numext::isfinite)(beta));
    VERIFY((numext::isfinite)(essential[0]));
    verify_large_householder_result(vector, essential, tau, beta);
  }

  {
    VectorXf vector(3);
    vector << 1e20f, 2e20f, 3e20f;
    VectorXf essential(2);
    float tau;
    float beta;

    vector.makeHouseholder(essential, tau, beta);

    verify_large_householder_result(vector, essential, tau, beta);
  }

  {
    // Accumulated overflow: every square is representable, their sum is not.
    VectorXf vector = VectorXf::Constant(65, 1e19f);
    VectorXf essential(64);
    float tau;
    float beta;

    vector.makeHouseholder(essential, tau, beta);

    verify_large_householder_result(vector, essential, tau, beta);
  }

  {
    VectorXd vector(3);
    vector << 1e160, -2e160, 3e160;
    VectorXd essential(2);
    double tau;
    double beta;

    vector.makeHouseholder(essential, tau, beta);

    verify_large_householder_result(vector, essential, tau, beta);
  }

  {
    const float largest = (std::numeric_limits<float>::max)();
    const Vector2cf vector(std::complex<float>(largest, 0.0f), std::complex<float>(largest, 0.0f));
    Matrix<std::complex<float>, 1, 1> essential;
    std::complex<float> tau;
    float beta;

    vector.makeHouseholder(essential, tau, beta);

    // The true norm exceeds the float range, so beta cannot be represented; the reflector itself still must be.
    VERIFY((numext::isfinite)(numext::real(tau)));
    VERIFY((numext::isfinite)(numext::imag(tau)));
    VERIFY((numext::isfinite)(numext::real(essential[0])));
    VERIFY((numext::isfinite)(numext::imag(essential[0])));
    Vector2cf householder;
    householder << std::complex<float>(1.0f, 0.0f), essential;
    const Matrix2cf transform = Matrix2cf::Identity() - tau * householder * householder.adjoint();
    VERIFY_IS_APPROX(transform.adjoint() * transform, Matrix2cf::Identity());
  }

  {
    // Reflectors this large must still compose into a usable decomposition.
    MatrixXf matrix(6, 4);
    for (Index i = 0; i < matrix.rows(); ++i)
      for (Index j = 0; j < matrix.cols(); ++j) matrix(i, j) = 1e19f * float(internal::random<double>(-1.0, 1.0));
    const HouseholderQR<MatrixXf> qr(matrix);
    const MatrixXf q = qr.householderQ() * MatrixXf::Identity(6, 4);
    const MatrixXf r = qr.matrixQR().topRows(4).template triangularView<Upper>();
    const MatrixXd scaled = (matrix / 1e19f).cast<double>();
    const double relative_residual = ((q * r).cast<double>() / 1e19 - scaled).norm() / scaled.norm();
    VERIFY(relative_residual <= 64 * double(NumTraits<float>::epsilon()));
  }
}

// Expected values follow reference LAPACK 3.12.1 xLARFG, whose tau is the conjugate of Eigen's:
//   xnorm = ||tail||; xnorm = 0 and Im(alpha) = 0  =>  tau = 0, beta = alpha, for any alpha (NaN, infinity included);
//   else beta = -sign(hypot(|alpha|, xnorm), Re(alpha)) by sign bit (-0 gives +hypot),
//   tau = ((beta - Re(alpha)) / beta, -Im(alpha) / beta), essential = tail / (alpha - beta).
// The test values are long double so that the exponents of the long double instantiation stay exact.
template <typename Scalar, bool IsComplex = NumTraits<Scalar>::IsComplex>
struct householder_make_scalar {
  static Scalar run(long double re, long double) { return Scalar(re); }
};
template <typename Scalar>
struct householder_make_scalar<Scalar, true> {
  using Real = typename NumTraits<Scalar>::Real;
  static Scalar run(long double re, long double im) { return Scalar(Real(re), Real(im)); }
};

template <typename Scalar>
Scalar householder_make(long double re, long double im = 0) {
  return householder_make_scalar<Scalar>::run(re, im);
}

// Powers of two for the scaled path: 2^kScaled squares below the direct construction's threshold, 2^kUnderflow
// squares to zero, 2^kDominant squares out of range, 2^kNegligible / 2^kDominant underflows to zero, and a tail of
// 2^kHuge scale squares out of range while its norm stays representable.
template <typename Real>
struct householder_exponents {
  static constexpr int kScaled = -70;
  static constexpr int kUnderflow = -76;
  static constexpr int kDominant = 100;
  static constexpr int kNegligible = -100;
  static constexpr int kHuge = 70;
};
template <>
struct householder_exponents<double> {
  static constexpr int kScaled = -700;
  static constexpr int kUnderflow = -700;
  static constexpr int kDominant = 1000;
  static constexpr int kNegligible = -1000;
  static constexpr int kHuge = 600;
};
template <>
struct householder_exponents<long double> {
  static constexpr int kScaled = -9000;
  static constexpr int kUnderflow = -9000;
  static constexpr int kDominant = 9000;
  static constexpr int kNegligible = -9000;
  static constexpr int kHuge = 9000;
};
template <>
struct householder_exponents<half> {
  static constexpr int kScaled = -10;
  static constexpr int kUnderflow = -13;
  static constexpr int kDominant = 15;
  static constexpr int kNegligible = -12;
  static constexpr int kHuge = 8;
};

// A tail of length m = 1, ..., 5 whose Euclidean norm is exactly representable, and a y with an exactly
// representable hypot(y, norm).
struct HouseholderExactTail {
  long double values[5];
  long double norm;
  long double y;
  long double hypot;
};

inline const HouseholderExactTail& householder_exact_tail(Index m) {
  static const HouseholderExactTail kTails[5] = {{{2}, 2, 1.5L, 2.5L},
                                                 {{3, 4}, 5, 12, 13},
                                                 {{2, -1, 2}, 3, 4, 5},
                                                 {{1, -1, 1, -1}, 2, 1.5L, 2.5L},
                                                 {{0, -1, 0, 2, -2}, 3, 4, 5}};
  return kTails[m - 1];
}

template <typename Real>
void verify_householder_real_close(const Real& actual, const Real& expected, int factor) {
  if (numext::is_exactly_zero(expected)) {
    VERIFY(numext::is_exactly_zero(actual));
  } else {
    VERIFY(numext::abs(actual - expected) <= Real(factor) * NumTraits<Real>::epsilon() * numext::abs(expected));
  }
}

template <typename Scalar>
void verify_householder_close(const Scalar& actual, const Scalar& expected, int factor) {
  verify_householder_real_close(numext::real(actual), numext::real(expected), factor);
  verify_householder_real_close(numext::imag(actual), numext::imag(expected), factor);
}

template <typename Real>
bool householder_is_negative(const Real& x) {
  return !numext::is_exactly_zero(numext::signbit(x));
}
// numext::signbit has no long double implementation.
inline bool householder_is_negative(const long double& x) { return std::signbit(x); }

template <typename Scalar>
void verify_householder_all_nan(const Scalar& tau, typename NumTraits<Scalar>::Real beta,
                                const Matrix<Scalar, Dynamic, 1>& essential) {
  VERIFY((numext::isnan)(beta));
  VERIFY((numext::isnan)(tau));
  for (Index i = 0; i < essential.size(); ++i) VERIFY((numext::isnan)(essential[i]));
}

// A NaN in the tail propagates to beta, tau, and every essential coefficient, also when it follows zeros, which the
// component maximum of the scaled path used to skip.
template <typename Scalar>
void verify_householder_nan_tail(Index m, bool nanInImaginaryPart, bool nanAmongNonzero, long double head) {
  using Real = typename NumTraits<Scalar>::Real;
  using Vector = Matrix<Scalar, Dynamic, 1>;
  const long double nan = std::numeric_limits<long double>::quiet_NaN();
  const HouseholderExactTail& tail = householder_exact_tail(m);
  Vector x(m + 1);
  x[0] = householder_make<Scalar>(head);
  for (Index i = 0; i < m; ++i) x[i + 1] = householder_make<Scalar>(nanAmongNonzero ? tail.values[i] : 0.0L);
  const Index nanIndex = nanAmongNonzero ? numext::maxi<Index>(m - 1, 1) : m;
  x[nanIndex] = nanInImaginaryPart ? householder_make<Scalar>(1, nan) : householder_make<Scalar>(nan);
  Vector essential(m);
  Scalar tau;
  Real beta;
  x.makeHouseholder(essential, tau, beta);
  verify_householder_all_nan(tau, beta, essential);
}

// -0 selects beta = +norm, as the sign bit of SIGN does, with the tail unscaled, scaled, and too large to square.
template <typename Scalar>
void verify_householder_negative_zero_head(Index m, int exponent, bool imaginaryTail) {
  using Real = typename NumTraits<Scalar>::Real;
  using Vector = Matrix<Scalar, Dynamic, 1>;
  const HouseholderExactTail& tail = householder_exact_tail(m);
  const long double scale = std::ldexp(1.0L, exponent);
  Vector x(m + 1);
  x[0] = householder_make<Scalar>(-0.0L);
  for (Index i = 0; i < m; ++i) {
    const long double value = tail.values[i] * scale;
    x[i + 1] = imaginaryTail ? householder_make<Scalar>(0, value) : householder_make<Scalar>(value);
  }
  Vector essential(m);
  Scalar tau;
  Real beta;
  x.makeHouseholder(essential, tau, beta);

  // The norm and (beta - alpha) / beta = 1 are exact for these inputs.
  VERIFY_IS_EQUAL(beta, Real(tail.norm * scale));
  VERIFY(!householder_is_negative(beta));
  VERIFY_IS_EQUAL(tau, householder_make<Scalar>(1));
  for (Index i = 0; i < m; ++i) {
    const long double ratio = -static_cast<long double>(tail.values[i]) / tail.norm;
    verify_householder_close(essential[i],
                             imaginaryTail ? householder_make<Scalar>(0, ratio) : householder_make<Scalar>(ratio), 8);
  }
}

// A complex head (-0, y) over a nonzero tail is reflected to beta = +hypot(y, norm); xLARFG's tau = (1, -y / beta),
// and Eigen's is the conjugate.
template <typename Scalar>
void verify_householder_negative_zero_complex_head(Index m, int exponent) {
  using Real = typename NumTraits<Scalar>::Real;
  using Vector = Matrix<Scalar, Dynamic, 1>;
  const HouseholderExactTail& tail = householder_exact_tail(m);
  const long double scale = std::ldexp(1.0L, exponent);
  for (long double sign : {1.0L, -1.0L}) {
    const long double y = sign * tail.y;
    Vector x(m + 1);
    x[0] = householder_make<Scalar>(-0.0L, y * scale);
    for (Index i = 0; i < m; ++i) x[i + 1] = householder_make<Scalar>(tail.values[i] * scale);
    Vector essential(m);
    Scalar tau;
    Real beta;
    x.makeHouseholder(essential, tau, beta);

    verify_householder_real_close(beta, Real(tail.hypot * scale), 2);
    VERIFY(!householder_is_negative(beta));
    verify_householder_close(tau, householder_make<Scalar>(1, y / tail.hypot), 4);
    // tail / (alpha - beta) with alpha - beta = (-hypot, y) * scale.
    const long double denominator = tail.hypot * tail.hypot + y * y;
    for (Index i = 0; i < m; ++i)
      verify_householder_close(
          essential[i],
          householder_make<Scalar>(-tail.values[i] * tail.hypot / denominator, -tail.values[i] * y / denominator), 8);
  }
}

// With a zero tail, a real head gives tau = 0 and keeps beta = alpha, including its sign. A complex head (-0, y)
// is reflected to beta = +|y|.
template <typename Scalar>
void verify_householder_zero_tail_signed_zero(Index m, int exponent) {
  using Real = typename NumTraits<Scalar>::Real;
  using Vector = Matrix<Scalar, Dynamic, 1>;
  Vector x = Vector::Zero(m + 1);
  x[0] = householder_make<Scalar>(-0.0L);
  Vector essential(m);
  Scalar tau;
  Real beta;
  x.makeHouseholder(essential, tau, beta);
  VERIFY(numext::is_exactly_zero(beta));
  VERIFY(householder_is_negative(beta));
  VERIFY(numext::is_exactly_zero(tau));
  for (Index i = 0; i < m; ++i) VERIFY(numext::is_exactly_zero(essential[i]));

  if (NumTraits<Scalar>::IsComplex) {
    const long double scale = std::ldexp(1.0L, exponent);
    for (long double y : {2.0L, -2.0L}) {
      x[0] = householder_make<Scalar>(-0.0L, y * scale);
      x.makeHouseholder(essential, tau, beta);
      VERIFY_IS_EQUAL(beta, Real(2.0L * scale));
      // xLARFG: tau = (1, -y / beta); Eigen's tau is its conjugate.
      VERIFY_IS_EQUAL(tau, householder_make<Scalar>(1, y > 0 ? 1 : -1));
      for (Index i = 0; i < m; ++i) VERIFY(numext::is_exactly_zero(essential[i]));
    }
  }
}

// A NaN head behind an exactly zero tail leaves tau = 0, so only beta is NaN. With a nonzero tail, or a complex head
// whose imaginary part is nonzero, the reflector is NaN throughout.
template <typename Scalar>
void verify_householder_nan_head(Index m) {
  using Real = typename NumTraits<Scalar>::Real;
  using Vector = Matrix<Scalar, Dynamic, 1>;
  const long double nan = std::numeric_limits<long double>::quiet_NaN();
  const HouseholderExactTail& tail = householder_exact_tail(m);
  Vector essential(m);
  Scalar tau;
  Real beta;

  Vector x = Vector::Zero(m + 1);
  x[0] = householder_make<Scalar>(nan);
  x.makeHouseholder(essential, tau, beta);
  VERIFY((numext::isnan)(beta));
  VERIFY(numext::is_exactly_zero(tau));
  for (Index i = 0; i < m; ++i) VERIFY(numext::is_exactly_zero(essential[i]));

  x[m] = householder_make<Scalar>(1);
  x.makeHouseholder(essential, tau, beta);
  verify_householder_all_nan(tau, beta, essential);

  // A tail whose squared norm underflows to zero is still nonzero.
  x[m] = householder_make<Scalar>(std::ldexp(1.0L, householder_exponents<Real>::kUnderflow));
  x.makeHouseholder(essential, tau, beta);
  verify_householder_all_nan(tau, beta, essential);

  if (NumTraits<Scalar>::IsComplex) {
    // A nonzero or NaN imaginary part keeps the reflector: NaN throughout, over a zero and a nonzero tail alike.
    const long double heads[3][2] = {{nan, 2.0L}, {1.0L, nan}, {nan, nan}};
    for (const auto& head : heads) {
      x = Vector::Zero(m + 1);
      x[0] = householder_make<Scalar>(head[0], head[1]);
      x.makeHouseholder(essential, tau, beta);
      verify_householder_all_nan(tau, beta, essential);
      for (Index i = 0; i < m; ++i) x[i + 1] = householder_make<Scalar>(tail.values[i]);
      x.makeHouseholder(essential, tau, beta);
      verify_householder_all_nan(tau, beta, essential);
    }
  }
}

// A real head of either sign and infinite magnitude gives beta = -sign * inf, tau = NaN, and a zero essential part
// (tail / (alpha - beta) = tail / inf), whatever the finite tail, as xLARFG does. Over an exactly zero tail it gives
// tau = 0 and beta = alpha.
template <typename Scalar>
void verify_householder_infinite_head(Index m) {
  using Real = typename NumTraits<Scalar>::Real;
  using Vector = Matrix<Scalar, Dynamic, 1>;
  const long double inf = std::numeric_limits<long double>::infinity();
  const HouseholderExactTail& tail = householder_exact_tail(m);
  Vector essential(m);
  Scalar tau;
  Real beta;

  for (long double sign : {1.0L, -1.0L}) {
    for (int exponent : {0, householder_exponents<Real>::kScaled, householder_exponents<Real>::kHuge}) {
      const long double scale = std::ldexp(1.0L, exponent);
      for (long double headImag : {0.0L, 1.0L}) {
        if (headImag != 0 && !NumTraits<Scalar>::IsComplex) continue;
        Vector x(m + 1);
        x[0] = householder_make<Scalar>(sign * inf, headImag);
        for (Index i = 0; i < m; ++i) x[i + 1] = householder_make<Scalar>(tail.values[i] * scale);
        x.makeHouseholder(essential, tau, beta);
        VERIFY((numext::isinf)(beta));
        VERIFY_IS_EQUAL(householder_is_negative(beta), sign > 0);
        VERIFY((numext::isnan)(tau));
        for (Index i = 0; i < m; ++i) VERIFY(numext::is_exactly_zero(essential[i]));
      }
    }

    Vector x = Vector::Zero(m + 1);
    x[0] = householder_make<Scalar>(sign * inf);
    x.makeHouseholder(essential, tau, beta);
    VERIFY((numext::isinf)(beta));
    VERIFY_IS_EQUAL(householder_is_negative(beta), sign < 0);
    VERIFY(numext::is_exactly_zero(tau));
    for (Index i = 0; i < m; ++i) VERIFY(numext::is_exactly_zero(essential[i]));
  }
}

// A tail whose ratio to the head underflows is negligible but nonzero: the reflector flips the sign of the head
// (tau = 2, essential part underflowing to zero) rather than being the identity.
template <typename Scalar>
void verify_householder_negligible_tail(Index m) {
  using Real = typename NumTraits<Scalar>::Real;
  using Vector = Matrix<Scalar, Dynamic, 1>;
  const long double negligible = std::ldexp(1.0L, householder_exponents<Real>::kNegligible);
  Vector essential(m);
  Scalar tau;
  Real beta;

  // The largest head also makes alpha - beta = 2 |alpha| overflow, which leaves the essential part zero all the same.
  const long double heads[2] = {std::ldexp(1.0L, householder_exponents<Real>::kDominant),
                                static_cast<long double>(NumTraits<Real>::highest())};
  for (long double dominant : heads) {
    for (long double sign : {1.0L, -1.0L}) {
      Vector x(m + 1);
      x[0] = householder_make<Scalar>(sign * dominant);
      for (Index i = 0; i < m; ++i) x[i + 1] = householder_make<Scalar>((i % 2 ? -1.0L : 1.0L) * negligible);
      x.makeHouseholder(essential, tau, beta);
      VERIFY_IS_EQUAL(beta, Real(-sign * dominant));
      VERIFY_IS_EQUAL(tau, householder_make<Scalar>(2));
      for (Index i = 0; i < m; ++i) VERIFY(numext::is_exactly_zero(essential[i]));
    }

    if (NumTraits<Scalar>::IsComplex) {
      // xLARFG: tau = (1, -Im(alpha) / beta) = (1, 1) for alpha = (0, dominant); Eigen's tau is the conjugate.
      Vector x(m + 1);
      x[0] = householder_make<Scalar>(0, dominant);
      for (Index i = 0; i < m; ++i) x[i + 1] = householder_make<Scalar>(negligible, -negligible);
      x.makeHouseholder(essential, tau, beta);
      VERIFY_IS_EQUAL(beta, Real(-dominant));
      VERIFY_IS_EQUAL(tau, householder_make<Scalar>(1, -1));
      for (Index i = 0; i < m; ++i) VERIFY(numext::is_exactly_zero(essential[i]));
    }
  }
}

// A tail with an infinite component gives xLARFG's reflector: beta = -copysign(inf, real(alpha)), tau = NaN, and an
// essential part that is NaN where the tail is infinite (inf / inf) and zero elsewhere (finite / inf). Neither a
// finite nor an overflowing head changes it.
template <typename Scalar>
void verify_householder_infinite_tail(const std::vector<std::pair<long double, long double>>& tailValues,
                                      long double headReal, const bool* tailIsInfinite) {
  using Real = typename NumTraits<Scalar>::Real;
  using Vector = Matrix<Scalar, Dynamic, 1>;
  const Index m = static_cast<Index>(tailValues.size());
  Vector x(m + 1);
  x[0] = householder_make<Scalar>(headReal);
  for (Index i = 0; i < m; ++i) x[i + 1] = householder_make<Scalar>(tailValues[i].first, tailValues[i].second);
  Vector essential(m);
  Scalar tau;
  Real beta;
  x.makeHouseholder(essential, tau, beta);
  VERIFY((numext::isinf)(beta));
  VERIFY(householder_is_negative(beta) == !householder_is_negative(numext::real(x[0])));
  VERIFY((numext::isnan)(tau));
  for (Index i = 0; i < m; ++i) {
    if (tailIsInfinite[i]) {
      VERIFY((numext::isnan)(essential[i]));
    } else {
      VERIFY(numext::is_exactly_zero(essential[i]));
    }
  }
}

template <typename Scalar>
void verify_householder_infinite_tails() {
  const long double inf = std::numeric_limits<long double>::infinity();
  using Entries = std::vector<std::pair<long double, long double>>;
  using Real = typename NumTraits<Scalar>::Real;
  for (int exponent : {0, householder_exponents<Real>::kHuge}) {
    const long double scale = std::ldexp(1.0L, exponent);
    for (long double head : {1.0L * scale, 0.0L, -1.0L * scale}) {
      bool second[] = {true};
      bool middle[] = {false, true, false};
      verify_householder_infinite_tail<Scalar>(Entries{{inf, 0}}, head, second);
      verify_householder_infinite_tail<Scalar>(Entries{{-inf, 0}}, head, second);
      verify_householder_infinite_tail<Scalar>(Entries{{1, 0}, {inf, 0}, {2, 0}}, head, middle);
      if (NumTraits<Scalar>::IsComplex) {
        verify_householder_infinite_tail<Scalar>(Entries{{0, inf}}, head, second);
        verify_householder_infinite_tail<Scalar>(Entries{{0, -inf}}, head, second);
        verify_householder_infinite_tail<Scalar>(Entries{{1, 1}, {0, inf}, {2, -1}}, head, middle);
      }
    }
  }
}

template <typename Scalar>
void householder_xlarfg_special_values() {
  using Real = typename NumTraits<Scalar>::Real;
  for (Index m : {1, 2, 4, 5}) {
    for (long double head : {1.0L, 0.0L}) {
      for (bool amongNonzero : {false, true}) {
        verify_householder_nan_tail<Scalar>(m, false, amongNonzero, head);
        if (NumTraits<Scalar>::IsComplex) verify_householder_nan_tail<Scalar>(m, true, amongNonzero, head);
      }
    }
    for (int exponent : {0, householder_exponents<Real>::kScaled, householder_exponents<Real>::kHuge}) {
      verify_householder_negative_zero_head<Scalar>(m, exponent, false);
      if (NumTraits<Scalar>::IsComplex) {
        verify_householder_negative_zero_head<Scalar>(m, exponent, true);
        verify_householder_negative_zero_complex_head<Scalar>(m, exponent);
      }
      verify_householder_zero_tail_signed_zero<Scalar>(m, exponent);
    }
    verify_householder_nan_head<Scalar>(m);
    verify_householder_infinite_head<Scalar>(m);
    verify_householder_negligible_tail<Scalar>(m);
  }
  verify_householder_infinite_tails<Scalar>();
  // No tail: a real alpha gives tau = 0 and beta = alpha whatever it is, as xLARFG does with N = 1.
  Real beta;
  Scalar tau;
  Matrix<Scalar, 1, 1> head;
  Matrix<Scalar, 0, 1> none;
  const long double values[2] = {std::numeric_limits<long double>::quiet_NaN(),
                                 std::numeric_limits<long double>::infinity()};
  for (long double value : values) {
    head[0] = householder_make<Scalar>(value);
    head.makeHouseholder(none, tau, beta);
    VERIFY(!(numext::isfinite)(beta));
    VERIFY(numext::is_exactly_zero(tau));
  }
  if (NumTraits<Scalar>::IsComplex) {
    // A complex alpha with a nonzero imaginary part is reflected, as zlarfg does with N = 1:
    // beta = -|alpha|, tau = (1, -Im(alpha) / beta) = (1, 1), and Eigen's tau is its conjugate.
    head[0] = householder_make<Scalar>(0, 2);
    head.makeHouseholder(none, tau, beta);
    VERIFY_IS_EQUAL(beta, Real(-2));
    VERIFY_IS_EQUAL(tau, householder_make<Scalar>(1, -1));
  }
}

// long double with the exponents above needs the range of x87 extended or binary128 precision.
template <typename Scalar>
void householder_xlarfg_special_values_long_double() {
  if (std::numeric_limits<long double>::max_exponent > 8192) householder_xlarfg_special_values<Scalar>();
}

// The same cases with subnormals flushed, which must not change any of them. A negative subnormal head is a signed
// zero to a DAZ comparison, so the sign bit decides it: beta = +norm.
template <typename Scalar>
void householder_xlarfg_special_values_flushed() {
  Eigen::ScopedFlushToZero flush_to_zero(Eigen::FlushToZeroMode::Both);
  if (!flush_to_zero.isSupported()) return;
  householder_xlarfg_special_values<Scalar>();
  using Real = typename NumTraits<Scalar>::Real;
  Matrix<Scalar, 3, 1> x;
  x << householder_make<Scalar>(-static_cast<long double>(std::numeric_limits<Real>::denorm_min())),
      householder_make<Scalar>(3), householder_make<Scalar>(4);
  Matrix<Scalar, 2, 1> essential;
  Scalar tau;
  Real beta;
  x.makeHouseholder(essential, tau, beta);
  VERIFY_IS_EQUAL(beta, Real(5));
  VERIFY_IS_EQUAL(tau, householder_make<Scalar>(1));
}

// The reflector reaches the decompositions: a NaN in a column's tail must show in the factor, a NaN head over a
// zero tail must stay in R(k, k) without spreading to the trailing columns, and an infinite head gives R(k, k) = beta.
template <typename Scalar>
void householder_xlarfg_qr() {
  using Matrix3 = Matrix<Scalar, 3, 3>;
  const Scalar nan = householder_make<Scalar>(std::numeric_limits<long double>::quiet_NaN());
  Matrix3 a = Matrix3::Zero();
  a(0, 1) = householder_make<Scalar>(2);
  a(1, 1) = householder_make<Scalar>(3);
  a(2, 1) = householder_make<Scalar>(4);
  a(1, 2) = householder_make<Scalar>(1);
  a(2, 2) = householder_make<Scalar>(1);
  Matrix3 b = a;

  a(0, 0) = householder_make<Scalar>(1);
  a(2, 0) = nan;
  const Matrix3 factored = HouseholderQR<Matrix3>(a).matrixQR();
  VERIFY((numext::isnan)(factored(0, 0)));
  VERIFY((numext::isnan)(factored(2, 0)));
  VERIFY((numext::isnan)(factored.col(1).sum()));

  b(0, 0) = nan;
  const HouseholderQR<Matrix3> qr(b);
  const Matrix3 r = qr.matrixQR();
  VERIFY((numext::isnan)(r(0, 0)));
  VERIFY(numext::is_exactly_zero(qr.hCoeffs()[0]));
  VERIFY(r.rightCols(2).allFinite());
  VERIFY_IS_EQUAL(r(0, 1), householder_make<Scalar>(2));

  // [[inf, 1], [1, 1]]: beta = -inf stays in R(0, 0), and tau = NaN poisons the trailing column, as in xLARFG.
  Matrix<Scalar, 2, 2> c;
  c << householder_make<Scalar>(std::numeric_limits<long double>::infinity()), householder_make<Scalar>(1),
      householder_make<Scalar>(1), householder_make<Scalar>(1);
  const HouseholderQR<Matrix<Scalar, 2, 2>> infQr(c);
  VERIFY((numext::isinf)(numext::real(infQr.matrixQR()(0, 0))));
  VERIFY(householder_is_negative(numext::real(infQr.matrixQR()(0, 0))));
  VERIFY((numext::isnan)(infQr.hCoeffs()[0]));
  VERIFY(numext::is_exactly_zero(infQr.matrixQR()(1, 0)));
}

namespace householder_autodiff_test {
// Forward-mode dual scalar whose copysign(x, y) injects an infinite/NaN impulse at y == 0 (as ceres::Jet does).
template <bool HasSignbit>
struct Dual {
  double v = 0.0;
  double dv = 0.0;
  Dual() = default;
  Dual(double val, double dval = 0.0) : v(val), dv(dval) {}
  Dual operator+() const { return *this; }
  Dual operator-() const { return Dual(-v, -dv); }
  friend Dual operator+(const Dual& a, const Dual& b) { return Dual(a.v + b.v, a.dv + b.dv); }
  friend Dual operator-(const Dual& a, const Dual& b) { return Dual(a.v - b.v, a.dv - b.dv); }
  friend Dual operator*(const Dual& a, const Dual& b) { return Dual(a.v * b.v, a.dv * b.v + a.v * b.dv); }
  friend Dual operator/(const Dual& a, const Dual& b) {
    const double q = a.v / b.v;
    return Dual(q, (a.dv - q * b.dv) / b.v);
  }
  Dual& operator+=(const Dual& b) { return *this = *this + b; }
  Dual& operator-=(const Dual& b) { return *this = *this - b; }
  Dual& operator*=(const Dual& b) { return *this = *this * b; }
  Dual& operator/=(const Dual& b) { return *this = *this / b; }
  friend bool operator==(const Dual& a, const Dual& b) { return a.v == b.v; }
  friend bool operator!=(const Dual& a, const Dual& b) { return a.v != b.v; }
  friend bool operator<(const Dual& a, const Dual& b) { return a.v < b.v; }
  friend bool operator<=(const Dual& a, const Dual& b) { return a.v <= b.v; }
  friend bool operator>(const Dual& a, const Dual& b) { return a.v > b.v; }
  friend bool operator>=(const Dual& a, const Dual& b) { return a.v >= b.v; }
  friend Dual abs(const Dual& x) { return Dual(std::abs(x.v), std::copysign(1.0, x.v) * x.dv); }
  friend Dual sqrt(const Dual& x) {
    const double s = std::sqrt(x.v);
    return Dual(s, 0.5 * x.dv / s);
  }
  friend Dual copysign(const Dual& x, const Dual& y) {
    const double dirac = y.v == 0.0 ? std::numeric_limits<double>::infinity() : 0.0;
    return Dual(std::copysign(x.v, y.v),
                std::copysign(1.0, x.v) * std::copysign(1.0, y.v) * x.dv + std::abs(x.v) * dirac * y.dv);
  }
  friend bool(isfinite)(const Dual& x) { return (std::isfinite)(x.v); }
  friend bool(isnan)(const Dual& x) { return (std::isnan)(x.v); }
  friend bool(isinf)(const Dual& x) { return (std::isinf)(x.v); }
  template <bool B = HasSignbit, std::enable_if_t<B, int> = 0>
  friend bool signbit(const Dual& x) {
    return std::signbit(x.v);
  }
};
}  // namespace householder_autodiff_test

namespace std {
template <bool B>
class numeric_limits<householder_autodiff_test::Dual<B>> : public numeric_limits<double> {};
}  // namespace std

namespace Eigen {
template <bool B>
struct NumTraits<householder_autodiff_test::Dual<B>> : GenericNumTraits<householder_autodiff_test::Dual<B>> {
  using Real = householder_autodiff_test::Dual<B>;
  static inline Real epsilon() { return Real(NumTraits<double>::epsilon()); }
  static inline Real dummy_precision() { return Real(NumTraits<double>::dummy_precision()); }
  static inline Real highest() { return Real(NumTraits<double>::highest()); }
  static inline Real lowest() { return Real(NumTraits<double>::lowest()); }
};
}  // namespace Eigen

template <bool HasSignbit>
void verify_householder_autodiff_zero_head() {
  using Dual = householder_autodiff_test::Dual<HasSignbit>;
  const double eps = NumTraits<double>::epsilon();
  for (double head : {0.0, -0.0}) {
    if (std::signbit(head) && !HasSignbit) continue;
    const double expectedSign = std::signbit(head) ? 1.0 : -1.0;

    for (int exp : {0, -600, 600}) {
      const double scale = std::ldexp(1.0, exp);
      for (bool seedHead : {true, false}) {
        // seedHead = false seeds the tail while dc0 == 0, where copysign's 0 * inf produces NaN rather than inf.
        Matrix<Dual, 3, 1> x(Dual(head, seedHead ? scale : 0.0), Dual(3.0 * scale, seedHead ? 0.0 : scale),
                             Dual(4.0 * scale, 0.0));
        Matrix<Dual, 2, 1> essential;
        Dual tau, beta;
        x.makeHouseholder(essential, tau, beta);
        const double expectedDBeta = seedHead ? 0.0 : 0.6 * scale * expectedSign;
        const double expectedDTau = seedHead ? -0.2 * expectedSign : 0.0;
        VERIFY_IS_EQUAL(beta.v, 5.0 * scale * expectedSign);
        VERIFY((std::isfinite)(beta.dv) && std::abs(beta.dv - expectedDBeta) <= 8 * eps * scale);
        VERIFY_IS_EQUAL(tau.v, 1.0);
        VERIFY((std::isfinite)(tau.dv) && std::abs(tau.dv - expectedDTau) <= 8 * eps);
        Matrix<Dual, 3, 1> v(Dual(1.0), essential[0], essential[1]);
        const Matrix<Dual, 3, 1> reflected = x - tau * v * (v.transpose() * x);
        VERIFY((std::isfinite)(reflected[0].dv) && std::abs(reflected[0].dv - beta.dv) <= 16 * eps * scale);
        for (Index i = 1; i < 3; ++i) {
          VERIFY(std::abs(reflected[i].v) <= 16 * eps * scale);
          VERIFY((std::isfinite)(reflected[i].dv) && std::abs(reflected[i].dv) <= 16 * eps * scale);
        }
      }
    }

#ifndef _MSC_VER
    using ComplexDual = std::complex<Dual>;
    Matrix<ComplexDual, 2, 1> z(ComplexDual(Dual(head, 1.0), Dual(2.0, 0.0)), ComplexDual(Dual(0.0), Dual(0.0)));
    Matrix<ComplexDual, 1, 1> zEssential;
    ComplexDual zTau;
    Dual zBeta;
    z.makeHouseholder(zEssential, zTau, zBeta);
    VERIFY_IS_EQUAL(zBeta.v, 2.0 * expectedSign);
    VERIFY((std::isfinite)(zBeta.dv) && std::abs(zBeta.dv) <= 8 * eps);
    VERIFY((std::isfinite)(zTau.real().dv) && std::abs(zTau.real().dv - (-0.5 * expectedSign)) <= 8 * eps);
    VERIFY((std::isfinite)(zTau.imag().dv) && std::abs(zTau.imag().dv) <= 8 * eps);
#endif
  }

  // [[t, 1], [2, 0]] * x = [1, 4]^T at t = 0 pivots column 0 ([0, 2]^T) first; x(t) = [2, 1 - 2t]^T.
  Matrix<Dual, 2, 2> a;
  a << Dual(0.0, 1.0), Dual(1.0), Dual(2.0), Dual(0.0);
  const Matrix<Dual, 2, 1> rhs(Dual(1.0), Dual(4.0));
  const Matrix<Dual, 2, 1> sol = a.colPivHouseholderQr().solve(rhs);
  VERIFY(std::abs(sol[0].v - 2.0) <= 8 * eps && (std::isfinite)(sol[0].dv) && std::abs(sol[0].dv) <= 8 * eps);
  VERIFY(std::abs(sol[1].v - 1.0) <= 8 * eps && (std::isfinite)(sol[1].dv) && std::abs(sol[1].dv + 2.0) <= 8 * eps);
}

inline void householder_autodiff_zero_head() {
  verify_householder_autodiff_zero_head<true>();
  verify_householder_autodiff_zero_head<false>();
}

template <typename Scalar, int Size, int StorageOrder>
void householder_short_strided() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, StorageOrder>;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Vector = Matrix<Scalar, Size, 1>;
  using Essential = Matrix<Scalar, Size - 1, 1>;
  const Vector vector = Vector::Random();
  Essential essential;
  Scalar tau;
  RealScalar beta;
  vector.makeHouseholder(essential, tau, beta);
  Vector v;
  v << Scalar(1), essential;
  const Matrix<Scalar, Size, Size> h = Matrix<Scalar, Size, Size>::Identity() - tau * v * v.adjoint();

  for (Index cols : {0, 1, 2, 17, 32, 33, 500, 512}) {
    for (Index innerStride : {1, 2}) {
      for (Index outerStride : {2048, 2055}) {
        Matrix<Scalar, Dynamic, 1> storage =
            Matrix<Scalar, Dynamic, 1>::Random(outerStride * (StorageOrder == ColMajor ? cols + 2 : Size + 2));
        Map<Mat, 0, Stride<Dynamic, Dynamic>> mapped(storage.data(), Size + 2, cols + 2,
                                                     Stride<Dynamic, Dynamic>(outerStride, innerStride));
        const auto originalStorage = storage.eval();
        const Mat original = mapped;
        auto block = mapped.block(1, 1, Size, cols);
        const Mat expected = h * original.block(1, 1, Size, cols);
        Matrix<Scalar, Dynamic, 1> workspace(cols + 2);
        workspace.setConstant(Scalar(7));
        // Fixed essential length selects the fused column-major path even with dynamic block dimensions.
        block.applyHouseholderOnTheLeft(essential, tau, workspace.data() + 1);
        const RealScalar bound = RealScalar(16 * Size) * NumTraits<RealScalar>::epsilon() * original.norm();
        VERIFY((block - expected).norm() <= bound);
        VERIFY_IS_EQUAL(workspace[0], Scalar(7));
        VERIFY_IS_EQUAL(workspace[cols + 1], Scalar(7));
        Mat expectedFull = original;
        expectedFull.block(1, 1, Size, cols) = block;
        const auto resultStorage = storage.eval();
        storage = originalStorage;
        mapped = expectedFull;
        VERIFY_IS_EQUAL(storage, resultStorage);

        block.applyHouseholderOnTheLeft(essential, Scalar(0), workspace.data() + 1);
        VERIFY_IS_EQUAL(storage, resultStorage);

        storage = originalStorage;
        const Matrix<Scalar, Dynamic, 1> dynamicEssential = essential;
        block.applyHouseholderOnTheLeft(dynamicEssential, tau, workspace.data() + 1);
        VERIFY((block - expected).norm() <= bound);

        if (cols > 0) {
          storage = originalStorage;
          const Scalar aliasedTau = block.coeff(0, 0);
          const Matrix<Scalar, Size, Size> aliasedH =
              Matrix<Scalar, Size, Size>::Identity() - aliasedTau * v * v.adjoint();
          const Mat aliasedExpected = aliasedH * original.block(1, 1, Size, cols);
          block.applyHouseholderOnTheLeft(essential, block.coeffRef(0, 0), workspace.data() + 1);
          VERIFY((block - aliasedExpected).norm() <= bound);
        }
      }
    }
  }
}

// Reference for H * a with v = [1; essential] and the operation order of applyHouseholderOnTheLeft, which a
// noncommutative scalar distinguishes: w_j = tau * (a_0j + sum_i conj(e_i) a_(i+1)j), a_0j -= w_j, a_(i+1)j -= e_i w_j.
template <typename MatrixType, typename EssentialType>
MatrixType householder_left_reference(const MatrixType& a, const EssentialType& essential,
                                      const typename MatrixType::Scalar& tau) {
  using Scalar = typename MatrixType::Scalar;
  MatrixType result = a;
  for (Index j = 0; j < a.cols(); ++j) {
    Scalar dot = a(0, j);
    for (Index i = 0; i < essential.size(); ++i) dot += numext::conj(essential(i)) * a(i + 1, j);
    const Scalar w = tau * dot;
    result(0, j) -= w;
    for (Index i = 0; i < essential.size(); ++i) result(i + 1, j) -= essential(i) * w;
  }
  return result;
}

// Whether applyHouseholderOnTheLeft on Derived with this essential type selects the fused column loop.
template <typename Derived, typename EssentialPart>
constexpr bool householder_fused() {
  return std::is_same<internal::householder_apply_left_impl<Derived, EssentialPart>,
                      internal::householder_apply_left_impl<Derived, EssentialPart, true>>::value;
}

template <bool ExpectFused, typename MatrixType, typename EssentialType>
void verify_householder_left(MatrixType a, const EssentialType& essential, const typename MatrixType::Scalar& tau) {
  using RealScalar = typename NumTraits<typename MatrixType::Scalar>::Real;
  STATIC_CHECK((householder_fused<MatrixType, EssentialType>() == ExpectFused));
  const MatrixType expected = householder_left_reference(a, essential.eval(), tau);
  // Applying a reflector is backward stable; both evaluations carry O(rows * eps * |a|) rounding.
  const RealScalar bound = RealScalar(16 * a.rows()) * NumTraits<RealScalar>::epsilon() * a.norm();
  Matrix<typename MatrixType::Scalar, Dynamic, 1> workspace(a.cols());
  a.applyHouseholderOnTheLeft(essential, tau, workspace.data());
  VERIFY((a - expected).norm() <= bound);
}

// The essential part may be any column-vector expression, not only a plain vector: a runtime one-column block
// (no compile-time vector shape), the adjoint of a row tail as CompleteOrthogonalDecomposition passes, or a
// fixed-size segment of either. Each must compile in C++14, where no branch of the dispatch is discarded, and
// only the fixed-size ones on column-major storage select the fused loop.
template <typename Scalar, int StorageOrder>
void householder_essential_expressions() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, StorageOrder>;
  using RowMajorMat = Matrix<Scalar, Dynamic, Dynamic, RowMajor>;
  using VectorType = Matrix<Scalar, Dynamic, 1>;
  constexpr bool kFusedIfFixed = StorageOrder == ColMajor;
  const Index cols = internal::random<Index>(1, 20);
  for (Index rows : {Index(2), Index(3), internal::random<Index>(4, 20)}) {
    const Mat a = Mat::Random(rows, cols);
    Scalar tau;
    typename NumTraits<Scalar>::Real beta;
    VectorType essential(rows - 1);
    // Materialize first: makeHouseholder reads its argument several times, and a lazy Random expression
    // redraws on every read, so essential and tau would not describe one reflector.
    const VectorType reflected = VectorType::Random(rows);
    reflected.makeHouseholder(essential, tau, beta);

    Mat column(rows + 1, 3);
    column.col(1).tail(rows - 1) = essential;
    verify_householder_left<false>(a, column.block(2, 1, rows - 1, 1), tau);

    Mat row(3, rows + 1);
    row.row(1).tail(rows - 1) = essential.adjoint();
    verify_householder_left<false>(a, row.row(1).tail(rows - 1).adjoint(), tau);

    RowMajorMat rowMajorColumn(rows + 1, 3);
    rowMajorColumn.col(1).tail(rows - 1) = essential;
    verify_householder_left<false>(a, rowMajorColumn.col(1).tail(rows - 1), tau);

    if (rows == 2) {
      verify_householder_left<kFusedIfFixed>(a, row.row(1).template segment<1>(2).adjoint(), tau);
      verify_householder_left<kFusedIfFixed>(a, rowMajorColumn.col(1).template tail<1>(), tau);
    }
    if (rows == 3) {
      verify_householder_left<kFusedIfFixed>(a, row.row(1).template segment<2>(2).adjoint(), tau);
      verify_householder_left<kFusedIfFixed>(a, rowMajorColumn.col(1).template tail<2>(), tau);
    }
  }
}

// The fused loop must multiply in the order the general update's expressions spell out, essential * tmp with
// tmp = tau * (essential.adjoint() * column + top), which only a noncommutative scalar observes. Small integer
// components keep every operation exact. The general path is not a reference here: its product kernels reorder
// operands (the gemv selector transposes essential.adjoint() * bottom, the column-major outer-product selector
// forms tmp_j * essential).
template <int Size>
void householder_noncommutative_scalar() {
  using Scalar = noncommutative_scalar::Quaternion;
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using EssentialType = Matrix<Scalar, Size - 1, 1>;
  STATIC_CHECK((householder_fused<Mat, EssentialType>()));
  const Scalar i(0, 1), j(0, 0, 1), k(0, 0, 0, 1);
  {
    // With essential = [i, 0...]^T, tau = 1 and input [j, 0...]^T, the second entry becomes -i * j = -k.
    Mat a = Mat::Constant(Size, 1, Scalar(0));
    a(0, 0) = j;
    EssentialType essential = EssentialType::Constant(Scalar(0));
    essential(0) = i;
    Scalar workspace[1];
    a.applyHouseholderOnTheLeft(essential, Scalar(1), workspace);
    VERIFY(a(0, 0) == Scalar(0));
    VERIFY(a(1, 0) == -k);
  }
  {
    // A one-row block has no essential part, so H = 1 - tau = i: H * j = k on the left, j * H = -k on the right.
    const Matrix<Scalar, 0, 1> none;
    Scalar workspace[2];
    Mat row = Mat::Constant(1, 2, j);
    row.applyHouseholderOnTheLeft(none, Scalar(1) - i, workspace);
    VERIFY(row == Mat::Constant(1, 2, k));
    Mat column = Mat::Constant(2, 1, j);
    column.applyHouseholderOnTheRight(none, Scalar(1) - i, workspace);
    VERIFY(column == Mat::Constant(2, 1, -k));
  }
  {
    // A zero essential part keeps the product kernels out: a H = a - (a v) tau v^* with v = [1, 0...]^T, so
    // tau = i turns a first column of j into j - j * i = j + k.
    Mat a = Mat::Constant(2, Size, Scalar(0));
    a.col(0).setConstant(j);
    const EssentialType essential = EssentialType::Constant(Scalar(0));
    Scalar workspace[2];
    a.applyHouseholderOnTheRight(essential, i, workspace);
    VERIFY(a.col(0) == Mat::Constant(2, 1, j + k));
  }
  const Index cols = internal::random<Index>(1, 7);
  Mat a(Size, cols);
  for (Index c = 0; c < cols; ++c)
    for (Index r = 0; r < Size; ++r)
      a(r, c) = Scalar(internal::random<int>(-2, 2), internal::random<int>(-2, 2), internal::random<int>(-2, 2),
                       internal::random<int>(-2, 2));
  EssentialType essential;
  for (Index r = 0; r + 1 < Size; ++r)
    essential(r) = Scalar(internal::random<int>(-2, 2), internal::random<int>(-2, 2), internal::random<int>(-2, 2),
                          internal::random<int>(-2, 2));
  const Scalar tau(internal::random<int>(-2, 2), internal::random<int>(-2, 2), internal::random<int>(-2, 2), 1);
  const Mat expected = householder_left_reference(a, essential, tau);
  Matrix<Scalar, Dynamic, 1> workspace(cols);
  a.applyHouseholderOnTheLeft(essential, tau, workspace.data());
  VERIFY(a == expected);
}

// Whether applyHouseholderOnTheRight on Derived with this essential type selects the fused row loop.
template <typename Derived, typename EssentialPart>
constexpr bool householder_right_fused() {
  return std::is_same<internal::householder_apply_right_impl<Derived, EssentialPart>,
                      internal::householder_apply_right_impl<Derived, EssentialPart, true>>::value;
}

// a H = a - (a v) tau v^* with v = [1; essential], formed explicitly.
template <typename MatrixType, typename EssentialType>
MatrixType householder_right_reference(const MatrixType& a, const EssentialType& essential,
                                       const typename MatrixType::Scalar& tau) {
  using Scalar = typename MatrixType::Scalar;
  Matrix<Scalar, Dynamic, 1> v(essential.size() + 1);
  v << Scalar(1), essential;
  return a - (a * v) * tau * v.adjoint();
}

// Applies [1; essential] and tau to columns j..j+K of a, through a fixed-width and a runtime-width block, with
// essential and tau either separate or stored in the block being updated.
template <int K, typename MatrixType>
void verify_householder_right_columns(const MatrixType& input, Index j) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using EssentialType = Matrix<Scalar, K, 1>;
  using FixedBlock = Block<MatrixType, Dynamic, K + 1>;
  using RuntimeBlock = Block<MatrixType, Dynamic, Dynamic>;
  constexpr bool kFused = !MatrixType::IsRowMajor && internal::packet_traits<Scalar>::Vectorizable;
  STATIC_CHECK((householder_right_fused<FixedBlock, EssentialType>() == kFused));
  STATIC_CHECK((householder_right_fused<RuntimeBlock, EssentialType>() == kFused));
  const Index rows = input.rows();
  Matrix<Scalar, Dynamic, 1> workspace(rows);

  const auto check = [&](const MatrixType& result, const EssentialType& essential, const Scalar& tau) {
    MatrixType expected = input;
    expected.middleCols(j, K + 1) = householder_right_reference(input.middleCols(j, K + 1).eval(), essential, tau);
    // Each entry is a (K + 2)-term expression in a, essential and tau; the other columns must not change.
    const RealScalar scale = RealScalar(1) + numext::abs(tau) * (RealScalar(1) + essential.squaredNorm());
    const RealScalar bound = RealScalar(16 * (K + 2)) * NumTraits<RealScalar>::epsilon() * scale * input.norm();
    VERIFY((result - expected).norm() <= bound);
  };

  Matrix<Scalar, K + 1, 1> x = Matrix<Scalar, K + 1, 1>::Random();
  EssentialType essential;
  Scalar tau;
  RealScalar beta;
  x.makeHouseholder(essential, tau, beta);
  MatrixType a = input;
  a.template middleCols<K + 1>(j).applyHouseholderOnTheRight(essential, tau, workspace.data());
  check(a, essential, tau);
  a = input;
  a.middleCols(j, K + 1).applyHouseholderOnTheRight(essential, tau, workspace.data());
  check(a, essential, tau);

  // The fused loop copies essential and tau before its first store; here they are coefficients of row r of the
  // block, so the reference uses the values read before the update.
  if (kFused) {
    const Index r = internal::random<Index>(0, rows - 1);
    const EssentialType aliasedEssential = input.row(r).segment(j + 1, K).transpose();
    const Scalar aliasedTau = input(r, j);
    a = input;
    a.template middleCols<K + 1>(j).applyHouseholderOnTheRight(a.row(r).template segment<K>(j + 1).transpose(), a(r, j),
                                                               workspace.data());
    check(a, aliasedEssential, aliasedTau);
  }
}

// The two- and three-column right updates of the Francis QR step, over row counts that cover the packet loop and
// the scalar tail, with spare columns on both sides of the updated block.
template <typename Scalar, int StorageOrder>
void householder_right_fixed_columns() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, StorageOrder>;
  const Index maxRows = 4 * internal::packet_traits<Scalar>::size + 3;
  for (Index rows = 1; rows <= maxRows; ++rows) {
    const Mat input = Mat::Random(rows, 5);
    verify_householder_right_columns<1>(input, internal::random<Index>(0, 3));
    verify_householder_right_columns<2>(input, internal::random<Index>(0, 2));
  }
}

// Q times a diagonal, triangular, self-adjoint or permutation matrix, both orders, against the products with the
// dense Q.
template <typename Scalar>
void householder_structured_products(Index size) {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  using VectorType = Matrix<Scalar, Dynamic, 1>;

  const MatrixType a = MatrixType::Random(size, size);
  const HouseholderQR<MatrixType> qr(a);
  const MatrixType q = qr.householderQ();
  const DiagonalMatrix<Scalar, Dynamic> d(VectorType::Random(size));
  PermutationMatrix<Dynamic> p(size);
  randomPermutationVector(p.indices(), size);
  const MatrixType dDense = d.toDenseMatrix();
  const MatrixType pDense = p.toDenseMatrix().template cast<Scalar>();
  const MatrixType upperDense = a.template triangularView<Upper>();
  const MatrixType selfadjointDense = a.template selfadjointView<Lower>();

  STATIC_CHECK((internal::is_same<decltype(qr.householderQ() * d), MatrixType>::value));
  STATIC_CHECK((internal::is_same<decltype(p * qr.householderQ()), MatrixType>::value));

  MatrixType result = qr.householderQ() * d;
  VERIFY_IS_APPROX(result, q * dDense);
  result = d * qr.householderQ();
  VERIFY_IS_APPROX(result, dDense * q);
  result = qr.householderQ() * a.template triangularView<Upper>();
  VERIFY_IS_APPROX(result, q * upperDense);
  result = a.template triangularView<UnitLower>() * qr.householderQ();
  VERIFY_IS_APPROX(result, MatrixType(a.template triangularView<UnitLower>()) * q);
  result = qr.householderQ() * a.template selfadjointView<Lower>();
  VERIFY_IS_APPROX(result, q * selfadjointDense);
  result = a.template selfadjointView<Upper>() * qr.householderQ();
  VERIFY_IS_APPROX(result, MatrixType(a.template selfadjointView<Upper>()) * q);
  result = qr.householderQ() * p;
  VERIFY_IS_APPROX(result, q * pDense);
  result = p * qr.householderQ();
  VERIFY_IS_APPROX(result, pDense * q);
  result = qr.householderQ() * p.inverse();
  VERIFY_IS_APPROX(result, q * pDense.transpose());
  result = p.transpose() * qr.householderQ();
  VERIFY_IS_APPROX(result, pDense.transpose() * q);
  result = qr.householderQ().adjoint() * d;
  VERIFY_IS_APPROX(result, q.adjoint() * dDense);
  result = qr.householderQ().adjoint() * p;
  VERIFY_IS_APPROX(result, q.adjoint() * pDense);

  // Upper triangular operands on the left take the identity path of the dense product; the other triangular modes
  // and the tall (non-square) case take the full path.
  result = qr.householderQ() * qr.matrixQR().template triangularView<Upper>();
  VERIFY_IS_APPROX(result, a);
  result = qr.householderQ() * a.template triangularView<StrictlyUpper>();
  VERIFY_IS_APPROX(result, q * MatrixType(a.template triangularView<StrictlyUpper>()));
  result = qr.householderQ() * a.template triangularView<UnitUpper>();
  VERIFY_IS_APPROX(result, q * MatrixType(a.template triangularView<UnitUpper>()));
  result = qr.householderQ() * a.template triangularView<Lower>();
  VERIFY_IS_APPROX(result, q * MatrixType(a.template triangularView<Lower>()));
  const MatrixType tall = MatrixType::Random(2 * size, size);
  const HouseholderQR<MatrixType> qrTall(tall);
  result = qrTall.householderQ() * qrTall.matrixQR().template triangularView<Upper>();
  VERIFY_IS_APPROX(result, tall);

  // Real operands of a complex sequence promote to the sequence's scalar type.
  using RealScalar = typename NumTraits<Scalar>::Real;
  using RealMatrixType = Matrix<RealScalar, Dynamic, Dynamic>;
  const DiagonalMatrix<RealScalar, Dynamic> dReal(Matrix<RealScalar, Dynamic, 1>::Random(size));
  const RealMatrixType aReal = RealMatrixType::Random(size, size);
  STATIC_CHECK((internal::is_same<decltype(qr.householderQ() * dReal), MatrixType>::value));
  result = qr.householderQ() * dReal;
  VERIFY_IS_APPROX(result, q * dReal.toDenseMatrix());
  result = dReal * qr.householderQ();
  VERIFY_IS_APPROX(result, dReal.toDenseMatrix() * q);
  result = qr.householderQ() * aReal.template triangularView<Upper>();
  VERIFY_IS_APPROX(result, q * RealMatrixType(aReal.template triangularView<Upper>()));
  result = aReal.template triangularView<Lower>() * qr.householderQ();
  VERIFY_IS_APPROX(result, RealMatrixType(aReal.template triangularView<Lower>()) * q);
}

// D * H scales the rows of H from the left, D(i,i) * H(i,j), which only a noncommutative scalar tells apart from
// scaling them on the right. Integer components keep every operation exact.
template <typename = void>
void householder_structured_products_noncommutative() {
  using Scalar = noncommutative_scalar::Quaternion;
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  {
    // H = 1 - tau = j and D = i: D * H = i * j = k, where scaling on the right gives j * i = -k.
    const Scalar i(0, 1), j(0, 0, 1), k(0, 0, 0, 1);
    DiagonalMatrix<Scalar, Dynamic> d(1);
    d.diagonal()(0) = i;
    const Mat result = d * householderSequence(Mat::Constant(1, 1, Scalar(0)), Vec::Constant(1, Scalar(1) - j));
    VERIFY(result(0, 0) == k);
  }
  const auto randomScalar = [] {
    return Scalar(internal::random<int>(-1, 1), internal::random<int>(-1, 1), internal::random<int>(-1, 1),
                  internal::random<int>(-1, 1));
  };
  const Index size = internal::random<Index>(1, 7);
  Mat vectors(size, size);
  Vec coeffs(size);
  DiagonalMatrix<Scalar, Dynamic> d(size);
  for (Index c = 0; c < size; ++c) {
    for (Index r = 0; r < size; ++r) vectors(r, c) = randomScalar();
    coeffs(c) = randomScalar();
    d.diagonal()(c) = randomScalar();
  }
  const HouseholderSequence<Mat, Vec> h(vectors, coeffs);
  const Mat hDense = h;
  Mat expected(size, size);
  for (Index c = 0; c < size; ++c)
    for (Index r = 0; r < size; ++r) expected(r, c) = d.diagonal()(r) * hDense(r, c);
  const Mat result = d * h;
  VERIFY(result == expected);
}

EIGEN_DECLARE_TEST(householder) {
  CALL_SUBTEST_26(householder_ordered_result_types<OnTheLeft>());
  CALL_SUBTEST_26(householder_ordered_result_types<OnTheRight>());
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(householder(Matrix<double, 2, 2>()));
    CALL_SUBTEST_2(householder(Matrix<float, 2, 3>()));
    CALL_SUBTEST_3(householder(Matrix<double, 3, 5>()));
    CALL_SUBTEST_4(householder(Matrix<float, 4, 4>()));
    CALL_SUBTEST_5(householder(
        MatrixXd(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_6(householder(
        MatrixXcf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_5(householder_row_vector_products<double>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    // Takes the blocked right-side application: length >= BlockSize (48) and rows >= 4 * BlockSize.
    CALL_SUBTEST_5(householder_row_vector_products<double>(200));
    CALL_SUBTEST_6(householder_row_vector_products<std::complex<float>>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    CALL_SUBTEST_7(householder(
        MatrixXf(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
    CALL_SUBTEST_8(householder(Matrix<double, 1, 1>()));

    CALL_SUBTEST_9(householder_update(Matrix<double, 3, 5>()));
    CALL_SUBTEST_9(householder_update(Matrix<float, 4, 2>()));
    // householder_update() costs O(rows * min(rows, cols)^3); at EIGEN_TEST_MAX_SIZE it exceeds CI's 3000 s
    // per-test timeout under QEMU's SME emulation.
    CALL_SUBTEST_9(householder_update(MatrixXcf(internal::random<Index>(1, EIGEN_TEST_MAX_SIZE / 2),
                                                internal::random<Index>(1, EIGEN_TEST_MAX_SIZE / 2))));
  }

  CALL_SUBTEST_10(householder_blocked_right_regression<double>());
  CALL_SUBTEST_11(householder_blocked_right_regression<std::complex<double>>());
  CALL_SUBTEST_12(householder_small_tail<>());
  CALL_SUBTEST_13(householder_large_components<>());
  CALL_SUBTEST_27(householder_xlarfg_special_values<float>());
  CALL_SUBTEST_27(householder_xlarfg_special_values<double>());
  CALL_SUBTEST_27(householder_xlarfg_special_values<std::complex<float>>());
  CALL_SUBTEST_27(householder_xlarfg_special_values<std::complex<double>>());
  CALL_SUBTEST_27(householder_xlarfg_qr<double>());
  CALL_SUBTEST_27(householder_xlarfg_qr<std::complex<double>>());
  CALL_SUBTEST_27(householder_xlarfg_special_values_flushed<float>());
  CALL_SUBTEST_27(householder_xlarfg_special_values_flushed<double>());
  CALL_SUBTEST_27(householder_xlarfg_special_values_flushed<std::complex<float>>());
  CALL_SUBTEST_27(householder_xlarfg_special_values_flushed<std::complex<double>>());
  CALL_SUBTEST_27(householder_autodiff_zero_head());
  CALL_SUBTEST_28(householder_xlarfg_special_values<half>());
  CALL_SUBTEST_28(householder_xlarfg_special_values<bfloat16>());
  CALL_SUBTEST_28(householder_xlarfg_special_values<std::complex<half>>());
  CALL_SUBTEST_28(householder_xlarfg_special_values<std::complex<bfloat16>>());
  CALL_SUBTEST_28(householder_xlarfg_special_values_long_double<long double>());
  CALL_SUBTEST_28(householder_xlarfg_special_values_long_double<std::complex<long double>>());
  CALL_SUBTEST_14((householder_short_strided<float, 2, ColMajor>()));
  CALL_SUBTEST_14((householder_short_strided<float, 3, ColMajor>()));
  CALL_SUBTEST_15((householder_short_strided<double, 2, ColMajor>()));
  CALL_SUBTEST_15((householder_short_strided<double, 3, ColMajor>()));
  CALL_SUBTEST_16((householder_short_strided<std::complex<float>, 2, ColMajor>()));
  CALL_SUBTEST_16((householder_short_strided<std::complex<float>, 3, ColMajor>()));
  CALL_SUBTEST_17((householder_short_strided<std::complex<double>, 2, ColMajor>()));
  CALL_SUBTEST_17((householder_short_strided<std::complex<double>, 3, ColMajor>()));
  CALL_SUBTEST_18((householder_short_strided<double, 2, RowMajor>()));
  CALL_SUBTEST_18((householder_short_strided<double, 3, RowMajor>()));
  CALL_SUBTEST_19((householder_short_strided<std::complex<double>, 3, RowMajor>()));
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_20((householder_essential_expressions<double, ColMajor>()));
    CALL_SUBTEST_20((householder_essential_expressions<double, RowMajor>()));
    CALL_SUBTEST_21((householder_essential_expressions<std::complex<double>, ColMajor>()));
    CALL_SUBTEST_21((householder_essential_expressions<std::complex<double>, RowMajor>()));
    CALL_SUBTEST_22(householder_noncommutative_scalar<2>());
    CALL_SUBTEST_22(householder_noncommutative_scalar<3>());
    CALL_SUBTEST_23(householder_structured_products<float>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    CALL_SUBTEST_23(householder_structured_products<double>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
    CALL_SUBTEST_23(
        householder_structured_products<std::complex<double>>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2)));
    CALL_SUBTEST_23(householder_structured_products_noncommutative<>());
    CALL_SUBTEST_24((householder_right_fixed_columns<float, ColMajor>()));
    CALL_SUBTEST_24((householder_right_fixed_columns<double, ColMajor>()));
    CALL_SUBTEST_24((householder_right_fixed_columns<double, RowMajor>()));
    CALL_SUBTEST_25((householder_right_fixed_columns<std::complex<float>, ColMajor>()));
    CALL_SUBTEST_25((householder_right_fixed_columns<std::complex<double>, ColMajor>()));
    CALL_SUBTEST_25((householder_right_fixed_columns<std::complex<double>, RowMajor>()));
  }
  CALL_SUBTEST_23(householder_structured_products<double>(1));
}
