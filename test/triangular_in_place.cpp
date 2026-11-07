// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include "product_test_helpers.h"

// The kernels run the column recurrences of xTRTI2/xTRTRI, whose right residual obeys the componentwise bound below.
// The left residual follows only through T X - I = T (X T - I) T^-1, which carries cond(T), so kResidualFactor is
// empirical: ||T X - I|| / (n*eps*||T||*||X||) over 20 seeds x 50 repeats of the families here, n <= 256, peaks at
// 1.10 (n = 1, complex) and 0.35 for 2 <= n <= 65; the non-normal family reaches 0.62 at n = 128 and 0.40 at n = 256
// (real), Kahan 0.06.
static constexpr int kResidualFactor = 16;
// Forward error against the reference solve, bounded by the residual bound times cond(T) <= |T|*|X|.
static constexpr int kForwardFactor = 16;
// Right residual ||X T - I||_F / ((n + 2) eps || |X||T| ||_F), X T formed in a wider type. The componentwise bound
// gamma_{n+1} = (n+1) eps/2 of xTRTRI (Higham, ASNA 14.2-14.3) gives (n+1)/(2(n+2)) < 1/2 for real scalars; complex
// arithmetic costs up to 3x (ASNA 3.6), hence 3/2. Over 2000 seeds the maxima are 0.062 (real) and 0.072 (complex).
template <typename Scalar>
constexpr double right_residual_factor() {
  return NumTraits<Scalar>::IsComplex ? 1.5 : 0.5;
}
// On random draws the row recurrence that the upper kernel replaced scores 0.02..0.03, with a heavy tail of 0.5..0.7,
// so the bounds above cannot tell it from the column recurrence. This fixed draw can: 0.2..0.3 (double) and 0.06..0.4
// (float) under the row recurrence, at most 0.0022 under the column recurrence, with or without FMA contraction.
// Float matters where long double is no wider than double, as only the float checks run there.
static constexpr double kFixedRightResidualFactor = 0.03;
static constexpr unsigned kFixedDrawSeed = 145;
// The wide products are unvectorized O(n^3); 72 still crosses the blocking threshold, with two row panels.
static constexpr Index kRightResidualMaxSize = 72;

// The part of the storage that a triangularView<mode>() does not own, and that an in-place kernel must
// therefore leave bit for bit unchanged: the opposite triangle, plus the diagonal under UnitDiag.
constexpr unsigned int untouched_mode(unsigned int mode) {
  return ((mode & Lower) != 0 ? unsigned(Upper) : unsigned(Lower)) | ((mode & UnitDiag) != 0 ? 0u : unsigned(ZeroDiag));
}

// The type the right residual is formed in. Long double is double under MSVC and on Apple arm64, and there the
// rounding of fl(X T), ~u || |X||T| ||, is comparable to the residual it would measure.
template <typename Scalar>
struct wide_scalar {
  using Real = typename NumTraits<Scalar>::Real;
  using WideReal = std::conditional_t<std::is_same<Real, float>::value, double, long double>;
  using type = std::conditional_t<NumTraits<Scalar>::IsComplex, std::complex<WideReal>, WideReal>;
  static constexpr bool kIsWider = std::numeric_limits<WideReal>::digits > std::numeric_limits<Real>::digits;
};

// Inverts the Mode triangle of t in x and returns ||X T - I||_F / ((n + 2) eps || |X||T| ||_F).
template <unsigned int Mode, typename MatrixType>
double inverse_right_residual_ratio(const MatrixType& t, MatrixType& x) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using DenseType = Matrix<Scalar, Dynamic, Dynamic>;
  using WideScalar = typename wide_scalar<Scalar>::type;
  using WideType = Matrix<WideScalar, Dynamic, Dynamic>;
  const Index n = t.rows();
  x = t;
  x.template triangularView<Mode>().inverseInPlace();
  const DenseType td = t.template triangularView<Mode>().toDenseMatrix();
  const DenseType xd = x.template triangularView<Mode>().toDenseMatrix();
  const WideType residual = xd.template cast<WideScalar>() * td.template cast<WideScalar>() - WideType::Identity(n, n);
  const double scale = abs_product_update_norm(1.0, xd, td, [](Index, Index) { return 0.0; });
  return double(residual.norm()) / (double(n + 2) * double(NumTraits<RealScalar>::epsilon()) * scale);
}

template <unsigned int Mode, typename MatrixType, typename OperandType>
void check_triangular_inverse(const MatrixType& t, OperandType& x, bool well_conditioned) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using DenseType = Matrix<Scalar, Dynamic, Dynamic>;

  const Index n = t.rows();
  const RealScalar eps = NumTraits<RealScalar>::epsilon();

  x = t;
  x.template triangularView<Mode>().inverseInPlace();

  constexpr unsigned int kUntouched = untouched_mode(Mode);
  VERIFY_IS_CWISE_EQUAL(x.template triangularView<kUntouched>().toDenseMatrix(),
                        t.template triangularView<kUntouched>().toDenseMatrix());

  const DenseType td = t.template triangularView<Mode>().toDenseMatrix();
  const DenseType xd = x.template triangularView<Mode>().toDenseMatrix();
  const RealScalar residual = (td * xd - DenseType::Identity(n, n)).norm();
  const RealScalar residual_bound = RealScalar(kResidualFactor * n) * eps * td.norm() * xd.norm();
  VERIFY((numext::isfinite)(residual_bound));
  VERIFY(residual <= residual_bound);

  if (!well_conditioned) return;
  MatrixType reference = MatrixType::Identity(n, n);
  t.template triangularView<Mode>().solveInPlace(reference);
  const DenseType referenced = reference.template triangularView<Mode>().toDenseMatrix();
  const RealScalar forward_bound = RealScalar(kForwardFactor * n) * eps * td.norm() * xd.norm() * xd.norm();
  VERIFY((numext::isfinite)(forward_bound));
  VERIFY((xd - referenced).norm() <= forward_bound);
}

template <unsigned int Mode, typename MatrixType>
void check_right_residual(const MatrixType& t, MatrixType& x) {
  using Scalar = typename MatrixType::Scalar;
  const Index n = t.rows();
  if (!wide_scalar<Scalar>::kIsWider || n == 0 || n > kRightResidualMaxSize) return;
  VERIFY(inverse_right_residual_ratio<Mode>(t, x) <= right_residual_factor<Scalar>());
}

// Strictly diagonally dominant, so cond(T) stays O(1) and the forward comparison is meaningful. The
// scaling also holds with a unit diagonal, where the stored diagonal is ignored.
template <typename MatrixType>
MatrixType well_conditioned_triangular(Index n) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  MatrixType t = MatrixType::Random(n, n) / RealScalar(n + 1);
  t.diagonal().array() += Scalar(1);
  return t;
}

// Unscaled off-diagonals, so |T^-1| can grow exponentially in n; only the residual bound applies.
template <typename MatrixType>
MatrixType ill_conditioned_triangular(Index n) {
  using Scalar = typename MatrixType::Scalar;
  MatrixType t = MatrixType::Random(n, n);
  t.diagonal().array() += Scalar(2);
  return t;
}

// Uniform off-diagonal entries in [-2, 2] against a unit diagonal, so that |T^-1| grows exponentially in n and the
// right residual of a row recurrence, T^-1 (T X - I) T, can far exceed that of the column recurrence.
template <typename MatrixType>
MatrixType non_normal_triangular(Index n) {
  using RealScalar = typename NumTraits<typename MatrixType::Scalar>::Real;
  MatrixType t = RealScalar(2) * MatrixType::Random(n, n);
  t.diagonal().setOnes();
  return t;
}

// Kahan: T(i, i) = s^i, T(i, j) = -c s^i above the diagonal, with s^2 + c^2 = 1; its transpose below.
template <typename MatrixType>
MatrixType kahan_triangular(Index n) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  const RealScalar theta = RealScalar(1.2);
  const RealScalar s = numext::sin(theta), c = numext::cos(theta);
  MatrixType t = MatrixType::Zero(n, n);
  RealScalar p = RealScalar(1);
  for (Index i = 0; i < n; ++i) {
    t(i, i) = Scalar(p);
    for (Index j = i + 1; j < n; ++j) t(i, j) = t(j, i) = Scalar(-c * p);
    p *= s;
  }
  return t;
}

// An upper non-normal draw that does not depend on the test seed: raw std::mt19937 output, whose sequence the
// standard specifies, mapped to [-2, 2) by hand since the standard's distributions are implementation-defined.
// kFixedDrawSeed was picked from 1..300 as one that separates the two recurrences.
template <typename MatrixType>
MatrixType fixed_non_normal_upper(Index n) {
  using Scalar = typename MatrixType::Scalar;
  std::mt19937 gen(kFixedDrawSeed);
  MatrixType t = MatrixType::Identity(n, n);
  for (Index j = 0; j < n; ++j)
    for (Index i = 0; i < j; ++i) t(i, j) = Scalar(4.0 * (double(gen()) / 4294967296.0) - 2.0);
  return t;
}

template <typename MatrixType>
void check_fixed_right_residual(Index n) {
  if (!wide_scalar<typename MatrixType::Scalar>::kIsWider) return;
  const MatrixType t = fixed_non_normal_upper<MatrixType>(n);
  const MatrixType tt = t.transpose();
  MatrixType x(n, n);
  VERIFY(inverse_right_residual_ratio<Upper>(t, x) <= kFixedRightResidualFactor);
  VERIFY(inverse_right_residual_ratio<UnitUpper>(t, x) <= kFixedRightResidualFactor);
  VERIFY(inverse_right_residual_ratio<Lower>(tt, x) <= kFixedRightResidualFactor);
  VERIFY(inverse_right_residual_ratio<UnitLower>(tt, x) <= kFixedRightResidualFactor);
}

template <typename MatrixType, typename OperandType>
void check_inverse_all_modes(const MatrixType& t, OperandType& x, bool well_conditioned) {
  check_triangular_inverse<Lower>(t, x, well_conditioned);
  check_triangular_inverse<Upper>(t, x, well_conditioned);
  check_triangular_inverse<UnitLower>(t, x, well_conditioned);
  check_triangular_inverse<UnitUpper>(t, x, well_conditioned);
}

template <typename MatrixType>
void check_non_normal_all_modes(const MatrixType& t, MatrixType& x) {
  check_inverse_all_modes(t, x, /*well_conditioned=*/false);
  check_right_residual<Lower>(t, x);
  check_right_residual<Upper>(t, x);
  check_right_residual<UnitLower>(t, x);
  check_right_residual<UnitUpper>(t, x);
}

template <typename MatrixType>
void triangular_inverse_all_modes(Index n) {
  MatrixType x(n, n);
  check_inverse_all_modes(well_conditioned_triangular<MatrixType>(n), x, /*well_conditioned=*/true);
  check_inverse_all_modes(ill_conditioned_triangular<MatrixType>(n), x, /*well_conditioned=*/false);
  // The ||T^-1|| of non_normal_triangular grows like 2^n or faster, so single precision overflows past n ~ 80; the
  // sizes up to 65 still cross the blocking threshold.
  if (std::is_same<typename NumTraits<typename MatrixType::Scalar>::Real, float>::value && n > 72) return;
  check_non_normal_all_modes(non_normal_triangular<MatrixType>(n), x);
  check_non_normal_all_modes(kahan_triangular<MatrixType>(n), x);
}

// The destination of LLT::inverse() need not be a plain object, so run the kernel through a block of a block
// of a larger matrix: a non-zero offset, inner stride 1 and an outer stride that does not match the block width.
template <typename MatrixType>
void triangular_inverse_on_block(Index n) {
  const Index row_offset = 2, col_offset = 3;
  MatrixType host = MatrixType::Random(n + 3, n + 5);
  MatrixType expected = host;
  Block<MatrixType> outer(host, 1, 1, n + 2, n + 4);
  Block<Block<MatrixType>> x(outer, row_offset - 1, col_offset - 1, n, n);

  check_inverse_all_modes(well_conditioned_triangular<MatrixType>(n), x, /*well_conditioned=*/true);

  // Nothing outside the block moved.
  expected.block(row_offset, col_offset, n, n) = x;
  VERIFY_IS_CWISE_EQUAL(host, expected);
}

// An operand whose inner stride is a run-time value, which is what a Map or Ref with a dynamic inner
// stride hands the blocked kernels: their TRSM and TRMM steps must reach each panel through that
// value, since the compile-time InnerStrideAtCompileTime they also carry is Dynamic.
template <typename MatrixType>
void triangular_inverse_on_strided_map(Index n) {
  using Scalar = typename MatrixType::Scalar;
  using VectorType = Matrix<Scalar, Dynamic, 1>;
  using StrideType = Stride<Dynamic, Dynamic>;
  const Index inner = 2, outer = 2 * n + 3;
  const Index buffer_size = numext::maxi(Index(1), (n - 1) * (outer + inner) + 1);
  VectorType buffer = VectorType::Random(buffer_size);
  const VectorType before = buffer;
  Map<MatrixType, 0, StrideType> x(buffer.data(), n, n, StrideType(outer, inner));

  check_triangular_inverse<Lower>(well_conditioned_triangular<MatrixType>(n), x, /*well_conditioned=*/true);
  check_triangular_inverse<Upper>(well_conditioned_triangular<MatrixType>(n), x, /*well_conditioned=*/true);

  // Nothing in the gaps between the mapped coefficients moved.
  std::vector<bool> mapped(buffer_size, false);
  for (Index j = 0; j < n; ++j)
    for (Index i = 0; i < n; ++i) mapped[static_cast<std::size_t>(&x.coeffRef(i, j) - buffer.data())] = true;
  for (Index k = 0; k < buffer_size; ++k)
    if (!mapped[k]) VERIFY_IS_EQUAL(buffer[k], before[k]);
}

// internal::triangular_adjoint_square_in_place() is what LLT::inverse() uses to turn L^-1 into
// L^-* L^-1. It is not public API, but it needs coverage of its own: a Cholesky factor always has a
// real diagonal, so the conjugations in the kernel are unobservable through LLT.
template <int UpLo, typename MatrixType>
void adjoint_square_in_place(Index n) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using DenseType = Matrix<Scalar, Dynamic, Dynamic>;

  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  const MatrixType t = ill_conditioned_triangular<MatrixType>(n);
  MatrixType a = t;
  internal::triangular_adjoint_square_in_place<UpLo>(a);
  VERIFY((a.diagonal().imag().array() == RealScalar(0)).all());

  const DenseType td = t.template triangularView<UpLo>().toDenseMatrix();
  const DenseType expected = UpLo == Lower ? DenseType(td.adjoint() * td) : DenseType(td * td.adjoint());
  const DenseType computed = a.template triangularView<UpLo>().toDenseMatrix();
  // n accumulations of products of entries of T, so the rounding error is bounded by ~n*eps*|T|^2.
  const RealScalar bound = RealScalar(kResidualFactor * n) * eps * td.norm() * td.norm();
  VERIFY((numext::isfinite)(bound));
  VERIFY((computed - expected.template triangularView<UpLo>().toDenseMatrix()).norm() <= bound);

  constexpr unsigned int kUntouched = untouched_mode(UpLo);
  VERIFY_IS_CWISE_EQUAL(a.template triangularView<kUntouched>().toDenseMatrix(),
                        t.template triangularView<kUntouched>().toDenseMatrix());
}

template <typename MatrixType>
void adjoint_square_both_triangles(Index n) {
  adjoint_square_in_place<Lower, MatrixType>(n);
  adjoint_square_in_place<Upper, MatrixType>(n);
}

// Overflow of the intermediate products follows the order of the recurrence. The upper kernel forms
// X11 * U(0:j, j) before scaling by x_jj, as xTRTRI 'U' does, so a row-graded factor, whose exact inverse is
// representable, stays finite. The row recurrence it replaced formed U(j, j+1:) * X22 first and overflowed there;
// the column-graded factor below is the mirror case, which now overflows as in the reference.
void triangular_inverse_graded() {
  const double big = std::ldexp(1.0, 600), small = std::ldexp(1.0, -600);
  MatrixXd u(2, 2);
  u << big, big, 0, small;
  MatrixXd x = u;
  x.triangularView<Upper>().inverseInPlace();
  VERIFY_IS_EQUAL(x(0, 0), small);
  VERIFY_IS_EQUAL(x(1, 1), big);
  VERIFY_IS_EQUAL(x(0, 1), -big);

  u << small, big, 0, big;
  x = u;
  x.triangularView<Upper>().inverseInPlace();
  VERIFY_IS_EQUAL(x(0, 0), big);
  VERIFY_IS_EQUAL(x(1, 1), small);
  // The exact entry is -big. As in xTRTI2, the recurrence forms X11 * U12 = inf first and scales it by -small.
  VERIFY_IS_EQUAL(x(0, 1), -std::numeric_limits<double>::infinity());
}

EIGEN_DECLARE_TEST(triangular_in_place) {
  // 63/64 straddles the unblocked-to-blocked switch (kTriangularInverseMinBlocked), 40 and 65 are not multiples of
  // the block size, and at 256 the heuristic first picks a block big enough for the nested kernels to block in turn.
  const Index sizes[] = {0, 1, 2, 3, 7, 16, 31, 32, 33, 40, 63, 64, 65, 128, 256};
  for (int i = 0; i < g_repeat; i++) {
    for (Index n : sizes) {
      EIGEN_UNUSED_VARIABLE(n);  // the part-6 build compiles no subtest that takes n
      CALL_SUBTEST_1(triangular_inverse_all_modes<MatrixXd>(n));
      CALL_SUBTEST_2(triangular_inverse_all_modes<MatrixXcd>(n));
      CALL_SUBTEST_2((triangular_inverse_all_modes<Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>>(n)));
      CALL_SUBTEST_3(triangular_inverse_all_modes<MatrixXf>(n));
      CALL_SUBTEST_4((triangular_inverse_all_modes<Matrix<double, Dynamic, Dynamic, RowMajor>>(n)));
      CALL_SUBTEST_5(triangular_inverse_on_block<MatrixXd>(n));
      CALL_SUBTEST_5(triangular_inverse_on_strided_map<MatrixXd>(n));
      CALL_SUBTEST_7(adjoint_square_both_triangles<MatrixXd>(n));
      CALL_SUBTEST_8(adjoint_square_both_triangles<MatrixXcd>(n));
      CALL_SUBTEST_8((adjoint_square_both_triangles<Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>>(n)));
      CALL_SUBTEST_9(adjoint_square_both_triangles<MatrixXf>(n));
      CALL_SUBTEST_10((adjoint_square_both_triangles<Matrix<double, Dynamic, Dynamic, RowMajor>>(n)));
    }
    CALL_SUBTEST_6((triangular_inverse_all_modes<Matrix<double, 1, 1>>(1)));
    CALL_SUBTEST_6(triangular_inverse_all_modes<Matrix4d>(4));
    CALL_SUBTEST_6((triangular_inverse_all_modes<Matrix<std::complex<float>, 8, 8>>(8)));
    // With a zero-sized fixed dimension col() is not a vector; inverseInPlace() must still compile.
    CALL_SUBTEST_6((triangular_inverse_all_modes<Matrix<double, 0, 0>>(0)));
    CALL_SUBTEST_6((triangular_inverse_all_modes<Matrix<float, Dynamic, 0, ColMajor, 0, 0>>(0)));
  }
  CALL_SUBTEST_1(triangular_inverse_graded());
  // The fixed draw is the same every time, so it runs once rather than per repeat. 48 is unblocked and 65 blocked
  // with two row panels.
  for (Index n : {Index(48), Index(65)}) {
    EIGEN_UNUSED_VARIABLE(n);  // only parts 1, 3 and 4 take n here
    CALL_SUBTEST_1(check_fixed_right_residual<MatrixXd>(n));
    CALL_SUBTEST_3(check_fixed_right_residual<MatrixXf>(n));
    CALL_SUBTEST_4((check_fixed_right_residual<Matrix<double, Dynamic, Dynamic, RowMajor>>(n)));
  }
}
