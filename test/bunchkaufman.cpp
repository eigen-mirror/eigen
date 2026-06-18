// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Enable Eigen's runtime malloc tracking so bunchkaufman_no_malloc() can assert that compute()
// performs no heap allocation when the workspace is pre-allocated. (Malloc stays allowed by default;
// only that one subtest toggles it off.) Must be defined before any Eigen header is included.
#define EIGEN_RUNTIME_NO_MALLOC

#include "main.h"
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>
#include "solverbase.h"

template <typename MatrixType, int UpLo>
typename MatrixType::RealScalar matrix_l1_norm(const MatrixType& m) {
  if (m.cols() == 0) return typename MatrixType::RealScalar(0);
  MatrixType symm = m.template selfadjointView<UpLo>();
  return symm.cwiseAbs().colwise().sum().maxCoeff();
}

// Reconstruct the block-diagonal D from vectorD() / subDiagonal() and check that
// P^T L D L^* P == A and that matrixL()/matrixU() are consistent.
template <typename MatrixType, typename BKType>
void verify_factorization(const MatrixType& A, const BKType& bk) {
  typedef typename MatrixType::Scalar Scalar;
  const Index n = A.rows();

  VERIFY_IS_APPROX(A, bk.reconstructedMatrix());

  // Build D explicitly from vectorD() and subDiagonal() and reconstruct manually.
  MatrixType D = MatrixType::Zero(n, n);
  D.diagonal() = bk.vectorD();
  for (Index k = 0; k + 1 < n; ++k) {
    Scalar s = bk.subDiagonal()(k);
    if (!numext::is_exactly_zero(s)) {
      D(k + 1, k) = s;
      D(k, k + 1) = numext::conj(s);
    }
  }
  // matrixU() should be the adjoint of matrixL().
  MatrixType L = bk.matrixL();
  MatrixType U = bk.matrixU();
  VERIFY_IS_APPROX(L.adjoint(), U);

  // P^T L D L^* P
  MatrixType PtL = bk.transpositionsP().transpose() * L;
  MatrixType recon = PtL * D * PtL.adjoint();
  VERIFY_IS_APPROX(A, recon);
}

// Core test on a Hermitian indefinite matrix `symm` (full, self-adjoint).
template <typename MatrixType>
void bunchkaufman_solve_and_reconstruct(const MatrixType& symm) {
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> SquareMatrixType;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;

  const Index rows = symm.rows();
  const Index cols = symm.cols();

  SquareMatrixType symmLo = symm.template triangularView<Lower>();
  SquareMatrixType symmUp = symm.template triangularView<Upper>();

  BunchKaufman<SquareMatrixType, Lower> bk_lo(symmLo);
  VERIFY(bk_lo.info() == Success);
  verify_factorization(SquareMatrixType(symm), bk_lo);
  check_solverbase<VectorType, VectorType>(symm, bk_lo, rows, rows, 1);
  check_solverbase<MatrixType, MatrixType>(symm, bk_lo, rows, cols, rows);

  BunchKaufman<SquareMatrixType, Upper> bk_up(symmUp);
  VERIFY(bk_up.info() == Success);
  verify_factorization(SquareMatrixType(symm), bk_up);
  check_solverbase<VectorType, VectorType>(symm, bk_up, rows, rows, 1);

  // MatrixBase / SelfAdjointView entry points.
  verify_factorization(SquareMatrixType(symm), SquareMatrixType(symm).bunchKaufman());
  verify_factorization(SquareMatrixType(symm), symm.template selfadjointView<Lower>().bunchKaufman());
  verify_factorization(SquareMatrixType(symm), symm.template selfadjointView<Upper>().bunchKaufman());

  // rcond() within a factor of 10 of the true reciprocal 1-norm condition number.
  if (rows > 0) {
    const SquareMatrixType inv = bk_lo.solve(SquareMatrixType::Identity(rows, rows));
    using RealScalar = typename MatrixType::RealScalar;
    RealScalar rcond = (RealScalar(1) / matrix_l1_norm<SquareMatrixType, Lower>(symmLo)) /
                       matrix_l1_norm<SquareMatrixType, Lower>(inv);
    RealScalar rcond_est = bk_lo.rcond();
    VERIFY(rcond_est >= rcond / 10 && rcond_est <= rcond * 10);
  }
}

// Build a Hermitian matrix with a prescribed (real) eigenvalue spectrum and check inertia + stability.
template <typename MatrixType>
void bunchkaufman_inertia_and_conditioning(Index n) {
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<RealScalar, Dynamic, 1> RealVectorType;

  // Random orthonormal/unitary U via QR.
  MatrixType R = MatrixType::Random(n, n);
  HouseholderQR<MatrixType> qr(R);
  MatrixType U = qr.householderQ();

  // Eigenvalues spanning a wide magnitude range, with mixed signs (indefinite).
  const RealScalar s = (std::min)(RealScalar(6), RealScalar(std::numeric_limits<RealScalar>::max_exponent10) / 8);
  RealVectorType d(n);
  Index expect_pos = 0, expect_neg = 0;
  for (Index k = 0; k < n; ++k) {
    RealScalar mag = pow(RealScalar(10), internal::random<RealScalar>(-s, s));
    RealScalar sign = internal::random<bool>() ? RealScalar(1) : RealScalar(-1);
    d(k) = sign * mag;
    if (d(k) > 0)
      ++expect_pos;
    else
      ++expect_neg;
  }
  MatrixType A = U * d.asDiagonal() * U.adjoint();
  // Force exact Hermitian symmetry (kill round-off asymmetry).
  A = (A + A.adjoint()).eval() * Scalar(RealScalar(0.5));

  BunchKaufman<MatrixType, Lower> bk(A);
  VERIFY(bk.info() == Success);
  VERIFY_IS_APPROX(A, bk.reconstructedMatrix());

  // isPositive()/isNegative() agree with the true inertia.
  VERIFY(bk.isPositive() == (expect_neg == 0));
  VERIFY(bk.isNegative() == (expect_pos == 0));

  // Backward-stable solve: relative residual is small even for ill-conditioned A.
  Matrix<Scalar, Dynamic, 1> b = Matrix<Scalar, Dynamic, 1>::Random(n);
  Matrix<Scalar, Dynamic, 1> x = bk.solve(b);
  RealScalar res = (A * x - b).norm() / b.norm();
  RealScalar tol = sqrt(test_precision<RealScalar>());
  VERIFY(res <= tol);
}

// Definiteness / 2x2-pivot regression cases.
template <typename Scalar>
void bunchkaufman_small_cases() {
  typedef Matrix<Scalar, 2, 2> Mat2;
  typedef Matrix<Scalar, 2, 1> Vec2;

  // Indefinite with zero diagonal -> requires a 2x2 pivot.
  {
    Mat2 A;
    A << Scalar(0), Scalar(1), Scalar(1), Scalar(0);
    BunchKaufman<Mat2> bk(A);
    VERIFY(bk.info() == Success);
    VERIFY_IS_APPROX(A, bk.reconstructedMatrix());
    VERIFY(!bk.isPositive());
    VERIFY(!bk.isNegative());
    Vec2 b(Scalar(3), Scalar(5));
    Vec2 x = bk.solve(b);
    VERIFY_IS_APPROX(A * x, b);
  }
  // Diagonal indefinite [[1,0],[0,-1]].
  {
    Mat2 A;
    A << Scalar(1), Scalar(0), Scalar(0), Scalar(-1);
    BunchKaufman<Mat2> bk(A);
    VERIFY(bk.info() == Success);
    VERIFY_IS_APPROX(A, bk.reconstructedMatrix());
    VERIFY(!bk.isPositive());
    VERIFY(!bk.isNegative());
  }
  // 1x1.
  {
    Matrix<Scalar, 1, 1> A;
    A << Scalar(-3);
    BunchKaufman<Matrix<Scalar, 1, 1> > bk(A);
    VERIFY(bk.info() == Success);
    VERIFY_IS_APPROX(A, bk.reconstructedMatrix());
    VERIFY(!bk.isPositive());
    VERIFY(bk.isNegative());
  }
}

// Exactly-singular matrix: factorization succeeds structurally but reports NumericalIssue,
// while still reconstructing the input exactly. Exercised at a small size (unblocked kernel) and a
// size larger than the panel width (blocked kernel, where the zero pivot lands inside a panel).
template <typename Scalar>
void bunchkaufman_singular(Index n) {
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
  // A zero column/row makes the matrix exactly singular with an exact zero pivot. Place it in the
  // interior so that, for n > blocksize, it falls inside a panel of the blocked algorithm.
  const Index z = n / 2;
  MatrixType M = MatrixType::Random(n, n);
  MatrixType A = M + M.adjoint();
  A.col(z).setZero();
  A.row(z).setZero();
  BunchKaufman<MatrixType, Lower> lo(A);
  VERIFY(lo.info() == NumericalIssue);
  VERIFY_IS_APPROX(A, lo.reconstructedMatrix());
  // matrixL() must be a well-formed unit lower triangular factor even on the singular column.
  VERIFY((lo.matrixL().toDenseMatrix().diagonal().array() == Scalar(1)).all());
  BunchKaufman<MatrixType, Upper> up(A);
  VERIFY(up.info() == NumericalIssue);
  VERIFY_IS_APPROX(A, up.reconstructedMatrix());
}

// A matrix containing a NaN must be reported as a numerical failure (matching LAPACK's DISNAN guard),
// not silently accepted.
template <typename Scalar>
void bunchkaufman_nan() {
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
  for (Index n : {5, 100}) {
    MatrixType M = MatrixType::Random(n, n);
    MatrixType A = M + M.adjoint();
    A(n / 2, n / 2) = std::numeric_limits<typename NumTraits<Scalar>::Real>::quiet_NaN();
    BunchKaufman<MatrixType> bk(A);
    VERIFY(bk.info() == NumericalIssue);
  }
}

// Rank-deficient PSD: A = a a^* with a of rank r < n. Reconstruct must match.
template <typename Scalar>
void bunchkaufman_rank_deficient() {
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
  const Index n = 16;
  const Index r = internal::random<Index>(1, n - 1);
  MatrixType a = MatrixType::Random(n, r);
  MatrixType A = a * a.adjoint();
  BunchKaufman<MatrixType> bk(A);
  VERIFY_IS_APPROX(A, bk.reconstructedMatrix());
  VERIFY(!bk.isNegative());  // PSD -> no negative eigenvalues
}

// Blocking and 2x2-panel-boundary stress across sizes that straddle the panel width.
template <typename Scalar>
void bunchkaufman_blocking_boundary() {
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  const Index PS = internal::packet_traits<Scalar>::size;
  const Index sizes[] = {1,  2,  3,  PS - 1, PS,  PS + 1, 2 * PS, 31,  32,         33,
                         63, 64, 65, 96,     127, 128,    129,    192, 2 * 64 + 3, 200};
  for (Index n : sizes) {
    if (n <= 0) continue;
    MatrixType M = MatrixType::Random(n, n);
    MatrixType A = M + M.adjoint();
    // Force several 2x2 pivots by shrinking the diagonal.
    A.diagonal() *= Scalar(RealScalar(1e-2));

    BunchKaufman<MatrixType, Lower> lo(A);
    VERIFY(lo.info() == Success);
    VERIFY_IS_APPROX(A, lo.reconstructedMatrix());

    BunchKaufman<MatrixType, Upper> up(A);
    VERIFY(up.info() == Success);
    VERIFY_IS_APPROX(A, up.reconstructedMatrix());

    // Lower and Upper must yield the same (Hermitian) decomposition of A.
    VERIFY_IS_APPROX(lo.reconstructedMatrix(), up.reconstructedMatrix());

    Matrix<Scalar, Dynamic, 1> b = Matrix<Scalar, Dynamic, 1>::Random(n);
    Matrix<Scalar, Dynamic, 1> x = lo.solve(b);
    const Matrix<Scalar, Dynamic, 1> residual = A * x - b;
    // The diagonal shrink above intentionally makes the pivot-boundary cases harder than the generic solve tests.
    VERIFY(residual.norm() <= RealScalar(10) * NumTraits<Scalar>::epsilon() * (A.norm() * x.norm() + b.norm()));
  }
}

template <typename MatrixType>
void bunchkaufman_verify_assert() {
  MatrixType tmp;
  BunchKaufman<MatrixType> bk;
  VERIFY_RAISES_ASSERT(bk.matrixL())
  VERIFY_RAISES_ASSERT(bk.matrixU())
  VERIFY_RAISES_ASSERT(bk.vectorD())
  VERIFY_RAISES_ASSERT(bk.subDiagonal())
  VERIFY_RAISES_ASSERT(bk.transpositionsP())
  VERIFY_RAISES_ASSERT(bk.isPositive())
  VERIFY_RAISES_ASSERT(bk.isNegative())
  VERIFY_RAISES_ASSERT(bk.matrixLDLT())
  VERIFY_RAISES_ASSERT(bk.reconstructedMatrix())
  VERIFY_RAISES_ASSERT(bk.solve(tmp))
}

// Build a random Hermitian (real symmetric) indefinite matrix of the same type/size as `m`.
template <typename MatrixType>
MatrixType make_hermitian_indefinite(const MatrixType& m) {
  MatrixType a = MatrixType::Random(m.rows(), m.cols());
  return MatrixType(a + a.adjoint());
}

template <typename MatrixType>
void bunchkaufman(const MatrixType& m) {
  // General Hermitian indefinite.
  bunchkaufman_solve_and_reconstruct(make_hermitian_indefinite(m));
  // Zero-diagonal Hermitian -> forces 2x2 pivots throughout (needs n >= 2 to stay non-singular).
  if (m.rows() >= 2) {
    MatrixType A = make_hermitian_indefinite(m);
    A.diagonal().setZero();
    bunchkaufman_solve_and_reconstruct(A);
  }
}

// Extreme-scale 2x2 pivot: an off-diagonal-only Hermitian 2x2 forces a single 2x2 pivot whose
// determinant is det = -|off|^2. The scaled (det-free) 2x2 formulas must stay finite and correct when
// |off| is huge or tiny. Regression for forming det = d11*d22 - |d21|^2 directly, which overflows to
// +-inf for off=1e200 (solve then returns 0, residual 1) and underflows to 0 for off=1e-200 (solve
// returns NaN/inf), and which also misclassifies the inertia. Requires a type that can hold 1e+-200,
// so this is exercised for double / complex<double> only.
template <typename Scalar>
void bunchkaufman_extreme_scale() {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, 2, 2> Mat2;
  typedef Matrix<Scalar, 2, 1> Vec2;
  const RealScalar tol = sqrt(test_precision<RealScalar>());
  for (RealScalar mag : {pow(RealScalar(10), RealScalar(200)), pow(RealScalar(10), RealScalar(-200))}) {
    const Scalar off = Scalar(mag);
    Mat2 A;
    A << Scalar(0), numext::conj(off), off, Scalar(0);
    BunchKaufman<Mat2> bk(A);
    VERIFY(bk.info() == Success);
    VERIFY(!bk.isPositive());  // det < 0 => one positive and one negative eigenvalue
    VERIFY(!bk.isNegative());
    // Use the max-abs (infinity) relative norm throughout: the Frobenius norm (.norm()) squares the
    // ~1e200 entries and would overflow/underflow even for a correct factorization.
    VERIFY((A - bk.reconstructedMatrix()).cwiseAbs().maxCoeff() <= tol * A.cwiseAbs().maxCoeff());
    // A x = b with b = [1,1]; the product A*x stays O(1), so its residual is safe to measure.
    const Vec2 b(Scalar(1), Scalar(1));
    const Vec2 x = bk.solve(b);
    VERIFY((A * x - b).cwiseAbs().maxCoeff() <= tol);
  }
}

// Extreme-scale factorization at the matrix level: a zero-diagonal Hermitian matrix (2x2 pivots
// throughout, exercising both the unblocked and -- for n > blocksize -- the blocked trailing update)
// scaled to an extreme magnitude. The factorization is scale-equivariant and must remain overflow-free;
// the solve must stay backward stable. double / complex<double> only (1e+-175 overflows float).
template <typename Scalar>
void bunchkaufman_extreme_scale_large(Index n) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;
  const RealScalar tol = sqrt(test_precision<RealScalar>());
  MatrixType M = MatrixType::Random(n, n);
  MatrixType A = M + M.adjoint();
  A.diagonal().setZero();
  // 1e+-175 is past the squaring threshold (|entry|^2 over/underflows double), so the pre-fix code
  // (which forms det = d11*d22 - |d21|^2) produces NaN/Inf here, while the scaled formulas stay exact.
  for (RealScalar sigma : {pow(RealScalar(10), RealScalar(175)), pow(RealScalar(10), RealScalar(-175))}) {
    const MatrixType As = A * Scalar(sigma);
    BunchKaufman<MatrixType, Lower> bk(As);
    VERIFY(bk.info() == Success);
    // Max-abs relative reconstruction error (avoid .norm(), whose squaring overflows the ~1e175 entries).
    VERIFY((As - bk.reconstructedMatrix()).cwiseAbs().maxCoeff() <= tol * As.cwiseAbs().maxCoeff());
    // As*x stays O(1) for a unit-scale rhs, so its residual norm is safe to form.
    const VectorType b = VectorType::Random(n);
    const VectorType x = bk.solve(b);
    VERIFY((As * x - b).norm() <= tol * b.norm());
  }
}

// Regression: the size constructor must pre-allocate the panel workspace so that a subsequent compute()
// on a problem of that size performs no heap allocation. n is chosen above the panel width so the
// blocked path (the one that uses the workspace) runs. (Uses the default stack-allocation limit so the
// trailing-update GEMM's small blocking buffers stay on the stack rather than the heap.)
template <typename Scalar>
void bunchkaufman_no_malloc() {
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;
  const Index n = internal::bunch_kaufman_blocksize<Scalar>() + 36;
  const MatrixType M = MatrixType::Random(n, n);
  const MatrixType A = M + M.adjoint();
  BunchKaufman<MatrixType> bk(n);  // pre-allocates m_matrix, m_transpositions, m_subdiag, m_workspace
  internal::set_is_malloc_allowed(false);
  bk.compute(A);
  internal::set_is_malloc_allowed(true);
  VERIFY(bk.info() == Success);
}

EIGEN_DECLARE_TEST(bunchkaufman) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(bunchkaufman(Matrix<double, 1, 1>()));
    CALL_SUBTEST_2(bunchkaufman(Matrix2d()));
    CALL_SUBTEST_3(bunchkaufman(Matrix3f()));
    CALL_SUBTEST_4(bunchkaufman(Matrix4d()));

    int s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE);
    CALL_SUBTEST_5(bunchkaufman(MatrixXd(s, s)));
    TEST_SET_BUT_UNUSED_VARIABLE(s);

    s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2);
    CALL_SUBTEST_6(bunchkaufman(MatrixXcd(s, s)));
    TEST_SET_BUT_UNUSED_VARIABLE(s);

    s = internal::random<int>(2, EIGEN_TEST_MAX_SIZE);
    CALL_SUBTEST_5(bunchkaufman_inertia_and_conditioning<MatrixXd>(s));
    s = internal::random<int>(2, EIGEN_TEST_MAX_SIZE / 2);
    CALL_SUBTEST_6(bunchkaufman_inertia_and_conditioning<MatrixXcd>(s));

    CALL_SUBTEST_7(bunchkaufman_small_cases<double>());
    CALL_SUBTEST_7(bunchkaufman_small_cases<std::complex<double> >());
    // Singular matrices in both the unblocked (n=8) and blocked (n=100 > blocksize) regimes.
    CALL_SUBTEST_5(bunchkaufman_singular<double>(8));
    CALL_SUBTEST_5(bunchkaufman_singular<double>(100));
    CALL_SUBTEST_6(bunchkaufman_singular<std::complex<double> >(8));
    CALL_SUBTEST_6(bunchkaufman_singular<std::complex<double> >(100));
    CALL_SUBTEST_5(bunchkaufman_nan<double>());
    CALL_SUBTEST_6(bunchkaufman_nan<std::complex<double> >());
    CALL_SUBTEST_5(bunchkaufman_rank_deficient<double>());
    CALL_SUBTEST_6(bunchkaufman_rank_deficient<std::complex<double> >());
  }

  // Empty-matrix edge case.
  CALL_SUBTEST_5(bunchkaufman(MatrixXd(0, 0)));

  // Problem-size constructors.
  CALL_SUBTEST_8(BunchKaufman<MatrixXf>(10));
  CALL_SUBTEST_8(BunchKaufman<MatrixXcd>(10));

  // Assertion checks on an uninitialized decomposition.
  CALL_SUBTEST_3(bunchkaufman_verify_assert<Matrix3f>());
  CALL_SUBTEST_5(bunchkaufman_verify_assert<MatrixXd>());
  CALL_SUBTEST_6(bunchkaufman_verify_assert<MatrixXcd>());

  // Deterministic blocking / panel-boundary tests (outside g_repeat).
  CALL_SUBTEST_8(bunchkaufman_blocking_boundary<double>());
  CALL_SUBTEST_8(bunchkaufman_blocking_boundary<float>());
  CALL_SUBTEST_8(bunchkaufman_blocking_boundary<std::complex<double> >());

  // Extreme-scale 2x2 pivots: the scaled (det-free) 2x2 formulas must not over/underflow.
  CALL_SUBTEST_7(bunchkaufman_extreme_scale<double>());
  CALL_SUBTEST_7(bunchkaufman_extreme_scale<std::complex<double> >());
  CALL_SUBTEST_5(bunchkaufman_extreme_scale_large<double>(8));
  CALL_SUBTEST_5(bunchkaufman_extreme_scale_large<double>(100));
  CALL_SUBTEST_6(bunchkaufman_extreme_scale_large<std::complex<double> >(8));
  CALL_SUBTEST_6(bunchkaufman_extreme_scale_large<std::complex<double> >(100));

  // No-malloc regression: the size constructor pre-allocates the panel workspace.
  CALL_SUBTEST_8(bunchkaufman_no_malloc<double>());
  CALL_SUBTEST_8(bunchkaufman_no_malloc<std::complex<double> >());
}
