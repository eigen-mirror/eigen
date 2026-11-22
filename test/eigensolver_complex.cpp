// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <limits>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include "CustomComplex.h"

template <typename MatrixType>
bool find_pivot(typename MatrixType::Scalar tol, MatrixType& diffs, Index col = 0) {
  bool match = diffs.diagonal().sum() <= tol;
  if (match || col == diffs.cols()) {
    return match;
  } else {
    Index n = diffs.cols();
    std::vector<std::pair<Index, Index> > transpositions;
    for (Index i = col; i < n; ++i) {
      Index best_index(0);
      if (diffs.col(col).segment(col, n - i).minCoeff(&best_index) > tol) break;

      best_index += col;

      diffs.row(col).swap(diffs.row(best_index));
      if (find_pivot(tol, diffs, col + 1)) return true;
      diffs.row(col).swap(diffs.row(best_index));

      // move current pivot to the end
      diffs.row(n - (i - col) - 1).swap(diffs.row(best_index));
      transpositions.push_back(std::pair<Index, Index>(n - (i - col) - 1, best_index));
    }
    // restore
    for (Index k = transpositions.size() - 1; k >= 0; --k)
      diffs.row(transpositions[k].first).swap(diffs.row(transpositions[k].second));
  }
  return false;
}

/* Check that two column vectors are approximately equal up to permutations.
 * Initially, this method checked that the k-th power sums are equal for all k = 1, ..., vec1.rows(),
 * however this strategy is numerically inaccurate because of numerical cancellation issues.
 */
template <typename VectorType>
void verify_is_approx_upto_permutation(const VectorType& vec1, const VectorType& vec2) {
  using Scalar = typename VectorType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;

  VERIFY(vec1.cols() == 1);
  VERIFY(vec2.cols() == 1);
  VERIFY(vec1.rows() == vec2.rows());

  Index n = vec1.rows();
  RealScalar tol = test_precision<RealScalar>() * test_precision<RealScalar>() *
                   numext::maxi(vec1.squaredNorm(), vec2.squaredNorm());
  Matrix<RealScalar, Dynamic, Dynamic> diffs =
      (vec1.rowwise().replicate(n) - vec2.rowwise().replicate(n).transpose()).cwiseAbs2();

  VERIFY(find_pivot(tol, diffs));
}

template <typename MatrixType>
void eigensolver(const MatrixType& m) {
  /* this test covers the following files:
     ComplexEigenSolver.h, and indirectly ComplexSchur.h
  */
  Index rows = m.rows();
  Index cols = m.cols();

  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;

  MatrixType a = MatrixType::Random(rows, cols);
  MatrixType symmA = a.adjoint() * a;

  ComplexEigenSolver<MatrixType> ei0(symmA);
  VERIFY_IS_EQUAL(ei0.info(), Success);
  VERIFY_IS_APPROX(symmA * ei0.eigenvectors(), ei0.eigenvectors() * ei0.eigenvalues().asDiagonal());

  ComplexEigenSolver<MatrixType> ei1(a);
  VERIFY_IS_EQUAL(ei1.info(), Success);
  VERIFY_IS_APPROX(a * ei1.eigenvectors(), ei1.eigenvectors() * ei1.eigenvalues().asDiagonal());
  // Note: If MatrixType is real then a.eigenvalues() uses EigenSolver and thus
  // another algorithm so results may differ slightly
  verify_is_approx_upto_permutation(a.eigenvalues(), ei1.eigenvalues());

  ComplexEigenSolver<MatrixType> ei2;
  ei2.setMaxIterations(ComplexSchur<MatrixType>::m_maxIterationsPerRow * rows).compute(a);
  VERIFY_IS_EQUAL(ei2.info(), Success);
  VERIFY_IS_EQUAL(ei2.eigenvectors(), ei1.eigenvectors());
  VERIFY_IS_EQUAL(ei2.eigenvalues(), ei1.eigenvalues());
  if (rows > 2) {
    ei2.setMaxIterations(1).compute(a);
    VERIFY_IS_EQUAL(ei2.info(), NoConvergence);
    VERIFY_IS_EQUAL(ei2.getMaxIterations(), 1);
  }

  ComplexEigenSolver<MatrixType> eiNoEivecs(a, false);
  VERIFY_IS_EQUAL(eiNoEivecs.info(), Success);
  VERIFY_IS_APPROX(ei1.eigenvalues(), eiNoEivecs.eigenvalues());

  // Regression test for issue #66
  MatrixType z = MatrixType::Zero(rows, cols);
  ComplexEigenSolver<MatrixType> eiz(z);
  VERIFY((eiz.eigenvalues().cwiseEqual(0)).all());

  MatrixType id = MatrixType::Identity(rows, cols);
  VERIFY_IS_APPROX(id.operatorNorm(), RealScalar(1));

  if (rows > 1 && rows < 20) {
    // Test matrix with NaN
    a(0, 0) = std::numeric_limits<typename MatrixType::RealScalar>::quiet_NaN();
    ComplexEigenSolver<MatrixType> eiNaN(a);
    VERIFY_IS_EQUAL(eiNaN.info(), NoConvergence);
  }

  // regression test for bug 1098
  {
    ComplexEigenSolver<MatrixType> eig(a.adjoint() * a);
    eig.compute(a.adjoint() * a);
  }

  // regression test for bug 478
  {
    a.setZero();
    ComplexEigenSolver<MatrixType> ei3(a);
    VERIFY_IS_EQUAL(ei3.info(), Success);
    RealScalar tol = 2 * a.cols() * NumTraits<RealScalar>::epsilon();
    VERIFY((ei3.eigenvectors().adjoint() * ei3.eigenvectors()).eval().isIdentity(tol));
  }
}

template <typename MatrixType>
void eigensolver_verify_assert(const MatrixType& m) {
  ComplexEigenSolver<MatrixType> eig;
  VERIFY_RAISES_ASSERT(eig.eigenvectors());
  VERIFY_RAISES_ASSERT(eig.eigenvalues());

  MatrixType a = MatrixType::Random(m.rows(), m.cols());
  eig.compute(a, false);
  VERIFY_RAISES_ASSERT(eig.eigenvectors());
}

void custom_complex_stable_normalization() {
  using Scalar = CustomComplex<double>;
  using Vector2 = Matrix<Scalar, 2, 1>;

  Vector2 input;
  input << Scalar(3.0, 4.0), Scalar(0.0, 0.0);
  const Vector2 normalized = input.stableNormalized();
  VERIFY_IS_APPROX(normalized(0).re, 0.6);
  VERIFY_IS_APPROX(normalized(0).im, 0.8);
  VERIFY_IS_EQUAL(normalized(1), Scalar(0.0, 0.0));

  input.stableNormalize();
  VERIFY_IS_APPROX(input(0).re, 0.6);
  VERIFY_IS_APPROX(input(0).im, 0.8);
  VERIFY_IS_EQUAL(input(1), Scalar(0.0, 0.0));
}

// A Jordan block has equal eigenvalues, so each back-substitution step divides by eps * |T| and the unit
// triangular factor X grows like eps^-k along the chain: it overflows for n >= 30 (double) and n >= 8 (float)
// unless the columns are rescaled. The eigenvectors are then not finite with info() == Success. The block is also
// multiplied by 2^exponent, so that |T|_F overflows (2^510 in double, 2^64 in float, for these sizes) or underflows
// (2^-1000, 2^-80) in the perturbation eps |T|_F that replaces the zero denominators.
template <typename Scalar>
void complex_eigensolver_jordan_block(int n, int exponent = 0) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using MatrixType = Matrix<std::complex<RealScalar>, Dynamic, Dynamic>;
  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  const RealScalar scale = std::ldexp(RealScalar(1), exponent);
  MatrixType unit = MatrixType::Identity(n, n);
  unit.template diagonal<1>().setOnes();
  const MatrixType A = unit * scale;
  ComplexEigenSolver<MatrixType> eig(A);
  VERIFY_IS_EQUAL(eig.info(), Success);
  const MatrixType& V = eig.eigenvectors();
  VERIFY(V.allFinite());
  VERIFY(eig.eigenvalues().allFinite());
  const MatrixType scaledEigenvalues = (eig.eigenvalues() / scale).matrix();
  const RealScalar residual = (unit * V - V * scaledEigenvalues.col(0).asDiagonal()).cwiseAbs().maxCoeff() /
                              (unit.cwiseAbs().maxCoeff() * V.cwiseAbs().maxCoeff());
  VERIFY(residual <= RealScalar(100 * n) * eps);
}

// The bound B = max / (4 n |T|_F) on the entries of X overflows in the denominator for |T|_F > max / (4 n), and |T|_F
// itself overflows for a finite T with |T|_F > max; neither may zero the columns of V. T = s (U + diag(1, ..., n)) with
// U the strictly upper triangular ones and s = 2^exponent.
template <typename RealScalar>
void complex_eigensolver_large_triangular(int n, int exponent) {
  using MatrixType = Matrix<std::complex<RealScalar>, Dynamic, Dynamic>;
  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  const RealScalar scale = std::ldexp(RealScalar(1), exponent);
  MatrixType unit = MatrixType::Zero(n, n);
  unit.template triangularView<StrictlyUpper>().setOnes();
  for (Index i = 0; i < n; ++i) unit(i, i) = RealScalar(i + 1);
  const MatrixType A = unit * scale;
  VERIFY(A.allFinite());
  ComplexEigenSolver<MatrixType> eig(A);
  VERIFY_IS_EQUAL(eig.info(), Success);
  const MatrixType& V = eig.eigenvectors();
  VERIFY(V.allFinite());
  for (Index k = 0; k < n; ++k) VERIFY(numext::abs(V.col(k).stableNorm() - RealScalar(1)) <= RealScalar(10 * n) * eps);
  const MatrixType scaledEigenvalues = (eig.eigenvalues() / scale).matrix();
  const RealScalar residual =
      (unit * V - V * scaledEigenvalues.col(0).asDiagonal()).cwiseAbs().maxCoeff() / unit.cwiseAbs().maxCoeff();
  VERIFY(residual <= RealScalar(100 * n) * eps);
}

EIGEN_DECLARE_TEST(eigensolver_complex) {
  int s = 0;
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(eigensolver(Matrix4cf()));
    s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4);
    CALL_SUBTEST_2(eigensolver(MatrixXcd(s, s)));
    CALL_SUBTEST_3(eigensolver(Matrix<std::complex<float>, 1, 1>()));
    CALL_SUBTEST_4(eigensolver(Matrix3f()));
    TEST_SET_BUT_UNUSED_VARIABLE(s);
  }
  CALL_SUBTEST_1(eigensolver_verify_assert(Matrix4cf()));
  s = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4);
  CALL_SUBTEST_2(eigensolver_verify_assert(MatrixXcd(s, s)));
  CALL_SUBTEST_3(eigensolver_verify_assert(Matrix<std::complex<float>, 1, 1>()));
  CALL_SUBTEST_4(eigensolver_verify_assert(Matrix3f()));

  // Test problem size constructors
  CALL_SUBTEST_5(ComplexEigenSolver<MatrixXf> tmp(s));

  // ComplexSchur took the maximum of no coefficients for an empty matrix.
  CALL_SUBTEST_2({
    MatrixXcd A(0, 0);
    ComplexEigenSolver<MatrixXcd> solver(A);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY_IS_EQUAL(solver.eigenvectors().size(), 0);
    VERIFY_IS_EQUAL(A.eigenvalues().size(), 0);
  });

  // Test custom complex scalar type.
  CALL_SUBTEST_6(eigensolver(Matrix<CustomComplex<double>, 5, 5>()));
  CALL_SUBTEST_6(custom_complex_stable_normalization());

  CALL_SUBTEST_5(complex_eigensolver_jordan_block<float>(7));
  CALL_SUBTEST_5(complex_eigensolver_jordan_block<float>(8));
  CALL_SUBTEST_5(complex_eigensolver_jordan_block<float>(24));
  CALL_SUBTEST_5(complex_eigensolver_jordan_block<double>(30));
  CALL_SUBTEST_5(complex_eigensolver_jordan_block<double>(50));
  CALL_SUBTEST_5(complex_eigensolver_jordan_block<float>(10, 64));
  CALL_SUBTEST_5(complex_eigensolver_jordan_block<float>(10, -80));
  CALL_SUBTEST_5(complex_eigensolver_jordan_block<double>(30, 510));
  CALL_SUBTEST_5(complex_eigensolver_jordan_block<double>(30, -1000));

  // |T|_F below max / (4 n) (exponents 1000, 1019, 123), between max / (4 n) and max (n = 2), above max (n = 5).
  CALL_SUBTEST_5(complex_eigensolver_large_triangular<double>(2, 1000));
  CALL_SUBTEST_5(complex_eigensolver_large_triangular<double>(2, 1019));
  CALL_SUBTEST_5(complex_eigensolver_large_triangular<double>(2, 1021));
  CALL_SUBTEST_5(complex_eigensolver_large_triangular<double>(2, 1022));
  CALL_SUBTEST_5(complex_eigensolver_large_triangular<double>(5, 1021));
  CALL_SUBTEST_5(complex_eigensolver_large_triangular<float>(2, 123));
  CALL_SUBTEST_5(complex_eigensolver_large_triangular<float>(2, 125));
  CALL_SUBTEST_5(complex_eigensolver_large_triangular<float>(2, 126));
  CALL_SUBTEST_5(complex_eigensolver_large_triangular<float>(5, 125));

  TEST_SET_BUT_UNUSED_VARIABLE(s);
}
