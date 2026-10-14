// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#if defined(EIGEN_TEST_PART_1)
#include <Eigen/Eigenvalues>
#include <contrib/Eigen/MatrixFunctions>

void dense_module_shapes() {
  const Matrix2d a = 2 * Matrix2d::Identity();
  const Vector2d v = Vector2d::Ones();
  LLT<Matrix2d> llt(a);
  LDLT<Matrix2d> ldlt(a);
  llt.rankUpdate(v);
  ldlt.rankUpdate(v);
  VERIFY_IS_APPROX(llt.reconstructedMatrix(), a + v * v.transpose());
  VERIFY_IS_APPROX(ldlt.reconstructedMatrix(), a + v * v.transpose());
  VectorXd wrong = VectorXd::Ones(3);
  VERIFY_RAISES_ASSERT(llt.rankUpdate(wrong));
  VERIFY_RAISES_ASSERT(ldlt.rankUpdate(wrong));
  SelfAdjointEigenSolver<Matrix2d> direct;
  direct.computeDirect(a);
  VERIFY_IS_EQUAL(direct.info(), Success);
  VERIFY_IS_EQUAL(direct.eigenvalues(), Vector2d::Constant(2));
  RealSchur<Matrix2d> schur(a);
  VERIFY_IS_EQUAL(schur.info(), Success);
  RealQZ<Matrix2d> real_qz(a, a);
  VERIFY_IS_EQUAL(real_qz.info(), Success);
  ComplexQZ<Matrix2cd> complex_qz(a.cast<std::complex<double>>(), a.cast<std::complex<double>>());
  VERIFY_IS_EQUAL(complex_qz.info(), Success);
  GeneralizedSelfAdjointEigenSolver<Matrix2d> generalized(a, a);
  VERIFY_IS_EQUAL(generalized.info(), Success);
  const Matrix2d solution = internal::matrix_function_solve_triangular_sylvester(a, a, a);
  VERIFY_IS_APPROX(a * solution + solution * a, a);
  MatrixXd rectangular = MatrixXd::Ones(2, 3);
  MatrixXd dynamic_a = a;
  VERIFY_RAISES_ASSERT(internal::matrix_function_solve_triangular_sylvester(dynamic_a, dynamic_a, rectangular));
  Vector2d coeffs;
  VERIFY_RAISES_ASSERT(internal::tridiagonalization_inplace_unblocked(rectangular, coeffs));
}
#endif

#if defined(EIGEN_TEST_PART_2)
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>

void sparse_module_shapes() {
  SparseMatrix<double> a(3, 3);
  a.setIdentity();
  SimplicialLLT<SparseMatrix<double>> llt(a);
  SimplicialLDLT<SparseMatrix<double>> ldlt(a);
  SparseLU<SparseMatrix<double>> lu(a);
  SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> qr(a);
  const VectorXd rhs = VectorXd::Ones(3);
  VERIFY_IS_APPROX(llt.solve(rhs), rhs);
  VERIFY_IS_APPROX(ldlt.solve(rhs), rhs);
  VERIFY_IS_APPROX(lu.solve(rhs), rhs);
  MatrixXd q = qr.matrixQ();
  VERIFY_IS_APPROX((qr.matrixQ() * rhs).eval(), q * rhs);
  VERIFY_IS_APPROX((qr.matrixQ().adjoint() * rhs).eval(), q.adjoint() * rhs);
  SparseMatrix<double> rectangular(2, 3);
  VERIFY_RAISES_ASSERT(rectangular.selfadjointView<Lower>());
  VERIFY_RAISES_ASSERT(llt.compute(rectangular));
  VERIFY_RAISES_ASSERT(ldlt.compute(rectangular));
  VERIFY_RAISES_ASSERT(lu.compute(rectangular));
  SparseVector<double> empty(0);
  VectorXd empty_dense(0);
  VERIFY_IS_EQUAL(empty.dot(empty_dense), 0.0);
}
#endif

#if defined(EIGEN_TEST_PART_3)
#include <contrib/Eigen/StructuredMatrices>

void structured_module_shapes() {
  Vector2d x(1, 2), y(4, 5);
  const Cauchy<double, 2, 2> c(x, y);
  VERIFY_IS_APPROX(c.determinant(), Matrix2d(c).determinant());
  const Vandermonde<double, 2, 2> v(x, 2);
  VERIFY_IS_APPROX(v.determinant(), Matrix2d(v).determinant());
  const Matrix2d identity = Matrix2d::Identity();
  const Matrix<double, 1, 1> one = Matrix<double, 1, 1>::Ones();
  const KroneckerOperator<Matrix2d, Matrix<double, 1, 1>> kron(identity, one);
  VERIFY_IS_EQUAL(kron.determinant(), 1);
  VERIFY_IS_APPROX(kron.solve(x), x);
  VERIFY_IS_APPROX(Matrix2d(kron.inverse()), identity);
  VERIFY_IS_APPROX(kron.eigenvalues(), Vector2cd::Ones());
  VERIFY_IS_EQUAL(kron.eigenvectors().rows(), 2);
  VectorXd three = VectorXd::LinSpaced(3, 4, 6);
  const Cauchy<double> rectangular_c(x, three);
  const Vandermonde<double> rectangular_v(x, 3);
  VERIFY_RAISES_ASSERT(rectangular_c.determinant());
  VERIFY_RAISES_ASSERT(rectangular_v.determinant());
  CauchyLU<double> clu;
  BjorckPereyra<double> bp;
  VERIFY_RAISES_ASSERT(clu.compute(rectangular_c));
  VERIFY_RAISES_ASSERT(bp.compute(rectangular_v));
  const Hankel<double> hankel(x, three);
  VERIFY_RAISES_ASSERT(hankel.solve(x));
  const Toeplitz<double> toeplitz(x, three);
  LookAheadLevinson<double> levinson;
  VERIFY_RAISES_ASSERT(levinson.compute(toeplitz));
  MatrixXd tall = MatrixXd::Ones(2, 1), wide = MatrixXd::Ones(1, 2);
  const KroneckerOperator<MatrixXd, MatrixXd> rectangular_factors(tall, wide);
  VERIFY_IS_EQUAL(rectangular_factors.rows(), rectangular_factors.cols());
  VERIFY_RAISES_ASSERT(rectangular_factors.determinant());
  VERIFY_RAISES_ASSERT(rectangular_factors.inverse());
  VERIFY_RAISES_ASSERT(rectangular_factors.eigenvalues());
  VERIFY_RAISES_ASSERT(rectangular_factors.eigenvectors());
  VERIFY_RAISES_ASSERT(rectangular_factors.solve(x));
  const Cauchy<double, 2, 3> fixed_rectangular(x, three);
  VERIFY_RAISES_ASSERT(fixed_rectangular.determinant());
}
#endif

#if defined(EIGEN_TEST_PART_4)
#include <contrib/Eigen/FFT>

void fft_shapes() {
  FFT<double> fft;
  Vector2d input(1, 2), recovered;
  Vector2cd spectrum;
  fft.fwd(spectrum, input);
  fft.inv(recovered, spectrum);
  VERIFY_IS_APPROX(recovered, input);
  MatrixXd not_vector = MatrixXd::Ones(2, 2);
  VectorXcd output;
  VERIFY_RAISES_ASSERT(fft.fwd(output, not_vector));
  MatrixXcd complex_not_vector = MatrixXcd::Ones(2, 2);
  VectorXd real_output;
  VERIFY_RAISES_ASSERT(fft.inv(real_output, complex_not_vector));
}
#endif

EIGEN_DECLARE_TEST(shape_checks_modules) {
  CALL_SUBTEST_1(dense_module_shapes());
  CALL_SUBTEST_2(sparse_module_shapes());
  CALL_SUBTEST_3(structured_module_shapes());
  CALL_SUBTEST_4(fft_shapes());
}
