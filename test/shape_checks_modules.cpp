// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#if defined(EIGEN_TEST_PART_1)
#include <Eigen/Dense>
#include <contrib/Eigen/MatrixFunctions>

template <int Order>
void axis_product_checks() {
  using Rect = Matrix<double, 3, 4, Order>;
  const Rect matrix = Rect::Random();
  const Vector3d left = Vector3d::Constant(2);
  const Vector4d right = Vector4d::Constant(3);
  Rect result = left.asDiagonal() * matrix.template triangularView<Upper>();
  VERIFY_IS_APPROX(result, 2 * matrix.template triangularView<Upper>().toDenseMatrix());
  result = matrix.template triangularView<Lower>() * right.asDiagonal();
  VERIFY_IS_APPROX(result, 3 * matrix.template triangularView<Lower>().toDenseMatrix());
  const Matrix3d identity = Matrix3d::Identity();
  Matrix3d square = identity.template selfadjointView<Lower>() * left.asDiagonal();
  VERIFY_IS_EQUAL(square, 2 * identity);
  const VectorXd wrong = VectorXd::Ones(2);
  VERIFY_RAISES_ASSERT(result.noalias() = wrong.asDiagonal() * matrix.template triangularView<Upper>());
  VERIFY_RAISES_ASSERT(result.noalias() = matrix.template triangularView<Lower>() * wrong.asDiagonal());
}

void dense_application_shapes() {
  CALL_SUBTEST((axis_product_checks<ColMajor>()));
  CALL_SUBTEST((axis_product_checks<RowMajor>()));
  const Matrix3d identity = Matrix3d::Identity();
  const Vector3d vector = Vector3d::Ones();
  const VectorXd wrong = VectorXd::Ones(2);
  LLT<Matrix3d> llt(identity);
  llt.rankUpdate(vector);
  VERIFY_IS_APPROX(llt.reconstructedMatrix(), identity + vector * vector.transpose());
  VERIFY_RAISES_ASSERT(llt.rankUpdate(wrong));
  LDLT<Matrix3d> ldlt(identity);
  ldlt.rankUpdate(vector);
  VERIFY_IS_APPROX(ldlt.reconstructedMatrix(), identity + vector * vector.transpose());
  VERIFY_RAISES_ASSERT(ldlt.rankUpdate(wrong));
  VERIFY_IS_APPROX(identity.partialPivLu().solve(vector), vector);
  SelfAdjointEigenSolver<Matrix3d> eig(identity);
  VERIFY_IS_EQUAL(eig.eigenvalues(), vector);

  Matrix<double, 3, 2> qr_matrix = Matrix<double, 3, 2>::Zero();
  Vector2d coefficients;
  double workspace[2];
  const Vector3d column = Vector3d::UnitX();
  internal::householder_qr_inplace_update(qr_matrix, coefficients, column, 0, workspace);
  VERIFY_RAISES_ASSERT(internal::householder_qr_inplace_update(qr_matrix, coefficients, wrong, 1, workspace));
  const Vector3d column2 = Vector3d::UnitY();
  internal::householder_qr_inplace_update(qr_matrix, coefficients, column2, 1, workspace);
  const Matrix3d q = householderSequence(qr_matrix, coefficients);
  VERIFY_IS_APPROX(q * qr_matrix.triangularView<Upper>().toDenseMatrix(), identity.leftCols<2>());
}

void dense_module_shapes() {
  const Matrix2d a = 2 * Matrix2d::Identity();
  const Vector2d v = Vector2d::Ones();
  LLT<Matrix2d> llt(a);
  LDLT<Matrix2d> ldlt(a);
  llt.rankUpdate(v);
  ldlt.rankUpdate(v);
  VERIFY_IS_APPROX(llt.reconstructedMatrix(), a + v * v.transpose());
  VERIFY_IS_APPROX(ldlt.reconstructedMatrix(), a + v * v.transpose());
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

void sparse_application_shapes() {
  const Matrix3d identity = Matrix3d::Identity();
  const Vector3d vector = Vector3d::Ones();
  const VectorXd wrong = VectorXd::Ones(2);
  const SparseMatrix<double> sparse_identity = identity.sparseView();
  DiagonalPreconditioner<double> preconditioner(sparse_identity);
  VERIFY_IS_APPROX(preconditioner.solve(vector), vector);
  VERIFY_RAISES_ASSERT(preconditioner.solve(wrong));
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

void structured_application_shapes() {
  const Matrix3d identity = Matrix3d::Identity();
  const Vector3d vector = Vector3d::Ones();
  const MatrixXd factor = MatrixXd::Ones(3, 1);
  DiagonalPlusLowRank<double> lowrank(vector, factor, factor);
  VERIFY_IS_APPROX((lowrank * vector).eval(), (identity + factor * factor.transpose()) * vector);
  const MatrixXd bad_factor = MatrixXd::Ones(2, 1);
  VERIFY_RAISES_ASSERT(DiagonalPlusLowRank<double>(vector, bad_factor, factor));
  VERIFY_RAISES_ASSERT(DiagonalPlusLowRank<double>(vector, factor, bad_factor));
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
  CALL_SUBTEST_1(dense_application_shapes());
  CALL_SUBTEST_2(sparse_module_shapes());
  CALL_SUBTEST_2(sparse_application_shapes());
  CALL_SUBTEST_3(structured_module_shapes());
  CALL_SUBTEST_3(structured_application_shapes());
  CALL_SUBTEST_4(fft_shapes());
}
