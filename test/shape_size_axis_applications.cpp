// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <contrib/Eigen/StructuredMatrices>

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

EIGEN_DECLARE_TEST(shape_size_axis_applications) {
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

  const SparseMatrix<double> sparse_identity = identity.sparseView();
  DiagonalPreconditioner<double> preconditioner(sparse_identity);
  VERIFY_IS_APPROX(preconditioner.solve(vector), vector);
  VERIFY_RAISES_ASSERT(preconditioner.solve(wrong));
  const MatrixXd factor = MatrixXd::Ones(3, 1);
  DiagonalPlusLowRank<double> lowrank(vector, factor, factor);
  VERIFY_IS_APPROX((lowrank * vector).eval(), (identity + factor * factor.transpose()) * vector);
  const MatrixXd bad_factor = MatrixXd::Ones(2, 1);
  VERIFY_RAISES_ASSERT(DiagonalPlusLowRank<double>(vector, bad_factor, factor));
  VERIFY_RAISES_ASSERT(DiagonalPlusLowRank<double>(vector, factor, bad_factor));
}
