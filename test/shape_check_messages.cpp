// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#ifndef EIGEN_DEBUG_ASSERTS
#define EIGEN_DEBUG_ASSERTS 1
#endif
#include "main.h"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseQR>

#if !defined(EIGEN_NO_ASSERTION_CHECKING)
template <typename Func>
void check_shape_assertion_message(const char* message, const Func& operation) {
  Eigen::eigen_assert_list.clear();
  Eigen::internal::push_assert = true;
  operation();
  Eigen::internal::push_assert = false;
  bool found = false;
  for (const auto& assertion : Eigen::eigen_assert_list) {
    found = found || assertion.find(message) != std::string::npos;
  }
  VERIFY(found);
}
#endif

EIGEN_DECLARE_TEST(shape_check_messages) {
#if !defined(EIGEN_NO_ASSERTION_CHECKING)
  MatrixXd matrix(1, 1);
  check_shape_assertion_message("Cannot comma-initialize a 0x0 matrix (operator<<)", [&] { matrix << 1.0; });

  DiagonalPreconditioner<double> preconditioner(matrix);
  const VectorXd rhs = VectorXd::Ones(1);
  check_shape_assertion_message("DiagonalPreconditioner::solve()", [&] { (void)preconditioner.solve(rhs); });

  SparseMatrix<double> sparse = matrix.sparseView();
  SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> qr(sparse);
  check_shape_assertion_message("SparseQR::solve()", [&] { (void)qr.solve(rhs); });

  BlockSparseMatrix<double, ColMajor, 2, 2> blocks(1, 1);
  blocks.setIdentity();
  const VectorXd block_rhs = VectorXd::Ones(2);
  check_shape_assertion_message("BlockSparseTriangularView * Dense: dimension mismatch", [&] {
    const VectorXd result = blocks.triangularView<Lower>() * block_rhs;
    VERIFY_IS_EQUAL(result, block_rhs);
  });
#endif
}
