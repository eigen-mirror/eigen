// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_JACOBI_SVD_BLOCKING_THRESHOLD 16
#include "main.h"
#include <Eigen/SVD>

template <typename MatrixType>
void jacobisvd_blocked(const MatrixType& input) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  const Index n = input.rows();
  MatrixType matrix = input;
  JacobiSVD<MatrixType, ComputeFullU | ComputeFullV> svd(n, n);
  JacobiSVD<MatrixType> valuesOnly(n, n);
  // O(n*eps) backward error and orthogonality bounds for accumulated rotations.
  const RealScalar tolerance = RealScalar(64 * n) * NumTraits<RealScalar>::epsilon();
  for (int pattern = 0; pattern < 4; ++pattern) {
    if (pattern == 1) matrix.col(n - 1) = matrix.col(0);
    if (pattern == 2) matrix.setIdentity();
    if (pattern == 3) matrix.setZero();
    svd.compute(matrix);
    valuesOnly.compute(matrix);
    VERIFY_IS_EQUAL(svd.info(), Success);
    VERIFY_IS_EQUAL(valuesOnly.info(), Success);
    const MatrixType reconstructed =
        svd.matrixU() * svd.singularValues().template cast<Scalar>().asDiagonal() * svd.matrixV().adjoint();
    VERIFY_LE((matrix - reconstructed).norm(), tolerance * matrix.norm());
    VERIFY_LE((svd.matrixU().adjoint() * svd.matrixU() - MatrixType::Identity(n, n)).norm(), tolerance);
    VERIFY_LE((svd.matrixV().adjoint() * svd.matrixV() - MatrixType::Identity(n, n)).norm(), tolerance);
    VERIFY_LE((valuesOnly.singularValues() - svd.singularValues()).norm(), tolerance * matrix.norm());
  }
}

EIGEN_DECLARE_TEST(jacobisvd_blocked) {
  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    // Default block size is 32: cover scalar tails, contiguous blocks, and distant rows.
    for (Index n : {31, 32, 33, 34, 65}) {
      TEST_SET_BUT_UNUSED_VARIABLE(n);
      CALL_SUBTEST_1(jacobisvd_blocked(MatrixXf(MatrixXf::Random(n, n))));
      CALL_SUBTEST_2((jacobisvd_blocked(Matrix<double, Dynamic, Dynamic, RowMajor>::Random(n, n).eval())));
      CALL_SUBTEST_3((jacobisvd_blocked(Matrix<std::complex<float>, Dynamic, Dynamic, RowMajor>::Random(n, n).eval())));
      CALL_SUBTEST_4(jacobisvd_blocked(MatrixXcd(MatrixXcd::Random(n, n))));
      CALL_SUBTEST_5((jacobisvd_blocked(Matrix<double, Dynamic, Dynamic, RowMajor, 65, 65>::Random(n, n).eval())));
    }
    CALL_SUBTEST_6((jacobisvd_blocked(Matrix<double, 34, 34>::Random().eval())));
  }
}
