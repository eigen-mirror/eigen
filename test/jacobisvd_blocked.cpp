// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_JACOBI_SVD_BLOCKING_THRESHOLD 16
#include "main.h"
#include <Eigen/SVD>
#include <random>
#include "fp_control.h"

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

// A rank-one n x 2n matrix u 1^T: the QR factor is triangular with a rapidly decaying diagonal, so the pivot blocks of
// the sweeps hold entries whose squares are subnormal. The norm of a block's first column must not be formed from them.
template <typename RealScalar>
void jacobisvd_blocked_rank_one_complex(Index n) {
  using Scalar = std::complex<RealScalar>;
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  const RealScalar tolerance = RealScalar(128 * n) * NumTraits<RealScalar>::epsilon();
  std::mt19937 gen(4242u + unsigned(n));
  std::normal_distribution<double> dist;
  for (int seed = 0; seed < 40; ++seed) {
    MatrixType u(n, 1);
    for (Index i = 0; i < n; ++i) {
      const RealScalar re = RealScalar(dist(gen));
      u(i) = Scalar(re, RealScalar(dist(gen)));
    }
    const MatrixType m = u * MatrixType::Ones(1, 2 * n);
    const JacobiSVD<MatrixType, ComputeThinU | ComputeThinV> svd(m);
    VERIFY_IS_EQUAL(svd.info(), Success);
    const MatrixType reconstructed =
        svd.matrixU() * svd.singularValues().template cast<Scalar>().asDiagonal() * svd.matrixV().adjoint();
    VERIFY_LE((m - reconstructed).norm(), tolerance * m.norm());
    VERIFY_LE((svd.matrixU().adjoint() * svd.matrixU() - MatrixType::Identity(n, n)).norm(), tolerance);
    VERIFY_LE((svd.matrixV().adjoint() * svd.matrixV() - MatrixType::Identity(n, n)).norm(), tolerance);
  }
}

// A complex n x n matrix whose diagonal and strict upper triangle, the pivot columns of the sweeps, are scale times a
// random matrix and whose strict lower triangle is of order 1: the first column of every 2x2 pivot block starts at
// the scale, from 1e-10 down through the subnormal range (k denorm_min and min / 2^j), where the squares of its entries
// underflow. The decomposition must stay unitary and reproduce the matrix in every flush-to-zero mode.
template <typename RealScalar>
void jacobisvd_blocked_tiny_pivot_columns(Index n) {
  using Scalar = std::complex<RealScalar>;
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  std::vector<RealScalar> scales;
  for (const int e : {10, 15, 19, 22, 30, 36}) scales.push_back(std::pow(RealScalar(10), -RealScalar(e)));
  const RealScalar denormMin = std::numeric_limits<RealScalar>::denorm_min();
  const RealScalar normalMin = (std::numeric_limits<RealScalar>::min)();
  for (const int k : {1, 3, 5, 1 << 10}) scales.push_back(RealScalar(k) * denormMin);
  for (const int j : {4, 20}) scales.push_back(std::ldexp(normalMin, -j));
  std::mt19937 gen(31u + unsigned(n));
  std::normal_distribution<double> dist;
  std::vector<MatrixType> matrices;
  for (const RealScalar scale : scales) {
    MatrixType m(n, n);
    for (Index j = 0; j < n; ++j) {
      for (Index i = 0; i < n; ++i) {
        const RealScalar re = RealScalar(dist(gen));
        const Scalar value(re, RealScalar(dist(gen)));
        m(i, j) = i <= j ? value * scale : value;
      }
    }
    matrices.push_back(m);
  }
  const RealScalar tolerance = RealScalar(64 * n) * NumTraits<RealScalar>::epsilon();
  const MatrixType identity = MatrixType::Identity(n, n);
  forEachFlushToZeroMode([&](FlushToZeroMode) {
    for (const MatrixType& m : matrices) {
      const JacobiSVD<MatrixType, ComputeFullU | ComputeFullV> svd(m);
      VERIFY_IS_EQUAL(svd.info(), Success);
      const MatrixType reconstructed =
          svd.matrixU() * svd.singularValues().template cast<Scalar>().asDiagonal() * svd.matrixV().adjoint();
      const RealScalar residual = (m - reconstructed).cwiseAbs().template maxCoeff<PropagateNaN>();
      VERIFY_LE(residual, tolerance * m.cwiseAbs().maxCoeff());
      VERIFY_LE((svd.matrixU().adjoint() * svd.matrixU() - identity).cwiseAbs().template maxCoeff<PropagateNaN>(),
                tolerance);
      VERIFY_LE((svd.matrixV().adjoint() * svd.matrixV() - identity).cwiseAbs().template maxCoeff<PropagateNaN>(),
                tolerance);
    }
  });
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
  CALL_SUBTEST_7(jacobisvd_blocked_rank_one_complex<float>(20));
  CALL_SUBTEST_7(jacobisvd_blocked_rank_one_complex<float>(40));
  CALL_SUBTEST_7(jacobisvd_blocked_rank_one_complex<double>(20));
  CALL_SUBTEST_8(jacobisvd_blocked_tiny_pivot_columns<float>(20));
  CALL_SUBTEST_8(jacobisvd_blocked_tiny_pivot_columns<float>(33));
  CALL_SUBTEST_8(jacobisvd_blocked_tiny_pivot_columns<double>(20));
}
