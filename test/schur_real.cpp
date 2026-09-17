// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010,2012 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include "fp_control.h"
#include <limits>
#include <Eigen/Eigenvalues>

template <typename MatrixType>
void verifyIsQuasiTriangular(const MatrixType& T) {
  const Index size = T.cols();
  typedef typename MatrixType::Scalar Scalar;

  // Check T is lower Hessenberg
  for (int row = 2; row < size; ++row) {
    for (int col = 0; col < row - 1; ++col) {
      VERIFY_IS_EQUAL(T(row, col), Scalar(0));
    }
  }

  // Check that any non-zero on the subdiagonal is followed by a zero and is
  // part of a 2x2 diagonal block with imaginary eigenvalues.
  for (int row = 1; row < size; ++row) {
    if (!numext::is_exactly_zero(T(row, row - 1))) {
      VERIFY(row == size - 1 || numext::is_exactly_zero(T(row + 1, row)));
      Scalar tr = T(row - 1, row - 1) + T(row, row);
      Scalar det = T(row - 1, row - 1) * T(row, row) - T(row - 1, row) * T(row, row - 1);
      VERIFY(4 * det > tr * tr);
    }
  }
}

template <typename MatrixType>
void schur(int size = MatrixType::ColsAtCompileTime) {
  // Test basic functionality: T is quasi-triangular and A = U T U*
  for (int counter = 0; counter < g_repeat; ++counter) {
    MatrixType A = MatrixType::Random(size, size);
    RealSchur<MatrixType> schurOfA(A);
    VERIFY_IS_EQUAL(schurOfA.info(), Success);
    MatrixType U = schurOfA.matrixU();
    MatrixType T = schurOfA.matrixT();
    verifyIsQuasiTriangular(T);
    VERIFY_IS_APPROX(A, U * T * U.transpose());
  }

  // Test asserts when not initialized
  RealSchur<MatrixType> rsUninitialized;
  VERIFY_RAISES_ASSERT(rsUninitialized.matrixT());
  VERIFY_RAISES_ASSERT(rsUninitialized.matrixU());
  VERIFY_RAISES_ASSERT(rsUninitialized.info());

  // Test whether compute() and constructor returns same result
  MatrixType A = MatrixType::Random(size, size);
  RealSchur<MatrixType> rs1;
  rs1.compute(A);
  RealSchur<MatrixType> rs2(A);
  VERIFY_IS_EQUAL(rs1.info(), Success);
  VERIFY_IS_EQUAL(rs2.info(), Success);
  VERIFY_IS_EQUAL(rs1.matrixT(), rs2.matrixT());
  VERIFY_IS_EQUAL(rs1.matrixU(), rs2.matrixU());

  // Test maximum number of iterations
  RealSchur<MatrixType> rs3;
  rs3.setMaxIterations(RealSchur<MatrixType>::m_maxIterationsPerRow * size).compute(A);
  VERIFY_IS_EQUAL(rs3.info(), Success);
  VERIFY_IS_EQUAL(rs3.matrixT(), rs1.matrixT());
  VERIFY_IS_EQUAL(rs3.matrixU(), rs1.matrixU());
  if (size > 2) {
    rs3.setMaxIterations(1).compute(A);
    VERIFY_IS_EQUAL(rs3.info(), NoConvergence);
    VERIFY_IS_EQUAL(rs3.getMaxIterations(), 1);
  }

  MatrixType Atriangular = A;
  Atriangular.template triangularView<StrictlyLower>().setZero();
  rs3.setMaxIterations(1).compute(Atriangular);  // triangular matrices do not need any iterations
  VERIFY_IS_EQUAL(rs3.info(), Success);
  VERIFY_IS_APPROX(rs3.matrixT(), Atriangular);  // approx because of scaling...
  VERIFY_IS_EQUAL(rs3.matrixU(), MatrixType::Identity(size, size));

  // Test computation of only T, not U
  RealSchur<MatrixType> rsOnlyT(A, false);
  VERIFY_IS_EQUAL(rsOnlyT.info(), Success);
  VERIFY_IS_EQUAL(rs1.matrixT(), rsOnlyT.matrixT());
  VERIFY_RAISES_ASSERT(rsOnlyT.matrixU());

  if (size > 2 && size < 20) {
    // Test matrix with NaN
    A(0, 0) = std::numeric_limits<typename MatrixType::Scalar>::quiet_NaN();
    RealSchur<MatrixType> rsNaN(A);
    VERIFY_IS_EQUAL(rsNaN.info(), NoConvergence);
  }
}

void test_bug2633() {
  Eigen::MatrixXd A(4, 4);
  A << 0, 0, 0, -2, 1, 0, 0, -0, 0, 1, 0, 2, 0, 0, 2, -0;
  RealSchur<Eigen::MatrixXd> schur(A);
  VERIFY(schur.info() == Eigen::Success);
}

void real_schur_power_of_two_scaling() {
  // Reciprocal scaling rounds the smaller diagonal entry up by one ULP.
  Matrix2f matrix = Matrix2f::Zero();
  matrix(0, 0) = numext::bit_cast<float>(numext::uint32_t(0x58f6aaed));
  matrix(0, 1) = numext::bit_cast<float>(numext::uint32_t(0x52123456));
  matrix(1, 1) = numext::bit_cast<float>(numext::uint32_t(0x537dcf0e));

  const RealSchur<Matrix2f> schur(matrix);
  VERIFY_IS_EQUAL(schur.info(), Success);
  VERIFY_IS_EQUAL(schur.matrixT(), matrix);

  // Probe arithmetic, not just loads: FTZ can preserve an input yet flush a subnormal result.
  volatile float normalMinInput = (std::numeric_limits<float>::min)();
  volatile float epsilonInput = std::numeric_limits<float>::epsilon();
  const float denormMin = normalMinInput * epsilonInput;
  if (!(denormMin > 0.0f)) return;
  matrix.setZero();
  matrix.diagonal() << 1.5f, denormMin;
  const RealSchur<Matrix2f> tailSchur(matrix);
  VERIFY_IS_EQUAL(tailSchur.matrixT()(1, 1), denormMin);
}

template <typename MatrixType>
void real_schur_subnormal_restoration(Index n = MatrixType::RowsAtCompileTime) {
  using Scalar = typename MatrixType::Scalar;
  using Bits = typename numext::get_integer_by_size<sizeof(Scalar)>::unsigned_type;
  const Scalar normal_min = (std::numeric_limits<Scalar>::min)();
  const Bits half_min_bits = numext::bit_cast<Bits>(normal_min) >> 1;
  MatrixType matrix = MatrixType::Zero(n, n);
  matrix(0, 0) = normal_min;
  matrix(1, 1) = numext::bit_cast<Scalar>(half_min_bits);
  const ScopedFlushToZero flush;
  for (bool compute_u : {false, true}) {
    const RealSchur<MatrixType> schur(matrix, compute_u);
    VERIFY_IS_EQUAL(schur.info(), Success);
    // Compare representations: DAZ can make an erroneous zero compare equal to a subnormal.
    VERIFY_IS_EQUAL(numext::bit_cast<Bits>(schur.matrixT()(1, 1)), half_min_bits);
    VERIFY_IS_EQUAL(numext::bit_cast<Bits>(schur.matrixT()(0, 0)), numext::bit_cast<Bits>(normal_min));
  }
}

template <typename Scalar, int StorageOrder>
void schur_workspace_stride() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, StorageOrder>;
  // The middle size selects the padded workspace for both float and double; its neighbours do not.
  const Index strideSize = 1024 / sizeof(Scalar);
  for (Index n : {strideSize - 1, strideSize, strideSize + 1}) {
    const Mat identity = Mat::Identity(n, n);
    Mat a = Mat::Random(n, n);
    const Scalar bound = Scalar(64 * n) * NumTraits<Scalar>::epsilon();
    RealSchur<Mat> solver(n);
    solver.compute(a);
    VERIFY_IS_EQUAL(solver.info(), Success);
    const Mat u = solver.matrixU(), t = solver.matrixT();
    verifyIsQuasiTriangular(t);
    VERIFY((a - u * t * u.transpose()).norm() <= bound * a.norm());
    VERIFY((u.transpose() * u - identity).norm() <= bound);

    solver.compute(a, false);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY_IS_EQUAL(solver.matrixT(), t);

    HessenbergDecomposition<Mat> hess(a);
    const Mat h = hess.matrixH(), q = hess.matrixQ();
    solver.computeFromHessenberg(h, q, true);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY((a - solver.matrixU() * solver.matrixT() * solver.matrixU().transpose()).norm() <= bound * a.norm());

    solver.setMaxIterations(1).computeFromHessenberg(h, q, true);
    VERIFY_IS_EQUAL(solver.info(), NoConvergence);
    // Partial results must be copied out of the workspace too.
    VERIFY((a - solver.matrixU() * solver.matrixT() * solver.matrixU().transpose()).norm() <= bound * a.norm());
    solver.setMaxIterations(-1);

    // A decoupled trailing block exercises active windows smaller than the padded matrix.
    a.bottomLeftCorner(n / 2, n - n / 2).setZero();
    EigenSolver<Mat> eig(a);
    VERIFY_IS_EQUAL(eig.info(), Success);
    VERIFY((a * eig.pseudoEigenvectors() - eig.pseudoEigenvectors() * eig.pseudoEigenvalueMatrix()).norm() <=
           bound * a.norm() * eig.pseudoEigenvectors().norm());
    solver.compute(identity);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY_IS_EQUAL(solver.matrixT(), identity);
    VERIFY_IS_EQUAL(solver.matrixU(), identity);

    // Exercise subnormal restoration across the padded-workspace boundary too.
    real_schur_subnormal_restoration<Mat>(n);

    a.setZero();
    for (Scalar nonfinite : {NumTraits<Scalar>::quiet_NaN(), NumTraits<Scalar>::infinity()}) {
      a(n - 1, n - 1) = nonfinite;
      solver.compute(a);
      VERIFY_IS_EQUAL(solver.info(), NoConvergence);
      VERIFY(!(numext::isfinite)(solver.matrixT()(n - 1, n - 1)));
      VERIFY_RAISES_ASSERT(solver.matrixU());
      eig.compute(a, false);
      VERIFY_IS_EQUAL(eig.info(), NumericalIssue);
    }
  }
}

template <typename Scalar, int StorageOrder>
void schur_workspace_ref() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, StorageOrder>;
  using StridedRef = Ref<Mat, 0, Stride<Dynamic, Dynamic>>;
  const Index strideSize = 1024 / sizeof(Scalar);
  for (Index n : {strideSize - 1, strideSize}) {
    const Mat a = Mat::Random(n, n);
    const Mat identity = Mat::Identity(n, n);
    const Scalar bound = Scalar(64 * n) * NumTraits<Scalar>::epsilon();
    HessenbergDecomposition<Mat> hess(a);
    const Mat h = hess.matrixH(), q = hess.matrixQ();
    for (Index inner : {Index(1), Index(2)}) {
      for (Index extra : {Index(0), Index(1)}) {
        const Index outer = n * inner + extra;
        Matrix<Scalar, Dynamic, 1> storage = Matrix<Scalar, Dynamic, 1>::Constant(n * outer + 2, Scalar(17));
        Map<Mat, 0, Stride<Dynamic, Dynamic>> matrix(storage.data() + 1, n, n, Stride<Dynamic, Dynamic>(outer, inner));
        matrix = a;
        RealSchur<StridedRef> solver(matrix);
        VERIFY_IS_EQUAL(solver.info(), Success);
        VERIFY(internal::is_same_dense(matrix, solver.matrixT()));
        VERIFY((a - solver.matrixU() * matrix * solver.matrixU().transpose()).norm() <= bound * a.norm());
        VERIFY((solver.matrixU().transpose() * solver.matrixU() - identity).norm() <= bound);

        const Mat input = a * Scalar(8);
        solver.compute(input, false);
        VERIFY_IS_EQUAL(solver.info(), Success);
        VERIFY_IS_EQUAL(input, a * Scalar(8));
        VERIFY_RAISES_ASSERT(solver.matrixU());
        for (Index maxIters : {Index(1), Index(-1)}) {
          solver.setMaxIterations(maxIters).compute(input);
          VERIFY_IS_EQUAL(solver.info(), maxIters == -1 ? Success : NoConvergence);
          VERIFY((input - solver.matrixU() * matrix * solver.matrixU().transpose()).norm() <= bound * input.norm());
          solver.computeFromHessenberg(h, q, true);
          VERIFY_IS_EQUAL(solver.info(), maxIters == -1 ? Success : NoConvergence);
          VERIFY((a - solver.matrixU() * matrix * solver.matrixU().transpose()).norm() <= bound * a.norm());
        }
        // Exercise aliased input with an already suitable stride as well as a cache-conflicting one.
        matrix = h;
        solver.computeFromHessenberg(matrix, q, true);
        VERIFY_IS_EQUAL(solver.info(), Success);
        VERIFY((a - solver.matrixU() * matrix * solver.matrixU().transpose()).norm() <= bound * a.norm());
        solver.compute(Mat::Zero(n, n));
        VERIFY_IS_EQUAL(solver.info(), Success);
        VERIFY_IS_EQUAL(matrix, Mat::Zero(n, n));
        VERIFY_IS_EQUAL(solver.matrixU(), identity);
        VERIFY(internal::is_same_dense(matrix, solver.matrixT()));
        for (Index i = 0; i < storage.size(); ++i) {
          const Index offset = i - 1;
          const bool inMatrix =
              offset >= 0 && offset / outer < n && offset % outer < n * inner && offset % outer % inner == 0;
          if (!inMatrix) VERIFY_IS_EQUAL(storage(i), Scalar(17));
        }
      }
    }
  }
}

EIGEN_DECLARE_TEST(schur_real) {
  CALL_SUBTEST_1((schur<Matrix4f>()));
  CALL_SUBTEST_2((schur<MatrixXd>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4))));
  CALL_SUBTEST_3((schur<Matrix<float, 1, 1> >()));
  CALL_SUBTEST_4((schur<Matrix<double, 3, 3, Eigen::RowMajor> >()));
  CALL_SUBTEST_1((schur<Matrix<float, Dynamic, Dynamic, RowMajor>>(17)));
  CALL_SUBTEST_2((schur<Matrix<double, Dynamic, Dynamic, RowMajor>>(17)));

  // Test problem size constructors
  CALL_SUBTEST_5(RealSchur<MatrixXf>(10));

  CALL_SUBTEST_6((test_bug2633()));
  CALL_SUBTEST_6((real_schur_power_of_two_scaling()));
  CALL_SUBTEST_6(real_schur_subnormal_restoration<Matrix2f>());
  CALL_SUBTEST_6(real_schur_subnormal_restoration<Matrix2d>());
  CALL_SUBTEST_7((schur_workspace_stride<float, ColMajor>()));
  CALL_SUBTEST_8((schur_workspace_stride<double, ColMajor>()));
  CALL_SUBTEST_9((schur_workspace_stride<float, RowMajor>()));
  CALL_SUBTEST_10((schur_workspace_stride<double, RowMajor>()));
  CALL_SUBTEST_11((schur_workspace_ref<float, ColMajor>()));
  CALL_SUBTEST_12((schur_workspace_ref<double, ColMajor>()));
  CALL_SUBTEST_13((schur_workspace_ref<float, RowMajor>()));
  CALL_SUBTEST_14((schur_workspace_ref<double, RowMajor>()));
}
