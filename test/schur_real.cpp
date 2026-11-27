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
  using Scalar = typename MatrixType::Scalar;

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
      // The test RealSchur::splitOffTwoRows keeps a block on: q = p^2 + bc < 0 with p = (a-d)/2.
      // tr^2 - 4*det would cancel the common diagonal shift.
      const Scalar half_difference = (T(row - 1, row - 1) - T(row, row)) / Scalar(2);
      VERIFY(half_difference * half_difference + T(row - 1, row) * T(row, row - 1) < Scalar(0));
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
    const MatrixType& U = schurOfA.matrixU();
    const MatrixType& T = schurOfA.matrixT();
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

// Structured inputs larger than one window of the Francis step's deferred left updates (32 reflectors). The
// permutation, companion and Jordan matrices give reflectors with exactly zero bulges (tau == 0), which the deferred
// path must skip as the immediate one does.
template <typename Scalar>
void real_schur_structured_windows() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  for (Index n : {Index(40), Index(70)}) {
    const MatrixType identity = MatrixType::Identity(n, n);
    // Backward stable: ||A - U T U^T|| and ||U^T U - I|| are O(n eps) relative to ||A|| and 1.
    const Scalar bound = Scalar(10 * n) * NumTraits<Scalar>::epsilon();
    MatrixType cyclic = MatrixType::Zero(n, n);
    for (Index i = 0; i < n; ++i) cyclic(i, (i + 1) % n) = Scalar(1);
    MatrixType companion = MatrixType::Zero(n, n);
    companion.diagonal(-1).setOnes();
    companion.col(n - 1).setRandom();
    MatrixType jordan = MatrixType::Identity(n, n) * Scalar(2);
    jordan.diagonal(1).setOnes();
    MatrixType decoupled = MatrixType::Random(n, n).template triangularView<Upper>();
    decoupled.diagonal(-1).setRandom();
    decoupled(n / 2, n / 2 - 1) = Scalar(0);
    for (const MatrixType* a : {&cyclic, &companion, &jordan, &decoupled}) {
      RealSchur<MatrixType> rs(*a);
      VERIFY_IS_EQUAL(rs.info(), Success);
      const MatrixType& T = rs.matrixT();
      const MatrixType& U = rs.matrixU();
      verifyIsQuasiTriangular(T);
      VERIFY((*a - U * T * U.transpose()).norm() <= bound * a->norm());
      VERIFY((U.transpose() * U - identity).norm() <= bound);
    }
  }
}

template <typename = void>
void test_bug2633() {
  Eigen::MatrixXd A(4, 4);
  A << 0, 0, 0, -2, 1, 0, 0, -0, 0, 1, 0, 2, 0, 0, 2, -0;
  RealSchur<Eigen::MatrixXd> schur(A);
  VERIFY(schur.info() == Eigen::Success);
}

template <typename = void>
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

// Reordering keeps A = U T U^T while moving the selected eigenvalues into the leading diagonal blocks, so that the
// leading columns of U span their invariant subspace.
template <typename MatrixType>
void schur_reorder(Index size = MatrixType::ColsAtCompileTime) {
  using Scalar = typename MatrixType::Scalar;
  using ComplexScalar = std::complex<Scalar>;
  using DynamicMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  const auto negativeRealPart = [](const ComplexScalar& lambda) { return lambda.real() < Scalar(0); };
  const MatrixType identity = MatrixType::Identity(size, size);
  for (int counter = 0; counter < g_repeat; ++counter) {
    const MatrixType a = MatrixType::Random(size, size);
    const Scalar bound = Scalar(500 * size) * NumTraits<Scalar>::epsilon() * a.norm();
    const Scalar orthogonalBound = Scalar(500 * size) * NumTraits<Scalar>::epsilon();

    // reorder() decides on the eigenvalues of T, so take the reference from A and skip the comparison when an
    // eigenvalue sits close enough to the selection boundary for the two to disagree.
    const EigenSolver<MatrixType> eigenvalues(a);
    Index expected = 0;
    bool ambiguous = false;
    for (Index i = 0; i < size; ++i) {
      const ComplexScalar lambda = eigenvalues.eigenvalues()(i);
      if (negativeRealPart(lambda)) ++expected;
      if (numext::abs(lambda.real()) <= bound) ambiguous = true;
    }

    RealSchur<MatrixType> schur(a);
    VERIFY_IS_EQUAL(schur.info(), Success);
    const Index selected = schur.reorder(negativeRealPart);
    const MatrixType& t = schur.matrixT();
    const MatrixType& u = schur.matrixU();
    verifyIsQuasiTriangular(t);
    VERIFY((a - u * t * u.transpose()).norm() <= bound * Scalar(size));
    VERIFY((u.transpose() * u - identity).norm() <= orthogonalBound * Scalar(size));
    if (!ambiguous) VERIFY_IS_EQUAL(selected, expected);

    // The leading block holds exactly the selected eigenvalues and spans an invariant subspace of A.
    if (selected > 0) {
      const EigenSolver<DynamicMatrix> leading(t.topLeftCorner(selected, selected));
      for (Index i = 0; i < selected; ++i) VERIFY(leading.eigenvalues()(i).real() < bound);
      const DynamicMatrix basis = u.leftCols(selected);
      VERIFY((a * basis - basis * t.topLeftCorner(selected, selected)).norm() <= bound * Scalar(size));
    }

    // Selecting everything or nothing moves nothing, and reordering a second time is a no-op.
    const MatrixType before = t;
    VERIFY_IS_EQUAL(schur.reorder([](const ComplexScalar&) { return true; }), size);
    VERIFY_IS_EQUAL(schur.matrixT(), before);
    VERIFY_IS_EQUAL(schur.reorder([](const ComplexScalar&) { return false; }), Index(0));
    VERIFY_IS_EQUAL(schur.matrixT(), before);
    if (!ambiguous) {
      VERIFY_IS_EQUAL(schur.reorder(negativeRealPart), selected);
      VERIFY_IS_EQUAL(schur.matrixT(), before);
    }
  }
}

// Every eigenvalue of a block diagonal of rotations is a complex pair, so the swaps have to move whole 2x2 blocks,
// including a 2x2 block past another 2x2 block.
template <typename Scalar>
void schur_reorder_complex_pairs() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  using ComplexScalar = std::complex<Scalar>;
  const Index size = 8;
  const Scalar angles[4] = {Scalar(0.3), Scalar(2.5), Scalar(1.2), Scalar(2.9)};
  MatrixType diagonal = MatrixType::Zero(size, size);
  Index expected = 0;
  for (int block = 0; block < 4; ++block) {
    const Scalar cosine = std::cos(angles[block]), sine = std::sin(angles[block]);
    diagonal(2 * block, 2 * block) = cosine;
    diagonal(2 * block, 2 * block + 1) = -sine;
    diagonal(2 * block + 1, 2 * block) = sine;
    diagonal(2 * block + 1, 2 * block + 1) = cosine;
    if (cosine < Scalar(0)) expected += 2;
  }
  // Spread the blocks with an orthogonal similarity so that T starts from a dense matrix.
  const MatrixType orthogonal = RealSchur<MatrixType>(MatrixType::Random(size, size)).matrixU();
  const MatrixType a = orthogonal * diagonal * orthogonal.transpose();
  const Scalar bound = Scalar(500 * size) * NumTraits<Scalar>::epsilon() * a.norm();

  RealSchur<MatrixType> schur(a);
  VERIFY_IS_EQUAL(schur.info(), Success);
  bool isComplexSelection = true;
  const Index selected = schur.reorder([&isComplexSelection](const ComplexScalar& lambda) {
    isComplexSelection = isComplexSelection && lambda.imag() != Scalar(0);
    return lambda.real() < Scalar(0);
  });
  VERIFY(isComplexSelection);
  VERIFY_IS_EQUAL(selected, expected);
  const MatrixType& t = schur.matrixT();
  const MatrixType& u = schur.matrixU();
  verifyIsQuasiTriangular(t);
  VERIFY((a - u * t * u.transpose()).norm() <= bound * Scalar(size));
  const MatrixType basis = u.leftCols(selected);
  VERIFY((a * basis - basis * t.topLeftCorner(selected, selected)).norm() <= bound * Scalar(size));
  for (Index i = 0; i < selected; i += 2) {
    VERIFY(!numext::is_exactly_zero(t(i + 1, i)));
    const Scalar halfDifference = (t(i, i) - t(i + 1, i + 1)) / 2;
    VERIFY(halfDifference * halfDifference + t(i, i + 1) * t(i + 1, i) < Scalar(0));
  }
}

// Equal eigenvalues cannot be separated in floating point: such a reordering must leave the matrix in a valid Schur
// form, with every eigenvalue it reports as leading really selected.
template <typename Scalar>
void schur_reorder_repeated() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  using ComplexScalar = std::complex<Scalar>;
  const Index size = 5;
  MatrixType a = MatrixType::Zero(size, size);
  a.diagonal() << Scalar(1), Scalar(1), Scalar(-1), Scalar(-1), Scalar(-2);
  a(0, 2) = Scalar(1);
  a(0, 3) = Scalar(-2);
  a(1, 4) = Scalar(1);
  const auto negativeRealPart = [](const ComplexScalar& lambda) { return lambda.real() < Scalar(0); };
  const Scalar bound = Scalar(500 * size) * NumTraits<Scalar>::epsilon() * a.norm();

  for (bool computeU : {true, false}) {
    RealSchur<MatrixType> schur(a, computeU);
    VERIFY_IS_EQUAL(schur.info(), Success);
    const Scalar traceBefore = schur.matrixT().diagonal().sum();
    const Index selected = schur.reorder(negativeRealPart);
    const MatrixType& t = schur.matrixT();
    verifyIsQuasiTriangular(t);
    // The similarity keeps the trace and the size of the reordered block. The two equal -1 entries never have to
    // pass each other, so all three negative eigenvalues can lead.
    VERIFY(numext::abs(t.diagonal().sum() - traceBefore) <= bound);
    VERIFY_IS_EQUAL(selected, 3);
    if (selected > 0) {
      const EigenSolver<MatrixType> leading(t.topLeftCorner(selected, selected));
      for (Index i = 0; i < selected; ++i) VERIFY(leading.eigenvalues()(i).real() < bound);
    }
    if (computeU) {
      const MatrixType& u = schur.matrixU();
      VERIFY((a - u * t * u.transpose()).norm() <= bound * Scalar(size));
      if (selected > 0) {
        const MatrixType basis = u.leftCols(selected);
        VERIFY((a * basis - basis * t.topLeftCorner(selected, selected)).norm() <= bound * Scalar(size));
      }
    } else {
      VERIFY_RAISES_ASSERT(schur.matrixU());
    }
  }
}

// A swap whose blocks cannot be separated must be rejected, and the count must stop short of the blocks that did not
// move. Two identical 2x2 blocks with a selector that accepts only the later one is the smallest such case.
template <typename Scalar>
void schur_reorder_rejected() {
  using MatrixType = Matrix<Scalar, 4, 4>;
  using ComplexScalar = std::complex<Scalar>;
  MatrixType a = MatrixType::Zero();
  for (int block = 0; block < 2; ++block) {
    a(2 * block, 2 * block) = Scalar(1);
    a(2 * block, 2 * block + 1) = Scalar(2);
    a(2 * block + 1, 2 * block) = Scalar(-2);
    a(2 * block + 1, 2 * block + 1) = Scalar(1);
  }

  RealSchur<MatrixType> schur;
  schur.computeFromHessenberg(a, MatrixType::Identity(), true);
  VERIFY_IS_EQUAL(schur.info(), Success);
  int calls = 0;
  const Index selected = schur.reorder([&calls](const ComplexScalar&) { return ++calls == 2; });
  VERIFY_IS_EQUAL(selected, 0);
  VERIFY_IS_EQUAL((schur.matrixT() - a).cwiseAbs().maxCoeff(), Scalar(0));
}

// Finite input near the scalar limits must not turn the decomposition into NaN.
template <typename Scalar>
void schur_reorder_overflow() {
  using MatrixType = Matrix<Scalar, 2, 2>;
  using ComplexScalar = std::complex<Scalar>;
  const Scalar highest = NumTraits<Scalar>::highest();

  {
    // The via-Sylvester solution is highest/4, whose squared norm overflows.
    MatrixType a;
    a << Scalar(1), highest / 4, Scalar(0), Scalar(2);
    RealSchur<MatrixType> schur;
    schur.computeFromHessenberg(a, MatrixType::Identity(), true);
    const Index selected = schur.reorder([](const ComplexScalar& lambda) { return lambda.real() > Scalar(1.5); });
    VERIFY_IS_EQUAL(selected, 1);
    VERIFY(schur.matrixT().allFinite());
    VERIFY(schur.matrixU().allFinite());
    VERIFY_IS_APPROX(schur.matrixT()(0, 0), Scalar(2));
    VERIFY_IS_APPROX(schur.matrixT()(1, 1), Scalar(1));
  }

  {
    // The eigenvalue difference overflows, so the uncoupled pair must take the Givens path.
    MatrixType a = MatrixType::Zero();
    a(0, 0) = highest;
    a(1, 1) = -highest;
    RealSchur<MatrixType> schur;
    schur.computeFromHessenberg(a, MatrixType::Identity(), true);
    const Index selected = schur.reorder([](const ComplexScalar& lambda) { return lambda.real() < Scalar(0); });
    VERIFY_IS_EQUAL(selected, 1);
    VERIFY(schur.matrixT().allFinite());
    VERIFY_IS_APPROX(schur.matrixT()(0, 0), -highest);
    VERIFY_IS_APPROX(schur.matrixT()(1, 1), highest);
  }

  // The discriminant must be formed without squaring the off-diagonal entries: highest/4 squares to infinity, and
  // the selector must still see the finite 10^154-sized eigenvalue rather than (0, inf).
  const Scalar magnitude = highest / Scalar(4);
  {
    MatrixType a = MatrixType::Zero();
    a(0, 1) = -magnitude;
    a(1, 0) = magnitude;
    RealSchur<MatrixType> schur;
    schur.computeFromHessenberg(a, MatrixType::Identity(), true);
    const Index selected = schur.reorder([magnitude](const ComplexScalar& lambda) {
      return lambda.imag() > Scalar(0) && lambda.imag() <= magnitude;
    });
    VERIFY_IS_EQUAL(selected, 2);
    VERIFY(schur.matrixT().allFinite());
  }
}

EIGEN_DECLARE_TEST(schur_real) {
  // RealSchur keeps these blocks (|c| > eps*(|a|+|d|)), but tr^2 and 4*det round to the same value.
  Matrix2f small_imaginary_float;
  small_imaginary_float << 1, 0.1f, -3e-7f, 1;
  CALL_SUBTEST_1(verifyIsQuasiTriangular(RealSchur<Matrix2f>(small_imaginary_float).matrixT()));
  Matrix2d small_imaginary_double;
  small_imaginary_double << 1, 0.1, -5e-16, 1;
  CALL_SUBTEST_2(verifyIsQuasiTriangular(RealSchur<Matrix2d>(small_imaginary_double).matrixT()));

  CALL_SUBTEST_1((schur<Matrix4f>()));
  CALL_SUBTEST_2((schur<MatrixXd>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 4))));
  CALL_SUBTEST_3((schur<Matrix<float, 1, 1> >()));
  CALL_SUBTEST_4((schur<Matrix<double, 3, 3, Eigen::RowMajor> >()));
  CALL_SUBTEST_1((schur<Matrix<float, Dynamic, Dynamic, RowMajor>>(17)));
  CALL_SUBTEST_2((schur<Matrix<double, Dynamic, Dynamic, RowMajor>>(17)));

  // Test problem size constructors
  CALL_SUBTEST_5(RealSchur<MatrixXf>(10));

  CALL_SUBTEST_6((test_bug2633<>()));
  CALL_SUBTEST_6((real_schur_power_of_two_scaling<>()));
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
  CALL_SUBTEST_15((real_schur_structured_windows<float>()));
  CALL_SUBTEST_15((real_schur_structured_windows<double>()));

  CALL_SUBTEST_16((schur_reorder<Matrix4f>()));
  CALL_SUBTEST_16((schur_reorder<MatrixXd>(internal::random<int>(2, EIGEN_TEST_MAX_SIZE / 4))));
  CALL_SUBTEST_16((schur_reorder<Matrix<double, Dynamic, Dynamic, RowMajor>>(9)));
  CALL_SUBTEST_17(schur_reorder_complex_pairs<float>());
  CALL_SUBTEST_17(schur_reorder_complex_pairs<double>());
  CALL_SUBTEST_18(schur_reorder_repeated<float>());
  CALL_SUBTEST_18(schur_reorder_repeated<double>());
  CALL_SUBTEST_19(schur_reorder_rejected<float>());
  CALL_SUBTEST_19(schur_reorder_rejected<double>());
  CALL_SUBTEST_20(schur_reorder_overflow<float>());
  CALL_SUBTEST_20(schur_reorder_overflow<double>());
}
