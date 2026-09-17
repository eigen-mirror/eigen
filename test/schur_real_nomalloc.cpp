// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#if defined(EIGEN_TEST_PART_2)
#define EIGEN_NO_MALLOC
#else
#define EIGEN_RUNTIME_NO_MALLOC
#endif

#include "main.h"
#include <Eigen/Eigenvalues>

template <typename MatrixType>
void schur_no_malloc() {
  using Scalar = typename MatrixType::Scalar;
  using VectorType = typename RealSchur<MatrixType>::ColumnVectorType;
  // These sizes would select padding if allocation and deallocation were allowed.
  const Index n = 1024 / sizeof(Scalar);
  MatrixType h = MatrixType::Random(n, n);
  h.bottomLeftCorner(n - 2, n - 2).template triangularView<Lower>().setZero();
  MatrixType q = MatrixType::Identity(n, n);
  for (Index i = 0; i < n; i += 2) q(i, i) = Scalar(-1);
  const MatrixType a = q.diagonal().asDiagonal() * h * q.diagonal().asDiagonal();
  const Scalar bound = Scalar(64 * n) * NumTraits<Scalar>::epsilon();
  MatrixType referenceT(n, n);
  VectorType residual(n);
  RealSchur<MatrixType> solver(n);

#ifdef EIGEN_RUNTIME_NO_MALLOC
  for (int restriction : {1, 2, 3, 0}) {
    internal::set_is_malloc_allowed((restriction & 1) == 0);
    internal::set_is_free_allowed((restriction & 2) == 0);
#endif
    solver.setMaxIterations(-1).compute(a, false);
    VERIFY_IS_EQUAL(solver.info(), Success);
    // An already Hessenberg input has no reflectors, so forming its initial Q must not allocate.
    solver.compute(h, true);
#ifdef EIGEN_RUNTIME_NO_MALLOC
    internal::set_is_malloc_allowed(true);
    internal::set_is_free_allowed(true);
#endif
    VERIFY_IS_EQUAL(solver.info(), Success);
    Scalar hessenbergError(0);
    for (Index j = 0; j < n; ++j) {
      residual.noalias() = h * solver.matrixU().col(j);
      residual.noalias() -= solver.matrixU() * solver.matrixT().col(j);
      hessenbergError += residual.squaredNorm();
    }
    VERIFY(numext::sqrt(hessenbergError) <= bound * h.norm());
    for (Index maxIters : {Index(-1), Index(1)}) {
      solver.setMaxIterations(maxIters);
      for (bool computeU : {true, false}) {
#ifdef EIGEN_RUNTIME_NO_MALLOC
        internal::set_is_malloc_allowed((restriction & 1) == 0);
        internal::set_is_free_allowed((restriction & 2) == 0);
#endif
        solver.computeFromHessenberg(h, q, computeU);
#ifdef EIGEN_RUNTIME_NO_MALLOC
        internal::set_is_malloc_allowed(true);
        internal::set_is_free_allowed(true);
#endif
        VERIFY_IS_EQUAL(solver.info(), maxIters == -1 ? Success : NoConvergence);
        if (computeU) {
          referenceT = solver.matrixT();
          const MatrixType& u = solver.matrixU();
          Scalar reconstructionError(0), orthogonalityError(0);
          // Matrix-vector products also keep verification allocation-free in the compile-time no-malloc part.
          for (Index j = 0; j < n; ++j) {
            residual.noalias() = a * u.col(j);
            residual.noalias() -= u * referenceT.col(j);
            reconstructionError += residual.squaredNorm();
            residual.noalias() = u.transpose() * u.col(j);
            residual(j) -= Scalar(1);
            orthogonalityError += residual.squaredNorm();
          }
          VERIFY(numext::sqrt(reconstructionError) <= bound * a.norm());
          VERIFY(numext::sqrt(orthogonalityError) <= bound);
        } else {
          VERIFY_IS_EQUAL(solver.matrixT(), referenceT);
        }
      }
    }
#ifdef EIGEN_RUNTIME_NO_MALLOC
  }
#endif
}

EIGEN_DECLARE_TEST(schur_real_nomalloc) {
  CALL_SUBTEST_1((schur_no_malloc<Matrix<float, Dynamic, Dynamic, ColMajor>>()));
  CALL_SUBTEST_1((schur_no_malloc<Matrix<float, Dynamic, Dynamic, RowMajor>>()));
  CALL_SUBTEST_1((schur_no_malloc<Matrix<double, Dynamic, Dynamic, ColMajor>>()));
  CALL_SUBTEST_1((schur_no_malloc<Matrix<double, Dynamic, Dynamic, RowMajor>>()));
  CALL_SUBTEST_2((schur_no_malloc<Matrix<double, 128, 128, ColMajor>>()));
  CALL_SUBTEST_2((schur_no_malloc<Matrix<double, 128, 128, RowMajor>>()));
  CALL_SUBTEST_2((schur_no_malloc<Matrix<double, Dynamic, Dynamic, ColMajor, 128, 128>>()));
  CALL_SUBTEST_2((schur_no_malloc<Matrix<double, Dynamic, Dynamic, RowMajor, 128, 128>>()));
}
