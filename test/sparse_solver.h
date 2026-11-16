// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TEST_SPARSE_SOLVER_H
#define EIGEN_TEST_SPARSE_SOLVER_H

#include "sparse.h"
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <sstream>

template <typename Solver, typename Rhs, typename Guess, typename Result>
void solve_with_guess(IterativeSolverBase<Solver>& solver, const MatrixBase<Rhs>& b, const Guess& g, Result& x) {
  if (internal::random<bool>()) {
    // With a temporary through evaluator<SolveWithGuess>
    x = solver.derived().solveWithGuess(b, g) + Result::Zero(x.rows(), x.cols());
  } else {
    // direct evaluation within x through Assignment<Result,SolveWithGuess>
    x = solver.derived().solveWithGuess(b.derived(), g);
  }
}

template <typename Solver, typename Rhs, typename Guess, typename Result>
void solve_with_guess(SparseSolverBase<Solver>& solver, const MatrixBase<Rhs>& b, const Guess&, Result& x) {
  if (internal::random<bool>())
    x = solver.derived().solve(b) + Result::Zero(x.rows(), x.cols());
  else
    x = solver.derived().solve(b);
}

template <typename Solver, typename Rhs, typename Guess, typename Result>
void solve_with_guess(SparseSolverBase<Solver>& solver, const SparseMatrixBase<Rhs>& b, const Guess&, Result& x) {
  x = solver.derived().solve(b);
}

template <typename Solver, typename Rhs, typename DenseMat, typename DenseRhs>
void check_sparse_solving(Solver& solver, const typename Solver::MatrixType& A, const Rhs& b, const DenseMat& dA,
                          const DenseRhs& db) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;
  using StorageIndex = typename Mat::StorageIndex;

  DenseRhs refX = dA.householderQr().solve(db);
  {
    Rhs x(A.cols(), b.cols());
    Rhs oldb = b;

    solver.compute(A);
    if (solver.info() != Success) {
      std::cerr << "ERROR | sparse solver testing, factorization failed (" << typeid(Solver).name() << ")\n";
      VERIFY(solver.info() == Success);
    }
    x = solver.solve(b);
    if (solver.info() != Success) {
      std::cerr << "ERROR | sparse solver testing, solving failed (" << typeid(Solver).name() << ")\n";
    }
    // A solver that does not converge must fail the test: the problems of these tests are well conditioned.
    VERIFY(solver.info() == Success && "sparse solver testing: solving failed");
    VERIFY(oldb.isApprox(b) && "sparse solver testing: the rhs should not be modified!");
    VERIFY(x.isApprox(refX, test_precision<Scalar>()));

    x.setZero();
    solve_with_guess(solver, b, x, x);
    VERIFY(solver.info() == Success && "solving failed when using solve_with_guess API");
    VERIFY(oldb.isApprox(b) && "sparse solver testing: the rhs should not be modified!");
    VERIFY(x.isApprox(refX, test_precision<Scalar>()));

    x.setZero();
    // test the analyze/factorize API
    solver.analyzePattern(A);
    solver.factorize(A);
    VERIFY(solver.info() == Success && "factorization failed when using analyzePattern/factorize API");
    x = solver.solve(b);
    VERIFY(solver.info() == Success && "solving failed when using analyzePattern/factorize API");
    VERIFY(oldb.isApprox(b) && "sparse solver testing: the rhs should not be modified!");
    VERIFY(x.isApprox(refX, test_precision<Scalar>()));

    x.setZero();
    // test with Map
    Map<SparseMatrix<Scalar, Mat::Options, StorageIndex>> Am(
        A.rows(), A.cols(), A.nonZeros(), const_cast<StorageIndex*>(A.outerIndexPtr()),
        const_cast<StorageIndex*>(A.innerIndexPtr()), const_cast<Scalar*>(A.valuePtr()));
    solver.compute(Am);
    VERIFY(solver.info() == Success && "factorization failed when using Map");
    DenseRhs dx(refX);
    dx.setZero();
    Map<DenseRhs> xm(dx.data(), dx.rows(), dx.cols());
    Map<const DenseRhs> bm(db.data(), db.rows(), db.cols());
    xm = solver.solve(bm);
    VERIFY(solver.info() == Success && "solving failed when using Map");
    VERIFY(oldb.isApprox(bm) && "sparse solver testing: the rhs should not be modified!");
    VERIFY(xm.isApprox(refX, test_precision<Scalar>()));

    // Test with a Map and non-unit stride.
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(2 * xm.rows(), 2 * xm.cols());
    out.setZero();
    // The outer stride spans DenseRhs's outer dimension, so it must be taken from outerStride()
    // rather than from rows(): the two differ when DenseRhs is row major.
    Eigen::Map<DenseRhs, 0, Stride<Eigen::Dynamic, 2>> outm(out.data(), xm.rows(), xm.cols(),
                                                            Stride<Eigen::Dynamic, 2>(2 * xm.outerStride(), 2));
    outm = solver.solve(bm);
    VERIFY(outm.isApprox(refX, test_precision<Scalar>()));

    // Test a right-hand side with non-unit strides.
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> bstore(2 * db.rows(), 2 * db.cols());
    bstore.setZero();
    Eigen::Map<DenseRhs, 0, Stride<Eigen::Dynamic, 2>> bstrided(bstore.data(), db.rows(), db.cols(),
                                                                Stride<Eigen::Dynamic, 2>(2 * db.outerStride(), 2));
    bstrided = db;
    dx = solver.solve(bstrided);
    VERIFY(dx.isApprox(refX, test_precision<Scalar>()));
  }

  // if not too large, do some extra check:
  if (A.rows() < 2000) {
    // test initialization ctor
    {
      Rhs x(b.rows(), b.cols());
      Solver solver2(A);
      VERIFY(solver2.info() == Success);
      x = solver2.solve(b);
      VERIFY(x.isApprox(refX, test_precision<Scalar>()));
    }

    // test dense Block as the result and rhs:
    {
      DenseRhs x(refX.rows(), refX.cols());
      DenseRhs oldb(db);
      x.setZero();
      x.block(0, 0, x.rows(), x.cols()) = solver.solve(db.block(0, 0, db.rows(), db.cols()));
      VERIFY(oldb.isApprox(db) && "sparse solver testing: the rhs should not be modified!");
      VERIFY(x.isApprox(refX, test_precision<Scalar>()));
    }

    // test uncompressed inputs
    {
      Mat A2 = A;
      A2.reserve((ArrayXf::Random(A.outerSize()) + 2).template cast<typename Mat::StorageIndex>().eval());
      solver.compute(A2);
      Rhs x = solver.solve(b);
      VERIFY(x.isApprox(refX, test_precision<Scalar>()));
    }

    // test expression as input
    {
      solver.compute(0.5 * (A + A));
      Rhs x = solver.solve(b);
      VERIFY(x.isApprox(refX, test_precision<Scalar>()));

      Solver solver2(0.5 * (A + A));
      Rhs x2 = solver2.solve(b);
      VERIFY(x2.isApprox(refX, test_precision<Scalar>()));
    }
  }
}

// specialization of generic check_sparse_solving for SuperLU in order to also test adjoint and transpose solves
template <typename Scalar, typename Rhs, typename DenseMat, typename DenseRhs>
void check_sparse_solving(Eigen::SparseLU<Eigen::SparseMatrix<Scalar>>& solver,
                          const typename Eigen::SparseMatrix<Scalar>& A, const Rhs& b, const DenseMat& dA,
                          const DenseRhs& db) {
  using Mat = typename Eigen::SparseMatrix<Scalar>;
  using StorageIndex = typename Mat::StorageIndex;
  using Solver = typename Eigen::SparseLU<Eigen::SparseMatrix<Scalar>>;

  // reference solutions computed by dense QR solver
  DenseRhs refX1 = dA.householderQr().solve(db);              // solution of A x = db
  DenseRhs refX2 = dA.transpose().householderQr().solve(db);  // solution of A^T * x = db (use transposed matrix A^T)
  DenseRhs refX3 = dA.adjoint().householderQr().solve(db);    // solution of A^* * x = db (use adjoint matrix A^*)

  {
    Rhs x1(A.cols(), b.cols());
    Rhs x2(A.cols(), b.cols());
    Rhs x3(A.cols(), b.cols());
    Rhs oldb = b;

    solver.compute(A);
    if (solver.info() != Success) {
      std::cerr << "ERROR | sparse solver testing, factorization failed (" << typeid(Solver).name() << ")\n";
      VERIFY(solver.info() == Success);
    }
    x1 = solver.solve(b);
    if (solver.info() != Success) {
      std::cerr << "ERROR | sparse solver testing, solving failed (" << typeid(Solver).name() << ")\n";
    }
    VERIFY(solver.info() == Success && "sparse solver testing: solving failed");
    VERIFY(oldb.isApprox(b, 0.0) && "sparse solver testing: the rhs should not be modified!");
    VERIFY(x1.isApprox(refX1, test_precision<Scalar>()));

    // test solve with transposed
    x2 = solver.transpose().solve(b);
    VERIFY(oldb.isApprox(b) && "sparse solver testing: the rhs should not be modified!");
    VERIFY(x2.isApprox(refX2, test_precision<Scalar>()));

    // test solve with adjoint
    // solver.template _solve_impl_transposed<true>(b, x3);
    x3 = solver.adjoint().solve(b);
    VERIFY(oldb.isApprox(b, 0.0) && "sparse solver testing: the rhs should not be modified!");
    VERIFY(x3.isApprox(refX3, test_precision<Scalar>()));

    x1.setZero();
    solve_with_guess(solver, b, x1, x1);
    VERIFY(solver.info() == Success && "solving failed when using analyzePattern/factorize API");
    VERIFY(oldb.isApprox(b, 0.0) && "sparse solver testing: the rhs should not be modified!");
    VERIFY(x1.isApprox(refX1, test_precision<Scalar>()));

    x1.setZero();
    x2.setZero();
    x3.setZero();
    // test the analyze/factorize API
    solver.analyzePattern(A);
    solver.factorize(A);
    VERIFY(solver.info() == Success && "factorization failed when using analyzePattern/factorize API");
    x1 = solver.solve(b);
    x2 = solver.transpose().solve(b);
    x3 = solver.adjoint().solve(b);

    VERIFY(solver.info() == Success && "solving failed when using analyzePattern/factorize API");
    VERIFY(oldb.isApprox(b, 0.0) && "sparse solver testing: the rhs should not be modified!");
    VERIFY(x1.isApprox(refX1, test_precision<Scalar>()));
    VERIFY(x2.isApprox(refX2, test_precision<Scalar>()));
    VERIFY(x3.isApprox(refX3, test_precision<Scalar>()));

    x1.setZero();
    // test with Map
    Map<SparseMatrix<Scalar, Mat::Options, StorageIndex>> Am(
        A.rows(), A.cols(), A.nonZeros(), const_cast<StorageIndex*>(A.outerIndexPtr()),
        const_cast<StorageIndex*>(A.innerIndexPtr()), const_cast<Scalar*>(A.valuePtr()));
    solver.compute(Am);
    VERIFY(solver.info() == Success && "factorization failed when using Map");
    DenseRhs dx(refX1);
    dx.setZero();
    Map<DenseRhs> xm(dx.data(), dx.rows(), dx.cols());
    Map<const DenseRhs> bm(db.data(), db.rows(), db.cols());
    xm = solver.solve(bm);
    VERIFY(solver.info() == Success && "solving failed when using Map");
    VERIFY(oldb.isApprox(bm, 0.0) && "sparse solver testing: the rhs should not be modified!");
    VERIFY(xm.isApprox(refX1, test_precision<Scalar>()));

    // Test with a Map and non-unit stride.
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(2 * xm.rows(), 2 * xm.cols());
    out.setZero();
    // The outer stride spans DenseRhs's outer dimension, so it must be taken from outerStride()
    // rather than from rows(): the two differ when DenseRhs is row major.
    Eigen::Map<DenseRhs, 0, Stride<Eigen::Dynamic, 2>> outm(out.data(), xm.rows(), xm.cols(),
                                                            Stride<Eigen::Dynamic, 2>(2 * xm.outerStride(), 2));
    outm = solver.solve(bm);
    VERIFY(outm.isApprox(refX1, test_precision<Scalar>()));

    // Test a right-hand side with non-unit strides.
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> bstore(2 * db.rows(), 2 * db.cols());
    bstore.setZero();
    Eigen::Map<DenseRhs, 0, Stride<Eigen::Dynamic, 2>> bstrided(bstore.data(), db.rows(), db.cols(),
                                                                Stride<Eigen::Dynamic, 2>(2 * db.outerStride(), 2));
    bstrided = db;
    dx = solver.solve(bstrided);
    VERIFY(dx.isApprox(refX1, test_precision<Scalar>()));
  }

  // if not too large, do some extra check:
  if (A.rows() < 2000) {
    // test initialization ctor
    {
      Rhs x(b.rows(), b.cols());
      Solver solver2(A);
      VERIFY(solver2.info() == Success);
      x = solver2.solve(b);
      VERIFY(x.isApprox(refX1, test_precision<Scalar>()));
    }

    // test dense Block as the result and rhs:
    {
      DenseRhs x(refX1.rows(), refX1.cols());
      DenseRhs oldb(db);
      x.setZero();
      x.block(0, 0, x.rows(), x.cols()) = solver.solve(db.block(0, 0, db.rows(), db.cols()));
      VERIFY(oldb.isApprox(db, 0.0) && "sparse solver testing: the rhs should not be modified!");
      VERIFY(x.isApprox(refX1, test_precision<Scalar>()));
    }

    // test uncompressed inputs
    {
      Mat A2 = A;
      A2.reserve((ArrayXf::Random(A.outerSize()) + 2).template cast<typename Mat::StorageIndex>().eval());
      solver.compute(A2);
      Rhs x = solver.solve(b);
      VERIFY(x.isApprox(refX1, test_precision<Scalar>()));
    }

    // test expression as input
    {
      solver.compute(0.5 * (A + A));
      Rhs x = solver.solve(b);
      VERIFY(x.isApprox(refX1, test_precision<Scalar>()));

      Solver solver2(0.5 * (A + A));
      Rhs x2 = solver2.solve(b);
      VERIFY(x2.isApprox(refX1, test_precision<Scalar>()));
    }
  }
}

template <typename Solver, typename Rhs>
void check_sparse_solving_real_cases(Solver& solver, const typename Solver::MatrixType& A, const Rhs& b,
                                     const typename Solver::MatrixType& fullA, const Rhs& refX) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;
  using RealScalar = typename Mat::RealScalar;

  Rhs x(A.cols(), b.cols());

  solver.compute(A);
  if (solver.info() != Success) {
    std::cerr << "ERROR | sparse solver testing, factorization failed (" << typeid(Solver).name() << ")\n";
    VERIFY(solver.info() == Success);
  }
  x = solver.solve(b);

  if (solver.info() != Success) {
    std::cerr << "WARNING | sparse solver testing, solving failed (" << typeid(Solver).name() << ")\n";
    return;
  }

  RealScalar res_error = (fullA * x - b).norm() / b.norm();
  VERIFY((res_error <= test_precision<Scalar>()) && "sparse solver failed without noticing it");

  if (refX.size() != 0 && (refX - x).norm() / refX.norm() > test_precision<Scalar>()) {
    std::cerr << "WARNING | found solution is different from the provided reference one\n";
  }
}
template <typename Solver, typename DenseMat>
void check_sparse_determinant(Solver& solver, const typename Solver::MatrixType& A, const DenseMat& dA) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;

  solver.compute(A);
  if (solver.info() != Success) {
    std::cerr << "WARNING | sparse solver testing: factorization failed (check_sparse_determinant)\n";
    return;
  }

  // d det(A) = det(A) tr(A^{-1} dA). With the LU backward error |dA| <= gamma_n |L||U| (Higham, ASNA, Thm. 9.3) and
  // |L||U| ~ |A|, each determinant is off by about n eps cond |det(A)|, cond = sum_ij |A^{-1}_ji| |A_ij| >= n, which
  // is unbounded for the matrices check_sparse_square_determinant draws. The factor 8 is empirical: partial pivoting
  // stays below 0.5. The test_precision floor admits the growth of threshold pivoting, which |L||U| ~ |A| ignores;
  // UMFPACK's symmetric strategy takes diagonal pivots down to 1e-3 of the column maximum.
  const RealScalar cond = dA.inverse().transpose().cwiseAbs().cwiseProduct(dA.cwiseAbs()).sum();
  const RealScalar tolerance =
      numext::maxi(test_precision<Scalar>(), RealScalar(8 * dA.rows()) * NumTraits<RealScalar>::epsilon() * cond);
  VERIFY((numext::isfinite)(tolerance));
  const Scalar refDet = dA.determinant();
  VERIFY(numext::abs(solver.determinant() - refDet) <= tolerance * numext::abs(refDet));
}
template <typename Solver, typename DenseMat>
void check_sparse_abs_determinant(Solver& solver, const typename Solver::MatrixType& A, const DenseMat& dA) {
  using std::abs;
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;

  solver.compute(A);
  if (solver.info() != Success) {
    std::cerr << "WARNING | sparse solver testing: factorization failed (check_sparse_abs_determinant)\n";
    return;
  }

  Scalar refDet = abs(dA.determinant());
  VERIFY_IS_APPROX(refDet, solver.absDeterminant());
}

template <typename Solver, typename DenseMat>
void check_sparse_log_abs_determinant(Solver& solver, const typename Solver::MatrixType& A, const DenseMat& dA) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;

  solver.compute(A);
  if (solver.info() != Success) {
    std::cerr << "WARNING | sparse solver testing: factorization failed (check_sparse_log_abs_determinant)\n";
    return;
  }

  Scalar refDet = dA.determinant();
  RealScalar refAbsDet = numext::abs(refDet);
  VERIFY_IS_APPROX(refAbsDet, solver.absDeterminant());
  VERIFY_IS_APPROX(numext::sign(refDet), solver.signDeterminant());
  // log|det| crosses zero, so the meaningful bound here is absolute once |log|det|| drops below one.
  RealScalar refLogAbsDet = numext::log(refAbsDet);
  VERIFY_IS_MUCH_SMALLER_THAN(solver.logAbsDeterminant() - refLogAbsDet,
                              numext::maxi(RealScalar(1), numext::abs(refLogAbsDet)));
}

template <typename Solver, typename DenseMat>
int generate_sparse_spd_problem(Solver&, typename Solver::MatrixType& A, typename Solver::MatrixType& halfA,
                                DenseMat& dA, int maxSize = 300) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;

  int size = internal::random<int>(1, maxSize);
  double density = (std::max)(8. / static_cast<double>(size * size), 0.01);

  Mat M(size, size);
  DenseMatrix dM(size, size);

  initSparse<Scalar>(density, dM, M, ForceNonZeroDiag);

  A = M * M.adjoint();
  dA = dM * dM.adjoint();

  halfA.resize(size, size);
  if (Solver::UpLo == (Lower | Upper))
    halfA = A;
  else
    halfA.template selfadjointView<Solver::UpLo>().rankUpdate(M);

  // With FMA contraction, m * conj(m) leaves a rounding-level imaginary part on the diagonal of M M^H.
  // CHOLMOD's simplicial factorizations reject a non-real diagonal as not positive definite.
  A.diagonal() = A.diagonal().real().template cast<Scalar>();
  halfA.diagonal() = halfA.diagonal().real().template cast<Scalar>();
  dA.diagonal() = dA.diagonal().real().template cast<Scalar>();

  return size;
}

#ifdef TEST_REAL_CASES
template <typename Scalar>
inline std::string get_matrixfolder() {
  std::string mat_folder = TEST_REAL_CASES;
  if (std::is_same<Scalar, std::complex<float>>::value || std::is_same<Scalar, std::complex<double>>::value)
    mat_folder = mat_folder + static_cast<std::string>("/complex/");
  else
    mat_folder = mat_folder + static_cast<std::string>("/real/");
  return mat_folder;
}
std::string sym_to_string(int sym) {
  if (sym == Symmetric) return "Symmetric ";
  if (sym == SPD) return "SPD ";
  return "";
}
template <typename Derived>
std::string solver_stats(const IterativeSolverBase<Derived>& solver) {
  std::stringstream ss;
  ss << solver.iterations() << " iters, error: " << solver.error();
  return ss.str();
}
template <typename Derived>
std::string solver_stats(const SparseSolverBase<Derived>& /*solver*/) {
  return "";
}
#endif

template <typename Solver>
void check_sparse_spd_solving(Solver& solver, int maxSize = (std::min)(300, EIGEN_TEST_MAX_SIZE),
                              int maxRealWorldSize = 100000) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;
  using StorageIndex = typename Mat::StorageIndex;
  using SpMat = SparseMatrix<Scalar, ColMajor, StorageIndex>;
  using SpVec = SparseVector<Scalar, 0, StorageIndex>;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;

  // generate the problem
  Mat A, halfA;
  DenseMatrix dA;
  for (int i = 0; i < g_repeat; i++) {
    int size = generate_sparse_spd_problem(solver, A, halfA, dA, maxSize);

    // generate the right hand sides
    int rhsCols = internal::random<int>(1, 16);
    double density = (std::max)(8. / static_cast<double>(size * rhsCols), 0.1);
    SpMat B(size, rhsCols);
    DenseVector b = DenseVector::Random(size);
    DenseMatrix dB(size, rhsCols);
    initSparse<Scalar>(density, dB, B, ForceNonZeroDiag);
    SpVec c = B.col(0);
    DenseVector dc = dB.col(0);

    CALL_SUBTEST(check_sparse_solving(solver, A, b, dA, b));
    CALL_SUBTEST(check_sparse_solving(solver, halfA, b, dA, b));
    CALL_SUBTEST(check_sparse_solving(solver, A, dB, dA, dB));
    CALL_SUBTEST(check_sparse_solving(solver, halfA, dB, dA, dB));
    CALL_SUBTEST(check_sparse_solving(solver, A, B, dA, dB));
    CALL_SUBTEST(check_sparse_solving(solver, halfA, B, dA, dB));
    CALL_SUBTEST(check_sparse_solving(solver, A, c, dA, dc));
    CALL_SUBTEST(check_sparse_solving(solver, halfA, c, dA, dc));

    // check only once
    if (i == 0) {
      b = DenseVector::Zero(size);
      check_sparse_solving(solver, A, b, dA, b);
    }
  }

  // First, get the folder
#ifdef TEST_REAL_CASES
  // Test real problems with double precision only
  if (std::is_same<typename NumTraits<Scalar>::Real, double>::value) {
    std::string mat_folder = get_matrixfolder<Scalar>();
    MatrixMarketIterator<Scalar> it(mat_folder);
    for (; it; ++it) {
      if (it.sym() == SPD) {
        A = it.matrix();
        if (A.diagonal().size() <= maxRealWorldSize) {
          DenseVector b = it.rhs();
          DenseVector refX = it.refX();
          PermutationMatrix<Dynamic, Dynamic, StorageIndex> pnull;
          halfA.resize(A.rows(), A.cols());
          if (Solver::UpLo == (Lower | Upper))
            halfA = A;
          else
            halfA.template selfadjointView<Solver::UpLo>() = A.template triangularView<Eigen::Lower>().twistedBy(pnull);

          std::cout << "INFO | Testing " << sym_to_string(it.sym()) << "sparse problem " << it.matname() << " ("
                    << A.rows() << "x" << A.cols() << ") using " << typeid(Solver).name() << "..." << std::endl;
          CALL_SUBTEST(check_sparse_solving_real_cases(solver, A, b, A, refX));
          std::string stats = solver_stats(solver);
          if (stats.size() > 0) std::cout << "INFO |  " << stats << std::endl;
          CALL_SUBTEST(check_sparse_solving_real_cases(solver, halfA, b, A, refX));
        } else {
          std::cout << "INFO | Skip sparse problem \"" << it.matname() << "\" (too large)" << std::endl;
        }
      }
    }
  }
#else
  EIGEN_UNUSED_VARIABLE(maxRealWorldSize);
#endif
}

template <typename Solver>
void check_sparse_spd_determinant(Solver& solver) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;

  // generate the problem
  Mat A, halfA;
  DenseMatrix dA;
  generate_sparse_spd_problem(solver, A, halfA, dA, 30);

  check_sparse_determinant(solver, A, dA);
  check_sparse_determinant(solver, halfA, dA);
}

template <typename Solver>
void check_sparse_spd_log_abs_determinant(Solver& solver) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;

  // generate the problem
  Mat A, halfA;
  DenseMatrix dA;
  generate_sparse_spd_problem(solver, A, halfA, dA, 30);

  check_sparse_log_abs_determinant(solver, A, dA);
  check_sparse_log_abs_determinant(solver, halfA, dA);
}

template <typename Solver, typename DenseMat>
int generate_sparse_nonhermitian_problem(Solver&, typename Solver::MatrixType& A, typename Solver::MatrixType& halfA,
                                         DenseMat& dA, int maxSize = 300) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;

  int size = internal::random<int>(1, maxSize);
  double density = (std::max)(8. / static_cast<double>(size * size), 0.01);

  Mat M(size, size);
  DenseMatrix dM(size, size);

  initSparse<Scalar>(density, dM, M, ForceNonZeroDiag);

  A = M * M.transpose();
  dA = dM * dM.transpose();

  halfA.resize(size, size);
  if (Solver::UpLo == (Lower | Upper))
    halfA = A;
  else
    halfA = A.template triangularView<Solver::UpLo>();

  return size;
}

template <typename Solver>
void check_sparse_nonhermitian_solving(Solver& solver, int maxSize = (std::min)(300, EIGEN_TEST_MAX_SIZE),
                                       int maxRealWorldSize = 100000) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;
  using StorageIndex = typename Mat::StorageIndex;
  using SpMat = SparseMatrix<Scalar, ColMajor, StorageIndex>;
  using SpVec = SparseVector<Scalar, 0, StorageIndex>;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;

  // generate the problem
  Mat A, halfA;
  DenseMatrix dA;
  for (int i = 0; i < g_repeat; i++) {
    int size = generate_sparse_nonhermitian_problem(solver, A, halfA, dA, maxSize);

    // generate the right hand sides
    int rhsCols = internal::random<int>(1, 16);
    double density = (std::max)(8. / static_cast<double>(size * rhsCols), 0.1);
    SpMat B(size, rhsCols);
    DenseVector b = DenseVector::Random(size);
    DenseMatrix dB(size, rhsCols);
    initSparse<Scalar>(density, dB, B, ForceNonZeroDiag);
    SpVec c = B.col(0);
    DenseVector dc = dB.col(0);

    CALL_SUBTEST(check_sparse_solving(solver, A, b, dA, b));
    CALL_SUBTEST(check_sparse_solving(solver, halfA, b, dA, b));
    CALL_SUBTEST(check_sparse_solving(solver, A, dB, dA, dB));
    CALL_SUBTEST(check_sparse_solving(solver, halfA, dB, dA, dB));
    CALL_SUBTEST(check_sparse_solving(solver, A, B, dA, dB));
    CALL_SUBTEST(check_sparse_solving(solver, halfA, B, dA, dB));
    CALL_SUBTEST(check_sparse_solving(solver, A, c, dA, dc));
    CALL_SUBTEST(check_sparse_solving(solver, halfA, c, dA, dc));

    // check only once
    if (i == 0) {
      b = DenseVector::Zero(size);
      check_sparse_solving(solver, A, b, dA, b);
    }
  }

  EIGEN_UNUSED_VARIABLE(maxRealWorldSize);
}

template <typename Solver>
void check_sparse_nonhermitian_determinant(Solver& solver) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;

  for (int i = 0; i < g_repeat; i++) {
    // generate a new problem for each repetition
    Mat A, halfA;
    DenseMatrix dA;
    generate_sparse_nonhermitian_problem(solver, A, halfA, dA, 30);

    check_sparse_determinant(solver, A, dA);
    check_sparse_determinant(solver, halfA, dA);
  }
}

template <typename Solver>
void check_sparse_nonhermitian_log_abs_determinant(Solver& solver) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;

  for (int i = 0; i < g_repeat; i++) {
    // generate a new problem for each repetition
    Mat A, halfA;
    DenseMatrix dA;
    generate_sparse_nonhermitian_problem(solver, A, halfA, dA, 30);

    check_sparse_log_abs_determinant(solver, A, dA);
    check_sparse_log_abs_determinant(solver, halfA, dA);
  }
}

template <typename Solver>
void check_sparse_zero_matrix(Solver& solver) {
  using Mat = typename Solver::MatrixType;

  Mat A(1, 1);
  solver.compute(A);
  VERIFY_IS_EQUAL(solver.info(), NumericalIssue);
}

template <typename Solver, typename DenseMat>
Index generate_sparse_square_problem(Solver&, typename Solver::MatrixType& A, DenseMat& dA, int maxSize = 300,
                                     int options = ForceNonZeroDiag) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;

  Index size = internal::random<int>(1, maxSize);
  double density = (std::max)(8. / static_cast<double>(size * size), 0.01);

  A.resize(size, size);
  dA.resize(size, size);

  initSparse<Scalar>(density, dA, A, options);

  return size;
}

struct prune_column {
  Index m_col;
  prune_column(Index col) : m_col(col) {}
  template <class Scalar>
  bool operator()(Index, Index col, const Scalar&) const {
    return col != m_col;
  }
};

template <typename Solver>
void check_sparse_square_solving(Solver& solver, int maxSize = 300, int maxRealWorldSize = 100000,
                                 bool checkDeficient = false) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;
  using SpMat = SparseMatrix<Scalar, ColMajor, typename Mat::StorageIndex>;
  using SpVec = SparseVector<Scalar, 0, typename Mat::StorageIndex>;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;

  int rhsCols = internal::random<int>(1, 16);

  Mat A;
  DenseMatrix dA;
  for (int i = 0; i < g_repeat; i++) {
    Index size = generate_sparse_square_problem(solver, A, dA, maxSize);

    A.makeCompressed();
    DenseVector b = DenseVector::Random(size);
    DenseMatrix dB(size, rhsCols);
    SpMat B(size, rhsCols);
    double density = (std::max)(8. / double(size * rhsCols), 0.1);
    initSparse<Scalar>(density, dB, B, ForceNonZeroDiag);
    B.makeCompressed();
    SpVec c = B.col(0);
    DenseVector dc = dB.col(0);
    CALL_SUBTEST(check_sparse_solving(solver, A, b, dA, b));
    CALL_SUBTEST(check_sparse_solving(solver, A, dB, dA, dB));
    CALL_SUBTEST(check_sparse_solving(solver, A, B, dA, dB));
    CALL_SUBTEST(check_sparse_solving(solver, A, c, dA, dc));

    // check only once
    if (i == 0) {
      CALL_SUBTEST(b = DenseVector::Zero(size); check_sparse_solving(solver, A, b, dA, b));
    }
    // regression test for Bug 792 (structurally rank deficient matrices):
    if (checkDeficient && size > 1) {
      Index col = internal::random<int>(0, int(size - 1));
      A.prune(prune_column(col));
      solver.compute(A);
      VERIFY_IS_EQUAL(solver.info(), NumericalIssue);
    }
  }

  // First, get the folder
#ifdef TEST_REAL_CASES
  // Test real problems with double precision only
  if (std::is_same<typename NumTraits<Scalar>::Real, double>::value) {
    std::string mat_folder = get_matrixfolder<Scalar>();
    MatrixMarketIterator<Scalar> it(mat_folder);
    for (; it; ++it) {
      A = it.matrix();
      if (A.diagonal().size() <= maxRealWorldSize) {
        DenseVector b = it.rhs();
        DenseVector refX = it.refX();
        std::cout << "INFO | Testing " << sym_to_string(it.sym()) << "sparse problem " << it.matname() << " ("
                  << A.rows() << "x" << A.cols() << ") using " << typeid(Solver).name() << "..." << std::endl;
        CALL_SUBTEST(check_sparse_solving_real_cases(solver, A, b, A, refX));
        std::string stats = solver_stats(solver);
        if (stats.size() > 0) std::cout << "INFO |  " << stats << std::endl;
      } else {
        std::cout << "INFO | SKIP sparse problem \"" << it.matname() << "\" (too large)" << std::endl;
      }
    }
  }
#else
  EIGEN_UNUSED_VARIABLE(maxRealWorldSize);
#endif
}

template <typename Solver>
void check_sparse_square_determinant(Solver& solver) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;

  for (int i = 0; i < g_repeat; i++) {
    // generate the problem
    Mat A;
    DenseMatrix dA;

    int size = internal::random<int>(1, 30);
    dA.setRandom(size, size);

    dA = (dA.array().abs() < 0.3).select(0, dA);
    dA.diagonal() = (dA.diagonal().array() == 0).select(1, dA.diagonal());
    A = dA.sparseView();
    A.makeCompressed();

    check_sparse_determinant(solver, A, dA);
  }
}

template <typename Solver>
void check_sparse_square_abs_determinant(Solver& solver) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;

  for (int i = 0; i < g_repeat; i++) {
    // generate the problem
    Mat A;
    DenseMatrix dA;
    generate_sparse_square_problem(solver, A, dA, 30);
    A.makeCompressed();
    check_sparse_abs_determinant(solver, A, dA);
  }
}

template <typename Solver, typename DenseMat>
void generate_sparse_leastsquare_problem(Solver&, typename Solver::MatrixType& A, DenseMat& dA, int maxSize = 300,
                                         int options = ForceNonZeroDiag) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;

  int rows = internal::random<int>(1, maxSize);
  int cols = internal::random<int>(1, rows);
  double density = (std::max)(8. / (rows * cols), 0.01);

  A.resize(rows, cols);
  dA.resize(rows, cols);

  initSparse<Scalar>(density, dA, A, options);
}

template <typename Solver>
void check_sparse_leastsquare_solving(Solver& solver) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;
  using SpMat = SparseMatrix<Scalar, ColMajor, typename Mat::StorageIndex>;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;

  int rhsCols = internal::random<int>(1, 16);

  Mat A;
  DenseMatrix dA;
  for (int i = 0; i < g_repeat; i++) {
    generate_sparse_leastsquare_problem(solver, A, dA);

    A.makeCompressed();
    DenseVector b = DenseVector::Random(A.rows());
    DenseMatrix dB(A.rows(), rhsCols);
    SpMat B(A.rows(), rhsCols);
    double density = (std::max)(8. / (A.rows() * rhsCols), 0.1);
    initSparse<Scalar>(density, dB, B, ForceNonZeroDiag);
    B.makeCompressed();
    check_sparse_solving(solver, A, b, dA, b);
    check_sparse_solving(solver, A, dB, dA, dB);
    check_sparse_solving(solver, A, B, dA, dB);

    // check only once
    if (i == 0) {
      b = DenseVector::Zero(A.rows());
      check_sparse_solving(solver, A, b, dA, b);
    }
  }
}

// Solves systems with tolerance 0, which asks the solver to iterate until the residual vanishes or maxIterations() is
// reached. The matrices have an exhausted Krylov space after one to a few steps (identity, three distinct eigenvalues),
// or reach a residual below the range of squares well before the iteration limit (diagonal, tridiagonal, with an
// iteration limit well above their size). The solvers used to divide by the zero residual and return NaN, or aborted in
// an assertion.
template <typename Solver>
void check_sparse_zero_tolerance(Solver& solver) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;

  for (int kind = 0; kind < 4; ++kind) {
    const Index n = kind == 3 ? 8 : 12;
    std::vector<Triplet<Scalar>> triplets;
    for (Index i = 0; i < n; ++i) {
      Scalar d(1);
      if (kind == 1) d = Scalar(1 + i % 3);
      if (kind == 2) d = Scalar(1 + i);
      if (kind == 3) d = Scalar(4 + i);
      triplets.emplace_back(i, i, d);
      if (kind == 3 && i + 1 < n) {
        triplets.emplace_back(i, i + 1, Scalar(1));
        triplets.emplace_back(i + 1, i, Scalar(1));
      }
    }
    Mat A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    const DenseVector xt = DenseVector::LinSpaced(n, Scalar(1), Scalar(2));
    const DenseVector b = A * xt;
    // Every system is SPD with condition number below 12, so a residual of 100 n eps bounds the rounding error.
    const RealScalar bound = RealScalar(100) * RealScalar(n) * NumTraits<RealScalar>::epsilon() * b.norm();

    for (Index maxit : {Index(-1), Index(300)}) {
      solver.setTolerance(0);
      solver.setMaxIterations(maxit);
      solver.compute(A);
      const DenseVector x = solver.solve(b);
      VERIFY(x.allFinite());
      VERIFY((numext::isfinite)(solver.error()));
      // The tolerance is met only by a residual of exactly 0, so the solver reports Success then and NoConvergence
      // otherwise; a breakdown at the exact solution is not a numerical issue.
      VERIFY(solver.info() == Success || solver.info() == NoConvergence);
      VERIFY_LE(solver.iterations(), solver.maxIterations());
      VERIFY_LE((b - A * x).norm(), bound);
    }
  }
}

// Solves the 2D Poisson problem on a 30 x 30 grid, an SPD matrix on which every iterative solver converges within about
// 700 iterations, with a solver constructed with its defaults. The solver must report Success after fewer than
// maxIterations() iterations, with a true residual within 100 n eps, so that a default that cannot be reached, which
// makes a solver run to maxIterations() with NoConvergence, does not go unnoticed.
template <typename Solver>
void check_sparse_default_tolerance(const Solver&) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;

  const Index m = 30, n = m * m;
  std::vector<Triplet<Scalar>> triplets;
  for (Index i = 0; i < m; ++i) {
    for (Index j = 0; j < m; ++j) {
      const Index k = i * m + j;
      triplets.emplace_back(k, k, Scalar(4));
      if (j + 1 < m) triplets.emplace_back(k, k + 1, Scalar(-1));
      if (j > 0) triplets.emplace_back(k, k - 1, Scalar(-1));
      if (i + 1 < m) triplets.emplace_back(k, k + m, Scalar(-1));
      if (i > 0) triplets.emplace_back(k, k - m, Scalar(-1));
    }
  }
  Mat A(n, n);
  A.setFromTriplets(triplets.begin(), triplets.end());
  const DenseVector b = DenseVector::Random(n);

  Solver solver(A);
  const DenseVector x = solver.solve(b);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY(solver.iterations() < solver.maxIterations());
  VERIFY_LE((b - A * x).norm(), RealScalar(100 * n) * NumTraits<RealScalar>::epsilon() * b.norm());
}

// The matrix-free operator of doc/examples/matrixfree_cg.cpp without its product: enough to attach it to a solver and
// query the solver.
class MatrixFreeOperator;
namespace Eigen {
namespace internal {
template <>
struct traits<MatrixFreeOperator> : public traits<SparseMatrix<double>> {};
}  // namespace internal
}  // namespace Eigen

class MatrixFreeOperator : public EigenBase<MatrixFreeOperator> {
 public:
  using Scalar = double;
  using RealScalar = double;
  using StorageIndex = int;
  static constexpr int ColsAtCompileTime = Dynamic;
  static constexpr int MaxColsAtCompileTime = Dynamic;
  static constexpr bool IsRowMajor = false;

  explicit MatrixFreeOperator(const SparseMatrix<double>& mat) : mp_mat(&mat) {}
  Index rows() const { return mp_mat->rows(); }
  Index cols() const { return mp_mat->cols(); }

 private:
  const SparseMatrix<double>* mp_mat;
};

// maxIterations() depends on n and used to dereference the null operator pointer of a matrix-free Solver before
// compute(), and so did the tolerance() of IDRS and MINRES that depended on n. With n = 0 until compute():
// maxIterations() == 2 n, and eps <= tolerance() <= 8 sqrt(max(n, 1)) eps.
template <typename Solver>
void check_matrix_free_accessors() {
  using RealScalar = typename Solver::RealScalar;
  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  const Index n = 100;
  SparseMatrix<double> S(n, n);
  S.setIdentity();
  const MatrixFreeOperator A(S);

  Solver detached;
  VERIFY_IS_EQUAL(detached.maxIterations(), Index(0));
  VERIFY_LE(eps, detached.tolerance());
  VERIFY_LE(detached.tolerance(), 8 * eps);
  detached.setMaxIterations(7).setTolerance(RealScalar(1e-3));
  VERIFY_IS_EQUAL(detached.maxIterations(), Index(7));
  VERIFY_IS_EQUAL(detached.tolerance(), RealScalar(1e-3));

  Solver attached;
  attached.compute(A);
  VERIFY_IS_EQUAL(attached.maxIterations(), 2 * n);
  VERIFY_LE(eps, attached.tolerance());
  VERIFY_LE(attached.tolerance(), 8 * numext::sqrt(RealScalar(n)) * eps);
}

// Solves A x = b for b = s A xt with A = tridiag(-1, 4, -1), n = 40, and |b| so small that tol |b| is below sqrt(min),
// though |b| is a normal number: a stopping threshold raised to an absolute floor, to keep a tolerance of 0 from
// dividing by zero, would then exceed tol |b|, and the solvers returned NoConvergence or an inaccurate x without
// reaching the tolerance (|b| < 1e-12 in float at the default tolerance). Each solver must reach Success with a forward
// error of at most 1000 cond^2 tol (cond(A) <= 3), at its default tolerance and at a loose one.
template <typename Solver>
void check_sparse_small_rhs(const Solver&) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;

  const Index n = 40;
  std::vector<Triplet<Scalar>> triplets;
  for (Index i = 0; i < n; ++i) {
    triplets.emplace_back(i, i, Scalar(4));
    if (i + 1 < n) {
      triplets.emplace_back(i, i + 1, Scalar(-1));
      triplets.emplace_back(i + 1, i, Scalar(-1));
    }
  }
  Mat A(n, n);
  A.setFromTriplets(triplets.begin(), triplets.end());
  const DenseVector xt = DenseVector::Random(n);
  const DenseVector b0 = A * xt;

  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  const RealScalar sqrtMin = numext::sqrt((std::numeric_limits<RealScalar>::min)());
  const RealScalar looseTol = RealScalar(1e-2) * numext::sqrt(eps);
  // tol |b| < sqrt(min) at the default tolerance for s1, and at the loose one for s1 and s2.
  const RealScalar s1 = sqrtMin / eps / RealScalar(1000), s2 = s1 / RealScalar(30);
  for (int c = 0; c < 3; ++c) {
    const RealScalar s = c == 2 ? s2 : s1;
    const DenseVector b = s * b0;
    Solver solver(A);
    if (c > 0) solver.setTolerance(looseTol);
    const DenseVector x = solver.solve(b);
    VERIFY(x.allFinite());
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY_LE((x / s - xt).stableNorm(), RealScalar(9000) * solver.tolerance() * xt.stableNorm());
  }
}

// Solves A x = b for b = s A xt with |b| of order 10 sqrt(min) (1e-18 in float), for A = tridiag(-1, 4, -1), n = 400
// and tridiag(-1, 2, -1), n = 200, which are well and moderately conditioned. The residual squares of a Krylov method
// are of the order of |b|^2 eps^2 there, far below the smallest normal number: CG and BiCGSTAB returned NaN, LSCG
// NoConvergence and IDRS NumericalIssue (or NaN) in float before the problem was solved for b / s, or the residual
// scaled, as the small-rhs tests do on ARMv7, where NEON flushes subnormals. When the solver succeeds for s = 1, it
// must succeed with a finite x and a forward error within a factor 30 of that of the s = 1 solve (the scaled problem
// rounds differently, and the errors of the second matrix are of order cond eps).
template <typename Solver>
void check_sparse_tiny_rhs(const Solver&) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;

  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  const RealScalar tiny = RealScalar(10) * numext::sqrt((std::numeric_limits<RealScalar>::min)());
  for (int c = 0; c < 2; ++c) {
    const Index n = c == 0 ? 400 : 200;
    const int diagonal = c == 0 ? 4 : 2;
    std::vector<Triplet<Scalar>> triplets;
    for (Index i = 0; i < n; ++i) {
      triplets.emplace_back(i, i, Scalar(diagonal));
      if (i + 1 < n) {
        triplets.emplace_back(i, i + 1, Scalar(-1));
        triplets.emplace_back(i + 1, i, Scalar(-1));
      }
    }
    Mat A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    const DenseVector xt = DenseVector::Random(n);
    const DenseVector b0 = A * xt;

    Solver reference(A);
    const DenseVector xr = reference.solve(b0);
    if (reference.info() != Success) continue;  // LSCG squares the condition number of the second matrix
    const RealScalar errorReference = (xr - xt).stableNorm() / xt.stableNorm();

    Solver solver(A);
    const DenseVector x = solver.solve(tiny * b0);
    VERIFY(x.allFinite());
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY_LE((x / tiny - xt).stableNorm() / xt.stableNorm(), RealScalar(30) * numext::maxi(errorReference, eps));
  }
}

#endif  // EIGEN_TEST_SPARSE_SOLVER_H
