// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Test resizing restrictions and assignment to empty matrices and arrays.

#define EIGEN_NO_AUTOMATIC_RESIZING
#include "main.h"
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Sparse>

template <typename Scalar>
void testNoAutomaticResizing() {
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using RowVector = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
  using Array = Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using ArrayVector = Eigen::Array<Scalar, Eigen::Dynamic, 1>;

  const Index rows = internal::random<Index>(1, 50);
  const Index cols = internal::random<Index>(1, 50);

  // Assignment of Zero expression to default-constructed matrix.
  {
    Matrix M;
    M = Matrix::Zero(rows, cols);
    VERIFY_IS_EQUAL(M.rows(), rows);
    VERIFY_IS_EQUAL(M.cols(), cols);
    VERIFY_IS_EQUAL(M.norm(), Scalar(0));
  }

  // Assignment of Zero expression to default-constructed array.
  {
    Array A;
    A = Array::Zero(rows, cols);
    VERIFY_IS_EQUAL(A.rows(), rows);
    VERIFY_IS_EQUAL(A.cols(), cols);
  }

  // Assignment of Ones expression to default-constructed matrix.
  {
    Matrix M;
    M = Matrix::Ones(rows, cols);
    VERIFY_IS_EQUAL(M.rows(), rows);
    VERIFY_IS_EQUAL(M.cols(), cols);
  }

  // Assignment of Random expression to default-constructed matrix.
  {
    Matrix M;
    M = Matrix::Random(rows, cols);
    VERIFY_IS_EQUAL(M.rows(), rows);
    VERIFY_IS_EQUAL(M.cols(), cols);
  }

  // Assignment from another matrix to default-constructed matrix.
  {
    Matrix src = Matrix::Random(rows, cols);
    Matrix dst;
    dst = src;
    VERIFY_IS_EQUAL(dst.rows(), rows);
    VERIFY_IS_EQUAL(dst.cols(), cols);
    VERIFY_IS_APPROX(dst, src);
  }

  // Vector assignment to default-constructed vector.
  {
    Vector v;
    v = Vector::Zero(rows);
    VERIFY_IS_EQUAL(v.size(), rows);
  }

  // RowVector assignment to default-constructed row vector.
  {
    RowVector v;
    v = RowVector::Zero(cols);
    VERIFY_IS_EQUAL(v.size(), cols);
  }

  // Array vector assignment to default-constructed array vector.
  {
    ArrayVector v;
    v = ArrayVector::Zero(rows);
    VERIFY_IS_EQUAL(v.size(), rows);
  }

  // Column access after Zero initialization (reproducer for reported bug).
  {
    Array A;
    A = Array::Zero(rows, cols);
    for (Index j = 0; j < cols; ++j) {
      auto c = A.col(j);
      VERIFY_IS_EQUAL(c.rows(), rows);
    }
  }
}

// Isolate this check from the additional shape assertions in assignment evaluators.
template <typename PlainObject>
struct ResizeToMatchProbe : PlainObject {
  using PlainObject::_resize_to_match;
};

template <typename Vector>
void testResizeToMatchVector() {
  ResizeToMatchProbe<Vector> dst;
  MatrixXd source = MatrixXd::Ones(2, 3);
  VERIFY_RAISES_ASSERT(dst._resize_to_match(source));
  dst.resize(6);
  dst.setConstant(-1);
  VERIFY_RAISES_ASSERT(dst._resize_to_match(source));
  VERIFY_IS_EQUAL(dst.sum(), -6);

  // A non-empty vector accepts a same-length source of either runtime orientation and keeps its shape.
  source.resize(6, 1);
  dst._resize_to_match(source);
  source.resize(1, 6);
  dst._resize_to_match(source);
  VERIFY_IS_EQUAL(dst.rows(), Vector::RowsAtCompileTime == 1 ? 1 : 6);
  VERIFY_IS_EQUAL(dst.cols(), Vector::RowsAtCompileTime == 1 ? 6 : 1);
  source.resize(1, 5);
  VERIFY_RAISES_ASSERT(dst._resize_to_match(source));

  ResizeToMatchProbe<Vector> empty;
  empty._resize_to_match(source);
  VERIFY_IS_EQUAL(empty.size(), 5);
  VERIFY_IS_EQUAL(empty.rows(), Vector::RowsAtCompileTime == 1 ? 1 : 5);
  VERIFY_IS_EQUAL(empty.cols(), Vector::RowsAtCompileTime == 1 ? 5 : 1);
}

template <int Order>
void testVectorOrientationNoAutomaticResizing() {
  using Matrix = Eigen::Matrix<double, Dynamic, Dynamic, Order>;
  using Array = Eigen::Array<double, Dynamic, Dynamic, Order>;
  const Matrix column = Matrix::Constant(3, 1, 4.0);
  const Matrix row = Matrix::Constant(1, 3, 5.0);
  RowVectorXd r = RowVectorXd::Constant(3, -1.0);
  VectorXd c = VectorXd::Constant(3, -1.0);
  VERIFY_RAISES_ASSERT(r = column);
  VERIFY_RAISES_ASSERT(c = row);
  VERIFY_IS_EQUAL(r.sum(), -3.0);
  VERIFY_IS_EQUAL(c.sum(), -3.0);

  Eigen::Array<double, 1, Dynamic> ar = Eigen::Array<double, 1, Dynamic>::Constant(3, -1.0);
  ArrayXd ac = ArrayXd::Constant(3, -1.0);
  const Array arrayColumn = Array::Constant(3, 1, 4.0);
  const Array arrayRow = Array::Constant(1, 3, 5.0);
  VERIFY_RAISES_ASSERT(ar = arrayColumn);
  VERIFY_RAISES_ASSERT(ac = arrayRow);

  const Matrix identity = Matrix::Identity(3, 3);
  VERIFY_RAISES_ASSERT(r.noalias() = identity * column);
  VERIFY_RAISES_ASSERT(c.noalias() = row * identity);
  VERIFY_RAISES_ASSERT(r.noalias() = 2.0 * (identity * column));
  VERIFY_RAISES_ASSERT(c.noalias() = 2.0 * (row * identity));

  // Compile-time vector orientation conversion remains supported.
  c << 1, 2, 3;
  r = c;
  VERIFY_IS_EQUAL(r, RowVector3d(1, 2, 3));
  c = RowVector3d(4, 5, 6);
  VERIFY_IS_EQUAL(c, Vector3d(4, 5, 6));
  ac << 1, 2, 3;
  ar = ac;
  VERIFY_IS_EQUAL(ar.matrix(), RowVector3d(1, 2, 3));
  ac = ar;
  VERIFY_IS_EQUAL(ac.matrix(), Vector3d(1, 2, 3));
  r = row;
  c = column;
  VERIFY_IS_EQUAL(r.sum(), 15.0);
  VERIFY_IS_EQUAL(c.sum(), 12.0);
}

template <int Order>
void testSparseNoAutomaticResizing() {
  SparseMatrix<double, Order> row(1, 3), column(3, 1);
  row.insert(0, 2) = 7.0;
  column.insert(2, 0) = 8.0;
  VectorXd c = VectorXd::Constant(3, -1.0);
  RowVectorXd r = RowVectorXd::Constant(3, -1.0);
  VERIFY_RAISES_ASSERT(c = row);
  VERIFY_RAISES_ASSERT(r = column);
  c = column;
  r = row;
  VERIFY_IS_EQUAL(c, Vector3d(0, 0, 8));
  VERIFY_IS_EQUAL(r, RowVector3d(0, 0, 7));
}

// A nonsymmetric, diagonally dominant sparse matrix (row- or column-major), well conditioned for every solver below.
template <typename SparseMatrixType>
SparseMatrixType makeSolverMatrix(Index n) {
  using Scalar = typename SparseMatrixType::Scalar;
  std::vector<Triplet<Scalar>> triplets;
  for (Index i = 0; i < n; ++i) {
    triplets.emplace_back(i, i, Scalar(4));
    if (i > 0) triplets.emplace_back(i, i - 1, Scalar(-1));
    if (i + 1 < n) triplets.emplace_back(i, i + 1, Scalar(-1.5));
    if (i + 5 < n) triplets.emplace_back(i, i + 5, Scalar(0.25));
    if (i >= 7) triplets.emplace_back(i, i - 7, Scalar(-0.5));
  }
  SparseMatrixType A(n, n);
  A.setFromTriplets(triplets.begin(), triplets.end());
  return A;
}

// Whether every column x of X satisfies the normwise backward error bound of A x = b (Rigal-Gaches):
// |A x - b| / (|A|_F |x| + |b|) <= tol. A backward stable direct solver achieves c n eps; an iterative solver with
// relative residual tolerance t stops with |A x - b| <= t |b|, hence at most t. A NaN fails the comparison.
template <typename MatrixType, typename Rhs>
bool hasSmallBackwardError(const MatrixType& A, const MatrixXd& X, const Rhs& B, double tol) {
  const double normA = A.norm();
  for (Index j = 0; j < B.cols(); ++j) {
    const double eta = (A * X.col(j) - B.col(j)).norm() / (normA * X.col(j).norm() + B.col(j).norm());
    if (!(eta <= tol)) return false;
  }
  return true;
}

// The solution is assigned to default-constructed destinations, which stay legal under the macro; anything the solver
// assigns to a non-empty member of the wrong size would assert. The default bound n eps is the backward stability of
// a direct solver (the largest error measured over seeds 1..20 on these matrices is 0.017 n eps); iterative solvers
// pass a multiple of their stopping tolerance.
template <typename Solver, typename MatrixType>
void checkSparseSolve(Solver& solver, const MatrixType& A, double tol = -1) {
  const Index n = A.rows();
  if (tol < 0) tol = double(n) * NumTraits<double>::epsilon();
  const VectorXd b = VectorXd::Random(n);
  const MatrixXd B = MatrixXd::Random(n, 3);
  solver.compute(A);
  VERIFY_IS_EQUAL(int(solver.info()), int(Success));
  VectorXd x;
  x = solver.solve(b);
  VERIFY(hasSmallBackwardError(A, x, b, tol));
  MatrixXd X;
  X = solver.solve(B);
  VERIFY(hasSmallBackwardError(A, X, B, tol));
  // Refactorize the same pattern.
  solver.analyzePattern(A);
  solver.factorize(A);
  VERIFY_IS_EQUAL(int(solver.info()), int(Success));
  x = solver.solve(b);
  VERIFY(hasSmallBackwardError(A, x, b, tol));
}

template <int Order>
void testSparseLUNoAutomaticResizing() {
  using SM = SparseMatrix<double, Order>;
  const SM A = makeSolverMatrix<SM>(internal::random<Index>(12, 60));
  for (int symmetric = 0; symmetric < 2; ++symmetric) {
    {
      SparseLU<SM> lu;
      lu.isSymmetric(symmetric != 0);
      checkSparseSolve(lu, A);
    }
    {
      SparseLU<SM, AMDOrdering<int>> lu;
      lu.isSymmetric(symmetric != 0);
      checkSparseSolve(lu, A);
    }
    {
      SparseLU<SM, NaturalOrdering<int>> lu;
      lu.isSymmetric(symmetric != 0);
      checkSparseSolve(lu, A);
    }
  }
}

template <int Order>
void testSparseQRNoAutomaticResizing() {
  using SM = SparseMatrix<double, Order>;
  SM A = makeSolverMatrix<SM>(internal::random<Index>(12, 60));
  A.makeCompressed();
  SparseQR<SM, COLAMDOrdering<int>> qr;
  checkSparseSolve(qr, A);
  SparseQR<SM, NaturalOrdering<int>> qrNatural;
  checkSparseSolve(qrNatural, A);
}

// A symmetric positive definite matrix built from the nonsymmetric one.
template <int Order>
SparseMatrix<double, Order> makeSpdMatrix(Index n) {
  using SM = SparseMatrix<double, Order>;
  const SM A = makeSolverMatrix<SM>(n);
  SM identity(n, n);
  identity.setIdentity();
  SM S = A.transpose() * A;
  S += identity;
  S.makeCompressed();
  return S;
}

template <int Order>
void testSimplicialNoAutomaticResizing() {
  using SM = SparseMatrix<double, Order>;
  const SM S = makeSpdMatrix<Order>(internal::random<Index>(12, 60));
  {
    SimplicialLLT<SM> llt;
    checkSparseSolve(llt, S);
  }
  {
    SimplicialLDLT<SM> ldlt;
    checkSparseSolve(ldlt, S);
  }
  {
    SimplicialLLT<SM, Upper, NaturalOrdering<int>> llt;
    checkSparseSolve(llt, S);
  }
  {
    SimplicialLDLT<SM, Upper, NaturalOrdering<int>> ldlt;
    checkSparseSolve(ldlt, S);
  }
  {
    SimplicialCholesky<SM> chol;
    checkSparseSolve(chol, S);
  }
}

template <int Order>
void testIterativeNoAutomaticResizing() {
  using SM = SparseMatrix<double, Order>;
  const Index n = internal::random<Index>(12, 60);
  const SM A = makeSolverMatrix<SM>(n);
  const SM S = makeSpdMatrix<Order>(n);
  const double solverTol = 1e-12;
  const double tol = 10 * solverTol;  // the largest error measured over seeds 1..20 is 0.9 solverTol

  {
    ConjugateGradient<SM, Lower | Upper> cg;
    cg.setTolerance(solverTol);
    checkSparseSolve(cg, S, tol);
  }
  {
    ConjugateGradient<SM, Lower, IncompleteCholesky<double, Lower, AMDOrdering<int>>> cg;
    cg.setTolerance(solverTol);
    checkSparseSolve(cg, S, tol);
  }
  {
    ConjugateGradient<SM, Lower | Upper, IdentityPreconditioner> cg;
    cg.setTolerance(solverTol);
    checkSparseSolve(cg, S, tol);
  }
  {
    MINRES<SM, Lower | Upper, DiagonalPreconditioner<double>> minres;
    minres.setTolerance(solverTol);
    checkSparseSolve(minres, S, tol);
  }
  {
    LeastSquaresConjugateGradient<SM, LeastSquareDiagonalPreconditioner<double>> lscg;
    lscg.setTolerance(solverTol);
    checkSparseSolve(lscg, A, tol);
  }
  {
    BiCGSTAB<SM> bicgstab;
    bicgstab.setTolerance(solverTol);
    checkSparseSolve(bicgstab, A, tol);
  }
  {
    BiCGSTAB<SM, IncompleteLUT<double>> bicgstab;
    bicgstab.setTolerance(solverTol);
    checkSparseSolve(bicgstab, A, tol);
  }
  {
    GMRES<SM, IncompleteLUT<double>> gmres;
    gmres.setTolerance(solverTol);
    checkSparseSolve(gmres, A, tol);
  }
  {
    DGMRES<SM, DiagonalPreconditioner<double>> dgmres;
    dgmres.setTolerance(solverTol);
    checkSparseSolve(dgmres, A, tol);
  }
  {
    BiCGSTABL<SM, DiagonalPreconditioner<double>> bicgstabl;
    bicgstabl.setTolerance(solverTol);
    checkSparseSolve(bicgstabl, A, tol);
  }
  {
    LSMR<SM> lsmr;
    lsmr.setTolerance(solverTol);
    checkSparseSolve(lsmr, A, tol);
  }
  {
    // solveWithGuess into an empty destination.
    BiCGSTAB<SM> bicgstab;
    bicgstab.setTolerance(solverTol);
    bicgstab.compute(A);
    const VectorXd b = VectorXd::Random(n);
    VectorXd x;
    x = bicgstab.solveWithGuess(b, VectorXd::Zero(n));
    VERIFY(hasSmallBackwardError(A, x, b, tol));
  }
}

// Sparse products, assignments, views and triangular solves into empty and correctly sized destinations.
template <int Order, int OtherOrder>
void testSparseOperationsNoAutomaticResizing() {
  using SM = SparseMatrix<double, Order>;
  using OtherSM = SparseMatrix<double, OtherOrder>;
  const Index n = internal::random<Index>(4, 40);
  const SM A = makeSolverMatrix<SM>(n);
  const OtherSM B = makeSolverMatrix<OtherSM>(n).transpose();
  const MatrixXd Ad = MatrixXd(A), Bd = MatrixXd(B);
  const VectorXd v = VectorXd::Random(n);
  const MatrixXd M = MatrixXd::Random(n, 3);

  for (int presized = 0; presized < 2; ++presized) {
    VectorXd x;
    MatrixXd X;
    if (presized) {
      x.resize(n);
      X.resize(n, 3);
    }
    x = A * v;
    VERIFY_IS_APPROX(x, Ad * v);
    x = A.transpose() * v;
    VERIFY_IS_APPROX(x, Ad.transpose() * v);
    X = A * M;
    VERIFY_IS_APPROX(X, Ad * M);
    X = A.transpose() * M;
    VERIFY_IS_APPROX(X, Ad.transpose() * M);
    MatrixXd Y;
    if (presized) Y.resize(3, n);
    Y = M.transpose() * A;
    VERIFY_IS_APPROX(Y, M.transpose() * Ad);
    x = A.template selfadjointView<Lower>() * v;
    VERIFY_IS_APPROX(x, MatrixXd(Ad.template selfadjointView<Lower>()) * v);
    MatrixXd D;
    if (presized) D.resize(n, n);
    D = A;
    VERIFY_IS_APPROX(D, Ad);
    D = A * B;
    VERIFY_IS_APPROX(D, Ad * Bd);
    D = A + SM(B);
    VERIFY_IS_APPROX(D, Ad + Bd);
  }

  SM C;
  C = A * B;
  VERIFY_IS_APPROX(MatrixXd(C), Ad * Bd);
  C = SM(A * B) + A;
  VERIFY_IS_APPROX(MatrixXd(C), Ad * Bd + Ad);
  C = A + SM(B);
  C += A;
  C -= SM(B);
  C = 2.0 * C;
  VERIFY_IS_APPROX(MatrixXd(C), 2 * (Ad + Bd + Ad - Bd));
  C = A.transpose();
  VERIFY_IS_APPROX(MatrixXd(C), Ad.transpose());
  C = (A * B).pruned();
  VERIFY_IS_APPROX(MatrixXd(C), Ad * Bd);
  OtherSM O = A;
  VERIFY_IS_APPROX(MatrixXd(O), Ad);
  C = MatrixXd(Ad).sparseView();
  VERIFY_IS_APPROX(MatrixXd(C), Ad);
  PermutationMatrix<Dynamic> P(n);
  P.setIdentity();
  std::swap(P.indices()(0), P.indices()(n - 1));
  C = P * A * P.transpose();
  VERIFY_IS_APPROX(MatrixXd(C), MatrixXd(P * Ad * P.transpose()));
  C = A.twistedBy(P);
  VERIFY_IS_APPROX(MatrixXd(C), MatrixXd(P * Ad * P.transpose()));
  C = A.block(1, 1, n - 2, n - 2) * SM(n - 2, n - 2);
  VERIFY_IS_EQUAL(C.nonZeros(), 0);

  SparseVector<double> sv(n);
  sv.insert(0) = 1.0;
  sv.insert(n - 1) = 2.0;
  SparseVector<double> sw;
  sw = A * sv;
  VERIFY_IS_APPROX(VectorXd(sw), Ad * VectorXd(sv));

  // Triangular solves with dense and sparse right-hand sides.
  const SM L = A.template triangularView<Lower>();
  const MatrixXd Ld = MatrixXd(L);
  VectorXd z;
  z = L.template triangularView<Lower>().solve(v);
  VERIFY_IS_APPROX(Ld * z, v);
  MatrixXd Z;
  Z = L.template triangularView<Lower>().solve(M);
  VERIFY_IS_APPROX(Ld * Z, M);
  SparseMatrix<double> rhs(n, 2);
  rhs.insert(0, 0) = 1.0;
  rhs.insert(n - 1, 1) = 1.0;
  SparseMatrix<double> sol = rhs;
  SparseMatrix<double> Lc = L;
  Lc.template triangularView<Lower>().solveInPlace(sol);
  VERIFY_IS_APPROX(MatrixXd(Lc.template triangularView<Lower>()) * MatrixXd(sol), MatrixXd(rhs));
}

template <typename = void>
void testProductNoAutomaticResizing() {
  const MatrixXd lhs = MatrixXd::Ones(3, 2);
  const MatrixXd rhs = MatrixXd::Constant(2, 4, 2.0);
  const MatrixXd expected = MatrixXd::Constant(3, 4, 4.0);

  MatrixXd dst;
  dst.noalias() = lhs * rhs;
  VERIFY_IS_APPROX(dst, expected);
  dst.noalias() = 2.0 * (lhs * rhs);
  VERIFY_IS_APPROX(dst, 2.0 * expected);
  dst.noalias() = lhs * rhs;
  VERIFY_IS_APPROX(dst, expected);

  dst.resize(2, 6);  // Same coefficient count does not imply compatible dimensions.
  dst.setOnes();
  VERIFY_RAISES_ASSERT(dst.noalias() = lhs * rhs);
  VERIFY_RAISES_ASSERT(dst.noalias() = 2.0 * (lhs * rhs));
  VERIFY_IS_EQUAL(dst.rows(), 2);
  VERIFY_IS_EQUAL(dst.cols(), 6);
  VERIFY_IS_EQUAL(dst.sum(), 12.0);

  RowVectorXd row;
  row.noalias() = lhs * Vector2d::Ones();
  VERIFY_IS_APPROX(row, RowVector3d::Constant(2.0));
  row.resize(2);
  VERIFY_RAISES_ASSERT(row.noalias() = lhs * Vector2d::Ones());
}

template <typename = void>
void testSolveNoAutomaticResizing() {
  const Matrix4d A = Matrix4d::Random() + 4 * Matrix4d::Identity();
  const PartialPivLU<MatrixXd> lu(A);
  const VectorXd b = VectorXd::Random(4);

  VectorXd x;
  x = lu.solve(b);
  VERIFY_IS_APPROX(A * x, b);
  x = lu.transpose().solve(b);
  VERIFY_IS_APPROX(A.transpose() * x, b);
  x = lu.adjoint().solve(b);
  VERIFY_IS_APPROX(A.adjoint() * x, b);

  x.setConstant(3, -1.0);
  VERIFY_RAISES_ASSERT(x = lu.solve(b));
  VERIFY_RAISES_ASSERT(x = lu.transpose().solve(b));
  VERIFY_RAISES_ASSERT(x = lu.adjoint().solve(b));
  VERIFY_IS_EQUAL(x.size(), 3);
}

EIGEN_DECLARE_TEST(no_automatic_resizing) {
  CALL_SUBTEST_1(testNoAutomaticResizing<float>());
  CALL_SUBTEST_2(testNoAutomaticResizing<double>());
  CALL_SUBTEST_2(testProductNoAutomaticResizing<>());
  CALL_SUBTEST_2(testSolveNoAutomaticResizing<>());
  CALL_SUBTEST_2(testVectorOrientationNoAutomaticResizing<ColMajor>());
  CALL_SUBTEST_2(testVectorOrientationNoAutomaticResizing<RowMajor>());
  CALL_SUBTEST_2(testResizeToMatchVector<VectorXd>());
  CALL_SUBTEST_2(testResizeToMatchVector<RowVectorXd>());
  CALL_SUBTEST_2(testResizeToMatchVector<ArrayXd>());
  CALL_SUBTEST_2((testResizeToMatchVector<Eigen::Array<double, 1, Dynamic>>()));
  CALL_SUBTEST_3(testNoAutomaticResizing<std::complex<double>>());
  CALL_SUBTEST_4(testSparseNoAutomaticResizing<ColMajor>());
  CALL_SUBTEST_4(testSparseNoAutomaticResizing<RowMajor>());
  CALL_SUBTEST_5(testSparseLUNoAutomaticResizing<ColMajor>());
  CALL_SUBTEST_5(testSparseLUNoAutomaticResizing<RowMajor>());
  CALL_SUBTEST_6(testSparseQRNoAutomaticResizing<ColMajor>());
  CALL_SUBTEST_7(testSimplicialNoAutomaticResizing<ColMajor>());
  CALL_SUBTEST_7(testSimplicialNoAutomaticResizing<RowMajor>());
  CALL_SUBTEST_8(testIterativeNoAutomaticResizing<ColMajor>());
  CALL_SUBTEST_8(testIterativeNoAutomaticResizing<RowMajor>());
  CALL_SUBTEST_9((testSparseOperationsNoAutomaticResizing<ColMajor, ColMajor>()));
  CALL_SUBTEST_9((testSparseOperationsNoAutomaticResizing<ColMajor, RowMajor>()));
  CALL_SUBTEST_9((testSparseOperationsNoAutomaticResizing<RowMajor, ColMajor>()));
  CALL_SUBTEST_9((testSparseOperationsNoAutomaticResizing<RowMajor, RowMajor>()));
}
