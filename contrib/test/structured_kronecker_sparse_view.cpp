// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include <contrib/Eigen/StructuredMatrices>

using namespace Eigen;

template <typename Scalar, int Options = ColMajor>
SparseMatrix<Scalar, Options> random_sparse(Index rows, Index cols) {
  Matrix<Scalar, Dynamic, Dynamic> dense = Matrix<Scalar, Dynamic, Dynamic>::Random(rows, cols);
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i)
      if (internal::random<int>(0, 1) == 0) dense(i, j) = Scalar(0);
  return dense.sparseView();
}

// The view must iterate exactly the entries the sparse materialization stores,
// in increasing inner index, and behave as that matrix in every sparse
// expression: assignment, sums, sparse-dense and sparse-sparse products in both
// orders, transposition and scaling.
template <int Options, typename Op>
void check_view(const Op& K) {
  using Scalar = typename Op::Scalar;
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using Sparse = SparseMatrix<Scalar, Options>;
  using View = KroneckerSparseView<Op, Options>;

  const View V = K.template sparseView<Options>();
  STATIC_CHECK(bool(View::IsRowMajor) == bool(Options & RowMajorBit));
  VERIFY_IS_EQUAL(V.rows(), K.rows());
  VERIFY_IS_EQUAL(V.cols(), K.cols());
  Sparse M;
  M = K;
  const Mat dense = M;

  Index count = 0;
  for (Index k = 0; k < V.outerSize(); ++k) {
    typename Sparse::InnerIterator ref(M, k);
    Index previous = -1;
    for (typename View::InnerIterator it(V, k); it; ++it, ++ref) {
      VERIFY(bool(ref));
      VERIFY(it.index() > previous);
      previous = it.index();
      VERIFY_IS_EQUAL(it.index(), ref.index());
      VERIFY_IS_EQUAL(it.row(), ref.row());
      VERIFY_IS_EQUAL(it.col(), ref.col());
      VERIFY_IS_APPROX(it.value(), ref.value());
      ++count;
    }
    VERIFY(!bool(ref));
  }
  VERIFY_IS_EQUAL(count, M.nonZeros());
  VERIFY(internal::evaluator<View>(V).nonZerosEstimate() >= count);

  Sparse S;
  S = V;
  VERIFY_IS_EQUAL(S.nonZeros(), M.nonZeros());
  VERIFY_IS_APPROX(Mat(S), dense);
  const Sparse R = random_sparse<Scalar, Options>(K.rows(), K.cols());
  VERIFY_IS_APPROX(Mat(Sparse(R + V)), Mat(Mat(R) + dense));
  VERIFY_IS_APPROX(Mat(Sparse(Scalar(2) * V - R)), Mat(Scalar(2) * dense - Mat(R)));
  VERIFY_IS_APPROX(Mat(Sparse(V.transpose())), Mat(dense.transpose()));

  const Vec x = Vec::Random(K.cols());
  VERIFY_IS_APPROX(Vec(V * x), Vec(dense * x));
  const Vec z = Vec::Random(K.rows());
  VERIFY_IS_APPROX(Vec(V.transpose() * z), Vec(dense.transpose() * z));
  VERIFY_IS_APPROX(Mat(z.transpose() * V), Mat(z.transpose() * dense));
  const Sparse P = random_sparse<Scalar, Options>(K.cols(), 3);
  VERIFY_IS_APPROX(Mat(Sparse(V * P)), Mat(dense * Mat(P)));
  const Sparse Q = random_sparse<Scalar, Options>(3, K.rows());
  VERIFY_IS_APPROX(Mat(Sparse(Q * V)), Mat(Mat(Q) * dense));
}

template <typename Op>
void check_view_both_orders(const Op& K) {
  check_view<ColMajor>(K);
  check_view<RowMajor>(K);
}

template <typename Scalar>
void test_view_kronecker(Index m1, Index n1, Index m2, Index n2) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using Diag = DiagonalMatrix<Scalar, Dynamic>;
  using RowSparse = SparseMatrix<Scalar, RowMajor>;

  const Mat A = Mat::Random(m1, n1);
  const SparseMatrix<Scalar> S = random_sparse<Scalar>(m2, n2);
  const RowSparse Sr = random_sparse<Scalar, RowMajor>(m1, n1);
  const Diag D(Vec::Random(m2));

  check_view_both_orders(makeKroneckerOperator(A, S));
  check_view_both_orders(makeKroneckerOperator(Sr, S));
  check_view_both_orders(makeKroneckerOperator(S, A));
  check_view_both_orders(makeKroneckerOperator(Sr, D));
  check_view_both_orders(makeKroneckerOperator(D, A));
  check_view_both_orders(makeKroneckerOperator(Mat::Identity(m1, m1), S));
  check_view_both_orders(makeKroneckerOperator(S, Mat::Identity(m1, m1)));
  check_view_both_orders(makeKroneckerOperator(Mat::Identity(m1, n1 + 1), Sr));
  check_view_both_orders(makeKroneckerOperator(Mat::Identity(m1, m1), S, Mat::Identity(n1, n1)));
  check_view_both_orders(makeKroneckerOperator(makeKroneckerOperator(Sr, D), A));
  // An empty inner vector of one factor empties the matching inner vectors of the product.
  SparseMatrix<Scalar> E = S;
  E.col(0) *= Scalar(0);
  E.prune(Scalar(0));
  check_view_both_orders(makeKroneckerOperator(A, E));
  // An explicitly stored zero is an entry of the view, as of the materialization.
  SparseMatrix<Scalar> Z = S;
  Z.coeffRef(0, 0) = Scalar(0);
  check_view_both_orders(makeKroneckerOperator(A, Z));
  check_view_both_orders(makeKroneckerOperator(Z, D));
}

template <typename Scalar>
void test_view_kronecker_sum(Index n1, Index n2) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using Diag = DiagonalMatrix<Scalar, Dynamic>;
  using RowSparse = SparseMatrix<Scalar, RowMajor>;

  const Mat A = Mat::Random(n1, n1);
  const SparseMatrix<Scalar> S = random_sparse<Scalar>(n2, n2);
  const RowSparse Sr = random_sparse<Scalar, RowMajor>(n1, n1);
  const Diag D(Vec::Random(n2));

  check_view_both_orders(makeKroneckerSum(A, S));
  check_view_both_orders(makeKroneckerSum(Sr, D));
  check_view_both_orders(makeKroneckerSum(S, Mat::Identity(n1, n1)));
  check_view_both_orders(makeKroneckerSum(Sr, S, A));
  check_view_both_orders(makeKroneckerOperator(Mat::Identity(n2, n2), makeKroneckerSum(Sr, D)));
  check_view_both_orders(makeKroneckerSum(makeKroneckerOperator(S, Mat::Identity(2, 2)), Sr));
}

// The finite-difference assembly the view exists for: I + tau L built in one
// sparse expression, and the Jacobi preconditioner read through the view.
void test_view_finite_difference(Index n1, Index n2) {
  using Sparse = SparseMatrix<double>;
  using RowSparse = SparseMatrix<double, RowMajor>;
  const double tau = 0.25;
  Sparse Dx(n2, n2), Dy(n1, n1);
  for (Index j = 0; j < n2; ++j) {
    Dx.insert(j, j) = 2.0 + double(j);  // a variable diagonal, so that Jacobi is not a scaling
    if (j + 1 < n2) {
      Dx.insert(j + 1, j) = -1.0;  // separate statements: an insert invalidates references
      Dx.insert(j, j + 1) = -1.0;
    }
  }
  for (Index j = 0; j < n1; ++j) {
    Dy.insert(j, j) = 2.0;
    if (j + 1 < n1) {
      Dy.insert(j + 1, j) = -1.0;
      Dy.insert(j, j + 1) = -1.0;
    }
  }
  auto L = makeKroneckerSum(Dy, Dx);
  Sparse Lm;
  Lm = L;
  RowSparse Id(n1 * n2, n1 * n2);
  Id.setIdentity();
  const RowSparse M = Id + tau * L.sparseView<RowMajor>();
  VERIFY_IS_EQUAL(MatrixXd(M), MatrixXd(MatrixXd(Id) + tau * MatrixXd(Lm)));

  const VectorXd b = VectorXd::Random(n1 * n2);
  DiagonalPreconditioner<double> jacobi;
  jacobi.compute(L.sparseView());
  VERIFY_IS_EQUAL(jacobi.info(), Success);
  VERIFY_IS_APPROX(VectorXd(jacobi.solve(b)), VectorXd(b.cwiseQuotient(MatrixXd(Lm).diagonal())));
}

// Factors outside the common case: a compressed factor with unsorted inner
// vectors, signed zeros in a Kronecker sum, and 64-bit sparse indices.
void test_view_factor_contracts() {
  using Sparse = SparseMatrix<double>;
  Sparse U(3, 3);
  U.startVec(0);
  U.insertBackByOuterInnerUnordered(0, 2) = 1.0;
  U.insertBackByOuterInnerUnordered(0, 0) = 2.0;
  U.startVec(1);
  U.insertBackByOuterInnerUnordered(1, 1) = 3.0;
  U.startVec(2);
  U.insertBackByOuterInnerUnordered(2, 1) = 4.0;
  U.insertBackByOuterInnerUnordered(2, 0) = 5.0;
  U.finalize();
  VERIFY(U.innerIndicesAreSorted() != U.outerSize());
  const MatrixXd A = MatrixXd::Random(2, 2);
  check_view_both_orders(makeKroneckerOperator(A, U));
  check_view_both_orders(makeKroneckerOperator(U, A));
  check_view_both_orders(makeKroneckerSum(U, A));

  // -0 + -0 on the diagonal, -0 + 0 off it.
  Sparse X(2, 2), Y(2, 2);
  X.insert(0, 0) = -0.0;
  X.insert(1, 0) = -0.0;
  Y.insert(0, 0) = -0.0;
  Y.insert(0, 1) = -0.0;
  const auto L = makeKroneckerSum(X, Y);
  Sparse M;
  M = L;
  VERIFY(std::signbit(M.coeff(0, 0)));
  using View = KroneckerSparseView<KroneckerSum<Sparse, Sparse>, ColMajor>;
  const View V = L.sparseView();
  for (Index k = 0; k < V.outerSize(); ++k) {
    Sparse::InnerIterator ref(M, k);
    for (View::InnerIterator it(V, k); it; ++it, ++ref) VERIFY(std::signbit(it.value()) == std::signbit(ref.value()));
  }

  // The plain matrices Eigen evaluates the view into take the widest index of
  // its sparse factors.
  using Sparse64 = SparseMatrix<double, ColMajor, std::int64_t>;
  using Op64 = KroneckerOperator<MatrixXd, KroneckerSum<Sparse, Sparse64>>;
  STATIC_CHECK((std::is_same<KroneckerSparseView<Op64, ColMajor>::StorageIndex, std::int64_t>::value));
  STATIC_CHECK(
      (std::is_same<KroneckerSparseView<KroneckerOperator<MatrixXd, Sparse>, RowMajor>::StorageIndex, int>::value));
  STATIC_CHECK(
      (std::is_same<KroneckerSparseView<KroneckerOperator<MatrixXd, MatrixXd>, ColMajor>::StorageIndex, int>::value));
  const Sparse64 B = Sparse64(random_sparse<double>(3, 3));
  const auto K64 = makeKroneckerOperator(A, B);
  const Sparse64 E64 = K64.sparseView().eval();
  Sparse64 M64;
  M64 = K64;
  VERIFY_IS_APPROX(MatrixXd(E64), MatrixXd(M64));
}

EIGEN_DECLARE_TEST(structured_kronecker_sparse_view) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1((test_view_kronecker<double>(3, 4, 4, 2)));
    CALL_SUBTEST_1((test_view_kronecker<std::complex<float>>(2, 3, 3, 3)));
    CALL_SUBTEST_1((test_view_kronecker<double>(1, 1, 1, 1)));
    CALL_SUBTEST_2((test_view_kronecker_sum<double>(3, 4)));
    CALL_SUBTEST_2((test_view_kronecker_sum<std::complex<double>>(2, 3)));
    CALL_SUBTEST_2(test_view_finite_difference(6, 5));
    CALL_SUBTEST_3(test_view_factor_contracts());
  }
}
