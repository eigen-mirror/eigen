// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_STRUCTURED_TEST_HELPERS_H
#define EIGEN_STRUCTURED_TEST_HELPERS_H

#include <limits>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace Eigen {

// Coefficient-wise reference independent of StructuredMatrices and KroneckerProduct.
template <typename Scalar>
Matrix<Scalar, Dynamic, Dynamic> reference_kron(const Matrix<Scalar, Dynamic, Dynamic>& A,
                                                const Matrix<Scalar, Dynamic, Dynamic>& B) {
  Matrix<Scalar, Dynamic, Dynamic> K(A.rows() * B.rows(), A.cols() * B.cols());
  for (Index j = 0; j < A.cols(); ++j) {
    for (Index i = 0; i < A.rows(); ++i) {
      K.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
    }
  }
  return K;
}

// Vectorized complex products can introduce Inf - Inf across component accumulators.
// Preserve scalar evaluation order for the non-finite reference results.
template <typename Scalar>
Matrix<Scalar, Dynamic, 1> reference_product_ieee(const Matrix<Scalar, Dynamic, Dynamic>& matrix,
                                                  const Matrix<Scalar, Dynamic, 1>& vector) {
  Matrix<Scalar, Dynamic, 1> result(matrix.rows());
  for (Index i = 0; i < matrix.rows(); ++i) {
    Scalar accumulator(0);
    for (Index j = 0; j < matrix.cols(); ++j) accumulator += matrix(i, j) * vector[j];
    result[i] = accumulator;
  }
  return result;
}

// The FFT-based products mix every input entry into every output entry, so an Inf or NaN may make more entries
// non-finite than the dense product does. \returns whether it makes at least those.
template <typename Actual, typename Reference>
bool nonfinite_covers_reference(const Actual& actual, const Reference& reference) {
  for (Index i = 0; i < reference.size(); ++i)
    if (!(numext::isfinite)(reference[i]) && (numext::isfinite)(actual[i])) return false;
  return true;
}

// A random sparse test matrix with about half of its entries stored, drawn
// densely so that the dense reference is exact.
template <typename Scalar, int Options = ColMajor>
SparseMatrix<Scalar, Options> random_sparse(Index rows, Index cols) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  Mat dense = Mat::Random(rows, cols);
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i)
      if (internal::random<int>(0, 1) == 0) dense(i, j) = Scalar(0);
  return dense.sparseView();
}

// An Inf or NaN in the right-hand side is never lost by the product with `op` and does not leak into other columns
// (see nonfinite_covers_reference()). `dense` is the reference matrix of `op`. \returns the vector with the Inf at
// inf_index.
template <typename Operator, typename Scalar>
Matrix<Scalar, Dynamic, 1> check_nonfinite_rhs_product(const Operator& op,
                                                       const Matrix<Scalar, Dynamic, Dynamic>& dense, Index inf_index) {
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;
  const Index n = dense.cols();

  Vec x = Vec::Random(n);
  x[inf_index] = Scalar(std::numeric_limits<RealScalar>::infinity());
  VERIFY(nonfinite_covers_reference((op * x).eval(), reference_product_ieee(dense, x)));
  Vec xn = Vec::Random(n);
  xn[n - 1] = Scalar(std::numeric_limits<RealScalar>::quiet_NaN());
  VERIFY(nonfinite_covers_reference((op * xn).eval(), reference_product_ieee(dense, xn)));

  Mat Xm(n, 2);
  Xm.col(0) = Vec::Random(n);
  Xm.col(1) = x;
  Mat Ym = op * Xm;
  VERIFY_IS_APPROX(Ym.col(0).eval(), (dense * Xm.col(0)).eval());
  VERIFY(nonfinite_covers_reference(Ym.col(1).eval(), reference_product_ieee(dense, Vec(Xm.col(1)))));
  return x;
}

// The product of an operator holding Inf or NaN data with a finite random vector keeps every non-finite entry of the
// dense product.
template <typename Operator, typename Scalar>
void check_nonfinite_data_product(const Operator& op, const Matrix<Scalar, Dynamic, Dynamic>& dense) {
  using Vec = Matrix<Scalar, Dynamic, 1>;
  const Vec x = Vec::Random(dense.cols());
  VERIFY(nonfinite_covers_reference((op * x).eval(), reference_product_ieee(dense, x)));
}

}  // namespace Eigen

#endif  // EIGEN_STRUCTURED_TEST_HELPERS_H
