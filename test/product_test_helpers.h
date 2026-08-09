// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TEST_PRODUCT_TEST_HELPERS_H
#define EIGEN_TEST_PRODUCT_TEST_HELPERS_H

#include <iostream>

#include <Eigen/Core>

namespace Eigen {

// Rounding error bounds for matrix products, based on:
//
//   Deterministic: Higham, "Accuracy and Stability of Numerical Algorithms",
//     Thm 3.5: |fl(A*B) - A*B| <= gamma_k * |A| * |B|,  gamma_k ~ k * epsilon.
//
//   Probabilistic: Higham & Mary, "A New Approach to Probabilistic Rounding
//     Error Analysis", SISC 2019, Thm 3.4: under the assumption that rounding
//     errors are independent with mean zero:
//       |fl(A*B) - A*B| <= gamma_tilde_k * |A| * |B|,
//       gamma_tilde_k ~ lambda * sqrt(k) * epsilon,
//     holding with probability >= 1 - 2*exp(-lambda^2/2) per inner product.
//
// Two overloads are provided:
//
// 1. product_tolerance<Scalar>(inner_dim, ...) — RELATIVE tolerance for use
//    with isApprox(). Assumes random matrices in [-1,1], where sign
//    cancellation gives || |A|*|B| ||_F / ||A*B||_F ~ (3/4)*sqrt(k).
//    Combined: tol ~ lambda * num_products * k * epsilon.
//
// 2. product_error_bound(A, B, ...) — ABSOLUTE error bound for arbitrary
//    matrices. Computes || |A|*|B| ||_F directly.
//    Bound: lambda * sqrt(k) * epsilon * num_products * || |A|*|B| ||_F.
//
// Parameters common to both:
//   num_products: number of independent products contributing error (default 1).
//                 Use 2 when comparing two different evaluations of A*B.
//   lambda:       probability parameter; P(lambda) = 1 - 2*exp(-lambda^2/2).
//                 lambda=5 gives P > 0.9999 per inner product.

// Overload 1: Relative tolerance for random [-1,1] matrices.
template <typename Scalar>
typename NumTraits<Scalar>::Real product_tolerance(Index inner_dim, int num_products = 1, double lambda = 5) {
  using Real = typename NumTraits<Scalar>::Real;
  const Real lambda_real(lambda);
  return lambda_real * Real(num_products) * Real(inner_dim) * NumTraits<Scalar>::epsilon();
}

// Overload 2: Absolute error bound for arbitrary matrices.
// Returns lambda * sqrt(k) * epsilon * num_products * || |A|*|B| ||_F.
//
// || |A|*|B| ||_F is accumulated in double with an explicit loop rather than formed as
// (A.cwiseAbs() * B.cwiseAbs()).norm(): a bound built with the product implementation would inherit
// that implementation's defects, and one that overflowed to infinity would admit any result at all.
template <typename DerivedA, typename DerivedB>
typename NumTraits<typename DerivedA::Scalar>::Real product_error_bound(const MatrixBase<DerivedA>& A,
                                                                        const MatrixBase<DerivedB>& B,
                                                                        int num_products = 1, double lambda = 5) {
  using Scalar = typename DerivedA::Scalar;
  using Real = typename NumTraits<Scalar>::Real;
  const Index k = A.cols();
  double squared_norm = 0.0;
  for (Index i = 0; i < A.rows(); ++i)
    for (Index j = 0; j < B.cols(); ++j) {
      double sum = 0.0;
      for (Index l = 0; l < k; ++l)
        sum += static_cast<double>(numext::abs(A.coeff(i, l))) * static_cast<double>(numext::abs(B.coeff(l, j)));
      squared_norm += sum * sum;
    }
  const double bound = lambda * numext::sqrt(double(k)) * static_cast<double>(NumTraits<Scalar>::epsilon()) *
                       double(num_products) * numext::sqrt(squared_norm);
  return static_cast<Real>(bound);
}

// Verify that two computations of A*B agree within the Higham-Mary bound.
// Returns true if ||actual - expected||_F <= product_error_bound(A, B, ...).
template <typename D1, typename D2, typename DA, typename DB>
inline bool verifyProduct(const MatrixBase<D1>& actual, const MatrixBase<D2>& expected, const MatrixBase<DA>& A,
                          const MatrixBase<DB>& B, int num_products = 2, double lambda = 5) {
  using Real = typename NumTraits<typename DA::Scalar>::Real;
  Real bound = product_error_bound(A, B, num_products, lambda);
  Real error = (actual - expected).norm();
  // Negated so that a NaN error fails rather than slipping through the comparison.
  if (!(error <= bound)) {
    std::cerr << "Product verification failed: error " << error << " exceeds bound " << bound << std::endl;
    return false;
  }
  return true;
}

}  // namespace Eigen

#endif  // EIGEN_TEST_PRODUCT_TEST_HELPERS_H
