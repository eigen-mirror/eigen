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
template <typename DerivedA, typename DerivedB>
typename NumTraits<typename DerivedA::Scalar>::Real product_error_bound(const MatrixBase<DerivedA>& A,
                                                                        const MatrixBase<DerivedB>& B,
                                                                        int num_products = 1, double lambda = 5) {
  using Scalar = typename DerivedA::Scalar;
  using Real = typename NumTraits<Scalar>::Real;
  Index k = A.cols();
  Real abs_prod_norm = (A.cwiseAbs() * B.cwiseAbs()).norm();
  const Real lambda_real(lambda);
  return lambda_real * numext::sqrt(Real(k)) * NumTraits<Scalar>::epsilon() * Real(num_products) * abs_prod_norm;
}

// Verify that two computations of A*B agree within the Higham-Mary bound.
// Returns true if ||actual - expected||_F <= product_error_bound(A, B, ...).
template <typename D1, typename D2, typename DA, typename DB>
inline bool verifyProduct(const MatrixBase<D1>& actual, const MatrixBase<D2>& expected, const MatrixBase<DA>& A,
                          const MatrixBase<DB>& B, int num_products = 2, double lambda = 5) {
  using Real = typename NumTraits<typename DA::Scalar>::Real;
  Real bound = product_error_bound(A, B, num_products, lambda);
  Real error = (actual - expected).norm();
  if (error > bound) {
    std::cerr << "Product verification failed: error " << error << " exceeds bound " << bound << std::endl;
    return false;
  }
  return true;
}

}  // namespace Eigen

#endif  // EIGEN_TEST_PRODUCT_TEST_HELPERS_H
