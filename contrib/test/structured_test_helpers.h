// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_STRUCTURED_TEST_HELPERS_H
#define EIGEN_STRUCTURED_TEST_HELPERS_H

#include <Eigen/Core>

namespace Eigen {

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

}  // namespace Eigen

#endif  // EIGEN_STRUCTURED_TEST_HELPERS_H
