// SPDX-FileCopyrightText: 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPARSE_TEST_HELPERS_H
#define EIGEN_SPARSE_TEST_HELPERS_H

#include <type_traits>

#include <Eigen/SparseCore>

namespace Eigen {

template <typename T, std::enable_if_t<(T::Flags & RowMajorBit) != 0, int> = 0>
auto innervec(T& matrix, Index index) -> decltype(matrix.row(index)) {
  return matrix.row(index);
}

template <typename T, std::enable_if_t<(T::Flags & RowMajorBit) == 0, int> = 0>
auto innervec(T& matrix, Index index) -> decltype(matrix.col(index)) {
  return matrix.col(index);
}

template <typename T>
typename T::Map make_map(T& matrix) {
  return typename T::Map(matrix.rows(), matrix.cols(), matrix.nonZeros(), matrix.outerIndexPtr(),
                         matrix.innerIndexPtr(), matrix.valuePtr(), matrix.innerNonZeroPtr());
}

template <typename T, std::enable_if_t<(T::Flags & RowMajorBit) != 0, int> = 0>
auto innerpanel(T& matrix, Index start, Index size) -> decltype(matrix.middleRows(start, size)) {
  return matrix.middleRows(start, size);
}

template <typename T, std::enable_if_t<(T::Flags & RowMajorBit) == 0, int> = 0>
auto innerpanel(T& matrix, Index start, Index size) -> decltype(matrix.middleCols(start, size)) {
  return matrix.middleCols(start, size);
}

template <typename MatrixType>
void init_sparse_diagonal(MatrixType& matrix) {
  using Scalar = typename MatrixType::Scalar;
  matrix.resize(4, 4);
  matrix.insert(0, 0) = Scalar(10);
  matrix.insert(1, 1) = Scalar(20);
  matrix.insert(2, 2) = Scalar(30);
  matrix.insert(3, 3) = Scalar(40);
  matrix.makeCompressed();
}

}  // namespace Eigen

#endif  // EIGEN_SPARSE_TEST_HELPERS_H
