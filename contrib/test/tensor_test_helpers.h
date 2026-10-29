// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TENSOR_TEST_HELPERS_H
#define EIGEN_TENSOR_TEST_HELPERS_H

#include <contrib/Eigen/Tensor>

namespace Eigen {

// Keep each suite's integer draw type so fixed seeds retain their inputs.
template <int NumDims, typename RandomIndex = Index>
DSizes<Index, NumDims> randomTensorDims(Index min = 1, Index max = 20) {
  DSizes<Index, NumDims> dims;
  for (int i = 0; i < NumDims; ++i) {
    dims[i] = internal::random<RandomIndex>(RandomIndex(min), RandomIndex(max));
  }
  return dims;
}

inline internal::TensorBlockShapeType randomTensorBlockShape() {
  return internal::random<bool>() ? internal::TensorBlockShapeType::kUniformAllDims
                                  : internal::TensorBlockShapeType::kSkewedInnerDims;
}

template <int NumDims>
size_t randomTensorBlockSize(const DSizes<Index, NumDims>& dims) {
  return internal::random<size_t>(1, dims.TotalSize());
}

}  // namespace Eigen

#endif  // EIGEN_TENSOR_TEST_HELPERS_H
