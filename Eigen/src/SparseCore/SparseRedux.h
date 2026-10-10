// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPARSEREDUX_H
#define EIGEN_SPARSEREDUX_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename Derived>
typename internal::traits<Derived>::Scalar SparseMatrixBase<Derived>::sum() const {
  // The sum of an empty matrix is zero, as for dense ones.
  if (size() == 0) return Scalar(0);
  Scalar res(0);
  internal::evaluator<Derived> thisEval(derived());
  for (Index j = 0; j < outerSize(); ++j)
    for (typename internal::evaluator<Derived>::InnerIterator iter(thisEval, j); iter; ++iter) res += iter.value();
  return res;
}

template <typename Scalar_, int Options_, typename Index_>
typename internal::traits<SparseMatrix<Scalar_, Options_, Index_> >::Scalar
SparseMatrix<Scalar_, Options_, Index_>::sum() const {
  if (this->isCompressed())
    return Matrix<Scalar, 1, Dynamic>::Map(m_data.valuePtr(), m_data.size()).sum();
  else
    return Base::sum();
}

template <typename Scalar_, int Options_, typename Index_>
typename internal::traits<SparseVector<Scalar_, Options_, Index_> >::Scalar
SparseVector<Scalar_, Options_, Index_>::sum() const {
  return Matrix<Scalar, 1, Dynamic>::Map(m_data.valuePtr(), m_data.size()).sum();
}

}  // end namespace Eigen

#endif  // EIGEN_SPARSEREDUX_H
