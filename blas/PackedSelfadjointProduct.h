// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SELFADJOINT_PACKED_PRODUCT_H
#define EIGEN_SELFADJOINT_PACKED_PRODUCT_H

namespace Eigen {
namespace internal {

/* Optimized matrix += alpha * vv', with the left v conjugated if ConjLhs and the right one if ConjRhs.
 * The matrix is in packed form.
 */
template <typename Scalar, typename Index, int StorageOrder, int UpLo, bool ConjLhs, bool ConjRhs>
struct selfadjoint_packed_rank1_update;

template <typename Scalar, typename Index, int UpLo, bool ConjLhs, bool ConjRhs>
struct selfadjoint_packed_rank1_update<Scalar, Index, ColMajor, UpLo, ConjLhs, ConjRhs> {
  using RealScalar = typename NumTraits<Scalar>::Real;
  static void run(Index size, Scalar* mat, const Scalar* vec, RealScalar alpha) {
    using OtherMap = Map<const Matrix<Scalar, Dynamic, 1> >;
    using ConjLhsType = conj_expr_if<ConjLhs, OtherMap>;
    conj_if<ConjRhs> cj;

    for (Index i = 0; i < size; ++i) {
      // Column i holds rows [start, start + len).
      const Index start = UpLo == Lower ? i : 0, len = UpLo == Lower ? size - i : i + 1;
      Map<Matrix<Scalar, Dynamic, 1> >(mat, len) += alpha * cj(vec[i]) * ConjLhsType(OtherMap(vec + start, len));
      // FIXME This should be handled outside.
      mat[i - start] = numext::real(mat[i - start]);
      mat += len;
    }
  }
};

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_SELFADJOINT_PACKED_PRODUCT_H
