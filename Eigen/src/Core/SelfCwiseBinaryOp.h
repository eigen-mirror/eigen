// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SELFCWISEBINARYOP_H
#define EIGEN_SELFCWISEBINARYOP_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& DenseBase<Derived>::operator*=(const Scalar& other) {
  using ConstantExpr = typename internal::plain_constant_type<Derived, Scalar>::type;
  using Op = internal::mul_assign_op<Scalar>;
  internal::call_assignment(derived(), ConstantExpr(rows(), cols(), other), Op());
  return derived();
}

template <typename Derived>
template <bool Enable, typename>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& DenseBase<Derived>::operator*=(const RealScalar& other) {
  realView() *= other;
  return derived();
}

template <typename Derived, typename Scalar, bool IsIntegral = std::is_integral<Scalar>::value>
struct div_assign_impl
{
  using ConstantExpr = typename internal::plain_constant_type<Derived, Scalar>::type;
  using Op = internal::div_assign_op<Scalar>;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Derived& derived, const Scalar& other)
  {
    internal::call_assignment(derived, ConstantExpr(derived.rows(), derived.cols(), other), Op());
  }
};
template <typename Derived, typename Scalar>
struct div_assign_impl<Derived,Scalar,true> {
  using FastDivOp = internal::fast_div_op<Scalar>;
  using FastDivXpr = CwiseUnaryOp<FastDivOp, Derived>;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Derived& derived, const Scalar& other) {
    internal::call_assignment(derived, FastDivXpr(derived, FastDivOp(other)));
  }
};

template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& DenseBase<Derived>::operator/=(const Scalar& other) {
  div_assign_impl<Derived, Scalar>::run(derived(), other);
  return derived();
}

template <typename Derived>
template <bool Enable, typename>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& DenseBase<Derived>::operator/=(const RealScalar& other) {
  realView() /= other;
  return derived();
}

}  // end namespace Eigen

#endif  // EIGEN_SELFCWISEBINARYOP_H
