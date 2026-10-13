// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPARSE_APPROX_COMPARISONS_H
#define EIGEN_SPARSE_APPROX_COMPARISONS_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename Scalar, bool UseScaledComparison>
struct sparse_approx_comparison {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Accumulator = RealScalar;
  // Custom scalars are summed as they are; a conversion to std::complex is not required of them.
  using Wide = Scalar;

  template <typename DifferenceNorm2, typename Scaled>
  static bool compare(const RealScalar& x2, const RealScalar& y2, Index, const RealScalar& prec,
                      const DifferenceNorm2& difference2, const Scaled&) {
    return difference2() <= prec * prec * numext::mini(x2, y2);
  }

  template <typename X, typename Y>
  static bool isApprox(const X& x, const Y& y, const RealScalar& prec) {
    return compare(
        x.squaredNorm(), y.squaredNorm(), 0, prec, [&] { return (x - y).squaredNorm(); }, [] { return false; });
  }
};

// |x - y| <= prec min(|x|, |y|) as in the dense approx_comparison_impl: from the squared norms of the stored values
// while they are in range, and otherwise from norms of values scaled by powers of two, so tiny and huge matrices are
// compared rather than their flushed or overflowed squares.
template <typename Scalar>
struct sparse_approx_comparison<Scalar, true> {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Accumulator = typename stable_norm_accumulator<RealScalar>::type;
  // Coefficients widened to the accumulator, as approx_comparison_wide_t widens an expression's.
  using Wide = std::conditional_t<NumTraits<Scalar>::IsComplex, std::complex<Accumulator>, Accumulator>;
  using Dense = approx_comparison_impl<Scalar, true>;

  // x2 and y2 are the squared norms of operands with at most size stored values between them; difference2() gives
  // that of their difference, and scaled() decides the comparison when the squares cannot.
  template <typename DifferenceNorm2, typename Scaled>
  static bool compare(const Accumulator& x2, const Accumulator& y2, Index size, const RealScalar& prec,
                      const DifferenceNorm2& difference2, const Scaled& scaled) {
    const Accumulator minimum = numext::mini(x2, y2);
    const Accumulator precision2 = Accumulator(prec) * Accumulator(prec);
    const Accumulator bound = precision2 * minimum;
    // Only the smaller norm enters the bound; overflow of the larger norm is harmless.
    if (Dense::safe_squared_norm(bound, size) &&
        minimum >= Dense::template squared_norm_lower_bound<RealScalar>(size) &&
        precision2 >= Dense::template squared_norm_lower_bound<RealScalar>(1))
      return Accumulator(difference2()) <= bound;
    return scaled();
  }

  template <typename X, typename Y>
  static bool isApprox(const X& x, const Y& y, const RealScalar& prec) {
    // Widening must not admit operands that cannot be subtracted, such as half and float.
    EIGEN_CHECK_BINARY_COMPATIBILITY(scalar_difference_op<typename X::Scalar EIGEN_COMMA typename Y::Scalar>,
                                     typename X::Scalar, typename Y::Scalar)
    const auto& wideX = x.template cast<approx_comparison_wide_t<X>>();
    const auto& wideY = y.template cast<approx_comparison_wide_t<Y>>();
    // x - y stores at most the entries of both operands.
    return compare(
        wideX.squaredNorm(), wideY.squaredNorm(), stored_size(x) + stored_size(y), prec,
        [&] { return (wideX - wideY).squaredNorm(); }, [&] { return isApprox_scaled(x, y, prec); });
  }

 private:
  // An upper bound on the number of stored values; only types with compressed access count them.
  template <typename X, std::enable_if_t<has_compressed_access<X>::value, int> = 0>
  static Index stored_size(const X& x) {
    return x.nonZeros();
  }
  template <typename X, std::enable_if_t<!has_compressed_access<X>::value, int> = 0>
  static Index stored_size(const X& x) {
    return x.rows() * x.cols();
  }

  // The norm of a compressed matrix is that of its stored values, which the dense scaled norm takes as a vector.
  template <typename Plain>
  static comparison_magnitude<Accumulator> norm(const Plain& m) {
    return scaled_comparison_norm(m.coeffs().matrix());
  }

  template <typename X, typename Y>
  static EIGEN_DONT_INLINE bool isApprox_scaled(const X& xExpr, const Y& yExpr, const RealScalar& prec) {
    constexpr int Order = X::IsRowMajor ? RowMajor : ColMajor;
    using StorageIndex = typename X::StorageIndex;
    using WideX = approx_comparison_wide_t<X>;
    using WideY = approx_comparison_wide_t<Y>;
    using WideDifference = std::conditional_t<NumTraits<WideX>::IsComplex, WideX, WideY>;
    const SparseMatrix<WideX, Order, StorageIndex> x = xExpr.template cast<WideX>();
    const SparseMatrix<WideY, Order, StorageIndex> y = yExpr.template cast<WideY>();
    const auto nx = norm(x);
    const auto ny = norm(y);
    if (!nx.isFinite() || !ny.isFinite()) return false;
    auto tolerance = nx <= ny ? nx : ny;
    tolerance.multiply(numext::abs(Accumulator(prec)));

    // As in scaled_comparison_distance_impl: lift a maximum below 1 by a power of two first, so FTZ does not flush a
    // subnormal difference of normal values, and halve if the difference of finite values overflows.
    const Accumulator maxCoeff =
        numext::mini(Accumulator(1), numext::maxi(scaled_comparison_max_coeff(x.coeffs().matrix().realView()),
                                                  scaled_comparison_max_coeff(y.coeffs().matrix().realView())));
    const safe_scaling_factors<Accumulator> factors = supports_power_of_two_scaling<Accumulator>::value
                                                          ? safe_scaling<Accumulator>::compute_floor_factors(maxCoeff)
                                                          : safe_scaling_factors<Accumulator>();
    SparseMatrix<WideDifference, Order, StorageIndex> difference = x * factors.invScale - y * factors.invScale;
    auto distance = norm(difference);
    if (!distance.isFinite()) {
      const Accumulator halfInvScale = factors.invScale * Accumulator(0.5);
      difference = x * halfInvScale - y * halfInvScale;
      distance = norm(difference);
      distance.multiply(Accumulator(2));
    }
    distance.multiply(factors.scale);
    return distance <= tolerance;
  }
};

}  // namespace internal

template <typename Derived>
template <typename OtherDerived>
bool SparseMatrixBase<Derived>::isApprox(const SparseMatrixBase<OtherDerived>& other, const RealScalar& prec) const {
  const typename internal::nested_eval<Derived, 2, PlainObject>::type actualA(derived());
  std::conditional_t<bool(IsRowMajor) == bool(OtherDerived::IsRowMajor),
                     const typename internal::nested_eval<OtherDerived, 2, PlainObject>::type, const PlainObject>
      actualB(other.derived());

  return internal::sparse_approx_comparison<Scalar>::isApprox(actualA, actualB, prec);
}

}  // end namespace Eigen

#endif  // EIGEN_SPARSE_APPROX_COMPARISONS_H
