// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TENSOR_TENSOR_INTDIV_H
#define EIGEN_TENSOR_TENSOR_INTDIV_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

namespace tensor_int_div {

// floor(m n / 2^(k-1+l)) for n < 2^(k-1). shift is l for k = 64, where 2 n < 2^64, and k - 1 + l for k < 64.
template <typename Unsigned, std::enable_if_t<sizeof(Unsigned) == 8, int> = 0>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Unsigned quotient(Unsigned n, Unsigned m, int shift) {
  return int_div::umulhi(m, static_cast<Unsigned>(n << 1)) >> shift;
}
template <typename Unsigned, std::enable_if_t<(sizeof(Unsigned) < 8), int> = 0>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Unsigned quotient(Unsigned n, Unsigned m, int shift) {
  using Wide = int_div::double_width_t<Unsigned>;
  return static_cast<Unsigned>((static_cast<Wide>(m) * static_cast<Wide>(n)) >> shift);
}

}  // namespace tensor_int_div

/** \internal
 *
 * \ingroup Tensor_Module
 *
 * \brief Division of a non-negative index by a positive divisor that is fixed at run time.
 *
 * For a k-bit \c T, numerators n and divisors d lie in the non-negative range of a signed index:
 * 0 <= n < 2^(k-1) and 0 < d < 2^(k-1). With l = ceil(log2(d)) and m = floor(2^(k-1+l) / d) + 1,
 *
 *   2^(k-1+l) < m d <= 2^(k-1+l) + 2^l,   so   floor(n / d) = (m n) >> (k - 1 + l)
 *
 * (Granlund and Montgomery, "Division by invariant integers using multiplication", PLDI 1994, Theorem 4.2 with
 * N = k - 1). d > 2^(l-1) gives m < 2^k, so the bit that a non-negative index leaves unused lets one multiplication
 * and one shift serve every divisor. IntDivider, which takes all 2^k numerators, needs a multiply-add or a fix-up
 * instead, and one more register. A default-constructed divisor divides by 1. \c div_gt_one is ignored; it once
 * selected a kernel restricted to divisors of at least 2.
 *
 * \sa IntDivider, Tensor
 */
template <typename T, bool div_gt_one = false>
struct TensorIntDivisor {
 public:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorIntDivisor() = default;

  // Must have 0 < divider < 2^(k-1).
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorIntDivisor(const T divider) {
    eigen_assert(divider > 0 && in_range(divider));
    const UnsignedType d = static_cast<UnsignedType>(divider);
    const int l = log_2_impl<UnsignedType>::run_ceil(d);
    // 2^(k-1+l) = r 2^k with r = 2^(l-1) < d. d = 1 has l = 0 and keeps m = 2^(k-1).
    if (l > 0) {
      const UnsignedType r = static_cast<UnsignedType>(UnsignedType(1) << (l - 1));
      m_multiplier = static_cast<UnsignedType>(int_div::div_shifted(r, d) + 1);
    }
    m_shift = kBaseShift + l;
  }

  // Must have 0 <= numerator < 2^(k-1).
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T divide(const T numerator) const {
    eigen_assert(in_range(numerator));
    return static_cast<T>(tensor_int_div::quotient(static_cast<UnsignedType>(numerator), m_multiplier, m_shift));
  }

 private:
  using UnsignedType = std::make_unsigned_t<T>;
  static constexpr int kBits = CHAR_BIT * sizeof(T);
  static constexpr int kBaseShift = sizeof(T) == 8 ? 0 : kBits - 1;

  EIGEN_DEVICE_FUNC static constexpr bool in_range(const T x) {
    return (static_cast<UnsignedType>(x) >> (kBits - 1)) == 0;
  }

  UnsignedType m_multiplier = static_cast<UnsignedType>(UnsignedType(1) << (kBits - 1));
  int m_shift = kBaseShift;
};

template <typename T, bool div_gt_one>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator/(const T& numerator, const TensorIntDivisor<T, div_gt_one>& divisor) {
  return divisor.divide(numerator);
}

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_TENSOR_TENSOR_INTDIV_H
