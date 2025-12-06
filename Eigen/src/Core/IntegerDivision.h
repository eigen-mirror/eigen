// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Charlie Schlosser <cs.schlosser@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_INTDIV_H
#define EIGEN_INTDIV_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename T>
struct DoubleWordInteger {
  EIGEN_STATIC_ASSERT((std::is_integral<T>::value) && (std::is_unsigned<T>::value),
                      "SCALAR MUST BE A BUILT IN UNSIGNED INTEGER")
  static constexpr int k = CHAR_BIT * sizeof(T);

  EIGEN_DEVICE_FUNC DoubleWordInteger(T highBits, T lowBits) : hi(highBits), lo(lowBits) {}
  EIGEN_DEVICE_FUNC DoubleWordInteger(T lowBits) : hi(0), lo(lowBits) {}

  static EIGEN_DEVICE_FUNC DoubleWordInteger FromSum(T a, T b) {
    T sum = a + b;
    return DoubleWordInteger(sum < a ? 1 : 0, sum);
  }
  static EIGEN_DEVICE_FUNC DoubleWordInteger FromProduct(T a, T b) {
    // convenient constructor that computes the full product of a*b
    constexpr int kh = k / 2;
    constexpr T kLowMask = T(-1) >> kh;

    T a_h = a >> kh;
    T a_l = a & kLowMask;
    T b_h = b >> kh;
    T b_l = b & kLowMask;

    T ab_hh = a_h * b_h;
    T ab_hl = a_h * b_l;
    T ab_lh = a_l * b_h;
    T ab_ll = a_l * b_l;

    DoubleWordInteger<T> result(ab_hh, ab_ll);
    result += DoubleWordInteger<T>(ab_hl >> kh, ab_hl << kh);
    result += DoubleWordInteger<T>(ab_lh >> kh, ab_lh << kh);

    eigen_assert(result.lo == T(a * b));
    return result;
  }

  EIGEN_DEVICE_FUNC DoubleWordInteger& operator+=(const DoubleWordInteger& rhs) {
    hi += rhs.hi;
    lo += rhs.lo;
    if (lo < rhs.lo) hi++;
    return *this;
  }
  EIGEN_DEVICE_FUNC DoubleWordInteger& operator+=(const T& rhs) {
    lo += rhs;
    if (lo < rhs) hi++;
    return *this;
  }
  EIGEN_DEVICE_FUNC DoubleWordInteger& operator-=(const DoubleWordInteger& rhs) {
    if (lo < rhs.lo) hi--;
    hi -= rhs.hi;
    lo -= rhs.lo;
    return *this;
  }
  EIGEN_DEVICE_FUNC DoubleWordInteger& operator-=(const T& rhs) {
    if (lo < rhs) hi--;
    lo -= rhs;
    return *this;
  }
  EIGEN_DEVICE_FUNC DoubleWordInteger& operator>>=(int shift) {
    if (shift >= k) {
      lo = hi << (shift - k);
      hi = 0;
    } else {
      lo >>= shift;
      lo |= hi << (k - shift);
      hi >>= shift;
    }
    return *this;
  }
  EIGEN_DEVICE_FUNC DoubleWordInteger& operator<<=(int shift) {
    if (shift >= k) {
      hi = lo << (shift - k);
      lo = 0;
    } else {
      hi <<= shift;
      hi |= lo >> (k - shift);
      lo <<= shift;
    }
    return *this;
  }

  EIGEN_DEVICE_FUNC DoubleWordInteger operator+(const DoubleWordInteger& rhs) const {
    DoubleWordInteger result = *this;
    result += rhs;
    return result;
  }
  EIGEN_DEVICE_FUNC DoubleWordInteger operator+(const T& rhs) const {
    DoubleWordInteger result = *this;
    result += rhs;
    return result;
  }
  EIGEN_DEVICE_FUNC DoubleWordInteger operator-(const DoubleWordInteger& rhs) const {
    DoubleWordInteger result = *this;
    result -= rhs;
    return result;
  }
  EIGEN_DEVICE_FUNC DoubleWordInteger operator-(const T& rhs) const {
    DoubleWordInteger result = *this;
    result -= rhs;
    return result;
  }
  EIGEN_DEVICE_FUNC DoubleWordInteger operator>>(int shift) const {
    DoubleWordInteger result = *this;
    result >>= shift;
    return result;
  }
  EIGEN_DEVICE_FUNC DoubleWordInteger operator<<(int shift) const {
    DoubleWordInteger result = *this;
    result <<= shift;
    return result;
  }

  EIGEN_DEVICE_FUNC bool operator==(const DoubleWordInteger& rhs) const { return hi == rhs.hi && lo == rhs.lo; }
  EIGEN_DEVICE_FUNC bool operator<(const DoubleWordInteger& rhs) const {
    return hi != rhs.hi ? hi < rhs.hi : lo < rhs.lo;
  }
  EIGEN_DEVICE_FUNC bool operator!=(const DoubleWordInteger& rhs) const { return !(*this == rhs); }
  EIGEN_DEVICE_FUNC bool operator>(const DoubleWordInteger& rhs) const { return rhs < *this; }
  EIGEN_DEVICE_FUNC bool operator<=(const DoubleWordInteger& rhs) const { return !(*this > rhs); }
  EIGEN_DEVICE_FUNC bool operator>=(const DoubleWordInteger& rhs) const { return !(*this < rhs); }

  T hi, lo;
};

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T calc_magic_generic(T d, int p) {
  EIGEN_STATIC_ASSERT((std::is_integral<T>::value) && (std::is_unsigned<T>::value),
                      "SCALAR MUST BE A BUILT IN UNSIGNED INTEGER")
  constexpr int k = CHAR_BIT * sizeof(T);

  // calculates ceil(2^(k+p) / d) mod 2^k
  // where p = log2_ceil(d)

  eigen_assert(d != 0 && "Error: Division by zero attempted!");

  // the logic below assumes that d > 1 and p > 0
  // if d == 1, then the magic number is 2^k mod 2^k == 0
  if (d == 1) return 0;

  // magic = 1 + floor(n / d) mod 2^k
  // n = 2^(k+p)-1, which is at least k+1 bits and at most 2k bits
  // p = log2_ceil(d), d <= 2^p
  // 2^k+1 > q >= 2^k
  // subtract 2^k * d, 2^k-1 * d ... and so forth until the high bits of q are depleted

  constexpr T nLowBits = T(-1);
  T nHighBits = nLowBits >> (k - p);

  DoubleWordInteger<T> n(nHighBits, nLowBits);
  DoubleWordInteger<T> q_inc(1, 0);   // the incremental amount to add to q
  DoubleWordInteger<T> qd_inc(d, 0);  // the incremental amount to subtract from n
  DoubleWordInteger<T> q(0, 0);       // the total number of times q is subtracted from n

  // in the worst-case scenario, this loop runs k+1 times
  while (n.hi) {
    if (n >= qd_inc) {
      q += q_inc;
      n -= qd_inc;
    }
    q_inc >>= 1;
    qd_inc >>= 1;
  }
  q += n.lo / d;
  return q.lo + 1;
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint8_t calc_magic(uint8_t d, int p) {
  uint16_t n = uint16_t(-1) >> (8 - p);
  uint16_t q = 1 + (n / d);
  uint8_t result = static_cast<uint8_t>(q);
  return result;
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint16_t calc_magic(uint16_t d, int p) {
  uint32_t n = uint32_t(-1) >> (16 - p);
  uint32_t q = 1 + (n / d);
  uint16_t result = static_cast<uint16_t>(q);
  return result;
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint32_t calc_magic(uint32_t d, int p) {
  uint64_t n = uint64_t(-1) >> (32 - p);
  uint64_t q = 1 + (n / d);
  uint32_t result = static_cast<uint32_t>(q);
  return result;
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint64_t calc_magic(uint64_t d, int p) { return calc_magic_generic(d, p); }

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T muluh_generic(T a, T b) {
  return DoubleWordInteger<T>::FromProduct(a, b).hi;
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint8_t muluh(uint8_t a, uint8_t b) {
  uint_fast16_t result = (uint_fast16_t(a) * uint_fast16_t(b)) >> 8;
  return static_cast<uint8_t>(result);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint16_t muluh(uint16_t a, uint16_t b) {
  uint_fast32_t result = (uint_fast32_t(a) * uint_fast32_t(b)) >> 16;
  return static_cast<uint16_t>(result);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint32_t muluh(uint32_t a, uint32_t b) {
  uint_fast64_t result = (uint_fast64_t(a) * uint_fast64_t(b)) >> 32;
  return static_cast<uint32_t>(result);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint64_t muluh(uint64_t a, uint64_t b) {
//#if defined(EIGEN_GPU_COMPILE_PHASE)
//  return __umul64hi(a, b);
//#elif defined(SYCL_DEVICE_ONLY)
//  return cl::sycl::mul_hi(a, b);
//#elif EIGEN_COMP_MSVC && (EIGEN_ARCH_x86_64 || EIGEN_ARCH_ARM64)
//  return __umulh(a, b);
//#elif EIGEN_HAS_BUILTIN_INT128
//  __uint128_t v = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
//  return static_cast<uint64_t>(v >> 64);
//#else
//  return muluh_generic(a, b);
//#endif
  return muluh_generic(a, b);
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T fast_int_div_generic(T a, T magic, int shift) {
  T b = muluh(a, magic);
  DoubleWordInteger<T> t = DoubleWordInteger<T>::FromSum(b, a) >> shift;
  return t.lo;
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint8_t fast_int_div(uint8_t a, uint8_t magic, int shift) {
  uint_fast16_t b = muluh(a, magic);
  uint_fast16_t t = (b + a) >> shift;
  return static_cast<uint8_t>(t);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint16_t fast_int_div(uint16_t a, uint16_t magic, int shift) {
  uint_fast32_t b = muluh(a, magic);
  uint_fast32_t t = (b + a) >> shift;
  return static_cast<uint16_t>(t);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint32_t fast_int_div(uint32_t a, uint32_t magic, int shift) {
  uint_fast64_t b = muluh(a, magic);
  uint_fast64_t t = (b + a) >> shift;
  return static_cast<uint32_t>(t);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint64_t fast_int_div(uint64_t a, uint64_t magic, int shift) {
#if EIGEN_HAS_BUILTIN_INT128
  __uint128_t b = muluh(a, magic);
  __uint128_t t = (b + a) >> shift;
  return static_cast<uint64_t>(t);
#else
  return fast_int_div_generic(a, magic, shift);
#endif
}

template <typename Scalar, bool Signed = std::is_signed<Scalar>::value>
struct fast_div_op_impl;

template <typename Scalar>
struct fast_div_op_impl<Scalar, false> {
  static constexpr int k = CHAR_BIT * sizeof(Scalar);
  template <typename Divisor>
  EIGEN_DEVICE_FUNC fast_div_op_impl(Divisor d) {
    eigen_assert(d != 0 && "Error: Division by zero attempted!");
    using UnsignedDivisor = typename std::make_unsigned<Divisor>::type;
    UnsignedDivisor abs_d = static_cast<UnsignedDivisor>(numext::abs(d));
    if (abs_d <= NumTraits<Scalar>::highest()) {
      // d is in range
      // reduce d to 2^tz * d_odd so that the magic number is smaller and easier to calculate
      int tz = ctz(abs_d);
      Scalar d_odd = static_cast<Scalar>(abs_d >> tz);
      int p = log2_ceil(d_odd);
      magic = calc_magic(d_odd, p);
      shift = p + tz;
    } else {
      // d is not in range
      // all divisions result in zero
      magic = 0;
      shift = k;
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& a) const {
    return fast_int_div(a, magic, shift);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a) const {
    return pfast_uint_div(a, magic, shift);
  }

  Scalar magic;
  int shift;
};

template <typename Scalar>
struct fast_div_op_impl<Scalar, true> : fast_div_op_impl<typename std::make_unsigned<Scalar>::type> {
  using UnsignedScalar = typename std::make_unsigned<Scalar>::type;
  using UnsignedImpl = fast_div_op_impl<UnsignedScalar>;
  template <typename Divisor>
  EIGEN_DEVICE_FUNC fast_div_op_impl(Divisor d) : UnsignedImpl(d), sign(d < 0) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& a) const {
    bool returnNegative = (a < 0) != sign;
    UnsignedScalar abs_a = numext::abs(a);
    Scalar result = UnsignedImpl::operator()(abs_a);
    return returnNegative ? -result : result;
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a) const {
    return pfast_sint_div(a, UnsignedImpl::magic, UnsignedImpl::shift, sign);
  }

  bool sign;
};

template <typename Scalar>
struct fast_div_op : fast_div_op_impl<Scalar> {
  EIGEN_STATIC_ASSERT((std::is_integral<Scalar>::value), "THE SCALAR MUST BE A BUILT IN INTEGER")
  using Base = fast_div_op_impl<Scalar>;
  template <typename Divisor>
  EIGEN_DEVICE_FUNC fast_div_op(Divisor d) : Base(d) {
    EIGEN_STATIC_ASSERT((std::is_integral<Divisor>::value), "THE DIVISOR MUST BE A BUILT IN INTEGER")
    eigen_assert(((std::is_signed<Scalar>::value) || (d > 0)) &&
                 "unable to divide an unsigned integer by a negative divisor!");
  }
};

template <typename Scalar>
struct functor_traits<fast_div_op<Scalar>> {
  enum {
    PacketAccess = packet_traits<Scalar>::HasFastIntDiv,
    Cost = functor_traits<scalar_product_op<Scalar>>::Cost + 4 * functor_traits<scalar_sum_op<Scalar>>::Cost
  };
};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_INTDIV_H
