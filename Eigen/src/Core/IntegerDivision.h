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

  static EIGEN_DEVICE_FUNC DoubleWordInteger FromSum(T a, T b) {
    // convenient constructor that returns the full sum a + b
    T sum = a + b;
    return DoubleWordInteger(sum < a ? 1 : 0, sum);
  }
  static EIGEN_DEVICE_FUNC DoubleWordInteger FromProduct(T a, T b) {
    // convenient constructor that returns the full product a * b
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
  uint_fast16_t n = uint16_t(-1) >> (8 - p);
  uint_fast16_t q = 1 + (n / d);
  uint8_t result = static_cast<uint8_t>(q);
  return result;
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint16_t calc_magic(uint16_t d, int p) {
  uint_fast32_t n = uint32_t(-1) >> (16 - p);
  uint_fast32_t q = 1 + (n / d);
  uint16_t result = static_cast<uint16_t>(q);
  return result;
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint32_t calc_magic(uint32_t d, int p) {
  uint_fast64_t n = uint64_t(-1) >> (32 - p);
  uint_fast64_t q = 1 + (n / d);
  uint32_t result = static_cast<uint32_t>(q);
  return result;
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint64_t calc_magic(uint64_t d, int p) { return calc_magic_generic(d, p); }

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T umuluh_generic(T a, T b) {
  return DoubleWordInteger<T>::FromProduct(a, b).hi;
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint8_t umuluh(uint8_t a, uint8_t b) {
  uint_fast16_t result = (uint_fast16_t(a) * uint_fast16_t(b)) >> 8;
  return static_cast<uint8_t>(result);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint16_t umuluh(uint16_t a, uint16_t b) {
  uint_fast32_t result = (uint_fast32_t(a) * uint_fast32_t(b)) >> 16;
  return static_cast<uint16_t>(result);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint32_t umuluh(uint32_t a, uint32_t b) {
  uint_fast64_t result = (uint_fast64_t(a) * uint_fast64_t(b)) >> 32;
  return static_cast<uint32_t>(result);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint64_t umuluh(uint64_t a, uint64_t b) {
#if defined(EIGEN_GPU_COMPILE_PHASE)
  return __umul64hi(a, b);
#elif defined(SYCL_DEVICE_ONLY)
  return cl::sycl::mul_hi(a, b);
#elif EIGEN_COMP_MSVC && (EIGEN_ARCH_x86_64 || EIGEN_ARCH_ARM64)
  return __umulh(a, b);
#elif EIGEN_HAS_BUILTIN_INT128
  __uint128_t v = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
  return static_cast<uint64_t>(v >> 64);
#else
  return umuluh_generic(a, b);
#endif
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T smuluh_generic(T a, T b) {
  using UT = std::make_unsigned_t<T>;
  UT ua = static_cast<UT>(a);
  UT ub = static_cast<UT>(b);
  UT result = umuluh(ua, ub);
  if (a < 0) result -= ub;
  if (b < 0) result -= ua;
  return static_cast<T>(result);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int8_t smuluh(int8_t a, int8_t b) {
  int_fast16_t result = (int_fast16_t(a) * int_fast16_t(b)) >> 8;
  return static_cast<int8_t>(result);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int16_t smuluh(int16_t a, int16_t b) {
  int_fast32_t result = (int_fast32_t(a) * int_fast32_t(b)) >> 16;
  return static_cast<int16_t>(result);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int32_t smuluh(int32_t a, int32_t b) {
  int_fast64_t result = (int_fast64_t(a) * int_fast64_t(b)) >> 32;
  return static_cast<int32_t>(result);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int64_t smuluh(int64_t a, int64_t b) {
#if EIGEN_HAS_BUILTIN_INT128
  __int128_t v = static_cast<__int128_t>(a) * static_cast<__int128_t>(b);
  return static_cast<__int128_t>(v >> 64);
#else
  return smuluh_generic(a, b);
#endif
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T uintdiv_generic(T a, T magic, int shift) {
  T b = umuluh(a, magic);
  DoubleWordInteger<T> t = DoubleWordInteger<T>::FromSum(b, a) >> shift;
  return t.lo;
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint8_t uintdiv(uint8_t a, uint8_t magic, int shift) {
  uint_fast16_t b = umuluh(a, magic);
  uint_fast16_t t = (b + a) >> shift;
  return static_cast<uint8_t>(t);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint16_t uintdiv(uint16_t a, uint16_t magic, int shift) {
  uint_fast32_t b = umuluh(a, magic);
  uint_fast32_t t = (b + a) >> shift;
  return static_cast<uint16_t>(t);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint32_t uintdiv(uint32_t a, uint32_t magic, int shift) {
  uint_fast64_t b = umuluh(a, magic);
  uint_fast64_t t = (b + a) >> shift;
  return static_cast<uint32_t>(t);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint64_t uintdiv(uint64_t a, uint64_t magic, int shift) {
#if EIGEN_HAS_BUILTIN_INT128
  __uint128_t b = umuluh(a, magic);
  __uint128_t t = (b + a) >> shift;
  return static_cast<uint64_t>(t);
#else
  return uintdiv_generic(a, magic, shift);
#endif
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T sintdiv_generic(T a, T magic, int shift, bool sign) {
  constexpr int k = CHAR_BIT * sizeof(T);
  using UT = std::make_unsigned_t<T>;
  if (shift >= k) return 0;
  T b = smuluh(a, magic);
  UT ua = static_cast<UT>(a);
  UT ub = static_cast<UT>(b);
  UT lo = ub + ua;
  T hi = lo < ua ? 1 : 0;
  if (a < 0) hi--;
  if (b < 0) hi--;
  T t = shift == 0 ? static_cast<T>(lo) : static_cast<T>((hi << (k - shift)) | (lo >> shift));
  if (a < 0) t++;
  return sign ? -t : t;
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int8_t sintdiv(int8_t a, int8_t magic, int shift, bool sign) {
  int_fast16_t b = smuluh(a, magic);
  int_fast16_t t = (b + a) >> shift;
  t -= a >> 7;
  return static_cast<int8_t>(sign ? -t : t);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int16_t sintdiv(int16_t a, int16_t magic, int shift, bool sign) {
  int_fast32_t b = smuluh(a, magic);
  int_fast32_t t = (b + a) >> shift;
  t -= a >> 15;
  return static_cast<int16_t>(sign ? -t : t);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int32_t sintdiv(int32_t a, int32_t magic, int shift, bool sign) {
  int_fast64_t b = smuluh(a, magic);
  int_fast64_t t = (b + a) >> shift;
  t -= a >> 31;
  return static_cast<int32_t>(sign ? -t : t);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int64_t sintdiv(int64_t a, int64_t magic, int shift, bool sign) {
#if EIGEN_HAS_BUILTIN_INT128
  __int128_t b = smuluh(a, magic);
  __int128_t t = (b + a) >> shift;
  t -= a >> 63;
  return static_cast<int64_t>(sign ? -t : t);
#else
  return sintdiv_generic(a, magic, shift, sign);
#endif
}

template <typename Scalar, bool Signed = std::is_signed<Scalar>::value>
struct fast_div_op_impl;

template <typename Scalar>
struct fast_div_op_impl<Scalar, false> {
  using UnsignedScalar = std::make_unsigned_t<Scalar>;
  static constexpr int k = CHAR_BIT * sizeof(Scalar);
  template <typename Divisor>
  EIGEN_DEVICE_FUNC fast_div_op_impl(Divisor d) {
    using UnsignedDivisor = std::make_unsigned_t<Divisor>;
    eigen_assert(d != 0 && "Error: Division by zero attempted!");
    int tz = ctz(d);
    Divisor d_odd = d >> tz;
    UnsignedDivisor d_odd_abs = static_cast<UnsignedDivisor>(numext::abs(d_odd));
    int p = std::is_signed<Scalar>::value ? log2_floor(d_odd_abs) : log2_ceil(d_odd_abs);
    shift = tz + p;
    if (shift <= k) {
      if (p == 0) {
        magic = std::is_signed<Scalar>::value ? 1 : 0;
      } else {
        magic = static_cast<Scalar>(calc_magic(static_cast<UnsignedScalar>(d_odd_abs), p));
      }
    } else {
      magic = 0;
      shift = k;
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& a) const {
    return uintdiv(a, magic, shift);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a) const {
    return puintdiv(a, magic, shift);
  }

  Scalar magic;
  int shift;
};

template <typename Scalar>
struct fast_div_op_impl<Scalar, true> : fast_div_op_impl<Scalar, false> {
  using Base = fast_div_op_impl<Scalar, false>;
  static constexpr int k = CHAR_BIT * sizeof(Scalar);
  using UnsignedScalar = std::make_unsigned_t<Scalar>;
  template <typename Divisor>
  EIGEN_DEVICE_FUNC fast_div_op_impl(Divisor d) : Base(d), sign(d < 0) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& a) const {
    return sintdiv(a, Base::magic, Base::shift, sign);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a) const {
    return psintdiv(a, Base::magic, Base::shift, sign);
  }

  bool sign;
};

template <typename Scalar>
struct fast_div_op : fast_div_op_impl<Scalar> {
  EIGEN_STATIC_ASSERT((std::is_integral<Scalar>::value), "THE SCALAR MUST BE A BUILT IN INTEGER")
  using result_type = Scalar;
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
    Cost = functor_traits<scalar_product_op<Scalar>>::Cost + 2 * functor_traits<scalar_sum_op<Scalar>>::Cost
  };
};

}  // namespace internal

template <typename Scalar>
struct IntDivider {
  using FastDivOp = internal::fast_div_op<Scalar>;
  template <typename Divisor>
  explicit EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE IntDivider(Divisor d) : op(d) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar divide(const Scalar& numerator) const { return op(numerator); }
  FastDivOp op;
};

template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator/(const Scalar& lhs, const IntDivider<Scalar>& rhs) {
  return rhs.divide(lhs);
}

}  // namespace Eigen

#endif  // EIGEN_INTDIV_H
