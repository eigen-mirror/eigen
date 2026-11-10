// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Charlie Schlosser <cs.schlosser@gmail.com>
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_INTEGER_DIVISION_H
#define EIGEN_INTEGER_DIVISION_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// Division of k-bit integers by an invariant divisor d != 0.
//
// Unsigned, multiply-add (Robison, "N-Bit Unsigned Division Via N-Bit Multiply-Add", ARITH 2005). With
// f = floor(log2(d)), q = floor(2^(k+f) / d) and e = d - (2^(k+f) mod d),
//
//   floor(n / d) = (m n + a) >> (k + s),   where
//
//   (m, a, s) = (2^k - 1, 2^k - 1, 0)   for d = 1,
//               (2^(k-1), 0, f - 1)     for d = 2^f, f >= 1,
//               (q + 1, 0, f)           for e <= 2^f,
//               (q, q, f)               otherwise.
//
// m n + a <= (2^k - 1) 2^k, so the double-width sum cannot overflow. A divisor d >= 2^k is stored as (0, 0, 0).
//
// Unsigned, subtract (Granlund and Montgomery, "Division by invariant integers using multiplication", PLDI 1994,
// Fig. 4.1). With l = ceil(log2(d)) and m = floor(2^k (2^l - d) / d) + 1, t = floor(m n / 2^k), s1 = min(l, 1):
//
//   floor(n / d) = (t + ((n - t) >> s1)) >> (l - s1).
//
// A divisor d >= 2^k is stored as m = 1, l = k. This needs only the high half of m n, where multiply-add also needs
// the low half to see its carry. Double-width products up to 64 bits and x86's mul return both halves at once, so
// those use multiply-add; 64-bit scalars elsewhere take the two halves from two multiplications, and use this form.
//
// Signed k < 64, after Granlund and Montgomery, Fig. 5.2. With D = |d|, l = max(ceil(log2(D)), 1) - 1 and
// M = floor(2^(k+l) / D) + 1, every |n| <= 2^(k-1) satisfies
//
//   trunc(n / D) = (n M >> (k + l)) - (n >> (k - 1)),
//
// arithmetic shifts in double width. 2^(k-1) < M < 2^k for D >= 2 and M = 2^k + 1 for D = 1, so M - 2^k fits in k
// signed bits and the high half of n M is n + mulhs(M - 2^k, n), one sequence for every d. Only n = lowest() with D = 1
// wraps, in n M or in that sum; there l = 0, so no shift sees the wrapped sign and the result is exact modulo 2^k.
// A divisor D > 2^k divides like D = 2^k: every quotient is 0.
namespace int_div {

template <typename T>
EIGEN_DEVICE_FUNC constexpr bool is_negative(const T& x, std::true_type) {
  return x < 0;
}
template <typename T>
EIGEN_DEVICE_FUNC constexpr bool is_negative(const T&, std::false_type) {
  return false;
}
template <typename T>
EIGEN_DEVICE_FUNC constexpr bool is_negative(const T& x) {
  return is_negative(x, std::is_signed<T>());
}

// |x| as the unsigned type of the same width; exact for lowest().
template <typename T>
EIGEN_DEVICE_FUNC constexpr std::make_unsigned_t<T> unsigned_abs(const T& x) {
  using Unsigned = std::make_unsigned_t<T>;
  return is_negative(x) ? static_cast<Unsigned>(Unsigned(0) - static_cast<Unsigned>(x)) : static_cast<Unsigned>(x);
}

// floor(a b / 2^64).
template <typename Unsigned, std::enable_if_t<sizeof(Unsigned) == 8, int> = 0>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Unsigned umulhi(Unsigned a, Unsigned b) {
#if defined(EIGEN_GPU_COMPILE_PHASE)
  return __umul64hi(a, b);
#elif defined(EIGEN_SYCL_DEVICE_ONLY)
  return cl::sycl::mul_hi(static_cast<uint64_t>(a), static_cast<uint64_t>(b));
#elif EIGEN_COMP_MSVC && (EIGEN_ARCH_x86_64 || EIGEN_ARCH_ARM64)
  return __umulh(a, b);
#elif EIGEN_HAS_BUILTIN_INT128
  return static_cast<Unsigned>((static_cast<__uint128_t>(a) * b) >> 64);
#else
  // Hacker's Delight, 2nd ed., Fig. 8-2: with 32-bit digits, u and w stay below 2^64.
  Unsigned a0 = a & 0xffffffffu, a1 = a >> 32, b0 = b & 0xffffffffu, b1 = b >> 32;
  Unsigned u = a1 * b0 + ((a0 * b0) >> 32);
  Unsigned w = a0 * b1 + (u & 0xffffffffu);
  return a1 * b1 + (u >> 32) + (w >> 32);
#endif
}

// floor(r 2^k / d) for r < d, which fits in k bits.
template <typename Unsigned, std::enable_if_t<(sizeof(Unsigned) < 8), int> = 0>
EIGEN_DEVICE_FUNC Unsigned div_shifted(Unsigned r, Unsigned d) {
  return static_cast<Unsigned>((static_cast<uint64_t>(r) << (CHAR_BIT * sizeof(Unsigned))) / d);
}
template <typename Unsigned, std::enable_if_t<sizeof(Unsigned) == 8, int> = 0>
EIGEN_DEVICE_FUNC Unsigned div_shifted(Unsigned r, Unsigned d) {
#if EIGEN_HAS_BUILTIN_INT128 && !defined(EIGEN_GPU_COMPILE_PHASE) && !defined(__SYCL_DEVICE_ONLY__)
  return static_cast<Unsigned>((static_cast<__uint128_t>(r) << 64) / d);
#else
  // Restoring division, one quotient bit per step, keeping r < d.
  Unsigned q = 0;
  for (int i = 0; i < 64; ++i) {
    bool carry = (r >> 63) != 0;
    r <<= 1;
    q <<= 1;
    if (carry || r >= d) {
      r -= d;
      q |= 1;
    }
  }
  return q;
#endif
}

// Whether k-bit division uses multiply-add: see above. Depends on the architecture alone, not on the vectorization
// flags, so the constants of a divider mean the same in translation units compiled for different instruction sets.
template <typename Unsigned>
struct use_multiply_add : std::integral_constant<bool, (sizeof(Unsigned) < 8) || EIGEN_ARCH_i386_OR_x86_64> {};

// The unsigned type holding the double-width product of k < 64 bit operands. It is never narrower than 32 bits: 16-bit
// operands would promote to int, where the product can overflow.
template <typename Unsigned>
using double_width_t = std::conditional_t<(sizeof(Unsigned) < 4), uint32_t, uint64_t>;

// floor((a b + c) / 2^64), which fits in 64 bits for a, b, c < 2^64.
template <typename Unsigned, std::enable_if_t<sizeof(Unsigned) == 8, int> = 0>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Unsigned umuladdhi(Unsigned a, Unsigned b, Unsigned c) {
#if EIGEN_HAS_BUILTIN_INT128 && !defined(EIGEN_GPU_COMPILE_PHASE) && !defined(__SYCL_DEVICE_ONLY__)
  return static_cast<Unsigned>((static_cast<__uint128_t>(a) * b + c) >> 64);
#else
  Unsigned low = a * b;
  return umulhi(a, b) + static_cast<Unsigned>(static_cast<Unsigned>(low + c) < low);
#endif
}

// floor(a / d) for the constants of d computed by params. Below 64 bits one shift of the whole sum: compilers that
// vectorize this keep the lanes wide until the end, where narrowing to the high half first costs more shuffles.
template <typename Unsigned, std::enable_if_t<(sizeof(Unsigned) < 8), int> = 0>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Unsigned udiv(Unsigned a, Unsigned magic, Unsigned addend, int shift) {
  using Wide = double_width_t<Unsigned>;
  return static_cast<Unsigned>((static_cast<Wide>(a) * static_cast<Wide>(magic) + static_cast<Wide>(addend)) >>
                               (CHAR_BIT * sizeof(Unsigned) + shift));
}
template <typename Unsigned, std::enable_if_t<sizeof(Unsigned) == 8 && use_multiply_add<Unsigned>::value, int> = 0>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Unsigned udiv(Unsigned a, Unsigned magic, Unsigned addend, int shift) {
  return umuladdhi(a, magic, addend) >> shift;
}
template <typename Unsigned, std::enable_if_t<!use_multiply_add<Unsigned>::value, int> = 0>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Unsigned udiv(Unsigned a, Unsigned magic, Unsigned /*addend*/, int shift) {
  Unsigned t = umulhi(a, magic);
  int shift1 = shift > 0 ? 1 : 0;
  return (t + ((a - t) >> shift1)) >> (shift - shift1);
}

template <typename Unsigned, bool MultiplyAdd = use_multiply_add<Unsigned>::value>
struct params;

template <typename Unsigned>
struct params<Unsigned, true> {
  static_assert(std::is_unsigned<Unsigned>::value, "params takes the unsigned type of the numerators");
  static constexpr int kBits = CHAR_BIT * sizeof(Unsigned);

  template <typename UnsignedDivisor>
  EIGEN_DEVICE_FUNC explicit params(UnsignedDivisor d) : magic(0), addend(0), shift(0) {
    eigen_assert(d != 0 && "IntDivider: division by zero");
    // A wider divisor type can hold d >= 2^k, which keeps the defaults above.
    if (d <= static_cast<UnsignedDivisor>(NumTraits<Unsigned>::highest())) {
      using Wide = std::conditional_t<(sizeof(Unsigned) < 4), uint32_t, Unsigned>;
      Unsigned du = static_cast<Unsigned>(d);
      int f = log_2_impl<Unsigned>::run_floor(du);
      Wide power = Wide(1) << f;
      if (du == 1) {
        magic = addend = NumTraits<Unsigned>::highest();
      } else if (Wide(du) == power) {
        magic = static_cast<Unsigned>(Wide(1) << (kBits - 1));
        shift = f - 1;
      } else {
        Unsigned q = div_shifted(static_cast<Unsigned>(power), du);
        // r = 2^(k+f) mod d = -q d mod 2^k, since r < d < 2^k.
        Unsigned r = static_cast<Unsigned>(Wide(0) - Wide(q) * Wide(du));
        bool round_up = Wide(du) - Wide(r) <= power;
        magic = static_cast<Unsigned>(q + (round_up ? 1 : 0));
        addend = round_up ? Unsigned(0) : q;
        shift = f;
      }
    }
  }

  Unsigned magic;
  Unsigned addend;
  int shift;
};

template <typename Unsigned>
struct params<Unsigned, false> {
  static_assert(std::is_unsigned<Unsigned>::value, "params takes the unsigned type of the numerators");
  static constexpr int kBits = CHAR_BIT * sizeof(Unsigned);

  template <typename UnsignedDivisor>
  EIGEN_DEVICE_FUNC explicit params(UnsignedDivisor d) : magic(1), addend(0), shift(kBits) {
    eigen_assert(d != 0 && "IntDivider: division by zero");
    // A wider divisor type can hold d >= 2^k, which keeps the defaults above.
    if (d <= static_cast<UnsignedDivisor>(NumTraits<Unsigned>::highest())) {
      Unsigned du = static_cast<Unsigned>(d);
      shift = log_2_impl<Unsigned>::run_ceil(du);
      // r = 2^l - d < d, reduced mod 2^k for l = k.
      Unsigned r =
          shift == kBits ? static_cast<Unsigned>(Unsigned(0) - du) : static_cast<Unsigned>((Unsigned(1) << shift) - du);
      magic = static_cast<Unsigned>(div_shifted(r, du) + 1);
    }
  }

  Unsigned magic;
  Unsigned addend;
  int shift;
};

// The constants of signed division for k < 64. magic holds M modulo 2^32, which only |d| = 1 with k = 32 reduces. It is
// not wider so that compilers see a 32 x 32 -> 64-bit product: GCC vectorizes a truncated 64-bit multiplier as a
// 64 x 64-bit product, three times the multiplications. shift is l, and total_shift is k + l, or 0 for |d| = 1 with
// k = 32, where the low bits of M are 1.
template <typename Scalar>
struct signed_params {
  static constexpr int kBits = CHAR_BIT * sizeof(Scalar);

  template <typename UnsignedDivisor>
  EIGEN_DEVICE_FUNC explicit signed_params(UnsignedDivisor d) {
    eigen_assert(d != 0 && "IntDivider: division by zero");
    using Unsigned = std::make_unsigned_t<Scalar>;
    // Only a wider divisor type can hold d > 2^k - 1; 2^k gives the same quotients, all 0.
    bool fits = d <= static_cast<UnsignedDivisor>(NumTraits<Unsigned>::highest());
    uint64_t magnitude = fits ? static_cast<uint64_t>(d) : uint64_t(1) << kBits;
    shift = !fits ? kBits - 1 : numext::maxi(log_2_impl<Unsigned>::run_ceil(static_cast<Unsigned>(d)), 1) - 1;
    uint64_t full_magic = (uint64_t(1) << (kBits + shift)) / magnitude + 1;
    magic = static_cast<uint32_t>(full_magic);
    total_shift = (full_magic >> 32) != 0 ? 0 : kBits + shift;
  }

  uint32_t magic;
  int shift;
  int total_shift;
};

// trunc(a / d) for the constants of signed_params and divisor_sign = -[d < 0]. lowest() / -1 wraps to lowest().
//
// 32 bits: compilers vectorize a loop over this badly in the signed form, which needs a 64-bit arithmetic shift.
// Multiply the magnitude and shift the whole product instead. The masks avoid a select, which mispredicts on random
// signs.
template <typename Scalar, std::enable_if_t<sizeof(Scalar) == 4, int> = 0>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar sdiv(Scalar a, uint32_t magic, int total_shift, Scalar divisor_sign) {
  uint32_t a_sign = static_cast<uint32_t>(a >> 31);
  uint32_t abs_a = (static_cast<uint32_t>(a) ^ a_sign) - a_sign;
  uint32_t quotient = static_cast<uint32_t>((static_cast<uint64_t>(abs_a) * magic) >> total_shift);
  uint32_t sign = a_sign ^ static_cast<uint32_t>(divisor_sign);
  return static_cast<Scalar>((quotient ^ sign) - sign);
}
template <typename Scalar, std::enable_if_t<(sizeof(Scalar) < 4), int> = 0>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar sdiv(Scalar a, uint32_t magic, int total_shift, Scalar divisor_sign) {
  using Unsigned = std::make_unsigned_t<Scalar>;
  int32_t wide = static_cast<int32_t>(a);
  int32_t product = static_cast<int32_t>(static_cast<uint32_t>(wide) * magic);
  Unsigned quotient =
      static_cast<Unsigned>(static_cast<Unsigned>(product >> total_shift) - static_cast<Unsigned>(wide >> 31));
  Unsigned sign = static_cast<Unsigned>(divisor_sign);
  return static_cast<Scalar>(static_cast<Unsigned>((quotient ^ sign) - sign));
}

}  // namespace int_div

/** \internal \returns floor(a / d) lane-wise, the lanes of \a a read as unsigned integers, for the \a magic,
 * \a addend and \a shift of d computed by int_div::params. This lane-by-lane fallback keeps every packet type correct;
 * backends specialize it where the instructions exist. */
template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet pudiv_magic(const Packet& a,
                                            std::make_unsigned_t<typename unpacket_traits<Packet>::type> magic,
                                            std::make_unsigned_t<typename unpacket_traits<Packet>::type> addend,
                                            int shift) {
  using Scalar = typename unpacket_traits<Packet>::type;
  using Unsigned = std::make_unsigned_t<Scalar>;
  constexpr int kSize = unpacket_traits<Packet>::size;
  Scalar lanes[kSize];
  pstoreu(lanes, a);
  for (int i = 0; i < kSize; ++i) {
    lanes[i] = static_cast<Scalar>(int_div::udiv(static_cast<Unsigned>(lanes[i]), magic, addend, shift));
  }
  return ploadu<Packet>(lanes);
}

/** \internal Truncating signed division: |q| = floor(|a| / |d|) with the sign of a * d. \a divisor_sign holds -1 in
 * every lane for d < 0, else 0. lowest() / -1 wraps to lowest(). */
template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet psdiv_magic(
    const Packet& a, std::make_unsigned_t<typename unpacket_traits<Packet>::type> magic,
    std::make_unsigned_t<typename unpacket_traits<Packet>::type> addend, int shift, const Packet& divisor_sign) {
  // For a real packet, pabs(lowest()) wraps to lowest(), whose unsigned reading is |lowest()|.
  Packet abs_quotient = pudiv_magic(pabs(a), magic, addend, shift);
  Packet quotient_sign = pxor(psignbit(a), divisor_sign);
  return psub(pxor(abs_quotient, quotient_sign), quotient_sign);
}

/** \internal \returns the truncated signed quotients lane-wise for the \a magic, \a shift and \a total_shift of d
 * computed by int_div::signed_params. \a divisor_sign holds -1 in every lane for d < 0, else 0. lowest() / -1 wraps to
 * lowest(). This lane-by-lane fallback keeps every packet type correct; backends specialize it where the instructions
 * exist. */
template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet psdiv_signed_magic(const Packet& a, uint32_t magic, int /*shift*/, int total_shift,
                                                   const Packet& divisor_sign) {
  using Scalar = typename unpacket_traits<Packet>::type;
  constexpr int kSize = unpacket_traits<Packet>::size;
  Scalar sign = pfirst(divisor_sign);
  Scalar lanes[kSize];
  pstoreu(lanes, a);
  for (int i = 0; i < kSize; ++i) lanes[i] = int_div::sdiv(lanes[i], magic, total_shift, sign);
  return ploadu<Packet>(lanes);
}

template <typename Scalar, bool IsSigned = std::is_signed<Scalar>::value, bool SignedMultiply = (sizeof(Scalar) < 8)>
struct scalar_int_div_op : int_div::params<Scalar> {
  using Base = int_div::params<Scalar>;
  using result_type = Scalar;

  template <typename Divisor>
  EIGEN_DEVICE_FUNC explicit scalar_int_div_op(const Divisor& d) : Base(int_div::unsigned_abs(d)) {
    eigen_assert(!int_div::is_negative(d) && "IntDivider: an unsigned numerator needs a positive divisor");
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a) const {
    return int_div::udiv(a, this->magic, this->addend, this->shift);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a) const {
    return pudiv_magic(a, this->magic, this->addend, this->shift);
  }
};

// Signed 64-bit: there is no double-width type to multiply in, so divide the magnitudes.
template <typename Scalar>
struct scalar_int_div_op<Scalar, true, false> : int_div::params<std::make_unsigned_t<Scalar>> {
  using Unsigned = std::make_unsigned_t<Scalar>;
  using Base = int_div::params<Unsigned>;
  using result_type = Scalar;

  template <typename Divisor>
  EIGEN_DEVICE_FUNC explicit scalar_int_div_op(const Divisor& d)
      : Base(int_div::unsigned_abs(d)), divisor_sign(int_div::is_negative(d) ? Scalar(-1) : Scalar(0)) {}

  // q = (|q| ^ s) - s for the mask s = -[q < 0]: a select here compiles to a mispredicted branch with some
  // compilers (GCC 12 for AArch64) when numerator signs are random. The conversion back to Scalar is modular; only
  // lowest() / -1 wraps.
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a) const {
    Unsigned abs_quotient = int_div::udiv(int_div::unsigned_abs(a), this->magic, this->addend, this->shift);
    Unsigned quotient_sign = (Unsigned(0) - Unsigned(a < 0)) ^ static_cast<Unsigned>(divisor_sign);
    return static_cast<Scalar>((abs_quotient ^ quotient_sign) - quotient_sign);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a) const {
    return psdiv_magic(a, this->magic, this->addend, this->shift, pset1<Packet>(divisor_sign));
  }

  Scalar divisor_sign;
};

// Signed below 64 bits: one multiplication and no absolute value, for every divisor.
template <typename Scalar>
struct scalar_int_div_op<Scalar, true, true> : int_div::signed_params<Scalar> {
  using Base = int_div::signed_params<Scalar>;
  using result_type = Scalar;

  template <typename Divisor>
  EIGEN_DEVICE_FUNC explicit scalar_int_div_op(const Divisor& d)
      : Base(int_div::unsigned_abs(d)), divisor_sign(int_div::is_negative(d) ? Scalar(-1) : Scalar(0)) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a) const {
    return int_div::sdiv(a, this->magic, this->total_shift, divisor_sign);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a) const {
    return psdiv_signed_magic(a, this->magic, this->shift, this->total_shift, pset1<Packet>(divisor_sign));
  }

  Scalar divisor_sign;
};

template <typename Scalar, bool IsSigned, bool SignedMultiply>
struct functor_traits<scalar_int_div_op<Scalar, IsSigned, SignedMultiply>> {
  static constexpr int Cost = 2 * NumTraits<Scalar>::MulCost + 5 * NumTraits<Scalar>::AddCost;
  static constexpr bool PacketAccess = packet_traits<Scalar>::HasFastIntDiv;
};

}  // namespace internal

/** \class IntDivider
 * \ingroup Core_Module
 *
 * \brief Divides integers by a divisor that is fixed at run time, with multiplications and shifts in place of
 * division instructions.
 *
 * \tparam Scalar_ the type of the numerators and quotients: a built-in integer type other than \c bool.
 *
 * The constructor precomputes a multiplier and a shift count from the divisor. Each quotient then costs a
 * double-width multiplication, a few additions and subtractions and a shift or two, which on most hardware is several
 * times cheaper than an integer division, and vectorizes where hardware integer division does not exist. Each quotient
 * is the mathematical quotient truncated toward zero, which is what the built-in \c / operator computes when both
 * operands have type \c Scalar_:
 *
 * \code
 * Eigen::IntDivider<int> by(d);           // d is known only at run time
 * int q = n / by;                         // == n / d
 * Eigen::ArrayXi Q = A / by;              // == A / d, coefficient-wise
 * n /= by;                                // n = n / by
 * A /= by;                                // A = A / by
 * \endcode
 *
 * The divisor may have any built-in integer type, and its magnitude may exceed the range of \c Scalar_. Quotients use
 * the divisor's mathematical value, without the built-in operator's usual arithmetic conversions, so a divisor of the
 * other signedness can give a different quotient: an \c IntDivider of \c int constructed from \c 3u divides -6 to -2,
 * while <tt>-6 / 3u</tt> is 1431655763. The divisor must be nonzero, and it must be positive when \c Scalar_ is
 * unsigned. \c lowest() divided by -1 wraps to \c lowest(), where the built-in operator overflows for \c int and wider
 * types.
 *
 * A scalar numerator of another integer type converts to \c Scalar_ first. Its type may be no wider than the wider of
 * \c Scalar_ and \c int, so that literals and arithmetic promoted to \c int, such as <tt>(a + b) / by</tt>, divide by
 * an \c IntDivider of a narrower type. A wider integer or a floating-point numerator does not compile, since it would
 * be truncated.
 *
 * A default-constructed \c IntDivider divides by 1; compute() sets the divisor later, for example in a member that is
 * initialized after construction.
 *
 * Use \c IntDivider when one divisor that is not a compile-time constant divides many numerators: constructing it costs
 * about as much as a few divisions. For a constant divisor, write the constant; the compiler then derives a similar
 * sequence itself.
 */
template <typename Scalar_>
class IntDivider {
  EIGEN_STATIC_ASSERT((std::is_integral<Scalar_>::value && !std::is_same<Scalar_, bool>::value),
                      THE_NUMERATOR_TYPE_MUST_BE_A_BUILT_IN_INTEGER)

 public:
  using Scalar = Scalar_;
  using Functor = internal::scalar_int_div_op<Scalar>;

  /** Default constructor: divides by 1 until compute() sets another divisor. */
  EIGEN_DEVICE_FUNC IntDivider() : m_functor(Scalar(1)) {}

  /** Precomputes the division by \a divisor. */
  template <typename Divisor>
  EIGEN_DEVICE_FUNC explicit IntDivider(const Divisor& divisor) : m_functor(divisor) {
    EIGEN_STATIC_ASSERT((std::is_integral<Divisor>::value && !std::is_same<Divisor, bool>::value),
                        THE_DIVISOR_TYPE_MUST_BE_A_BUILT_IN_INTEGER)
  }

  /** Precomputes the division by \a divisor, replacing the previous divisor. \returns a reference to \c *this. */
  template <typename Divisor>
  EIGEN_DEVICE_FUNC IntDivider& compute(const Divisor& divisor) {
    EIGEN_STATIC_ASSERT((std::is_integral<Divisor>::value && !std::is_same<Divisor, bool>::value),
                        THE_DIVISOR_TYPE_MUST_BE_A_BUILT_IN_INTEGER)
    m_functor = Functor(divisor);
    return *this;
  }

  /** \returns \a numerator divided by the divisor, truncated toward zero. */
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar divide(const Scalar& numerator) const { return m_functor(numerator); }

  /** \returns an expression of the coefficient-wise quotients of \a numerators by the divisor. */
  template <typename Derived>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CwiseUnaryOp<Functor, const Derived> divide(
      const DenseBase<Derived>& numerators) const {
    EIGEN_STATIC_ASSERT(
        (std::is_same<typename Derived::Scalar, Scalar>::value),
        YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
    return CwiseUnaryOp<Functor, const Derived>(numerators.derived(), m_functor);
  }

  /** \returns the functor that divides one numerator. */
  EIGEN_DEVICE_FUNC const Functor& functor() const { return m_functor; }

 private:
  Functor m_functor;
};

/** \relates IntDivider \returns \a numerator divided by the divisor of \a divider, truncated toward zero.
 *
 * \a numerator is an integer no wider than the wider of \c Scalar and \c int, and converts to \c Scalar first. */
template <typename Numerator, typename Scalar, std::enable_if_t<std::is_arithmetic<Numerator>::value, int> = 0>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator/(const Numerator& numerator, const IntDivider<Scalar>& divider) {
  // Conversions would truncate these silently: 7.5 / IntDivider<int>(2) == 3, (1LL << 40) / IntDivider<int>(1) == 0.
  // int stays allowed for literals and integer promotion.
  EIGEN_STATIC_ASSERT(
      (std::is_integral<Numerator>::value && (sizeof(Numerator) <= sizeof(Scalar) || sizeof(Numerator) <= sizeof(int))),
      THE_NUMERATOR_MUST_BE_AN_INTEGER_NO_WIDER_THAN_THE_DIVIDER_SCALAR_OR_INT)
  return divider.divide(static_cast<Scalar>(numerator));
}

/** \relates IntDivider Divides \a numerator by the divisor of \a divider, truncated toward zero.
 * \returns a reference to \a numerator. */
template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& operator/=(Scalar& numerator, const IntDivider<Scalar>& divider) {
  numerator = divider.divide(numerator);
  return numerator;
}

/** \relates IntDivider \returns an expression of the coefficient-wise quotients of \a numerators by the divisor of
 * \a divider. */
template <typename Derived, typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CwiseUnaryOp<internal::scalar_int_div_op<Scalar>, const Derived> operator/(
    const DenseBase<Derived>& numerators, const IntDivider<Scalar>& divider) {
  return divider.divide(numerators);
}

}  // namespace Eigen

#endif  // EIGEN_INTEGER_DIVISION_H
