// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2024 Charles Schlosser <cs.schlosser@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_RANDOM_IMPL_H
#define EIGEN_RANDOM_IMPL_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/****************************************************************************
 * Implementation of random                                               *
 ****************************************************************************/

// The SplitMix64 finalizer (Steele, Lea and Flood, "Fast splittable pseudorandom number generators", OOPSLA
// 2014): a bijection on 64-bit words with full avalanche, Stafford's "Mix13" variant of the MurmurHash3 finalizer.
// Applied to a Weyl sequence (state += 0x9E3779B97F4A7C15) it is the SplitMix64 generator; applied to a seed plus
// an index it seeds independent per-index streams.
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE numext::uint64_t splitmix64_mix(numext::uint64_t z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

template <typename Scalar, bool IsComplex, bool IsInteger>
struct random_default_impl {};

template <typename Scalar>
struct random_impl : random_default_impl<Scalar, NumTraits<Scalar>::IsComplex, NumTraits<Scalar>::IsInteger> {};

template <typename Scalar>
inline Scalar random(const Scalar& x, const Scalar& y) {
  return EIGEN_MATHFUNC_IMPL(random, Scalar)::run(x, y);
}

template <typename Scalar>
inline Scalar random() {
  return EIGEN_MATHFUNC_IMPL(random, Scalar)::run();
}

// PCG-XSH-RS: minimal, fast, high-quality 32-bit PRNG.
// Shared step function used by both core Random and TensorRandom.
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE unsigned pcg_xsh_rs_step(uint64_t* state, uint64_t increment) {
  uint64_t current = *state;
  *state = current * 6364136223846793005ULL + (increment | 1);
  return static_cast<unsigned>((current ^ (current >> 22)) >> (22 + (current >> 61)));
}

// Define EIGEN_USE_PCG_RANDOM before including Eigen to enable the thread-safe PCG-XSH-RS
// backend instead of std::rand(). This provides 32 bits of entropy per call, is thread-safe
// via thread_local state, and avoids the data races inherent in std::rand().
//
// thread_local is part of the C++14 baseline, so as in Memory.h only EIGEN_AVOID_THREAD_LOCAL and device passes
// (CUDA/HIP, SYCL) disable it. Clang additionally reports per-target TLS support through cxx_thread_local.
// __SYCL_DEVICE_ONLY__ rather than SYCL_DEVICE_ONLY, which Macros.h defines only under EIGEN_USE_SYCL.
#if defined(EIGEN_USE_PCG_RANDOM) && !defined(EIGEN_AVOID_THREAD_LOCAL) && !defined(EIGEN_GPU_COMPILE_PHASE) && \
    !defined(__SYCL_DEVICE_ONLY__) && (!EIGEN_COMP_CLANG || __has_feature(cxx_thread_local))
#define EIGEN_HAS_THREAD_LOCAL_RANDOM 1
#else
#define EIGEN_HAS_THREAD_LOCAL_RANDOM 0
#endif

#if EIGEN_HAS_THREAD_LOCAL_RANDOM

struct eigen_pcg_state {
  uint64_t state;
  uint64_t inc;
};

inline eigen_pcg_state& eigen_tl_random_state() {
  // Default seed: mix of two arbitrary constants to avoid zero-state.
  thread_local eigen_pcg_state tl_state = {0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL};
  return tl_state;
}

// pcg32_srandom_r (O'Neill, "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random
// Number Generation", HMC-CS-2014-0905) seeds from two words, initstate and initseq. Deriving both from one seed as
// inc = 2*seed+1 would give seed and seed ^ 2^63 the same increment and states differing by 2^63 forever
// (2^63 * a = 2^63 mod 2^64 for odd a), so each output of one stream would be the other's shifted by 4 bits.
// Hashing the seed with splitmix64_mix for the increment decorrelates the two.
inline void set_random_seed(uint64_t seed) {
  eigen_pcg_state& s = eigen_tl_random_state();
  s.state = 0;
  s.inc = (splitmix64_mix(seed) << 1) | 1;
  pcg_xsh_rs_step(&s.state, s.inc);
  s.state += seed;
  pcg_xsh_rs_step(&s.state, s.inc);
}

#else

inline void set_random_seed(uint64_t seed) { std::srand(static_cast<unsigned>(seed)); }

#endif  // EIGEN_HAS_THREAD_LOCAL_RANDOM

struct eigen_random_device {
  using ReturnType = unsigned;
#if EIGEN_HAS_THREAD_LOCAL_RANDOM
  static constexpr int Entropy = 32;
  static constexpr ReturnType Highest = 0xFFFFFFFFu;
#else
  static constexpr int Entropy = floor_log2((unsigned int)(RAND_MAX) + 1);
  static constexpr ReturnType Highest = RAND_MAX;
#endif
  static EIGEN_DEVICE_FUNC inline ReturnType run() {
#if EIGEN_HAS_THREAD_LOCAL_RANDOM && !defined(EIGEN_GPU_COMPILE_PHASE)
    eigen_pcg_state& s = eigen_tl_random_state();
    return pcg_xsh_rs_step(&s.state, s.inc);
#else
    // GPU or no thread_local: fall back to std::rand().
    return static_cast<unsigned>(std::rand());
#endif
  }
};

// Fill a built-in unsigned integer with numRandomBits beginning with the least significant bit
template <typename Scalar>
struct random_bits_impl {
  EIGEN_STATIC_ASSERT(std::is_unsigned<Scalar>::value, SCALAR MUST BE A BUILT - IN UNSIGNED INTEGER)
  using RandomDevice = eigen_random_device;
  using RandomReturnType = typename RandomDevice::ReturnType;
  static constexpr int kTotalBits = sizeof(Scalar) * CHAR_BIT;
  static constexpr int kEntropy = plain_enum_min(kTotalBits, RandomDevice::Entropy);
  // return a Scalar filled with numRandomBits beginning from the least significant bit
  static EIGEN_DEVICE_FUNC inline Scalar run(int numRandomBits) {
    eigen_assert((numRandomBits >= 0) && (numRandomBits <= kTotalBits));
    Scalar randomBits = 0;
    for (int filledBits = 0; filledBits < numRandomBits; filledBits += kEntropy) {
      Scalar r = static_cast<Scalar>(RandomDevice::run());
      int remainingBits = numRandomBits - filledBits;
      if (remainingBits < kEntropy) {
        // clear the excess bits to avoid UB and rounding bias
        r >>= kEntropy - remainingBits;
      }
      randomBits |= r << filledBits;
    }
    return randomBits;
  }
};

template <typename BitsType>
EIGEN_DEVICE_FUNC inline BitsType getRandomBits(int numRandomBits) {
  return random_bits_impl<BitsType>::run(numRandomBits);
}

// random implementation for a built-in floating point type
template <typename Scalar, bool BuiltIn = std::is_floating_point<Scalar>::value>
struct random_float_impl {
  using BitsType = typename numext::get_integer_by_size<sizeof(Scalar)>::unsigned_type;
  static constexpr EIGEN_DEVICE_FUNC inline int mantissaBits() {
    const int digits = NumTraits<Scalar>::digits();
    return digits - 1;
  }
  static EIGEN_DEVICE_FUNC inline Scalar run(int numRandomBits) {
    eigen_assert(numRandomBits >= 0 && numRandomBits <= mantissaBits());
    BitsType randomBits = getRandomBits<BitsType>(numRandomBits);
    // if fewer than MantissaBits is requested, shift them to the left
    randomBits <<= (mantissaBits() - numRandomBits);
    // randomBits is in the half-open interval [2,4)
    randomBits |= numext::bit_cast<BitsType>(Scalar(2));
    // result is in the half-open interval [-1,1)
    Scalar result = numext::bit_cast<Scalar>(randomBits) - Scalar(3);
    return result;
  }
};
// random implementation for a custom floating point type
// uses double as the implementation with a mantissa with a size equal to either the target scalar's mantissa or that of
// double, whichever is smaller
template <typename Scalar>
struct random_float_impl<Scalar, false> {
  static EIGEN_DEVICE_FUNC inline int mantissaBits() {
    const int digits = NumTraits<Scalar>::digits();
    constexpr int kDoubleDigits = NumTraits<double>::digits();
    return numext::mini(digits, kDoubleDigits) - 1;
  }
  static EIGEN_DEVICE_FUNC inline Scalar run(int numRandomBits) {
    eigen_assert(numRandomBits >= 0 && numRandomBits <= mantissaBits());
    Scalar result = static_cast<Scalar>(random_float_impl<double>::run(numRandomBits));
    return result;
  }
};

#if !EIGEN_COMP_NVCC
// random implementation for long double
// this specialization is not compatible with double-double scalars
template <bool Specialize = (sizeof(long double) == 2 * sizeof(uint64_t)) &&
                            ((std::numeric_limits<long double>::digits != (2 * std::numeric_limits<double>::digits)))>
struct random_longdouble_impl {
  static constexpr int Size = sizeof(long double);
  static constexpr EIGEN_DEVICE_FUNC int mantissaBits() { return NumTraits<long double>::digits() - 1; }
  static EIGEN_DEVICE_FUNC inline long double run(int numRandomBits) {
    eigen_assert(numRandomBits >= 0 && numRandomBits <= mantissaBits());
    EIGEN_USING_STD(memcpy);
    int numLowBits = numext::mini(numRandomBits, 64);
    int numHighBits = numext::maxi(numRandomBits - 64, 0);
    uint64_t randomBits[2];
    long double result = 2.0L;
    memcpy(&randomBits, &result, Size);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    randomBits[0] |= getRandomBits<uint64_t>(numLowBits);
    randomBits[1] |= getRandomBits<uint64_t>(numHighBits);
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    randomBits[0] |= getRandomBits<uint64_t>(numHighBits);
    randomBits[1] |= getRandomBits<uint64_t>(numLowBits);
#else
#error Unexpected or undefined __BYTE_ORDER__
#endif
    memcpy(&result, &randomBits, Size);
    result -= 3.0L;
    return result;
  }
};
template <>
struct random_longdouble_impl<false> {
  static constexpr EIGEN_DEVICE_FUNC int mantissaBits() { return NumTraits<double>::digits() - 1; }
  static EIGEN_DEVICE_FUNC inline long double run(int numRandomBits) {
    return static_cast<long double>(random_float_impl<double>::run(numRandomBits));
  }
};
template <>
struct random_float_impl<long double> : random_longdouble_impl<> {};
#endif

template <typename Scalar>
struct random_default_impl<Scalar, false, false> {
  using Impl = random_float_impl<Scalar>;
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y, int numRandomBits) {
    Scalar half_x = Scalar(0.5) * x;
    Scalar half_y = Scalar(0.5) * y;
    Scalar result = (half_x + half_y) + (half_y - half_x) * run(numRandomBits);
    // result is in the half-open interval [x, y) -- provided that x < y
    return result;
  }
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y) {
    return run(x, y, Impl::mantissaBits());
  }
  static EIGEN_DEVICE_FUNC inline Scalar run(int numRandomBits) { return Impl::run(numRandomBits); }
  static EIGEN_DEVICE_FUNC inline Scalar run() { return run(Impl::mantissaBits()); }
};

template <typename Scalar, bool IsSigned = NumTraits<Scalar>::IsSigned, bool BuiltIn = std::is_integral<Scalar>::value>
struct random_int_impl;

// random implementation for a built-in unsigned integer type
template <typename Scalar>
struct random_int_impl<Scalar, false, true> {
  static constexpr int kTotalBits = sizeof(Scalar) * CHAR_BIT;
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y) {
    if (y <= x) return x;
    Scalar range = y - x;
    // handle edge case where [x,y] spans the entire range of Scalar
    if (range == NumTraits<Scalar>::highest()) return run();
    Scalar count = range + 1;
    // calculate the number of random bits needed to fill range
    int numRandomBits = log2_ceil(count);
    Scalar randomBits;
    do {
      randomBits = getRandomBits<Scalar>(numRandomBits);
      // if the random draw is outside [0, range], try again (rejection sampling)
      // in the worst-case scenario, the probability of rejection is: 1/2 - 1/2^numRandomBits < 50%
    } while (randomBits >= count);
    Scalar result = x + randomBits;
    return result;
  }
  static EIGEN_DEVICE_FUNC inline Scalar run() { return getRandomBits<Scalar>(kTotalBits); }
};

// random implementation for a built-in signed integer type
template <typename Scalar>
struct random_int_impl<Scalar, true, true> {
  static constexpr int kTotalBits = sizeof(Scalar) * CHAR_BIT;
  // avoid implicit integral promotion to `int`
  using BitsType = std::conditional_t<(sizeof(Scalar) < sizeof(int)), unsigned int, std::make_unsigned_t<Scalar> >;
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y) {
    if (y <= x) return x;
    // Avoid overflow by representing `range` as an unsigned type
    BitsType range = static_cast<BitsType>(y) - static_cast<BitsType>(x);
    BitsType randomBits = random_int_impl<BitsType>::run(0, range);
    // Avoid overflow in the case where `x` is negative and there is a large range so
    // `randomBits` would also be negative if cast to `Scalar` first.
    Scalar result = static_cast<Scalar>(static_cast<BitsType>(x) + randomBits);
    return result;
  }
  static EIGEN_DEVICE_FUNC inline Scalar run() { return static_cast<Scalar>(getRandomBits<BitsType>(kTotalBits)); }
};

// todo: custom integers
template <typename Scalar, bool IsSigned>
struct random_int_impl<Scalar, IsSigned, false> {
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar&, const Scalar&) { return run(); }
  static EIGEN_DEVICE_FUNC inline Scalar run() {
    eigen_assert(std::false_type::value && "RANDOM FOR CUSTOM INTEGERS NOT YET SUPPORTED");
    return Scalar(0);
  }
};

template <typename Scalar>
struct random_default_impl<Scalar, false, true> : random_int_impl<Scalar> {};

template <>
struct random_impl<bool> {
  static EIGEN_DEVICE_FUNC inline bool run(const bool& x, const bool& y) {
    if (y <= x) return x;
    return run();
  }
  static EIGEN_DEVICE_FUNC inline bool run() { return getRandomBits<unsigned>(1) != 0; }
};

template <typename Scalar>
struct random_default_impl<Scalar, true, false> {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Impl = random_impl<RealScalar>;
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y, int numRandomBits) {
    return Scalar(Impl::run(x.real(), y.real(), numRandomBits), Impl::run(x.imag(), y.imag(), numRandomBits));
  }
  static EIGEN_DEVICE_FUNC inline Scalar run(const Scalar& x, const Scalar& y) {
    return Scalar(Impl::run(x.real(), y.real()), Impl::run(x.imag(), y.imag()));
  }
  static EIGEN_DEVICE_FUNC inline Scalar run(int numRandomBits) {
    return Scalar(Impl::run(numRandomBits), Impl::run(numRandomBits));
  }
  static EIGEN_DEVICE_FUNC inline Scalar run() { return Scalar(Impl::run(), Impl::run()); }
};

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_RANDOM_IMPL_H
