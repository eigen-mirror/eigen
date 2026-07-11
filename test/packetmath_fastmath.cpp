// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

template <typename Scalar, typename Packet>
EIGEN_DONT_INLINE void store_ptrue(Scalar* output) {
  const Packet zero = Eigen::internal::pset1<Packet>(Scalar(0));
  Eigen::internal::pstoreu<Scalar, Packet>(output, Eigen::internal::ptrue(zero));
}

template <typename Scalar>
EIGEN_DONT_INLINE void store_extended_scalar_constants(Scalar* output) {
  output[0] = Eigen::internal::psignmask<Scalar>();
  output[1] = Eigen::internal::pinf<Scalar>();
  output[2] = Eigen::internal::pnan<Scalar>();
}

template <typename Scalar>
EIGEN_DONT_INLINE Eigen::numext::uint32_t extended_to_float_bits(const volatile Scalar* input) {
  const float narrowed = static_cast<float>(*input);
  return Eigen::numext::bit_cast<Eigen::numext::uint32_t>(narrowed);
}

template <typename Scalar, bool Vectorizable = Eigen::internal::packet_traits<Scalar>::Vectorizable>
struct packetmath_fastmath_runner {
  static void run() {}
};

template <typename Scalar>
struct packetmath_fastmath_runner<Scalar, true> {
  static void run() {
    typedef typename Eigen::internal::packet_traits<Scalar>::type Packet;
    const int packet_size = Eigen::internal::unpacket_traits<Packet>::size;
    Scalar output[packet_size];
    for (int i = 0; i < packet_size; ++i) {
      output[i] = Scalar(0);
    }

    store_ptrue<Scalar, Packet>(output);

    for (int i = 0; i < packet_size; ++i) {
      const unsigned char* lane_bytes = reinterpret_cast<const unsigned char*>(output + i);
      bool has_nonzero_byte = false;
      for (std::size_t j = 0; j < sizeof(Scalar); ++j) {
        has_nonzero_byte = has_nonzero_byte || lane_bytes[j] != 0;
      }
      VERIFY(has_nonzero_byte);
    }
  }
};

template <typename Scalar,
          bool HasIntegerBits =
              !std::is_void<typename Eigen::numext::get_integer_by_size<sizeof(Scalar)>::unsigned_type>::value>
struct extended_scalar_constant_runner {
  static void run() {}
};

template <typename Scalar>
struct extended_scalar_constant_runner<Scalar, false> {
  static void run() {
    Scalar actual[3];
    store_extended_scalar_constants(actual);

    // Floating-point classification is optimized away under -ffinite-math-only. Narrow through a volatile pointer and
    // inspect the resulting integer bits instead; this also avoids indeterminate padding in x87 long double objects.
    typedef Eigen::numext::uint32_t Bits;
    const Bits sign_bits = extended_to_float_bits(actual + 0);
    const Bits inf_bits = extended_to_float_bits(actual + 1);
    const Bits nan_bits = extended_to_float_bits(actual + 2);
    VERIFY_IS_EQUAL(sign_bits, Bits(0x80000000u));
    VERIFY_IS_EQUAL(inf_bits, Bits(0x7f800000u));
    VERIFY_IS_EQUAL(nan_bits & Bits(0x7fc00000u), Bits(0x7fc00000u));
  }
};

EIGEN_DECLARE_TEST(packetmath_fastmath) {
  CALL_SUBTEST(packetmath_fastmath_runner<float>::run());
  CALL_SUBTEST(packetmath_fastmath_runner<double>::run());
  CALL_SUBTEST(packetmath_fastmath_runner<Eigen::half>::run());
  CALL_SUBTEST(packetmath_fastmath_runner<Eigen::bfloat16>::run());
  CALL_SUBTEST(extended_scalar_constant_runner<long double>::run());
}
