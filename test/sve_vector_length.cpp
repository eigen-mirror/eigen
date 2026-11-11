// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// SVE packet types are fixed-length vectors of exactly EIGEN_ARM64_SVE_VL bits. Running such a
// binary where the hardware vector length differs makes every packet type a lie and lets the
// rest of the suite pass vacuously. qemu is the common way in: above 512 bits it keeps the
// default 512-bit task vector length, so `-cpu max,sve2048=on` alone runs at 512 unless
// `sve-default-vector-length=<bytes>` is also given.

// The backend raises Eigen's own default to fit its blocked kernel panels, but an explicit limit
// is the caller's policy and must survive. 1 is below the requirement at every vector length, so
// the assertion is never vacuous, and nothing in this file allocates a fixed-size Eigen object.
#define EIGEN_STACK_ALLOCATION_LIMIT 1
#include "main.h"
static_assert(EIGEN_STACK_ALLOCATION_LIMIT == 1, "SVE must preserve an explicit stack allocation limit");

// Registered only under EIGEN_TEST_SVE (test/CMakeLists.txt). Without an SVE -march this would
// compile against NEON and trivially pass.
#if !defined(EIGEN_VECTORIZE_SVE)
#error \
    "sve_vector_length requires the SVE backend.  Build with an SVE -march (e.g. " \
    "-march=armv8.2-a+sve) and -DEIGEN_TEST_SVE=ON, which supplies -DEIGEN_ARM64_USE_SVE and " \
    "-msve-vector-bits."
#endif

// Must stay inline asm: under -msve-vector-bits the compiler folds svcntb() to the compile-time
// constant, making the comparison below a tautology that holds on a mismatched machine.
static EIGEN_DONT_INLINE int runtime_vector_length_bits() {
  numext::uint64_t vl_bytes;
  asm("rdvl %0, #1" : "=r"(vl_bytes));
  return int(vl_bytes) * CHAR_BIT;
}

void check_runtime_vector_length() { VERIFY_IS_EQUAL(runtime_vector_length_bits(), int(EIGEN_ARM64_SVE_VL)); }

// Cross-checked against RDVL rather than svcntw()/svcntd(), which fold as above.
void check_packet_sizes() {
  const int vl = runtime_vector_length_bits();
  VERIFY_IS_EQUAL(int(internal::packet_traits<numext::int32_t>::size), vl / (8 * int(sizeof(numext::int32_t))));
  VERIFY_IS_EQUAL(int(internal::packet_traits<float>::size), vl / (8 * int(sizeof(float))));
  VERIFY_IS_EQUAL(int(internal::packet_traits<double>::size), vl / (8 * int(sizeof(double))));
}

// GCC can lose lanes in full-register constructors of at least eight elements when the even or odd half has
// leading constants followed by nonconstant elements. SLP can produce these from ordinary scalar stores.
EIGEN_DONT_INLINE void scale_zero_prefixed_values(float* dst, float a, float b, float c, float d, float e, float f,
                                                  float g, float h, float i, float j, float scale) {
  dst[0] = dst[1] = dst[2] = dst[3] = dst[4] = dst[5] = 0 * scale;
  dst[6] = a * scale;
  dst[7] = b * scale;
  dst[8] = c * scale;
  dst[9] = d * scale;
  dst[10] = e * scale;
  dst[11] = f * scale;
  dst[12] = g * scale;
  dst[13] = h * scale;
  dst[14] = i * scale;
  dst[15] = j * scale;
}

template <typename Scalar, int Prefix>
EIGEN_DONT_INLINE void scale_constant_prefixed_values(Scalar* dst, Scalar a, Scalar b, Scalar c, Scalar d, Scalar e,
                                                      Scalar scale) {
  dst[0] = Scalar(Prefix) * scale;
  dst[1] = Scalar(Prefix == 0 ? 0 : Prefix + 1) * scale;
  dst[2] = Scalar(Prefix == 0 ? 0 : Prefix + 2) * scale;
  dst[3] = a * scale;
  dst[4] = b * scale;
  dst[5] = c * scale;
  dst[6] = d * scale;
  dst[7] = e * scale;
}

void check_scaled_zero_prefix() {
  float values[10], result[16];
  for (float& value : values) value = internal::random<float>(1, 2);
  const float scale = internal::random<float>(1, 2);
  scale_zero_prefixed_values(result, values[0], values[1], values[2], values[3], values[4], values[5], values[6],
                             values[7], values[8], values[9], scale);
  for (int i = 0; i < 16; ++i) VERIFY_IS_EQUAL(result[i], i < 6 ? 0.0f : values[i - 6] * scale);
}

template <typename Scalar, int Prefix>
void check_scaled_constant_prefix() {
  Scalar values[5], result[8];
  for (Scalar& value : values) value = internal::random<Scalar>(1, 2);
  const Scalar scale = internal::random<Scalar>(1, 2);
  scale_constant_prefixed_values<Scalar, Prefix>(result, values[0], values[1], values[2], values[3], values[4], scale);
  for (int i = 0; i < 8; ++i) {
    const Scalar expected = i < 3 ? Scalar(Prefix == 0 ? 0 : Prefix + i) : values[i - 3];
    VERIFY_IS_EQUAL(result[i], expected * scale);
  }
}

// The opaque barriers keep the store/reload from being constant-folded away; on a narrower
// machine the tail of the buffer then keeps its sentinel.
template <typename Scalar, typename Packet>
void check_full_width_store() {
  constexpr int PacketSize = internal::unpacket_traits<Packet>::size;
  EIGEN_ALIGN_MAX Scalar buffer[PacketSize];
  Scalar value = Scalar(1);

  Scalar* p = buffer;
  asm volatile("" : "+r"(p) : : "memory");
  for (int i = 0; i < PacketSize; ++i) p[i] = Scalar(-12345);
  asm volatile("" : "+r"(p), "+r"(value) : : "memory");

  internal::pstore(p, internal::pset1<Packet>(value));
  asm volatile("" : "+r"(p) : : "memory");

  for (int i = 0; i < PacketSize; ++i) {
    VERIFY_IS_EQUAL(p[i], Scalar(1));
  }
}

EIGEN_DECLARE_TEST(sve_vector_length) {
  CALL_SUBTEST(check_runtime_vector_length());
  CALL_SUBTEST(check_packet_sizes());
  CALL_SUBTEST(check_scaled_zero_prefix());
  CALL_SUBTEST((check_scaled_constant_prefix<float, 0>()));
  CALL_SUBTEST((check_scaled_constant_prefix<float, 1>()));
  CALL_SUBTEST((check_scaled_constant_prefix<double, 0>()));
  CALL_SUBTEST((check_scaled_constant_prefix<double, 1>()));
  CALL_SUBTEST((check_full_width_store<numext::int32_t, internal::PacketXi>()));
  CALL_SUBTEST((check_full_width_store<float, internal::PacketXf>()));
  CALL_SUBTEST((check_full_width_store<double, internal::PacketXd>()));
}
