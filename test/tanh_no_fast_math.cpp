// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// EIGEN_FAST_MATH=0 selects the accurate float tanh kernel, which backends such as the generic clang one vectorize
// whatever the setting.
#define EIGEN_FAST_MATH 0

#include "main.h"

#include <cfenv>
#include <cstring>

namespace {

numext::int64_t float_ulp_distance(float a, float b) {
  numext::int32_t ia, ib;
  std::memcpy(&ia, &a, sizeof(float));
  std::memcpy(&ib, &b, sizeof(float));
  // Map the sign-magnitude bit patterns onto a monotonic integer line.
  const numext::int64_t la = ia < 0 ? numext::int64_t(INT32_MIN) - ia : ia;
  const numext::int64_t lb = ib < 0 ? numext::int64_t(INT32_MIN) - ib : ib;
  return la > lb ? la - lb : lb - la;
}

template <typename Packet>
void tanh_float_accuracy() {
  constexpr int PacketSize = internal::unpacket_traits<Packet>::size;
  // Every third float in [0.5, 10], where the fast kernel is up to 6 ulps off; the accurate one is within 2.
  const numext::uint32_t first = 0x3f000000u, last = 0x41200000u;
  float x[PacketSize], y[PacketSize];
  numext::int64_t max_error = 0;
  float worst = 0;
  for (numext::uint32_t bits = first; bits < last;) {
    for (int k = 0; k < PacketSize; ++k, bits += 3) {
      std::memcpy(&x[k], &bits, sizeof(float));
      if (k % 2 == 1) x[k] = -x[k];
    }
    internal::pstoreu(y, internal::ptanh_float(internal::ploadu<Packet>(x)));
    for (int k = 0; k < PacketSize; ++k) {
      const numext::int64_t error = float_ulp_distance(y[k], static_cast<float>(std::tanh(double(x[k]))));
      if (error > max_error) {
        max_error = error;
        worst = x[k];
      }
    }
  }
  if (max_error > 2) std::cout << "tanh(" << worst << ") is " << max_error << " ulps off" << std::endl;
  VERIFY(max_error <= 2);

  const float inf = NumTraits<float>::infinity();
  const float special[] = {0.0f, -0.0f, inf, -inf, NumTraits<float>::quiet_NaN(), 20.0f, -20.0f, 1e-30f};
  for (const float s : special) {
    for (int k = 0; k < PacketSize; ++k) x[k] = s;
    internal::pstoreu(y, internal::ptanh_float(internal::ploadu<Packet>(x)));
    const float expected = std::tanh(s);
    const bool success = ((numext::isnan)(expected) && (numext::isnan)(y[0])) ||
                         (y[0] == expected && std::signbit(y[0]) == std::signbit(expected));
    if (!success) std::cout << "tanh(" << s << ") = " << y[0] << " != " << expected << std::endl;
    VERIFY(success);
  }
}

template <typename Packet>
Packet tanh_kernel(const Packet& x, float) {
  return internal::ptanh_float(x);
}
template <typename Packet>
Packet tanh_kernel(const Packet& x, double) {
  return internal::ptanh_double(x);
}

// The FE_OVERFLOW, FE_INVALID and FE_DIVBYZERO flags the tanh kernel raises on the packet in, with the result in out.
// The operands and results go through volatile copies after feholdexcept and before fetestexcept, which keeps the
// compiler from folding the kernel or moving it across the flag accesses.
template <typename Packet, typename Scalar>
int tanh_exceptions(const Scalar* in, Scalar* out) {
  constexpr int PacketSize = internal::unpacket_traits<Packet>::size;
  const volatile Scalar* volatile_in = in;
  volatile Scalar* volatile_out = out;
  Scalar x[PacketSize], y[PacketSize];
  std::fenv_t saved_environment;
  if (std::feholdexcept(&saved_environment) != 0) return 0;
  for (int k = 0; k < PacketSize; ++k) x[k] = volatile_in[k];
  internal::pstoreu(y, tanh_kernel(internal::ploadu<Packet>(x), Scalar()));
  for (int k = 0; k < PacketSize; ++k) volatile_out[k] = y[k];
  const int raised = std::fetestexcept(FE_OVERFLOW | FE_INVALID | FE_DIVBYZERO);
  std::fesetenv(&saved_environment);
  return raised;
}

// tanh of large and infinite arguments is +-1 and raises no FE_OVERFLOW, FE_INVALID or FE_DIVBYZERO. A large element
// beside a NaN raises nothing that a small one does not: the NaN's own comparison may raise FE_INVALID.
template <typename Packet>
void tanh_saturation_exceptions() {
  using Scalar = typename internal::unpacket_traits<Packet>::type;
  constexpr int PacketSize = internal::unpacket_traits<Packet>::size;
  const Scalar inf = NumTraits<Scalar>::infinity();
  const Scalar values[] = {Scalar(20), Scalar(100), Scalar(1e20f), NumTraits<Scalar>::highest(), inf};
  Scalar x[PacketSize], y[PacketSize];
  for (const Scalar value : values) {
    for (const Scalar s : {value, -value}) {
      for (int k = 0; k < PacketSize; ++k) x[k] = s;
      const int raised = tanh_exceptions<Packet>(x, y);
      if (raised != 0) std::cout << "tanh(" << s << ") raised " << raised << std::endl;
      VERIFY_IS_EQUAL(raised, 0);
      for (int k = 0; k < PacketSize; ++k) VERIFY_IS_EQUAL(y[k], s > 0 ? Scalar(1) : Scalar(-1));
    }
  }
  if (PacketSize > 1) {
    int raised[2];
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < PacketSize; ++k) x[k] = k == 0 ? NumTraits<Scalar>::quiet_NaN() : Scalar(j == 0 ? 0.5 : 100);
      raised[j] = tanh_exceptions<Packet>(x, y);
      VERIFY((numext::isnan)(y[0]));
    }
    VERIFY_IS_EQUAL(raised[1] & ~raised[0], 0);
  }
}

}  // namespace

EIGEN_DECLARE_TEST(tanh_no_fast_math) {
  CALL_SUBTEST(tanh_float_accuracy<internal::packet_traits<float>::type>());
  CALL_SUBTEST(tanh_float_accuracy<float>());
  CALL_SUBTEST(tanh_saturation_exceptions<internal::packet_traits<float>::type>());
  CALL_SUBTEST(tanh_saturation_exceptions<float>());
  CALL_SUBTEST(tanh_saturation_exceptions<internal::packet_traits<double>::type>());
  CALL_SUBTEST(tanh_saturation_exceptions<double>());
}
