// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <cfenv>

namespace pnmsub_test {
template <int Mode>
struct Packet {
  float value;
  friend Packet operator+(Packet a, Packet b) { return {a.value + b.value}; }
  friend Packet operator-(Packet a, Packet b) { return {a.value - b.value}; }
  friend Packet operator*(Packet a, Packet b) { return {a.value * b.value}; }
  // The barrier keeps the generic pnmsub, -a * b + -c, from folding into -c - a * b, which rounds a * b before
  // negating it: GCC 10 and 11 emit vnmla for it on 32-bit Arm, even with -frounding-math.
  friend Packet operator-(Packet a) {
    float negated = -a.value;
    EIGEN_OPTIMIZATION_BARRIER(negated);
    return {negated};
  }
};

// The generic dispatch must not accidentally select an unrelated ADL overload.
inline Packet<1> pmadd(Packet<1>, Packet<1>, Packet<1>) { return {99}; }

struct SubtractOnly {
  unsigned value;
  friend SubtractOnly operator*(SubtractOnly a, SubtractOnly b) { return {a.value * b.value}; }
  friend SubtractOnly operator-(SubtractOnly a, SubtractOnly b) { return {a.value - b.value}; }
};

// Provides exactly pmadd and pnegate, which is all that the generic pnmsub has always required.
struct MaddNegateOnly {
  float value;
};
}  // namespace pnmsub_test

namespace Eigen {
template <>
struct NumTraits<pnmsub_test::Packet<3>> : GenericNumTraits<pnmsub_test::Packet<3>> {
  static constexpr bool IsSigned = true;
};
template <>
struct NumTraits<pnmsub_test::SubtractOnly> : GenericNumTraits<pnmsub_test::SubtractOnly> {
  static constexpr bool IsSigned = false;
};
namespace internal {
template <int Mode>
struct unpacket_traits<pnmsub_test::Packet<Mode>> : unpacket_traits<float> {};

template <>
struct unpacket_traits<pnmsub_test::Packet<3>> : unpacket_traits<float> {
  using type = pnmsub_test::Packet<3>;
};

template <>
struct unpacket_traits<pnmsub_test::Packet<4>> : unpacket_traits<uint32_t> {};

template <>
struct unpacket_traits<pnmsub_test::MaddNegateOnly> : unpacket_traits<float> {};
template <>
EIGEN_STRONG_INLINE pnmsub_test::MaddNegateOnly pmadd(const pnmsub_test::MaddNegateOnly& a,
                                                      const pnmsub_test::MaddNegateOnly& b,
                                                      const pnmsub_test::MaddNegateOnly& c) {
  return {a.value * b.value + c.value};
}
template <>
EIGEN_STRONG_INLINE pnmsub_test::MaddNegateOnly pnegate(const pnmsub_test::MaddNegateOnly& a) {
  return {-a.value};
}

template <>
struct is_fused_madd<pnmsub_test::Packet<1>> : true_type {};
template <>
EIGEN_STRONG_INLINE pnmsub_test::Packet<1> pmadd(const pnmsub_test::Packet<1>& a, const pnmsub_test::Packet<1>& b,
                                                 const pnmsub_test::Packet<1>& c) {
  return {numext::fma(a.value, b.value, c.value)};
}

// A standalone pmadd specialization does not prove fusion to the generic dispatch.
template <>
EIGEN_STRONG_INLINE pnmsub_test::Packet<5> pmadd(const pnmsub_test::Packet<5>& a, const pnmsub_test::Packet<5>& b,
                                                 const pnmsub_test::Packet<5>& c) {
  return {numext::fma(a.value, b.value, c.value)};
}

template <>
EIGEN_STRONG_INLINE pnmsub_test::Packet<4> pmadd(const pnmsub_test::Packet<4>&, const pnmsub_test::Packet<4>&,
                                                 const pnmsub_test::Packet<4>&) {
  return {99};
}

template <>
struct pmadd_impl<pnmsub_test::Packet<2>> {
  static pnmsub_test::Packet<2> pnmadd(const pnmsub_test::Packet<2>&, const pnmsub_test::Packet<2>&,
                                       const pnmsub_test::Packet<2>&) {
    return {7};
  }
  static pnmsub_test::Packet<2> pnmsub(const pnmsub_test::Packet<2>&, const pnmsub_test::Packet<2>&,
                                       const pnmsub_test::Packet<2>&) {
    return {7};
  }
};
}  // namespace internal
}  // namespace Eigen

template <typename Packet>
EIGEN_DONT_INLINE float generic_pnmsub_value(float a, float b, float c) {
  return internal::pnmsub(Packet{a}, Packet{b}, Packet{c}).value;
}

void generic_pnmsub_dispatch() {
  using Fused = pnmsub_test::Packet<1>;
  using Unfused = pnmsub_test::Packet<0>;
  STATIC_CHECK((!internal::is_scalar<Fused>::value));
  const float delta = numext::ldexp(1.0f, -13);
  const auto fused = internal::pnmsub(Fused{1 + delta}, Fused{1 - delta}, Fused{-1});
  VERIFY_IS_EQUAL(fused.value, delta * delta);
  const auto unfused = internal::pnmsub(Unfused{0}, Unfused{1}, Unfused{0});
  VERIFY_IS_EQUAL(unfused.value, 0.0f);
  VERIFY(std::signbit(unfused.value));
  using Custom = pnmsub_test::Packet<2>;
  VERIFY_IS_EQUAL(internal::pnmsub(Custom{1}, Custom{2}, Custom{3}).value, 7.0f);
  using Scalar = pnmsub_test::Packet<3>;
  STATIC_CHECK((internal::is_scalar<Scalar>::value));
  VERIFY_IS_EQUAL(internal::pnmsub(Scalar{2}, Scalar{3}, Scalar{4}).value, -10.0f);
  VERIFY_IS_EQUAL(internal::pnmsub(2, 3, 4), -10);
  VERIFY_IS_EQUAL(internal::pnmsub(2u, 3u, 4u), (std::numeric_limits<unsigned>::max)() - 9u);
}

// Negating an operand of a signed integer would overflow here, though the result, -1, is representable. The operands
// are read through volatile, so that the compiler cannot fold the call away.
void integer_pnmsub_boundary() {
  const volatile int int_min = (std::numeric_limits<int>::min)();
  const volatile int zero = 0;
  VERIFY_IS_EQUAL(internal::pnmsub(int(int_min), int(zero), 1), -1);
  const volatile int64_t int64_min = (std::numeric_limits<int64_t>::min)();
  VERIFY_IS_EQUAL(internal::pnmsub(int64_t(int64_min), int64_t(zero), int64_t(1)), int64_t(-1));
}

// A packet that provides pmadd and pnegate, and nothing else the generic pnmsub could use, must keep compiling.
void generic_pnmsub_minimal_operations() {
  using Minimal = pnmsub_test::MaddNegateOnly;
  VERIFY_IS_EQUAL(internal::pnmsub(Minimal{1}, Minimal{2}, Minimal{3}).value, -5.0f);
}

template <typename Op>
void verify_directed_roundings(const Op& op, float expected_upward, float expected_downward) {
  std::fenv_t environment;
  if (std::fegetenv(&environment) != 0) return;
  for (int mode : {FE_UPWARD, FE_DOWNWARD}) {
    if (std::fesetround(mode) != 0) continue;
    const float result = op();
    std::fesetenv(&environment);
    VERIFY_IS_EQUAL(result, mode == FE_UPWARD ? expected_upward : expected_downward);
  }
}

// Without fusion, pnmsub is pmadd(-a, b, -c) in every rounding mode: -a * b rounds the negated product, which is not
// the negation of the rounded product when the mode is directed. (1 + e)^2 = 1 + 2e + e^2 lies between 1 + 2e and
// 1 + 3e, with e the epsilon of float.
void generic_pnmsub_directed_rounding() {
  using Unfused = pnmsub_test::Packet<0>;
  const float e = NumTraits<float>::epsilon();
  verify_directed_roundings([e] { return generic_pnmsub_value<Unfused>(1 + e, 1 + e, 0); }, -(1 + 2 * e), -(1 + 3 * e));
  verify_directed_roundings([e] { return generic_pnmsub_value<Unfused>(1 + e, 1 + e, -0.0f); }, -(1 + 2 * e),
                            -(1 + 3 * e));
}

template <typename Packet>
EIGEN_DONT_INLINE float generic_pmsub_value(float a, float b, float c) {
  return internal::pmsub(Packet{a}, Packet{b}, Packet{c}).value;
}

void generic_pmsub_dispatch() {
  using Fused = pnmsub_test::Packet<1>;
  using Unfused = pnmsub_test::Packet<0>;
  using Scalar = pnmsub_test::SubtractOnly;
  using IntegerPacket = pnmsub_test::Packet<4>;
  STATIC_CHECK((!internal::is_scalar<IntegerPacket>::value));
  STATIC_CHECK((!internal::is_fused_madd<IntegerPacket>::value));
  STATIC_CHECK((!internal::is_fused_madd<Unfused>::value));
  STATIC_CHECK((internal::is_fused_madd<Fused>::value));
  VERIFY_IS_EQUAL(generic_pmsub_value<IntegerPacket>(2, 3, 4), 2.0f);
  STATIC_CHECK((internal::is_scalar<Scalar>::value));
  VERIFY_IS_EQUAL(internal::pmsub(Scalar{2}, Scalar{3}, Scalar{4}).value, 2u);
  const float delta = numext::ldexp(1.0f, -13);
  VERIFY_IS_EQUAL(generic_pmsub_value<Fused>(1 + delta, 1 - delta, 1), -delta * delta);
  VERIFY_IS_EQUAL(generic_pmsub_value<Unfused>(2, 3, 4), 2.0f);
  for (float a : {0.0f, -0.0f}) {
    for (float b : {1.0f, -1.0f}) {
      for (float c : {0.0f, -0.0f}) {
        const float expected = a * b - c;
        const float actual = generic_pmsub_value<Fused>(a, b, c);
        VERIFY_IS_EQUAL(actual, expected);
        VERIFY_IS_EQUAL(std::signbit(actual), std::signbit(expected));
      }
    }
  }
  verify_directed_roundings([] { return generic_pmsub_value<Fused>(1, 1, -NumTraits<float>::epsilon() / 4); },
                            1 + NumTraits<float>::epsilon(), 1.0f);
}

template <typename Packet>
EIGEN_DONT_INLINE float generic_pnmadd_value(float a, float b, float c) {
  return internal::pnmadd(Packet{a}, Packet{b}, Packet{c}).value;
}

void generic_pnmadd_dispatch() {
  using Fused = pnmsub_test::Packet<1>;
  using Unfused = pnmsub_test::Packet<0>;
  using Unknown = pnmsub_test::Packet<5>;
  using Scalar = pnmsub_test::SubtractOnly;
  using Custom = pnmsub_test::Packet<2>;
  VERIFY_IS_EQUAL(internal::pnmadd(Scalar{2}, Scalar{3}, Scalar{4}).value, (std::numeric_limits<unsigned>::max)() - 1u);
  VERIFY_IS_EQUAL(internal::pnmadd(Custom{2}, Custom{3}, Custom{4}).value, 7.0f);
  VERIFY_IS_EQUAL(generic_pnmadd_value<pnmsub_test::Packet<4>>(2, 3, 4), -2.0f);
  const float delta = numext::ldexp(1.0f, -13);
  STATIC_CHECK((internal::is_fused_madd<Fused>::value));
  STATIC_CHECK((!internal::is_fused_madd<Unfused>::value));
  STATIC_CHECK((!internal::is_fused_madd<Unknown>::value));
  VERIFY_IS_EQUAL(generic_pnmadd_value<Unfused>(1 + delta, 1 - delta, 1), 0.0f);
  VERIFY_IS_EQUAL(generic_pnmadd_value<Unknown>(1 + delta, 1 - delta, 1), 0.0f);
  VERIFY_IS_EQUAL(generic_pnmadd_value<Fused>(1 + delta, 1 - delta, 1), delta * delta);
  const float infinity = NumTraits<float>::infinity();
  const float largest = (std::numeric_limits<float>::max)();
  const float smallest = (std::numeric_limits<float>::denorm_min)();
  const float values[] = {0.0f,
                          -0.0f,
                          1.0f,
                          -1.0f,
                          2.0f,
                          -2.0f,
                          largest,
                          -largest,
                          smallest,
                          -smallest,
                          infinity,
                          -infinity,
                          NumTraits<float>::quiet_NaN()};
  for (float a : values) {
    for (float b : values) {
      for (float c : values) {
        const float expected = numext::fma(-a, b, c);
        const float actual = generic_pnmadd_value<Fused>(a, b, c);
        if ((numext::isnan)(expected)) {
          VERIFY((numext::isnan)(actual));
        } else {
          VERIFY_IS_EQUAL(actual, expected);
          if (actual == 0) VERIFY_IS_EQUAL(std::signbit(actual), std::signbit(expected));
        }
      }
    }
  }
  verify_directed_roundings([] { return generic_pnmadd_value<Fused>(1, 1, NumTraits<float>::epsilon() / 4); },
                            -1 + NumTraits<float>::epsilon() / 2, -1.0f);
}

template <typename Scalar, typename Packet>
void integer_packet_subtract() {
  constexpr int Size = internal::unpacket_traits<Packet>::size;
  Scalar values[Size], result[Size];
  const Scalar largest = (std::numeric_limits<Scalar>::max)();
  const Scalar cases[] = {Scalar(0), Scalar(1), Scalar(largest / 2), Scalar(largest / 2 + 1), largest};
  for (int offset = 0; offset < 5; ++offset) {
    for (int i = 0; i < Size; ++i) values[i] = cases[(i + offset) % 5];
    const Packet input = internal::ploadu<Packet>(values);
    internal::pstoreu(result, internal::pmsub(input, input, input));
    for (int i = 0; i < Size; ++i) {
      const uint64_t value = values[i];
      VERIFY_IS_EQUAL(result[i], Scalar(value * value - value));
    }
  }
}

template <typename Scalar>
void integer_packet_subtract() {
  integer_packet_subtract<Scalar, typename internal::packet_traits<Scalar>::type>();
  integer_packet_subtract<Scalar, typename internal::packet_traits<Scalar>::half>();
}

EIGEN_DECLARE_TEST(packet_madd_dispatch) {
  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    CALL_SUBTEST_1(generic_pnmsub_dispatch());
    CALL_SUBTEST_1(integer_pnmsub_boundary());
    CALL_SUBTEST_1(generic_pnmsub_minimal_operations());
    CALL_SUBTEST_1(generic_pnmsub_directed_rounding());
    CALL_SUBTEST_1(generic_pmsub_dispatch());
    CALL_SUBTEST_1(generic_pnmadd_dispatch());
    CALL_SUBTEST_1(integer_packet_subtract<uint8_t>());
    CALL_SUBTEST_1(integer_packet_subtract<uint16_t>());
    CALL_SUBTEST_1(integer_packet_subtract<uint32_t>());
    CALL_SUBTEST_1(integer_packet_subtract<uint64_t>());
  }
}
