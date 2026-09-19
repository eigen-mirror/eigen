// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include "twoprod_helpers.h"

template <typename Scalar, typename Packet>
void check_doubleword_product(const Scalar (&data)[7]) {
  constexpr Index size = internal::unpacket_traits<Packet>::size;
  EIGEN_ALIGN_MAX Scalar hi[size], lo[size];
  const Scalar epsilon = NumTraits<Scalar>::epsilon();
  const Scalar bound = Scalar(1.25) * epsilon * epsilon * numext::abs(data[4]);  // 5*u^2, u = eps/2.
  for (int exponent : {-20, 0, 20}) {
    // Keep the operands runtime so constant folding cannot replace the arithmetic under test.
    volatile Scalar x_scale = numext::ldexp(Scalar(1), exponent), y_scale = numext::ldexp(Scalar(1), -exponent);
    for (int sign : {-1, 1}) {
      const Scalar x_hi = Scalar(sign) * data[0] * x_scale;
      const Scalar x_lo = Scalar(sign) * data[1] * x_scale;
      const Scalar y_hi = data[2] * y_scale;
      const Scalar y_lo = data[3] * y_scale;
      VERIFY_IS_EQUAL(x_hi + x_lo, x_hi);
      VERIFY_IS_EQUAL(y_hi + y_lo, y_hi);
      Packet p_hi, p_lo;
      internal::twoprod(internal::pset1<Packet>(x_hi), internal::pset1<Packet>(x_lo), internal::pset1<Packet>(y_hi),
                        internal::pset1<Packet>(y_lo), p_hi, p_lo);
      internal::pstore(hi, p_hi);
      internal::pstore(lo, p_lo);
      for (Index i = 0; i < size; ++i) {
        VERIFY_IS_EQUAL(hi[i], Scalar(sign) * data[4]);
        const Scalar error = numext::abs((lo[i] - Scalar(sign) * data[5]) - Scalar(sign) * data[6]);
        VERIFY(error < bound);
      }
    }
  }
}

EIGEN_DECLARE_TEST(twoprod) {
  using FloatPacket = internal::packet_traits<float>::type;
  using DoublePacket = internal::packet_traits<double>::type;
  check_twoprod_packet<float>();
  check_twoprod_packet<double>();
  check_twoprod_packet<FloatPacket>();
  check_twoprod_packet<DoublePacket>();

  // {x_hi, x_lo, y_hi, y_lo, ref_hi, ref_lo, ref_tail}. The reference product is
  // decomposed with 600-bit MPFR, rounding each word to the tested scalar type.
  // These normalized inputs exceed the former 2*u^2 bound in both precisions.
  const float float_data[] = {1.1675887107849121f,     -3.9186936362511915e-08f, 1.7997856140136719f,
                              5.2592689314678864e-08f, 2.1014094352722168f,      -7.963821957446271e-08f,
                              1.2973732602550997e-15f};
  const double double_data[] = {1.4946183601517316,      9.493375847471864e-17, 1.4603476327002127,
                                -8.3856771222625945e-17, 2.182662384037855,     1.9088477361435543e-16,
                                -3.7615759087187478e-33};
  check_doubleword_product<float, float>(float_data);
  check_doubleword_product<double, double>(double_data);
  check_doubleword_product<float, FloatPacket>(float_data);
  check_doubleword_product<double, DoublePacket>(double_data);
}
