// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include "twoprod_helpers.h"

// Checks twoprod, or fast_twoprod if `fast` is set, on {x_hi, x_lo, y_hi, y_lo, ref_hi, ref_lo, ref_tail}.
template <typename Scalar, typename Packet>
void check_doubleword_product(const Scalar (&data)[7], bool fast = false) {
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
      const Packet px_hi = internal::pset1<Packet>(x_hi), px_lo = internal::pset1<Packet>(x_lo);
      const Packet py_hi = internal::pset1<Packet>(y_hi), py_lo = internal::pset1<Packet>(y_lo);
      if (fast) {
        internal::fast_twoprod(px_hi, px_lo, py_hi, py_lo, p_hi, p_lo);
      } else {
        internal::twoprod(px_hi, px_lo, py_hi, py_lo, p_hi, p_lo);
      }
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

// Where scalar and packet pmadd are both fused, fast_twoprod takes the same path for both, so int_pow's scalar and
// vectorized paths round alike.
template <typename Scalar, typename Packet>
void check_fast_twoprod_scalar_matches_packet(const Scalar (&data)[7]) {
#ifdef EIGEN_VECTORIZE_FMA
  if (!internal::is_fused_madd<Packet>::value) return;
  constexpr Index size = internal::unpacket_traits<Packet>::size;
  EIGEN_ALIGN_MAX Scalar hi[size], lo[size];
  // Keep the operands runtime so constant folding cannot replace the arithmetic under test.
  volatile Scalar operands[4] = {data[0], data[1], data[2], data[3]};
  const Scalar x_hi = operands[0], x_lo = operands[1], y_hi = operands[2], y_lo = operands[3];
  Scalar s_hi, s_lo;
  internal::fast_twoprod<Scalar>(x_hi, x_lo, y_hi, y_lo, s_hi, s_lo);
  const Packet px_hi = internal::pset1<Packet>(x_hi), px_lo = internal::pset1<Packet>(x_lo);
  const Packet py_hi = internal::pset1<Packet>(y_hi), py_lo = internal::pset1<Packet>(y_lo);
  Packet p_hi, p_lo;
  internal::fast_twoprod(px_hi, px_lo, py_hi, py_lo, p_hi, p_lo);
  internal::pstore(hi, p_hi);
  internal::pstore(lo, p_lo);
  for (Index i = 0; i < size; ++i) {
    VERIFY_IS_EQUAL(hi[i], s_hi);
    VERIFY_IS_EQUAL(lo[i], s_lo);
  }
#else
  EIGEN_UNUSED_VARIABLE(data);
#endif
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

  // Worst cases found by search for fast_twoprod: a chain of fused pmadds, with error up to 5*u^2, and (where pmadd
  // is not fused) Algorithm 10 of Joldes, Muller, & Popescu, which stays below 5*u^2 on these operands. An unfused
  // pmadd chain reaches 6.9*u^2 on the "unfused" operands. References are decomposed as above.
  const float fast_float_data[][7] = {
      // fused general.
      {-0.250000864f, -1.48879558e-08f, -0.508609235f, -2.97569827e-08f, 0.127152771f, -7.35021777e-09f,
       -4.21284906e-17f},
      // fused square.
      {-1.00657368f, -5.95824865e-08f, -1.00657368f, -5.95824865e-08f, 1.01319075f, -5.90029394e-08f, -2.8103879e-18f},
      // unfused general.
      {8.21313381f, 4.76301125e-07f, 1.00737178f, 5.95087641e-08f, 8.27368069f, -4.64115629e-07f, -1.17886943e-15f},
      // unfused square.
      {-2.0135932f, 1.19185245e-07f, -2.0135932f, 1.19185245e-07f, 4.05455685f, 2.35876868e-07f, -1.45683893e-17f},
      // Algorithm 10 general.
      {-1.00040495f, -5.96022929e-08f, 0.0625946224f, 3.72474185e-09f, -0.0626199767f, -1.07125442e-09f,
       5.91333584e-19f},
      // Algorithm 10 square.
      {8.00068283f, 4.76670664e-07f, 8.00068283f, 4.76670664e-07f, 64.0109329f, 4.64245204e-07f, -1.60054737e-16f},
  };
  const double fast_double_data[][7] = {
      // fused general.
      {0.25002041616236204, 2.7754358076265322e-17, 1.0410038358637521, 1.1094124097677933e-16, 0.26027221226927061,
       -2.7065762180931695e-17, -5.4781762140699477e-36},
      // fused square.
      {-4.0061437100749089, -4.4397637134168864e-16, -4.0061437100749089, -4.4397637134168864e-16, 16.049187425772761,
       -1.7728980890933963e-15, -1.3739735310886058e-34},
      // unfused general.
      {-8.8249679712992339, -8.8761466869110876e-16, 0.062562782408520978, 6.915930645409851e-18, -0.55211455095056094,
       4.9974548464202032e-17, 8.7122901865361381e-35},
      // unfused square.
      {2.0092063436530401, 2.2171312984987409e-16, 2.0092063436530401, 2.2171312984987409e-16, 4.0369101313756195,
       -4.417540122138019e-16, -2.2355717681196923e-34},
      // Algorithm 10 general.
      {-16.239637864125285, 1.7758716492754939e-15, 16.01732882694828, -1.7740284508235665e-15, -260.11561970025468,
       2.0571644619504938e-14, 8.9404545263427861e-33},
      // Algorithm 10 square.
      {16.000001245592074, 1.7762960189520815e-15, 16.000001245592074, 1.7762960189520815e-15, 256.00003985894796,
       1.6725363562404628e-14, -2.1609212338112685e-34},
  };
  for (const auto& data : fast_float_data) {
    check_doubleword_product<float, float>(data, true);
    check_doubleword_product<float, FloatPacket>(data, true);
    check_fast_twoprod_scalar_matches_packet<float, FloatPacket>(data);
  }
  for (const auto& data : fast_double_data) {
    check_doubleword_product<double, double>(data, true);
    check_doubleword_product<double, DoublePacket>(data, true);
    check_fast_twoprod_scalar_matches_packet<double, DoublePacket>(data);
  }
}
