// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "packetmath_test_shared.h"
#include "trig_reduction_cases.h"

void check_trig_reduction() {
  const Index count = test::kNumTrigReductionCases;
  const auto& cases = test::kTrigReductionCases;
  ArrayXd x(2 * count), sin_ref(2 * count), cos_ref(2 * count), tan_ref(2 * count);
  for (Index i = 0; i < count; ++i) {
    x[2 * i] = cases[i][0];
    x[2 * i + 1] = -cases[i][0];
    sin_ref[2 * i] = cases[i][1];
    sin_ref[2 * i + 1] = -cases[i][1];
    cos_ref[2 * i] = cos_ref[2 * i + 1] = cases[i][2];
    tan_ref[2 * i] = cases[i][3];
    tan_ref[2 * i + 1] = -cases[i][3];
  }
  for (Index offset = 0; offset < 2; ++offset) {
    for (Index size = 1; size <= x.size() - offset; ++size) {
      const ArrayXd sine = x.segment(offset, size).sin();
      const ArrayXd cosine = x.segment(offset, size).cos();
      const ArrayXd tangent = x.segment(offset, size).tan();
      VERIFY(test::areWithinUlps(sin_ref.data() + offset, sine.data(), static_cast<int>(size), 4));
      VERIFY(test::areWithinUlps(cos_ref.data() + offset, cosine.data(), static_cast<int>(size), 4));
      VERIFY(test::areWithinUlps(tan_ref.data() + offset, tangent.data(), static_cast<int>(size), 4));
    }
  }
}

template <typename Packet>
void check_error_free_transformations() {
  const double low = std::ldexp(1.0, -54);
  Packet hi, lo;
  internal::twosum(internal::pset1<Packet>(low), internal::pset1<Packet>(1.0), hi, lo);
  VERIFY_IS_EQUAL(internal::pfirst(hi), 1.0);
  VERIFY_IS_EQUAL(internal::pfirst(lo), low);
  const Packet x = internal::pset1<Packet>(1.0 + std::ldexp(1.0, -30));
  internal::twoprod<Packet, true>(x, x, hi, lo);
  VERIFY_IS_EQUAL(internal::pfirst(hi), 1.0 + std::ldexp(1.0, -29));
  VERIFY_IS_EQUAL(internal::pfirst(lo), std::ldexp(1.0, -60));
}

EIGEN_DECLARE_TEST(trig_reduction_fastmath) {
  CALL_SUBTEST(check_trig_reduction());
  CALL_SUBTEST(check_error_free_transformations<double>());
  CALL_SUBTEST(check_error_free_transformations<internal::packet_traits<double>::type>());
}
