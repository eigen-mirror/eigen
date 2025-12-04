// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template <typename Numerator, typename Divisor = Numerator>
void test_division() {
  using UnsignedNumerator = typename std::make_unsigned<Numerator>::type;
  using UnsignedDivisor = typename std::make_unsigned<Divisor>::type;
  using PlainType = VectorX<Numerator>;
  using FastDivOp = internal::fast_div_op<Numerator>;
  using FastDivXpr = CwiseUnaryOp<FastDivOp, PlainType>;
  using RefRhs = typename internal::plain_constant_type<PlainType>::type;
  using RefXpr = CwiseBinaryOp<internal::scalar_quotient_op<Numerator>, PlainType, RefRhs>;

  Index size = 4096;
  PlainType numerator(size);
  for (int repeat = 0; repeat < EIGEN_TEST_MAX_SIZE; repeat++) {
    numerator.setRandom();
    Divisor d = internal::random<Divisor>(1, NumTraits<Divisor>::highest() / 4);
    {
      FastDivXpr xpr(numerator, FastDivOp(d));
      for (Index i = 0; i < size; i++) {
        UnsignedNumerator absNumerator = numext::abs(numerator.coeff(i));
        Numerator ref = absNumerator / d;
        if (numerator.coeff(i) < 0) ref = -ref;
        VERIFY_IS_EQUAL(xpr.coeff(i), ref);
      }
    }
    if (std::is_signed<Divisor>::value) {
      Divisor neg_d = 0 - d;
      FastDivXpr xpr(numerator, FastDivOp(neg_d));
      for (Index i = 0; i < size; i++) {
        UnsignedNumerator absNumerator = numext::abs(numerator.coeff(i));
        Numerator ref = absNumerator / d;
        if (numerator.coeff(i) > 0) ref = -ref;
        VERIFY_IS_EQUAL(xpr.coeff(i), ref);
      }
    }
  }
}

template <typename Numerator, bool Signed = std::is_signed<Numerator>::value>
struct test_division_driver {
  static void run() {
    // divide unsigned numerators by unsigned divisors only
    test_division<Numerator, uint8_t>();
    test_division<Numerator, uint16_t>();
    test_division<Numerator, uint32_t>();
    test_division<Numerator, uint64_t>();
  }
};
template <typename Numerator>
struct test_division_driver<Numerator, true> {
  static void run() {
    // divide signed numerators by unsigned and signed divisors
    test_division_driver<Numerator, false>::run();
    test_division<Numerator, int8_t>();
    test_division<Numerator, int16_t>();
    test_division<Numerator, int32_t>();
    test_division<Numerator, int64_t>();
  }
};
template <typename Numerator>
void run_division_tests() {
  test_division_driver<Numerator>::run();
}

EIGEN_DECLARE_TEST(integer_division) {
  CALL_SUBTEST_1(run_division_tests<uint8_t>());
  CALL_SUBTEST_1(run_division_tests<uint16_t>());
  CALL_SUBTEST_1(run_division_tests<uint32_t>());
  CALL_SUBTEST_1(run_division_tests<uint64_t>());
  CALL_SUBTEST_1(run_division_tests<int8_t>());
  CALL_SUBTEST_1(run_division_tests<int16_t>());
  CALL_SUBTEST_1(run_division_tests<int32_t>());
  CALL_SUBTEST_1(run_division_tests<int64_t>());
}
