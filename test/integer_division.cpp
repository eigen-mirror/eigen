// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template <typename Numerator, typename Divisor>
Numerator ref_div(Numerator n, Divisor d) {
  EIGEN_STATIC_ASSERT(std::is_signed<Numerator>::value || !std::is_signed<Divisor>::value,
                      CANT DIVIDE AN UNSIGNED INTEGER BY A SIGNED INTEGER)
  using UnsignedNumerator = typename std::make_unsigned<Numerator>::type;
  using UnsignedDivisor = typename std::make_unsigned<Divisor>::type;
  bool n_is_negative = n < 0;
  bool d_is_negative = d < 0;
  UnsignedNumerator abs_n = numext::abs(n);
  UnsignedDivisor abs_d = numext::abs(d);
  Numerator result = static_cast<Numerator>(abs_n / abs_d);
  if (n_is_negative != d_is_negative) {
    result = 0 - result;
  }
  return result;
}

template <typename Numerator, typename Divisor>
void test_division_exhaustive() {
  Numerator n = NumTraits<Numerator>::lowest();
  Divisor d = NumTraits<Divisor>::lowest();
  while (true) {
    if (d != 0) {
      internal::fast_div_op<Numerator> fast_div(d);
      while (true) {
        Numerator q = fast_div.operator()(n);
        Numerator ref = ref_div(n, d);
        VERIFY_IS_EQUAL(q, ref);
        if (n == NumTraits<Numerator>::highest()) break;
        n++;
      }
    }
    if (d == NumTraits<Divisor>::highest()) break;
    d++;
  }
}

template <typename Numerator, typename Divisor>
void test_division() {
  using PlainType = VectorX<Numerator>;
  using FastDivOp = internal::fast_div_op<Numerator>;
  using FastDivXpr = CwiseUnaryOp<FastDivOp, PlainType>;

  Index size = 4096;
  PlainType numerator(size);
  for (int repeat = 0; repeat < EIGEN_TEST_MAX_SIZE; repeat++) {
    numerator.setRandom();
    Divisor d = internal::random<Divisor>(1, NumTraits<Divisor>::highest() / 4);
    {
      FastDivXpr xpr(numerator, FastDivOp(d));
      PlainType evalXpr = xpr;
      for (Index i = 0; i < size; i++) {
        Numerator ref = ref_div(numerator.coeff(i), d);
        VERIFY_IS_EQUAL(evalXpr.coeff(i), ref);
      }
    }
    if (std::is_signed<Divisor>::value) {
      Divisor neg_d = 0 - d;
      FastDivXpr xpr(numerator, FastDivOp(neg_d));
      for (Index i = 0; i < size; i++) {
        Numerator ref = ref_div(numerator.coeff(i), neg_d);
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
  CALL_SUBTEST_1((test_division_exhaustive<uint8_t, uint8_t>()));
  CALL_SUBTEST_1((test_division_exhaustive<int8_t, int8_t>()));
  CALL_SUBTEST_1((test_division_exhaustive<int8_t, uint8_t>()));
  CALL_SUBTEST_2((test_division_exhaustive<uint16_t, uint16_t>()));
  CALL_SUBTEST_2((test_division_exhaustive<int16_t, int16_t>()));
  CALL_SUBTEST_2((test_division_exhaustive<int16_t, uint16_t>()));
  CALL_SUBTEST_3((run_division_tests<uint8_t>()));
  CALL_SUBTEST_3((run_division_tests<uint16_t>()));
  CALL_SUBTEST_4((run_division_tests<uint32_t>()));
  CALL_SUBTEST_4((run_division_tests<uint64_t>()));
  CALL_SUBTEST_5((run_division_tests<int8_t>()));
  CALL_SUBTEST_5((run_division_tests<int16_t>()));
  CALL_SUBTEST_6((run_division_tests<int32_t>()));
  CALL_SUBTEST_6((run_division_tests<int64_t>()));
}
