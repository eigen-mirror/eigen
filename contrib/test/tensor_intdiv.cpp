// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014-2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include <Eigen/Tensor>

void test_signed_32bit() {
  // Divide by one
  const Eigen::internal::TensorIntDivisor<int32_t, false> div_by_one(1);

  for (int32_t j = 0; j < 25000; ++j) {
    const int32_t fast_div = j / div_by_one;
    const int32_t slow_div = j / 1;
    VERIFY_IS_EQUAL(fast_div, slow_div);
  }

  // Standard divide by 2 or more
  for (int32_t i = 2; i < 25000; ++i) {
    const Eigen::internal::TensorIntDivisor<int32_t, false> div(i);

    for (int32_t j = 0; j < 25000; ++j) {
      const int32_t fast_div = j / div;
      const int32_t slow_div = j / i;
      VERIFY_IS_EQUAL(fast_div, slow_div);
    }
  }

  // The div_gt_one argument is accepted and ignored.
  for (int32_t i = 2; i < 25000; ++i) {
    const Eigen::internal::TensorIntDivisor<int32_t, true> div(i);

    for (int32_t j = 0; j < 25000; ++j) {
      const int32_t fast_div = j / div;
      const int32_t slow_div = j / i;
      VERIFY_IS_EQUAL(fast_div, slow_div);
    }
  }
}

void test_unsigned_32bit() {
  for (uint32_t i = 1; i < 25000; ++i) {
    const Eigen::internal::TensorIntDivisor<uint32_t> div(i);

    for (uint32_t j = 0; j < 25000; ++j) {
      const uint32_t fast_div = j / div;
      const uint32_t slow_div = j / i;
      VERIFY_IS_EQUAL(fast_div, slow_div);
    }
  }
}

void test_signed_64bit() {
  for (int64_t i = 1; i < 25000; ++i) {
    const Eigen::internal::TensorIntDivisor<int64_t> div(i);

    for (int64_t j = 0; j < 25000; ++j) {
      const int64_t fast_div = j / div;
      const int64_t slow_div = j / i;
      VERIFY_IS_EQUAL(fast_div, slow_div);
    }
  }
}

void test_unsigned_64bit() {
  for (uint64_t i = 1; i < 25000; ++i) {
    const Eigen::internal::TensorIntDivisor<uint64_t> div(i);

    for (uint64_t j = 0; j < 25000; ++j) {
      const uint64_t fast_div = j / div;
      const uint64_t slow_div = j / i;
      VERIFY_IS_EQUAL(fast_div, slow_div);
    }
  }
}

void test_powers_32bit() {
  const int32_t int32_max = (std::numeric_limits<int32_t>::max)();
  for (int expon = 1; expon < 31; expon++) {
    int32_t div = (1 << expon);
    for (int num_expon = 0; num_expon < 32; num_expon++) {
      const int64_t pivot = int64_t(1) << num_expon;
      int32_t start_num = pivot > int32_max ? int32_max : static_cast<int32_t>(pivot);
      int32_t end_num = start_num;
      start_num = (std::max)(int32_t(0), start_num - 100);
      end_num = static_cast<int32_t>((std::min)(int64_t(int32_max), int64_t(end_num) + 100));
      if (start_num < 0) start_num = 0;
      for (int32_t num = start_num; num < end_num; num++) {
        Eigen::internal::TensorIntDivisor<int32_t> divider = Eigen::internal::TensorIntDivisor<int32_t>(div);
        int32_t result = num / div;
        int32_t result_op = divider.divide(num);
        VERIFY_IS_EQUAL(result_op, result);
      }
    }
  }
}

void test_powers_64bit() {
  for (int expon = 0; expon < 63; expon++) {
    int64_t div = (1ull << expon);
    for (int num_expon = 0; num_expon < 63; num_expon++) {
      int64_t start_num = (1ull << num_expon) - 10;
      int64_t end_num = (1ull << num_expon) + 10;
      if (start_num < 0) start_num = 0;
      for (int64_t num = start_num; num < end_num; num++) {
        Eigen::internal::TensorIntDivisor<int64_t> divider(div);
        int64_t result = num / div;
        int64_t result_op = divider.divide(num);
        VERIFY_IS_EQUAL(result_op, result);
      }
    }
  }
}

void test_specific() {
  // A particular combination that was previously failing
  int64_t div = 209715200;
  int64_t num = 3238002688ll;
  Eigen::internal::TensorIntDivisor<int64_t> divider(div);
  int64_t result = num / div;
  int64_t result_op = divider.divide(num);
  VERIFY_IS_EQUAL(result, result_op);
}

// Divisors and numerators up to the largest non-negative index, 2^(k-1) - 1.
template <typename T>
void test_full_range() {
  const T highest = static_cast<T>((std::numeric_limits<std::make_signed_t<T>>::max)());
  const T divisors[] = {T(1),           T(2),   T(3), T(7), T(10), T(highest / 3), T(highest / 2), T(highest / 2 + 1),
                        T(highest - 1), highest};
  for (T d : divisors) {
    const Eigen::internal::TensorIntDivisor<T> div(d);
    const T q = T(highest / d);
    // Both sides of the quotient boundaries j * d, and the ends of the range.
    for (T j : {T(0), T(1), T(q / 2), T(q - 1), q}) {
      for (int offset = -2; offset <= 2; ++offset) {
        const T base = T(j * d);
        if (offset < 0 && base < T(-offset)) continue;
        if (offset > 0 && T(highest - base) < T(offset)) continue;
        const T num = T(base + T(offset));
        VERIFY_IS_EQUAL(num / div, T(num / d));
      }
    }
    for (T num : {T(0), T(1), T(highest / 2), T(highest - 1), highest}) {
      VERIFY_IS_EQUAL(div.divide(num), T(num / d));
    }
    for (int i = 0; i < 1000; ++i) {
      const T num = internal::random<T>(T(0), highest);
      VERIFY_IS_EQUAL(num / div, T(num / d));
    }
  }
}

// Every divisor of a 16-bit index. The computed quotient and floor(n / d) are both nondecreasing in n, so agreeing at
// n = j d - 1 and n = j d for every j makes them agree at every n.
void test_all_divisors_16bit() {
  const int highest = (std::numeric_limits<int16_t>::max)();
  for (int d = 1; d <= highest; ++d) {
    const Eigen::internal::TensorIntDivisor<int16_t> div(static_cast<int16_t>(d));
    VERIFY_IS_EQUAL(int16_t(0) / div, int16_t(0));
    VERIFY_IS_EQUAL(static_cast<int16_t>(highest) / div, static_cast<int16_t>(highest / d));
    for (int boundary = d; boundary <= highest; boundary += d) {
      VERIFY_IS_EQUAL(static_cast<int16_t>(boundary - 1) / div, static_cast<int16_t>(boundary / d - 1));
      VERIFY_IS_EQUAL(static_cast<int16_t>(boundary) / div, static_cast<int16_t>(boundary / d));
    }
  }
}

void test_default_constructed() {
  const Eigen::internal::TensorIntDivisor<Eigen::Index> div;
  const Eigen::Index highest = (std::numeric_limits<Eigen::Index>::max)();
  for (Eigen::Index num : {Eigen::Index(0), Eigen::Index(1), Eigen::Index(12345), highest}) {
    VERIFY_IS_EQUAL(num / div, num);
  }
}

EIGEN_DECLARE_TEST(tensor_intdiv) {
  CALL_SUBTEST_1(test_signed_32bit());
  CALL_SUBTEST_2(test_unsigned_32bit());
  CALL_SUBTEST_3(test_signed_64bit());
  CALL_SUBTEST_4(test_unsigned_64bit());
  CALL_SUBTEST_5(test_powers_32bit());
  CALL_SUBTEST_6(test_powers_64bit());
  CALL_SUBTEST_7(test_specific());
  CALL_SUBTEST_8(test_full_range<int8_t>());
  CALL_SUBTEST_8(test_full_range<uint8_t>());
  CALL_SUBTEST_8(test_full_range<int16_t>());
  CALL_SUBTEST_8(test_full_range<int32_t>());
  CALL_SUBTEST_8(test_full_range<uint32_t>());
  CALL_SUBTEST_8(test_full_range<int64_t>());
  CALL_SUBTEST_8(test_full_range<uint64_t>());
  CALL_SUBTEST_8(test_default_constructed());
  CALL_SUBTEST_8(test_all_divisors_16bit());
}
