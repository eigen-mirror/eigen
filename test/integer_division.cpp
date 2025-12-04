// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template <typename T>
void test_generic_magic() {
  constexpr int k = 8 * sizeof(T);
  int max_iter = 0;
  T max_d = 1;
  for (T d = 1; d != 0; d++) {
    iter = 0;
    int p = internal::log2_ceil(d);
    T magic = internal::calc_magic_generic(d, p);
    T ref = internal::calc_magic(d, p);
    VERIFY_IS_EQUAL(magic, ref);
  }

}

EIGEN_DECLARE_TEST(integer_division) {
  CALL_SUBTEST_1(test_generic_magic<uint8_t>());
  //CALL_SUBTEST_1(test_generic_magic<uint16_t>());
  //CALL_SUBTEST_1(test_generic_magic<uint32_t>());
  // for (int i = 0; i < g_repeat; i++) {
  //   // CALL_SUBTEST_1((test_realview<Dynamic, Dynamic, Dynamic, Dynamic>()));
  // }
}
