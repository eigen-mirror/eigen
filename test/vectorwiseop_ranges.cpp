// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define TEST_ENABLE_TEMPORARY_TRACKING

#include "main.h"

#if defined(__cpp_lib_ranges) && __cpp_lib_ranges >= 201911L
#include <ranges>
#endif

void vectorwiseop_use_in_std_ranges() {
  // verify basic std::ranges functionality; noop if ranges not present
#if defined(__cpp_lib_ranges) && __cpp_lib_ranges >= 201911L
  Matrix3f a = Matrix3f::Random();
  int count = 0;
  std::ranges::for_each(a.colwise(), [&count](auto&& col) { count += col.count(); });
  VERIFY_IS_EQUAL(count, 9);
  std::ranges::for_each(a.rowwise(), [&count](auto&& row) { count += row.count(); });
  VERIFY_IS_EQUAL(count, 18);
#endif
}

EIGEN_DECLARE_TEST(vectorwiseop_ranges) { CALL_SUBTEST_1(vectorwiseop_use_in_std_ranges()); }
