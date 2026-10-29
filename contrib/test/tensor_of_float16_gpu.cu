// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU

#include "main.h"
#include "tensor_low_precision_gpu_test_helpers.h"

EIGEN_DECLARE_TEST(tensor_of_float16_gpu) {
  CALL_SUBTEST_1(tensor_test::test_gpu_numext<Eigen::half>());

  CALL_SUBTEST_1(tensor_test::test_gpu_unary<Eigen::half>());
  CALL_SUBTEST_1(tensor_test::test_gpu_elementwise<Eigen::half>());
  CALL_SUBTEST_1(tensor_test::test_gpu_transcendental<Eigen::half>());
  CALL_SUBTEST_2(tensor_test::test_gpu_contractions<Eigen::half>(16.f));
  CALL_SUBTEST_3(tensor_test::test_gpu_reductions<Eigen::half>());
  CALL_SUBTEST_4(tensor_test::test_gpu_full_reductions<Eigen::half>());
  CALL_SUBTEST_5(tensor_test::test_gpu_forced_evals<Eigen::half>());
}
