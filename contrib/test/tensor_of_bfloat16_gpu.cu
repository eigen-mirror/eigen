// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2021 Rohit Santhanam <rohit.santhanam@amd.com>
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

EIGEN_DECLARE_TEST(tensor_of_bfloat16_gpu) {
  CALL_SUBTEST_1(tensor_test::test_gpu_numext<Eigen::bfloat16>());

#ifdef EIGEN_HAS_GPU_BF16
  CALL_SUBTEST_2(tensor_test::test_gpu_conversion<Eigen::bfloat16>());
  CALL_SUBTEST_3(tensor_test::test_gpu_unary<Eigen::bfloat16>());
  CALL_SUBTEST_4(tensor_test::test_gpu_elementwise<Eigen::bfloat16>());
  CALL_SUBTEST_5(tensor_test::test_gpu_transcendental<Eigen::bfloat16>());
  CALL_SUBTEST_6(tensor_test::test_gpu_contractions<Eigen::bfloat16>(2.f));
  CALL_SUBTEST_7(tensor_test::test_gpu_reductions<Eigen::bfloat16>());
  CALL_SUBTEST_8(tensor_test::test_gpu_full_reductions<Eigen::bfloat16>());
  CALL_SUBTEST_9(tensor_test::test_gpu_forced_evals<Eigen::bfloat16>());
#else
  // Without device bfloat16 support parts 2-9 have nothing to run.  Report them as skipped rather than let an
  // empty part pass: under EIGEN_SPLIT_LARGE_TESTS each part is its own executable, and a green one would claim
  // coverage that never ran.  Detected at run time: an EIGEN_TEST_PART_<n> token in the source would make
  // ei_add_test split the test even with EIGEN_SPLIT_LARGE_TESTS=OFF.
  bool ran_numext = false;
  CALL_SUBTEST_1(ran_numext = true);
  if (!ran_numext) skip_test("bfloat16 device support (EIGEN_HAS_GPU_BF16) is not available with this toolkit.");
  std::cout << "bfloat16 device support (EIGEN_HAS_GPU_BF16) is not available with this toolkit: only the numext "
               "part runs."
            << std::endl;
#endif
}
