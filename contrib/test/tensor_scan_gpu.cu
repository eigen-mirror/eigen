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
#include <contrib/Eigen/Tensor>
#include "tensor_gpu_test_helpers.h"

using Eigen::Tensor;
using DimPair = Tensor<float, 1>::DimensionPair;

template <int DataLayout>
void test_gpu_cumsum(int m_size, int k_size, int n_size) {
  std::cout << "Testing for (" << m_size << "," << k_size << "," << n_size << ")" << std::endl;
  Tensor<float, 3, DataLayout> t_input(m_size, k_size, n_size);
  Tensor<float, 3, DataLayout> t_result(m_size, k_size, n_size);

  t_input.setRandom();

  tensor_test::GpuTensorTestFixture fixture;
  auto gpu_t_input = fixture.fromHost(t_input);
  auto gpu_t_result = fixture.tensor<float, 3, Eigen::Aligned, DataLayout>(m_size, k_size, n_size);

  gpu_t_result.device(fixture.device()) = gpu_t_input.cumsum(1);
  t_result = t_input.cumsum(1);

  const auto t_result_gpu = fixture.toHost(gpu_t_result);
  for (DenseIndex i = 0; i < t_result.size(); i++) {
    // Scans accumulate float rounding error; 1024 * eps of absolute slack covers it before the relative check.
    if (numext::abs(t_result(i) - t_result_gpu(i)) < 1024 * NumTraits<float>::epsilon()) {
      continue;
    }
    if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i), 1024 * NumTraits<float>::epsilon())) {
      continue;
    }
    std::cout << "mismatch detected at index " << i << ": " << t_result(i) << " vs " << t_result_gpu(i) << std::endl;
    assert(false);
  }
}

EIGEN_DECLARE_TEST(tensor_scan_gpu) {
  CALL_SUBTEST_1(test_gpu_cumsum<ColMajor>(128, 128, 128));
  CALL_SUBTEST_1(test_gpu_cumsum<ColMajor>(4, 0, 4));
  CALL_SUBTEST_2(test_gpu_cumsum<RowMajor>(128, 128, 128));
  CALL_SUBTEST_2(test_gpu_cumsum<RowMajor>(0, 4, 4));
}
