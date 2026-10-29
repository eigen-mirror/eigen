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

void test_gpu_conversion() {
  tensor_test::GpuTensorTestFixture fixture;
  int num_elem = 101;

  Tensor<float, 1> floats(num_elem);
  floats.setRandom();

  auto gpu_float = fixture.tensor<float, 1, Eigen::Aligned>(num_elem);
  auto gpu_half = fixture.tensor<Eigen::half, 1, Eigen::Aligned>(num_elem);
  auto gpu_conv = fixture.tensor<float, 1, Eigen::Aligned>(num_elem);

  fixture.device().memcpyHostToDevice(gpu_float.data(), floats.data(), num_elem * sizeof(float));

  gpu_half.device(fixture.device()) = gpu_float.cast<Eigen::half>();
  gpu_conv.device(fixture.device()) = gpu_half.cast<float>();

  const auto initial = fixture.toHost(gpu_float);
  const auto final = fixture.toHost(gpu_conv);

  for (int i = 0; i < num_elem; ++i) {
    VERIFY_IS_APPROX(initial(i), final(i));
  }
}

void test_fallback_conversion() {
  int num_elem = 101;
  Tensor<float, 1> floats(num_elem);
  floats.setRandom();

  Eigen::Tensor<Eigen::half, 1> halfs = floats.cast<Eigen::half>();
  Eigen::Tensor<float, 1> conv = halfs.cast<float>();

  for (int i = 0; i < num_elem; ++i) {
    VERIFY_IS_APPROX(floats(i), conv(i));
  }
}

EIGEN_DECLARE_TEST(tensor_cast_float16_gpu) {
  CALL_SUBTEST(test_gpu_conversion());
  CALL_SUBTEST(test_fallback_conversion());
}
