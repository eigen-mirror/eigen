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

#define EIGEN_USE_GPU

#include "main.h"
#include <contrib/Eigen/Tensor>
#include "tensor_gpu_test_helpers.h"

using Eigen::Tensor;

void test_cuda_nullary() {
  Tensor<std::complex<float>, 1, 0, int> in1(2);
  Tensor<std::complex<float>, 1, 0, int> in2(2);
  in1.setRandom();
  in2.setRandom();

  tensor_test::GpuTensorTestFixture fixture;
  auto gpu_in1 = fixture.fromHost(in1);
  auto gpu_in2 = fixture.fromHost(in2);
  auto gpu_out2 = fixture.tensor<float, 1, Eigen::Aligned, 0, int>(2);

  gpu_in1.device(fixture.device()) = gpu_in1.constant(std::complex<float>(3.14f, 2.7f));
  gpu_out2.device(fixture.device()) = gpu_in2.abs();

  const auto new1 = fixture.toHost(gpu_in1);
  const auto new2 = fixture.toHost(gpu_out2);

  for (int i = 0; i < 2; ++i) {
    VERIFY_IS_APPROX(new1(i), std::complex<float>(3.14f, 2.7f));
    VERIFY_IS_APPROX(new2(i), std::abs(in2(i)));
  }
}

static void test_cuda_sum_reductions() {
  tensor_test::GpuTensorTestFixture fixture;

  const int num_rows = internal::random<int>(1024, 5 * 1024);
  const int num_cols = internal::random<int>(1024, 5 * 1024);

  Tensor<std::complex<float>, 2> in(num_rows, num_cols);
  in.setRandom();

  Tensor<std::complex<float>, 0> full_redux;
  full_redux = in.sum();

  auto in_gpu = fixture.fromHost(in);
  auto out_gpu = fixture.tensor<std::complex<float>, 0>();

  out_gpu.device(fixture.device()) = in_gpu.sum();

  // Check that the CPU and GPU reductions return the same result.
  VERIFY_IS_APPROX(full_redux(), fixture.toHost(out_gpu)());
}

static void test_cuda_mean_reductions() {
  tensor_test::GpuTensorTestFixture fixture;

  const int num_rows = internal::random<int>(1024, 5 * 1024);
  const int num_cols = internal::random<int>(1024, 5 * 1024);

  Tensor<std::complex<float>, 2> in(num_rows, num_cols);
  in.setRandom();

  Tensor<std::complex<float>, 0> full_redux;
  full_redux = in.mean();

  auto in_gpu = fixture.fromHost(in);
  auto out_gpu = fixture.tensor<std::complex<float>, 0>();

  out_gpu.device(fixture.device()) = in_gpu.mean();

  // Check that the CPU and GPU reductions return the same result.
  VERIFY_IS_APPROX(full_redux(), fixture.toHost(out_gpu)());
}

static void test_cuda_product_reductions() {
  tensor_test::GpuTensorTestFixture fixture;

  const int num_rows = internal::random<int>(1024, 5 * 1024);
  const int num_cols = internal::random<int>(1024, 5 * 1024);

  Tensor<std::complex<float>, 2> in(num_rows, num_cols);
  in.setRandom();

  Tensor<std::complex<float>, 0> full_redux;
  full_redux = in.prod();

  auto in_gpu = fixture.fromHost(in);
  auto out_gpu = fixture.tensor<std::complex<float>, 0>();

  out_gpu.device(fixture.device()) = in_gpu.prod();

  // Check that the CPU and GPU reductions return the same result.
  VERIFY_IS_APPROX(full_redux(), fixture.toHost(out_gpu)());
}

EIGEN_DECLARE_TEST(test_tensor_complex) {
  CALL_SUBTEST(test_cuda_nullary());
  CALL_SUBTEST(test_cuda_sum_reductions());
  CALL_SUBTEST(test_cuda_mean_reductions());
  CALL_SUBTEST(test_cuda_product_reductions());
}
