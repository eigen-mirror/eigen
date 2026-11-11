// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX

#define EIGEN_USE_GPU

#include "main.h"
#include <contrib/Eigen/Tensor>
#include "tensor_gpu_test_helpers.h"

template <typename Type, int DataLayout>
static void test_full_reductions() {
  tensor_test::GpuTensorTestFixture fixture;

  const int num_rows = internal::random<int>(1024, 5 * 1024);
  const int num_cols = internal::random<int>(1024, 5 * 1024);

  Tensor<Type, 2, DataLayout> in(num_rows, num_cols);
  in.setRandom();

  Tensor<Type, 0, DataLayout> full_redux;
  full_redux = in.sum();

  auto in_gpu = fixture.fromHost(in);
  auto out_gpu = fixture.tensor<Type, 0, Eigen::Aligned, DataLayout>();

  out_gpu.device(fixture.device()) = in_gpu.sum();

  // Check that the CPU and GPU reductions return the same result.
  VERIFY_IS_APPROX(full_redux(), fixture.toHost(out_gpu)());
}

// A full reduction of an empty tensor writes the reducer's identity.
template <typename Type>
static void test_empty_full_reduction() {
  tensor_test::GpuTensorTestFixture fixture;
  auto in_gpu = fixture.tensor<Type, 2, Eigen::Unaligned>(0, 3);
  Tensor<Type, 0> sentinel;
  sentinel() = Type(7);
  auto out_gpu = fixture.fromHost(sentinel);
  out_gpu.device(fixture.device()) = in_gpu.sum();
  VERIFY_IS_EQUAL(fixture.toHost(out_gpu)(), Type(0));
}

template <typename Type, int DataLayout>
static void test_first_dim_reductions(int dim_x = 33, int dim_y = 1, int dim_z = 128) {
  Tensor<Type, 3, DataLayout> in(dim_x, dim_y, dim_z);
  in.setRandom();

  Eigen::array<int, 1> red_axis;
  red_axis[0] = 0;
  Tensor<Type, 2, DataLayout> redux = in.sum(red_axis);

  tensor_test::GpuTensorTestFixture fixture;
  auto gpu_in = fixture.fromHost(in);
  auto gpu_out = fixture.tensor<Type, 2, Eigen::Aligned, DataLayout>(dim_y, dim_z);

  // Perform operation
  gpu_out.device(fixture.device()) = gpu_in.sum(red_axis);
  gpu_out.device(fixture.device()) += gpu_in.sum(red_axis);
  const auto redux_gpu = fixture.toHost(gpu_out);

  // Check that the CPU and GPU reductions return the same result.
  for (int i = 0; i < gpu_out.size(); ++i) {
    VERIFY_IS_APPROX(2 * redux(i), redux_gpu(i));
  }
}

template <typename Type, int DataLayout>
static void test_last_dim_reductions(int dim_x = 128, int dim_y = 1, int dim_z = 33) {
  Tensor<Type, 3, DataLayout> in(dim_x, dim_y, dim_z);
  in.setRandom();

  Eigen::array<int, 1> red_axis;
  red_axis[0] = 2;
  Tensor<Type, 2, DataLayout> redux = in.sum(red_axis);

  tensor_test::GpuTensorTestFixture fixture;
  auto gpu_in = fixture.fromHost(in);
  auto gpu_out = fixture.tensor<Type, 2, Eigen::Aligned, DataLayout>(dim_x, dim_y);

  // Perform operation
  gpu_out.device(fixture.device()) = gpu_in.sum(red_axis);
  gpu_out.device(fixture.device()) += gpu_in.sum(red_axis);
  const auto redux_gpu = fixture.toHost(gpu_out);

  // Check that the CPU and GPU reductions return the same result.
  for (int i = 0; i < gpu_out.size(); ++i) {
    VERIFY_IS_APPROX(2 * redux(i), redux_gpu(i));
  }
}

EIGEN_DECLARE_TEST(tensor_reduction_gpu) {
  CALL_SUBTEST_1((test_full_reductions<float, ColMajor>()));
  CALL_SUBTEST_1((test_full_reductions<double, ColMajor>()));
  CALL_SUBTEST_1((test_empty_full_reduction<float>()));
  CALL_SUBTEST_1((test_empty_full_reduction<double>()));
  CALL_SUBTEST_2((test_full_reductions<float, RowMajor>()));
  CALL_SUBTEST_2((test_full_reductions<double, RowMajor>()));

  CALL_SUBTEST_3((test_first_dim_reductions<float, ColMajor>()));
  CALL_SUBTEST_3((test_first_dim_reductions<double, ColMajor>()));
  CALL_SUBTEST_4((test_first_dim_reductions<float, RowMajor>()));
  CALL_SUBTEST_4((test_first_dim_reductions<double, RowMajor>()));

  CALL_SUBTEST_5((test_last_dim_reductions<float, ColMajor>()));
  CALL_SUBTEST_5((test_last_dim_reductions<double, ColMajor>()));
  CALL_SUBTEST_6((test_last_dim_reductions<float, RowMajor>()));
  CALL_SUBTEST_6((test_last_dim_reductions<double, RowMajor>()));

  // CUDA double sums also reach OuterReductionKernel below and above the former shape band. HIP keeps that band.
  {
    Eigen::GpuStreamDevice stream;
    Eigen::GpuDevice device(&stream);
    const int multi_processors = device.getNumGpuMultiProcessors();
    for (int outputs : {4 * multi_processors, 16 * multi_processors, 65 * multi_processors}) {
      EIGEN_UNUSED_VARIABLE(outputs);
      CALL_SUBTEST_7((test_last_dim_reductions<double, ColMajor>(outputs, 1, 128)));
      CALL_SUBTEST_7((test_first_dim_reductions<double, RowMajor>(128, 1, outputs)));
    }
    CALL_SUBTEST_7((test_last_dim_reductions<double, ColMajor>(16 * multi_processors, 1, 33)));
    CALL_SUBTEST_7((test_first_dim_reductions<double, RowMajor>(33, 1, 16 * multi_processors)));
  }
}
