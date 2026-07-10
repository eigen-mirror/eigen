// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
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
#include <Eigen/Tensor>

void test_gpu_random_uniform() {
  Tensor<float, 2> out(72, 97);
  out.setZero();

  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_out;
  gpuMalloc((void**)(&d_out), out_bytes);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_out(d_out, 72, 97);

  gpu_out.device(gpu_device) = gpu_out.random();

  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  // For now we just check this code doesn't crash.
  // TODO: come up with a valid test of randomness
}

void test_gpu_random_normal() {
  Tensor<float, 2> out(72, 97);
  out.setZero();

  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_out;
  gpuMalloc((void**)(&d_out), out_bytes);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_out(d_out, 72, 97);

  Eigen::internal::NormalRandomGenerator<float> gen(true);
  gpu_out.device(gpu_device) = gpu_out.random(gen);

  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
}

template <typename Scalar>
void test_gpu_random_uniform_range(int rows, int cols) {
  Tensor<Scalar, 2> out(rows, cols);
  out.setZero();

  std::size_t out_bytes = out.size() * sizeof(Scalar);

  Scalar* d_out;
  gpuMalloc((void**)(&d_out), out_bytes);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, rows, cols);

  gpu_out.device(gpu_device) = gpu_out.random();

  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  // All uniform draws must lie in [0, 1).
  int num_out_of_range = 0;
  for (int i = 0; i < out.size(); ++i) {
    if (!(out.data()[i] >= Scalar(0.0f) && out.data()[i] < Scalar(1.0f))) ++num_out_of_range;
  }
  VERIFY_IS_EQUAL(num_out_of_range, 0);
}

template <typename Scalar>
void test_gpu_random_normal_all_finite(int rows, int cols) {
  Tensor<Scalar, 2> out(rows, cols);
  out.setZero();

  std::size_t out_bytes = out.size() * sizeof(Scalar);

  Scalar* d_out;
  gpuMalloc((void**)(&d_out), out_bytes);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<Scalar, 2> > gpu_out(d_out, rows, cols);

  Eigen::internal::NormalRandomGenerator<Scalar> gen(true);
  gpu_out.device(gpu_device) = gpu_out.random(gen);

  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  // Regression test for 16-bit types: the deviate must be computed in float,
  // otherwise log(0) in the rejection algorithm emits NaN/Inf.
  int num_not_finite = 0;
  for (int i = 0; i < out.size(); ++i) {
    if (!(numext::isfinite)(out.data()[i])) ++num_not_finite;
  }
  VERIFY_IS_EQUAL(num_not_finite, 0);
}

static void test_complex() {
  Tensor<std::complex<float>, 1> vec(6);
  vec.setRandom();

  // Fixme: we should check that the generated numbers follow a uniform
  // distribution instead.
  for (int i = 1; i < 6; ++i) {
    VERIFY_IS_NOT_EQUAL(vec(i), vec(i - 1));
  }
}

EIGEN_DECLARE_TEST(tensor_random_gpu) {
  CALL_SUBTEST(test_gpu_random_uniform());
  CALL_SUBTEST(test_gpu_random_normal());
  CALL_SUBTEST(test_gpu_random_uniform_range<Eigen::half>(256, 256));
  CALL_SUBTEST(test_gpu_random_uniform_range<Eigen::bfloat16>(256, 256));
  CALL_SUBTEST(test_gpu_random_normal_all_finite<Eigen::half>(1024, 2048));
  CALL_SUBTEST(test_gpu_random_normal_all_finite<Eigen::bfloat16>(512, 512));
  CALL_SUBTEST(test_complex());
}
