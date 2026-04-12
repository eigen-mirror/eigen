// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Smoke test for GPU library test infrastructure.
// Verifies gpu::Context, GpuBuffer, and host<->device matrix transfers
// without requiring any NVIDIA library (cuBLAS, cuSOLVER, etc.).

#define EIGEN_USE_GPU
#include "main.h"
#include "gpu_context.h"
#include "gpu_library_test_helper.h"

using namespace Eigen;
using namespace Eigen::test;

// Test that gpu::Context initializes, reports valid device info, and owns a cuSOLVER handle.
void test_gpu_context() {
  gpu::Context ctx;
  VERIFY(ctx.device() >= 0);
  VERIFY(ctx.deviceProperties().major >= 7);  // sm_70 minimum
  VERIFY(ctx.stream != nullptr);
  VERIFY(ctx.cusolver != nullptr);
  std::cout << "  GPU: " << ctx.deviceProperties().name << " (sm_" << ctx.deviceProperties().major
            << ctx.deviceProperties().minor << ")\n";
}

// Test dense matrix roundtrip: host -> device -> host.
template <typename MatrixType>
void test_dense_roundtrip() {
  gpu::Context ctx;
  const Index rows = 64;
  const Index cols = 32;

  MatrixType A = MatrixType::Random(rows, cols);
  auto buf = gpu_copy_to_device(ctx.stream, A);
  VERIFY(buf.data != nullptr);
  VERIFY(buf.size == rows * cols);

  MatrixType B(rows, cols);
  B.setZero();
  gpu_copy_to_host(ctx.stream, buf, B);
  ctx.synchronize();

  VERIFY_IS_EQUAL(A, B);
}

// Test GpuBuffer RAII: move semantics, async zero-init.
void test_gpu_buffer() {
  gpu::Context ctx;

  GpuBuffer<float> a(128);
  VERIFY(a.data != nullptr);
  VERIFY(a.size == 128);

  // Move construction.
  GpuBuffer<float> b(std::move(a));
  VERIFY(a.data == nullptr);
  VERIFY(b.data != nullptr);
  VERIFY(b.size == 128);

  // Move assignment.
  GpuBuffer<float> c;
  c = std::move(b);
  VERIFY(b.data == nullptr);
  VERIFY(c.data != nullptr);

  // setZeroAsync.
  c.setZeroAsync(ctx.stream);
  ctx.synchronize();

  std::vector<float> host(128);
  GPU_CHECK(cudaMemcpy(host.data(), c.data, 128 * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 128; ++i) {
    VERIFY_IS_EQUAL(host[i], 0.0f);
  }
}

// Test with vectors (1D).
template <typename Scalar>
void test_vector_roundtrip() {
  gpu::Context ctx;
  const Index n = 256;
  Matrix<Scalar, Dynamic, 1> v = Matrix<Scalar, Dynamic, 1>::Random(n);
  auto buf = gpu_copy_to_device(ctx.stream, v);

  Matrix<Scalar, Dynamic, 1> w(n);
  w.setZero();
  gpu_copy_to_host(ctx.stream, buf, w);
  ctx.synchronize();

  VERIFY_IS_EQUAL(v, w);
}

EIGEN_DECLARE_TEST(gpu_library_example) {
  CALL_SUBTEST(test_gpu_context());
  CALL_SUBTEST(test_gpu_buffer());
  CALL_SUBTEST(test_dense_roundtrip<MatrixXf>());
  CALL_SUBTEST(test_dense_roundtrip<MatrixXd>());
  CALL_SUBTEST((test_dense_roundtrip<Matrix<float, Dynamic, Dynamic, RowMajor>>()));
  CALL_SUBTEST((test_dense_roundtrip<Matrix<double, Dynamic, Dynamic, RowMajor>>()));
  CALL_SUBTEST(test_vector_roundtrip<float>());
  CALL_SUBTEST(test_vector_roundtrip<double>());
  CALL_SUBTEST(test_vector_roundtrip<std::complex<float>>());
}
