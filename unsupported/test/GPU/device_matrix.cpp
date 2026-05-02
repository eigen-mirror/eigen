// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Tests for DeviceMatrix and HostTransfer: typed RAII GPU memory wrapper.
// No cuSOLVER dependency — only CUDA runtime.

#define EIGEN_USE_GPU
#include "main.h"
#include <unsupported/Eigen/GPU>

using namespace Eigen;

// ---- Default construction ---------------------------------------------------

void test_default_construct() {
  gpu::DeviceMatrix<double> dm;
  VERIFY(dm.empty());
  VERIFY_IS_EQUAL(dm.rows(), 0);
  VERIFY_IS_EQUAL(dm.cols(), 0);
  VERIFY(dm.data() == nullptr);
  VERIFY_IS_EQUAL(dm.sizeInBytes(), size_t(0));
}

// ---- Allocate uninitialized -------------------------------------------------

template <typename Scalar>
void test_allocate(Index rows, Index cols) {
  gpu::DeviceMatrix<Scalar> dm(rows, cols);
  VERIFY(!dm.empty());
  VERIFY_IS_EQUAL(dm.rows(), rows);
  VERIFY_IS_EQUAL(dm.cols(), cols);
  VERIFY_IS_EQUAL(dm.outerStride(), rows);
  VERIFY(dm.data() != nullptr);
  VERIFY_IS_EQUAL(dm.sizeInBytes(), size_t(rows) * size_t(cols) * sizeof(Scalar));
}

// ---- fromHost / toHost roundtrip (synchronous) ------------------------------

template <typename Scalar>
void test_roundtrip(Index rows, Index cols) {
  using MatrixType = Eigen::Matrix<Scalar, Dynamic, Dynamic>;
  MatrixType host = MatrixType::Random(rows, cols);

  auto dm = gpu::DeviceMatrix<Scalar>::fromHost(host);
  VERIFY_IS_EQUAL(dm.rows(), rows);
  VERIFY_IS_EQUAL(dm.cols(), cols);
  VERIFY(!dm.empty());

  MatrixType result = dm.toHost();
  VERIFY_IS_EQUAL(result.rows(), rows);
  VERIFY_IS_EQUAL(result.cols(), cols);
  VERIFY_IS_APPROX(result, host);
}

// ---- fromHostAsync / toHostAsync roundtrip -----------------------------------

template <typename Scalar>
void test_roundtrip_async(Index rows, Index cols) {
  using MatrixType = Eigen::Matrix<Scalar, Dynamic, Dynamic>;
  MatrixType host = MatrixType::Random(rows, cols);

  cudaStream_t stream;
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreate(&stream));

  // Async upload from raw pointer.
  auto dm = gpu::DeviceMatrix<Scalar>::fromHostAsync(host.data(), rows, cols, rows, stream);
  VERIFY_IS_EQUAL(dm.rows(), rows);
  VERIFY_IS_EQUAL(dm.cols(), cols);

  // Async download via HostTransfer future.
  auto transfer = dm.toHostAsync(stream);

  // get() blocks and returns the matrix.
  MatrixType result = transfer.get();
  VERIFY_IS_APPROX(result, host);

  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamDestroy(stream));

  cudaStream_t producer_stream;
  cudaStream_t consumer_stream;
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreate(&producer_stream));
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreate(&consumer_stream));

  auto cross_stream_dm = gpu::DeviceMatrix<Scalar>::fromHostAsync(host.data(), rows, cols, rows, producer_stream);
  auto cross_stream_transfer = cross_stream_dm.toHostAsync(consumer_stream);
  MatrixType cross_stream_result = cross_stream_transfer.get();
  VERIFY_IS_APPROX(cross_stream_result, host);

  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamDestroy(consumer_stream));
  EIGEN_CUDA_RUNTIME_CHECK(cudaStreamDestroy(producer_stream));
}

// ---- HostTransfer::ready() and idempotent get() -----------------------------

void test_host_transfer_ready() {
  using MatrixType = Eigen::Matrix<double, Dynamic, Dynamic>;
  MatrixType host = MatrixType::Random(100, 100);

  auto dm = gpu::DeviceMatrix<double>::fromHost(host);
  auto transfer = dm.toHostAsync();

  // After get(), ready() must return true.
  MatrixType result = transfer.get();
  VERIFY(transfer.ready());
  VERIFY_IS_APPROX(result, host);

  // get() is idempotent.
  MatrixType& result2 = transfer.get();
  VERIFY_IS_APPROX(result2, host);
}

// ---- HostTransfer move ------------------------------------------------------

void test_host_transfer_move() {
  using MatrixType = Eigen::Matrix<double, Dynamic, Dynamic>;
  MatrixType host = MatrixType::Random(50, 50);

  auto dm = gpu::DeviceMatrix<double>::fromHost(host);
  auto transfer = dm.toHostAsync();

  gpu::HostTransfer<double> moved(std::move(transfer));
  MatrixType result = moved.get();
  VERIFY_IS_APPROX(result, host);
}

// ---- clone() produces independent copy --------------------------------------

template <typename Scalar>
void test_clone(Index rows, Index cols) {
  using MatrixType = Eigen::Matrix<Scalar, Dynamic, Dynamic>;
  MatrixType host = MatrixType::Random(rows, cols);

  auto dm = gpu::DeviceMatrix<Scalar>::fromHost(host);
  auto cloned = dm.clone();

  // Overwrite original with different data.
  MatrixType other = MatrixType::Random(rows, cols);
  dm = gpu::DeviceMatrix<Scalar>::fromHost(other);

  // Clone still holds the original data.
  MatrixType clone_result = cloned.toHost();
  VERIFY_IS_APPROX(clone_result, host);

  // Original holds the new data.
  MatrixType dm_result = dm.toHost();
  VERIFY_IS_APPROX(dm_result, other);
}

// ---- Move construct ---------------------------------------------------------

template <typename Scalar>
void test_move_construct(Index rows, Index cols) {
  using MatrixType = Eigen::Matrix<Scalar, Dynamic, Dynamic>;
  MatrixType host = MatrixType::Random(rows, cols);

  auto dm = gpu::DeviceMatrix<Scalar>::fromHost(host);
  gpu::DeviceMatrix<Scalar> moved(std::move(dm));

  VERIFY(dm.empty());
  VERIFY(dm.data() == nullptr);

  VERIFY_IS_EQUAL(moved.rows(), rows);
  VERIFY_IS_EQUAL(moved.cols(), cols);
  MatrixType result = moved.toHost();
  VERIFY_IS_APPROX(result, host);
}

// ---- Move assign ------------------------------------------------------------

template <typename Scalar>
void test_move_assign(Index rows, Index cols) {
  using MatrixType = Eigen::Matrix<Scalar, Dynamic, Dynamic>;
  MatrixType host = MatrixType::Random(rows, cols);

  auto dm = gpu::DeviceMatrix<Scalar>::fromHost(host);
  gpu::DeviceMatrix<Scalar> dest;
  dest = std::move(dm);

  VERIFY(dm.empty());
  VERIFY_IS_EQUAL(dest.rows(), rows);
  MatrixType result = dest.toHost();
  VERIFY_IS_APPROX(result, host);
}

// ---- resize() ---------------------------------------------------------------

void test_resize() {
  gpu::DeviceMatrix<double> dm(10, 20);
  VERIFY_IS_EQUAL(dm.rows(), 10);
  VERIFY_IS_EQUAL(dm.cols(), 20);

  dm.resize(50, 30);
  VERIFY_IS_EQUAL(dm.rows(), 50);
  VERIFY_IS_EQUAL(dm.cols(), 30);
  VERIFY_IS_EQUAL(dm.outerStride(), 50);
  VERIFY(dm.data() != nullptr);

  // Resize to same dimensions is a no-op.
  double* ptr_before = dm.data();
  dm.resize(50, 30);
  VERIFY(dm.data() == ptr_before);
}

// ---- Empty / 0x0 matrix -----------------------------------------------------

void test_empty() {
  using MatrixType = Eigen::Matrix<double, Dynamic, Dynamic>;
  MatrixType empty_mat(0, 0);

  auto dm = gpu::DeviceMatrix<double>::fromHost(empty_mat);
  VERIFY(dm.empty());
  VERIFY_IS_EQUAL(dm.rows(), 0);
  VERIFY_IS_EQUAL(dm.cols(), 0);

  MatrixType result = dm.toHost();
  VERIFY_IS_EQUAL(result.rows(), 0);
  VERIFY_IS_EQUAL(result.cols(), 0);
}

// ---- Per-scalar driver ------------------------------------------------------

template <typename Scalar>
void test_scalar() {
  // Square.
  CALL_SUBTEST(test_roundtrip<Scalar>(1, 1));
  CALL_SUBTEST(test_roundtrip<Scalar>(64, 64));
  CALL_SUBTEST(test_roundtrip<Scalar>(256, 256));

  // Rectangular.
  CALL_SUBTEST(test_roundtrip<Scalar>(100, 7));
  CALL_SUBTEST(test_roundtrip<Scalar>(7, 100));

  // Async roundtrip.
  CALL_SUBTEST(test_roundtrip_async<Scalar>(64, 64));
  CALL_SUBTEST(test_roundtrip_async<Scalar>(100, 7));

  CALL_SUBTEST(test_clone<Scalar>(64, 64));
  CALL_SUBTEST(test_move_construct<Scalar>(64, 64));
  CALL_SUBTEST(test_move_assign<Scalar>(64, 64));
}

EIGEN_DECLARE_TEST(gpu_device_matrix) {
  CALL_SUBTEST(test_default_construct());
  CALL_SUBTEST(test_empty());
  CALL_SUBTEST(test_resize());
  CALL_SUBTEST(test_host_transfer_ready());
  CALL_SUBTEST(test_host_transfer_move());
  CALL_SUBTEST((test_allocate<float>(100, 50)));
  CALL_SUBTEST((test_allocate<double>(100, 50)));
  CALL_SUBTEST(test_scalar<float>());
  CALL_SUBTEST(test_scalar<double>());
  CALL_SUBTEST(test_scalar<std::complex<float>>());
  CALL_SUBTEST(test_scalar<std::complex<double>>());
}
