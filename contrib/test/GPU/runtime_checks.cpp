// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// A failed CUDA runtime or library call has to reach EIGEN_GPU_CHECK_FAILED in a
// release build as much as in a debug one (runtime_checks_ndebug.cpp compiles
// this file with EIGEN_NO_DEBUG). The hook is overridden here to record the
// failure instead of stopping the process.

#include <string>

struct RecordedGpuFailure {
  std::string error;
  std::string expression;
  std::string file;
  int line = 0;
};
static RecordedGpuFailure g_last_failure;
static int g_num_failures = 0;
static bool g_throw_failures = false;  // throw the record, as a user's hook may

static void record_gpu_failure(const char* error, const char* expression, const char* file, int line) {
  ++g_num_failures;
  g_last_failure.error = error;
  g_last_failure.expression = expression;
  g_last_failure.file = file;
  g_last_failure.line = line;
  if (g_throw_failures) throw g_last_failure;
}
#define EIGEN_GPU_CHECK_FAILED(error, expression, file, line) ::record_gpu_failure(error, expression, file, line)

#define EIGEN_USE_GPU
#include "main.h"
#include <contrib/Eigen/GPU>

#include "./gpu_test_helpers.h"

using namespace Eigen;

static bool starts_with(const std::string& s, const std::string& prefix) {
  return s.compare(0, prefix.size(), prefix) == 0;
}

static bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// A runtime call that fails inside the module is reported with the call's own
// text and location: downloading a view of a null pointer copies from null.
void test_runtime_call() {
  gpu::Context ctx;
  g_num_failures = 0;
  gpu::DeviceMatrix<double> bogus = gpu::DeviceMatrix<double>::view(nullptr, 4, 1);
  (void)bogus.toHost(ctx.stream());
  VERIFY_IS_EQUAL(g_num_failures, 1);
  VERIFY_IS_EQUAL(g_last_failure.error, std::string("cudaErrorInvalidValue"));
  VERIFY(starts_with(g_last_failure.expression, "cudaMemcpyAsync"));
  VERIFY(ends_with(g_last_failure.file, "DeviceMatrix.h"));
  VERIFY(g_last_failure.line > 0);

  // The context is still usable, and a successful call reports nothing.
  const VectorXd x = VectorXd::Random(8);
  VERIFY_IS_APPROX(VectorXd(gpu::DeviceMatrix<double>::fromHost(x, ctx.stream()).toHost(ctx.stream())), x);
  VERIFY_IS_EQUAL(g_num_failures, 1);
}

// cuBLAS before 11.6.1 cannot name a status, see cublas_check_failed().
static std::string cublas_status_text(cublasStatus_t status, const char* name) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 110601
  EIGEN_UNUSED_VARIABLE(status);
  return name;
#else
  EIGEN_UNUSED_VARIABLE(name);
  return "cuBLAS status " + std::to_string(int(status));
#endif
}

// Each library's check macro reports a failed status, named where the library
// can name it. The macros are given the statuses directly: what a library
// returns for invalid arguments differs between versions, and some calls
// dereference a null handle instead of returning a status.
void test_library_checks() {
  g_num_failures = 0;
  EIGEN_CUBLAS_CHECK(CUBLAS_STATUS_SUCCESS);
  EIGEN_NPP_CHECK(NPP_NO_OPERATION_WARNING);  // positive NPP statuses are warnings
  VERIFY_IS_EQUAL(g_num_failures, 0);

  EIGEN_CUBLAS_CHECK(CUBLAS_STATUS_INVALID_VALUE);
  VERIFY_IS_EQUAL(g_num_failures, 1);
  VERIFY_IS_EQUAL(g_last_failure.error, cublas_status_text(CUBLAS_STATUS_INVALID_VALUE, "CUBLAS_STATUS_INVALID_VALUE"));
  VERIFY_IS_EQUAL(g_last_failure.expression, std::string("CUBLAS_STATUS_INVALID_VALUE"));
  VERIFY(ends_with(g_last_failure.file, "runtime_checks.cpp"));
  VERIFY(g_last_failure.line > 0);

  EIGEN_CUBLASLT_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
  VERIFY_IS_EQUAL(g_num_failures, 2);
  VERIFY_IS_EQUAL(g_last_failure.error, cublas_status_text(CUBLAS_STATUS_NOT_SUPPORTED, "CUBLAS_STATUS_NOT_SUPPORTED"));

  EIGEN_CUSOLVER_CHECK(CUSOLVER_STATUS_INVALID_VALUE);
  VERIFY_IS_EQUAL(g_num_failures, 3);
  VERIFY_IS_EQUAL(g_last_failure.error, std::string("CUSOLVER_STATUS_INVALID_VALUE"));

  EIGEN_CUSPARSE_CHECK(CUSPARSE_STATUS_INVALID_VALUE);
  VERIFY_IS_EQUAL(g_num_failures, 4);
  VERIFY_IS_EQUAL(g_last_failure.error, std::string("CUSPARSE_STATUS_INVALID_VALUE"));

  EIGEN_CUFFT_CHECK(CUFFT_INVALID_PLAN);
  VERIFY_IS_EQUAL(g_num_failures, 5);
  VERIFY_IS_EQUAL(g_last_failure.error, "cuFFT status " + std::to_string(int(CUFFT_INVALID_PLAN)));

  EIGEN_NPP_CHECK(NPP_NULL_POINTER_ERROR);
  VERIFY_IS_EQUAL(g_num_failures, 6);
  VERIFY_IS_EQUAL(g_last_failure.error, "NPP status " + std::to_string(int(NPP_NULL_POINTER_ERROR)));
}

// A hook that throws unwinds through the module with the cuBLAS handle back in
// its pointer mode, so that later host-scalar calls on the context still work.
void test_throwing_hook() {
  gpu::Context ctx;
  cublasPointerMode_t mode = CUBLAS_POINTER_MODE_DEVICE;
  EIGEN_CUBLAS_CHECK(cublasGetPointerMode(ctx.cublasHandle(), &mode));
  VERIFY(mode == CUBLAS_POINTER_MODE_HOST);

  bool thrown = false;
  g_throw_failures = true;
  try {
    gpu::internal::with_device_pointer_mode(ctx.cublasHandle(), [] { EIGEN_CUBLAS_CHECK(CUBLAS_STATUS_ALLOC_FAILED); });
  } catch (const RecordedGpuFailure&) {
    thrown = true;
  }
  g_throw_failures = false;
  VERIFY(thrown);
  EIGEN_CUBLAS_CHECK(cublasGetPointerMode(ctx.cublasHandle(), &mode));
  VERIFY(mode == CUBLAS_POINTER_MODE_HOST);

  const VectorXd x = VectorXd::Random(16);
  gpu::DeviceMatrix<double> d_x = gpu::DeviceMatrix<double>::fromHost(x, ctx.stream());
  d_x.scale(ctx, 2.0);
  VERIFY_IS_APPROX(VectorXd(d_x.toHost(ctx.stream())), VectorXd(2.0 * x));
}

EIGEN_DECLARE_TEST(gpu_runtime_checks) {
  gpu_test::require_cuda_device();
  CALL_SUBTEST(test_runtime_call());
  CALL_SUBTEST(test_library_checks());
  CALL_SUBTEST(test_throwing_hook());
}
