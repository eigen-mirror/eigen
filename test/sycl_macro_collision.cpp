// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// EIGEN_SUFFIXES;1;2

#if defined(EIGEN_TEST_PART_2)
// Older DPC++ NVPTX passes define __CUDA_ARCH__ without CUDA language mode.
#define __SYCL_DEVICE_ONLY__  // NOLINT(bugprone-reserved-identifier): simulate the compiler's device pass.
#define __CUDA_ARCH__ 500     // NOLINT(bugprone-reserved-identifier): simulate older DPC++ NVPTX targets.
#ifndef EIGEN_NO_ASSERTION_CHECKING
#define EIGEN_NO_ASSERTION_CHECKING
#endif
#endif

// AdaptiveCpp defines this flag even without Eigen's explicit SYCL integration.
#define SYCL_DEVICE_ONLY
#include "main.h"
#include "fp_control.h"
#if !defined(EIGEN_TEST_PART_2)
#include <contrib/Eigen/Tensor>
#endif

#ifdef EIGEN_SYCL_DEVICE_ONLY
#error Eigen SYCL integration requires EIGEN_USE_SYCL
#endif
#ifndef SYCL_DEVICE_ONLY
#error Eigen must preserve externally defined macros
#endif

#if defined(EIGEN_TEST_PART_2)
#if defined(EIGEN_CUDA_ARCH) || defined(EIGEN_GPU_COMPILE_PHASE)
#error SYCL device compilation alone must not select CUDA intrinsic paths
#endif
static_assert(sizeof(EIGEN_MAKESTRING(EIGEN_DEVICE_FUNC)) == 1,
              "Implicit SYCL use must not enable Eigen's flattening policy");
#endif

EIGEN_DECLARE_TEST(sycl_macro_collision) {
#if !defined(EIGEN_TEST_PART_2) && EIGEN_ARCH_x86_64
  ScopedFlushToZero fp_mode(FlushToZeroMode::None);
  VERIFY(fp_mode.isSupported());
#endif
#if defined(EIGEN_TEST_PART_2)
#if EIGEN_COMP_GNUC
  VERIFY_IS_EQUAL(std::string(EIGEN_MAKESTRING(EIGEN_ALWAYS_INLINE)),
                  std::string("__attribute__((always_inline)) inline"));
  VERIFY_IS_EQUAL(std::string(EIGEN_MAKESTRING(EIGEN_LAMBDA_ALWAYS_INLINE)),
                  std::string("__attribute__((always_inline))"));
#endif
  const half half_a(33.f / 32), half_b(65.f / 64), half_c((std::numeric_limits<half>::denorm_min)());
  VERIFY_IS_EQUAL(float(numext::fma(half_a, half_b, half_c)),
                  float(half(numext::fma(float(half_a), float(half_b), float(half_c)))));
#endif
  Vector4d a, b;
  a << 1, 2, 3, 4;
  b << 5, 6, 7, 8;
  Vector4d expected;
  expected << 6, 8, 10, 12;
  VERIFY_IS_EQUAL((a + b).eval(), expected);
  VERIFY_IS_EQUAL(numext::mini(2, 3), 2);
  VERIFY_IS_EQUAL(numext::maxi(2.0, 3.0), 3.0);
#if !defined(EIGEN_TEST_PART_2)
  Tensor<float, 1> tensor(3), result(3);
  tensor.setConstant(2.0f);
  result = tensor + tensor;
  for (Index i = 0; i < result.size(); ++i) VERIFY_IS_EQUAL(result(i), 4.0f);
#endif
}
