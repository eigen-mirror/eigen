// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/Core>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

enum class BarrierEnum : unsigned long long { Value = 0xfedcba9876543210ull };
struct BarrierRecord {
  unsigned long long value;
  bool operator==(const BarrierRecord& other) const { return value == other.value; }
};
union BarrierUnion {
  unsigned long long value;
  double real;
  bool operator==(const BarrierUnion& other) const { return value == other.value; }
};
// GCC rejects register and general asm constraints for this struct on every target.
struct BarrierTriple {
  double x, y, z;
  bool operator==(const BarrierTriple& other) const { return x == other.x && y == other.y && z == other.z; }
};

template <typename T>
void check_barrier_value(T value) {
  const T expected = value;
  EIGEN_OPTIMIZATION_BARRIER(value);
  VERIFY(value == expected);
  internal::pbarrier(value);
  VERIFY(value == expected);
}

template <typename Scalar>
EIGEN_DONT_INLINE Scalar rounded_by_barrier(Scalar x) {
  // A constant bias: MSVC /fp:fast folds (x + bias) - bias only when it can see the constant.
  constexpr Scalar bias = Scalar(1ull << (NumTraits<Scalar>::digits() - 1));
  Scalar rounded = x + bias;
  EIGEN_OPTIMIZATION_BARRIER(rounded);
  return rounded - bias;
}

EIGEN_DECLARE_TEST(optimization_barrier) {
  check_barrier_value(false);
  check_barrier_value(static_cast<signed char>(-17));
  check_barrier_value(static_cast<unsigned short>(65521));
  check_barrier_value(-1234567);
  check_barrier_value(0xfedcba9876543210ull);
  check_barrier_value(BarrierEnum::Value);
  check_barrier_value(nullptr);
  check_barrier_value(1.25f);
  check_barrier_value(1.25);
  check_barrier_value(1.25L);
  int pointee = 7;
  check_barrier_value(&pointee);
  volatile decltype(nullptr) null_value = nullptr;
  EIGEN_OPTIMIZATION_BARRIER(null_value);
  VERIFY(null_value == nullptr);
  volatile double real_value = 1.25;
  EIGEN_OPTIMIZATION_BARRIER(real_value);
  VERIFY(real_value == 1.25);

  check_barrier_value(BarrierRecord{0xfedcba9876543210ull});
  check_barrier_value(BarrierUnion{0xfedcba9876543210ull});
  check_barrier_value(BarrierTriple{1.25, -2.5, 3.75});
  check_barrier_value(std::complex<double>(1.25, -2.5));
  int array[] = {1, -2, 3, -4};
  EIGEN_OPTIMIZATION_BARRIER(array);
  for (int i = 0; i < 4; ++i) VERIFY_IS_EQUAL(array[i], (i % 2 ? -1 : 1) * (i + 1));
  double wide_array[8] = {1, -2, 3, -4, 5, -6, 7, -8};
  EIGEN_OPTIMIZATION_BARRIER(wide_array);
  for (int i = 0; i < 8; ++i) VERIFY_IS_EQUAL(wide_array[i], (i % 2 ? -1 : 1) * (i + 1));
  internal::eigen_packet_wrapper<int, 7> wrapped(-1234567);
  internal::pbarrier(wrapped);
  VERIFY_IS_EQUAL(static_cast<int&>(wrapped), -1234567);
#ifdef __ARM_FEATURE_SVE
  // Sizeless: sizeof(svfloat64_t) is ill-formed, and there is no operator== for check_barrier_value.
  svfloat64_t sizeless = svdup_n_f64(1.25);
  EIGEN_OPTIMIZATION_BARRIER(sizeless);
  VERIFY(!svptest_any(svptrue_b64(), svcmpne_n_f64(svptrue_b64(), sizeless, 1.25)));
  internal::pbarrier(sizeless);
  VERIFY(!svptest_any(svptrue_b64(), svcmpne_n_f64(svptrue_b64(), sizeless, 1.25)));
#endif

#if !defined(EIGEN_GPU_COMPILE_PHASE) && !defined(__SYCL_DEVICE_ONLY__) && \
    (EIGEN_COMP_GNUC || (defined(__clang__) && defined(_MSC_VER)) || (EIGEN_COMP_MSVC && defined(_M_FP_FAST)))
  VERIFY_IS_EQUAL(rounded_by_barrier(1.25f), 1.0f);
  VERIFY_IS_EQUAL(rounded_by_barrier(1.25), 1.0);
#endif

#if !defined(EIGEN_GPU_COMPILE_PHASE) && (EIGEN_COMP_GNUC_STRICT || EIGEN_COMP_CLANG || EIGEN_COMP_MINGW) && \
    (EIGEN_ARCH_PPC || EIGEN_ARCH_ARM_OR_ARM64 || EIGEN_ARCH_i386_OR_x86_64)
  using internal::optimization_barrier_in_memory;
  STATIC_CHECK(optimization_barrier_in_memory<BarrierRecord>::value);
  STATIC_CHECK(optimization_barrier_in_memory<BarrierUnion>::value);
  STATIC_CHECK(optimization_barrier_in_memory<BarrierTriple>::value);
  STATIC_CHECK(optimization_barrier_in_memory<std::complex<double>>::value);
  STATIC_CHECK(optimization_barrier_in_memory<Eigen::half>::value);
  STATIC_CHECK(optimization_barrier_in_memory<int[4]>::value);
  STATIC_CHECK(optimization_barrier_in_memory<double[8]>::value);
  STATIC_CHECK(!optimization_barrier_in_memory<double>::value);
  STATIC_CHECK(!optimization_barrier_in_memory<long double>::value);
  STATIC_CHECK(!optimization_barrier_in_memory<int*>::value);
  STATIC_CHECK(!optimization_barrier_in_memory<decltype(nullptr)>::value);
  STATIC_CHECK(!optimization_barrier_in_memory<BarrierEnum>::value);
#endif

#if !defined(EIGEN_GPU_COMPILE_PHASE) && (EIGEN_COMP_GNUC_STRICT || EIGEN_COMP_CLANG || EIGEN_COMP_MINGW) && \
    (EIGEN_ARCH_ARM_OR_ARM64 || EIGEN_ARCH_i386_OR_x86_64)
  using internal::optimization_barrier_in_register;
#if (EIGEN_ARCH_ARM_OR_ARM64 && defined(__ARM_FP) && (__ARM_FP & 4)) || (EIGEN_ARCH_i386_OR_x86_64 && defined(__SSE2__))
  STATIC_CHECK(optimization_barrier_in_register<float>::value);
#endif
#if (EIGEN_ARCH_ARM_OR_ARM64 && defined(__ARM_FP) && (__ARM_FP & 8)) || (EIGEN_ARCH_i386_OR_x86_64 && defined(__SSE2__))
  STATIC_CHECK(optimization_barrier_in_register<double>::value);
#endif
  STATIC_CHECK(!optimization_barrier_in_register<long double>::value);
  STATIC_CHECK(!optimization_barrier_in_register<int>::value);
  STATIC_CHECK(!optimization_barrier_in_register<int*>::value);
  STATIC_CHECK(!optimization_barrier_in_register<decltype(nullptr)>::value);
  STATIC_CHECK(!optimization_barrier_in_register<BarrierEnum>::value);
  STATIC_CHECK(!optimization_barrier_in_register<BarrierRecord>::value);
  STATIC_CHECK(!optimization_barrier_in_register<BarrierUnion>::value);
  STATIC_CHECK(!optimization_barrier_in_register<Eigen::half>::value);
  STATIC_CHECK(!optimization_barrier_in_register<Eigen::bfloat16>::value);
  STATIC_CHECK(!optimization_barrier_in_register<std::complex<double>>::value);
  STATIC_CHECK(!optimization_barrier_in_register<int[4]>::value);
#ifdef __ARM_FEATURE_SVE
  STATIC_CHECK(!optimization_barrier_in_register<svfloat64_t>::value);
#endif
#if (EIGEN_ARCH_i386_OR_x86_64 && defined(__SSE2__)) || defined(__ARM_NEON)
  using BarrierVector = float __attribute__((vector_size(16)));
  using WideBarrierVector = float __attribute__((vector_size(128)));
  STATIC_CHECK(optimization_barrier_in_register<BarrierVector>::value);
  STATIC_CHECK(!optimization_barrier_in_register<WideBarrierVector>::value);
#endif
#endif
}
