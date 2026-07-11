// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TEST_FP_CONTROL_H
#define EIGEN_TEST_FP_CONTROL_H

#include <cfenv>
#include <cstdint>

#include <Eigen/Core>

#if defined(EIGEN_GPU_COMPILE_PHASE) || defined(SYCL_DEVICE_ONLY)
#define EIGEN_TEST_DEVICE_COMPILE 1
#else
#define EIGEN_TEST_DEVICE_COMPILE 0
#endif

#if EIGEN_ARCH_ARM || defined(_M_ARM)
#define EIGEN_TEST_ARCH_ARM32 1
#else
#define EIGEN_TEST_ARCH_ARM32 0
#endif

#if !EIGEN_TEST_DEVICE_COMPILE && EIGEN_ARCH_i386_OR_x86_64 && \
    (defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1))
#include <xmmintrin.h>
#define EIGEN_TEST_HAS_X86_FTZ 1
#else
#define EIGEN_TEST_HAS_X86_FTZ 0
#endif

#if !EIGEN_TEST_DEVICE_COMPILE && defined(_MSC_VER) && (EIGEN_TEST_ARCH_ARM32 || EIGEN_ARCH_ARM64)
#include <float.h>
#include <intrin.h>
#endif

#if !EIGEN_TEST_DEVICE_COMPILE &&                                                              \
    (EIGEN_TEST_HAS_X86_FTZ || EIGEN_ARCH_ARM64 ||                                             \
     (EIGEN_TEST_ARCH_ARM32 && ((defined(__ARM_FP) && __ARM_FP != 0) || defined(_MSC_VER))) || \
     (EIGEN_ARCH_MIPS && defined(__mips_hard_float)))
#define EIGEN_TEST_HAS_RUNTIME_FTZ 1
#else
#define EIGEN_TEST_HAS_RUNTIME_FTZ 0
#endif

namespace Eigen {

// C++14 has no standard FTZ control. <cfenv> preserves the standard state;
// architecture-specific registers preserve and control FTZ.
//
// PowerPC has no portable FTZ mode; ZVector, LSX, and RVV have no runtime FTZ
// control; Hexagon/HVX has fixed QFloat subnormal semantics (Qualcomm Hexagon
// V68 HVX Programmer's Reference Manual). Device compiler modes are unchanged.
// Control-bit definitions follow the Intel 64 and IA-32 Architectures
// Software Developer's Manual (MXCSR), the Arm Architecture Reference Manual
// (FPSCR/FPCR FZ and FZ16), and MIPS Architecture for Programmers Volumes I
// and IV-j (FCSR/MSACSR FS).
class ScopedFlushToZero {
 public:
  ScopedFlushToZero() : environment_saved_(false), active_(false), control_state_(0), vector_control_state_(0) {
    static_cast<void>(vector_control_state_);
#if EIGEN_TEST_HAS_RUNTIME_FTZ
    environment_saved_ = std::fegetenv(&environment_) == 0;

#if EIGEN_TEST_HAS_X86_FTZ
    control_state_ = static_cast<std::uint32_t>(_mm_getcsr());
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    active_ = true;
#elif EIGEN_ARCH_ARM64 && defined(_MSC_VER)
    control_state_ = arm64ControlState();
    const std::uint64_t flush_to_zero = control_state_ | arm64FlushToZeroMask();
    _WriteStatusReg(0x5a20, static_cast<__int64>(flush_to_zero));
    __isb(_ARM64_BARRIER_SY);
    active_ = true;
#elif EIGEN_ARCH_ARM64 && (defined(__GNUC__) || defined(__clang__))
    asm volatile("mrs %0, fpcr" : "=r"(control_state_));
    const std::uint64_t flush_to_zero = control_state_ | arm64FlushToZeroMask();
    asm volatile("msr fpcr, %0" : : "r"(flush_to_zero) : "memory");
    asm volatile("isb" : : : "memory");
    active_ = true;
#elif EIGEN_TEST_ARCH_ARM32 && defined(_MSC_VER)
    unsigned int current_control = 0;
    if (_controlfp_s(&current_control, 0, 0) == 0) {
      control_state_ = current_control;
      active_ = _controlfp_s(&current_control, _DN_FLUSH, _MCW_DN) == 0;
    }
#elif EIGEN_TEST_ARCH_ARM32 && (defined(__GNUC__) || defined(__clang__))
    std::uint32_t fpscr = 0;
    asm volatile("vmrs %0, fpscr" : "=r"(fpscr));
    control_state_ = fpscr;
    fpscr |= armFlushToZeroMask();
    asm volatile("vmsr fpscr, %0" : : "r"(fpscr) : "memory");
    asm volatile("isb" : : : "memory");
    active_ = true;
#elif EIGEN_ARCH_MIPS && defined(__mips_hard_float) && (defined(__GNUC__) || defined(__clang__))
    std::uint32_t fcsr = 0;
    asm volatile("cfc1 %0, $31" : "=r"(fcsr));
    control_state_ = fcsr;
    fcsr |= mipsFlushToZeroMask();
    asm volatile("ctc1 %0, $31" : : "r"(fcsr) : "memory");
#if defined(__mips_msa)
    asm volatile("cfcmsa %0, $1" : "=r"(vector_control_state_));
    const std::uint32_t msacsr = vector_control_state_ | mipsFlushToZeroMask();
    asm volatile("ctcmsa $1, %0" : : "r"(msacsr) : "memory");
#endif
    active_ = true;
#endif
#endif
  }

  ~ScopedFlushToZero() {
#if EIGEN_TEST_HAS_RUNTIME_FTZ
    if (!active_) return;

    // Restore both forms because the C++ floating-point environment is not
    // required to represent non-standard FTZ controls.
    if (environment_saved_) {
      std::fesetenv(&environment_);
    }

#if EIGEN_TEST_HAS_X86_FTZ
    _mm_setcsr(static_cast<unsigned int>(control_state_));
#elif EIGEN_ARCH_ARM64 && defined(_MSC_VER)
    _WriteStatusReg(0x5a20, static_cast<__int64>(control_state_));
    __isb(_ARM64_BARRIER_SY);
#elif EIGEN_ARCH_ARM64 && (defined(__GNUC__) || defined(__clang__))
    asm volatile("msr fpcr, %0" : : "r"(control_state_) : "memory");
    asm volatile("isb" : : : "memory");
#elif EIGEN_TEST_ARCH_ARM32 && defined(_MSC_VER)
    unsigned int current_control = 0;
    _controlfp_s(&current_control, static_cast<unsigned int>(control_state_), _MCW_DN);
#elif EIGEN_TEST_ARCH_ARM32 && (defined(__GNUC__) || defined(__clang__))
    const std::uint32_t fpscr = static_cast<std::uint32_t>(control_state_);
    asm volatile("vmsr fpscr, %0" : : "r"(fpscr) : "memory");
    asm volatile("isb" : : : "memory");
#elif EIGEN_ARCH_MIPS && defined(__mips_hard_float) && (defined(__GNUC__) || defined(__clang__))
#if defined(__mips_msa)
    asm volatile("ctcmsa $1, %0" : : "r"(vector_control_state_) : "memory");
#endif
    const std::uint32_t fcsr = static_cast<std::uint32_t>(control_state_);
    asm volatile("ctc1 %0, $31" : : "r"(fcsr) : "memory");
#endif
#endif
  }

  bool isSupported() const { return active_; }

  ScopedFlushToZero(const ScopedFlushToZero&) = delete;
  ScopedFlushToZero& operator=(const ScopedFlushToZero&) = delete;

 private:
  static std::uint32_t armFlushToZeroMask() { return std::uint32_t(1) << 24; }

  static std::uint64_t arm64FlushToZeroMask() {
    std::uint64_t mask = std::uint64_t(1) << 24;
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    mask |= std::uint64_t(1) << 19;
#endif
    return mask;
  }

#if EIGEN_ARCH_ARM64 && defined(_MSC_VER)
  static std::uint64_t arm64ControlState() { return static_cast<std::uint64_t>(_ReadStatusReg(0x5a20)); }
#endif

  static std::uint32_t mipsFlushToZeroMask() { return std::uint32_t(1) << 24; }

  std::fenv_t environment_;
  bool environment_saved_;
  bool active_;
  std::uint64_t control_state_;
  std::uint32_t vector_control_state_;
};

}  // namespace Eigen

#undef EIGEN_TEST_HAS_RUNTIME_FTZ
#undef EIGEN_TEST_HAS_X86_FTZ
#undef EIGEN_TEST_ARCH_ARM32
#undef EIGEN_TEST_DEVICE_COMPILE

#endif  // EIGEN_TEST_FP_CONTROL_H
