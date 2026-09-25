// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TEST_FP_CONTROL_H
#define EIGEN_TEST_FP_CONTROL_H

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>

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
// The DAZ controls live here, not in <xmmintrin.h>.
#include <pmmintrin.h>
#define EIGEN_TEST_HAS_X86_FTZ 1
#else
#define EIGEN_TEST_HAS_X86_FTZ 0
#endif

// MXCSR.DAZ is reserved, and writing it faults, on the SSE2 processors that predate SSE3; every x86-64 processor has
// it.
#if EIGEN_TEST_HAS_X86_FTZ && (EIGEN_ARCH_x86_64 || defined(__SSE3__))
#define EIGEN_TEST_HAS_X86_DAZ 1
#else
#define EIGEN_TEST_HAS_X86_DAZ 0
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

// Which subnormals a flush-to-zero mode replaces with zero: x86 controls results (MXCSR.FTZ) and inputs (MXCSR.DAZ)
// separately, while the Arm FZ and MIPS FS bits flush both.
enum class FlushToZeroMode { None = 0, Results = 1, Inputs = 2, Both = 3 };

inline bool flushToZeroModeIncludes(FlushToZeroMode mode, FlushToZeroMode part) {
  return (static_cast<int>(mode) & static_cast<int>(part)) == static_cast<int>(part);
}

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
  // Selects at least the requested flushing and clears the rest: exactly the request where results and inputs have
  // separate controls, both where one bit flushes both, gradual underflow for None. mode() is the control register
  // read back afterwards, isSupported() whether it covers the request; the default request is FTZ alone on x86.
  explicit ScopedFlushToZero(FlushToZeroMode requested = FlushToZeroMode::Results) {
#if EIGEN_TEST_HAS_RUNTIME_FTZ
    environment_saved_ = std::fegetenv(&environment_) == 0;
    const bool flush = requested != FlushToZeroMode::None;
    EIGEN_UNUSED_VARIABLE(flush);

#if EIGEN_TEST_HAS_X86_FTZ
    control_state_ = static_cast<std::uint32_t>(_mm_getcsr());
    std::uint32_t wanted = 0;
    if (flushToZeroModeIncludes(requested, FlushToZeroMode::Results)) wanted |= x86FlushToZeroMask();
#if EIGEN_TEST_HAS_X86_DAZ
    if (flushToZeroModeIncludes(requested, FlushToZeroMode::Inputs)) wanted |= x86DenormalsAreZeroMask();
#endif
    _mm_setcsr(static_cast<unsigned int>((control_state_ & ~x86FlushMask()) | wanted));
    restore_ = true;
    const std::uint32_t actual = static_cast<std::uint32_t>(_mm_getcsr()) & x86FlushMask();
    mode_ = static_cast<FlushToZeroMode>(((actual & x86FlushToZeroMask()) != 0 ? 1 : 0) |
                                         ((actual & x86DenormalsAreZeroMask()) != 0 ? 2 : 0));
#elif EIGEN_ARCH_ARM64 && defined(_MSC_VER)
    control_state_ = arm64ControlState();
    const std::uint64_t wanted = flush ? arm64FlushToZeroMask() : std::uint64_t(0);
    _WriteStatusReg(0x5a20, static_cast<__int64>((control_state_ & ~arm64FlushToZeroMask()) | wanted));
    __isb(_ARM64_BARRIER_SY);
    restore_ = true;
    mode_ = (arm64ControlState() & armFlushToZeroMask()) != 0 ? FlushToZeroMode::Both : FlushToZeroMode::None;
#elif EIGEN_ARCH_ARM64 && (defined(__GNUC__) || defined(__clang__))
    asm volatile("mrs %0, fpcr" : "=r"(control_state_));
    const std::uint64_t wanted = flush ? arm64FlushToZeroMask() : std::uint64_t(0);
    const std::uint64_t fpcr = (control_state_ & ~arm64FlushToZeroMask()) | wanted;
    asm volatile("msr fpcr, %0" : : "r"(fpcr) : "memory");
    asm volatile("isb" : : : "memory");
    restore_ = true;
    std::uint64_t actual = 0;
    asm volatile("mrs %0, fpcr" : "=r"(actual));
    mode_ = (actual & armFlushToZeroMask()) != 0 ? FlushToZeroMode::Both : FlushToZeroMode::None;
#elif EIGEN_TEST_ARCH_ARM32 && defined(_MSC_VER)
    unsigned int current_control = 0;
    if (_controlfp_s(&current_control, 0, 0) == 0) {
      control_state_ = current_control;
      restore_ = _controlfp_s(&current_control, flush ? _DN_FLUSH : _DN_SAVE, _MCW_DN) == 0;
      if (restore_ && _controlfp_s(&current_control, 0, 0) == 0) {
        mode_ = (current_control & _MCW_DN) == _DN_FLUSH ? FlushToZeroMode::Both : FlushToZeroMode::None;
      }
    }
#elif EIGEN_TEST_ARCH_ARM32 && (defined(__GNUC__) || defined(__clang__))
    std::uint32_t fpscr = 0;
    asm volatile("vmrs %0, fpscr" : "=r"(fpscr));
    control_state_ = fpscr;
    fpscr = (fpscr & ~armFlushToZeroMask()) | (flush ? armFlushToZeroMask() : std::uint32_t(0));
    asm volatile("vmsr fpscr, %0" : : "r"(fpscr) : "memory");
    asm volatile("isb" : : : "memory");
    restore_ = true;
    asm volatile("vmrs %0, fpscr" : "=r"(fpscr));
    mode_ = (fpscr & armFlushToZeroMask()) != 0 ? FlushToZeroMode::Both : FlushToZeroMode::None;
#elif EIGEN_ARCH_MIPS && defined(__mips_hard_float) && (defined(__GNUC__) || defined(__clang__))
    std::uint32_t fcsr = 0;
    asm volatile("cfc1 %0, $31" : "=r"(fcsr));
    control_state_ = fcsr;
    fcsr = (fcsr & ~mipsFlushToZeroMask()) | (flush ? mipsFlushToZeroMask() : std::uint32_t(0));
    asm volatile("ctc1 %0, $31" : : "r"(fcsr) : "memory");
#if defined(__mips_msa)
    asm volatile("cfcmsa %0, $1" : "=r"(vector_control_state_));
    const std::uint32_t msacsr =
        (vector_control_state_ & ~mipsFlushToZeroMask()) | (flush ? mipsFlushToZeroMask() : std::uint32_t(0));
    asm volatile("ctcmsa $1, %0" : : "r"(msacsr) : "memory");
#endif
    restore_ = true;
    asm volatile("cfc1 %0, $31" : "=r"(fcsr));
    mode_ = (fcsr & mipsFlushToZeroMask()) != 0 ? FlushToZeroMode::Both : FlushToZeroMode::None;
#endif
#endif
    supported_ = requested == FlushToZeroMode::None ? mode_ == FlushToZeroMode::None
                                                    : (restore_ && flushToZeroModeIncludes(mode_, requested));
  }

  ~ScopedFlushToZero() {
#if EIGEN_TEST_HAS_RUNTIME_FTZ
    if (!restore_) return;

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

  // Whether the flushing read back from the control register covers the request.
  bool isSupported() const { return supported_; }

  // The flushing the control register reads back after the constructor wrote it.
  FlushToZeroMode mode() const { return mode_; }

  // Whether the hardware currently flushes subnormal inputs to zero, read from
  // the registers the constructor writes.  On x86 that is DAZ; the FTZ bit
  // only flushes results.  The Arm and MIPS controls flush both.  A compiler
  // told it may treat subnormals as zero leaves every bit clear, so this is a
  // diagnostic, not a gate.  Returns false where there is no runtime control
  // to read.
  static bool hardwareFlushesSubnormalInputs() {
#if !EIGEN_TEST_HAS_RUNTIME_FTZ
    return false;
#elif EIGEN_TEST_HAS_X86_FTZ
    return (static_cast<std::uint32_t>(_mm_getcsr()) & x86DenormalsAreZeroMask()) != 0;
#elif EIGEN_ARCH_ARM64 && defined(_MSC_VER)
    return (arm64ControlState() & arm64FlushToZeroMask()) != 0;
#elif EIGEN_ARCH_ARM64 && (defined(__GNUC__) || defined(__clang__))
    std::uint64_t fpcr = 0;
    asm volatile("mrs %0, fpcr" : "=r"(fpcr));
    return (fpcr & arm64FlushToZeroMask()) != 0;
#elif EIGEN_TEST_ARCH_ARM32 && defined(_MSC_VER)
    unsigned int current_control = 0;
    if (_controlfp_s(&current_control, 0, 0) != 0) return false;
    return (current_control & _MCW_DN) == _DN_FLUSH;
#elif EIGEN_TEST_ARCH_ARM32 && (defined(__GNUC__) || defined(__clang__))
    std::uint32_t fpscr = 0;
    asm volatile("vmrs %0, fpscr" : "=r"(fpscr));
    return (fpscr & armFlushToZeroMask()) != 0;
#elif EIGEN_ARCH_MIPS && defined(__mips_hard_float) && (defined(__GNUC__) || defined(__clang__))
    std::uint32_t fcsr = 0;
    asm volatile("cfc1 %0, $31" : "=r"(fcsr));
    return (fcsr & mipsFlushToZeroMask()) != 0;
#else
    return false;
#endif
  }

  ScopedFlushToZero(const ScopedFlushToZero&) = delete;
  ScopedFlushToZero& operator=(const ScopedFlushToZero&) = delete;

 private:
  // MXCSR bits 15 (FTZ) and 6 (DAZ).
  static std::uint32_t x86FlushToZeroMask() { return std::uint32_t(1) << 15; }
  static std::uint32_t x86DenormalsAreZeroMask() { return std::uint32_t(1) << 6; }
  static std::uint32_t x86FlushMask() { return x86FlushToZeroMask() | x86DenormalsAreZeroMask(); }

  // FPSCR/FPCR bit 24 (FZ); the Arm64 mask adds bit 19 (FZ16) where half precision is in use.
  static std::uint32_t armFlushToZeroMask() { return std::uint32_t(1) << 24; }

  static std::uint64_t arm64FlushToZeroMask() {
    std::uint64_t mask = armFlushToZeroMask();
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    mask |= std::uint64_t(1) << 19;
#endif
    return mask;
  }

#if EIGEN_ARCH_ARM64 && defined(_MSC_VER)
  static std::uint64_t arm64ControlState() { return static_cast<std::uint64_t>(_ReadStatusReg(0x5a20)); }
#endif

  static std::uint32_t mipsFlushToZeroMask() { return std::uint32_t(1) << 24; }

  // Declared only on the paths that use them: GCC 10 ignores `unused` on data
  // members, and clang warns on a private field no member function touches.
  bool restore_ = false;
  bool supported_ = false;
  FlushToZeroMode mode_ = FlushToZeroMode::None;
#if EIGEN_TEST_HAS_RUNTIME_FTZ
  std::fenv_t environment_;
  bool environment_saved_ = false;
  std::uint64_t control_state_ = 0;
#if defined(__mips_msa)
  std::uint32_t vector_control_state_ = 0;
#endif
#endif
};

// Calls func(mode) under gradual underflow and then under each flush-to-zero mode this host selects exactly, each
// read back from the control register: the four x86 combinations, None and Both on Arm and MIPS, and None alone
// where there is no runtime control.
template <typename Func>
void forEachFlushToZeroMode(const Func& func) {
  for (const FlushToZeroMode requested :
       {FlushToZeroMode::None, FlushToZeroMode::Results, FlushToZeroMode::Inputs, FlushToZeroMode::Both}) {
    const ScopedFlushToZero scope(requested);
    if (!scope.isSupported() || scope.mode() != requested) continue;
    func(requested);
  }
}

// min * 1/2 is subnormal, so the product is exactly zero precisely when results
// are flushed to zero. The volatile operands keep the compiler from folding the
// probe under IEEE semantics the run time does not use.
template <typename Scalar>
EIGEN_DONT_INLINE Scalar underflowProbe() {
  volatile Scalar normal_min = (std::numeric_limits<Scalar>::min)();
  volatile Scalar one_half = Scalar(0.5);
  return normal_min * one_half;
}

// denorm_min * 2^digits is the normal 2^min_exponent, so the product is exactly zero precisely when subnormal inputs
// are flushed to zero (x86 DAZ, Arm FZ, MIPS FS); x86 FTZ alone leaves this normal result alone.
template <typename Scalar>
EIGEN_DONT_INLINE Scalar subnormalInputProbe() {
  volatile Scalar denorm_min = std::numeric_limits<Scalar>::denorm_min();
  volatile Scalar scale = std::ldexp(Scalar(1), std::numeric_limits<Scalar>::digits);
  return denorm_min * scale;
}

// Whether dividing a normal value by a subnormal divisor yields its IEEE 754
// quotient here.  Two things defeat it: hardware that flushes subnormal inputs
// (x86 DAZ, Arm FZ, MIPS FS) reads the divisor as zero, and a compiler
// permitted to relax floating point may rewrite `x / c` as
// `x * (Scalar(1) / c)`, whose reciprocal overflows.  The x86 FTZ bit alone
// does not: it flushes results, and this quotient is normal.
//
// The division is the packet primitive that `Matrix / scalar` lowers to, plain
// `/` where `Scalar` is not vectorized, because a compiler may relax the one
// and leave the other alone: MSVC /fp:fast rewrites scalar division but not
// _mm_div_pd.  The lanes are distinct runtime values, all of them checked, so
// the packet division cannot be narrowed back to a scalar one.  The divisor
// stays a compile-time constant, because the rewrite is a folding step the
// compiler only reaches while it can see the value.  The numerators are read
// through `volatile`, because a compiler free to fold the whole quotient never
// forms the reciprocal at all; EIGEN_OPTIMIZATION_BARRIER would not do, since
// it expands to nothing on MSVC cl.exe.  The result is compared with the exact
// quotient rather than tested for finiteness: under -ffinite-math-only GCC
// folds `quotient <= max` to true before the division runs.
template <typename Scalar>
bool subnormalDivisionIsExact() {
  using Packet = typename internal::packet_traits<Scalar>::type;
  constexpr int packet_size = internal::packet_traits<Scalar>::size;

  volatile Scalar opaque_numerator;
  Scalar numerators[packet_size];
  for (int k = 0; k < packet_size; ++k) {
    opaque_numerator = Scalar(k + 1) * (std::numeric_limits<Scalar>::min)();
    numerators[k] = opaque_numerator;
  }

  const Packet divisor = internal::pset1<Packet>((std::numeric_limits<Scalar>::denorm_min)());
  Scalar quotients[packet_size];
  internal::pstoreu(quotients, internal::pdiv<Packet>(internal::ploadu<Packet>(numerators), divisor));

  // (k + 1) * min / denorm_min == (k + 1) * 2^(digits - 1) == (k + 1) / epsilon.
  for (int k = 0; k < packet_size; ++k) {
    if (quotients[k] != Scalar(k + 1) / std::numeric_limits<Scalar>::epsilon()) return false;
  }
  return true;
}

}  // namespace Eigen

#undef EIGEN_TEST_HAS_RUNTIME_FTZ
#undef EIGEN_TEST_HAS_X86_DAZ
#undef EIGEN_TEST_HAS_X86_FTZ
#undef EIGEN_TEST_ARCH_ARM32
#undef EIGEN_TEST_DEVICE_COMPILE

#endif  // EIGEN_TEST_FP_CONTROL_H
