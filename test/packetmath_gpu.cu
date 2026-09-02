// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Device-side packet math: test/packetmath.cpp never runs on a GPU. Thread i applies one operation to packet i
// (gpu_common.h's run_on_gpu); the host compares against a reference using bit-exact results, full-bit masks, or a
// named ULP budget. Parts: 1/3 float4/double2 core, 2/4 float4/double2 math, 5/6 half core/math, 7 scalar fallbacks
// and preinterpret.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_USE_GPU

// packetmath_test_shared.h brings in main.h, which has no include guard.
#include "packetmath_test_shared.h"
#include "gpu_common.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace {

using Eigen::Index;
using Eigen::internal::packet_traits;
namespace test = Eigen::test;

template <typename Scalar>
using Buffer = Eigen::Array<Scalar, Eigen::Dynamic, 1>;

// The scalar type and lane count of a device packet, as the host side of this test needs them: the device packet
// types are not all described by unpacket_traits in the host pass, where Packet4h2's traits do not exist at all.
// A type that is not one of the device packets stands for itself, which is how the scalar fallbacks are tested.
template <typename Packet>
struct gpu_packet {
  using Scalar = Packet;
  static constexpr int kSize = 1;
};
template <>
struct gpu_packet<float4> {
  using Scalar = float;
  static constexpr int kSize = 4;
};
template <>
struct gpu_packet<double2> {
  using Scalar = double;
  static constexpr int kSize = 2;
};
// Packet4h2 is an alias of ulonglong2, which exists in both passes; eight halves in four half2 lanes.
template <>
struct gpu_packet<ulonglong2> {
  using Scalar = Eigen::half;
  static constexpr int kSize = 8;
};

// Named budgets. rsqrt is the one approximate intrinsic among the core operations: the CUDA Math API documents
// 2 ulp for rsqrtf and 1 ulp for rsqrt, and the reference below is rounded once more from the wider type.
const uint64_t kRsqrtFloatUlps = 3;
const uint64_t kRsqrtDoubleUlps = 2;
// The CUDA Math API's documented maximum error for each function, plus one for the rounding of the reference from
// the wider type. Everything here is a named budget the test pins, so a toolkit regression is a test failure.
struct MathUlpBudget {
  uint64_t log, log1p, exp, exp2, expm1;
};
const MathUlpBudget kFloatUlps = {2, 2, 3, 3, 2};
const MathUlpBudget kDoubleUlps = {2, 2, 2, 2, 2};

// The reference is computed one type wider and rounded once.
template <typename Scalar>
struct wider_type {
  using type = double;
};
template <>
struct wider_type<double> {
  using type = long double;
};

// nvcc compiles sqrtf to a correctly rounded sqrt (-prec-sqrt=true is its default); clang as the CUDA compiler
// lowers it to an approximation one ulp off, and only the __fsqrt_rn intrinsic is exact there. Double sqrt is
// correctly rounded under both.
#if EIGEN_COMP_NVCC
const uint64_t kSqrtFloatUlps = 0;
#else
const uint64_t kSqrtFloatUlps = 1;
#endif

// ------------------------------------------------------------------------------------------------------------------
// Operations: a static run() per packet type, with device-only bodies (EIGEN_TEST_DEVICE_ONLY).

#define EIGEN_GPU_TEST_UNARY_OP(NAME, EXPR)           \
  struct NAME {                                       \
    static const char* name() { return #NAME; }       \
    template <typename P>                             \
    EIGEN_TEST_DEVICE_ONLY static P run(const P& a) { \
      return EXPR;                                    \
    }                                                 \
  };
#define EIGEN_GPU_TEST_BINARY_OP(NAME, EXPR)                      \
  struct NAME {                                                   \
    static const char* name() { return #NAME; }                   \
    template <typename P>                                         \
    EIGEN_TEST_DEVICE_ONLY static P run(const P& a, const P& b) { \
      return EXPR;                                                \
    }                                                             \
  };
#define EIGEN_GPU_TEST_TERNARY_OP(NAME, EXPR)                                 \
  struct NAME {                                                               \
    static const char* name() { return #NAME; }                               \
    template <typename P>                                                     \
    EIGEN_TEST_DEVICE_ONLY static P run(const P& a, const P& b, const P& c) { \
      return EXPR;                                                            \
    }                                                                         \
  };
#define EIGEN_GPU_TEST_REDUX_OP(NAME, EXPR)                                        \
  struct NAME {                                                                    \
    static const char* name() { return #NAME; }                                    \
    template <typename P>                                                          \
    EIGEN_TEST_DEVICE_ONLY static typename gpu_packet<P>::Scalar run(const P& a) { \
      return EXPR;                                                                 \
    }                                                                              \
  };

EIGEN_GPU_TEST_UNARY_OP(op_identity, a)
EIGEN_GPU_TEST_UNARY_OP(op_pnegate, Eigen::internal::pnegate(a))
EIGEN_GPU_TEST_UNARY_OP(op_pconj, Eigen::internal::pconj(a))
EIGEN_GPU_TEST_UNARY_OP(op_pabs, Eigen::internal::pabs(a))
EIGEN_GPU_TEST_UNARY_OP(op_pfloor, Eigen::internal::pfloor(a))
EIGEN_GPU_TEST_UNARY_OP(op_pceil, Eigen::internal::pceil(a))
EIGEN_GPU_TEST_UNARY_OP(op_print, Eigen::internal::print(a))
EIGEN_GPU_TEST_UNARY_OP(op_ptrunc, Eigen::internal::ptrunc(a))
EIGEN_GPU_TEST_UNARY_OP(op_pround, Eigen::internal::pround(a))
EIGEN_GPU_TEST_UNARY_OP(op_psqrt, Eigen::internal::psqrt(a))
EIGEN_GPU_TEST_UNARY_OP(op_prsqrt, Eigen::internal::prsqrt(a))
EIGEN_GPU_TEST_UNARY_OP(op_plog, Eigen::internal::plog(a))
EIGEN_GPU_TEST_UNARY_OP(op_plog1p, Eigen::internal::plog1p(a))
EIGEN_GPU_TEST_UNARY_OP(op_pexp, Eigen::internal::pexp(a))
EIGEN_GPU_TEST_UNARY_OP(op_pexp2, Eigen::internal::pexp2(a))
EIGEN_GPU_TEST_UNARY_OP(op_pexpm1, Eigen::internal::pexpm1(a))
EIGEN_GPU_TEST_UNARY_OP(op_preverse, Eigen::internal::preverse(a))
EIGEN_GPU_TEST_UNARY_OP(op_ptrue, Eigen::internal::ptrue(a))
EIGEN_GPU_TEST_UNARY_OP(op_pzero, Eigen::internal::pzero(a))
EIGEN_GPU_TEST_UNARY_OP(op_preinterpret_self, Eigen::internal::preinterpret<P>(a))
EIGEN_GPU_TEST_UNARY_OP(op_psign, Eigen::internal::psign(a))

EIGEN_GPU_TEST_BINARY_OP(op_padd, Eigen::internal::padd(a, b))
EIGEN_GPU_TEST_BINARY_OP(op_psub, Eigen::internal::psub(a, b))
EIGEN_GPU_TEST_BINARY_OP(op_pmul, Eigen::internal::pmul(a, b))
EIGEN_GPU_TEST_BINARY_OP(op_pdiv, Eigen::internal::pdiv(a, b))
EIGEN_GPU_TEST_BINARY_OP(op_pmin, Eigen::internal::pmin(a, b))
EIGEN_GPU_TEST_BINARY_OP(op_pmax, Eigen::internal::pmax(a, b))
EIGEN_GPU_TEST_BINARY_OP(op_pmin_numbers, Eigen::internal::pmin<Eigen::PropagateNumbers>(a, b))
EIGEN_GPU_TEST_BINARY_OP(op_pmax_numbers, Eigen::internal::pmax<Eigen::PropagateNumbers>(a, b))
EIGEN_GPU_TEST_BINARY_OP(op_pmin_nan, Eigen::internal::pmin<Eigen::PropagateNaN>(a, b))
EIGEN_GPU_TEST_BINARY_OP(op_pmax_nan, Eigen::internal::pmax<Eigen::PropagateNaN>(a, b))
EIGEN_GPU_TEST_BINARY_OP(op_pand, Eigen::internal::pand(a, b))
EIGEN_GPU_TEST_BINARY_OP(op_por, Eigen::internal::por(a, b))
EIGEN_GPU_TEST_BINARY_OP(op_pxor, Eigen::internal::pxor(a, b))
EIGEN_GPU_TEST_BINARY_OP(op_pandnot, Eigen::internal::pandnot(a, b))
EIGEN_GPU_TEST_BINARY_OP(op_pcmp_eq, Eigen::internal::pcmp_eq(a, b))
EIGEN_GPU_TEST_BINARY_OP(op_pcmp_lt, Eigen::internal::pcmp_lt(a, b))
EIGEN_GPU_TEST_BINARY_OP(op_pcmp_le, Eigen::internal::pcmp_le(a, b))
EIGEN_GPU_TEST_BINARY_OP(op_pcmp_lt_or_nan, Eigen::internal::pcmp_lt_or_nan(a, b))
EIGEN_GPU_TEST_BINARY_OP(op_pabsdiff, Eigen::internal::pabsdiff(a, b))

EIGEN_GPU_TEST_TERNARY_OP(op_pmadd, Eigen::internal::pmadd(a, b, c))
EIGEN_GPU_TEST_TERNARY_OP(op_pselect, Eigen::internal::pselect(a, b, c))

EIGEN_GPU_TEST_REDUX_OP(op_pfirst, Eigen::internal::pfirst(a))
EIGEN_GPU_TEST_REDUX_OP(op_predux, Eigen::internal::predux(a))
EIGEN_GPU_TEST_REDUX_OP(op_predux_mul, Eigen::internal::predux_mul(a))
EIGEN_GPU_TEST_REDUX_OP(op_predux_min, Eigen::internal::predux_min(a))
EIGEN_GPU_TEST_REDUX_OP(op_predux_max, Eigen::internal::predux_max(a))

// ------------------------------------------------------------------------------------------------------------------
// Kernels: thread i owns packet i. Aligned loads are legitimate because gpuMalloc returns 256-byte aligned memory
// and every packet here is at most 16 bytes.

template <typename Packet, typename Op>
struct unary_kernel {
  using Scalar = typename gpu_packet<Packet>::Scalar;
  static constexpr int kSize = gpu_packet<Packet>::kSize;
  EIGEN_TEST_DEVICE_ONLY void operator()(int i, const Scalar* in, Scalar* out) const {
#if defined(EIGEN_GPU_COMPILE_PHASE)
    Eigen::internal::pstore(out + i * kSize, Op::run(Eigen::internal::pload<Packet>(in + i * kSize)));
#endif
  }
};

// The operands of packet i are stored back to back: a at 2i, b at 2i + 1 (and c at 3i + 2 for three operands).
template <typename Packet, typename Op>
struct binary_kernel {
  using Scalar = typename gpu_packet<Packet>::Scalar;
  static constexpr int kSize = gpu_packet<Packet>::kSize;
  EIGEN_TEST_DEVICE_ONLY void operator()(int i, const Scalar* in, Scalar* out) const {
#if defined(EIGEN_GPU_COMPILE_PHASE)
    const Packet a = Eigen::internal::pload<Packet>(in + (2 * i) * kSize);
    const Packet b = Eigen::internal::pload<Packet>(in + (2 * i + 1) * kSize);
    Eigen::internal::pstore(out + i * kSize, Op::run(a, b));
#endif
  }
};

template <typename Packet, typename Op>
struct ternary_kernel {
  using Scalar = typename gpu_packet<Packet>::Scalar;
  static constexpr int kSize = gpu_packet<Packet>::kSize;
  EIGEN_TEST_DEVICE_ONLY void operator()(int i, const Scalar* in, Scalar* out) const {
#if defined(EIGEN_GPU_COMPILE_PHASE)
    const Packet a = Eigen::internal::pload<Packet>(in + (3 * i) * kSize);
    const Packet b = Eigen::internal::pload<Packet>(in + (3 * i + 1) * kSize);
    const Packet c = Eigen::internal::pload<Packet>(in + (3 * i + 2) * kSize);
    Eigen::internal::pstore(out + i * kSize, Op::run(a, b, c));
#endif
  }
};

template <typename Packet, typename Op>
struct redux_kernel {
  using Scalar = typename gpu_packet<Packet>::Scalar;
  static constexpr int kSize = gpu_packet<Packet>::kSize;
  EIGEN_TEST_DEVICE_ONLY void operator()(int i, const Scalar* in, Scalar* out) const {
#if defined(EIGEN_GPU_COMPILE_PHASE)
    out[i] = Op::run(Eigen::internal::pload<Packet>(in + i * kSize));
#endif
  }
};

// Loads and stores with their own addressing. `offset` misaligns the unaligned forms.
template <typename Packet>
struct ploadu_kernel {
  using Scalar = typename gpu_packet<Packet>::Scalar;
  static constexpr int kSize = gpu_packet<Packet>::kSize;
  int offset;
  EIGEN_TEST_DEVICE_ONLY void operator()(int i, const Scalar* in, Scalar* out) const {
#if defined(EIGEN_GPU_COMPILE_PHASE)
    Eigen::internal::pstore(out + i * kSize, Eigen::internal::ploadu<Packet>(in + i * kSize + offset));
#endif
  }
};
template <typename Packet>
struct pstoreu_kernel {
  using Scalar = typename gpu_packet<Packet>::Scalar;
  static constexpr int kSize = gpu_packet<Packet>::kSize;
  int offset;
  EIGEN_TEST_DEVICE_ONLY void operator()(int i, const Scalar* in, Scalar* out) const {
#if defined(EIGEN_GPU_COMPILE_PHASE)
    Eigen::internal::pstoreu(out + i * kSize + offset, Eigen::internal::pload<Packet>(in + i * kSize));
#endif
  }
};
template <typename Packet, int Alignment>
struct ploadt_ro_kernel {
  using Scalar = typename gpu_packet<Packet>::Scalar;
  static constexpr int kSize = gpu_packet<Packet>::kSize;
  int offset;
  EIGEN_TEST_DEVICE_ONLY void operator()(int i, const Scalar* in, Scalar* out) const {
#if defined(EIGEN_GPU_COMPILE_PHASE)
    Eigen::internal::pstore(out + i * kSize, Eigen::internal::ploadt_ro<Packet, Alignment>(in + i * kSize + offset));
#endif
  }
};
template <typename Packet>
struct ploaddup_kernel {
  using Scalar = typename gpu_packet<Packet>::Scalar;
  static constexpr int kSize = gpu_packet<Packet>::kSize;
  EIGEN_TEST_DEVICE_ONLY void operator()(int i, const Scalar* in, Scalar* out) const {
#if defined(EIGEN_GPU_COMPILE_PHASE)
    Eigen::internal::pstore(out + i * kSize, Eigen::internal::ploaddup<Packet>(in + i * (kSize / 2)));
#endif
  }
};
template <typename Packet>
struct pset1_kernel {
  using Scalar = typename gpu_packet<Packet>::Scalar;
  static constexpr int kSize = gpu_packet<Packet>::kSize;
  EIGEN_TEST_DEVICE_ONLY void operator()(int i, const Scalar* in, Scalar* out) const {
#if defined(EIGEN_GPU_COMPILE_PHASE)
    Eigen::internal::pstore(out + i * kSize, Eigen::internal::pset1<Packet>(in[i]));
#endif
  }
};
template <typename Packet>
struct plset_kernel {
  using Scalar = typename gpu_packet<Packet>::Scalar;
  static constexpr int kSize = gpu_packet<Packet>::kSize;
  EIGEN_TEST_DEVICE_ONLY void operator()(int i, const Scalar* in, Scalar* out) const {
#if defined(EIGEN_GPU_COMPILE_PHASE)
    Eigen::internal::pstore(out + i * kSize, Eigen::internal::plset<Packet>(in[i]));
#endif
  }
};
template <typename Packet>
struct pgather_kernel {
  using Scalar = typename gpu_packet<Packet>::Scalar;
  static constexpr int kSize = gpu_packet<Packet>::kSize;
  int stride;
  EIGEN_TEST_DEVICE_ONLY void operator()(int i, const Scalar* in, Scalar* out) const {
#if defined(EIGEN_GPU_COMPILE_PHASE)
    Eigen::internal::pstore(out + i * kSize, Eigen::internal::pgather<Scalar, Packet>(in + i * kSize * stride, stride));
#endif
  }
};
template <typename Packet>
struct pscatter_kernel {
  using Scalar = typename gpu_packet<Packet>::Scalar;
  static constexpr int kSize = gpu_packet<Packet>::kSize;
  int stride;
  EIGEN_TEST_DEVICE_ONLY void operator()(int i, const Scalar* in, Scalar* out) const {
#if defined(EIGEN_GPU_COMPILE_PHASE)
    Eigen::internal::pscatter<Scalar, Packet>(out + i * kSize * stride, Eigen::internal::pload<Packet>(in + i * kSize),
                                              stride);
#endif
  }
};
template <typename Packet>
struct ptranspose_kernel {
  using Scalar = typename gpu_packet<Packet>::Scalar;
  static constexpr int kSize = gpu_packet<Packet>::kSize;
  EIGEN_TEST_DEVICE_ONLY void operator()(int i, const Scalar* in, Scalar* out) const {
#if defined(EIGEN_GPU_COMPILE_PHASE)
    Eigen::internal::PacketBlock<Packet, kSize> block;
    for (int r = 0; r < kSize; ++r) block.packet[r] = Eigen::internal::pload<Packet>(in + (i * kSize + r) * kSize);
    Eigen::internal::ptranspose(block);
    for (int r = 0; r < kSize; ++r) Eigen::internal::pstore(out + (i * kSize + r) * kSize, block.packet[r]);
#endif
  }
};
// Thread i copies its first (i mod (kSize + 1)) lanes; the rest of the output keeps its sentinel.
template <typename Packet>
struct partial_kernel {
  using Scalar = typename gpu_packet<Packet>::Scalar;
  static constexpr int kSize = gpu_packet<Packet>::kSize;
  EIGEN_TEST_DEVICE_ONLY void operator()(int i, const Scalar* in, Scalar* out) const {
#if defined(EIGEN_GPU_COMPILE_PHASE)
    const Index n = i % (kSize + 1);
    Eigen::internal::pstore_partial(out + i * kSize, Eigen::internal::pload_partial<Packet>(in + i * kSize, n), n);
#endif
  }
};
// preinterpret between a scalar and its same-size integer, as the evaluator's cast path uses it on the device.
template <typename Scalar, typename Bits>
struct preinterpret_scalar_kernel {
  EIGEN_TEST_DEVICE_ONLY void operator()(int i, const Scalar* in, Bits* out) const {
#if defined(EIGEN_GPU_COMPILE_PHASE)
    out[i] = Eigen::internal::preinterpret<Bits>(in[i]);
#endif
  }
};

// Casts. The evaluator hands the narrowing form two source packets and the widening form one, from which it
// takes the leading DstPacketSize lanes; these kernels use exactly those two forms.
struct half_to_float_kernel {
  EIGEN_TEST_DEVICE_ONLY void operator()(int i, const Eigen::half* in, float* out) const {
#if defined(EIGEN_GPU_COMPILE_PHASE)
    const ulonglong2 a = Eigen::internal::pload<ulonglong2>(in + 8 * i);
    Eigen::internal::pstore(out + 4 * i, Eigen::internal::pcast<ulonglong2, float4>(a));
#else
    EIGEN_UNUSED_VARIABLE(i);
    EIGEN_UNUSED_VARIABLE(in);
    EIGEN_UNUSED_VARIABLE(out);
#endif
  }
};

struct float_to_half_kernel {
  EIGEN_TEST_DEVICE_ONLY void operator()(int i, const float* in, Eigen::half* out) const {
#if defined(EIGEN_GPU_COMPILE_PHASE)
    const float4 a = Eigen::internal::pload<float4>(in + 8 * i);
    const float4 b = Eigen::internal::pload<float4>(in + 8 * i + 4);
    Eigen::internal::pstore(out + 8 * i, Eigen::internal::pcast<float4, ulonglong2>(a, b));
#else
    EIGEN_UNUSED_VARIABLE(i);
    EIGEN_UNUSED_VARIABLE(in);
    EIGEN_UNUSED_VARIABLE(out);
#endif
  }
};

// The device pass's packet traits, written into an int array: the flags the host selects operations by are the
// device's, not the host pass's, which differ under nvcc (EIGEN_HAS_GPU_DEVICE_FUNCTIONS).
#define EIGEN_GPU_TEST_TRAIT_FLAGS(X) \
  X(Vectorizable)                     \
  X(size)                             \
  X(HasAdd)                           \
  X(HasSub)                           \
  X(HasMul)                           \
  X(HasDiv)                           \
  X(HasNegate)                        \
  X(HasAbs)                           \
  X(HasMin)                           \
  X(HasMax)                           \
  X(HasCmp)                           \
  X(HasRound)                         \
  X(HasSqrt)                          \
  X(HasRsqrt)                         \
  X(HasSign) X(HasAbsDiff) X(HasSetLinear) X(HasConj) X(HasReciprocal) X(HasExp) X(HasExpm1) X(HasLog) X(HasLog1p)
#define EIGEN_GPU_TEST_TRAIT_NAME(FLAG) #FLAG,
#define EIGEN_GPU_TEST_TRAIT_ENUM(FLAG) k##FLAG,
#define EIGEN_GPU_TEST_TRAIT_VALUE(FLAG) out[k++] = static_cast<int>(packet_traits<Scalar>::FLAG);
const char* const kTraitNames[] = {EIGEN_GPU_TEST_TRAIT_FLAGS(EIGEN_GPU_TEST_TRAIT_NAME)};
enum Trait { EIGEN_GPU_TEST_TRAIT_FLAGS(EIGEN_GPU_TEST_TRAIT_ENUM) kNumTraits };

template <typename Scalar>
struct traits_kernel {
  EIGEN_TEST_DEVICE_ONLY void operator()(int i, const int*, int* out) const {
#if defined(EIGEN_GPU_COMPILE_PHASE)
    if (i != 0) return;
    int k = 0;
    EIGEN_GPU_TEST_TRAIT_FLAGS(EIGEN_GPU_TEST_TRAIT_VALUE)
#endif
  }
};

template <typename Scalar>
std::vector<int> device_traits() {
  Buffer<int> dummy(1), report(kNumTraits);
  dummy.setZero();  // the kernel ignores it, but it is still copied to the device
  report.setConstant(-1);
  run_on_gpu(traits_kernel<Scalar>(), 1, dummy, report);
  return std::vector<int>(report.data(), report.data() + kNumTraits);
}

// Every flag the device advertises must be covered by a part of this test or deferred to a reserved part by name.
template <typename Scalar>
void check_advertised_ops_are_covered(const std::vector<int>& traits, const std::vector<Trait>& covered_here) {
  const std::vector<Trait> deferred = {kHasExp, kHasExpm1, kHasLog, kHasLog1p};
  for (int k = kHasAdd; k < kNumTraits; ++k) {
    const Trait flag = static_cast<Trait>(k);
    if (traits[k] == 0) continue;
    const auto covered = [flag](const std::vector<Trait>& list) {
      return std::find(list.begin(), list.end(), flag) != list.end();
    };
    const bool ok = covered(covered_here) || covered(deferred);
    if (!ok) std::cout << "advertised device op without a test: " << kTraitNames[k] << std::endl;
    VERIFY(ok);
  }
}

// ------------------------------------------------------------------------------------------------------------------
// Inputs.

template <typename Scalar>
std::vector<Scalar> special_values() {
  using L = std::numeric_limits<Scalar>;
  const Scalar largest_subnormal = (L::min)() - L::denorm_min();
  return {Scalar(0),
          Scalar(-0.0),
          Scalar(1),
          Scalar(-1),
          Scalar(0.5),
          Scalar(-0.5),
          Scalar(2),
          Scalar(-2),
          Scalar(3),
          Scalar(-3),
          Scalar(0.25),
          Scalar(0.75),
          Scalar(1.5),
          Scalar(-1.5),
          Scalar(2.5),
          Scalar(-2.5),
          Scalar(-0.4),
          Scalar(0.4),
          Scalar(1e-3),
          Scalar(-1e-3),
          L::epsilon(),
          -L::epsilon(),
          Scalar(1) + L::epsilon(),
          Scalar(1) - L::epsilon() / Scalar(2),
          L::denorm_min(),
          -L::denorm_min(),
          largest_subnormal,
          -largest_subnormal,
          (L::min)(),
          -(L::min)(),
          (L::max)(),
          -(L::max)(),
          L::infinity(),
          -L::infinity(),
          L::quiet_NaN(),
          -L::quiet_NaN()};
}

// Every exponent step of the type, at a few mantissas, both signs.
template <typename Scalar>
std::vector<Scalar> exponent_grid(int exponent_step) {
  using L = std::numeric_limits<Scalar>;
  std::vector<Scalar> values;
  const Scalar mantissas[] = {Scalar(1), Scalar(1.25), Scalar(1.5), Scalar(1.75), Scalar(2) - L::epsilon()};
  for (int e = L::min_exponent - L::digits + 1; e <= L::max_exponent - 1; e += exponent_step) {
    for (Scalar m : mantissas) {
      const Scalar v = std::ldexp(m, e);
      if ((std::isfinite)(v) && v != Scalar(0)) {
        values.push_back(v);
        values.push_back(-v);
      }
    }
  }
  return values;
}

template <typename Scalar>
std::vector<Scalar> random_values(int count) {
  std::vector<Scalar> values(count);
  for (Scalar& v : values) {
    v = Eigen::internal::random<Scalar>(Scalar(-1), Scalar(1)) *
        std::ldexp(Scalar(1), Eigen::internal::random<int>(-30, 30));
  }
  return values;
}

// The special values at every lane position (a run of ones shifts them), then the grid and the random values,
// padded to whole packets with ones.
template <typename Scalar>
Buffer<Scalar> unary_inputs(int packet_size, int random_count) {
  std::vector<Scalar> values;
  const std::vector<Scalar> specials = special_values<Scalar>();
  for (int shift = 0; shift < packet_size; ++shift) {
    values.insert(values.end(), shift, Scalar(1));
    values.insert(values.end(), specials.begin(), specials.end());
  }
  const std::vector<Scalar> grid = exponent_grid<Scalar>(std::is_same<Scalar, double>::value ? 8 : 2);
  values.insert(values.end(), grid.begin(), grid.end());
  const std::vector<Scalar> random = random_values<Scalar>(random_count);
  values.insert(values.end(), random.begin(), random.end());
  while (values.size() % packet_size != 0) values.push_back(Scalar(1));
  return Eigen::Map<const Buffer<Scalar>>(values.data(), values.size());
}

template <typename Scalar>
struct binary_inputs {
  std::vector<Scalar> a, b;
  binary_inputs() = default;
  // The cross product of the special values, then random pairs, then a shifted copy of the specials so that a
  // pair reaches every lane. `exclude` drops pairs the operation leaves implementation-defined.
  template <typename Exclude>
  binary_inputs(int packet_size, int random_count, Exclude exclude) {
    const std::vector<Scalar> specials = special_values<Scalar>();
    for (int shift = 0; shift < packet_size; ++shift) {
      for (int k = 0; k < shift; ++k) push(Scalar(1), Scalar(1));
      for (Scalar x : specials) {
        for (Scalar y : specials) {
          if (!exclude(x, y)) push(x, y);
        }
      }
    }
    const std::vector<Scalar> ra = random_values<Scalar>(random_count);
    const std::vector<Scalar> rb = random_values<Scalar>(random_count);
    for (int k = 0; k < random_count; ++k) {
      if (!exclude(ra[k], rb[k])) push(ra[k], rb[k]);
    }
    while (a.size() % packet_size != 0) push(Scalar(1), Scalar(1));
  }
  void push(Scalar x, Scalar y) {
    a.push_back(x);
    b.push_back(y);
  }
  int size() const { return int(a.size()); }
  // Packet-interleaved device layout: a-packet, b-packet, a-packet, ...
  Buffer<Scalar> interleaved(int packet_size) const {
    Buffer<Scalar> in(2 * size());
    for (int p = 0; p < size() / packet_size; ++p) {
      for (int l = 0; l < packet_size; ++l) {
        in[(2 * p) * packet_size + l] = a[p * packet_size + l];
        in[(2 * p + 1) * packet_size + l] = b[p * packet_size + l];
      }
    }
    return in;
  }
};

// Every value a half can hold, in bit-pattern order. 65536 lanes is 8192 packets, which the device sweeps in
// microseconds, so the unary operations are tested exhaustively rather than sampled.
std::vector<Eigen::half> all_half_values() {
  std::vector<Eigen::half> values;
  values.reserve(1 << 16);
  for (int bits = 0; bits < (1 << 16); ++bits) {
    values.push_back(Eigen::numext::bit_cast<Eigen::half>(static_cast<Eigen::numext::uint16_t>(bits)));
  }
  return values;
}

// Pairs covering the whole domain in both operands: every value against a value a stride away, plus the cross
// product of the special values. 40503 is coprime with 65536, so the second operand also runs over everything.
binary_inputs<Eigen::half> all_half_pairs(bool exclude_nan) {
  const std::vector<Eigen::half> values = all_half_values();
  binary_inputs<Eigen::half> pairs;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const Eigen::half a = values[i], b = values[(i * 40503) % values.size()];
    if (exclude_nan && ((Eigen::numext::isnan)(a) || (Eigen::numext::isnan)(b))) continue;
    pairs.push(a, b);
  }
  const std::vector<float> specials = special_values<float>();
  for (float x : specials) {
    for (float y : specials) {
      const Eigen::half a(x), b(y);
      if (exclude_nan && ((Eigen::numext::isnan)(a) || (Eigen::numext::isnan)(b))) continue;
      pairs.push(a, b);
    }
  }
  while (pairs.size() % 8 != 0) pairs.push(Eigen::half(1), Eigen::half(1));
  return pairs;
}

template <typename Scalar>
bool never(Scalar, Scalar) {
  return false;
}
template <typename Scalar>
bool either_nan(Scalar x, Scalar y) {
  return (std::isnan)(x) || (std::isnan)(y);
}

// ------------------------------------------------------------------------------------------------------------------
// Comparators.

// Bit-level view of a scalar, for the reference side of the bitwise operations.
template <typename Scalar>
using scalar_bits_t = typename Eigen::numext::get_integer_by_size<sizeof(Scalar)>::unsigned_type;
template <typename Scalar>
scalar_bits_t<Scalar> bits_of(Scalar x) {
  return Eigen::numext::bit_cast<scalar_bits_t<Scalar>>(x);
}
template <typename Scalar>
Scalar from_bits(scalar_bits_t<Scalar> b) {
  return Eigen::numext::bit_cast<Scalar>(b);
}

template <typename Scalar>
struct compare_bits {
  bool nan_is_nan;
  bool operator()(const Scalar* ref, const Scalar* vec, int n) const {
    return test::areEqualBits(ref, vec, n, nan_is_nan);
  }
};
// Value equality: +0 and -0 agree, NaN matches NaN.
template <typename Scalar>
struct compare_values {
  bool operator()(const Scalar* ref, const Scalar* vec, int n) const { return test::areEqual(ref, vec, n); }
};
template <typename Scalar>
struct compare_ulps {
  uint64_t ulps;
  bool operator()(const Scalar* ref, const Scalar* vec, int n) const { return test::areWithinUlps(ref, vec, n, ulps); }
};

// ------------------------------------------------------------------------------------------------------------------
// Drivers.

// VERIFY with the operation's name in the failure report.
#define VERIFY_OP(COND)                                                      \
  do {                                                                       \
    const bool ok_ = (COND);                                                 \
    if (!ok_) std::cout << "failing operation: " << Op::name() << std::endl; \
    VERIFY(ok_);                                                             \
  } while (0)

template <typename Packet, typename Op, typename Ref, typename Compare>
void check_unary(const Buffer<typename gpu_packet<Packet>::Scalar>& in, Ref ref, Compare compare) {
  using Scalar = typename gpu_packet<Packet>::Scalar;
  const int kSize = gpu_packet<Packet>::kSize;
  Buffer<Scalar> out(in.size());
  out.setConstant(Scalar(-7));
  run_on_gpu(unary_kernel<Packet, Op>(), int(in.size()) / kSize, in, out);
  Buffer<Scalar> expected(in.size());
  for (Index k = 0; k < in.size(); ++k) expected[k] = ref(in[k]);
  VERIFY_OP(compare(expected.data(), out.data(), int(in.size())));
}

template <typename Packet, typename Op, typename Ref, typename Compare>
void check_binary(const binary_inputs<typename gpu_packet<Packet>::Scalar>& inputs, Ref ref, Compare compare) {
  using Scalar = typename gpu_packet<Packet>::Scalar;
  const int kSize = gpu_packet<Packet>::kSize;
  const Buffer<Scalar> in = inputs.interleaved(kSize);
  Buffer<Scalar> out(inputs.size());
  out.setConstant(Scalar(-7));
  run_on_gpu(binary_kernel<Packet, Op>(), inputs.size() / kSize, in, out);
  Buffer<Scalar> expected(inputs.size());
  for (int k = 0; k < inputs.size(); ++k) expected[k] = ref(inputs.a[k], inputs.b[k]);
  VERIFY_OP(compare(expected.data(), out.data(), inputs.size()));
}

// A comparison must return an all-ones lane where the predicate holds and an all-zero lane elsewhere.
template <typename Packet, typename Op, typename Pred>
void check_compare(const binary_inputs<typename gpu_packet<Packet>::Scalar>& inputs, Pred pred) {
  using Scalar = typename gpu_packet<Packet>::Scalar;
  const int kSize = gpu_packet<Packet>::kSize;
  const Buffer<Scalar> in = inputs.interleaved(kSize);
  Buffer<Scalar> out(inputs.size());
  out.setConstant(Scalar(-7));
  run_on_gpu(binary_kernel<Packet, Op>(), inputs.size() / kSize, in, out);
  Buffer<bool> zero_mask(inputs.size());
  for (int k = 0; k < inputs.size(); ++k) zero_mask[k] = !pred(inputs.a[k], inputs.b[k]);
  VERIFY_OP(test::areFullBitMasks(out.data(), zero_mask.data(), inputs.size()));
}

template <typename Packet, typename Op, typename Compare>
void check_ternary(const std::vector<typename gpu_packet<Packet>::Scalar>& a,
                   const std::vector<typename gpu_packet<Packet>::Scalar>& b,
                   const std::vector<typename gpu_packet<Packet>::Scalar>& c,
                   const Buffer<typename gpu_packet<Packet>::Scalar>& expected, Compare compare) {
  using Scalar = typename gpu_packet<Packet>::Scalar;
  const int kSize = gpu_packet<Packet>::kSize;
  const int n = int(a.size());
  Buffer<Scalar> in(3 * n);
  for (int p = 0; p < n / kSize; ++p) {
    for (int l = 0; l < kSize; ++l) {
      in[(3 * p) * kSize + l] = a[p * kSize + l];
      in[(3 * p + 1) * kSize + l] = b[p * kSize + l];
      in[(3 * p + 2) * kSize + l] = c[p * kSize + l];
    }
  }
  Buffer<Scalar> out(n);
  out.setConstant(Scalar(-7));
  run_on_gpu(ternary_kernel<Packet, Op>(), n / kSize, in, out);
  VERIFY_OP(compare(expected.data(), out.data(), n));
}

template <typename Packet, typename Op, typename Ref, typename Compare>
void check_redux(const Buffer<typename gpu_packet<Packet>::Scalar>& in, Ref ref, Compare compare) {
  using Scalar = typename gpu_packet<Packet>::Scalar;
  const int kSize = gpu_packet<Packet>::kSize;
  const int n = int(in.size()) / kSize;
  Buffer<Scalar> out(n);
  out.setConstant(Scalar(-7));
  run_on_gpu(redux_kernel<Packet, Op>(), n, in, out);
  Buffer<Scalar> expected(n);
  for (int p = 0; p < n; ++p) expected[p] = ref(in.data() + p * kSize);
  VERIFY_OP(compare(expected.data(), out.data(), n));
}

// Data movement: what goes in comes out, lane for lane. `nan_is_nan` says how a NaN is compared, since the float
// packets carry a payload through these unchanged while the half2 intrinsics quiet a signalling NaN. `plset_ref`
// supplies the type's own addition, because a half sum has to round exactly once.
template <typename Packet, typename PlsetRef>
void check_memory_ops(const Buffer<typename gpu_packet<Packet>::Scalar>& in, bool nan_is_nan, PlsetRef plset_ref) {
  using Scalar = typename gpu_packet<Packet>::Scalar;
  const int kSize = gpu_packet<Packet>::kSize;
  const int n = int(in.size()) / kSize;
  Buffer<Scalar> out(in.size()), expected(in.size());
  const auto matches = [nan_is_nan](const Scalar* ref, const Scalar* got, int count) {
    return test::areEqualBits(ref, got, count, nan_is_nan);
  };
  const auto reset = [&out]() { out.setConstant(Scalar(-7)); };

  reset();
  run_on_gpu(unary_kernel<Packet, op_preverse>(), n, in, out);
  for (Index k = 0; k < in.size(); ++k) expected[k] = in[(k / kSize) * kSize + (kSize - 1 - k % kSize)];
  VERIFY(matches(expected.data(), out.data(), int(in.size())) && "preverse");

  reset();
  run_on_gpu(pset1_kernel<Packet>(), n, in, out);
  for (Index k = 0; k < in.size(); ++k) expected[k] = in[k / kSize];
  VERIFY(matches(expected.data(), out.data(), int(in.size())) && "pset1");

  reset();
  run_on_gpu(ploaddup_kernel<Packet>(), n, in, out);
  for (Index k = 0; k < in.size(); ++k) expected[k] = in[(k / kSize) * (kSize / 2) + (k % kSize) / 2];
  VERIFY(matches(expected.data(), out.data(), int(in.size())) && "ploaddup");

  // The lane index is added in the packet's own arithmetic, which for a NaN input may not preserve the payload,
  // so this one comparison is NaN-tolerant whatever the caller asked for.
  {
    // Every input must be used as a base, including large half values and negative zero.
    Buffer<Scalar> linear_out(in.size() * kSize), linear_expected(in.size() * kSize);
    linear_out.setConstant(Scalar(-7));
    run_on_gpu(plset_kernel<Packet>(), int(in.size()), in, linear_out);
    for (Index k = 0; k < linear_out.size(); ++k)
      linear_expected[k] = (k % kSize == 0) ? in[k / kSize] : plset_ref(in[k / kSize], int(k % kSize));
    VERIFY(test::areEqualBits(linear_expected.data(), linear_out.data(), int(linear_out.size())) && "plset");
  }

  for (int offset = 1; offset < kSize; ++offset) {
    Buffer<Scalar> padded(in.size() + kSize);
    padded << in, Buffer<Scalar>::Constant(kSize, Scalar(1));
    reset();
    run_on_gpu(ploadu_kernel<Packet>{offset}, n, padded, out);
    VERIFY(matches(padded.data() + offset, out.data(), int(in.size())) && "ploadu");
    Buffer<Scalar> out_padded(in.size() + kSize);
    out_padded.setConstant(Scalar(-7));
    run_on_gpu(pstoreu_kernel<Packet>{offset}, n, in, out_padded);
    VERIFY(matches(in.data(), out_padded.data() + offset, int(in.size())) && "pstoreu");
    VERIFY(out_padded[0] == Scalar(-7) && out_padded[in.size() + kSize - 1] == Scalar(-7) && "pstoreu bounds");
  }

  {
    const int stride = 3;
    Buffer<Scalar> strided(in.size() * stride);
    strided.setConstant(Scalar(-7));
    for (Index k = 0; k < in.size(); ++k) strided[k * stride] = in[k];
    reset();
    run_on_gpu(pgather_kernel<Packet>{stride}, n, strided, out);
    VERIFY(matches(in.data(), out.data(), int(in.size())) && "pgather");
    Buffer<Scalar> scattered(in.size() * stride);
    scattered.setConstant(Scalar(-7));
    run_on_gpu(pscatter_kernel<Packet>{stride}, n, in, scattered);
    VERIFY(matches(strided.data(), scattered.data(), int(strided.size())) && "pscatter");
  }

  {
    // A kSize x kSize block per thread.
    const int blocks = n / kSize;
    const int block_elements = kSize * kSize;
    Buffer<Scalar> transposed(blocks * block_elements);
    transposed.setConstant(Scalar(-7));
    run_on_gpu(ptranspose_kernel<Packet>(), blocks, in, transposed);
    Buffer<Scalar> want(blocks * block_elements);
    for (int block = 0; block < blocks; ++block) {
      for (int r = 0; r < kSize; ++r) {
        for (int c = 0; c < kSize; ++c) {
          want[block * block_elements + r * kSize + c] = in[block * block_elements + c * kSize + r];
        }
      }
    }
    VERIFY(matches(want.data(), transposed.data(), int(want.size())) && "ptranspose");
  }
}

// ptrue and pzero: every bit of every lane set, or every bit cleared.
template <typename Packet, typename Op>
void check_full_mask(const Buffer<typename gpu_packet<Packet>::Scalar>& in, bool expect_zero) {
  using Scalar = typename gpu_packet<Packet>::Scalar;
  Buffer<Scalar> out(in.size());
  out.setConstant(Scalar(-7));
  run_on_gpu(unary_kernel<Packet, Op>(), int(in.size()) / gpu_packet<Packet>::kSize, in, out);
  const Buffer<bool> zero_mask = Buffer<bool>::Constant(in.size(), expect_zero);
  VERIFY_OP(test::areFullBitMasks(out.data(), zero_mask.data(), int(in.size())));
}

// Points where the standard fixes the result exactly, whatever the implementation's accuracy elsewhere. The
// comparison is bitwise, so the sign of a zero counts; NaN matches NaN whatever the payload.
template <typename Packet, typename Op>
void check_special_values(
    const std::vector<std::pair<typename gpu_packet<Packet>::Scalar, typename gpu_packet<Packet>::Scalar>>& cases) {
  using Scalar = typename gpu_packet<Packet>::Scalar;
  const int kSize = gpu_packet<Packet>::kSize;
  std::vector<Scalar> in_values, expected_values;
  for (const auto& one : cases) {
    in_values.push_back(one.first);
    expected_values.push_back(one.second);
  }
  // Pad to a whole number of packets by repeating the first case; only the real cases are compared.
  while (in_values.size() % kSize != 0) {
    in_values.push_back(cases[0].first);
    expected_values.push_back(cases[0].second);
  }
  const Buffer<Scalar> in = Eigen::Map<const Buffer<Scalar>>(in_values.data(), in_values.size());
  Buffer<Scalar> out(in.size());
  out.setConstant(Scalar(-7));
  run_on_gpu(unary_kernel<Packet, Op>(), int(in.size()) / kSize, in, out);
  VERIFY_OP(test::areEqualBits(expected_values.data(), out.data(), int(cases.size())));
}

// ------------------------------------------------------------------------------------------------------------------
// Parts 2 and 4: the transcendental operations. Accuracy over the ordinary range is a named ULP budget against a
// reference computed one type wider; the values the standard fixes are checked exactly.

template <typename Scalar>
void packetmath_gpu_real_math() {
  using Packet = typename packet_traits<Scalar>::type;
  using Wider = typename wider_type<Scalar>::type;
  const int kSize = gpu_packet<Packet>::kSize;
  const MathUlpBudget& budget = std::is_same<Scalar, double>::value ? kDoubleUlps : kFloatUlps;

  const std::vector<int> traits = device_traits<Scalar>();
  VERIFY_IS_EQUAL(traits[kHasExp], 1);
  VERIFY_IS_EQUAL(traits[kHasLog], 1);

  const Scalar zero(0), one(1), inf = std::numeric_limits<Scalar>::infinity();
  const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
  const Scalar lowest = std::numeric_limits<Scalar>::lowest(), largest = (std::numeric_limits<Scalar>::max)();

  // Inputs the logarithms accept: positive, plus the special values, which the budgeted comparison tolerates
  // because a NaN reference matches a NaN result.
  const Buffer<Scalar> in = unary_inputs<Scalar>(kSize, 1 << 18);
  Buffer<Scalar> positive(in.size());
  for (Index k = 0; k < in.size(); ++k) positive[k] = std::abs(in[k]);
  // log1p's domain is (-1, +inf), and the interesting half of it is the run just above -1: |x| - 0.5 never
  // reaches below -0.5, so it would leave the argument reduction there untested. Even lanes therefore hold
  // -1 + u with u drawn log-uniformly over [2^-digits, 1), which walks (-1, 0) from a single ulp above -1 up to
  // 0; odd lanes keep |x| - 0.5, which covers (-0.5, +inf) and carries the special values.
  Buffer<Scalar> above_minus_one(in.size());
  for (Index k = 0; k < in.size(); ++k) {
    if (k % 2 == 0) {
      const int exponent = Eigen::internal::random<int>(1, Eigen::NumTraits<Scalar>::digits());
      const Scalar significand = Eigen::internal::random<Scalar>(Scalar(1), Scalar(2));
      above_minus_one[k] = Scalar(-1) + std::ldexp(significand, -exponent);
    } else {
      above_minus_one[k] = std::abs(in[k]) - Scalar(0.5);
    }
  }

  check_unary<Packet, op_pexp>(
      in, [](Scalar x) { return static_cast<Scalar>(std::exp(static_cast<Wider>(x))); },
      compare_ulps<Scalar>{budget.exp});
  check_unary<Packet, op_pexp2>(
      in, [](Scalar x) { return static_cast<Scalar>(std::exp2(static_cast<Wider>(x))); },
      compare_ulps<Scalar>{budget.exp2});
  check_unary<Packet, op_pexpm1>(
      in, [](Scalar x) { return static_cast<Scalar>(std::expm1(static_cast<Wider>(x))); },
      compare_ulps<Scalar>{budget.expm1});
  check_unary<Packet, op_plog>(
      positive, [](Scalar x) { return static_cast<Scalar>(std::log(static_cast<Wider>(x))); },
      compare_ulps<Scalar>{budget.log});
  check_unary<Packet, op_plog1p>(
      above_minus_one, [](Scalar x) { return static_cast<Scalar>(std::log1p(static_cast<Wider>(x))); },
      compare_ulps<Scalar>{budget.log1p});

  check_special_values<Packet, op_pexp>(
      {{zero, one}, {-zero, one}, {-inf, zero}, {inf, inf}, {nan, nan}, {lowest, zero}, {largest, inf}});
  check_special_values<Packet, op_pexp2>(
      {{zero, one}, {-zero, one}, {one, Scalar(2)}, {-inf, zero}, {inf, inf}, {nan, nan}});
  // expm1 preserves the sign of a zero, and saturates to -1 at minus infinity.
  check_special_values<Packet, op_pexpm1>(
      {{zero, zero}, {-zero, -zero}, {-inf, Scalar(-1)}, {inf, inf}, {nan, nan}, {lowest, Scalar(-1)}});
  check_special_values<Packet, op_plog>(
      {{one, zero}, {zero, -inf}, {-zero, -inf}, {-one, nan}, {-inf, nan}, {inf, inf}, {nan, nan}});
  check_special_values<Packet, op_plog1p>(
      {{zero, zero}, {-zero, -zero}, {Scalar(-1), -inf}, {Scalar(-2), nan}, {-inf, nan}, {inf, inf}, {nan, nan}});
}

// ------------------------------------------------------------------------------------------------------------------
// The core of a floating-point packet type: loads and stores, arithmetic, min/max, rounding, bit operations,
// comparisons, select, reductions, sqrt and rsqrt.

template <typename Scalar>
Scalar rsqrt_reference(Scalar x) {
  using Wider = typename wider_type<Scalar>::type;
  return static_cast<Scalar>(Wider(1) / std::sqrt(static_cast<Wider>(x)));
}

template <typename Scalar>
void packetmath_gpu_real_core() {
  using Packet = typename packet_traits<Scalar>::type;
  const int kSize = gpu_packet<Packet>::kSize;
  using Bits = typename Eigen::numext::get_integer_by_size<sizeof(Scalar)>::unsigned_type;
  const compare_bits<Scalar> bits{true};
  const compare_bits<Scalar> bits_and_payload{false};
  const compare_values<Scalar> values;
  const uint64_t rsqrt_ulps = std::is_same<Scalar, double>::value ? kRsqrtDoubleUlps : kRsqrtFloatUlps;

  // The device's view of the type, and what this part covers of it.
  const std::vector<int> traits = device_traits<Scalar>();
  std::cout << "device packet_traits<" << typeid(Scalar).name() << ">:";
  for (int k = 0; k < kNumTraits; ++k) std::cout << " " << kTraitNames[k] << "=" << traits[k];
  std::cout << std::endl;
  VERIFY_IS_EQUAL(traits[kVectorizable], 1);
  VERIFY_IS_EQUAL(traits[ksize], kSize);
  VERIFY_IS_EQUAL(traits[kHasCmp], 1);
  check_advertised_ops_are_covered<Scalar>(
      traits, {kHasAdd, kHasSub, kHasMul, kHasDiv, kHasNegate, kHasAbs, kHasMin, kHasMax, kHasCmp, kHasRound, kHasSqrt,
               kHasRsqrt, kHasSign, kHasAbsDiff, kHasSetLinear, kHasConj});

  const Buffer<Scalar> in = unary_inputs<Scalar>(kSize, 1 << 18);
  const int n = int(in.size()) / kSize;

  // Loads and stores keep every bit, NaN payloads included.
  check_unary<Packet, op_identity>(
      in, [](Scalar x) { return x; }, bits_and_payload);
  check_memory_ops<Packet>(in, /*nan_is_nan=*/false, [](Scalar base, int lane) { return Scalar(base + Scalar(lane)); });

  // Read-only loads and the partial load/store pair, which the half packets do not have.
  for (int offset = 1; offset < kSize; ++offset) {
    Buffer<Scalar> padded(in.size() + kSize);
    padded << in, Buffer<Scalar>::Constant(kSize, Scalar(1));
    Buffer<Scalar> out(in.size());
    out.setConstant(Scalar(-7));
    run_on_gpu(ploadt_ro_kernel<Packet, Eigen::Unaligned>{offset}, n, padded, out);
    VERIFY(test::areEqualBits(padded.data() + offset, out.data(), int(in.size()), false) && "ploadt_ro<Unaligned>");
  }
  {
    Buffer<Scalar> out(in.size());
    out.setConstant(Scalar(-7));
    run_on_gpu(ploadt_ro_kernel<Packet, Eigen::Aligned>{0}, n, in, out);
    VERIFY(test::areEqualBits(in.data(), out.data(), int(in.size()), false) && "ploadt_ro<Aligned>");
    Buffer<Scalar> expected(in.size());
    out.setConstant(Scalar(-7));
    run_on_gpu(partial_kernel<Packet>(), n, in, out);
    for (Index k = 0; k < in.size(); ++k) {
      const Index lanes = (k / kSize) % (kSize + 1);
      expected[k] = (k % kSize) < lanes ? in[k] : Scalar(-7);
    }
    VERIFY(test::areEqualBits(expected.data(), out.data(), int(in.size()), false) && "pload_partial/pstore_partial");
  }

  // Sign manipulation and rounding: fixed bit for bit, including the sign of a zero.
  check_unary<Packet, op_pnegate>(
      in, [](Scalar x) { return -x; }, bits);
  check_unary<Packet, op_pconj>(
      in, [](Scalar x) { return x; }, bits);
  check_unary<Packet, op_pabs>(
      in, [](Scalar x) { return std::abs(x); }, bits);
  check_unary<Packet, op_pfloor>(
      in, [](Scalar x) { return std::floor(x); }, bits);
  check_unary<Packet, op_pceil>(
      in, [](Scalar x) { return std::ceil(x); }, bits);
  check_unary<Packet, op_print>(
      in, [](Scalar x) { return std::rint(x); }, bits);
  check_unary<Packet, op_ptrunc>(
      in, [](Scalar x) { return std::trunc(x); }, bits);
  check_unary<Packet, op_pround>(
      in, [](Scalar x) { return std::round(x); }, bits);
  check_unary<Packet, op_preinterpret_self>(
      in, [](Scalar x) { return x; }, bits_and_payload);
  check_unary<Packet, op_psign>(
      in, [](Scalar x) { return Eigen::numext::sign(x); }, bits);
  if (std::is_same<Scalar, float>::value) {
    check_unary<Packet, op_psqrt>(
        in, [](Scalar x) { return std::sqrt(x); }, compare_ulps<Scalar>{kSqrtFloatUlps});
  } else {
    check_unary<Packet, op_psqrt>(
        in, [](Scalar x) { return std::sqrt(x); }, bits);
  }
  check_unary<Packet, op_prsqrt>(in, rsqrt_reference<Scalar>, compare_ulps<Scalar>{rsqrt_ulps});
  check_full_mask<Packet, op_ptrue>(in, /*expect_zero=*/false);
  check_full_mask<Packet, op_pzero>(in, /*expect_zero=*/true);

  // Arithmetic: IEEE operations, so the host's own results are the reference, bit for bit.
  const binary_inputs<Scalar> pairs(kSize, 1 << 17, never<Scalar>);
  check_binary<Packet, op_padd>(
      pairs, [](Scalar x, Scalar y) { return x + y; }, bits);
  check_binary<Packet, op_psub>(
      pairs, [](Scalar x, Scalar y) { return x - y; }, bits);
  check_binary<Packet, op_pmul>(
      pairs, [](Scalar x, Scalar y) { return x * y; }, bits);
  check_binary<Packet, op_pdiv>(
      pairs, [](Scalar x, Scalar y) { return x / y; }, bits);
  // The sign of a zero difference is not part of pabsdiff's contract.
  check_binary<Packet, op_pabsdiff>(
      pairs, [](Scalar x, Scalar y) { return x < y ? y - x : x - y; }, values);
  // Plain pmin/pmax leave NaN operands implementation-defined (the device uses fminf/fmaxf); the propagating
  // variants fix them.
  const binary_inputs<Scalar> number_pairs(kSize, 1 << 16, either_nan<Scalar>);
  check_binary<Packet, op_pmin>(
      number_pairs, [](Scalar x, Scalar y) { return (std::min)(x, y); }, values);
  check_binary<Packet, op_pmax>(
      number_pairs, [](Scalar x, Scalar y) { return (std::max)(x, y); }, values);
  check_binary<Packet, op_pmin_numbers>(
      pairs, [](Scalar x, Scalar y) { return std::fmin(x, y); }, values);
  check_binary<Packet, op_pmax_numbers>(
      pairs, [](Scalar x, Scalar y) { return std::fmax(x, y); }, values);
  check_binary<Packet, op_pmin_nan>(
      pairs,
      [](Scalar x, Scalar y) {
        return ((std::isnan)(x) || (std::isnan)(y)) ? std::numeric_limits<Scalar>::quiet_NaN() : (std::min)(x, y);
      },
      values);
  check_binary<Packet, op_pmax_nan>(
      pairs,
      [](Scalar x, Scalar y) {
        return ((std::isnan)(x) || (std::isnan)(y)) ? std::numeric_limits<Scalar>::quiet_NaN() : (std::max)(x, y);
      },
      values);

  // Bit operations act on the representation, payloads included.
  check_binary<Packet, op_pand>(
      pairs, [](Scalar x, Scalar y) { return from_bits<Scalar>(Bits(bits_of(x) & bits_of(y))); }, bits_and_payload);
  check_binary<Packet, op_por>(
      pairs, [](Scalar x, Scalar y) { return from_bits<Scalar>(Bits(bits_of(x) | bits_of(y))); }, bits_and_payload);
  check_binary<Packet, op_pxor>(
      pairs, [](Scalar x, Scalar y) { return from_bits<Scalar>(Bits(bits_of(x) ^ bits_of(y))); }, bits_and_payload);
  check_binary<Packet, op_pandnot>(
      pairs, [](Scalar x, Scalar y) { return from_bits<Scalar>(Bits(bits_of(x) & ~bits_of(y))); }, bits_and_payload);

  // Comparisons: full-bit masks, false on any NaN operand.
  check_compare<Packet, op_pcmp_eq>(pairs, [](Scalar x, Scalar y) { return x == y; });
  check_compare<Packet, op_pcmp_lt>(pairs, [](Scalar x, Scalar y) { return x < y; });
  check_compare<Packet, op_pcmp_le>(pairs, [](Scalar x, Scalar y) { return x <= y; });
  check_compare<Packet, op_pcmp_lt_or_nan>(pairs, [](Scalar x, Scalar y) { return !(x >= y); });

  {
    // pmadd may or may not be contracted by the device compiler (ptxas fuses a*b+c by default): either the
    // fused or the separately rounded result is acceptable until the backend provides an explicit fma.
    const int count = 1 << 16;
    std::vector<Scalar> a = random_values<Scalar>(count), b = random_values<Scalar>(count),
                        c = random_values<Scalar>(count);
    const std::vector<Scalar> specials = special_values<Scalar>();
    for (Scalar x : specials) {
      for (Scalar y : specials) {
        a.push_back(x);
        b.push_back(y);
        c.push_back(specials[(a.size() * 7) % specials.size()]);
      }
    }
    while (a.size() % kSize != 0) {
      a.push_back(Scalar(1));
      b.push_back(Scalar(1));
      c.push_back(Scalar(1));
    }
    const int n3 = int(a.size());
    Buffer<Scalar> fused(n3), unfused(n3);
    for (int k = 0; k < n3; ++k) {
      fused[k] = std::fma(a[k], b[k], c[k]);
      unfused[k] = a[k] * b[k] + c[k];
    }
    const auto fused_or_not = [&](const Scalar* ref, const Scalar* vec, int m) {
      for (int k = 0; k < m; ++k) {
        const bool ok = test::ulp_distance(ref[k], vec[k]) == 0 || test::ulp_distance(unfused[k], vec[k]) == 0;
        if (!ok) {
          std::cout << "pmadd lane " << k << ": " << vec[k] << " is neither fma " << ref[k] << " nor a*b+c "
                    << unfused[k] << std::endl;
          return false;
        }
      }
      return true;
    };
    check_ternary<Packet, op_pmadd>(a, b, c, fused, fused_or_not);

    // pselect with full-bit masks picks the second operand where the mask is set, bit for bit.
    std::vector<Scalar> mask(n3);
    Buffer<Scalar> selected(n3);
    const Scalar all_ones = Eigen::numext::bit_cast<Scalar>(Bits(~Bits(0)));
    for (int k = 0; k < n3; ++k) {
      mask[k] = Eigen::internal::random<bool>() ? all_ones : Scalar(0);
      selected[k] = Eigen::numext::bit_cast<Bits>(mask[k]) != Bits(0) ? a[k] : b[k];
    }
    check_ternary<Packet, op_pselect>(mask, a, b, selected, bits_and_payload);
  }

  // Reductions. Sums and products follow the device's lane order, which the host reproduces.
  check_redux<Packet, op_pfirst>(
      in, [](const Scalar* p) { return p[0]; }, bits_and_payload);
  check_redux<Packet, op_predux>(
      in,
      [kSize](const Scalar* p) {
        Scalar s = p[0];
        for (int l = 1; l < kSize; ++l) s = s + p[l];
        return s;
      },
      bits);
  check_redux<Packet, op_predux_mul>(
      in,
      [kSize](const Scalar* p) {
        Scalar s = p[0];
        for (int l = 1; l < kSize; ++l) s = s * p[l];
        return s;
      },
      bits);
  Buffer<Scalar> numbers(in.size());
  for (Index k = 0; k < in.size(); ++k) numbers[k] = (std::isnan)(in[k]) ? Scalar(1) : in[k];
  check_redux<Packet, op_predux_min>(
      numbers,
      [kSize](const Scalar* p) {
        Scalar s = p[0];
        for (int l = 1; l < kSize; ++l) s = (std::min)(s, p[l]);
        return s;
      },
      values);
  check_redux<Packet, op_predux_max>(
      numbers,
      [kSize](const Scalar* p) {
        Scalar s = p[0];
        for (int l = 1; l < kSize; ++l) s = (std::max)(s, p[l]);
        return s;
      },
      values);
}

// ------------------------------------------------------------------------------------------------------------------
// Parts 5 and 6: the half packets. Packet4h2 is eight halves in four half2 lanes and exists only in the device
// pass. References are computed in float and rounded once, which is exact for the arithmetic operations because a
// float mantissa is at least twice a half's plus two bits.

using HalfPacket = ulonglong2;

// A half operation whose reference is the float computation rounded back.
Eigen::half half_of(float x) { return Eigen::half(x); }
Eigen::half half_add(Eigen::half a, Eigen::half b) { return half_of(float(a) + float(b)); }
Eigen::half half_mul(Eigen::half a, Eigen::half b) { return half_of(float(a) * float(b)); }

void packetmath_gpu_half_core() {
  using Scalar = Eigen::half;
  using Packet = HalfPacket;
  const int kSize = gpu_packet<Packet>::kSize;
  const compare_bits<Scalar> bits{true};
  const compare_bits<Scalar> bits_and_payload{false};
  const compare_values<Scalar> values;

  const std::vector<int> traits = device_traits<Scalar>();
  std::cout << "device packet_traits<half>:";
  for (int k = 0; k < kNumTraits; ++k) std::cout << " " << kTraitNames[k] << "=" << traits[k];
  std::cout << std::endl;
  VERIFY_IS_EQUAL(traits[kVectorizable], 1);
  VERIFY_IS_EQUAL(traits[ksize], kSize);
  // The hygiene sweep turned these off: there is no rounding operation and no psign for the half packets.
  VERIFY_IS_EQUAL(traits[kHasRound], 0);
  VERIFY_IS_EQUAL(traits[kHasSign], 0);
  check_advertised_ops_are_covered<Scalar>(
      traits, {kHasAdd, kHasSub, kHasMul, kHasDiv, kHasNegate, kHasAbs, kHasMin, kHasMax, kHasSqrt, kHasRsqrt,
               kHasSetLinear, kHasConj, kHasAbsDiff});

  // Exhaustive: every value a half can hold.
  const std::vector<Scalar> all = all_half_values();
  const Buffer<Scalar> in = Eigen::Map<const Buffer<Scalar>>(all.data(), all.size());
  const int n = int(in.size()) / kSize;

  check_unary<Packet, op_identity>(
      in, [](Scalar x) { return x; }, bits_and_payload);
  // The sign of a NaN survives neither __hneg2 nor __habs2, as on the float packets; the value stays NaN.
  check_unary<Packet, op_pnegate>(
      in, [](Scalar x) { return from_bits<Scalar>(bits_of(x) ^ 0x8000u); }, bits);
  check_unary<Packet, op_pabs>(
      in, [](Scalar x) { return from_bits<Scalar>(bits_of(x) & 0x7fffu); }, bits);
  check_unary<Packet, op_pconj>(
      in, [](Scalar x) { return x; }, bits_and_payload);

  // The lane index is added in half, so the reference rounds once, as the packet does.
  check_memory_ops<Packet>(in, /*nan_is_nan=*/true,
                           [](Scalar base, int lane) { return half_add(base, half_of(float(lane))); });

  // Arithmetic. Rounding once from float is the correctly rounded half result.
  const binary_inputs<Scalar> pairs = all_half_pairs(/*exclude_nan=*/false);
  const binary_inputs<Scalar> numbers = all_half_pairs(/*exclude_nan=*/true);
  check_binary<Packet, op_padd>(pairs, half_add, bits);
  check_binary<Packet, op_psub>(
      pairs, [](Scalar x, Scalar y) { return half_of(float(x) - float(y)); }, bits);
  check_binary<Packet, op_pmul>(pairs, half_mul, bits);
  check_binary<Packet, op_pdiv>(
      pairs, [](Scalar x, Scalar y) { return half_of(float(x) / float(y)); }, bits);
  // Plain min/max leave a NaN operand implementation-defined, as for the float packets.
  check_binary<Packet, op_pmin>(
      numbers, [](Scalar x, Scalar y) { return float(x) < float(y) ? x : y; }, values);
  check_binary<Packet, op_pmax>(
      numbers, [](Scalar x, Scalar y) { return float(x) > float(y) ? x : y; }, values);

  // Bit operations act on the 16-bit representation.
  check_binary<Packet, op_pand>(
      pairs, [](Scalar x, Scalar y) { return from_bits<Scalar>(bits_of(x) & bits_of(y)); }, bits_and_payload);
  check_binary<Packet, op_por>(
      pairs, [](Scalar x, Scalar y) { return from_bits<Scalar>(bits_of(x) | bits_of(y)); }, bits_and_payload);
  check_binary<Packet, op_pxor>(
      pairs, [](Scalar x, Scalar y) { return from_bits<Scalar>(bits_of(x) ^ bits_of(y)); }, bits_and_payload);
  check_binary<Packet, op_pandnot>(
      pairs, [](Scalar x, Scalar y) { return from_bits<Scalar>(bits_of(x) & ~bits_of(y)); }, bits_and_payload);

  // Comparisons produce all-ones or all-zero 16-bit lanes.
  check_compare<Packet, op_pcmp_eq>(pairs, [](Scalar x, Scalar y) { return float(x) == float(y); });
  check_compare<Packet, op_pcmp_lt>(pairs, [](Scalar x, Scalar y) { return float(x) < float(y); });
  check_compare<Packet, op_pcmp_le>(pairs, [](Scalar x, Scalar y) { return float(x) <= float(y); });

  check_full_mask<Packet, op_ptrue>(in, /*expect_zero=*/false);
  check_full_mask<Packet, op_pzero>(in, /*expect_zero=*/true);
  {
    // pselect with full-bit masks picks lane by lane.
    const int count = numbers.size();
    std::vector<Scalar> mask(count);
    Buffer<Scalar> selected(count);
    for (int k = 0; k < count; ++k) {
      const bool set = Eigen::internal::random<bool>();
      mask[k] = set ? from_bits<Scalar>(0xffffu) : from_bits<Scalar>(0x0000u);
      selected[k] = set ? numbers.a[k] : numbers.b[k];
    }
    check_ternary<Packet, op_pselect>(mask, numbers.a, numbers.b, selected, bits_and_payload);
  }
  {
    // pmadd is __hfma2: one rounding, so the reference is the float fma rounded once.
    const int count = numbers.size();
    Buffer<Scalar> expected(count);
    for (int k = 0; k < count; ++k) {
      expected[k] = half_of(std::fma(float(numbers.a[k]), float(numbers.b[k]), float(numbers.a[k])));
    }
    check_ternary<Packet, op_pmadd>(numbers.a, numbers.b, numbers.a, expected, bits);
  }

  // Reductions follow the lane tree of the implementation, so the reference rounds in the same order.
  const Buffer<Scalar> finite = [&] {
    std::vector<Scalar> v;
    for (Index k = 0; k < in.size(); ++k) {
      const float x = float(in[k]);
      v.push_back((std::isfinite)(x) && std::abs(x) < 4.0f ? in[k] : Scalar(1));
    }
    return Eigen::Map<const Buffer<Scalar>>(v.data(), v.size()).eval();
  }();
  check_redux<Packet, op_predux>(
      finite,
      [](const Scalar* p) {
        return half_add(half_add(half_add(p[0], p[2]), half_add(p[4], p[6])),
                        half_add(half_add(p[1], p[3]), half_add(p[5], p[7])));
      },
      bits);
  check_redux<Packet, op_predux_mul>(
      finite,
      [](const Scalar* p) {
        return half_mul(half_mul(half_mul(p[0], p[2]), half_mul(p[4], p[6])),
                        half_mul(half_mul(p[1], p[3]), half_mul(p[5], p[7])));
      },
      bits);
  check_redux<Packet, op_predux_min>(
      finite,
      [](const Scalar* p) {
        Scalar m = p[0];
        for (int l = 1; l < 8; ++l) m = float(p[l]) < float(m) ? p[l] : m;
        return m;
      },
      values);
  check_redux<Packet, op_predux_max>(
      finite,
      [](const Scalar* p) {
        Scalar m = p[0];
        for (int l = 1; l < 8; ++l) m = float(p[l]) > float(m) ? p[l] : m;
        return m;
      },
      values);
  check_redux<Packet, op_pfirst>(
      in, [](const Scalar* p) { return p[0]; }, bits_and_payload);

  // Casts, in the two forms the evaluator uses.
  {
    const int packets = int(in.size()) / kSize;
    Buffer<float> as_float(packets * 4);
    as_float.setConstant(-7.0f);
    run_on_gpu(half_to_float_kernel(), packets, in, as_float);
    for (int k = 0; k < packets * 4; ++k) {
      const float want = static_cast<float>(in[(k / 4) * kSize + (k % 4)]);
      VERIFY(test::areEqualBits(&want, as_float.data() + k, 1) && "pcast<Packet4h2, float4>");
    }

    // The narrowing cast has to round, so its inputs must not be representable as half to begin with: a float
    // widened from a half is, and would let any rounding rule pass. Every tie of the half grid is generated
    // instead -- the midpoint between each pair of adjacent halves, from the subnormals up to the largest finite
    // one -- with the neighbours of each tie, then the overflow and underflow boundaries.
    std::vector<float> float_values;
    for (unsigned bits = 0; bits + 1 < 0x7c00u; ++bits) {
      const float below = float(from_bits<Scalar>(uint16_t(bits)));
      const float above = float(from_bits<Scalar>(uint16_t(bits + 1)));
      const float tie = 0.5f * (below + above);  // rounds to even, so to one of the two only every other time
      for (float value : {tie, std::nextafterf(tie, below), std::nextafterf(tie, above)}) {
        float_values.push_back(value);
        float_values.push_back(-value);
      }
    }
    const float largest_half = float(from_bits<Scalar>(uint16_t(0x7bffu)));
    const float overflow_tie = 65520.0f;  // halfway from the largest half to where the next one would be
    for (float value :
         {largest_half, std::nextafterf(largest_half, 1e30f), std::nextafterf(overflow_tie, 0.0f), overflow_tie,
          std::nextafterf(overflow_tie, 1e30f), 65536.0f, (std::numeric_limits<float>::max)(),
          std::numeric_limits<float>::infinity(), std::numeric_limits<float>::denorm_min(), 0.0f}) {
      float_values.push_back(value);
      float_values.push_back(-value);
    }
    while (float_values.size() % 8 != 0) float_values.push_back(0.0f);

    const int cast_packets = int(float_values.size()) / 8;
    const Buffer<float> floats = Eigen::Map<const Buffer<float>>(float_values.data(), Index(float_values.size()));
    Buffer<Scalar> as_half(float_values.size());
    as_half.setConstant(Scalar(-7));
    run_on_gpu(float_to_half_kernel(), cast_packets, floats, as_half);
    Buffer<Scalar> want(float_values.size());
    for (Index k = 0; k < want.size(); ++k) want[k] = half_of(floats[k]);
    VERIFY(test::areEqualBits(want.data(), as_half.data(), int(want.size())) && "pcast<float4, Packet4h2>");
  }
}

void packetmath_gpu_half_math() {
  using Scalar = Eigen::half;
  using Packet = HalfPacket;

  const std::vector<Scalar> all = all_half_values();
  const Buffer<Scalar> in = Eigen::Map<const Buffer<Scalar>>(all.data(), all.size());
  Buffer<Scalar> positive(in.size());
  for (Index k = 0; k < in.size(); ++k) positive[k] = from_bits<Scalar>(bits_of(in[k]) & 0x7fffu);

  // h2sqrt, h2rsqrt, h2log and h2exp against the float computation rounded to half. The device functions are not
  // documented to be correctly rounded in half, so allow one half ULP.
  const uint64_t kHalfMathUlps = 1;
  check_unary<Packet, op_psqrt>(
      positive, [](Scalar x) { return half_of(std::sqrt(float(x))); }, compare_ulps<Scalar>{kHalfMathUlps});
  check_unary<Packet, op_prsqrt>(
      positive, [](Scalar x) { return half_of(1.0f / std::sqrt(float(x))); }, compare_ulps<Scalar>{kHalfMathUlps});
  check_unary<Packet, op_plog>(
      positive, [](Scalar x) { return half_of(std::log(float(x))); }, compare_ulps<Scalar>{kHalfMathUlps});
  check_unary<Packet, op_pexp>(
      in, [](Scalar x) { return half_of(std::exp(float(x))); }, compare_ulps<Scalar>{kHalfMathUlps});
  check_unary<Packet, op_plog1p>(
      in, [](Scalar x) { return half_of(std::log1p(float(x))); }, compare_ulps<Scalar>{kHalfMathUlps});
  check_unary<Packet, op_pexpm1>(
      in, [](Scalar x) { return half_of(std::expm1(float(x))); }, compare_ulps<Scalar>{kHalfMathUlps});

  const Scalar zero(0.0f), one(1.0f), inf = std::numeric_limits<Scalar>::infinity();
  const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
  check_special_values<Packet, op_pexp>({{zero, one}, {-zero, one}, {-inf, zero}, {inf, inf}, {nan, nan}});
  check_special_values<Packet, op_plog>({{one, zero}, {zero, -inf}, {-one, nan}, {inf, inf}, {nan, nan}});
  check_special_values<Packet, op_psqrt>({{zero, zero}, {-zero, -zero}, {one, one}, {inf, inf}, {nan, nan}});
}

// ------------------------------------------------------------------------------------------------------------------
// Part 7: types without a device packet run the scalar fallbacks of GenericPacketMath.h, which must agree with the
// host's scalar fallbacks bit for bit; preinterpret is the bit_cast the cast path relies on.

template <typename Scalar>
void check_scalar_fallback_common(const binary_inputs<Scalar>& pairs, const Buffer<Scalar>& in) {
  const compare_bits<Scalar> bits{true};
  const std::vector<int> traits = device_traits<Scalar>();
  VERIFY_IS_EQUAL(traits[kVectorizable], 0);
  VERIFY_IS_EQUAL(traits[ksize], 1);
  check_binary<Scalar, op_padd>(
      pairs, [](Scalar x, Scalar y) { return Eigen::internal::padd(x, y); }, bits);
  check_binary<Scalar, op_pmul>(
      pairs, [](Scalar x, Scalar y) { return Eigen::internal::pmul(x, y); }, bits);
  check_binary<Scalar, op_pmin>(
      pairs, [](Scalar x, Scalar y) { return Eigen::internal::pmin(x, y); }, bits);
  check_binary<Scalar, op_pmax>(
      pairs, [](Scalar x, Scalar y) { return Eigen::internal::pmax(x, y); }, bits);
  check_binary<Scalar, op_pand>(
      pairs, [](Scalar x, Scalar y) { return Eigen::internal::pand(x, y); }, bits);
  check_binary<Scalar, op_por>(
      pairs, [](Scalar x, Scalar y) { return Eigen::internal::por(x, y); }, bits);
  check_binary<Scalar, op_pxor>(
      pairs, [](Scalar x, Scalar y) { return Eigen::internal::pxor(x, y); }, bits);
  check_binary<Scalar, op_pcmp_eq>(
      pairs, [](Scalar x, Scalar y) { return Eigen::internal::pcmp_eq(x, y); }, bits);
  check_binary<Scalar, op_pcmp_lt>(
      pairs, [](Scalar x, Scalar y) { return Eigen::internal::pcmp_lt(x, y); }, bits);
  check_binary<Scalar, op_pcmp_le>(
      pairs, [](Scalar x, Scalar y) { return Eigen::internal::pcmp_le(x, y); }, bits);
  check_unary<Scalar, op_identity>(
      in, [](Scalar x) { return x; }, bits);
  {
    // pselect on a scalar is a ternary on the mask's truth value.
    const int n = pairs.size();
    std::vector<Scalar> mask(n);
    Buffer<Scalar> selected(n);
    for (int k = 0; k < n; ++k) {
      mask[k] = Eigen::internal::random<bool>() ? Eigen::internal::ptrue(Scalar(0)) : Eigen::internal::pzero(Scalar(0));
      selected[k] = Eigen::internal::pselect(Scalar(mask[k]), Scalar(pairs.a[k]), Scalar(pairs.b[k]));
    }
    check_ternary<Scalar, op_pselect>(mask, pairs.a, pairs.b, selected, bits);
  }
}

template <typename Scalar>
binary_inputs<Scalar> integer_pairs(int count) {
  binary_inputs<Scalar> pairs;
  // Magnitudes small enough that the sum and product of two stay in range for every integer type tested.
  const Scalar bound = Scalar(NumTraits<Scalar>::IsSigned ? 100 : 200);
  for (int k = 0; k < count; ++k) {
    pairs.push(Eigen::internal::random<Scalar>(NumTraits<Scalar>::IsSigned ? Scalar(-bound) : Scalar(0), bound),
               Eigen::internal::random<Scalar>(NumTraits<Scalar>::IsSigned ? Scalar(-bound) : Scalar(0), bound));
  }
  return pairs;
}

template <typename Scalar>
void packetmath_gpu_integer_fallback() {
  const compare_bits<Scalar> bits{true};
  const binary_inputs<Scalar> pairs = integer_pairs<Scalar>(1 << 12);
  const Buffer<Scalar> in = Eigen::Map<const Buffer<Scalar>>(pairs.a.data(), pairs.size());
  check_scalar_fallback_common<Scalar>(pairs, in);
  check_unary<Scalar, op_pnegate>(
      in, [](Scalar x) { return Eigen::internal::pnegate(x); }, bits);
  check_unary<Scalar, op_pabs>(
      in, [](Scalar x) { return Eigen::internal::pabs(x); }, bits);
  check_binary<Scalar, op_psub>(
      pairs, [](Scalar x, Scalar y) { return Eigen::internal::psub(x, y); }, bits);
  binary_inputs<Scalar> nonzero = pairs;
  for (Scalar& y : nonzero.b) {
    if (y == Scalar(0)) y = Scalar(1);
  }
  check_binary<Scalar, op_pdiv>(
      nonzero, [](Scalar x, Scalar y) { return Eigen::internal::pdiv(x, y); }, bits);
  check_binary<Scalar, op_pandnot>(
      pairs, [](Scalar x, Scalar y) { return Eigen::internal::pandnot(x, y); }, bits);
}

void packetmath_gpu_bool_fallback() {
  binary_inputs<bool> pairs;
  for (int k = 0; k < 1 << 10; ++k) pairs.push(Eigen::internal::random<bool>(), Eigen::internal::random<bool>());
  // std::vector<bool> has no data(): copy lane by lane.
  Buffer<bool> in(pairs.size());
  for (int k = 0; k < pairs.size(); ++k) in[k] = pairs.a[k];
  check_scalar_fallback_common<bool>(pairs, in);
}

void packetmath_gpu_bfloat16_fallback() {
  using Scalar = Eigen::bfloat16;
  const compare_bits<Scalar> bits{true};
  const std::vector<float> specials = special_values<float>();
  binary_inputs<Scalar> pairs;
  for (float x : specials) {
    for (float y : specials) pairs.push(Scalar(x), Scalar(y));
  }
  for (int k = 0; k < 1 << 12; ++k) {
    pairs.push(Scalar(Eigen::internal::random<float>(-4.0f, 4.0f)),
               Scalar(Eigen::internal::random<float>(-4.0f, 4.0f)));
  }
  const Buffer<Scalar> in = Eigen::Map<const Buffer<Scalar>>(pairs.a.data(), pairs.size());
  check_scalar_fallback_common<Scalar>(pairs, in);
  check_unary<Scalar, op_pnegate>(
      in, [](Scalar x) { return Eigen::internal::pnegate(x); }, bits);
  check_unary<Scalar, op_pabs>(
      in, [](Scalar x) { return Eigen::internal::pabs(x); }, bits);
  check_binary<Scalar, op_psub>(
      pairs, [](Scalar x, Scalar y) { return Eigen::internal::psub(x, y); }, bits);
  check_binary<Scalar, op_pdiv>(
      pairs, [](Scalar x, Scalar y) { return Eigen::internal::pdiv(x, y); }, bits);
  check_unary<Scalar, op_psqrt>(
      in, [](Scalar x) { return Eigen::internal::psqrt(x); }, bits);
  check_unary<Scalar, op_pfloor>(
      in, [](Scalar x) { return Eigen::internal::pfloor(x); }, bits);
}

template <typename Scalar>
void packetmath_gpu_preinterpret() {
  using Bits = typename Eigen::numext::get_integer_by_size<sizeof(Scalar)>::signed_type;
  const int kSize = gpu_packet<typename packet_traits<Scalar>::type>::kSize;
  const Buffer<Scalar> in = unary_inputs<Scalar>(kSize, 1 << 10);
  Buffer<Bits> out(in.size());
  out.setZero();
  run_on_gpu(preinterpret_scalar_kernel<Scalar, Bits>(), int(in.size()), in, out);
  for (Index k = 0; k < in.size(); ++k) {
    VERIFY_IS_EQUAL(out[k], Eigen::numext::bit_cast<Bits>(in[k]));
  }
}

// Compiled in the host pass as well as the device pass: with EIGEN_USE_GPU defined, packet_traits<float>::type is
// float4 in both, so an expression reaching psign, a comparison or a bit operation has to compile on the host too.
// It did not before those operations moved out of the device-only block.
void host_pass_instantiation() {
  Eigen::Array<float, 32, 1> a = Eigen::Array<float, 32, 1>::Random();
  a(0) = std::numeric_limits<float>::quiet_NaN();
  const Eigen::Array<float, 32, 1> s = a.sign();
  const Eigen::Array<bool, 32, 1> nans = a.isNaN();
  const Eigen::Array<float, 32, 1> clamped = a.cwiseMax(0.0f).cwiseMin(1.0f);
  VERIFY((Eigen::numext::isnan)(s(0)));
  VERIFY(nans(0));
  VERIFY(clamped(1) >= 0.0f && clamped(1) <= 1.0f);
}

}  // namespace

EIGEN_DECLARE_TEST(packetmath_gpu) {
  ei_test_init_gpu();
  CALL_SUBTEST_1(host_pass_instantiation());
  CALL_SUBTEST_1(packetmath_gpu_real_core<float>());
  CALL_SUBTEST_2(packetmath_gpu_real_math<float>());
  CALL_SUBTEST_3(packetmath_gpu_real_core<double>());
  CALL_SUBTEST_4(packetmath_gpu_real_math<double>());
  CALL_SUBTEST_5(packetmath_gpu_half_core());
  CALL_SUBTEST_6(packetmath_gpu_half_math());
  CALL_SUBTEST_7(packetmath_gpu_integer_fallback<int32_t>());
  CALL_SUBTEST_7(packetmath_gpu_integer_fallback<int64_t>());
  CALL_SUBTEST_7(packetmath_gpu_integer_fallback<uint8_t>());
  CALL_SUBTEST_7(packetmath_gpu_bool_fallback());
  CALL_SUBTEST_7(packetmath_gpu_bfloat16_fallback());
  CALL_SUBTEST_7(packetmath_gpu_preinterpret<float>());
  CALL_SUBTEST_7(packetmath_gpu_preinterpret<double>());
}
