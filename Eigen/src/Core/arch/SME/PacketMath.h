// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_PACKET_MATH_SME_H
#define EIGEN_PACKET_MATH_SME_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

// ---------------------------------------------------------------------------
// Streaming-mode packet layer of the SME backend.
//
// The evaluators keep NEON packets; the streaming SVE registers and the ZA array
// are reached only from the product kernels, through the operations below, which
// cannot be the packet_traits type or p* specializations:
//   - a streaming vector (svfloat32_t, ...) is sizeless -- no sizeof, not a
//     member or array element type, lane count known at run time -- so neither
//     unpacket_traits<>::size nor PacketBlock can describe it;
//   - code touching one must carry __arm_streaming (or _compatible), which is
//     part of the function type: a p* specialization cannot add it, and an
//     evaluator cannot call it without a mode switch, nor at all where the
//     target has no non-streaming SVE (-mcpu=apple-m4);
//   - streaming mode runs scalar FP 30-80x slower and NEON not at all on Apple M4.
// So these are __arm_streaming overloads on the streaming vector types: Eigen's
// names and argument order where a counterpart exists (pset1, ploadu, pstoreu,
// padd, pmul, pnegate, pmadd, pnmadd, predux), the arithmetic and memory
// operations taking the SVE governing predicate first; the SVE name with Eigen's
// p prefix where none does (pget, pcreate, pld2, pst2, puzp1/2, pzip1/2, psplice); an sme_
// prefix for the ZA tile operations; and the predicates and lane counts as
// members of sme_packet_traits. Eigen's templates of the same names drop out of
// overload resolution, by arity or by substitution failure on the empty
// unpacket_traits below.
// ---------------------------------------------------------------------------

// ZA tile count at a given element width: 4 ZA.S tiles, 8 ZA.D tiles. Architectural rather than
// feature-dependent, so it stays outside sme_packet_traits, which has no double specialization
// without FEAT_SME_F64F64.
template <typename RealScalar>
struct sme_tile_count;
template <>
struct sme_tile_count<float> {
  static constexpr int value = 4;
};
template <>
struct sme_tile_count<double> {
  static constexpr int value = 8;
};

// Clang 23 and later never inline a private-ZA function (one with neither a shared-ZA attribute nor
// __arm_agnostic("sme_za_state")) into a caller with ZA state, always_inline or not: the call stays
// out of line behind a TPIDR2 lazy-save setup. So every function reachable from __arm_inout("za") or
// __arm_new("za") code needs a shared-ZA attribute or EIGEN_SME_ZA_AGNOSTIC, whether or not it uses
// intrinsics; agnostic rather than shared because private-ZA streaming entry points such as
// pack_direct call the same helpers. GCC and Clang before 20 lack the keyword and inline these anyway.
#if EIGEN_COMP_CLANG
#if !__is_identifier(__arm_agnostic)
#define EIGEN_SME_ZA_AGNOSTIC __arm_agnostic("sme_za_state")
#endif
#endif
#ifndef EIGEN_SME_ZA_AGNOSTIC
#define EIGEN_SME_ZA_AGNOSTIC
#endif

// Scalar -> streaming vector, its two- and four-vector tuples, and the predicates of its element
// width; size() is the lane count of one streaming vector, a runtime value. whilelt takes int64_t:
// svwhilelt_b* is overloaded on the four fixed-width types only, so Index -- `long` where int64_t
// is `long long` -- matches none of them exactly; every bound the kernels pass is a non-negative
// Index.
template <typename Scalar>
struct sme_packet_traits {};

template <>
struct sme_packet_traits<float> {
  using type = svfloat32_t;
  using type_x2 = svfloat32x2_t;
  using type_x4 = svfloat32x4_t;
  static EIGEN_ALWAYS_INLINE int size() __arm_streaming_compatible EIGEN_SME_ZA_AGNOSTIC {
    return static_cast<int>(svcntsw());
  }
  static EIGEN_ALWAYS_INLINE svbool_t ptrue() __arm_streaming EIGEN_SME_ZA_AGNOSTIC { return svptrue_b32(); }
  static EIGEN_ALWAYS_INLINE svcount_t ptrue_c() __arm_streaming EIGEN_SME_ZA_AGNOSTIC { return svptrue_c32(); }
  static EIGEN_ALWAYS_INLINE svbool_t whilelt(int64_t begin, int64_t end) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
    return svwhilelt_b32(begin, end);
  }
  static EIGEN_ALWAYS_INLINE svcount_t whilelt_c4(int64_t begin, int64_t end) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
    return svwhilelt_c32_s64(begin, end, 4);
  }
};

#ifdef EIGEN_VECTORIZE_SME_F64F64
template <>
struct sme_packet_traits<double> {
  using type = svfloat64_t;
  using type_x2 = svfloat64x2_t;
  using type_x4 = svfloat64x4_t;
  static EIGEN_ALWAYS_INLINE int size() __arm_streaming_compatible EIGEN_SME_ZA_AGNOSTIC {
    return static_cast<int>(svcntsd());
  }
  static EIGEN_ALWAYS_INLINE svbool_t ptrue() __arm_streaming EIGEN_SME_ZA_AGNOSTIC { return svptrue_b64(); }
  static EIGEN_ALWAYS_INLINE svcount_t ptrue_c() __arm_streaming EIGEN_SME_ZA_AGNOSTIC { return svptrue_c64(); }
  static EIGEN_ALWAYS_INLINE svbool_t whilelt(int64_t begin, int64_t end) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
    return svwhilelt_b64(begin, end);
  }
  static EIGEN_ALWAYS_INLINE svcount_t whilelt_c4(int64_t begin, int64_t end) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
    return svwhilelt_c64_s64(begin, end, 4);
  }
};
#endif

// Streaming vector -> scalar. The primary template has no members, so pset1's parameter type
// below fails substitution for every other packet type and Eigen's own pset1 is selected there.
template <typename Packet>
struct sme_unpacket_traits {};
template <>
struct sme_unpacket_traits<svfloat32_t> {
  using type = float;
  static EIGEN_ALWAYS_INLINE svfloat32_t dup(float from) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
    return svdup_f32(from);
  }
};
#ifdef EIGEN_VECTORIZE_SME_F64F64
template <>
struct sme_unpacket_traits<svfloat64_t> {
  using type = double;
  static EIGEN_ALWAYS_INLINE svfloat64_t dup(double from) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
    return svdup_f64(from);
  }
};
#endif

// A streaming vector is neither a packet nor a scalar, so unpacket_traits has no members for it: a
// signature naming unpacket_traits<Packet>::type, such as Eigen's pset1, fails substitution, and a
// generic operation called without its predicate (predux(v), pfirst(v)) does not compile, where
// mapping the vector to itself would make it a one-lane "scalar" and resolve those to the identity.
// The primary template cannot serve: it applies sizeof, which a sizeless type does not have.
template <>
struct unpacket_traits<svfloat32_t> {};
#ifdef EIGEN_VECTORIZE_SME_F64F64
template <>
struct unpacket_traits<svfloat64_t> {};
#endif

// An overload of Eigen's pset1 rather than a specialization (see the file comment); the two never
// compete, since each fails substitution on the other's vector types.
template <typename Packet>
EIGEN_ALWAYS_INLINE Packet
pset1(typename sme_unpacket_traits<Packet>::type from) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  return sme_unpacket_traits<Packet>::dup(from);
}

// Every operation takes its governing predicate, as the ACLE intrinsics do: the kernels keep the
// predicate of the enclosing loop live, and an all-true one materialized per operation instead
// costs a PTRUE and a P register inside the ZA store loops (48 PTRUEs and 10 predicate spills
// against 10 and none in the float GEMM kernel). The vector type follows the scalar pointer or the
// vector arguments, so one template per operation covers every element width through the
// type-generic ACLE overloads. SVE loads have no alignment requirement, hence only the unaligned
// spellings; _x2/_x4 move two and four consecutive vectors under one SME2 predicate-as-counter.
template <typename Scalar>
EIGEN_ALWAYS_INLINE typename sme_packet_traits<Scalar>::type ploadu(
    svbool_t pg, const Scalar* from) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  return svld1(pg, from);
}
template <typename Scalar>
EIGEN_ALWAYS_INLINE typename sme_packet_traits<Scalar>::type_x2 ploadu_x2(
    svcount_t pn, const Scalar* from) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  return svld1_x2(pn, from);
}
template <typename Scalar>
EIGEN_ALWAYS_INLINE typename sme_packet_traits<Scalar>::type_x4 ploadu_x4(
    svcount_t pn, const Scalar* from) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  return svld1_x4(pn, from);
}
template <typename Scalar>
EIGEN_ALWAYS_INLINE void pstoreu(svbool_t pg, Scalar* to,
                                 typename sme_packet_traits<Scalar>::type from) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  svst1(pg, to, from);
}
template <typename Scalar>
EIGEN_ALWAYS_INLINE void pstoreu_x2(
    svcount_t pn, Scalar* to, typename sme_packet_traits<Scalar>::type_x2 from) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  svst1(pn, to, from);
}
template <typename Scalar>
EIGEN_ALWAYS_INLINE void pstoreu_x4(
    svcount_t pn, Scalar* to, typename sme_packet_traits<Scalar>::type_x4 from) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  svst1(pn, to, from);
}
// Two-element structure load and store (LD2/ST2): pld2 splits interleaved pairs into the even and odd
// lanes, the real and imaginary parts of a complex array, and pst2 interleaves them back.
template <typename Scalar>
EIGEN_ALWAYS_INLINE typename sme_packet_traits<Scalar>::type_x2 pld2(
    svbool_t pg, const Scalar* from) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  return svld2(pg, from);
}
template <typename Scalar>
EIGEN_ALWAYS_INLINE void pst2(svbool_t pg, Scalar* to,
                              typename sme_packet_traits<Scalar>::type_x2 from) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  svst2(pg, to, from);
}

// Arithmetic in Eigen's argument order after the predicate: pmadd(pg, a, b, c) = a * b + c and
// pnmadd(pg, a, b, c) = c - a * b, each one fused operation. Inactive lanes are unspecified (the
// _x forms); every consumer stores through a predicate of its own.
template <typename Packet>
EIGEN_ALWAYS_INLINE Packet padd(svbool_t pg, Packet a, Packet b) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  return svadd_x(pg, a, b);
}
template <typename Packet>
EIGEN_ALWAYS_INLINE Packet pmul(svbool_t pg, Packet a, Packet b) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  return svmul_x(pg, a, b);
}
template <typename Packet>
EIGEN_ALWAYS_INLINE Packet pnegate(svbool_t pg, Packet a) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  return svneg_x(pg, a);
}
template <typename Packet>
EIGEN_ALWAYS_INLINE Packet pmadd(svbool_t pg, Packet a, Packet b, Packet c) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  return svmla_x(pg, c, a, b);
}
template <typename Packet>
EIGEN_ALWAYS_INLINE Packet pnmadd(svbool_t pg, Packet a, Packet b, Packet c) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  return svmls_x(pg, c, a, b);
}
// Merging form: inactive lanes keep c, in the one FMLA that a select after the _x form does not fold to.
template <typename Packet>
EIGEN_ALWAYS_INLINE Packet pmadd_m(svbool_t pg, Packet a, Packet b, Packet c) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  return svmla_m(pg, c, a, b);
}
// Sum of the active lanes.
template <typename Packet>
EIGEN_ALWAYS_INLINE typename sme_unpacket_traits<Packet>::type predux(svbool_t pg,
                                                                      Packet a) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  return svaddv(pg, a);
}

// Vector tuples: the lane of a tuple is an instruction immediate, hence a template parameter.
template <int Lane>
EIGEN_ALWAYS_INLINE svfloat32_t pget(svfloat32x2_t v) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  return svget2_f32(v, Lane);
}
template <int Lane>
EIGEN_ALWAYS_INLINE svfloat32_t pget(svfloat32x4_t v) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  return svget4_f32(v, Lane);
}
#ifdef EIGEN_VECTORIZE_SME_F64F64
template <int Lane>
EIGEN_ALWAYS_INLINE svfloat64_t pget(svfloat64x2_t v) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  return svget2_f64(v, Lane);
}
template <int Lane>
EIGEN_ALWAYS_INLINE svfloat64_t pget(svfloat64x4_t v) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  return svget4_f64(v, Lane);
}
#endif
template <typename Packet>
EIGEN_ALWAYS_INLINE auto pcreate(Packet a, Packet b) EIGEN_SME_ZA_AGNOSTIC __arm_streaming
    -> decltype(svcreate2(a, b)) {
  return svcreate2(a, b);
}
template <typename Packet>
EIGEN_ALWAYS_INLINE auto pcreate(Packet a, Packet b, Packet c, Packet d) EIGEN_SME_ZA_AGNOSTIC __arm_streaming
    -> decltype(svcreate4(a, b, c, d)) {
  return svcreate4(a, b, c, d);
}

// Permutes: puzp1/puzp2 gather the even/odd lanes of the concatenation (a, b), pzip1/pzip2
// interleave its low/high halves, psplice(pg, a, b) is the active lanes of a followed by the leading
// lanes of b.
template <typename Packet>
EIGEN_ALWAYS_INLINE Packet puzp1(Packet a, Packet b) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  return svuzp1(a, b);
}
template <typename Packet>
EIGEN_ALWAYS_INLINE Packet puzp2(Packet a, Packet b) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  return svuzp2(a, b);
}
template <typename Packet>
EIGEN_ALWAYS_INLINE Packet pzip1(Packet a, Packet b) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  return svzip1(a, b);
}
template <typename Packet>
EIGEN_ALWAYS_INLINE Packet pzip2(Packet a, Packet b) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  return svzip2(a, b);
}
template <typename Packet>
EIGEN_ALWAYS_INLINE Packet psplice(svbool_t pg, Packet a, Packet b) __arm_streaming EIGEN_SME_ZA_AGNOSTIC {
  return svsplice(pg, a, b);
}

// Entering and leaving streaming mode sets FPSR's cumulative exception flags (Arm DDI0616, RMHTLZ), also when the OS
// resumes a thread in streaming mode, and FP arithmetic into ZA raises none. So a call into streaming code keeps the
// caller's flags and reports none of its own.
struct sme_fpsr_guard {
  EIGEN_ALWAYS_INLINE sme_fpsr_guard() { asm volatile("mrs %0, fpsr" : "=r"(value) : : "memory"); }
  EIGEN_ALWAYS_INLINE ~sme_fpsr_guard() { asm volatile("msr fpsr, %0" : : "r"(value) : "memory"); }
  sme_fpsr_guard(const sme_fpsr_guard&) = delete;
  sme_fpsr_guard& operator=(const sme_fpsr_guard&) = delete;

 private:
  std::uint64_t value;
};

// min() usable from streaming functions (numext::mini lacks the __arm_streaming_compatible attribute).
template <typename T>
EIGEN_ALWAYS_INLINE T sme_min(T a, T b) __arm_streaming_compatible EIGEN_SME_ZA_AGNOSTIC {
  return a < b ? a : b;
}

// Offset a pointer by n elements without forming the pointer value: an access whose predicate is
// empty makes no memory reference, but computing an address more than one past the end of the
// object is undefined regardless, so the second-vector accesses of the kernels reach their address
// through uintptr_t.
template <typename T>
EIGEN_ALWAYS_INLINE T* sme_offset(T* p, Index n) __arm_streaming_compatible EIGEN_SME_ZA_AGNOSTIC {
  return reinterpret_cast<T*>(uintptr_t(p) + ptrdiff_t(n) * sizeof(T));
}

// ZA tile access. A tile number is an instruction immediate, hence a template parameter; a slice
// number is a register operand and stays a value. The element width follows the vector or pointer
// argument, or is given as the scalar type where there is neither (sme_za_read).
template <int Tile>
EIGEN_ALWAYS_INLINE void sme_ld1_hor_za(uint32_t slice, svbool_t pg, const float* p) __arm_streaming __arm_inout("za") {
  svld1_hor_za32(Tile, slice, pg, p);
}
template <int Tile>
EIGEN_ALWAYS_INLINE svfloat32_t sme_read_hor_za(svfloat32_t zero, svbool_t pg,
                                                uint32_t slice) __arm_streaming __arm_in("za") {
  return svread_hor_za32_f32_m(zero, pg, Tile, slice);
}
template <int Tile>
EIGEN_ALWAYS_INLINE svfloat32_t sme_read_ver_za(svfloat32_t zero, svbool_t pg,
                                                uint32_t slice) __arm_streaming __arm_in("za") {
  return svread_ver_za32_f32_m(zero, pg, Tile, slice);
}
// Four slices at once (SME2 MOVA ... vg4).
template <int Tile>
EIGEN_ALWAYS_INLINE void sme_write_hor_za_vg4(uint32_t slice, svfloat32_t a, svfloat32_t b, svfloat32_t c,
                                              svfloat32_t d) __arm_streaming __arm_inout("za") {
  svwrite_hor_za32_f32_vg4(Tile, slice, svcreate4_f32(a, b, c, d));
}
template <int Tile>
EIGEN_ALWAYS_INLINE void sme_write_ver_za_vg4(uint32_t slice, svfloat32x4_t v) __arm_streaming __arm_inout("za") {
  svwrite_ver_za32_f32_vg4(Tile, slice, v);
}
// Outer product accumulate (FMOPA) and subtract (FMOPS) into a tile.
template <int Tile>
EIGEN_ALWAYS_INLINE void sme_mopa(svbool_t pm, svbool_t pn, svfloat32_t a,
                                  svfloat32_t b) __arm_streaming __arm_inout("za") {
  svmopa_za32_f32_m(Tile, pm, pn, a, b);
}
template <int Tile>
EIGEN_ALWAYS_INLINE void sme_mops(svbool_t pm, svbool_t pn, svfloat32_t a,
                                  svfloat32_t b) __arm_streaming __arm_inout("za") {
  svmops_za32_f32_m(Tile, pm, pn, a, b);
}
// SME2 multi-vector FMLA into four ZA array vectors, the vector-length accumulators of the vector
// kernels; four independent groups hide the accumulator latency.
EIGEN_ALWAYS_INLINE void sme_madd_za_vg1x4(uint32_t slice, svfloat32x4_t x,
                                           svfloat32_t y) __arm_streaming __arm_inout("za") {
  svmla_single_za32_f32_vg1x4(slice, x, y);
}
EIGEN_ALWAYS_INLINE void sme_madd_za_vg1x4(uint32_t slice, svfloat32x4_t x,
                                           svfloat32x4_t y) __arm_streaming __arm_inout("za") {
  svmla_za32_f32_vg1x4(slice, x, y);
}
EIGEN_ALWAYS_INLINE void sme_write_za_vg1x4(uint32_t slice, svfloat32x4_t x) __arm_streaming __arm_inout("za") {
  svwrite_za32_f32_vg1x4(slice, x);
}

#ifdef EIGEN_VECTORIZE_SME_F64F64
template <int Tile>
EIGEN_ALWAYS_INLINE void sme_ld1_hor_za(uint32_t slice, svbool_t pg,
                                        const double* p) __arm_streaming __arm_inout("za") {
  svld1_hor_za64(Tile, slice, pg, p);
}
template <int Tile>
EIGEN_ALWAYS_INLINE svfloat64_t sme_read_hor_za(svfloat64_t zero, svbool_t pg,
                                                uint32_t slice) __arm_streaming __arm_in("za") {
  return svread_hor_za64_f64_m(zero, pg, Tile, slice);
}
template <int Tile>
EIGEN_ALWAYS_INLINE svfloat64_t sme_read_ver_za(svfloat64_t zero, svbool_t pg,
                                                uint32_t slice) __arm_streaming __arm_in("za") {
  return svread_ver_za64_f64_m(zero, pg, Tile, slice);
}
template <int Tile>
EIGEN_ALWAYS_INLINE void sme_write_hor_za_vg4(uint32_t slice, svfloat64_t a, svfloat64_t b, svfloat64_t c,
                                              svfloat64_t d) __arm_streaming __arm_inout("za") {
  svwrite_hor_za64_f64_vg4(Tile, slice, svcreate4_f64(a, b, c, d));
}
template <int Tile>
EIGEN_ALWAYS_INLINE void sme_write_ver_za_vg4(uint32_t slice, svfloat64x4_t v) __arm_streaming __arm_inout("za") {
  svwrite_ver_za64_f64_vg4(Tile, slice, v);
}
template <int Tile>
EIGEN_ALWAYS_INLINE void sme_mopa(svbool_t pm, svbool_t pn, svfloat64_t a,
                                  svfloat64_t b) __arm_streaming __arm_inout("za") {
  svmopa_za64_f64_m(Tile, pm, pn, a, b);
}
template <int Tile>
EIGEN_ALWAYS_INLINE void sme_mops(svbool_t pm, svbool_t pn, svfloat64_t a,
                                  svfloat64_t b) __arm_streaming __arm_inout("za") {
  svmops_za64_f64_m(Tile, pm, pn, a, b);
}
EIGEN_ALWAYS_INLINE void sme_madd_za_vg1x4(uint32_t slice, svfloat64x4_t x,
                                           svfloat64_t y) __arm_streaming __arm_inout("za") {
  svmla_single_za64_f64_vg1x4(slice, x, y);
}
EIGEN_ALWAYS_INLINE void sme_madd_za_vg1x4(uint32_t slice, svfloat64x4_t x,
                                           svfloat64x4_t y) __arm_streaming __arm_inout("za") {
  svmla_za64_f64_vg1x4(slice, x, y);
}
EIGEN_ALWAYS_INLINE void sme_write_za_vg1x4(uint32_t slice, svfloat64x4_t x) __arm_streaming __arm_inout("za") {
  svwrite_za64_f64_vg1x4(slice, x);
}
#endif  // EIGEN_VECTORIZE_SME_F64F64

// The two reads with no width-carrying argument: four vertical slices of a tile, and four ZA array
// vectors.
template <typename Scalar>
struct sme_za_read;
template <>
struct sme_za_read<float> {
  template <int Tile>
  static EIGEN_ALWAYS_INLINE svfloat32x4_t ver_vg4(uint32_t slice) __arm_streaming __arm_in("za") {
    return svread_ver_za32_f32_vg4(Tile, slice);
  }
  static EIGEN_ALWAYS_INLINE svfloat32x4_t vg1x4(uint32_t slice) __arm_streaming __arm_in("za") {
    return svread_za32_f32_vg1x4(slice);
  }
};
#ifdef EIGEN_VECTORIZE_SME_F64F64
template <>
struct sme_za_read<double> {
  template <int Tile>
  static EIGEN_ALWAYS_INLINE svfloat64x4_t ver_vg4(uint32_t slice) __arm_streaming __arm_in("za") {
    return svread_ver_za64_f64_vg4(Tile, slice);
  }
  static EIGEN_ALWAYS_INLINE svfloat64x4_t vg1x4(uint32_t slice) __arm_streaming __arm_in("za") {
    return svread_za64_f64_vg1x4(slice);
  }
};
#endif
template <int Tile, typename Scalar>
EIGEN_ALWAYS_INLINE typename sme_packet_traits<Scalar>::type_x4 sme_read_ver_za_vg4(
    uint32_t slice) __arm_streaming __arm_in("za") {
  return sme_za_read<Scalar>::template ver_vg4<Tile>(slice);
}
template <typename Scalar>
EIGEN_ALWAYS_INLINE typename sme_packet_traits<Scalar>::type_x4 sme_read_za_vg1x4(
    uint32_t slice) __arm_streaming __arm_in("za") {
  return sme_za_read<Scalar>::vg1x4(slice);
}

// Outer-product accumulate with a compile-time sign: the complex kernel's four real products differ
// only in whether they add or subtract into the tile.
template <int Tile, bool Subtract, typename Packet>
EIGEN_ALWAYS_INLINE void sme_mopa_signed(svbool_t pm, svbool_t pn, Packet a,
                                         Packet b) __arm_streaming __arm_inout("za") {
  EIGEN_IF_CONSTEXPR (Subtract) {
    sme_mops<Tile>(pm, pn, a, b);
  } else {
    sme_mopa<Tile>(pm, pn, a, b);
  }
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_PACKET_MATH_SME_H
