// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_MATH_FUNCTIONS_NEON_H
#define EIGEN_MATH_FUNCTIONS_NEON_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_FLOAT(Packet2f)
EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_FLOAT(Packet4f)

#if EIGEN_ARCH_ARM64 && EIGEN_HAS_ARM64_FP16
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet4hf ptanh<Packet4hf>(const Packet4hf& x) {
  // Convert to float, call the float ptanh, and then convert back.
  return vcvt_f16_f32(ptanh<Packet4f>(vcvt_f32_f16(x)));
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet8hf ptanh<Packet8hf>(const Packet8hf& x) {
  // Convert each 4 half types to float, call the float ptanh, and then convert back.
  return vcombine_f16(vcvt_f16_f32(ptanh<Packet4f>(vcvt_f32_f16(vget_low_f16(x)))),
                      vcvt_f16_f32(ptanh<Packet4f>(vcvt_f32_f16(vget_high_f16(x)))));
}
#endif  // EIGEN_ARCH_ARM64 && EIGEN_HAS_ARM64_FP16

// psqrt and prsqrt have their own kernels below.
EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_BF16(Packet4f, Packet4bf)

template <>
EIGEN_STRONG_INLINE Packet4bf psqrt<Packet4bf>(const Packet4bf& a) {
  // y = x r with r = FRSQRTE(x), then one FRSQRTS step y <- y (3 - y r) / 2. FRSQRTE is specified
  // bit for bit, and rounding y to bfloat16 equals bfloat16(sqrtf(x)) for all 65536 inputs
  // (packetmath checks them all); the raw estimate alone misrounds 7366 of them. RSHRN adds half a
  // bfloat16 ulp and truncates. That is round half up, which differs from F32ToBf16 only on an
  // exact float midpoint, and no input produces one; NaNs stay NaN since their low half is zero.
  Packet4f x = Bf16ToF32(a);
  Packet4f r = vrsqrteq_f32(x);
  Packet4f y = pmul(x, r);
  y = pmul(y, vrsqrtsq_f32(y, r));
  Packet4bf ybf = vrshrn_n_u32(vreinterpretq_u32_f32(y), 16);
  // y is NaN where r is +-inf (x is +-0, or a subnormal the FPU flushes) or +0 (x is +inf), and
  // the answer there is the sign and exponent of a. Reading r rather than a keeps that right under
  // FPCR.FZ and on ARMv7. The high half of r plus 0x80 is 0x80, 0 or 0x8000 on exactly those
  // lanes; every other r, a positive normal or a NaN, has a bit set under 0x7f7f.
  uint16x4_t u = vadd_u16(vshrn_n_u32(vreinterpretq_u32_f32(r), 16), vdup_n_u16(0x80));
  uint16x4_t step_lane = vtst_u16(u, vdup_n_u16(0x7f7f));
  return vbsl_u16(step_lane, ybf, vand_u16(a, vdup_n_u16(0xff80)));
}

template <>
EIGEN_STRONG_INLINE Packet4bf prsqrt<Packet4bf>(const Packet4bf& a) {
  // r = FRSQRTE(x) refined by one FRSQRTS step, r1 = r (3 - x r r) / 2, then the same RSHRN
  // narrowing as psqrt. r1 rounds to the correctly rounded bfloat16 rsqrt for all 65536 inputs
  // (packetmath checks them all). FRSQRTE has 8 fraction bits, so the raw estimate always lands on
  // a bfloat16 value or midpoint and misrounds 1270 of them.
  Packet4f x = Bf16ToF32(a);
  Packet4f r = vrsqrteq_f32(x);
#if EIGEN_ARCH_ARM64
  // FMULX gives 2 for 0 * inf where FMUL gives NaN, so no lane needs a select. Where the FPU reads
  // x as +-0 (a subnormal too under FPCR.FZ), r is +-inf, y = 2 and the step is -+inf, and |step|
  // restores r1 = +-inf. Where x is +inf, r is 0, y = 2 and r1 = 0 * 1.5. Negative and NaN x stay
  // NaN. On every other lane FMULX is FMUL and the step is positive, so r1 is the plain step.
  Packet4f y = vmulxq_f32(x, r);
  Packet4f r1 = pmul(r, vabsq_f32(vrsqrtsq_f32(y, r)));
  return vrshrn_n_u32(vreinterpretq_u32_f32(r1), 16);
#else
  // ARMv7 has no FMULX. r1 is NaN where x is +-0, +inf or a subnormal, which ARMv7 NEON flushes,
  // and there r is already the answer (+-inf, 0, +-inf) with a zero low half, so those lanes take
  // r truncated. They are the bit patterns with a - 0x80 >= 0x7f00 unsigned: the exponent-0 range
  // wraps around, and +inf, NaN and every negative value sit above.
  Packet4f r1 = pmul(r, vrsqrtsq_f32(pmul(x, r), r));
  Packet4bf refined = vrshrn_n_u32(vreinterpretq_u32_f32(r1), 16);
  Packet4bf raw = vshrn_n_u32(vreinterpretq_u32_f32(r), 16);
  uint16x4_t special = vcge_u16(vsub_u16(a, vdup_n_u16(0x80)), vdup_n_u16(0x7f00));
  return vbsl_u16(special, raw, refined);
#endif
}

template <>
EIGEN_STRONG_INLINE Packet4bf pfrexp(const Packet4bf& a, Packet4bf& exponent) {
  // Both results are exact: the mantissa keeps the input's significand, and the exponent is a
  // small integer.
  Packet4f fexponent;
  const Packet4bf out = F32ToBf16Truncate(pfrexp<Packet4f>(Bf16ToF32(a), fexponent));
  exponent = F32ToBf16Truncate(fexponent);
  return out;
}

template <>
EIGEN_STRONG_INLINE Packet4bf pldexp(const Packet4bf& a, const Packet4bf& exponent) {
  return F32ToBf16(pldexp<Packet4f>(Bf16ToF32(a), Bf16ToF32(exponent)));
}

//---------- double ----------

#if EIGEN_ARCH_ARM64

EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_DOUBLE(Packet2d)

#endif

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_MATH_FUNCTIONS_NEON_H
