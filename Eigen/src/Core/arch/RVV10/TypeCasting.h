// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2024 Kseniya Zaytseva <kseniya.zaytseva@syntacore.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TYPE_CASTING_RVV10_H
#define EIGEN_TYPE_CASTING_RVV10_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

/********************************* 32 bits ************************************/

template <>
struct type_casting_traits<float, numext::int32_t> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 };
};

template <>
struct type_casting_traits<numext::int32_t, float> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE PacketMul1Xf pcast<PacketMul1Xi, PacketMul1Xf>(const PacketMul1Xi& a) {
  return __riscv_vfcvt_f_x_v_f32m1(a, unpacket_traits<PacketMul1Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi pcast<PacketMul1Xf, PacketMul1Xi>(const PacketMul1Xf& a) {
  return __riscv_vfcvt_rtz_x_f_v_i32m1(a, unpacket_traits<PacketMul1Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xf preinterpret<PacketMul1Xf, PacketMul1Xi>(const PacketMul1Xi& a) {
  return __riscv_vreinterpret_v_i32m1_f32m1(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xi preinterpret<PacketMul1Xi, PacketMul1Xf>(const PacketMul1Xf& a) {
  return __riscv_vreinterpret_v_f32m1_i32m1(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pcast<PacketMul4Xi, PacketMul4Xf>(const PacketMul4Xi& a) {
  return __riscv_vfcvt_f_x_v_f32m4(a, unpacket_traits<PacketMul4Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pcast<PacketMul4Xf, PacketMul4Xi>(const PacketMul4Xf& a) {
  return __riscv_vfcvt_rtz_x_f_v_i32m4(a, unpacket_traits<PacketMul4Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf preinterpret<PacketMul4Xf, PacketMul4Xi>(const PacketMul4Xi& a) {
  return __riscv_vreinterpret_v_i32m4_f32m4(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi preinterpret<PacketMul4Xi, PacketMul4Xf>(const PacketMul4Xf& a) {
  return __riscv_vreinterpret_v_f32m4_i32m4(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pcast<PacketMul2Xi, PacketMul2Xf>(const PacketMul2Xi& a) {
  return __riscv_vfcvt_f_x_v_f32m2(a, unpacket_traits<PacketMul2Xi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pcast<PacketMul2Xf, PacketMul2Xi>(const PacketMul2Xf& a) {
  return __riscv_vfcvt_rtz_x_f_v_i32m2(a, unpacket_traits<PacketMul2Xf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf preinterpret<PacketMul2Xf, PacketMul2Xi>(const PacketMul2Xi& a) {
  return __riscv_vreinterpret_v_i32m2_f32m2(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi preinterpret<PacketMul2Xi, PacketMul2Xf>(const PacketMul2Xf& a) {
  return __riscv_vreinterpret_v_f32m2_i32m2(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pcast<PacketMul1Xi, PacketMul4Xi>(const PacketMul1Xi& a, const PacketMul1Xi& b, const PacketMul1Xi& c,
                                                               const PacketMul1Xi& d) {
  return __riscv_vcreate_v_i32m1_i32m4(a, b, c, d);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pcast<PacketMul1Xi, PacketMul4Xf>(const PacketMul1Xi& a, const PacketMul1Xi& b, const PacketMul1Xi& c,
                                                               const PacketMul1Xi& d) {
  return __riscv_vcreate_v_f32m1_f32m4(__riscv_vfcvt_f_x_v_f32m1(a, unpacket_traits<PacketMul1Xi>::size),
                                       __riscv_vfcvt_f_x_v_f32m1(b, unpacket_traits<PacketMul1Xi>::size),
                                       __riscv_vfcvt_f_x_v_f32m1(c, unpacket_traits<PacketMul1Xi>::size),
                                       __riscv_vfcvt_f_x_v_f32m1(d, unpacket_traits<PacketMul1Xi>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pcast<PacketMul1Xf, PacketMul4Xf>(const PacketMul1Xf& a, const PacketMul1Xf& b, const PacketMul1Xf& c,
                                                               const PacketMul1Xf& d) {
  return __riscv_vcreate_v_f32m1_f32m4(a, b, c, d);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pcast<PacketMul1Xf, PacketMul4Xi>(const PacketMul1Xf& a, const PacketMul1Xf& b, const PacketMul1Xf& c,
                                                               const PacketMul1Xf& d) {
  return __riscv_vcreate_v_i32m1_i32m4(__riscv_vfcvt_rtz_x_f_v_i32m1(a, unpacket_traits<PacketMul1Xf>::size),
                                       __riscv_vfcvt_rtz_x_f_v_i32m1(b, unpacket_traits<PacketMul1Xf>::size),
                                       __riscv_vfcvt_rtz_x_f_v_i32m1(c, unpacket_traits<PacketMul1Xf>::size),
                                       __riscv_vfcvt_rtz_x_f_v_i32m1(d, unpacket_traits<PacketMul1Xf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pcast<PacketMul1Xi, PacketMul2Xi>(const PacketMul1Xi& a, const PacketMul1Xi& b) {
  return __riscv_vcreate_v_i32m1_i32m2(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pcast<PacketMul1Xi, PacketMul2Xf>(const PacketMul1Xi& a, const PacketMul1Xi& b) {
  return __riscv_vcreate_v_f32m1_f32m2(__riscv_vfcvt_f_x_v_f32m1(a, unpacket_traits<PacketMul1Xi>::size),
                                       __riscv_vfcvt_f_x_v_f32m1(b, unpacket_traits<PacketMul1Xi>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pcast<PacketMul1Xf, PacketMul2Xf>(const PacketMul1Xf& a, const PacketMul1Xf& b) {
  return __riscv_vcreate_v_f32m1_f32m2(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pcast<PacketMul1Xf, PacketMul2Xi>(const PacketMul1Xf& a, const PacketMul1Xf& b) {
  return __riscv_vcreate_v_i32m1_i32m2(__riscv_vfcvt_rtz_x_f_v_i32m1(a, unpacket_traits<PacketMul1Xf>::size),
                                       __riscv_vfcvt_rtz_x_f_v_i32m1(b, unpacket_traits<PacketMul1Xf>::size));
}

/********************************* 64 bits ************************************/

template <>
struct type_casting_traits<double, numext::int64_t> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 };
};

template <>
struct type_casting_traits<numext::int64_t, double> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE PacketMul1Xd pcast<PacketMul1Xl, PacketMul1Xd>(const PacketMul1Xl& a) {
  return __riscv_vfcvt_f_x_v_f64m1(a, unpacket_traits<PacketMul1Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl pcast<PacketMul1Xd, PacketMul1Xl>(const PacketMul1Xd& a) {
  return __riscv_vfcvt_rtz_x_f_v_i64m1(a, unpacket_traits<PacketMul1Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xd preinterpret<PacketMul1Xd, PacketMul1Xl>(const PacketMul1Xl& a) {
  return __riscv_vreinterpret_v_i64m1_f64m1(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul1Xl preinterpret<PacketMul1Xl, PacketMul1Xd>(const PacketMul1Xd& a) {
  return __riscv_vreinterpret_v_f64m1_i64m1(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pcast<PacketMul4Xl, PacketMul4Xd>(const PacketMul4Xl& a) {
  return __riscv_vfcvt_f_x_v_f64m4(a, unpacket_traits<PacketMul4Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pcast<PacketMul4Xd, PacketMul4Xl>(const PacketMul4Xd& a) {
  return __riscv_vfcvt_rtz_x_f_v_i64m4(a, unpacket_traits<PacketMul4Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd preinterpret<PacketMul4Xd, PacketMul4Xl>(const PacketMul4Xl& a) {
  return __riscv_vreinterpret_v_i64m4_f64m4(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl preinterpret<PacketMul4Xl, PacketMul4Xd>(const PacketMul4Xd& a) {
  return __riscv_vreinterpret_v_f64m4_i64m4(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pcast<PacketMul2Xl, PacketMul2Xd>(const PacketMul2Xl& a) {
  return __riscv_vfcvt_f_x_v_f64m2(a, unpacket_traits<PacketMul2Xl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pcast<PacketMul2Xd, PacketMul2Xl>(const PacketMul2Xd& a) {
  return __riscv_vfcvt_rtz_x_f_v_i64m2(a, unpacket_traits<PacketMul2Xd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd preinterpret<PacketMul2Xd, PacketMul2Xl>(const PacketMul2Xl& a) {
  return __riscv_vreinterpret_v_i64m2_f64m2(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl preinterpret<PacketMul2Xl, PacketMul2Xd>(const PacketMul2Xd& a) {
  return __riscv_vreinterpret_v_f64m2_i64m2(a);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pcast<PacketMul1Xl, PacketMul4Xl>(const PacketMul1Xl& a, const PacketMul1Xl& b, const PacketMul1Xl& c,
                                                               const PacketMul1Xl& d) {
  return __riscv_vcreate_v_i64m1_i64m4(a, b, c, d);
  ;
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pcast<PacketMul1Xl, PacketMul4Xd>(const PacketMul1Xl& a, const PacketMul1Xl& b, const PacketMul1Xl& c,
                                                               const PacketMul1Xl& d) {
  return __riscv_vcreate_v_f64m1_f64m4(__riscv_vfcvt_f_x_v_f64m1(a, unpacket_traits<PacketMul1Xl>::size),
                                       __riscv_vfcvt_f_x_v_f64m1(b, unpacket_traits<PacketMul1Xl>::size),
                                       __riscv_vfcvt_f_x_v_f64m1(c, unpacket_traits<PacketMul1Xl>::size),
                                       __riscv_vfcvt_f_x_v_f64m1(d, unpacket_traits<PacketMul1Xl>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pcast<PacketMul1Xd, PacketMul4Xd>(const PacketMul1Xd& a, const PacketMul1Xd& b, const PacketMul1Xd& c,
                                                               const PacketMul1Xd& d) {
  return __riscv_vcreate_v_f64m1_f64m4(a, b, c, d);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pcast<PacketMul1Xd, PacketMul4Xl>(const PacketMul1Xd& a, const PacketMul1Xd& b, const PacketMul1Xd& c,
                                                               const PacketMul1Xd& d) {
  return __riscv_vcreate_v_i64m1_i64m4(__riscv_vfcvt_rtz_x_f_v_i64m1(a, unpacket_traits<PacketMul1Xd>::size),
                                       __riscv_vfcvt_rtz_x_f_v_i64m1(b, unpacket_traits<PacketMul1Xd>::size),
                                       __riscv_vfcvt_rtz_x_f_v_i64m1(c, unpacket_traits<PacketMul1Xd>::size),
                                       __riscv_vfcvt_rtz_x_f_v_i64m1(d, unpacket_traits<PacketMul1Xd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pcast<PacketMul1Xl, PacketMul2Xl>(const PacketMul1Xl& a, const PacketMul1Xl& b) {
  return __riscv_vcreate_v_i64m1_i64m2(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pcast<PacketMul1Xl, PacketMul2Xd>(const PacketMul1Xl& a, const PacketMul1Xl& b) {
  return __riscv_vcreate_v_f64m1_f64m2(__riscv_vfcvt_f_x_v_f64m1(a, unpacket_traits<PacketMul1Xl>::size),
                                       __riscv_vfcvt_f_x_v_f64m1(b, unpacket_traits<PacketMul1Xl>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pcast<PacketMul1Xd, PacketMul2Xd>(const PacketMul1Xd& a, const PacketMul1Xd& b) {
  return __riscv_vcreate_v_f64m1_f64m2(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pcast<PacketMul1Xd, PacketMul2Xl>(const PacketMul1Xd& a, const PacketMul1Xd& b) {
  return __riscv_vcreate_v_i64m1_i64m2(__riscv_vfcvt_rtz_x_f_v_i64m1(a, unpacket_traits<PacketMul1Xd>::size),
                                       __riscv_vfcvt_rtz_x_f_v_i64m1(b, unpacket_traits<PacketMul1Xd>::size));
}

/********************************* 16 bits ************************************/

template <>
EIGEN_STRONG_INLINE PacketMul2Xs pcast<PacketMul1Xs, PacketMul2Xs>(const PacketMul1Xs& a, const PacketMul1Xs& b) {
  return __riscv_vcreate_v_i16m1_i16m2(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs pcast<PacketMul1Xs, PacketMul4Xs>(const PacketMul1Xs& a, const PacketMul1Xs& b, const PacketMul1Xs& c,
                                                               const PacketMul1Xs& d) {
  return __riscv_vcreate_v_i16m1_i16m4(a, b, c, d);
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_TYPE_CASTING_RVV10_H
