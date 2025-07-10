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
EIGEN_STRONG_INLINE PacketXf pcast<PacketXi, PacketXf>(const PacketXi& a) {
  return __riscv_vfcvt_f_x_v_f32m1(a, unpacket_traits<PacketXi>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXi pcast<PacketXf, PacketXi>(const PacketXf& a) {
  return __riscv_vfcvt_rtz_x_f_v_i32m1(a, unpacket_traits<PacketXf>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXf preinterpret<PacketXf, PacketXi>(const PacketXi& a) {
  return __riscv_vreinterpret_v_i32m1_f32m1(a);
}

template <>
EIGEN_STRONG_INLINE PacketXi preinterpret<PacketXi, PacketXf>(const PacketXf& a) {
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
EIGEN_STRONG_INLINE PacketMul4Xi pcast<PacketXi, PacketMul4Xi>(const PacketXi& a, const PacketXi& b, const PacketXi& c,
                                                               const PacketXi& d) {
  return __riscv_vcreate_v_i32m1_i32m4(a, b, c, d);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pcast<PacketXi, PacketMul4Xf>(const PacketXi& a, const PacketXi& b, const PacketXi& c,
                                                               const PacketXi& d) {
  return __riscv_vcreate_v_f32m1_f32m4(__riscv_vfcvt_f_x_v_f32m1(a, unpacket_traits<PacketXi>::size),
                                       __riscv_vfcvt_f_x_v_f32m1(b, unpacket_traits<PacketXi>::size),
                                       __riscv_vfcvt_f_x_v_f32m1(c, unpacket_traits<PacketXi>::size),
                                       __riscv_vfcvt_f_x_v_f32m1(d, unpacket_traits<PacketXi>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xf pcast<PacketXf, PacketMul4Xf>(const PacketXf& a, const PacketXf& b, const PacketXf& c,
                                                               const PacketXf& d) {
  return __riscv_vcreate_v_f32m1_f32m4(a, b, c, d);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xi pcast<PacketXf, PacketMul4Xi>(const PacketXf& a, const PacketXf& b, const PacketXf& c,
                                                               const PacketXf& d) {
  return __riscv_vcreate_v_i32m1_i32m4(__riscv_vfcvt_rtz_x_f_v_i32m1(a, unpacket_traits<PacketXf>::size),
                                       __riscv_vfcvt_rtz_x_f_v_i32m1(b, unpacket_traits<PacketXf>::size),
                                       __riscv_vfcvt_rtz_x_f_v_i32m1(c, unpacket_traits<PacketXf>::size),
                                       __riscv_vfcvt_rtz_x_f_v_i32m1(d, unpacket_traits<PacketXf>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pcast<PacketXi, PacketMul2Xi>(const PacketXi& a, const PacketXi& b) {
  return __riscv_vcreate_v_i32m1_i32m2(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pcast<PacketXi, PacketMul2Xf>(const PacketXi& a, const PacketXi& b) {
  return __riscv_vcreate_v_f32m1_f32m2(__riscv_vfcvt_f_x_v_f32m1(a, unpacket_traits<PacketXi>::size),
                                       __riscv_vfcvt_f_x_v_f32m1(b, unpacket_traits<PacketXi>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xf pcast<PacketXf, PacketMul2Xf>(const PacketXf& a, const PacketXf& b) {
  return __riscv_vcreate_v_f32m1_f32m2(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xi pcast<PacketXf, PacketMul2Xi>(const PacketXf& a, const PacketXf& b) {
  return __riscv_vcreate_v_i32m1_i32m2(__riscv_vfcvt_rtz_x_f_v_i32m1(a, unpacket_traits<PacketXf>::size),
                                       __riscv_vfcvt_rtz_x_f_v_i32m1(b, unpacket_traits<PacketXf>::size));
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
EIGEN_STRONG_INLINE PacketXd pcast<PacketXl, PacketXd>(const PacketXl& a) {
  return __riscv_vfcvt_f_x_v_f64m1(a, unpacket_traits<PacketXl>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXl pcast<PacketXd, PacketXl>(const PacketXd& a) {
  return __riscv_vfcvt_rtz_x_f_v_i64m1(a, unpacket_traits<PacketXd>::size);
}

template <>
EIGEN_STRONG_INLINE PacketXd preinterpret<PacketXd, PacketXl>(const PacketXl& a) {
  return __riscv_vreinterpret_v_i64m1_f64m1(a);
}

template <>
EIGEN_STRONG_INLINE PacketXl preinterpret<PacketXl, PacketXd>(const PacketXd& a) {
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
EIGEN_STRONG_INLINE PacketMul4Xl pcast<PacketXl, PacketMul4Xl>(const PacketXl& a, const PacketXl& b, const PacketXl& c,
                                                               const PacketXl& d) {
  return __riscv_vcreate_v_i64m1_i64m4(a, b, c, d);
  ;
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pcast<PacketXl, PacketMul4Xd>(const PacketXl& a, const PacketXl& b, const PacketXl& c,
                                                               const PacketXl& d) {
  return __riscv_vcreate_v_f64m1_f64m4(__riscv_vfcvt_f_x_v_f64m1(a, unpacket_traits<PacketXl>::size),
                                       __riscv_vfcvt_f_x_v_f64m1(b, unpacket_traits<PacketXl>::size),
                                       __riscv_vfcvt_f_x_v_f64m1(c, unpacket_traits<PacketXl>::size),
                                       __riscv_vfcvt_f_x_v_f64m1(d, unpacket_traits<PacketXl>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xd pcast<PacketXd, PacketMul4Xd>(const PacketXd& a, const PacketXd& b, const PacketXd& c,
                                                               const PacketXd& d) {
  return __riscv_vcreate_v_f64m1_f64m4(a, b, c, d);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xl pcast<PacketXd, PacketMul4Xl>(const PacketXd& a, const PacketXd& b, const PacketXd& c,
                                                               const PacketXd& d) {
  return __riscv_vcreate_v_i64m1_i64m4(__riscv_vfcvt_rtz_x_f_v_i64m1(a, unpacket_traits<PacketXd>::size),
                                       __riscv_vfcvt_rtz_x_f_v_i64m1(b, unpacket_traits<PacketXd>::size),
                                       __riscv_vfcvt_rtz_x_f_v_i64m1(c, unpacket_traits<PacketXd>::size),
                                       __riscv_vfcvt_rtz_x_f_v_i64m1(d, unpacket_traits<PacketXd>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pcast<PacketXl, PacketMul2Xl>(const PacketXl& a, const PacketXl& b) {
  return __riscv_vcreate_v_i64m1_i64m2(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pcast<PacketXl, PacketMul2Xd>(const PacketXl& a, const PacketXl& b) {
  return __riscv_vcreate_v_f64m1_f64m2(__riscv_vfcvt_f_x_v_f64m1(a, unpacket_traits<PacketXl>::size),
                                       __riscv_vfcvt_f_x_v_f64m1(b, unpacket_traits<PacketXl>::size));
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xd pcast<PacketXd, PacketMul2Xd>(const PacketXd& a, const PacketXd& b) {
  return __riscv_vcreate_v_f64m1_f64m2(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketMul2Xl pcast<PacketXd, PacketMul2Xl>(const PacketXd& a, const PacketXd& b) {
  return __riscv_vcreate_v_i64m1_i64m2(__riscv_vfcvt_rtz_x_f_v_i64m1(a, unpacket_traits<PacketXd>::size),
                                       __riscv_vfcvt_rtz_x_f_v_i64m1(b, unpacket_traits<PacketXd>::size));
}

/********************************* 16 bits ************************************/

template <>
EIGEN_STRONG_INLINE PacketMul2Xs pcast<PacketXs, PacketMul2Xs>(const PacketXs& a, const PacketXs& b) {
  return __riscv_vcreate_v_i16m1_i16m2(a, b);
}

template <>
EIGEN_STRONG_INLINE PacketMul4Xs pcast<PacketXs, PacketMul4Xs>(const PacketXs& a, const PacketXs& b, const PacketXs& c,
                                                               const PacketXs& d) {
  return __riscv_vcreate_v_i16m1_i16m4(a, b, c, d);
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_TYPE_CASTING_RVV10_H
