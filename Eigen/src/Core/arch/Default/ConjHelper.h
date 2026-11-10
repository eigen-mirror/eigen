// SPDX-License-Identifier: MPL-2.0

// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_ARCH_CONJ_HELPER_H
#define EIGEN_ARCH_CONJ_HELPER_H

#define EIGEN_MAKE_CONJ_HELPER_CPLX_REAL(PACKET_CPLX, PACKET_REAL)                                                  \
  template <bool ConjLhs, bool ConjRhs>                                                                             \
  struct conj_helper<PACKET_REAL, PACKET_CPLX, ConjLhs, ConjRhs> {                                                  \
    EIGEN_STRONG_INLINE PACKET_CPLX pmadd(const PACKET_REAL& x, const PACKET_CPLX& y, const PACKET_CPLX& c) const { \
      return PACKET_CPLX(Eigen::internal::pmadd<PACKET_REAL>(x, conj_if<ConjRhs>().pconj(y).v, c.v));               \
    }                                                                                                               \
    EIGEN_STRONG_INLINE PACKET_CPLX pmsub(const PACKET_REAL& x, const PACKET_CPLX& y, const PACKET_CPLX& c) const { \
      return PACKET_CPLX(Eigen::internal::pmsub<PACKET_REAL>(x, conj_if<ConjRhs>().pconj(y).v, c.v));               \
    }                                                                                                               \
    EIGEN_STRONG_INLINE PACKET_CPLX pmul(const PACKET_REAL& x, const PACKET_CPLX& y) const {                        \
      return PACKET_CPLX(Eigen::internal::pmul<PACKET_REAL>(x, conj_if<ConjRhs>().pconj(y).v));                     \
    }                                                                                                               \
  };                                                                                                                \
                                                                                                                    \
  template <>                                                                                                       \
  struct conj_helper<PACKET_REAL, PACKET_CPLX, true, true> : conj_helper<PACKET_REAL, PACKET_CPLX, false, true> {}; \
                                                                                                                    \
  template <bool ConjLhs, bool ConjRhs>                                                                             \
  struct conj_helper<PACKET_CPLX, PACKET_REAL, ConjLhs, ConjRhs> {                                                  \
    EIGEN_STRONG_INLINE PACKET_CPLX pmadd(const PACKET_CPLX& x, const PACKET_REAL& y, const PACKET_CPLX& c) const { \
      return PACKET_CPLX(Eigen::internal::pmadd<PACKET_REAL>(conj_if<ConjLhs>().pconj(x).v, y, c.v));               \
    }                                                                                                               \
    EIGEN_STRONG_INLINE PACKET_CPLX pmsub(const PACKET_CPLX& x, const PACKET_REAL& y, const PACKET_CPLX& c) const { \
      return PACKET_CPLX(Eigen::internal::pmsub<PACKET_REAL>(conj_if<ConjLhs>().pconj(x).v, y, c.v));               \
    }                                                                                                               \
    EIGEN_STRONG_INLINE PACKET_CPLX pmul(const PACKET_CPLX& x, const PACKET_REAL& y) const {                        \
      return PACKET_CPLX(Eigen::internal::pmul<PACKET_REAL>(conj_if<ConjLhs>().pconj(x).v, y));                     \
    }                                                                                                               \
  };                                                                                                                \
  template <>                                                                                                       \
  struct conj_helper<PACKET_CPLX, PACKET_REAL, true, true> : conj_helper<PACKET_CPLX, PACKET_REAL, true, false> {};

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

template <bool Conjugate>
struct conj_if;

template <>
struct conj_if<true> {
  template <typename T>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x) const {
    return numext::conj(x);
  }
  template <typename T>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T pconj(const T& x) const {
    return internal::pconj(x);
  }
};

template <>
struct conj_if<false> {
  template <typename T>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T& operator()(const T& x) const {
    return x;
  }
  template <typename T>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T& pconj(const T& x) const {
    return x;
  }
};

// Generic Implementation, assume scalars since the packet-version is
// specialized below.
template <typename LhsType, typename RhsType, bool ConjLhs, bool ConjRhs>
struct conj_helper {
  using ResultType = typename ScalarBinaryOpTraits<LhsType, RhsType>::ReturnType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ResultType pmadd(const LhsType& x, const RhsType& y,
                                                         const ResultType& c) const {
    return this->pmul(x, y) + c;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ResultType pmsub(const LhsType& x, const RhsType& y,
                                                         const ResultType& c) const {
    return this->pmul(x, y) - c;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ResultType pmul(const LhsType& x, const RhsType& y) const {
    return internal::mul(conj_if<ConjLhs>()(x), conj_if<ConjRhs>()(y));
  }
};

template <typename LhsScalar, typename RhsScalar>
struct conj_helper<LhsScalar, RhsScalar, true, true> {
  using ResultType = typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ResultType pmadd(const LhsScalar& x, const RhsScalar& y,
                                                         const ResultType& c) const {
    return this->pmul(x, y) + c;
  }

  // Conjugation can reverse multiplication for noncommutative scalars.
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ResultType pmul(const LhsScalar& x, const RhsScalar& y) const {
    EIGEN_IF_CONSTEXPR ((functor_is_commutative<scalar_product_op<LhsScalar, RhsScalar>>::value)) {
      return numext::conj(internal::mul(x, y));
    } else {
      return internal::mul(numext::conj(x), numext::conj(y));
    }
  }
};

// Implementation with equal type, use packet operations.
template <typename Packet, bool ConjLhs, bool ConjRhs>
struct conj_helper<Packet, Packet, ConjLhs, ConjRhs> {
  using ResultType = Packet;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet pmadd(const Packet& x, const Packet& y, const Packet& c) const {
    return Eigen::internal::pmadd(conj_if<ConjLhs>().pconj(x), conj_if<ConjRhs>().pconj(y), c);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet pmsub(const Packet& x, const Packet& y, const Packet& c) const {
    return Eigen::internal::pmsub(conj_if<ConjLhs>().pconj(x), conj_if<ConjRhs>().pconj(y), c);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet pmul(const Packet& x, const Packet& y) const {
    return Eigen::internal::pmul(conj_if<ConjLhs>().pconj(x), conj_if<ConjRhs>().pconj(y));
  }
};

template <typename Packet>
struct conj_helper<Packet, Packet, true, true> {
  using ResultType = Packet;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet pmadd(const Packet& x, const Packet& y, const Packet& c) const {
    return Eigen::internal::pmadd(pconj(x), pconj(y), c);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet pmsub(const Packet& x, const Packet& y, const Packet& c) const {
    return Eigen::internal::pmsub(pconj(x), pconj(y), c);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet pmul(const Packet& x, const Packet& y) const {
    EIGEN_IF_CONSTEXPR ((functor_is_commutative<scalar_product_op<typename unpacket_traits<Packet>::type>>::value)) {
      return pconj(Eigen::internal::pmul(x, y));
    } else {
      return Eigen::internal::pmul(pconj(x), pconj(y));
    }
  }
};

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_ARCH_CONJ_HELPER_H
