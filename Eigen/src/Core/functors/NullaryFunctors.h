// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_NULLARY_FUNCTORS_H
#define EIGEN_NULLARY_FUNCTORS_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename Scalar>
struct scalar_constant_op {
  EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE scalar_constant_op(const Scalar& other) : m_other(other) {}
  EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE Scalar operator()() const { return m_other; }
  template <typename PacketType>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketType packetOp() const {
    return internal::pset1<PacketType>(m_other);
  }
  const Scalar m_other;
};
template <typename Scalar>
struct functor_traits<scalar_constant_op<Scalar>> {
  enum {
    Cost = 0 /* as the constant value should be loaded in register only once for the whole expression */,
    PacketAccess = packet_traits<Scalar>::Vectorizable,
    IsRepeatable = true
  };
};

template <typename Scalar>
struct scalar_zero_op {
  EIGEN_DEVICE_FUNC scalar_zero_op() = default;
  EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE Scalar operator()() const { return Scalar(0); }
  template <typename PacketType>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketType packetOp() const {
    return internal::pzero<PacketType>(PacketType());
  }
};
template <typename Scalar>
struct functor_traits<scalar_zero_op<Scalar>> : functor_traits<scalar_constant_op<Scalar>> {};

template <typename Scalar>
struct scalar_identity_op {
  template <typename IndexType>
  EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE Scalar operator()(IndexType row, IndexType col) const {
    return row == col ? Scalar(1) : Scalar(0);
  }
};
template <typename Scalar>
struct functor_traits<scalar_identity_op<Scalar>> {
  enum { Cost = NumTraits<Scalar>::AddCost, PacketAccess = false, IsRepeatable = true };
};

template <typename Scalar, bool IsInteger>
struct linspaced_op_impl;

template <typename Scalar>
struct linspaced_op_impl<Scalar, /*IsInteger*/ false> {
  using RealScalar = typename NumTraits<Scalar>::Real;

  EIGEN_DEVICE_FUNC constexpr linspaced_op_impl(const Scalar& low, const Scalar& high, Index num_steps)
      : m_low(low),
        m_high(high),
        m_size1(num_steps == 1 ? 1 : num_steps - 1),
        m_step(num_steps == 1 ? Scalar() : Scalar((high - low) / RealScalar(num_steps - 1))),
        m_flip(numext::abs(high) < numext::abs(low)) {}

  template <typename IndexType>
  EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE Scalar operator()(IndexType i) const {
    if (m_flip)
      return (i == 0) ? m_low : Scalar(m_high - RealScalar(m_size1 - i) * m_step);
    else
      return (i == m_size1) ? m_high : Scalar(m_low + RealScalar(i) * m_step);
  }

  template <typename Packet, typename IndexType>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(IndexType i) const {
    // Principle:
    // [low, ..., low] + ( [step, ..., step] * ( [i, ..., i] + [0, ..., size] ) )
    Packet low = pset1<Packet>(m_low);
    Packet high = pset1<Packet>(m_high);
    Packet step = pset1<Packet>(m_step);
    if (m_flip) {
      Packet pi = plset<Packet>(Scalar(i - m_size1));
      Packet res = pmadd(step, pi, high);
      Packet mask = pcmp_lt(pzero(res), plset<Packet>(Scalar(i)));
      return pselect<Packet>(mask, res, low);
    } else {
      Packet pi = plset<Packet>(Scalar(i));
      Packet res = pmadd(step, pi, low);
      Packet mask = pcmp_lt(pi, pset1<Packet>(Scalar(m_size1)));
      return pselect<Packet>(mask, res, high);
    }
  }

  const Scalar m_low;
  const Scalar m_high;
  const Index m_size1;
  const Scalar m_step;
  const bool m_flip;
};

// Built-in integers compute in the unsigned type of Scalar's width, widened to Index's, which holds num_steps, the
// index i, |high - low| and the offset from low where high - low overflows a signed Scalar or wraps an unsigned one;
// low +- offset then wraps to the coefficient. Other scalars, custom integers included, compute in Scalar.
template <typename Scalar, bool IsBuiltin = std::is_integral<Scalar>::value>
struct linspaced_integer_traits {
  using Wide = std::make_unsigned_t<std::conditional_t<(sizeof(Scalar) < sizeof(Index)), Index, Scalar>>;
  EIGEN_DEVICE_FUNC static constexpr Scalar narrow(const Wide& x) {
    return static_cast<Scalar>(numext::bit_cast<std::make_signed_t<Wide>>(x));
  }
};

template <typename Scalar>
struct linspaced_integer_traits<Scalar, false> {
  using Wide = Scalar;
  EIGEN_DEVICE_FUNC static constexpr Scalar narrow(const Wide& x) { return x; }
};

template <typename Scalar>
struct linspaced_op_impl<Scalar, /*IsInteger*/ true> {
  using Traits = linspaced_integer_traits<Scalar>;
  using Wide = typename Traits::Wide;

  // Coefficient i is low +- i * m_step with m_step = |high - low| / (num_steps - 1), or, where values repeat because
  // |high - low| + 1 < num_steps, low +- i / m_step with m_step = (num_steps + |high - low|) / (|high - low| + 1).
  EIGEN_DEVICE_FUNC constexpr linspaced_op_impl(const Scalar& low, const Scalar& high, Index num_steps)
      : m_low(low),
        m_decreasing(high < low),
        m_range(high < low ? Wide(Wide(low) - Wide(high)) : Wide(Wide(high) - Wide(low))),
        m_use_divisor(num_steps > 1 && m_range < convert_index<Wide>(num_steps - 1)),
        m_step(m_use_divisor ? (convert_index<Wide>(num_steps) + m_range) / (m_range + Wide(1))
                             : m_range / convert_index<Wide>(num_steps <= 1 ? 1 : num_steps - 1)) {}

  template <typename IndexType>
  EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE Scalar operator()(IndexType i) const {
    const Wide offset = m_use_divisor ? Wide(convert_index<Wide>(i) / m_step) : Wide(convert_index<Wide>(i) * m_step);
    return Traits::narrow(m_decreasing ? Wide(Wide(m_low) - offset) : Wide(Wide(m_low) + offset));
  }

  const Scalar m_low;
  const bool m_decreasing;
  const Wide m_range;
  const bool m_use_divisor;
  const Wide m_step;
};

// ----- Linspace functor ----------------------------------------------------------------

// Forward declaration (we default to random access which does not really give
// us a speed gain when using packet access but it allows to use the functor in
// nested expressions).
template <typename Scalar>
struct linspaced_op;
template <typename Scalar>
struct functor_traits<linspaced_op<Scalar>> {
  enum {
    Cost = 1,
    PacketAccess = (!NumTraits<Scalar>::IsInteger) && packet_traits<Scalar>::HasSetLinear,
    /*&& ((!NumTraits<Scalar>::IsInteger) || packet_traits<Scalar>::HasDiv),*/  // <- vectorization for integer is
                                                                                // currently disabled
    IsRepeatable = true
  };
};
template <typename Scalar>
struct linspaced_op {
  EIGEN_DEVICE_FUNC constexpr linspaced_op(const Scalar& low, const Scalar& high, Index num_steps)
      : impl((num_steps == 1 ? high : low), high, num_steps) {}

  template <typename IndexType>
  EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE Scalar operator()(IndexType i) const {
    return impl(i);
  }

  template <typename Packet, typename IndexType>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(IndexType i) const {
    return impl.template packetOp<Packet>(i);
  }

  // This proxy object handles the actual required temporaries and the different
  // implementations (integer vs. floating point).
  const linspaced_op_impl<Scalar, NumTraits<Scalar>::IsInteger> impl;
};

template <typename Scalar>
struct equalspaced_op {
  EIGEN_DEVICE_FUNC constexpr equalspaced_op(const Scalar& start, const Scalar& step) : m_start(start), m_step(step) {}
  template <typename IndexType>
  EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE Scalar operator()(IndexType i) const {
    return m_start + m_step * static_cast<Scalar>(i);
  }
  template <typename Packet, typename IndexType>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(IndexType i) const {
    const Packet cst_start = pset1<Packet>(m_start);
    const Packet cst_step = pset1<Packet>(m_step);
    const Packet cst_lin0 = plset<Packet>(Scalar(0));
    const Packet cst_offset = pmadd(cst_lin0, cst_step, cst_start);

    Packet i_packet = pset1<Packet>(static_cast<Scalar>(i));
    return pmadd(i_packet, cst_step, cst_offset);
  }
  const Scalar m_start;
  const Scalar m_step;
};

template <typename Scalar>
struct functor_traits<equalspaced_op<Scalar>> {
  enum {
    Cost = NumTraits<Scalar>::AddCost + NumTraits<Scalar>::MulCost,
    PacketAccess =
        packet_traits<Scalar>::HasSetLinear && packet_traits<Scalar>::HasMul && packet_traits<Scalar>::HasAdd,
    IsRepeatable = true
  };
};

// Linear access is automatically determined from the operator() prototypes available for the given functor.
// If it exposes an operator()(i,j), then we assume the i and j coefficients are required independently
// and linear access is not possible. In all other cases, linear access is enabled.
// Users should not have to deal with this structure.
template <typename Functor>
using functor_has_linear_access = bool_constant<!has_binary_operator<Functor>::value>;

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_NULLARY_FUNCTORS_H
