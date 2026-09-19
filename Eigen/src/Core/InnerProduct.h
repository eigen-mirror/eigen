// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2024 Charlie Schlosser <cs.schlosser@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_INNER_PRODUCT_EVAL_H
#define EIGEN_INNER_PRODUCT_EVAL_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// Shared accumulation kernel for dot() (Conj = true) and vector products (Conj = false).
template <typename Lhs, typename Rhs>
struct inner_product_assert {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Lhs)
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Rhs)
  EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(Lhs, Rhs)
#ifndef EIGEN_NO_DEBUG
  static EIGEN_DEVICE_FUNC void run(const Lhs& lhs, const Rhs& rhs) {
    eigen_assert((lhs.size() == rhs.size()) && "Inner product: lhs and rhs vectors must have same size");
  }
#else
  static EIGEN_DEVICE_FUNC void run(const Lhs&, const Rhs&) {}
#endif
};

template <typename Func, typename Lhs, typename Rhs>
struct inner_product_evaluator {
  static constexpr int LhsFlags = evaluator<Lhs>::Flags;
  static constexpr int RhsFlags = evaluator<Rhs>::Flags;
  static constexpr int SizeAtCompileTime = size_prefer_fixed(Lhs::SizeAtCompileTime, Rhs::SizeAtCompileTime);
  static constexpr int MaxSizeAtCompileTime =
      min_size_prefer_fixed(Lhs::MaxSizeAtCompileTime, Rhs::MaxSizeAtCompileTime);
  static constexpr int LhsAlignment = evaluator<Lhs>::Alignment;
  static constexpr int RhsAlignment = evaluator<Rhs>::Alignment;

  using Scalar = typename Func::result_type;
  using Packet = typename find_largest_packet<Scalar, SizeAtCompileTime>::type;

  static constexpr bool Vectorize =
      bool(LhsFlags & RhsFlags & PacketAccessBit) && Func::PacketAccess &&
      ((MaxSizeAtCompileTime == Dynamic) || (unpacket_traits<Packet>::size <= MaxSizeAtCompileTime));

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE explicit inner_product_evaluator(const Lhs& lhs, const Rhs& rhs,
                                                                         Func func = Func())
      : m_func(func), m_lhs(lhs), m_rhs(rhs), m_size(lhs.size()) {
    inner_product_assert<Lhs, Rhs>::run(lhs, rhs);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index size() const { return m_size.value(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar coeff(Index index) const {
    return m_func.coeff(m_lhs.coeff(index), m_rhs.coeff(index));
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar coeff(const Scalar& value, Index index) const {
    return m_func.coeff(value, m_lhs.coeff(index), m_rhs.coeff(index));
  }

  template <typename PacketType, int LhsMode = LhsAlignment, int RhsMode = RhsAlignment>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketType packet(Index index) const {
    return m_func.packet(m_lhs.template packet<LhsMode, PacketType>(index),
                         m_rhs.template packet<RhsMode, PacketType>(index));
  }

  template <typename PacketType, int LhsMode = LhsAlignment, int RhsMode = RhsAlignment>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketType packet(const PacketType& value, Index index) const {
    return m_func.packet(value, m_lhs.template packet<LhsMode, PacketType>(index),
                         m_rhs.template packet<RhsMode, PacketType>(index));
  }

  const Func m_func;
  const evaluator<Lhs> m_lhs;
  const evaluator<Rhs> m_rhs;
  const variable_if_dynamic<Index, SizeAtCompileTime> m_size;
};

template <typename Evaluator, bool Vectorize = Evaluator::Vectorize>
struct inner_product_impl;

// scalar loop
template <typename Evaluator>
struct inner_product_impl<Evaluator, false> {
  using Scalar = typename Evaluator::Scalar;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar run(const Evaluator& eval) {
    const Index size = eval.size();
    if (size == 0) return Scalar(0);

    Scalar result = eval.coeff(0);
    for (Index k = 1; k < size; k++) {
      result = eval.coeff(result, k);
    }

    return result;
  }
};

// vector loop
template <typename Evaluator>
struct inner_product_impl<Evaluator, true> {
  using UnsignedIndex = std::make_unsigned_t<Index>;
  using Scalar = typename Evaluator::Scalar;
  using Packet = typename Evaluator::Packet;
  static constexpr int PacketSize = unpacket_traits<Packet>::size;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar run(const Evaluator& eval) {
    const UnsignedIndex size = static_cast<UnsignedIndex>(eval.size());
    if (size < PacketSize) return inner_product_impl<Evaluator, false>::run(eval);

    const UnsignedIndex packetEnd = numext::round_down(size, PacketSize);
    const UnsignedIndex quadEnd = numext::round_down(size, 4 * PacketSize);
    const UnsignedIndex numPackets = size / PacketSize;
    const UnsignedIndex numRemPackets = (packetEnd - quadEnd) / PacketSize;

    Packet presult0 = eval.template packet<Packet>(0 * PacketSize);
    if (numPackets >= 2) {
      Packet presult1 = eval.template packet<Packet>(1 * PacketSize);
      if (numPackets >= 3) {
        Packet presult2 = eval.template packet<Packet>(2 * PacketSize);
        if (numPackets >= 4) {
          Packet presult3 = eval.template packet<Packet>(3 * PacketSize);

          for (UnsignedIndex k = 4 * PacketSize; k < quadEnd; k += 4 * PacketSize) {
            presult0 = eval.packet(presult0, k + 0 * PacketSize);
            presult1 = eval.packet(presult1, k + 1 * PacketSize);
            presult2 = eval.packet(presult2, k + 2 * PacketSize);
            presult3 = eval.packet(presult3, k + 3 * PacketSize);
          }

          if (numRemPackets >= 1) {
            presult0 = eval.packet(presult0, quadEnd + 0 * PacketSize);
            if (numRemPackets >= 2) {
              presult1 = eval.packet(presult1, quadEnd + 1 * PacketSize);
              if (numRemPackets == 3) presult2 = eval.packet(presult2, quadEnd + 2 * PacketSize);
            }
          }

          presult2 = padd(presult2, presult3);
        }
        presult1 = padd(presult1, presult2);
      }
      presult0 = padd(presult0, presult1);
    }

    Scalar result = predux(presult0);
    for (UnsignedIndex k = packetEnd; k < size; k++) {
      result = eval.coeff(result, k);
    }

    return result;
  }
};

template <typename LhsScalar, typename RhsScalar, bool Conj>
struct scalar_inner_product_op {
  using result_type = typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type coeff(const LhsScalar& a, const RhsScalar& b) const {
    return (conj_if<Conj>()(a) * b);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type coeff(const result_type& accum, const LhsScalar& a,
                                                          const RhsScalar& b) const {
    return (conj_if<Conj>()(a) * b) + accum;
  }
  static constexpr bool PacketAccess = false;
};

// Partial specialization for packet access if and only if
// LhsScalar == RhsScalar == ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType.
template <typename Scalar, bool Conj>
struct scalar_inner_product_op<
    Scalar,
    std::enable_if_t<std::is_same<typename ScalarBinaryOpTraits<Scalar, Scalar>::ReturnType, Scalar>::value, Scalar>,
    Conj> {
  using result_type = Scalar;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar coeff(const Scalar& a, const Scalar& b) const {
    return pmul(conj_if<Conj>()(a), b);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar coeff(const Scalar& accum, const Scalar& a, const Scalar& b) const {
    return pmadd(conj_if<Conj>()(a), b, accum);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packet(const Packet& a, const Packet& b) const {
    return pmul(conj_if<Conj>().pconj(a), b);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packet(const Packet& accum, const Packet& a, const Packet& b) const {
    return pmadd(conj_if<Conj>().pconj(a), b, accum);
  }
  static constexpr bool PacketAccess = packet_traits<Scalar>::HasMul && packet_traits<Scalar>::HasAdd;
};

template <typename Lhs, typename Rhs, bool Conj>
struct default_inner_product_impl {
  using LhsScalar = typename traits<Lhs>::Scalar;
  using RhsScalar = typename traits<Rhs>::Scalar;
  using Op = scalar_inner_product_op<LhsScalar, RhsScalar, Conj>;
  using Evaluator = inner_product_evaluator<Op, Lhs, Rhs>;
  using result_type = typename Evaluator::Scalar;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type run(const MatrixBase<Lhs>& a, const MatrixBase<Rhs>& b) {
    Evaluator eval(a.derived(), b.derived(), Op());
    return inner_product_impl<Evaluator>::run(eval);
  }
};

template <typename T>
struct unwrap_unary {
  using type = T;
  static constexpr bool HasDirectAccess = bool(traits<type>::Flags & DirectAccessBit);

  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE constexpr T const& get(T const& xpr) { return xpr; }
};

template <typename T>
struct unwrap_unary<T const> : unwrap_unary<T> {};

template <typename Op, typename Xpr>
struct unwrap_unary<CwiseUnaryOp<Op, Xpr>> : unwrap_unary<Xpr> {
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE constexpr typename unwrap_unary<Xpr>::type const& get(
      CwiseUnaryOp<Op, Xpr> const& xpr) {
    return unwrap_unary<Xpr>::get(xpr.nestedExpression());
  }
};

template <typename T, typename Target>
struct rewrap_unary {
  using type = Target;

  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE constexpr Target const& apply(T const&, Target const& target) {
    return target;
  }
};

template <typename T, typename Target>
struct rewrap_unary<T const, Target> : rewrap_unary<T, Target const> {};

template <typename Op, typename Xpr, typename Target>
struct rewrap_unary<CwiseUnaryOp<Op, Xpr>, Target> {
  // unaryExpr() nests its operand as const; Xpr itself need not be (adjoint() nests a non-const Transpose).
  using type = CwiseUnaryOp<Op, std::add_const_t<typename rewrap_unary<Xpr, Target>::type>>;

  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE constexpr type apply(CwiseUnaryOp<Op, Xpr> const& base,
                                                                    Target const& target) {
    return rewrap_unary<Xpr, Target>::apply(base.nestedExpression(), target).unaryExpr(base.functor());
  }
};

// Rewrap coefficient-wise unary operations around contiguous maps only when both
// operands expose storage and at least one lacks a compile-time unit inner stride.
template <typename Lhs, typename Rhs, bool Conj,
          bool MayMap = unwrap_unary<Lhs>::HasDirectAccess && unwrap_unary<Rhs>::HasDirectAccess &&
                        (inner_stride_at_compile_time<typename unwrap_unary<Lhs>::type>::value != 1 ||
                         inner_stride_at_compile_time<typename unwrap_unary<Rhs>::type>::value != 1)>
struct inner_product_dispatch : default_inner_product_impl<Lhs, Rhs, Conj> {};

template <typename Lhs, typename Rhs, bool Conj>
struct inner_product_dispatch<Lhs, Rhs, Conj, true> {
  using Impl = default_inner_product_impl<Lhs, Rhs, Conj>;
  using result_type = typename Impl::result_type;

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type run(const MatrixBase<Lhs>& a, const MatrixBase<Rhs>& b) {
    EIGEN_IF_CONSTEXPR (Conj) {
      return run_general(a, b);
    }
    // Keep tiny products inlined without the remapping and packet-loop setup.
    if (a.size() <= 4) {
      typename Impl::Evaluator eval(a.derived(), b.derived());
      if (eval.size() == 0) return result_type(0);
      result_type result = eval.coeff(0);
      if (eval.size() > 1) result = eval.coeff(result, 1);
      if (eval.size() > 2) result = eval.coeff(result, 2);
      if (eval.size() > 3) result = eval.coeff(result, 3);
      return result;
    }
    return run_large_product(a, b);
  }

  // Keep the larger product kernel out of tiny callers, while dot() retains its existing inlining.
  static EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE result_type run_large_product(const MatrixBase<Lhs>& a,
                                                                           const MatrixBase<Rhs>& b) {
    return run_general(a, b);
  }

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type run_general(const MatrixBase<Lhs>& a,
                                                                       const MatrixBase<Rhs>& b) {
    using LhsUnwrapper = unwrap_unary<Lhs>;
    using RhsUnwrapper = unwrap_unary<Rhs>;
    using LhsInner = typename LhsUnwrapper::type;
    using RhsInner = typename RhsUnwrapper::type;

    LhsInner const& lhs_inner = LhsUnwrapper::get(a.derived());
    RhsInner const& rhs_inner = RhsUnwrapper::get(b.derived());

    if (lhs_inner.innerStride() == 1 && rhs_inner.innerStride() == 1) {
      using LhsMap = Map<Vector<typename LhsInner::Scalar, size_of_xpr_at_compile_time<LhsInner>::value> const,
                         evaluator<LhsInner>::Alignment>;
      using RhsMap = Map<Vector<typename RhsInner::Scalar, size_of_xpr_at_compile_time<RhsInner>::value> const,
                         evaluator<RhsInner>::Alignment>;

      LhsMap const lhs_map(lhs_inner.data(), lhs_inner.size());
      RhsMap const rhs_map(rhs_inner.data(), rhs_inner.size());

      using LhsRewrap = rewrap_unary<Lhs, LhsMap>;
      using RhsRewrap = rewrap_unary<Rhs, RhsMap>;

      return default_inner_product_impl<typename LhsRewrap::type, typename RhsRewrap::type, Conj>::run(
          LhsRewrap::apply(a.derived(), lhs_map), RhsRewrap::apply(b.derived(), rhs_map));
    }

    return Impl::run(a, b);
  }
};

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_INNER_PRODUCT_EVAL_H
