// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TENSOR_TENSOR_EXPR_H
#define EIGEN_TENSOR_TENSOR_EXPR_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {
template <typename NullaryOp, typename XprType>
struct traits<TensorCwiseNullaryOp<NullaryOp, XprType> > : traits<XprType> {
  using XprTraits = traits<XprType>;
  using Scalar = typename XprType::Scalar;
  static constexpr int NumDimensions = XprTraits::NumDimensions;
  static constexpr int Layout = XprTraits::Layout;
  using PointerType = typename XprTraits::PointerType;
  enum { Flags = 0 };
};

}  // end namespace internal

/**
 * \ingroup Tensor_Module
 *
 * \brief Tensor nullary expression.
 *
 * The TensorCwiseNullaryOp class applies a nullary operator to an expression.
 * This is typically used to generate constants.
 */
template <typename NullaryOp, typename XprType>
class TensorCwiseNullaryOp : public TensorBase<TensorCwiseNullaryOp<NullaryOp, XprType>, ReadOnlyAccessors> {
 public:
  using Scalar = typename Eigen::internal::traits<TensorCwiseNullaryOp>::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using CoeffReturnType = typename XprType::CoeffReturnType;
  using Nested = typename Eigen::internal::ref_selector<TensorCwiseNullaryOp>::non_const_type;
  using StorageKind = typename Eigen::internal::traits<TensorCwiseNullaryOp>::StorageKind;
  using Index = typename Eigen::internal::traits<TensorCwiseNullaryOp>::Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorCwiseNullaryOp(const XprType& xpr, const NullaryOp& func = NullaryOp())
      : m_xpr(xpr), m_functor(func) {}

  EIGEN_DEVICE_FUNC const internal::remove_all_t<typename XprType::Nested>& nestedExpression() const { return m_xpr; }

  EIGEN_DEVICE_FUNC const NullaryOp& functor() const { return m_functor; }

 protected:
  typename XprType::Nested m_xpr;
  const NullaryOp m_functor;
};

namespace internal {
template <typename UnaryOp, typename XprType>
struct traits<TensorCwiseUnaryOp<UnaryOp, XprType> > : traits<XprType> {
  // TODO(phli): Add InputScalar, InputPacket.  Check references to
  // current Scalar/Packet to see if the intent is Input or Output.
  using Scalar = typename result_of<UnaryOp(typename XprType::Scalar)>::type;
  using XprTraits = traits<XprType>;
  static constexpr int NumDimensions = XprTraits::NumDimensions;
  static constexpr int Layout = XprTraits::Layout;
  using PointerType = typename TypeConversion<Scalar, typename XprTraits::PointerType>::type;
};

template <typename UnaryOp, typename XprType>
struct eval<TensorCwiseUnaryOp<UnaryOp, XprType>, Eigen::Dense> {
  using type = const TensorCwiseUnaryOp<UnaryOp, XprType>&;
};

}  // end namespace internal

/**
 * \ingroup Tensor_Module
 *
 * \brief Tensor unary expression.
 *
 * The TensorCwiseUnaryOp class represents an expression where a unary operator
 * (e.g. cwiseSqrt) is applied to an expression.
 */
template <typename UnaryOp, typename XprType>
class TensorCwiseUnaryOp : public TensorBase<TensorCwiseUnaryOp<UnaryOp, XprType>, ReadOnlyAccessors> {
 public:
  // TODO(phli): Add InputScalar, InputPacket.  Check references to
  // current Scalar/Packet to see if the intent is Input or Output.
  using Scalar = typename Eigen::internal::traits<TensorCwiseUnaryOp>::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using CoeffReturnType = Scalar;
  using Nested = typename Eigen::internal::ref_selector<TensorCwiseUnaryOp>::type;
  using StorageKind = typename Eigen::internal::traits<TensorCwiseUnaryOp>::StorageKind;
  using Index = typename Eigen::internal::traits<TensorCwiseUnaryOp>::Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorCwiseUnaryOp(const XprType& xpr, const UnaryOp& func = UnaryOp())
      : m_xpr(xpr), m_functor(func) {}

  EIGEN_DEVICE_FUNC const UnaryOp& functor() const { return m_functor; }

  /** \returns the nested expression */
  EIGEN_DEVICE_FUNC const internal::remove_all_t<typename XprType::Nested>& nestedExpression() const { return m_xpr; }

 protected:
  typename XprType::Nested m_xpr;
  const UnaryOp m_functor;
};

namespace internal {
template <typename BinaryOp, typename LhsXprType, typename RhsXprType>
struct traits<TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType> > {
  // Type promotion to handle the case where the types of the lhs and the rhs
  // are different.
  // TODO(phli): Add Lhs/RhsScalar, Lhs/RhsPacket.  Check references to
  // current Scalar/Packet to see if the intent is Inputs or Output.
  using Scalar = typename result_of<BinaryOp(typename LhsXprType::Scalar, typename RhsXprType::Scalar)>::type;
  using XprTraits = traits<LhsXprType>;
  using StorageKind = typename promote_storage_type<typename traits<LhsXprType>::StorageKind,
                                                    typename traits<RhsXprType>::StorageKind>::ret;
  using Index =
      typename promote_index_type<typename traits<LhsXprType>::Index, typename traits<RhsXprType>::Index>::type;
  static constexpr int NumDimensions = XprTraits::NumDimensions;
  static constexpr int Layout = XprTraits::Layout;
  using PointerType = typename TypeConversion<
      Scalar,
      std::conditional_t<Pointer_type_promotion<typename LhsXprType::Scalar, Scalar>::val,
                         typename traits<LhsXprType>::PointerType, typename traits<RhsXprType>::PointerType> >::type;
  enum { Flags = 0 };
};

template <typename BinaryOp, typename LhsXprType, typename RhsXprType>
struct eval<TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>, Eigen::Dense> {
  using type = const TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>&;
};

}  // end namespace internal

/**
 * \ingroup Tensor_Module
 *
 * \brief Tensor binary expression.
 *
 * The TensorCwiseBinaryOp class represents an expression where a binary
 * operator (e.g. addition) is applied to a lhs and a rhs expression.
 */
template <typename BinaryOp, typename LhsXprType, typename RhsXprType>
class TensorCwiseBinaryOp
    : public TensorBase<TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>, ReadOnlyAccessors> {
 public:
  // TODO(phli): Add Lhs/RhsScalar, Lhs/RhsPacket.  Check references to
  // current Scalar/Packet to see if the intent is Inputs or Output.
  using Scalar = typename Eigen::internal::traits<TensorCwiseBinaryOp>::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using CoeffReturnType = Scalar;
  using Nested = typename Eigen::internal::ref_selector<TensorCwiseBinaryOp>::type;
  using StorageKind = typename Eigen::internal::traits<TensorCwiseBinaryOp>::StorageKind;
  using Index = typename Eigen::internal::traits<TensorCwiseBinaryOp>::Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorCwiseBinaryOp(const LhsXprType& lhs, const RhsXprType& rhs,
                                                            const BinaryOp& func = BinaryOp())
      : m_lhs_xpr(lhs), m_rhs_xpr(rhs), m_functor(func) {}

  EIGEN_DEVICE_FUNC const BinaryOp& functor() const { return m_functor; }

  /** \returns the nested expressions */
  EIGEN_DEVICE_FUNC const internal::remove_all_t<typename LhsXprType::Nested>& lhsExpression() const {
    return m_lhs_xpr;
  }

  EIGEN_DEVICE_FUNC const internal::remove_all_t<typename RhsXprType::Nested>& rhsExpression() const {
    return m_rhs_xpr;
  }

 protected:
  typename LhsXprType::Nested m_lhs_xpr;
  typename RhsXprType::Nested m_rhs_xpr;
  const BinaryOp m_functor;
};

namespace internal {
template <typename TernaryOp, typename Arg1XprType, typename Arg2XprType, typename Arg3XprType>
struct traits<TensorCwiseTernaryOp<TernaryOp, Arg1XprType, Arg2XprType, Arg3XprType> > {
  // Type promotion to handle the case where the types of the args are different.
  using Scalar = typename result_of<TernaryOp(typename Arg1XprType::Scalar, typename Arg2XprType::Scalar,
                                              typename Arg3XprType::Scalar)>::type;
  using XprTraits = traits<Arg1XprType>;
  using StorageKind = typename traits<Arg1XprType>::StorageKind;
  using Index = typename traits<Arg1XprType>::Index;
  static constexpr int NumDimensions = XprTraits::NumDimensions;
  static constexpr int Layout = XprTraits::Layout;
  using PointerType = typename TypeConversion<
      Scalar,
      std::conditional_t<Pointer_type_promotion<typename Arg2XprType::Scalar, Scalar>::val,
                         typename traits<Arg2XprType>::PointerType, typename traits<Arg3XprType>::PointerType> >::type;
  enum { Flags = 0 };
};

template <typename TernaryOp, typename Arg1XprType, typename Arg2XprType, typename Arg3XprType>
struct eval<TensorCwiseTernaryOp<TernaryOp, Arg1XprType, Arg2XprType, Arg3XprType>, Eigen::Dense> {
  using type = const TensorCwiseTernaryOp<TernaryOp, Arg1XprType, Arg2XprType, Arg3XprType>&;
};

}  // end namespace internal

template <typename TernaryOp, typename Arg1XprType, typename Arg2XprType, typename Arg3XprType>
class TensorCwiseTernaryOp
    : public TensorBase<TensorCwiseTernaryOp<TernaryOp, Arg1XprType, Arg2XprType, Arg3XprType>, ReadOnlyAccessors> {
 public:
  using Scalar = typename Eigen::internal::traits<TensorCwiseTernaryOp>::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using CoeffReturnType = Scalar;
  using Nested = typename Eigen::internal::ref_selector<TensorCwiseTernaryOp>::type;
  using StorageKind = typename Eigen::internal::traits<TensorCwiseTernaryOp>::StorageKind;
  using Index = typename Eigen::internal::traits<TensorCwiseTernaryOp>::Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorCwiseTernaryOp(const Arg1XprType& arg1, const Arg2XprType& arg2,
                                                             const Arg3XprType& arg3,
                                                             const TernaryOp& func = TernaryOp())
      : m_arg1_xpr(arg1), m_arg2_xpr(arg2), m_arg3_xpr(arg3), m_functor(func) {}

  EIGEN_DEVICE_FUNC const TernaryOp& functor() const { return m_functor; }

  /** \returns the nested expressions */
  EIGEN_DEVICE_FUNC const internal::remove_all_t<typename Arg1XprType::Nested>& arg1Expression() const {
    return m_arg1_xpr;
  }

  EIGEN_DEVICE_FUNC const internal::remove_all_t<typename Arg2XprType::Nested>& arg2Expression() const {
    return m_arg2_xpr;
  }

  EIGEN_DEVICE_FUNC const internal::remove_all_t<typename Arg3XprType::Nested>& arg3Expression() const {
    return m_arg3_xpr;
  }

 protected:
  typename Arg1XprType::Nested m_arg1_xpr;
  typename Arg2XprType::Nested m_arg2_xpr;
  typename Arg3XprType::Nested m_arg3_xpr;
  const TernaryOp m_functor;
};

namespace internal {
template <typename IfXprType, typename ThenXprType, typename ElseXprType>
struct traits<TensorSelectOp<IfXprType, ThenXprType, ElseXprType> > : traits<ThenXprType> {
  using Scalar = typename traits<ThenXprType>::Scalar;
  using XprTraits = traits<ThenXprType>;
  using StorageKind = typename promote_storage_type<typename traits<ThenXprType>::StorageKind,
                                                    typename traits<ElseXprType>::StorageKind>::ret;
  using Index =
      typename promote_index_type<typename traits<ElseXprType>::Index, typename traits<ThenXprType>::Index>::type;
  static constexpr int NumDimensions = XprTraits::NumDimensions;
  static constexpr int Layout = XprTraits::Layout;
  using PointerType =
      std::conditional_t<Pointer_type_promotion<typename ThenXprType::Scalar, Scalar>::val,
                         typename traits<ThenXprType>::PointerType, typename traits<ElseXprType>::PointerType>;
};

template <typename IfXprType, typename ThenXprType, typename ElseXprType>
struct eval<TensorSelectOp<IfXprType, ThenXprType, ElseXprType>, Eigen::Dense> {
  using type = const TensorSelectOp<IfXprType, ThenXprType, ElseXprType>&;
};

}  // end namespace internal

template <typename IfXprType, typename ThenXprType, typename ElseXprType>
class TensorSelectOp : public TensorBase<TensorSelectOp<IfXprType, ThenXprType, ElseXprType>, ReadOnlyAccessors> {
 public:
  using Scalar = typename Eigen::internal::traits<TensorSelectOp>::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using CoeffReturnType = typename internal::promote_storage_type<typename ThenXprType::CoeffReturnType,
                                                                  typename ElseXprType::CoeffReturnType>::ret;
  using Nested = typename Eigen::internal::ref_selector<TensorSelectOp>::type;
  using StorageKind = typename Eigen::internal::traits<TensorSelectOp>::StorageKind;
  using Index = typename Eigen::internal::traits<TensorSelectOp>::Index;

  EIGEN_DEVICE_FUNC TensorSelectOp(const IfXprType& a_condition, const ThenXprType& a_then, const ElseXprType& a_else)
      : m_condition(a_condition), m_then(a_then), m_else(a_else) {}

  EIGEN_DEVICE_FUNC const IfXprType& ifExpression() const { return m_condition; }

  EIGEN_DEVICE_FUNC const ThenXprType& thenExpression() const { return m_then; }

  EIGEN_DEVICE_FUNC const ElseXprType& elseExpression() const { return m_else; }

 protected:
  typename IfXprType::Nested m_condition;
  typename ThenXprType::Nested m_then;
  typename ElseXprType::Nested m_else;
};

}  // end namespace Eigen

#endif  // EIGEN_TENSOR_TENSOR_EXPR_H
