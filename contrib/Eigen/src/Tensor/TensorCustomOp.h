// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TENSOR_TENSOR_CUSTOM_OP_H
#define EIGEN_TENSOR_TENSOR_CUSTOM_OP_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {
template <typename CustomUnaryFunc, typename XprType>
struct traits<TensorCustomUnaryOp<CustomUnaryFunc, XprType> > {
  using Scalar = typename XprType::Scalar;
  using StorageKind = typename XprType::StorageKind;
  using Index = typename XprType::Index;
  // The functor's dimensions() determines the output shape, so the rank of the
  // result may differ from the rank of the input. The argument is spelled
  // exactly as in the evaluator's call so both resolve to the same overload.
  using CustomDimensions = remove_all_t<decltype(std::declval<const CustomUnaryFunc&>().dimensions(
      std::declval<const remove_all_t<typename XprType::Nested>&>()))>;
  static constexpr ptrdiff_t CustomRank = array_size<CustomDimensions>::value;
  static_assert(CustomRank >= 0,
                "The dimensions() method of a custom tensor functor must return a fixed-rank "
                "array-like type such as DSizes<Index, Rank>.");
  // Clamped so a failed assertion doesn't cascade into DSizes<Index, -1> errors.
  static constexpr int NumDimensions = CustomRank < 0 ? 1 : static_cast<int>(CustomRank);
  static constexpr int Layout = traits<XprType>::Layout;
  using PointerType = typename traits<XprType>::PointerType;
  enum { Flags = 0 };
};

template <typename CustomUnaryFunc, typename XprType>
struct eval<TensorCustomUnaryOp<CustomUnaryFunc, XprType>, Eigen::Dense> {
  using type = const TensorCustomUnaryOp<CustomUnaryFunc, XprType> EIGEN_DEVICE_REF;
};

}  // end namespace internal

/**
 * \ingroup Tensor_Module
 *
 * \brief Tensor custom class.
 */
template <typename CustomUnaryFunc, typename XprType>
class TensorCustomUnaryOp : public TensorBase<TensorCustomUnaryOp<CustomUnaryFunc, XprType>, ReadOnlyAccessors> {
 public:
  using Scalar = typename internal::traits<TensorCustomUnaryOp>::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using CoeffReturnType = typename XprType::CoeffReturnType;
  using Nested = typename internal::ref_selector<TensorCustomUnaryOp>::non_const_type;
  using StorageKind = typename internal::traits<TensorCustomUnaryOp>::StorageKind;
  using Index = typename internal::traits<TensorCustomUnaryOp>::Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorCustomUnaryOp(const XprType& expr, const CustomUnaryFunc& func)
      : m_expr(expr), m_func(func) {}

  EIGEN_DEVICE_FUNC const CustomUnaryFunc& func() const { return m_func; }

  EIGEN_DEVICE_FUNC const internal::remove_all_t<typename XprType::Nested>& expression() const { return m_expr; }

 protected:
  typename XprType::Nested m_expr;
  const CustomUnaryFunc m_func;
};

// Eval as rvalue
template <typename CustomUnaryFunc, typename XprType, typename Device>
struct TensorEvaluator<const TensorCustomUnaryOp<CustomUnaryFunc, XprType>, Device> {
  using ArgType = TensorCustomUnaryOp<CustomUnaryFunc, XprType>;
  using Index = typename internal::traits<ArgType>::Index;
  static constexpr int NumDims = internal::traits<ArgType>::NumDimensions;
  using Dimensions = DSizes<Index, NumDims>;
  using Scalar = std::remove_const_t<typename ArgType::Scalar>;
  using CoeffReturnType = std::remove_const_t<typename XprType::CoeffReturnType>;
  using PacketReturnType = typename PacketType<CoeffReturnType, Device>::type;
  static constexpr int PacketSize = PacketType<CoeffReturnType, Device>::size;
  using TensorPointerType = typename Eigen::internal::traits<XprType>::PointerType;
  using Storage = StorageMemory<CoeffReturnType, Device>;
  using EvaluatorPointerType = typename Storage::Type;

  static constexpr int Layout = TensorEvaluator<XprType, Device>::Layout;
  enum {
    IsAligned = false,
    PacketAccess = (PacketType<CoeffReturnType, Device>::size > 1),
    // The custom op is eagerly evaluated into a dense buffer (m_result), so
    // blocks and raw storage can be served straight from it, exactly like
    // TensorForcedEvalOp. Without these flags a custom op disables tiled
    // evaluation for any expression containing it and hides its buffer from
    // consumers with data()-based fast paths.
    BlockAccess = internal::is_arithmetic<CoeffReturnType>::value,
    PreferBlockAccess = false,
    CoordAccess = false,  // to be implemented
    RawAccess = true
  };

  //===- Tensor block evaluation strategy (see TensorBlock.h) -------------===//
  using TensorBlockDesc = internal::TensorBlockDescriptor<NumDims, Index>;
  using TensorBlockScratch = internal::TensorBlockScratchAllocator<Device>;

  using TensorBlock = typename internal::TensorMaterializedBlock<CoeffReturnType, NumDims, Layout, Index>;
  //===--------------------------------------------------------------------===//

  // The functor's dimensions() may return an index type that promotes to Index.
  EIGEN_STRONG_INLINE TensorEvaluator(const ArgType& op, const Device& device)
      : m_dimensions(op.func().dimensions(op.expression())), m_op(op), m_device(device), m_result(nullptr) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(EvaluatorPointerType data) {
    if (data) {
      evalTo(data);
      return false;
    } else {
      m_result = static_cast<EvaluatorPointerType>(
          m_device.get((CoeffReturnType*)m_device.allocate_temp(dimensions().TotalSize() * sizeof(CoeffReturnType))));
      evalTo(m_result);
      return true;
    }
  }

  EIGEN_STRONG_INLINE void cleanup() {
    if (m_result) {
      m_device.deallocate_temp(m_result);
      m_result = nullptr;
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const { return m_result[index]; }

  template <int LoadMode>
  EIGEN_DEVICE_FUNC PacketReturnType packet(Index index) const {
    return internal::ploadt<PacketReturnType, LoadMode>(m_result + index);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost costPerCoeff(bool vectorized) const {
    // TODO(rmlarsen): Extend CustomOp API to return its cost estimate.
    return TensorOpCost(sizeof(CoeffReturnType), 0, 0, vectorized, PacketSize);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE internal::TensorBlockResourceRequirements getResourceRequirements() const {
    return internal::TensorBlockResourceRequirements::any();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorBlock block(TensorBlockDesc& desc, TensorBlockScratch& scratch,
                                                          bool /*root_of_expr_ast*/ = false) const {
    eigen_assert(m_result != nullptr);
    return TensorBlock::materialize(m_result, m_dimensions, desc, scratch);
  }

  EIGEN_DEVICE_FUNC EvaluatorPointerType data() const { return m_result; }

 protected:
  void evalTo(EvaluatorPointerType data) {
    TensorMap<Tensor<CoeffReturnType, NumDims, Layout, Index> > result(m_device.get(data), m_dimensions);
    m_op.func().eval(m_op.expression(), result, m_device);
  }

  Dimensions m_dimensions;
  const ArgType m_op;
  const Device EIGEN_DEVICE_REF m_device;
  EvaluatorPointerType m_result;
};

/** \class TensorCustomBinaryOp
 * \ingroup Tensor_Module
 *
 * \brief Tensor custom class.
 *
 *
 */
namespace internal {
template <typename CustomBinaryFunc, typename LhsXprType, typename RhsXprType>
struct traits<TensorCustomBinaryOp<CustomBinaryFunc, LhsXprType, RhsXprType> > {
  using Scalar = typename internal::promote_storage_type<typename LhsXprType::Scalar, typename RhsXprType::Scalar>::ret;
  using CoeffReturnType = typename internal::promote_storage_type<typename LhsXprType::CoeffReturnType,
                                                                  typename RhsXprType::CoeffReturnType>::ret;
  using StorageKind = typename promote_storage_type<typename traits<LhsXprType>::StorageKind,
                                                    typename traits<RhsXprType>::StorageKind>::ret;
  using Index =
      typename promote_index_type<typename traits<LhsXprType>::Index, typename traits<RhsXprType>::Index>::type;
  // The functor's dimensions() determines the output shape, so the rank of the
  // result may differ from the ranks of the inputs. The arguments are spelled
  // exactly as in the evaluator's call so both resolve to the same overload.
  using CustomDimensions = remove_all_t<decltype(std::declval<const CustomBinaryFunc&>().dimensions(
      std::declval<const remove_all_t<typename LhsXprType::Nested>&>(),
      std::declval<const remove_all_t<typename RhsXprType::Nested>&>()))>;
  static constexpr ptrdiff_t CustomRank = array_size<CustomDimensions>::value;
  static_assert(CustomRank >= 0,
                "The dimensions() method of a custom tensor functor must return a fixed-rank "
                "array-like type such as DSizes<Index, Rank>.");
  // Clamped so a failed assertion doesn't cascade into DSizes<Index, -1> errors.
  static constexpr int NumDimensions = CustomRank < 0 ? 1 : static_cast<int>(CustomRank);
  static constexpr int Layout = traits<LhsXprType>::Layout;
  using PointerType =
      std::conditional_t<Pointer_type_promotion<typename LhsXprType::Scalar, Scalar>::val,
                         typename traits<LhsXprType>::PointerType, typename traits<RhsXprType>::PointerType>;
  enum { Flags = 0 };
};

template <typename CustomBinaryFunc, typename LhsXprType, typename RhsXprType>
struct eval<TensorCustomBinaryOp<CustomBinaryFunc, LhsXprType, RhsXprType>, Eigen::Dense> {
  using type = const TensorCustomBinaryOp<CustomBinaryFunc, LhsXprType, RhsXprType>&;
};

}  // end namespace internal

template <typename CustomBinaryFunc, typename LhsXprType, typename RhsXprType>
class TensorCustomBinaryOp
    : public TensorBase<TensorCustomBinaryOp<CustomBinaryFunc, LhsXprType, RhsXprType>, ReadOnlyAccessors> {
 public:
  using Scalar = typename internal::traits<TensorCustomBinaryOp>::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using CoeffReturnType = typename internal::traits<TensorCustomBinaryOp>::CoeffReturnType;
  using Nested = typename internal::ref_selector<TensorCustomBinaryOp>::non_const_type;
  using StorageKind = typename internal::traits<TensorCustomBinaryOp>::StorageKind;
  using Index = typename internal::traits<TensorCustomBinaryOp>::Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorCustomBinaryOp(const LhsXprType& lhs, const RhsXprType& rhs,
                                                             const CustomBinaryFunc& func)

      : m_lhs_xpr(lhs), m_rhs_xpr(rhs), m_func(func) {}

  EIGEN_DEVICE_FUNC const CustomBinaryFunc& func() const { return m_func; }

  EIGEN_DEVICE_FUNC const internal::remove_all_t<typename LhsXprType::Nested>& lhsExpression() const {
    return m_lhs_xpr;
  }

  EIGEN_DEVICE_FUNC const internal::remove_all_t<typename RhsXprType::Nested>& rhsExpression() const {
    return m_rhs_xpr;
  }

 protected:
  typename LhsXprType::Nested m_lhs_xpr;
  typename RhsXprType::Nested m_rhs_xpr;
  const CustomBinaryFunc m_func;
};

// Eval as rvalue
template <typename CustomBinaryFunc, typename LhsXprType, typename RhsXprType, typename Device>
struct TensorEvaluator<const TensorCustomBinaryOp<CustomBinaryFunc, LhsXprType, RhsXprType>, Device> {
  using XprType = TensorCustomBinaryOp<CustomBinaryFunc, LhsXprType, RhsXprType>;
  using Index = typename internal::traits<XprType>::Index;
  static constexpr int NumDims = internal::traits<XprType>::NumDimensions;
  using Dimensions = DSizes<Index, NumDims>;
  using Scalar = typename XprType::Scalar;
  using CoeffReturnType = std::remove_const_t<typename XprType::CoeffReturnType>;
  using PacketReturnType = typename PacketType<CoeffReturnType, Device>::type;
  static constexpr int PacketSize = PacketType<CoeffReturnType, Device>::size;

  using TensorPointerType = typename Eigen::internal::traits<XprType>::PointerType;
  using Storage = StorageMemory<CoeffReturnType, Device>;
  using EvaluatorPointerType = typename Storage::Type;

  static constexpr int Layout = TensorEvaluator<LhsXprType, Device>::Layout;
  enum {
    IsAligned = false,
    PacketAccess = (PacketType<CoeffReturnType, Device>::size > 1),
    // See the unary evaluator above: serve blocks and raw storage from the
    // eagerly materialized buffer, like TensorForcedEvalOp.
    BlockAccess = internal::is_arithmetic<CoeffReturnType>::value,
    PreferBlockAccess = false,
    CoordAccess = false,  // to be implemented
    RawAccess = true
  };

  //===- Tensor block evaluation strategy (see TensorBlock.h) -------------===//
  using TensorBlockDesc = internal::TensorBlockDescriptor<NumDims, Index>;
  using TensorBlockScratch = internal::TensorBlockScratchAllocator<Device>;

  using TensorBlock = typename internal::TensorMaterializedBlock<CoeffReturnType, NumDims, Layout, Index>;
  //===--------------------------------------------------------------------===//

  // The functor's dimensions() may return an index type that promotes to Index.
  EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_dimensions(op.func().dimensions(op.lhsExpression(), op.rhsExpression())),
        m_op(op),
        m_device(device),
        m_result(nullptr) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(EvaluatorPointerType data) {
    if (data) {
      evalTo(data);
      return false;
    } else {
      m_result = static_cast<EvaluatorPointerType>(
          m_device.get((CoeffReturnType*)m_device.allocate_temp(dimensions().TotalSize() * sizeof(CoeffReturnType))));
      evalTo(m_result);
      return true;
    }
  }

  EIGEN_STRONG_INLINE void cleanup() {
    if (m_result != nullptr) {
      m_device.deallocate_temp(m_result);
      m_result = nullptr;
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const { return m_result[index]; }

  template <int LoadMode>
  EIGEN_DEVICE_FUNC PacketReturnType packet(Index index) const {
    return internal::ploadt<PacketReturnType, LoadMode>(m_result + index);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost costPerCoeff(bool vectorized) const {
    // TODO(rmlarsen): Extend CustomOp API to return its cost estimate.
    return TensorOpCost(sizeof(CoeffReturnType), 0, 0, vectorized, PacketSize);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE internal::TensorBlockResourceRequirements getResourceRequirements() const {
    return internal::TensorBlockResourceRequirements::any();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorBlock block(TensorBlockDesc& desc, TensorBlockScratch& scratch,
                                                          bool /*root_of_expr_ast*/ = false) const {
    eigen_assert(m_result != nullptr);
    return TensorBlock::materialize(m_result, m_dimensions, desc, scratch);
  }

  EIGEN_DEVICE_FUNC EvaluatorPointerType data() const { return m_result; }

 protected:
  void evalTo(EvaluatorPointerType data) {
    // The Output type handed to eval() is a compatibility surface: functors are
    // compiled against a DenseIndex-typed map, so widen its index type only
    // when the expressions' promoted Index is strictly wider than DenseIndex.
    // DenseIndex must stay the first argument: promote_index_type keeps that
    // one on a tie, which preserves the map type for equal-width distinct
    // index types such as long long versus long.
    using MapIndex = typename internal::promote_index_type<DenseIndex, Index>::type;
    TensorMap<Tensor<CoeffReturnType, NumDims, Layout, MapIndex> > result(m_device.get(data), m_dimensions);
    m_op.func().eval(m_op.lhsExpression(), m_op.rhsExpression(), result, m_device);
  }

  Dimensions m_dimensions;
  const XprType m_op;
  const Device EIGEN_DEVICE_REF m_device;
  EvaluatorPointerType m_result;
};

}  // end namespace Eigen

#endif  // EIGEN_TENSOR_TENSOR_CUSTOM_OP_H
