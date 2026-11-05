// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Eugene Brevdo <ebrevdo@gmail.com>
//                    Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TENSOR_TENSOR_ARG_MAX_H
#define EIGEN_TENSOR_TENSOR_ARG_MAX_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

template <typename XprType>
struct traits<TensorIndexPairOp<XprType>> : public traits<XprType> {
  using XprTraits = traits<XprType>;
  using StorageKind = typename XprTraits::StorageKind;
  using Index = typename XprTraits::Index;
  using Scalar = Pair<Index, typename XprTraits::Scalar>;
  static constexpr int NumDimensions = XprTraits::NumDimensions;
  static constexpr int Layout = XprTraits::Layout;
};

template <typename XprType>
struct eval<TensorIndexPairOp<XprType>, Eigen::Dense> {
  using type = const TensorIndexPairOp<XprType> EIGEN_DEVICE_REF;
};

}  // end namespace internal

/**
 * \ingroup Tensor_Module
 *
 * \brief Tensor + Index Pair class.
 */
template <typename XprType>
class TensorIndexPairOp : public TensorBase<TensorIndexPairOp<XprType>, ReadOnlyAccessors> {
 public:
  using Scalar = typename Eigen::internal::traits<TensorIndexPairOp>::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using Nested = typename Eigen::internal::ref_selector<TensorIndexPairOp>::type;
  using StorageKind = typename Eigen::internal::traits<TensorIndexPairOp>::StorageKind;
  using Index = typename Eigen::internal::traits<TensorIndexPairOp>::Index;
  using CoeffReturnType = Pair<Index, typename XprType::CoeffReturnType>;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorIndexPairOp(const XprType& expr) : m_xpr(expr) {}

  EIGEN_DEVICE_FUNC const internal::remove_all_t<typename XprType::Nested>& expression() const { return m_xpr; }

 protected:
  typename XprType::Nested m_xpr;
};

// Eval as rvalue
template <typename ArgType, typename Device>
struct TensorEvaluator<const TensorIndexPairOp<ArgType>, Device> {
  using XprType = TensorIndexPairOp<ArgType>;
  using Index = typename XprType::Index;
  using Scalar = typename XprType::Scalar;
  using CoeffReturnType = typename XprType::CoeffReturnType;

  using Dimensions = typename TensorEvaluator<ArgType, Device>::Dimensions;
  static constexpr int NumDims = internal::array_size<Dimensions>::value;
  using Storage = StorageMemory<CoeffReturnType, Device>;
  using EvaluatorPointerType = typename Storage::Type;

  enum {
    IsAligned = false,
    PacketAccess = false,
    BlockAccess = false,
    PreferBlockAccess = TensorEvaluator<ArgType, Device>::PreferBlockAccess,
    CoordAccess = false,  // to be implemented
    RawAccess = false
  };
  static constexpr int Layout = TensorEvaluator<ArgType, Device>::Layout;

  //===- Tensor block evaluation strategy (see TensorBlock.h) -------------===//
  using TensorBlock = internal::TensorBlockNotImplemented;
  //===--------------------------------------------------------------------===//

  EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device) : m_impl(op.expression(), device) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_impl.dimensions(); }

  EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(EvaluatorPointerType /*data*/) {
    m_impl.evalSubExprsIfNeeded(nullptr);
    return true;
  }
  EIGEN_STRONG_INLINE void cleanup() { m_impl.cleanup(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const {
    return CoeffReturnType(index, m_impl.coeff(index));
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost costPerCoeff(bool vectorized) const {
    return m_impl.costPerCoeff(vectorized) + TensorOpCost(0, 0, 1);
  }

  EIGEN_DEVICE_FUNC EvaluatorPointerType data() const { return nullptr; }

 protected:
  TensorEvaluator<ArgType, Device> m_impl;
};

namespace internal {

/** \class TensorPairReducerOp
 * \ingroup Tensor_Module
 *
 * \brief Converts to Tensor<Pair<Index, Scalar> > and reduces to Tensor<Index>.
 *
 */
template <typename ReduceOp, typename Dims, typename XprType>
struct traits<TensorPairReducerOp<ReduceOp, Dims, XprType>> : public traits<XprType> {
  using XprTraits = traits<XprType>;
  using StorageKind = typename XprTraits::StorageKind;
  using Index = typename XprTraits::Index;
  using Scalar = Index;
  static constexpr int NumDimensions = XprTraits::NumDimensions - array_size<Dims>::value;
  static constexpr int Layout = XprTraits::Layout;
};

template <typename ReduceOp, typename Dims, typename XprType>
struct eval<TensorPairReducerOp<ReduceOp, Dims, XprType>, Eigen::Dense> {
  using type = const TensorPairReducerOp<ReduceOp, Dims, XprType> EIGEN_DEVICE_REF;
};

}  // end namespace internal

template <typename ReduceOp, typename Dims, typename XprType>
class TensorPairReducerOp : public TensorBase<TensorPairReducerOp<ReduceOp, Dims, XprType>, ReadOnlyAccessors> {
 public:
  using Scalar = typename Eigen::internal::traits<TensorPairReducerOp>::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using Nested = typename Eigen::internal::ref_selector<TensorPairReducerOp>::type;
  using StorageKind = typename Eigen::internal::traits<TensorPairReducerOp>::StorageKind;
  using Index = typename Eigen::internal::traits<TensorPairReducerOp>::Index;
  using CoeffReturnType = Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorPairReducerOp(const XprType& expr, const ReduceOp& reduce_op,
                                                            const Index return_dim, const Dims& reduce_dims)
      : m_xpr(expr), m_reduce_op(reduce_op), m_return_dim(return_dim), m_reduce_dims(reduce_dims) {}

  EIGEN_DEVICE_FUNC const internal::remove_all_t<typename XprType::Nested>& expression() const { return m_xpr; }

  EIGEN_DEVICE_FUNC const ReduceOp& reduce_op() const { return m_reduce_op; }

  EIGEN_DEVICE_FUNC const Dims& reduce_dims() const { return m_reduce_dims; }

  EIGEN_DEVICE_FUNC Index return_dim() const { return m_return_dim; }

 protected:
  typename XprType::Nested m_xpr;
  const ReduceOp m_reduce_op;
  const Index m_return_dim;
  const Dims m_reduce_dims;
};

// Eval as rvalue
template <typename ReduceOp, typename Dims, typename ArgType, typename Device>
struct TensorEvaluator<const TensorPairReducerOp<ReduceOp, Dims, ArgType>, Device> {
  using XprType = TensorPairReducerOp<ReduceOp, Dims, ArgType>;
  using Index = typename XprType::Index;
  using Scalar = typename XprType::Scalar;
  using CoeffReturnType = typename XprType::CoeffReturnType;
  using PairType = typename TensorIndexPairOp<ArgType>::CoeffReturnType;
  using Dimensions = typename TensorEvaluator<const TensorReductionOp<ReduceOp, Dims, const TensorIndexPairOp<ArgType>>,
                                              Device>::Dimensions;
  using InputDimensions = typename TensorEvaluator<const TensorIndexPairOp<ArgType>, Device>::Dimensions;
  static constexpr int NumDims = internal::array_size<InputDimensions>::value;
  using StrideDims = array<Index, NumDims>;
  using Storage = StorageMemory<CoeffReturnType, Device>;
  using EvaluatorPointerType = typename Storage::Type;

  enum {
    IsAligned = false,
    PacketAccess = false,
    BlockAccess = false,
    PreferBlockAccess = TensorEvaluator<ArgType, Device>::PreferBlockAccess,
    CoordAccess = false,  // to be implemented
    RawAccess = false
  };
  static constexpr int Layout =
      TensorEvaluator<const TensorReductionOp<ReduceOp, Dims, const TensorIndexPairOp<ArgType>>, Device>::Layout;
  //===- Tensor block evaluation strategy (see TensorBlock.h) -------------===//
  using TensorBlock = internal::TensorBlockNotImplemented;
  //===--------------------------------------------------------------------===//

  EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_orig_impl(op.expression(), device),
        m_impl(op.expression().index_pairs().reduce(op.reduce_dims(), op.reduce_op()), device),
        m_return_dim(op.return_dim()) {
    gen_strides(m_orig_impl.dimensions(), m_strides);
    EIGEN_IF_CONSTEXPR (Layout == static_cast<int>(ColMajor)) {
      const Index total_size = internal::array_prod(m_orig_impl.dimensions());
      m_stride_mod = (m_return_dim < NumDims - 1) ? m_strides[m_return_dim + 1] : total_size;
    } else {
      const Index total_size = internal::array_prod(m_orig_impl.dimensions());
      m_stride_mod = (m_return_dim > 0) ? m_strides[m_return_dim - 1] : total_size;
    }
    // If m_return_dim is not a valid index, returns 1 or this can crash on Windows.
    m_stride_div =
        ((m_return_dim >= 0) && (m_return_dim < static_cast<Index>(m_strides.size()))) ? m_strides[m_return_dim] : 1;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_impl.dimensions(); }

  EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(EvaluatorPointerType /*data*/) {
    m_impl.evalSubExprsIfNeeded(nullptr);
    return true;
  }
  EIGEN_STRONG_INLINE void cleanup() { m_impl.cleanup(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const {
    const PairType v = m_impl.coeff(index);
    return (m_return_dim < 0) ? v.first : (v.first % m_stride_mod) / m_stride_div;
  }

  EIGEN_DEVICE_FUNC EvaluatorPointerType data() const { return nullptr; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost costPerCoeff(bool vectorized) const {
    const double compute_cost =
        1.0 + (m_return_dim < 0 ? 0.0 : (TensorOpCost::ModCost<Index>() + TensorOpCost::DivCost<Index>()));
    return m_orig_impl.costPerCoeff(vectorized) + m_impl.costPerCoeff(vectorized) + TensorOpCost(0, 0, compute_cost);
  }

 private:
  EIGEN_DEVICE_FUNC void gen_strides(const InputDimensions& dims, StrideDims& strides) {
    if (m_return_dim < 0) {
      return;  // Won't be using the strides.
    }
    eigen_assert(m_return_dim < NumDims && "Asking to convert index to a dimension outside of the rank");

    // Calculate m_stride_div and m_stride_mod, which are used to
    // calculate the value of an index w.r.t. the m_return_dim.
    EIGEN_IF_CONSTEXPR (Layout == static_cast<int>(ColMajor)) {
      strides[0] = 1;
      for (int i = 1; i < NumDims; ++i) {
        strides[i] = strides[i - 1] * dims[i - 1];
      }
    } else {
      strides[NumDims - 1] = 1;
      for (int i = NumDims - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * dims[i + 1];
      }
    }
  }

 protected:
  TensorEvaluator<const TensorIndexPairOp<ArgType>, Device> m_orig_impl;
  TensorEvaluator<const TensorReductionOp<ReduceOp, Dims, const TensorIndexPairOp<ArgType>>, Device> m_impl;
  const Index m_return_dim;
  StrideDims m_strides;
  Index m_stride_mod;
  Index m_stride_div;
};

}  // end namespace Eigen

#endif  // EIGEN_TENSOR_TENSOR_ARG_MAX_H
