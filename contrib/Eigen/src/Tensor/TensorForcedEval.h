// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TENSOR_TENSOR_FORCED_EVAL_H
#define EIGEN_TENSOR_TENSOR_FORCED_EVAL_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include <memory>

namespace Eigen {

namespace internal {
template <typename XprType>
struct traits<TensorForcedEvalOp<XprType>> {
  using Scalar = typename XprType::Scalar;
  using XprTraits = traits<XprType>;
  using StorageKind = typename traits<XprType>::StorageKind;
  using Index = typename traits<XprType>::Index;
  static constexpr int NumDimensions = XprTraits::NumDimensions;
  static constexpr int Layout = XprTraits::Layout;
  using PointerType = typename XprTraits::PointerType;

  enum { Flags = 0 };
};

template <typename XprType>
struct eval<TensorForcedEvalOp<XprType>, Eigen::Dense> {
  using type = const TensorForcedEvalOp<XprType>&;
};

}  // end namespace internal

/**
 * \ingroup Tensor_Module
 *
 * \brief Tensor forced evaluation class.
 */
template <typename XprType>
class TensorForcedEvalOp : public TensorBase<TensorForcedEvalOp<XprType>, ReadOnlyAccessors> {
 public:
  using Scalar = typename Eigen::internal::traits<TensorForcedEvalOp>::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using CoeffReturnType = std::remove_const_t<typename XprType::CoeffReturnType>;
  using Nested = typename Eigen::internal::ref_selector<TensorForcedEvalOp>::type;
  using StorageKind = typename Eigen::internal::traits<TensorForcedEvalOp>::StorageKind;
  using Index = typename Eigen::internal::traits<TensorForcedEvalOp>::Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorForcedEvalOp(const XprType& expr) : m_xpr(expr) {}

  EIGEN_DEVICE_FUNC const internal::remove_all_t<typename XprType::Nested>& expression() const { return m_xpr; }

 protected:
  typename XprType::Nested m_xpr;
};

namespace internal {
// Returns true iff it actually placement-new'd any elements (so cleanup
// knows whether destructors must be run — see bug #1530).
template <typename Device, typename CoeffReturnType>
struct non_integral_type_placement_new {
  template <typename StorageType>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(Index numValues, StorageType m_buffer) const {
    EIGEN_IF_CONSTEXPR (NumTraits<CoeffReturnType>::RequireInitialization) {
      default_construct_elements_of_array(m_buffer, numValues);
      return true;
    }
    return false;
  }
};

// SYCL does not support non-integral types
// having new (m_buffer + i) CoeffReturnType() causes the following compiler error for SYCL Devices
// no matching function for call to 'operator new'
template <typename CoeffReturnType>
struct non_integral_type_placement_new<Eigen::SyclDevice, CoeffReturnType> {
  template <typename StorageType>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(Index, StorageType) const {
    return false;
  }
};
}  // end namespace internal

template <typename Device>
class DeviceTempPointerHolder {
 public:
  DeviceTempPointerHolder(const Device& device, size_t size)
      : device_(device), size_(size), ptr_(device.allocate_temp(size)) {}

  ~DeviceTempPointerHolder() {
    device_.deallocate_temp(ptr_);
    size_ = 0;
    ptr_ = nullptr;
  }

  void* ptr() { return ptr_; }

 private:
  Device device_;
  size_t size_;
  void* ptr_;
};

template <typename ArgType_, typename Device>
struct TensorEvaluator<const TensorForcedEvalOp<ArgType_>, Device> {
  using ArgType = const internal::remove_all_t<ArgType_>;
  using XprType = TensorForcedEvalOp<ArgType>;
  using Scalar = typename ArgType::Scalar;
  using Dimensions = typename TensorEvaluator<ArgType, Device>::Dimensions;
  using Index = typename XprType::Index;
  using CoeffReturnType = typename XprType::CoeffReturnType;
  using PacketReturnType = typename PacketType<CoeffReturnType, Device>::type;
  static constexpr int PacketSize = PacketType<CoeffReturnType, Device>::size;
  using TensorPointerType = typename Eigen::internal::traits<XprType>::PointerType;
  using Storage = StorageMemory<CoeffReturnType, Device>;
  using EvaluatorPointerType = typename Storage::Type;

  enum {
    IsAligned = true,
    PacketAccess = (PacketType<CoeffReturnType, Device>::size > 1),
    BlockAccess = internal::is_arithmetic<CoeffReturnType>::value,
    PreferBlockAccess = false,
    RawAccess = true
  };

  static constexpr int Layout = TensorEvaluator<ArgType, Device>::Layout;
  static constexpr int NumDims = internal::traits<ArgType>::NumDimensions;

  //===- Tensor block evaluation strategy (see TensorBlock.h) -------------===//
  using TensorBlockDesc = internal::TensorBlockDescriptor<NumDims, Index>;
  using TensorBlockScratch = internal::TensorBlockScratchAllocator<Device>;

  using TensorBlock = typename internal::TensorMaterializedBlock<CoeffReturnType, NumDims, Layout, Index>;
  //===--------------------------------------------------------------------===//

  TensorEvaluator(const XprType& op, const Device& device)
      : m_impl(op.expression(), device),
        m_op(op.expression()),
        m_device(device),
        m_buffer_holder(nullptr),
        m_buffer(nullptr),
        m_placement_constructed(false) {}

  ~TensorEvaluator() { cleanup(); }

  EIGEN_DEVICE_FUNC const Dimensions& dimensions() const { return m_impl.dimensions(); }

  EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(EvaluatorPointerType) {
    const Index numValues = internal::array_prod(m_impl.dimensions());
    m_buffer_holder = std::make_shared<DeviceTempPointerHolder<Device>>(m_device, numValues * sizeof(CoeffReturnType));
    m_buffer = static_cast<EvaluatorPointerType>(m_buffer_holder->ptr());

    m_placement_constructed = internal::non_integral_type_placement_new<Device, CoeffReturnType>()(numValues, m_buffer);

    using EvalTo = TensorEvalToOp<const std::remove_const_t<ArgType>>;
    EvalTo evalToTmp(m_device.get(m_buffer), m_op);

    internal::TensorExecutor<const EvalTo, std::remove_const_t<Device>,
                             /*Vectorizable=*/internal::IsVectorizable<Device, const ArgType>::value,
                             /*Tiling=*/internal::IsTileable<Device, const ArgType>::value>::run(evalToTmp, m_device);

    return true;
  }

#ifdef EIGEN_USE_THREADS
  template <typename EvalSubExprsCallback>
  EIGEN_STRONG_INLINE void evalSubExprsIfNeededAsync(EvaluatorPointerType, EvalSubExprsCallback done) {
    const Index numValues = internal::array_prod(m_impl.dimensions());
    m_buffer_holder = std::make_shared<DeviceTempPointerHolder<Device>>(m_device, numValues * sizeof(CoeffReturnType));
    m_buffer = static_cast<EvaluatorPointerType>(m_buffer_holder->ptr());

    m_placement_constructed = internal::non_integral_type_placement_new<Device, CoeffReturnType>()(numValues, m_buffer);

    using EvalTo = TensorEvalToOp<const std::remove_const_t<ArgType>>;
    EvalTo evalToTmp(m_device.get(m_buffer), m_op);

    auto on_done = std::bind([](EvalSubExprsCallback done_) { done_(true); }, std::move(done));
    internal::TensorAsyncExecutor<
        const EvalTo, std::remove_const_t<Device>, decltype(on_done),
        /*Vectorizable=*/internal::IsVectorizable<Device, const ArgType>::value,
        /*Tiling=*/internal::IsTileable<Device, const ArgType>::value>::runAsync(evalToTmp, m_device,
                                                                                 std::move(on_done));
  }
#endif

  EIGEN_STRONG_INLINE void cleanup() {
    // Executors copy evaluators per task; only the last owner of the shared buffer destroys its elements.
    if (m_placement_constructed && m_buffer_holder.use_count() == 1) {
      internal::destruct_elements_of_array(m_buffer, internal::array_prod(m_impl.dimensions()));
      m_placement_constructed = false;
    }
    m_buffer_holder = nullptr;
    m_buffer = nullptr;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const { return m_buffer[index]; }

  template <int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const {
    return internal::ploadt<PacketReturnType, LoadMode>(m_buffer + index);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE internal::TensorBlockResourceRequirements getResourceRequirements() const {
    return internal::TensorBlockResourceRequirements::any();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorBlock block(TensorBlockDesc& desc, TensorBlockScratch& scratch,
                                                          bool /*root_of_expr_ast*/ = false) const {
    eigen_assert(m_buffer != nullptr);
    return TensorBlock::materialize(m_buffer, m_impl.dimensions(), desc, scratch);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost costPerCoeff(bool vectorized) const {
    return TensorOpCost(sizeof(CoeffReturnType), 0, 0, vectorized, PacketSize);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE EvaluatorPointerType data() const { return m_buffer; }

 private:
  TensorEvaluator<ArgType, Device> m_impl;
  const ArgType m_op;
  const Device EIGEN_DEVICE_REF m_device;
  std::shared_ptr<DeviceTempPointerHolder<Device>> m_buffer_holder;
  EvaluatorPointerType m_buffer;  // Cached copy of the value stored in m_buffer_holder.
  bool m_placement_constructed;   // See bug #1530.
};

}  // end namespace Eigen

#endif  // EIGEN_TENSOR_TENSOR_FORCED_EVAL_H
