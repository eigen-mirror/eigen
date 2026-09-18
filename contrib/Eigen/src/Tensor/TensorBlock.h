// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TENSOR_TENSOR_BLOCK_H
#define EIGEN_TENSOR_TENSOR_BLOCK_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

// -------------------------------------------------------------------------- //
// Forward declarations for templates defined below.
template <typename Scalar, typename IndexType, int NumDims, int Layout>
class TensorBlockIO;

// -------------------------------------------------------------------------- //
// Helper function to compute strides for densely stored buffer of given
// dimensions.

// TODO(ezhulenev): We compute strides many times in different evaluators, use
// this function instead everywhere.
template <int Layout, typename IndexType, int NumDims>
EIGEN_ALWAYS_INLINE std::enable_if_t<NumDims == 0, DSizes<IndexType, NumDims> > strides_impl(
    const DSizes<IndexType, NumDims>& /*dimensions*/) {
  DSizes<IndexType, NumDims> strides;
  return strides;
}

template <int Layout, typename IndexType, int NumDims>
EIGEN_ALWAYS_INLINE std::enable_if_t<(NumDims > 0), DSizes<IndexType, NumDims> > strides_impl(
    const DSizes<IndexType, NumDims>& dimensions) {
  DSizes<IndexType, NumDims> strides;
  // TODO(ezhulenev): Benchmark whether template-unrolling this loop is beneficial.
  EIGEN_IF_CONSTEXPR (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
    strides[0] = 1;
    for (int i = 1; i < NumDims; ++i) {
      strides[i] = strides[i - 1] * dimensions[i - 1];
    }
  } else {
    strides[NumDims - 1] = 1;
    for (int i = NumDims - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * dimensions[i + 1];
    }
  }

  return strides;
}

template <int Layout, typename IndexType, int NumDims>
EIGEN_ALWAYS_INLINE DSizes<IndexType, NumDims> strides(const DSizes<IndexType, NumDims>& dimensions) {
  return strides_impl<Layout>(dimensions);
}

template <int Layout, typename IndexType, size_t NumDims>
EIGEN_ALWAYS_INLINE DSizes<IndexType, NumDims> strides(const Eigen::array<IndexType, NumDims>& dimensions) {
  return strides<Layout>(DSizes<IndexType, NumDims>(dimensions));
}

template <int Layout, std::ptrdiff_t... Indices>
EIGEN_STRONG_INLINE DSizes<std::ptrdiff_t, sizeof...(Indices)> strides(const Sizes<Indices...>& sizes) {
  return strides<Layout>(DSizes<std::ptrdiff_t, sizeof...(Indices)>(sizes));
}

// -------------------------------------------------------------------------- //

// Tensor block shape type defines what are the shape preference for the blocks
// extracted from the larger tensor.
//
// Example: blocks of 100 elements from the large 100x100 tensor:
// - tensor: 100x100
// - target_block_size: 100
//
// TensorBlockShapeType:
//  - kUniformAllDims: 100 blocks of size 10x10
//  - kSkewedInnerDims: 100 blocks of size 100x1 (or 1x100 depending on a column
//                      or row major layout)
enum class TensorBlockShapeType { kUniformAllDims, kSkewedInnerDims };

struct TensorBlockResourceRequirements {
  TensorBlockShapeType shape_type;  // target block shape
  size_t size;                      // target block size
  TensorOpCost cost_per_coeff;      // cost of computing a single block element

#ifdef EIGEN_HIPCC
  // For HIPCC, we need to explicitly declare as a "device fun", the constructor
  // which is implicitly invoked in the "merge" / "any" routines. else HIPCC
  // errors out complaining about the lack of a matching constructor
  EIGEN_DEVICE_FUNC TensorBlockResourceRequirements(TensorBlockShapeType shape_type_, size_t size_, TensorOpCost cost_)
      : shape_type(shape_type_), size(size_), cost_per_coeff(cost_) {}
#endif

  template <typename Scalar>
  EIGEN_DEVICE_FUNC static TensorBlockResourceRequirements withShapeAndSize(TensorBlockShapeType shape_type,
                                                                            size_t size_in_bytes, TensorOpCost cost) {
    const size_t size = numext::maxi(size_t(1), size_in_bytes / sizeof(Scalar));
    return {shape_type, size, cost};
  }

  template <typename Scalar>
  EIGEN_DEVICE_FUNC static TensorBlockResourceRequirements withShapeAndSize(TensorBlockShapeType shape_type,
                                                                            size_t size_in_bytes) {
    // This default cost per coefficient is valid for most materialized tensor
    // block evaluation implementations, because they typically just read
    // coefficients from the underlying tensor storage, and write to the tensor
    // block buffer (scratch or destination memory, reads and writes have linear
    // access pattern). We ignore the fixed cost of block evaluation, because in
    // practice it should be negligible.
    //
    // Lazy block evaluation adds the cost of calling a functor for each
    // coefficient.
    //
    // All non-trivial block evaluation implementations must provide their own
    // cost approximation (e.g. shuffling inner dimension has a much higher cost
    // because it reads memory randomly, although the total number of moved
    // bytes is the same).
    return withShapeAndSize<Scalar>(shape_type, size_in_bytes,
                                    {/*bytes_loaded=*/sizeof(Scalar),
                                     /*bytes_stored=*/sizeof(Scalar),
                                     /*compute_cycles=*/0});
  }

  template <typename Scalar>
  EIGEN_DEVICE_FUNC static TensorBlockResourceRequirements skewed(size_t size_in_bytes) {
    return withShapeAndSize<Scalar>(TensorBlockShapeType::kSkewedInnerDims, size_in_bytes);
  }

  template <typename Scalar>
  EIGEN_DEVICE_FUNC static TensorBlockResourceRequirements uniform(size_t size_in_bytes) {
    return withShapeAndSize<Scalar>(TensorBlockShapeType::kUniformAllDims, size_in_bytes);
  }

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE TensorBlockResourceRequirements
  merge(const TensorBlockResourceRequirements& lhs, const TensorBlockResourceRequirements& rhs) {
    return {merge(lhs.shape_type, rhs.shape_type),           // shape_type
            merge(lhs.size, rhs.size),                       // size
            merge(lhs.cost_per_coeff, rhs.cost_per_coeff)};  // cost_per_coeff
  }

  EIGEN_DEVICE_FUNC TensorBlockResourceRequirements& addCostPerCoeff(TensorOpCost cost) {
    cost_per_coeff += cost;
    return *this;
  }

  // This is a resource requirement that should be returned from expressions
  // that do not have any block evaluation preference (e.g. default tensor
  // expression with raw buffer access).
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE TensorBlockResourceRequirements any() {
    return {TensorBlockShapeType::kUniformAllDims, 1, {0, 0, 0}};
  }

 private:
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE size_t merge(size_t lhs_size, size_t rhs_size) {
    return numext::maxi(lhs_size, rhs_size);
  }

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE TensorBlockShapeType merge(TensorBlockShapeType lhs,
                                                                          TensorBlockShapeType rhs) {
    return (lhs == TensorBlockShapeType::kSkewedInnerDims || rhs == TensorBlockShapeType::kSkewedInnerDims)
               ? TensorBlockShapeType::kSkewedInnerDims
               : TensorBlockShapeType::kUniformAllDims;
  }

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE TensorOpCost merge(TensorOpCost lhs_cost, TensorOpCost rhs_cost) {
    return lhs_cost + rhs_cost;
  }
};

// -------------------------------------------------------------------------- //
// TensorBlockDescriptor specifies a block offset within a tensor and the block
// sizes along each of the tensor dimensions.

template <int NumDims, typename IndexType = Eigen::Index>
class TensorBlockDescriptor {
 public:
  typedef DSizes<IndexType, NumDims> Dimensions;

  // If we evaluate a Tensor assignment, and expression on the left, already has
  // a memory buffer, then we might do performance optimization, and evaluate
  // the root expression directly into the final output memory. Some time it's
  // possible to reuse it for materializing subexpressions inside an expression
  // tree, to avoid dynamic memory allocation.
  //
  // The pointer type of the underlying storage is erased, because passing
  // Scalar type through all the expression evaluation layers is way too many
  // templates. In practice destination buffer type should always match the
  // evaluated expression scalar type.
  class DestinationBuffer {
   public:
    enum DestinationBufferKind : int {
      // The above explicit specification of "int" as the enum basetype is
      // needed to get around a HIPCC link error ("the field type is not
      // amp-compatible")
      // which is issued for class members with the enum type.
      // TODO(rocm):
      // remove the "int" basetype once HIPCC has been fixed to not error out
      // in the above scenario.

      // Destination buffer is not defined (`m_data` == nullptr).
      kEmpty,

      // Tensor block defined by an owning tensor block descriptor can fit
      // contiguously into the destination buffer. In this case it's safe to
      // materialize a tensor block in the destination buffer and build an
      // expression over a dense view of it.
      kContiguous,

      // Destination buffer strides do not match strides of the contiguously
      // stored block, and it's impossible to define a TensorMap over this
      // buffer. However if we are evaluating a root of an expression tree, we
      // still can materialize an output into this destination, because we can
      // guarantee that no one will ever access it through block API.
      //
      // Strided input views are represented by TensorBlockView. Output
      // materialization still reserves this destination for the root so that
      // a child cannot overwrite another operand before it has been consumed.
      kStrided
    };

    template <typename Scalar>
    Scalar* data() const {
      eigen_assert(m_data_type_size == sizeof(Scalar));
      return static_cast<Scalar*>(m_data);
    }

    const Dimensions& strides() const { return m_strides; }
    const DestinationBufferKind& kind() const { return m_kind; }

   private:
    friend class TensorBlockDescriptor<NumDims, IndexType>;

    DestinationBuffer() = default;

    template <typename Scalar>
    DestinationBuffer(Scalar* data, const Dimensions& strides, DestinationBufferKind kind)
        : m_data(static_cast<void*>(data)), m_data_type_size(sizeof(Scalar)), m_strides(strides), m_kind(kind) {}

    template <int Layout, typename Scalar>
    static DestinationBuffer make(const TensorBlockDescriptor& desc, Scalar* data, const Dimensions& strides) {
      return DestinationBuffer(data, strides, kind<Layout>(desc, strides));
    }

    template <int Layout>
    static DestinationBufferKind kind(const TensorBlockDescriptor& desc, const Dimensions& strides) {
      const Dimensions& desc_dims = desc.dimensions();
      const Dimensions& desc_strides = internal::strides<Layout>(desc_dims);
      for (int i = 0; i < NumDims; ++i) {
        if (desc_dims[i] == 1) continue;
        if (desc_strides[i] != strides[i]) return kStrided;
      }
      return kContiguous;
    }

    // Storage pointer is type erased, to reduce template bloat, but we still
    // keep the size of the underlying element type for error checking.
    void* m_data = nullptr;
    size_t m_data_type_size = 0;

    // Destination buffer dimensions always match the dimensions of a tensor
    // block descriptor it belongs to, however strides might be different.
    Dimensions m_strides;

    DestinationBufferKind m_kind = kEmpty;
  };

  TensorBlockDescriptor(const IndexType offset, const Dimensions& dimensions, const DestinationBuffer& destination)
      : m_offset(offset), m_dimensions(dimensions), m_destination(destination) {}

  TensorBlockDescriptor(const IndexType offset, const Dimensions& dimensions)
      : m_offset(offset), m_dimensions(dimensions), m_destination(DestinationBuffer()) {}

  IndexType offset() const { return m_offset; }
  const Dimensions& dimensions() const { return m_dimensions; }
  IndexType dimension(int index) const { return m_dimensions[index]; }
  IndexType size() const { return array_prod<IndexType>(m_dimensions); }

  const DestinationBuffer& destination() const { return m_destination; }

  template <int Layout, typename Scalar>
  void AddDestinationBuffer(Scalar* dst_base, const Dimensions& dst_strides) {
    eigen_assert(dst_base != nullptr);
    m_destination = DestinationBuffer::template make<Layout>(*this, dst_base, dst_strides);
  }

  template <int Layout, typename Scalar, typename DstStridesIndexType>
  void AddDestinationBuffer(Scalar* dst_base, const DSizes<DstStridesIndexType, NumDims>& dst_strides) {
    // DSizes constructor will do index type promotion if it's safe.
    AddDestinationBuffer<Layout>(dst_base, Dimensions(dst_strides));
  }

  TensorBlockDescriptor& DropDestinationBuffer() {
    m_destination.m_data = nullptr;
    m_destination.m_kind = DestinationBuffer::kEmpty;
    return *this;
  }

  bool HasDestinationBuffer() const { return m_destination.kind() != DestinationBuffer::kEmpty; }

  // Returns a copy of `*this` with updated offset.
  TensorBlockDescriptor WithOffset(IndexType offset) const {
    return TensorBlockDescriptor(offset, m_dimensions, m_destination);
  }

 private:
  // Offset and dimensions are immutable after construction. Block descriptor
  // can only be mutated by adding or dropping destination.
  const IndexType m_offset;
  const Dimensions m_dimensions;
  DestinationBuffer m_destination;
};

// -------------------------------------------------------------------------- //
// TensorBlockMapper is responsible for iterating over the blocks of a tensor.

template <int NumDims, int Layout, typename IndexType = Eigen::Index>
class TensorBlockMapper {
  typedef TensorBlockDescriptor<NumDims, IndexType> BlockDescriptor;

 public:
  typedef DSizes<IndexType, NumDims> Dimensions;

  TensorBlockMapper() = default;
  TensorBlockMapper(const DSizes<IndexType, NumDims>& dimensions, const TensorBlockResourceRequirements& requirements)
      : m_tensor_dimensions(dimensions), m_requirements(requirements) {
    // Compute block dimensions and the total number of blocks.
    InitializeBlockDimensions();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE IndexType blockCount() const { return m_total_block_count; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE IndexType blockTotalSize() const { return m_block_dimensions.TotalSize(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const DSizes<IndexType, NumDims>& blockDimensions() const {
    return m_block_dimensions;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE BlockDescriptor blockDescriptor(IndexType block_index) const {
    static constexpr bool isColMajor = Layout == static_cast<int>(ColMajor);

    IndexType offset = 0;
    DSizes<IndexType, NumDims> dimensions;

    EIGEN_IF_CONSTEXPR (NumDims == 0) return BlockDescriptor(offset, dimensions);

    // Iterate outer -> inner dimensions.
    for (int i = NumDims - 1; i >= 0; --i) {
      const int dim = isColMajor ? i : NumDims - i - 1;

      const IndexType idx = block_index / m_block_strides[dim];
      block_index -= idx * m_block_strides[dim];

      const IndexType coord = idx * m_block_dimensions[dim];
      dimensions[dim] = numext::mini(m_tensor_dimensions[dim] - coord, m_block_dimensions[dim]);
      offset += coord * m_tensor_strides[dim];
    }

    return {offset, dimensions};
  }

 private:
  void InitializeBlockDimensions() {
    // Requested block shape and size.
    const TensorBlockShapeType shape_type = m_requirements.shape_type;
    IndexType target_block_size = numext::maxi<IndexType>(1, static_cast<IndexType>(m_requirements.size));

    IndexType tensor_size = m_tensor_dimensions.TotalSize();

    // Corner case: one of the dimensions is zero. Logic below is too complex
    // to handle this case on a general basis, just use unit block size.
    // Note: we must not yield blocks with zero dimensions (recipe for
    // overflows/underflows, divisions by zero and NaNs later).
    if (tensor_size == 0) {
      for (int i = 0; i < NumDims; ++i) {
        m_block_dimensions[i] = 1;
      }
      m_total_block_count = 0;
      return;
    }

    // If tensor fits into a target block size, evaluate it as a single block.
    if (tensor_size <= target_block_size) {
      m_block_dimensions = m_tensor_dimensions;
      m_total_block_count = 1;
      // The only valid block index is `0`, and in this case we do not need
      // to compute real strides for tensor or blocks (see blockDescriptor).
      for (int i = 0; i < NumDims; ++i) {
        m_tensor_strides[i] = 0;
        m_block_strides[i] = 1;
      }
      return;
    }

    static constexpr bool isColMajor = Layout == static_cast<int>(ColMajor);

    // Block shape skewed towards inner dimension.
    if (shape_type == TensorBlockShapeType::kSkewedInnerDims) {
      IndexType coeff_to_allocate = target_block_size;

      for (int i = 0; i < NumDims; ++i) {
        const int dim = isColMajor ? i : NumDims - i - 1;
        m_block_dimensions[dim] = numext::mini(coeff_to_allocate, m_tensor_dimensions[dim]);
        coeff_to_allocate =
            numext::div_ceil(coeff_to_allocate, numext::maxi(static_cast<IndexType>(1), m_block_dimensions[dim]));
      }
      eigen_assert(coeff_to_allocate == 1);

    } else if (shape_type == TensorBlockShapeType::kUniformAllDims) {
      // Tensor will not fit within 'target_block_size' budget: calculate tensor
      // block dimension sizes based on "square" dimension size target.
      const IndexType dim_size_target = convert_index<IndexType>(
          numext::pow(static_cast<float>(target_block_size), 1.0f / static_cast<float>(m_block_dimensions.rank())));

      for (int i = 0; i < NumDims; ++i) {
        // TODO(andydavis): Adjust the inner most 'block_dim_size' to make it
        // a multiple of the packet size. Note that reducing
        // 'block_dim_size' in this manner can increase the number of
        // blocks, and so will amplify any per-block overhead.
        m_block_dimensions[i] = numext::mini(dim_size_target, m_tensor_dimensions[i]);
      }

      // Add any un-allocated coefficients to inner dimension(s).
      IndexType total_size = m_block_dimensions.TotalSize();
      for (int i = 0; i < NumDims; ++i) {
        const int dim = isColMajor ? i : NumDims - i - 1;

        if (m_block_dimensions[dim] < m_tensor_dimensions[dim]) {
          const IndexType total_size_other_dims = total_size / m_block_dimensions[dim];
          const IndexType alloc_avail = numext::div_ceil<IndexType>(target_block_size, total_size_other_dims);
          if (alloc_avail == m_block_dimensions[dim]) {
            // Insufficient excess coefficients to allocate.
            break;
          }
          m_block_dimensions[dim] = numext::mini(m_tensor_dimensions[dim], alloc_avail);
          total_size = total_size_other_dims * m_block_dimensions[dim];
        }
      }

    } else {
      eigen_assert(false);  // unknown block shape
    }

    eigen_assert(m_block_dimensions.TotalSize() >=
                 numext::mini<IndexType>(target_block_size, m_tensor_dimensions.TotalSize()));

    // Calculate block counts by dimension and total block count.
    DSizes<IndexType, NumDims> block_count;
    for (int i = 0; i < NumDims; ++i) {
      block_count[i] = numext::div_ceil(m_tensor_dimensions[i], m_block_dimensions[i]);
    }
    m_total_block_count = array_prod(block_count);

    // Calculate block strides (used for enumerating blocks).
    m_tensor_strides = strides<Layout>(m_tensor_dimensions);
    m_block_strides = strides<Layout>(block_count);
  }

  DSizes<IndexType, NumDims> m_tensor_dimensions;
  TensorBlockResourceRequirements m_requirements;

  DSizes<IndexType, NumDims> m_block_dimensions;
  IndexType m_total_block_count;

  DSizes<IndexType, NumDims> m_tensor_strides;
  DSizes<IndexType, NumDims> m_block_strides;
};

// -------------------------------------------------------------------------- //
// TensorBlockScratchAllocator is responsible for allocating temporary buffers
// for block evaluation (output or input block materialization). Given that
// Eigen expression traversal order is deterministic, all temporary allocations
// are happening in the same order, and usually have exactly the same size.
// Scratch allocator keeps a trace of all dynamic allocations, and after the
// first block evaluation is completed, we should be able to reuse all the
// temporary buffers for the next block evaluation.

template <typename Device>
class TensorBlockScratchAllocator {
 public:
  explicit TensorBlockScratchAllocator(const Device& device) : m_device(device), m_allocation_index(0) {}

  ~TensorBlockScratchAllocator() {
    for (size_t i = 0; i < m_allocations.size(); ++i) {
      m_device.deallocate(m_allocations[i].ptr);
    }
  }

  void* allocate(size_t size) {
    // TODO(ezhulenev): Remove when replaced with inlined vector.
    if (m_allocations.capacity() == 0) m_allocations.reserve(8);

    // Check if we already have an existing allocation at current index.
    const int num_allocations = static_cast<int>(m_allocations.size());
    const bool has_allocation = m_allocation_index < num_allocations;

    // Allocation index can't be larger than the number of allocations.
    eigen_assert(m_allocation_index <= num_allocations);

    // If we have existing allocation, and its size is larger or equal to
    // requested size, we do nothing.

    // If current allocation can't fit requested size, we deallocate it, and
    // replace with a larger allocation.
    if (has_allocation && m_allocations[m_allocation_index].size < size) {
      m_device.deallocate(m_allocations[m_allocation_index].ptr);
      m_allocations[m_allocation_index].ptr = m_device.allocate(size);
      m_allocations[m_allocation_index].size = size;
    }

    // Make a new allocation if we don't have an existing one.
    if (!has_allocation) {
      Allocation allocation;
      allocation.ptr = m_device.allocate(size);
      allocation.size = size;
      m_allocations.push_back(allocation);
    }

    eigen_assert(m_allocations[m_allocation_index].ptr != nullptr);
    eigen_assert(m_allocations[m_allocation_index].size >= size);

    return m_allocations[m_allocation_index++].ptr;
  }

  void reset() { m_allocation_index = 0; }

 private:
  struct Allocation {
    void* ptr;
    size_t size;
  };

  const Device& m_device;
  int m_allocation_index;
  // TODO(ezhulenev): This should be an inlined vector.
  std::vector<Allocation> m_allocations;
};

// -------------------------------------------------------------------------- //
// TensorBlockKind represents all possible block kinds, that can be produced by
// TensorEvaluator::evalBlock function.
enum TensorBlockKind {
  // Tensor block that is a lazy expression that must be assigned to a
  // destination using TensorBlockAssign.
  kExpr,

  // Tensor block that is a view into a memory buffer owned by an underlying
  // Tensor expression (e.g. it can be a view into a Tensor buffer).
  kView,

  // Tensor block that was materialized in a scratch memory buffer, allocated
  // with TensorBlockScratchAllocator. This block must be copied to a
  // destination, similar to a block of `kExpr` type.
  kMaterializedInScratch,

  // Tensor block that was materialized directly into the final output memory
  // buffer. For example if the left side of an assignment is a Tensor, we can
  // directly materialize the block in the destination memory.
  //
  // If strides in the output buffer do not match tensor block strides, the
  // Tensor expression will be invalid, and should not be used by
  // TensorBlockAssign or for constructing another block expression.
  kMaterializedInOutput
};

// -------------------------------------------------------------------------- //
// TensorBlockNotImplemented should be used to define TensorBlock typedef in
// TensorEvaluators that do not support block evaluation.

class TensorBlockNotImplemented {
 public:
  typedef void XprType;
};

template <typename Scalar, int NumDims, int Layout, typename IndexType>
class TensorBlockView;

template <typename Scalar, int NumDims, int Layout, typename IndexType>
struct traits<TensorBlockView<Scalar, NumDims, Layout, IndexType>>
    : traits<Tensor<Scalar, NumDims, Layout, IndexType>> {
  static constexpr unsigned int Flags = 0;
};

// A block's inner runs are contiguous, but successive runs can belong to different
// rows, columns, or planes of the underlying tensor.
template <typename Scalar_, int NumDims, int Layout, typename IndexType>
class TensorBlockView : public TensorBase<TensorBlockView<Scalar_, NumDims, Layout, IndexType>> {
 public:
  using Scalar = Scalar_;
  using Index = IndexType;
  using Dimensions = DSizes<Index, NumDims>;
  using Nested = TensorBlockView;
  using StorageKind = Dense;
  using CoeffReturnType = Scalar;

  TensorBlockView(const Scalar* data, const Dimensions& dimensions)
      : TensorBlockView(data, dimensions, internal::strides<Layout>(dimensions)) {}

  TensorBlockView(const Scalar* data, const Dimensions& dimensions, const Dimensions& strides)
      : m_data(data), m_dimensions(dimensions), m_strides(strides), m_contiguous(true) {
    eigen_assert(NumDims == 0 || strides[Layout == ColMajor ? 0 : NumDims - 1] == 1);
    Index stride = 1;
    for (int i = 0; i < NumDims; ++i) {
      const int dim = Layout == ColMajor ? i : NumDims - 1 - i;
      if (dimensions[dim] > 1 && strides[dim] != stride) m_contiguous = false;
      stride *= dimensions[dim];
    }
  }

  EIGEN_DEVICE_FUNC const Dimensions& dimensions() const { return m_dimensions; }
  EIGEN_DEVICE_FUNC const Dimensions& strides() const { return m_strides; }
  EIGEN_DEVICE_FUNC const Scalar* data() const { return m_contiguous ? m_data : nullptr; }
  EIGEN_DEVICE_FUNC const Scalar* rawData() const { return m_data; }

 private:
  const Scalar* m_data;
  Dimensions m_dimensions;
  Dimensions m_strides;
  bool m_contiguous;
};

}  // namespace internal

template <typename Scalar_, int NumDims, int Layout_, typename IndexType, typename Device>
struct TensorEvaluator<const internal::TensorBlockView<Scalar_, NumDims, Layout_, IndexType>, Device> {
  using XprType = internal::TensorBlockView<Scalar_, NumDims, Layout_, IndexType>;
  using Scalar = Scalar_;
  using Index = IndexType;
  using Dimensions = DSizes<Index, NumDims>;
  using CoeffReturnType = Scalar;
  using PacketReturnType = typename PacketType<Scalar, Device>::type;
  using EvaluatorPointerType = const Scalar*;
  using TensorBlock = internal::TensorBlockNotImplemented;
  static constexpr int Layout = Layout_;
  static constexpr bool IsAligned = false;
  static constexpr bool PacketAccess = internal::packet_traits<Scalar>::Vectorizable;
  static constexpr bool BlockAccess = false;
  static constexpr bool PreferBlockAccess = false;
  static constexpr bool CoordAccess = false;
  static constexpr bool RawAccess = false;

  TensorEvaluator(const XprType& expression, const Device&)
      : m_expression(expression), m_output_strides(internal::strides<Layout>(expression.dimensions())) {
    if (!expression.data()) {
      for (int i = 0; i < NumDims; ++i) {
        m_divisors[i] = internal::TensorIntDivisor<Index>(numext::maxi(Index(1), m_output_strides[i]));
      }
    }
  }

  EIGEN_DEVICE_FUNC const Dimensions& dimensions() const { return m_expression.dimensions(); }
  EIGEN_DEVICE_FUNC bool evalSubExprsIfNeeded(EvaluatorPointerType) { return true; }
  EIGEN_DEVICE_FUNC void cleanup() {}
  EIGEN_DEVICE_FUNC const Scalar* data() const { return m_expression.data(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index srcCoeff(Index index) const {
    if (m_expression.data()) return index;
    Index offset = 0;
    for (int i = NumDims - 1; i > 0; --i) {
      const int dim = Layout == ColMajor ? i : NumDims - 1 - i;
      const Index coordinate = index / m_divisors[dim];
      offset += coordinate * m_expression.strides()[dim];
      index -= coordinate * m_output_strides[dim];
    }
    return offset + index;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const {
    return m_expression.rawData()[srcCoeff(index)];
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar* coeffAddress(Index index) const {
    return m_expression.rawData() + srcCoeff(index);
  }

  template <int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const {
    constexpr int PacketSize = PacketType<Scalar, Device>::size;
    const Index first = srcCoeff(index);
    if (m_expression.data() || srcCoeff(index + PacketSize - 1) == first + PacketSize - 1) {
      return internal::ploadu<PacketReturnType>(m_expression.rawData() + first);
    }
    EIGEN_ALIGN_MAX Scalar values[PacketSize];
    for (int i = 0; i < PacketSize; ++i) values[i] = coeff(index + i);
    return internal::ploadu<PacketReturnType>(values);
  }

  EIGEN_DEVICE_FUNC TensorOpCost costPerCoeff(bool vectorized) const {
    return TensorOpCost(sizeof(Scalar), 0, m_expression.data() ? 0 : NumDims, vectorized,
                        PacketType<Scalar, Device>::size);
  }

 private:
  XprType m_expression;
  Dimensions m_output_strides;
  array<internal::TensorIntDivisor<Index>, NumDims> m_divisors;
};

template <typename Scalar, int Layout, typename Index, typename Device>
struct TensorEvaluator<const internal::TensorBlockView<Scalar, 1, Layout, Index>, Device>
    : TensorEvaluator<const TensorMap<const Tensor<Scalar, 1, Layout, Index>>, Device> {
  using XprType = internal::TensorBlockView<Scalar, 1, Layout, Index>;
  using MapType = TensorMap<const Tensor<Scalar, 1, Layout, Index>>;
  using Base = TensorEvaluator<const MapType, Device>;
  TensorEvaluator(const XprType& expression, const Device& device)
      : Base(MapType(expression.rawData(), expression.dimensions()), device) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar* coeffAddress(Index index) const { return this->data() + index; }
};

namespace internal {

// -------------------------------------------------------------------------- //
// XprScalar extracts Scalar type from the Eigen expressions (if expression type
// is not void). It's required to be able to define lazy block expression for
// argument types, that do not support block evaluation.

template <typename XprType>
struct XprScalar {
  typedef typename XprType::Scalar type;
};
template <>
struct XprScalar<void> {
  typedef void type;
};

// -------------------------------------------------------------------------- //
// TensorMaterializedBlock is a fully evaluated block of the original tensor,
// and XprType is a view over its data, allowing non-unit outer strides. It is
// used to materialize blocks of tensor expressions, that can't be efficiently
// represented as lazy Tensor expressions with fast coeff/packet operations,
// e.g. we materialize all broadcasts into evaluated blocks.
//
// TensorMaterializedBlock does not own its memory buffer, it's either a memory
// buffer that backs the original expression (e.g. block is just a view into a
// Tensor), or a memory buffer allocated with scratch allocator, and in this
// case the scratch allocator will deallocate it at the end of block based
// expression execution.
//
// If the block was evaluated directly into the output buffer, and strides in
// the output buffer do not match block strides, the dense block expression will
// be invalid, and should never be used in block assignment or any other tensor
// expression.

template <typename Scalar, int NumDims, int Layout, typename IndexType = Eigen::Index>
class TensorMaterializedBlock {
 public:
  typedef DSizes<IndexType, NumDims> Dimensions;
  using XprType = TensorBlockView<Scalar, NumDims, Layout, IndexType>;

  TensorMaterializedBlock(TensorBlockKind kind, const Scalar* data, const Dimensions& dimensions,
                          bool valid_expr = true)
      : m_kind(kind), m_data(data), m_dimensions(dimensions), m_expr(m_data, m_dimensions), m_valid_expr(valid_expr) {
    eigen_assert(m_kind == internal::TensorBlockKind::kView ||
                 m_kind == internal::TensorBlockKind::kMaterializedInScratch ||
                 m_kind == internal::TensorBlockKind::kMaterializedInOutput);
  }

  TensorBlockKind kind() const { return m_kind; }
  const XprType& expr() const {
    eigen_assert(m_valid_expr);
    return m_expr;
  }
  // Consumers of data() require a dense buffer; strided views must use expr().
  const Scalar* data() const { return m_valid_expr ? m_expr.data() : m_data; }
  void cleanup() {}

  typedef internal::TensorBlockDescriptor<NumDims, IndexType> TensorBlockDesc;

  // TensorMaterializedBlock can be backed by different types of storage:
  //
  //   (1) Contiguous block of memory allocated with scratch allocator.
  //   (2) Contiguous block of memory reused from tensor block descriptor
  //       destination buffer.
  //   (3) Strided block of memory reused from tensor block descriptor
  //       destination buffer.
  //
  class Storage {
   public:
    Scalar* data() const { return m_data; }
    const Dimensions& dimensions() const { return m_dimensions; }
    const Dimensions& strides() const { return m_strides; }

    TensorMaterializedBlock AsTensorMaterializedBlock() const {
      return TensorMaterializedBlock(m_materialized_in_output ? internal::TensorBlockKind::kMaterializedInOutput
                                                              : internal::TensorBlockKind::kMaterializedInScratch,
                                     m_data, m_dimensions, !m_strided_storage);
    }

   private:
    friend class TensorMaterializedBlock<Scalar, NumDims, Layout, IndexType>;

    Storage(Scalar* data, const Dimensions& dimensions, const Dimensions& strides, bool materialized_in_output,
            bool strided_storage)
        : m_data(data),
          m_dimensions(dimensions),
          m_strides(strides),
          m_materialized_in_output(materialized_in_output),
          m_strided_storage(strided_storage) {}

    Scalar* m_data;
    Dimensions m_dimensions;
    Dimensions m_strides;
    bool m_materialized_in_output;
    bool m_strided_storage;
  };

  // Creates a storage for materialized block either from the block descriptor
  // destination buffer, or allocates a new buffer with scratch allocator.
  template <typename TensorBlockScratch>
  EIGEN_STRONG_INLINE static Storage prepareStorage(TensorBlockDesc& desc, TensorBlockScratch& scratch,
                                                    bool allow_strided_storage = false) {
    // Try to reuse destination as an output block buffer.
    typedef typename TensorBlockDesc::DestinationBuffer DestinationBuffer;

    if (desc.destination().kind() == DestinationBuffer::kContiguous) {
      Scalar* buffer = desc.destination().template data<Scalar>();
      desc.DropDestinationBuffer();
      return Storage(buffer, desc.dimensions(), internal::strides<Layout>(desc.dimensions()),
                     /*materialized_in_output=*/true,
                     /*strided_storage=*/false);

    } else if (desc.destination().kind() == DestinationBuffer::kStrided && allow_strided_storage) {
      Scalar* buffer = desc.destination().template data<Scalar>();
      desc.DropDestinationBuffer();
      return Storage(buffer, desc.dimensions(), desc.destination().strides(),
                     /*materialized_in_output=*/true, /*strided_storage=*/true);

    } else {
      void* mem = scratch.allocate(desc.size() * sizeof(Scalar));
      return Storage(static_cast<Scalar*>(mem), desc.dimensions(), internal::strides<Layout>(desc.dimensions()),
                     /*materialized_in_output=*/false,
                     /*strided_storage=*/false);
    }
  }

  // Creates a materialized block for the given descriptor from a memory buffer.
  template <typename DataDimensions, typename TensorBlockScratch>
  EIGEN_STRONG_INLINE static TensorMaterializedBlock materialize(const Scalar* data, const DataDimensions& data_dims,
                                                                 TensorBlockDesc& desc,
                                                                 TensorBlockScratch& /*scratch*/) {
    eigen_assert(array_size<DataDimensions>::value == desc.dimensions().size());

    TensorMaterializedBlock block(internal::TensorBlockKind::kView, data + desc.offset(), desc.dimensions());
    block.m_expr = XprType(data + desc.offset(), desc.dimensions(), internal::strides<Layout>(Dimensions(data_dims)));
    return block;
  }

 private:
  TensorBlockKind m_kind;
  const Scalar* m_data;
  Dimensions m_dimensions;
  XprType m_expr;
  bool m_valid_expr;
};

// -------------------------------------------------------------------------- //
// TensorCwiseUnaryBlock is a lazy tensor expression block that applies UnaryOp
// functor to the blocks produced by the underlying Tensor expression.

template <typename UnaryOp, typename ArgTensorBlock>
class TensorCwiseUnaryBlock {
  static constexpr bool NoArgBlockAccess = std::is_void<typename ArgTensorBlock::XprType>::value;

 public:
  typedef std::conditional_t<NoArgBlockAccess, void,
                             TensorCwiseUnaryOp<UnaryOp, const typename ArgTensorBlock::XprType> >
      XprType;

  typedef typename XprScalar<XprType>::type Scalar;

  TensorCwiseUnaryBlock(const ArgTensorBlock& arg_block, const UnaryOp& functor)
      : m_arg_block(arg_block), m_functor(functor) {}

  TensorBlockKind kind() const { return internal::TensorBlockKind::kExpr; }

  XprType expr() const { return XprType(m_arg_block.expr(), m_functor); }
  const Scalar* data() const { return nullptr; }
  void cleanup() { m_arg_block.cleanup(); }

 private:
  ArgTensorBlock m_arg_block;
  UnaryOp m_functor;
};

// -------------------------------------------------------------------------- //
// TensorCwiseBinaryBlock is a lazy tensor expression block that applies BinaryOp
// functor to the blocks produced by the underlying Tensor expression.

template <typename BinaryOp, typename LhsTensorBlock, typename RhsTensorBlock>
class TensorCwiseBinaryBlock {
  static constexpr bool NoArgBlockAccess =
      std::is_void<typename LhsTensorBlock::XprType>::value || std::is_void<typename RhsTensorBlock::XprType>::value;

 public:
  typedef std::conditional_t<
      NoArgBlockAccess, void,
      TensorCwiseBinaryOp<BinaryOp, const typename LhsTensorBlock::XprType, const typename RhsTensorBlock::XprType> >
      XprType;

  typedef typename XprScalar<XprType>::type Scalar;

  TensorCwiseBinaryBlock(const LhsTensorBlock& left_block, const RhsTensorBlock& right_block, const BinaryOp& functor)
      : m_left_block(left_block), m_right_block(right_block), m_functor(functor) {}

  TensorBlockKind kind() const { return internal::TensorBlockKind::kExpr; }

  XprType expr() const { return XprType(m_left_block.expr(), m_right_block.expr(), m_functor); }

  const Scalar* data() const { return nullptr; }

  void cleanup() {
    m_left_block.cleanup();
    m_right_block.cleanup();
  }

 private:
  LhsTensorBlock m_left_block;
  RhsTensorBlock m_right_block;
  BinaryOp m_functor;
};

// -------------------------------------------------------------------------- //
// TensorUnaryExprBlock is a lazy tensor expression block that can construct
// an arbitrary tensor expression from a block of the underlying type (this is a
// generalization of the TensorCwiseUnaryBlock for arbitrary expressions).

template <typename BlockFactory, typename ArgTensorBlock>
class TensorUnaryExprBlock {
  typedef typename ArgTensorBlock::XprType ArgXprType;
  static constexpr bool NoArgBlockAccess = std::is_void<ArgXprType>::value;

 public:
  typedef std::conditional_t<NoArgBlockAccess, void, typename BlockFactory::template XprType<ArgXprType>::type> XprType;

  typedef typename XprScalar<XprType>::type Scalar;

  TensorUnaryExprBlock(const ArgTensorBlock& arg_block, const BlockFactory& factory)
      : m_arg_block(arg_block), m_factory(factory) {}

  TensorBlockKind kind() const { return internal::TensorBlockKind::kExpr; }
  XprType expr() const { return m_factory.expr(m_arg_block.expr()); }
  const Scalar* data() const { return nullptr; }
  void cleanup() { m_arg_block.cleanup(); }

 private:
  ArgTensorBlock m_arg_block;
  BlockFactory m_factory;
};

// -------------------------------------------------------------------------- //
// TensorTernaryExprBlock is a lazy tensor expression block that can construct
// an arbitrary tensor expression from three blocks of the underlying type.

template <typename BlockFactory, typename Arg1TensorBlock, typename Arg2TensorBlock, typename Arg3TensorBlock>
class TensorTernaryExprBlock {
  typedef typename Arg1TensorBlock::XprType Arg1XprType;
  typedef typename Arg2TensorBlock::XprType Arg2XprType;
  typedef typename Arg3TensorBlock::XprType Arg3XprType;

  static constexpr bool NoArgBlockAccess =
      std::is_void<Arg1XprType>::value || std::is_void<Arg2XprType>::value || std::is_void<Arg3XprType>::value;

 public:
  typedef std::conditional_t<NoArgBlockAccess, void,
                             typename BlockFactory::template XprType<Arg1XprType, Arg2XprType, Arg3XprType>::type>
      XprType;

  typedef typename XprScalar<XprType>::type Scalar;

  TensorTernaryExprBlock(const Arg1TensorBlock& arg1_block, const Arg2TensorBlock& arg2_block,
                         const Arg3TensorBlock& arg3_block, const BlockFactory& factory)
      : m_arg1_block(arg1_block), m_arg2_block(arg2_block), m_arg3_block(arg3_block), m_factory(factory) {}

  TensorBlockKind kind() const { return internal::TensorBlockKind::kExpr; }
  XprType expr() const { return m_factory.expr(m_arg1_block.expr(), m_arg2_block.expr(), m_arg3_block.expr()); }
  const Scalar* data() const { return nullptr; }
  void cleanup() {
    m_arg1_block.cleanup();
    m_arg2_block.cleanup();
    m_arg3_block.cleanup();
  }

 private:
  Arg1TensorBlock m_arg1_block;
  Arg2TensorBlock m_arg2_block;
  Arg3TensorBlock m_arg3_block;
  BlockFactory m_factory;
};

// -------------------------------------------------------------------------- //
// StridedLinearBufferCopy provides a method to copy data between two linear
// buffers with different strides, with optimized paths for scatter/gather.

template <typename Scalar, typename IndexType>
class StridedLinearBufferCopy {
  typedef typename packet_traits<Scalar>::type Packet;
  typedef typename unpacket_traits<Packet>::half HalfPacket;
  enum {
    Vectorizable = packet_traits<Scalar>::Vectorizable,
    PacketSize = packet_traits<Scalar>::size,
    HalfPacketSize = unpacket_traits<HalfPacket>::size,
    HasHalfPacket = static_cast<int>(HalfPacketSize) < static_cast<int>(PacketSize)
  };

 public:
  // Specifying linear copy kind statically gives ~30% speedup for small sizes.
  enum class Kind {
    Linear = 0,        // src_stride == 1 && dst_stride == 1
    Scatter = 1,       // src_stride == 1 && dst_stride != 1 && dst_stride != -1
    FillLinear = 2,    // src_stride == 0 && dst_stride == 1
    FillScatter = 3,   // src_stride == 0 && dst_stride != 1
    Gather = 4,        // dst_stride == 1 && src_stride != -1
    Random = 5,        // everything else
    ReverseStore = 6,  // src_stride == 1 && dst_stride == -1
    ReverseLoad = 7,   // src_stride == -1 && dst_stride == 1
    ReverseBoth = 8    // src_stride == -1 && dst_stride == -1
  };

  struct Dst {
    Dst(IndexType o, IndexType s, Scalar* d) : offset(o), stride(s), data(d) {}

    IndexType offset;
    IndexType stride;
    Scalar* data;
  };

  struct Src {
    Src(IndexType o, IndexType s, const Scalar* d) : offset(o), stride(s), data(d) {}

    IndexType offset;
    IndexType stride;
    const Scalar* data;
  };

  template <typename StridedLinearBufferCopy::Kind kind>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void Run(const Dst& dst, const Src& src, const size_t count) {
    Run<kind>(count, dst.offset, dst.stride, dst.data, src.offset, src.stride, src.data);
  }

 private:
  template <typename StridedLinearBufferCopy::Kind kind>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void Run(const IndexType count, const IndexType dst_offset,
                                                        const IndexType dst_stride, Scalar* EIGEN_RESTRICT dst_data,
                                                        const IndexType src_offset, const IndexType src_stride,
                                                        const Scalar* EIGEN_RESTRICT src_data) {
    const Scalar* src = &src_data[src_offset];
    Scalar* dst = &dst_data[dst_offset];

    EIGEN_IF_CONSTEXPR (!Vectorizable) {
      for (Index i = 0; i < count; ++i) {
        dst[i * dst_stride] = src[i * src_stride];
      }
      return;
    }

    const IndexType vectorized_size = PacketSize * (count / PacketSize);
    IndexType i = 0;

    EIGEN_IF_CONSTEXPR (kind == StridedLinearBufferCopy::Kind::Linear ||
                        kind == StridedLinearBufferCopy::Kind::ReverseBoth) {
      // ******************************************************************** //
      // Linear copy from `src` to `dst`. `ReverseBoth` walks both runs
      // backwards, which leaves the elements contiguous and in the same order
      // in both buffers, so it is this same copy once each pointer is moved to
      // the low end of its run. No evaluator produces a reversed run on both
      // sides today; the kind exists so that such a run does not fall back to
      // `Random`.
      constexpr IndexType run_stride = kind == StridedLinearBufferCopy::Kind::ReverseBoth ? -1 : 1;
      eigen_assert(src_stride == run_stride && dst_stride == run_stride);
      const IndexType run_offset = run_stride == 1 ? 0 : count - 1;
      const Scalar* run_src = src - run_offset;
      Scalar* run_dst = dst - run_offset;
      const IndexType unrolled_size = (4 * PacketSize) * (count / (4 * PacketSize));
      for (; i < unrolled_size; i += 4 * PacketSize) {
        for (int j = 0; j < 4; ++j) {
          Packet p = ploadu<Packet>(run_src + i + j * PacketSize);
          pstoreu<Scalar, Packet>(run_dst + i + j * PacketSize, p);
        }
      }
      for (; i < vectorized_size; i += PacketSize) {
        Packet p = ploadu<Packet>(run_src + i);
        pstoreu<Scalar, Packet>(run_dst + i, p);
      }
      EIGEN_IF_CONSTEXPR (HasHalfPacket) {
        const IndexType vectorized_half_size = HalfPacketSize * (count / HalfPacketSize);
        if (i < vectorized_half_size) {
          HalfPacket p = ploadu<HalfPacket>(run_src + i);
          pstoreu<Scalar, HalfPacket>(run_dst + i, p);
          i += HalfPacketSize;
        }
      }
      for (; i < count; ++i) {
        run_dst[i] = run_src[i];
      }
      // ******************************************************************** //
    } else EIGEN_IF_CONSTEXPR (kind == StridedLinearBufferCopy::Kind::Scatter) {
      // Scatter from `src` to `dst`.
      eigen_assert(src_stride == 1 && dst_stride != 1);
      for (; i < vectorized_size; i += PacketSize) {
        Packet p = ploadu<Packet>(src + i);
        pscatter<Scalar, Packet>(dst + i * dst_stride, p, dst_stride);
      }
      EIGEN_IF_CONSTEXPR (HasHalfPacket) {
        const IndexType vectorized_half_size = HalfPacketSize * (count / HalfPacketSize);
        if (i < vectorized_half_size) {
          HalfPacket p = ploadu<HalfPacket>(src + i);
          pscatter<Scalar, HalfPacket>(dst + i * dst_stride, p, dst_stride);
          i += HalfPacketSize;
        }
      }
      for (; i < count; ++i) {
        dst[i * dst_stride] = src[i];
      }
      // ******************************************************************** //
    } else EIGEN_IF_CONSTEXPR (kind == StridedLinearBufferCopy::Kind::FillLinear) {
      // Fill `dst` with value at `*src`.
      eigen_assert(src_stride == 0 && dst_stride == 1);

      const IndexType unrolled_size = (4 * PacketSize) * (count / (4 * PacketSize));
      Scalar s = *src;
      Packet p = pset1<Packet>(s);
      for (; i < unrolled_size; i += 4 * PacketSize) {
        for (int j = 0; j < 4; ++j) {
          pstoreu<Scalar, Packet>(dst + i + j * PacketSize, p);
        }
      }
      for (; i < vectorized_size; i += PacketSize) {
        pstoreu<Scalar, Packet>(dst + i, p);
      }
      EIGEN_IF_CONSTEXPR (HasHalfPacket) {
        const IndexType vectorized_half_size = HalfPacketSize * (count / HalfPacketSize);
        if (i < vectorized_half_size) {
          HalfPacket hp = pset1<HalfPacket>(s);
          pstoreu<Scalar, HalfPacket>(dst + i, hp);
          i += HalfPacketSize;
        }
      }
      for (; i < count; ++i) {
        dst[i] = s;
      }
      // ******************************************************************** //
    } else EIGEN_IF_CONSTEXPR (kind == StridedLinearBufferCopy::Kind::FillScatter) {
      // Scatter `*src` into `dst`.
      eigen_assert(src_stride == 0 && dst_stride != 1);
      Scalar s = *src;
      Packet p = pset1<Packet>(s);
      for (; i < vectorized_size; i += PacketSize) {
        pscatter<Scalar, Packet>(dst + i * dst_stride, p, dst_stride);
      }
      EIGEN_IF_CONSTEXPR (HasHalfPacket) {
        const IndexType vectorized_half_size = HalfPacketSize * (count / HalfPacketSize);
        if (i < vectorized_half_size) {
          HalfPacket hp = pset1<HalfPacket>(s);
          pscatter<Scalar, HalfPacket>(dst + i * dst_stride, hp, dst_stride);
          i += HalfPacketSize;
        }
      }
      for (; i < count; ++i) {
        dst[i * dst_stride] = s;
      }
      // ******************************************************************** //
    } else EIGEN_IF_CONSTEXPR (kind == StridedLinearBufferCopy::Kind::Gather) {
      // Gather from `src` into `dst`.
      eigen_assert(dst_stride == 1);
      for (; i < vectorized_size; i += PacketSize) {
        Packet p = pgather<Scalar, Packet>(src + i * src_stride, src_stride);
        pstoreu<Scalar, Packet>(dst + i, p);
      }
      EIGEN_IF_CONSTEXPR (HasHalfPacket) {
        const IndexType vectorized_half_size = HalfPacketSize * (count / HalfPacketSize);
        if (i < vectorized_half_size) {
          HalfPacket p = pgather<Scalar, HalfPacket>(src + i * src_stride, src_stride);
          pstoreu<Scalar, HalfPacket>(dst + i, p);
          i += HalfPacketSize;
        }
      }
      for (; i < count; ++i) {
        dst[i] = src[i * src_stride];
      }
      // ******************************************************************** //
    } else EIGEN_IF_CONSTEXPR (kind == StridedLinearBufferCopy::Kind::ReverseStore) {
      // ******************************************************************** //
      // Contiguous read, reversed write: `dst[-i] = src[i]`. The destination
      // run covers [dst - count + 1, dst], so a packet is one contiguous load,
      // one `preverse` and one contiguous store -- instead of the `pscatter`
      // that a stride of -1 would otherwise fall into.
      eigen_assert(src_stride == 1 && dst_stride == -1);
      for (; i < vectorized_size; i += PacketSize) {
        Packet p = ploadu<Packet>(src + i);
        pstoreu<Scalar, Packet>(dst - i - (PacketSize - 1), preverse(p));
      }
      EIGEN_IF_CONSTEXPR (HasHalfPacket) {
        const IndexType vectorized_half_size = HalfPacketSize * (count / HalfPacketSize);
        if (i < vectorized_half_size) {
          HalfPacket p = ploadu<HalfPacket>(src + i);
          pstoreu<Scalar, HalfPacket>(dst - i - (HalfPacketSize - 1), preverse(p));
          i += HalfPacketSize;
        }
      }
      for (; i < count; ++i) {
        dst[-i] = src[i];
      }
      // ******************************************************************** //
    } else EIGEN_IF_CONSTEXPR (kind == StridedLinearBufferCopy::Kind::ReverseLoad) {
      // Reversed read, contiguous write: `dst[i] = src[-i]`.
      eigen_assert(src_stride == -1 && dst_stride == 1);
      for (; i < vectorized_size; i += PacketSize) {
        Packet p = ploadu<Packet>(src - i - (PacketSize - 1));
        pstoreu<Scalar, Packet>(dst + i, preverse(p));
      }
      EIGEN_IF_CONSTEXPR (HasHalfPacket) {
        const IndexType vectorized_half_size = HalfPacketSize * (count / HalfPacketSize);
        if (i < vectorized_half_size) {
          HalfPacket p = ploadu<HalfPacket>(src - i - (HalfPacketSize - 1));
          pstoreu<Scalar, HalfPacket>(dst + i, preverse(p));
          i += HalfPacketSize;
        }
      }
      for (; i < count; ++i) {
        dst[i] = src[-i];
      }
      // ******************************************************************** //
    } else EIGEN_IF_CONSTEXPR (kind == StridedLinearBufferCopy::Kind::Random) {
      // Random.
      for (; i < count; ++i) {
        dst[i * dst_stride] = src[i * src_stride];
      }
    } else {
      eigen_assert(false);
    }
  }
};

// -------------------------------------------------------------------------- //
// TensorBlockIO copies data from `src` tensor block, to the `dst` tensor block.
// It's possible to specify src->dst dimension mapping for the copy operation.
// Dimensions of `dst` specify how many elements have to be copied, for the
// `src` we need to know only stride to navigate through source memory buffer.
//
// Strides may be non-unit (strided/dilated views), negative (reversed views),
// or, on the `src` side only, zero (the broadcasting trick). Inner dimensions
// are fused into one copy only while the elements keep forming a single
// arithmetic progression at the inner stride on both sides.

template <typename Scalar, typename IndexType, int NumDims, int Layout>
class TensorBlockIO {
  static constexpr bool IsColMajor = Layout == ColMajor;

  typedef StridedLinearBufferCopy<Scalar, IndexType> LinCopy;

 public:
  typedef DSizes<IndexType, NumDims> Dimensions;
  typedef DSizes<int, NumDims> DimensionsMap;

  struct Dst {
    Dst(const Dimensions& dst_dims, const Dimensions& dst_strides, Scalar* dst, IndexType dst_offset = 0)
        : dims(dst_dims), strides(dst_strides), data(dst), offset(dst_offset) {}

    Dimensions dims;
    Dimensions strides;
    Scalar* data;
    IndexType offset;
  };

  struct Src {
    Src(const Dimensions& src_strides, const Scalar* src, IndexType src_offset = 0)
        : strides(src_strides), data(src), offset(src_offset) {}

    Dimensions strides;
    const Scalar* data;
    IndexType offset;
  };

  // Copies data to `dst` from `src`, using provided dimensions mapping:
  //
  //   src_dimension_index = dst_to_src_dim_map[dst_dimension_index]
  //
  // Returns the number of copied elements.
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE IndexType Copy(const Dst& dst, const Src& src,
                                                              const DimensionsMap& dst_to_src_dim_map) {
    // Copy single scalar value from `src` to `dst`.
    EIGEN_IF_CONSTEXPR (NumDims == 0) {
      *(dst.data + dst.offset) = *(src.data + src.offset);
      return 1;
    }

    // Give a shorter name to `dst_to_src_dim_map`.
    const DimensionsMap& dim_map = dst_to_src_dim_map;

    // Do not squeeze reordered inner dimensions.
    int num_squeezable_dims = NumSqueezableInnerDims(dim_map);

    // NOTE: We find the innermost dimension (contiguous in memory) in the dst
    // block, and we write data linearly into that dimension, reading it from
    // the src. If dimensions are reordered, we might end up reading data from
    // the src with `stride != 1`.
    //
    // NOTE: Random-Read/Linear-Write can be up to ~2X faster than
    // Linear-Read/Random-Write: https://stackoverflow.com/a/54935680

    // Find the innermost dimension in the dst whose size is not 1. This is the
    // effective inner dim.
    int num_size_one_inner_dims = 0;
    for (int i = 0; i < num_squeezable_dims; ++i) {
      const int dst_dim = IsColMajor ? i : NumDims - i - 1;
      if (dst.dims[dst_dim] != 1) break;
      num_size_one_inner_dims++;
    }

    // If all dimensions are of size 1, just copy a scalar from `src` to `dst`.
    if (num_size_one_inner_dims == NumDims) {
      *(dst.data + dst.offset) = *(src.data + src.offset);
      return 1;
    }

    // Innermost dimension in the dst that still has to be copied. Its stride
    // need not be 1: the run may be dilated or reversed.
    const int dst_inner_dim = IsColMajor ? num_size_one_inner_dims : NumDims - num_size_one_inner_dims - 1;

    // Dimension in the src that corresponds to the dst innermost dimension.
    const int src_dim_for_dst_inner_dim = NumDims == 0 ? 1 : dim_map[dst_inner_dim];

    // Number of elements copied per line.
    IndexType dst_inner_dim_size = NumDims == 0 ? 1 : dst.dims[dst_inner_dim];

    // Squeeze multiple inner dims into one if the elements keep forming a
    // single arithmetic progression at the inner stride across the dimension
    // boundary in both `dst` and `src` memory, so we can do less linear copy
    // calls.
    const IndexType output_stride = NumDims == 0 ? 1 : dst.strides[dst_inner_dim];
    const IndexType input_stride = NumDims == 0 ? 1 : src.strides[src_dim_for_dst_inner_dim];
    for (int i = num_size_one_inner_dims + 1; i < num_squeezable_dims; ++i) {
      const int dst_dim = IsColMajor ? i : NumDims - i - 1;
      const IndexType dst_stride = dst.strides[dst_dim];
      const IndexType src_stride = src.strides[dim_map[dst_dim]];
      if (dst_stride == dst_inner_dim_size * output_stride && src_stride == dst_inner_dim_size * input_stride) {
        dst_inner_dim_size *= dst.dims[dst_dim];
        ++num_size_one_inner_dims;
      } else {
        break;
      }
    }

    // Setup strides to read data from `src` and write to `dst`.
    IndexType input_offset = src.offset;
    IndexType output_offset = dst.offset;

    constexpr int at_least_1_dim = NumDims <= 1 ? 1 : NumDims - 1;
    array<BlockIteratorState, at_least_1_dim> it;

    // Initialize block iterator state. Squeeze away any dimension of size 1.
    int idx = 0;  // currently initialized iterator state index
    for (int i = num_size_one_inner_dims; i < NumDims - 1; ++i) {
      const int dst_dim = IsColMajor ? i + 1 : NumDims - i - 2;
      if (dst.dims[dst_dim] == 1) continue;

      it[idx].size = dst.dims[dst_dim];
      it[idx].input_stride = src.strides[dim_map[dst_dim]];
      it[idx].output_stride = dst.strides[dst_dim];

      it[idx].input_span = it[idx].input_stride * (it[idx].size - 1);
      it[idx].output_span = it[idx].output_stride * (it[idx].size - 1);

      idx++;
    }

    // Iterate copying data from src to dst.
    const IndexType block_total_size = NumDims == 0 ? 1 : dst.dims.TotalSize();

#define COPY_INNER_DIM(KIND)                                                                                      \
  IndexType num_copied = 0;                                                                                       \
  for (num_copied = 0; num_copied < block_total_size; num_copied += dst_inner_dim_size) {                         \
    LinCopy::template Run<KIND>(typename LinCopy::Dst(output_offset, output_stride, dst.data),                    \
                                typename LinCopy::Src(input_offset, input_stride, src.data), dst_inner_dim_size); \
                                                                                                                  \
    for (int j = 0; j < idx; ++j) {                                                                               \
      if (++it[j].count < it[j].size) {                                                                           \
        input_offset += it[j].input_stride;                                                                       \
        output_offset += it[j].output_stride;                                                                     \
        break;                                                                                                    \
      }                                                                                                           \
      it[j].count = 0;                                                                                            \
      input_offset -= it[j].input_span;                                                                           \
      output_offset -= it[j].output_span;                                                                         \
    }                                                                                                             \
  }                                                                                                               \
  return num_copied;

    if (input_stride == 1 && output_stride == 1) {
      COPY_INNER_DIM(LinCopy::Kind::Linear);
    } else if (input_stride == 1 && output_stride == -1) {
      COPY_INNER_DIM(LinCopy::Kind::ReverseStore);
    } else if (input_stride == -1 && output_stride == 1) {
      COPY_INNER_DIM(LinCopy::Kind::ReverseLoad);
    } else if (input_stride == -1 && output_stride == -1) {
      COPY_INNER_DIM(LinCopy::Kind::ReverseBoth);
    } else if (input_stride == 1 && output_stride != 1) {
      COPY_INNER_DIM(LinCopy::Kind::Scatter);
    } else if (input_stride == 0 && output_stride == 1) {
      COPY_INNER_DIM(LinCopy::Kind::FillLinear);
    } else if (input_stride == 0 && output_stride != 1) {
      COPY_INNER_DIM(LinCopy::Kind::FillScatter);
    } else if (output_stride == 1) {
      COPY_INNER_DIM(LinCopy::Kind::Gather);
    } else {
      COPY_INNER_DIM(LinCopy::Kind::Random);
    }

#undef COPY_INNER_DIM
  }

  // Copy from `src` to `dst` with an identity src->dst dimension map. Returns
  // the number of copied elements.
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE IndexType Copy(const Dst& dst, const Src& src) {
    DimensionsMap dst_to_src_map;
    for (int i = 0; i < NumDims; ++i) dst_to_src_map[i] = i;
    return Copy(dst, src, dst_to_src_map);
  }

 private:
  struct BlockIteratorState {
    BlockIteratorState() = default;

    IndexType size = 0;
    IndexType count = 0;
    IndexType input_stride = 0;
    IndexType output_stride = 0;
    IndexType input_span = 0;
    IndexType output_span = 0;
  };

  // Compute how many inner dimensions it's allowed to squeeze when doing IO
  // between two tensor blocks. It's safe to squeeze inner dimensions, only
  // if they are not reordered.
  static int NumSqueezableInnerDims(const DimensionsMap& dim_map) {
    int num_squeezable_dims = 0;
    for (int i = 0; i < NumDims; ++i) {
      const int dim = IsColMajor ? i : NumDims - i - 1;
      if (dim_map[dim] != dim) break;
      num_squeezable_dims++;
    }
    return num_squeezable_dims;
  }
};

// -------------------------------------------------------------------------- //
// Bind coefficient-wise block expressions to one contiguous inner run at a time.
// This moves strided source index calculations out of the coefficient/packet loop.
template <typename XprType>
struct TensorBlockRead {
  static constexpr bool Supported = false;
  using Expression = XprType;
  using Index = typename XprType::Index;
  explicit TensorBlockRead(const XprType& expression) : m_expression(expression) {}
  Index innerSize() const { return NumTraits<Index>::highest(); }
  const Expression& expr(Index, Index) const { return m_expression; }

 private:
  const XprType& m_expression;
};

template <typename Scalar, int NumDims, int Layout, typename Index>
struct TensorBlockRead<TensorBlockView<Scalar, NumDims, Layout, Index>> {
  static constexpr bool Supported = true;
  using XprType = TensorBlockView<Scalar, NumDims, Layout, Index>;
  using Expression = TensorBlockView<Scalar, 1, Layout, Index>;
  explicit TensorBlockRead(const XprType& expression) : m_evaluator(expression, m_device), m_inner_size(1) {
    for (int i = 0; i < NumDims; ++i) {
      const int dim = Layout == ColMajor ? i : NumDims - 1 - i;
      if (expression.dimensions()[dim] > 1 && expression.strides()[dim] != m_inner_size) break;
      m_inner_size *= expression.dimensions()[dim];
    }
  }
  Index innerSize() const { return m_inner_size; }
  Expression expr(Index offset, Index size) const {
    return Expression(m_evaluator.coeffAddress(offset), DSizes<Index, 1>(size));
  }

 private:
  DefaultDevice m_device;
  TensorEvaluator<const XprType, DefaultDevice> m_evaluator;
  Index m_inner_size;
};

// Rebinding a run must not restart a functor's mutable state.
template <typename Functor>
class TensorBlockReadFunctor {
 public:
  explicit TensorBlockReadFunctor(const Functor& functor) : m_functor(&functor) {}
  template <typename... Args>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE auto operator()(Args&&... args) const
      -> decltype(std::declval<const Functor&>()(std::forward<Args>(args)...)) {
    return (*m_functor)(std::forward<Args>(args)...);
  }
  template <typename... Args, typename F = Functor>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE auto packetOp(Args&&... args) const
      -> decltype(std::declval<const F&>().packetOp(std::forward<Args>(args)...)) {
    return m_functor->packetOp(std::forward<Args>(args)...);
  }

 private:
  const Functor* m_functor;
};

template <typename Functor>
struct functor_traits<TensorBlockReadFunctor<Functor>> : functor_traits<Functor> {};

template <typename UnaryOp, typename Arg>
struct TensorBlockRead<TensorCwiseUnaryOp<UnaryOp, Arg>> {
  using ArgRead = TensorBlockRead<remove_all_t<Arg>>;
  static constexpr bool Supported = ArgRead::Supported;
  using XprType = TensorCwiseUnaryOp<UnaryOp, Arg>;
  using Expression = TensorCwiseUnaryOp<TensorBlockReadFunctor<UnaryOp>, const typename ArgRead::Expression>;
  using Index = typename XprType::Index;
  explicit TensorBlockRead(const XprType& expression)
      : m_arg(expression.nestedExpression()), m_functor(expression.functor()) {}
  Index innerSize() const { return m_arg.innerSize(); }
  Expression expr(Index offset, Index size) const {
    return Expression(m_arg.expr(offset, size), TensorBlockReadFunctor<UnaryOp>(m_functor));
  }

 private:
  ArgRead m_arg;
  UnaryOp m_functor;
};

template <typename BinaryOp, typename Left, typename Right>
struct TensorBlockRead<TensorCwiseBinaryOp<BinaryOp, Left, Right>> {
  using LeftRead = TensorBlockRead<remove_all_t<Left>>;
  using RightRead = TensorBlockRead<remove_all_t<Right>>;
  static constexpr bool Supported = LeftRead::Supported && RightRead::Supported;
  using XprType = TensorCwiseBinaryOp<BinaryOp, Left, Right>;
  using Expression = TensorCwiseBinaryOp<TensorBlockReadFunctor<BinaryOp>, const typename LeftRead::Expression,
                                         const typename RightRead::Expression>;
  using Index = typename XprType::Index;
  explicit TensorBlockRead(const XprType& expression)
      : m_left(expression.lhsExpression()), m_right(expression.rhsExpression()), m_functor(expression.functor()) {}
  Index innerSize() const { return numext::mini(m_left.innerSize(), m_right.innerSize()); }
  Expression expr(Index offset, Index size) const {
    return Expression(m_left.expr(offset, size), m_right.expr(offset, size),
                      TensorBlockReadFunctor<BinaryOp>(m_functor));
  }

 private:
  LeftRead m_left;
  RightRead m_right;
  BinaryOp m_functor;
};

template <typename NullaryOp, typename Arg>
struct TensorBlockRead<TensorCwiseNullaryOp<NullaryOp, Arg>> {
  using XprType = TensorCwiseNullaryOp<NullaryOp, Arg>;
  using Evaluator = TensorEvaluator<const XprType, DefaultDevice>;
  static constexpr bool Supported = Evaluator::IndexIndependentFunctor;
  using Index = typename XprType::Index;
  using RunView = TensorBlockView<typename XprType::Scalar, 1, traits<XprType>::Layout, Index>;
  using Expression = TensorCwiseNullaryOp<NullaryOp, const RunView>;
  explicit TensorBlockRead(const XprType& expression) : m_functor(expression.functor()) {}
  Index innerSize() const { return NumTraits<Index>::highest(); }
  Expression expr(Index, Index size) const { return Expression(RunView(nullptr, DSizes<Index, 1>(size)), m_functor); }

 private:
  NullaryOp m_functor;
};

template <typename TernaryOp, typename Arg1, typename Arg2, typename Arg3>
struct TensorBlockRead<TensorCwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>> {
  using Arg1Read = TensorBlockRead<remove_all_t<Arg1>>;
  using Arg2Read = TensorBlockRead<remove_all_t<Arg2>>;
  using Arg3Read = TensorBlockRead<remove_all_t<Arg3>>;
  static constexpr bool Supported = Arg1Read::Supported && Arg2Read::Supported && Arg3Read::Supported;
  using XprType = TensorCwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>;
  using Expression = TensorCwiseTernaryOp<TensorBlockReadFunctor<TernaryOp>, const typename Arg1Read::Expression,
                                          const typename Arg2Read::Expression, const typename Arg3Read::Expression>;
  using Index = typename XprType::Index;
  explicit TensorBlockRead(const XprType& expression)
      : m_arg1(expression.arg1Expression()),
        m_arg2(expression.arg2Expression()),
        m_arg3(expression.arg3Expression()),
        m_functor(expression.functor()) {}
  Index innerSize() const {
    return numext::mini(m_arg1.innerSize(), numext::mini(m_arg2.innerSize(), m_arg3.innerSize()));
  }
  Expression expr(Index offset, Index size) const {
    return Expression(m_arg1.expr(offset, size), m_arg2.expr(offset, size), m_arg3.expr(offset, size),
                      TensorBlockReadFunctor<TernaryOp>(m_functor));
  }

 private:
  Arg1Read m_arg1;
  Arg2Read m_arg2;
  Arg3Read m_arg3;
  TernaryOp m_functor;
};

template <typename Cond, typename Then, typename Else>
struct TensorBlockRead<TensorSelectOp<Cond, Then, Else>> {
  using CondRead = TensorBlockRead<remove_all_t<Cond>>;
  using ThenRead = TensorBlockRead<remove_all_t<Then>>;
  using ElseRead = TensorBlockRead<remove_all_t<Else>>;
  static constexpr bool Supported = CondRead::Supported && ThenRead::Supported && ElseRead::Supported;
  using XprType = TensorSelectOp<Cond, Then, Else>;
  using Expression = TensorSelectOp<const typename CondRead::Expression, const typename ThenRead::Expression,
                                    const typename ElseRead::Expression>;
  using Index = typename XprType::Index;
  explicit TensorBlockRead(const XprType& expression)
      : m_cond(expression.ifExpression()), m_then(expression.thenExpression()), m_else(expression.elseExpression()) {}
  Index innerSize() const {
    return numext::mini(m_cond.innerSize(), numext::mini(m_then.innerSize(), m_else.innerSize()));
  }
  Expression expr(Index offset, Index size) const {
    return Expression(m_cond.expr(offset, size), m_then.expr(offset, size), m_else.expr(offset, size));
  }

 private:
  CondRead m_cond;
  ThenRead m_then;
  ElseRead m_else;
};

template <typename Scalar, typename Arg>
struct TensorBlockRead<TensorConversionOp<Scalar, Arg>> {
  using ArgRead = TensorBlockRead<remove_all_t<Arg>>;
  static constexpr bool Supported = ArgRead::Supported;
  using XprType = TensorConversionOp<Scalar, Arg>;
  using Expression = TensorConversionOp<Scalar, const typename ArgRead::Expression>;
  using Index = typename XprType::Index;
  explicit TensorBlockRead(const XprType& expression) : m_arg(expression.expression()) {}
  Index innerSize() const { return m_arg.innerSize(); }
  Expression expr(Index offset, Index size) const { return Expression(m_arg.expr(offset, size)); }

 private:
  ArgRead m_arg;
};

// TensorBlockAssignment assigns a block expression of type `TensorBlockExpr` to
// a Tensor block defined by `desc`, backed by a memory buffer at `target`.
//
// Currently there is no way to write from a Tensor expression to a block of
// memory, if dimensions are reordered. If you need to do that, you should
// materialize a Tensor block expression into a memory buffer, and then use
// TensorBlockIO to copy data between two memory buffers with a custom
// `target->src` dimension map (see definition above).
//
// Also currently the innermost dimension of `target` must have a stride '1'
// (contiguous in memory). This restriction could be lifted with a `pscatter`,
// but in practice it's never needed, and there is a similar TensorBlockIO
// workaround for that.
//
// TODO(ezhulenev): TensorBlockAssignment is a special case of TensorBlockIO
// where `src` is a tensor expression. Explore if it is possible to rewrite IO
// to use expressions instead of pointers, and after that TensorBlockAssignment
// will become an alias to IO.
template <typename Scalar, int NumDims, typename TensorBlockExpr, typename IndexType = Eigen::Index>
class TensorBlockAssignment {
  // We will use coeff/packet path to evaluate block expressions.
  typedef TensorEvaluator<const TensorBlockExpr, DefaultDevice> TensorBlockEvaluator;

  typedef DSizes<IndexType, NumDims> Dimensions;
  using BlockRead = TensorBlockRead<TensorBlockExpr>;

  enum { Vectorizable = packet_traits<Scalar>::Vectorizable, PacketSize = packet_traits<Scalar>::size };

  template <bool Vectorizable, typename Evaluator>
  struct InnerDimAssign {
    EIGEN_ALWAYS_INLINE static void Run(Scalar* target, IndexType count, const Evaluator& eval, IndexType eval_offset) {
      for (IndexType i = 0; i < count; ++i) {
        target[i] = eval.coeff(eval_offset + i);
      }
    }
  };

  template <typename Evaluator>
  struct InnerDimAssign<true, Evaluator> {
    EIGEN_ALWAYS_INLINE static void Run(Scalar* target, IndexType count, const Evaluator& eval, IndexType eval_offset) {
      typedef typename packet_traits<Scalar>::type Packet;

      const IndexType unrolled_size = (4 * PacketSize) * (count / (4 * PacketSize));
      const IndexType vectorized_size = PacketSize * (count / PacketSize);
      IndexType i = 0;

      for (; i < unrolled_size; i += 4 * PacketSize) {
        for (int j = 0; j < 4; ++j) {
          const IndexType idx = eval_offset + i + j * PacketSize;
          Packet p = eval.template packet<Unaligned>(idx);
          pstoreu<Scalar>(target + i + j * PacketSize, p);
        }
      }

      for (; i < vectorized_size; i += PacketSize) {
        Packet p = eval.template packet<Unaligned>(eval_offset + i);
        pstoreu<Scalar>(target + i, p);
      }

      for (; i < count; ++i) {
        target[i] = eval.coeff(eval_offset + i);
      }
    }
  };

  template <typename Evaluator>
  static EIGEN_STRONG_INLINE void AssignInner(Scalar* target, IndexType count, const Evaluator& eval, const BlockRead&,
                                              IndexType offset, std::false_type) {
    InnerDimAssign<Vectorizable && Evaluator::PacketAccess, Evaluator>::Run(target, count, eval, offset);
  }

  static EIGEN_STRONG_INLINE void AssignInner(Scalar* target, IndexType count, const TensorBlockEvaluator&,
                                              const BlockRead& reader, IndexType offset, std::true_type) {
    const auto expression = reader.expr(offset, count);
    using RunEvaluator = TensorEvaluator<const typename BlockRead::Expression, DefaultDevice>;
    const DefaultDevice device;
    const RunEvaluator eval(expression, device);
    InnerDimAssign<Vectorizable && RunEvaluator::PacketAccess, RunEvaluator>::Run(target, count, eval, 0);
  }

 public:
  struct Target {
    Target(const Dimensions& target_dims, const Dimensions& target_strides, Scalar* target_data,
           IndexType target_offset = 0)
        : dims(target_dims), strides(target_strides), data(target_data), offset(target_offset) {}

    Dimensions dims;
    Dimensions strides;
    Scalar* data;
    IndexType offset;
  };

  static Target target(const Dimensions& target_dims, const Dimensions& target_strides, Scalar* target_data,
                       IndexType target_offset = 0) {
    return Target(target_dims, target_strides, target_data, target_offset);
  }

  template <typename TargetDimsIndexType, typename TargetStridesIndexType>
  static Target target(const DSizes<TargetDimsIndexType, NumDims>& target_dims,
                       const DSizes<TargetStridesIndexType, NumDims>& target_strides, Scalar* target_data,
                       IndexType target_offset = 0) {
    // DSizes constructor will do index type promotion if it's safe.
    return Target(Dimensions(target_dims), Dimensions(target_strides), target_data, target_offset);
  }

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void Run(const Target& target, const TensorBlockExpr& expr) {
    // Prepare evaluator for block expression.
    DefaultDevice default_device;
    TensorBlockEvaluator eval(expr, default_device);
    const BlockRead reader(expr);

    // Tensor block expression dimension should match destination dimensions.
    eigen_assert(dimensions_match(target.dims, eval.dimensions()));

    Run(target, eval, reader, bool_constant<BlockRead::Supported>());
  }

 private:
  static void Run(const Target& target, const TensorBlockEvaluator& eval, const BlockRead& reader, std::false_type) {
    RunImpl<false>(target, eval, reader);
  }

  static void Run(const Target& target, const TensorBlockEvaluator& eval, const BlockRead& reader, std::true_type) {
    const IndexType size = NumDims == 0 ? 1 : target.dims.TotalSize();
    if (reader.innerSize() >= size) {
      // Bind a dense source once, even when the destination has strided rows.
      const auto expression = reader.expr(0, size);
      using RunEvaluator = TensorEvaluator<const typename BlockRead::Expression, DefaultDevice>;
      const DefaultDevice device;
      const RunEvaluator dense_eval(expression, device);
      RunImpl<false>(target, dense_eval, reader);
    } else {
      RunImpl<true>(target, eval, reader);
    }
  }

  template <bool UseInnerRuns, typename Evaluator>
  static EIGEN_STRONG_INLINE void RunImpl(const Target& target, const Evaluator& eval, const BlockRead& reader) {
    static constexpr int Layout = Evaluator::Layout;
    static constexpr bool is_col_major = Layout == ColMajor;

    // Initialize output inner dimension size based on a layout.
    const IndexType output_size = NumDims == 0 ? 1 : target.dims.TotalSize();
    constexpr int inner_dim_idx = NumDims == 0 ? 0 : (is_col_major ? 0 : NumDims - 1);
    IndexType output_inner_dim_size = NumDims == 0 ? 1 : target.dims[inner_dim_idx];

    // Target inner dimension stride must be '1'.
    EIGEN_IF_CONSTEXPR (NumDims > 0) {
      eigen_assert(target.strides[inner_dim_idx] == 1);
    }

    // Squeeze multiple inner dims into one if they are contiguous in `target`.
    IndexType num_squeezed_dims = 0;
    for (Index i = 1; i < NumDims; ++i) {
      const Index dim = is_col_major ? i : NumDims - i - 1;
      const IndexType target_stride = target.strides[dim];

      if (output_inner_dim_size == target_stride &&
          (!UseInnerRuns || output_inner_dim_size * target.dims[dim] <= reader.innerSize())) {
        output_inner_dim_size *= target.dims[dim];
        num_squeezed_dims++;
      } else {
        break;
      }
    }

    // Initialize output block iterator state. Dimension in this array are
    // always in inner_most -> outer_most order (col major layout).
    array<BlockIteratorState, NumDims> it;

    int idx = 0;  // currently initialized iterator state index
    for (Index i = num_squeezed_dims; i < NumDims - 1; ++i) {
      const Index dim = is_col_major ? i + 1 : NumDims - i - 2;

      it[idx].count = 0;
      it[idx].size = target.dims[dim];
      it[idx].output_stride = target.strides[dim];
      it[idx].output_span = it[idx].output_stride * (it[idx].size - 1);
      idx++;
    }

    // We read block expression from the beginning, and start writing data to
    // `target` at given offset.
    IndexType input_offset = 0;
    IndexType output_offset = target.offset;

    // Iterate copying data from `eval` to `target`.
    for (IndexType i = 0; i < output_size; i += output_inner_dim_size) {
      // Assign to `target` at current offset.
      AssignInner(target.data + output_offset, output_inner_dim_size, eval, reader, input_offset,
                  bool_constant<UseInnerRuns>());

      // Move input offset forward by the number of assigned coefficients.
      input_offset += output_inner_dim_size;

      // Update index.
      for (int j = 0; j < idx; ++j) {
        if (++it[j].count < it[j].size) {
          output_offset += it[j].output_stride;
          break;
        }
        it[j].count = 0;
        output_offset -= it[j].output_span;
      }
    }
  }

 private:
  struct BlockIteratorState {
    BlockIteratorState() = default;

    IndexType count = 0;
    IndexType size = 0;
    IndexType output_stride = 0;
    IndexType output_span = 0;
  };
};

// -------------------------------------------------------------------------- //

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_TENSOR_TENSOR_BLOCK_H
