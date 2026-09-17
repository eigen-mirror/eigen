// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPARSE_SCATTER_H
#define EIGEN_SPARSE_SCATTER_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

constexpr Index kSparseScatterPacketMinSize = 32;

template <typename Scalar, bool Vectorize>
struct sparse_scatter_sub_impl {
  template <typename StorageIndex, typename ValuesEvaluator>
  static EIGEN_STRONG_INLINE Index run(Scalar*, const StorageIndex*, const ValuesEvaluator&, Index) {
    return 0;
  }
};

template <typename Scalar>
struct sparse_scatter_sub_impl<Scalar, true> {
  // Keep the packet loop out of the common path for short updates.
  template <typename StorageIndex, typename ValuesEvaluator>
  static EIGEN_DONT_INLINE Index run(Scalar* EIGEN_RESTRICT dense, const StorageIndex* indices,
                                     const ValuesEvaluator& values, Index size) {
    // Limit register pressure from the indirect addresses.
    using Packet = typename find_best_packet<Scalar, 4>::type;
    constexpr Index PacketSize = unpacket_traits<Packet>::size;
    const Index end = size - size % PacketSize;
    for (Index i = 0; i < end; i += PacketSize) {
      EIGEN_ALIGN_MAX Scalar gathered[PacketSize];
      for (Index lane = 0; lane < PacketSize; ++lane) gathered[lane] = dense[indices[i + lane]];
      pstoreu(gathered, psub(ploadu<Packet>(gathered), values.template packet<Unaligned, Packet>(i)));
      for (Index lane = 0; lane < PacketSize; ++lane) dense[indices[i + lane]] = gathered[lane];
    }
    return end;
  }
};

// Subtract a packet-sized prefix of values from dense at distinct indices, returning its length.
// Neither input may alias dense. Callers retain their scalar loop for short updates and the tail.
// Packing complex destinations costs more than the packet arithmetic saves.
template <typename Scalar, typename StorageIndex, typename Values>
EIGEN_STRONG_INLINE Index sparse_scatter_sub_packets(Scalar* EIGEN_RESTRICT dense, const StorageIndex* indices,
                                                     const MatrixBase<Values>& values) {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Values)
  EIGEN_STATIC_ASSERT((std::is_same<Scalar, typename Values::Scalar>::value), YOU_MIXED_DIFFERENT_NUMERIC_TYPES)
  constexpr bool Vectorize = (evaluator<Values>::Flags & PacketAccessBit) &&
                             (evaluator<Values>::Flags & LinearAccessBit) && packet_traits<Scalar>::HasSub &&
                             !NumTraits<Scalar>::IsComplex;
  if (!Vectorize || values.size() < kSparseScatterPacketMinSize) return 0;
  evaluator<Values> valuesEval(values.derived());
  return sparse_scatter_sub_impl<Scalar, Vectorize>::run(dense, indices, valuesEval, values.size());
}

template <bool Conjugate, typename Scalar, typename StorageIndex>
EIGEN_STRONG_INLINE Index sparse_scatter_sub_packets(Scalar* dense, const StorageIndex* indices, const Scalar* values,
                                                     Index size, const Scalar& scale) {
  constexpr bool Vectorize = packet_traits<Scalar>::Vectorizable && packet_traits<Scalar>::HasMul &&
                             packet_traits<Scalar>::HasSub && !NumTraits<Scalar>::IsComplex;
  if (!Vectorize || size < kSparseScatterPacketMinSize) return 0;
  const Map<const Matrix<Scalar, Dynamic, 1>> mappedValues(values, size);
  return sparse_scatter_sub_packets(dense, indices, mappedValues.template conjugateIf<Conjugate>() * scale);
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_SPARSE_SCATTER_H
