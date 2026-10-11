// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_PREDICATOR_H
#define EIGEN_PREDICATOR_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// Predicate reductions all_of and any_of. A predicate tests a scalar with operator() and a packet with
// packetOp, which returns a truth packet: each lane is zero where the predicate is false and, where it is true, the
// lane mask pcmp_* produces (all ones, or one for bool packets). Truth packets combine with pand and por, so the
// traversals below reduce and branch once per block of packets rather than once per packet. Results do not depend on
// the grouping, but a short-circuit may read past the deciding coefficient, up to the end of its block.
template <typename Scalar>
struct nonzero_predicate {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const Scalar& x) const { return x != Scalar(0); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    return pandnot(ptrue(x), pcmp_eq(x, pzero(x)));
  }
};
template <typename Scalar>
struct functor_traits<nonzero_predicate<Scalar>> {
  static constexpr bool PacketAccess = packet_traits<Scalar>::HasCmp;
  // Accepts that ARMv7 NEON reads a subnormal float as zero: packets compare there already, and grouped scalar tests
  // may be vectorized onto it. Testing float one at a time instead cost 0.7-0.85x at sizes with a scalar tail on
  // Cortex-A72.
  static constexpr bool GroupScalars = true;
};

template <typename Scalar>
struct isfinite_predicate {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const Scalar& x) const { return (numext::isfinite)(x); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    return pisfinite(x);
  }
};
template <typename Scalar>
struct functor_traits<isfinite_predicate<Scalar>> {
  static constexpr bool PacketAccess = packet_traits<Scalar>::HasCmp;
  static constexpr bool GroupScalars = true;
};

// Separate specializations keep custom scalars that support only one of isApprox and isMuchSmallerThan compiling.
template <typename Scalar, bool Approximate>
struct fuzzy_constant_predicate {
  using RealScalar = typename NumTraits<Scalar>::Real;
  Scalar value;
  RealScalar precision;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const Scalar& x) const {
    return internal::isApprox(x, value, precision);
  }
  // Requires 0 <= precision <= 1 and a finite value. Then min(|x|, |v|) p = min(|x| p, |v| p) exactly, and |x| p is
  // x copysign(p, v) where x has the sign of v; where it does not, |x - v| >= max(|x|, |v|) > p min(|x|, |v|) unless
  // x = v = 0, and x copysign(p, v) <= 0 rejects it as well.
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    const Packet bound = pset1<Packet>(numext::abs(value) * precision);
    const Packet scale = pset1<Packet>(numext::copysign(precision, value));
    return pcmp_le(pabs(psub(x, pset1<Packet>(value))), pmin(pmul(x, scale), bound));
  }
  EIGEN_DEVICE_FUNC bool packetOpApplies() const {
    return precision >= RealScalar(0) && precision <= RealScalar(1) && (numext::isfinite)(value);
  }
};
template <typename Scalar>
struct fuzzy_constant_predicate<Scalar, false> {
  Scalar value;
  typename NumTraits<Scalar>::Real precision;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const Scalar& x) const {
    return internal::isMuchSmallerThan(x, Scalar(1), precision);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    return pcmp_le(pabs(x), pset1<Packet>(precision));
  }
};
template <typename Scalar, bool Approximate>
struct functor_traits<fuzzy_constant_predicate<Scalar, Approximate>> {
  // ARMv7 NEON flushes subnormal float operands and results; scalar VFP does not.
  // Explicitly flushing would change exact-zero checks; normal operands can also have a subnormal difference.
  static constexpr bool PacketAccess =
      (std::is_same<Scalar, float>::value || std::is_same<Scalar, double>::value) &&
      !(EIGEN_ARCH_ARM && std::is_same<Scalar, float>::value) && packet_traits<Scalar>::HasAbs &&
      packet_traits<Scalar>::HasCmp &&
      (!Approximate ||
       (packet_traits<Scalar>::HasSub && packet_traits<Scalar>::HasMin && packet_traits<Scalar>::HasMul));
  // Grouped scalar compares may be vectorized onto the unit PacketAccess excludes.
  static constexpr bool GroupScalars = PacketAccess;
  // The approximate test saturates the vector units, so larger blocks barely speed up full scans but read further
  // past a deciding coefficient: 2 packets per exit is never slower than master's exit per packet.
  static constexpr int BlockPackets = Approximate ? 2 : 8;
};

// A predicate's scalar test alone, tested one coefficient at a time.
template <typename Predicate>
struct scalar_predicate {
  Predicate pred;
  template <typename Scalar>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const Scalar& x) const {
    return pred(x);
  }
};
template <typename Predicate>
struct functor_traits<scalar_predicate<Predicate>> {
  static constexpr bool PacketAccess = false;
};

// Searches for a coefficient on which the predicate is Target: any_of searches for true, all_of for false. Packets
// combine truth masks and test the combined mask; scalars search for a true test_scalar by OR, which vectorizes
// better than an AND of test results.
struct packet_predicate {
  template <typename Predicate, typename Packet>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet apply(const Predicate& pred, const Packet& x) {
    return pred.packetOp(x);
  }
};
template <bool Target>
struct predicate_search;
template <>
struct predicate_search<true> : packet_predicate {
  template <typename Packet>
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet combine(const Packet& a, const Packet& b) {
    return por(a, b);
  }
  template <typename Packet>
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool found(const Packet& m) {
    return predux_any(m);
  }
  template <typename Predicate, typename Scalar>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool test_scalar(const Predicate& pred, const Scalar& x) {
    return pred(x);
  }
};
template <>
struct predicate_search<false> : packet_predicate {
  template <typename Packet>
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet combine(const Packet& a, const Packet& b) {
    return pand(a, b);
  }
  // pandnot rather than pnot, which some targets implement only bytewise.
  template <typename Packet>
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool found(const Packet& m) {
    return predux_any(pandnot(ptrue(m), m));
  }
  template <typename Predicate, typename Scalar>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool test_scalar(const Predicate& pred, const Scalar& x) {
    return !pred(x);
  }
};

// all_of(nonzero) searches for zero lanes, x == 0 being the exact complement of x != 0: a single compare per packet
// and no complement of the combined mask, which some compilers otherwise lower through narrowing and widening.
// Scalars still test the predicate: a coefficient is found where it is false.
struct zero_lane_search : predicate_search<true> {
  template <typename Predicate, typename Packet>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet apply(const Predicate&, const Packet& x) {
    return pcmp_eq(x, pzero(x));
  }
  template <typename Predicate, typename Scalar>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool test_scalar(const Predicate& pred, const Scalar& x) {
    return !pred(x);
  }
};
// any_of on bool lanes reduces them directly, treating every nonzero byte as true as the scalar reduction does.
struct bool_lane_search : predicate_search<true> {
  template <typename Predicate, typename Packet>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet apply(const Predicate&, const Packet& x) {
    return x;
  }
};
template <bool Target, typename Predicate>
struct predicate_search_for {
  using type = predicate_search<Target>;
};
template <typename Scalar>
struct predicate_search_for<false, nonzero_predicate<Scalar>> {
  using type = zero_lane_search;
};
template <>
struct predicate_search_for<true, nonzero_predicate<bool>> {
  using type = bool_lane_search;
};

// Coefficient access along one inner vector, or along the whole expression with linear access.
template <typename Evaluator, bool Linear = Evaluator::LinearAccess>
struct predicate_segment {
  using Scalar = typename Evaluator::Scalar;
  const Evaluator& eval;
  Index length;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE predicate_segment(const Evaluator& evaluator, Index, Index size)
      : eval(evaluator), length(size) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index size() const { return length; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename Evaluator::CoeffReturnType coeff(Index i) const {
    return eval.coeff(i);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packet(Index i) const {
    return eval.template packet<Packet>(i);
  }
};
template <typename Evaluator>
struct predicate_segment<Evaluator, false> {
  using Scalar = typename Evaluator::Scalar;
  const Evaluator& eval;
  Index outer;
  Index length;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE predicate_segment(const Evaluator& evaluator, Index j, Index size)
      : eval(evaluator), outer(j), length(size) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index size() const { return length; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename Evaluator::CoeffReturnType coeff(Index i) const {
    return Evaluator::IsRowMajor ? eval.coeff(outer, i) : eval.coeff(i, outer);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packet(Index i) const {
    return Evaluator::IsRowMajor ? eval.template packet<Packet>(outer, i) : eval.template packet<Packet>(i, outer);
  }
};

// Whether a predicate's scalar tests may be grouped; predicates that do not say are tested one at a time.
template <typename Predicate, typename = void>
struct predicate_groups_scalars : std::false_type {};
template <typename Predicate>
struct predicate_groups_scalars<Predicate, void_t<decltype(functor_traits<Predicate>::GroupScalars)>>
    : bool_constant<static_cast<bool>(functor_traits<Predicate>::GroupScalars)> {};

// Packets tested between exits on the long path; predicates that do not say test blocks of 8.
template <typename Predicate, typename = void>
struct predicate_block_packets : std::integral_constant<int, 8> {};
template <typename Predicate>
struct predicate_block_packets<Predicate, void_t<decltype(functor_traits<Predicate>::BlockPackets)>>
    : std::integral_constant<int, functor_traits<Predicate>::BlockPackets> {};

// Scalar searches OR their tests without a branch per test; spelling a | b on the calls directly draws clang's
// -Wbitwise-instead-of-logical.
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool predicate_or(bool a, bool b) { return a | b; }

// At most three coefficients from i on, as straight-line tests.
template <typename Search, typename Segment, typename Predicate>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool predicate_search_upto3(const Segment& seg, Index i, Predicate pred) {
  const Index size = seg.size();
  if (i == size) return false;
  bool b = Search::test_scalar(pred, seg.coeff(i));
  if (i + 1 < size) b = predicate_or(b, Search::test_scalar(pred, seg.coeff(i + 1)));
  if (i + 2 < size) b = predicate_or(b, Search::test_scalar(pred, seg.coeff(i + 2)));
  return b;
}

template <typename Search, typename Segment, typename Predicate>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool predicate_search_scalar(const Segment& seg, Index i, Predicate pred) {
  const Index size = seg.size();
  for (; i + 3 < size; i += 4) {
    bool b = predicate_or(
        predicate_or(Search::test_scalar(pred, seg.coeff(i)), Search::test_scalar(pred, seg.coeff(i + 1))),
        predicate_or(Search::test_scalar(pred, seg.coeff(i + 2)), Search::test_scalar(pred, seg.coeff(i + 3))));
    if EIGEN_PREDICT_FALSE (b) return true;
  }
  return predicate_search_upto3<Search>(seg, i, pred);
}

// One coefficient at a time with an exit after each, a shape compilers do not vectorize.
template <typename Search, typename Segment, typename Predicate>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool predicate_search_sequential(const Segment& seg, Index i, Predicate pred) {
  for (const Index size = seg.size(); i < size; ++i)
    if (Search::test_scalar(pred, seg.coeff(i))) return true;
  return false;
}

// Scalar tests from i on: grouped only where the predicate allows it, since grouped scalar tests may be vectorized
// onto the unit a predicate excludes (ARMv7 NEON, which flushes subnormals, turns grouped float compares into vceq).
template <typename Search, typename Segment, typename Predicate>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool predicate_search_tail(const Segment& seg, Index i, Predicate pred) {
  if (predicate_groups_scalars<Predicate>::value) return predicate_search_scalar<Search>(seg, i, pred);
  return predicate_search_sequential<Search>(seg, i, pred);
}

// Scalar blocks OR their tests into an unsigned integer as wide as the scalar. gcc vectorizes that reduction but not an
// OR of bools, and a wider accumulator widens every test.
template <typename Scalar, typename Unsigned = typename numext::get_integer_by_size<sizeof(Scalar)>::unsigned_type>
struct predicate_block_accumulator {
  using type = Unsigned;
};
template <typename Scalar>
struct predicate_block_accumulator<Scalar, void> {
  using type = bool;
};

template <typename Search, typename Packet, bool Vectorize>
struct predicate_search_segment {
  static constexpr Index ShortSize = 0;
  // Without packets, blocks of 64 tests without an exit, which compilers vectorize, and one exit check per block;
  // starting each block from the identity keeps its trip count regular once inlined. Only where the predicate allows
  // grouped tests, as for predicate_search_tail.
  template <typename Segment, typename Predicate>
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool run(const Segment& seg, Predicate pred) {
    if (!predicate_groups_scalars<Predicate>::value) return predicate_search_sequential<Search>(seg, 0, pred);
    const Index size = seg.size();
    // The shortest segments first, so they skip the block and group guards.
    if (EIGEN_PREDICT_FALSE(size < 4)) return predicate_search_upto3<Search>(seg, 0, pred);
    Index i = 0;
    using Acc = typename predicate_block_accumulator<typename Segment::Scalar>::type;
    for (; i + 64 <= size; i += 64) {
      Acc b = 0;
      for (Index k = 0; k < 64; ++k) b |= Acc(Search::test_scalar(pred, seg.coeff(i + k)));
      if EIGEN_PREDICT_FALSE (b != Acc(0)) return true;
    }
    return predicate_search_scalar<Search>(seg, i, pred);
  }
  template <typename Segment, typename Predicate>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool run_short(const Segment& seg, Predicate pred) {
    return run(seg, pred);
  }
};
template <typename Search, typename Packet>
struct predicate_search_segment<Search, Packet, true> {
  // One helper per packet, taking its operands by reference and returning the mask by value: if the early inliner
  // leaves it as a call, the caller needs no addressable packet temporaries.
  template <typename Segment, typename Predicate>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet test(const Segment& seg, Index i, const Predicate& pred) {
    return Search::apply(pred, seg.template packet<Packet>(i));
  }
  static constexpr Index PacketSize = unpacket_traits<Packet>::size;
  // Segments shorter than ShortSize take run_short.
  static constexpr Index ShortSize = 4 * PacketSize;

  // Up to three packets (plus a ragged tail), the common short call: test them directly.
  template <typename Segment, typename Predicate>
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool run_short(const Segment& seg, Predicate pred) {
    const Index size = seg.size();
    Index i = 0;
    if (PacketSize <= size) {
      if (size < 2 * PacketSize) {
        if (Search::found(test(seg, 0, pred))) return true;
        i = PacketSize;
      } else if (size < 3 * PacketSize) {
        if (Search::found(Search::combine(test(seg, 0, pred), test(seg, PacketSize, pred)))) return true;
        i = 2 * PacketSize;
      } else {
        if (Search::found(Search::combine(Search::combine(test(seg, 0, pred), test(seg, PacketSize, pred)),
                                          test(seg, 2 * PacketSize, pred))))
          return true;
        i = 3 * PacketSize;
      }
      if (EIGEN_PREDICT_TRUE(i == size)) return false;
    } else if (size < 4 && predicate_groups_scalars<Predicate>::value) {
      // Shorter than one packet and than a group of four: only these sizes reach this check.
      return predicate_search_upto3<Search>(seg, 0, pred);
    }
    return predicate_search_tail<Search>(seg, i, pred);
  }

  template <int Block, typename Segment, typename Predicate>
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool run_blocks(const Segment& seg, Predicate pred) {
    const Index size = seg.size();
    Index i = 0;
    for (; i + Block * PacketSize <= size; i += Block * PacketSize) {
      Packet m = test(seg, i, pred);
      for (int k = 1; k < Block; ++k) m = Search::combine(m, test(seg, i + k * PacketSize, pred));
      if EIGEN_PREDICT_FALSE (Search::found(m)) return true;
    }
    if (Block > 1 && i + PacketSize <= size) {
      Packet m = test(seg, i, pred);
      for (i += PacketSize; i + PacketSize <= size; i += PacketSize) m = Search::combine(m, test(seg, i, pred));
      if (Search::found(m)) return true;
    }
    if (EIGEN_PREDICT_TRUE(i == size)) return false;
    return predicate_search_tail<Search>(seg, i, pred);
  }

  template <typename Segment, typename Predicate>
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool run(const Segment& seg, Predicate pred) {
    const Index size = seg.size();
    if (size < ShortSize) return run_short(seg, pred);
    EIGEN_IF_CONSTEXPR (predicate_block_packets<Predicate>::value < 8) {
      return run_blocks<predicate_block_packets<Predicate>::value>(seg, pred);
    }
    Index i = 0;
    // Longer segments: the block loop is laid out off the fall-through path.
    if (EIGEN_PREDICT_FALSE(8 * PacketSize <= size)) {
      do {
        Packet m = Search::combine(
            Search::combine(Search::combine(test(seg, i + 0 * PacketSize, pred), test(seg, i + 1 * PacketSize, pred)),
                            Search::combine(test(seg, i + 2 * PacketSize, pred), test(seg, i + 3 * PacketSize, pred))),
            Search::combine(Search::combine(test(seg, i + 4 * PacketSize, pred), test(seg, i + 5 * PacketSize, pred)),
                            Search::combine(test(seg, i + 6 * PacketSize, pred), test(seg, i + 7 * PacketSize, pred))));
        if EIGEN_PREDICT_FALSE (Search::found(m)) return true;
        i += 8 * PacketSize;
      } while (i + 8 * PacketSize <= size);
    }
    if (i + 4 * PacketSize <= size) {
      Packet m =
          Search::combine(Search::combine(test(seg, i + 0 * PacketSize, pred), test(seg, i + 1 * PacketSize, pred)),
                          Search::combine(test(seg, i + 2 * PacketSize, pred), test(seg, i + 3 * PacketSize, pred)));
      if (Search::found(m)) return true;
      i += 4 * PacketSize;
    }
    if (i + PacketSize <= size) {
      Packet m = test(seg, i, pred);
      for (i += PacketSize; i + PacketSize <= size; i += PacketSize) m = Search::combine(m, test(seg, i, pred));
      if (Search::found(m)) return true;
    }
    if (EIGEN_PREDICT_TRUE(i == size)) return false;
    return predicate_search_tail<Search>(seg, i, pred);
  }
};

template <typename Derived, typename Predicate>
struct predicate_reduction {
  using Evaluator = visitor_evaluator<Derived>;
  using Scalar = typename Derived::Scalar;
  static constexpr bool LinearAccess = Evaluator::LinearAccess;
  // A segment's compile-time length, so a short fixed-size segment uses a packet that fits it, as assignment does.
  static constexpr int SegmentSizeAtCompileTime =
      LinearAccess ? int(Derived::SizeAtCompileTime)
                   : int(Evaluator::IsRowMajor ? Derived::ColsAtCompileTime : Derived::RowsAtCompileTime);
  using Packet = typename find_largest_packet<Scalar, SegmentSizeAtCompileTime>::type;
  static constexpr bool Vectorize = Evaluator::PacketAccess && packet_traits<Scalar>::Vectorizable &&
                                    (unpacket_traits<Packet>::size > 1) &&
                                    static_cast<bool>(functor_traits<Predicate>::PacketAccess);
  using Segment = predicate_segment<Evaluator>;

  template <bool Target>
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool search(const Derived& xpr, Predicate pred) {
    using impl = predicate_search_segment<typename predicate_search_for<Target, Predicate>::type, Packet, Vectorize>;
    Evaluator eval(xpr);
    EIGEN_IF_CONSTEXPR (LinearAccess) {
      return impl::run(Segment(eval, 0, eval.size()), pred);
    } else {
      // The inner size is read once: through the evaluator's reference to the expression, compilers reload it for
      // every inner vector.
      const Index innerSize = Evaluator::IsRowMajor ? eval.cols() : eval.rows();
      const Index outerSize = Evaluator::IsRowMajor ? eval.rows() : eval.cols();
      // Choosing the segment path once keeps the length dispatch out of the loop over short inner vectors.
      if (innerSize < impl::ShortSize) {
        for (Index j = 0; j < outerSize; ++j)
          if (impl::run_short(Segment(eval, j, innerSize), pred)) return true;
      } else {
        for (Index j = 0; j < outerSize; ++j)
          if (impl::run(Segment(eval, j, innerSize), pred)) return true;
      }
      return false;
    }
  }
};

/** \internal \returns true if \a pred holds for every coefficient of \a xpr. */
template <typename Derived, typename Predicate>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool all_of(const Derived& xpr, const Predicate& pred) {
  return !predicate_reduction<Derived, Predicate>::template search<false>(xpr, pred);
}

/** \internal \returns true if \a pred holds for some coefficient of \a xpr. */
template <typename Derived, typename Predicate>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool any_of(const Derived& xpr, const Predicate& pred) {
  return predicate_reduction<Derived, Predicate>::template search<true>(xpr, pred);
}

// The packet test of fuzzy_constant_predicate holds only where packetOpApplies; elsewhere it tests scalars.
template <typename Derived, typename Predicate>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool fuzzy_constant_all_of(const Derived& xpr, const Predicate& pred,
                                                                 std::true_type) {
  if (EIGEN_PREDICT_FALSE(!pred.packetOpApplies())) return all_of(xpr, scalar_predicate<Predicate>{pred});
  return all_of(xpr, pred);
}
template <typename Derived, typename Predicate>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool fuzzy_constant_all_of(const Derived& xpr, const Predicate& pred,
                                                                 std::false_type) {
  return all_of(xpr, pred);
}

template <typename Derived, bool AlwaysTrue = NumTraits<typename traits<Derived>::Scalar>::IsInteger>
struct all_finite_impl {
  static EIGEN_DEVICE_FUNC inline bool run(const Derived& /*derived*/) { return true; }
};
#if !defined(__FINITE_MATH_ONLY__) || !(__FINITE_MATH_ONLY__)
template <typename Derived>
struct all_finite_impl<Derived, false> {
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool run(const Derived& derived) {
    return all_of(derived, isfinite_predicate<typename traits<Derived>::Scalar>());
  }
};
#endif

}  // end namespace internal

/** \returns true if all coefficients are true
 *
 * Example: \include MatrixBase_all.cpp
 * Output: \verbinclude MatrixBase_all.out
 *
 * \sa any(), Cwise::operator<()
 */
template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool DenseBase<Derived>::all() const {
  return internal::all_of(derived(), internal::nonzero_predicate<Scalar>());
}

/** \returns true if at least one coefficient is true
 *
 * \sa all()
 */
template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool DenseBase<Derived>::any() const {
  return internal::any_of(derived(), internal::nonzero_predicate<Scalar>());
}

template <typename Derived>
EIGEN_DEVICE_FUNC inline bool DenseBase<Derived>::hasNaN() const {
  return derived().cwiseTypedNotEqual(derived()).any();
}

/** \returns true if \c *this contains only finite numbers, i.e., no NaN and no +/-INF values.
 *
 * \sa hasNaN()
 */
template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool DenseBase<Derived>::allFinite() const {
  return internal::all_finite_impl<Derived>::run(derived());
}

/** \returns true if all coefficients in this matrix are approximately equal to \a val, to within precision \a prec */
template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool DenseBase<Derived>::isApproxToConstant(const Scalar& val,
                                                                                  const RealScalar& prec) const {
  using Predicate = internal::fuzzy_constant_predicate<Scalar, true>;
  return internal::fuzzy_constant_all_of(
      derived(), Predicate{val, prec},
      std::integral_constant<bool, internal::functor_traits<Predicate>::PacketAccess>());
}

/** \returns true if *this is approximately equal to the zero matrix,
 *          within the precision given by \a prec.
 *
 * Example: \include MatrixBase_isZero.cpp
 * Output: \verbinclude MatrixBase_isZero.out
 *
 * \sa class CwiseNullaryOp, Zero()
 */
template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool DenseBase<Derived>::isZero(const RealScalar& prec) const {
  return internal::all_of(derived(), internal::fuzzy_constant_predicate<Scalar, false>{Scalar(0), prec});
}

}  // end namespace Eigen

#endif  // EIGEN_PREDICATOR_H
