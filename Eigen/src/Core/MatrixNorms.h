// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_MATRIX_NORMS_H
#define EIGEN_MATRIX_NORMS_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

// *_abs_sum sums magnitudes of stored coefficients; *_l1_norm takes the maximum
// absolute column sum of the represented matrix, including any implicit entries.

// Sum |a_ij| over i <= j + ExtraDiagonals (Upper), or j <= i + ExtraDiagonals (Lower).
template <unsigned int UpLo, int ExtraDiagonals, typename Derived>
EIGEN_DEVICE_FUNC typename Derived::RealScalar triangular_band_abs_sum(const MatrixBase<Derived>& matrix) {
  static_assert(UpLo == Upper || UpLo == Lower, "UpLo must be Upper or Lower");
  static_assert(ExtraDiagonals == 0 || ExtraDiagonals == 1, "Expected a triangular or Hessenberg envelope");
  using RealScalar = typename Derived::RealScalar;
  RealScalar result(0);
  if (matrix.rows() == 0 || matrix.cols() == 0) return result;

  constexpr bool Leading = (UpLo == Upper) != bool(Derived::IsRowMajor);
  const Index inner = matrix.innerSize();
  // Inner-vector slices preserve packet access for contiguous storage and also work for strided expressions.
  for (Index j = 0; j < matrix.outerSize(); ++j) {
    const Index start = Leading ? Index(0) : numext::mini(inner, numext::maxi(Index(0), j - ExtraDiagonals));
    const Index end = Leading ? numext::mini(inner, j + ExtraDiagonals + 1) : inner;
    result += matrix.derived().innerVector(j).segment(start, end - start).template lpNorm<1>();
  }
  return result;
}

// Coefficient-wise l1 norm of the stored upper/lower triangle, including the diagonal.
// Entries outside the selected triangle are never read; rectangular and empty expressions are supported.
template <unsigned int UpLo, typename Derived>
EIGEN_DEVICE_FUNC typename Derived::RealScalar triangular_abs_sum(const MatrixBase<Derived>& matrix) {
  return triangular_band_abs_sum<UpLo, 0>(matrix);
}

// Coefficient-wise l1 norm of the upper/lower Hessenberg envelope, including its one extra off-diagonal.
template <unsigned int UpLo, typename Derived>
EIGEN_DEVICE_FUNC typename Derived::RealScalar hessenberg_abs_sum(const MatrixBase<Derived>& matrix) {
  return triangular_band_abs_sum<UpLo, 1>(matrix);
}

// Column step of the self-adjoint 1-norm, on two columns sharing a range of rows: sums[i] +=
// |m(i, j0)| + |m(i, j1)|, and each column's own sum of those rows goes to sums[j0] and sums[j1].
// Walking two columns at once halves the traffic on sums, and one packet pass does everything, so
// short columns pay no per-expression setup.
//
// Real scalars use pabs. Complex ones have no packet abs, so |z| = sqrt(re^2 + im^2) is computed
// on the real lanes of the complex packet. That leaves |z|^2 in both lanes of each slot, so one
// square root serves both columns: even lanes from j0 and odd lanes from j1 give |u0| |v0| |u1|
// |v1| ..., whose sum with its flip is the update of sums, and whose reduction as a complex packet
// is (sum |u|, sum |v|). The accumulator has the matrix's scalar type and only its real parts are
// read, so what lands in the imaginary lanes is harmless. Squaring overflows above sqrt(max) and
// loses precision below sqrt(min), so the pass also records the largest component it has seen and
// the caller recomputes the norm through numext::abs when that is out of range. The record is
// exact and the range test finite, so neither depends on infinities surviving fast-math.
template <typename Scalar_>
struct selfadjoint_l1norm_real_lanes {
  using Scalar = Scalar_;
  using Real = Scalar_;
  using Packet = typename packet_traits<Scalar>::type;
  using RPacket = Packet;
  static constexpr Index PacketSize = unpacket_traits<Packet>::size;
  // Up to this size the per-column form (mirrored term read as a row) beats the column pass, whose
  // accumulator costs more to set up than these columns cost to read.
  static constexpr Index PerColumnUpTo = 4;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE RPacket lanes(const Packet& p) { return p; }
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Real abs(const Scalar& x) { return numext::abs(x); }
  // Nothing to record: pabs is exact.
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Real component(const Scalar&) { return Real(0); }
  static EIGEN_DEVICE_FUNC bool inRange(Real, Index) { return true; }

  // The running sums of a pass over two columns.
  struct Pass {
    RPacket acc0 = pzero(RPacket());
    RPacket acc1 = pzero(RPacket());
    template <typename SumsEvaluator>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void step(const RPacket& u, const RPacket& v, SumsEvaluator& s, Index i) {
      RPacket a = pabs(u);
      RPacket b = pabs(v);
      acc0 = padd(acc0, a);
      acc1 = padd(acc1, b);
      s.template writePacket<Unaligned>(i, padd(s.template packet<Unaligned, Packet>(i), padd(a, b)));
    }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Real sum0() const { return predux(acc0); }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Real sum1() const { return predux(acc1); }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Real peak() const { return Real(0); }
  };
};

template <typename T>
struct selfadjoint_l1norm_complex_lanes {
  using Scalar = std::complex<T>;
  using Real = T;
  using Packet = typename packet_traits<Scalar>::type;
  using RPacket = typename unpacket_traits<Packet>::as_real;
  static constexpr Index PacketSize = unpacket_traits<Packet>::size;
  static constexpr Index PerColumnUpTo = 0;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE RPacket lanes(const Packet& p) { return p.v; }
  // Same formula as the packets, for the diagonal and the tails: hypot costs more than the packets
  // spend on the rest of a short column.
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Real abs(const Scalar& z) { return numext::sqrt(numext::abs2(z)); }
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Real component(const Scalar& z) {
    return numext::maxi(numext::abs(numext::real(z)), numext::abs(numext::imag(z)));
  }
  // Components this large overflow when squared, and below the lower bound the squares lose
  // precision the sum of n of them cannot hide.
  static EIGEN_DEVICE_FUNC bool inRange(Real peak, Index n) {
    Real tiny = Real(n) * numext::sqrt((std::numeric_limits<Real>::min)()) / NumTraits<Real>::epsilon();
    Real huge = numext::sqrt(NumTraits<Real>::highest()) / Real(2);
    return peak > tiny && peak < huge;
  }

  struct Pass {
    RPacket acc = pzero(RPacket());    // |u| in the even lanes, |v| in the odd ones
    RPacket peak_ = pzero(RPacket());  // the largest component seen
    template <typename SumsEvaluator>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void step(const RPacket& u, const RPacket& v, SumsEvaluator& s, Index i) {
      peak_ = pmax(peak_, pmax(pabs(u), pabs(v)));
      RPacket r = psqrt(pselect(peven_mask(u), abs2(u), abs2(v)));  // |u0| |v0| |u1| |v1| ...
      acc = padd(acc, r);
      s.template writePacket<Unaligned>(i,
                                        Packet(padd(lanes(s.template packet<Unaligned, Packet>(i)), padd(r, flip(r)))));
    }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Real sum0() const { return numext::real(predux(Packet(acc))); }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Real sum1() const { return numext::imag(predux(Packet(acc))); }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Real peak() const { return predux_max(peak_); }
  };

 private:
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE RPacket flip(const RPacket& r) { return pcplxflip(Packet(r)).v; }
  // |z|^2 in both lanes of its slot.
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE RPacket abs2(const RPacket& v) {
    RPacket s = pmul(v, v);
    return padd(s, flip(s));
  }
};

// The pass over two columns: packets over the shared rows, coefficients for the tail. An instance
// remembers the largest component it has seen, for inRange().
template <typename Lanes>
struct selfadjoint_l1norm_packet_impl : Lanes {
  using Lanes::PacketSize;
  using typename Lanes::Packet;
  using typename Lanes::Real;
  using typename Lanes::RPacket;
  using typename Lanes::Scalar;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Real abs(const Scalar& x) {
    m_peak = numext::maxi(m_peak, Lanes::component(x));
    return Lanes::abs(x);
  }
  EIGEN_DEVICE_FUNC bool inRange(Index n) const { return Lanes::inRange(m_peak, n); }

  template <typename SumsDerived, typename Derived>
  EIGEN_DEVICE_FUNC void accumulate(DenseBase<SumsDerived>& sums, const DenseBase<Derived>& m, Index j0, Index j1,
                                    Index begin, Index end) {
    accumulateCast(sums, j0, j1, begin, m.col(j0).segment(begin, end - begin).template cast<Scalar>(),
                   m.col(j1).segment(begin, end - begin).template cast<Scalar>());
  }

 private:
  Real m_peak = Real(0);

  template <typename SumsDerived, typename Derived0, typename Derived1>
  EIGEN_DEVICE_FUNC void accumulateCast(DenseBase<SumsDerived>& sums, Index j0, Index j1, Index begin,
                                        const DenseBase<Derived0>& x0, const DenseBase<Derived1>& x1) {
    using SumsEvaluator = evaluator<SumsDerived>;
    using Evaluator0 = evaluator<Derived0>;
    using Evaluator1 = evaluator<Derived1>;
    constexpr int Needed = PacketAccessBit | LinearAccessBit;
    constexpr bool Vectorize = (SumsEvaluator::Flags & Needed) == Needed && (Evaluator0::Flags & Needed) == Needed &&
                               (Evaluator1::Flags & Needed) == Needed;
    SumsEvaluator s(sums.derived());
    accumulate(s, j0, j1, begin, Evaluator0(x0.derived()), Evaluator1(x1.derived()), x0.size(),
               bool_constant<Vectorize>());
  }
  template <typename Evaluator>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE RPacket load(const Evaluator& x, Index i) {
    return Lanes::lanes(x.template packet<Unaligned, Packet>(i));
  }
  template <typename SumsEvaluator, typename Evaluator0, typename Evaluator1>
  EIGEN_DEVICE_FUNC void accumulate(SumsEvaluator& s, Index j0, Index j1, Index begin, const Evaluator0& x0,
                                    const Evaluator1& x1, Index from, Index to) {
    Real sum0 = Real(0);
    Real sum1 = Real(0);
    for (Index i = from; i < to; ++i) {
      Real a = abs(x0.coeff(i));
      Real b = abs(x1.coeff(i));
      s.coeffRef(begin + i) += a + b;
      sum0 += a;
      sum1 += b;
    }
    s.coeffRef(j0) += sum0;
    s.coeffRef(j1) += sum1;
  }
  template <typename SumsEvaluator, typename Evaluator0, typename Evaluator1>
  EIGEN_DEVICE_FUNC void accumulate(SumsEvaluator& s, Index j0, Index j1, Index begin, const Evaluator0& x0,
                                    const Evaluator1& x1, Index n, std::false_type) {
    accumulate(s, j0, j1, begin, x0, x1, Index(0), n);
  }
  template <typename SumsEvaluator, typename Evaluator0, typename Evaluator1>
  EIGEN_DEVICE_FUNC void accumulate(SumsEvaluator& s, Index j0, Index j1, Index begin, const Evaluator0& x0,
                                    const Evaluator1& x1, Index n, std::true_type) {
    if (n < PacketSize) return accumulate(s, j0, j1, begin, x0, x1, Index(0), n);
    typename Lanes::Pass pass;
    Index i = 0;
    for (; i + PacketSize <= n; i += PacketSize) pass.step(load(x0, i), load(x1, i), s, begin + i);
    accumulate(s, j0, j1, begin, x0, x1, i, n);
    s.coeffRef(j0) += pass.sum0();
    s.coeffRef(j1) += pass.sum1();
    m_peak = numext::maxi(m_peak, pass.peak());
  }
};

// Coefficient fallback: custom complex types, or complex packets without a plain real view.
template <typename Scalar_, typename Enable = void>
struct selfadjoint_l1norm_impl {
  using Scalar = Scalar_;
  using Real = typename NumTraits<Scalar>::Real;
  static constexpr Index PerColumnUpTo = 16;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Real abs(const Scalar& x) const { return numext::abs(x); }
  EIGEN_DEVICE_FUNC bool inRange(Index) const { return true; }
  template <typename SumsDerived, typename Derived>
  EIGEN_DEVICE_FUNC void accumulate(DenseBase<SumsDerived>& sums, const DenseBase<Derived>& m, Index j0, Index j1,
                                    Index begin, Index end) const {
    Real sum0 = Real(0);
    Real sum1 = Real(0);
    for (Index i = begin; i < end; ++i) {
      Real a = numext::abs(m.coeff(i, j0));
      Real b = numext::abs(m.coeff(i, j1));
      sums.coeffRef(i) += Scalar(a + b);
      sum0 += a;
      sum1 += b;
    }
    sums.coeffRef(j0) += Scalar(sum0);
    sums.coeffRef(j1) += Scalar(sum1);
  }
};
// half and bfloat16 accumulate in float, as stableNorm does.
template <typename Scalar>
struct selfadjoint_l1norm_impl<Scalar, std::enable_if_t<!NumTraits<Scalar>::IsComplex>>
    : selfadjoint_l1norm_packet_impl<selfadjoint_l1norm_real_lanes<typename stable_norm_accumulator<Scalar>::type>> {};
// The real view must lay the components out one per lane: Z13 stores four floats in two double
// packets, which the lane masks do not describe.
template <typename Packet, typename Enable = void>
struct selfadjoint_l1norm_plain_real_view : std::false_type {};
template <typename Packet>
struct selfadjoint_l1norm_plain_real_view<Packet, void_t<typename unpacket_traits<Packet>::as_real>>
    : bool_constant<sizeof(typename unpacket_traits<Packet>::as_real) ==
                    unpacket_traits<Packet>::size * sizeof(typename unpacket_traits<Packet>::type)> {};
template <typename T>
struct selfadjoint_l1norm_impl<
    std::complex<T>,
    std::enable_if_t<selfadjoint_l1norm_plain_real_view<typename packet_traits<std::complex<T>>::type>::value>>
    : selfadjoint_l1norm_packet_impl<selfadjoint_l1norm_complex_lanes<T>> {};

template <typename MatrixType, unsigned int UpLo>
class selfadjoint_l1norm {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;

 public:
  EIGEN_DEVICE_FUNC explicit selfadjoint_l1norm(const MatrixType& matrix) : m_matrix(matrix) {}

  EIGEN_DEVICE_FUNC RealScalar run() const {
#ifdef EIGEN_GPU_COMPILE_PHASE
    // No per-thread accumulator on a device.
    return l1NormPerColumn();
#else
    if (m_matrix.rows() <= L1NormImpl::PerColumnUpTo) return l1NormPerColumn();
    // The stored triangle of a row-major matrix is the complementary triangle of its column-major
    // transpose, which has the same norm.
    EIGEN_IF_CONSTEXPR (bool(MatrixType::IsRowMajor)) {
      return l1NormStreaming<(UpLo == Lower ? Upper : Lower)>(m_matrix.transpose());
    } else {
      return l1NormStreaming<UpLo>(m_matrix);
    }
#endif
  }

 private:
  using L1NormImpl = internal::selfadjoint_l1norm_impl<Scalar>;
  // float for half and bfloat16, Scalar otherwise.
  using L1NormScalar = typename L1NormImpl::Scalar;
  using L1NormAccumulator = typename L1NormImpl::Real;

  // Each column is read once, top to bottom, two at a time: |a_ij| goes to column j's sum and, as
  // the mirrored a_ji, to sums[i]. Lower walks the columns forward and Upper backward so that
  // sums[j] is complete when column j is reached. Of a pair (j0, j1) only j0's element in row j1
  // lies outside the rows the two share.
  template <int Mode, typename Mat>
  RealScalar l1NormStreaming(const Mat& m) const {
    const Index n = m.rows();
    // The accumulator lives in the object for bounded sizes and on the stack otherwise, so that
    // neither fixed-size nor preallocated dynamic-size decompositions allocate.
    internal::gemv_static_vector_if<L1NormScalar, Mat::RowsAtCompileTime, Mat::MaxRowsAtCompileTime, true> static_sums;
    ei_declare_aligned_stack_constructed_variable(L1NormScalar, sums_data, n, static_sums.data());
    Map<Matrix<L1NormScalar, Dynamic, 1>> sums(sums_data, n);
    sums.setZero();
    L1NormImpl impl;
    L1NormAccumulator norm = L1NormAccumulator(0);
    Index k = 0;
    for (; k + 1 < n; k += 2) {
      Index j0 = Mode == Lower ? k : n - 1 - k;
      Index j1 = Mode == Lower ? j0 + 1 : j0 - 1;
      Index rowBegin = Mode == Lower ? j1 + 1 : 0;
      Index rowEnd = Mode == Lower ? n : j1;
      impl.accumulate(sums, m, j0, j1, rowBegin, rowEnd);
      // The element of j0 in row j1 lies outside the shared rows: it counts for both columns.
      L1NormAccumulator boundary = impl.abs(m.coeff(j1, j0));
      // Totals are materialized so that maxi compares two accumulators (an integer sum promotes,
      // an autodiff sum is an expression).
      L1NormAccumulator col0 = numext::real(sums.coeff(j0)) + impl.abs(m.coeff(j0, j0)) + boundary;
      L1NormAccumulator col1 = numext::real(sums.coeff(j1)) + impl.abs(m.coeff(j1, j1)) + boundary;
      norm = numext::maxi(norm, col0);
      norm = numext::maxi(norm, col1);
    }
    if (k < n) {
      Index j = Mode == Lower ? k : 0;
      L1NormAccumulator col = numext::real(sums.coeff(j)) + impl.abs(m.coeff(j, j));
      norm = numext::maxi(norm, col);
    }
    return impl.inRange(n) ? RealScalar(norm) : l1NormPerColumn();
  }

  // One column at a time, the mirrored term read as a row; no workspace.
  EIGEN_DEVICE_FUNC RealScalar l1NormPerColumn() const {
    L1NormAccumulator norm = L1NormAccumulator(0);
    const Index n = m_matrix.rows();
    for (Index col = 0; col < n; ++col) {
      L1NormAccumulator abs_col_sum;
      EIGEN_IF_CONSTEXPR (UpLo == Lower) {
        abs_col_sum = m_matrix.col(col).tail(n - col).template cast<L1NormScalar>().template lpNorm<1>() +
                      m_matrix.row(col).head(col).template cast<L1NormScalar>().template lpNorm<1>();
      } else {
        abs_col_sum = m_matrix.col(col).head(col).template cast<L1NormScalar>().template lpNorm<1>() +
                      m_matrix.row(col).tail(n - col).template cast<L1NormScalar>().template lpNorm<1>();
      }
      norm = numext::maxi(norm, abs_col_sum);
    }
    return RealScalar(norm);
  }

  const MatrixType& m_matrix;
};

// Matrix 1-norm of the implicit full self-adjoint matrix; only the stored triangle is read.
template <unsigned int UpLo, typename Derived>
EIGEN_DEVICE_FUNC typename Derived::RealScalar selfadjoint_l1_norm(const MatrixBase<Derived>& matrix) {
  static_assert(UpLo == Upper || UpLo == Lower, "UpLo must be Upper or Lower");
  eigen_assert(matrix.rows() == matrix.cols());
  return selfadjoint_l1norm<Derived, UpLo>(matrix.derived()).run();
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_MATRIX_NORMS_H
