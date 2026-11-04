// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TRIANGULAR_MATRIX_MATRIX_H
#define EIGEN_TRIANGULAR_MATRIX_MATRIX_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/* Optimized triangular matrix * matrix (_TRMM++) product built on top of
 * the general matrix matrix product.
 */
template <typename Scalar, typename Index, int Mode, bool LhsIsTriangular, int LhsStorageOrder, bool ConjugateLhs,
          int RhsStorageOrder, bool ConjugateRhs, int ResStorageOrder, int ResInnerStride, int Version = Specialized>
struct product_triangular_matrix_matrix;

template <typename Scalar, typename Index, int Mode, bool LhsIsTriangular, int LhsStorageOrder, bool ConjugateLhs,
          int RhsStorageOrder, bool ConjugateRhs, int ResInnerStride, int Version>
struct product_triangular_matrix_matrix<Scalar, Index, Mode, LhsIsTriangular, LhsStorageOrder, ConjugateLhs,
                                        RhsStorageOrder, ConjugateRhs, RowMajor, ResInnerStride, Version> {
  static EIGEN_STRONG_INLINE void run(Index rows, Index cols, Index depth, const Scalar* lhs, Index lhsStride,
                                      const Scalar* rhs, Index rhsStride, Scalar* res, Index resIncr, Index resStride,
                                      const Scalar& alpha, level3_blocking<Scalar, Scalar>& blocking) {
    product_triangular_matrix_matrix<Scalar, Index, (Mode & (UnitDiag | ZeroDiag)) | ((Mode & Upper) ? Lower : Upper),
                                     (!LhsIsTriangular), RhsStorageOrder == RowMajor ? ColMajor : RowMajor,
                                     ConjugateRhs, LhsStorageOrder == RowMajor ? ColMajor : RowMajor, ConjugateLhs,
                                     ColMajor, ResInnerStride>::run(cols, rows, depth, rhs, rhsStride, lhs, lhsStride,
                                                                    res, resIncr, resStride, alpha, blocking);
  }
};

// implements col-major += alpha * op(triangular) * op(general)
template <typename Scalar, typename Index, int Mode, int LhsStorageOrder, bool ConjugateLhs, int RhsStorageOrder,
          bool ConjugateRhs, int ResInnerStride, int Version>
struct product_triangular_matrix_matrix<Scalar, Index, Mode, true, LhsStorageOrder, ConjugateLhs, RhsStorageOrder,
                                        ConjugateRhs, ColMajor, ResInnerStride, Version> {
  using Traits = gebp_traits<Scalar, Scalar>;
  enum {
    SmallPanelWidth = 2 * plain_enum_max(Traits::mr, Traits::nr),
    // A multiple of mr, so that horizontal panels start on lhs micro-panel boundaries. It exceeds SmallPanelWidth only
    // if nr > mr, e.g. for NEON double (18 vs 16), whose vertical panels ran 1-3% slower at 18 on Cortex-A76.
    HorizontalPanelWidth = numext::div_ceil(int(SmallPanelWidth), int(Traits::mr)) * int(Traits::mr),
    IsLower = (Mode & Lower) == Lower,
    SetDiag = (Mode & (ZeroDiag | UnitDiag)) ? 0 : 1
  };

  static EIGEN_DONT_INLINE void run(Index _rows, Index _cols, Index _depth, const Scalar* lhs_, Index lhsStride,
                                    const Scalar* rhs_, Index rhsStride, Scalar* res, Index resIncr, Index resStride,
                                    const Scalar& alpha, level3_blocking<Scalar, Scalar>& blocking);
};

template <typename Scalar, typename Index, int Mode, int LhsStorageOrder, bool ConjugateLhs, int RhsStorageOrder,
          bool ConjugateRhs, int ResInnerStride, int Version>
EIGEN_DONT_INLINE void product_triangular_matrix_matrix<
    Scalar, Index, Mode, true, LhsStorageOrder, ConjugateLhs, RhsStorageOrder, ConjugateRhs, ColMajor, ResInnerStride,
    Version>::run(Index _rows, Index _cols, Index _depth, const Scalar* lhs_, Index lhsStride, const Scalar* rhs_,
                  Index rhsStride, Scalar* res_, Index resIncr, Index resStride, const Scalar& alpha,
                  level3_blocking<Scalar, Scalar>& blocking) {
  // strip zeros
  Index diagSize = (std::min)(_rows, _depth);
  Index rows = IsLower ? _rows : diagSize;
  Index depth = IsLower ? diagSize : _depth;
  Index cols = _cols;

  using LhsMapper = const_blas_data_mapper<Scalar, Index, LhsStorageOrder>;
  using RhsMapper = const_blas_data_mapper<Scalar, Index, RhsStorageOrder>;
  using ResMapper = blas_data_mapper<typename Traits::ResScalar, Index, ColMajor, Unaligned, ResInnerStride>;
  LhsMapper lhs(lhs_, lhsStride);
  RhsMapper rhs(rhs_, rhsStride);
  ResMapper res(res_, resStride, resIncr);

  Index kc = blocking.kc();                    // cache block size along the K direction
  Index mc = (std::min)(rows, blocking.mc());  // cache block size along the M direction
#ifdef EIGEN_VECTORIZE_SME
  // The SME kernel moves each output block through ZA on every call, which a panelWidth-deep update does not amortize.
  constexpr bool HorizontalBelowNr = sme_has_gebp_kernel<Scalar, Scalar>::value;
#else
  constexpr bool HorizontalBelowNr = false;
#endif
  // P horizontal panels update each output panel once instead of (P+1)/2 times on average, with the same packed
  // coefficients and flops. Below nr columns, the generic kernel's partial register blocks run faster over the short
  // depth of vertical panels.
  const bool horizontal =
      (HorizontalBelowNr || cols >= Traits::nr) && Index(HorizontalPanelWidth) <= (std::min)(kc, mc);
  // The small panel size must not be larger than blocking size.
  // Usually this should never be the case because SmallPanelWidth^2 is very small
  // compared to L2 cache size, but let's be safe:
  const Index panelWidth =
      horizontal ? Index(HorizontalPanelWidth) : (std::min)(Index(SmallPanelWidth), (std::min)(kc, mc));

  std::size_t sizeA = kc * mc;
  std::size_t sizeB = kc * cols;

  ei_declare_aligned_stack_constructed_variable(Scalar, blockA, sizeA, blocking.blockA());
  ei_declare_aligned_stack_constructed_variable(Scalar, blockB, sizeB, blocking.blockB());

  static constexpr int PanelCapacity =
      (EIGEN_STACK_ALLOCATION_LIMIT == 0 ||
       HorizontalPanelWidth * HorizontalPanelWidth * sizeof(Scalar) <= EIGEN_STACK_ALLOCATION_LIMIT)
          ? int(HorizontalPanelWidth)
          : Dynamic;
  Matrix<Scalar, PanelCapacity, PanelCapacity, LhsStorageOrder> triangularBuffer(HorizontalPanelWidth,
                                                                                 HorizontalPanelWidth);
  triangularBuffer.topLeftCorner(panelWidth, panelWidth).setZero();
  EIGEN_IF_CONSTEXPR ((Mode & ZeroDiag) == ZeroDiag)
    triangularBuffer.diagonal().setZero();
  else
    triangularBuffer.diagonal().setOnes();

  gebp_kernel<Scalar, Scalar, Index, ResMapper, Traits::mr, Traits::nr, ConjugateLhs, ConjugateRhs> gebp_kernel;
  // Panel mode; with stride == depth and offset 0 it packs exactly as plain mode does.
  gemm_pack_lhs<Scalar, Index, LhsMapper, Traits::mr, Traits::LhsProgress, typename Traits::LhsPacket4Packing,
                LhsStorageOrder, false, true>
      pack_lhs;
  gemm_pack_rhs<Scalar, Index, RhsMapper, Traits::nr, RhsStorageOrder> pack_rhs;

  for (Index k2 = IsLower ? depth : 0; IsLower ? k2 > 0 : k2 < depth; IsLower ? k2 -= kc : k2 += kc) {
    Index actual_kc = (std::min)(IsLower ? k2 : depth - k2, kc);
    Index actual_k2 = IsLower ? k2 - actual_kc : k2;

    // align blocks with the end of the triangular part for trapezoidal lhs
    EIGEN_IF_CONSTEXPR (!IsLower) {
      if ((k2 < rows) && (k2 + actual_kc > rows)) {
        actual_kc = rows - k2;
        k2 = k2 + actual_kc - kc;
      }
    }

    pack_rhs(blockB, rhs.getSubMapper(actual_k2, 0), actual_kc, cols);

    // the selected lhs's panel has to be split in three different parts:
    //  1 - the part which is zero => skip it
    //  2 - the diagonal block => special kernel
    //  3 - the dense panel below (lower case) or above (upper case) the diagonal block => GEPP

    // the block diagonal, if any:
    if (IsLower || actual_k2 < rows) {
      // The trick is to pack each micro triangular block while filling the opposite triangular part with zeros.
      // To this end we do an extra triangular copy to a small temporary buffer
      const auto packTriangle = [&](Scalar* dst, Index k1, Index width, Index stride, Index offset) {
        const Index start = actual_k2 + k1;
        for (Index k = 0; k < width; ++k) {
          EIGEN_IF_CONSTEXPR (SetDiag) triangularBuffer.coeffRef(k, k) = lhs(start + k, start + k);
          for (Index i = IsLower ? k + 1 : 0; IsLower ? i < width : i < k; ++i)
            triangularBuffer.coeffRef(i, k) = lhs(start + i, start + k);
        }
        pack_lhs(dst, LhsMapper(triangularBuffer.data(), triangularBuffer.outerStride()), width, width, stride, offset);
      };
      // Each group of horizontal panels that fits blockA is packed in the vertical order, whose tall column segments
      // stream from memory, at stride actual_kc.
      if (horizontal) {
        const Index groupRows = mc / panelWidth * panelWidth;
        for (Index g0 = 0; g0 < actual_kc; g0 += groupRows) {
          const Index g1 = (std::min)(actual_kc, g0 + groupRows);
          for (Index k1 = IsLower ? 0 : g0; k1 < (IsLower ? g1 : actual_kc); k1 += panelWidth) {
            const Index width = (std::min)(actual_kc - k1, panelWidth);
            if (k1 >= g0 && k1 < g1) packTriangle(blockA + (k1 - g0) * actual_kc, k1, width, actual_kc, k1);
            const Index begin = IsLower ? (std::max)(k1 + width, g0) : g0;
            const Index end = IsLower ? g1 : (std::min)(k1, g1);
            if (end > begin)
              pack_lhs(blockA + (begin - g0) * actual_kc, lhs.getSubMapper(actual_k2 + begin, actual_k2 + k1), width,
                       end - begin, actual_kc, k1);
          }
          for (Index k1 = g0; k1 < g1; k1 += panelWidth) {
            const Index width = (std::min)(actual_kc - k1, panelWidth);
            const Index offset = IsLower ? 0 : k1;
            gebp_kernel(res.getSubMapper(actual_k2 + k1, 0), blockA + (k1 - g0) * actual_kc, blockB, width,
                        IsLower ? k1 + width : actual_kc - k1, cols, alpha, actual_kc, actual_kc, offset, offset);
          }
        }
      } else {
        // for each small vertical panels of lhs
        for (Index k1 = 0; k1 < actual_kc; k1 += panelWidth) {
          Index actualPanelWidth = std::min<Index>(actual_kc - k1, panelWidth);
          Index lengthTarget = IsLower ? actual_kc - k1 - actualPanelWidth : k1;
          Index startBlock = actual_k2 + k1;
          Index blockBOffset = k1;

          // => GEBP with the micro triangular block
          packTriangle(blockA, k1, actualPanelWidth, actualPanelWidth, 0);

          gebp_kernel(res.getSubMapper(startBlock, 0), blockA, blockB, actualPanelWidth, actualPanelWidth, cols, alpha,
                      actualPanelWidth, actual_kc, 0, blockBOffset);

          // GEBP with remaining micro panel
          if (lengthTarget > 0) {
            Index startTarget = IsLower ? actual_k2 + k1 + actualPanelWidth : actual_k2;

            pack_lhs(blockA, lhs.getSubMapper(startTarget, startBlock), actualPanelWidth, lengthTarget,
                     actualPanelWidth, 0);

            gebp_kernel(res.getSubMapper(startTarget, 0), blockA, blockB, lengthTarget, actualPanelWidth, cols, alpha,
                        actualPanelWidth, actual_kc, 0, blockBOffset);
          }
        }
      }
    }
    // the part below (lower case) or above (upper case) the diagonal => GEPP
    {
      Index start = IsLower ? k2 : 0;
      Index end = IsLower ? rows : (std::min)(actual_k2, rows);
      for (Index i2 = start; i2 < end; i2 += mc) {
        const Index actual_mc = (std::min)(i2 + mc, end) - i2;
        gemm_pack_lhs<Scalar, Index, LhsMapper, Traits::mr, Traits::LhsProgress, typename Traits::LhsPacket4Packing,
                      LhsStorageOrder, false>()(blockA, lhs.getSubMapper(i2, actual_k2), actual_kc, actual_mc);

        gebp_kernel(res.getSubMapper(i2, 0), blockA, blockB, actual_mc, actual_kc, cols, alpha, -1, -1, 0, 0);
      }
    }
  }
}

// implements col-major += alpha * op(general) * op(triangular)
template <typename Scalar, typename Index, int Mode, int LhsStorageOrder, bool ConjugateLhs, int RhsStorageOrder,
          bool ConjugateRhs, int ResInnerStride, int Version>
struct product_triangular_matrix_matrix<Scalar, Index, Mode, false, LhsStorageOrder, ConjugateLhs, RhsStorageOrder,
                                        ConjugateRhs, ColMajor, ResInnerStride, Version> {
  using Traits = gebp_traits<Scalar, Scalar>;
  enum {
    SmallPanelWidth = plain_enum_max(Traits::mr, Traits::nr),
    IsLower = (Mode & Lower) == Lower,
    SetDiag = (Mode & (ZeroDiag | UnitDiag)) ? 0 : 1
  };

  static EIGEN_DONT_INLINE void run(Index _rows, Index _cols, Index _depth, const Scalar* lhs_, Index lhsStride,
                                    const Scalar* rhs_, Index rhsStride, Scalar* res, Index resIncr, Index resStride,
                                    const Scalar& alpha, level3_blocking<Scalar, Scalar>& blocking);
};

template <typename Scalar, typename Index, int Mode, int LhsStorageOrder, bool ConjugateLhs, int RhsStorageOrder,
          bool ConjugateRhs, int ResInnerStride, int Version>
EIGEN_DONT_INLINE void product_triangular_matrix_matrix<
    Scalar, Index, Mode, false, LhsStorageOrder, ConjugateLhs, RhsStorageOrder, ConjugateRhs, ColMajor, ResInnerStride,
    Version>::run(Index _rows, Index _cols, Index _depth, const Scalar* lhs_, Index lhsStride, const Scalar* rhs_,
                  Index rhsStride, Scalar* res_, Index resIncr, Index resStride, const Scalar& alpha,
                  level3_blocking<Scalar, Scalar>& blocking) {
  const Index PacketBytes = packet_traits<Scalar>::size * sizeof(Scalar);
  // strip zeros
  Index diagSize = (std::min)(_cols, _depth);
  Index rows = _rows;
  Index depth = IsLower ? _depth : diagSize;
  Index cols = IsLower ? diagSize : _cols;

  using LhsMapper = const_blas_data_mapper<Scalar, Index, LhsStorageOrder>;
  using RhsMapper = const_blas_data_mapper<Scalar, Index, RhsStorageOrder>;
  using ResMapper = blas_data_mapper<typename Traits::ResScalar, Index, ColMajor, Unaligned, ResInnerStride>;
  LhsMapper lhs(lhs_, lhsStride);
  RhsMapper rhs(rhs_, rhsStride);
  ResMapper res(res_, resStride, resIncr);

  Index kc = blocking.kc();                    // cache block size along the K direction
  Index mc = (std::min)(rows, blocking.mc());  // cache block size along the M direction

  std::size_t sizeA = kc * mc;
  // Room for realigning geb below, which moves it by less than one packet: more than EIGEN_MAX_ALIGN_BYTES when
  // packets are wider than the allocation alignment (RVV with LMUL > 1, or a lowered EIGEN_MAX_ALIGN_BYTES).
  std::size_t sizeB = kc * cols + PacketBytes / sizeof(Scalar);

  ei_declare_aligned_stack_constructed_variable(Scalar, blockA, sizeA, blocking.blockA());
  ei_declare_aligned_stack_constructed_variable(Scalar, blockB, sizeB, blocking.blockB());

  static constexpr int PanelCapacity =
      (EIGEN_STACK_ALLOCATION_LIMIT == 0 ||
       SmallPanelWidth * SmallPanelWidth * sizeof(Scalar) <= EIGEN_STACK_ALLOCATION_LIMIT)
          ? int(SmallPanelWidth)
          : Dynamic;
  Matrix<Scalar, PanelCapacity, PanelCapacity, RhsStorageOrder> triangularBuffer(SmallPanelWidth, SmallPanelWidth);
  triangularBuffer.setZero();
  EIGEN_IF_CONSTEXPR ((Mode & ZeroDiag) == ZeroDiag)
    triangularBuffer.diagonal().setZero();
  else
    triangularBuffer.diagonal().setOnes();

  gebp_kernel<Scalar, Scalar, Index, ResMapper, Traits::mr, Traits::nr, ConjugateLhs, ConjugateRhs> gebp_kernel;
  gemm_pack_lhs<Scalar, Index, LhsMapper, Traits::mr, Traits::LhsProgress, typename Traits::LhsPacket4Packing,
                LhsStorageOrder>
      pack_lhs;
  gemm_pack_rhs<Scalar, Index, RhsMapper, Traits::nr, RhsStorageOrder> pack_rhs;
  gemm_pack_rhs<Scalar, Index, RhsMapper, Traits::nr, RhsStorageOrder, false, true> pack_rhs_panel;

  for (Index k2 = IsLower ? 0 : depth; IsLower ? k2 < depth : k2 > 0; IsLower ? k2 += kc : k2 -= kc) {
    Index actual_kc = (std::min)(IsLower ? depth - k2 : k2, kc);
    Index actual_k2 = IsLower ? k2 : k2 - actual_kc;

    // align blocks with the end of the triangular part for trapezoidal rhs
    EIGEN_IF_CONSTEXPR (IsLower) {
      if ((k2 < cols) && (actual_k2 + actual_kc > cols)) {
        actual_kc = cols - k2;
        k2 = actual_k2 + actual_kc - kc;
      }
    }

    // remaining size
    Index rs = IsLower ? (std::min)(cols, actual_k2) : cols - k2;
    // size of the triangular part
    Index ts = (IsLower && actual_k2 >= cols) ? 0 : actual_kc;

    Scalar* geb = blockB + ts * ts;
    geb = geb + internal::first_aligned<PacketBytes>(geb, PacketBytes / sizeof(Scalar));

    pack_rhs(geb, rhs.getSubMapper(actual_k2, IsLower ? 0 : k2), actual_kc, rs);

    // pack the triangular part of the rhs padding the unrolled blocks with zeros
    if (ts > 0) {
      for (Index j2 = 0; j2 < actual_kc; j2 += SmallPanelWidth) {
        Index actualPanelWidth = std::min<Index>(actual_kc - j2, SmallPanelWidth);
        Index actual_j2 = actual_k2 + j2;
        Index panelOffset = IsLower ? j2 + actualPanelWidth : 0;
        Index panelLength = IsLower ? actual_kc - j2 - actualPanelWidth : j2;
        // general part
        pack_rhs_panel(blockB + j2 * actual_kc, rhs.getSubMapper(actual_k2 + panelOffset, actual_j2), panelLength,
                       actualPanelWidth, actual_kc, panelOffset);

        // append the triangular part via a temporary buffer
        for (Index j = 0; j < actualPanelWidth; ++j) {
          EIGEN_IF_CONSTEXPR (SetDiag) triangularBuffer.coeffRef(j, j) = rhs(actual_j2 + j, actual_j2 + j);
          for (Index k = IsLower ? j + 1 : 0; IsLower ? k < actualPanelWidth : k < j; ++k)
            triangularBuffer.coeffRef(k, j) = rhs(actual_j2 + k, actual_j2 + j);
        }

        pack_rhs_panel(blockB + j2 * actual_kc, RhsMapper(triangularBuffer.data(), triangularBuffer.outerStride()),
                       actualPanelWidth, actualPanelWidth, actual_kc, j2);
      }
    }

    for (Index i2 = 0; i2 < rows; i2 += mc) {
      const Index actual_mc = (std::min)(mc, rows - i2);
      pack_lhs(blockA, lhs.getSubMapper(i2, actual_k2), actual_kc, actual_mc);

      // triangular kernel
      if (ts > 0) {
        for (Index j2 = 0; j2 < actual_kc; j2 += SmallPanelWidth) {
          Index actualPanelWidth = std::min<Index>(actual_kc - j2, SmallPanelWidth);
          Index panelLength = IsLower ? actual_kc - j2 : j2 + actualPanelWidth;
          Index blockOffset = IsLower ? j2 : 0;

          gebp_kernel(res.getSubMapper(i2, actual_k2 + j2), blockA, blockB + j2 * actual_kc, actual_mc, panelLength,
                      actualPanelWidth, alpha, actual_kc, actual_kc,  // strides
                      blockOffset, blockOffset);                      // offsets
        }
      }
      gebp_kernel(res.getSubMapper(i2, IsLower ? 0 : k2), blockA, geb, actual_mc, actual_kc, rs, alpha, -1, -1, 0, 0);
    }
  }
}

/***************************************************************************
 * Wrapper to product_triangular_matrix_matrix
 ***************************************************************************/

}  // end namespace internal

namespace internal {
template <int Mode, bool LhsIsTriangular, typename Lhs, typename Rhs>
struct triangular_product_impl<Mode, LhsIsTriangular, Lhs, false, Rhs, false> {
  template <typename Dest>
  static void run(Dest& dst, const Lhs& a_lhs, const Rhs& a_rhs, const typename Dest::Scalar& alpha) {
    using LhsScalar = typename Lhs::Scalar;
    using RhsScalar = typename Rhs::Scalar;
    using Scalar = typename Dest::Scalar;

    if (a_lhs.rows() == 0 || a_lhs.cols() == 0 || a_rhs.cols() == 0) return;
    // Reuse GEMM's generic crossover: below it, packing costs more than a coefficient-based product.
    if (a_lhs.rows() + a_lhs.cols() + a_rhs.cols() < EIGEN_GEMM_TO_COEFFBASED_THRESHOLD) {
      constexpr bool lower = (Mode & Lower) != 0;
      constexpr bool implicitDiag = (Mode & (UnitDiag | ZeroDiag)) != 0;
      evaluator<Lhs> lhs(a_lhs);
      evaluator<Rhs> rhs(a_rhs);
      const Index depth = a_lhs.cols();
      for (Index j = 0; j < a_rhs.cols(); ++j) {
        for (Index i = 0; i < a_lhs.rows(); ++i) {
          const Index diagonal = LhsIsTriangular ? i : j;
          const bool prefix = LhsIsTriangular ? lower : !lower;
          const Index begin = prefix ? 0 : diagonal + Index(implicitDiag);
          const Index end = prefix ? (numext::mini)(depth, diagonal + Index(!implicitDiag)) : depth;
          Scalar sum(0);
          for (Index k = begin; k < end; ++k) sum += lhs.coeff(i, k) * rhs.coeff(k, j);
          EIGEN_IF_CONSTEXPR ((Mode & UnitDiag) != 0) {
            if (diagonal < depth) sum += LhsIsTriangular ? rhs.coeff(i, j) : lhs.coeff(i, j);
          }
          dst.coeffRef(i, j) += alpha * sum;
        }
      }
      return;
    }

    using LhsBlasTraits = internal::blas_traits<Lhs>;
    using ActualLhsType = typename LhsBlasTraits::DirectLinearAccessType;
    using ActualLhsTypeCleaned = internal::remove_all_t<ActualLhsType>;
    using RhsBlasTraits = internal::blas_traits<Rhs>;
    using ActualRhsType = typename RhsBlasTraits::DirectLinearAccessType;
    using ActualRhsTypeCleaned = internal::remove_all_t<ActualRhsType>;

    internal::add_const_on_value_type_t<ActualLhsType> lhs = LhsBlasTraits::extract(a_lhs);
    internal::add_const_on_value_type_t<ActualRhsType> rhs = RhsBlasTraits::extract(a_rhs);

    LhsScalar lhs_alpha = LhsBlasTraits::extractScalarFactor(a_lhs);
    RhsScalar rhs_alpha = RhsBlasTraits::extractScalarFactor(a_rhs);
    Scalar actualAlpha = alpha * lhs_alpha * rhs_alpha;

    using BlockingType = internal::gemm_blocking_space<(Dest::Flags & RowMajorBit) ? RowMajor : ColMajor, Scalar,
                                                       Scalar, Lhs::MaxRowsAtCompileTime, Rhs::MaxColsAtCompileTime,
                                                       Lhs::MaxColsAtCompileTime, 4>;

    enum { IsLower = (Mode & Lower) == Lower };
    Index stripedRows = ((!LhsIsTriangular) || (IsLower)) ? lhs.rows() : (std::min)(lhs.rows(), lhs.cols());
    Index stripedCols = ((LhsIsTriangular) || (!IsLower)) ? rhs.cols() : (std::min)(rhs.cols(), rhs.rows());
    Index stripedDepth = LhsIsTriangular ? ((!IsLower) ? lhs.cols() : (std::min)(lhs.cols(), lhs.rows()))
                                         : ((IsLower) ? rhs.rows() : (std::min)(rhs.rows(), rhs.cols()));

    using Trmm = internal::product_triangular_matrix_matrix<
        Scalar, Index, Mode, LhsIsTriangular,
        (internal::traits<ActualLhsTypeCleaned>::Flags & RowMajorBit) ? RowMajor : ColMajor,
        LhsBlasTraits::NeedToConjugate,
        (internal::traits<ActualRhsTypeCleaned>::Flags & RowMajorBit) ? RowMajor : ColMajor,
        RhsBlasTraits::NeedToConjugate, (internal::traits<Dest>::Flags & RowMajorBit) ? RowMajor : ColMajor,
        Dest::InnerStrideAtCompileTime>;
    const auto runWith = [&](level3_blocking<Scalar, Scalar>& blocking) {
      Trmm::run(stripedRows, stripedCols, stripedDepth,                     // sizes
                &lhs.coeffRef(0, 0), lhs.outerStride(),                     // lhs info
                &rhs.coeffRef(0, 0), rhs.outerStride(),                     // rhs info
                &dst.coeffRef(0, 0), dst.innerStride(), dst.outerStride(),  // result info
                actualAlpha, blocking);
    };

#ifdef EIGEN_VECTORIZE_AVX512
    // The AVX512 physical-L1 cap on kc already keeps the micro-panels in L1, so on large outputs BlockingType's
    // extra KcFactor = 4 makes depth blocks too short to amortize result loads/stores; small and narrow products
    // still favor it. Fixed-size blocking packs the whole product into inline buffers and ignores KcFactor, so only
    // dynamic blocking can change; a second fixed-size blocking object would double the stack frame.
    constexpr bool DynamicBlocking = Lhs::MaxRowsAtCompileTime == Dynamic || Rhs::MaxColsAtCompileTime == Dynamic ||
                                     Lhs::MaxColsAtCompileTime == Dynamic;
    if (DynamicBlocking && (is_same<Scalar, float>::value || is_same<Scalar, double>::value) &&
        gebp_traits<Scalar, Scalar>::nr == 8 && stripedRows >= 512 && stripedCols >= 512) {
      using LargeBlockingType = internal::gemm_blocking_space<(Dest::Flags & RowMajorBit) ? RowMajor : ColMajor, Scalar,
                                                              Scalar, Dynamic, Dynamic, Dynamic, 1>;
      LargeBlockingType blocking(stripedRows, stripedCols, stripedDepth, 1, false);
      runWith(blocking);
    } else
#endif
    {
      BlockingType blocking(stripedRows, stripedCols, stripedDepth, 1, false);
      runWith(blocking);
    }

    // A factor s nested in a unit triangle reached the kernel through actualAlpha, so the unit diagonal
    // contributed alpha * s * other instead of alpha * other: subtract alpha * (s - 1) * other.
    EIGEN_IF_CONSTEXPR ((Mode & UnitDiag) == UnitDiag) {
      EIGEN_IF_CONSTEXPR (LhsIsTriangular) {
        if (!numext::is_exactly_one(lhs_alpha)) {
          Index diagSize = (std::min)(lhs.rows(), lhs.cols());
          dst.matrix().topRows(diagSize) -= ((alpha * (lhs_alpha - LhsScalar(1))) * a_rhs).topRows(diagSize);
        }
      } else {
        if (!numext::is_exactly_one(rhs_alpha)) {
          Index diagSize = (std::min)(rhs.rows(), rhs.cols());
          dst.matrix().leftCols(diagSize) -= (alpha * (rhs_alpha - RhsScalar(1))) * a_lhs.leftCols(diagSize);
        }
      }
    }
  }
};

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_TRIANGULAR_MATRIX_MATRIX_H
