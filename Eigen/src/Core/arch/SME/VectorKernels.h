// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#ifndef EIGEN_SME_VECTOR_KERNELS_H
#define EIGEN_SME_VECTOR_KERNELS_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

#if EIGEN_COMP_CLANG
#define EIGEN_SME_VECTOR_UNROLL4 _Pragma("unroll")
#elif EIGEN_COMP_GNUC
#define EIGEN_SME_VECTOR_UNROLL4 _Pragma("GCC unroll 4")
#else
#define EIGEN_SME_VECTOR_UNROLL4
#endif

// FPCR.RMode == 0b10 (roundTowardNegative), the one mode where (+0) + (-0) is -0.
static EIGEN_ALWAYS_INLINE bool sme_rounds_toward_negative() {
  std::uint64_t fpcr;
  asm volatile("mrs %0, fpcr" : "=r"(fpcr));
  return ((fpcr >> 22) & 3) == 2;
}

template <typename Scalar, typename Index>
__arm_new("za") __arm_locally_streaming
    EIGEN_DONT_INLINE void sme_axpy(Index n, const Scalar* x, Scalar* y, Scalar alpha, Index prefix) {
  using Traits = sme_packet_traits<Scalar>;
  using Vec = typename Traits::type;
  const Index lanes = Traits::size();
  const auto pn = Traits::ptrue_c();
  const auto a = pset1<Vec>(alpha);
  Index i = 0;
  for (; i < prefix; i += prefix - i < lanes ? prefix - i : lanes) {
    auto active = Traits::whilelt(i, prefix);
    pstoreu(active, y + i, pmadd(active, ploadu(active, x + i), a, ploadu(active, y + i)));
  }
  for (; i <= n - 16 * lanes; i += 16 * lanes) {
    EIGEN_SME_VECTOR_UNROLL4
    for (int k = 0; k < 4; ++k) {
      auto xv = ploadu_x4(pn, x + i + k * 4 * lanes);
      sme_write_za_vg1x4(k, ploadu_x4(pn, y + i + k * 4 * lanes));
      sme_madd_za_vg1x4(k, xv, a);
    }
    EIGEN_SME_VECTOR_UNROLL4
    for (int k = 0; k < 4; ++k) pstoreu_x4(pn, y + i + k * 4 * lanes, sme_read_za_vg1x4<Scalar>(k));
  }
  for (; i <= n - 4 * lanes; i += 4 * lanes) {
    auto xv = ploadu_x4(pn, x + i);
    sme_write_za_vg1x4(0, ploadu_x4(pn, y + i));
    sme_madd_za_vg1x4(0, xv, a);
    pstoreu_x4(pn, y + i, sme_read_za_vg1x4<Scalar>(0));
  }
  for (; i < n; i += n - i < lanes ? n - i : lanes) {
    auto tail = Traits::whilelt(i, n);
    pstoreu(tail, y + i, pmadd(tail, ploadu(tail, x + i), a, ploadu(tail, y + i)));
  }
}

template <typename Scalar, typename Index>
__arm_new("za") __arm_locally_streaming EIGEN_DONT_INLINE Scalar sme_dot(Index n, const Scalar* x, const Scalar* y) {
  using Traits = sme_packet_traits<Scalar>;
  using Vec = typename Traits::type;
  const Index lanes = Traits::size();
  const auto pg = Traits::ptrue();
  const auto pn = Traits::ptrue_c();
  // -0 is the additive identity except under roundTowardNegative, so a zero sum gets the IEEE sign.
  const auto negative_zero = pset1<Vec>(Scalar(-0.0));
  EIGEN_SME_VECTOR_UNROLL4
  for (int k = 0; k < 4; ++k)
    sme_write_za_vg1x4(k, pcreate(negative_zero, negative_zero, negative_zero, negative_zero));
  Index i = 0;
  for (; i <= n - 16 * lanes; i += 16 * lanes) {
    EIGEN_SME_VECTOR_UNROLL4
    for (int k = 0; k < 4; ++k)
      sme_madd_za_vg1x4(k, ploadu_x4(pn, x + i + k * 4 * lanes), ploadu_x4(pn, y + i + k * 4 * lanes));
  }
  for (; i <= n - 4 * lanes; i += 4 * lanes) sme_madd_za_vg1x4(0, ploadu_x4(pn, x + i), ploadu_x4(pn, y + i));
  auto tail_accumulator = negative_zero;
  for (; i < n; i += n - i < lanes ? n - i : lanes) {
    auto tail = Traits::whilelt(i, n);
    tail_accumulator = pmadd_m(tail, ploadu(tail, x + i), ploadu(tail, y + i), tail_accumulator);
  }
  auto sum = tail_accumulator;
  EIGEN_SME_VECTOR_UNROLL4
  for (int k = 0; k < 4; ++k) {
    auto v = sme_read_za_vg1x4<Scalar>(k);
    sum = padd(pg, sum, padd(pg, padd(pg, pget<0>(v), pget<1>(v)), padd(pg, pget<2>(v), pget<3>(v))));
  }
  return predux(pg, sum);
}

// Bounds the columns, and so the pages, one row sweep touches; a multiple of every block_cols below cols.
static constexpr Index kSmeGemvPanelCols = 128;
static_assert(kSmeGemvPanelCols % 16 == 0, "a panel holds whole batches of 16 or 4 columns");

// Needs SVL >= 256 bits (sme_gemv_svl_suitable): eight vgx4 groups, group g inside 64-bit tile g mod 8.
template <typename Scalar, typename Index>
__arm_new("za") __arm_locally_streaming
    EIGEN_DONT_INLINE void sme_gemv(Index rows, Index cols, const Scalar* a, Index stride, const Scalar* x, Scalar* y,
                                    Scalar alpha, Index block_cols) {
  using Traits = sme_packet_traits<Scalar>;
  using Vec = typename Traits::type;
  const Index lanes = Traits::size();
  const auto scale = pset1<Vec>(alpha);
  // 4-column batches mean a column stride past L1; a wider panel then costs 8-16% at 16384-65536 rows on Apple M4.
  const Index panel_cols = block_cols < 16 ? block_cols : Index(kSmeGemvPanelCols);
  for (Index panel_first = 0; panel_first < cols;) {
    const Index panel_end = cols - panel_first < panel_cols ? cols : panel_first + panel_cols;
    for (Index i = 0; i < rows;) {
      const Index remaining = rows - i;
      // Four predicated chunks per row block keep four independent FMLA chains, also in the final partial block.
      // An inactive chunk addresses the block start and accesses no memory.
      const auto p0 = Traits::whilelt_c4(i, rows), p1 = Traits::whilelt_c4(i + std::int64_t(4) * lanes, rows),
                 p2 = Traits::whilelt_c4(i + std::int64_t(8) * lanes, rows),
                 p3 = Traits::whilelt_c4(i + std::int64_t(12) * lanes, rows);
      const Index o1 = remaining > 4 * lanes ? 4 * lanes : 0, o2 = remaining > 8 * lanes ? 8 * lanes : 0,
                  o3 = remaining > 12 * lanes ? 12 * lanes : 0;
      // y stays in groups 4-7 for the panel, as y traffic per batch is 2.5x slower unless y is 64-byte aligned.
      sme_write_za_vg1x4(4, ploadu_x4(p0, y + i));
      sme_write_za_vg1x4(5, ploadu_x4(p1, y + i + o1));
      sme_write_za_vg1x4(6, ploadu_x4(p2, y + i + o2));
      sme_write_za_vg1x4(7, ploadu_x4(p3, y + i + o3));
      // Match the generic GEMV's scaled column batches to bound the unscaled sums: y += alpha * sum_batch(A x).
      for (Index first = panel_first; first < panel_end;) {
        const Index end = first + (panel_end - first < block_cols ? panel_end - first : block_cols);
        svzero_mask_za(0x0f);  // za0.d-za3.d, which hold groups 0-3
        for (Index j = first; j < end; ++j) {
          const auto b = pset1<Vec>(x[j]);
          const Scalar* column = a + i + j * stride;
          sme_madd_za_vg1x4(0, ploadu_x4(p0, column), b);
          sme_madd_za_vg1x4(1, ploadu_x4(p1, column + o1), b);
          sme_madd_za_vg1x4(2, ploadu_x4(p2, column + o2), b);
          sme_madd_za_vg1x4(3, ploadu_x4(p3, column + o3), b);
        }
        for (unsigned int k = 0; k < 4; ++k) sme_madd_za_vg1x4(k + 4, sme_read_za_vg1x4<Scalar>(k), scale);
        first = end;
      }
      pstoreu_x4(p0, y + i, sme_read_za_vg1x4<Scalar>(4));
      pstoreu_x4(p1, y + i + o1, sme_read_za_vg1x4<Scalar>(5));
      pstoreu_x4(p2, y + i + o2, sme_read_za_vg1x4<Scalar>(6));
      pstoreu_x4(p3, y + i + o3, sme_read_za_vg1x4<Scalar>(7));
      // Bound the final increment to avoid signed overflow with a 32-bit Index near its maximum.
      i += remaining < 16 * lanes ? remaining : 16 * lanes;
    }
    panel_first = panel_end;
  }
}

// Accumulates dot(row + k * stride, x) for k < block <= 8 into ZA groups 2k and 2k + 1.
template <bool Full, typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE void sme_gemv_rowmajor_block(Index block, Index cols, const Scalar* row, Index stride,
                                                 const Scalar* x) __arm_streaming __arm_inout("za") {
  using Traits = sme_packet_traits<Scalar>;
  const Index lanes = Traits::size();
  // As in sme_gemv: the bounded increment avoids signed overflow, and an inactive chunk accesses no memory.
  for (Index i = 0; i < cols; i += cols - i < 8 * lanes ? cols - i : 8 * lanes) {
    const auto p0 = Traits::whilelt_c4(i, cols), p1 = Traits::whilelt_c4(i + std::int64_t(4) * lanes, cols);
    const Index o1 = cols - i > 4 * lanes ? 4 * lanes : 0;
    const auto x0 = ploadu_x4(p0, x + i), x1 = ploadu_x4(p1, x + i + o1);
    const unsigned int count = Full ? 8u : static_cast<unsigned int>(block);
    for (unsigned int k = 0; k < count; ++k) {
      const Scalar* chunk = row + k * stride + i;
      sme_madd_za_vg1x4(2 * k, ploadu_x4(p0, chunk), x0);
      sme_madd_za_vg1x4(2 * k + 1, ploadu_x4(p1, chunk + o1), x1);
    }
  }
}

// Sum of the eight vectors of ZA groups 2k and 2k + 1.
template <typename Scalar>
EIGEN_ALWAYS_INLINE typename sme_packet_traits<Scalar>::type sme_gemv_rowmajor_sum(
    unsigned int k) __arm_streaming __arm_in("za") {
  const auto pg = sme_packet_traits<Scalar>::ptrue();
  const auto u = sme_read_za_vg1x4<Scalar>(2 * k), v = sme_read_za_vg1x4<Scalar>(2 * k + 1);
  return padd(pg, padd(pg, padd(pg, pget<0>(u), pget<1>(u)), padd(pg, pget<2>(u), pget<3>(u))),
              padd(pg, padd(pg, pget<0>(v), pget<1>(v)), padd(pg, pget<2>(v), pget<3>(v))));
}

// Row-major GEMV, y[j] += alpha * dot(a + j * stride, x); needs SVL >= 512 bits: 16 vgx4 groups, 8 rows in tile 0.
template <typename Scalar, typename Index>
__arm_new("za") __arm_locally_streaming
    EIGEN_DONT_INLINE void sme_gemv_rowmajor(Index rows, Index cols, const Scalar* a, Index stride, const Scalar* x,
                                             Scalar* y, Scalar alpha) {
  using Traits = sme_packet_traits<Scalar>;
  using Vec = typename Traits::type;
  const Index lanes = Traits::size();
  const auto pg = Traits::ptrue();
  const auto scale = pset1<Vec>(alpha);
  for (Index j = 0; j < rows;) {
    const Index block = rows - j < 8 ? rows - j : 8;
    svzero_za();
    if (block == 8)
      sme_gemv_rowmajor_block<true>(block, cols, a + j * stride, stride, x);
    else
      sme_gemv_rowmajor_block<false>(block, cols, a + j * stride, stride, x);
    const auto s0 = sme_gemv_rowmajor_sum<Scalar>(0), s1 = sme_gemv_rowmajor_sum<Scalar>(1),
               s2 = sme_gemv_rowmajor_sum<Scalar>(2), s3 = sme_gemv_rowmajor_sum<Scalar>(3),
               s4 = sme_gemv_rowmajor_sum<Scalar>(4), s5 = sme_gemv_rowmajor_sum<Scalar>(5),
               s6 = sme_gemv_rowmajor_sum<Scalar>(6), s7 = sme_gemv_rowmajor_sum<Scalar>(7);
    // Transpose through tile 0: row k holds sk, so lane k of the sum of all vertical slices is dot k.
    svzero_za();
    sme_write_hor_za_vg4<0>(0, s0, s1, s2, s3);
    sme_write_hor_za_vg4<0>(4, s4, s5, s6, s7);
    auto dots = pset1<Vec>(Scalar(0));
    for (unsigned int p = 0; p < static_cast<unsigned int>(lanes); p += 4) {
      const auto v = sme_read_ver_za_vg4<0, Scalar>(p);
      dots = padd(pg, dots, padd(pg, padd(pg, pget<0>(v), pget<1>(v)), padd(pg, pget<2>(v), pget<3>(v))));
    }
    const auto active = Traits::whilelt(0, block);
    pstoreu(active, y + j, pmadd(active, dots, scale, ploadu(active, y + j)));
    j += block;
  }
}

#undef EIGEN_SME_VECTOR_UNROLL4

template <typename Scalar>
struct sme_vector_scalar : false_type {};
template <>
struct sme_vector_scalar<float> : true_type {};
#ifdef EIGEN_VECTORIZE_SME_F64F64
template <>
struct sme_vector_scalar<double> : true_type {};
#endif

template <typename Xpr>
struct sme_vector_access : bool_constant<(sme_vector_scalar<typename traits<Xpr>::Scalar>::value &&
                                          bool(Xpr::IsVectorAtCompileTime) && has_direct_access<Xpr>::value)> {};

template <typename Lhs, typename Rhs>
struct sme_dot_supported : bool_constant<sme_vector_access<Lhs>::value && sme_vector_access<Rhs>::value &&
                                         is_same<typename traits<Lhs>::Scalar, typename traits<Rhs>::Scalar>::value> {};

// Fixed crossovers, fitted on Apple M4 (SVL 512): below them the NEON/SME memory handoff outweighs the
// streaming kernels. Core's cache estimates, including setCpuCacheSizes overrides, supply the others.
static constexpr std::size_t kSmeDotMinBytes = 16384;  // per operand
static constexpr std::size_t kSmeAxpyMinBytes = 4096;  // per operand
static constexpr Index kSmeGemvMinRows = 128;
static constexpr Index kSmeGemvMinCols = 4;
// A GEMV whose RHS reaches kSmeGemvWideRhsBytes needs only kSmeGemvWideMinBytes of matrix, not Core's L1 estimate.
static constexpr std::size_t kSmeGemvWideRhsBytes = 128;
static constexpr std::size_t kSmeGemvWideMinBytes = 32768;
// Row-major GEMV: shorter dots, in streaming vectors, lose to NEON as the reduction per output outweighs the loop.
static constexpr Index kSmeGemvRowMajorMinDotVectors = 24;

// Keep cache queries out of callers so the small vector fallback can inline.
// Per operand, the L1 estimate (divided by L1Divisor) is the lower crossover and the L2 estimate the upper one.
template <typename Scalar, int L1Divisor = 1>
EIGEN_DONT_INLINE bool sme_vector_size_suitable(Index size) {
  std::ptrdiff_t l1, l2, l3;
  manage_caching_sizes(GetAction, &l1, &l2, &l3);
  return l1 > 0 && l2 > 0 && size >= 0 &&
         static_cast<std::size_t>(size) >= (static_cast<std::size_t>(l1) - 1) / (L1Divisor * sizeof(Scalar)) + 1 &&
         static_cast<std::size_t>(size) <= static_cast<std::size_t>(l2) / sizeof(Scalar);
}

template <typename Lhs, typename Rhs>
struct default_inner_product_impl<Lhs, Rhs, true, std::enable_if_t<sme_dot_supported<Lhs, Rhs>::value>>
    : default_inner_product_impl<Lhs, Rhs, true, false_type> {
  using Base = default_inner_product_impl<Lhs, Rhs, true, false_type>;
  using Scalar = typename traits<Lhs>::Scalar;
  static EIGEN_STRONG_INLINE Scalar run(const MatrixBase<Lhs>& lhs, const MatrixBase<Rhs>& rhs) {
    inner_product_assert<Lhs, Rhs>::run(lhs.derived(), rhs.derived());
    // DOT has no streaming stores for a following NEON consumer: half L1 per operand suffices.
    if (lhs.size() >= Index(kSmeDotMinBytes / sizeof(Scalar)) && sme_vector_size_suitable<Scalar, 2>(lhs.size()) &&
        lhs.innerStride() == 1 && rhs.innerStride() == 1) {
      sme_fpsr_guard status;
      const Scalar result = sme_dot(lhs.size(), lhs.derived().data(), rhs.derived().data());
      // Under roundTowardNegative the -0 seeds turn +0 sums into -0.
      if (result != Scalar(0) || !sme_rounds_toward_negative()) return result;
    }
    return Base::run(lhs, rhs);
  }
};

template <typename Dst, typename Src>
EIGEN_STRONG_INLINE bool sme_try_axpy(Dst& dst, const Src& src, typename traits<Src>::Scalar alpha) {
  using Scalar = typename traits<Src>::Scalar;
  eigen_assert(dst.rows() == src.rows() && dst.cols() == src.cols());
  if (dst.size() < Index(kSmeAxpyMinBytes / sizeof(Scalar)) || !sme_vector_size_suitable<Scalar>(dst.size()) ||
      dst.innerStride() != 1 || src.innerStride() != 1)
    return false;
  const std::uintptr_t dst_address = reinterpret_cast<std::uintptr_t>(dst.data());
  const std::uintptr_t src_address = reinterpret_cast<std::uintptr_t>(src.data());
  const std::uintptr_t distance = dst_address > src_address ? dst_address - src_address : src_address - dst_address;
  // Exact aliasing is coefficient-wise; partial overlap must retain the default traversal.
  if (distance != 0 && distance / sizeof(Scalar) < static_cast<std::uintptr_t>(dst.size())) return false;
  // Align streaming stores to 64 bytes without changing Eigen's allocation alignment.
  sme_fpsr_guard status;
  sme_axpy(dst.size(), src.data(), dst.data(), alpha, first_aligned<64>(dst.data(), dst.size()));
  return true;
}

template <typename Dst, typename Scalar, typename Lhs, typename Rhs>
struct Assignment<
    Dst, CwiseBinaryOp<scalar_product_op<Scalar, Scalar>, Lhs, Rhs>, add_assign_op<Scalar, Scalar>, Dense2Dense,
    std::enable_if_t<sme_vector_access<Dst>::value &&
                     (sme_vector_access<Lhs>::value || sme_vector_access<Rhs>::value) &&
                     blas_traits<CwiseBinaryOp<scalar_product_op<Scalar, Scalar>, Lhs, Rhs>>::HasScalarFactor &&
                     is_same<Scalar, typename traits<Dst>::Scalar>::value>> {
  using Source = CwiseBinaryOp<scalar_product_op<Scalar, Scalar>, Lhs, Rhs>;
  using BlasTraits = blas_traits<Source>;
  static EIGEN_STRONG_INLINE void run(Dst& dst, const Source& src, const add_assign_op<Scalar, Scalar>& func) {
    // A direct operand restricts extraction to one scale, preserving nested products' evaluation order.
    if (!sme_try_axpy(dst, BlasTraits::extract(src), BlasTraits::extractScalarFactor(src)))
      Assignment<Dst, Source, add_assign_op<Scalar, Scalar>, Dense2Dense, false_type>::run(dst, src, func);
  }
};

#ifndef EIGEN_USE_BLAS
// sme_gemv's ZA layout needs SVL >= 256 bits.
EIGEN_ALWAYS_INLINE bool sme_gemv_svl_suitable() { return svcntsb() >= 32; }

template <typename Scalar, typename Index>
EIGEN_STRONG_INLINE bool sme_gemv_size_suitable(Index rows, Index cols) {
  if (rows < Index(kSmeGemvMinRows) || cols < Index(kSmeGemvMinCols)) return false;
  if (cols >= Index(kSmeGemvWideRhsBytes / sizeof(Scalar)) &&
      cols > Index(kSmeGemvWideMinBytes / sizeof(Scalar) - 1) / rows)
    return true;
  // Thin products amortize the NEON/SME handoff once the matrix reaches Core's L1 estimate.
  const std::ptrdiff_t l1 = l1CacheSize();
  return l1 > 0 && static_cast<std::size_t>(cols) >
                       (static_cast<std::size_t>(l1) - 1) / sizeof(Scalar) / static_cast<std::size_t>(rows);
}

#define EIGEN_SME_GEMV_SPECIALIZATION(Scalar)                                                                    \
  template <typename Index, bool ConjugateLhs, bool ConjugateRhs>                                                \
  struct general_matrix_vector_product<Index, Scalar, const_blas_data_mapper<Scalar, Index, ColMajor>, ColMajor, \
                                       ConjugateLhs, Scalar, const_blas_data_mapper<Scalar, Index, RowMajor>,    \
                                       ConjugateRhs, Specialized> {                                              \
    static EIGEN_STRONG_INLINE void run(Index rows, Index cols,                                                  \
                                        const const_blas_data_mapper<Scalar, Index, ColMajor>& lhs,              \
                                        const const_blas_data_mapper<Scalar, Index, RowMajor>& rhs, Scalar* res, \
                                        Index resIncr, Scalar alpha) {                                           \
      if (sme_gemv_size_suitable<Scalar>(rows, cols) && rhs.stride() == 1 && resIncr == 1 &&                     \
          sme_gemv_svl_suitable()) {                                                                             \
        const std::ptrdiff_t l1 = l1CacheSize();                                                                 \
        const Index block_cols = cols < 128 ? cols                                                               \
                                            : (l1 > 0 && static_cast<std::size_t>(lhs.stride()) <=               \
                                                             (static_cast<std::size_t>(l1) - 1) / sizeof(Scalar) \
                                                   ? Index(16)                                                   \
                                                   : Index(4));                                                  \
        /* BLAS contract: alpha == 0 leaves the result unchanged. */                                             \
        if (alpha == Scalar(0)) return;                                                                          \
        sme_fpsr_guard status;                                                                                   \
        sme_gemv(rows, cols, lhs.data(), lhs.stride(), rhs.data(), res, alpha, block_cols);                      \
      } else {                                                                                                   \
        general_matrix_vector_product<Index, Scalar, const_blas_data_mapper<Scalar, Index, ColMajor>, ColMajor,  \
                                      ConjugateLhs, Scalar, const_blas_data_mapper<Scalar, Index, RowMajor>,     \
                                      ConjugateRhs, BuiltIn>::run(rows, cols, lhs, rhs, res, resIncr, alpha);    \
      }                                                                                                          \
    }                                                                                                            \
  };
// rows outputs, each a dot of length cols.
template <typename Scalar, typename Index>
EIGEN_STRONG_INLINE bool sme_gemv_rowmajor_suitable(Index rows, Index cols) {
  const Index svl_bytes = Index(svcntsb());
  return svl_bytes >= 64 && cols >= kSmeGemvRowMajorMinDotVectors * (svl_bytes / Index(sizeof(Scalar))) &&
         sme_gemv_size_suitable<Scalar>(cols, rows);
}

#define EIGEN_SME_GEMV_ROWMAJOR_SPECIALIZATION(Scalar)                                                           \
  template <typename Index, bool ConjugateLhs, bool ConjugateRhs>                                                \
  struct general_matrix_vector_product<Index, Scalar, const_blas_data_mapper<Scalar, Index, RowMajor>, RowMajor, \
                                       ConjugateLhs, Scalar, const_blas_data_mapper<Scalar, Index, ColMajor>,    \
                                       ConjugateRhs, Specialized> {                                              \
    static EIGEN_STRONG_INLINE void run(Index rows, Index cols,                                                  \
                                        const const_blas_data_mapper<Scalar, Index, RowMajor>& lhs,              \
                                        const const_blas_data_mapper<Scalar, Index, ColMajor>& rhs, Scalar* res, \
                                        Index resIncr, Scalar alpha) {                                           \
      if (sme_gemv_rowmajor_suitable<Scalar>(rows, cols) && rhs.stride() == 1 && resIncr == 1) {                 \
        /* BLAS contract: alpha == 0 leaves the result unchanged. */                                             \
        if (alpha == Scalar(0)) return;                                                                          \
        sme_fpsr_guard status;                                                                                   \
        sme_gemv_rowmajor(rows, cols, lhs.data(), lhs.stride(), rhs.data(), res, alpha);                         \
      } else {                                                                                                   \
        general_matrix_vector_product<Index, Scalar, const_blas_data_mapper<Scalar, Index, RowMajor>, RowMajor,  \
                                      ConjugateLhs, Scalar, const_blas_data_mapper<Scalar, Index, ColMajor>,     \
                                      ConjugateRhs, BuiltIn>::run(rows, cols, lhs, rhs, res, resIncr, alpha);    \
      }                                                                                                          \
    }                                                                                                            \
  };
EIGEN_SME_GEMV_SPECIALIZATION(float)
EIGEN_SME_GEMV_ROWMAJOR_SPECIALIZATION(float)
#ifdef EIGEN_VECTORIZE_SME_F64F64
EIGEN_SME_GEMV_SPECIALIZATION(double)
EIGEN_SME_GEMV_ROWMAJOR_SPECIALIZATION(double)
#endif
#undef EIGEN_SME_GEMV_ROWMAJOR_SPECIALIZATION
#undef EIGEN_SME_GEMV_SPECIALIZATION
#endif

}  // namespace internal
}  // namespace Eigen

#endif
