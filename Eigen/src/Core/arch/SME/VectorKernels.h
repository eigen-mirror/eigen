// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#ifndef EIGEN_SME_VECTOR_KERNELS_H
#define EIGEN_SME_VECTOR_KERNELS_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

// SME2 multi-vector FMLA updates four streaming vectors per ZA group. Four independent groups
// hide accumulator latency. ACLE: https://arm-software.github.io/acle/main/acle.html

static EIGEN_ALWAYS_INLINE void sme_vector_madd(unsigned int slice, svfloat32x4_t x,
                                                svfloat32_t y) __arm_streaming __arm_inout("za") {
  svmla_single_za32_f32_vg1x4(slice, x, y);
}
static EIGEN_ALWAYS_INLINE void sme_vector_madd(unsigned int slice, svfloat32x4_t x,
                                                svfloat32x4_t y) __arm_streaming __arm_inout("za") {
  svmla_za32_f32_vg1x4(slice, x, y);
}
static EIGEN_ALWAYS_INLINE svfloat32x4_t sme_vector_read(unsigned int slice, float) __arm_streaming __arm_in("za") {
  return svread_za32_f32_vg1x4(slice);
}
static EIGEN_ALWAYS_INLINE void sme_vector_write(unsigned int slice,
                                                 svfloat32x4_t x) __arm_streaming __arm_inout("za") {
  svwrite_za32_f32_vg1x4(slice, x);
}

#ifdef EIGEN_VECTORIZE_SME_F64F64
static EIGEN_ALWAYS_INLINE void sme_vector_madd(unsigned int slice, svfloat64x4_t x,
                                                svfloat64_t y) __arm_streaming __arm_inout("za") {
  svmla_single_za64_f64_vg1x4(slice, x, y);
}
static EIGEN_ALWAYS_INLINE void sme_vector_madd(unsigned int slice, svfloat64x4_t x,
                                                svfloat64x4_t y) __arm_streaming __arm_inout("za") {
  svmla_za64_f64_vg1x4(slice, x, y);
}
static EIGEN_ALWAYS_INLINE svfloat64x4_t sme_vector_read(unsigned int slice, double) __arm_streaming __arm_in("za") {
  return svread_za64_f64_vg1x4(slice);
}
static EIGEN_ALWAYS_INLINE void sme_vector_write(unsigned int slice,
                                                 svfloat64x4_t x) __arm_streaming __arm_inout("za") {
  svwrite_za64_f64_vg1x4(slice, x);
}
#endif

#if EIGEN_COMP_CLANG
#define EIGEN_SME_VECTOR_UNROLL4 _Pragma("unroll")
#elif EIGEN_COMP_GNUC
#define EIGEN_SME_VECTOR_UNROLL4 _Pragma("GCC unroll 4")
#else
#define EIGEN_SME_VECTOR_UNROLL4
#endif

// SMSTART/SMSTOP set FPSR's cumulative flags (Arm DDI0616 A.a, RMHTLZ), and so can resuming a thread preempted
// in streaming mode, so no FPSR value read while streaming is reliable. ZA arithmetic raises no flags either.
// Restoring the caller's FPSR around the streaming call is deterministic; the kernels report no exceptions.
struct sme_vector_fpsr {
  EIGEN_ALWAYS_INLINE sme_vector_fpsr() { asm volatile("mrs %0, fpsr" : "=r"(value) : : "memory"); }
  EIGEN_ALWAYS_INLINE ~sme_vector_fpsr() { asm volatile("msr fpsr, %0" : : "r"(value) : "memory"); }
  sme_vector_fpsr(const sme_vector_fpsr&) = delete;
  sme_vector_fpsr& operator=(const sme_vector_fpsr&) = delete;

 private:
  std::uint64_t value;
};

// FPCR.RMode == 0b10 (roundTowardNegative), the one mode where (+0) + (-0) is -0.
static EIGEN_ALWAYS_INLINE bool sme_rounds_toward_negative() {
  std::uint64_t fpcr;
  asm volatile("mrs %0, fpcr" : "=r"(fpcr));
  return ((fpcr >> 22) & 3) == 2;
}

template <typename Scalar, typename Index>
__arm_new("za") __arm_locally_streaming
    EIGEN_DONT_INLINE void sme_axpy(Index n, const Scalar* x, Scalar* y, Scalar alpha, Index prefix) {
  using Traits = sme_traits<Scalar>;
  const Index lanes = Traits::svl();
  const auto pn = Traits::ptrue_c();
  const auto a = Traits::dup(alpha);
  Index i = 0;
  for (; i < prefix; i += prefix - i < lanes ? prefix - i : lanes) {
    auto active = Traits::whilelt(i, prefix);
    sme_st1(active, y + i, svmla_x(active, sme_ld1(active, y + i), sme_ld1(active, x + i), alpha));
  }
  for (; i <= n - 16 * lanes; i += 16 * lanes) {
    EIGEN_SME_VECTOR_UNROLL4
    for (int k = 0; k < 4; ++k) {
      auto xv = sme_ld1_x4(pn, x + i + k * 4 * lanes);
      sme_vector_write(k, sme_ld1_x4(pn, y + i + k * 4 * lanes));
      sme_vector_madd(k, xv, a);
    }
    EIGEN_SME_VECTOR_UNROLL4
    for (int k = 0; k < 4; ++k) svst1(pn, y + i + k * 4 * lanes, sme_vector_read(k, Scalar(0)));
  }
  for (; i <= n - 4 * lanes; i += 4 * lanes) {
    auto xv = sme_ld1_x4(pn, x + i);
    sme_vector_write(0, sme_ld1_x4(pn, y + i));
    sme_vector_madd(0, xv, a);
    svst1(pn, y + i, sme_vector_read(0, Scalar(0)));
  }
  for (; i < n; i += n - i < lanes ? n - i : lanes) {
    auto tail = Traits::whilelt(i, n);
    sme_st1(tail, y + i, svmla_x(tail, sme_ld1(tail, y + i), sme_ld1(tail, x + i), alpha));
  }
}

template <typename Scalar, typename Index>
__arm_new("za") __arm_locally_streaming EIGEN_DONT_INLINE Scalar sme_dot(Index n, const Scalar* x, const Scalar* y) {
  using Traits = sme_traits<Scalar>;
  const Index lanes = Traits::svl();
  const auto pg = Traits::ptrue();
  const auto pn = Traits::ptrue_c();
  // -0 is the additive identity except under roundTowardNegative, so a zero sum gets the IEEE sign.
  const auto negative_zero = Traits::dup(Scalar(-0.0));
  EIGEN_SME_VECTOR_UNROLL4
  for (int k = 0; k < 4; ++k)
    sme_vector_write(k, svcreate4(negative_zero, negative_zero, negative_zero, negative_zero));
  Index i = 0;
  for (; i <= n - 16 * lanes; i += 16 * lanes) {
    EIGEN_SME_VECTOR_UNROLL4
    for (int k = 0; k < 4; ++k)
      sme_vector_madd(k, sme_ld1_x4(pn, x + i + k * 4 * lanes), sme_ld1_x4(pn, y + i + k * 4 * lanes));
  }
  for (; i <= n - 4 * lanes; i += 4 * lanes) sme_vector_madd(0, sme_ld1_x4(pn, x + i), sme_ld1_x4(pn, y + i));
  auto tail_accumulator = negative_zero;
  for (; i < n; i += n - i < lanes ? n - i : lanes) {
    auto tail = Traits::whilelt(i, n);
    tail_accumulator = svmla_m(tail, tail_accumulator, sme_ld1(tail, x + i), sme_ld1(tail, y + i));
  }
  auto sum = tail_accumulator;
  EIGEN_SME_VECTOR_UNROLL4
  for (int k = 0; k < 4; ++k) {
    auto v = sme_vector_read(k, Scalar(0));
    sum = svadd_x(pg, sum,
                  svadd_x(pg, svadd_x(pg, sme_get<0>(v), sme_get<1>(v)), svadd_x(pg, sme_get<2>(v), sme_get<3>(v))));
  }
  return svaddv(pg, sum);
}

// y += alpha * ZA group k, updated in ZA like the accumulation.
template <typename Scalar, typename Vec>
static EIGEN_ALWAYS_INLINE void sme_gemv_update(unsigned int k, svcount_t active, Scalar* y,
                                                Vec alpha) __arm_streaming __arm_inout("za") {
  const auto sum = sme_vector_read(k, Scalar(0));
  sme_vector_write(k, sme_ld1_x4(active, y));
  sme_vector_madd(k, sum, alpha);
  svst1(active, y, sme_vector_read(k, Scalar(0)));
}

template <typename Scalar, typename Index>
__arm_new("za") __arm_locally_streaming
    EIGEN_DONT_INLINE void sme_gemv(Index rows, Index cols, const Scalar* a, Index stride, const Scalar* x, Scalar* y,
                                    Scalar alpha, Index block_cols) {
  using Traits = sme_traits<Scalar>;
  const Index lanes = Traits::svl();
  const auto scale = Traits::dup(alpha);
  // Match the generic GEMV's scaled column batches to bound the unscaled sums.
  for (Index first = 0; first < cols;) {
    const Index end = first + (cols - first < block_cols ? cols - first : block_cols);
    // Four predicated chunks per row block keep four independent FMLA chains, also in the final partial block.
    // An inactive chunk addresses the block start and accesses no memory.
    for (Index i = 0; i < rows;) {
      const Index remaining = rows - i;
      const auto p0 = Traits::whilelt_c4(i, rows), p1 = Traits::whilelt_c4(i + std::int64_t(4) * lanes, rows),
                 p2 = Traits::whilelt_c4(i + std::int64_t(8) * lanes, rows),
                 p3 = Traits::whilelt_c4(i + std::int64_t(12) * lanes, rows);
      const Index o1 = remaining > 4 * lanes ? 4 * lanes : 0, o2 = remaining > 8 * lanes ? 8 * lanes : 0,
                  o3 = remaining > 12 * lanes ? 12 * lanes : 0;
      svzero_za();
      for (Index j = first; j < end; ++j) {
        const auto b = Traits::dup(x[j]);
        const Scalar* column = a + i + j * stride;
        sme_vector_madd(0, sme_ld1_x4(p0, column), b);
        sme_vector_madd(1, sme_ld1_x4(p1, column + o1), b);
        sme_vector_madd(2, sme_ld1_x4(p2, column + o2), b);
        sme_vector_madd(3, sme_ld1_x4(p3, column + o3), b);
      }
      sme_gemv_update(0, p0, y + i, scale);
      sme_gemv_update(1, p1, y + i + o1, scale);
      sme_gemv_update(2, p2, y + i + o2, scale);
      sme_gemv_update(3, p3, y + i + o3, scale);
      // Bound the final increment to avoid signed overflow with a 32-bit Index near its maximum.
      i += remaining < 16 * lanes ? remaining : 16 * lanes;
    }
    first = end;
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
      sme_vector_fpsr status;
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
  sme_vector_fpsr status;
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
      if (sme_gemv_size_suitable<Scalar>(rows, cols) && rhs.stride() == 1 && resIncr == 1) {                     \
        const std::ptrdiff_t l1 = l1CacheSize();                                                                 \
        const Index block_cols = cols < 128 ? cols                                                               \
                                            : (l1 > 0 && static_cast<std::size_t>(lhs.stride()) <=               \
                                                             (static_cast<std::size_t>(l1) - 1) / sizeof(Scalar) \
                                                   ? Index(16)                                                   \
                                                   : Index(4));                                                  \
        /* BLAS contract: alpha == 0 leaves the result unchanged. */                                             \
        if (alpha == Scalar(0)) return;                                                                          \
        sme_vector_fpsr status;                                                                                  \
        sme_gemv(rows, cols, lhs.data(), lhs.stride(), rhs.data(), res, alpha, block_cols);                      \
      } else {                                                                                                   \
        general_matrix_vector_product<Index, Scalar, const_blas_data_mapper<Scalar, Index, ColMajor>, ColMajor,  \
                                      ConjugateLhs, Scalar, const_blas_data_mapper<Scalar, Index, RowMajor>,     \
                                      ConjugateRhs, BuiltIn>::run(rows, cols, lhs, rhs, res, resIncr, alpha);    \
      }                                                                                                          \
    }                                                                                                            \
  };
EIGEN_SME_GEMV_SPECIALIZATION(float)
#ifdef EIGEN_VECTORIZE_SME_F64F64
EIGEN_SME_GEMV_SPECIALIZATION(double)
#endif
#undef EIGEN_SME_GEMV_SPECIALIZATION
#endif

}  // namespace internal
}  // namespace Eigen

#endif
