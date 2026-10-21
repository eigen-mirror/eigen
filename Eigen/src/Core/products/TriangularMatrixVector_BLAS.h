/*
 Copyright (c) 2011, Intel Corporation. All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of Intel Corporation nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 ********************************************************************************
 *   Content : Eigen bindings to BLAS F77
 *   Triangular matrix-vector product functionality based on ?TRMV.
 ********************************************************************************
*/
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EIGEN_TRIANGULAR_MATRIX_VECTOR_BLAS_H
#define EIGEN_TRIANGULAR_MATRIX_VECTOR_BLAS_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/**********************************************************************
 * This file implements triangular matrix-vector multiplication using BLAS
 **********************************************************************/

// trmv/hemv specialization

template <typename Index, int Mode, typename LhsScalar, bool ConjLhs, typename RhsScalar, bool ConjRhs,
          int StorageOrder>
struct triangular_matrix_vector_product_trmv
    : triangular_matrix_vector_product<Index, Mode, LhsScalar, ConjLhs, RhsScalar, ConjRhs, StorageOrder, BuiltIn> {};

template <typename Index, int Mode, typename Scalar, bool ConjLhs, bool ConjRhs, int Order>
void trmv_builtin(Index rows, Index cols, const Scalar* lhs, Index lhsStride, const Scalar* rhs, Index rhsIncr,
                  Scalar* res, Index resIncr, Scalar alpha) {
  using Vector = Matrix<Scalar, Dynamic, 1>;
  Vector rhs_copy, result;
  Scalar* output = res;
  Index output_stride = resIncr;
  const Index size = (std::min)(rows, cols);
  const Index output_size = (Mode & Lower) ? rows : size;
  // The row-major kernel needs contiguous input; the column-major kernel needs contiguous output.
  EIGEN_IF_CONSTEXPR (Order == RowMajor) {
    if (rhsIncr != 1) {
      rhs_copy = Map<const Vector, 0, InnerStride<>>(rhs, (Mode & Lower) ? size : cols, InnerStride<>(rhsIncr));
      rhs = rhs_copy.data();
      rhsIncr = 1;
    }
  } else {
    if (resIncr != 1) {
      result = Vector::Zero(output_size);
      output = result.data();
      output_stride = 1;
    }
  }
  triangular_matrix_vector_product<Index, Mode, Scalar, ConjLhs, Scalar, ConjRhs, Order, BuiltIn>::run(
      rows, cols, lhs, lhsStride, rhs, rhsIncr, output, output_stride, alpha);
  if (result.size() != 0) Map<Vector, 0, InnerStride<>>(res, output_size, InnerStride<>(resIncr)) += result;
}

#define EIGEN_BLAS_TRMV_SPECIALIZE(Scalar)                                                                            \
  template <typename Index, int Mode, bool ConjLhs, bool ConjRhs>                                                     \
  struct triangular_matrix_vector_product<Index, Mode, Scalar, ConjLhs, Scalar, ConjRhs, ColMajor, Specialized> {     \
    static void run(Index rows_, Index cols_, const Scalar* lhs_, Index lhsStride, const Scalar* rhs_, Index rhsIncr, \
                    Scalar* res_, Index resIncr, Scalar alpha) {                                                      \
      triangular_matrix_vector_product_trmv<Index, Mode, Scalar, ConjLhs, Scalar, ConjRhs, ColMajor>::run(            \
          rows_, cols_, lhs_, lhsStride, rhs_, rhsIncr, res_, resIncr, alpha);                                        \
    }                                                                                                                 \
  };                                                                                                                  \
  template <typename Index, int Mode, bool ConjLhs, bool ConjRhs>                                                     \
  struct triangular_matrix_vector_product<Index, Mode, Scalar, ConjLhs, Scalar, ConjRhs, RowMajor, Specialized> {     \
    static void run(Index rows_, Index cols_, const Scalar* lhs_, Index lhsStride, const Scalar* rhs_, Index rhsIncr, \
                    Scalar* res_, Index resIncr, Scalar alpha) {                                                      \
      triangular_matrix_vector_product_trmv<Index, Mode, Scalar, ConjLhs, Scalar, ConjRhs, RowMajor>::run(            \
          rows_, cols_, lhs_, lhsStride, rhs_, rhsIncr, res_, resIncr, alpha);                                        \
    }                                                                                                                 \
  };

EIGEN_BLAS_TRMV_SPECIALIZE(double)
EIGEN_BLAS_TRMV_SPECIALIZE(float)
EIGEN_BLAS_TRMV_SPECIALIZE(dcomplex)
EIGEN_BLAS_TRMV_SPECIALIZE(scomplex)

// implements col-major: res += alpha * op(triangular) * vector
#define EIGEN_BLAS_TRMV_CM(EIGTYPE, BLASTYPE, EIGPREFIX, BLASPREFIX, BLASPOSTFIX)                                    \
  template <typename Index, int Mode, bool ConjLhs, bool ConjRhs>                                                    \
  struct triangular_matrix_vector_product_trmv<Index, Mode, EIGTYPE, ConjLhs, EIGTYPE, ConjRhs, ColMajor> {          \
    using Scalar = EIGTYPE;                                                                                          \
    enum {                                                                                                           \
      IsLower = (Mode & Lower) == Lower,                                                                             \
      SetDiag = (Mode & (ZeroDiag | UnitDiag)) ? 0 : 1,                                                              \
      IsUnitDiag = (Mode & UnitDiag) ? 1 : 0,                                                                        \
      IsZeroDiag = (Mode & ZeroDiag) ? 1 : 0,                                                                        \
      LowUp = IsLower ? Lower : Upper                                                                                \
    };                                                                                                               \
    static void run(Index rows_, Index cols_, const EIGTYPE* lhs_, Index lhsStride, const EIGTYPE* rhs_,             \
                    Index rhsIncr, Scalar* res_, Index resIncr, EIGTYPE alpha) {                                     \
      if (rows_ == 0 || cols_ == 0 || numext::is_exactly_zero(alpha)) return;                                        \
      if (!blas_compatible_stride<ColMajor>(rows_, cols_, lhsStride) || (rows_ > 1 && resIncr == 0)) {               \
        trmv_builtin<Index, Mode, EIGTYPE, ConjLhs, ConjRhs, ColMajor>(rows_, cols_, lhs_, lhsStride, rhs_, rhsIncr, \
                                                                       res_, resIncr, alpha);                        \
        return;                                                                                                      \
      }                                                                                                              \
      EIGEN_IF_CONSTEXPR (ConjLhs || IsZeroDiag) {                                                                   \
        trmv_builtin<Index, Mode, EIGTYPE, ConjLhs, ConjRhs, ColMajor>(rows_, cols_, lhs_, lhsStride, rhs_, rhsIncr, \
                                                                       res_, resIncr, alpha);                        \
        return;                                                                                                      \
      }                                                                                                              \
      Index size = (std::min)(rows_, cols_);                                                                         \
      Index rows = IsLower ? rows_ : size;                                                                           \
      Index cols = IsLower ? size : cols_;                                                                           \
                                                                                                                     \
      typedef VectorX##EIGPREFIX VectorRhs;                                                                          \
      Scalar *x, *y;                                                                                                 \
                                                                                                                     \
      /* Set x*/                                                                                                     \
      Map<const VectorRhs, 0, InnerStride<>> rhs(rhs_, cols, InnerStride<>(rhsIncr));                                \
      VectorRhs x_tmp;                                                                                               \
      EIGEN_IF_CONSTEXPR (ConjRhs)                                                                                   \
        x_tmp = rhs.conjugate();                                                                                     \
      else                                                                                                           \
        x_tmp = rhs;                                                                                                 \
      x = x_tmp.data();                                                                                              \
                                                                                                                     \
      /* Square part handling */                                                                                     \
                                                                                                                     \
      char trans, uplo, diag;                                                                                        \
      BlasIndex m, n, lda, incx, incy;                                                                               \
      EIGTYPE const* a;                                                                                              \
      EIGTYPE beta(1);                                                                                               \
                                                                                                                     \
      /* Set m, n */                                                                                                 \
      n = convert_index<BlasIndex>(size);                                                                            \
      lda = blas_leading_dim<ColMajor>(rows, cols, lhsStride);                                                       \
      incx = 1;                                                                                                      \
      incy = blas_vector_increment(rows, resIncr);                                                                   \
                                                                                                                     \
      /* Set uplo, trans and diag*/                                                                                  \
      trans = 'N';                                                                                                   \
      uplo = IsLower ? 'L' : 'U';                                                                                    \
      diag = IsUnitDiag ? 'U' : 'N';                                                                                 \
                                                                                                                     \
      /* call ?TRMV*/                                                                                                \
      EIGEN_CAT(EIGEN_CAT(BLASPREFIX, trmv), BLASPOSTFIX)                                                            \
      (&uplo, &trans, &diag, &n, (const BLASTYPE*)lhs_, &lda, (BLASTYPE*)x, &incx);                                  \
                                                                                                                     \
      /* Add op(a_tr)rhs into res*/                                                                                  \
      /* BLAS addresses a vector with a negative increment from its lowest address, Eigen from entry 0. */           \
      EIGEN_CAT(EIGEN_CAT(BLASPREFIX, axpy), BLASPOSTFIX)                                                            \
      (&n, (const BLASTYPE*)&numext::real_ref(alpha), (const BLASTYPE*)x, &incx,                                     \
       (BLASTYPE*)(resIncr < 0 ? res_ + (size - 1) * resIncr : res_), &incy);                                        \
      /* Non-square case - doesn't fit to BLAS ?TRMV. Fall to default triangular product*/                           \
      if (size < (std::max)(rows, cols)) {                                                                           \
        EIGEN_IF_CONSTEXPR (ConjRhs)                                                                                 \
          x_tmp = rhs.conjugate();                                                                                   \
        else                                                                                                         \
          x_tmp = rhs;                                                                                               \
        x = x_tmp.data();                                                                                            \
        if (size < rows) {                                                                                           \
          y = res_ + size * resIncr;                                                                                 \
          a = lhs_ + size;                                                                                           \
          m = convert_index<BlasIndex>(rows - size);                                                                 \
          n = convert_index<BlasIndex>(size);                                                                        \
        } else {                                                                                                     \
          x += size;                                                                                                 \
          y = res_;                                                                                                  \
          a = lhs_ + size * lda;                                                                                     \
          m = convert_index<BlasIndex>(size);                                                                        \
          n = convert_index<BlasIndex>(cols - size);                                                                 \
        }                                                                                                            \
        if (resIncr < 0) y += (m - 1) * resIncr;                                                                     \
        EIGEN_CAT(EIGEN_CAT(BLASPREFIX, gemv), BLASPOSTFIX)                                                          \
        (&trans, &m, &n, (const BLASTYPE*)&numext::real_ref(alpha), (const BLASTYPE*)a, &lda, (const BLASTYPE*)x,    \
         &incx, (const BLASTYPE*)&numext::real_ref(beta), (BLASTYPE*)y, &incy);                                      \
      }                                                                                                              \
    }                                                                                                                \
  };

#ifdef EIGEN_USE_MKL
EIGEN_BLAS_TRMV_CM(double, double, d, d, )
EIGEN_BLAS_TRMV_CM(dcomplex, MKL_Complex16, cd, z, )
EIGEN_BLAS_TRMV_CM(float, float, f, s, )
EIGEN_BLAS_TRMV_CM(scomplex, MKL_Complex8, cf, c, )
#else
EIGEN_BLAS_TRMV_CM(double, double, d, d, EIGEN_BLAS_POSTFIX)
EIGEN_BLAS_TRMV_CM(dcomplex, double, cd, z, EIGEN_BLAS_POSTFIX)
EIGEN_BLAS_TRMV_CM(float, float, f, s, EIGEN_BLAS_POSTFIX)
EIGEN_BLAS_TRMV_CM(scomplex, float, cf, c, EIGEN_BLAS_POSTFIX)
#endif

// implements row-major: res += alpha * op(triangular) * vector
#define EIGEN_BLAS_TRMV_RM(EIGTYPE, BLASTYPE, EIGPREFIX, BLASPREFIX, BLASPOSTFIX)                                    \
  template <typename Index, int Mode, bool ConjLhs, bool ConjRhs>                                                    \
  struct triangular_matrix_vector_product_trmv<Index, Mode, EIGTYPE, ConjLhs, EIGTYPE, ConjRhs, RowMajor> {          \
    using Scalar = EIGTYPE;                                                                                          \
    enum {                                                                                                           \
      IsLower = (Mode & Lower) == Lower,                                                                             \
      SetDiag = (Mode & (ZeroDiag | UnitDiag)) ? 0 : 1,                                                              \
      IsUnitDiag = (Mode & UnitDiag) ? 1 : 0,                                                                        \
      IsZeroDiag = (Mode & ZeroDiag) ? 1 : 0,                                                                        \
      LowUp = IsLower ? Lower : Upper                                                                                \
    };                                                                                                               \
    static void run(Index rows_, Index cols_, const EIGTYPE* lhs_, Index lhsStride, const EIGTYPE* rhs_,             \
                    Index rhsIncr, Scalar* res_, Index resIncr, EIGTYPE alpha) {                                     \
      if (rows_ == 0 || cols_ == 0 || numext::is_exactly_zero(alpha)) return;                                        \
      if (!blas_compatible_stride<RowMajor>(rows_, cols_, lhsStride) || (rows_ > 1 && resIncr == 0)) {               \
        trmv_builtin<Index, Mode, EIGTYPE, ConjLhs, ConjRhs, RowMajor>(rows_, cols_, lhs_, lhsStride, rhs_, rhsIncr, \
                                                                       res_, resIncr, alpha);                        \
        return;                                                                                                      \
      }                                                                                                              \
      EIGEN_IF_CONSTEXPR (IsZeroDiag) {                                                                              \
        trmv_builtin<Index, Mode, EIGTYPE, ConjLhs, ConjRhs, RowMajor>(rows_, cols_, lhs_, lhsStride, rhs_, rhsIncr, \
                                                                       res_, resIncr, alpha);                        \
        return;                                                                                                      \
      }                                                                                                              \
      Index size = (std::min)(rows_, cols_);                                                                         \
      Index rows = IsLower ? rows_ : size;                                                                           \
      Index cols = IsLower ? size : cols_;                                                                           \
                                                                                                                     \
      typedef VectorX##EIGPREFIX VectorRhs;                                                                          \
      Scalar *x, *y;                                                                                                 \
                                                                                                                     \
      /* Set x*/                                                                                                     \
      Map<const VectorRhs, 0, InnerStride<>> rhs(rhs_, cols, InnerStride<>(rhsIncr));                                \
      VectorRhs x_tmp;                                                                                               \
      EIGEN_IF_CONSTEXPR (ConjRhs)                                                                                   \
        x_tmp = rhs.conjugate();                                                                                     \
      else                                                                                                           \
        x_tmp = rhs;                                                                                                 \
      x = x_tmp.data();                                                                                              \
                                                                                                                     \
      /* Square part handling */                                                                                     \
                                                                                                                     \
      char trans, uplo, diag;                                                                                        \
      BlasIndex m, n, lda, incx, incy;                                                                               \
      EIGTYPE const* a;                                                                                              \
      EIGTYPE beta(1);                                                                                               \
                                                                                                                     \
      /* Set m, n */                                                                                                 \
      n = convert_index<BlasIndex>(size);                                                                            \
      lda = blas_leading_dim<RowMajor>(rows, cols, lhsStride);                                                       \
      incx = 1;                                                                                                      \
      incy = blas_vector_increment(rows, resIncr);                                                                   \
                                                                                                                     \
      /* Set uplo, trans and diag*/                                                                                  \
      trans = ConjLhs ? 'C' : 'T';                                                                                   \
      uplo = IsLower ? 'U' : 'L';                                                                                    \
      diag = IsUnitDiag ? 'U' : 'N';                                                                                 \
                                                                                                                     \
      /* call ?TRMV*/                                                                                                \
      EIGEN_CAT(EIGEN_CAT(BLASPREFIX, trmv), BLASPOSTFIX)                                                            \
      (&uplo, &trans, &diag, &n, (const BLASTYPE*)lhs_, &lda, (BLASTYPE*)x, &incx);                                  \
                                                                                                                     \
      /* Add op(a_tr)rhs into res*/                                                                                  \
      /* BLAS addresses a vector with a negative increment from its lowest address, Eigen from entry 0. */           \
      EIGEN_CAT(EIGEN_CAT(BLASPREFIX, axpy), BLASPOSTFIX)                                                            \
      (&n, (const BLASTYPE*)&numext::real_ref(alpha), (const BLASTYPE*)x, &incx,                                     \
       (BLASTYPE*)(resIncr < 0 ? res_ + (size - 1) * resIncr : res_), &incy);                                        \
      /* Non-square case - doesn't fit to BLAS ?TRMV. Fall to default triangular product*/                           \
      if (size < (std::max)(rows, cols)) {                                                                           \
        EIGEN_IF_CONSTEXPR (ConjRhs)                                                                                 \
          x_tmp = rhs.conjugate();                                                                                   \
        else                                                                                                         \
          x_tmp = rhs;                                                                                               \
        x = x_tmp.data();                                                                                            \
        if (size < rows) {                                                                                           \
          y = res_ + size * resIncr;                                                                                 \
          a = lhs_ + size * lda;                                                                                     \
          m = convert_index<BlasIndex>(rows - size);                                                                 \
          n = convert_index<BlasIndex>(size);                                                                        \
        } else {                                                                                                     \
          x += size;                                                                                                 \
          y = res_;                                                                                                  \
          a = lhs_ + size;                                                                                           \
          m = convert_index<BlasIndex>(size);                                                                        \
          n = convert_index<BlasIndex>(cols - size);                                                                 \
        }                                                                                                            \
        if (resIncr < 0) y += (m - 1) * resIncr;                                                                     \
        EIGEN_CAT(EIGEN_CAT(BLASPREFIX, gemv), BLASPOSTFIX)                                                          \
        (&trans, &n, &m, (const BLASTYPE*)&numext::real_ref(alpha), (const BLASTYPE*)a, &lda, (const BLASTYPE*)x,    \
         &incx, (const BLASTYPE*)&numext::real_ref(beta), (BLASTYPE*)y, &incy);                                      \
      }                                                                                                              \
    }                                                                                                                \
  };

#ifdef EIGEN_USE_MKL
EIGEN_BLAS_TRMV_RM(double, double, d, d, )
EIGEN_BLAS_TRMV_RM(dcomplex, MKL_Complex16, cd, z, )
EIGEN_BLAS_TRMV_RM(float, float, f, s, )
EIGEN_BLAS_TRMV_RM(scomplex, MKL_Complex8, cf, c, )
#else
EIGEN_BLAS_TRMV_RM(double, double, d, d, EIGEN_BLAS_POSTFIX)
EIGEN_BLAS_TRMV_RM(dcomplex, double, cd, z, EIGEN_BLAS_POSTFIX)
EIGEN_BLAS_TRMV_RM(float, float, f, s, EIGEN_BLAS_POSTFIX)
EIGEN_BLAS_TRMV_RM(scomplex, float, cf, c, EIGEN_BLAS_POSTFIX)
#endif

#undef EIGEN_BLAS_TRMV_RM
#undef EIGEN_BLAS_TRMV_SPECIALIZE
#undef EIGEN_BLAS_TRMV_CM
}  // namespace internal

}  // end namespace Eigen

#endif  // EIGEN_TRIANGULAR_MATRIX_VECTOR_BLAS_H
