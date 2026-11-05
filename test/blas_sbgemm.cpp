// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Runs the EIGEN_USE_OPENBLAS_BFLOAT16 specialization of the bfloat16 GEMM against the reference sbgemm defined below,
// so that it needs no OpenBLAS built with BUILD_BFLOAT16 (#3215). Part 2 repeats it with row-major default storage.
#if defined(EIGEN_TEST_PART_2) && !defined(EIGEN_DEFAULT_TO_ROW_MAJOR)
#define EIGEN_DEFAULT_TO_ROW_MAJOR
#endif
#ifndef EIGEN_USE_BLAS
#define EIGEN_USE_BLAS
#endif
#ifndef EIGEN_USE_OPENBLAS_BFLOAT16
#define EIGEN_USE_OPENBLAS_BFLOAT16 1
#endif

#include "main.h"

static int sbgemm_calls = 0;

// C = alpha * op(A) * op(B) + beta * C with column-major operands, summed in float.
extern "C" void EIGEN_BLAS_SYM(sbgemm)(const char* trans_a, const char* trans_b, const EIGEN_BLAS_INT* M,
                                       const EIGEN_BLAS_INT* N, const EIGEN_BLAS_INT* K, const float* alpha,
                                       const Eigen::bfloat16* A, const EIGEN_BLAS_INT* lda, const Eigen::bfloat16* B,
                                       const EIGEN_BLAS_INT* ldb, const float* beta, float* C,
                                       const EIGEN_BLAS_INT* ldc) {
  ++sbgemm_calls;
  for (EIGEN_BLAS_INT j = 0; j < *N; ++j)
    for (EIGEN_BLAS_INT i = 0; i < *M; ++i) {
      float sum = 0;
      for (EIGEN_BLAS_INT l = 0; l < *K; ++l) {
        const float a = static_cast<float>(*trans_a == 'N' ? A[i + l * *lda] : A[l + i * *lda]);
        const float b = static_cast<float>(*trans_b == 'N' ? B[l + j * *ldb] : B[j + l * *ldb]);
        sum += a * b;
      }
      C[i + j * *ldc] = *alpha * sum + *beta * C[i + j * *ldc];
    }
}

// Small integer entries keep every float sum exact, so the one rounding to bfloat16 is the only error allowed.
template <typename MatrixType>
MatrixType sbgemm_operand(Index rows, Index cols, int max_entry) {
  MatrixType m(rows, cols);
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i) m(i, j) = bfloat16(float(internal::random<int>(0, max_entry)));
  return m;
}

// The bfloat16 rounding of c0 + alpha * lhs * rhs, computed without a GEMM, which EIGEN_USE_BLAS would send to BLAS.
template <typename Dst, typename Lhs, typename Rhs>
Dst sbgemm_expected(const Dst& c0, const Lhs& lhs, const Rhs& rhs, float alpha) {
  Dst expected(c0.rows(), c0.cols());
  for (Index j = 0; j < c0.cols(); ++j)
    for (Index i = 0; i < c0.rows(); ++i) {
      float sum = 0;
      for (Index l = 0; l < lhs.cols(); ++l) sum += float(lhs(i, l)) * float(rhs(l, j));
      expected(i, j) = bfloat16(float(c0(i, j)) + alpha * sum);
    }
  return expected;
}

template <int LhsOrder, int RhsOrder, int DstOrder>
void blas_sbgemm(Index rows, Index depth, Index cols) {
  using Lhs = Matrix<bfloat16, Dynamic, Dynamic, LhsOrder>;
  using Rhs = Matrix<bfloat16, Dynamic, Dynamic, RhsOrder>;
  using Dst = Matrix<bfloat16, Dynamic, Dynamic, DstOrder>;
  // The products exceed 256, where bfloat16 starts rounding integers, so adding c0 before or after rounding differs.
  const Lhs lhs = sbgemm_operand<Lhs>(rows, depth, 4);
  const Rhs rhs = sbgemm_operand<Rhs>(depth, cols, 4);
  const Dst c0 = sbgemm_operand<Dst>(rows, cols, 7);
  const Dst zero = Dst::Zero(rows, cols);
  const Dst product = sbgemm_expected(zero, lhs, rhs, 1.f);

  int calls = sbgemm_calls;
  // A result that held garbage must not leak into an assignment.
  Dst dst = Dst::Constant(rows, cols, bfloat16(-7.f));
  dst.noalias() = lhs * rhs;
  VERIFY_IS_CWISE_EQUAL(dst, product);
  VERIFY(sbgemm_calls > calls);

  calls = sbgemm_calls;
  dst = c0;
  dst = lhs * rhs;
  VERIFY_IS_CWISE_EQUAL(dst, product);
  VERIFY(sbgemm_calls > calls);

  calls = sbgemm_calls;
  dst = c0;
  dst.noalias() += lhs * rhs;
  VERIFY_IS_CWISE_EQUAL(dst, sbgemm_expected(c0, lhs, rhs, 1.f));
  VERIFY(sbgemm_calls > calls);

  calls = sbgemm_calls;
  dst = c0;
  dst.noalias() -= bfloat16(2.f) * lhs * rhs;
  VERIFY_IS_CWISE_EQUAL(dst, sbgemm_expected(c0, lhs, rhs, -2.f));
  VERIFY(sbgemm_calls > calls);

  // A block destination has an outer stride larger than its inner size; the coefficients around it stay untouched.
  calls = sbgemm_calls;
  Dst big = Dst::Constant(rows + 3, cols + 2, bfloat16(-5.f));
  Dst big_expected = big;
  big.block(1, 2, rows, cols) = c0;
  big.block(1, 2, rows, cols).noalias() += lhs * rhs;
  big_expected.block(1, 2, rows, cols) = sbgemm_expected(c0, lhs, rhs, 1.f);
  VERIFY_IS_CWISE_EQUAL(big, big_expected);
  VERIFY(sbgemm_calls > calls);
}

template <int DstOrder>
void blas_sbgemm_orders(Index rows, Index depth, Index cols) {
  blas_sbgemm<ColMajor, ColMajor, DstOrder>(rows, depth, cols);
  blas_sbgemm<ColMajor, RowMajor, DstOrder>(rows, depth, cols);
  blas_sbgemm<RowMajor, ColMajor, DstOrder>(rows, depth, cols);
  blas_sbgemm<RowMajor, RowMajor, DstOrder>(rows, depth, cols);
}

EIGEN_DECLARE_TEST(blas_sbgemm) {
  // Distinct sizes, so that a transposed or misstrided result cannot pass.
  CALL_SUBTEST_1(blas_sbgemm_orders<ColMajor>(40, 300, 30));
  CALL_SUBTEST_1(blas_sbgemm_orders<RowMajor>(37, 64, 23));
  CALL_SUBTEST_2(blas_sbgemm_orders<ColMajor>(37, 64, 23));
  CALL_SUBTEST_2(blas_sbgemm_orders<RowMajor>(40, 300, 30));
}
