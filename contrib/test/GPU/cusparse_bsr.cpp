// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Tests for gpu::SparseContext with BlockSparseMatrix input: GPU BSR SpMV/SpMM
// via cuSPARSE.

#define EIGEN_USE_GPU
#include "main.h"
#include <Eigen/Sparse>
#include <contrib/Eigen/GPU>
#include "gpu_test_helpers.h"

using namespace Eigen;

#if EIGEN_HAS_CUSPARSE_BSR

template <typename Scalar, int Options, int B>
using Bsm = BlockSparseMatrix<Scalar, Options, B, B, int>;

// Random block pattern with random dense blocks; Random() fills both parts of
// a complex scalar, so conjugation is observable.
template <typename Scalar, int Options, int B>
Bsm<Scalar, Options, B> make_block_sparse(Index block_rows, Index block_cols, double density = 0.3) {
  using Triplet = typename Bsm<Scalar, Options, B>::TripletType;
  using Block = typename Bsm<Scalar, Options, B>::BlockType;
  std::vector<Triplet> triplets;
  for (Index bi = 0; bi < block_rows; ++bi) {
    for (Index bj = 0; bj < block_cols; ++bj) {
      if (internal::random<double>(0.0, 1.0) < density) triplets.emplace_back(int(bi), int(bj), Block::Random());
    }
  }
  Bsm<Scalar, Options, B> A(block_rows, block_cols);
  A.setFromTriplets(triplets.begin(), triplets.end());
  return A;
}

// Relative bound for a product by A: each entry sums at most
// max(A.rows(), A.cols()) terms, so 10 * max(A.rows(), A.cols()) * eps covers
// the accumulation error.
template <typename Lhs, typename Rhs, typename MatrixType>
void verify_product(const Lhs& gpu, const Rhs& cpu, const MatrixType& A) {
  using Scalar = typename Lhs::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  const RealScalar tol = RealScalar(10) * RealScalar((std::max)(A.rows(), A.cols())) * NumTraits<Scalar>::epsilon();
  VERIFY_IS_EQUAL(gpu.rows(), cpu.rows());
  VERIFY_IS_EQUAL(gpu.cols(), cpu.cols());
  VERIFY((gpu - cpu).norm() / (cpu.norm() + RealScalar(1)) < tol);
}

// ---- Host SpMV: y = op(A) * x for every op -----------------------------------

template <typename Scalar, int Options, int B>
void test_bsr_spmv(Index block_rows, Index block_cols) {
  using Vec = Matrix<Scalar, Dynamic, 1>;

  const Bsm<Scalar, Options, B> A = make_block_sparse<Scalar, Options, B>(block_rows, block_cols);
  const Vec x = Vec::Random(A.cols());
  const Vec xt = Vec::Random(A.rows());

  gpu::SparseContext<Scalar> ctx;
  const Vec y = ctx.multiply(A, x);
  verify_product(y, Vec(A * x), A);
  const Vec yt = ctx.multiplyT(A, xt);
  verify_product(yt, Vec(A.transpose() * xt), A);
  const Vec yh = ctx.multiplyAdjoint(A, xt);
  verify_product(yh, Vec(A.adjoint() * xt), A);
}

// ---- Host SpMV in place: y = alpha * op(A) * x + beta * y ---------------------

template <typename Scalar, int Options, int B>
void test_bsr_spmv_alpha_beta(Index block_rows, Index block_cols) {
  using Vec = Matrix<Scalar, Dynamic, 1>;

  const Bsm<Scalar, Options, B> A = make_block_sparse<Scalar, Options, B>(block_rows, block_cols);
  const Scalar alpha(2);
  const Scalar beta(3);
  gpu::SparseContext<Scalar> ctx;

  const Vec x = Vec::Random(A.cols());
  const Vec y_init = Vec::Random(A.rows());
  Vec y = y_init;
  ctx.multiply(A, x, y, alpha, beta);
  verify_product(y, Vec(alpha * (A * x) + beta * y_init), A);

  const Vec xt = Vec::Random(A.rows());
  const Vec yt_init = Vec::Random(A.cols());
  Vec yt = yt_init;
  ctx.multiply(A, xt, yt, alpha, beta, gpu::GpuOp::ConjTrans);
  verify_product(yt, Vec(alpha * (A.adjoint() * xt) + beta * yt_init), A);
}

// ---- Host SpMM: Y = op(A) * X -------------------------------------------------

template <typename Scalar, int Options, int B>
void test_bsr_spmm(Index block_rows, Index block_cols, Index nrhs) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;

  const Bsm<Scalar, Options, B> A = make_block_sparse<Scalar, Options, B>(block_rows, block_cols);
  const Mat X = Mat::Random(A.cols(), nrhs);
  const Mat Xt = Mat::Random(A.rows(), nrhs);

  gpu::SparseContext<Scalar> ctx;
  verify_product(ctx.multiplyMat(A, X), Mat(A * X), A);
  verify_product(ctx.multiplyMat(A, Xt, gpu::GpuOp::Trans), Mat(A.transpose() * Xt), A);
  verify_product(ctx.multiplyMat(A, Xt, gpu::GpuOp::ConjTrans), Mat(A.adjoint() * Xt), A);
}

// ---- DeviceMatrix in/out with an explicit op ----------------------------------

template <typename Scalar, int Options, int B>
void test_bsr_device_multiply(Index block_rows, Index block_cols) {
  using Vec = Matrix<Scalar, Dynamic, 1>;

  const Bsm<Scalar, Options, B> A = make_block_sparse<Scalar, Options, B>(block_rows, block_cols);
  gpu::Context gctx;
  gpu::SparseContext<Scalar> ctx(gctx);

  const Vec x = Vec::Random(A.cols());
  auto d_x = gpu::DeviceMatrix<Scalar>::fromHost(x, gctx.stream());
  gpu::DeviceMatrix<Scalar> d_y;
  ctx.multiply(A, d_x, d_y);
  verify_product(d_y.toHost(gctx.stream()), Vec(A * x), A);

  const Vec xt = Vec::Random(A.rows());
  auto d_xt = gpu::DeviceMatrix<Scalar>::fromHost(xt, gctx.stream());
  gpu::DeviceMatrix<Scalar> d_yt;
  ctx.multiply(A, d_xt, d_yt, Scalar(1), Scalar(0), gpu::GpuOp::Trans);
  verify_product(d_yt.toHost(gctx.stream()), Vec(A.transpose() * xt), A);
  ctx.multiply(A, d_xt, d_yt, Scalar(1), Scalar(0), gpu::GpuOp::ConjTrans);
  verify_product(d_yt.toHost(gctx.stream()), Vec(A.adjoint() * xt), A);
}

// ---- deviceView: upload once, SpMV and SpMM by expression ----------------------

template <typename Scalar, int Options, int B>
void test_bsr_device_view(Index block_rows, Index block_cols, Index nrhs) {
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;

  const Bsm<Scalar, Options, B> A = make_block_sparse<Scalar, Options, B>(block_rows, block_cols);
  gpu::Context gctx;
  gpu::SparseContext<Scalar> ctx(gctx);
  auto view = ctx.deviceView(A);
  VERIFY(view.generation() == ctx.uploadGeneration());
  VERIFY_IS_EQUAL(view.rows(), A.rows());
  VERIFY_IS_EQUAL(view.cols(), A.cols());

  const Vec x = Vec::Random(A.cols());
  auto d_x = gpu::DeviceMatrix<Scalar>::fromHost(x, gctx.stream());
  gpu::DeviceMatrix<Scalar> d_y = view * d_x;
  verify_product(d_y.toHost(gctx.stream()), Vec(A * x), A);

  const Mat X = Mat::Random(A.cols(), nrhs);
  auto d_X = gpu::DeviceMatrix<Scalar>::fromHost(X, gctx.stream());
  gpu::DeviceMatrix<Scalar> d_Y;
  d_Y.noalias() = view * d_X;
  verify_product(d_Y.toHost(gctx.stream()), Mat(A * X), A);

  // cuSPARSE runs no transposed BSR product; the exec entry points reject any
  // other op against a BSR upload before queuing work.
  VERIFY_RAISES_ASSERT(ctx.spmv_device_exec(d_x, d_y, Scalar(1), Scalar(0), gpu::GpuOp::Trans));
  VERIFY_RAISES_ASSERT(ctx.spmm_device_exec(d_X, d_Y, Scalar(1), Scalar(0), gpu::GpuOp::Trans));
}

// ---- Descriptor cache across formats ------------------------------------------

// A CSC matrix with the same shape and nonzero count as a BSR one must not
// reuse its descriptor: S = (A^T as scalar sparse) has both, and a different
// value for every x unless A is symmetric.
template <typename Scalar, int Options, int B>
void test_bsr_format_switch(Index block_n) {
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using SpMat = SparseMatrix<Scalar, ColMajor, int>;

  const Bsm<Scalar, Options, B> A = make_block_sparse<Scalar, Options, B>(block_n, block_n);
  const SpMat S = SpMat(A.transpose().toSparse());
  VERIFY_IS_EQUAL(S.nonZeros(), A.nonZeros());
  const Vec x = Vec::Random(A.cols());

  gpu::SparseContext<Scalar> ctx;
  verify_product(ctx.multiply(A, x), Vec(A * x), A);
  verify_product(ctx.multiply(S, x), Vec(S * x), A);
  verify_product(ctx.multiply(A, x), Vec(A * x), A);
}

// ---- Pattern rewrite at unchanged host pointers -------------------------------

// The index arrays are re-uploaded on every host-input call, so rewriting the
// block-column indices in place is picked up. Blocks sit at (i, i) in an
// n x (n + 1) grid and move to (i, i + 1), which keeps every block row sorted.
template <typename Scalar, int Options, int B>
void test_bsr_pattern_rewrite(Index n) {
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using Triplet = typename Bsm<Scalar, Options, B>::TripletType;
  using Block = typename Bsm<Scalar, Options, B>::BlockType;

  std::vector<Triplet> triplets;
  for (Index i = 0; i < n; ++i) triplets.emplace_back(int(i), int(i), Block::Random());
  Bsm<Scalar, Options, B> A(n, n + 1);
  A.setFromTriplets(triplets.begin(), triplets.end());
  const Vec x = Vec::Random(A.cols());

  gpu::SparseContext<Scalar> ctx;
  verify_product(ctx.multiply(A, x), Vec(A * x), A);

  // RowMajor: inner index = block column, shift it. ColMajor: inner index =
  // block row; shifting the *outer* boundaries instead moves block i to
  // column i + 1, so both orders end at the same pattern.
  if (Options & RowMajor) {
    for (Index k = 0; k < A.nonZeroBlocks(); ++k) A.innerIndexPtr()[k] += 1;
  } else {
    for (Index j = n; j >= 1; --j) A.outerIndexPtr()[j] = A.outerIndexPtr()[j - 1];
    A.outerIndexPtr()[0] = 0;
    A.outerIndexPtr()[n + 1] = int(n);
  }
  VERIFY_IS_EQUAL(A.nonZeroBlocks(), n);
  verify_product(ctx.multiply(A, x), Vec(A * x), A);
}

// ---- Empty matrices ------------------------------------------------------------

template <typename Scalar, int Options, int B>
void test_bsr_empty() {
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;

  gpu::Context gctx;
  gpu::SparseContext<Scalar> ctx(gctx);

  const Bsm<Scalar, Options, B> A0(0, 0);
  VERIFY_IS_EQUAL(ctx.multiply(A0, Vec(0)).size(), 0);
  VERIFY_IS_EQUAL(ctx.multiplyMat(A0, Mat(0, 3)).cols(), 3);

  // Nonzero dimensions, no stored blocks: y <- beta * y on the host and
  // y <- 0 on the device.
  const Bsm<Scalar, Options, B> A(3, 2);
  const Vec x = Vec::Random(A.cols());
  const Vec y_init = Vec::Random(A.rows());
  Vec y = y_init;
  ctx.multiply(A, x, y, Scalar(1), Scalar(2));
  VERIFY_IS_APPROX(y, Vec(Scalar(2) * y_init));
  VERIFY(ctx.multiply(A, x).isZero());

  auto d_x = gpu::DeviceMatrix<Scalar>::fromHost(x, gctx.stream());
  gpu::DeviceMatrix<Scalar> d_y;
  ctx.multiply(A, d_x, d_y);
  VERIFY_IS_EQUAL(d_y.rows(), A.rows());
  VERIFY(d_y.toHost(gctx.stream()).isZero());
}

// ---- Driver ---------------------------------------------------------------------

template <typename Scalar, int Options, int B>
void test_order() {
  CALL_SUBTEST((test_bsr_spmv<Scalar, Options, B>(7, 5)));
  CALL_SUBTEST((test_bsr_spmv_alpha_beta<Scalar, Options, B>(5, 7)));
  CALL_SUBTEST((test_bsr_spmm<Scalar, Options, B>(6, 4, 3)));
  CALL_SUBTEST((test_bsr_device_multiply<Scalar, Options, B>(4, 6)));
  CALL_SUBTEST((test_bsr_device_view<Scalar, Options, B>(6, 6, 4)));
  CALL_SUBTEST((test_bsr_format_switch<Scalar, Options, B>(5)));
  CALL_SUBTEST((test_bsr_pattern_rewrite<Scalar, Options, B>(4)));
  CALL_SUBTEST((test_bsr_empty<Scalar, Options, B>()));
}

template <typename Scalar>
void test_scalar() {
  // RowMajor binds without a host copy for NoTrans, ColMajor for Trans; the
  // other op / order combinations take the transposing copy.
  test_order<Scalar, RowMajor, 2>();
  test_order<Scalar, ColMajor, 2>();
  test_order<Scalar, RowMajor, 3>();
  test_order<Scalar, ColMajor, 3>();
}

#endif  // EIGEN_HAS_CUSPARSE_BSR

EIGEN_DECLARE_TEST(gpu_cusparse_bsr) {
#if EIGEN_HAS_CUSPARSE_BSR
  gpu_test::require_cusparse_context();

  // Split by scalar so each part compiles in parallel.
  CALL_SUBTEST_1(test_scalar<float>());
  CALL_SUBTEST_2(test_scalar<double>());
  CALL_SUBTEST_3(test_scalar<std::complex<float>>());
  CALL_SUBTEST_4(test_scalar<std::complex<double>>());
  // Block size 4, in one scalar type to bound compile time.
  CALL_SUBTEST_5((test_order<double, RowMajor, 4>()));
  CALL_SUBTEST_5((test_order<double, ColMajor, 4>()));
#else
  std::cout << "SKIP: BSR products need cuSPARSE >= 12.6.3 (CUDA 13.0 Update 1); this build has CUSPARSE_VERSION "
            << CUSPARSE_VERSION << std::endl;
  std::exit(77);
#endif
}
