// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "sparse.h"

using namespace Eigen;

// ---------------------------------------------------------------------------
// Helper: build a BlockSparseMatrix from a dense matrix
// ---------------------------------------------------------------------------

// Treats each BlockRows x BlockCols tile of `dense` as one block whenever the
// tile is non-zero.
template <int BlockRows, int BlockCols, typename Scalar, int Options, typename StorageIndex>
BlockSparseMatrix<Scalar, Options, BlockRows, BlockCols, StorageIndex> denseToBlock(
    const Matrix<Scalar, Dynamic, Dynamic>& dense) {
  using BSM = BlockSparseMatrix<Scalar, Options, BlockRows, BlockCols, StorageIndex>;
  using Triplet = typename BSM::TripletType;

  const Eigen::Index bRows = dense.rows() / BlockRows;
  const Eigen::Index bCols = dense.cols() / BlockCols;

  std::vector<Triplet> triplets;
  for (Eigen::Index bi = 0; bi < bRows; ++bi) {
    for (Eigen::Index bj = 0; bj < bCols; ++bj) {
      auto tile = dense.block(bi * BlockRows, bj * BlockCols, BlockRows, BlockCols);
      if (tile.squaredNorm() > Scalar(0)) {
        triplets.emplace_back(StorageIndex(bi), StorageIndex(bj),
                              Matrix<Scalar, BlockRows, BlockCols>(tile));
      }
    }
  }

  BSM bsm(bRows, bCols);
  bsm.setFromTriplets(triplets.begin(), triplets.end());
  return bsm;
}

// ---------------------------------------------------------------------------
// Core test driver templated on block size and storage order
// ---------------------------------------------------------------------------

template <int BlockRows, int BlockCols, int Options>
void test_block_sparse(int bRows, int bCols) {
  using Scalar = double;
  using StorageIndex = int;
  using BSM = BlockSparseMatrix<Scalar, Options, BlockRows, BlockCols, StorageIndex>;
  using SpMat = SparseMatrix<Scalar, Options, StorageIndex>;
  using DenseMat = Matrix<Scalar, Dynamic, Dynamic>;

  const int rows = bRows * BlockRows;
  const int cols = bCols * BlockCols;

  // Build two random dense matrices with a coarse block structure.
  DenseMat dA = DenseMat::Zero(rows, cols);
  DenseMat dB = DenseMat::Zero(rows, cols);
  for (int bi = 0; bi < bRows; ++bi) {
    for (int bj = 0; bj < bCols; ++bj) {
      if (internal::random<double>(0.0, 1.0) < 0.4) {
        dA.block(bi * BlockRows, bj * BlockCols, BlockRows, BlockCols) =
            DenseMat::Random(BlockRows, BlockCols);
      }
      if (internal::random<double>(0.0, 1.0) < 0.4) {
        dB.block(bi * BlockRows, bj * BlockCols, BlockRows, BlockCols) =
            DenseMat::Random(BlockRows, BlockCols);
      }
    }
  }

  BSM A = denseToBlock<BlockRows, BlockCols, Scalar, Options, StorageIndex>(dA);
  BSM B = denseToBlock<BlockRows, BlockCols, Scalar, Options, StorageIndex>(dB);

  // ---- toSparse / fromSparse round-trip -----------------------------------
  {
    SpMat spA = A.toSparse();
    DenseMat dense_spA(spA);
    VERIFY_IS_APPROX(dense_spA, dA);

    BSM A2 = BSM::fromSparse(spA);
    VERIFY_IS_APPROX(A2.toSparse(), spA);

    // Implicit conversion operator
    SpMat spA3 = A;
    VERIFY_IS_APPROX(DenseMat(spA3), dA);
  }

  // ---- Addition -----------------------------------------------------------
  {
    BSM C = A + B;
    DenseMat dC(C.toSparse());
    VERIFY_IS_APPROX(dC, dA + dB);

    BSM D = A;
    D += B;
    VERIFY_IS_APPROX(DenseMat(D.toSparse()), dA + dB);
  }

  // ---- Subtraction --------------------------------------------------------
  {
    BSM C = A - B;
    VERIFY_IS_APPROX(DenseMat(C.toSparse()), dA - dB);

    BSM D = A;
    D -= B;
    VERIFY_IS_APPROX(DenseMat(D.toSparse()), dA - dB);
  }

  // ---- Unary minus --------------------------------------------------------
  {
    BSM C = -A;
    VERIFY_IS_APPROX(DenseMat(C.toSparse()), -dA);
  }

  // ---- Scalar multiplication ----------------------------------------------
  {
    const Scalar s = Scalar(3.14);
    BSM C = A * s;
    VERIFY_IS_APPROX(DenseMat(C.toSparse()), dA * s);

    BSM D = s * A;
    VERIFY_IS_APPROX(DenseMat(D.toSparse()), s * dA);

    BSM E = A;
    E *= s;
    VERIFY_IS_APPROX(DenseMat(E.toSparse()), dA * s);
  }

  // ---- Element access (coeff) ---------------------------------------------
  {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        VERIFY_IS_APPROX(A.coeff(i, j), dA(i, j));
      }
    }
  }

  // ---- setFromTriplets with duplicate blocks (accumulation) ---------------
  {
    using Trip = typename BSM::TripletType;
    using BlockMat = Matrix<Scalar, BlockRows, BlockCols>;
    BlockMat half = BlockMat::Ones() * Scalar(0.5);

    std::vector<Trip> trips;
    trips.emplace_back(StorageIndex(0), StorageIndex(0), half);
    trips.emplace_back(StorageIndex(0), StorageIndex(0), half);  // duplicate -> sum

    BSM M(bRows, bCols);
    M.setFromTriplets(trips.begin(), trips.end());

    VERIFY(M.nonZeroBlocks() == 1);
    VERIFY_IS_APPROX(DenseMat(M.blockRef(0)), DenseMat(BlockMat::Ones()));
  }
}

// ---------------------------------------------------------------------------
// Square-block product test
// ---------------------------------------------------------------------------

template <int B, int Options>
void test_block_sparse_product(int bM, int bK, int bN) {
  using Scalar = double;
  using StorageIndex = int;
  using BSMA = BlockSparseMatrix<Scalar, Options, B, B, StorageIndex>;
  using DenseMat = Matrix<Scalar, Dynamic, Dynamic>;

  const int rowsA = bM * B, colsA = bK * B;
  const int colsB = bN * B;

  DenseMat dA = DenseMat::Zero(rowsA, colsA);
  DenseMat dB = DenseMat::Zero(colsA, colsB);

  for (int bi = 0; bi < bM; ++bi)
    for (int bk = 0; bk < bK; ++bk)
      if (internal::random<double>(0.0, 1.0) < 0.4)
        dA.block(bi * B, bk * B, B, B) = DenseMat::Random(B, B);

  for (int bk = 0; bk < bK; ++bk)
    for (int bj = 0; bj < bN; ++bj)
      if (internal::random<double>(0.0, 1.0) < 0.4)
        dB.block(bk * B, bj * B, B, B) = DenseMat::Random(B, B);

  BSMA A = denseToBlock<B, B, Scalar, Options, StorageIndex>(dA);
  BSMA Bmat = denseToBlock<B, B, Scalar, Options, StorageIndex>(dB);

  auto C = A * Bmat;
  DenseMat dC(C.toSparse());
  VERIFY_IS_APPROX(dC, dA * dB);
}

// ---------------------------------------------------------------------------
// Block-sparse * dense and dense * block-sparse products
// ---------------------------------------------------------------------------

template <int BlockRows, int BlockCols, int Options>
void test_block_sparse_dense_product(int bRows, int bCols) {
  using Scalar = double;
  using StorageIndex = int;
  using BSM = BlockSparseMatrix<Scalar, Options, BlockRows, BlockCols, StorageIndex>;
  using DenseMat = Matrix<Scalar, Dynamic, Dynamic>;
  using DenseVec = Matrix<Scalar, Dynamic, 1>;
  using RowVec = Matrix<Scalar, 1, Dynamic>;

  const int rows = bRows * BlockRows;
  const int cols = bCols * BlockCols;

  DenseMat dA = DenseMat::Zero(rows, cols);
  for (int bi = 0; bi < bRows; ++bi)
    for (int bj = 0; bj < bCols; ++bj)
      if (internal::random<double>(0.0, 1.0) < 0.5)
        dA.block(bi * BlockRows, bj * BlockCols, BlockRows, BlockCols) =
            DenseMat::Random(BlockRows, BlockCols);

  BSM A = denseToBlock<BlockRows, BlockCols, Scalar, Options, StorageIndex>(dA);

  // BSM * dense matrix
  {
    DenseMat rhs = DenseMat::Random(cols, 5);
    VERIFY_IS_APPROX(A * rhs, dA * rhs);
  }

  // BSM * column vector
  {
    DenseVec v = DenseVec::Random(cols);
    VERIFY_IS_APPROX(A * v, dA * v);
  }

  // dense matrix * BSM
  {
    DenseMat lhs = DenseMat::Random(5, rows);
    VERIFY_IS_APPROX(lhs * A, lhs * dA);
  }

  // row vector * BSM
  {
    RowVec v = RowVec::Random(rows);
    VERIFY_IS_APPROX(v * A, v * dA);
  }
}

// ---------------------------------------------------------------------------
// Non-square block product: A(BR x BC) * B(BC x BC2) -> C(BR x BC2)
// ---------------------------------------------------------------------------

void test_nonsquare_block_product() {
  using Scalar = double;
  using StorageIndex = int;
  constexpr int BR = 2, BC = 3, BC2 = 4;
  using BSMA = BlockSparseMatrix<Scalar, ColMajor, BR, BC, StorageIndex>;
  using BSMB = BlockSparseMatrix<Scalar, ColMajor, BC, BC2, StorageIndex>;
  using DenseMat = Matrix<Scalar, Dynamic, Dynamic>;

  const int bM = 4, bK = 3, bN = 5;
  DenseMat dA = DenseMat::Zero(bM * BR, bK * BC);
  DenseMat dB = DenseMat::Zero(bK * BC, bN * BC2);

  for (int bi = 0; bi < bM; ++bi)
    for (int bk = 0; bk < bK; ++bk)
      if (internal::random<double>() > 0.0)
        dA.block(bi * BR, bk * BC, BR, BC) = DenseMat::Random(BR, BC);

  for (int bk = 0; bk < bK; ++bk)
    for (int bj = 0; bj < bN; ++bj)
      if (internal::random<double>() > 0.0)
        dB.block(bk * BC, bj * BC2, BC, BC2) = DenseMat::Random(BC, BC2);

  BSMA A = denseToBlock<BR, BC, Scalar, ColMajor, StorageIndex>(dA);
  BSMB Bmat = denseToBlock<BC, BC2, Scalar, ColMajor, StorageIndex>(dB);

  auto C = A * Bmat;
  DenseMat dC(C.toSparse());
  VERIFY_IS_APPROX(dC, dA * dB);
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

EIGEN_DECLARE_TEST(block_sparse_matrix) {
  // ColMajor, various block sizes and matrix sizes
  CALL_SUBTEST_1((test_block_sparse<1, 1, ColMajor>(6, 8)));
  CALL_SUBTEST_2((test_block_sparse<2, 2, ColMajor>(4, 6)));
  CALL_SUBTEST_3((test_block_sparse<3, 3, ColMajor>(4, 5)));
  CALL_SUBTEST_4((test_block_sparse<4, 4, ColMajor>(3, 3)));
  CALL_SUBTEST_5((test_block_sparse<2, 3, ColMajor>(5, 4)));

  // RowMajor
  CALL_SUBTEST_6((test_block_sparse<2, 2, RowMajor>(4, 6)));
  CALL_SUBTEST_7((test_block_sparse<3, 3, RowMajor>(4, 5)));
  CALL_SUBTEST_8((test_block_sparse<2, 3, RowMajor>(5, 4)));

  // Products (ColMajor)
  CALL_SUBTEST_9((test_block_sparse_product<2, ColMajor>(4, 5, 3)));
  CALL_SUBTEST_9((test_block_sparse_product<3, ColMajor>(3, 4, 5)));

  // Products (RowMajor)
  CALL_SUBTEST_10((test_block_sparse_product<2, RowMajor>(4, 5, 3)));
  CALL_SUBTEST_10((test_block_sparse_product<3, RowMajor>(3, 4, 5)));

  // Non-square block product
  CALL_SUBTEST_11(test_nonsquare_block_product());

  // Block-sparse * dense and dense * block-sparse (ColMajor)
  CALL_SUBTEST_12((test_block_sparse_dense_product<2, 2, ColMajor>(4, 5)));
  CALL_SUBTEST_12((test_block_sparse_dense_product<3, 3, ColMajor>(3, 4)));
  CALL_SUBTEST_12((test_block_sparse_dense_product<2, 3, ColMajor>(4, 3)));

  // Block-sparse * dense and dense * block-sparse (RowMajor)
  CALL_SUBTEST_13((test_block_sparse_dense_product<2, 2, RowMajor>(4, 5)));
  CALL_SUBTEST_13((test_block_sparse_dense_product<3, 3, RowMajor>(3, 4)));
  CALL_SUBTEST_13((test_block_sparse_dense_product<2, 3, RowMajor>(4, 3)));
}
