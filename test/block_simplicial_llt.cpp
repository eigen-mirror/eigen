// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Test for BlockSimplicialLLT: block-sparse Cholesky factorization.
//
// For each block size and scalar type, we:
//   1. Generate a random block-sparse SPD matrix A.
//   2. Factor with BlockSimplicialLLT, solve a random rhs, check residual.
//   3. Compare result with scalar SimplicialLLT on the expanded matrix.

#include "sparse.h"

#include <Eigen/SparseCholesky>

using namespace Eigen;

// ---------------------------------------------------------------------------
// Generate a random SPD BlockSparseMatrix.
//
// Strategy: generate a random block-sparse lower-triangular L0, form
//   A = L0 * L0^T + shift * I
// so that A is guaranteed SPD.
// ---------------------------------------------------------------------------
template <typename Scalar, int B, typename StorageIndex = int>
BlockSparseMatrix<Scalar, ColMajor, B, B, StorageIndex> makeRandomSPDBlockSparse(
    Index nb,         // number of block rows/cols
    double density,   // probability that off-diagonal block is nonzero
    Scalar shift = Scalar(B * 4)  // diagonal shift to ensure positive definiteness
) {
  using BSM = BlockSparseMatrix<Scalar, ColMajor, B, B, StorageIndex>;
  using BlockT = Matrix<Scalar, B, B>;
  using Triplet = typename BSM::TripletType;

  // Build lower-triangular L0 (block-level lower triangle including diagonal).
  std::vector<Triplet> triplets_L0;
  for (Index bj = 0; bj < nb; ++bj) {
    for (Index bi = bj; bi < nb; ++bi) {
      bool include = (bi == bj) || (internal::random<double>(0.0, 1.0) < density);
      if (!include) continue;
      BlockT blk = BlockT::Random();
      if (bi == bj) {
        // Make diagonal block of L0 lower triangular with positive diagonal.
        for (int r = 0; r < B; ++r)
          for (int c = r + 1; c < B; ++c) blk(r, c) = Scalar(0);
        for (int d = 0; d < B; ++d)
          blk(d, d) = Scalar(std::abs(static_cast<double>(blk(d, d))) + Scalar(1));
      }
      triplets_L0.emplace_back(StorageIndex(bi), StorageIndex(bj), blk);
    }
  }
  BSM L0(nb, nb);
  L0.setFromTriplets(triplets_L0.begin(), triplets_L0.end());

  // A = L0 * L0^T (block product).
  BSM A = L0 * L0.transpose();

  // Add diagonal shift: A(bi,bi) += shift * I for all bi.
  // We do this by adding shift*I to each diagonal block.
  std::vector<Triplet> diag_triplets;
  BlockT shift_block = shift * BlockT::Identity();
  for (Index bi = 0; bi < nb; ++bi)
    diag_triplets.emplace_back(StorageIndex(bi), StorageIndex(bi), shift_block);

  BSM Shift(nb, nb);
  Shift.setFromTriplets(diag_triplets.begin(), diag_triplets.end());

  return A + Shift;
}

// ---------------------------------------------------------------------------
// Core test for a single block size and scalar type.
// ---------------------------------------------------------------------------
template <typename Scalar, int B>
void test_block_simplicial_llt(Index nb, double density) {
  using BSM = BlockSparseMatrix<Scalar, ColMajor, B, B>;
  using SpMat = SparseMatrix<Scalar, ColMajor>;
  using VecX  = Matrix<Scalar, Dynamic, 1>;

  // Generate random SPD block-sparse matrix.
  BSM A = makeRandomSPDBlockSparse<Scalar, B>(nb, density);

  // Random rhs.
  VecX b = VecX::Random(nb * B);

  // ---- BlockSimplicialLLT ---------------------------------------------------
  BlockSimplicialLLT<BSM> solver(A);
  VERIFY(solver.info() == Success);

  VecX x_block = solver.solve(b);

  // Check residual: A * x ≈ b.
  // Compute A * x via scalar expansion.
  SpMat Asp = A.toSparse();
  VecX residual = Asp * x_block - b;
  // Use a generous tolerance scaled by matrix size.
  Scalar tol = Scalar(1e-4) * b.norm();
  VERIFY(residual.norm() <= tol);

  // ---- ScalarSimplicialLLT for comparison -----------------------------------
  SimplicialLLT<SpMat> scalar_solver(Asp);
  VERIFY(scalar_solver.info() == Success);
  VecX x_scalar = scalar_solver.solve(b);

  // The two solutions should agree.
  VERIFY_IS_APPROX(x_block, x_scalar);
}

// ---------------------------------------------------------------------------
// Test entry point
// ---------------------------------------------------------------------------
EIGEN_DECLARE_TEST(block_simplicial_llt) {
  // 2x2 blocks, double
  CALL_SUBTEST_1((test_block_simplicial_llt<double, 2>(8, 0.3)));
  CALL_SUBTEST_1((test_block_simplicial_llt<double, 2>(12, 0.4)));

  // 3x3 blocks, double
  CALL_SUBTEST_2((test_block_simplicial_llt<double, 3>(6, 0.3)));
  CALL_SUBTEST_2((test_block_simplicial_llt<double, 3>(10, 0.35)));

  // 4x4 blocks, float (looser tolerance in VERIFY_IS_APPROX)
  CALL_SUBTEST_3((test_block_simplicial_llt<float, 4>(4, 0.4)));
  CALL_SUBTEST_3((test_block_simplicial_llt<float, 4>(6, 0.3)));

  // 2x2 blocks, larger matrix
  CALL_SUBTEST_4((test_block_simplicial_llt<double, 2>(20, 0.2)));

  // 3x3 blocks, larger
  CALL_SUBTEST_5((test_block_simplicial_llt<double, 3>(15, 0.2)));
}
