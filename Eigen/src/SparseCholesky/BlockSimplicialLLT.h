// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_BLOCK_SIMPLICIAL_LLT_H
#define EIGEN_BLOCK_SIMPLICIAL_LLT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \ingroup SparseCholesky_Module
 * \class BlockSimplicialLLT
 * \brief Block-sparse simplicial LLT (Cholesky) factorization.
 *
 * Operates on a \c BlockSparseMatrix<Scalar, ColMajor, B, B> with fixed B×B
 * square blocks.  AMD reordering and symbolic fill-in analysis are performed
 * at the block level via \c internal::simpl_chol_helper; the numerical
 * factorization replaces every scalar operation with a B×B block operation.
 *
 * \tparam BSM  A \c BlockSparseMatrix<Scalar, ColMajor, B, B, StorageIndex>
 *              with square blocks (BlockRows == BlockCols).
 *
 * Usage:
 * \code
 *   BlockSimplicialLLT<BSM> solver;
 *   solver.analyzePattern(A);   // symbolic: AMD + fill-in pattern
 *   solver.factorize(A);        // numeric
 *   VectorXd x = solver.solve(b);
 * \endcode
 * Or the short form:
 * \code
 *   BlockSimplicialLLT<BSM> solver(A);
 *   VectorXd x = solver.solve(b);
 * \endcode
 */
template <typename BSM>
class BlockSimplicialLLT {
  static_assert(BSM::BlockRows == BSM::BlockCols,
                "BlockSimplicialLLT requires square blocks (BlockRows == BlockCols)");
  static_assert(!BSM::IsRowMajor, "BlockSimplicialLLT requires ColMajor BlockSparseMatrix");

 public:
  using Scalar        = typename BSM::Scalar;
  using StorageIndex  = typename BSM::StorageIndex;
  using BlockType     = typename BSM::BlockType;
  static constexpr int B         = BSM::BlockRows;
  static constexpr int BlockSize = BSM::BlockSize;

  using ScalarSparseMatrix = SparseMatrix<Scalar, ColMajor, StorageIndex>;
  using VectorI            = Matrix<StorageIndex, Dynamic, 1>;

  BlockSimplicialLLT()
      : m_info(Success), m_isInitialized(false), m_analysisIsOk(false), m_factorizationIsOk(false) {}

  explicit BlockSimplicialLLT(const BSM& A)
      : m_info(Success), m_isInitialized(false), m_analysisIsOk(false), m_factorizationIsOk(false) {
    compute(A);
  }

  void compute(const BSM& A) {
    analyzePattern(A);
    if (m_info == Success) factorize(A);
  }

  ComputationInfo info() const {
    eigen_assert(m_isInitialized && "BlockSimplicialLLT is not initialized.");
    return m_info;
  }

  // -------------------------------------------------------------------------
  // analyzePattern: AMD reordering + fill-in pattern at block level
  // -------------------------------------------------------------------------

  void analyzePattern(const BSM& A) {
    const Index nb = A.blockRows();
    eigen_assert(A.blockCols() == nb && "BlockSimplicialLLT: matrix must be square");

    // Build a scalar sparse matrix with A's block sparsity pattern.
    // Values are irrelevant for symbolic analysis; set to 1 as placeholders.
    const Index nnzBlocks = A.nonZeroBlocks();
    ScalarSparseMatrix S(nb, nb);
    S.resizeNonZeros(nnzBlocks);
    std::copy_n(A.outerIndexPtr(), nb + 1, S.outerIndexPtr());
    std::copy_n(A.innerIndexPtr(), nnzBlocks, S.innerIndexPtr());
    std::fill_n(S.valuePtr(), nnzBlocks, Scalar(1));

    // AMD ordering on the lower-triangular block pattern.
    // AMDOrdering fills m_Pinv (old-to-new); m_P is then new-to-old.
    AMDOrdering<StorageIndex> amd;
    amd(S.template selfadjointView<Lower>(), m_Pinv);
    m_P = m_Pinv.inverse();

    // Apply permutation: read lower triangle of S, write upper triangle of ap_perm.
    ScalarSparseMatrix ap_perm;
    internal::permute_symm_to_symm<Lower, Upper, /*NonHermitian=*/false>(
        S, ap_perm, m_P.indices().data());

    // Symbolic analysis: fills L_scalar.outerIndexPtr(), allocates and fills L_scalar.innerIndexPtr().
    using Helper = internal::simpl_chol_helper<Scalar, StorageIndex>;
    ScalarSparseMatrix L_scalar;
    VectorI parent, workSpace;
    Helper::run(StorageIndex(nb), ap_perm, L_scalar, parent, workSpace, /*doLDLT=*/false);

    // Build m_L block structure directly from L_scalar index arrays.
    m_L.setFromOuterInner(nb, nb, L_scalar.nonZeros(),
                          L_scalar.outerIndexPtr(), L_scalar.innerIndexPtr());

    m_info              = Success;
    m_isInitialized     = true;
    m_analysisIsOk      = true;
    m_factorizationIsOk = false;
  }

  // -------------------------------------------------------------------------
  // factorize: block column-by-column left-looking Cholesky
  // -------------------------------------------------------------------------
  //
  // AMD factors the permuted matrix P*A*P^T = L*L^T, where:
  //   m_Pinv.indices()[orig_i] = perm_i  (old-to-new)
  //   m_P.indices()[perm_i]   = orig_i  (new-to-old)
  // -------------------------------------------------------------------------

  void factorize(const BSM& A) {
    eigen_assert(m_analysisIsOk && "Call analyzePattern() before factorize()");

    const Index nb = A.blockRows();
    eigen_assert(A.blockCols() == nb);

    for (Index k = 0; k < m_L.nonZeroBlocks(); ++k)
      m_L.blockRef(k).setZero();

    bool ok = true;

    // Dense work buffer: one B×B block per block-row, stored as a flat scalar array.
    Array<Scalar, Dynamic, 1> work(nb * Index(BlockSize));

    for (Index j = 0; j < nb; ++j) {
      const Index orig_j = (m_Pinv.size() > 0) ? static_cast<Index>(m_Pinv.indices()[j]) : j;

      // Step 1: gather column j of the permuted A into work[j..nb-1].
      for (Index i = j; i < nb; ++i)
        Map<BlockType>(work.data() + i * Index(BlockSize)).setZero();

      {
        const StorageIndex* beg = A.innerIndexPtr() + A.outerIndexPtr()[orig_j];
        const StorageIndex* fin = A.innerIndexPtr() + A.outerIndexPtr()[orig_j + 1];
        for (const StorageIndex* it = beg; it != fin; ++it) {
          const Index orig_i = static_cast<Index>(*it);
          const Index perm_i = (m_P.size() > 0) ? static_cast<Index>(m_P.indices()[orig_i]) : orig_i;
          if (perm_i >= j) {
            const Index idx = static_cast<Index>(it - A.innerIndexPtr());
            Map<BlockType>(work.data() + perm_i * Index(BlockSize)) = A.blockRef(idx);
          }
        }
      }

      // Step 2: subtract contributions from completed columns k < j.
      // work[i] -= L(i,k) * L(j,k)^T
      for (Index k = 0; k < j; ++k) {
        const Index Ljk_idx = findBlockInL(j, k);
        if (Ljk_idx < 0) continue;
        const BlockType Ljk = m_L.blockRef(Ljk_idx);

        const StorageIndex* beg = m_L.innerIndexPtr() + m_L.outerIndexPtr()[k];
        const StorageIndex* fin = m_L.innerIndexPtr() + m_L.outerIndexPtr()[k + 1];
        for (const StorageIndex* it = beg; it != fin; ++it) {
          const Index i = static_cast<Index>(*it);
          if (i >= j) {
            const Index idx = static_cast<Index>(it - m_L.innerIndexPtr());
            Map<BlockType>(work.data() + i * Index(BlockSize)).noalias() -=
                m_L.blockRef(idx) * Ljk.transpose();
          }
        }
      }

      // Step 3: Cholesky factor the diagonal block.
      LLT<BlockType> llt(Map<const BlockType>(work.data() + j * Index(BlockSize)));
      if (llt.info() != Success) { ok = false; break; }
      const Index Ljj_idx = findBlockInL(j, j);
      eigen_assert(Ljj_idx >= 0 && "Diagonal block missing from L sparsity pattern");
      m_L.blockRef(Ljj_idx) = llt.matrixL();

      // Step 4: off-diagonal blocks. L(i,j) = work[i] * L(j,j)^{-T}.
      // Solve L(j,j) * X^T = work[i]^T  =>  X = work[i] * L(j,j)^{-T}.
      const BlockType Ljj = m_L.blockRef(Ljj_idx);
      const StorageIndex* beg = m_L.innerIndexPtr() + m_L.outerIndexPtr()[j];
      const StorageIndex* fin = m_L.innerIndexPtr() + m_L.outerIndexPtr()[j + 1];
      for (const StorageIndex* it = beg; it != fin; ++it) {
        const Index i = static_cast<Index>(*it);
        if (i > j) {
          const Index idx = static_cast<Index>(it - m_L.innerIndexPtr());
          BlockType Xt = Map<const BlockType>(work.data() + i * Index(BlockSize)).transpose();
          Ljj.template triangularView<Lower>().solveInPlace(Xt);
          m_L.blockRef(idx) = Xt.transpose();
        }
      }
    }

    m_info              = ok ? Success : NumericalIssue;
    m_factorizationIsOk = true;
  }

  // -------------------------------------------------------------------------
  // solve: A x = b
  // -------------------------------------------------------------------------

  template <typename Rhs>
  Matrix<Scalar, Dynamic, 1> solve(const MatrixBase<Rhs>& b) const {
    eigen_assert(m_factorizationIsOk && "Call factorize() before solve()");
    eigen_assert(m_info == Success && "BlockSimplicialLLT factorization failed");

    const Index nb = m_L.blockCols();
    const Index n  = nb * Index(B);
    eigen_assert(b.rows() == n);

    // 1. Permute rhs: bperm[perm_i] = b[orig_i],  orig_i = m_Pinv.indices()[perm_i].
    Matrix<Scalar, Dynamic, 1> bperm(n);
    if (m_Pinv.size() > 0) {
      for (Index perm_i = 0; perm_i < nb; ++perm_i) {
        const Index orig_i = static_cast<Index>(m_Pinv.indices()[perm_i]);
        bperm.segment(perm_i * B, B) = b.derived().segment(orig_i * B, B);
      }
    } else {
      bperm = b.derived();
    }

    // 2. Forward block triangular solve: L y = bperm.
    Matrix<Scalar, Dynamic, 1> y = bperm;
    for (Index j = 0; j < nb; ++j) {
      const StorageIndex* beg = m_L.innerIndexPtr() + m_L.outerIndexPtr()[j];
      const StorageIndex* fin = m_L.innerIndexPtr() + m_L.outerIndexPtr()[j + 1];
      eigen_assert(beg < fin && static_cast<Index>(*beg) == j);
      const Index Ljj_idx = static_cast<Index>(beg - m_L.innerIndexPtr());
      m_L.blockRef(Ljj_idx).template triangularView<Lower>().solveInPlace(y.segment(j * B, B));
      for (const StorageIndex* it = beg + 1; it != fin; ++it) {
        const Index i   = static_cast<Index>(*it);
        const Index idx = static_cast<Index>(it - m_L.innerIndexPtr());
        y.segment(i * B, B).noalias() -= m_L.blockRef(idx) * y.segment(j * B, B);
      }
    }

    // 3. Backward block triangular solve: L^T x = y.
    Matrix<Scalar, Dynamic, 1> x = y;
    for (Index j = nb - 1; j >= 0; --j) {
      const StorageIndex* beg = m_L.innerIndexPtr() + m_L.outerIndexPtr()[j];
      const StorageIndex* fin = m_L.innerIndexPtr() + m_L.outerIndexPtr()[j + 1];
      eigen_assert(beg < fin && static_cast<Index>(*beg) == j);
      const Index Ljj_idx = static_cast<Index>(beg - m_L.innerIndexPtr());
      for (const StorageIndex* it = beg + 1; it != fin; ++it) {
        const Index i   = static_cast<Index>(*it);
        const Index idx = static_cast<Index>(it - m_L.innerIndexPtr());
        x.segment(j * B, B).noalias() -= m_L.blockRef(idx).transpose() * x.segment(i * B, B);
      }
      m_L.blockRef(Ljj_idx).template triangularView<Lower>().transpose().solveInPlace(
          x.segment(j * B, B));
    }

    // 4. Inverse permutation: result[orig_i] = x[perm_i],  perm_i = m_P.indices()[orig_i].
    Matrix<Scalar, Dynamic, 1> result(n);
    if (m_P.size() > 0) {
      for (Index orig_i = 0; orig_i < nb; ++orig_i) {
        const Index perm_i = static_cast<Index>(m_P.indices()[orig_i]);
        result.segment(orig_i * B, B) = x.segment(perm_i * B, B);
      }
    } else {
      result = x;
    }
    return result;
  }

  const BSM& matrixL() const {
    eigen_assert(m_factorizationIsOk && "Call factorize() first");
    return m_L;
  }

  const PermutationMatrix<Dynamic, Dynamic, StorageIndex>& permutationP() const { return m_P; }

 private:
  // Returns the flat block index of (row, col) in m_L (ColMajor), or -1 if absent.
  Index findBlockInL(Index row, Index col) const {
    const StorageIndex* beg = m_L.innerIndexPtr() + m_L.outerIndexPtr()[col];
    const StorageIndex* fin = m_L.innerIndexPtr() + m_L.outerIndexPtr()[col + 1];
    const StorageIndex* it  = std::lower_bound(beg, fin, StorageIndex(row));
    if (it == fin || static_cast<Index>(*it) != row) return Index(-1);
    return static_cast<Index>(it - m_L.innerIndexPtr());
  }

  PermutationMatrix<Dynamic, Dynamic, StorageIndex> m_P;     // AMD perm (new-to-old)
  PermutationMatrix<Dynamic, Dynamic, StorageIndex> m_Pinv;  // AMD perm (old-to-new)
  BSM             m_L;
  ComputationInfo m_info;
  bool m_isInitialized;
  bool m_analysisIsOk;
  bool m_factorizationIsOk;
};

}  // end namespace Eigen

#endif  // EIGEN_BLOCK_SIMPLICIAL_LLT_H
