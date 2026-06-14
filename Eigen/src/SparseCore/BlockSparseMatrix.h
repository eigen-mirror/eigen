// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_BLOCKSPARSEMATRIX_H
#define EIGEN_BLOCKSPARSEMATRIX_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include <algorithm>
#include <numeric>
#include <utility>

namespace Eigen {

/** \class BlockTriplet
 * \ingroup SparseCore_Module
 * \brief A (blockRow, blockCol, blockValue) triplet for assembling a BlockSparseMatrix.
 *
 * Coordinates are in \em block space, not element space.
 *
 * \tparam Scalar_       Numeric scalar type.
 * \tparam BlockRows_    Number of rows in each block.
 * \tparam BlockCols_    Number of columns in each block.
 * \tparam StorageIndex_ Signed integer index type (default: int).
 */
template <typename Scalar_, int BlockRows_, int BlockCols_, typename StorageIndex_ = int>
class BlockTriplet {
 public:
  using Scalar = Scalar_;
  using StorageIndex = StorageIndex_;
  using BlockType = Matrix<Scalar, BlockRows_, BlockCols_>;

  BlockTriplet() = default;

  BlockTriplet(StorageIndex blockRow, StorageIndex blockCol, const BlockType& value)
      : m_row(blockRow), m_col(blockCol), m_value(value) {}

  StorageIndex row() const { return m_row; }
  StorageIndex col() const { return m_col; }
  const BlockType& value() const { return m_value; }

 private:
  StorageIndex m_row = StorageIndex(0);
  StorageIndex m_col = StorageIndex(0);
  BlockType m_value = BlockType::Zero();
};

/** \class BlockSparseMatrix
 * \ingroup SparseCore_Module
 * \brief A sparse matrix whose stored nonzeros are fixed-size dense blocks.
 *
 * Each nonzero entry is a \c BlockRows x \c BlockCols dense matrix.  The
 * block sparsity pattern is stored in block-level compressed-column
 * (ColMajor) or compressed-row (RowMajor) format.
 *
 * \tparam Scalar_       Numeric scalar type.
 * \tparam Options_      ColMajor (0) or RowMajor.  Affects the outer
 *                       iteration direction over blocks, not the storage
 *                       layout within each block (which is always
 *                       column-major).
 * \tparam BlockRows_    Rows per block; must be a fixed positive integer.
 * \tparam BlockCols_    Columns per block; must be a fixed positive integer.
 * \tparam StorageIndex_ Signed integer type for internal index arrays
 *                       (default: int).
 *
 * ### Assembly
 * Populate the matrix via setFromTriplets(), passing an iterator range of
 * BlockTriplet objects in block coordinates.  Triplets with the same
 * (blockRow, blockCol) are summed.
 *
 * ### Arithmetic
 * Addition, subtraction, and matrix product are supported between compatible
 * BlockSparseMatrix instances.  Scalar multiplication is also available.
 *
 * The matrix product \c C = A * B requires
 * \c A.BlockCols == B.BlockRows (enforced at compile time by the template
 * constraint) and \c A.blockCols() == B.blockRows() (checked at runtime).
 * The result has block type \c Matrix<Scalar,A.BlockRows,B.BlockCols>.
 *
 * ### Conversion
 * toSparse() converts to a standard SparseMatrix with element-level
 * sparsity.  fromSparse() reconstructs a BlockSparseMatrix from an
 * element-level SparseMatrix whose dimensions are divisible by BlockRows
 * and BlockCols.  An implicit conversion operator to SparseMatrix is
 * also provided.
 */
template <typename Scalar_, int Options_, int BlockRows_, int BlockCols_,
          typename StorageIndex_ = int>
class BlockSparseMatrix {
  static_assert(BlockRows_ >= 1, "BlockRows must be >= 1");
  static_assert(BlockCols_ >= 1, "BlockCols must be >= 1");
  static_assert(BlockRows_ != Dynamic, "BlockRows must be a fixed compile-time size");
  static_assert(BlockCols_ != Dynamic, "BlockCols must be a fixed compile-time size");

 public:
  // -------------------------------------------------------------------------
  // Type aliases & compile-time constants
  // -------------------------------------------------------------------------
  using Scalar = Scalar_;
  using StorageIndex = StorageIndex_;
  using Index = Eigen::Index;
  using BlockType = Matrix<Scalar, BlockRows_, BlockCols_>;
  using BlockMap = Map<BlockType>;
  using ConstBlockMap = Map<const BlockType>;
  using TripletType = BlockTriplet<Scalar, BlockRows_, BlockCols_, StorageIndex>;

  enum {
    Options = Options_,
    BlockRows = BlockRows_,
    BlockCols = BlockCols_,
    IsRowMajor = (Options_ & RowMajorBit) != 0,
    BlockSize = BlockRows_ * BlockCols_
  };

  // -------------------------------------------------------------------------
  // Constructors / copy / move
  // -------------------------------------------------------------------------

  /** Default constructor; creates a 0×0 matrix. */
  BlockSparseMatrix() : m_blockOuterSize(0), m_blockInnerSize(0) {
    m_outerIndex.resize(1);
    m_outerIndex(0) = StorageIndex(0);
  }

  /** Construct a zero matrix with the given number of block-rows and block-columns. */
  BlockSparseMatrix(Index blockRows, Index blockCols)
      : m_blockOuterSize(IsRowMajor ? blockRows : blockCols),
        m_blockInnerSize(IsRowMajor ? blockCols : blockRows) {
    m_outerIndex.resize(m_blockOuterSize + 1);
    m_outerIndex.setZero();
  }

  BlockSparseMatrix(const BlockSparseMatrix&) = default;
  BlockSparseMatrix(BlockSparseMatrix&&) noexcept = default;
  BlockSparseMatrix& operator=(const BlockSparseMatrix&) = default;
  BlockSparseMatrix& operator=(BlockSparseMatrix&&) noexcept = default;

  // -------------------------------------------------------------------------
  // Dimensions
  // -------------------------------------------------------------------------

  /** Total number of element rows. */
  Index rows() const { return (IsRowMajor ? m_blockOuterSize : m_blockInnerSize) * BlockRows_; }
  /** Total number of element columns. */
  Index cols() const { return (IsRowMajor ? m_blockInnerSize : m_blockOuterSize) * BlockCols_; }

  /** Number of block-rows. */
  Index blockRows() const { return IsRowMajor ? m_blockOuterSize : m_blockInnerSize; }
  /** Number of block-columns. */
  Index blockCols() const { return IsRowMajor ? m_blockInnerSize : m_blockOuterSize; }

  /** Outer block dimension (block-cols for ColMajor, block-rows for RowMajor). */
  Index blockOuterSize() const { return m_blockOuterSize; }
  /** Inner block dimension (block-rows for ColMajor, block-cols for RowMajor). */
  Index blockInnerSize() const { return m_blockInnerSize; }

  /** Number of stored (structurally non-zero) blocks. */
  Index nonZeroBlocks() const { return m_innerIndex.size(); }
  /** Total number of stored scalar coefficients (= nonZeroBlocks() * BlockRows * BlockCols). */
  Index nonZeros() const { return nonZeroBlocks() * Index(BlockSize); }

  // -------------------------------------------------------------------------
  // Raw pointer access (for interoperability)
  // -------------------------------------------------------------------------
  const StorageIndex* outerIndexPtr() const { return m_outerIndex.data(); }
  StorageIndex* outerIndexPtr() { return m_outerIndex.data(); }
  const StorageIndex* innerIndexPtr() const { return m_innerIndex.data(); }
  StorageIndex* innerIndexPtr() { return m_innerIndex.data(); }
  const Scalar* valuePtr() const { return m_values.data(); }
  Scalar* valuePtr() { return m_values.data(); }

  // -------------------------------------------------------------------------
  // Block access by sequential nonzero index
  // -------------------------------------------------------------------------

  /** Read-only Map to the \a k-th stored block (column-major layout within block). */
  ConstBlockMap blockRef(Index k) const {
    return ConstBlockMap(m_values.data() + k * Index(BlockSize));
  }
  /** Mutable Map to the \a k-th stored block. */
  BlockMap blockRef(Index k) {
    return BlockMap(m_values.data() + k * Index(BlockSize));
  }

  // -------------------------------------------------------------------------
  // Inner iterator over blocks within one outer vector
  // -------------------------------------------------------------------------

  /** \brief Iterates over stored blocks in outer vector \a outer.
   *
   * Usage mirrors SparseMatrix::InnerIterator but value() returns a
   * Map to a BlockRows×BlockCols matrix, not a scalar.
   */
  class InnerIterator {
   public:
    EIGEN_STRONG_INLINE InnerIterator(const BlockSparseMatrix& mat, Index outer)
        : m_mat(mat),
          m_id(static_cast<Index>(mat.m_outerIndex(outer))),
          m_end(static_cast<Index>(mat.m_outerIndex(outer + 1))),
          m_outer(outer) {}

    EIGEN_STRONG_INLINE operator bool() const { return m_id < m_end; }
    EIGEN_STRONG_INLINE InnerIterator& operator++() {
      ++m_id;
      return *this;
    }

    /** Current block outer index (block-col for ColMajor, block-row for RowMajor). */
    EIGEN_STRONG_INLINE Index outer() const { return m_outer; }
    /** Current block inner index (block-row for ColMajor, block-col for RowMajor). */
    EIGEN_STRONG_INLINE Index index() const { return static_cast<Index>(m_mat.m_innerIndex(m_id)); }
    /** Block-row of the current block. */
    EIGEN_STRONG_INLINE Index blockRow() const { return IsRowMajor ? m_outer : index(); }
    /** Block-column of the current block. */
    EIGEN_STRONG_INLINE Index blockCol() const { return IsRowMajor ? index() : m_outer; }

    /** Read-only Map to the current block value. */
    EIGEN_STRONG_INLINE ConstBlockMap value() const { return m_mat.blockRef(m_id); }
    /** Mutable Map to the current block value. */
    EIGEN_STRONG_INLINE BlockMap valueRef() {
      return BlockMap(const_cast<Scalar*>(m_mat.m_values.data()) + m_id * Index(BlockSize));
    }

   private:
    const BlockSparseMatrix& m_mat;
    Index m_id;
    Index m_end;
    Index m_outer;
  };

  // -------------------------------------------------------------------------
  // Resize / clear
  // -------------------------------------------------------------------------

  /** Resize to \a blockRows × \a blockCols blocks and set all blocks to zero. */
  void resize(Index blockRows, Index blockCols) {
    m_blockOuterSize = IsRowMajor ? blockRows : blockCols;
    m_blockInnerSize = IsRowMajor ? blockCols : blockRows;
    m_outerIndex.resize(m_blockOuterSize + 1);
    m_outerIndex.setZero();
    m_innerIndex.resize(0);
    m_values.resize(0);
  }

  /** Clear all stored blocks while keeping the matrix dimensions. */
  void setZero() {
    m_outerIndex.resize(m_blockOuterSize + 1);
    m_outerIndex.setZero();
    m_innerIndex.resize(0);
    m_values.resize(0);
  }

  // -------------------------------------------------------------------------
  // Assembly
  // -------------------------------------------------------------------------

  /** Fill the matrix from an iterator range of BlockTriplet objects.
   *
   * Triplet coordinates are in block space.  Triplets with the same
   * (blockRow, blockCol) pair are summed (their block values are added).
   * The input range may be in any order.
   *
   * \tparam InputIterator  Must dereference to a type with \c row(),
   *                        \c col(), and \c value() members, matching
   *                        BlockTriplet's interface.
   */
  template <typename InputIterator>
  void setFromTriplets(InputIterator begin, InputIterator end);

  // -------------------------------------------------------------------------
  // Conversion to / from SparseMatrix
  // -------------------------------------------------------------------------

  /** Convert to a SparseMatrix with scalar nonzeros.
   *
   * Each stored block of size BlockRows×BlockCols expands into up to
   * BlockRows*BlockCols scalar nonzeros.  The resulting SparseMatrix has
   * the same storage order as \c *this.
   */
  SparseMatrix<Scalar, Options_, StorageIndex_> toSparse() const;

  /** Construct a BlockSparseMatrix from an element-level SparseMatrix.
   *
   * \pre  \c sp.rows() % BlockRows == 0 and \c sp.cols() % BlockCols == 0.
   *
   * Each scalar entry \c sp(i,j) is placed into position
   * \c (i%BlockRows, j%BlockCols) of block \c (i/BlockRows, j/BlockCols).
   * Multiple entries mapping to the same element of the same block are
   * accumulated with \c +=.
   */
  static BlockSparseMatrix fromSparse(const SparseMatrix<Scalar_, Options_, StorageIndex_>& sp);

  /** Implicit conversion to SparseMatrix. */
  operator SparseMatrix<Scalar_, Options_, StorageIndex_>() const { return toSparse(); }

  // -------------------------------------------------------------------------
  // Element access
  // -------------------------------------------------------------------------

  /** Read element \c (row, col); returns 0 if no block covers that position. */
  Scalar coeff(Index row, Index col) const {
    eigen_assert(row >= 0 && row < rows() && col >= 0 && col < cols());
    const Index bOuter = IsRowMajor ? (row / BlockRows_) : (col / BlockCols_);
    const Index bInner = IsRowMajor ? (col / BlockCols_) : (row / BlockRows_);
    const Index localRow = row % Index(BlockRows_);
    const Index localCol = col % Index(BlockCols_);
    const StorageIndex* beg = m_innerIndex.data() + m_outerIndex(bOuter);
    const StorageIndex* fin = m_innerIndex.data() + m_outerIndex(bOuter + 1);
    const StorageIndex* it = std::lower_bound(beg, fin, StorageIndex(bInner));
    if (it == fin || static_cast<Index>(*it) != bInner) return Scalar(0);
    return blockRef(static_cast<Index>(it - m_innerIndex.data()))(localRow, localCol);
  }

  // -------------------------------------------------------------------------
  // Arithmetic
  // -------------------------------------------------------------------------

  /** Element-wise addition.  Both matrices must have the same block dimensions. */
  BlockSparseMatrix operator+(const BlockSparseMatrix& other) const;

  /** Element-wise subtraction. */
  BlockSparseMatrix operator-(const BlockSparseMatrix& other) const { return *this + (-other); }

  /** Unary negation. */
  BlockSparseMatrix operator-() const {
    BlockSparseMatrix result(*this);
    result.m_values = -result.m_values;
    return result;
  }

  BlockSparseMatrix& operator+=(const BlockSparseMatrix& other) { return *this = *this + other; }
  BlockSparseMatrix& operator-=(const BlockSparseMatrix& other) { return *this = *this - other; }

  /** Scalar multiplication (returns a new matrix). */
  BlockSparseMatrix operator*(const Scalar& s) const {
    BlockSparseMatrix result(*this);
    result.m_values *= s;
    return result;
  }
  BlockSparseMatrix& operator*=(const Scalar& s) {
    m_values *= s;
    return *this;
  }
  BlockSparseMatrix operator/(const Scalar& s) const { return *this * (Scalar(1) / s); }
  BlockSparseMatrix& operator/=(const Scalar& s) { return *this *= (Scalar(1) / s); }

  /** Scalar-on-left multiplication. */
  friend BlockSparseMatrix operator*(const Scalar& s, const BlockSparseMatrix& m) { return m * s; }

  /** Block-sparse times dense matrix (or vector) product.
   *
   * Returns a dense matrix.  If \a rhs has a fixed column count at compile
   * time, that count is preserved in the result type.
   *
   * \pre  \c this->cols() == rhs.rows().
   * \pre  Scalar types must match.
   */
  template <typename OtherDerived>
  Matrix<Scalar, Dynamic, OtherDerived::ColsAtCompileTime>
  operator*(const MatrixBase<OtherDerived>& rhs) const {
    EIGEN_STATIC_ASSERT(
        (std::is_same<Scalar, typename OtherDerived::Scalar>::value),
        YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
    eigen_assert(cols() == rhs.rows() && "BlockSparseMatrix * Dense: dimension mismatch");
    using ResultType = Matrix<Scalar, Dynamic, OtherDerived::ColsAtCompileTime>;
    ResultType result = ResultType::Zero(rows(), rhs.cols());
    for (Index out = 0; out < m_blockOuterSize; ++out) {
      for (Index id = static_cast<Index>(m_outerIndex(out));
           id < static_cast<Index>(m_outerIndex(out + 1)); ++id) {
        const Index inner = static_cast<Index>(m_innerIndex(id));
        const Index bi = IsRowMajor ? out : inner;
        const Index bj = IsRowMajor ? inner : out;
        result.middleRows(bi * Index(BlockRows_), Index(BlockRows_)).noalias() +=
            blockRef(id) * rhs.middleRows(bj * Index(BlockCols_), Index(BlockCols_));
      }
    }
    return result;
  }

  /** Dense matrix (or vector) times block-sparse product (hidden friend).
   *
   * Returns a dense matrix.  If \a lhs has a fixed row count at compile
   * time, that count is preserved in the result type.
   *
   * \pre  \c lhs.cols() == bsm.rows().
   * \pre  Scalar types must match.
   */
  template <typename OtherDerived>
  friend Matrix<Scalar_, OtherDerived::RowsAtCompileTime, Dynamic>
  operator*(const MatrixBase<OtherDerived>& lhs, const BlockSparseMatrix& bsm) {
    EIGEN_STATIC_ASSERT(
        (std::is_same<Scalar_, typename OtherDerived::Scalar>::value),
        YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
    eigen_assert(lhs.cols() == bsm.rows() && "Dense * BlockSparseMatrix: dimension mismatch");
    constexpr bool isRM = (Options_ & RowMajorBit) != 0;
    using ResultType = Matrix<Scalar_, OtherDerived::RowsAtCompileTime, Dynamic>;
    ResultType result = ResultType::Zero(lhs.rows(), bsm.cols());
    for (Index out = 0; out < bsm.m_blockOuterSize; ++out) {
      for (Index id = static_cast<Index>(bsm.m_outerIndex(out));
           id < static_cast<Index>(bsm.m_outerIndex(out + 1)); ++id) {
        const Index inner = static_cast<Index>(bsm.m_innerIndex(id));
        const Index bi = isRM ? out : inner;
        const Index bj = isRM ? inner : out;
        result.middleCols(bj * Index(BlockCols_), Index(BlockCols_)).noalias() +=
            lhs.middleCols(bi * Index(BlockRows_), Index(BlockRows_)) * bsm.blockRef(id);
      }
    }
    return result;
  }

  /** Block-sparse matrix product.
   *
   * \tparam RhsBlockCols  Block-column count of the right-hand side.
   *
   * The left-hand side has block type \c Matrix<Scalar,BlockRows,BlockCols>
   * and the right-hand side must have block type
   * \c Matrix<Scalar,BlockCols,RhsBlockCols> (enforced by the template
   * parameter).  The result has block type
   * \c Matrix<Scalar,BlockRows,RhsBlockCols>.
   *
   * \pre  \c this->blockCols() == rhs.blockRows() (checked at runtime).
   */
  template <int RhsBlockCols>
  BlockSparseMatrix<Scalar_, Options_, BlockRows_, RhsBlockCols, StorageIndex_>
  operator*(const BlockSparseMatrix<Scalar_, Options_, BlockCols_, RhsBlockCols, StorageIndex_>& rhs) const;

  // -------------------------------------------------------------------------
  // Approximate equality (useful for testing)
  // -------------------------------------------------------------------------
  bool isApprox(const BlockSparseMatrix& other,
                const typename NumTraits<Scalar>::Real& prec =
                    NumTraits<Scalar>::dummy_precision()) const {
    return toSparse().isApprox(other.toSparse(), prec);
  }

 private:
  // -------------------------------------------------------------------------
  // Storage
  // -------------------------------------------------------------------------
  Index m_blockOuterSize;  // block-cols (ColMajor) or block-rows (RowMajor)
  Index m_blockInnerSize;  // block-rows (ColMajor) or block-cols (RowMajor)

  Array<StorageIndex, Dynamic, 1> m_outerIndex;  // size: m_blockOuterSize + 1
  Array<StorageIndex, Dynamic, 1> m_innerIndex;  // size: nonZeroBlocks()
  // Block values stored consecutively in column-major order; block k occupies
  // m_values[k*BlockSize .. (k+1)*BlockSize - 1].
  Array<Scalar, Dynamic, 1> m_values;  // size: nonZeroBlocks() * BlockSize

  // -------------------------------------------------------------------------
  // MultiInnerIterator
  //
  // Simultaneously walks BlockOuterSize_ consecutive outer vectors of a
  // compressed SparseMatrix, yielding scalar entries one at a time in
  // non-decreasing inner-index order.  On each step the sub-iterator with
  // the smallest current inner index is the "active" one.
  //
  // BlockOuterSize_ is BlockCols_ for ColMajor (one block-column at a time)
  // and BlockRows_ for RowMajor (one block-row at a time).
  // -------------------------------------------------------------------------
  template <typename SparseMatrixType>
  class MultiInnerIterator {
    using StorageIndex = typename SparseMatrixType::StorageIndex;
    static constexpr int BlockOuterSize_ = IsRowMajor ? BlockRows_ : BlockCols_;

   public:
    MultiInnerIterator(const SparseMatrixType& mat, Index outerBase)
        : m_outerPtr(mat.outerIndexPtr()),
          m_innerPtr(mat.innerIndexPtr()),
          m_valuePtr(mat.valuePtr()),
          m_outerBase(outerBase) {
      for (int k = 0; k < BlockOuterSize_; ++k)
        m_pos[k] = m_outerPtr[outerBase + k];
      advance();
    }

    EIGEN_STRONG_INLINE operator bool() const { return m_valid; }

    EIGEN_STRONG_INLINE MultiInnerIterator& operator++() {
      ++m_pos[m_active];
      advance();
      return *this;
    }

    // Absolute outer index of the current entry.
    EIGEN_STRONG_INLINE Index outer() const { return m_outerBase + m_active; }
    // Inner index of the current entry.
    EIGEN_STRONG_INLINE StorageIndex index() const { return m_innerPtr[m_pos[m_active]]; }
    // Scalar value of the current entry.
    EIGEN_STRONG_INLINE Scalar value() const { return m_valuePtr[m_pos[m_active]]; }

   private:
    // Find the sub-iterator with the smallest inner index.
    void advance() {
      m_valid = false;
      for (int k = 0; k < BlockOuterSize_; ++k) {
        if (m_pos[k] < m_outerPtr[m_outerBase + k + 1]) {
          if (!m_valid || m_innerPtr[m_pos[k]] < m_innerPtr[m_pos[m_active]]) {
            m_active = k;
            m_valid = true;
          }
        }
      }
    }

    const StorageIndex* m_outerPtr;
    const StorageIndex* m_innerPtr;
    const Scalar* m_valuePtr;
    Index m_outerBase;
    StorageIndex m_pos[BlockOuterSize_];
    int m_active = 0;
    bool m_valid = false;
  };

  // Allow other instantiations of BlockSparseMatrix to access private members
  // (needed by operator*).
  template <typename, int, int, int, typename>
  friend class BlockSparseMatrix;
};

// =============================================================================
// Out-of-line method definitions
// =============================================================================

// -----------------------------------------------------------------------------
// setFromTriplets
// -----------------------------------------------------------------------------

template <typename Scalar_, int Options_, int BlockRows_, int BlockCols_, typename StorageIndex_>
template <typename InputIterator>
void BlockSparseMatrix<Scalar_, Options_, BlockRows_, BlockCols_, StorageIndex_>::setFromTriplets(
    InputIterator begin, InputIterator end) {
  const Index n = static_cast<Index>(std::distance(begin, end));

  // Copy triplet coordinates and block values into Eigen arrays.
  Array<StorageIndex, Dynamic, 1> tOuter(n), tInner(n);
  Array<Scalar, Dynamic, 1> tValues(n * Index(BlockSize));

  Index k = 0;
  for (InputIterator it = begin; it != end; ++it, ++k) {
    tOuter(k) = IsRowMajor ? StorageIndex(it->row()) : StorageIndex(it->col());
    tInner(k) = IsRowMajor ? StorageIndex(it->col()) : StorageIndex(it->row());
    Map<BlockType>(tValues.data() + k * Index(BlockSize)) = it->value();
  }

  // Compute a sort permutation by (outer, inner).
  Array<Index, Dynamic, 1> perm(n);
  std::iota(perm.data(), perm.data() + n, Index(0));
  std::sort(perm.data(), perm.data() + n, [&](Index a, Index b) {
    return tOuter(a) != tOuter(b) ? tOuter(a) < tOuter(b) : tInner(a) < tInner(b);
  });

  // Reset and pre-allocate (worst case: all n triplets are distinct blocks).
  m_outerIndex.resize(m_blockOuterSize + 1);
  m_outerIndex.setZero();
  m_innerIndex.resize(n);
  m_values.resize(n * Index(BlockSize));

  Index nnz = 0;
  k = 0;
  while (k < n) {
    const Index pi = perm(k);
    const StorageIndex outer = tOuter(pi);
    const StorageIndex inner = tInner(pi);

    BlockType block = Map<const BlockType>(tValues.data() + pi * Index(BlockSize));
    ++k;

    // Accumulate duplicate entries at the same (outer, inner) position.
    while (k < n) {
      const Index pk = perm(k);
      if (tOuter(pk) != outer || tInner(pk) != inner) break;
      block += Map<const BlockType>(tValues.data() + pk * Index(BlockSize));
      ++k;
    }

    m_innerIndex(nnz) = inner;
    m_values.segment(nnz * Index(BlockSize), Index(BlockSize)) =
        Map<const Array<Scalar, BlockSize, 1>>(block.data());
    m_outerIndex(Index(outer) + 1)++;
    ++nnz;
  }

  // Trim to actual number of unique blocks.
  m_innerIndex.conservativeResize(nnz);
  m_values.conservativeResize(nnz * Index(BlockSize));

  // Convert per-outer block counts to prefix sums.
  for (Index j = 0; j < m_blockOuterSize; ++j) {
    m_outerIndex(j + 1) += m_outerIndex(j);
  }
}

// -----------------------------------------------------------------------------
// toSparse
// -----------------------------------------------------------------------------

template <typename Scalar_, int Options_, int BlockRows_, int BlockCols_, typename StorageIndex_>
SparseMatrix<Scalar_, Options_, StorageIndex_>
BlockSparseMatrix<Scalar_, Options_, BlockRows_, BlockCols_, StorageIndex_>::toSparse() const {
  SparseMatrix<Scalar_, Options_, StorageIndex_> result(rows(), cols());
  result.reserve(nonZeroBlocks() * Index(BlockSize));

  if (!IsRowMajor) {
    // ColMajor: outer = block-column J.  Emit scalar columns J*BlockCols+c
    // in order c = 0..BlockCols-1.  Within each scalar column, blocks are
    // sorted by bI (block-row), so scalar rows bI*BlockRows+r are increasing.
    for (Index J = 0; J < m_blockOuterSize; ++J) {
      for (Index c = 0; c < BlockCols_; ++c) {
        result.startVec(J * BlockCols_ + c);
        for (Index id = m_outerIndex(J); id < m_outerIndex(J + 1); ++id) {
          const Index bI = m_innerIndex(id);
          ConstBlockMap blk = blockRef(id);
          for (Index r = 0; r < BlockRows_; ++r) {
            result.insertBack(bI * BlockRows_ + r, J * BlockCols_ + c) = blk(r, c);
          }
        }
      }
    }
  } else {
    // RowMajor: outer = block-row bI.  Emit scalar rows bI*BlockRows+r
    // in order r = 0..BlockRows-1.  Within each scalar row, blocks are
    // sorted by J (block-col), so scalar cols J*BlockCols+c are increasing.
    for (Index bI = 0; bI < m_blockOuterSize; ++bI) {
      for (Index r = 0; r < BlockRows_; ++r) {
        result.startVec(bI * BlockRows_ + r);
        for (Index id = m_outerIndex(bI); id < m_outerIndex(bI + 1); ++id) {
          const Index J = m_innerIndex(id);
          ConstBlockMap blk = blockRef(id);
          for (Index c = 0; c < BlockCols_; ++c) {
            result.insertBack(bI * BlockRows_ + r, J * BlockCols_ + c) = blk(r, c);
          }
        }
      }
    }
  }

  result.finalize();
  return result;
}

// -----------------------------------------------------------------------------
// fromSparse
// -----------------------------------------------------------------------------

template <typename Scalar_, int Options_, int BlockRows_, int BlockCols_, typename StorageIndex_>
BlockSparseMatrix<Scalar_, Options_, BlockRows_, BlockCols_, StorageIndex_>
BlockSparseMatrix<Scalar_, Options_, BlockRows_, BlockCols_, StorageIndex_>::fromSparse(
    const SparseMatrix<Scalar_, Options_, StorageIndex_>& sp) {
  eigen_assert(sp.rows() % BlockRows_ == 0 && "matrix rows not divisible by BlockRows");
  eigen_assert(sp.cols() % BlockCols_ == 0 && "matrix cols not divisible by BlockCols");
  eigen_assert(sp.isCompressed() && "fromSparse requires a compressed SparseMatrix");

  const Index bRows = sp.rows() / BlockRows_;
  const Index bCols = sp.cols() / BlockCols_;

  // BlockOuterSize: how many consecutive outer vectors form one block-outer strip.
  // BlockInnerSize: the inner dimension of each block.
  constexpr Index BlockOuterSize = IsRowMajor ? BlockRows_ : BlockCols_;
  constexpr Index BlockInnerSize = IsRowMajor ? BlockCols_ : BlockRows_;

  using SpMat = SparseMatrix<Scalar_, Options_, StorageIndex_>;

  BlockSparseMatrix result(bRows, bCols);

  // Pass 1: count the number of unique block-inner indices per block-outer,
  // by scanning each group of BlockOuterSize consecutive outer vectors together.
  for (Index outerBlock = 0; outerBlock < result.m_blockOuterSize; ++outerBlock) {
    StorageIndex_ prevInnerBlock = StorageIndex_(-1);
    for (MultiInnerIterator<SpMat> it(sp, outerBlock * BlockOuterSize); it; ++it) {
      const StorageIndex_ innerBlock = it.index() / StorageIndex_(BlockInnerSize);
      if (innerBlock != prevInnerBlock) {
        result.m_outerIndex(outerBlock + 1)++;
        prevInnerBlock = innerBlock;
      }
    }
  }

  // Prefix sum → result.m_outerIndex becomes the standard CSC/CSR outer pointer.
  for (Index j = 0; j < result.m_blockOuterSize; ++j)
    result.m_outerIndex(j + 1) += result.m_outerIndex(j);

  const Index nBlocks = result.m_outerIndex(result.m_blockOuterSize);
  result.m_innerIndex.resize(nBlocks);
  result.m_values.setZero(nBlocks * Index(BlockSize));

  // Pass 2: scatter each scalar entry directly into its position within the
  // pre-zeroed block value array.
  for (Index outerBlock = 0; outerBlock < result.m_blockOuterSize; ++outerBlock) {
    Index blockId = result.m_outerIndex(outerBlock) - 1;  // incremented on first new block
    StorageIndex_ prevInnerBlock = StorageIndex_(-1);

    for (MultiInnerIterator<SpMat> it(sp, outerBlock * BlockOuterSize); it; ++it) {
      const Index absOuter = it.outer();           // absolute outer index in sp
      const StorageIndex_ innerIdx = it.index();   // inner index in sp
      const StorageIndex_ innerBlock = innerIdx / StorageIndex_(BlockInnerSize);

      if (innerBlock != prevInnerBlock) {
        ++blockId;
        result.m_innerIndex(blockId) = innerBlock;
        prevInnerBlock = innerBlock;
      }

      // Scatter into column-major block storage.
      // ColMajor: outer=col, inner=row → offset = localOuter*BlockRows_ + localInner
      // RowMajor: outer=row, inner=col → offset = localInner*BlockRows_ + localOuter
      const Index localOuter = absOuter % BlockOuterSize;
      const Index localInner = Index(innerIdx) % BlockInnerSize;
      const Index offset = IsRowMajor ? (localInner * Index(BlockRows_) + localOuter)
                                      : (localOuter * Index(BlockRows_) + localInner);

      result.m_values(blockId * Index(BlockSize) + offset) = it.value();
    }
  }

  return result;
}

// -----------------------------------------------------------------------------
// operator+
// -----------------------------------------------------------------------------

template <typename Scalar_, int Options_, int BlockRows_, int BlockCols_, typename StorageIndex_>
BlockSparseMatrix<Scalar_, Options_, BlockRows_, BlockCols_, StorageIndex_>
BlockSparseMatrix<Scalar_, Options_, BlockRows_, BlockCols_, StorageIndex_>::operator+(
    const BlockSparseMatrix& other) const {
  eigen_assert(blockRows() == other.blockRows() && blockCols() == other.blockCols() &&
               "BlockSparseMatrix size mismatch in operator+");

  BlockSparseMatrix result(blockRows(), blockCols());

  // Pre-allocate worst case (union of both sparsity patterns).
  const Index maxNnz = nonZeroBlocks() + other.nonZeroBlocks();
  result.m_innerIndex.resize(maxNnz);
  result.m_values.resize(maxNnz * Index(BlockSize));

  Index nnz = 0;

  for (Index J = 0; J < m_blockOuterSize; ++J) {
    result.m_outerIndex(J) = StorageIndex_(nnz);

    Index aId = static_cast<Index>(m_outerIndex(J));
    const Index aEnd = static_cast<Index>(m_outerIndex(J + 1));
    Index bId = static_cast<Index>(other.m_outerIndex(J));
    const Index bEnd = static_cast<Index>(other.m_outerIndex(J + 1));

    while (aId < aEnd || bId < bEnd) {
      const bool hasA = aId < aEnd;
      const bool hasB = bId < bEnd;
      // Use -1 as a sentinel "infinity" for the absent side (inner indices >= 0).
      const Index aInner = hasA ? static_cast<Index>(m_innerIndex(aId)) : Index(-1);
      const Index bInner = hasB ? static_cast<Index>(other.m_innerIndex(bId)) : Index(-1);

      BlockType block;
      StorageIndex_ inner;

      if (hasA && (!hasB || aInner < bInner)) {
        inner = StorageIndex_(aInner);
        block = blockRef(aId);
        ++aId;
      } else if (hasB && (!hasA || bInner < aInner)) {
        inner = StorageIndex_(bInner);
        block = other.blockRef(bId);
        ++bId;
      } else {
        // aInner == bInner: both are present, merge.
        inner = StorageIndex_(aInner);
        block = blockRef(aId) + other.blockRef(bId);
        ++aId;
        ++bId;
      }

      result.m_innerIndex(nnz) = inner;
      result.m_values.segment(nnz * Index(BlockSize), Index(BlockSize)) =
          Map<const Array<Scalar, BlockSize, 1>>(block.data());
      ++nnz;
    }
  }
  result.m_outerIndex(m_blockOuterSize) = StorageIndex_(nnz);

  // Trim to actual size.
  result.m_innerIndex.conservativeResize(nnz);
  result.m_values.conservativeResize(nnz * Index(BlockSize));

  return result;
}

// -----------------------------------------------------------------------------
// operator* (block-sparse product)
// -----------------------------------------------------------------------------

template <typename Scalar_, int Options_, int BlockRows_, int BlockCols_, typename StorageIndex_>
template <int RhsBlockCols>
BlockSparseMatrix<Scalar_, Options_, BlockRows_, RhsBlockCols, StorageIndex_>
BlockSparseMatrix<Scalar_, Options_, BlockRows_, BlockCols_, StorageIndex_>::operator*(
    const BlockSparseMatrix<Scalar_, Options_, BlockCols_, RhsBlockCols, StorageIndex_>& rhs) const {
  using ResultMatrix = BlockSparseMatrix<Scalar_, Options_, BlockRows_, RhsBlockCols, StorageIndex_>;
  using ResultBlock = Matrix<Scalar_, BlockRows_, RhsBlockCols>;
  constexpr int ResultBlockSize = BlockRows_ * RhsBlockCols;

  eigen_assert(blockCols() == rhs.blockRows() &&
               "BlockSparseMatrix product: lhs.blockCols() != rhs.blockRows()");

  const Index cBlockRows = blockRows();
  const Index cBlockCols = rhs.blockCols();
  ResultMatrix result(cBlockRows, cBlockCols);

  // For ColMajor: mask / accum indexed by block-row (size = cBlockRows).
  // For RowMajor: mask / accum indexed by block-col (size = cBlockCols).
  const Index maskSize = IsRowMajor ? cBlockCols : cBlockRows;
  Array<uint8_t, Dynamic, 1> mask = Array<uint8_t, Dynamic, 1>::Zero(maskSize);
  Array<Scalar_, Dynamic, 1> accumData(maskSize * Index(ResultBlockSize));
  Array<Index, Dynamic, 1> indices(maskSize);
  Index nIndices = 0;

  // Pre-allocate result storage (worst case = dense block pattern).
  const Index cOuterSize = result.m_blockOuterSize;
  const Index maxResultNnz = cBlockRows * cBlockCols;
  result.m_innerIndex.resize(maxResultNnz);
  result.m_values.resize(maxResultNnz * Index(ResultBlockSize));
  Index nnz = 0;

  for (Index out = 0; out < cOuterSize; ++out) {
    result.m_outerIndex(out) = StorageIndex_(nnz);

    if (!IsRowMajor) {
      // ColMajor: out is block-column J of the result.
      // For each block B(K,J) and each block A(bI,K): C(bI,J) += A(bI,K)*B(K,J).
      const Index J = out;
      for (Index rhsId = rhs.m_outerIndex(J); rhsId < rhs.m_outerIndex(J + 1); ++rhsId) {
        const Index K = rhs.m_innerIndex(rhsId);
        const auto Bkj = rhs.blockRef(rhsId);
        for (Index lhsId = m_outerIndex(K); lhsId < m_outerIndex(K + 1); ++lhsId) {
          const Index bI = m_innerIndex(lhsId);
          if (!mask(bI)) {
            mask(bI) = 1;
            Map<ResultBlock>(accumData.data() + bI * Index(ResultBlockSize)).noalias() =
                blockRef(lhsId) * Bkj;
            indices(nIndices++) = bI;
          } else {
            Map<ResultBlock>(accumData.data() + bI * Index(ResultBlockSize)).noalias() +=
                blockRef(lhsId) * Bkj;
          }
        }
      }
    } else {
      // RowMajor: out is block-row bI of the result.
      // For each block A(bI,K) and each block B(K,J): C(bI,J) += A(bI,K)*B(K,J).
      const Index bI = out;
      for (Index lhsId = m_outerIndex(bI); lhsId < m_outerIndex(bI + 1); ++lhsId) {
        const Index K = m_innerIndex(lhsId);
        const auto Aik = blockRef(lhsId);
        for (Index rhsId = rhs.m_outerIndex(K); rhsId < rhs.m_outerIndex(K + 1); ++rhsId) {
          const Index J = rhs.m_innerIndex(rhsId);
          if (!mask(J)) {
            mask(J) = 1;
            Map<ResultBlock>(accumData.data() + J * Index(ResultBlockSize)).noalias() =
                Aik * rhs.blockRef(rhsId);
            indices(nIndices++) = J;
          } else {
            Map<ResultBlock>(accumData.data() + J * Index(ResultBlockSize)).noalias() +=
                Aik * rhs.blockRef(rhsId);
          }
        }
      }
    }

    // Sort the accumulated indices so the result's inner index array is sorted.
    std::sort(indices.data(), indices.data() + nIndices);
    for (Index ki = 0; ki < nIndices; ++ki) {
      const Index idx = indices(ki);
      result.m_innerIndex(nnz) = StorageIndex_(idx);
      result.m_values.segment(nnz * Index(ResultBlockSize), Index(ResultBlockSize)) =
          Map<const Array<Scalar_, ResultBlockSize, 1>>(accumData.data() + idx * Index(ResultBlockSize));
      mask(idx) = 0;
      ++nnz;
    }
    nIndices = 0;
  }
  result.m_outerIndex(cOuterSize) = StorageIndex_(nnz);

  // Trim to actual number of result blocks.
  result.m_innerIndex.conservativeResize(nnz);
  result.m_values.conservativeResize(nnz * Index(ResultBlockSize));

  return result;
}

}  // end namespace Eigen

#endif  // EIGEN_BLOCKSPARSEMATRIX_H
