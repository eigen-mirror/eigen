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
#include <map>
#include <utility>
#include <vector>

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
  BlockSparseMatrix() : m_blockOuterSize(0), m_blockInnerSize(0) { m_outerIndex.push_back(StorageIndex(0)); }

  /** Construct a zero matrix with the given number of block-rows and block-columns. */
  BlockSparseMatrix(Index blockRows, Index blockCols)
      : m_blockOuterSize(IsRowMajor ? blockRows : blockCols),
        m_blockInnerSize(IsRowMajor ? blockCols : blockRows) {
    m_outerIndex.assign(m_blockOuterSize + 1, StorageIndex(0));
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
  Index nonZeroBlocks() const { return static_cast<Index>(m_innerIndex.size()); }
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
          m_id(static_cast<Index>(mat.m_outerIndex[outer])),
          m_end(static_cast<Index>(mat.m_outerIndex[outer + 1])),
          m_outer(outer) {}

    EIGEN_STRONG_INLINE operator bool() const { return m_id < m_end; }
    EIGEN_STRONG_INLINE InnerIterator& operator++() {
      ++m_id;
      return *this;
    }

    /** Current block outer index (block-col for ColMajor, block-row for RowMajor). */
    EIGEN_STRONG_INLINE Index outer() const { return m_outer; }
    /** Current block inner index (block-row for ColMajor, block-col for RowMajor). */
    EIGEN_STRONG_INLINE Index index() const { return static_cast<Index>(m_mat.m_innerIndex[m_id]); }
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
    m_outerIndex.assign(m_blockOuterSize + 1, StorageIndex(0));
    m_innerIndex.clear();
    m_values.clear();
  }

  /** Clear all stored blocks while keeping the matrix dimensions. */
  void setZero() {
    m_outerIndex.assign(m_blockOuterSize + 1, StorageIndex(0));
    m_innerIndex.clear();
    m_values.clear();
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
    const StorageIndex* beg = m_innerIndex.data() + m_outerIndex[bOuter];
    const StorageIndex* fin = m_innerIndex.data() + m_outerIndex[bOuter + 1];
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
    for (Scalar& v : result.m_values) v = -v;
    return result;
  }

  BlockSparseMatrix& operator+=(const BlockSparseMatrix& other) { return *this = *this + other; }
  BlockSparseMatrix& operator-=(const BlockSparseMatrix& other) { return *this = *this - other; }

  /** Scalar multiplication (returns a new matrix). */
  BlockSparseMatrix operator*(const Scalar& s) const {
    BlockSparseMatrix result(*this);
    for (Scalar& v : result.m_values) v *= s;
    return result;
  }
  BlockSparseMatrix& operator*=(const Scalar& s) {
    for (Scalar& v : m_values) v *= s;
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
      for (Index id = static_cast<Index>(m_outerIndex[out]);
           id < static_cast<Index>(m_outerIndex[out + 1]); ++id) {
        const Index inner = static_cast<Index>(m_innerIndex[id]);
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
      for (Index id = static_cast<Index>(bsm.m_outerIndex[out]);
           id < static_cast<Index>(bsm.m_outerIndex[out + 1]); ++id) {
        const Index inner = static_cast<Index>(bsm.m_innerIndex[id]);
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

  std::vector<StorageIndex> m_outerIndex;  // size: m_blockOuterSize + 1
  std::vector<StorageIndex> m_innerIndex;  // size: nonZeroBlocks()
  // Block values stored consecutively in column-major order; block k occupies
  // m_values[k*BlockSize .. (k+1)*BlockSize - 1].
  std::vector<Scalar> m_values;  // size: nonZeroBlocks() * BlockSize

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
  using TripType = typename std::iterator_traits<InputIterator>::value_type;

  std::vector<TripType> triplets(begin, end);

  // Sort by (outer, inner) to enable a single-pass merge of duplicates.
  std::sort(triplets.begin(), triplets.end(), [](const TripType& a, const TripType& b) {
    const Index aOuter = IsRowMajor ? Index(a.row()) : Index(a.col());
    const Index bOuter = IsRowMajor ? Index(b.row()) : Index(b.col());
    if (aOuter != bOuter) return aOuter < bOuter;
    const Index aInner = IsRowMajor ? Index(a.col()) : Index(a.row());
    const Index bInner = IsRowMajor ? Index(b.col()) : Index(b.row());
    return aInner < bInner;
  });

  m_outerIndex.assign(m_blockOuterSize + 1, StorageIndex(0));
  m_innerIndex.clear();
  m_values.clear();

  const Index n = static_cast<Index>(triplets.size());
  Index k = 0;
  while (k < n) {
    const Index outer = IsRowMajor ? Index(triplets[k].row()) : Index(triplets[k].col());
    const Index inner = IsRowMajor ? Index(triplets[k].col()) : Index(triplets[k].row());

    BlockType block = triplets[k].value();
    ++k;

    // Accumulate duplicate entries at the same (outer, inner) position.
    while (k < n) {
      const Index ko = IsRowMajor ? Index(triplets[k].row()) : Index(triplets[k].col());
      const Index ki = IsRowMajor ? Index(triplets[k].col()) : Index(triplets[k].row());
      if (ko != outer || ki != inner) break;
      block += triplets[k].value();
      ++k;
    }

    m_innerIndex.push_back(StorageIndex(inner));
    m_values.insert(m_values.end(), block.data(), block.data() + Index(BlockSize));
    m_outerIndex[outer + 1]++;
  }

  // Convert per-outer block counts to prefix sums.
  for (Index j = 0; j < m_blockOuterSize; ++j) {
    m_outerIndex[j + 1] += m_outerIndex[j];
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
        for (Index id = m_outerIndex[J]; id < m_outerIndex[J + 1]; ++id) {
          const Index bI = m_innerIndex[id];
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
        for (Index id = m_outerIndex[bI]; id < m_outerIndex[bI + 1]; ++id) {
          const Index J = m_innerIndex[id];
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

  const Index bRows = sp.rows() / BlockRows_;
  const Index bCols = sp.cols() / BlockCols_;

  // Accumulate scalar entries into their block positions.
  using Key = std::pair<StorageIndex_, StorageIndex_>;
  std::map<Key, BlockType> blockMap;

  for (Index j = 0; j < sp.outerSize(); ++j) {
    for (typename SparseMatrix<Scalar_, Options_, StorageIndex_>::InnerIterator spIt(sp, j); spIt;
         ++spIt) {
      const Index row = spIt.row();
      const Index col = spIt.col();
      const StorageIndex_ bI = StorageIndex_(row / BlockRows_);
      const StorageIndex_ bJ = StorageIndex_(col / BlockCols_);
      const Index r = row % Index(BlockRows_);
      const Index c = col % Index(BlockCols_);

      auto [mapIt, inserted] = blockMap.try_emplace(Key{bI, bJ}, BlockType::Zero());
      mapIt->second(r, c) += spIt.value();
    }
  }

  std::vector<TripletType> triplets;
  triplets.reserve(blockMap.size());
  for (const auto& [key, block] : blockMap) {
    triplets.emplace_back(key.first, key.second, block);
  }

  BlockSparseMatrix result(bRows, bCols);
  result.setFromTriplets(triplets.begin(), triplets.end());
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
  Index nnz = 0;

  for (Index J = 0; J < m_blockOuterSize; ++J) {
    result.m_outerIndex[J] = StorageIndex_(nnz);

    Index aId = static_cast<Index>(m_outerIndex[J]);
    const Index aEnd = static_cast<Index>(m_outerIndex[J + 1]);
    Index bId = static_cast<Index>(other.m_outerIndex[J]);
    const Index bEnd = static_cast<Index>(other.m_outerIndex[J + 1]);

    while (aId < aEnd || bId < bEnd) {
      const bool hasA = aId < aEnd;
      const bool hasB = bId < bEnd;
      // Use -1 as a sentinel "infinity" for the absent side (inner indices >= 0).
      const Index aInner = hasA ? static_cast<Index>(m_innerIndex[aId]) : Index(-1);
      const Index bInner = hasB ? static_cast<Index>(other.m_innerIndex[bId]) : Index(-1);

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

      result.m_innerIndex.push_back(inner);
      result.m_values.insert(result.m_values.end(), block.data(), block.data() + Index(BlockSize));
      ++nnz;
    }
  }
  result.m_outerIndex[m_blockOuterSize] = StorageIndex_(nnz);
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

  eigen_assert(blockCols() == rhs.blockRows() &&
               "BlockSparseMatrix product: lhs.blockCols() != rhs.blockRows()");

  const Index cBlockRows = blockRows();
  const Index cBlockCols = rhs.blockCols();
  ResultMatrix result(cBlockRows, cBlockCols);

  // For ColMajor: mask / accum are indexed by block-row I (size = cBlockRows).
  // For RowMajor: mask / accum are indexed by block-col J (size = cBlockCols).
  const Index maskSize = IsRowMajor ? cBlockCols : cBlockRows;
  std::vector<uint8_t> mask(maskSize, 0);
  std::vector<ResultBlock> accum(maskSize);
  std::vector<Index> indices;

  // cOuterSize == result.m_blockOuterSize.
  // ColMajor: result outer = block-cols of B = rhs.blockOuterSize().
  // RowMajor: result outer = block-rows of A = this->blockOuterSize().
  const Index cOuterSize = result.m_blockOuterSize;
  Index nnz = 0;

  for (Index out = 0; out < cOuterSize; ++out) {
    result.m_outerIndex[out] = StorageIndex_(nnz);

    if (!IsRowMajor) {
      // ColMajor: out is block-column J of the result.
      // For each block B(K,J) and each block A(bI,K): C(bI,J) += A(bI,K)*B(K,J).
      const Index J = out;
      for (Index rhsId = rhs.m_outerIndex[J]; rhsId < rhs.m_outerIndex[J + 1]; ++rhsId) {
        const Index K = rhs.m_innerIndex[rhsId];
        const auto Bkj = rhs.blockRef(rhsId);  // Map<const Matrix<Scalar_,BlockCols_,RhsBlockCols>>
        for (Index lhsId = m_outerIndex[K]; lhsId < m_outerIndex[K + 1]; ++lhsId) {
          const Index bI = m_innerIndex[lhsId];
          if (!mask[bI]) {
            mask[bI] = 1;
            accum[bI].noalias() = blockRef(lhsId) * Bkj;
            indices.push_back(bI);
          } else {
            accum[bI].noalias() += blockRef(lhsId) * Bkj;
          }
        }
      }
    } else {
      // RowMajor: out is block-row bI of the result.
      // For each block A(bI,K) and each block B(K,J): C(bI,J) += A(bI,K)*B(K,J).
      const Index bI = out;
      for (Index lhsId = m_outerIndex[bI]; lhsId < m_outerIndex[bI + 1]; ++lhsId) {
        const Index K = m_innerIndex[lhsId];
        const auto Aik = blockRef(lhsId);  // Map<const Matrix<Scalar_,BlockRows_,BlockCols_>>
        for (Index rhsId = rhs.m_outerIndex[K]; rhsId < rhs.m_outerIndex[K + 1]; ++rhsId) {
          const Index J = rhs.m_innerIndex[rhsId];
          if (!mask[J]) {
            mask[J] = 1;
            accum[J].noalias() = Aik * rhs.blockRef(rhsId);
            indices.push_back(J);
          } else {
            accum[J].noalias() += Aik * rhs.blockRef(rhsId);
          }
        }
      }
    }

    // Sort the accumulated indices so the result's inner index array is sorted.
    std::sort(indices.begin(), indices.end());
    for (const Index idx : indices) {
      result.m_innerIndex.push_back(StorageIndex_(idx));
      result.m_values.insert(result.m_values.end(), accum[idx].data(),
                             accum[idx].data() + Index(BlockRows_ * RhsBlockCols));
      mask[idx] = 0;
      ++nnz;
    }
    indices.clear();
  }
  result.m_outerIndex[cOuterSize] = StorageIndex_(nnz);
  return result;
}

}  // end namespace Eigen

#endif  // EIGEN_BLOCKSPARSEMATRIX_H
