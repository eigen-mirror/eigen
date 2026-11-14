// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_MULTIFRONTAL_LDLT_H
#define EIGEN_MULTIFRONTAL_LDLT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \ingroup SparseCholesky_Module
 * \class MultifrontalLDLT
 * \brief Sparse multifrontal LDL^* factorization of a symmetric or Hermitian indefinite matrix
 *
 * This class computes \f$ P A P^T = L D L^* \f$, where \f$ P \f$ is a permutation, \f$ L \f$ is unit lower triangular
 * and \f$ D \f$ is block diagonal with 1x1 and 2x2 blocks. This is the form of the dense BunchKaufman, with different
 * pivot tests.
 *
 * The matrix is factored as a tree of small dense frontal matrices. A front can only pivot on its fully summed
 * columns, so a pivot must also bound the entries in rows that aren't fully summed yet. With threshold \f$ u \f$, a 1x1
 * pivot on column \f$ j \f$ is accepted if
 * \f[ |a_{jj}| \ge u \max_{i \ne j} |a_{ij}| \f]
 * and a 2x2 pivot \f$ D \f$ on columns \f$ j \f$ and \f$ r \f$ is accepted if
 * \f[ |D^{-1}| \left( \max_{i \ne j,r} |a_{ij}|, \max_{i \ne j,r} |a_{ir}| \right)^T \le (1/u, 1/u)^T
 *     \quad \text{and} \quad |\det D| \ge |a_{rj}|^2 / 2 . \f]
 * Columns with no acceptable pivot are delayed to the parent front, which adds them to its fully summed columns. The
 * root front has no other rows to bound, so \f$ u \le 1/2 \f$ guarantees it can pivot (Duff and Reid, 1983).
 *
 * analyzePattern() merges fronts into their parents if the merged front would hold few enough explicit zeros (Ashcraft
 * and Grimes, 1989). This costs some fill for larger, fewer fronts: they need less assembly, spend more time in dense
 * kernels, and have more fully summed columns to pivot on, so they delay fewer pivots.
 *
 * Delayed pivots add fill that the symbolic analysis doesn't predict, so the memory the factorization needs depends on
 * the matrix's values. This class doesn't scale the matrix. The pivot tests compare entries with each other, so a
 * single large row can delay columns all the way up the tree. Equilibrate such a matrix first (e.g., with Ruiz
 * scaling).
 *
 * The matrix is read as Hermitian, so only the real part of each diagonal entry is read, even if its imaginary part is
 * not finite.
 *
 * The elimination tree and the assembly maps come from analyzePattern() and are reused by each factorize(). The
 * workspace keeps its capacity, so its storage only grows if a factorization needs more memory than any before it.
 * Dense products on large fronts can allocate temporaries above \c EIGEN_STACK_ALLOCATION_LIMIT.
 *
 * Each front is factored in panels: a pivot updates the columns of its panel, and the rest of the front is updated
 * once per panel by a level-3 product. Solving likewise applies each front's factor to all right-hand sides at once.
 *
 * References:
 *  - I. S. Duff and J. K. Reid, "The multifrontal solution of indefinite sparse symmetric linear equations", ACM TOMS
 *    9 (1983), pp. 302-325.
 *  - J. R. Bunch and L. Kaufman, "Some stable methods for calculating inertia and solving symmetric linear systems",
 *    Math. Comp. 31 (1977), pp. 163-179.
 *  - J. W. H. Liu, "The role of elimination trees in sparse factorization", SIAM J. Matrix Anal. Appl. 11 (1990),
 *    pp. 134-172.
 *  - C. Ashcraft and R. Grimes, "The influence of relaxed supernode partitions on the multifrontal method", ACM TOMS
 *    15 (1989), pp. 291-309.
 *  - J. R. Gilbert, E. G. Ng and B. W. Peyton, "An efficient algorithm to compute row and column counts for sparse
 *    Cholesky factorization", SIAM J. Matrix Anal. Appl. 15 (1994), pp. 1075-1091.
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam UpLo_ the triangular part that will be used for the factorization: Lower (default) or Upper. The other
 *               triangular part won't be read.
 * \tparam Ordering_ The fill-reducing ordering method. See the \link OrderingMethods_Module OrderingMethods \endlink
 *                   module for the list of built-in and external ordering methods.
 *
 * \implsparsesolverconcept
 *
 * \sa class SimplicialLDLT, class BunchKaufman, class AMDOrdering, class NaturalOrdering
 */
template <typename MatrixType_, int UpLo_ = Lower, typename Ordering_ = AMDOrdering<typename MatrixType_::StorageIndex>>
class MultifrontalLDLT : public SparseSolverBase<MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>> {
  using Base = SparseSolverBase<MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>>;
  using Base::m_isInitialized;

 public:
  using Base::_solve_impl;

  using MatrixType = MatrixType_;
  using OrderingType = Ordering_;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using StorageIndex = typename MatrixType::StorageIndex;
  using VectorType = Matrix<Scalar, Dynamic, 1>;
  using RealVectorType = Matrix<RealScalar, Dynamic, 1>;
  using PermutationType = PermutationMatrix<Dynamic, Dynamic, StorageIndex>;
  using FactorType = SparseMatrix<Scalar, ColMajor, StorageIndex>;

  EIGEN_STATIC_ASSERT_NON_INTEGER(Scalar)
  EIGEN_STATIC_ASSERT(UpLo_ == Lower || UpLo_ == Upper, MULTIFRONTALLDLT_ACCEPTS_UPPER_AND_LOWER_MODE_ONLY)

  static constexpr int UpLo = UpLo_;
  static constexpr int ColsAtCompileTime = Dynamic;
  static constexpr int MaxColsAtCompileTime = Dynamic;

  /** Default constructor */
  MultifrontalLDLT() = default;

  /** Constructs the factorization of \a matrix. */
  explicit MultifrontalLDLT(const MatrixType& matrix) { compute(matrix); }

  /** Sets the threshold \f$ u \f$ for accepting a pivot. It must be in (0, 1/2]. The default is 0.01.
   *
   * Smaller values delay fewer pivots, so they add less fill, but they allow larger entries in L.
   *
   * \returns a reference to \c *this.
   */
  MultifrontalLDLT& setPivotThreshold(const RealScalar& threshold) {
    eigen_assert(threshold > RealScalar(0) && threshold <= RealScalar(0.5));
    m_threshold = threshold;
    return *this;
  }

  /** \returns the pivot threshold. \sa setPivotThreshold() */
  const RealScalar& pivotThreshold() const { return m_threshold; }

  /** Computes the symbolic analysis and the factorization of \a matrix.
   *
   * \returns a reference to \c *this.
   */
  MultifrontalLDLT& compute(const MatrixType& matrix) {
    analyzePattern(matrix);
    factorize(matrix);
    return *this;
  }

  /** Performs the symbolic analysis of the sparsity pattern of \a matrix.
   *
   * \sa factorize(), compute()
   */
  void analyzePattern(const MatrixType& matrix);

  /** Performs the numeric factorization of \a matrix, which must have the sparsity pattern given to analyzePattern().
   *
   * \sa analyzePattern(), compute()
   */
  void factorize(const MatrixType& matrix);

  /** \brief Reports whether the previous factorization was successful.
   *
   * \returns \c Success if every pivot was finite and nonzero, \c NumericalIssue if a pivot is zero (the matrix is
   *          singular to working precision) or not finite (from a non-finite entry, a complex entry whose magnitude
   *          overflows, or an update overflowing on finite input), and \c InvalidInput if no factorization was
   *          computed. The imaginary part of a diagonal entry isn't read, so it doesn't affect the result. A zero
   *          pivot gets a zero solution, so solve() still returns the solution of the remaining blocks if every pivot
   *          is finite. If a pivot isn't finite, the result of solve() isn't meaningful.
   */
  ComputationInfo info() const {
    eigen_assert(m_isInitialized && "Decomposition is not initialized.");
    return m_info;
  }

  Index rows() const { return m_P.rows(); }
  Index cols() const { return m_P.cols(); }

  /** \returns the diagonal of D, in pivot order. */
  const RealVectorType& vectorD() const {
    eigen_assert(m_info != InvalidInput && "factorize() must be called before reading D.");
    return m_D;
  }

  /** \returns the subdiagonal of D, in pivot order. Entry \c k is nonzero if and only if pivots \c k and \c k+1 form a
   * 2x2 block. */
  const VectorType& subDiagonal() const {
    eigen_assert(m_info != InvalidInput && "factorize() must be called before reading D.");
    return m_subdiagonal;
  }

  /** \returns the unit lower triangular factor \f$ L \f$ of the previous factorization, in pivot order, with its unit
   * diagonal stored. It is assembled from the fronts on each call, in O(nonZeros() + rows()) time and memory, and it
   * keeps the explicit zeros that nonZeros() counts. \f$ L_{k+1,k} = 0 \f$ where pivots \c k and \c k+1 form a
   * 2x2 block of D.
   *
   * \sa matrixU(), vectorD(), subDiagonal(), permutationP()
   */
  FactorType matrixL() const;

  /** \returns the unit upper triangular factor \f$ U = L^* \f$ of the previous factorization. Like matrixL(), it is
   * assembled on each call.
   *
   * \sa matrixL()
   */
  FactorType matrixU() const { return matrixL().adjoint(); }

  /** \returns the permutation \f$ P \f$ of the previous factorization. It combines the fill-reducing ordering, the
   * reordering of the elimination tree, and the delayed pivots, so unlike in SimplicialLDLT, it depends on the matrix's
   * values and can change with each factorize().
   *
   * \sa permutationPinv()
   */
  const PermutationType& permutationP() const {
    eigen_assert(m_info != InvalidInput && "factorize() must be called before reading the permutation.");
    return m_P;
  }

  /** \returns the inverse of the permutation \f$ P \f$. Entry \c k of its \c indices() is the row of pivot \c k in the
   * matrix given to factorize().
   *
   * \sa permutationP()
   */
  const PermutationType& permutationPinv() const {
    eigen_assert(m_info != InvalidInput && "factorize() must be called before reading the permutation.");
    return m_Pinv;
  }

  /** \returns the determinant of the matrix, \f$ \exp(\log|\det A|) \f$ times signDeterminant(). It is not finite if
   * any pivot is not finite, and otherwise zero if any pivot is zero. It has O(n) complexity, since the factorization
   * has already been computed.
   *
   * \warning A determinant can overflow or underflow even if the factorization is accurate. logAbsDeterminant() and
   * signDeterminant() don't.
   *
   * \sa absDeterminant(), logAbsDeterminant(), signDeterminant()
   */
  Scalar determinant() const { return Scalar(absDeterminant()) * signDeterminant(); }

  /** \returns the absolute value of the determinant of the matrix.
   *
   * \warning Like determinant(), this can overflow or underflow.
   *
   * \sa determinant(), logAbsDeterminant()
   */
  RealScalar absDeterminant() const { return numext::exp(logAbsDeterminant()); }

  /** \returns the natural log of the absolute value of the determinant of the matrix,
   * \f[ \log|\det A| = \sum_{1 \times 1} \log|d_k| + \sum_{2 \times 2} \left( 2 \log s_k + \log|\delta_k| \right), \f]
   * where a 2x2 block of D is \f$ s_k \f$ times a block with largest magnitude 1 and determinant \f$ \delta_k \f$. It
   * is \f$ -\infty \f$ if a pivot is zero, and not finite if a pivot is not finite.
   *
   * \sa determinant(), signDeterminant()
   */
  RealScalar logAbsDeterminant() const;

  /** \returns the sign of the determinant of the matrix: \c 1, \c -1, or \c 0 if a pivot is zero or not finite. It is
   * \f$ (-1)^{n_-} \f$ for the inertia numNegative() = \f$ n_- \f$, so it doesn't overflow.
   *
   * \sa determinant(), logAbsDeterminant(), numNegative()
   */
  Scalar signDeterminant() const {
    if (numZero() > 0 || numPositive() + numNegative() < rows()) return Scalar(0);
    return Scalar(numNegative() % 2 == 0 ? 1 : -1);
  }

  /** \returns the number of positive eigenvalues, read off the pivots of D by Sylvester's law of inertia.
   *
   * The signs come from the computed pivots, so this is the inertia of a matrix close to A rather than necessarily of
   * A. A non-finite pivot block is left out of all three counts, so they add up to rows() only if every pivot is
   * finite; info() reports \c NumericalIssue otherwise.
   *
   * \sa numNegative(), numZero()
   */
  Index numPositive() const {
    eigen_assert(m_info != InvalidInput && "factorize() must be called before reading the inertia.");
    return m_numPositive;
  }

  /** \returns the number of negative eigenvalues. \sa numPositive() */
  Index numNegative() const {
    eigen_assert(m_info != InvalidInput && "factorize() must be called before reading the inertia.");
    return m_numNegative;
  }

  /** \returns the number of pivots that are exactly zero.
   *
   * \warning A nonzero count means that A is singular to working precision, not that it is exactly singular:
   * rounding can turn a tiny pivot into an exact zero, and equally leave a tiny pivot in place of one.
   *
   * \sa numPositive()
   */
  Index numZero() const {
    eigen_assert(m_info != InvalidInput && "factorize() must be called before reading the inertia.");
    return m_numZero;
  }

  /** \returns true if the matrix is positive semidefinite: every pivot is finite and no eigenvalue is negative. */
  bool isPositive() const { return numNegative() == 0 && numPositive() + numZero() == rows(); }

  /** \returns true if the matrix is negative semidefinite: every pivot is finite and no eigenvalue is positive. */
  bool isNegative() const { return numPositive() == 0 && numNegative() + numZero() == rows(); }

  /** \returns the number of entries of L stored below its unit diagonal in the previous factorization. This includes
   * the explicit zeros of merged fronts, so it can exceed the fill of an unmerged factorization such as SimplicialLDLT.
   */
  Index nonZeros() const {
    eigen_assert(m_info != InvalidInput && "factorize() must be called before reading the size of L.");
    return m_LNonZeros;
  }

#ifndef EIGEN_PARSED_BY_DOXYGEN
  /** \internal */
  template <typename Rhs, typename Dest>
  void _solve_impl(const MatrixBase<Rhs>& rhs, MatrixBase<Dest>& dest) const;
#endif  // EIGEN_PARSED_BY_DOXYGEN

 private:
  using FrontMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using FrontMap = Map<FrontMatrix>;
  using ConstFrontMap = Map<const FrontMatrix>;
  using VectorMap = Map<VectorType>;
  using ConstVectorMap = Map<const VectorType>;
  using IndexVectorType = Matrix<StorageIndex, Dynamic, 1>;

  // Columns of a front that a pivot updates directly. Wider panels do more of the work in level-2 updates, narrower
  // ones pay for more calls to the level-3 product.
  static constexpr Index kPanelWidth = 32;
  // Below this order a front is its own panel: there's too little work to repay a level-3 product.
  static constexpr Index kBlockedFrontMinOrder = 32;
  // Below this length, a plain loop beats the alignment peeling of a vectorized expression. Sparse KKT matrices are
  // dominated by fronts of a few columns, where this decides the factorization's speed.
  static constexpr Index kVectorizedMinLength = 16;
  // Up to this order, the loops over a front have compile-time bounds, so they unroll.
  static constexpr Index kSmallFrontMaxOrder = 8;
  // Below this mean length of the runs of a child's map into its parent, extend-add stays a scatter: walking short runs
  // costs more than the additions they'd vectorize.
  static constexpr Index kExtendAddMinRunLength = 4;
  // Below this size of L (order * pivots), solve() applies a front's factor in place rather than through dense products
  // on a gathered copy of its rows.
  static constexpr Index kDenseSolveMinSize = 4096;

  // An array whose storage survives popping and clearing, so refactoring doesn't reallocate or reinitialize it.
  //
  // This, m_front, and m_pivotScratch are aligned so that the factorization is bitwise reproducible. Vectorized kernels
  // switch between scalar and packet code at alignment boundaries, and the two can round differently.
  template <typename T>
  struct WorkStack {
    std::vector<T, aligned_allocator<T>> storage;
    std::size_t size = 0;

    // Returns where to write count new elements. This can invalidate earlier pointers into the array.
    T* push(std::size_t count) {
      if (storage.size() < size + count) storage.resize(numext::maxi(size + count, 2 * storage.size()));
      T* end = storage.data() + size;
      size += count;
      return end;
    }

    // Returns the last count elements.
    T* top(std::size_t count) { return storage.data() + (size - count); }

    void pop(std::size_t count) { size -= count; }
    void reserve(std::size_t count) {
      if (storage.size() < count) storage.resize(count);
    }
    void clear() { size = 0; }
  };

  // An entry of the input matrix and where it's assembled in its front. value is the entry's index in the values of the
  // compressed matrix. row and col don't count delayed columns.
  struct Assembly {
    StorageIndex value;
    StorageIndex row;
    StorageIndex col;
    // Whether the permutation moved the entry to the other triangle.
    bool conjugate;
  };

  // Where a front's factor is stored after factorize().
  struct FactoredFront {
    StorageIndex numPivots;
    StorageIndex order;
    // Start of the front's rows in m_factorRows.
    std::size_t rows;
    // Start of the front's order x numPivots columns of L in m_L. Only the part below the diagonal is meaningful.
    std::size_t values;
  };

  // The sizes are computed in std::size_t because Index can be narrower than the product.
  static std::size_t rectangleSize(Index rows, Index cols) {
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  }

  // Entries in the lower triangle of an n x n matrix, including its diagonal.
  static std::size_t packedSize(Index n) { return static_cast<std::size_t>(n) * (static_cast<std::size_t>(n) + 1) / 2; }

  // Entries strictly below the diagonal in the first numPivots columns of an order x order matrix.
  static std::size_t factorSize(Index numPivots, Index order) {
    return rectangleSize(numPivots, order) - packedSize(numPivots);
  }

  // dst -= alpha * src, where the arrays don't overlap.
  static void subtractScaled(Scalar* EIGEN_RESTRICT dst, const Scalar* EIGEN_RESTRICT src, const Scalar& alpha,
                             Index length) {
    if (length < kVectorizedMinLength) {
      for (Index i = 0; i < length; ++i) dst[i] -= src[i] * alpha;
    } else {
      VectorMap(dst, length).noalias() -= ConstVectorMap(src, length) * alpha;
    }
  }

  // dst -= alpha1 * src1 + alpha2 * src2, where dst doesn't overlap the others.
  static void subtractScaled(Scalar* EIGEN_RESTRICT dst, const Scalar* EIGEN_RESTRICT src1, const Scalar& alpha1,
                             const Scalar* EIGEN_RESTRICT src2, const Scalar& alpha2, Index length) {
    if (length < kVectorizedMinLength) {
      for (Index i = 0; i < length; ++i) dst[i] -= src1[i] * alpha1 + src2[i] * alpha2;
    } else {
      VectorMap(dst, length).noalias() -= ConstVectorMap(src1, length) * alpha1 + ConstVectorMap(src2, length) * alpha2;
    }
  }

  // dst += src, where the arrays don't overlap.
  static void addTo(Scalar* EIGEN_RESTRICT dst, const Scalar* EIGEN_RESTRICT src, Index length) {
    if (length < kVectorizedMinLength) {
      for (Index i = 0; i < length; ++i) dst[i] += src[i];
    } else {
      VectorMap(dst, length).noalias() += ConstVectorMap(src, length);
    }
  }

  // Returns the largest magnitude of a column segment, or zero if it's empty.
  static RealScalar largestMagnitude(const Scalar* column, Index length) {
    if (length >= kVectorizedMinLength) return ConstVectorMap(column, length).cwiseAbs().maxCoeff();
    RealScalar result(0);
    for (Index i = 0; i < length; ++i) result = numext::maxi(result, RealScalar(numext::abs(column[i])));
    return result;
  }

  // The rows of the order x order front being factored.
  StorageIndex* frontRows(Index order) { return m_factorRows.top(static_cast<std::size_t>(order)); }

  // The 2x2 block [d1 conj(d21); d21 d2] = scale [a conj(b); b c], where scale is its largest magnitude. det = a c -
  // |b|^2 is in [-2, 1].
  struct ScaledBlock {
    RealScalar scale;
    RealScalar a;
    RealScalar c;
    RealScalar absB;
    RealScalar det;

    ScaledBlock(const RealScalar& d1, const RealScalar& d2, const RealScalar& absD21)
        : scale(numext::maxi(numext::maxi(RealScalar(numext::abs(d1)), RealScalar(numext::abs(d2))), absD21)),
          a(d1 / scale),
          c(d2 / scale),
          absB(absD21 / scale),
          det(a * c - absB * absB) {}
  };

  // Calls function(row, col, position) on the structural entries of the UpLo_ triangle in storage order, where position
  // counts all of the matrix's entries before this one in storage order. That's the entry's index in the values of the
  // matrix once it's compressed.
  template <typename Function>
  static void visitTriangle(const MatrixType& matrix, Function&& function) {
    Index position = 0;
    const Index outerSize = matrix.outerSize();
    for (Index outer = 0; outer < outerSize; ++outer) {
      for (typename MatrixType::InnerIterator it(matrix, outer); it; ++it, ++position) {
        const Index row = it.row();
        const Index col = it.col();
        if (UpLo_ == Lower ? row >= col : row <= col) function(row, col, position);
      }
    }
  }

  // Largest fraction of explicit zeros allowed in a merged front's L, including its diagonal. An explicit zero costs
  // more in a larger front, while a smaller one gains more from losing an extend-add and from gaining fully summed
  // columns.
  static double maxExplicitZeroFraction(Index pivots) { return pivots < 4 ? 0.3 : (pivots < 32 ? 0.1 : 0.02); }

  // Merges fronts into their parents if the result has few enough explicit zeros, then renumbers the columns so that
  // each merged front's are contiguous. frontParent and frontBelow hold each front's parent, or -1 for a root, and its
  // number of rows below its pivots. Updates columnFront, m_frontColumns, m_perm and m_invPerm, and sets newColumn to
  // each column's new index.
  void mergeRelaxedFronts(const std::vector<StorageIndex>& frontParent, const std::vector<StorageIndex>& frontBelow,
                          std::vector<StorageIndex>& columnFront, std::vector<StorageIndex>& newColumn);

  // Records a factored front and returns where to write its order x numPivots columns of L.
  Scalar* appendFactor(Index s, Index numPivots, Index order);

  // Returns the largest off-diagonal magnitude of column j in the front's active part, which starts at row active,
  // leaving out row skip (or -1 for none).
  //
  // This, largestFullySummed() and largestMagnitude() skip NaN entries, so a pivot can pass its test against a column
  // holding a NaN. The update carries the NaN to the diagonal of a later pivot, and factorize() reports a non-finite
  // pivot as NumericalIssue.
  RealScalar frontColumnMax(Index order, Index active, Index j, Index skip) const;

  // Returns whether column j is exactly zero off its diagonal in the front's active part, which starts at row active. A
  // NaN entry counts as nonzero.
  bool frontColumnIsZero(Index order, Index active, Index j) const;

  // Returns the row r in [active, fullySummed) other than j with the largest magnitude in column j, or -1 if they're
  // all zero, and sets rest to frontColumnMax(order, active, j, r).
  Index largestFullySummed(Index order, Index active, Index fullySummed, Index j, RealScalar& rest) const;

  // These add a child's packed lower triangle to the order x order front, where map gives each of its childOrder rows'
  // row in the front. extendAdd() adds the rows that stay adjacent in the front as vectorized segments, if there are
  // enough of them. A call is too expensive for the short children of a tree of small fronts, so they scatter inline.
  EIGEN_ALWAYS_INLINE void scatterAdd(Index order, const Scalar* block, const StorageIndex* map, Index childOrder) {
    FrontMap front(m_front.data(), order, order);
    for (Index j = 0; j < childOrder; ++j) {
      Scalar* column = &front.coeffRef(0, map[j]);
      for (Index i = j; i < childOrder; ++i) column[map[i]] += *block++;
    }
  }
  void extendAdd(Index order, const Scalar* block, const StorageIndex* map, Index childOrder);

  // Eliminates pivots from the fully summed columns of m_front and returns how many. Pivots end up at the front's start
  // and delayed columns between them and the rows that aren't fully summed, with the front's rows permuted to match.
  Index factorFront(Index order, Index fullySummed, bool isRoot);

  // Factors a front with one fully summed column in a single pass, without updating the front in place and then copying
  // its results, or delays the column if it fails the 1x1 test. Returns false, leaving the front unchanged, if the
  // pivot is zero.
  EIGEN_ALWAYS_INLINE bool factorOnePivotFront(Index s, Index order, bool isRoot);

  // Records the factored front's fullySummed - numPivots delayed columns and pushes the packed lower triangle of its
  // trailing order - numPivots rows and columns onto m_stack, for its parent's extend-add to pop. The parent reads the
  // delayed rows from m_factorRows, after the front's pivots.
  void pushToParent(Index s, Index order, Index fullySummed, Index numPivots);

  // Swaps two rows and columns of the front.
  void swapFront(Index order, Index a, Index b);

  // These eliminate the pivot at column k from the columns before panelEnd. updateTrailing() applies it to the others.
  void eliminate1x1(Index order, Index k, Index panelEnd);
  void eliminate2x2(Index order, Index k, Index panelEnd);

  // Applies the last numPivots pivots to the columns from panelEnd on.
  void updateTrailing(Index order, Index firstPivot, Index numPivots, Index panelEnd);

  // The pivot tests bound |L| by 1/u, so a smaller u trades backward error for fill.
  RealScalar m_threshold = RealScalar(0.01);

  ComputationInfo m_info = InvalidInput;

  // Symbolic analysis

  // Each input row's row in elimination order, and its inverse.
  std::vector<StorageIndex> m_perm;
  std::vector<StorageIndex> m_invPerm;
  // First column of each front, with one extra element at the end.
  std::vector<StorageIndex> m_frontColumns;
  // Each front's parent, or -1 for a root.
  std::vector<StorageIndex> m_frontParent;
  // The number of rows below each front's pivots, as m_belowStart[s + 1] - m_belowStart[s].
  std::vector<std::size_t> m_belowStart;
  // Each front's columns followed by the rows below its pivots, starting at m_belowStart[s] + m_frontColumns[s].
  std::vector<StorageIndex> m_frontRowList;
  // Position in the parent front of each row below a front's pivots, from m_belowStart[s], not counting the parent's
  // delayed columns.
  std::vector<StorageIndex> m_belowToParent;
  // Each front's children, in increasing order.
  std::vector<std::size_t> m_childStart;
  std::vector<StorageIndex> m_children;
  // Where each input matrix entry is assembled.
  std::vector<std::size_t> m_assemblyStart;
  std::vector<Assembly> m_assembly;
  // Number of entries the input matrix stores, in both triangles.
  std::size_t m_numValues = 0;
  // What a factorization without delayed pivots stores in m_L, and its largest m_stack and front order.
  std::size_t m_symbolicFactorSize = 0;
  std::size_t m_symbolicStackSize = 0;
  Index m_symbolicFrontOrder = 0;

  // Numeric factorization

  // The input matrix's values in compressed order, if it isn't compressed.
  std::vector<Scalar> m_values;
  // The front being factored.
  std::vector<Scalar, aligned_allocator<Scalar>> m_front;
  // Rows of L in a 2x2 pivot.
  std::vector<Scalar, aligned_allocator<Scalar>> m_pivotScratch;
  // Maps a child's rows into its parent.
  std::vector<StorageIndex> m_childMap;
  // For each row of a child, the end of its run of rows that stay adjacent in the parent.
  std::vector<StorageIndex> m_childRunEnd;
  // Stack of contribution blocks. Each is a packed lower triangle.
  WorkStack<Scalar> m_stack;
  // Number of columns each front delayed to its parent.
  std::vector<StorageIndex> m_numDelayed;
  // Each front's pivot columns.
  WorkStack<Scalar> m_L;
  // Each front's rows after pivoting, in elimination order. The current front's are the last ones, permuted in place,
  // so a tree of small fronts doesn't copy them from a scratch array.
  WorkStack<StorageIndex> m_factorRows;
  std::vector<FactoredFront> m_factoredFronts;
  // D's 2x2 blocks, in pivot order.
  std::vector<ScaledBlock> m_blocks;
  RealVectorType m_D;
  VectorType m_subdiagonal;
  PermutationType m_P;
  PermutationType m_Pinv;
  // Each pivot's row in elimination order.
  std::vector<StorageIndex> m_pivots;
  Index m_LNonZeros = 0;
  Index m_numPivots = 0;
  Index m_numPositive = 0;
  Index m_numNegative = 0;
  Index m_numZero = 0;
  Index m_maxFrontOrder = 0;
  bool m_hasZeroPivot = false;
};

template <typename MatrixType_, int UpLo_, typename Ordering_>
constexpr int MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::UpLo;
template <typename MatrixType_, int UpLo_, typename Ordering_>
constexpr int MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::ColsAtCompileTime;
template <typename MatrixType_, int UpLo_, typename Ordering_>
constexpr int MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::MaxColsAtCompileTime;

template <typename MatrixType_, int UpLo_, typename Ordering_>
void MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::mergeRelaxedFronts(const std::vector<StorageIndex>& frontParent,
                                                                         const std::vector<StorageIndex>& frontBelow,
                                                                         std::vector<StorageIndex>& columnFront,
                                                                         std::vector<StorageIndex>& newColumn) {
  const auto toStorageIndex = [](auto value) { return internal::convert_index<StorageIndex>(value); };
  constexpr StorageIndex none(-1);
  const Index size = static_cast<Index>(columnFront.size());
  const Index numFronts = static_cast<Index>(m_frontColumns.size()) - 1;
  newColumn.resize(size);
  std::iota(newColumn.begin(), newColumn.end(), StorageIndex(0));

  const auto storedEntries = [](Index pivots, Index below) {
    return factorSize(pivots, pivots + below) + static_cast<std::size_t>(pivots);
  };

  // below(child) is a subset of columns(parent) U below(parent) (Liu, 1990), so a merged front keeps its parent's rows
  // below. Each front's descendants are final when it's visited, because the fronts are postordered.
  std::vector<StorageIndex> mergedInto(numFronts, none);
  std::vector<StorageIndex> numPivots(numFronts);
  // Entries of L in a front, including its diagonal and leaving out explicit zeros.
  std::vector<std::size_t> numEntries(numFronts);
  for (Index s = 0; s < numFronts; ++s) {
    numPivots[s] = m_frontColumns[s + 1] - m_frontColumns[s];
    numEntries[s] = storedEntries(numPivots[s], frontBelow[s]);
  }
  Index numMerged = 0;
  for (Index s = 0; s < numFronts; ++s) {
    const StorageIndex p = frontParent[s];
    if (p == none) continue;
    const Index pivots = numPivots[s] + numPivots[p];
    const std::size_t stored = storedEntries(pivots, frontBelow[p]);
    const std::size_t entries = numEntries[s] + numEntries[p];
    if (double(stored - entries) <= maxExplicitZeroFraction(pivots) * double(stored)) {
      mergedInto[s] = p;
      numPivots[p] = toStorageIndex(pivots);
      numEntries[p] = entries;
      ++numMerged;
    }
  }
  if (numMerged == 0) return;

  // Each original front's merged front, in the order of the fronts that weren't merged into a parent.
  std::vector<StorageIndex> merged(numFronts);
  const StorageIndex numMergedFronts = toStorageIndex(numFronts - numMerged);
  {
    StorageIndex next = numMergedFronts;
    for (Index s = numFronts - 1; s >= 0; --s) {
      merged[s] = mergedInto[s] == none ? --next : merged[mergedInto[s]];
    }
  }

  // A merged child generally isn't its parent's last child, so the merged tree needs a new postorder to make each
  // front's columns contiguous and keep contribution blocks on a stack. Any postorder of a tree has the same fill.
  // Visiting children in increasing order keeps the fronts that weren't merged in their relative order.
  IndexVectorType rootedParent(numMergedFronts);
  for (Index s = 0; s < numFronts; ++s) {
    if (mergedInto[s] == none) {
      rootedParent[merged[s]] = frontParent[s] == none ? numMergedFronts : merged[frontParent[s]];
    }
  }
  IndexVectorType postorder;
  internal::treePostorder(numMergedFronts, rootedParent, postorder);

  std::vector<StorageIndex> firstColumn(numMergedFronts + 1, 0);
  for (Index s = 0; s < numFronts; ++s) {
    if (mergedInto[s] == none) firstColumn[postorder[merged[s]] + 1] = numPivots[s];
  }
  for (Index s = 0; s < numMergedFronts; ++s) firstColumn[s + 1] += firstColumn[s];

  // A front's columns keep their relative order, so every entry stays in the same triangle.
  std::vector<StorageIndex> newColumnFront(size);
  {
    std::vector<StorageIndex> next(firstColumn.begin(), firstColumn.end() - 1);
    for (Index s = 0; s < numFronts; ++s) {
      const StorageIndex front = postorder[merged[s]];
      const Index end = m_frontColumns[s + 1];
      for (Index j = m_frontColumns[s]; j < end; ++j) {
        newColumn[j] = next[front]++;
        newColumnFront[newColumn[j]] = front;
      }
    }
  }

  columnFront.swap(newColumnFront);
  m_frontColumns.swap(firstColumn);
  for (Index i = 0; i < size; ++i) {
    m_perm[i] = newColumn[m_perm[i]];
    m_invPerm[m_perm[i]] = toStorageIndex(i);
  }
}

template <typename MatrixType_, int UpLo_, typename Ordering_>
void MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::analyzePattern(const MatrixType& matrix) {
  eigen_assert(matrix.rows() == matrix.cols() && "The matrix must be square.");

  const Index size = matrix.rows();

  // Fill-reducing ordering. An empty permutation means the identity.
  PermutationMatrix<Dynamic, Dynamic, StorageIndex> Pinv;
  EIGEN_IF_CONSTEXPR ((!std::is_same<OrderingType, NaturalOrdering<StorageIndex>>::value)) {
    SparseMatrix<Scalar, ColMajor, StorageIndex> C;
    constexpr bool kUseAMDFastPath = std::is_same<OrderingType, AMDOrdering<StorageIndex>>::value;
    internal::simplicial_cholesky_amd_dispatch<kUseAMDFastPath>::template run<UpLo_, false, OrderingType>(matrix, C,
                                                                                                          Pinv);
  }
  const auto toStorageIndex = [](auto value) { return internal::convert_index<StorageIndex>(value); };
  std::vector<StorageIndex> ordered(size);
  for (Index i = 0; i < size; ++i) ordered[Pinv.size() > 0 ? Index(Pinv.indices()(i)) : i] = toStorageIndex(i);

  // The upper triangle of the ordered matrix.
  FactorType upper;
  internal::permute_symm_to_symm<UpLo_, Upper, false>(matrix, upper, Pinv.size() > 0 ? ordered.data() : nullptr);

  // Each column's parent in the elimination tree, or -1 for a root.
  constexpr StorageIndex none(-1);
  std::vector<StorageIndex> parent(size);
  {
    std::vector<StorageIndex> ancestors(size);
    internal::simpl_chol_helper<Scalar, StorageIndex>::calc_etree(toStorageIndex(size), upper, parent.data(),
                                                                  ancestors.data());
  }

  // A postorder doesn't change the fill and lets factorize() keep contribution blocks on a stack.
  IndexVectorType rootedParent(size);
  for (Index j = 0; j < size; ++j) rootedParent[j] = parent[j] == none ? toStorageIndex(size) : parent[j];

  // Visiting children from largest to smallest column index places each leaf of an AMD-ordered KKT chain's tree right
  // before its parent, not O(N) rows away, so a front's rows stay close together in the solution vectors once those
  // vectors outgrow the cache.
  IndexVectorType postorder;
  internal::treePostorder(toStorageIndex(size), rootedParent, postorder, internal::PostorderChildren::Decreasing);

  m_perm.resize(size);
  m_invPerm.resize(size);
  for (Index i = 0; i < size; ++i) {
    m_perm[i] = postorder[ordered[i]];
    m_invPerm[m_perm[i]] = toStorageIndex(i);
  }

  // Each column's number of rows below its diagonal in L, from the rows of the upper triangle (Gilbert, Ng and Peyton,
  // 1994).
  using SymbolicHelper = internal::simpl_chol_helper<Scalar, StorageIndex>;
  const StorageIndex n = toStorageIndex(size);
  std::vector<StorageIndex> rowStart(size + 1, 0);
  SymbolicHelper::calc_hadj_outer(n, upper, rowStart.data());
  std::vector<StorageIndex> rowColumns(rowStart[size]);
  std::vector<StorageIndex> count(size);
  {
    std::vector<StorageIndex> work(size);
    IndexVectorType previousLeaf = IndexVectorType::Constant(size, none);
    SymbolicHelper::calc_hadj_inner(n, upper, rowStart.data(), rowColumns.data(), work.data());
    IndexVectorType inPostorder(size);
    for (Index j = 0; j < size; ++j) inPostorder[postorder[j]] = toStorageIndex(j);
    SymbolicHelper::calc_colcount(n, rowStart.data(), rowColumns.data(), parent.data(), previousLeaf.data(),
                                  work.data(), inPostorder.data(), count.data(), /*doLDLT=*/true);
  }

  // Fundamental supernodes: column j joins column j - 1's front if it's j - 1's parent and only child, and
  // count(j - 1) = count(j) + 1. pattern(L_{j-1}) \ {j} is a subset of pattern(L_j), so the counts make them equal.
  // mergeRelaxedFronts() then relaxes these fronts.
  std::vector<StorageIndex> columnParent(size, none);
  std::vector<StorageIndex> columnBelow(size);
  std::vector<StorageIndex> numChildren(size, 0);
  for (Index j = 0; j < size; ++j) {
    columnBelow[postorder[j]] = count[j];
    if (parent[j] != none) {
      columnParent[postorder[j]] = postorder[parent[j]];
      ++numChildren[postorder[parent[j]]];
    }
  }
  m_frontColumns.clear();
  std::vector<StorageIndex> columnFront(size);
  for (Index j = 0; j < size; ++j) {
    const bool extends =
        j > 0 && columnParent[j - 1] == j && numChildren[j] == 1 && columnBelow[j - 1] == columnBelow[j] + 1;
    if (!extends) m_frontColumns.push_back(toStorageIndex(j));
    columnFront[j] = toStorageIndex(m_frontColumns.size() - 1);
  }
  m_frontColumns.push_back(n);
  {
    const Index numFundamental = static_cast<Index>(m_frontColumns.size()) - 1;
    std::vector<StorageIndex> frontParent(numFundamental);
    std::vector<StorageIndex> frontBelow(numFundamental);
    for (Index s = 0; s < numFundamental; ++s) {
      const StorageIndex last = m_frontColumns[s + 1] - 1;
      frontParent[s] = columnParent[last] == none ? none : columnFront[columnParent[last]];
      frontBelow[s] = columnBelow[last];
    }
    std::vector<StorageIndex> newColumn;
    mergeRelaxedFronts(frontParent, frontBelow, columnFront, newColumn);
    for (Index j = 0; j < size; ++j) postorder[j] = newColumn[postorder[j]];
  }
  const Index numFronts = static_cast<Index>(m_frontColumns.size()) - 1;
  // Each column of the ordered matrix in m_perm's numbering, and the reverse.
  const IndexVectorType& columnOf = postorder;
  std::vector<StorageIndex> orderedColumn(size);
  for (Index j = 0; j < size; ++j) orderedColumn[columnOf[j]] = toStorageIndex(j);

  // Each front's parent and children. A front's parent holds the parent of its last column.
  m_frontParent.assign(numFronts, none);
  m_childStart.assign(numFronts + 1, 0);
  for (Index s = 0; s < numFronts; ++s) {
    const StorageIndex last = orderedColumn[m_frontColumns[s + 1] - 1];
    if (parent[last] != none) {
      m_frontParent[s] = columnFront[columnOf[parent[last]]];
      ++m_childStart[m_frontParent[s] + 1];
    }
  }
  for (Index s = 0; s < numFronts; ++s) m_childStart[s + 1] += m_childStart[s];
  m_children.resize(m_childStart[numFronts]);
  {
    std::vector<std::size_t> next(m_childStart.begin(), m_childStart.end() - 1);
    for (Index s = 0; s < numFronts; ++s) {
      if (m_frontParent[s] != none) m_children[next[m_frontParent[s]]++] = toStorageIndex(s);
    }
  }

  // Each front's columns, then its rows below them: pattern(L_last) for its last column, which is the rows past its
  // columns in its columns of A and in its children's rows below.
  m_belowStart.assign(numFronts + 1, 0);
  m_frontRowList.clear();
  {
    std::size_t numRows = static_cast<std::size_t>(size);
    for (Index s = 0; s < numFronts; ++s) {
      numRows += static_cast<std::size_t>(count[orderedColumn[m_frontColumns[s + 1] - 1]]);
    }
    m_frontRowList.reserve(numRows);
  }
  {
    std::vector<StorageIndex> mark(size, none);
    for (Index s = 0; s < numFronts; ++s) {
      const StorageIndex front = toStorageIndex(s);
      const StorageIndex end = m_frontColumns[s + 1];
      for (StorageIndex j = m_frontColumns[s]; j < end; ++j) m_frontRowList.push_back(j);
      const std::size_t first = m_frontRowList.size();
      const auto add = [&](StorageIndex row) {
        if (row >= end && mark[row] != front) {
          mark[row] = front;
          m_frontRowList.push_back(row);
        }
      };
      for (StorageIndex j = m_frontColumns[s]; j < end; ++j) {
        const StorageIndex column = orderedColumn[j];
        const StorageIndex rowEnd = rowStart[column + 1];
        for (StorageIndex k = rowStart[column]; k < rowEnd; ++k) add(columnOf[rowColumns[k]]);
      }
      const std::size_t childEnd = m_childStart[s + 1];
      for (std::size_t c = m_childStart[s]; c < childEnd; ++c) {
        const StorageIndex child = m_children[c];
        const std::size_t childFirst = m_belowStart[child] + static_cast<std::size_t>(m_frontColumns[child + 1]);
        const std::size_t childLast = m_belowStart[child + 1] + static_cast<std::size_t>(m_frontColumns[child + 1]);
        for (std::size_t k = childFirst; k < childLast; ++k) add(m_frontRowList[k]);
      }
      m_belowStart[s + 1] = m_belowStart[s] + (m_frontRowList.size() - first);
      eigen_internal_assert(m_belowStart[s + 1] - m_belowStart[s] ==
                            static_cast<std::size_t>(count[orderedColumn[end - 1]]));
    }
  }
  // Sorts each front's rows below with a counting transpose: list each row's fronts, then write the rows back to their
  // fronts in increasing order.
  {
    const auto belowRows = [&](Index s) {
      return m_frontRowList.begin() + static_cast<std::ptrdiff_t>(m_belowStart[s] + m_frontColumns[s + 1]);
    };
    std::vector<std::size_t> rowFrontStart(size + 1, 0);
    for (Index s = 0; s < numFronts; ++s) {
      const auto end = belowRows(s) + static_cast<std::ptrdiff_t>(m_belowStart[s + 1] - m_belowStart[s]);
      for (auto row = belowRows(s); row != end; ++row) {
        ++rowFrontStart[*row + 1];
      }
    }
    for (Index i = 0; i < size; ++i) rowFrontStart[i + 1] += rowFrontStart[i];
    std::vector<StorageIndex> rowFronts(rowFrontStart[size]);
    {
      std::vector<std::size_t> next(rowFrontStart.begin(), rowFrontStart.end() - 1);
      for (Index s = 0; s < numFronts; ++s) {
        const auto end = belowRows(s) + static_cast<std::ptrdiff_t>(m_belowStart[s + 1] - m_belowStart[s]);
        for (auto row = belowRows(s); row != end; ++row) {
          rowFronts[next[*row]++] = toStorageIndex(s);
        }
      }
    }
    std::vector<std::size_t> next(numFronts);
    for (Index s = 0; s < numFronts; ++s) next[s] = m_belowStart[s] + static_cast<std::size_t>(m_frontColumns[s + 1]);
    for (Index i = 0; i < size; ++i) {
      const std::size_t end = rowFrontStart[i + 1];
      for (std::size_t k = rowFrontStart[i]; k < end; ++k) {
        m_frontRowList[next[rowFronts[k]]++] = toStorageIndex(i);
      }
    }
  }

  // Where each entry of the input matrix is assembled. The row is global until the loop over fronts below.
  m_assemblyStart.assign(numFronts + 1, 0);
  m_numValues = static_cast<std::size_t>(matrix.nonZeros());
  visitTriangle(matrix, [&](Index row, Index col, Index) {
    ++m_assemblyStart[columnFront[numext::mini(m_perm[row], m_perm[col])] + 1];
  });
  for (Index s = 0; s < numFronts; ++s) m_assemblyStart[s + 1] += m_assemblyStart[s];
  m_assembly.resize(m_assemblyStart[numFronts]);
  {
    std::vector<std::size_t> next(m_assemblyStart.begin(), m_assemblyStart.end() - 1);
    visitTriangle(matrix, [&](Index row, Index col, Index value) {
      const StorageIndex i = m_perm[row];
      const StorageIndex j = m_perm[col];
      const StorageIndex s = columnFront[numext::mini(i, j)];
      m_assembly[next[s]++] = Assembly{toStorageIndex(value), numext::maxi(i, j),
                                       toStorageIndex(numext::mini(i, j) - m_frontColumns[s]), i < j};
    });
  }

  // The position of each row in front s, not counting delayed columns: row - first column for its columns, then the
  // rows below them in order. Each front's children and entries of A take their positions from it.
  m_belowToParent.resize(m_belowStart[numFronts]);
  {
    std::vector<StorageIndex> position(size);
    for (Index s = 0; s < numFronts; ++s) {
      const StorageIndex first = m_frontColumns[s];
      const StorageIndex end = m_frontColumns[s + 1];
      for (StorageIndex j = first; j < end; ++j) position[j] = j - first;
      const std::size_t rows = m_belowStart[s] + static_cast<std::size_t>(end);
      for (std::size_t k = 0; k < m_belowStart[s + 1] - m_belowStart[s]; ++k) {
        position[m_frontRowList[rows + k]] = toStorageIndex(Index(end - first) + Index(k));
      }
      for (std::size_t c = m_childStart[s]; c < m_childStart[s + 1]; ++c) {
        const StorageIndex child = m_children[c];
        for (std::size_t k = m_belowStart[child]; k < m_belowStart[child + 1]; ++k) {
          m_belowToParent[k] = position[m_frontRowList[k + static_cast<std::size_t>(m_frontColumns[child + 1])]];
        }
      }
      for (std::size_t k = m_assemblyStart[s]; k < m_assemblyStart[s + 1]; ++k) {
        m_assembly[k].row = position[m_assembly[k].row];
      }
    }
  }

  m_symbolicFactorSize = 0;
  m_symbolicStackSize = 0;
  m_symbolicFrontOrder = 0;
  {
    std::size_t stack = 0;
    for (Index s = 0; s < numFronts; ++s) {
      const Index numColumns = m_frontColumns[s + 1] - m_frontColumns[s];
      const Index numBelow = static_cast<Index>(m_belowStart[s + 1] - m_belowStart[s]);
      m_symbolicFactorSize += rectangleSize(numColumns + numBelow, numColumns);
      m_symbolicFrontOrder = numext::maxi(m_symbolicFrontOrder, numColumns + numBelow);
      for (std::size_t c = m_childStart[s]; c < m_childStart[s + 1]; ++c) {
        const StorageIndex child = m_children[c];
        stack -= packedSize(static_cast<Index>(m_belowStart[child + 1] - m_belowStart[child]));
      }
      stack += packedSize(numBelow);
      m_symbolicStackSize = numext::maxi(m_symbolicStackSize, stack);
    }
  }

  m_numDelayed.resize(numFronts);
  m_factoredFronts.resize(numFronts);
  m_pivots.resize(size);
  m_D.resize(size);
  m_subdiagonal.resize(size);
  m_P.resize(size);
  m_Pinv.resize(size);

  m_info = InvalidInput;
  m_isInitialized = true;
}

template <typename MatrixType_, int UpLo_, typename Ordering_>
void MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::factorize(const MatrixType& matrix) {
  eigen_assert(m_isInitialized && "analyzePattern() must be called first.");
  eigen_assert(matrix.rows() == rows() && matrix.cols() == rows());

  eigen_assert(static_cast<std::size_t>(matrix.nonZeros()) == m_numValues &&
               "The matrix must have the analyzed sparsity pattern.");
  // A compressed matrix is read in place.
  const Scalar* values = matrix.valuePtr();
  if (!matrix.isCompressed()) {
    m_values.resize(m_numValues);
    std::size_t value = 0;
    for (Index outer = 0; outer < matrix.outerSize(); ++outer) {
      for (typename MatrixType::InnerIterator it(matrix, outer); it; ++it) m_values[value++] = it.value();
    }
    values = m_values.data();
  }

  // Delayed pivots can still outgrow these.
  m_L.reserve(m_symbolicFactorSize);
  m_stack.reserve(m_symbolicStackSize);
  m_factorRows.reserve(m_frontRowList.size());
  if (m_front.size() < rectangleSize(m_symbolicFrontOrder, m_symbolicFrontOrder)) {
    m_front.resize(rectangleSize(m_symbolicFrontOrder, m_symbolicFrontOrder));
  }

  m_L.clear();
  m_LNonZeros = 0;
  m_factorRows.clear();
  m_stack.clear();
  m_numPivots = 0;
  m_numPositive = 0;
  m_numNegative = 0;
  m_numZero = 0;
  m_maxFrontOrder = 0;
  m_hasZeroPivot = false;

  const Index numFronts = static_cast<Index>(m_frontColumns.size()) - 1;
  // None of these are reallocated during the factorization, but the calls below stop the compiler from proving it.
  const StorageIndex* const frontColumns = m_frontColumns.data();
  const std::size_t* const belowStart = m_belowStart.data();
  const std::size_t* const childStart = m_childStart.data();
  const StorageIndex* const children = m_children.data();
  const StorageIndex* const numDelayed = m_numDelayed.data();
  const std::size_t* const assemblyStart = m_assemblyStart.data();
  const Assembly* const assembly = m_assembly.data();
  const StorageIndex* const frontRowList = m_frontRowList.data();
  const StorageIndex* const belowToParent = m_belowToParent.data();
  const StorageIndex* const frontParent = m_frontParent.data();
  for (Index s = 0; s < numFronts; ++s) {
    const Index firstColumn = frontColumns[s];
    const Index numColumns = frontColumns[s + 1] - firstColumn;
    const Index numBelow = static_cast<Index>(belowStart[s + 1] - belowStart[s]);

    Index delayedIn = 0;
    for (std::size_t c = childStart[s]; c < childStart[s + 1]; ++c) delayedIn += numDelayed[children[c]];

    const Index fullySummed = delayedIn + numColumns;
    const Index order = fullySummed + numBelow;
    m_maxFrontOrder = numext::maxi(m_maxFrontOrder, order);

    const std::size_t frontSize = rectangleSize(order, order);
    if (m_front.size() < frontSize) m_front.resize(frontSize);
    StorageIndex* rows = m_factorRows.push(static_cast<std::size_t>(order));
    FrontMap front(m_front.data(), order, order);
    // Only the lower triangle is read, but a single fill is cheaper than one per column for a small front. Filling the
    // whole of a larger one would also push the working set out of the L1 cache.
    if (order < kBlockedFrontMinOrder) {
      front.setZero();
    } else {
      for (Index j = 0; j < order; ++j) front.col(j).tail(order - j).setZero();
    }

    for (std::size_t k = assemblyStart[s]; k < assemblyStart[s + 1]; ++k) {
      const Assembly& entry = assembly[k];
      const Scalar& value = values[entry.value];
      front(entry.row + delayedIn, entry.col + delayedIn) += entry.conjugate ? Scalar(numext::conj(value)) : value;
    }

    // The delayed rows come first, and the children fill them in below.
    const StorageIndex* rowList = frontRowList + belowStart[s] + firstColumn;
    internal::smart_copy(rowList, rowList + numColumns + numBelow, rows + delayedIn);

    // Extend-add the children's contribution blocks. They're on a stack, so the last child's is on top.
    Index delayedSlot = delayedIn;
    const std::size_t childBegin = childStart[s];
    for (std::size_t c = childStart[s + 1]; c > childBegin; --c) {
      const Index child = children[c - 1];
      const Index delayed = numDelayed[child];
      const Index childBelow = static_cast<Index>(belowStart[child + 1] - belowStart[child]);
      const Index childOrder = delayed + childBelow;
      delayedSlot -= delayed;

      // Without delayed columns on either side, the symbolic map is the map.
      const StorageIndex* childMap = belowToParent + belowStart[child];
      if (delayedIn > 0) {
        if (m_childMap.size() < static_cast<std::size_t>(childOrder)) m_childMap.resize(childOrder);
        const FactoredFront& factored = m_factoredFronts[child];
        const StorageIndex* delayedRows = m_factorRows.storage.data() + factored.rows + factored.numPivots;
        for (Index i = 0; i < delayed; ++i) {
          m_childMap[i] = internal::convert_index<StorageIndex>(delayedSlot + i);
          rows[delayedSlot + i] = delayedRows[i];
        }
        for (Index i = 0; i < childBelow; ++i) {
          m_childMap[delayed + i] = internal::convert_index<StorageIndex>(delayedIn + childMap[i]);
        }
        childMap = m_childMap.data();
      }

      // Delayed rows come before the rows below in both the child and the parent, and the rows below keep their
      // relative order. So the map is increasing and the lower triangle stays the lower triangle.
      const std::size_t blockSize = packedSize(childOrder);
      const Scalar* block = m_stack.top(blockSize);
      if (childOrder < kVectorizedMinLength) {
        scatterAdd(order, block, childMap, childOrder);
      } else {
        extendAdd(order, block, childMap, childOrder);
      }
      m_stack.pop(blockSize);
    }

    const bool isRoot = frontParent[s] == -1;
    if (fullySummed == 1 && factorOnePivotFront(s, order, isRoot)) continue;

    const Index numPivots = factorFront(order, fullySummed, isRoot);

    Scalar* factor = appendFactor(s, numPivots, order);
    internal::smart_copy(m_front.data(), m_front.data() + rectangleSize(order, numPivots), factor);

    pushToParent(s, order, fullySummed, numPivots);
  }

  // Roots can't delay, so every pivot, and with it every entry of m_D and m_subdiagonal, has been written.
  const Index size = rows();
  eigen_internal_assert(m_numPivots == size);

  for (Index k = 0; k < size; ++k) {
    const StorageIndex row = m_invPerm[m_pivots[k]];
    m_Pinv.indices()[k] = row;
    m_P.indices()[row] = internal::convert_index<StorageIndex>(k);
  }

  // Inertia from the signs of D (Sylvester's law). A 2x2 block with det < 0 has one eigenvalue of each sign; otherwise
  // both eigenvalues have the sign of its trace.
  bool finite = true;
  m_blocks.clear();
  for (Index k = 0; k < size; ++k) {
    const RealScalar d1 = m_D[k];
    const RealScalar sub = numext::abs(m_subdiagonal[k]);
    if (sub != RealScalar(0)) {
      const RealScalar d2 = m_D[k + 1];
      m_blocks.emplace_back(d1, d2, sub);
      if (!(numext::isfinite)(d1) || !(numext::isfinite)(sub) || !(numext::isfinite)(d2)) {
        finite = false;
      } else if (m_blocks.back().det < RealScalar(0)) {
        ++m_numPositive;
        ++m_numNegative;
      } else if (d1 + d2 > RealScalar(0)) {
        m_numPositive += 2;
      } else {
        m_numNegative += 2;
      }
      ++k;
    } else if (!(numext::isfinite)(d1)) {
      finite = false;
    } else if (d1 > RealScalar(0)) {
      ++m_numPositive;
    } else if (d1 < RealScalar(0)) {
      ++m_numNegative;
    } else {
      ++m_numZero;
    }
  }

  m_info = finite && !m_hasZeroPivot ? Success : NumericalIssue;
}

template <typename MatrixType_, int UpLo_, typename Ordering_>
typename MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::RealScalar
MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::logAbsDeterminant() const {
  eigen_assert(m_isInitialized && "Decomposition is not initialized.");
  eigen_assert(m_info != InvalidInput && "factorize() must be called before logAbsDeterminant().");

  RealScalar result(0);
  for (Index k = 0; k < rows(); ++k) {
    const RealScalar sub = numext::abs(m_subdiagonal[k]);
    if (sub != RealScalar(0)) {
      const ScaledBlock block(m_D[k], m_D[k + 1], sub);
      result += RealScalar(2) * numext::log(block.scale) + numext::log(numext::abs(block.det));
      ++k;
    } else {
      result += numext::log(numext::abs(m_D[k]));
    }
  }
  return result;
}

template <typename MatrixType_, int UpLo_, typename Ordering_>
typename MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::Scalar*
MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::appendFactor(Index s, Index numPivots, Index order) {
  m_factoredFronts[s] =
      FactoredFront{internal::convert_index<StorageIndex>(numPivots), internal::convert_index<StorageIndex>(order),
                    m_factorRows.size - order, m_L.size};
  m_LNonZeros += static_cast<Index>(factorSize(numPivots, order));
  return m_L.push(rectangleSize(order, numPivots));
}

template <typename MatrixType_, int UpLo_, typename Ordering_>
void MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::extendAdd(Index order, const Scalar* block,
                                                                const StorageIndex* map, Index childOrder) {
  if (m_childRunEnd.size() < static_cast<std::size_t>(childOrder)) m_childRunEnd.resize(childOrder);
  Index numRuns = 1;
  Index runEnd = childOrder;
  m_childRunEnd[childOrder - 1] = internal::convert_index<StorageIndex>(runEnd);
  for (Index i = childOrder - 2; i >= 0; --i) {
    if (map[i + 1] != map[i] + 1) {
      runEnd = i + 1;
      ++numRuns;
    }
    m_childRunEnd[i] = internal::convert_index<StorageIndex>(runEnd);
  }

  if (childOrder < kExtendAddMinRunLength * numRuns) {
    scatterAdd(order, block, map, childOrder);
    return;
  }

  FrontMap front(m_front.data(), order, order);
  for (Index j = 0; j < childOrder; ++j) {
    Scalar* column = &front.coeffRef(0, map[j]);
    for (Index i = j; i < childOrder;) {
      const Index end = m_childRunEnd[i];
      addTo(column + map[i], block, end - i);
      block += end - i;
      i = end;
    }
  }
}

template <typename MatrixType_, int UpLo_, typename Ordering_>
typename MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::RealScalar
MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::frontColumnMax(Index order, Index active, Index j, Index skip) const {
  ConstFrontMap front(m_front.data(), order, order);
  RealScalar result(0);
  for (Index i = active; i < j; ++i) {
    if (i != skip) result = numext::maxi(result, RealScalar(numext::abs(front(j, i))));
  }
  const Scalar* column = &front.coeffRef(0, j);
  const Index split = skip > j ? skip : order;
  result = numext::maxi(result, largestMagnitude(column + j + 1, split - j - 1));
  if (order > split + 1) result = numext::maxi(result, largestMagnitude(column + split + 1, order - split - 1));
  return result;
}

template <typename MatrixType_, int UpLo_, typename Ordering_>
bool MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::frontColumnIsZero(Index order, Index active, Index j) const {
  ConstFrontMap front(m_front.data(), order, order);
  return (front.row(j).segment(active, j - active).array() == Scalar(0)).all() &&
         (front.col(j).tail(order - j - 1).array() == Scalar(0)).all();
}

template <typename MatrixType_, int UpLo_, typename Ordering_>
Index MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::largestFullySummed(Index order, Index active, Index fullySummed,
                                                                          Index j, RealScalar& rest) const {
  ConstFrontMap front(m_front.data(), order, order);
  Index r = -1;
  RealScalar largest(0);
  rest = RealScalar(0);
  const auto visit = [&](Index i, const RealScalar& magnitude) {
    if (magnitude > largest) {
      rest = largest;
      largest = magnitude;
      r = i;
    } else {
      rest = numext::maxi(rest, magnitude);
    }
  };
  for (Index i = active; i < j; ++i) visit(i, numext::abs(front(j, i)));
  for (Index i = j + 1; i < fullySummed; ++i) visit(i, numext::abs(front(i, j)));
  rest = numext::maxi(rest, largestMagnitude(&front.coeffRef(0, j) + fullySummed, order - fullySummed));
  return r;
}

template <typename MatrixType_, int UpLo_, typename Ordering_>
EIGEN_ALWAYS_INLINE bool MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::factorOnePivotFront(Index s, Index order,
                                                                                              bool isRoot) {
  ConstFrontMap front(m_front.data(), order, order);
  const RealScalar d = numext::real(front(0, 0));
  if (d == RealScalar(0)) return false;

  // The root has no parent to delay to, and has no rows below its only diagonal to bound. With one fully summed column,
  // there's no 2x2 pivot to try instead.
  if (!isRoot && !(numext::abs(d) >= m_threshold * frontColumnMax(order, 0, 0, -1))) {
    appendFactor(s, 0, order);
    pushToParent(s, order, 1, 0);
    return true;
  }

  m_D[m_numPivots] = d;
  m_subdiagonal[m_numPivots] = Scalar(0);
  m_pivots[m_numPivots] = frontRows(order)[0];
  ++m_numPivots;
  m_numDelayed[s] = 0;

  const Index numBelow = order - 1;
  Scalar* L = appendFactor(s, 1, order);
  Scalar* contribution = m_stack.push(packedSize(numBelow));
  const Scalar* u = &front.coeffRef(0, 0);
  for (Index j = 1; j < order; ++j) {
    const Index length = order - j;
    const Scalar lj = numext::conj(u[j]) / d;
    const Scalar* column = &front.coeffRef(j, j);
    if (length < kVectorizedMinLength) {
      for (Index i = 0; i < length; ++i) contribution[i] = column[i] - u[j + i] * lj;
    } else {
      VectorMap(contribution, length) = ConstVectorMap(column, length) - ConstVectorMap(u + j, length) * lj;
    }
    contribution += length;
    L[j] = numext::conj(lj);
  }

  return true;
}

template <typename MatrixType_, int UpLo_, typename Ordering_>
void MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::pushToParent(Index s, Index order, Index fullySummed,
                                                                   Index numPivots) {
  const Index numDelayed = fullySummed - numPivots;
  m_numDelayed[s] = internal::convert_index<StorageIndex>(numDelayed);

  ConstFrontMap front(m_front.data(), order, order);
  Scalar* contribution = m_stack.push(packedSize(order - numPivots));
  if (order <= kSmallFrontMaxOrder) {
    const Index end = numext::mini(order, Index(kSmallFrontMaxOrder));
    for (Index j = numPivots; j < end; ++j) {
      for (Index i = j; i < end; ++i) *contribution++ = front(i, j);
    }
    return;
  }
  for (Index j = numPivots; j < order; ++j) {
    internal::smart_copy(&front.coeffRef(j, j), &front.coeffRef(j, j) + (order - j), contribution);
    contribution += order - j;
  }
}

template <typename MatrixType_, int UpLo_, typename Ordering_>
void MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::swapFront(Index order, Index a, Index b) {
  if (a == b) return;
  if (a > b) std::swap(a, b);

  FrontMap front(m_front.data(), order, order);
  internal::selfadjoint_swap_lower(front, a, b);
  std::swap(frontRows(order)[a], frontRows(order)[b]);
}

template <typename MatrixType_, int UpLo_, typename Ordering_>
void MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::eliminate1x1(Index order, Index k, Index panelEnd) {
  FrontMap front(m_front.data(), order, order);
  const RealScalar d = numext::real(front(k, k));
  m_D[m_numPivots] = d;
  m_subdiagonal[m_numPivots] = Scalar(0);
  m_pivots[m_numPivots] = frontRows(order)[k];
  ++m_numPivots;

  if (d == RealScalar(0)) {
    // The pivot test only accepts a zero pivot if its column is zero, so L's column is already zero.
    m_hasZeroPivot = true;
    return;
  }

  // Column k becomes L's column as each quotient is computed. Later columns only read the entries below their own row,
  // which are still unscaled.
  if (order <= kSmallFrontMaxOrder) {
    // A front this small is its own panel.
    const Index end = numext::mini(order, Index(kSmallFrontMaxOrder));
    for (Index j = k + 1; j < end; ++j) {
      const Scalar lj = numext::conj(front(j, k)) / d;
      for (Index i = j; i < end; ++i) front(i, j) -= front(i, k) * lj;
      front(j, k) = numext::conj(lj);
    }
    return;
  }
  for (Index j = k + 1; j < panelEnd; ++j) {
    const Scalar lj = numext::conj(front(j, k)) / d;
    subtractScaled(&front.coeffRef(j, j), &front.coeffRef(j, k), lj, order - j);
    front(j, k) = numext::conj(lj);
  }
  front.col(k).tail(order - panelEnd) = front.col(k).tail(order - panelEnd) / d;
}

template <typename MatrixType_, int UpLo_, typename Ordering_>
void MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::eliminate2x2(Index order, Index k, Index panelEnd) {
  FrontMap front(m_front.data(), order, order);
  const RealScalar d11 = numext::real(front(k, k));
  const Scalar d21 = front(k + 1, k);
  const RealScalar d22 = numext::real(front(k + 1, k + 1));
  m_D[m_numPivots] = d11;
  m_D[m_numPivots + 1] = d22;
  m_subdiagonal[m_numPivots] = d21;
  m_subdiagonal[m_numPivots + 1] = Scalar(0);
  m_pivots[m_numPivots] = frontRows(order)[k];
  m_pivots[m_numPivots + 1] = frontRows(order)[k + 1];
  m_numPivots += 2;

  // L = U D^-1, with D scaled by its largest entry to keep its determinant in range.
  const ScaledBlock block(d11, d22, numext::abs(d21));
  const Scalar b = d21 / block.scale;
  const Scalar conjB = numext::conj(b);

  const Index numRows = order - k - 2;
  // numRows can be zero, where coeffRef() would be out of bounds.
  Scalar* u1 = front.data() + rectangleSize(order, k) + (k + 2);
  Scalar* u2 = u1 + order;
  if (order <= kSmallFrontMaxOrder) {
    // A front this small is its own panel.
    constexpr Index kMaxRows = kSmallFrontMaxOrder - 2;
    Scalar l1[kMaxRows], l2[kMaxRows];
    const Index smallRows = numext::mini(numRows, kMaxRows);
    for (Index i = 0; i < smallRows; ++i) {
      l1[i] = ((u1[i] * block.c - u2[i] * b) / block.scale) / block.det;
      l2[i] = ((u2[i] * block.a - u1[i] * conjB) / block.scale) / block.det;
    }
    for (Index j = 0; j < smallRows; ++j) {
      const Scalar alpha1 = numext::conj(u1[j]);
      const Scalar alpha2 = numext::conj(u2[j]);
      Scalar* column = &front.coeffRef(k + 2, k + 2 + j);
      for (Index i = j; i < smallRows; ++i) column[i] -= l1[i] * alpha1 + l2[i] * alpha2;
    }
    for (Index i = 0; i < smallRows; ++i) {
      u1[i] = l1[i];
      u2[i] = l2[i];
    }
    front(k + 1, k) = Scalar(0);
    return;
  }
  if (m_pivotScratch.size() < rectangleSize(2, numRows)) m_pivotScratch.resize(rectangleSize(2, numRows));
  Scalar* l1 = m_pivotScratch.data();
  Scalar* l2 = l1 + numRows;
  if (numRows < kVectorizedMinLength) {
    for (Index i = 0; i < numRows; ++i) {
      l1[i] = ((u1[i] * block.c - u2[i] * b) / block.scale) / block.det;
      l2[i] = ((u2[i] * block.a - u1[i] * conjB) / block.scale) / block.det;
    }
  } else {
    const ConstVectorMap mapU1(u1, numRows);
    const ConstVectorMap mapU2(u2, numRows);
    VectorMap(l1, numRows) = ((mapU1 * block.c - mapU2 * b) / block.scale) / block.det;
    VectorMap(l2, numRows) = ((mapU2 * block.a - mapU1 * conjB) / block.scale) / block.det;
  }

  for (Index j = k + 2; j < panelEnd; ++j) {
    const Index offset = j - k - 2;
    subtractScaled(&front.coeffRef(j, j), l1 + offset, numext::conj(u1[offset]), l2 + offset, numext::conj(u2[offset]),
                   numRows - offset);
  }

  front(k + 1, k) = Scalar(0);
  internal::smart_copy(l1, l1 + numRows, u1);
  internal::smart_copy(l2, l2 + numRows, u2);
}

template <typename MatrixType_, int UpLo_, typename Ordering_>
void MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::updateTrailing(Index order, Index firstPivot, Index numPivots,
                                                                     Index panelEnd) {
  const Index trailing = order - panelEnd;
  if (trailing == 0 || numPivots == 0) return;

  FrontMap front(m_front.data(), order, order);
  if (m_pivotScratch.size() < rectangleSize(trailing, numPivots)) {
    m_pivotScratch.resize(rectangleSize(trailing, numPivots));
  }
  FrontMap W(m_pivotScratch.data(), trailing, numPivots);
  const auto L21 = front.block(panelEnd, firstPivot, trailing, numPivots);

  // W = L21 D
  Index p = m_numPivots - numPivots;
  for (Index c = 0; c < numPivots; ++c, ++p) {
    if (m_subdiagonal[p] != Scalar(0)) {
      W.col(c) = L21.col(c) * m_D[p] + L21.col(c + 1) * m_subdiagonal[p];
      W.col(c + 1) = L21.col(c) * numext::conj(m_subdiagonal[p]) + L21.col(c + 1) * m_D[p + 1];
      ++c;
      ++p;
    } else {
      W.col(c) = L21.col(c) * m_D[p];
    }
  }

  front.bottomRightCorner(trailing, trailing).template triangularView<Lower>() -= W * L21.adjoint();
}

template <typename MatrixType_, int UpLo_, typename Ordering_>
Index MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::factorFront(Index order, Index fullySummed, bool isRoot) {
  FrontMap front(m_front.data(), order, order);

  // Pivots [0, k) have been applied to the columns before panelEnd, and pivots [applied, k) have yet to be applied to
  // the others. Every pivot decision reads only current entries, so it's the one an unblocked factorization would make:
  // the panel is flushed before reading a column past its end. After a flush, applied == k and any column can be read.
  Index k = 0;
  Index applied = 0;
  Index panelEnd = order < kBlockedFrontMinOrder ? order : numext::mini(Index(kPanelWidth), fullySummed);
  const auto flush = [&] {
    updateTrailing(order, applied, k - applied, panelEnd);
    applied = k;
    panelEnd = numext::mini(k + kPanelWidth, fullySummed);
  };

  // A sweep tests each active column once, continuing past accepted pivots, so a column that keeps failing costs one
  // test per sweep instead of one per pivot. Columns [next, tail) are untested in this sweep. The front has no
  // acceptable pivot once every active column has failed since the last pivot, because those tests all read the same
  // values.
  Index next = 0;
  Index tail = fullySummed;
  bool acceptedPivot = false;
  // Without moved columns, the columns from here to the end have failed since the last pivot.
  Index lastPivotEnd = 0;

  while (k < fullySummed) {
    if (k >= panelEnd) flush();

    if (next >= tail) {
      next = k;
      if (acceptedPivot) {
        tail = tail == fullySummed ? lastPivotEnd : fullySummed;
        acceptedPivot = false;
        continue;
      }
      tail = fullySummed;
      if (!isRoot) break;

      if (applied < k) flush();

      // The pivot tests guarantee a pivot at the root, so only rounding error or a non-finite entry gets here. Take the
      // largest diagonal instead of leaving the column unfactored.
      Index largest;
      front.diagonal().segment(k, fullySummed - k).real().cwiseAbs().template maxCoeff<PropagateNumbers>(&largest);
      largest += k;
      swapFront(order, k, largest);

      // A zero here means the active diagonal is zero and every column is nonzero. The largest finite entry would pass
      // the 2x2 test, so some entry's magnitude isn't finite, and the pivot is NaN.
      if (numext::real(front(k, k)) == RealScalar(0)) {
        RealScalar nan = NumTraits<RealScalar>::quiet_NaN();
        EIGEN_FAST_MATH_CONSTANT_BARRIER(nan);
        front(k, k) = Scalar(nan);
      }

      eliminate1x1(order, k, panelEnd);
      ++k;
      next = k;
      continue;
    }

    if (next >= panelEnd) {
      flush();

      // Move the columns that failed in [k, next) past the untested ones, which fills the panel with untested columns
      // and lets it accept pivots without a flush for each one.
      const Index numUntested = tail - next;
      const Index numMoved = numext::mini(next - k, numUntested);
      for (Index i = 0; i < numMoved; ++i) swapFront(order, k + i, tail - 1 - i);
      next = k;
      tail = k + numUntested;
    }

    const Index j = next;
    const RealScalar djj = numext::real(front(j, j));
    const RealScalar absDjj = numext::abs(djj);

    // A zero pivot eliminates nothing, so it's accepted only if its column is exactly zero.
    if (djj != RealScalar(0) ? absDjj >= m_threshold * frontColumnMax(order, k, j, -1)
                             : frontColumnIsZero(order, k, j)) {
      swapFront(order, k, j);
      eliminate1x1(order, k, panelEnd);
      ++k;
      next = j + 1;
      lastPivotEnd = next;
      acceptedPivot = true;
      continue;
    }

    // Try a 2x2 pivot with the largest fully summed off-diagonal in column j.
    RealScalar columnJ;
    const Index r = largestFullySummed(order, k, fullySummed, j, columnJ);
    if (r == -1) {
      ++next;
      continue;
    }
    if (r >= panelEnd && applied < k) flush();

    // With D = scale [a b*; b c], |D^-1| = 1/(scale |det|) [|c| |b|; |b| |a|] where det = a c - |b|^2.
    //
    // |det| >= |b|^2 / 2 gives |a c| + |b|^2 <= 5 |det|, so det and the solves with D lose at most a few digits to
    // cancellation. For the pair holding a root front's largest off-diagonal, b = 1 and |a|, |c| < u <= 1/2 if neither
    // passes the 1x1 test, so |det| >= 1 - u^2 >= 3/4 and the root guarantee still holds.
    const RealScalar drr = numext::real(front(r, r));
    const ScaledBlock block(djj, drr, numext::abs(r > j ? front(r, j) : front(j, r)));
    const RealScalar a = numext::abs(block.a);
    const RealScalar b = block.absB;
    const RealScalar c = numext::abs(block.c);
    const RealScalar absDet = numext::abs(block.det);
    const RealScalar bound = absDet * block.scale / m_threshold;
    const RealScalar columnR = frontColumnMax(order, k, r, j);
    if (block.det != RealScalar(0) && absDet >= b * b / RealScalar(2) && c * columnJ + b * columnR <= bound &&
        b * columnJ + a * columnR <= bound) {
      swapFront(order, k, j);
      // Swapping k and j moved whatever was at k to j.
      swapFront(order, k + 1, r == k ? j : r);
      eliminate2x2(order, k, panelEnd);
      k += 2;
      next = numext::maxi(j + 1, k);
      lastPivotEnd = next;
      acceptedPivot = true;
    } else {
      ++next;
    }
  }

  flush();
  return k;
}

template <typename MatrixType_, int UpLo_, typename Ordering_>
typename MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::FactorType
MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::matrixL() const {
  eigen_assert(m_info != InvalidInput && "factorize() must be called before reading L.");

  // A front's rows are in elimination order, and L's rows and columns are in pivot order.
  std::vector<StorageIndex> pivotOf(rows());
  for (Index k = 0; k < rows(); ++k) pivotOf[m_pivots[k]] = internal::convert_index<StorageIndex>(k);

  std::vector<Triplet<Scalar, StorageIndex>> entries;
  entries.reserve(static_cast<std::size_t>(m_LNonZeros) + static_cast<std::size_t>(rows()));
  for (const FactoredFront& factored : m_factoredFronts) {
    const Index order = factored.order;
    const StorageIndex* rows = m_factorRows.storage.data() + factored.rows;
    const Index numPivots = factored.numPivots;
    ConstFrontMap L(m_L.storage.data() + factored.values, order, numPivots);
    for (Index j = 0; j < numPivots; ++j) {
      const StorageIndex col = pivotOf[rows[j]];
      entries.emplace_back(col, col, Scalar(1));
      for (Index i = j + 1; i < order; ++i) entries.emplace_back(pivotOf[rows[i]], col, L(i, j));
    }
  }

  FactorType result(rows(), cols());
  result.setFromTriplets(entries.begin(), entries.end());
  return result;
}

#ifndef EIGEN_PARSED_BY_DOXYGEN
template <typename MatrixType_, int UpLo_, typename Ordering_>
template <typename Rhs, typename Dest>
void MultifrontalLDLT<MatrixType_, UpLo_, Ordering_>::_solve_impl(const MatrixBase<Rhs>& rhs,
                                                                  MatrixBase<Dest>& dest) const {
  eigen_assert(m_isInitialized && "Decomposition is not initialized.");
  eigen_assert(m_info != InvalidInput && "factorize() must be called before solve().");
  eigen_assert(rhs.rows() == rows());

  const typename internal::nested_eval<Rhs, Dynamic>::type actualRhs(rhs.derived());
  const Index numRhs = rhs.cols();
  FrontMatrix X(rows(), numRhs);
  for (Index col = 0; col < numRhs; ++col) {
    for (Index i = 0; i < rows(); ++i) X(m_perm[i], col) = actualRhs.coeff(i, col);
  }

  // Each front's rows of X, so a large factor applies as one dense product.
  FrontMatrix Y(m_maxFrontOrder, numRhs);
  const auto gather = [&](const StorageIndex* rows, Index count) {
    for (Index col = 0; col < numRhs; ++col) {
      for (Index i = 0; i < count; ++i) Y(i, col) = X(rows[i], col);
    }
  };
  const auto scatter = [&](const StorageIndex* rows, Index count) {
    for (Index col = 0; col < numRhs; ++col) {
      for (Index i = 0; i < count; ++i) X(rows[i], col) = Y(i, col);
    }
  };

  // L Y = B
  for (const FactoredFront& factored : m_factoredFronts) {
    const Index order = factored.order;
    const Index numPivots = factored.numPivots;
    const StorageIndex* rows = m_factorRows.storage.data() + factored.rows;
    ConstFrontMap L(m_L.storage.data() + factored.values, order, numPivots);
    if (order * numPivots < kDenseSolveMinSize) {
      for (Index col = 0; col < numRhs; ++col) {
        Scalar* x = X.col(col).data();
        for (Index j = 0; j < numPivots; ++j) {
          const Scalar value = x[rows[j]];
          const Scalar* column = &L.coeffRef(0, j);
          for (Index i = j + 1; i < order; ++i) x[rows[i]] -= column[i] * value;
        }
      }
    } else {
      const Index numBelow = order - numPivots;
      gather(rows, order);
      // One right-hand side takes the vector kernels, which don't pack L.
      if (numRhs == 1) {
        auto y = Y.col(0);
        L.topRows(numPivots).template triangularView<UnitLower>().solveInPlace(y.head(numPivots));
        y.segment(numPivots, numBelow).noalias() -= L.bottomRows(numBelow) * y.head(numPivots);
      } else {
        L.topRows(numPivots).template triangularView<UnitLower>().solveInPlace(Y.topRows(numPivots));
        Y.middleRows(numPivots, numBelow).noalias() -= L.bottomRows(numBelow) * Y.topRows(numPivots);
      }
      scatter(rows, order);
    }
  }

  // D Z = Y. A zero pivot gets a zero solution, the minimum norm solution for that block.
  for (Index col = 0; col < numRhs; ++col) {
    Scalar* x = X.col(col).data();
    auto block = m_blocks.cbegin();
    for (Index k = 0; k < rows(); ++k) {
      const Index row = m_pivots[k];
      if (m_subdiagonal[k] != Scalar(0)) {
        const Index nextRow = m_pivots[k + 1];
        const Scalar b = m_subdiagonal[k] / block->scale;
        const Scalar y1 = x[row];
        const Scalar y2 = x[nextRow];
        x[row] = ((y1 * block->c - y2 * numext::conj(b)) / block->scale) / block->det;
        x[nextRow] = ((y2 * block->a - y1 * b) / block->scale) / block->det;
        ++block;
        ++k;
      } else if (m_D[k] != RealScalar(0)) {
        x[row] /= m_D[k];
      } else {
        x[row] = Scalar(0);
      }
    }
  }

  // L^* W = Z
  for (auto factored = m_factoredFronts.rbegin(); factored != m_factoredFronts.rend(); ++factored) {
    const Index order = factored->order;
    const Index numPivots = factored->numPivots;
    const StorageIndex* rows = m_factorRows.storage.data() + factored->rows;
    ConstFrontMap L(m_L.storage.data() + factored->values, order, numPivots);
    if (order * numPivots < kDenseSolveMinSize) {
      for (Index col = 0; col < numRhs; ++col) {
        Scalar* x = X.col(col).data();
        for (Index j = numPivots - 1; j >= 0; --j) {
          Scalar value = x[rows[j]];
          const Scalar* column = &L.coeffRef(0, j);
          for (Index i = order - 1; i > j; --i) value -= numext::conj(column[i]) * x[rows[i]];
          x[rows[j]] = value;
        }
      }
    } else {
      const Index numBelow = order - numPivots;
      gather(rows, order);
      if (numRhs == 1) {
        auto y = Y.col(0);
        y.head(numPivots).noalias() -= L.bottomRows(numBelow).adjoint() * y.segment(numPivots, numBelow);
        L.topRows(numPivots).adjoint().template triangularView<UnitUpper>().solveInPlace(y.head(numPivots));
      } else {
        Y.topRows(numPivots).noalias() -= L.bottomRows(numBelow).adjoint() * Y.middleRows(numPivots, numBelow);
        L.topRows(numPivots).adjoint().template triangularView<UnitUpper>().solveInPlace(Y.topRows(numPivots));
      }
      scatter(rows, numPivots);
    }
  }

  for (Index col = 0; col < numRhs; ++col) {
    for (Index i = 0; i < rows(); ++i) dest.coeffRef(i, col) = X(m_perm[i], col);
  }
}
#endif  // EIGEN_PARSED_BY_DOXYGEN

}  // end namespace Eigen

#endif  // EIGEN_MULTIFRONTAL_LDLT_H
