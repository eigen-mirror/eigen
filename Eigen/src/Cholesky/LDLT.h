// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Keir Mierle <mierle@gmail.com>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2011 Timothy E. Holy <tim.holy@gmail.com >
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_LDLT_H
#define EIGEN_LDLT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {
template <typename MatrixType_, int UpLo_>
struct traits<LDLT<MatrixType_, UpLo_> > : traits<MatrixType_> {
  using XprKind = MatrixXpr;
  using StorageKind = SolverStorage;
  using StorageIndex = int;
  enum { Flags = 0 };
};

template <typename MatrixType, int UpLo>
struct LDLT_Traits;

// PositiveSemiDef means positive semi-definite and non-zero; same for NegativeSemiDef
enum SignMatrix { PositiveSemiDef, NegativeSemiDef, ZeroSign, Indefinite };

// Panel width of the blocked factorization (defined below); declared here for the size constructor.
template <typename MatrixType>
inline Index ldlt_blocksize(Index size);
}  // namespace internal

/** \ingroup Cholesky_Module
 *
 * \class LDLT
 *
 * \brief Robust Cholesky decomposition of a matrix with pivoting
 *
 * \tparam MatrixType_ the type of the matrix of which to compute the LDL^T Cholesky decomposition
 * \tparam UpLo_ the triangular part that will be used for the decomposition: Lower (default) or Upper.
 *             The other triangular part won't be read.
 *
 * Perform a robust Cholesky decomposition of a positive semidefinite or negative semidefinite
 * matrix \f$ A \f$ such that \f$ A =  P^TLDL^*P \f$, where P is a permutation matrix, L
 * is lower triangular with a unit diagonal and D is a diagonal matrix.
 *
 * Each step pivots on the diagonal entry of largest magnitude of the remaining Schur complement
 * (complete diagonal pivoting, as in LAPACK's xPSTRF), so that for a semidefinite matrix of rank r
 * the n - r entries of D that vanish in exact arithmetic come last, at the level of rounding errors
 * and of either sign. Avoiding the square root on D also stabilizes the computation.
 *
 * LDLT does not determine a numerical rank, though. Also, do not use a Cholesky decomposition to
 * determine whether a system of equations has a solution.
 *
 * This class supports the \link InplaceDecomposition inplace decomposition \endlink mechanism.
 *
 * D is purely diagonal, so this class is not backward stable on indefinite matrices: it can report
 * \c Success for a factorization with a large backward error. If isPositive() and isNegative() both
 * return false, D has pivots of both signs: unless that comes from the rounding-level pivots of a
 * singular semidefinite matrix, the matrix is indefinite, and BunchKaufman, which produces a
 * block-diagonal D, should be used instead.
 *
 * \sa MatrixBase::ldlt(), SelfAdjointView::ldlt(), class LLT, class BunchKaufman
 */
template <typename MatrixType_, int UpLo_>
class LDLT : public SolverBase<LDLT<MatrixType_, UpLo_> > {
 public:
  using MatrixType = MatrixType_;
  using Base = SolverBase<LDLT>;
  friend class SolverBase<LDLT>;

  EIGEN_GENERIC_PUBLIC_INTERFACE(LDLT)
  enum {
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
    UpLo = UpLo_
  };
  using TmpMatrixType = Matrix<Scalar, RowsAtCompileTime, 1, 0, MaxRowsAtCompileTime, 1>;

  using TranspositionType = Transpositions<RowsAtCompileTime, MaxRowsAtCompileTime>;
  using PermutationType = PermutationMatrix<RowsAtCompileTime, MaxRowsAtCompileTime>;

  using Traits = internal::LDLT_Traits<MatrixType, UpLo>;

  /** \brief Default Constructor.
   *
   * The default constructor is useful in cases in which the user intends to
   * perform decompositions via LDLT::compute(const MatrixType&).
   */
  LDLT()
      : m_matrix(),
        m_l1_norm(0),
        m_transpositions(),
        m_sign(internal::ZeroSign),
        m_isInitialized(false),
        m_info(InvalidInput) {}

  /** \brief Default Constructor with memory preallocation
   *
   * Like the default constructor but with preallocation of the internal data
   * according to the specified problem \a size.
   *
   * Above 96 rows, compute() factors a dynamic-size matrix in blocks, and the matrix products of the blocked
   * factorization can allocate their packing buffers, as those of LLT and PartialPivLU do. Fixed-size and bounded
   * matrices are factored unblocked and allocate nothing.
   * \sa LDLT()
   */
  explicit LDLT(Index size)
      : m_matrix(size, size),
        m_l1_norm(0),
        m_transpositions(size),
        m_temporary(size),
        // The shape compute() gives the panel workspace, so that it does not reallocate it for this size.
        m_workspace(size, internal::ldlt_blocksize<MatrixType>(size)),
        m_sign(internal::ZeroSign),
        m_isInitialized(false),
        m_info(InvalidInput) {}

  /** \brief Constructor with decomposition
   *
   * This calculates the decomposition for the input \a matrix.
   *
   * \sa LDLT(Index size)
   */
  template <typename InputType>
  explicit LDLT(const EigenBase<InputType>& matrix)
      : m_matrix(matrix.rows(), matrix.cols()),
        m_l1_norm(0),
        m_transpositions(matrix.rows()),
        m_temporary(matrix.rows()),
        m_sign(internal::ZeroSign),
        m_isInitialized(false),
        m_info(InvalidInput) {
    compute(matrix.derived());
  }

  /** \brief Constructs a LDLT factorization from a given matrix
   *
   * This overloaded constructor is provided for \link InplaceDecomposition inplace decomposition \endlink when \c
   * MatrixType is a Eigen::Ref.
   *
   * \sa LDLT(const EigenBase&)
   */
  template <typename InputType>
  explicit LDLT(EigenBase<InputType>& matrix)
      : m_matrix(matrix.derived()),
        m_l1_norm(0),
        m_transpositions(matrix.rows()),
        m_temporary(matrix.rows()),
        m_sign(internal::ZeroSign),
        m_isInitialized(false),
        m_info(InvalidInput) {
    compute(matrix.derived());
  }

  /** Clear any existing decomposition
   * \sa rankUpdate(w,sigma)
   */
  void setZero() { m_isInitialized = false; }

  /** \returns a view of the upper triangular matrix U */
  inline typename Traits::MatrixU matrixU() const {
    eigen_assert(m_isInitialized && "LDLT is not initialized.");
    return Traits::getU(m_matrix);
  }

  /** \returns a view of the lower triangular matrix L */
  inline typename Traits::MatrixL matrixL() const {
    eigen_assert(m_isInitialized && "LDLT is not initialized.");
    return Traits::getL(m_matrix);
  }

  /** \returns the permutation matrix P as a transposition sequence.
   */
  inline const TranspositionType& transpositionsP() const {
    eigen_assert(m_isInitialized && "LDLT is not initialized.");
    return m_transpositions;
  }

  /** \returns the coefficients of the diagonal matrix D */
  inline Diagonal<const MatrixType> vectorD() const {
    eigen_assert(m_isInitialized && "LDLT is not initialized.");
    return m_matrix.diagonal();
  }

  /** \returns true if the matrix is positive (semidefinite) */
  inline bool isPositive() const {
    eigen_assert(m_isInitialized && "LDLT is not initialized.");
    return m_sign == internal::PositiveSemiDef || m_sign == internal::ZeroSign;
  }

  /** \returns true if the matrix is negative (semidefinite) */
  inline bool isNegative(void) const {
    eigen_assert(m_isInitialized && "LDLT is not initialized.");
    return m_sign == internal::NegativeSemiDef || m_sign == internal::ZeroSign;
  }

#ifdef EIGEN_PARSED_BY_DOXYGEN
  /** \returns a solution x of \f$ A x = b \f$ using the current decomposition of A.
   *
   * This function also supports in-place solves using the syntax <tt>x = decompositionObject.solve(x)</tt> .
   *
   * \note_about_checking_solutions
   *
   * More precisely, this method solves \f$ A x = b \f$ using the decomposition \f$ A = P^T L D L^* P \f$
   * by solving the systems \f$ P^T y_1 = b \f$, \f$ L y_2 = y_1 \f$, \f$ D y_3 = y_2 \f$,
   * \f$ L^* y_4 = y_3 \f$ and \f$ P x = y_4 \f$ in succession. If the matrix \f$ A \f$ is singular, then
   * \f$ D \f$ will also be singular (all the other matrices are invertible). In that case, the
   * least-square solution of \f$ D y_3 = y_2 \f$ is computed. This does not mean that this function
   * computes the least-square solution of \f$ A x = b \f$ if \f$ A \f$ is singular.
   *
   * \sa MatrixBase::ldlt(), SelfAdjointView::ldlt()
   */
  template <typename Rhs>
  inline Solve<LDLT, Rhs> solve(const MatrixBase<Rhs>& b) const;
#endif

  template <typename Derived>
  bool solveInPlace(MatrixBase<Derived>& bAndX) const;

  template <typename InputType>
  LDLT& compute(const EigenBase<InputType>& matrix);

  /** \returns an estimate of the reciprocal condition number of the matrix of
   *  which \c *this is the LDLT decomposition.
   */
  RealScalar rcond() const {
    eigen_assert(m_isInitialized && "LDLT is not initialized.");
    return internal::rcond_estimate_helper(m_l1_norm, *this);
  }

  template <typename Derived>
  LDLT& rankUpdate(const MatrixBase<Derived>& w, const RealScalar& alpha = 1);

  /** \returns the determinant of the matrix of which *this is the Cholesky decomposition.
   *
   * It has only linear complexity (that is, O(n) where n is the dimension of the square matrix)
   * as the Cholesky decomposition has already been computed.
   *
   * \warning a determinant can be very big or small, so for matrices
   * of large enough dimension, there is a risk of overflow/underflow.
   * One way to work around that is to use logAbsDeterminant() and signDeterminant() instead.
   * Also, do not rely on the determinant being exactly zero for testing
   * singularity or rank-deficiency.
   *
   * \pre info() returns \c Success. A failed factorization does not represent the input matrix.
   *
   * \sa absDeterminant(), logAbsDeterminant(), signDeterminant(), MatrixBase::determinant()
   */
  Scalar determinant() const;

  /** \returns the absolute value of the determinant of the matrix of which *this is the Cholesky decomposition.
   *
   * It has only linear complexity (that is, O(n) where n is the dimension of the square matrix)
   * as the Cholesky decomposition has already been computed.
   *
   * \warning a determinant can be very big or small, so for matrices
   * of large enough dimension, there is a risk of overflow/underflow.
   * One way to work around that is to use logAbsDeterminant() instead.
   *
   * \pre info() returns \c Success. A failed factorization does not represent the input matrix.
   *
   * \sa determinant(), logAbsDeterminant(), signDeterminant(), MatrixBase::determinant()
   */
  RealScalar absDeterminant() const;

  /** \returns the natural log of the absolute value of the determinant of the matrix of which *this is the Cholesky
   * decomposition.
   *
   * It has only linear complexity (that is, O(n) where n is the dimension of the square matrix)
   * as the Cholesky decomposition has already been computed.
   *
   * \note This method is useful to work around the risk of overflow/underflow that's inherent
   * to determinant computation.
   *
   * \pre info() returns \c Success. A failed factorization does not represent the input matrix.
   *
   * \sa determinant(), absDeterminant(), signDeterminant(), MatrixBase::determinant()
   */
  RealScalar logAbsDeterminant() const;

  /** \returns the sign of the determinant of the matrix of which *this is the Cholesky decomposition,
   * that is, \c 1, \c -1, or \c 0 if the matrix is singular.
   *
   * It has only linear complexity (that is, O(n) where n is the dimension of the square matrix)
   * as the Cholesky decomposition has already been computed.
   *
   * \pre info() returns \c Success. A failed factorization does not represent the input matrix.
   *
   * \sa determinant(), absDeterminant(), logAbsDeterminant(), MatrixBase::determinant()
   */
  Scalar signDeterminant() const;

  /** \returns the internal LDLT decomposition matrix
   *
   * TODO: document the storage layout.
   */
  inline const MatrixType& matrixLDLT() const {
    eigen_assert(m_isInitialized && "LDLT is not initialized.");
    return m_matrix;
  }

  MatrixType reconstructedMatrix() const;

  /** \returns the adjoint of \c *this, that is, a const reference to the decomposition itself as the underlying matrix
   * is self-adjoint.
   *
   * This method is provided for compatibility with other matrix decompositions, thus enabling generic code such as:
   * \code x = decomposition.adjoint().solve(b) \endcode
   */
  const LDLT& adjoint() const { return *this; }

  EIGEN_DEVICE_FUNC constexpr Index rows() const noexcept { return m_matrix.rows(); }
  EIGEN_DEVICE_FUNC constexpr Index cols() const noexcept { return m_matrix.cols(); }

  /** \brief Reports whether previous computation was successful.
   *
   * \returns \c Success if computation was successful,
   *          \c NumericalIssue if the factorization failed: a pivot is not finite, or is zero above a
   *          non-zero column. In exact arithmetic only an indefinite matrix has the latter, but rounding
   *          errors can give it to a singular semidefinite matrix too.
   */
  ComputationInfo info() const {
    eigen_assert(m_isInitialized && "LDLT is not initialized.");
    return m_info;
  }

#ifndef EIGEN_PARSED_BY_DOXYGEN
  template <typename RhsType, typename DstType>
  void _solve_impl(const RhsType& rhs, DstType& dst) const;

  template <bool Conjugate, typename RhsType, typename DstType>
  void _solve_impl_transposed(const RhsType& rhs, DstType& dst) const;
#endif

 protected:
  EIGEN_STATIC_ASSERT_NON_INTEGER(Scalar)

  /** \internal
   * Used to compute and store the Cholesky decomposition A = L D L^* = U^* D U.
   * The strict upper part is used during the decomposition, the strict lower
   * part correspond to the coefficients of L (its diagonal is equal to 1 and
   * is not stored), and the diagonal entries correspond to D.
   */
  MatrixType m_matrix;
  RealScalar m_l1_norm;
  TranspositionType m_transpositions;
  TmpMatrixType m_temporary;
  // L21 D for the trailing update of the blocked factorization; it stays empty below the blocking threshold.
  Matrix<Scalar, Dynamic, Dynamic> m_workspace;
  internal::SignMatrix m_sign;
  bool m_isInitialized;
  ComputationInfo m_info;
};

namespace internal {

/** \internal Panel width of the blocked factorization of a \a size x \a size matrix, or 0 where a single
 * unblocked panel covers the whole matrix: up to three panel widths, below which the level-3 update does not pay
 * off, and for fixed-size or bounded matrices, which must not allocate the panel workspace.
 *
 * The panel's level-2 updates run at a rate R, and the trailing update at P / (1 + kappa / nb), kappa being the
 * fixed cost of a pass in steps of its depth. Then T(nb) = n^3 / (3 P) (1 + kappa / nb) + n^2 nb / (2 R) is least at
 * nb = sqrt(c n), c = 2 kappa R / (3 P). Measured c ranges from about 0.7 to 4 with AVX2, AVX-512, NEON and SME
 * products, and grows once the trailing matrix outgrows the last-level cache. c = 3, with nb rounded to a multiple
 * of 16 and clamped to [32, 128], the range the measured optimum spans, loses about 2% to the best width on average.
 * Can be overridden (mainly for testing the panel logic at small sizes) by defining EIGEN_LDLT_BLOCKSIZE. */
template <typename MatrixType>
inline Index ldlt_blocksize(Index size) {
  if (int(MatrixType::MaxRowsAtCompileTime) != Dynamic || int(MatrixType::MaxColsAtCompileTime) != Dynamic) return 0;
#ifdef EIGEN_LDLT_BLOCKSIZE
  const Index nb = Index(EIGEN_LDLT_BLOCKSIZE);
#else
  constexpr Index kMinBlockSize = 32;
  // Three of the narrowest panels are not blocked either, so small matrices skip the square root.
  if (size <= 3 * kMinBlockSize) return 0;
  const Index unclamped = 16 * Index(numext::round(numext::sqrt(3.0 * double(size)) / 16));
  const Index nb = numext::mini(numext::maxi(unclamped, kMinBlockSize), Index(128));
#endif
  return nb > 0 && size > 3 * nb ? nb : 0;
}

template <int UpLo>
struct ldlt_inplace;

template <>
struct ldlt_inplace<Lower> {
  // Settles the cases that leave nothing to factor, and returns whether it did: an empty or 1x1 matrix, and a
  // diagonal that is entirely zero, which offers no pivot and leaves the matrix either zero or indefinite. Otherwise
  // it copies the diagonal into temp, where factor_panel() expects it.
  template <typename MatrixType, typename TranspositionType, typename Workspace>
  static bool factor_trivial(const MatrixType& mat, TranspositionType& transpositions, Workspace& temp,
                             SignMatrix& sign, bool& ret) {
    using Scalar = typename MatrixType::Scalar;
    using RealScalar = typename MatrixType::RealScalar;
    const Index size = mat.rows();

    if (size <= 1) {
      transpositions.setIdentity();
      if (size == 0)
        sign = ZeroSign;
      else if (numext::real(mat.coeff(0, 0)) > static_cast<RealScalar>(0))
        sign = PositiveSemiDef;
      else if (numext::real(mat.coeff(0, 0)) < static_cast<RealScalar>(0))
        sign = NegativeSemiDef;
      else
        sign = ZeroSign;
      ret = size == 0 || (numext::isfinite)(numext::real(mat.coeff(0, 0)));
      return true;
    }

    temp = mat.diagonal().real().template cast<Scalar>();
    if ((temp.array() == Scalar(0)).all()) {
      transpositions.setIdentity();
      sign = ZeroSign;
      ret = true;
      for (Index j = 0; ret && j < size; ++j) ret = (mat.col(j).tail(size - j - 1).array() == Scalar(0)).all();
      return true;
    }
    return false;
  }

  // Factors columns [k0, k0 + kb) left-looking, column k pivoting on max |diag(S_k)|, S_k = A22 - L21 D L21^*
  // (complete diagonal pivoting, as in LAPACK's xPSTF2). Invariant: mat(k0:n, k0:n) holds S_k0 off its diagonal and
  // temp(k:n) holds diag(S_k). The caller interchanges the rows of the columns before k0.
  template <typename MatrixType, typename TranspositionType, typename Workspace>
  static bool factor_panel(MatrixType& mat, Index k0, Index kb, TranspositionType& transpositions, Workspace& temp,
                           SignMatrix& sign) {
    using Scalar = typename MatrixType::Scalar;
    using RealScalar = typename MatrixType::RealScalar;
    using IndexType = typename TranspositionType::StorageIndex;
    const Index size = mat.rows();
    bool ret = true;

    for (Index k = k0; k < k0 + kb; ++k) {
      // Find largest diagonal element
      Index index_of_biggest_in_corner;
      temp.tail(size - k).real().cwiseAbs().maxCoeff(&index_of_biggest_in_corner);
      index_of_biggest_in_corner += k;

      transpositions.coeffRef(k) = IndexType(index_of_biggest_in_corner);
      if (k != index_of_biggest_in_corner) {
        auto trailing = mat.bottomRightCorner(size - k0, size - k0);
        selfadjoint_swap_lower(trailing, k - k0, index_of_biggest_in_corner - k0);
        numext::swap(temp.coeffRef(k), temp.coeffRef(index_of_biggest_in_corner));
      }

      // partition the matrix:
      //       A00 |  -  |  -
      // lu  = A10 | A11 |  -
      //       A20 | A21 | A22
      // where A10 and A20 span the panel columns [k0, k); those before k0 are already applied to A21. The
      // product's operand D A10^* takes temp(k0:k), whose diagonal entries of S have all been used as pivots.
      Index rs = size - k - 1;
      Index done = k - k0;
      auto A21 = mat.block(k + 1, k, rs, fix<1>);

      if (done > 0 && rs > 0) {
        auto A10 = mat.block(k, k0, fix<1>, done);
        auto A20 = mat.block(k + 1, k0, rs, done);
        temp.segment(k0, done) = mat.diagonal().real().segment(k0, done).asDiagonal() * A10.adjoint();
        A21.noalias() -= A20 * temp.segment(k0, done);
      }

      // In some previous versions of Eigen (e.g., 3.2.1), the scaling was omitted if the pivot
      // was smaller than the cutoff value. However, since LDLT is not rank-revealing
      // we should only make sure that we do not introduce INF or NaN values.
      // Remark that LAPACK also uses 0 as the cutoff value.
      RealScalar realAkk = numext::real(temp.coeff(k));
      mat.coeffRef(k, k) = Scalar(realAkk);
      bool pivot_is_valid = (numext::abs(realAkk) > RealScalar(0));
      if (!(numext::isfinite)(realAkk)) ret = false;

      // A zero pivot is the largest remaining, so the rest of D is zero too, and its column must be zero as well.
      if ((rs > 0) && pivot_is_valid) {
        A21 /= realAkk;
        // diag(S_{k+1}) = diag(S_k) - conj(L21) .* (D_k L21), where the next pivot is searched for. Squaring L21
        // first would underflow for |L21| < sqrt(min) on a graded matrix, where D_k |L21|^2 is representable.
        temp.tail(rs).real() -= A21.conjugate().cwiseProduct(realAkk * A21).real();
      } else if (rs > 0) {
        ret = ret && (A21.array() == Scalar(0)).all();
      }

      if (sign == PositiveSemiDef) {
        if (realAkk < static_cast<RealScalar>(0)) sign = Indefinite;
      } else if (sign == NegativeSemiDef) {
        if (realAkk > static_cast<RealScalar>(0)) sign = Indefinite;
      } else if (sign == ZeroSign) {
        if (realAkk > static_cast<RealScalar>(0))
          sign = PositiveSemiDef;
        else if (realAkk < static_cast<RealScalar>(0))
          sign = NegativeSemiDef;
      }
    }

    return ret;
  }

  template <typename MatrixType, typename TranspositionType, typename Workspace>
  static bool unblocked(MatrixType& mat, TranspositionType& transpositions, Workspace& temp, SignMatrix& sign) {
    EIGEN_INTERNAL_ASSERT_SQUARE(mat);
    bool ret = true;
    if (factor_trivial(mat, transpositions, temp, sign, ret)) return ret;
    return factor_panel(mat, 0, mat.rows(), transpositions, temp, sign);
  }

  // Blocked (level-3) variant, with the panel/trailing-update structure of LAPACK's xPSTRF: each panel is factored
  // by factor_panel(), then a single product applies it to the trailing Schur complement, A22 -= L21 D L21^*.
  template <typename MatrixType, typename TranspositionType, typename Workspace, typename BlockWorkspace>
  static bool blocked(MatrixType& mat, TranspositionType& transpositions, Workspace& temp, BlockWorkspace& workspace,
                      SignMatrix& sign) {
    EIGEN_INTERNAL_ASSERT_SQUARE(mat);
    const Index size = mat.rows();
    const Index nb = ldlt_blocksize<MatrixType>(size);
    if (nb == 0) return unblocked(mat, transpositions, temp, sign);
    bool ret = true;
    if (factor_trivial(mat, transpositions, temp, sign, ret)) return ret;

    workspace.resize(size, nb);
    for (Index k0 = 0; k0 < size; k0 += nb) {
      const Index kb = numext::mini(nb, size - k0);
      const Index k1 = k0 + kb;
      if (!factor_panel(mat, k0, kb, transpositions, temp, sign)) ret = false;

      // The panel's interchanges, applied to the rows of the columns before it. Column-major storage takes them a
      // column at a time, where a row swap would stride through memory.
      EIGEN_IF_CONSTEXPR (MatrixType::IsRowMajor) {
        for (Index k = k0; k < k1; ++k) {
          const Index p = transpositions.coeff(k);
          if (p != k) mat.row(k).head(k0).swap(mat.row(p).head(k0));
        }
      } else {
        for (Index c = 0; c < k0; ++c) {
          for (Index k = k0; k < k1; ++k) {
            const Index p = transpositions.coeff(k);
            if (p != k) numext::swap(mat.coeffRef(k, c), mat.coeffRef(p, c));
          }
        }
      }

      const Index rs = size - k1;
      if (rs > 0) {
        auto L21 = mat.block(k1, k0, rs, kb);
        auto W = workspace.topLeftCorner(rs, kb);
        W.noalias() = L21 * mat.diagonal().real().segment(k0, kb).asDiagonal();
        mat.block(k1, k1, rs, rs).template triangularView<Lower>() -= L21 * W.adjoint();
      }
    }
    return ret;
  }

  // Reference for the algorithm: Davis and Hager, "Multiple Rank
  // Modifications of a Sparse Cholesky Factorization" (Algorithm 1)
  // Trivial rearrangements of their computations (Timothy E. Holy)
  // allow their algorithm to work for rank-1 updates even if the
  // original matrix is not of full rank.
  // Here only rank-1 updates are implemented, to reduce the
  // requirement for intermediate storage and improve accuracy
  template <typename MatrixType, typename WDerived>
  static bool updateInPlace(MatrixType& mat, MatrixBase<WDerived>& w,
                            const typename MatrixType::RealScalar& sigma = 1) {
    using Scalar = typename MatrixType::Scalar;
    using RealScalar = typename MatrixType::RealScalar;

    const Index size = mat.rows();
    EIGEN_INTERNAL_ASSERT_SQUARE(mat);
    EIGEN_INTERNAL_ASSERT_SIZE_MATCHES_ROWS(w, mat);

    RealScalar alpha = 1;

    // Apply the update
    for (Index j = 0; j < size; j++) {
      // Check for termination due to an original decomposition of low-rank
      if (!(numext::isfinite)(alpha)) break;

      // Update the diagonal terms
      RealScalar dj = numext::real(mat.coeff(j, j));
      Scalar wj = w.coeff(j);
      RealScalar swj2 = sigma * numext::abs2(wj);
      RealScalar gamma = dj * alpha + swj2;

      // A zero contribution leaves both quantities unchanged, but spelling that out as an addition of swj2/alpha and
      // swj2/dj evaluates 0/0 when alpha or the pivot is zero. The resulting NaN reaches the termination test above
      // on the next iteration, which reads it as a low-rank signal and abandons the remainder of the update. Skip the
      // no-op instead. A zero pivot with a nonzero contribution still yields an infinite alpha, so genuine low-rank
      // termination is unaffected.
      if (!numext::is_exactly_zero(swj2)) {
        mat.coeffRef(j, j) += swj2 / alpha;
        alpha += swj2 / dj;
      }

      // Update the terms of L
      Index rs = size - j - 1;
      w.tail(rs) -= wj * mat.col(j).tail(rs);
      if (!numext::is_exactly_zero(gamma)) mat.col(j).tail(rs) += (sigma * numext::conj(wj) / gamma) * w.tail(rs);
    }
    return true;
  }

  template <typename MatrixType, typename TranspositionType, typename Workspace, typename WType>
  static bool update(MatrixType& mat, const TranspositionType& transpositions, Workspace& tmp, const WType& w,
                     const typename MatrixType::RealScalar& sigma = 1) {
    // Apply the permutation to the input w
    tmp = transpositions * w;

    return ldlt_inplace<Lower>::updateInPlace(mat, tmp, sigma);
  }
};

template <>
struct ldlt_inplace<Upper> {
  template <typename MatrixType, typename TranspositionType, typename Workspace>
  static EIGEN_STRONG_INLINE bool unblocked(MatrixType& mat, TranspositionType& transpositions, Workspace& temp,
                                            SignMatrix& sign) {
    Transpose<MatrixType> matt(mat);
    return ldlt_inplace<Lower>::unblocked(matt, transpositions, temp, sign);
  }

  template <typename MatrixType, typename TranspositionType, typename Workspace, typename BlockWorkspace>
  static EIGEN_STRONG_INLINE bool blocked(MatrixType& mat, TranspositionType& transpositions, Workspace& temp,
                                          BlockWorkspace& workspace, SignMatrix& sign) {
    Transpose<MatrixType> matt(mat);
    return ldlt_inplace<Lower>::blocked(matt, transpositions, temp, workspace, sign);
  }

  template <typename MatrixType, typename TranspositionType, typename Workspace, typename WType>
  static EIGEN_STRONG_INLINE bool update(MatrixType& mat, TranspositionType& transpositions, Workspace& tmp, WType& w,
                                         const typename MatrixType::RealScalar& sigma = 1) {
    Transpose<MatrixType> matt(mat);
    return ldlt_inplace<Lower>::update(matt, transpositions, tmp, w.conjugate(), sigma);
  }
};

template <typename MatrixType>
struct LDLT_Traits<MatrixType, Lower> {
  using MatrixL = const TriangularView<const MatrixType, UnitLower>;
  using MatrixU = const TriangularView<const typename MatrixType::AdjointReturnType, UnitUpper>;
  static inline MatrixL getL(const MatrixType& m) { return MatrixL(m); }
  static inline MatrixU getU(const MatrixType& m) { return MatrixU(m.adjoint()); }
};

template <typename MatrixType>
struct LDLT_Traits<MatrixType, Upper> {
  using MatrixL = const TriangularView<const typename MatrixType::AdjointReturnType, UnitLower>;
  using MatrixU = const TriangularView<const MatrixType, UnitUpper>;
  static inline MatrixL getL(const MatrixType& m) { return MatrixL(m.adjoint()); }
  static inline MatrixU getU(const MatrixType& m) { return MatrixU(m); }
};

}  // end namespace internal

/** Compute / recompute the LDLT decomposition A = L D L^* = U^* D U of \a matrix
 */
template <typename MatrixType, int UpLo_>
template <typename InputType>
LDLT<MatrixType, UpLo_>& LDLT<MatrixType, UpLo_>::compute(const EigenBase<InputType>& a) {
  EIGEN_INTERNAL_ASSERT_SQUARE(a);
  const Index size = a.rows();

  m_matrix = a.derived();

  // Compute matrix L1 norm = max abs column sum over the implicit self-adjoint matrix.
  m_l1_norm = m_matrix.template selfadjointView<UpLo_>().l1Norm();

  m_transpositions.resize(size);
  m_isInitialized = false;
  m_temporary.resize(size);
  m_sign = internal::ZeroSign;

  m_info = internal::ldlt_inplace<UpLo>::blocked(m_matrix, m_transpositions, m_temporary, m_workspace, m_sign)
               ? Success
               : NumericalIssue;

  m_isInitialized = true;
  return *this;
}

/** Update the LDLT decomposition: given a decomposition of \f$ A = P^TLDL^*P \f$, efficiently compute the
 * decomposition of \f$ A + \sigma w w^* \f$.
 *
 * If \c *this holds no factorization yet, A is taken to be zero and the decomposition is built from scratch; info()
 * then reports \c Success. An update applied to an existing factorization leaves info() unchanged, so a
 * \c NumericalIssue already reported for that factorization stands until compute() replaces it or setZero() discards
 * it. LLT::rankUpdate() differs on both counts: it requires an existing factorization and re-reports the status on
 * every call.
 *
 * \note rcond(), isPositive() and isNegative() are not maintained across rank updates. rcond() keeps using the L1
 * norm recorded by the last compute(), which is zero when there was none, and the definiteness flags are assigned
 * only where this function builds a factorization from scratch, from the sign of \a sigma alone.
 *
 * \param w a vector to be incorporated into the decomposition.
 * \param sigma a scalar, +1 for updates and -1 for "downdates," which correspond to removing previously-added column
 * vectors. Optional; default value is +1.
 *
 * \sa setZero()
 */
template <typename MatrixType, int UpLo_>
template <typename Derived>
LDLT<MatrixType, UpLo_>& LDLT<MatrixType, UpLo_>::rankUpdate(
    const MatrixBase<Derived>& w, const typename LDLT<MatrixType, UpLo_>::RealScalar& sigma) {
  using IndexType = typename TranspositionType::StorageIndex;
  const Index size = w.rows();
  if (m_isInitialized) {
    EIGEN_INTERNAL_ASSERT_SAME_ROWS(m_matrix, w);
  } else {
    m_matrix.resize(size, size);
    m_matrix.setZero();
    m_transpositions.resize(size);
    for (Index i = 0; i < size; i++) m_transpositions.coeffRef(i) = IndexType(i);
    m_temporary.resize(size);
    m_sign = sigma >= 0 ? internal::PositiveSemiDef : internal::NegativeSemiDef;
    m_isInitialized = true;
    // Assigned here rather than after the branch: updating an existing factorization keeps the status it already has.
    m_info = Success;
  }

  internal::ldlt_inplace<UpLo>::update(m_matrix, m_transpositions, m_temporary, w, sigma);

  return *this;
}

// A = P^T L D L^* P with L unit lower triangular and D real diagonal, so det(A) = prod(D_ii).

template <typename MatrixType_, int UpLo_>
typename LDLT<MatrixType_, UpLo_>::Scalar LDLT<MatrixType_, UpLo_>::determinant() const {
  eigen_assert(m_isInitialized && "LDLT is not initialized.");
  eigen_assert(m_info == Success && "LDLT failed because of a zero pivot.");
  return Scalar(vectorD().real().prod());
}

template <typename MatrixType_, int UpLo_>
typename LDLT<MatrixType_, UpLo_>::RealScalar LDLT<MatrixType_, UpLo_>::absDeterminant() const {
  eigen_assert(m_isInitialized && "LDLT is not initialized.");
  eigen_assert(m_info == Success && "LDLT failed because of a zero pivot.");
  return numext::abs(vectorD().real().prod());
}

template <typename MatrixType_, int UpLo_>
typename LDLT<MatrixType_, UpLo_>::RealScalar LDLT<MatrixType_, UpLo_>::logAbsDeterminant() const {
  eigen_assert(m_isInitialized && "LDLT is not initialized.");
  eigen_assert(m_info == Success && "LDLT failed because of a zero pivot.");
  return vectorD().real().cwiseAbs().array().log().sum();
}

template <typename MatrixType_, int UpLo_>
typename LDLT<MatrixType_, UpLo_>::Scalar LDLT<MatrixType_, UpLo_>::signDeterminant() const {
  eigen_assert(m_isInitialized && "LDLT is not initialized.");
  eigen_assert(m_info == Success && "LDLT failed because of a zero pivot.");
  return Scalar(vectorD().real().array().sign().prod());
}

#ifndef EIGEN_PARSED_BY_DOXYGEN
template <typename MatrixType_, int UpLo_>
template <typename RhsType, typename DstType>
void LDLT<MatrixType_, UpLo_>::_solve_impl(const RhsType& rhs, DstType& dst) const {
  _solve_impl_transposed<true>(rhs, dst);
}

template <typename MatrixType_, int UpLo_>
template <bool Conjugate, typename RhsType, typename DstType>
void LDLT<MatrixType_, UpLo_>::_solve_impl_transposed(const RhsType& rhs, DstType& dst) const {
  // dst = P b
  dst = m_transpositions * rhs;

  // dst = L^-1 (P b)
  // dst = L^-*T (P b)
  matrixL().template conjugateIf<!Conjugate>().solveInPlace(dst);

  // dst = D^-* (L^-1 P b)
  // dst = D^-1 (L^-*T P b)
  // more precisely, use pseudo-inverse of D (see bug 241)
  const typename Diagonal<const MatrixType>::RealReturnType vecD(vectorD());
  // In some previous versions, tolerance was set to the max of 1/highest (or rather numeric_limits::min())
  // and the maximal diagonal entry * epsilon as motivated by LAPACK's xGELSS:
  // RealScalar tolerance = numext::maxi(vecD.array().abs().maxCoeff() * NumTraits<RealScalar>::epsilon(),RealScalar(1)
  // / NumTraits<RealScalar>::highest()); However, LDLT is not rank revealing, and so adjusting the tolerance wrt to the
  // highest diagonal element is not well justified and leads to numerical issues in some cases. Moreover, Lapack's
  // xSYTRS routines use 0 for the tolerance. Using numeric_limits::min() gives us more robustness to denormals.
  RealScalar tolerance = (std::numeric_limits<RealScalar>::min)();
  for (Index i = 0; i < vecD.size(); ++i) {
    if (numext::abs(vecD(i)) > tolerance)
      dst.row(i) /= vecD(i);
    else
      dst.row(i).setZero();
  }

  // dst = L^-* (D^-* L^-1 P b)
  // dst = L^-T (D^-1 L^-*T P b)
  matrixL().transpose().template conjugateIf<Conjugate>().solveInPlace(dst);

  // dst = P^T (L^-* D^-* L^-1 P b) = A^-1 b
  // dst = P^-T (L^-T D^-1 L^-*T P b) = A^-1 b
  dst = m_transpositions.transpose() * dst;
}
#endif

/** \internal use x = ldlt_object.solve(x);
 *
 * This is the \em in-place version of solve().
 *
 * \param bAndX represents both the right-hand side matrix b and result x.
 *
 * \returns true always! If you need to check for existence of solutions, use another decomposition like LU, QR, or SVD.
 *
 * This version avoids a copy when the right hand side matrix b is not
 * needed anymore.
 *
 * \sa LDLT::solve(), MatrixBase::ldlt()
 */
template <typename MatrixType, int UpLo_>
template <typename Derived>
bool LDLT<MatrixType, UpLo_>::solveInPlace(MatrixBase<Derived>& bAndX) const {
  eigen_assert(m_isInitialized && "LDLT is not initialized.");
  EIGEN_INTERNAL_ASSERT_SAME_ROWS(m_matrix, bAndX);

  bAndX = this->solve(bAndX);

  return true;
}

/** \returns the matrix represented by the decomposition,
 * i.e., it returns the product: P^T L D L^* P.
 * This function is provided for debug purpose. */
template <typename MatrixType, int UpLo_>
MatrixType LDLT<MatrixType, UpLo_>::reconstructedMatrix() const {
  eigen_assert(m_isInitialized && "LDLT is not initialized.");
  MatrixType res = matrixU();
  res = vectorD().real().asDiagonal() * res;
  res = matrixL() * res;
  // P^T (L D L^*) P; the stored pivots cannot alias the new result.
  auto lower_triangle = res.template selfadjointView<Lower>();
  for (Index k = m_matrix.rows() - 1; k >= 0; --k) lower_triangle.applyTransposition(k, m_transpositions.coeff(k));
  res.template triangularView<StrictlyUpper>() = res.adjoint();

  return res;
}

/** \cholesky_module
 * \returns the Cholesky decomposition with full pivoting without square root of \c *this
 * \sa MatrixBase::ldlt()
 */
template <typename MatrixType, unsigned int UpLo>
inline LDLT<typename SelfAdjointView<MatrixType, UpLo>::PlainObject, UpLo> SelfAdjointView<MatrixType, UpLo>::ldlt()
    const {
  return LDLT<PlainObject, UpLo>(m_matrix);
}

/** \cholesky_module
 * \returns the Cholesky decomposition with full pivoting without square root of \c *this
 * \sa SelfAdjointView::ldlt()
 */
template <typename Derived>
inline LDLT<typename MatrixBase<Derived>::PlainObject> MatrixBase<Derived>::ldlt() const {
  return LDLT<PlainObject>(derived());
}

}  // end namespace Eigen

#endif  // EIGEN_LDLT_H
