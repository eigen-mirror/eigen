// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
//
// References:
//  [1] J. J. Dongarra, J. R. Bunch, C. B. Moler and G. W. Stewart, "LINPACK
//      Users' Guide", SIAM, 1979. determinant()'s balanced accumulation follows
//      the convention of its xGEDI routines, which return determinants as a
//      (fraction, exponent) pair to avoid spurious overflow/underflow.
//  [2] P. H. Sterbenz, "Floating-Point Computation", Prentice-Hall, 1974.
//      Scaling by a power of two is exact, the property the balanced
//      accumulation relies on.

#ifndef EIGEN_STRUCTURED_CAUCHY_H
#define EIGEN_STRUCTURED_CAUCHY_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename Scalar_, int Rows_ = Dynamic, int Cols_ = Dynamic>
class Cauchy;

template <typename Scalar_>
class CauchyLU;

namespace internal {

template <typename Scalar_, int Rows_, int Cols_>
struct traits<Cauchy<Scalar_, Rows_, Cols_>> {
  using Scalar = Scalar_;
  using StorageKind = Dense;
  using XprKind = MatrixXpr;
  using StorageIndex = int;
  static constexpr int RowsAtCompileTime = Rows_;
  static constexpr int ColsAtCompileTime = Cols_;
  static constexpr int MaxRowsAtCompileTime = Rows_;
  static constexpr int MaxColsAtCompileTime = Cols_;
  // Deliberately no NestByRefBit: transpose(), conjugate() and adjoint() return
  // owning temporaries, so Product must nest the operator by value for a
  // delayed-evaluated product expression to keep its left factor alive. The copy
  // is O(m+n), negligible against the O(mn) product evaluation.
  static constexpr unsigned int Flags = 0;
};

template <typename Scalar_, int Rows_, int Cols_>
struct evaluator_traits<Cauchy<Scalar_, Rows_, Cols_>> {
  using Kind = IndexBased;
  using Shape = StructuredShape;
};

template <typename Scalar_>
struct traits<CauchyLU<Scalar_>> : traits<Matrix<Scalar_, Dynamic, Dynamic>> {
  using XprKind = MatrixXpr;
  using StorageKind = SolverStorage;
  using StorageIndex = int;
  using BaseTraits = traits<Matrix<Scalar_, Dynamic, Dynamic>>;
  static constexpr unsigned int Flags = BaseTraits::Flags & RowMajorBit;
  static constexpr int CoeffReadCost = Dynamic;
};

}  // namespace internal

/** \ingroup StructuredMatrices_Module
 * \class Cauchy
 * \brief An \c m x \c n Cauchy matrix represented by its two node vectors.
 *
 * A Cauchy matrix has entry \c (i,j) equal to \f$ 1/(x_i - y_j) \f$; the class
 * stores only the \c m + \c n nodes. Products are evaluated directly at O(mn)
 * operations -- the same cost as a dense product, but with O(m+n) storage and
 * without ever forming the matrix. (Fast approximate products via multipole
 * expansions are out of scope.)
 *
 * The class is closed under transposition: \f$ C(x,y)^T = C(-y,-x) \f$, and the
 * determinant of a square Cauchy matrix has the classical closed form
 * \f$ \prod_{i<j}(x_j-x_i)(y_i-y_j) \big/ \prod_{i,j}(x_i-y_j) \f$.
 *
 * Because \c operator* returns an Eigen product expression, a \c Cauchy also
 * drops into the matrix-free iterative solvers, and it can be assigned to a
 * dense matrix when an explicit representation is needed. As with any
 * matrix-free operator, the iterative solvers must be instantiated with
 * \c IdentityPreconditioner (e.g.
 * \c GMRES<Cauchy<double>,IdentityPreconditioner>): the default preconditioners
 * read individual coefficients through \c col() or \c InnerIterator, which the
 * structured operators do not expose.
 *
 * Square systems are solved in O(n^2) by \ref CauchyLU, a partially pivoted LU
 * factorization computed through the displacement structure
 * (Gohberg-Kailath-Olshevsky): unlike the Toeplitz/Levinson world, Cauchy
 * structure survives row permutations, which is what makes fast \em pivoted
 * factorization possible. The Hilbert matrix is the Cauchy matrix with
 * \c x_i = i+1, \c y_j = -j.
 *
 * All nodes \c x_i must differ from all nodes \c y_j (entries would be infinite
 * otherwise); this is not checked.
 *
 * \tparam Scalar_ the scalar type, real or complex.
 * \tparam Rows_ the number of rows at compile time, or \c Dynamic (the default).
 * \tparam Cols_ the number of columns at compile time, or \c Dynamic (the default).
 *
 * \sa class CauchyLU, makeCauchy()
 */
template <typename Scalar_, int Rows_, int Cols_>
class Cauchy : public EigenBase<Cauchy<Scalar_, Rows_, Cols_>> {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using StorageIndex = int;
  using RowNodeVector = Matrix<Scalar, Rows_, 1>;
  using ColNodeVector = Matrix<Scalar, Cols_, 1>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(bool(RowNodeVector::NeedsToAlign || ColNodeVector::NeedsToAlign))

  static constexpr int RowsAtCompileTime = Rows_;
  static constexpr int ColsAtCompileTime = Cols_;
  static constexpr int MaxRowsAtCompileTime = Rows_;
  static constexpr int MaxColsAtCompileTime = Cols_;
  static constexpr int SizeAtCompileTime = internal::size_at_compile_time(Rows_, Cols_);
  static constexpr int MaxSizeAtCompileTime = SizeAtCompileTime;
  static constexpr bool IsRowMajor = false;
  // Deliberately no IsVectorAtCompileTime: Ref<const Cauchy>'s default StrideType
  // argument reads it, so its absence makes internal::is_ref_compatible SFINAE to
  // false and keeps the iterative solvers on their matrix-free path.

  /** Builds a Cauchy matrix from the row nodes \a x and the column nodes \a y:
   * entry \c (i,j) is \c 1/(x[i] - y[j]). */
  template <typename XDerived, typename YDerived>
  Cauchy(const MatrixBase<XDerived>& x, const MatrixBase<YDerived>& y) : m_x(x), m_y(y) {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(XDerived)
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(YDerived)
    eigen_assert(m_x.size() > 0 && m_y.size() > 0 && "Cauchy node vectors must be non-empty");
  }

  EIGEN_DEVICE_FUNC Index rows() const { return m_x.size(); }
  EIGEN_DEVICE_FUNC Index cols() const { return m_y.size(); }

  /** \returns the row node vector \c x. */
  const RowNodeVector& rowNodes() const { return m_x; }
  /** \returns the column node vector \c y. */
  const ColNodeVector& colNodes() const { return m_y; }

  /** \returns the coefficient at row \a row and column \a col. */
  Scalar coeff(Index row, Index col) const { return Scalar(1) / (m_x.coeff(row) - m_y.coeff(col)); }

  /** \returns the transpose of \c *this, itself a Cauchy operator:
   * \f$ C(x,y)^T = C(-y,-x) \f$. */
  Cauchy<Scalar, Cols_, Rows_> transpose() const { return Cauchy<Scalar, Cols_, Rows_>(-m_y, -m_x); }

  /** \returns the complex conjugate of \c *this, itself a Cauchy operator (the
   * one with conjugated nodes). */
  Cauchy conjugate() const { return Cauchy(m_x.conjugate(), m_y.conjugate()); }

  /** \returns the adjoint of \c *this, itself a Cauchy operator:
   * \f$ C(x,y)^H = C(-\bar y, -\bar x) \f$. */
  Cauchy<Scalar, Cols_, Rows_> adjoint() const {
    return Cauchy<Scalar, Cols_, Rows_>(-m_y.conjugate(), -m_x.conjugate());
  }

  /** \returns the determinant of a \b square Cauchy matrix through the classical
   * closed form \f$ \prod_{i<j}(x_j-x_i)(y_i-y_j) \big/ \prod_{i,j}(x_i-y_j) \f$,
   * in O(n^2) operations. All factors enter a single accumulation kept in the
   * balanced form \c m * 2^e (the split fraction/exponent determinant
   * convention of LINPACK's xGEDI [1]) -- every factor and the running value
   * are renormalized to unit magnitude with the power of two tracked separately
   * (exact frexp/ldexp rescaling [2]), numerator factors multiplied in and
   * denominator factors divided out -- so no partial product can overflow or
   * underflow when the determinant itself is representable. Zero factors
   * (coincident \c x or \c y nodes) and non-finite factors propagate exactly;
   * a node difference that overflows makes its factor infinite, which turns the
   * determinant into NaN when it meets a zero or another infinite factor. */
  Scalar determinant() const {
    eigen_assert(rows() == cols() && "Cauchy::determinant requires a square matrix");
    const Index n = rows();
    Scalar det(1);
    Index exponent = 0;
    for (Index j = 1; j < n; ++j)
      for (Index i = 0; i < j; ++i) {
        const Scalar xDiff = internal::structured_balance(Scalar(m_x.coeff(j) - m_x.coeff(i)), exponent);
        det = internal::structured_balance(Scalar(det * xDiff), exponent);
        const Scalar yDiff = internal::structured_balance(Scalar(m_y.coeff(i) - m_y.coeff(j)), exponent);
        det = internal::structured_balance(Scalar(det * yDiff), exponent);
      }
    for (Index j = 0; j < n; ++j)
      for (Index i = 0; i < n; ++i) {
        Index denomExponent = 0;
        const Scalar d = internal::structured_balance(Scalar(m_x.coeff(i) - m_y.coeff(j)), denomExponent);
        exponent -= denomExponent;
        det = internal::structured_balance(Scalar(det / d), exponent);
      }
    return internal::structured_ldexp_clamped(det, exponent);
  }

  /** \internal Writes the dense representation into \a dst, one vectorized
   * column at a time. Invoked through \c dense = cauchy; */
  template <typename Dest>
  void evalTo(Dest& dst) const {
    applyAssignment(dst, internal::assign_op<typename Dest::Scalar, Scalar>());
  }

  /** \internal Computes \c dst += (*this), see evalTo(). */
  template <typename Dest>
  void addTo(Dest& dst) const {
    applyAssignment(dst, internal::add_assign_op<typename Dest::Scalar, Scalar>());
  }

  /** \internal Computes \c dst -= (*this), see evalTo(). */
  template <typename Dest>
  void subTo(Dest& dst) const {
    applyAssignment(dst, internal::sub_assign_op<typename Dest::Scalar, Scalar>());
  }

  /** \returns the product expression \c (*this) * \a v, evaluated directly at
   * O(mn) operations per right-hand side and at most O(m) extra storage. The
   * expression carries the default product tag, so assigning it behaves like any
   * dense product: a temporary resolves aliasing between the destination and
   * \a v, and \c .noalias() skips it. */
  template <typename Rhs>
  Product<Cauchy, Rhs> operator*(const MatrixBase<Rhs>& v) const {
    EIGEN_STATIC_ASSERT(ColsAtCompileTime == Dynamic || Rhs::RowsAtCompileTime == Dynamic ||
                            int(ColsAtCompileTime) == int(Rhs::RowsAtCompileTime),
                        INVALID_MATRIX_PRODUCT)
    eigen_assert(v.rows() == cols() && "invalid product: dimensions do not match");
    return Product<Cauchy, Rhs>(*this, v.derived());
  }

  /** \internal Computes \c dst += alpha * (*this) * rhs. \c ProductScalar is the
   * promoted scalar of the product (complex when a real operator is applied to a
   * complex right-hand side); the accumulation runs in the promoted type. Each
   * column of the operator is formed on the fly, vectorized over the rows; for
   * several right-hand sides of the operator's own scalar type it is formed once
   * per block of rows and reused across them, with identical results. */
  template <typename Dest, typename Rhs, typename ProductScalar>
  void addProduct(Dest& dst, const Rhs& rhs, const ProductScalar& alpha) const {
    const Index n = cols();
    eigen_assert(rhs.rows() == n && "invalid product: dimensions do not match");
    // A unit alpha must not multiply: even the identity complex scalar (1,0)
    // pollutes an (Inf,0) value with NaN through the 0*Inf cross term.
    const bool unitAlpha = alpha == ProductScalar(1);
    const auto weight = [&](Index j, Index k) {
      return unitAlpha ? ProductScalar(rhs.coeff(j, k)) : ProductScalar(alpha * rhs.coeff(j, k));
    };
    // Reuse saves (r-1)/r of the divisions. A real operator applied to a complex
    // right-hand side gains nothing: the complex updates dominate its real divisions.
    if (rhs.cols() > 1 && std::is_same<ProductScalar, Scalar>::value) {
      // Row blocks keep the r destination columns of a block in cache across all
      // n reciprocal columns; the floor keeps the per-column updates long.
      constexpr Index kBlockBytes = Index(1) << 19;
      constexpr Index kMinBlockRows = 1024;
      const Index m = rows(), r = rhs.cols();
      const Index blockRows = numext::mini(m, numext::maxi(kMinBlockRows, kBlockBytes / (r * Index(sizeof(Scalar)))));
      Matrix<Scalar, Dynamic, 1, ColMajor, Rows_> reciprocals(blockRows);
      for (Index i = 0; i < m; i += blockRows) {
        const Index b = numext::mini(blockRows, m - i);
        for (Index j = 0; j < n; ++j) {
          reciprocals.head(b) = (m_x.segment(i, b).array() - m_y.coeff(j)).inverse();
          for (Index k = 0; k < r; ++k) dst.col(k).segment(i, b) += weight(j, k) * reciprocals.head(b);
        }
      }
      return;
    }
    for (Index k = 0; k < rhs.cols(); ++k)
      for (Index j = 0; j < n; ++j) dst.col(k) += weight(j, k) * (m_x.array() - m_y.coeff(j)).inverse().matrix();
  }

 private:
  template <typename Dest, typename Assignment>
  void applyAssignment(Dest& dst, const Assignment& assignment) const {
    for (Index j = 0; j < cols(); ++j) {
      auto dstColumn = dst.col(j);
      internal::call_assignment_no_alias(dstColumn, (m_x.array() - m_y.coeff(j)).inverse().matrix(), assignment);
    }
  }

  RowNodeVector m_x;
  ColNodeVector m_y;
};

/** \ingroup StructuredMatrices_Module
 * \returns a \ref Cauchy operator with row nodes \a x and column nodes \a y. The
 * compile-time dimensions of the operator are deduced from the node vectors. */
template <typename XDerived, typename YDerived>
Cauchy<typename XDerived::Scalar, XDerived::SizeAtCompileTime, YDerived::SizeAtCompileTime> makeCauchy(
    const MatrixBase<XDerived>& x, const MatrixBase<YDerived>& y) {
  return Cauchy<typename XDerived::Scalar, XDerived::SizeAtCompileTime, YDerived::SizeAtCompileTime>(x, y);
}

/** \ingroup StructuredMatrices_Module
 * \class CauchyLU
 * \brief Partially pivoted O(n^2) LU solver for square Cauchy systems
 * (Gohberg-Kailath-Olshevsky).
 *
 * A Cauchy matrix satisfies the displacement equation
 * \f$ D_x C - C D_y = \mathbf{1}\mathbf{1}^T \f$, and -- crucially -- this
 * structure survives row permutations. The GKO algorithm exploits it to compute
 * the row-pivoted factorization \c P*C = L*U in O(n^2) operations: at each step
 * the current column of the Schur complement is generated from O(n) data, the
 * largest entry is chosen as pivot, and the Schur complement stays Cauchy-like
 * with updated generators. This gives Cauchy systems what the Levinson world
 * lacks: genuine partial pivoting at fast-algorithm cost.
 *
 * Usage follows the usual decomposition style, including the transposed and
 * adjoint solves of \c SolverBase (from the same factorization):
 * \code
 *   CauchyLU<double> lu(C);              // or lu.compute(C);
 *   VectorXd u = lu.solve(b);             // solve C   * u = b
 *   VectorXd v = lu.transpose().solve(b); // solve C^T * v = b
 *   VectorXd w = lu.adjoint().solve(b);   // solve C^H * w = b
 * \endcode
 *
 * The factors are stored densely (O(n^2) memory, like the dense LU
 * decompositions).
 *
 * References: I. Gohberg, T. Kailath, V. Olshevsky, "Fast Gaussian elimination
 * with partial pivoting for matrices with displacement structure," Math. Comp.
 * 64 (1995).
 *
 * \tparam Scalar_ the scalar type, real or complex.
 *
 * \sa class Cauchy
 */
template <typename Scalar_>
class CauchyLU : public SolverBase<CauchyLU<Scalar_>> {
 public:
  using Base = SolverBase<CauchyLU>;
  friend class SolverBase<CauchyLU>;
  EIGEN_GENERIC_PUBLIC_INTERFACE(CauchyLU)
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;

  /** Default constructor; call \ref compute before \ref solve. */
  CauchyLU() : m_isInitialized(false), m_info(InvalidInput) {}

  /** Constructs and factorizes from the square Cauchy matrix \a C. */
  template <int Rows_, int Cols_>
  explicit CauchyLU(const Cauchy<Scalar, Rows_, Cols_>& C) : m_isInitialized(false), m_info(InvalidInput) {
    compute(C);
  }

  /** Factorizes the square Cauchy matrix \a C as \c P*C = L*U by the GKO
   * recursion with partial pivoting. The Schur complement retains the form
   * \f[ S_{ij}=\frac{a_i b_j}{x_i-y_j}, \qquad
   * a_i\leftarrow a_i\frac{x_i-x_k}{x_i-y_k},\quad
   * b_j\leftarrow b_j\frac{y_j-y_k}{y_j-x_k}. \f]
   * This follows from the rank-one displacement
   * \f$D_x C-C D_y=\mathbf{1}\mathbf{1}^T\f$. \sa solve */
  template <int Rows_, int Cols_>
  CauchyLU& compute(const Cauchy<Scalar, Rows_, Cols_>& C) {
    eigen_assert(C.rows() == C.cols() && "CauchyLU requires a square Cauchy matrix");
    const Index n = C.rows();
    DenseVector x = C.rowNodes();
    const DenseVector y = C.colNodes();
    DenseVector a = DenseVector::Ones(n);
    DenseVector b = DenseVector::Ones(n);
    m_lu.resize(n, n);
    m_perm.resize(static_cast<std::size_t>(n));
    m_info = Success;

    for (Index k = 0; k < n; ++k) {
      Index piv = k;
      RealScalar best(-1);
      for (Index i = k; i < n; ++i) {
        m_lu(i, k) = a[i] * b[k] / (x[i] - y[k]);
        const RealScalar mag = numext::abs(m_lu(i, k));
        if ((numext::isnan)(mag) || mag > best) {
          best = mag;
          piv = i;
        }
      }
      if (piv != k) {
        m_lu.row(piv).head(k + 1).swap(m_lu.row(k).head(k + 1));
        std::swap(x[piv], x[k]);
        std::swap(a[piv], a[k]);
      }
      m_perm[static_cast<std::size_t>(k)] = piv;
      const Scalar pivot = m_lu(k, k);
      if (pivot == Scalar(0) || !(numext::isfinite)(pivot)) {
        m_info = NumericalIssue;
        m_lu.row(k).tail(n - k - 1).setZero();
        m_lu.col(k).tail(n - k - 1).setZero();
        continue;
      }
      m_lu.col(k).tail(n - k - 1) /= pivot;
      for (Index j = k + 1; j < n; ++j) {
        m_lu(k, j) = a[k] * b[j] / (x[k] - y[j]);
        if (!(numext::isfinite)(m_lu(k, j))) m_info = NumericalIssue;
      }
      for (Index i = k + 1; i < n; ++i) a[i] *= (x[i] - x[k]) / (x[i] - y[k]);
      for (Index j = k + 1; j < n; ++j) b[j] *= (y[j] - y[k]) / (y[j] - x[k]);
    }
    m_isInitialized = true;
    return *this;
  }

  Index rows() const noexcept { return m_lu.rows(); }
  Index cols() const noexcept { return m_lu.cols(); }

  /** \returns \c Success, or \c NumericalIssue when the factorization encounters
   * a zero or non-finite entry that prevents a usable factorization. */
  ComputationInfo info() const {
    eigen_assert(m_isInitialized && "CauchyLU is not initialized.");
    return m_info;
  }

#ifdef EIGEN_PARSED_BY_DOXYGEN
  /** \returns the solution \c u of \c C*u = \a b, as a lazily evaluated
   * expression. Supports multiple right-hand sides. The transposed and adjoint
   * systems reuse the same factorization through \c transpose().solve(b) and
   * \c adjoint().solve(b). \pre \ref compute has been called. */
  template <typename Rhs>
  inline const Solve<CauchyLU, Rhs> solve(const MatrixBase<Rhs>& b) const;
#endif

#ifndef EIGEN_PARSED_BY_DOXYGEN
  /** \internal P*C = L*U, so C*u = b becomes L*U*u = P*b. */
  template <typename RhsType, typename DstType>
  void _solve_impl(const RhsType& rhs, DstType& dst) const {
    dst = rhs;
    for (Index k = 0; k < rows(); ++k) {
      const Index piv = m_perm[static_cast<std::size_t>(k)];
      if (piv != k) dst.row(k).swap(dst.row(piv));
    }
    m_lu.template triangularView<UnitLower>().solveInPlace(dst);
    m_lu.template triangularView<Upper>().solveInPlace(dst);
  }

  /** \internal C^T = U^T L^T P, so C^T*w = b becomes U^T (L^T (P w)) = b: solve
   * the transposed triangles, then undo the transpositions in reverse order;
   * conjugated on the way in and out for the adjoint. */
  template <bool Conjugate, typename RhsType, typename DstType>
  void _solve_impl_transposed(const RhsType& rhs, DstType& dst) const {
    dst = rhs.template conjugateIf<Conjugate>();
    m_lu.template triangularView<Upper>().transpose().solveInPlace(dst);
    m_lu.template triangularView<UnitLower>().transpose().solveInPlace(dst);
    for (Index k = rows() - 1; k >= 0; --k) {
      const Index piv = m_perm[static_cast<std::size_t>(k)];
      if (piv != k) dst.row(k).swap(dst.row(piv));
    }
    if (Conjugate) dst = dst.conjugate().eval();
  }
#endif

 private:
  DenseMatrix m_lu;
  std::vector<Index> m_perm;  // transposition applied at each elimination step
  bool m_isInitialized;
  ComputationInfo m_info;
};

namespace internal {

// Single product specialization covering every product tag; see the note in
// Circulant.h.
template <typename Scalar_, int Rows_, int Cols_, typename Rhs, int ProductTag>
struct generic_product_impl<Cauchy<Scalar_, Rows_, Cols_>, Rhs, StructuredShape, DenseShape, ProductTag>
    : structured_product_impl<Cauchy<Scalar_, Rows_, Cols_>, Rhs> {};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_STRUCTURED_CAUCHY_H
