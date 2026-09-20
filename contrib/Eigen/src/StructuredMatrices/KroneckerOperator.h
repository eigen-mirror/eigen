// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// References:
//  [1] C. F. Van Loan, "The ubiquitous Kronecker product", Journal of
//      Computational and Applied Mathematics 123 (2000), 85-100. The vec
//      identity driving every product and solve below -- stated there as
//      Y = C X B^T <=> vec(Y) = (B (x) C) vec(X), i.e. with this file's
//      operand naming (A (x) B) vec(X) = vec(B X A^T) -- together with the
//      factor-wise identities for the inverse, pseudo-inverse,
//      eigendecomposition, SVD and determinant of a Kronecker product.
//  [2] N. J. Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed.,
//      SIAM, 2002, chapter 27. Avoiding spurious overflow by rescaling with
//      powers of two, the technique behind the scaled vec-trick products, the
//      normalized-frame solves, the exponent-split pseudo-inversion and the
//      exponent-balanced determinant.
//  [3] P. H. Sterbenz, "Floating-Point Computation", Prentice-Hall, 1974.
//      Scaling by a power of two is exact, the property every rescaling in this
//      file relies on.

#ifndef EIGEN_STRUCTURED_KRONECKER_OPERATOR_H
#define EIGEN_STRUCTURED_KRONECKER_OPERATOR_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename LhsMatrix, typename RhsMatrix>
class KroneckerOperator;

namespace internal {

template <typename LhsMatrix, typename RhsMatrix>
struct traits<KroneckerOperator<LhsMatrix, RhsMatrix>> {
  using Scalar = typename LhsMatrix::Scalar;
  using StorageKind = Dense;
  using XprKind = MatrixXpr;
  using StorageIndex = int;
  // size_at_compile_time is the compile-time dimension product, Dynamic when a
  // factor is unknown or the product would overflow int.
  static constexpr int RowsAtCompileTime =
      size_at_compile_time(traits<LhsMatrix>::RowsAtCompileTime, traits<RhsMatrix>::RowsAtCompileTime);
  static constexpr int ColsAtCompileTime =
      size_at_compile_time(traits<LhsMatrix>::ColsAtCompileTime, traits<RhsMatrix>::ColsAtCompileTime);
  static constexpr int MaxRowsAtCompileTime = RowsAtCompileTime;
  static constexpr int MaxColsAtCompileTime = ColsAtCompileTime;
  // Deliberately no NestByRefBit: transpose(), conjugate(), adjoint(), inverse(),
  // eigenvectors(), matrixU() and matrixV() return owning temporaries (the
  // operator stores its factors by value), so Product must nest the operator by
  // value for a delayed-evaluated product expression to keep its left factor
  // alive. The copy is O(m1 n1 + m2 n2), negligible against the product
  // evaluation.
  static constexpr int Flags = 0;
};

template <typename LhsMatrix, typename RhsMatrix>
struct evaluator_traits<KroneckerOperator<LhsMatrix, RhsMatrix>> {
  using Kind = IndexBased;
  using Shape = StructuredShape;
};

// Three factor kinds -- dense, diagonal, sparse -- with every kind-specific
// operation in the kron_factor_* helpers below, so the operator has a single
// implementation. Diagonal: stored as its diagonal, applied as a scaling, solved
// entrywise, closed under transposition, inverse and determinant. Sparse: stored
// compressed, applied by sparse-dense products, solved via SparseLU; inverse dense.

template <typename Factor>
struct kron_factor_is_diagonal : std::false_type {};
template <typename Scalar, int Size, int MaxSize>
struct kron_factor_is_diagonal<DiagonalMatrix<Scalar, Size, MaxSize>> : std::true_type {};

template <typename Factor>
struct kron_factor_is_dense_matrix : std::false_type {};
template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct kron_factor_is_dense_matrix<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> : std::true_type {};

template <typename Factor>
struct kron_factor_is_sparse_matrix : std::false_type {};
template <typename Scalar, int Options, typename StorageIndex>
struct kron_factor_is_sparse_matrix<SparseMatrix<Scalar, Options, StorageIndex>> : std::true_type {};

// The factor kind, the dispatch key of kron_factor_ops and kron_factor_solver.
constexpr int kKronDenseFactor = 0;
constexpr int kKronDiagonalFactor = 1;
constexpr int kKronSparseFactor = 2;

template <typename Factor>
constexpr int kron_factor_kind() {
  return kron_factor_is_diagonal<Factor>::value        ? kKronDiagonalFactor
         : kron_factor_is_sparse_matrix<Factor>::value ? kKronSparseFactor
                                                       : kKronDenseFactor;
}

// Apply 2^e without forming a possibly unrepresentable scale factor. For
// std::complex storage, realView exposes both components to the real ldexp packets.
template <typename Xpr, std::enable_if_t<!NumTraits<typename Xpr::Scalar>::IsComplex, bool> = true>
void kron_ldexp_entries(Xpr& M, int e) {
  M.array() = M.array().ldexp(e);
}

template <typename Xpr, std::enable_if_t<complex_array_access<typename Xpr::Scalar>::value, bool> = true>
void kron_ldexp_entries(Xpr& M, int e) {
  M.realView().array() = M.realView().array().ldexp(e);
}

template <typename Xpr, std::enable_if_t<NumTraits<typename Xpr::Scalar>::IsComplex &&
                                             !complex_array_access<typename Xpr::Scalar>::value,
                                         bool> = true>
void kron_ldexp_entries(Xpr& M, int e) {
  using Scalar = typename Xpr::Scalar;
  M = M.unaryExpr([e](const Scalar& z) { return structured_ldexp_clamped(z, Index(e)); });
}

template <typename Factor, int Kind = kron_factor_kind<Factor>()>
struct kron_factor_ops {
  // Dense factor.
  using Scalar = typename Factor::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using TransposedFactor = Matrix<Scalar, Factor::ColsAtCompileTime, Factor::RowsAtCompileTime>;
  using InverseFactor = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  // Every entry is stored: a materialization has no structurally zero blocks
  // to clear, and forEachNonZero visits all m*n entries in column-major order.
  static constexpr bool StoresAllEntries = true;

  static void prepare(Factor&) {}
  static bool coeffIfStored(const Factor& f, Index row, Index col, Scalar& value) {
    value = f.coeff(row, col);
    return true;
  }
  template <typename Visitor>
  static void forEachNonZero(const Factor& f, Visitor&& visit) {
    for (Index j = 0; j < f.cols(); ++j)
      for (Index i = 0; i < f.rows(); ++i) visit(i, j, f.coeff(i, j));
  }
  /** \internal \returns the number of stored entries of each inner vector, for a
   * destination of the storage order selected by \a rowMajor. */
  static Matrix<Index, Dynamic, 1> innerNonZeros(const Factor& f, bool rowMajor) {
    return Matrix<Index, Dynamic, 1>::Constant(rowMajor ? f.rows() : f.cols(), rowMajor ? f.cols() : f.rows());
  }
  static auto transposed(const Factor& f) { return f.transpose(); }
  static auto conjugated(const Factor& f) { return f.conjugate(); }
  static auto adjointed(const Factor& f) { return f.adjoint(); }
  static auto inversed(const Factor& f) { return f.inverse(); }
  // The factor as a dense expression, for the decomposition family.
  static const Factor& denseFactor(const Factor& f) { return f; }
  // The right operand op(F) of the vec-trick product Y = B X op(A): the
  // transpose for a dense factor, the factor itself for a diagonal one.
  static auto transposedOperand(const Factor& f) { return f.transpose(); }
  static bool allFinite(const Factor& f) { return f.allFinite(); }
  static int exponentBound(const Factor& f) { return structured_exponent_bound(f); }
  /** \internal \returns the dot-product growth bits of a product contracting
   * over \a n entries of this factor. Cast to an unsigned type deliberately:
   * the generic count_bits_impl fallback static_asserts on unsignedness, and
   * only the GCC/Clang and MSVC specialisations accept a signed Index. */
  static int growthBits(Index n) { return log2_floor(static_cast<numext::uint64_t>(n)) + 1; }
  /** \internal \returns the mantissa of \c det(M) in the balanced form
   * \c m * 2^e, adding \c e into \a exponent: the determinant is accumulated
   * directly from the LU diagonal (times the permutation sign), each entry and
   * the running product renormalized by \c structured_balance. */
  static Scalar balancedDet(const Factor& M, Index& exponent) {
    PartialPivLU<Factor> lu(M);
    Scalar m = Scalar(RealScalar(lu.permutationP().determinant()));  // +-1
    for (Index i = 0; i < M.rows(); ++i)
      m = structured_balance(m * structured_balance(lu.matrixLU().coeff(i, i), exponent), exponent);
    return m;
  }
};

template <typename Factor>
struct kron_factor_ops<Factor, kKronDiagonalFactor> {
  // Diagonal factor: everything runs on the stored diagonal vector.
  using Scalar = typename Factor::Scalar;
  using TransposedFactor = typename Factor::PlainObject;  // a diagonal matrix is its own transpose
  using InverseFactor = typename Factor::PlainObject;     // entrywise reciprocals stay diagonal
  static constexpr bool StoresAllEntries = false;

  static void prepare(Factor&) {}
  static bool coeffIfStored(const Factor& f, Index row, Index col, Scalar& value) {
    if (row != col) return false;
    value = f.diagonal().coeff(row);
    return true;
  }
  template <typename Visitor>
  static void forEachNonZero(const Factor& f, Visitor&& visit) {
    for (Index k = 0; k < f.rows(); ++k) visit(k, k, f.diagonal().coeff(k));
  }
  static Matrix<Index, Dynamic, 1> innerNonZeros(const Factor& f, bool) {
    return Matrix<Index, Dynamic, 1>::Ones(f.rows());
  }
  static const Factor& transposed(const Factor& f) { return f; }
  static auto conjugated(const Factor& f) { return f.diagonal().conjugate().asDiagonal(); }
  static auto adjointed(const Factor& f) { return f.diagonal().conjugate().asDiagonal(); }
  static auto inversed(const Factor& f) { return f.inverse(); }
  static typename Factor::DenseMatrixType denseFactor(const Factor& f) { return f.toDenseMatrix(); }
  static const Factor& transposedOperand(const Factor& f) { return f; }
  static bool allFinite(const Factor& f) { return f.diagonal().allFinite(); }
  static int exponentBound(const Factor& f) { return structured_exponent_bound(f.diagonal()); }
  // A diagonal product has no accumulation, hence no dot-product growth.
  static int growthBits(Index) { return 0; }
  static Scalar balancedDet(const Factor& D, Index& exponent) {
    Scalar m(1);
    for (Index i = 0; i < D.rows(); ++i)
      m = structured_balance(m * structured_balance(D.diagonal().coeff(i), exponent), exponent);
    return m;
  }
};

/** \internal SparseLU extended with the balanced determinant of the dense path:
 * the U diagonal -- stored on the diagonal of the supernodal L, where
 * SparseLU::determinant() reads it -- times the permutation signs, each entry
 * and the running product renormalized by structured_balance. */
template <typename SparseType>
class kron_sparse_lu : public SparseLU<SparseType> {
 public:
  using Base = SparseLU<SparseType>;
  using Scalar = typename SparseType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;

  Scalar balancedDet(Index& exponent) const {
    eigen_assert(Base::info() == Success);
    Scalar m = Scalar(RealScalar(Base::m_detPermR * Base::m_detPermC));
    for (Index j = 0; j < Base::cols(); ++j) {
      typename Base::SCMatrix::InnerIterator it(Base::m_Lstore, j);
      while (it && it.index() != j) ++it;
      if (it) m = structured_balance(m * structured_balance(it.value(), exponent), exponent);
    }
    return m;
  }
};

template <typename Factor>
struct kron_factor_ops<Factor, kKronSparseFactor> {
  // Sparse factor: the products run through the sparse-dense kernels, the
  // solve and the determinant through a SparseLU factorization.
  using Scalar = typename Factor::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using ColMajorFactor = SparseMatrix<Scalar, ColMajor, typename Factor::StorageIndex>;
  using TransposedFactor = Factor;  // the transpose is stored sparse, in the same storage order
  using InverseFactor = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;  // dense in general
  static constexpr bool StoresAllEntries = false;

  // coeffs() below reads the compressed value array.
  static void prepare(Factor& f) { f.makeCompressed(); }
  static bool coeffIfStored(const Factor& f, Index row, Index col, Scalar& value) {
    const Index outer = Factor::IsRowMajor ? row : col;
    const Index inner = Factor::IsRowMajor ? col : row;
    const Index start = f.outerIndexPtr()[outer], end = f.outerIndexPtr()[outer + 1];
    if (start == end) return false;
    const Index position = f.data().searchLowerIndex(start, end, inner);
    if (position == end || f.innerIndexPtr()[position] != inner) return false;
    value = f.valuePtr()[position];
    return true;
  }
  template <typename Visitor>
  static void forEachNonZero(const Factor& f, Visitor&& visit) {
    for (Index k = 0; k < f.outerSize(); ++k)
      for (typename Factor::InnerIterator it(f, k); it; ++it) visit(it.row(), it.col(), it.value());
  }
  static Matrix<Index, Dynamic, 1> innerNonZeros(const Factor& f, bool rowMajor) {
    Matrix<Index, Dynamic, 1> counts = Matrix<Index, Dynamic, 1>::Zero(rowMajor ? f.rows() : f.cols());
    for (Index k = 0; k < f.outerSize(); ++k)
      for (typename Factor::InnerIterator it(f, k); it; ++it) ++counts[rowMajor ? it.row() : it.col()];
    return counts;
  }
  static auto transposed(const Factor& f) { return f.transpose(); }
  static auto conjugated(const Factor& f) { return f.conjugate(); }
  static auto adjointed(const Factor& f) { return f.adjoint(); }
  /** \internal \returns the dense inverse, one SparseLU solve against the
   * identity: O(n^2) storage and n substitutions; NaN when the factorization
   * fails, see nanMatrix(). */
  static InverseFactor inversed(const Factor& f) {
    const ColMajorFactor colMajor(f);
    SparseLU<ColMajorFactor> lu(colMajor);
    if (lu.info() != Success) return nanMatrix(f.rows(), f.cols());
    return lu.solve(InverseFactor::Identity(f.rows(), f.cols()));
  }
  /** \internal SparseLU aborts on an exactly zero or non-finite pivot column
   * and leaves no usable factors, so such a factor solves and inverts to NaN:
   * the sparse counterpart of the Inf/NaN a dense LU substitutes through. */
  static InverseFactor nanMatrix(Index rows, Index cols) {
    return InverseFactor::Constant(rows, cols, Scalar(NumTraits<RealScalar>::quiet_NaN()));
  }
  static InverseFactor denseFactor(const Factor& f) { return InverseFactor(f); }
  static auto transposedOperand(const Factor& f) { return f.transpose(); }
  static bool allFinite(const Factor& f) { return f.coeffs().allFinite(); }
  static int exponentBound(const Factor& f) { return structured_exponent_bound(f.coeffs()); }
  // The dense contraction bound: a row holds at most n stored entries.
  static int growthBits(Index n) { return log2_floor(static_cast<numext::uint64_t>(n)) + 1; }
  static Scalar balancedDet(const Factor& M, Index& exponent) {
    // Scale small factors up, exactly, so that the elimination runs on normal
    // numbers: a subnormal near 2^e carries only e - min_exponent + digits bits.
    // Scaling down could erase small pivots in factors with a wide exponent range.
    const int scaleExponent = numext::mini(exponentBound(M), 0);
    ColMajorFactor normalized(M);
    if (scaleExponent != 0) {
      auto values = normalized.coeffs();
      kron_ldexp_entries(values, -scaleExponent);
    }
    kron_sparse_lu<ColMajorFactor> lu;
    lu.compute(normalized);
    if (lu.info() == Success) {
      exponent += M.rows() * scaleExponent;
      return lu.balancedDet(exponent);
    }
    // An aborted factorization met an exactly zero pivot column -- an exactly
    // singular factor -- unless a non-finite entry defeated the pivot search.
    return M.coeffs().allFinite() ? Scalar(0) : Scalar(NumTraits<RealScalar>::quiet_NaN());
  }
};

/** \internal Per-factor solve adapter for KroneckerOperator::solve(). All
 * kinds normalize the factor by an exact power of two up front and then solve
 * per right-hand-side column in the shared normalized frame; the caller folds
 * the removed \c exponent() back together with the other factor's and the
 * column's. A dense factor is LU-factorized once, a sparse factor once by
 * SparseLU; a diagonal factor is solved by entrywise division, whose
 * intermediates are bounded by the factor's conditioning exactly like the
 * substitutions of the LU paths (and each division saturates per entry, like
 * every step of a substitution). */
template <typename Factor, int Kind = kron_factor_kind<Factor>()>
class kron_factor_solver {
 public:
  using Scalar = typename Factor::Scalar;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;

  explicit kron_factor_solver(const Factor& f) : m_exponent(structured_exponent_bound(f)) {
    DenseMatrix normalized = f;
    kron_ldexp_entries(normalized, -m_exponent);
    m_lu.compute(normalized);
  }
  int exponent() const { return m_exponent; }
  /** \internal \returns \f$ F_{norm}^{-1} M \f$. */
  template <typename Xpr>
  DenseMatrix solveLeft(const Xpr& M) const {
    return m_lu.solve(M);
  }
  /** \internal \returns \f$ M F_{norm}^{-T} \f$. */
  template <typename Xpr>
  DenseMatrix solveTransposedRight(const Xpr& M) const {
    return m_lu.solve(M.transpose()).transpose();
  }

 private:
  int m_exponent;
  PartialPivLU<DenseMatrix> m_lu;
};

template <typename Factor>
class kron_factor_solver<Factor, kKronDiagonalFactor> {
 public:
  using Scalar = typename Factor::Scalar;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;

  explicit kron_factor_solver(const Factor& f)
      : m_exponent(structured_exponent_bound(f.diagonal())), m_d(f.diagonal()) {
    kron_ldexp_entries(m_d, -m_exponent);
  }
  int exponent() const { return m_exponent; }
  template <typename Xpr>
  DenseMatrix solveLeft(const Xpr& M) const {
    return (M.array().colwise() / m_d.array()).matrix();
  }
  template <typename Xpr>
  DenseMatrix solveTransposedRight(const Xpr& M) const {
    return (M.array().rowwise() / m_d.transpose().array()).matrix();
  }

 private:
  int m_exponent;
  typename Factor::DiagonalVectorType m_d;
};

template <typename Factor>
class kron_factor_solver<Factor, kKronSparseFactor> {
 public:
  using Scalar = typename Factor::Scalar;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using ColMajorFactor = SparseMatrix<Scalar, ColMajor, typename Factor::StorageIndex>;

  explicit kron_factor_solver(const Factor& f) : m_exponent(structured_exponent_bound(f.coeffs())) {
    ColMajorFactor normalized(f);
    normalized.makeCompressed();
    if (m_exponent != 0) {
      auto values = normalized.coeffs();
      kron_ldexp_entries(values, -m_exponent);
    }
    m_lu.compute(normalized);
    m_factorized = m_lu.info() == Success;  // else the solves fill with NaN, see kron_factor_ops::nanMatrix
  }
  int exponent() const { return m_exponent; }
  template <typename Xpr>
  DenseMatrix solveLeft(const Xpr& M) const {
    if (!m_factorized) return kron_factor_ops<Factor>::nanMatrix(M.rows(), M.cols());
    return m_lu.solve(M);
  }
  template <typename Xpr>
  DenseMatrix solveTransposedRight(const Xpr& M) const {
    if (!m_factorized) return kron_factor_ops<Factor>::nanMatrix(M.rows(), M.cols());
    // SparseLU solves into column-major storage only, which a transposed Solve
    // expression would not evaluate into; the right-hand side stays a view.
    const DenseMatrix Xt = m_lu.solve(M.transpose());
    return Xt.transpose();
  }

 private:
  int m_exponent;
  bool m_factorized;
  SparseLU<ColMajorFactor> m_lu;
};

}  // namespace internal

/** \ingroup StructuredMatrices_Module
 * \class KroneckerOperator
 * \brief The Kronecker product \f$ A \otimes B \f$ as an implicit operator that is
 * never materialized.
 *
 * For \c A of size \c m1 x \c n1 and \c B of size \c m2 x \c n2, the Kronecker
 * product is the \c m1*m2 x \c n1*n2 block matrix whose block \c (i,j) is
 * \c A(i,j)*B. This class stores only the two factors and evaluates every
 * operation through them:
 *
 * - the matrix-vector product uses the vec identity
 *   \f$ (A \otimes B)\,\mathrm{vec}(X) = \mathrm{vec}(B X A^T) \f$, costing
 *   O(m2 n2 n1 + m2 n1 m1) instead of the O(m1 m2 n1 n2) of a materialized
 *   product;
 * - linear solves and \ref inverse factor through decompositions of \c A and
 *   \c B (\f$ (A \otimes B)^{-1} = A^{-1} \otimes B^{-1} \f$); \ref rank and
 *   minimum-norm least-squares solves (\f$ (A \otimes B)^+ = A^+ \otimes B^+ \f$)
 *   go through the factor SVDs, thresholding the pairwise singular-value
 *   products \f$ \sigma_i(A)\,\sigma_j(B) \f$ -- the singular values of the
 *   Kronecker product -- at the product level;
 * - the eigendecomposition and the (thin) SVD are Kronecker products of the
 *   factor decompositions: the eigenvector and singular-vector matrices are
 *   returned as \c KroneckerOperator objects themselves, never materialized;
 * - \ref determinant uses \f$ \det(A \otimes B) = \det(A)^{n_2}\det(B)^{n_1} \f$,
 *   accumulated in an exponent-balanced form so it neither overflows nor
 *   underflows when the result is representable.
 *
 * The class is closed under \ref transpose, \ref conjugate and \ref adjoint
 * (\f$ (A \otimes B)^T = A^T \otimes B^T \f$). \c operator* returns an Eigen
 * product expression, so the operator plugs into the matrix-free iterative
 * solvers, and it can be assigned to a dense matrix when an explicit
 * representation is needed. As with any matrix-free operator, the iterative
 * solvers must be instantiated with \c IdentityPreconditioner (e.g.
 * \c ConjugateGradient<KroneckerOperator<MatrixXd,MatrixXd>,Lower|Upper,IdentityPreconditioner>):
 * the default preconditioners read individual coefficients through \c col() or
 * \c InnerIterator, which the structured operators do not expose.
 *
 * In contrast to \c kroneckerProduct() (the KroneckerProduct module), which
 * builds an expression meant to be evaluated into a dense matrix, this class is
 * an operator meant to be applied and solved with, without ever forming the
 * product.
 *
 * Either factor may be a \c DiagonalMatrix, with the identity as the
 * unit-diagonal special case. A diagonal factor is stored as its diagonal --
 * O(n) instead of O(n^2) -- its side of every product is a diagonal scaling
 * instead of a GEMM, \ref solve divides entrywise instead of factorizing, and
 * \ref transpose, \ref conjugate, \ref adjoint, \ref inverse and
 * \ref determinant never leave diagonal form. This covers the
 * identity-Kronecker operators \f$ I \otimes A \f$ and \f$ A \otimes I \f$
 * ubiquitous in finite-difference and Sylvester/Lyapunov settings, e.g.
 * \code
 * auto K = makeKroneckerOperator(VectorXd::Ones(p).asDiagonal(), A);  // I_p (x) A, applied in O(p m n)
 * \endcode
 * The decomposition family (\ref eigenvalues, \ref eigenvectors,
 * \ref singularValues, \ref matrixU, \ref matrixV, \ref leastSquaresSolve,
 * \ref rank) currently materializes a diagonal factor densely for the factor
 * decomposition.
 *
 * Either factor may also be a \c SparseMatrix, stored compressed. For finite
 * inputs, its side of a product is sparse-dense -- O(nnz) per right-hand-side column
 * instead of O(m n) -- \ref solve factorizes it once with \c SparseLU and
 * back-substitutes per column, \ref transpose, \ref conjugate and \ref adjoint
 * stay sparse, and \ref determinant accumulates the SparseLU pivots in the same
 * balanced form as the dense LU path, scaling small factors up before
 * factorization to keep the elimination out of the subnormal range. Its \ref inverse is
 * dense (one SparseLU solve against the identity), and the decomposition family
 * densifies it like a diagonal factor. As in every sparse kernel, absent entries
 * are exact zeros: a structural zero annihilates the Inf or NaN it meets, where a stored zero
 * (or a dense factor) would produce NaN. Products with non-finite inputs visit
 * pairs of stored factor entries to preserve this distinction without
 * materializing the product. A sparse factor whose \c SparseLU factorization
 * fails -- an exactly zero or non-finite pivot column -- solves
 * and inverts to NaN, and has determinant 0 when exactly singular (NaN when
 * non-finite). Assigning the operator to a \c SparseMatrix materializes the
 * product sparsely, every inner vector reserved to its exact size. For a sparse
 * \c A of size \c n with \c nnz(A) stored entries:
 * \code
 * auto K = makeKroneckerOperator(VectorXd::Ones(p).asDiagonal(), A);  // I_p (x) A
 * VectorXd y = K * x;             // O(p nnz(A)), no p n x p n matrix
 * VectorXd z = K.solve(b);        // one SparseLU of A, then p column solves
 * SparseMatrix<double> M;
 * M = K;                          // p nnz(A) stored entries, when the matrix itself is needed
 * \endcode
 *
 * \tparam LhsMatrix the plain type of the left factor \c A: a dense \c Matrix,
 *         a \c DiagonalMatrix to exploit diagonal structure, or a
 *         \c SparseMatrix to exploit sparsity.
 * \tparam RhsMatrix the plain type of the right factor \c B, under the same
 *         convention; its scalar type must match that of \c LhsMatrix.
 *
 * \sa makeKroneckerOperator(), class Circulant, class Toeplitz
 */
template <typename LhsMatrix, typename RhsMatrix>
class KroneckerOperator : public EigenBase<KroneckerOperator<LhsMatrix, RhsMatrix>> {
 public:
  using Scalar = typename LhsMatrix::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using StorageIndex = int;
  using ComplexScalar = std::complex<RealScalar>;
  // The vec-trick reshapes below identify a vector of length n1*n2 with an
  // n2 x n1 matrix whose columns are stacked, so every workspace taking part
  // in a reshape is pinned to ColMajor explicitly: the semantics must not
  // change under EIGEN_DEFAULT_TO_ROW_MAJOR.
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;
  using RealVector = Matrix<RealScalar, Dynamic, 1>;
  using ComplexMatrix = Matrix<ComplexScalar, Dynamic, Dynamic, ColMajor>;
  using ComplexVector = Matrix<ComplexScalar, Dynamic, 1>;

  static_assert(std::is_same<Scalar, typename RhsMatrix::Scalar>::value,
                "KroneckerOperator requires both factors to have the same scalar type");
  static_assert((internal::kron_factor_is_dense_matrix<LhsMatrix>::value ||
                 internal::kron_factor_is_diagonal<LhsMatrix>::value ||
                 internal::kron_factor_is_sparse_matrix<LhsMatrix>::value) &&
                    (internal::kron_factor_is_dense_matrix<RhsMatrix>::value ||
                     internal::kron_factor_is_diagonal<RhsMatrix>::value ||
                     internal::kron_factor_is_sparse_matrix<RhsMatrix>::value),
                "KroneckerOperator factors must be plain Matrix, DiagonalMatrix or SparseMatrix types (owning their "
                "storage: views and expressions would dangle)");

 private:
  // Factor-kind dispatch (dense, diagonal or sparse), see kron_factor_ops.
  using LhsOps = internal::kron_factor_ops<LhsMatrix>;
  using RhsOps = internal::kron_factor_ops<RhsMatrix>;
  static constexpr bool kHasSparseFactor = internal::kron_factor_is_sparse_matrix<LhsMatrix>::value ||
                                           internal::kron_factor_is_sparse_matrix<RhsMatrix>::value;

 public:
  static constexpr int RowsAtCompileTime =
      internal::size_at_compile_time(LhsMatrix::RowsAtCompileTime, RhsMatrix::RowsAtCompileTime);
  static constexpr int ColsAtCompileTime =
      internal::size_at_compile_time(LhsMatrix::ColsAtCompileTime, RhsMatrix::ColsAtCompileTime);
  static constexpr int MaxRowsAtCompileTime = RowsAtCompileTime;
  static constexpr int MaxColsAtCompileTime = ColsAtCompileTime;
  static constexpr int SizeAtCompileTime = internal::size_at_compile_time(RowsAtCompileTime, ColsAtCompileTime);
  static constexpr int MaxSizeAtCompileTime = SizeAtCompileTime;
  static constexpr bool IsRowMajor = false;
  // Deliberately no IsVectorAtCompileTime: Ref<const KroneckerOperator>'s default
  // StrideType argument reads it, so its absence makes internal::is_ref_compatible
  // SFINAE to false and keeps the iterative solvers on their matrix-free path.

  /** Builds the operator \c A (x) \c B from the two factors, evaluated into the
   * operator's factor types: a dense expression into a \c Matrix, a diagonal
   * one into a \c DiagonalMatrix (stored as its diagonal), a sparse one into a
   * compressed \c SparseMatrix. */
  template <typename LhsDerived, typename RhsDerived>
  KroneckerOperator(const EigenBase<LhsDerived>& a, const EigenBase<RhsDerived>& b)
      : m_A(a.derived()), m_B(b.derived()) {
    eigen_assert(m_A.size() > 0 && m_B.size() > 0 && "KroneckerOperator factors must be non-empty");
    LhsOps::prepare(m_A);
    RhsOps::prepare(m_B);
  }

  EIGEN_DEVICE_FUNC Index rows() const { return m_A.rows() * m_B.rows(); }
  EIGEN_DEVICE_FUNC Index cols() const { return m_A.cols() * m_B.cols(); }

  /** \returns the left factor \c A. */
  const LhsMatrix& lhs() const { return m_A; }
  /** \returns the right factor \c B. */
  const RhsMatrix& rhs() const { return m_B; }

  /** \returns the coefficient at row \a row and column \a col. */
  Scalar coeff(Index row, Index col) const {
    eigen_assert(row >= 0 && row < rows() && col >= 0 && col < cols());
    const Index m2 = m_B.rows(), n2 = m_B.cols();
    Scalar a, b;
    if (!LhsOps::coeffIfStored(m_A, row / m2, col / n2, a) || !RhsOps::coeffIfStored(m_B, row % m2, col % n2, b))
      return Scalar(0);
    return a * b;
  }

  /** \returns the transpose \f$ A^T \otimes B^T \f$, itself a Kronecker
   * operator. A diagonal factor stays diagonal (it is its own transpose). */
  KroneckerOperator<typename LhsOps::TransposedFactor, typename RhsOps::TransposedFactor> transpose() const {
    return {LhsOps::transposed(m_A), RhsOps::transposed(m_B)};
  }

  /** \returns the conjugate \f$ \bar A \otimes \bar B \f$, itself a Kronecker operator. */
  KroneckerOperator conjugate() const { return {LhsOps::conjugated(m_A), RhsOps::conjugated(m_B)}; }

  /** \returns the adjoint \f$ A^H \otimes B^H \f$, itself a Kronecker
   * operator. A diagonal factor stays diagonal (its adjoint is its conjugate). */
  KroneckerOperator<typename LhsOps::TransposedFactor, typename RhsOps::TransposedFactor> adjoint() const {
    return {LhsOps::adjointed(m_A), RhsOps::adjointed(m_B)};
  }

  /** \returns the solution of \c (*this) * x = b for \b square factors, obtained
   * from one LU decomposition per dense factor and one \c SparseLU per sparse
   * factor (a diagonal factor is solved by entrywise division instead):
   * reshaping \c b column-wise as \c mat(b) of size \c n2 x \c n1, the system
   * reads \f$ B X A^T = \mathrm{mat}(b) \f$, so
   * \f$ X = B^{-1} \mathrm{mat}(b) A^{-T} \f$. Supports multiple right-hand
   * sides at O(n1^3 + n2^3 + nrhs (n1 + n2) n1 n2) total cost for dense
   * factors; a diagonal factor contributes only O(nrhs n1 n2) per application,
   * a sparse one its factorization plus \c nrhs times its column solves.
   *
   * The solves run in a normalized frame [2][3]: the factors and each right-hand
   * side are rescaled to unit magnitude by exact powers of two, and the combined
   * exponent is folded back into the solution entrywise afterwards. In the raw
   * frame the intermediate \f$ B^{-1} \mathrm{mat}(b) \f$ can overflow, or
   * silently underflow to zero, on factor magnitudes alone even when the
   * solution is representable (the subsequent \f$ A^{-T} \f$ rescales it: think
   * \c B huge and \c A tiny); in the normalized frame the intermediates are
   * bounded by the conditioning of the factors, and the final fold saturates to
   * 0/Inf exactly when the true solution leaves the representable range.
   * Partial pivoting is invariant under the uniform power-of-two normalization
   * and every substitution step scales exactly with it, so the result is
   * bit-identical to the unnormalized evaluation whenever that evaluation
   * encounters no intermediate over- or underflow.
   * \warning Both factors must be invertible, like in \c PartialPivLU: a
   * singular dense factor substitutes Inf/NaN through the solution, and a
   * sparse factor whose \c SparseLU factorization fails solves to NaN. Use
   * \ref leastSquaresSolve for rank-deficient or rectangular factors. */
  template <typename Rhs>
  Matrix<Scalar, ColsAtCompileTime, Rhs::ColsAtCompileTime> solve(const MatrixBase<Rhs>& b) const {
    EIGEN_STATIC_ASSERT(RowsAtCompileTime == Dynamic || Rhs::RowsAtCompileTime == Dynamic ||
                            int(RowsAtCompileTime) == int(Rhs::RowsAtCompileTime),
                        YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES)
    const Index n1 = m_A.cols(), n2 = m_B.cols();
    eigen_assert(m_A.rows() == n1 && m_B.rows() == n2 && "KroneckerOperator::solve requires square factors");
    eigen_assert(b.rows() == n1 * n2 && "right-hand side has the wrong number of rows");
    // The exponent bounds are 0 for zero or non-finite data, which is then left
    // unnormalized (Inf/NaN propagate through the substitutions as usual).
    const internal::kron_factor_solver<LhsMatrix> solverA(m_A);
    const internal::kron_factor_solver<RhsMatrix> solverB(m_B);
    Matrix<Scalar, ColsAtCompileTime, Rhs::ColsAtCompileTime> x(n1 * n2, b.cols());
    DenseVector bc(n1 * n2);
    DenseMatrix X(n2, n1);
    for (Index k = 0; k < b.cols(); ++k) {
      bc = b.col(k);
      const int ec = internal::structured_exponent_bound(bc);
      if (ec != 0) internal::kron_ldexp_entries(bc, -ec);
      X = solverA.solveTransposedRight(solverB.solveLeft(bc.reshaped(n2, n1)));
      // Fold the combined exponent back. The entrywise ldexp saturates exactly
      // where the true solution over- or underflows; a multiplicative fold could
      // not (the combined exponent can exceed the representable range of any
      // fixed number of power-of-two factors).
      const int e = ec - solverA.exponent() - solverB.exponent();
      if (e != 0) internal::kron_ldexp_entries(X, e);
      x.col(k) = X.reshaped();
    }
    return x;
  }

  /** \returns the minimum-norm least-squares solution of \c (*this) * x = b,
   * through one SVD per factor: with \f$ A = U_A \Sigma_A V_A^H \f$ and
   * \f$ B = U_B \Sigma_B V_B^H \f$, the SVD of the product is
   * \f$ (U_A \otimes U_B)(\Sigma_A \otimes \Sigma_B)(V_A \otimes V_B)^H \f$, and
   * the pseudo-inverse is applied via the vec identity. The singular values of
   * the product are the pairwise products \f$ \sigma_i(A)\,\sigma_j(B) \f$, so
   * the truncation thresholds those products against the product-level
   * threshold of \ref rank, in the same overflow-safe ratio form -- factor-wise
   * truncation would invert modes that are negligible at the product level.
   * Handles rectangular and rank-deficient factors. Supports multiple
   * right-hand sides.
   *
   * Unlike \ref solve, no rescaling of the intermediates is needed here: the
   * projection and back-transformation matrices have orthonormal columns, so no
   * intermediate can outgrow the data by more than a dimension factor, exactly
   * as in a dense product. */
  template <typename Rhs>
  Matrix<Scalar, ColsAtCompileTime, Rhs::ColsAtCompileTime> leastSquaresSolve(const MatrixBase<Rhs>& b) const {
    EIGEN_STATIC_ASSERT(RowsAtCompileTime == Dynamic || Rhs::RowsAtCompileTime == Dynamic ||
                            int(RowsAtCompileTime) == int(Rhs::RowsAtCompileTime),
                        YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES)
    const Index m1 = m_A.rows(), m2 = m_B.rows();
    eigen_assert(b.rows() == m1 * m2 && "right-hand side has the wrong number of rows");
    BDCSVD<DenseMatrix, ComputeThinU | ComputeThinV> svdA(LhsOps::denseFactor(m_A)), svdB(RhsOps::denseFactor(m_B));
    const RealVector sa = svdA.singularValues(), sb = svdB.singularValues();
    const RealScalar tol = relativeRankThreshold();
    const Index kA = sa.size(), kB = sb.size();
    Matrix<Scalar, ColsAtCompileTime, Rhs::ColsAtCompileTime> x(cols(), b.cols());
    if (sa[0] == RealScalar(0) || sb[0] == RealScalar(0)) {
      // An exactly zero factor zeroes the whole operator, whose pseudo-inverse is
      // zero (and the singular-value ratios below would be 0/0).
      x.setZero();
      return x;
    }
    const SingularModes modesA(sa), modesB(sb);
    Matrix<RealScalar, Dynamic, Dynamic, ColMajor> divisors(kB, kA);
    Matrix<int, Dynamic, Dynamic, ColMajor> exponents(kB, kA);
    Matrix<bool, Dynamic, Dynamic, ColMajor> retained(kB, kA);
    for (Index j = 0; j < kA; ++j)
      for (Index i = 0; i < kB; ++i) {
        retained(i, j) = modesA.retains(modesB, j, i, tol);
        divisors(i, j) = modesA.mantissas[j] * modesB.mantissas[i];
        exponents(i, j) = -(modesA.exponents[j] + modesB.exponents[i]);
      }
    DenseVector bc(m1 * m2);
    DenseMatrix M(kB, kA), X(m_B.cols(), m_A.cols());
    for (Index k = 0; k < b.cols(); ++k) {
      bc = b.col(k);
      // By [1], M = U_B^H mat(b) conj(U_A) matricizes (U_A (x) U_B)^H b.
      M.noalias() = svdB.matrixU().adjoint() * bc.reshaped(m2, m1) * svdA.matrixU().conjugate();
      // Apply the exponent before dividing by the mantissa product [2]: the
      // singular-value product or its reciprocal need not be representable.
      for (Index j = 0; j < kA; ++j)
        for (Index i = 0; i < kB; ++i) {
          if (retained(i, j))
            M(i, j) = internal::structured_ldexp_clamped(M(i, j), Index(exponents(i, j))) / divisors(i, j);
          else
            M(i, j) = Scalar(0);
        }
      X.noalias() = svdB.matrixV() * M * svdA.matrixV().transpose();
      x.col(k) = X.reshaped();
    }
    return x;
  }

  /** \returns the numerical rank: the number of pairwise singular-value products
   * \f$ \sigma_i(A)\,\sigma_j(B) \f$ -- the singular values of the Kronecker
   * product -- that reach the threshold
   * \c min(rows(),cols()) * epsilon * sigma_max(A) * sigma_max(B) (the
   * \c SVDBase convention), and that reach the smallest normal number (the
   * \c SVDBase threshold clamp, so subnormal products count as exact zeros).
   * The relative comparison is made in ratio space,
   * \f$ (\sigma_i(A)/\sigma_{max}(A))(\sigma_j(B)/\sigma_{max}(B)) \f$ against
   * \c min(rows(),cols()) * epsilon, and the clamp in exponent space, so that
   * neither the thresholds nor the products can spuriously under- or overflow.
   * This is the same threshold \ref leastSquaresSolve uses to decide which
   * modes to invert. Thresholding the products matters: factors that are each
   * full rank against their own threshold can still form pairwise products
   * that are negligible at the product level, so the rank can be smaller than
   * the product of the factor ranks. */
  Index rank() const {
    BDCSVD<DenseMatrix> svdA(LhsOps::denseFactor(m_A)), svdB(RhsOps::denseFactor(m_B));
    const RealVector sa = svdA.singularValues(), sb = svdB.singularValues();
    // An exactly zero factor zeroes the whole operator (and would make the
    // ratios below 0/0).
    if (sa[0] == RealScalar(0) || sb[0] == RealScalar(0)) return 0;
    const RealScalar tol = relativeRankThreshold();
    const SingularModes modesA(sa), modesB(sb);
    Index r = 0;
    for (Index i = 0; i < sa.size(); ++i)
      for (Index j = 0; j < sb.size(); ++j)
        // Negated ratio comparison so NaN ratios count as non-zero.
        if (modesA.retains(modesB, i, j, tol)) ++r;
    return r;
  }

  /** \returns the inverse \f$ A^{-1} \otimes B^{-1} \f$, itself a Kronecker
   * operator, for square invertible factors. A diagonal factor's inverse stays
   * diagonal (entrywise reciprocals); a sparse factor's inverse is a dense
   * matrix, computed by one \c SparseLU solve against the identity (NaN when
   * the factorization fails). */
  KroneckerOperator<typename LhsOps::InverseFactor, typename RhsOps::InverseFactor> inverse() const {
    eigen_assert(m_A.rows() == m_A.cols() && m_B.rows() == m_B.cols() &&
                 "KroneckerOperator::inverse requires square factors");
    return {LhsOps::inversed(m_A), RhsOps::inversed(m_B)};
  }

  /** \returns the determinant \f$ \det(A)^{n_2} \det(B)^{n_1} \f$ for square
   * factors \c A of size \c n1 and \c B of size \c n2. The product is
   * accumulated from the factor LU diagonals (the \c SparseLU pivots for a
   * sparse factor, the diagonal itself for a diagonal factor, skipping the LU)
   * in the balanced form \c m * 2^e --
   * every factor and the running product are renormalized to unit magnitude
   * with the power of two tracked separately -- so the partial products (in
   * particular \c det(A) and \c det(B) themselves, which can overflow or
   * underflow on their own) never leave the representable range when the
   * determinant itself is representable. */
  Scalar determinant() const {
    eigen_assert(m_A.rows() == m_A.cols() && m_B.rows() == m_B.cols() &&
                 "KroneckerOperator::determinant requires square factors");
    Index exponent = 0;
    Scalar mant = balancedDetPow(m_A, m_B.cols(), exponent);
    mant = internal::structured_balance(mant * balancedDetPow(m_B, m_A.cols(), exponent), exponent);
    return internal::structured_ldexp_clamped(mant, exponent);
  }

  /** \returns the eigenvalues for square factors, in Kronecker order: entry
   * \c i*n2 + j is \f$ \lambda_i(A)\,\mu_j(B) \f$, matching column \c i*n2 + j of
   * \ref eigenvectors. The set is not sorted -- there is no canonical eigenvalue
   * order, and sorting would break the Kronecker structure of the eigenvector
   * matrix. */
  ComplexVector eigenvalues() const {
    eigen_assert(m_A.rows() == m_A.cols() && m_B.rows() == m_B.cols() &&
                 "KroneckerOperator::eigenvalues requires square factors");
    ComplexEigenSolver<ComplexMatrix> esA(LhsOps::denseFactor(m_A).template cast<ComplexScalar>(),
                                          /*computeEigenvectors=*/false);
    ComplexEigenSolver<ComplexMatrix> esB(RhsOps::denseFactor(m_B).template cast<ComplexScalar>(),
                                          /*computeEigenvectors=*/false);
    eigen_assert(esA.info() == Success && esB.info() == Success);
    // Column-major stacking of the nB x nA outer product puts mu_j(B) lambda_i(A)
    // at index i*nB + j, the Kronecker order.
    return (esB.eigenvalues() * esA.eigenvalues().transpose()).reshaped();
  }

  /** \returns the matrix of eigenvectors \f$ V_A \otimes V_B \f$ for square
   * factors -- itself a Kronecker operator, never materialized. Column
   * \c i*n2 + j is \f$ v_i(A) \otimes v_j(B) \f$ and matches
   * \c eigenvalues()[i*n2 + j]. Assign it to a dense matrix to materialize. */
  KroneckerOperator<ComplexMatrix, ComplexMatrix> eigenvectors() const {
    eigen_assert(m_A.rows() == m_A.cols() && m_B.rows() == m_B.cols() &&
                 "KroneckerOperator::eigenvectors requires square factors");
    ComplexEigenSolver<ComplexMatrix> esA(LhsOps::denseFactor(m_A).template cast<ComplexScalar>());
    ComplexEigenSolver<ComplexMatrix> esB(RhsOps::denseFactor(m_B).template cast<ComplexScalar>());
    eigen_assert(esA.info() == Success && esB.info() == Success);
    return {esA.eigenvectors(), esB.eigenvectors()};
  }

  /** \returns the singular values of the thin SVD
   * \f$ A \otimes B = U \Sigma V^H \f$ in Kronecker order: entry \c i*k_B + j is
   * \f$ \sigma_i(A)\,\sigma_j(B) \f$ with \c k_A, \c k_B the factor thin ranks
   * \c min(rows, cols), matching the columns of \ref matrixU and \ref matrixV.
   * The values are not sorted (sorting would break the Kronecker structure of
   * \c U and \c V); for rectangular shapes the full SVD pads this set with
   * \c min(rows(),cols()) - k_A*k_B structural zeros. */
  RealVector singularValues() const {
    BDCSVD<DenseMatrix> svdA(LhsOps::denseFactor(m_A)), svdB(RhsOps::denseFactor(m_B));
    // Column-major stacking of the kB x kA outer product puts sigma_j(B) sigma_i(A)
    // at index i*kB + j, the Kronecker order.
    return (svdB.singularValues() * svdA.singularValues().transpose()).reshaped();
  }

  /** \returns the left singular vectors \f$ U_A \otimes U_B \f$ of the thin SVD,
   * itself a Kronecker operator with orthonormal columns; column \c i*k_B + j
   * matches \c singularValues()[i*k_B + j]. */
  KroneckerOperator<DenseMatrix, DenseMatrix> matrixU() const {
    BDCSVD<DenseMatrix, ComputeThinU> svdA(LhsOps::denseFactor(m_A)), svdB(RhsOps::denseFactor(m_B));
    return {svdA.matrixU(), svdB.matrixU()};
  }

  /** \returns the right singular vectors \f$ V_A \otimes V_B \f$ of the thin SVD,
   * itself a Kronecker operator with orthonormal columns; column \c i*k_B + j
   * matches \c singularValues()[i*k_B + j]. */
  KroneckerOperator<DenseMatrix, DenseMatrix> matrixV() const {
    BDCSVD<DenseMatrix, ComputeThinV> svdA(LhsOps::denseFactor(m_A)), svdB(RhsOps::denseFactor(m_B));
    return {svdA.matrixV(), svdB.matrixV()};
  }

  /** \internal Writes the representation into \a dst, dense or sparse according
   * to the destination's storage kind; invoked through \c dense = kron; and
   * \c sparse = kron; (there is no converting constructor for a SparseMatrix). */
  template <typename Dest>
  void evalTo(Dest& dst) const {
    evalToImpl(dst, IsSparseDestination<Dest>());
  }

  /** \internal Computes \c dst += (*this), see evalTo(). */
  template <typename Dest>
  void addTo(Dest& dst) const {
    addToImpl(dst, IsSparseDestination<Dest>());
  }

  /** \internal Computes \c dst -= (*this), see evalTo(). */
  template <typename Dest>
  void subTo(Dest& dst) const {
    subToImpl(dst, IsSparseDestination<Dest>());
  }

 private:
  template <typename Dest>
  using IsSparseDestination = std::is_same<typename internal::traits<Dest>::StorageKind, Sparse>;

  /** \internal The dense representation, one block per stored entry of \c A. A
   * diagonal or sparse \c B writes each block through the diagonal- or
   * sparse-to-dense assignment (zeros plus the stored entries); the structurally
   * zero blocks of a diagonal or sparse \c A are cleared up front instead of
   * visited. */
  template <typename Dest>
  void evalToImpl(Dest& dst, std::false_type) const {
    const Index m2 = m_B.rows(), n2 = m_B.cols();
    if (!LhsOps::StoresAllEntries) dst.setZero();
    LhsOps::forEachNonZero(
        m_A, [&dst, m2, n2, this](Index i, Index j, const Scalar& a) { dst.block(i * m2, j * n2, m2, n2) = a * m_B; });
  }

  template <typename Dest>
  void addToImpl(Dest& dst, std::false_type) const {
    const Index m2 = m_B.rows(), n2 = m_B.cols();
    LhsOps::forEachNonZero(
        m_A, [&dst, m2, n2, this](Index i, Index j, const Scalar& a) { dst.block(i * m2, j * n2, m2, n2) += a * m_B; });
  }

  template <typename Dest>
  void subToImpl(Dest& dst, std::false_type) const {
    const Index m2 = m_B.rows(), n2 = m_B.cols();
    LhsOps::forEachNonZero(
        m_A, [&dst, m2, n2, this](Index i, Index j, const Scalar& a) { dst.block(i * m2, j * n2, m2, n2) -= a * m_B; });
  }

  /** \internal The sparse representation. Every stored entry of \c A (all
   * entries of a dense factor, the diagonal of a diagonal one, the nonzeros of a
   * sparse one) meets every stored entry of \c B, so \a dst holds exactly the
   * products of the stored entries, explicit zeros included (\c prune() drops
   * them). Each inner vector is reserved to its exact final size, the product of
   * the factors' inner-vector counts, and receives its entries in increasing
   * inner index for every mix of storage orders: with the loop over \c A
   * outermost, the visits to a fixed column <tt>(jA, jB)</tt> are lexicographic
   * in <tt>(iA, iB)</tt>, and those to a fixed row in <tt>(jA, jB)</tt>. */
  template <typename Dest>
  void evalToImpl(Dest& S, std::true_type) const {
    const Index m2 = m_B.rows(), n2 = m_B.cols();
    S.resize(rows(), cols());
    using IndexVector = Matrix<Index, Dynamic, 1>;
    const IndexVector nnzA = LhsOps::innerNonZeros(m_A, Dest::IsRowMajor);
    const IndexVector nnzB = RhsOps::innerNonZeros(m_B, Dest::IsRowMajor);
    // Inner vectors kA of A and kB of B meet in inner vector kA * nnzB.size() + kB
    // of the product: the column-major stacking of the count outer product.
    const Matrix<Index, Dynamic, Dynamic, ColMajor> counts = nnzB * nnzA.transpose();
    S.reserve(counts.reshaped());
    LhsOps::forEachNonZero(m_A, [&S, m2, n2, this](Index iA, Index jA, const Scalar& a) {
      RhsOps::forEachNonZero(m_B, [&S, m2, n2, iA, jA, &a](Index iB, Index jB, const Scalar& b) {
        S.insert(iA * m2 + iB, jA * n2 + jB) = a * b;
      });
    });
    S.makeCompressed();
  }

  template <typename Dest>
  void addToImpl(Dest& dst, std::true_type) const {
    typename Dest::PlainObject product;
    evalTo(product);
    dst += product;
  }

  template <typename Dest>
  void subToImpl(Dest& dst, std::true_type) const {
    typename Dest::PlainObject product;
    evalTo(product);
    dst -= product;
  }

 public:
  /** \returns the product expression \c (*this) * \a x, evaluated through the vec
   * identity without materializing the Kronecker product. The expression carries
   * the default product tag, so assigning it behaves like any dense product: a
   * temporary resolves aliasing between the destination and \a x, and
   * \c .noalias() skips it. */
  template <typename Rhs>
  Product<KroneckerOperator, Rhs> operator*(const MatrixBase<Rhs>& x) const {
    EIGEN_STATIC_ASSERT(ColsAtCompileTime == Dynamic || Rhs::RowsAtCompileTime == Dynamic ||
                            int(ColsAtCompileTime) == int(Rhs::RowsAtCompileTime),
                        INVALID_MATRIX_PRODUCT)
    eigen_assert(x.rows() == cols() && "invalid product: dimensions do not match");
    return Product<KroneckerOperator, Rhs>(*this, x.derived());
  }

  /** \internal Computes \c dst += alpha * (*this) * rhs through the vec identity
   * \f$ (A \otimes B)\,\mathrm{vec}(X) = \mathrm{vec}(B X A^T) \f$ of [1], column
   * by column. \c ProductScalar is the promoted scalar of the product (complex
   * when a real operator is applied to a complex right-hand side); the
   * workspaces and the accumulation run in the promoted type.
   *
   * Evaluated as \c (B X) A^T, the first factor \c B X can overflow even when
   * the product itself is representable (the multiplication by \c A^T rescales
   * it: think \c B huge and \c A tiny). Each column is therefore pre-scaled by
   * an exact power of two [2][3] derived from the component-wise exponent
   * bounds of \c A, \c B and the column -- never from complex moduli, which
   * overflow near the threshold -- so that no intermediate can spuriously
   * overflow, and the
   * exponent is folded back into the result through two representable
   * half-factors, saturating to 0/Inf exactly when the true result leaves the
   * representable range. The scale is one whenever the conservative bound shows
   * no overflow risk, so results of moderate magnitude are bit-identical to the
   * unscaled evaluation. Non-finite data is never scaled (the bounds cannot see
   * past an Inf/NaN, and 0 * Inf would manufacture NaNs); the unscaled GEMMs
   * propagate it entrywise exactly like a dense product. With a sparse factor,
   * non-finite inputs instead use the stored-entry product to preserve the
   * distinction between absent entries and stored zeros. */
  template <typename Dest, typename Rhs, typename ProductScalar>
  void addProduct(Dest& dst, const Rhs& rhs, const ProductScalar& alpha) const {
    using ProductVector = Matrix<ProductScalar, Dynamic, 1>;
    using ProductMatrix = Matrix<ProductScalar, Dynamic, Dynamic, ColMajor>;  // ColMajor: see DenseMatrix
    using ProductReal = typename NumTraits<ProductScalar>::Real;
    // In particular, materialize a nested matrix product once before taking
    // per-column expressions from it. Otherwise each column evaluator may
    // independently evaluate the entire nested product.
    typename internal::nested_eval<Rhs, 1>::type actualRhs(rhs);
    const Index n1 = m_A.cols(), n2 = m_B.cols();
    eigen_assert(actualRhs.rows() == n1 * n2 && "invalid product: dimensions do not match");
    // If max|X| < 2^eX, the two factor applications are bounded by
    // 2^(eB+eX+bits2) and 2^(eA+eB+eX+bits1+bits2), including dot-product growth.
    const bool factorsFinite = LhsOps::allFinite(m_A) && RhsOps::allFinite(m_B);
    const int expA = LhsOps::exponentBound(m_A);
    const int expB = RhsOps::exponentBound(m_B);
    // Bit widths of the contracted dimensions, i.e. the dot-product growth
    // bound; zero for a diagonal factor, whose application has no accumulation.
    const int bits1 = LhsOps::growthBits(n1);
    const int bits2 = RhsOps::growthBits(n2);
    const int budget = std::numeric_limits<ProductReal>::max_exponent - 2;
    ProductVector xc(n1 * n2);
    ProductMatrix Y(m_B.rows(), m_A.rows());
    for (Index k = 0; k < actualRhs.cols(); ++k) {
      xc = actualRhs.col(k).template cast<ProductScalar>();
      int e = 0;
      const bool finite = factorsFinite && xc.allFinite();
      if (finite) {
        const int expX = internal::structured_exponent_bound(xc);  // 0 for an all-zero column: no scaling
        e = numext::maxi(0, numext::maxi(expB + expX + bits2 - budget, expA + expB + expX + bits1 + bits2 - budget));
        // The cap keeps the half-factors below overflow; it only binds when the
        // true result overflows anyway, so the saturation to Inf is genuine.
        e = numext::mini(e, 2 * budget);
      }
      if (e > 0) {
        // Each power of two is split in halves so that the factors themselves
        // stay representable; scaling by the two exact factors in sequence is
        // still an exact shift wherever the result is representable.
        const ProductReal down1 = ProductReal(std::ldexp(ProductReal(1), -(e / 2)));
        const ProductReal down2 = ProductReal(std::ldexp(ProductReal(1), -(e - e / 2)));
        xc = (xc * down1) * down2;
      }
      // For a diagonal factor its side degenerates to a diagonal scaling
      // (transposedOperand: a diagonal matrix is its own transpose), a single
      // stored term per entry: only a sparse factor can leave an exact 0 in B X
      // (an empty row) for an Inf/NaN of A to meet.
      if (EIGEN_PREDICT_FALSE(!finite && kHasSparseFactor))
        productNonFinite(Y, xc);
      else
        Y.noalias() = m_B * xc.reshaped(n2, n1) * LhsOps::transposedOperand(m_A);
      if (e > 0) {
        const ProductReal up1 = ProductReal(std::ldexp(ProductReal(1), e / 2));
        const ProductReal up2 = ProductReal(std::ldexp(ProductReal(1), e - e / 2));
        dst.col(k) += alpha * ((Y.reshaped() * up1) * up2);
      } else {
        dst.col(k) += alpha * Y.reshaped();
      }
    }
  }

 private:
  // A dense intermediate loses which zeros were structural; a later Inf/NaN
  // factor would turn those absent terms into NaNs. Preserve the stored pairs.
  template <typename ProductScalar>
  EIGEN_DONT_INLINE void productNonFinite(Matrix<ProductScalar, Dynamic, Dynamic, ColMajor>& result,
                                          const Matrix<ProductScalar, Dynamic, 1>& x) const {
    result.setZero();
    LhsOps::forEachNonZero(m_A, [&result, &x, this](Index iA, Index jA, const Scalar& a) {
      RhsOps::forEachNonZero(m_B, [&result, &x, jA, iA, &a, this](Index iB, Index jB, const Scalar& b) {
        result.coeffRef(iB, iA) += (a * b) * x.coeff(jA * m_B.cols() + jB);
      });
    });
  }

  /** \internal \returns the relative rank/pseudo-inversion threshold for the
   * pairwise singular-value products, in the spirit of the SVD-based
   * pseudo-inverse: a mode \c (i,j) is kept when
   * \c (sa[i]/sa[0]) * (sb[j]/sb[0]) >= min(rows,cols) * epsilon, the \c SVDBase
   * convention for \c sa[i]*sb[j] measured against \c smax = sa[0]*sb[0]. The
   * comparison must happen in ratio space: the absolute form both underflows
   * (\c min(rows,cols) * epsilon * sa[0] can flush to zero before the \c sb[0]
   * multiplication, silently accepting every mode) and overflows
   * (\c sa[0]*sb[0] can exceed the representable range). The ratios never
   * exceed one, and a ratio product that underflows is genuinely below any
   * epsilon-sized threshold. */
  RealScalar relativeRankThreshold() const {
    return RealScalar(numext::mini(rows(), cols())) * NumTraits<RealScalar>::epsilon();
  }

  // Factor-level ratios and frexp decompositions are independent of the RHS and
  // of the other factor. Keep rank() and pseudo-inversion on the same predicate.
  struct SingularModes {
    explicit SingularModes(const RealVector& s) : ratios(s.size()), mantissas(s.size()), exponents(s.size()) {
      for (Index i = 0; i < s.size(); ++i) {
        ratios[i] = s[i] / s[0];
        exponents[i] = 0;
        EIGEN_USING_STD(frexp);
        mantissas[i] = frexp(s[i], &exponents[i]);
      }
    }

    bool retains(const SingularModes& other, Index i, Index j, RealScalar tol) const {
      // Negation keeps NaN ratios in the inverted set, matching SVDBase.
      if (ratios[i] * other.ratios[j] < tol) return false;
      const RealScalar m = mantissas[i] * other.mantissas[j];
      if (!(numext::isfinite)(m)) return true;
      if (m == RealScalar(0)) return false;
      // s_i*s_j = m*2^(e_i+e_j), with m in [0.25,1). Test the
      // smallest-normal clamp without forming an overflowing/underflowing product.
      const int e = exponents[i] + other.exponents[j] - (m < RealScalar(0.5) ? 1 : 0);
      return e >= std::numeric_limits<RealScalar>::min_exponent;
    }

    RealVector ratios, mantissas;
    Matrix<int, Dynamic, 1> exponents;
  };

  /** \internal \returns the mantissa of \c det(M)^power in the balanced form
   * \c m * 2^e, adding \c e into \a exponent. The determinant is accumulated
   * directly from the LU diagonal, times the permutation sign (from the
   * diagonal itself for a diagonal factor, see kron_factor_ops::balancedDet),
   * each entry and the running product renormalized by \ref balance, and the
   * integer power is applied by balanced repeated multiplication -- exact
   * integer-power semantics for negative real and complex determinants (no
   * exp/log branch-cut roundoff), with the bulk of the magnitude carried
   * exactly on the exponent side. */
  template <typename Factor>
  static Scalar balancedDetPow(const Factor& M, Index power, Index& exponent) {
    Index e = 0;
    const Scalar m = internal::kron_factor_ops<Factor>::balancedDet(M, e);
    Scalar r(1);
    Index er = 0;
    for (Index k = 0; k < power; ++k) r = internal::structured_balance(r * m, er);
    exponent += power * e + er;
    return r;
  }

  LhsMatrix m_A;
  RhsMatrix m_B;
};

/** \ingroup StructuredMatrices_Module
 * \returns a \ref KroneckerOperator \c a (x) \c b holding evaluated copies of the
 * factors. The operator type is deduced from the plain types of \a a and \a b:
 * a dense argument becomes a \c Matrix factor, a diagonal one (e.g.
 * \c VectorXd::Ones(n).asDiagonal() for an identity factor) an owning
 * \c DiagonalMatrix, a sparse one a compressed \c SparseMatrix, each exploited
 * structurally, see \ref KroneckerOperator. */
template <typename LhsDerived, typename RhsDerived>
KroneckerOperator<typename LhsDerived::PlainObject, typename RhsDerived::PlainObject> makeKroneckerOperator(
    const EigenBase<LhsDerived>& a, const EigenBase<RhsDerived>& b) {
  return {a.derived(), b.derived()};
}

namespace internal {

template <typename LhsMatrix, typename RhsMatrix, typename Rhs, int ProductTag>
struct generic_product_impl<KroneckerOperator<LhsMatrix, RhsMatrix>, Rhs, StructuredShape, DenseShape, ProductTag>
    : structured_product_impl<KroneckerOperator<LhsMatrix, RhsMatrix>, Rhs> {};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_STRUCTURED_KRONECKER_OPERATOR_H
