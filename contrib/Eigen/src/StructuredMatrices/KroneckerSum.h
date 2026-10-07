// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// References:
//  [1] R. H. Bartels and G. W. Stewart, "Solution of the matrix equation
//      AX + XB = C", Communications of the ACM 15 (1972), 820-826. The Schur
//      form solve of the Sylvester equation B X + X A^T = mat(b) that
//      BartelsStewart generalizes to Kronecker sums of any number of factors.
//  [2] R. E. Lynch, J. R. Rice and D. H. Thomas, "Direct solution of partial
//      difference equations by tensor product methods", Numerische Mathematik 6
//      (1964), 185-199. The fast diagonalization method: with Hermitian factors
//      the Schur forms are diagonal and the solve is a division by the
//      eigenvalue sums.

#ifndef EIGEN_STRUCTURED_KRONECKER_SUM_H
#define EIGEN_STRUCTURED_KRONECKER_SUM_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename KroneckerSumType>
class BartelsStewart;

namespace internal {

template <typename LhsMatrix, typename RhsMatrix>
struct traits<KroneckerSum<LhsMatrix, RhsMatrix>> {
  using Scalar = typename LhsMatrix::Scalar;
  using StorageKind = Dense;
  using XprKind = MatrixXpr;
  using StorageIndex = int;
  static constexpr int RowsAtCompileTime =
      size_at_compile_time(traits<LhsMatrix>::RowsAtCompileTime, traits<RhsMatrix>::RowsAtCompileTime);
  static constexpr int ColsAtCompileTime = RowsAtCompileTime;
  static constexpr int MaxRowsAtCompileTime = RowsAtCompileTime;
  static constexpr int MaxColsAtCompileTime = RowsAtCompileTime;
  // No NestByRefBit, for the reason given at traits<KroneckerOperator>.
  static constexpr unsigned int Flags = 0;
};

template <typename LhsMatrix, typename RhsMatrix>
struct evaluator_traits<KroneckerSum<LhsMatrix, RhsMatrix>> {
  using Kind = IndexBased;
  using Shape = StructuredShape;
};

template <typename KroneckerSumType>
struct traits<BartelsStewart<KroneckerSumType>> : traits<Matrix<typename KroneckerSumType::Scalar, Dynamic, Dynamic>> {
  using XprKind = MatrixXpr;
  using StorageKind = SolverStorage;
  using StorageIndex = int;
  using BaseTraits = traits<Matrix<typename KroneckerSumType::Scalar, Dynamic, Dynamic>>;
  static constexpr unsigned int Flags = BaseTraits::Flags & RowMajorBit;
  static constexpr int CoeffReadCost = Dynamic;
};

/** \internal A Kronecker sum S = L (+) R as a factor of a KroneckerOperator or
 * of another KroneckerSum. Left products go through S itself, right products
 * through its factors; the visitors read the sparse materialization, solves go
 * through BartelsStewart, and the inverse and determinant through the dense
 * matrix. */
template <typename LhsMatrix, typename RhsMatrix>
struct kron_factor_ops<KroneckerSum<LhsMatrix, RhsMatrix>, kKronSumFactor> {
  using Factor = KroneckerSum<LhsMatrix, RhsMatrix>;
  using Scalar = typename Factor::Scalar;
  using LhsOps = kron_factor_ops<LhsMatrix>;
  using RhsOps = kron_factor_ops<RhsMatrix>;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using SparseFactor = SparseMatrix<Scalar>;
  using TransposedFactor = KroneckerSum<typename LhsOps::TransposedFactor, typename RhsOps::TransposedFactor>;
  using InverseFactor = DenseMatrix;
  static constexpr bool StoresAllEntries = false;

  static void prepare(Factor&) {}
  static bool coeffIfStored(const Factor& f, Index row, Index col, Scalar& value) {
    const Index n2 = f.rhs().rows();
    Scalar a(0), b(0);
    const bool storedA = row % n2 == col % n2 && LhsOps::coeffIfStored(f.lhs(), row / n2, col / n2, a);
    const bool storedB = row / n2 == col / n2 && RhsOps::coeffIfStored(f.rhs(), row % n2, col % n2, b);
    if (!storedA && !storedB) return false;
    value = a + b;
    return true;
  }
  template <typename Visitor>
  static void forEachNonZero(const Factor& f, Visitor&& visit) {
    kron_factor_ops<SparseFactor>::forEachNonZero(kron_factor_visitable<Factor>::get(f), visit);
  }
  static Matrix<Index, Dynamic, 1> innerNonZeros(const Factor& f, bool rowMajor) {
    return kron_factor_ops<SparseFactor>::innerNonZeros(kron_factor_visitable<Factor>::get(f), rowMajor);
  }
  static TransposedFactor transposed(const Factor& f) { return f.transpose(); }
  static Factor conjugated(const Factor& f) { return f.conjugate(); }
  static TransposedFactor adjointed(const Factor& f) { return f.adjoint(); }
  static InverseFactor inversed(const Factor& f) { return f.solve(DenseMatrix::Identity(f.rows(), f.cols())); }
  static DenseMatrix denseFactor(const Factor& f) { return DenseMatrix(f); }
  static DenseMatrix blockOperand(const Factor& f) { return denseFactor(f); }
  static bool isSquareIdentity(const Factor&) { return false; }
  template <typename Dst, typename Alpha, typename Xpr>
  static void addLeftProduct(Dst& dst, const Alpha& alpha, const Factor& f, const Xpr& X) {
    f.addProduct(dst, X, alpha);
  }
  // With X_j = X(:, j n2 : (j+1) n2 - 1), j < n1, the p x n2 slices of X,
  //   X (L (+) R)^T = X (L (x) I)^T + X (I (x) R)^T = X_{[p n2 x n1]} L^T + [X_j R^T]_j.
  template <typename Dst, typename Alpha, typename Xpr, typename Work>
  static void addRightProduct(Dst& dst, const Alpha& alpha, const Xpr& X, const Factor& f, Work& work) {
    const Index p = X.rows(), n1 = f.lhs().rows(), n2 = f.rhs().rows();
    auto dstL = dst.reshaped(p * n2, n1);
    LhsOps::addRightProduct(dstL, alpha, X.reshaped(p * n2, n1), f.lhs(), work);
    for (Index j = 0; j < n1; ++j) {
      auto dstj = dst.middleCols(j * n2, n2);
      RhsOps::addRightProduct(dstj, alpha, X.middleCols(j * n2, n2), f.rhs(), work);
    }
  }
  // det = prod_{i,j} (lambda_i + mu_j) would cost only the factor Schur forms,
  // but every delta lambda_i recurs in n_R factors of the product, and a shift
  // split across L and R makes |lambda_i + mu_j| << |lambda_i| + |mu_j|. Against
  // a 256-bit reference that product lost 1 to 6 digits to the LU of the
  // materialized sum.
  static Scalar balancedDet(const Factor& f, Index& exponent) {
    return kron_factor_ops<DenseMatrix>::balancedDet(denseFactor(f), exponent);
  }
};

template <typename LhsMatrix, typename RhsMatrix>
struct kron_factor_visitable<KroneckerSum<LhsMatrix, RhsMatrix>, kKronSumFactor> {
  using Factor = KroneckerSum<LhsMatrix, RhsMatrix>;
  using type = SparseMatrix<typename Factor::Scalar>;
  static type get(const Factor& f) {
    type S;
    S = f;
    return S;
  }
};

/** \internal (L (+) R)(v_i (x) w_j) = (lambda_i + mu_j)(v_i (x) w_j), so the
 * eigenvalues and eigenvectors recurse into the factors, as for a Kronecker
 * factor. A sum has no separable SVD, which stays the dense one of the primary
 * template. */
template <typename LhsMatrix, typename RhsMatrix>
struct kron_factor_spectrum<KroneckerSum<LhsMatrix, RhsMatrix>, kKronSumFactor>
    : kron_factor_spectrum<KroneckerSum<LhsMatrix, RhsMatrix>, kKronDenseFactor> {
  using Factor = KroneckerSum<LhsMatrix, RhsMatrix>;
  using Eigenvectors = KroneckerOperator<typename kron_factor_spectrum<LhsMatrix>::Eigenvectors,
                                         typename kron_factor_spectrum<RhsMatrix>::Eigenvectors>;

  static typename Factor::ComplexVector eigenvalues(const Factor& f) { return f.eigenvalues(); }
  static Eigenvectors eigenvectors(const Factor& f) { return f.eigenvectors(); }
};

template <typename LhsMatrix, typename RhsMatrix>
class kron_factor_solver<KroneckerSum<LhsMatrix, RhsMatrix>, kKronSumFactor> {
 public:
  using Factor = KroneckerSum<LhsMatrix, RhsMatrix>;
  using Scalar = typename Factor::Scalar;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;

  explicit kron_factor_solver(const Factor& f) : m_solver(f) {}
  template <typename Xpr>
  DenseMatrix solveLeft(const Xpr& M) const {
    return m_solver.solve(M);
  }
  // X = M S^{-T} solves S X^T = M^T, written through a transposed view of X.
  template <typename Xpr>
  DenseMatrix solveTransposedRight(const Xpr& M) const {
    DenseMatrix X(M.rows(), M.cols());
    X.transpose() = m_solver.solve(M.transpose());
    return X;
  }

 private:
  BartelsStewart<Factor> m_solver;
};

}  // namespace internal

/** \ingroup StructuredMatrices_Module
 * \class KroneckerSum
 * \brief The Kronecker sum \f$ A \oplus B = A \otimes I + I \otimes B \f$ of two
 * square matrices as an implicit operator that is never materialized.
 *
 * For \c A of size \c n1 and \c B of size \c n2 the Kronecker sum is the
 * \c n1*n2 x \c n1*n2 matrix whose block \c (i,j) is \c A(i,j)*I plus \c B on the
 * diagonal blocks. It is the matrix of the separable operators that
 * finite-difference and spectral discretizations produce on tensor-product
 * grids: with \c Dx and \c Dy the 1-D second-difference matrices
 * tridiag(1, -2, 1)/h^2, the 2-D Laplacian is \f$ D_y \oplus D_x \f$, the 3-D
 * one \f$ D_z \oplus D_y \oplus D_x \f$, and an implicit Euler step of the
 * heat equation, \f$ (I - \tau L) u = b \f$ with \f$ L = D_y \oplus D_x \f$, is
 * \f$ \big((I - \tau D_y) \oplus (-\tau D_x)\big) u = b \f$, since
 * \f$ I \otimes I = I \f$ lets a shift go into either factor.
 *
 * With \f$ \mathrm{vec} \f$ stacking columns as for \ref KroneckerOperator, the
 * product is \f$ (A \oplus B)\,\mathrm{vec}(X) = \mathrm{vec}(B X + X A^T) \f$
 * for \c X of size \c n2 x \c n1: one product with each factor, O(n1 n2 (n1 +
 * n2)) for dense factors and O(n1 nnz(B) + n2 nnz(A)) for sparse ones, with no
 * identity ever formed. The factors may be of any kind \ref KroneckerOperator
 * accepts -- dense, diagonal, sparse, \c Identity() or a \c KroneckerOperator --
 * or a \c KroneckerSum itself, which is how sums of three or more factors are
 * built (\c makeKroneckerSum(a, b, c, ...) nests to the right). A Kronecker sum
 * may in turn be a factor of a \ref KroneckerOperator.
 *
 * The operator is closed under \ref transpose, \ref conjugate and \ref adjoint
 * (\f$ (A \oplus B)^T = A^T \oplus B^T \f$), materializes into a dense or a
 * sparse matrix on assignment, and plugs into the matrix-free iterative solvers
 * (with \c IdentityPreconditioner). \ref solve and the reusable
 * \ref BartelsStewart solver use the Schur forms of the factors;
 * \ref eigenvalues are the pairwise sums \f$ \lambda_i(A) + \mu_j(B) \f$, with
 * \ref eigenvectors \f$ V_A \otimes V_B \f$, also as a \ref KroneckerOperator
 * factor.
 * \code
 * SparseMatrix<double> Dx = ..., Dy = ...;                       // tridiag(1, -2, 1) / h^2
 * SparseMatrix<double> Iy(ny, ny); Iy.setIdentity();
 * auto M = makeKroneckerSum(Iy - tau * Dy, -tau * Dx);           // I - tau (Dy (+) Dx)
 * BartelsStewart<decltype(M)> step(M);                           // Schur forms, once
 * for (int k = 0; k < steps; ++k) u = step.solve(u);             // implicit Euler
 * \endcode
 *
 * \tparam LhsMatrix the type of the left factor \c A, see \ref KroneckerOperator.
 * \tparam RhsMatrix the type of the right factor \c B, under the same
 *         convention; its scalar type must match that of \c LhsMatrix.
 *
 * \sa makeKroneckerSum(), class BartelsStewart, class KroneckerOperator
 */
template <typename LhsMatrix, typename RhsMatrix>
class KroneckerSum : public EigenBase<KroneckerSum<LhsMatrix, RhsMatrix>> {
 public:
  using Scalar = typename LhsMatrix::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using StorageIndex = int;

  static_assert(std::is_same<Scalar, typename RhsMatrix::Scalar>::value,
                "KroneckerSum requires both factors to have the same scalar type");
  static_assert((internal::kron_factor_is_dense_matrix<LhsMatrix>::value ||
                 internal::kron_factor_kind<LhsMatrix>() != internal::kKronDenseFactor) &&
                    (internal::kron_factor_is_dense_matrix<RhsMatrix>::value ||
                     internal::kron_factor_kind<RhsMatrix>() != internal::kKronDenseFactor),
                "KroneckerSum factors must be plain Matrix, DiagonalMatrix or SparseMatrix types, identity factors "
                "(makeKroneckerSum stores an Identity() expression as one), KroneckerOperators or KroneckerSums "
                "(owning their storage)");

 private:
  using LhsOps = internal::kron_factor_ops<LhsMatrix>;
  using RhsOps = internal::kron_factor_ops<RhsMatrix>;
  using LhsSpectrum = internal::kron_factor_spectrum<LhsMatrix>;
  using RhsSpectrum = internal::kron_factor_spectrum<RhsMatrix>;

 public:
  using ComplexScalar = std::complex<RealScalar>;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using ComplexVector = Matrix<ComplexScalar, Dynamic, 1>;

  static constexpr int RowsAtCompileTime =
      internal::size_at_compile_time(LhsMatrix::RowsAtCompileTime, RhsMatrix::RowsAtCompileTime);
  static constexpr int ColsAtCompileTime = RowsAtCompileTime;
  static constexpr int MaxRowsAtCompileTime = RowsAtCompileTime;
  static constexpr int MaxColsAtCompileTime = RowsAtCompileTime;
  static constexpr int SizeAtCompileTime = internal::size_at_compile_time(RowsAtCompileTime, ColsAtCompileTime);
  static constexpr int MaxSizeAtCompileTime = SizeAtCompileTime;
  static constexpr bool IsRowMajor = false;
  // Deliberately no IsVectorAtCompileTime, for the reason given in KroneckerOperator.

  /** Builds the operator \c A (+) \c B from the two square factors, evaluated
   * into the operator's factor types as for \ref KroneckerOperator. */
  template <typename LhsDerived, typename RhsDerived>
  KroneckerSum(const EigenBase<LhsDerived>& a, const EigenBase<RhsDerived>& b) : m_A(a.derived()), m_B(b.derived()) {
    eigen_assert(m_A.size() > 0 && m_B.size() > 0 && "KroneckerSum factors must be non-empty");
    eigen_assert(m_A.rows() == m_A.cols() && m_B.rows() == m_B.cols() && "KroneckerSum factors must be square");
    LhsOps::prepare(m_A);
    RhsOps::prepare(m_B);
  }

  EIGEN_DEVICE_FUNC Index rows() const { return m_A.rows() * m_B.rows(); }
  EIGEN_DEVICE_FUNC Index cols() const { return rows(); }

  /** \returns the left factor \c A. */
  const LhsMatrix& lhs() const { return m_A; }
  /** \returns the right factor \c B. */
  const RhsMatrix& rhs() const { return m_B; }

  /** \returns the coefficient at row \a row and column \a col:
   * \f$ A(i_1, j_1)\,\delta_{i_2 j_2} + \delta_{i_1 j_1} B(i_2, j_2) \f$ with
   * \c row = i1*n2 + i2 and \c col = j1*n2 + j2. */
  Scalar coeff(Index row, Index col) const {
    eigen_assert(row >= 0 && row < rows() && col >= 0 && col < cols());
    Scalar value;
    return internal::kron_factor_ops<KroneckerSum>::coeffIfStored(*this, row, col, value) ? value : Scalar(0);
  }

  /** \returns the transpose \f$ A^T \oplus B^T \f$, itself a Kronecker sum. */
  KroneckerSum<typename LhsOps::TransposedFactor, typename RhsOps::TransposedFactor> transpose() const {
    return {LhsOps::transposed(m_A), RhsOps::transposed(m_B)};
  }

  /** \returns the conjugate \f$ \bar A \oplus \bar B \f$, itself a Kronecker sum. */
  KroneckerSum conjugate() const { return {LhsOps::conjugated(m_A), RhsOps::conjugated(m_B)}; }

  /** \returns the adjoint \f$ A^H \oplus B^H \f$, itself a Kronecker sum. */
  KroneckerSum<typename LhsOps::TransposedFactor, typename RhsOps::TransposedFactor> adjoint() const {
    return {LhsOps::adjointed(m_A), RhsOps::adjointed(m_B)};
  }

  /** \returns the solution of \c (*this) * x = b through a \ref BartelsStewart
   * solver set up for this call; construct the solver once instead to reuse its
   * Schur forms across calls. */
  template <typename Rhs>
  Matrix<Scalar, ColsAtCompileTime, Rhs::ColsAtCompileTime> solve(const MatrixBase<Rhs>& b) const {
    const BartelsStewart<KroneckerSum> solver(*this);
    return solver.solve(b);
  }

  /** \returns the eigenvalues in Kronecker order: entry \c i*n2 + j is
   * \f$ \lambda_i(A) + \mu_j(B) \f$, from one eigenvalue solve per factor
   * (recursively for a Kronecker-sum factor, and for a Kronecker factor, whose
   * own factors must then be square), matching column \c i*n2 + j of
   * \ref eigenvectors. The set is not sorted.
   *
   * Each sum carries the errors of the factor eigenvalues, of order
   * \f$ \epsilon\,(\kappa(\lambda_i) \|A\| + \kappa(\mu_j) \|B\|) \f$ for simple
   * eigenvalues with condition numbers \f$ \kappa \f$, where a dense eigensolver
   * of \f$ A \oplus B \f$ errs by
   * \f$ \epsilon\,\kappa(\lambda_i)\,\kappa(\mu_j)\,\|A \oplus B\| \f$. The sums
   * are thus the more accurate for ill-conditioned or defective factors, whose
   * Jordan blocks lengthen in the sum, and the less accurate when a shift
   * splits across the factors, as in \f$ (A + cI) \oplus (B - cI) \f$. */
  ComplexVector eigenvalues() const {
    const ComplexVector lambda = LhsSpectrum::eigenvalues(m_A), mu = RhsSpectrum::eigenvalues(m_B);
    return (mu.replicate(fix<1>, lambda.size()) + lambda.transpose().replicate(mu.size(), fix<1>)).reshaped();
  }

  /** \returns the matrix of eigenvectors \f$ V_A \otimes V_B \f$ -- a
   * \ref KroneckerOperator, never materialized -- since
   * \f$ (A \oplus B)(v_i \otimes w_j) = (\lambda_i + \mu_j)(v_i \otimes w_j) \f$:
   * column \c i*n2 + j matches \c eigenvalues()[i*n2 + j]. Assign it to a dense
   * matrix to materialize. */
  KroneckerOperator<typename LhsSpectrum::Eigenvectors, typename RhsSpectrum::Eigenvectors> eigenvectors() const {
    return {LhsSpectrum::eigenvectors(m_A), RhsSpectrum::eigenvectors(m_B)};
  }

  /** \returns the product expression \c (*this) * \a x, evaluated through
   * \f$ \mathrm{mat}(y) = B\,\mathrm{mat}(x) + \mathrm{mat}(x)\,A^T \f$. */
  template <typename Rhs>
  Product<KroneckerSum, Rhs> operator*(const MatrixBase<Rhs>& x) const {
    EIGEN_STATIC_ASSERT(ColsAtCompileTime == Dynamic || Rhs::RowsAtCompileTime == Dynamic ||
                            int(ColsAtCompileTime) == int(Rhs::RowsAtCompileTime),
                        INVALID_MATRIX_PRODUCT)
    eigen_assert(x.rows() == cols() && "invalid product: dimensions do not match");
    return Product<KroneckerSum, Rhs>(*this, x.derived());
  }

  /** \internal Computes \c dst += alpha * (*this) * rhs as
   * \f$ \mathrm{mat}(y) \mathrel{+}= \alpha (B X + X A^T) \f$, \f$ X = \mathrm{mat}(x) \f$:
   * in place for a single right-hand side, and for several on the stacked layout
   * of KroneckerOperator::addProduct(), one product per factor per batch. */
  template <typename Dest, typename Rhs, typename ProductScalar>
  void addProduct(Dest& dst, const Rhs& rhs, const ProductScalar& alpha) const {
    using ProductMatrix = Matrix<ProductScalar, Dynamic, Dynamic, ColMajor>;
    const Index n1 = m_A.rows(), n2 = m_B.rows(), r = rhs.cols();
    eigen_assert(rhs.rows() == n1 * n2 && "invalid product: dimensions do not match");
    typename internal::nested_eval<Rhs, 1>::type actualRhs(rhs);
    ProductMatrix X, Y, work;
    const Index chunk = internal::kron_rhs_chunk<ProductScalar>(2 * n1 * n2);
    for (Index k0 = 0; k0 < r; k0 += chunk) {
      const Index c = numext::mini(chunk, r - k0);
      if (c == 1) {
        const auto Xk = actualRhs.col(k0).reshaped(n2, n1);
        auto Yk = dst.col(k0).reshaped(n2, n1);
        RhsOps::addLeftProduct(Yk, alpha, m_B, Xk);
        LhsOps::addRightProduct(Yk, alpha, Xk, m_A, work);
        continue;
      }
      internal::kron_stack_columns(X, actualRhs.middleCols(k0, c), n2, n1);
      Y.setZero(n2 * c, n1);
      auto Yflat = Y.reshaped(n2, c * n1);
      RhsOps::addLeftProduct(Yflat, ProductScalar(1), m_B, X.reshaped(n2, c * n1));
      LhsOps::addRightProduct(Y, ProductScalar(1), X, m_A, work);
      for (Index k = 0; k < c; ++k) dst.col(k0 + k).reshaped(n2, n1) += alpha * Y.middleRows(k * n2, n2);
    }
  }

  /** \internal Writes the representation into \a dst, dense or sparse according
   * to the destination's storage kind; invoked through \c dense = sum; and
   * \c sparse = sum;. */
  template <typename Dest>
  void evalTo(Dest& dst) const {
    evalToImpl(dst, IsSparseDestination<Dest>());
  }

  /** \internal Computes \c dst += (*this), see evalTo(). */
  template <typename Dest>
  void addTo(Dest& dst) const {
    addToImpl(dst, Scalar(1), IsSparseDestination<Dest>());
  }

  /** \internal Computes \c dst -= (*this), see evalTo(). */
  template <typename Dest>
  void subTo(Dest& dst) const {
    addToImpl(dst, Scalar(-1), IsSparseDestination<Dest>());
  }

 private:
  template <typename Dest>
  using IsSparseDestination = std::is_same<typename internal::traits<Dest>::StorageKind, Sparse>;

  template <typename Dest>
  void evalToImpl(Dest& dst, std::false_type) const {
    dst.setZero();
    addToImpl(dst, Scalar(1), std::false_type());
  }

  /** \internal dst += s (A (x) I + I (x) B): each stored entry of A on the
   * diagonal of its block, B on every diagonal block. */
  template <typename Dest>
  void addToImpl(Dest& dst, const Scalar& s, std::false_type) const {
    const Index n1 = m_A.rows(), n2 = m_B.rows();
    LhsOps::forEachNonZero(m_A, [&dst, &s, n2](Index i, Index j, const Scalar& a) {
      dst.block(i * n2, j * n2, n2, n2).diagonal().array() += s * a;
    });
    const auto& B = RhsOps::blockOperand(m_B);
    for (Index k = 0; k < n1; ++k) dst.block(k * n2, k * n2, n2, n2) += s * B;
  }

  /** \internal The sparse representation, as the sum of A (x) I and I (x) B
   * materialized with exactly reserved inner vectors: inner vector k1*n2 + k2 of
   * A (x) I holds the entries of inner vector k1 of A, of I (x) B those of inner
   * vector k2 of B. Both visitors insert each inner vector in increasing inner
   * index, see kron_factor_ops. */
  template <typename Dest>
  void evalToImpl(Dest& S, std::true_type) const {
    using IndexVector = Matrix<Index, Dynamic, 1>;
    using LhsVisitable = internal::kron_factor_visitable<LhsMatrix>;
    using RhsVisitable = internal::kron_factor_visitable<RhsMatrix>;
    using VisitedLhsOps = internal::kron_factor_ops<typename LhsVisitable::type>;
    using VisitedRhsOps = internal::kron_factor_ops<typename RhsVisitable::type>;
    const Index n1 = m_A.rows(), n2 = m_B.rows();
    const auto& A = LhsVisitable::get(m_A);
    const auto& B = RhsVisitable::get(m_B);
    const IndexVector nnzA = VisitedLhsOps::innerNonZeros(A, Dest::IsRowMajor);
    const IndexVector nnzB = VisitedRhsOps::innerNonZeros(B, Dest::IsRowMajor);
    Dest SA(rows(), cols()), SB(rows(), cols());
    SA.reserve(IndexVector(nnzA.transpose().replicate(n2, fix<1>).reshaped()));
    VisitedLhsOps::forEachNonZero(A, [&SA, n2](Index i, Index j, const Scalar& a) {
      for (Index k = 0; k < n2; ++k) SA.insert(i * n2 + k, j * n2 + k) = a;
    });
    SB.reserve(IndexVector(nnzB.replicate(n1, fix<1>)));
    for (Index k = 0; k < n1; ++k)
      VisitedRhsOps::forEachNonZero(
          B, [&SB, n2, k](Index i, Index j, const Scalar& b) { SB.insert(k * n2 + i, k * n2 + j) = b; });
    S = SA + SB;
  }

  template <typename Dest>
  void addToImpl(Dest& dst, const Scalar& s, std::true_type) const {
    typename Dest::PlainObject sum;
    evalTo(sum);
    if (s == Scalar(1))
      dst += sum;
    else
      dst -= sum;
  }

  LhsMatrix m_A;
  RhsMatrix m_B;
};

/** \ingroup StructuredMatrices_Module
 * \returns the \ref KroneckerSum \c a (+) \c b, with the factor types deduced as
 * for makeKroneckerOperator(). */
template <typename LhsDerived, typename RhsDerived>
KroneckerSum<typename internal::kron_factor_storage<LhsDerived>::type,
             typename internal::kron_factor_storage<RhsDerived>::type>
makeKroneckerSum(const EigenBase<LhsDerived>& a, const EigenBase<RhsDerived>& b) {
  return {a.derived(), b.derived()};
}

/** \ingroup StructuredMatrices_Module
 * \returns the \ref KroneckerSum \c a (+) \c b (+) \c c (+) ..., nested to the
 * right: \c makeKroneckerSum(a, makeKroneckerSum(b, c, ...)). */
template <typename D1, typename D2, typename D3, typename... Rest>
auto makeKroneckerSum(const EigenBase<D1>& a, const EigenBase<D2>& b, const EigenBase<D3>& c, const Rest&... rest) {
  return makeKroneckerSum(a, makeKroneckerSum(b, c, rest...));
}

/** \ingroup StructuredMatrices_Module
 * \class BartelsStewart
 * \brief Direct solver for Kronecker-sum systems
 * \f$ (A_1 \oplus A_2 \oplus \cdots \oplus A_d)\,x = b \f$.
 *
 * Flattening the (possibly nested) \ref KroneckerSum into its non-sum factors
 * \f$ A_k = Q_k T_k Q_k^H \f$ (complex Schur forms, \f$ Q_k \f$ unitary,
 * \f$ T_k \f$ upper triangular) gives
 * \f[ A_1 \oplus \cdots \oplus A_d = Q\,(T_1 \oplus \cdots \oplus T_d)\,Q^H,
 *     \qquad Q = Q_1 \otimes \cdots \otimes Q_d, \f]
 * so a solve applies \f$ Q^H \f$, back-substitutes the upper triangular
 * \f$ T_1 \oplus \cdots \oplus T_d \f$ and applies \f$ Q \f$. Both transforms
 * are one product per factor along its own index; the back substitution
 * recurses over the factors with accumulated shifts,
 * \f[ \big((\sigma + T_1(i,i)) I + T_2 \oplus \cdots \oplus T_d\big)\,y_i
 *     = c_i - \textstyle\sum_{j > i} T_1(i,j)\,y_j, \f]
 * the Bartels-Stewart algorithm [1] for two factors. When every factor is
 * exactly Hermitian, the Schur forms are the eigendecompositions, \f$ T_k \f$
 * is real diagonal, and the triangular solve is a division by the eigenvalue
 * sums -- the fast diagonalization method [2], in real arithmetic for real
 * factors. Either way the setup costs one \f$ O(n_k^3) \f$ decomposition per
 * factor, each solve \f$ O(N \sum_k n_k) \f$ per right-hand side,
 * \f$ N = \prod_k n_k \f$; sparse factors are densified for the decomposition.
 * \c transpose().solve() and \c adjoint().solve() reuse the decompositions:
 * \f[ (A_1 \oplus \cdots \oplus A_d)^H = Q\,(T_1^H \oplus \cdots \oplus T_d^H)\,Q^H \f]
 * is solved with the same transforms around a forward substitution, and
 * \f$ M^T x = b \f$ as \f$ M^H \bar x = \bar b \f$.
 *
 * The system is singular exactly when some sum
 * \f$ \lambda_{i_1}(A_1) + \cdots + \lambda_{i_d}(A_d) \f$ vanishes. As with
 * \c PartialPivLU nothing detects it: the computed sum is typically of order
 * \f$ \epsilon \sum_k \|A_k\| \f$ rather than zero, and the solution huge; it
 * is non-finite when a sum vanishes exactly, as it can for diagonal,
 * triangular or identity factors, whose decompositions are exact.
 *
 * The solve is backward stable relative to the factors, with residual
 * \f$ \|b - Mx\| = O\big((\sum_k n_k)\,\epsilon\,(\sum_k \|A_k\|)\,\|x\|\big) \f$.
 * That is relative to \f$ \sum_k \|A_k\| \f$, not \f$ \|M\| \f$; with a shift
 * split across the factors, as in \f$ (A + cI) \oplus (B - cI) \f$, the first
 * grows with \f$ c \f$ and the second does not.
 *
 * \c info() reports \c InvalidInput for a non-finite factor and
 * \c NoConvergence when a Schur or eigenvalue iteration fails; the solve then
 * returns NaN.
 *
 * \tparam KroneckerSumType the \ref KroneckerSum type to solve with.
 *
 * \sa class KroneckerSum
 */
template <typename KroneckerSumType>
class BartelsStewart : public SolverBase<BartelsStewart<KroneckerSumType>> {
 public:
  using Base = SolverBase<BartelsStewart>;
  friend class SolverBase<BartelsStewart>;
  EIGEN_GENERIC_PUBLIC_INTERFACE(BartelsStewart)
  EIGEN_STATIC_ASSERT_NON_INTEGER(RealScalar)
  using MatrixType = KroneckerSumType;
  using ComplexScalar = std::complex<RealScalar>;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using ComplexMatrix = Matrix<ComplexScalar, Dynamic, Dynamic, ColMajor>;
  using RealVector = Matrix<RealScalar, Dynamic, 1>;

  /** Default constructor; call \ref compute before \ref solve. */
  BartelsStewart() = default;

  /** Computes the factor decompositions of \a op. */
  explicit BartelsStewart(const KroneckerSumType& op) { compute(op); }

  /** Computes the Schur forms (eigendecompositions, when every factor is
   * exactly Hermitian) of the non-sum factors of \a op. */
  BartelsStewart& compute(const KroneckerSumType& op) {
    std::vector<DenseMatrix> leaves;
    collectLeaves(op, leaves);
    const std::size_t d = leaves.size();
    m_sizes.resize(d);
    m_inner.resize(d);
    m_unitary.clear();
    m_triangular.clear();
    m_basis.clear();
    m_info = Success;
    m_hermitian = true;
    for (std::size_t k = 0; k < d; ++k) {
      m_sizes[k] = leaves[k].rows();
      if (!leaves[k].allFinite()) m_info = InvalidInput;
      m_hermitian = m_hermitian && leaves[k] == leaves[k].adjoint();
    }
    m_size = 1;
    for (std::size_t k = d; k-- > 0;) {
      m_inner[k] = m_size;
      m_size *= m_sizes[k];
    }
    m_isInitialized = true;
    if (m_info != Success) return *this;
    if (m_hermitian) {
      m_spectrum = RealVector::Zero(1);
      SelfAdjointEigenSolver<DenseMatrix> es;
      for (std::size_t k = 0; k < d; ++k) {
        es.compute(leaves[k]);
        if (es.info() != Success) m_info = NoConvergence;
        m_basis.push_back(es.eigenvectors());
        // Kronecker order: the new factor's index runs fastest.
        const RealVector& lambda = es.eigenvalues();
        const RealVector previous = m_spectrum;
        m_spectrum = (lambda.replicate(fix<1>, previous.size()) + previous.transpose().replicate(lambda.size(), fix<1>))
                         .reshaped();
      }
    } else {
      ComplexSchur<ComplexMatrix> schur;
      for (std::size_t k = 0; k < d; ++k) {
        schur.compute(leaves[k].template cast<ComplexScalar>());
        if (schur.info() != Success) m_info = NoConvergence;
        m_unitary.push_back(schur.matrixU());
        m_triangular.push_back(schur.matrixT());
      }
    }
    return *this;
  }

  Index rows() const noexcept { return m_size; }
  Index cols() const noexcept { return m_size; }

  /** \returns \c Success, \c InvalidInput for a non-finite factor, or
   * \c NoConvergence when a factor decomposition did not converge. */
  ComputationInfo info() const {
    eigen_assert(m_isInitialized && "BartelsStewart is not initialized.");
    return m_info;
  }

  /** \returns whether every factor is exactly Hermitian, so that the solve runs
   * the fast diagonalization method. */
  bool isHermitian() const {
    eigen_assert(m_isInitialized && "BartelsStewart is not initialized.");
    return m_hermitian;
  }

#ifdef EIGEN_PARSED_BY_DOXYGEN
  /** \returns the solution \c x of \c op * x = \a b, as a lazily evaluated
   * expression. Supports multiple right-hand sides.
   * \pre \ref compute has been called. */
  template <typename Rhs>
  inline const Solve<BartelsStewart, Rhs> solve(const MatrixBase<Rhs>& b) const;
#endif

#ifndef EIGEN_PARSED_BY_DOXYGEN
  template <typename RhsType, typename DstType>
  void _solve_impl(const RhsType& rhs, DstType& dst) const {
    solveImpl(rhs, dst, /*adjoint=*/false);
  }

  // M^T = conj(M^H): M^{-T} b = conj(M^{-H} conj(b)).
  template <bool Conjugate, typename RhsType, typename DstType>
  void _solve_impl_transposed(const RhsType& rhs, DstType& dst) const {
    constexpr bool ConjugateRhs = !Conjugate && NumTraits<Scalar>::IsComplex;
    solveImpl(rhs.template conjugateIf<ConjugateRhs>(), dst, /*adjoint=*/true);
    if (ConjugateRhs) dst = dst.conjugate();
  }
#endif

 private:
  /** \internal x = M^{-1} b, or M^{-H} b when \a adjoint; on the Hermitian path
   * M^H = M. */
  template <typename RhsType, typename DstType>
  void solveImpl(const RhsType& rhs, DstType& dst, bool adjoint) const {
    if (m_info != Success) {
      // No usable decompositions; the nested solves never see info().
      dst.setConstant(Scalar(NumTraits<RealScalar>::quiet_NaN()));
      return;
    }
    if (m_hermitian) {
      DenseMatrix W = rhs;
      applyBasis(W, m_basis, /*adjoint=*/true);
      W.array().colwise() /= m_spectrum.array();
      applyBasis(W, m_basis, /*adjoint=*/false);
      dst = W;
    } else {
      ComplexMatrix W = rhs.template cast<ComplexScalar>();
      applyBasis(W, m_unitary, /*adjoint=*/true);
      for (Index j = 0; j < W.cols(); ++j) {
        if (adjoint)
          adjointTriangularSolve(0, ComplexScalar(0), W.col(j).data());
        else
          triangularSolve(0, ComplexScalar(0), W.col(j).data());
      }
      applyBasis(W, m_unitary, /*adjoint=*/false);
      dst = internal::structured_scalar_part_impl<Scalar>::run(W);
    }
  }

  template <typename Factor>
  static void collectLeaves(const Factor& f, std::vector<DenseMatrix>& leaves) {
    collectLeaves(f, leaves, internal::kron_factor_is_kronecker_sum<Factor>());
  }
  template <typename Factor>
  static void collectLeaves(const Factor& f, std::vector<DenseMatrix>& leaves, std::true_type) {
    collectLeaves(f.lhs(), leaves);
    collectLeaves(f.rhs(), leaves);
  }
  template <typename Factor>
  static void collectLeaves(const Factor& f, std::vector<DenseMatrix>& leaves, std::false_type) {
    leaves.push_back(DenseMatrix(internal::kron_factor_ops<Factor>::denseFactor(f)));
  }

  /** \internal W.col(j) <- (M_1 (x) ... (x) M_d) W.col(j) with M_k = Q_k^H when
   * \a adjoint, Q_k otherwise, one factor at a time: along factor k with inner
   * stride s = n_{k+1} ... n_d, every length n_k s block of the columns is an
   * s x n_k matrix V and the factor acts as V <- V M_k^T, or for s = 1 on all
   * blocks at once as U <- M_k U. */
  template <typename Work, typename Basis>
  void applyBasis(Work& W, const std::vector<Basis>& Q, bool adjoint) const {
    using WorkMatrix = Matrix<typename Work::Scalar, Dynamic, Dynamic, ColMajor>;
    WorkMatrix T;
    for (std::size_t k = 0; k < Q.size(); ++k) {
      const Index n = m_sizes[k], s = m_inner[k], blocks = W.size() / (n * s);
      if (s == 1) {
        Map<WorkMatrix> U(W.data(), n, blocks);
        if (adjoint)
          T.noalias() = Q[k].adjoint() * U;
        else
          T.noalias() = Q[k] * U;
        U = T;
        continue;
      }
      for (Index b = 0; b < blocks; ++b) {
        Map<WorkMatrix> V(W.data() + b * n * s, s, n);
        if (adjoint)
          T.noalias() = V * Q[k].conjugate();
        else
          T.noalias() = V * Q[k].transpose();
        V = T;
      }
    }
  }

  /** \internal Solves (sigma I + T_k (+) ... (+) T_d) y = y in place for the
   * length n_k s_k segment \a y, viewed as the s_k x n_k matrix Y whose column i
   * belongs to index i of factor k; see the recurrence in the class
   * documentation. */
  void triangularSolve(std::size_t k, const ComplexScalar& sigma, ComplexScalar* y) const {
    const Index n = m_sizes[k], s = m_inner[k];
    const TriangularMatrix& T = m_triangular[k];
    if (k + 1 == m_sizes.size()) {
      // Last factor, s = 1: back substitution on the shifted T_d, by rows.
      Map<Matrix<ComplexScalar, 1, Dynamic>> Y(y, n);
      for (Index i = n - 1; i >= 0; --i) {
        const Index tail = n - 1 - i;
        Y(i) = (Y(i) - Y.tail(tail).cwiseProduct(T.row(i).tail(tail)).sum()) / (sigma + T(i, i));
      }
      return;
    }
    Map<ComplexMatrix> Y(y, s, n);
    for (Index i = n - 1; i >= 0; --i) {
      const Index tail = n - 1 - i;
      if (tail > 0) Y.col(i).noalias() -= Y.rightCols(tail) * T.row(i).tail(tail).transpose();
      triangularSolve(k + 1, sigma + T(i, i), Y.col(i).data());
    }
  }

  /** \internal The adjoint of triangularSolve: solves
   * (sigma I + T_k^H (+) ... (+) T_d^H) y = y in place by forward substitution,
   * row i of T_k^H being the conjugated column i of T_k. */
  void adjointTriangularSolve(std::size_t k, const ComplexScalar& sigma, ComplexScalar* y) const {
    const Index n = m_sizes[k], s = m_inner[k];
    const TriangularMatrix& T = m_triangular[k];
    if (k + 1 == m_sizes.size()) {
      // Last factor, s = 1: by columns of T_d^H, which are the rows of T_d,
      // contiguous and conjugated.
      Map<Matrix<ComplexScalar, 1, Dynamic>> Y(y, n);
      for (Index i = 0; i < n; ++i) {
        Y(i) /= sigma + numext::conj(T(i, i));
        const Index tail = n - 1 - i;
        Y.tail(tail) -= Y(i) * T.row(i).tail(tail).conjugate();
      }
      return;
    }
    Map<ComplexMatrix> Y(y, s, n);
    for (Index i = 0; i < n; ++i) {
      if (i > 0) Y.col(i).noalias() -= Y.leftCols(i) * T.col(i).head(i).conjugate();
      adjointTriangularSolve(k + 1, sigma + numext::conj(T(i, i)), Y.col(i).data());
    }
  }

  std::vector<Index> m_sizes, m_inner;
  Index m_size = 0;
  std::vector<DenseMatrix> m_basis;
  RealVector m_spectrum;
  // Row-major: the back substitution reads T_k by rows.
  using TriangularMatrix = Matrix<ComplexScalar, Dynamic, Dynamic, RowMajor>;
  std::vector<ComplexMatrix> m_unitary;
  std::vector<TriangularMatrix> m_triangular;
  bool m_hermitian = false;
  bool m_isInitialized = false;
  ComputationInfo m_info = InvalidInput;
};

namespace internal {

template <typename LhsMatrix, typename RhsMatrix, typename Rhs, int ProductTag>
struct generic_product_impl<KroneckerSum<LhsMatrix, RhsMatrix>, Rhs, StructuredShape, DenseShape, ProductTag>
    : structured_product_impl<KroneckerSum<LhsMatrix, RhsMatrix>, Rhs> {};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_STRUCTURED_KRONECKER_SUM_H
