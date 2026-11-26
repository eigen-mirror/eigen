// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Claire Maurice
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010,2012 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_COMPLEX_SCHUR_H
#define EIGEN_COMPLEX_SCHUR_H

#include "../Jacobi/Jacobi.h"
#include "./HessenbergDecomposition.h"

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {
template <typename MatrixType, bool IsComplex>
struct complex_schur_reduce_to_hessenberg;
}

template <typename MatrixType_>
class ComplexEigenSolver;

/** \eigenvalues_module \ingroup Eigenvalues_Module
 *
 *
 * \class ComplexSchur
 *
 * \brief Performs a complex Schur decomposition of a real or complex square matrix
 *
 * \tparam MatrixType_ the type of the matrix of which we are
 * computing the Schur decomposition; this is expected to be an
 * instantiation of the Matrix class template.
 *
 * Given a real or complex square matrix A, this class computes the
 * Schur decomposition: \f$ A = U T U^*\f$ where U is a unitary
 * complex matrix, and T is a complex upper triangular matrix.  The
 * diagonal of the matrix T corresponds to the eigenvalues of the
 * matrix A.
 *
 * Call the function compute() to compute the Schur decomposition of
 * a given matrix. Alternatively, you can use the
 * ComplexSchur(const MatrixType&, bool) constructor which computes
 * the Schur decomposition at construction time. Once the
 * decomposition is computed, you can use the matrixU() and matrixT()
 * functions to retrieve the matrices U and T in the decomposition.
 *
 * \note This code is inspired from Jampack
 *
 * \sa class RealSchur, class EigenSolver, class ComplexEigenSolver
 */
template <typename MatrixType_>
class ComplexSchur {
 public:
  using MatrixType = MatrixType_;
  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    Options = internal::plain_object_options<MatrixType>::value,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
  };

  /** \brief Scalar type for matrices of type \p MatrixType_. */
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Index = Eigen::Index;  ///< \deprecated since Eigen 3.3

  /** \brief Complex scalar type for \p MatrixType_.
   *
   * This is \c std::complex<Scalar> if #Scalar is real (e.g.,
   * \c float or \c double) and just \c Scalar if #Scalar is
   * complex.
   */
  using ComplexScalar = internal::make_complex_t<Scalar>;

  /** \brief Type for the matrices in the Schur decomposition.
   *
   * This is a square matrix with entries of type #ComplexScalar.
   * The size is the same as the size of \p MatrixType_.
   */
  using ComplexMatrixType =
      Matrix<ComplexScalar, RowsAtCompileTime, ColsAtCompileTime, Options, MaxRowsAtCompileTime, MaxColsAtCompileTime>;

  /** \brief Type of the matrix returned by matrixT().
   *
   * This is #ComplexMatrixType, or \p MatrixType_ itself when that is a Ref<>: the \link InplaceDecomposition
   * inplace decomposition \endlink then stores T in the referenced matrix. */
  using MatrixTType = std::conditional_t<internal::is_ref<MatrixType>::value, MatrixType, ComplexMatrixType>;

  /** \brief Default constructor.
   *
   * \param [in] size  Positive integer, size of the matrix whose Schur decomposition will be computed.
   *
   * The default constructor is useful in cases in which the user
   * intends to perform decompositions via compute().  The \p size
   * parameter is only used as a hint. It is not an error to give a
   * wrong \p size, but it may impair performance.
   *
   * \sa compute() for an example.
   */
  explicit ComplexSchur(Index size = RowsAtCompileTime == Dynamic ? 1 : RowsAtCompileTime)
      : m_matT(size, size),
        m_matU(size, size),
        m_hess(size),
        m_isInitialized(false),
        m_matUisUptodate(false),
        m_maxIters(-1) {}

  /** \brief Constructor; computes Schur decomposition of given matrix.
   *
   * \param[in]  matrix    Square matrix whose Schur decomposition is to be computed.
   * \param[in]  computeU  If true, both T and U are computed; if false, only T is computed.
   *
   * This constructor calls compute() to compute the Schur decomposition.
   *
   * \sa matrixT() and matrixU() for examples.
   */
  template <typename InputType>
  explicit ComplexSchur(const EigenBase<InputType>& matrix, bool computeU = true)
      : m_matT(matrix.rows(), matrix.cols()),
        m_matU(matrix.rows(), matrix.cols()),
        m_hess(matrix.rows()),
        m_isInitialized(false),
        m_matUisUptodate(false),
        m_maxIters(-1) {
    compute(matrix.derived(), computeU);
  }

  /** \brief Constructor for \link InplaceDecomposition inplace decomposition \endlink
   *
   * \param[in,out]  matrix    Square complex matrix whose Schur decomposition is to be computed.
   * \param[in]      computeU  If true, both T and U are computed; if false, only T is computed.
   *
   * When \p MatrixType is a Ref<>, the decomposition is computed within the memory of \p matrix, which then holds
   * the triangular matrix T returned by matrixT(); U is stored in the decomposition object. This overload is only
   * available for a complex Ref<>; owning matrix types use the constructor taking a const input.
   */
  template <typename InputType, bool IsRef = internal::is_ref<MatrixType>::value, std::enable_if_t<IsRef, int> = 0>
  explicit ComplexSchur(EigenBase<InputType>& matrix, bool computeU = true)
      : m_matT(matrix.derived()),
        m_matU(matrix.rows(), matrix.cols()),
        m_hess(matrix.rows()),
        m_isInitialized(false),
        m_matUisUptodate(false),
        m_maxIters(-1) {
    computeInPlace(computeU);
  }

  /** \brief Returns the unitary matrix in the Schur decomposition.
   *
   * \returns A const reference to the matrix U.
   *
   * It is assumed that either the constructor
   * ComplexSchur(const MatrixType& matrix, bool computeU) or the
   * member function compute(const MatrixType& matrix, bool computeU)
   * has been called before to compute the Schur decomposition of a
   * matrix, and that \p computeU was set to true (the default
   * value).
   *
   * Example: \include ComplexSchur_matrixU.cpp
   * Output: \verbinclude ComplexSchur_matrixU.out
   */
  const ComplexMatrixType& matrixU() const {
    eigen_assert(m_isInitialized && "ComplexSchur is not initialized.");
    eigen_assert(m_matUisUptodate && "The matrix U has not been computed during the ComplexSchur decomposition.");
    return m_matU;
  }

  /** \brief Returns the triangular matrix in the Schur decomposition.
   *
   * \returns A const reference to the matrix T.
   *
   * It is assumed that either the constructor
   * ComplexSchur(const MatrixType& matrix, bool computeU) or the
   * member function compute(const MatrixType& matrix, bool computeU)
   * has been called before to compute the Schur decomposition of a
   * matrix.
   *
   * Note that this function returns a plain square matrix. If you want to reference
   * only the upper triangular part, use:
   * \code schur.matrixT().triangularView<Upper>() \endcode
   *
   * Example: \include ComplexSchur_matrixT.cpp
   * Output: \verbinclude ComplexSchur_matrixT.out
   */
  const MatrixTType& matrixT() const {
    eigen_assert(m_isInitialized && "ComplexSchur is not initialized.");
    return m_matT;
  }

  /** \brief Computes Schur decomposition of given matrix.
    *
    * \param[in]  matrix  Square matrix whose Schur decomposition is to be computed.
    * \param[in]  computeU  If true, both T and U are computed; if false, only T is computed.

    * \returns    Reference to \c *this
    *
    * The Schur decomposition is computed by first reducing the
    * matrix to Hessenberg form using the class
    * HessenbergDecomposition. The Hessenberg matrix is then reduced
    * to triangular form by performing QR iterations with a single
    * shift. The cost of computing the Schur decomposition depends
    * on the number of iterations; as a rough guide, it may be taken
    * to be \f$25n^3\f$ complex flops, or \f$10n^3\f$ complex flops
    * if \a computeU is false.
    *
    * Example: \include ComplexSchur_compute.cpp
    * Output: \verbinclude ComplexSchur_compute.out
    *
    * \sa compute(const MatrixType&, bool, Index)
    */
  template <typename InputType>
  ComplexSchur& compute(const EigenBase<InputType>& matrix, bool computeU = true);

  /** \brief Compute Schur decomposition from a given Hessenberg matrix
   *  \param[in] matrixH Matrix in Hessenberg form H
   *  \param[in] matrixQ orthogonal matrix Q that transform a matrix A to H : A = Q H Q^T
   *  \param computeU Computes the matrix U of the Schur vectors
   * \return Reference to \c *this
   *
   *  This routine assumes that the matrix is already reduced in Hessenberg form matrixH
   *  using either the class HessenbergDecomposition or another mean.
   *  It computes the upper quasi-triangular matrix T of the Schur decomposition of H
   *  When computeU is true, this routine computes the matrix U such that
   *  A = U T U^T =  (QZ) T (QZ)^T = Q H Q^T where A is the initial matrix
   *
   * NOTE Q is referenced if computeU is true; so, if the initial orthogonal matrix
   * is not available, the user should give an identity matrix (Q.setIdentity())
   *
   * \sa compute(const MatrixType&, bool)
   */
  template <typename HessMatrixType, typename OrthMatrixType>
  ComplexSchur& computeFromHessenberg(const HessMatrixType& matrixH, const OrthMatrixType& matrixQ,
                                      bool computeU = true);

  /** \brief Reorders the diagonal of the Schur form so that a selected set of eigenvalues comes first
   *
   * \param[in] select  Callable taking an eigenvalue and returning whether it belongs to the leading set.
   *
   * \returns The number of selected eigenvalues.
   *
   * The reordering applies unitary similarity transformations to the stored decomposition, so that
   * \f$ A = U T U^* \f$ still holds and the selected eigenvalues lead the diagonal of \f$ T \f$, in the order the
   * decomposition gave them. With \f$ m \f$ selected eigenvalues, the leading \f$ m \f$ columns of #matrixU() span the
   * invariant subspace belonging to them:
   * \f[
   *   A U_1 = U_1 T_{11}, \qquad T = \begin{pmatrix} T_{11} & T_{12} \\ 0 & T_{22} \end{pmatrix},
   * \f]
   * where \f$ T_{11} \f$ is the leading \f$ m \times m \f$ block of \f$ T \f$.
   *
   * This is the reordering that LAPACK's \c xTREXC provides for the complex Schur form. Each selected eigenvalue is
   * moved up by adjacent transpositions, each one a Givens rotation on a 2x2 block, for a total cost of \f$ O(n^2) \f$
   * rotations of \f$ O(n) \f$ work.
   *
   * Only #matrixT() is reordered when the decomposition was computed with \c computeU = \c false. This method
   * requires that #compute() or the constructor has completed successfully, that is #info() must be \c Success, but
   * not that \c computeU was set.
   *
   * \code
   * ComplexSchur<MatrixXcd> schur(A);
   * const Index stable = schur.reorder([](const std::complex<double>& lambda) { return lambda.real() < 0; });
   * MatrixXcd basis = schur.matrixU().leftCols(stable);  // spans the stable invariant subspace
   * \endcode
   */
  template <typename SelectType>
  Index reorder(SelectType select);

  /** \brief Reports whether previous computation was successful.
   *
   * \returns \c Success if computation was successful, \c NoConvergence otherwise.
   */
  ComputationInfo info() const {
    eigen_assert(m_isInitialized && "ComplexSchur is not initialized.");
    return m_info;
  }

  /** \brief Sets the maximum number of iterations allowed.
   *
   * If not specified by the user, the maximum number of iterations is m_maxIterationsPerRow times the size
   * of the matrix.
   */
  ComplexSchur& setMaxIterations(Index maxIters) {
    m_maxIters = maxIters;
    return *this;
  }

  /** \brief Returns the maximum number of iterations. */
  Index getMaxIterations() const { return m_maxIters; }

  /** \brief Maximum number of iterations per row.
   *
   * If not otherwise specified, the maximum number of iterations is this number times the size of the
   * matrix. It is currently set to 30.
   */
  static const int m_maxIterationsPerRow = 30;

 protected:
  EIGEN_STATIC_ASSERT(NumTraits<Scalar>::IsComplex || !internal::is_ref<MatrixType>::value,
                      INPLACE_COMPLEXSCHUR_REQUIRES_A_COMPLEX_MATRIX_TYPE)

  MatrixTType m_matT;
  ComplexMatrixType m_matU;
  internal::complex_schur_reduce_to_hessenberg<MatrixType, NumTraits<Scalar>::IsComplex> m_hess;
  ComputationInfo m_info;
  bool m_isInitialized;
  bool m_matUisUptodate;
  Index m_maxIters;

 private:
  // ComplexEigenSolver computes the eigenvectors of T within this storage.
  friend class ComplexEigenSolver<MatrixType>;

  ComplexSchur& computeInPlace(bool computeU);
  void swapAdjacentEigenvalues(Index k);
  bool subdiagonalEntryIsNeglegible(Index i);
  ComplexScalar computeShift(Index iu, Index iter);
  void reduceToTriangularForm(bool computeU);
  friend struct internal::complex_schur_reduce_to_hessenberg<MatrixType, NumTraits<Scalar>::IsComplex>;
};

/** If m_matT(i+1,i) is negligible in floating point arithmetic
 * compared to m_matT(i,i) and m_matT(j,j), then set it to zero and
 * return true, else return false. */
template <typename MatrixType>
inline bool ComplexSchur<MatrixType>::subdiagonalEntryIsNeglegible(Index i) {
  RealScalar d = numext::norm1(m_matT.coeff(i, i)) + numext::norm1(m_matT.coeff(i + 1, i + 1));
  RealScalar sd = numext::norm1(m_matT.coeff(i + 1, i));
  if (internal::isMuchSmallerThan(sd, d, NumTraits<RealScalar>::epsilon())) {
    m_matT.coeffRef(i + 1, i) = ComplexScalar(0);
    return true;
  }
  return false;
}

/** Compute the shift in the current QR iteration. */
template <typename MatrixType>
typename ComplexSchur<MatrixType>::ComplexScalar ComplexSchur<MatrixType>::computeShift(Index iu, Index iter) {
  if ((iter == 10 || iter == 20) && iu > 1) {
    // exceptional shift, taken from http://www.netlib.org/eispack/comqr.f
    return ComplexSchur<MatrixType>::ComplexScalar(numext::abs(numext::real(m_matT.coeff(iu, iu - 1))) +
                                                   numext::abs(numext::real(m_matT.coeff(iu - 1, iu - 2))));
  }

  // compute the shift as one of the eigenvalues of t, the 2x2
  // diagonal block on the bottom of the active submatrix
  Matrix<ComplexScalar, 2, 2> t = m_matT.template block<2, 2>(iu - 1, iu - 1);
  const RealScalar normt = t.cwiseAbs().sum();
  const auto factors = internal::safe_scaling<RealScalar>::scale_to(t, t, normt);

  ComplexScalar b = t.coeff(0, 1) * t.coeff(1, 0);
  ComplexScalar c = t.coeff(0, 0) - t.coeff(1, 1);
  ComplexScalar disc = numext::sqrt(c * c + RealScalar(4) * b);
  ComplexScalar det = t.coeff(0, 0) * t.coeff(1, 1) - b;
  ComplexScalar trace = t.coeff(0, 0) + t.coeff(1, 1);
  ComplexScalar eival1 = (trace + disc) / RealScalar(2);
  ComplexScalar eival2 = (trace - disc) / RealScalar(2);
  RealScalar eival1_norm = numext::norm1(eival1);
  RealScalar eival2_norm = numext::norm1(eival2);
  // A division by zero can only occur if eival1==eival2==0.
  // In this case, det==0, and all we have to do is checking that eival2_norm!=0
  if (eival1_norm > eival2_norm)
    eival2 = det / eival1;
  else if (!numext::is_exactly_zero(eival2_norm))
    eival1 = det / eival2;

  // choose the eigenvalue closest to the bottom entry of the diagonal
  ComplexScalar shift = numext::norm1(eival1 - t.coeff(1, 1)) < numext::norm1(eival2 - t.coeff(1, 1)) ? eival1 : eival2;
  internal::safe_scaling<RealScalar>::unscale_in_place(shift, factors);
  return shift;
}

template <typename MatrixType>
template <typename InputType>
ComplexSchur<MatrixType>& ComplexSchur<MatrixType>::compute(const EigenBase<InputType>& matrix, bool computeU) {
  m_matUisUptodate = false;
  EIGEN_INTERNAL_ASSERT_SQUARE(matrix);

  if (matrix.cols() <= 1) {
    m_matT = matrix.derived().template cast<ComplexScalar>();
    if (computeU) m_matU = ComplexMatrixType::Identity(matrix.rows(), matrix.cols());
    m_info = Success;
    m_isInitialized = true;
    m_matUisUptodate = computeU;
    return *this;
  }

  // Reduce to Hessenberg form at unit scale, as RealSchur does: HessenbergDecomposition treats a subdiagonal tail
  // whose squared norm underflows as already zero, so an unscaled matrix near the bottom of the exponent range loses
  // its whole subdiagonal. For binary scalars the scale is the power of two just below the largest real or imaginary
  // part, clamped so that it and its reciprocal are normal; the modulus of a finite coefficient can overflow. When
  // maxCoeff < min/eps, both steps round through integer significands, which FTZ/DAZ and ARMv7 NEON cannot flush. A
  // scale above one underflows coefficients more than the exponent range below the largest one, a perturbation bounded
  // by the smallest subnormal relative to that largest coefficient. maxCoeff propagates NaN; a zero or non-finite
  // maxCoeff carries no usable exponent and is left unscaled.
  const RealScalar maxCoeff = internal::safe_scaling<RealScalar>::max_abs_component(matrix.derived());
  const internal::safe_scaling_factors<RealScalar> factors = internal::safe_scaling<RealScalar>::with_scaled(
      matrix.derived(), maxCoeff, [&](const auto& scaled) { m_hess.run(*this, scaled, computeU); });
  computeFromHessenberg(m_matT, m_matU, computeU);
  // m_matU is unitary either way; only the triangular factor carries the scale.
  internal::safe_scaling<RealScalar>::unscale_in_place(m_matT, maxCoeff, factors);
  return *this;
}

/** \internal Computes the Schur decomposition of the complex matrix held in m_matT, which is overwritten by T. */
template <typename MatrixType>
ComplexSchur<MatrixType>& ComplexSchur<MatrixType>::computeInPlace(bool computeU) {
  m_matUisUptodate = false;
  EIGEN_INTERNAL_ASSERT_SQUARE(m_matT);

  if (m_matT.cols() <= 1) {
    if (computeU) m_matU = ComplexMatrixType::Identity(m_matT.rows(), m_matT.cols());
    m_info = Success;
    m_isInitialized = true;
    m_matUisUptodate = computeU;
    return *this;
  }

  const RealScalar maxCoeff = internal::safe_scaling<RealScalar>::max_abs_component(m_matT);
  const internal::safe_scaling_factors<RealScalar> factors =
      internal::safe_scaling<RealScalar>::scale_in_place(m_matT, maxCoeff);
  m_hess.runInPlace(*this, computeU);
  computeFromHessenberg(m_matT, m_matU, computeU);
  internal::safe_scaling<RealScalar>::unscale_in_place(m_matT, maxCoeff, factors);
  return *this;
}

template <typename MatrixType>
template <typename HessMatrixType, typename OrthMatrixType>
ComplexSchur<MatrixType>& ComplexSchur<MatrixType>::computeFromHessenberg(const HessMatrixType& matrixH,
                                                                          const OrthMatrixType& matrixQ,
                                                                          bool computeU) {
  if (!internal::is_same_dense(m_matT, matrixH)) m_matT = matrixH;
  if (computeU && !internal::is_same_dense(m_matU, matrixQ)) m_matU = matrixQ;
  reduceToTriangularForm(computeU);
  return *this;
}

/** \internal Swaps the eigenvalues at diagonal positions \p k and \p k+1.
 *
 * For T(k:k+2,k:k+2) = [a b; 0 c] the eigenvector of c is [b; c-a], so the rotation J whose first column has that
 * direction satisfies J^* [a b; 0 c] J = [c b; 0 a], as in LAPACK's xTREXC. Building it with makeGivens from b and
 * c-a rather than from a normalized eigenvector avoids dividing by |b| when b is small. Both are divided by the
 * largest entry first, so a diagonal difference near the overflow threshold cannot form an infinity. */
template <typename MatrixType>
void ComplexSchur<MatrixType>::swapAdjacentEigenvalues(Index k) {
  const ComplexScalar first = m_matT.coeff(k, k);
  const ComplexScalar second = m_matT.coeff(k + 1, k + 1);
  const ComplexScalar coupling = m_matT.coeff(k, k + 1);
  const RealScalar magnitude = numext::maxi(
      numext::maxi(numext::abs(first), numext::abs(second)), numext::abs(coupling));
  // An uncoupled block with equal diagonal entries is a multiple of the identity, which the swap leaves unchanged.
  if (numext::is_exactly_zero(magnitude)) return;
  const ComplexScalar scaledCoupling = coupling / magnitude;
  const ComplexScalar scaledSpread = second / magnitude - first / magnitude;
  if (numext::is_exactly_zero(scaledCoupling) && numext::is_exactly_zero(scaledSpread)) return;
  JacobiRotation<ComplexScalar> rotation;
  rotation.makeGivens(scaledCoupling, scaledSpread);
  // As in xTREXC, rotate only the entries outside the 2x2 block, then exchange its diagonal entries exactly. The
  // block's off-diagonal entry is unchanged by J^* [a b; 0 c] J, and row k has no entries left of column k.
  if (k > 0) m_matT.topRows(k).applyOnTheRight(k, k + 1, rotation);
  if (k + 2 < m_matT.cols())
    m_matT.rightCols(m_matT.cols() - k - 2).applyOnTheLeft(k, k + 1, rotation.adjoint());
  m_matT.coeffRef(k, k) = second;
  m_matT.coeffRef(k + 1, k + 1) = first;
  m_matT.coeffRef(k + 1, k) = ComplexScalar(0);
  if (m_matUisUptodate) m_matU.applyOnTheRight(k, k + 1, rotation);
}

template <typename MatrixType>
template <typename SelectType>
Index ComplexSchur<MatrixType>::reorder(SelectType select) {
  eigen_assert(m_isInitialized && "ComplexSchur is not initialized.");
  eigen_assert(m_info == Success && "ComplexSchur::reorder requires a successful decomposition.");
  const Index n = m_matT.rows();
  // Invariant at the top of each iteration: positions below k are untouched, so T(k,k) is still the k-th eigenvalue
  // of the unreordered form and the first `leading` positions already hold the selected ones.
  Index leading = 0;
  for (Index k = 0; k < n; ++k) {
    if (!select(m_matT.coeff(k, k))) continue;
    for (Index j = k; j > leading; --j) swapAdjacentEigenvalues(j - 1);
    ++leading;
  }
  return leading;
}

namespace internal {

/* Reduce given matrix to Hessenberg form, and hold the workspace this takes. */
template <typename MatrixType, bool IsComplex>
struct complex_schur_reduce_to_hessenberg {
  // this is the implementation for the case IsComplex = true: the reduction runs in place in m_matT.
  using Schur = ComplexSchur<MatrixType>;
  using Scalar = typename Schur::ComplexScalar;
  using CoeffVectorType = Matrix<Scalar, Schur::RowsAtCompileTime == Dynamic ? Dynamic : Schur::RowsAtCompileTime - 1,
                                 1, Schur::Options & ~RowMajor,
                                 Schur::MaxRowsAtCompileTime == Dynamic ? Dynamic : Schur::MaxRowsAtCompileTime - 1, 1>;
  using WorkspaceType =
      Matrix<Scalar, Schur::ColsAtCompileTime, 1, Schur::Options & ~RowMajor, Schur::MaxColsAtCompileTime, 1>;

  explicit complex_schur_reduce_to_hessenberg(Index size) : m_workspace(size) {
    if (size > 1) m_hCoeffs.resize(size - 1);
  }

  template <typename InputType>
  void run(Schur& _this, const InputType& matrix, bool computeU) {
    _this.m_matT = matrix;
    runInPlace(_this, computeU);
  }

  void runInPlace(Schur& _this, bool computeU) {
    hessenberg_decomposition_inplace(_this.m_matT, m_hCoeffs, m_workspace, _this.m_matU, computeU);
  }

  CoeffVectorType m_hCoeffs;
  WorkspaceType m_workspace;
};

template <typename MatrixType>
struct complex_schur_reduce_to_hessenberg<MatrixType, false> {
  using Schur = ComplexSchur<MatrixType>;

  explicit complex_schur_reduce_to_hessenberg(Index size) : m_hess(size) {}

  template <typename InputType>
  void run(Schur& _this, const InputType& matrix, bool computeU) {
    using ComplexScalar = typename Schur::ComplexScalar;

    // Note: m_hess is over RealScalar; m_matT and m_matU is over ComplexScalar
    m_hess.compute(matrix);
    _this.m_matT = m_hess.matrixH().template cast<ComplexScalar>();
    if (computeU) {
      // TODO: this temporary allocation could potentially be avoided.
      MatrixType Q = m_hess.matrixQ();
      _this.m_matU = Q.template cast<ComplexScalar>();
    }
  }

  HessenbergDecomposition<MatrixType> m_hess;
};

}  // end namespace internal

// Reduce the Hessenberg matrix m_matT to triangular form by QR iteration.
template <typename MatrixType>
void ComplexSchur<MatrixType>::reduceToTriangularForm(bool computeU) {
  Index maxIters = m_maxIters;
  if (maxIters == -1) maxIters = m_maxIterationsPerRow * m_matT.rows();

  // The matrix m_matT is divided in three parts.
  // Rows 0,...,il-1 are decoupled from the rest because m_matT(il,il-1) is zero.
  // Rows il,...,iu is the part we are working on (the active submatrix).
  // Rows iu+1,...,end are already brought in triangular form.
  Index iu = m_matT.cols() - 1;
  Index il;
  Index iter = 0;       // number of iterations we are working on the (iu,iu) element
  Index totalIter = 0;  // number of iterations for whole matrix

  while (true) {
    // find iu, the bottom row of the active submatrix
    while (iu > 0) {
      if (!subdiagonalEntryIsNeglegible(iu - 1)) break;
      iter = 0;
      --iu;
    }

    // if iu is zero then we are done; the whole matrix is triangularized
    if (iu == 0) break;

    // if we spent too many iterations, we give up
    iter++;
    totalIter++;
    if (totalIter > maxIters) break;

    // find il, the top row of the active submatrix
    il = iu - 1;
    while (il > 0 && !subdiagonalEntryIsNeglegible(il - 1)) {
      --il;
    }

    /* perform the QR step using Givens rotations. The first rotation
       creates a bulge; the (il+2,il) element becomes nonzero. This
       bulge is chased down to the bottom of the active submatrix. */

    ComplexScalar shift = computeShift(iu, iter);
    JacobiRotation<ComplexScalar> rot;
    rot.makeGivens(m_matT.coeff(il, il) - shift, m_matT.coeff(il + 1, il));
    m_matT.rightCols(m_matT.cols() - il).applyOnTheLeft(il, il + 1, rot.adjoint());
    m_matT.topRows((std::min)(il + 2, iu) + 1).applyOnTheRight(il, il + 1, rot);
    if (computeU) m_matU.applyOnTheRight(il, il + 1, rot);

    for (Index i = il + 1; i < iu; i++) {
      rot.makeGivens(m_matT.coeffRef(i, i - 1), m_matT.coeffRef(i + 1, i - 1), &m_matT.coeffRef(i, i - 1));
      m_matT.coeffRef(i + 1, i - 1) = ComplexScalar(0);
      m_matT.rightCols(m_matT.cols() - i).applyOnTheLeft(i, i + 1, rot.adjoint());
      m_matT.topRows((std::min)(i + 2, iu) + 1).applyOnTheRight(i, i + 1, rot);
      if (computeU) m_matU.applyOnTheRight(i, i + 1, rot);
    }
  }

  if (totalIter <= maxIters)
    m_info = Success;
  else
    m_info = NoConvergence;

  m_isInitialized = true;
  m_matUisUptodate = computeU;
}

}  // end namespace Eigen

#endif  // EIGEN_COMPLEX_SCHUR_H
