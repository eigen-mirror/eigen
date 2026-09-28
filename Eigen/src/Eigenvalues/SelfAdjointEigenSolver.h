// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SELFADJOINTEIGENSOLVER_H
#define EIGEN_SELFADJOINTEIGENSOLVER_H

#include "./Tridiagonalization.h"

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename MatrixType_>
class GeneralizedSelfAdjointEigenSolver;

namespace internal {
template <typename SolverType, int Size, bool IsComplex, bool IsDirect = !IsComplex && (Size == 2 || Size == 3)>
struct direct_selfadjoint_eigenvalues;

template <bool PerBlockScaling, typename MatrixType, typename DiagType, typename SubDiagType>
EIGEN_DEVICE_FUNC ComputationInfo computeFromTridiagonal_impl(DiagType& diag, SubDiagType& subdiag,
                                                              const Index maxIterations, bool computeEigenvectors,
                                                              MatrixType& eivec);
}  // namespace internal

/** \eigenvalues_module \ingroup Eigenvalues_Module
 *
 *
 * \class SelfAdjointEigenSolver
 *
 * \brief Computes eigenvalues and eigenvectors of selfadjoint matrices
 *
 * \tparam MatrixType_ the type of the matrix of which we are computing the
 * eigendecomposition; this is expected to be an instantiation of the Matrix
 * class template.
 *
 * A matrix \f$ A \f$ is selfadjoint if it equals its adjoint. For real
 * matrices, this means that the matrix is symmetric: it equals its
 * transpose. This class computes the eigenvalues and eigenvectors of a
 * selfadjoint matrix. These are the scalars \f$ \lambda \f$ and vectors
 * \f$ v \f$ such that \f$ Av = \lambda v \f$.  The eigenvalues of a
 * selfadjoint matrix are always real. If \f$ D \f$ is a diagonal matrix with
 * the eigenvalues on the diagonal, and \f$ V \f$ is a matrix with the
 * eigenvectors as its columns, then \f$ A = V D V^{-1} \f$. This is called the
 * eigendecomposition.
 *
 * For a selfadjoint matrix, \f$ V \f$ is unitary, meaning its inverse is equal
 * to its adjoint, \f$ V^{-1} = V^{\dagger} \f$. If \f$ A \f$ is real, then
 * \f$ V \f$ is also real and therefore orthogonal, meaning its inverse is
 * equal to its transpose, \f$ V^{-1} = V^T \f$.
 *
 * The algorithm exploits the fact that the matrix is selfadjoint, making it
 * faster and more accurate than the general purpose eigenvalue algorithms
 * implemented in EigenSolver and ComplexEigenSolver.
 *
 * Only the \b lower \b triangular \b part of the input matrix is referenced.
 *
 * Call the function compute() to compute the eigenvalues and eigenvectors of
 * a given matrix. Alternatively, you can use the
 * SelfAdjointEigenSolver(const MatrixType&, int) constructor which computes
 * the eigenvalues and eigenvectors at construction time. Once the eigenvalue
 * and eigenvectors are computed, they can be retrieved with the eigenvalues()
 * and eigenvectors() functions.
 *
 * The documentation for SelfAdjointEigenSolver(const MatrixType&, int)
 * contains an example of the typical use of this class.
 *
 * To solve the \em generalized eigenvalue problem \f$ Av = \lambda Bv \f$ and
 * the likes, see the class GeneralizedSelfAdjointEigenSolver.
 *
 * \sa MatrixBase::eigenvalues(), class EigenSolver, class ComplexEigenSolver
 */
template <typename MatrixType_>
class SelfAdjointEigenSolver {
 public:
  using MatrixType = MatrixType_;
  enum {
    Size = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    Options = internal::plain_object_options<MatrixType>::value,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
  };

  /** \brief Scalar type for matrices of type \p MatrixType_. */
  using Scalar = typename MatrixType::Scalar;
  using Index = Eigen::Index;  ///< \deprecated since Eigen 3.3

  /** \brief Plain matrix type with the shape and storage options of \p MatrixType_; \p MatrixType_ itself unless
   * that is a Ref<>. */
  using PlainMatrixType =
      Matrix<Scalar, Size, ColsAtCompileTime, Options, MatrixType::MaxRowsAtCompileTime, MaxColsAtCompileTime>;

  /** \brief Type of the matrix returned by eigenvectors().
   *
   * A column-major plain matrix, or \p MatrixType_ itself when that is a Ref<>: the \link InplaceDecomposition
   * inplace decomposition \endlink then stores the eigenvectors in the referenced matrix. */
  using EigenvectorsType =
      std::conditional_t<internal::is_ref<MatrixType>::value, MatrixType,
                         Matrix<Scalar, Size, Size, ColMajor, MaxColsAtCompileTime, MaxColsAtCompileTime>>;

  /** \brief Real scalar type for \p MatrixType_.
   *
   * This is just \c Scalar if #Scalar is real (e.g., \c float or
   * \c double), and the type of the real part of \c Scalar if #Scalar is
   * complex.
   */
  using RealScalar = typename NumTraits<Scalar>::Real;

  friend struct internal::direct_selfadjoint_eigenvalues<SelfAdjointEigenSolver, Size, NumTraits<Scalar>::IsComplex>;

  /** \brief Type for vector of eigenvalues as returned by eigenvalues().
   *
   * This is a column vector with entries of type #RealScalar.
   * The length of the vector is the size of \p MatrixType_.
   */
  using VectorType = typename internal::plain_col_type<MatrixType, Scalar>::type;
  using RealVectorType = typename internal::plain_col_type<MatrixType, RealScalar>::type;
  using TridiagonalizationType = Tridiagonalization<MatrixType>;
  using SubDiagonalType = typename TridiagonalizationType::SubDiagonalType;

  /** \brief Default constructor for fixed-size matrices.
   *
   * The default constructor is useful in cases in which the user intends to
   * perform decompositions via compute(). This constructor
   * can only be used if \p MatrixType_ is a fixed-size matrix; use
   * SelfAdjointEigenSolver(Index) for dynamic-size matrices.
   *
   * Example: \include SelfAdjointEigenSolver_SelfAdjointEigenSolver.cpp
   * Output: \verbinclude SelfAdjointEigenSolver_SelfAdjointEigenSolver.out
   */
  EIGEN_DEVICE_FUNC SelfAdjointEigenSolver()
      : m_eivec(),
        m_workspace(),
        m_eivalues(),
        m_subdiag(),
        m_hcoeffs(),
        m_info(InvalidInput),
        m_isInitialized(false),
        m_eigenvectorsOk(false) {}

  /** \brief Constructor, pre-allocates memory for dynamic-size matrices.
   *
   * \param [in]  size  Positive integer, size of the matrix whose
   * eigenvalues and eigenvectors will be computed.
   *
   * This constructor is useful for dynamic-size matrices, when the user
   * intends to perform decompositions via compute(). The \p size
   * parameter is only used as a hint. It is not an error to give a wrong
   * \p size, but it may impair performance.
   *
   * \sa compute() for an example
   */
  EIGEN_DEVICE_FUNC explicit SelfAdjointEigenSolver(Index size)
      : m_eivec(size, size),
        m_workspace(size),
        m_eivalues(size),
        m_subdiag(size > 1 ? size - 1 : 1),
        m_hcoeffs(size > 1 ? size - 1 : 1),
        m_isInitialized(false),
        m_eigenvectorsOk(false) {}

  /** \brief Constructor; computes eigendecomposition of given matrix.
   *
   * \param[in]  matrix  Selfadjoint matrix whose eigendecomposition is to
   *    be computed. Only the lower triangular part of the matrix is referenced.
   * \param[in]  options Can be #ComputeEigenvectors (default) or #EigenvaluesOnly.
   *
   * This constructor calls compute(const MatrixType&, int) to compute the
   * eigenvalues of the matrix \p matrix. The eigenvectors are computed if
   * \p options equals #ComputeEigenvectors.
   *
   * Example: \include SelfAdjointEigenSolver_SelfAdjointEigenSolver_MatrixType.cpp
   * Output: \verbinclude SelfAdjointEigenSolver_SelfAdjointEigenSolver_MatrixType.out
   *
   * \sa compute(const MatrixType&, int)
   */
  template <typename InputType>
  EIGEN_DEVICE_FUNC explicit SelfAdjointEigenSolver(const EigenBase<InputType>& matrix,
                                                    int options = ComputeEigenvectors)
      : m_eivec(matrix.rows(), matrix.cols()),
        m_workspace(matrix.cols()),
        m_eivalues(matrix.cols()),
        m_subdiag(matrix.rows() > 1 ? matrix.rows() - 1 : 1),
        m_hcoeffs(matrix.cols() > 1 ? matrix.cols() - 1 : 1),
        m_isInitialized(false),
        m_eigenvectorsOk(false) {
    compute(matrix.derived(), options);
  }

  /** \brief Constructor for \link InplaceDecomposition inplace decomposition \endlink
   *
   * \param[in,out]  matrix  Selfadjoint matrix whose eigendecomposition is to
   *    be computed. Only the lower triangular part of the matrix is referenced.
   * \param[in]  options Can be #ComputeEigenvectors (default) or #EigenvaluesOnly.
   *
   * When \p MatrixType is a Ref<>, the decomposition is computed within the memory of \p matrix, whose content is
   * destroyed: with #ComputeEigenvectors it holds the eigenvectors afterwards and eigenvectors() refers to it, with
   * #EigenvaluesOnly its content is unspecified. This overload is only available for Ref<>; owning matrix types
   * use the constructor taking a const input.
   */
  template <typename InputType, bool IsRef = internal::is_ref<MatrixType>::value, std::enable_if_t<IsRef, int> = 0>
  EIGEN_DEVICE_FUNC explicit SelfAdjointEigenSolver(EigenBase<InputType>& matrix, int options = ComputeEigenvectors)
      : SelfAdjointEigenSolver(matrix, BindStorageTag()) {
    m_eivec.template triangularView<StrictlyUpper>().setZero();
    computeInPlace(options);
  }

  /** \brief Computes eigendecomposition of given matrix.
   *
   * \param[in]  matrix  Selfadjoint matrix whose eigendecomposition is to
   *    be computed. Only the lower triangular part of the matrix is referenced.
   * \param[in]  options Can be #ComputeEigenvectors (default) or #EigenvaluesOnly.
   * \returns    Reference to \c *this
   *
   * This function computes the eigenvalues of \p matrix.  The eigenvalues()
   * function can be used to retrieve them.  If \p options equals #ComputeEigenvectors,
   * then the eigenvectors are also computed and can be retrieved by
   * calling eigenvectors().
   *
   * This implementation uses a symmetric QR algorithm. The matrix is first
   * reduced to tridiagonal form using the Tridiagonalization class. The
   * tridiagonal matrix is then brought to diagonal form with implicit
   * symmetric QR steps with Wilkinson shift. Details can be found in
   * Section 8.3 of Golub \& Van Loan, <i>%Matrix Computations</i>.
   *
   * The cost of the computation is about \f$ 9n^3 \f$ if the eigenvectors
   * are required and \f$ 4n^3/3 \f$ if they are not required.
   *
   * This method reuses the memory in the SelfAdjointEigenSolver object that
   * was allocated when the object was constructed, if the size of the
   * matrix does not change.
   *
   * Example: \include SelfAdjointEigenSolver_compute_MatrixType.cpp
   * Output: \verbinclude SelfAdjointEigenSolver_compute_MatrixType.out
   *
   * \sa SelfAdjointEigenSolver(const MatrixType&, int)
   */
  template <typename InputType>
  EIGEN_DEVICE_FUNC SelfAdjointEigenSolver& compute(const EigenBase<InputType>& matrix,
                                                    int options = ComputeEigenvectors);

  /** \brief Computes eigendecomposition of given matrix primarily using a closed-form algorithm
   *
   * This is a variant of compute(const MatrixType&, int options) which
   * normally solves the underlying polynomial equation directly.
   * For supported real binary floating-point 3x3 matrices with coefficients near the
   * overflow limit, it instead uses the iterative solver in scaled coordinates to
   * avoid overflow from roundoff near repeated eigenvalues. This fallback has the
   * runtime characteristics of compute() and may fail to converge. Check info()
   * for Success before using the results; the fallback can report NoConvergence.
   * Non-finite inputs have no guaranteed decomposition. In particular, infinite
   * coefficients can also enter the 3x3 fallback and report NoConvergence;
   * a Success status alone does not validate non-finite input.
   *
   * Currently only 2x2 and 3x3 matrices for which the sizes are known at compile time are supported (e.g., Matrix3d).
   *
   * This method is usually significantly faster than the QR iterative algorithm
   * but it might also be less accurate. It is also worth noting that
   * for 3x3 matrices it involves trigonometric operations which are
   * not necessarily available for all scalar types.
   *
   * For the 3x3 case, we observed the following worst case relative error regarding the eigenvalues:
   *   - double: 1e-8
   *   - float:  1e-3
   *
   * \sa compute(const MatrixType&, int options)
   */
  EIGEN_DEVICE_FUNC SelfAdjointEigenSolver& computeDirect(const MatrixType& matrix, int options = ComputeEigenvectors);

  /**
   *\brief Computes the eigen decomposition from a tridiagonal symmetric matrix
   *
   * \param[in] diag The vector containing the diagonal of the matrix.
   * \param[in] subdiag The subdiagonal of the matrix.
   * \param[in] options Can be #ComputeEigenvectors (default) or #EigenvaluesOnly.
   * \returns Reference to \c *this
   *
   * This function assumes that the matrix has been reduced to tridiagonal form.
   *
   * \sa compute(const MatrixType&, int) for more information
   */
  SelfAdjointEigenSolver& computeFromTridiagonal(const RealVectorType& diag, const SubDiagonalType& subdiag,
                                                 int options = ComputeEigenvectors);

  /** \brief Returns the eigenvectors of given matrix.
   *
   * \returns  A const reference to the matrix whose columns are the eigenvectors.
   *
   * \pre The eigenvectors have been computed before.
   *
   * Column \f$ k \f$ of the returned matrix is an eigenvector corresponding
   * to eigenvalue number \f$ k \f$ as returned by eigenvalues().  The
   * eigenvectors are normalized to have (Euclidean) norm equal to one. If
   * this object was used to solve the eigenproblem for the selfadjoint
   * matrix \f$ A \f$, then the matrix returned by this function is the
   * matrix \f$ V \f$ in the eigendecomposition \f$ A = V D V^{-1} \f$.
   *
   * For a selfadjoint matrix, \f$ V \f$ is unitary, meaning its inverse is equal
   * to its adjoint, \f$ V^{-1} = V^{\dagger} \f$. If \f$ A \f$ is real, then
   * \f$ V \f$ is also real and therefore orthogonal, meaning its inverse is
   * equal to its transpose, \f$ V^{-1} = V^T \f$.
   *
   * Example: \include SelfAdjointEigenSolver_eigenvectors.cpp
   * Output: \verbinclude SelfAdjointEigenSolver_eigenvectors.out
   *
   * \sa eigenvalues()
   */
  EIGEN_DEVICE_FUNC const EigenvectorsType& eigenvectors() const {
    eigen_assert(m_isInitialized && "SelfAdjointEigenSolver is not initialized.");
    eigen_assert(m_eigenvectorsOk && "The eigenvectors have not been computed together with the eigenvalues.");
    return m_eivec;
  }

  /** \brief Returns the eigenvalues of given matrix.
   *
   * \returns A const reference to the column vector containing the eigenvalues.
   *
   * \pre The eigenvalues have been computed before.
   *
   * The eigenvalues are repeated according to their algebraic multiplicity,
   * so there are as many eigenvalues as rows in the matrix. The eigenvalues
   * are sorted in increasing order.
   *
   * Example: \include SelfAdjointEigenSolver_eigenvalues.cpp
   * Output: \verbinclude SelfAdjointEigenSolver_eigenvalues.out
   *
   * \sa eigenvectors(), MatrixBase::eigenvalues()
   */
  EIGEN_DEVICE_FUNC const RealVectorType& eigenvalues() const {
    eigen_assert(m_isInitialized && "SelfAdjointEigenSolver is not initialized.");
    return m_eivalues;
  }

  /** \brief Computes the positive-definite square root of the matrix.
   *
   * \returns the positive-definite square root of the matrix
   *
   * \pre The eigenvalues and eigenvectors of a positive-definite matrix
   * have been computed before.
   *
   * The square root of a positive-definite matrix \f$ A \f$ is the
   * positive-definite matrix whose square equals \f$ A \f$. This function
   * uses the eigendecomposition \f$ A = V D V^{-1} \f$ to compute the
   * square root as \f$ A^{1/2} = V D^{1/2} V^{-1} \f$.
   *
   * Example: \include SelfAdjointEigenSolver_operatorSqrt.cpp
   * Output: \verbinclude SelfAdjointEigenSolver_operatorSqrt.out
   *
   * \sa operatorInverseSqrt(), <a href="contrib/group__MatrixFunctions__Module.html">MatrixFunctions Module</a>
   */
  EIGEN_DEVICE_FUNC PlainMatrixType operatorSqrt() const {
    eigen_assert(m_isInitialized && "SelfAdjointEigenSolver is not initialized.");
    eigen_assert(m_eigenvectorsOk && "The eigenvectors have not been computed together with the eigenvalues.");
    return m_eivec * m_eivalues.cwiseSqrt().asDiagonal() * m_eivec.adjoint();
  }

  /** \brief Computes the matrix exponential of the matrix.
   *
   * \returns the matrix exponential of the matrix.
   *
   * \pre The eigenvalues and eigenvectors of a positive-definite matrix
   * have been computed before.
   *
   * \sa operatorInverseSqrt(), operatorSqrt(),
   * <a href="contrib/group__MatrixFunctions__Module.html">MatrixFunctions Module</a>
   */
  EIGEN_DEVICE_FUNC PlainMatrixType operatorExp() const {
    eigen_assert(m_isInitialized && "SelfAdjointEigenSolver is not initialized.");
    eigen_assert(m_eigenvectorsOk && "The eigenvectors have not been computed together with the eigenvalues.");
    return m_eivec * m_eivalues.array().exp().matrix().asDiagonal() * m_eivec.adjoint();
  }

  /** \brief Computes the inverse square root of the matrix.
   *
   * \returns the inverse positive-definite square root of the matrix
   *
   * \pre The eigenvalues and eigenvectors of a positive-definite matrix
   * have been computed before.
   *
   * This function uses the eigendecomposition \f$ A = V D V^{-1} \f$ to
   * compute the inverse square root as \f$ V D^{-1/2} V^{-1} \f$. This is
   * cheaper than first computing the square root with operatorSqrt() and
   * then its inverse with MatrixBase::inverse().
   *
   * Example: \include SelfAdjointEigenSolver_operatorInverseSqrt.cpp
   * Output: \verbinclude SelfAdjointEigenSolver_operatorInverseSqrt.out
   *
   * \sa operatorSqrt(), MatrixBase::inverse(), <a
   * href="contrib/group__MatrixFunctions__Module.html">MatrixFunctions Module</a>
   */
  EIGEN_DEVICE_FUNC PlainMatrixType operatorInverseSqrt() const {
    eigen_assert(m_isInitialized && "SelfAdjointEigenSolver is not initialized.");
    eigen_assert(m_eigenvectorsOk && "The eigenvectors have not been computed together with the eigenvalues.");
    return m_eivec * m_eivalues.cwiseInverse().cwiseSqrt().asDiagonal() * m_eivec.adjoint();
  }

  /** \brief Reports whether previous computation was successful.
   *
   * \returns \c Success if computation was successful, \c NoConvergence otherwise.
   */
  EIGEN_DEVICE_FUNC ComputationInfo info() const {
    eigen_assert(m_isInitialized && "SelfAdjointEigenSolver is not initialized.");
    return m_info;
  }

  /** \brief Maximum number of iterations.
   *
   * The algorithm terminates if it does not converge within m_maxIterations * n iterations, where n
   * denotes the size of the matrix. This value is currently set to 30 (copied from LAPACK).
   */
  static const int m_maxIterations = 30;

 protected:
  EIGEN_STATIC_ASSERT_NON_INTEGER(Scalar)

  /** \internal Tag of the constructor that binds the eigenvector storage without computing anything. */
  struct BindStorageTag {};

  /** \internal Binds the eigenvector storage to \a matrix, or copies it when #EigenvectorsType is a plain matrix, and
   * allocates the workspace without computing anything. */
  template <typename InputType>
  EIGEN_DEVICE_FUNC SelfAdjointEigenSolver(EigenBase<InputType>& matrix, BindStorageTag)
      : m_eivec(matrix.derived()),
        m_workspace(matrix.cols()),
        m_eivalues(matrix.cols()),
        m_subdiag(matrix.rows() > 1 ? matrix.rows() - 1 : 1),
        m_hcoeffs(matrix.cols() > 1 ? matrix.cols() - 1 : 1),
        m_isInitialized(false),
        m_eigenvectorsOk(false) {}

  /** \internal Computes the eigendecomposition of the selfadjoint matrix held in m_eivec, whose lower triangle is
   * referenced; the eigenvectors overwrite it when requested. */
  EIGEN_DEVICE_FUNC SelfAdjointEigenSolver& computeInPlace(int options);

  EigenvectorsType m_eivec;
  VectorType m_workspace;
  RealVectorType m_eivalues;
  typename TridiagonalizationType::SubDiagonalType m_subdiag;
  typename TridiagonalizationType::CoeffVectorType m_hcoeffs;
  ComputationInfo m_info;
  bool m_isInitialized;
  bool m_eigenvectorsOk;
};

namespace internal {
/** \internal
 *
 * \eigenvalues_module \ingroup Eigenvalues_Module
 *
 * Performs a QR step on a tridiagonal symmetric matrix represented as a
 * pair of two vectors \a diag and \a subdiag.
 *
 * \param diag the diagonal part of the input selfadjoint tridiagonal matrix
 * \param subdiag the sub-diagonal part of the input selfadjoint tridiagonal matrix
 * \param start starting index of the submatrix to work on
 * \param end last+1 index of the submatrix to work on
 * \param matrixQ pointer to the matrix accumulating the eigenvectors, can be null
 *
 * For compilation efficiency reasons, the tridiagonal matrix is passed as raw pointers.
 *
 * Implemented from Golub's "Matrix Computations", algorithm 8.3.2:
 * "implicit symmetric QR step with Wilkinson shift"
 */
template <typename RealScalar, typename Index, typename MatrixQType>
EIGEN_DEVICE_FUNC static void tridiagonal_qr_step(RealScalar* diag, RealScalar* subdiag, Index start, Index end,
                                                  MatrixQType* matrixQ);
}  // namespace internal

template <typename MatrixType>
template <typename InputType>
EIGEN_DEVICE_FUNC SelfAdjointEigenSolver<MatrixType>& SelfAdjointEigenSolver<MatrixType>::compute(
    const EigenBase<InputType>& a_matrix, int options) {
  const InputType& matrix(a_matrix.derived());
  eigen_assert(matrix.cols() == matrix.rows());
  m_eivec = matrix.template triangularView<Lower>();
  return computeInPlace(options);
}

template <typename MatrixType>
EIGEN_DEVICE_FUNC SelfAdjointEigenSolver<MatrixType>& SelfAdjointEigenSolver<MatrixType>::computeInPlace(int options) {
  eigen_assert(m_eivec.cols() == m_eivec.rows());
  eigen_assert((options & ~(EigVecMask | GenEigMask)) == 0 && (options & EigVecMask) != EigVecMask &&
               "invalid option parameter");
  bool computeEigenvectors = (options & ComputeEigenvectors) == ComputeEigenvectors;
  Index n = m_eivec.cols();
  m_eivalues.resize(n, 1);

  if (n == 1) {
    m_eivalues.coeffRef(0, 0) = numext::real(m_eivec.coeff(0, 0));
    if (computeEigenvectors) m_eivec.setOnes();
    m_info = (numext::isfinite)(m_eivalues.coeffRef(0, 0)) ? Success : NoConvergence;
    m_isInitialized = true;
    m_eigenvectorsOk = computeEigenvectors;
    return *this;
  }

  // declare some aliases
  RealVectorType& diag = m_eivalues;
  EigenvectorsType& mat = m_eivec;

  // Scale the matrix to O(1) to avoid overflow/underflow during tridiagonalization
  // and subsequent QR iteration. Power-of-two factors avoid rounding representable scaled coefficients.
  // Note: for block-diagonal matrices with widely separated scales, this
  // can underflow small blocks. Users with such matrices should tridiagonalize separately
  // and call computeFromTridiagonal(), which uses per-block scaling.
  const RealScalar maxCoeff = internal::safe_scaling<RealScalar>::recover_flushed_max_coeff(
      mat, mat.cwiseAbs().template maxCoeff<PropagateNaN>());
  if (!(numext::isfinite)(maxCoeff)) {
    // Input contains Inf or NaN.
    m_info = NoConvergence;
    m_isInitialized = true;
    m_eigenvectorsOk = false;
    return *this;
  }
  const auto factors = internal::safe_scaling<RealScalar>::with_scaled(mat, maxCoeff, [&](const auto& scaled) {
    mat.template triangularView<Lower>() = scaled.template triangularView<Lower>();
  });
  m_subdiag.resize(n - 1);
  m_hcoeffs.resize(n - 1);
  internal::tridiagonalization_inplace(mat, diag, m_subdiag, m_hcoeffs, m_workspace, computeEigenvectors);

  m_info = internal::computeFromTridiagonal_impl<false>(diag, m_subdiag, m_maxIterations, computeEigenvectors, m_eivec);

  // Scale back the eigenvalues.
  internal::safe_scaling<RealScalar>::unscale_in_place(m_eivalues, maxCoeff, factors);

  m_isInitialized = true;
  m_eigenvectorsOk = computeEigenvectors;
  return *this;
}

template <typename MatrixType>
SelfAdjointEigenSolver<MatrixType>& SelfAdjointEigenSolver<MatrixType>::computeFromTridiagonal(
    const RealVectorType& diag, const SubDiagonalType& subdiag, int options) {
  bool computeEigenvectors = (options & ComputeEigenvectors) == ComputeEigenvectors;

  m_eivalues = diag;
  m_subdiag = subdiag;

  // Check for Inf/NaN in the input.
  {
    RealScalar scale = RealScalar(0);
    if (m_eivalues.size() > 0) scale = m_eivalues.cwiseAbs().maxCoeff();
    if (m_subdiag.size() > 0) scale = numext::maxi(scale, m_subdiag.cwiseAbs().maxCoeff());
    if (!(numext::isfinite)(scale)) {
      m_info = NoConvergence;
      m_isInitialized = true;
      m_eigenvectorsOk = false;
      return *this;
    }
  }

  if (computeEigenvectors) {
    m_eivec.setIdentity(diag.size(), diag.size());
  }
  // Use per-deflation-block scaling (like LAPACK's DSTERF) to avoid losing
  // precision when the tridiagonal entries span a wide range of magnitudes.
  m_info =
      internal::computeFromTridiagonal_impl<true>(m_eivalues, m_subdiag, m_maxIterations, computeEigenvectors, m_eivec);

  m_isInitialized = true;
  m_eigenvectorsOk = computeEigenvectors;
  return *this;
}

namespace internal {
/**
 * \internal
 * \brief Compute the eigendecomposition from a tridiagonal matrix
 *
 * \tparam PerBlockScaling If true, each deflation block is independently scaled to O(1) before
 *         QR iteration, following LAPACK's DSTERF approach. This prevents precision loss when entries
 *         span a wide range of magnitudes. When false, the caller is responsible for ensuring the
 *         entries are in a safe range (e.g. by pre-scaling the dense matrix before tridiagonalization).
 * \param[in,out] diag : On input, the diagonal of the matrix, on output the eigenvalues
 * \param[in,out] subdiag : The subdiagonal part of the matrix (entries are modified during the decomposition)
 * \param[in] maxIterations : the maximum number of iterations
 * \param[in] computeEigenvectors : whether the eigenvectors have to be computed or not
 * \param[out] eivec : The matrix to store the eigenvectors if computeEigenvectors==true. Must be allocated on input.
 * \returns \c Success or \c NoConvergence
 */
template <bool PerBlockScaling, typename MatrixType, typename DiagType, typename SubDiagType>
EIGEN_DEVICE_FUNC ComputationInfo computeFromTridiagonal_impl(DiagType& diag, SubDiagType& subdiag,
                                                              const Index maxIterations, bool computeEigenvectors,
                                                              MatrixType& eivec) {
  ComputationInfo info;

  Index n = diag.size();
  Index end = n - 1;
  Index start = 0;
  Index iter = 0;  // total number of iterations

  using RealScalar = typename DiagType::RealScalar;
  const RealScalar considerAsZero = (std::numeric_limits<RealScalar>::min)();
  const RealScalar precision = NumTraits<RealScalar>::epsilon();
  const RealScalar precision_inv = RealScalar(1) / precision;

  // Helper lambda for the deflation test.
  auto deflate = [&](Index lo, Index hi) {
    for (Index i = lo; i < hi; ++i) {
      const RealScalar absSubdiag = numext::abs(subdiag[i]);
      if (absSubdiag < considerAsZero) {
        subdiag[i] = RealScalar(0);
      } else if (!PerBlockScaling) {
        // Homogeneous in the global scale: both sides are linear in it, so prescaling cannot change which couplings
        // are discarded (the previous test was quadratic on the left).
        if (absSubdiag <= precision * numext::maxi(numext::abs(diag[i]), numext::abs(diag[i + 1]))) {
          subdiag[i] = RealScalar(0);
        }
      } else {
        const RealScalar scaled_subdiag = precision_inv * subdiag[i];
        if (scaled_subdiag * scaled_subdiag <= (numext::abs(diag[i]) + numext::abs(diag[i + 1]))) {
          subdiag[i] = RealScalar(0);
        }
      }
    }
  };

  // For per-block scaling, track the currently scaled block and its scale factor.
  // When the outer loop identifies a block outside the scaled region, unscale the old
  // block and scale the new one. This keeps the same outer loop structure (one QR step
  // per iteration) while ensuring each block is processed in scaled coordinates.
  Index scaled_start = -1, scaled_end = -1;
  RealScalar block_norm = RealScalar(0);
  safe_scaling_factors<RealScalar> blockFactors;
  const auto restore_block = [&]() {
    if (scaled_start < 0) return;
    auto diagonal = diag.segment(scaled_start, scaled_end - scaled_start + 1);
    auto offDiagonal = subdiag.segment(scaled_start, scaled_end - scaled_start);
    safe_scaling<RealScalar>::unscale_in_place(diagonal, block_norm, blockFactors);
    safe_scaling<RealScalar>::unscale_in_place(offDiagonal, block_norm, blockFactors);
  };

  while (end > 0) {
    deflate(start, end);

    // Find the largest unreduced block at the end of the matrix.
    while (end > 0 && numext::is_exactly_zero(subdiag[end - 1])) {
      end--;
    }
    if (end <= 0) break;

    // if we spent too many iterations, we give up
    iter++;
    if (iter > maxIterations * n) break;

    start = end - 1;
    while (start > 0 && !numext::is_exactly_zero(subdiag[start - 1])) start--;

    if (PerBlockScaling) {
      // Check if we've moved to a different block than the one currently scaled.
      if (start != scaled_start || end != scaled_end) {
        restore_block();
        // Compute the norm and scale the new block to O(1).
        block_norm = RealScalar(0);
        for (Index i = start; i <= end; ++i) block_norm = numext::maxi(block_norm, numext::abs(diag[i]));
        for (Index i = start; i < end; ++i) block_norm = numext::maxi(block_norm, numext::abs(subdiag[i]));
        auto diagonal = diag.segment(start, end - start + 1);
        auto offDiagonal = subdiag.segment(start, end - start);
        blockFactors = safe_scaling<RealScalar>::scale_to(diagonal, diagonal, block_norm);
        safe_scaling<RealScalar>::scale_to(offDiagonal, offDiagonal, block_norm, blockFactors);
        scaled_start = start;
        scaled_end = end;
      }
    }

    internal::tridiagonal_qr_step(diag.data(), subdiag.data(), start, end,
                                  computeEigenvectors ? &eivec : static_cast<MatrixType*>(nullptr));
  }

  // Unscale any remaining scaled block.
  if (PerBlockScaling) restore_block();
  if (iter <= maxIterations * n)
    info = Success;
  else
    info = NoConvergence;

  // Sort eigenvalues and corresponding vectors.
  // TODO: make the sort optional and use a more efficient sorting algorithm.
  if (info == Success) {
    for (Index i = 0; i < n - 1; ++i) {
      // Scalar argmin: the vectorized minCoeff path can return the wrong
      // index on targets whose SIMD reduction flushes subnormals (32-bit ARM
      // NEON always treats subnormal inputs to vminq_f32 as zero), which
      // corrupts ordering when eigenvalues span the subnormal range.
      Index k = i;
      RealScalar min_val = diag[i];
      for (Index j = i + 1; j < n; ++j) {
        if (diag[j] < min_val) {
          min_val = diag[j];
          k = j;
        }
      }
      if (k != i) {
        numext::swap(diag[i], diag[k]);
        if (computeEigenvectors) eivec.col(i).swap(eivec.col(k));
      }
    }
  }
  return info;
}

template <typename SolverType, int Size, bool IsComplex, bool IsDirect>
struct direct_selfadjoint_eigenvalues {
  EIGEN_DEVICE_FUNC static inline void run(SolverType& eig, const typename SolverType::MatrixType& A, int options) {
    eig.compute(A, options);
  }
};

template <typename SolverType, int Size>
struct direct_selfadjoint_eigensolver_kernel;

template <typename SolverType>
struct direct_selfadjoint_eigensolver_kernel<SolverType, 3> {
  using MatrixType = typename SolverType::MatrixType;
  using VectorType = typename SolverType::RealVectorType;
  using Scalar = typename SolverType::Scalar;
  using EigenvectorsType = typename SolverType::EigenvectorsType;
  using PlainMatrixType = typename SolverType::PlainMatrixType;

  /** \internal
   * Computes the roots of the characteristic polynomial of \a m.
   * For numerical stability m.trace() should be near zero and to avoid over- or underflow m should be normalized.
   */
  EIGEN_DEVICE_FUNC static inline void computeRoots(const MatrixType& m, VectorType& roots) {
    EIGEN_USING_STD(sqrt)
    EIGEN_USING_STD(atan2)
    EIGEN_USING_STD(cos)
    EIGEN_USING_STD(sin)
    const Scalar s_inv3 = Scalar(1) / Scalar(3);
    const Scalar s_sqrt3 = sqrt(Scalar(3));

    // The characteristic equation is x^3 - c2*x^2 + c1*x - c0 = 0.  The
    // eigenvalues are the roots to this equation, all guaranteed to be
    // real-valued, because the matrix is symmetric.
    Scalar c0 = m(0, 0) * m(1, 1) * m(2, 2) + Scalar(2) * m(1, 0) * m(2, 0) * m(2, 1) - m(0, 0) * m(2, 1) * m(2, 1) -
                m(1, 1) * m(2, 0) * m(2, 0) - m(2, 2) * m(1, 0) * m(1, 0);
    Scalar c1 = m(0, 0) * m(1, 1) - m(1, 0) * m(1, 0) + m(0, 0) * m(2, 2) - m(2, 0) * m(2, 0) + m(1, 1) * m(2, 2) -
                m(2, 1) * m(2, 1);
    Scalar c2 = m(0, 0) + m(1, 1) + m(2, 2);

    // Construct the parameters used in classifying the roots of the equation
    // and in solving the equation for the roots in closed form.
    Scalar c2_over_3 = c2 * s_inv3;
    Scalar a_over_3 = (c2 * c2_over_3 - c1) * s_inv3;
    a_over_3 = numext::maxi(a_over_3, Scalar(0));

    Scalar half_b = Scalar(0.5) * (c0 + c2_over_3 * (Scalar(2) * c2_over_3 * c2_over_3 - c1));

    Scalar q = a_over_3 * a_over_3 * a_over_3 - half_b * half_b;
    q = numext::maxi(q, Scalar(0));

    // Compute the eigenvalues by solving for the roots of the polynomial.
    Scalar rho = sqrt(a_over_3);
    Scalar theta = atan2(sqrt(q), half_b) * s_inv3;  // since sqrt(q) > 0, atan2 is in [0, pi] and theta is in [0, pi/3]
    Scalar cos_theta = cos(theta);
    Scalar sin_theta = sin(theta);
    // roots are already sorted, since cos is monotonically decreasing on [0, pi]
    roots(0) = c2_over_3 - rho * (cos_theta + s_sqrt3 * sin_theta);  // == 2*rho*cos(theta+2pi/3)
    roots(1) = c2_over_3 - rho * (cos_theta - s_sqrt3 * sin_theta);  // == 2*rho*cos(theta+ pi/3)
    roots(2) = c2_over_3 + Scalar(2) * rho * cos_theta;
  }

  EIGEN_DEVICE_FUNC static inline bool extract_kernel(PlainMatrixType& mat, Ref<VectorType> res,
                                                      Ref<VectorType> representative) {
    EIGEN_USING_STD(abs);
    EIGEN_USING_STD(sqrt);
    Index i0;
    // Find non-zero column i0 (by construction, there must exist a non zero coefficient on the diagonal):
    mat.diagonal().cwiseAbs().maxCoeff(&i0);
    // mat.col(i0) is a good candidate for an orthogonal vector to the current eigenvector,
    // so let's save it:
    representative = mat.col(i0);
    Scalar n0, n1;
    VectorType c0, c1;
    n0 = (c0 = representative.cross(mat.col((i0 + 1) % 3))).squaredNorm();
    n1 = (c1 = representative.cross(mat.col((i0 + 2) % 3))).squaredNorm();
    if (n0 > n1)
      res = c0 / sqrt(n0);
    else
      res = c1 / sqrt(n1);

    return true;
  }

  EIGEN_DEVICE_FUNC static void run(PlainMatrixType& scaledMat, VectorType& eivals, EigenvectorsType& eivecs,
                                    bool computeEigenvectors, const Scalar& maxCoeff) {
    // compute the eigenvalues
    computeRoots(scaledMat, eivals);

    // computeRoots produces theoretically sorted roots, but floating-point
    // rounding in the trigonometric formulas can break the ordering.
    // Enforce sorting with a fixed 3-element compare-swap network.
    if (eivals(0) > eivals(1)) numext::swap(eivals(0), eivals(1));
    if (eivals(1) > eivals(2)) numext::swap(eivals(1), eivals(2));
    if (eivals(0) > eivals(1)) numext::swap(eivals(0), eivals(1));

    // compute the eigenvectors
    if (computeEigenvectors) {
      if ((eivals(2) - eivals(0)) <= Eigen::NumTraits<Scalar>::epsilon() * maxCoeff) {
        // All three eigenvalues are numerically the same
        eivecs.setIdentity();
      } else {
        PlainMatrixType tmp;
        tmp = scaledMat;

        // Compute the eigenvector of the most distinct eigenvalue
        Scalar d0 = eivals(2) - eivals(1);
        Scalar d1 = eivals(1) - eivals(0);
        Index k(0), l(2);
        if (d0 > d1) {
          numext::swap(k, l);
          d0 = d1;
        }

        // Compute the eigenvector of index k
        {
          tmp.diagonal().array() -= eivals(k);
          // By construction, 'tmp' is of rank 2, and its kernel corresponds to the respective eigenvector.
          extract_kernel(tmp, eivecs.col(k), eivecs.col(l));
        }

        // Compute eigenvector of index l
        if (d0 <= 2 * Eigen::NumTraits<Scalar>::epsilon() * d1) {
          // If d0 is too small, then the two other eigenvalues are numerically the same,
          // and thus we only have to ortho-normalize the near orthogonal vector we saved above.
          eivecs.col(l) -= eivecs.col(k).dot(eivecs.col(l)) * eivecs.col(k);
          eivecs.col(l).normalize();
        } else {
          tmp = scaledMat;
          tmp.diagonal().array() -= eivals(l);

          VectorType dummy;
          extract_kernel(tmp, eivecs.col(l), dummy);
        }

        // Compute last eigenvector from the other two
        eivecs.col(1) = eivecs.col(2).cross(eivecs.col(0)).normalized();
      }
    }
  }
};

// 2x2 direct eigenvalues decomposition, code from Hauke Heibel
template <typename SolverType>
struct direct_selfadjoint_eigensolver_kernel<SolverType, 2> {
  using MatrixType = typename SolverType::MatrixType;
  using VectorType = typename SolverType::RealVectorType;
  using Scalar = typename SolverType::Scalar;
  using EigenvectorsType = typename SolverType::EigenvectorsType;
  using PlainMatrixType = typename SolverType::PlainMatrixType;

  EIGEN_DEVICE_FUNC static inline void computeRoots(const MatrixType& m, VectorType& roots) {
    EIGEN_USING_STD(sqrt);
    const Scalar t0 = Scalar(0.5) * sqrt(numext::abs2(m(0, 0) - m(1, 1)) + Scalar(4) * numext::abs2(m(1, 0)));
    const Scalar t1 = Scalar(0.5) * (m(0, 0) + m(1, 1));
    roots(0) = t1 - t0;
    roots(1) = t1 + t0;
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE static void run(PlainMatrixType& scaledMat, VectorType& eivals,
                                                        EigenvectorsType& eivecs, bool computeEigenvectors,
                                                        const Scalar&) {
    EIGEN_USING_STD(sqrt);
    EIGEN_USING_STD(abs);

    // Compute the eigenvalues
    computeRoots(scaledMat, eivals);

    // compute the eigen vectors
    if (computeEigenvectors) {
      if ((eivals(1) - eivals(0)) <= abs(eivals(1)) * Eigen::NumTraits<Scalar>::epsilon()) {
        eivecs.setIdentity();
      } else {
        scaledMat.diagonal().array() -= eivals(1);
        Scalar a2 = numext::abs2(scaledMat(0, 0));
        Scalar c2 = numext::abs2(scaledMat(1, 1));
        Scalar b2 = numext::abs2(scaledMat(1, 0));
        if (a2 > c2) {
          eivecs.col(1) << -scaledMat(1, 0), scaledMat(0, 0);
          eivecs.col(1) /= sqrt(a2 + b2);
        } else {
          eivecs.col(1) << -scaledMat(1, 1), scaledMat(1, 0);
          eivecs.col(1) /= sqrt(c2 + b2);
        }

        // The partner is already normalized; this rotation preserves its norm.
        eivecs.col(0) << -eivecs(1, 1), eivecs(0, 1);
      }
    }
  }
};

template <typename SolverType, int Size>
struct direct_selfadjoint_eigenvalues<SolverType, Size, false, true> {
  using MatrixType = typename SolverType::MatrixType;
  using PlainMatrixType = typename SolverType::PlainMatrixType;
  using Scalar = typename SolverType::Scalar;

  // Only host-side long double limits need more range than double.
  using Limit = std::conditional_t<std::is_same<Scalar, long double>::value, long double, double>;

  EIGEN_DEVICE_FUNC static constexpr Limit power_of_two(int exponent) {
    Limit result = 1;
    for (; exponent > 0; --exponent) result *= 2;
    for (; exponent < 0; ++exponent) result /= 2;
    return result;
  }

  EIGEN_DEVICE_FUNC static bool safe_without_scaling(const Scalar& magnitude, true_type) {
    // The quadratic/cubic kernels have degree 2/6. Reserve 12 exponent bits
    // for intermediate growth and keep flushed terms below epsilon*mu^degree.
    constexpr int degree = Size == 3 ? 6 : 2;
    constexpr int lower = std::numeric_limits<Scalar>::min_exponent - 1 + std::numeric_limits<Scalar>::digits - 1 + 12;
    constexpr int upper = std::numeric_limits<Scalar>::max_exponent - 1 - 12;
    constexpr int minExponent = lower >= 0 ? (lower + degree - 1) / degree : lower / degree;
    constexpr int maxExponent = upper >= 0 ? upper / degree : (upper - degree + 1) / degree;
    constexpr Limit minimum = power_of_two(minExponent);
    constexpr Limit maximum = power_of_two(maxExponent);
    return magnitude >= Scalar(minimum) && magnitude <= Scalar(maximum);
  }

  EIGEN_DEVICE_FUNC static bool safe_without_scaling(const Scalar& magnitude, false_type) {
    return magnitude >= Scalar(0.25) && magnitude <= Scalar(2) &&
           NumTraits<Scalar>::epsilon() / Scalar(4096) >= (std::numeric_limits<Scalar>::min)();
  }

  EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE static void run_scaled(SolverType& solver, const MatrixType& mat,
                                                             const Scalar& shift, Scalar centeredMax, int options) {
    PlainMatrixType scaledMat = mat.template selfadjointView<Lower>();
    scaledMat.diagonal().array() -= shift;
    centeredMax = safe_scaling<Scalar>::recover_flushed_max_coeff(scaledMat, centeredMax);
    const auto factors = safe_scaling<Scalar>::scale_to(scaledMat, scaledMat, centeredMax);
    direct_selfadjoint_eigensolver_kernel<SolverType, Size>::run(scaledMat, solver.m_eivalues, solver.m_eivec,
                                                                 (options & ComputeEigenvectors) == ComputeEigenvectors,
                                                                 scaledMat.cwiseAbs().maxCoeff());
    safe_scaling<Scalar>::unscale_in_place(solver.m_eivalues, centeredMax, factors);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE static void run_centered(SolverType& solver, const MatrixType& mat,
                                                                 int options) {
    const bool computeEigenvectors = (options & ComputeEigenvectors) == ComputeEigenvectors;
    PlainMatrixType scaledMat = mat.template selfadjointView<Lower>();
    const Scalar first = scaledMat(0, 0);
    Scalar shift = first;
    for (Index i = 1; i < Size; ++i) shift += (scaledMat(i, i) - first) / Scalar(Size);
    scaledMat.diagonal().array() -= shift;
    // Centering can leave a much smaller matrix; normalize again for the cubic and cross-product formulas.
    const Scalar centeredMax = scaledMat.cwiseAbs().maxCoeff();
    if (safe_without_scaling(centeredMax, supports_power_of_two_scaling<Scalar>())) {
      direct_selfadjoint_eigensolver_kernel<SolverType, Size>::run(scaledMat, solver.m_eivalues, solver.m_eivec,
                                                                   computeEigenvectors, centeredMax);
    } else {
      run_scaled(solver, mat, shift, centeredMax, options);
    }
    solver.m_eivalues.array() += shift;
    solver.m_info = Success;
    solver.m_isInitialized = true;
    solver.m_eigenvectorsOk = computeEigenvectors;
  }

  EIGEN_DEVICE_FUNC static void run_large(SolverType& solver, PlainMatrixType& scaledMat, int options, false_type) {
    run_centered(solver, scaledMat, options);
  }

  EIGEN_DEVICE_FUNC static void run_large(SolverType& solver, PlainMatrixType& scaledMat, int options, true_type) {
    // The cubic loses accuracy near repeated roots; at the exponent limit that error can overflow a finite result.
    solver.compute(scaledMat, options);
  }

  EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE static void run_prescaled(SolverType& solver, const MatrixType& mat,
                                                                const Scalar& maxCoeff, int options) {
    PlainMatrixType scaledMat = mat.template selfadjointView<Lower>();
    const Scalar recoveredMax = safe_scaling<Scalar>::recover_flushed_max_coeff(scaledMat, maxCoeff);
    const auto inputFactors = safe_scaling<Scalar>::scale_to(scaledMat, scaledMat, recoveredMax);
    if (maxCoeff > (NumTraits<Scalar>::highest() * Scalar(0.5)) / Scalar(Size)) {
      run_large(solver, scaledMat, options,
                bool_constant<(Size == 3 && supports_power_of_two_scaling<Scalar>::value)>());
    } else {
      run_centered(solver, scaledMat, options);
    }
    safe_scaling<Scalar>::unscale_in_place(solver.m_eivalues, recoveredMax, inputFactors);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE static void run(SolverType& solver, const MatrixType& mat, int options) {
    eigen_assert(mat.rows() == Size && mat.cols() == Size);
    eigen_assert((options & ~(EigVecMask | GenEigMask)) == 0 && (options & EigVecMask) != EigVecMask &&
                 "invalid option parameter");
    PlainMatrixType scaledMat = mat.template selfadjointView<Lower>();
    const Scalar maxCoeff = scaledMat.cwiseAbs().maxCoeff();
    // Bound centered coefficients by 2*M and eigenvalues by Size*M. Prescale tiny inputs before any FTZ-sensitive
    // subtraction, and large inputs before centering or restoring the centered eigenvalues could overflow.
    // Halve highest() first: IBM double-double division can overflow internally at the maximum value.
    if (EIGEN_PREDICT_FALSE(maxCoeff < (std::numeric_limits<Scalar>::min)() / NumTraits<Scalar>::epsilon() ||
                            maxCoeff > (NumTraits<Scalar>::highest() * Scalar(0.5)) / Scalar(Size))) {
      run_prescaled(solver, mat, maxCoeff, options);
    } else {
      run_centered(solver, mat, options);
    }
  }
};

}  // namespace internal

template <typename MatrixType>
EIGEN_DEVICE_FUNC SelfAdjointEigenSolver<MatrixType>& SelfAdjointEigenSolver<MatrixType>::computeDirect(
    const MatrixType& matrix, int options) {
  internal::direct_selfadjoint_eigenvalues<SelfAdjointEigenSolver, Size, NumTraits<Scalar>::IsComplex>::run(
      *this, matrix, options);
  return *this;
}

namespace internal {

// Francis implicit QR step; the rotations accumulate into *matrixQ when it is non-null.
template <typename RealScalar, typename Index, typename MatrixQType>
EIGEN_DEVICE_FUNC static void tridiagonal_qr_step(RealScalar* diag, RealScalar* subdiag, Index start, Index end,
                                                  MatrixQType* matrixQ) {
  // Wilkinson Shift.
  RealScalar td = (diag[end - 1] - diag[end]) * RealScalar(0.5);
  RealScalar e = subdiag[end - 1];
  RealScalar mu = diag[end];
  if (numext::is_exactly_zero(td)) {
    mu -= numext::abs(e);
  } else if (!numext::is_exactly_zero(e)) {
    const RealScalar e2 = numext::abs2(e);
    if (numext::is_exactly_zero(e2)) {
      // Scaling prevents e^2 and td^2 from overflowing, but e^2 can still underflow. A subdiagonal that survives
      // deflation satisfies (e/epsilon)^2 > 2*abs(td), so td/e remains bounded in this branch. LAPACK's xSTEQR uses
      // this dimensionless form; keeping td/e inside hypot prevents fast-math from reassociating the correction into
      // an underflowing e^2 expression.
      const RealScalar ratio = td / e;
      const RealScalar h = numext::hypot(ratio, RealScalar(1));
      mu -= e / (ratio + (ratio > RealScalar(0) ? h : -h));
    } else {
      const RealScalar h = numext::hypot(td, e);
      mu -= e2 / (td + (td > RealScalar(0) ? h : -h));
    }
  }

  RealScalar x = diag[start] - mu;
  RealScalar z = subdiag[start];
  // If z ever becomes zero, the Givens rotation will be the identity and
  // z will stay zero for all future iterations.
  for (Index k = start; k < end && !numext::is_exactly_zero(z); ++k) {
    JacobiRotation<RealScalar> rot;
    rot.makeGivens(x, z);

    // do T = G' T G. The two new diagonal entries and the new off-diagonal are all expressed below
    // as functions of the original difference diff = diag[k] - diag[k+1] and the original subdiag[k],
    // so capture diff and compute delta/subdiag[k] before overwriting the diagonal.
    const RealScalar diff = diag[k] - diag[k + 1];

    // Update the 2x2 diagonal block by the symmetric change delta, applied with opposite signs to
    // diag[k] and diag[k+1]. A similarity leaves the trace invariant, so this preserves diag[k] +
    // diag[k+1] exactly. Forming the two new diagonal entries from independent explicit formulas
    // (diag[k] = c^2 d_k - 2 c s e_k + s^2 d_{k+1}, diag[k+1] = s^2 d_k + 2 c s e_k + c^2 d_{k+1})
    // instead lets the two roundings drift the trace a little on every rotation, an error that
    // accumulates over the O(n^2) rotations of a full solve and measurably degrades the eigenvalues.
    // Algebraically delta = s^2 (d_k - d_{k+1}) + 2 c s e_k, which is exactly d_k minus the formula above.
    const RealScalar delta = rot.s() * (rot.s() * diff + RealScalar(2) * rot.c() * subdiag[k]);
    // The exact (G' T G) off-diagonal, c*s*(d_k - d_{k+1}) + (c^2 - s^2)*e_k, written in the same diff
    // form rather than via the c*sdk - s*dkp1 temporaries.
    subdiag[k] = rot.c() * rot.s() * diff + (rot.c() * rot.c() - rot.s() * rot.s()) * subdiag[k];
    diag[k] -= delta;
    diag[k + 1] += delta;

    if (k > start) subdiag[k - 1] = rot.c() * subdiag[k - 1] - rot.s() * z;

    // "Chasing the bulge" to return to triangular form.
    x = subdiag[k];
    if (k < end - 1) {
      z = -rot.s() * subdiag[k + 1];
      subdiag[k + 1] = rot.c() * subdiag[k + 1];
    }

    // apply the givens rotation to the unit matrix Q = Q * G
    if (matrixQ) {
      // Contiguous column-major storage is rotated through a dynamic-size map: the rotation kernel then dispatches
      // on the runtime alignment for fixed-size matrices too, and rounds as it always did. Other storage (a
      // row-major Ref<>) is rotated through the matrix type itself.
      EIGEN_IF_CONSTEXPR (!MatrixQType::IsRowMajor && int(MatrixQType::InnerStrideAtCompileTime) == 1) {
        using Scalar = typename MatrixQType::Scalar;
        Map<Matrix<Scalar, Dynamic, Dynamic, ColMajor>, Unaligned, OuterStride<>> q(
            matrixQ->data(), matrixQ->rows(), matrixQ->cols(), OuterStride<>(matrixQ->outerStride()));
        q.applyOnTheRight(k, k + 1, rot);
      } else {
        matrixQ->applyOnTheRight(k, k + 1, rot);
      }
    }
  }
}

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_SELFADJOINTEIGENSOLVER_H
