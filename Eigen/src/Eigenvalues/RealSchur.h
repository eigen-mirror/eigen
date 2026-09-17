// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010,2012 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_REAL_SCHUR_H
#define EIGEN_REAL_SCHUR_H

#include "./HessenbergDecomposition.h"

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename MatrixType_>
class EigenSolver;

/** \eigenvalues_module \ingroup Eigenvalues_Module
 *
 *
 * \class RealSchur
 *
 * \brief Performs a real Schur decomposition of a square matrix
 *
 * \tparam MatrixType_ the type of the matrix of which we are computing the
 * real Schur decomposition; this is expected to be an instantiation of the
 * Matrix class template.
 *
 * Given a real square matrix A, this class computes the real Schur
 * decomposition: \f$ A = U T U^T \f$ where U is a real orthogonal matrix and
 * T is a real quasi-triangular matrix. An orthogonal matrix is a matrix whose
 * inverse is equal to its transpose, \f$ U^{-1} = U^T \f$. A quasi-triangular
 * matrix is a block-triangular matrix whose diagonal consists of 1-by-1
 * blocks and 2-by-2 blocks with complex eigenvalues. The eigenvalues of the
 * blocks on the diagonal of T are the same as the eigenvalues of the matrix
 * A, and thus the real Schur decomposition is used in EigenSolver to compute
 * the eigendecomposition of a matrix.
 *
 * Call the function compute() to compute the real Schur decomposition of a
 * given matrix. Alternatively, you can use the RealSchur(const MatrixType&, bool)
 * constructor which computes the real Schur decomposition at construction
 * time. Once the decomposition is computed, you can use the matrixU() and
 * matrixT() functions to retrieve the matrices U and T in the decomposition.
 *
 * The documentation of RealSchur(const MatrixType&, bool) contains an example
 * of the typical use of this class.
 *
 * \note The implementation is adapted from
 * <a href="http://math.nist.gov/javanumerics/jama/">JAMA</a> (public domain).
 * Their code is based on EISPACK.
 *
 * \sa class ComplexSchur, class EigenSolver, class ComplexEigenSolver
 */
template <typename MatrixType_>
class RealSchur {
 public:
  using MatrixType = MatrixType_;
  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    Options = internal::plain_object_options<MatrixType>::value,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
  };
  using Scalar = typename MatrixType::Scalar;
  using ComplexScalar = internal::make_complex_t<Scalar>;
  using Index = Eigen::Index;  ///< \deprecated since Eigen 3.3

  using EigenvalueType = Matrix<ComplexScalar, ColsAtCompileTime, 1, Options & ~RowMajor, MaxColsAtCompileTime, 1>;
  using ColumnVectorType = Matrix<Scalar, ColsAtCompileTime, 1, Options & ~RowMajor, MaxColsAtCompileTime, 1>;

  /** \brief Type of the matrix returned by matrixU(): a plain matrix with the shape and storage options of
   * \p MatrixType_, and \p MatrixType_ itself unless that is a Ref<>. */
  using MatrixUType =
      Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options, MaxRowsAtCompileTime, MaxColsAtCompileTime>;

  /** \brief Default constructor.
   *
   * \param [in] size  Positive integer, size of the matrix whose Schur decomposition will be computed.
   *
   * The default constructor is useful in cases in which the user intends to
   * perform decompositions via compute().  The \p size parameter is only
   * used as a hint. It is not an error to give a wrong \p size, but it may
   * impair performance.
   *
   * \sa compute() for an example.
   */
  explicit RealSchur(Index size = RowsAtCompileTime == Dynamic ? 1 : RowsAtCompileTime)
      : m_matT(size, size),
        m_matU(size, size),
        m_workspaceVector(size),
        m_isInitialized(false),
        m_matUisUptodate(false),
        m_maxIters(-1) {
    if (size > 1) m_hCoeffs.resize(size - 1);
  }

  /** \brief Constructor; computes real Schur decomposition of given matrix.
   *
   * \param[in]  matrix    Square matrix whose Schur decomposition is to be computed.
   * \param[in]  computeU  If true, both T and U are computed; if false, only T is computed.
   *
   * This constructor calls compute() to compute the Schur decomposition.
   *
   * Example: \include RealSchur_RealSchur_MatrixType.cpp
   * Output: \verbinclude RealSchur_RealSchur_MatrixType.out
   */
  template <typename InputType>
  explicit RealSchur(const EigenBase<InputType>& matrix, bool computeU = true)
      : m_matT(matrix.rows(), matrix.cols()),
        m_matU(matrix.rows(), matrix.cols()),
        m_workspaceVector(matrix.rows()),
        m_isInitialized(false),
        m_matUisUptodate(false),
        m_maxIters(-1) {
    compute(matrix.derived(), computeU);
  }

  /** \brief Constructor for \link InplaceDecomposition inplace decomposition \endlink
   *
   * \param[in,out]  matrix    Square matrix whose Schur decomposition is to be computed.
   * \param[in]      computeU  If true, both T and U are computed; if false, only T is computed.
   *
   * When \p MatrixType is a Ref<>, \p matrix holds the quasi-triangular matrix T returned by matrixT(); U is stored
   * in the decomposition object. For an n-by-n matrix with n >= 128 and an outer stride in bytes divisible by 1024,
   * the computation uses a temporary padded workspace of n*(n+1) coefficients and copies T back to \p matrix.
   * Otherwise, it computes directly in the bound storage.
   *
   * To avoid padding, bind a Ref<> whose outer stride in bytes is not divisible by 1024. EIGEN_NO_MALLOC, or
   * disabling allocation or deallocation through EIGEN_RUNTIME_NO_MALLOC, also suppresses padding, which may
   * reduce performance. These controls do not remove other allocations, such as solver storage or Schur vectors;
   * the inplace constructor is not guaranteed to be allocation-free.
   *
   * This overload is only available for Ref<>; owning matrix types use the constructor taking a const input.
   */
  template <typename InputType, bool IsRef = internal::is_ref<MatrixType>::value, std::enable_if_t<IsRef, int> = 0>
  explicit RealSchur(EigenBase<InputType>& matrix, bool computeU = true)
      : m_matT(matrix.derived()),
        m_matU(matrix.rows(), matrix.cols()),
        m_workspaceVector(matrix.rows()),
        m_isInitialized(false),
        m_matUisUptodate(false),
        m_maxIters(-1) {
    compute(m_matT, computeU);
  }

  /** \brief Returns the orthogonal matrix in the Schur decomposition.
   *
   * \returns A const reference to the matrix U.
   *
   * \pre Either the constructor RealSchur(const MatrixType&, bool) or the
   * member function compute(const MatrixType&, bool) has been called before
   * to compute the Schur decomposition of a matrix, and \p computeU was set
   * to true (the default value).
   *
   * \sa RealSchur(const MatrixType&, bool) for an example
   */
  const MatrixUType& matrixU() const {
    eigen_assert(m_isInitialized && "RealSchur is not initialized.");
    eigen_assert(m_matUisUptodate && "The matrix U has not been computed during the RealSchur decomposition.");
    return m_matU;
  }

  /** \brief Returns the quasi-triangular matrix in the Schur decomposition.
   *
   * \returns A const reference to the matrix T.
   *
   * \pre Either the constructor RealSchur(const MatrixType&, bool) or the
   * member function compute(const MatrixType&, bool) has been called before
   * to compute the Schur decomposition of a matrix.
   *
   * \sa RealSchur(const MatrixType&, bool) for an example
   */
  const MatrixType& matrixT() const {
    eigen_assert(m_isInitialized && "RealSchur is not initialized.");
    return m_matT;
  }

  /** \brief Computes Schur decomposition of given matrix.
   *
   * \param[in]  matrix    Square matrix whose Schur decomposition is to be computed.
   * \param[in]  computeU  If true, both T and U are computed; if false, only T is computed.
   * \returns    Reference to \c *this
   *
   * The Schur decomposition is computed by first reducing the matrix to
   * Hessenberg form using the class HessenbergDecomposition. The Hessenberg
   * matrix is then reduced to triangular form by performing Francis QR
   * iterations with implicit double shift. The cost of computing the Schur
   * decomposition depends on the number of iterations; as a rough guide, it
   * may be taken to be \f$25n^3\f$ flops if \a computeU is true and
   * \f$10n^3\f$ flops if \a computeU is false.
   *
   * To avoid cache-conflicting strides, this routine may allocate a temporary padded workspace even when
   * the solver was constructed with the correct size. With EIGEN_NO_MALLOC, or when allocation or deallocation
   * is disabled through EIGEN_RUNTIME_NO_MALLOC, it uses the unpadded workspace instead, which may be slower.
   * This only suppresses padding; resizing solver storage and constructing Schur vectors may still allocate.
   *
   * Example: \include RealSchur_compute.cpp
   * Output: \verbinclude RealSchur_compute.out
   *
   * \sa compute(const MatrixType&, bool, Index)
   */
  template <typename InputType>
  RealSchur& compute(const EigenBase<InputType>& matrix, bool computeU = true);

  /** \brief Computes Schur decomposition of a Hessenberg matrix H = Z T Z^T
   *  \param[in] matrixH Matrix in Hessenberg form H
   *  \param[in] matrixQ orthogonal matrix Q that transforms a matrix A to H : A = Q H Q^T
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
   * This routine may allocate a temporary padded workspace to avoid cache-conflicting strides. With EIGEN_NO_MALLOC,
   * or when allocation or deallocation is disabled through EIGEN_RUNTIME_NO_MALLOC, it uses the unpadded workspace
   * instead, which may be slower. Other allocations, such as resizing solver storage or evaluating input
   * expressions, are unaffected.
   *
   * \sa compute(const MatrixType&, bool)
   */
  template <typename HessMatrixType, typename OrthMatrixType>
  RealSchur& computeFromHessenberg(const HessMatrixType& matrixH, const OrthMatrixType& matrixQ, bool computeU);
  /** \brief Reports whether previous computation was successful.
   *
   * \returns \c Success if computation was successful, \c NoConvergence otherwise.
   */
  ComputationInfo info() const {
    eigen_assert(m_isInitialized && "RealSchur is not initialized.");
    return m_info;
  }

  /** \brief Sets the maximum number of iterations allowed.
   *
   * If not specified by the user, the maximum number of iterations is m_maxIterationsPerRow times the size
   * of the matrix.
   */
  RealSchur& setMaxIterations(Index maxIters) {
    m_maxIters = maxIters;
    return *this;
  }

  /** \brief Returns the maximum number of iterations. */
  Index getMaxIterations() const { return m_maxIters; }

  /** \brief Maximum number of iterations per row.
   *
   * If not otherwise specified, the maximum number of iterations is this number times the size of the
   * matrix. It is currently set to 40.
   */
  static const int m_maxIterationsPerRow = 40;

 private:
  // EigenSolver computes the eigenvectors of T and back-transforms U within this storage.
  friend class EigenSolver<MatrixType>;

  using CoeffVectorType =
      Matrix<Scalar, RowsAtCompileTime == Dynamic ? Dynamic : RowsAtCompileTime - 1, 1, Options & ~RowMajor,
             MaxRowsAtCompileTime == Dynamic ? Dynamic : MaxRowsAtCompileTime - 1, 1>;

  MatrixType m_matT;
  MatrixUType m_matU;
  ColumnVectorType m_workspaceVector;
  CoeffVectorType m_hCoeffs;
  ComputationInfo m_info;
  bool m_isInitialized;
  bool m_matUisUptodate;
  Index m_maxIters;

  template <typename TMatrix>
  EIGEN_DONT_INLINE RealSchur& computeFromHessenbergInPlace(TMatrix& matT, bool computeU);
  using Vector3s = Matrix<Scalar, 3, 1>;

  using WorkspaceMatrix = Matrix<Scalar, Dynamic, Dynamic, MatrixType::IsRowMajor ? RowMajor : ColMajor>;

  bool usePaddedWorkspace(Index size) const;
  template <typename TMatrix>
  RealSchur& computeInPlace(TMatrix& matT, bool computeU);
  template <typename TMatrix>
  Scalar computeNormOfT(TMatrix& matT);
  template <typename TMatrix>
  Index findSmallSubdiagEntry(TMatrix& matT, Index iu, const Scalar& considerAsZero);
  template <typename TMatrix>
  void splitOffTwoRows(TMatrix& matT, Index iu, bool computeU, const Scalar& exshift);
  template <typename TMatrix>
  void computeShift(TMatrix& matT, Index iu, Index iter, Scalar& exshift, Vector3s& shiftInfo);
  template <typename TMatrix>
  void initFrancisQRStep(TMatrix& matT, Index il, Index iu, const Vector3s& shiftInfo, Index& im,
                         Vector3s& firstHouseholderVector);
  template <typename TMatrix>
  void performFrancisQRStep(TMatrix& matT, Index il, Index im, Index iu, bool computeU,
                            const Vector3s& firstHouseholderVector, Scalar* workspace);
};

template <typename MatrixType>
template <typename InputType>
RealSchur<MatrixType>& RealSchur<MatrixType>::compute(const EigenBase<InputType>& matrix, bool computeU) {
  eigen_assert(matrix.cols() == matrix.rows());
  const Index size = matrix.rows();
  if (usePaddedWorkspace(size)) {
    WorkspaceMatrix storage(size + (MatrixType::IsRowMajor ? 0 : 1), size + (MatrixType::IsRowMajor ? 1 : 0));
    auto matT = storage.topLeftCorner(size, size);
    matT = matrix.derived();
    computeInPlace(matT, computeU);
  } else {
    if (!internal::is_same_dense(m_matT, matrix.derived())) m_matT = matrix.derived();
    computeInPlace(m_matT, computeU);
  }
  return *this;
}

template <typename MatrixType>
bool RealSchur<MatrixType>::usePaddedWorkspace(Index size) const {
  // Short reflectors traverse the outer dimension. Strides divisible by 1 KiB concentrate those accesses in
  // very few cache sets. A Ref may already have a suitable stride; owning matrices resize to size-by-size.
  const Index stride = internal::is_ref<MatrixType>::value ? m_matT.outerStride() : size;
  bool usePadding = size >= 128 && (stride * sizeof(Scalar)) % 1024 == 0;
#ifdef EIGEN_NO_MALLOC
  usePadding = false;
#elif defined(EIGEN_RUNTIME_NO_MALLOC)
  usePadding = usePadding && internal::is_malloc_allowed() && internal::is_free_allowed();
#endif
  return usePadding;
}

/** \internal Reduces matT in place and writes the rescaled Schur form to m_matT. */
template <typename MatrixType>
template <typename TMatrix>
RealSchur<MatrixType>& RealSchur<MatrixType>::computeInPlace(TMatrix& matT, bool computeU) {
  const Scalar considerAsZero = (std::numeric_limits<Scalar>::min)();
  const Index n = matT.rows();
  eigen_assert(matT.cols() == n);

  const Scalar maxCoeff = matT.cwiseAbs().template maxCoeff<PropagateNaN>();
  if (!(numext::isfinite)(maxCoeff)) {
    if (!internal::is_same_dense(m_matT, matT)) m_matT = matT;
    m_info = NoConvergence;
    m_isInitialized = true;
    m_matUisUptodate = false;
    return *this;
  }
  if (maxCoeff < considerAsZero) {
    // Ref has no setZero(rows, cols) overload.
    m_matT.resize(n, n);
    m_matT.setZero();
    if (computeU) m_matU.setIdentity(n, n);
    m_info = Success;
    m_isInitialized = true;
    m_matUisUptodate = computeU;
    return *this;
  }
  const auto factors = internal::safe_scaling<Scalar>::scale_in_place(matT, maxCoeff);

  // Step 1. Reduce to Hessenberg form
  internal::hessenberg_decomposition_inplace(matT, m_hCoeffs, m_workspaceVector, m_matU, computeU);

  // Step 2. Reduce to real Schur form
  computeFromHessenbergInPlace(matT, computeU);

  // Keep aliasing explicit for the compiler on the in-place path.
  if (internal::is_same_dense(m_matT, matT))
    internal::safe_scaling<Scalar>::unscale_in_place(matT, maxCoeff, factors);
  else
    internal::safe_scaling<Scalar>::unscale_to(m_matT, matT, maxCoeff, factors);

  return *this;
}
template <typename MatrixType>
template <typename HessMatrixType, typename OrthMatrixType>
RealSchur<MatrixType>& RealSchur<MatrixType>::computeFromHessenberg(const HessMatrixType& matrixH,
                                                                    const OrthMatrixType& matrixQ, bool computeU) {
  const Index size = matrixH.rows();
  m_workspaceVector.resize(size);
  if (usePaddedWorkspace(size)) {
    WorkspaceMatrix storage(size + (MatrixType::IsRowMajor ? 0 : 1), size + (MatrixType::IsRowMajor ? 1 : 0));
    auto matT = storage.topLeftCorner(size, size);
    matT = matrixH;
    if (computeU && !internal::is_same_dense(m_matU, matrixQ)) m_matU = matrixQ;
    computeFromHessenbergInPlace(matT, computeU);
    m_matT = matT;
  } else {
    if (!internal::is_same_dense(m_matT, matrixH)) m_matT = matrixH;
    if (computeU && !internal::is_same_dense(m_matU, matrixQ)) m_matU = matrixQ;
    computeFromHessenbergInPlace(m_matT, computeU);
  }
  return *this;
}

template <typename MatrixType>
template <typename TMatrix>
RealSchur<MatrixType>& RealSchur<MatrixType>::computeFromHessenbergInPlace(TMatrix& matT, bool computeU) {
  Index maxIters = m_maxIters;
  if (maxIters == -1) maxIters = m_maxIterationsPerRow * matT.rows();
  Scalar* workspace = &m_workspaceVector.coeffRef(0);

  // The matrix matT is divided in three parts.
  // Rows 0,...,il-1 are decoupled from the rest because matT(il,il-1) is zero.
  // Rows il,...,iu is the part we are working on (the active window).
  // Rows iu+1,...,end are already brought in triangular form.
  Index iu = matT.cols() - 1;
  Index iter = 0;       // iteration count for current eigenvalue
  Index totalIter = 0;  // iteration count for whole matrix
  Scalar exshift(0);    // sum of exceptional shifts
  Scalar norm = computeNormOfT(matT);
  // sub-diagonal entries smaller than considerAsZero will be treated as zero.
  // We use eps^2 to enable more precision in small eigenvalues.
  Scalar considerAsZero =
      numext::maxi<Scalar>(norm * numext::abs2(NumTraits<Scalar>::epsilon()), (std::numeric_limits<Scalar>::min)());

  if (!numext::is_exactly_zero(norm)) {
    while (iu >= 0) {
      Index il = findSmallSubdiagEntry(matT, iu, considerAsZero);

      // Check for convergence
      if (il == iu)  // One root found
      {
        matT.coeffRef(iu, iu) = matT.coeff(iu, iu) + exshift;
        if (iu > 0) matT.coeffRef(iu, iu - 1) = Scalar(0);
        iu--;
        iter = 0;
      } else if (il == iu - 1)  // Two roots found
      {
        splitOffTwoRows(matT, iu, computeU, exshift);
        iu -= 2;
        iter = 0;
      } else  // No convergence yet
      {
        // The firstHouseholderVector vector has to be initialized to something to get rid of a silly GCC warning (-O1
        // -Wall -DNDEBUG )
        Vector3s firstHouseholderVector = Vector3s::Zero(), shiftInfo;
        computeShift(matT, iu, iter, exshift, shiftInfo);
        iter = iter + 1;
        totalIter = totalIter + 1;
        if (totalIter > maxIters) break;
        Index im;
        initFrancisQRStep(matT, il, iu, shiftInfo, im, firstHouseholderVector);
        performFrancisQRStep(matT, il, im, iu, computeU, firstHouseholderVector, workspace);
      }
    }
  }
  if (totalIter <= maxIters)
    m_info = Success;
  else
    m_info = NoConvergence;

  m_isInitialized = true;
  m_matUisUptodate = computeU;
  return *this;
}

/** \internal Computes and returns vector L1 norm of T */
template <typename MatrixType>
template <typename TMatrix>
inline typename MatrixType::Scalar RealSchur<MatrixType>::computeNormOfT(TMatrix& matT) {
  return internal::hessenberg_abs_sum<Upper>(matT);
}

/** \internal Look for single small sub-diagonal element and returns its index */
template <typename MatrixType>
template <typename TMatrix>
inline Index RealSchur<MatrixType>::findSmallSubdiagEntry(TMatrix& matT, Index iu, const Scalar& considerAsZero) {
  using std::abs;
  Index res = iu;
  while (res > 0) {
    Scalar s = abs(matT.coeff(res - 1, res - 1)) + abs(matT.coeff(res, res));

    s = numext::maxi<Scalar>(s * NumTraits<Scalar>::epsilon(), considerAsZero);

    if (abs(matT.coeff(res, res - 1)) <= s) break;
    res--;
  }
  return res;
}

/** \internal Update T given that rows iu-1 and iu decouple from the rest. */
template <typename MatrixType>
template <typename TMatrix>
inline void RealSchur<MatrixType>::splitOffTwoRows(TMatrix& matT, Index iu, bool computeU, const Scalar& exshift) {
  using std::abs;
  using std::sqrt;
  const Index size = matT.cols();

  // The eigenvalues of the 2x2 matrix [a b; c d] are
  // trace +/- sqrt(discr/4) where discr = tr^2 - 4*det, tr = a + d, det = ad - bc
  Scalar p = Scalar(0.5) * (matT.coeff(iu - 1, iu - 1) - matT.coeff(iu, iu));
  Scalar q = p * p + matT.coeff(iu, iu - 1) * matT.coeff(iu - 1, iu);  // q = tr^2 / 4 - det = discr/4
  matT.coeffRef(iu, iu) += exshift;
  matT.coeffRef(iu - 1, iu - 1) += exshift;

  if (q >= Scalar(0))  // Two real eigenvalues
  {
    Scalar z = sqrt(abs(q));
    JacobiRotation<Scalar> rot;
    if (p >= Scalar(0))
      rot.makeGivens(p + z, matT.coeff(iu, iu - 1));
    else
      rot.makeGivens(p - z, matT.coeff(iu, iu - 1));

    matT.rightCols(size - iu + 1).applyOnTheLeft(iu - 1, iu, rot.adjoint());
    matT.topRows(iu + 1).applyOnTheRight(iu - 1, iu, rot);
    matT.coeffRef(iu, iu - 1) = Scalar(0);
    if (computeU) m_matU.applyOnTheRight(iu - 1, iu, rot);
  }

  if (iu > 1) matT.coeffRef(iu - 1, iu - 2) = Scalar(0);
}

/** \internal Form shift in shiftInfo, and update exshift if an exceptional shift is performed. */
template <typename MatrixType>
template <typename TMatrix>
inline void RealSchur<MatrixType>::computeShift(TMatrix& matT, Index iu, Index iter, Scalar& exshift,
                                                Vector3s& shiftInfo) {
  using std::abs;
  using std::sqrt;
  shiftInfo.coeffRef(0) = matT.coeff(iu, iu);
  shiftInfo.coeffRef(1) = matT.coeff(iu - 1, iu - 1);
  shiftInfo.coeffRef(2) = matT.coeff(iu, iu - 1) * matT.coeff(iu - 1, iu);

  // Alternate exceptional shifting strategy every 16 iterations.
  if (iter > 0 && iter % 16 == 0) {
    // Wilkinson's original ad hoc shift
    if (iter % 32 != 0) {
      exshift += shiftInfo.coeff(0);
      matT.diagonal().head(iu + 1).array() -= shiftInfo.coeff(0);
      Scalar s = abs(matT.coeff(iu, iu - 1)) + abs(matT.coeff(iu - 1, iu - 2));
      shiftInfo.coeffRef(0) = Scalar(0.75) * s;
      shiftInfo.coeffRef(1) = Scalar(0.75) * s;
      shiftInfo.coeffRef(2) = Scalar(-0.4375) * s * s;
    } else {
      // MATLAB's new ad hoc shift
      Scalar s = (shiftInfo.coeff(1) - shiftInfo.coeff(0)) / Scalar(2.0);
      s = s * s + shiftInfo.coeff(2);
      if (s > Scalar(0)) {
        s = sqrt(s);
        if (shiftInfo.coeff(1) < shiftInfo.coeff(0)) s = -s;
        s = s + (shiftInfo.coeff(1) - shiftInfo.coeff(0)) / Scalar(2.0);
        s = shiftInfo.coeff(0) - shiftInfo.coeff(2) / s;
        exshift += s;
        matT.diagonal().head(iu + 1).array() -= s;
        shiftInfo.setConstant(Scalar(0.964));
      }
    }
  }
}

/** \internal Compute index im at which Francis QR step starts and the first Householder vector. */
template <typename MatrixType>
template <typename TMatrix>
inline void RealSchur<MatrixType>::initFrancisQRStep(TMatrix& matT, Index il, Index iu, const Vector3s& shiftInfo,
                                                     Index& im, Vector3s& firstHouseholderVector) {
  using std::abs;
  Vector3s& v = firstHouseholderVector;  // alias to save typing

  for (im = iu - 2; im >= il; --im) {
    const Scalar Tmm = matT.coeff(im, im);
    const Scalar r = shiftInfo.coeff(0) - Tmm;
    const Scalar s = shiftInfo.coeff(1) - Tmm;
    v.coeffRef(0) = (r * s - shiftInfo.coeff(2)) / matT.coeff(im + 1, im) + matT.coeff(im, im + 1);
    v.coeffRef(1) = matT.coeff(im + 1, im + 1) - Tmm - r - s;
    v.coeffRef(2) = matT.coeff(im + 2, im + 1);
    if (im == il) {
      break;
    }
    const Scalar lhs = matT.coeff(im, im - 1) * (abs(v.coeff(1)) + abs(v.coeff(2)));
    const Scalar rhs = v.coeff(0) * (abs(matT.coeff(im - 1, im - 1)) + abs(Tmm) + abs(matT.coeff(im + 1, im + 1)));
    if (abs(lhs) < NumTraits<Scalar>::epsilon() * rhs) break;
  }
}

/** \internal Perform a Francis QR step involving rows il:iu and columns im:iu. */
template <typename MatrixType>
template <typename TMatrix>
inline void RealSchur<MatrixType>::performFrancisQRStep(TMatrix& matT, Index il, Index im, Index iu, bool computeU,
                                                        const Vector3s& firstHouseholderVector, Scalar* workspace) {
  eigen_assert(im >= il);
  eigen_assert(im <= iu - 2);

  const Index size = matT.cols();

  for (Index k = im; k <= iu - 2; ++k) {
    bool firstIteration = (k == im);

    Vector3s v;
    if (firstIteration)
      v = firstHouseholderVector;
    else
      v = matT.template block<3, 1>(k, k - 1);

    Scalar tau, beta;
    Matrix<Scalar, 2, 1> ess;
    v.makeHouseholder(ess, tau, beta);

    if (!numext::is_exactly_zero(beta))  // if v is not zero
    {
      if (firstIteration && k > il)
        matT.coeffRef(k, k - 1) = -matT.coeff(k, k - 1);
      else if (!firstIteration)
        matT.coeffRef(k, k - 1) = beta;

      // These Householder transformations form the O(n^3) part of the algorithm
      matT.block(k, k, 3, size - k).applyHouseholderOnTheLeft(ess, tau, workspace);
      matT.block(0, k, (std::min)(iu, k + 3) + 1, 3).applyHouseholderOnTheRight(ess, tau, workspace);
      if (computeU) m_matU.block(0, k, size, 3).applyHouseholderOnTheRight(ess, tau, workspace);
    }
  }

  Matrix<Scalar, 2, 1> v = matT.template block<2, 1>(iu - 1, iu - 2);
  Scalar tau, beta;
  Matrix<Scalar, 1, 1> ess;
  v.makeHouseholder(ess, tau, beta);

  if (!numext::is_exactly_zero(beta))  // if v is not zero
  {
    matT.coeffRef(iu - 1, iu - 2) = beta;
    matT.block(iu - 1, iu - 1, 2, size - iu + 1).applyHouseholderOnTheLeft(ess, tau, workspace);
    matT.block(0, iu - 1, iu + 1, 2).applyHouseholderOnTheRight(ess, tau, workspace);
    if (computeU) m_matU.block(0, iu - 1, size, 2).applyHouseholderOnTheRight(ess, tau, workspace);
  }

  // clean up pollution due to round-off errors
  for (Index i = im + 2; i <= iu; ++i) {
    matT.coeffRef(i, i - 2) = Scalar(0);
    if (i > im + 2) matT.coeffRef(i, i - 3) = Scalar(0);
  }
}

}  // end namespace Eigen

#endif  // EIGEN_REAL_SCHUR_H
