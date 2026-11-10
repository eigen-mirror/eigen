// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012-2016 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010,2012 Jitse Niesen <jitse@maths.leeds.ac.uk>
// Copyright (C) 2016 Tobias Wood <tobias@spinicist.org.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_GENERALIZEDEIGENSOLVER_H
#define EIGEN_GENERALIZEDEIGENSOLVER_H

#include "./RealQZ.h"

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// T_i = diag(a, b) = s diag(a', b') with |a'|, |b'| <= 1, for a complex pair; returns beta = (s a') b' = a b'. float
// and double take s = 2^e, the power of two at or above max(|a|, |b|), through the representation, so a' and b' are
// exact and s need not be representable. Other scalars need not provide frexp or ldexp: s = max(|a|, |b|).
template <typename RealScalar>
RealScalar generalized_eigen_split_pair_diagonal(const RealScalar& a, const RealScalar& b, RealScalar& aScaled,
                                                 RealScalar& bScaled, true_type) {
  int e = 0;
  const RealScalar maxT = max_preserving_subnormals(abs_preserving_subnormals(a), abs_preserving_subnormals(b));
  if (numext::equal_strict(frexp_preserving_subnormals(maxT, e), RealScalar(0.5))) --e;
  aScaled = ldexp_preserving_subnormals(a, -e);
  bScaled = ldexp_preserving_subnormals(b, -e);
  return ldexp_preserving_subnormals(aScaled, e) * bScaled;
}
template <typename RealScalar>
RealScalar generalized_eigen_split_pair_diagonal(const RealScalar& a, const RealScalar& b, RealScalar& aScaled,
                                                 RealScalar& bScaled, false_type) {
  const RealScalar maxT = numext::maxi(numext::abs(a), numext::abs(b));
  aScaled = a;
  bScaled = b;
  if (!numext::is_exactly_zero(maxT)) {
    aScaled /= maxT;
    bScaled /= maxT;
  }
  return a * bScaled;
}

}  // namespace internal

/** \eigenvalues_module \ingroup Eigenvalues_Module
 *
 *
 * \class GeneralizedEigenSolver
 *
 * \brief Computes the generalized eigenvalues and eigenvectors of a pair of general matrices
 *
 * \tparam MatrixType_ the type of the matrices of which we are computing the
 * eigen-decomposition; this is expected to be an instantiation of the Matrix
 * class template. Currently, only real matrices are supported.
 *
 * The generalized eigenvalues and eigenvectors of a matrix pair \f$ A \f$ and \f$ B \f$ are scalars
 * \f$ \lambda \f$ and vectors \f$ v \f$ such that \f$ Av = \lambda Bv \f$.  If
 * \f$ D \f$ is a diagonal matrix with the eigenvalues on the diagonal, and
 * \f$ V \f$ is a matrix with the eigenvectors as its columns, then \f$ A V =
 * B V D \f$. The matrix \f$ V \f$ is almost always invertible, in which case we
 * have \f$ A = B V D V^{-1} \f$. This is called the generalized eigen-decomposition.
 *
 * The generalized eigenvalues and eigenvectors of a matrix pair may be complex, even when the
 * matrices are real. Moreover, the generalized eigenvalue might be infinite if the matrix B is
 * singular. To workaround this difficulty, the eigenvalues are provided as a pair of complex \f$ \alpha \f$
 * and real \f$ \beta \f$ such that: \f$ \lambda_i = \alpha_i / \beta_i \f$. If \f$ \beta_i \f$ is (nearly) zero,
 * then one can consider the well defined left eigenvalue \f$ \mu = \beta_i / \alpha_i\f$ such that:
 * \f$ \mu_i A v_i = B v_i \f$, or even \f$ \mu_i u_i^T A  = u_i^T B \f$ where \f$ u_i \f$ is
 * called the left eigenvector.
 *
 * Call the function compute() to compute the generalized eigenvalues and eigenvectors of
 * a given matrix pair. Alternatively, you can use the
 * GeneralizedEigenSolver(const MatrixType&, const MatrixType&, bool) constructor which computes the
 * eigenvalues and eigenvectors at construction time. Once the eigenvalue and
 * eigenvectors are computed, they can be retrieved with the eigenvalues() and
 * eigenvectors() functions.
 *
 * Here is an usage example of this class:
 * Example: \include GeneralizedEigenSolver.cpp
 * Output: \verbinclude GeneralizedEigenSolver.out
 *
 * \sa MatrixBase::eigenvalues(), class ComplexEigenSolver, class SelfAdjointEigenSolver
 */
template <typename MatrixType_>
class GeneralizedEigenSolver {
 public:
  /** \brief Synonym for the template parameter \p MatrixType_. */
  using MatrixType = MatrixType_;

  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    Options = internal::plain_object_options<MatrixType>::value,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
  };

  /** \brief Scalar type for matrices of type #MatrixType. */
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Index = Eigen::Index;  ///< \deprecated since Eigen 3.3

  /** \brief Complex scalar type for #MatrixType.
   *
   * This is \c std::complex<Scalar> if #Scalar is real (e.g.,
   * \c float or \c double) and just \c Scalar if #Scalar is
   * complex.
   */
  using ComplexScalar = internal::make_complex_t<Scalar>;

  /** \brief Type for vector of real scalar values eigenvalues as returned by betas().
   *
   * This is a column vector with entries of type #Scalar.
   * The length of the vector is the size of #MatrixType.
   */
  using VectorType = Matrix<Scalar, ColsAtCompileTime, 1, Options & ~RowMajor, MaxColsAtCompileTime, 1>;

  /** \brief Type for vector of complex scalar values eigenvalues as returned by alphas().
   *
   * This is a column vector with entries of type #ComplexScalar.
   * The length of the vector is the size of #MatrixType.
   */
  using ComplexVectorType = Matrix<ComplexScalar, ColsAtCompileTime, 1, Options & ~RowMajor, MaxColsAtCompileTime, 1>;

  /** \brief Expression type for the eigenvalues as returned by eigenvalues().
   */
  using EigenvalueType =
      CwiseBinaryOp<internal::scalar_quotient_op<ComplexScalar, Scalar>, ComplexVectorType, VectorType>;

  /** \brief Type for matrix of eigenvectors as returned by eigenvectors().
   *
   * This is a square matrix with entries of type #ComplexScalar.
   * The size is the same as the size of #MatrixType.
   */
  using EigenvectorsType =
      Matrix<ComplexScalar, RowsAtCompileTime, ColsAtCompileTime, Options, MaxRowsAtCompileTime, MaxColsAtCompileTime>;

  /** \brief Default constructor.
   *
   * The default constructor is useful in cases in which the user intends to
   * perform decompositions via EigenSolver::compute(const MatrixType&, bool).
   *
   * \sa compute() for an example.
   */
  GeneralizedEigenSolver()
      : m_eivec(), m_alphas(), m_betas(), m_computeEigenvectors(false), m_isInitialized(false), m_realQZ() {}

  /** \brief Default constructor with memory preallocation
   *
   * Like the default constructor but with preallocation of the internal data
   * according to the specified problem \a size.
   * \sa GeneralizedEigenSolver()
   */
  explicit GeneralizedEigenSolver(Index size)
      : m_eivec(size, size),
        m_alphas(size),
        m_betas(size),
        m_computeEigenvectors(false),
        m_isInitialized(false),
        m_realQZ(size),
        m_tmp(size) {}

  /** \brief Constructor; computes the generalized eigendecomposition of given matrix pair.
   *
   * \param[in]  A  Square matrix whose eigendecomposition is to be computed.
   * \param[in]  B  Square matrix whose eigendecomposition is to be computed.
   * \param[in]  computeEigenvectors  If true, both the eigenvectors and the
   *    eigenvalues are computed; if false, only the eigenvalues are computed.
   *
   * This constructor computes the generalized eigenvalues and eigenvectors
   * as compute() does.
   *
   * \sa compute()
   */
  template <typename InputTypeA, typename InputTypeB>
  GeneralizedEigenSolver(const EigenBase<InputTypeA>& A, const EigenBase<InputTypeB>& B,
                         bool computeEigenvectors = true)
      : m_eivec(A.rows(), A.cols()),
        m_alphas(A.cols()),
        m_betas(A.cols()),
        m_computeEigenvectors(false),
        m_isInitialized(false),
        m_realQZ(A.derived(), B.derived(), computeEigenvectors),
        m_tmp(A.cols()) {
    computeFromQZ(computeEigenvectors);
  }

  /** \brief Constructor for \link InplaceDecomposition inplace decomposition \endlink
   *
   * \param[in,out]  A  Square matrix whose eigendecomposition is to be computed.
   * \param[in,out]  B  Square matrix whose eigendecomposition is to be computed.
   * \param[in]  computeEigenvectors  If true, both the eigenvectors and the
   *    eigenvalues are computed; if false, only the eigenvalues are computed.
   *
   * When \p MatrixType is a Ref<>, the decomposition is computed within the memory of \p A and \p B, whose
   * contents are destroyed; the results are stored in the decomposition object. Otherwise this constructor behaves
   * like GeneralizedEigenSolver(const EigenBase<InputTypeA>&, const EigenBase<InputTypeB>&, bool).
   */
  template <typename InputTypeA, typename InputTypeB>
  GeneralizedEigenSolver(EigenBase<InputTypeA>& A, EigenBase<InputTypeB>& B, bool computeEigenvectors = true)
      : m_eivec(A.rows(), A.cols()),
        m_alphas(A.cols()),
        m_betas(A.cols()),
        m_computeEigenvectors(false),
        m_isInitialized(false),
        m_realQZ(A.derived(), B.derived(), computeEigenvectors),
        m_tmp(A.cols()) {
    computeFromQZ(computeEigenvectors);
  }

  /** \brief Returns the computed generalized eigenvectors.
   *
   * \returns  %Matrix whose columns are the (possibly complex) right eigenvectors.
   * i.e. the eigenvectors that solve (A - l*B)x = 0. The ordering matches the eigenvalues.
   *
   * \pre Either the constructor
   * GeneralizedEigenSolver(const MatrixType&,const MatrixType&, bool) or the member function
   * compute(const MatrixType&, const MatrixType& bool) has been called before, and
   * \p computeEigenvectors was set to true (the default).
   *
   * \sa eigenvalues()
   */
  EigenvectorsType eigenvectors() const {
    eigen_assert(info() == Success && "GeneralizedEigenSolver failed to compute eigenvectors");
    eigen_assert(m_computeEigenvectors && "Eigenvectors for GeneralizedEigenSolver were not calculated");
    return m_eivec;
  }

  /** \brief Returns an expression of the computed generalized eigenvalues.
   *
   * \returns An expression of the column vector containing the eigenvalues.
   *
   * It is a shortcut for \code this->alphas().cwiseQuotient(this->betas()); \endcode
   * Note that betas might contain zeros. It is therefore not recommended to use this function,
   * but rather directly deal with the alphas and betas vectors.
   *
   * \pre Either the constructor
   * GeneralizedEigenSolver(const MatrixType&,const MatrixType&,bool) or the member function
   * compute(const MatrixType&,const MatrixType&,bool) has been called before.
   *
   * The eigenvalues are repeated according to their algebraic multiplicity,
   * so there are as many eigenvalues as rows in the matrix. The eigenvalues
   * are not sorted in any particular order.
   *
   * \sa alphas(), betas(), eigenvectors()
   */
  EigenvalueType eigenvalues() const {
    eigen_assert(info() == Success && "GeneralizedEigenSolver failed to compute eigenvalues.");
    return EigenvalueType(m_alphas, m_betas);
  }

  /** \returns A const reference to the vectors containing the alpha values
   *
   * This vector permits to reconstruct the j-th eigenvalues as alphas(i)/betas(j).
   *
   * \sa betas(), eigenvalues() */
  const ComplexVectorType& alphas() const {
    eigen_assert(info() == Success && "GeneralizedEigenSolver failed to compute alphas.");
    return m_alphas;
  }

  /** \returns A const reference to the vectors containing the beta values
   *
   * This vector permits to reconstruct the j-th eigenvalues as alphas(i)/betas(j).
   *
   * \sa alphas(), eigenvalues() */
  const VectorType& betas() const {
    eigen_assert(info() == Success && "GeneralizedEigenSolver failed to compute betas.");
    return m_betas;
  }

  /** \brief Computes generalized eigendecomposition of given matrix.
   *
   * \param[in]  A  Square matrix whose eigendecomposition is to be computed.
   * \param[in]  B  Square matrix whose eigendecomposition is to be computed.
   * \param[in]  computeEigenvectors  If true, both the eigenvectors and the
   *    eigenvalues are computed; if false, only the eigenvalues are
   *    computed.
   * \returns    Reference to \c *this
   *
   * This function computes the eigenvalues of the real matrix \p matrix.
   * The eigenvalues() function can be used to retrieve them.  If
   * \p computeEigenvectors is true, then the eigenvectors are also computed
   * and can be retrieved by calling eigenvectors().
   *
   * The matrix is first reduced to real generalized Schur form using the RealQZ
   * class. The generalized Schur decomposition is then used to compute the eigenvalues
   * and eigenvectors.
   *
   * The cost of the computation is dominated by the cost of the
   * generalized Schur decomposition.
   *
   * This method reuses the allocated data in the GeneralizedEigenSolver object.
   */
  template <typename InputTypeA, typename InputTypeB>
  GeneralizedEigenSolver& compute(const EigenBase<InputTypeA>& A, const EigenBase<InputTypeB>& B,
                                  bool computeEigenvectors = true);

  ComputationInfo info() const {
    eigen_assert(m_isInitialized && "EigenSolver is not initialized.");
    return m_realQZ.info();
  }

  /** Sets the maximal number of iterations allowed.
   */
  GeneralizedEigenSolver& setMaxIterations(Index maxIters) {
    m_realQZ.setMaxIterations(maxIters);
    return *this;
  }

 protected:
  EIGEN_STATIC_ASSERT_NON_INTEGER(Scalar)
  EIGEN_STATIC_ASSERT(!NumTraits<Scalar>::IsComplex, NUMERIC_TYPE_MUST_BE_REAL)

  EigenvectorsType m_eivec;
  ComplexVectorType m_alphas;
  VectorType m_betas;
  bool m_computeEigenvectors;
  bool m_isInitialized;
  RealQZ<MatrixType> m_realQZ;
  ComplexVectorType m_tmp;

 private:
  GeneralizedEigenSolver& computeFromQZ(bool computeEigenvectors);

  // Solves m x = r with complete pivoting, replacing a pivot of modulus below smin by smin, after LAPACK's xLALN2.
  template <typename T>
  static Matrix<T, 2, 1> solveWithPerturbedPivots(const Matrix<T, 2, 2>& m, const Matrix<T, 2, 1>& r,
                                                  const RealScalar& smin) {
    Index row, col;
    m.cwiseAbs().maxCoeff(&row, &col);
    const Index otherRow = 1 - row, otherCol = 1 - col;
    T pivot = m.coeff(row, col);
    if (numext::abs(pivot) < smin) pivot = T(smin);
    const T multiplier = m.coeff(otherRow, col) / pivot;
    T otherPivot = m.coeff(otherRow, otherCol) - multiplier * m.coeff(row, otherCol);
    if (numext::abs(otherPivot) < smin) otherPivot = T(smin);
    Matrix<T, 2, 1> x;
    x.coeffRef(otherCol) = (r.coeff(otherRow) - multiplier * r.coeff(row)) / otherPivot;
    x.coeffRef(col) = (r.coeff(row) - m.coeff(row, otherCol) * x.coeff(otherCol)) / pivot;
    return x;
  }
};

template <typename MatrixType>
template <typename InputTypeA, typename InputTypeB>
GeneralizedEigenSolver<MatrixType>& GeneralizedEigenSolver<MatrixType>::compute(const EigenBase<InputTypeA>& A,
                                                                                const EigenBase<InputTypeB>& B,
                                                                                bool computeEigenvectors) {
  EIGEN_INTERNAL_ASSERT_SQUARE(A);
  EIGEN_INTERNAL_ASSERT_SAME_SHAPE(A, B);
  // Reduce to generalized real Schur form:
  // A = Q S Z and B = Q T Z
  m_realQZ.compute(A.derived(), B.derived(), computeEigenvectors);
  return computeFromQZ(computeEigenvectors);
}

/** \internal Computes the generalized eigenvalues, and the eigenvectors when requested, from the QZ decomposition
 * held by m_realQZ. */
template <typename MatrixType>
GeneralizedEigenSolver<MatrixType>& GeneralizedEigenSolver<MatrixType>::computeFromQZ(bool computeEigenvectors) {
  const Index size = m_realQZ.matrixS().cols();
  if (m_realQZ.info() == Success) {
    // Resize storage
    m_alphas.resize(size);
    m_betas.resize(size);
    if (computeEigenvectors) {
      m_eivec.resize(size, size);
      m_tmp.resize(size);
    }

    // Aliases:
    Map<VectorType> v(reinterpret_cast<Scalar*>(m_tmp.data()), size);
    ComplexVectorType& cv = m_tmp;
    const MatrixType& mS = m_realQZ.matrixS();
    const MatrixType& mT = m_realQZ.matrixT();
    // The back substitutions follow LAPACK's xTGEVC, with
    //   (alpha, beta) -> (alpha / (t max|S|) * r / max|T|,  beta / (t max|T|) * r / max|S|),
    //   t = max(|beta| / max|T|, |alpha| / max|S|, min),
    //   r = max(1, min(max(max|S|, max|T|) min / eps, min(max|S|, max|T|) highest / 2)),
    // so that |beta| |S(j, k)| <= r and |alpha| |T(j, k)| <= r. With r = 1, 1 / max|S| is subnormal for max|S| > 1 /
    // min and flushes to 0 under FTZ. r / max|S| and r / max|T| are at most highest / 2 and, unless max|S| / max|T| or
    // max|T| / max|S| exceeds highest eps / (2 min), at least min / eps, so the scaled beta underflows only where
    // |beta| / (t max|T|) < eps, and likewise alpha. A pivot beta S(j, j) - alpha T(j, j) of modulus below
    //   smin_j = max(eps sum_k (|beta| |S(j, k)| + |alpha| |T(j, k)|), min),  k over the columns of the solve,
    // the larger of its two rows' for a 2x2 block, is replaced by smin_j: it is 0 where eigenvalue j repeats eigenvalue
    // i, or for beta = 0 = T(j, j). Unlike xTGEVC's floor eps ||.||, the row sums leave accurate small 1x1 pivots of a
    // graded pencil alone, down to a scaled pivot of min. Row j's numerator is at most max |v| smin_j / eps, so a step
    // multiplies max |v| by at most 3 / eps (solveWithPerturbedPivots for a 2x2 block), and v is divided by max |v|
    // once that exceeds 1 / eps, which keeps it finite.
    const RealScalar eps = NumTraits<RealScalar>::epsilon();
    const RealScalar big = RealScalar(1) / eps;
    const RealScalar smallestNormal = RealScalar((std::numeric_limits<RealScalar>::min)());
    const RealScalar maxS = computeEigenvectors && size > 0
                                ? numext::maxi(RealScalar(mS.cwiseAbs().maxCoeff()), smallestNormal)
                                : smallestNormal;
    const RealScalar maxT = computeEigenvectors && size > 0
                                ? numext::maxi(RealScalar(mT.cwiseAbs().maxCoeff()), smallestNormal)
                                : smallestNormal;
    // Compares against wanted / halfHighest rather than forming min(max|S|, max|T|) highest / 2, which overflows.
    const RealScalar wanted = numext::maxi(maxS, maxT) * (smallestNormal / eps);
    const RealScalar halfHighest = NumTraits<RealScalar>::highest() / RealScalar(2);
    const RealScalar smaller = numext::mini(maxS, maxT);
    const RealScalar bound =
        numext::maxi(RealScalar(1), smaller < wanted / halfHighest ? RealScalar(smaller * halfHighest) : wanted);
    const RealScalar scaleS = bound / maxS, scaleT = bound / maxT;
    const auto scaleDivisor = [&](const RealScalar& absAlpha, const RealScalar& absBeta) {
      return numext::maxi(numext::maxi(RealScalar(absBeta / maxT), RealScalar(absAlpha / maxS)), smallestNormal);
    };
    const auto pivotFloor = [&](Index row, Index first, Index last, const RealScalar& absAlpha,
                                const RealScalar& absBeta) {
      const Index count = last - first + 1;
      return numext::maxi(RealScalar(eps * (absBeta * mS.row(row).segment(first, count).cwiseAbs() +
                                            absAlpha * mT.row(row).segment(first, count).cwiseAbs())
                                               .sum()),
                          smallestNormal);
    };

    Index i = 0;
    while (i < size) {
      if (i == size - 1 || mS.coeff(i + 1, i) == Scalar(0)) {
        // Real eigenvalue
        m_alphas.coeffRef(i) = mS.diagonal().coeff(i);
        m_betas.coeffRef(i) = mT.diagonal().coeff(i);
        if (computeEigenvectors) {
          v.setConstant(Scalar(0.0));
          v.coeffRef(i) = Scalar(1.0);
          const Scalar alpha = numext::real(m_alphas.coeffRef(i)), beta = m_betas.coeffRef(i);
          const RealScalar divisor = scaleDivisor(numext::abs(alpha), numext::abs(beta));
          const Scalar scaledAlpha = alpha / divisor / maxS * scaleT, scaledBeta = beta / divisor / maxT * scaleS;
          const RealScalar absAlpha = numext::abs(scaledAlpha), absBeta = numext::abs(scaledBeta);
          for (Index j = i - 1; j >= 0; j--) {
            const Index st = j + 1;
            const Index sz = i - j;
            RealScalar largest;
            if (j > 0 && mS.coeff(j, j - 1) != Scalar(0)) {
              // 2x2 block
              Matrix<Scalar, 2, 1> rhs = (scaledAlpha * mT.template block<2, Dynamic>(j - 1, st, 2, sz) -
                                          scaledBeta * mS.template block<2, Dynamic>(j - 1, st, 2, sz))
                                             .lazyProduct(v.segment(st, sz));
              Matrix<Scalar, 2, 2> lhs = scaledBeta * mS.template block<2, 2>(j - 1, j - 1) -
                                         scaledAlpha * mT.template block<2, 2>(j - 1, j - 1);
              const RealScalar smin = numext::maxi(pivotFloor(j - 1, j - 1, i, absAlpha, absBeta),
                                                   pivotFloor(j, j - 1, i, absAlpha, absBeta));
              v.template segment<2>(j - 1) = solveWithPerturbedPivots(lhs, rhs, smin);
              largest = v.template segment<2>(j - 1).cwiseAbs().maxCoeff();
              j--;
            } else {
              Scalar pivot = scaledBeta * mS.coeff(j, j) - scaledAlpha * mT.coeff(j, j);
              const RealScalar smin = pivotFloor(j, j, i, absAlpha, absBeta);
              if (numext::abs(pivot) < smin) pivot = Scalar(smin);
              v.coeffRef(j) =
                  -v.segment(st, sz)
                       .transpose()
                       .cwiseProduct(scaledBeta * mS.block(j, st, 1, sz) - scaledAlpha * mT.block(j, st, 1, sz))
                       .sum() /
                  pivot;
              largest = numext::abs(v.coeff(j));
            }
            if (largest > big) v /= largest;
          }
          m_eivec.col(i).real().noalias() = m_realQZ.matrixZ().transpose() * v;
          m_eivec.col(i).real().normalize();
          m_eivec.col(i).imag().setConstant(0);
        }
        ++i;
      } else {
        // The 2x2 pencil (S_i, T_i), T_i = diag(a, b), has the eigenvalues eig(S_i diag(b, a)) / (a b). With
        // T_i = s diag(a', b'), s >= max(|a|, |b|) (see generalized_eigen_split_pair_diagonal()),
        //   alpha = eig(M), M = S_i diag(b', a'),  beta = (s a') b' = a b',
        // so |a'|, |b'| <= 1, |M| <= |S_i|, |alpha| ~ |S_i| and |beta| ~ min(|a|, |b|), as alpha = S(i, i) and
        // beta = T(i, i) for a real eigenvalue; a b over- or underflows with |T_i|^2. The discriminant p^2 + m01 m10
        // is formed relative to its largest term, as in LAPACK's xLANV2. For a complex pair p^2 and m01 m10 have
        // opposite signs, so p, z and m11 + p are at most |S_i|, and alpha is finite whenever S_i is.
        // NOTE: using diagonal() instead of coeff(i,i) works around an MSVC bug.
        const RealScalar a = mT.diagonal().coeff(i), b = mT.diagonal().coeff(i + 1);
        RealScalar aScaled, bScaled;
        const RealScalar beta = m_betas.coeffRef(i) = m_betas.coeffRef(i + 1) =
            internal::generalized_eigen_split_pair_diagonal(
                a, b, aScaled, bScaled,
                internal::bool_constant<internal::use_subnormal_preserving_scaling<RealScalar, RealScalar>::value>());

        const Matrix<RealScalar, 2, 2> M =
            mS.template block<2, 2>(i, i) * Matrix<RealScalar, 2, 1>(bScaled, aScaled).asDiagonal();
        const RealScalar p = RealScalar(0.5) * M.coeff(0, 0) - RealScalar(0.5) * M.coeff(1, 1);
        const RealScalar m01 = M.coeff(0, 1), m10 = M.coeff(1, 0);
        const RealScalar bcMax = numext::maxi(numext::abs(m01), numext::abs(m10));
        const RealScalar bcMin = numext::mini(numext::abs(m01), numext::abs(m10));
        const RealScalar scale = numext::maxi(numext::abs(p), bcMax);
        RealScalar z(0);
        if (!numext::is_exactly_zero(scale)) {
          const RealScalar bcMinSigned = ((m01 < RealScalar(0)) == (m10 < RealScalar(0))) ? bcMin : -bcMin;
          z = numext::sqrt(scale) * numext::sqrt(numext::abs((p / scale) * p + (bcMax / scale) * bcMinSigned));
        }
        const ComplexScalar alpha = ComplexScalar(M.coeff(1, 1) + p, (beta > 0) ? z : -z);
        m_alphas.coeffRef(i) = numext::conj(alpha);
        m_alphas.coeffRef(i + 1) = alpha;

        if (computeEigenvectors) {
          // Compute eigenvector in position (i+1) and then position (i) is just the conjugate
          cv.setZero();
          cv.coeffRef(i + 1) = Scalar(1.0);
          const RealScalar divisor = scaleDivisor(numext::abs(alpha), numext::abs(beta));
          const ComplexScalar scaledAlpha = alpha / divisor / maxS * scaleT;
          const RealScalar scaledBeta = beta / divisor / maxT * scaleS;
          const RealScalar absAlpha = numext::abs(scaledAlpha), absBeta = numext::abs(scaledBeta);
          // here, the "static_cast" works around expression template issues.
          cv.coeffRef(i) =
              -(static_cast<Scalar>(scaledBeta * mS.coeffRef(i, i + 1)) - scaledAlpha * mT.coeffRef(i, i + 1)) /
              (static_cast<Scalar>(scaledBeta * mS.coeffRef(i, i)) - scaledAlpha * mT.coeffRef(i, i));
          if (numext::abs(cv.coeff(i)) > big) cv /= numext::abs(cv.coeff(i));
          for (Index j = i - 1; j >= 0; j--) {
            const Index st = j + 1;
            const Index sz = i + 1 - j;
            RealScalar largest;
            if (j > 0 && mS.coeff(j, j - 1) != Scalar(0)) {
              // 2x2 block
              Matrix<ComplexScalar, 2, 1> rhs = (scaledAlpha * mT.template block<2, Dynamic>(j - 1, st, 2, sz) -
                                                 scaledBeta * mS.template block<2, Dynamic>(j - 1, st, 2, sz))
                                                    .lazyProduct(cv.segment(st, sz));
              Matrix<ComplexScalar, 2, 2> lhs = scaledBeta * mS.template block<2, 2>(j - 1, j - 1) -
                                                scaledAlpha * mT.template block<2, 2>(j - 1, j - 1);
              const RealScalar smin = numext::maxi(pivotFloor(j - 1, j - 1, i + 1, absAlpha, absBeta),
                                                   pivotFloor(j, j - 1, i + 1, absAlpha, absBeta));
              cv.template segment<2>(j - 1) = solveWithPerturbedPivots(lhs, rhs, smin);
              largest = cv.template segment<2>(j - 1).cwiseAbs().maxCoeff();
              j--;
            } else {
              ComplexScalar pivot = scaledAlpha * mT.coeff(j, j) - static_cast<Scalar>(scaledBeta * mS.coeff(j, j));
              const RealScalar smin = pivotFloor(j, j, i + 1, absAlpha, absBeta);
              if (numext::abs(pivot) < smin) pivot = ComplexScalar(smin);
              cv.coeffRef(j) =
                  cv.segment(st, sz)
                      .transpose()
                      .cwiseProduct(scaledBeta * mS.block(j, st, 1, sz) - scaledAlpha * mT.block(j, st, 1, sz))
                      .sum() /
                  pivot;
              largest = numext::abs(cv.coeff(j));
            }
            if (largest > big) cv /= largest;
          }
          m_eivec.col(i + 1).noalias() = m_realQZ.matrixZ().transpose() * cv;
          m_eivec.col(i + 1).normalize();
          m_eivec.col(i) = m_eivec.col(i + 1).conjugate();
        }
        i += 2;
      }
    }
  }
  m_computeEigenvectors = computeEigenvectors;
  m_isInitialized = true;
  return *this;
}

}  // end namespace Eigen

#endif  // EIGEN_GENERALIZEDEIGENSOLVER_H
