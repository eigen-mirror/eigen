// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_DGMRES_H
#define EIGEN_DGMRES_H

#include "../../Eigenvalues"

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename MatrixType_, typename Preconditioner_ = DiagonalPreconditioner<typename MatrixType_::Scalar> >
class DGMRES;

namespace internal {

template <typename MatrixType_, typename Preconditioner_>
struct traits<DGMRES<MatrixType_, Preconditioner_> > {
  using MatrixType = MatrixType_;
  using Preconditioner = Preconditioner_;
};

/** \brief Computes a permutation vector to have a sorted sequence
 * \param vec The vector to reorder.
 * \param perm gives the sorted sequence on output. Must be initialized with 0..n-1
 * \param ncut Put  the ncut smallest elements at the end of the vector
 * WARNING This is an expensive sort, so should be used only
 * for small size vectors
 * TODO: Use modified QuickSplit or std::nth_element to get the smallest values
 */
template <typename VectorType, typename IndexType>
void sortWithPermutation(VectorType& vec, IndexType& perm, typename IndexType::Scalar& ncut) {
  eigen_assert(vec.size() == perm.size());
  for (Index k = 0; k < ncut; k++) {
    bool flag = false;
    for (Index j = 0; j < vec.size() - 1; j++) {
      if (vec(perm(j)) < vec(perm(j + 1))) {
        std::swap(perm(j), perm(j + 1));
        flag = true;
      }
    }
    if (!flag) break;  // The vector is in sorted order
  }
}

}  // namespace internal
/**
 * \ingroup IterativeLinearSolvers_Module
 * \brief A Restarted GMRES with deflation.
 * This class implements a modification of the GMRES solver for
 * sparse linear systems. The basis is built with modified
 * Gram-Schmidt. At each restart, a few approximated eigenvectors
 * corresponding to the smallest eigenvalues are used to build a
 * preconditioner for the next cycle. This preconditioner
 * for deflation can be combined with any other preconditioner,
 * the IncompleteLUT for instance. The preconditioner is applied
 * at right of the matrix and the combination is multiplicative.
 *
 * \tparam MatrixType_ the type of the sparse matrix A, can be a dense or a sparse matrix.
 * \tparam Preconditioner_ the type of the preconditioner. Default is DiagonalPreconditioner
 * Typical usage :
 * \code
 * SparseMatrix<double> A;
 * VectorXd x, b;
 * //Fill A and b ...
 * DGMRES<SparseMatrix<double> > solver;
 * solver.set_restart(30); // Set restarting value
 * solver.setEigenv(1); // Set the number of eigenvalues to deflate
 * solver.compute(A);
 * x = solver.solve(b);
 * \endcode
 *
 * DGMRES can also be used in a matrix-free context, see the following \link MatrixfreeSolverExample example \endlink.
 *
 * References :
 * [1] D. NUENTSA WAKAM and F. PACULL, Memory Efficient Hybrid
 *  Algebraic Solvers for Linear Systems Arising from Compressible
 *  Flows, Computers and Fluids, In Press,
 *  https://doi.org/10.1016/j.compfluid.2012.03.023
 * [2] K. Burrage and J. Erhel, On the performance of various
 * adaptive preconditioned GMRES strategies, 5(1998), 101-121.
 * [3] J. Erhel, K. Burrage and B. Pohl, Restarted GMRES
 *  preconditioned by deflation,J. Computational and Applied
 *  Mathematics, 69(1996), 303-318.

 *
 */
template <typename MatrixType_, typename Preconditioner_>
class DGMRES : public IterativeSolverBase<DGMRES<MatrixType_, Preconditioner_> > {
 protected:
  using Base = IterativeSolverBase<DGMRES>;
  using Base::m_error;
  using Base::m_info;
  using Base::m_isInitialized;
  using Base::m_iterations;
  using Base::m_tolerance;
  using Base::matrix;

 public:
  using Base::_solve_impl;
  using Base::_solve_with_guess_impl;
  using MatrixType = MatrixType_;
  using Scalar = typename MatrixType::Scalar;
  using StorageIndex = typename MatrixType::StorageIndex;
  using RealScalar = typename MatrixType::RealScalar;
  using ComplexScalar = internal::make_complex_t<Scalar>;
  using Preconditioner = Preconditioner_;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  using DenseRealMatrix = Matrix<RealScalar, Dynamic, Dynamic>;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;
  using DenseRealVector = Matrix<RealScalar, Dynamic, 1>;
  using ComplexVector = Matrix<ComplexScalar, Dynamic, 1>;

  /** Default constructor. */
  DGMRES() : Base(), m_restart(30), m_neig(0), m_r(0), m_maxNeig(5), m_isDeflInitialized(false) {}

  /** Initialize the solver with matrix \a A for further \c Ax=b solving.
   *
   * This constructor is a shortcut for the default constructor followed
   * by a call to compute().
   *
   * \warning this class stores a reference to the matrix A as well as some
   * precomputed values that depend on it. Therefore, if \a A is changed
   * this class becomes invalid. Call compute() to update it with the new
   * matrix A, or modify a copy of A.
   */
  template <typename MatrixDerived>
  explicit DGMRES(const EigenBase<MatrixDerived>& A)
      : Base(A.derived()), m_restart(30), m_neig(0), m_r(0), m_maxNeig(5), m_isDeflInitialized(false) {}

  /** \internal */
  template <typename Rhs, typename Dest>
  void _solve_vector_with_guess_impl(const Rhs& b, Dest& x) const {
    EIGEN_STATIC_ASSERT(Rhs::ColsAtCompileTime == 1 || Dest::ColsAtCompileTime == 1,
                        YOU_TRIED_CALLING_A_VECTOR_METHOD_ON_A_MATRIX);

    m_iterations = Base::maxIterations();
    m_error = Base::m_tolerance;

    dgmres(matrix(), b, x, Base::m_preconditioner);
  }

  /**
   * Get the restart value
   */
  Index restart() const { return m_restart; }

  /**
   * Set the restart value (default is 30)
   */
  void set_restart(const Index restart) { m_restart = restart; }

  /**
   * Set the number of eigenvalues to deflate at each restart
   */
  void setEigenv(const Index neig) {
    m_neig = neig;
    if (neig + 1 > m_maxNeig) m_maxNeig = neig + 1;  // To allow for complex conjugates
  }

  /**
   * Get the size of the deflation subspace size
   */
  Index deflSize() const { return m_r; }

  /**
   * Set the maximum size of the deflation subspace
   */
  void setMaxEigenv(const Index maxNeig) { m_maxNeig = maxNeig; }

 protected:
  // DGMRES algorithm
  template <typename Rhs, typename Dest>
  void dgmres(const MatrixType& mat, const Rhs& rhs, Dest& x, const Preconditioner& precond) const;
  // Perform one cycle of GMRES
  template <typename Dest>
  Index dgmresCycle(const MatrixType& mat, const Preconditioner& precond, Dest& x, DenseVector& r0, RealScalar& beta,
                    const RealScalar& normRhs, Index& nbIts) const;
  // Compute data to use for deflation
  Index dgmresComputeDeflationData(const MatrixType& mat, const Preconditioner& precond, const Index& it,
                                   StorageIndex& neig) const;
  // Apply deflation to a vector
  template <typename RhsType, typename DestType>
  Index dgmresApplyDeflation(const RhsType& In, DestType& Out) const;
  // Ritz vectors of the Hessenberg matrix; for a real conjugate pair, its real and imaginary parts in adjacent columns
  const DenseMatrix& ritzVectors(const ComplexEigenSolver<DenseMatrix>& eigH) const { return eigH.eigenvectors(); }
  const DenseMatrix& ritzVectors(const EigenSolver<DenseMatrix>& eigH) const { return eigH.pseudoEigenvectors(); }
  // Init data for deflation
  void dgmresInitDeflation(Index& rows) const;
  mutable DenseMatrix m_V;                  // Krylov basis vectors
  mutable DenseMatrix m_H;                  // Hessenberg matrix
  mutable DenseMatrix m_Hes;                // Initial hessenberg matrix without Givens rotations applied
  mutable Index m_restart;                  // Maximum size of the Krylov subspace
  mutable DenseMatrix m_U;                  // Vectors that form the basis of the invariant subspace
  mutable DenseMatrix m_MU;                 // A*M^{-1}*U, the preconditioned operator applied to m_U
  mutable DenseMatrix m_T;                  /* T=U^H*A*M^{-1}*U */
  mutable PartialPivLU<DenseMatrix> m_luT;  // LU factorization of m_T
  mutable StorageIndex m_neig;              // Number of eigenvalues to extract at each restart
  mutable Index m_r;                        // Current number of deflated eigenvalues, size of m_U
  mutable Index m_maxNeig;                  // Maximum number of eigenvalues to deflate
  mutable RealScalar m_lambdaN = 0;         // Modulus of the largest eigenvalue of A
  mutable bool m_isDeflInitialized;

  // Adaptive strategy
  mutable RealScalar m_smv;  // Smaller multiple of the remaining number of steps allowed
  mutable bool m_force;      // Force the use of deflation at each restart
};
/**
 * \brief Perform several cycles of restarted GMRES with modified Gram Schmidt,
 *
 * A right preconditioner is used combined with deflation.
 *
 */
template <typename MatrixType_, typename Preconditioner_>
template <typename Rhs, typename Dest>
void DGMRES<MatrixType_, Preconditioner_>::dgmres(const MatrixType& mat, const Rhs& rhs, Dest& x,
                                                  const Preconditioner& precond) const {
  const RealScalar considerAsZero = (std::numeric_limits<RealScalar>::min)();

  // The deflation subspace is rebuilt for every right-hand side.
  m_isDeflInitialized = false;
  m_r = 0;
  m_lambdaN = 0;

  RealScalar normRhs = rhs.norm();
  if (normRhs <= considerAsZero) {
    x.setZero();
    m_error = 0;
    m_iterations = 0;
    m_info = Success;
    return;
  }

  // Initialization
  Index n = mat.rows();
  DenseVector r0(n);
  Index nbIts = 0;
  m_H.resize(m_restart + 1, m_restart);
  m_Hes.setZero(m_restart, m_restart);
  m_V.resize(n, m_restart + 1);
  // Initial residual vector and initial norm
  if (x.squaredNorm() == 0) x = precond.solve(rhs);
  r0.noalias() = rhs - mat * x;
  RealScalar beta = r0.norm();

  m_error = beta / normRhs;
  if (m_error < m_tolerance)
    m_info = Success;
  else
    m_info = NoConvergence;

  // Iterative process
  while (nbIts < m_iterations && m_info == NoConvergence) {
    dgmresCycle(mat, precond, x, r0, beta, normRhs, nbIts);

    // Compute the new residual vector for the restart
    if (nbIts < m_iterations && m_info == NoConvergence) {
      r0.noalias() = rhs - mat * x;
      beta = r0.norm();
    }
  }
  // m_iterations carried the iteration cap for the loops above; report the number actually performed.
  m_iterations = nbIts;
}

/**
 * \brief Perform one restart cycle of DGMRES
 * \param mat The coefficient matrix
 * \param precond The preconditioner
 * \param x the new approximated solution
 * \param r0 The initial residual vector
 * \param beta The norm of the residual computed so far
 * \param normRhs The norm of the right hand side vector
 * \param nbIts The number of iterations
 */
template <typename MatrixType_, typename Preconditioner_>
template <typename Dest>
Index DGMRES<MatrixType_, Preconditioner_>::dgmresCycle(const MatrixType& mat, const Preconditioner& precond, Dest& x,
                                                        DenseVector& r0, RealScalar& beta, const RealScalar& normRhs,
                                                        Index& nbIts) const {
  // Initialization
  DenseVector g(m_restart + 1);  // Right hand side of the least square problem
  g.setZero();
  g(0) = Scalar(beta);
  m_V.col(0) = r0 / beta;
  m_info = NoConvergence;
  std::vector<JacobiRotation<Scalar> > gr(m_restart);  // Givens rotations
  Index it = 0;                                        // Number of inner iterations
  Index n = mat.rows();
  DenseVector tv1(n), tv2(n);  // Temporary vectors
  while (m_info == NoConvergence && it < m_restart && nbIts < m_iterations) {
    // Apply preconditioner(s) at right
    if (m_isDeflInitialized) {
      dgmresApplyDeflation(m_V.col(it), tv1);  // Deflation
      tv2 = precond.solve(tv1);
    } else {
      tv2 = precond.solve(m_V.col(it));  // User's selected preconditioner
    }
    tv1.noalias() = mat * tv2;

    // Orthogonalize it with the previous basis in the basis using modified Gram-Schmidt
    Scalar coef;
    for (Index i = 0; i <= it; ++i) {
      coef = m_V.col(i).dot(tv1);
      tv1 = tv1 - coef * m_V.col(i);
      m_H(i, it) = coef;
      m_Hes(i, it) = coef;
    }
    // Normalize the vector. coef == 0 is an Arnoldi happy breakdown: the new
    // direction lies in span(V[0..it]), so skip the division (which would
    // poison m_V.col(it+1) with NaN) and fall through to the termination
    // check below.
    coef = tv1.norm();
    const bool happy_breakdown = numext::is_exactly_zero(coef);
    if (!happy_breakdown) {
      m_V.col(it + 1) = tv1 / coef;
    }
    m_H(it + 1, it) = coef;
    if (it + 1 < m_restart) m_Hes(it + 1, it) = coef;

    // Update Hessenberg matrix with Givens rotations
    for (Index i = 1; i <= it; ++i) {
      m_H.col(it).applyOnTheLeft(i - 1, i, gr[i - 1].adjoint());
    }

    // If the rotated diagonal is also zero, the reduced triangular system
    // becomes singular and the back-substitution below would produce Inf/NaN.
    // Stop with NumericalIssue instead of polluting x.
    if (happy_breakdown && numext::is_exactly_zero(m_H(it, it))) {
      m_info = NumericalIssue;
      break;
    }

    // Compute the new plane rotation
    gr[it].makeGivens(m_H(it, it), m_H(it + 1, it));
    // Apply the new rotation
    m_H.col(it).applyOnTheLeft(it, it + 1, gr[it].adjoint());
    g.applyOnTheLeft(it, it + 1, gr[it].adjoint());

    beta = numext::abs(g(it + 1));
    m_error = beta / normRhs;
    it++;
    nbIts++;

    if (m_error < m_tolerance || happy_breakdown) {
      // Happy breakdown: residual on the current subspace is exactly zero, so
      // the it-dim triangular system yields the exact solution.
      m_info = Success;
      break;
    }
  }

  // Compute the new coefficients by solving the least square problem.
  DenseVector nrs = m_H.topLeftCorner(it, it).template triangularView<Upper>().solve(g.head(it));

  // Form the new solution
  if (m_isDeflInitialized) {
    tv1.noalias() = m_V.leftCols(it) * nrs;
    dgmresApplyDeflation(tv1, tv2);
    x = x + precond.solve(tv2);
  } else
    x = x + precond.solve(m_V.leftCols(it) * nrs);

  // Go for a new cycle and compute data for deflation. A conjugate pair adds m_neig + 1 columns to m_U, which has
  // m_maxNeig columns, at most n of them independent.
  if (nbIts < m_iterations && m_info == NoConvergence && m_neig > 0 && (m_r + m_neig) < (std::min)(m_maxNeig, n))
    dgmresComputeDeflationData(mat, precond, it, m_neig);
  return 0;
}

template <typename MatrixType_, typename Preconditioner_>
void DGMRES<MatrixType_, Preconditioner_>::dgmresInitDeflation(Index& rows) const {
  m_U.resize(rows, m_maxNeig);
  m_MU.resize(rows, m_maxNeig);
  m_T.resize(m_maxNeig, m_maxNeig);
}

template <typename MatrixType_, typename Preconditioner_>
Index DGMRES<MatrixType_, Preconditioner_>::dgmresComputeDeflationData(const MatrixType& mat,
                                                                       const Preconditioner& precond, const Index& it,
                                                                       StorageIndex& neig) const {
  // Ritz pairs of the Hessenberg matrix H; perm ends with the neig Ritz values of smallest modulus, smallest last
  std::conditional_t<NumTraits<Scalar>::IsComplex, ComplexEigenSolver<DenseMatrix>, EigenSolver<DenseMatrix> > eigH(
      m_Hes.topLeftCorner(it, it));
  if (eigH.info() != Success) return 0;
  const ComplexVector& eig = eigH.eigenvalues();
  DenseRealVector modulEig = eig.cwiseAbs();
  Matrix<StorageIndex, Dynamic, 1> perm(it);
  perm.setLinSpaced(it, 0, internal::convert_index<StorageIndex>(it - 1));
  internal::sortWithPermutation(modulEig, perm, neig);

  if (!m_lambdaN) {
    m_lambdaN = modulEig.maxCoeff();
  }
  // Basis of the invariant subspace of H for those Ritz values. A real conjugate pair is taken whole, so up to
  // neig + 1 columns are extracted; each position visited adds a column or holds the partner of a pair already taken,
  // so the loop reads only the neig sorted positions.
  const DenseMatrix& ritzVecs = ritzVectors(eigH);
  DenseMatrix Y(it, neig + 1);
  std::vector<bool> taken(it, false);
  Index nbrEig = 0;
  for (Index k = it - 1; k >= 0 && nbrEig < neig; --k) {
    Index first = perm(k), count = 1;
    if (taken[first]) continue;
    if (!NumTraits<Scalar>::IsComplex && numext::imag(eig(first)) != RealScalar(0)) {
      // The pair's eigenvalue with positive imaginary part comes first.
      if (numext::imag(eig(first)) < RealScalar(0)) --first;
      count = 2;
    }
    Y.middleCols(nbrEig, count) = ritzVecs.middleCols(first, count);
    for (Index j = first; j < first + count; ++j) taken[j] = true;
    nbrEig += count;
  }

  // Lift to the Krylov basis, orthogonalize against the current deflation vectors, and orthonormalize. QR leaves
  // ||U^H X|| ~ eps cond(X), so a second pass restores orthogonality when the Ritz vectors are nearly dependent.
  Index m = m_V.rows();
  DenseMatrix X = m_V.leftCols(it) * Y.leftCols(nbrEig);
  for (int pass = 0; pass < (m_r > 0 ? 2 : 1); ++pass) {
    if (m_r > 0) X -= m_U.leftCols(m_r) * (m_U.leftCols(m_r).adjoint() * X);
    X = HouseholderQR<DenseMatrix>(X).householderQ() * DenseMatrix::Identity(m, nbrEig);
  }

  // Compute MX = A * M^-1 * X: deflation acts on the right-preconditioned operator
  if (m_r == 0) dgmresInitDeflation(m);
  DenseMatrix MX(m, nbrEig);
  DenseVector tv1(m);
  for (Index j = 0; j < nbrEig; j++) {
    tv1 = precond.solve(X.col(j));
    MX.col(j).noalias() = mat * tv1;
  }

  // Update m_T = [U'MU U'MX; X'MU X'MX]
  m_T.block(m_r, m_r, nbrEig, nbrEig).noalias() = X.adjoint() * MX;
  if (m_r) {
    m_T.block(0, m_r, m_r, nbrEig).noalias() = m_U.leftCols(m_r).adjoint() * MX;
    m_T.block(m_r, 0, nbrEig, m_r).noalias() = X.adjoint() * m_MU.leftCols(m_r);
  }

  // Save X into m_U and m_MX in m_MU
  for (Index j = 0; j < nbrEig; j++) m_U.col(m_r + j) = X.col(j);
  for (Index j = 0; j < nbrEig; j++) m_MU.col(m_r + j) = MX.col(j);
  // Increase the size of the invariant subspace
  m_r += nbrEig;

  // Factorize m_T into m_luT
  m_luT.compute(m_T.topLeftCorner(m_r, m_r));

  // FIXME: Check if the factorization was correctly done (nonsingular matrix).
  m_isDeflInitialized = true;
  return 0;
}
template <typename MatrixType_, typename Preconditioner_>
template <typename RhsType, typename DestType>
Index DGMRES<MatrixType_, Preconditioner_>::dgmresApplyDeflation(const RhsType& x, DestType& y) const {
  DenseVector x1 = m_U.leftCols(m_r).adjoint() * x;
  y = x + m_U.leftCols(m_r) * (m_lambdaN * m_luT.solve(x1) - x1);
  return 0;
}

}  // end namespace Eigen
#endif
