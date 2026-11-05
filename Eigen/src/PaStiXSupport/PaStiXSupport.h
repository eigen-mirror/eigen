// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: The Eigen Authors

#ifndef EIGEN_PASTIXSUPPORT_H
#define EIGEN_PASTIXSUPPORT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \ingroup PaStiXSupport_Module
 * \brief Interface to the PaStix solver
 *
 * This class is used to solve the linear systems A.X = B via the PaStix library.
 * The matrix can be either real or complex, symmetric or not.
 *
 * \sa TutorialSparseDirectSolvers
 */
template <typename MatrixType_, bool IsStrSym = false>
class PastixLU;
template <typename MatrixType_, int Options>
class PastixLLT;
template <typename MatrixType_, int Options>
class PastixLDLT;

namespace internal {

template <class Pastix>
struct pastix_traits;

template <typename MatrixType_>
struct pastix_traits<PastixLU<MatrixType_> > {
  using MatrixType = MatrixType_;
  using Scalar = typename MatrixType_::Scalar;
  using RealScalar = typename MatrixType_::RealScalar;
  using StorageIndex = typename MatrixType_::StorageIndex;
};

template <typename MatrixType_, int Options>
struct pastix_traits<PastixLLT<MatrixType_, Options> > {
  using MatrixType = MatrixType_;
  using Scalar = typename MatrixType_::Scalar;
  using RealScalar = typename MatrixType_::RealScalar;
  using StorageIndex = typename MatrixType_::StorageIndex;
};

template <typename MatrixType_, int Options>
struct pastix_traits<PastixLDLT<MatrixType_, Options> > {
  using MatrixType = MatrixType_;
  using Scalar = typename MatrixType_::Scalar;
  using RealScalar = typename MatrixType_::RealScalar;
  using StorageIndex = typename MatrixType_::StorageIndex;
};

}  // namespace internal

// PaStiX 6 task interface:
// https://solverstack.gitlabpages.inria.fr/pastix/group__pastix__users.html
template <typename Derived>
class PastixBase : public SparseSolverBase<Derived> {
 protected:
  using Base = SparseSolverBase<Derived>;
  using Base::m_isInitialized;

 public:
  using Base::_solve_impl;
  using MatrixType = typename internal::pastix_traits<Derived>::MatrixType;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using StorageIndex = typename MatrixType::StorageIndex;
  using Vector = Matrix<Scalar, Dynamic, 1>;
  using ColSpMatrix = SparseMatrix<Scalar, ColMajor, pastix_int_t>;
  static constexpr int ColsAtCompileTime = MatrixType::ColsAtCompileTime;
  static constexpr int MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime;

  PastixBase() {
    EIGEN_STATIC_ASSERT(
        (std::is_same<Scalar, float>::value || std::is_same<Scalar, double>::value ||
         std::is_same<Scalar, std::complex<float>>::value || std::is_same<Scalar, std::complex<double>>::value),
        THIS_TYPE_IS_NOT_SUPPORTED)
    pastixInitParam(m_iparm.data(), m_dparm.data());
    m_iparm(IPARM_VERBOSE) = PastixVerboseNot;
  }

  ~PastixBase() { clean(); }

  /** Returns the integer PaStiX parameters. Set input parameters before analyzePattern(). */
  Array<pastix_int_t, IPARM_SIZE, 1> &iparm() { return m_iparm; }
  /** Returns the integer PaStiX parameter at \p parameter. */
  pastix_int_t &iparm(int parameter) { return m_iparm(parameter); }
  /** Returns the floating-point PaStiX parameters. */
  Array<double, DPARM_SIZE, 1> &dparm() { return m_dparm; }
  /** Returns the floating-point PaStiX parameter at \p parameter. */
  double &dparm(int parameter) { return m_dparm(parameter); }
  Index rows() const { return m_matrix.rows(); }
  Index cols() const { return m_matrix.cols(); }
  /** Returns Success if the last task succeeded, NumericalIssue for a task error or a non-finite solution,
   * and InvalidInput before computation. PaStiX may not detect singularity during factorization.
   * Some library errors terminate the process instead of returning an error code; in particular,
   * PastixLLT requires a positive definite matrix.
   */
  ComputationInfo info() const { return m_info; }

  template <typename Rhs, typename Dest>
  bool _solve_impl(const MatrixBase<Rhs> &b, MatrixBase<Dest> &x) const {
    eigen_assert(m_isInitialized && "The matrix should be factorized first");
    eigen_assert(b.rows() == rows());
    // PaStiX overwrites a contiguous, column-major RHS. Refine needs the original RHS as well.
    Matrix<Scalar, Dynamic, Dynamic, ColMajor> solution = b;
    Matrix<Scalar, Dynamic, Dynamic, ColMajor> rhs = b;
    const pastix_int_t n = internal::convert_index<pastix_int_t>(rows());
    const pastix_int_t nrhs = internal::convert_index<pastix_int_t>(b.cols());
    int error = 0;
    if (n != 0 && nrhs != 0) {
#if PASTIX_VERSION_MINOR >= 3
      error = pastix_task_solve(m_pastixdata, n, nrhs, solution.data(), n);
#else
      error = pastix_task_solve(m_pastixdata, nrhs, solution.data(), n);
#endif
      // PaStiX refinement accepts only one right-hand side, even when solve accepts several.
      for (pastix_int_t col = 0; error == PASTIX_SUCCESS && col < nrhs; ++col) {
        if (!solution.col(col).allFinite()) break;
        error = pastix_task_refine(m_pastixdata, n, 1, rhs.col(col).data(), n, solution.col(col).data(), n);
      }
    }
    m_info = error == PASTIX_SUCCESS && solution.allFinite() ? Success : NumericalIssue;
    x = solution;
    return m_info == Success;
  }

 protected:
  void clean() {
    if (m_pastixdata) pastixFinalize(&m_pastixdata);
    m_isInitialized = false;
    m_analysisIsOk = false;
  }

  void compute(ColSpMatrix &matrix) {
    analyzePattern(matrix);
    if (m_analysisIsOk) factorize(m_matrix);
  }

  void analyzePattern(ColSpMatrix &matrix) {
    eigen_assert(matrix.rows() == matrix.cols() && "The input matrix should be square");
    clean();
    m_matrix.swap(matrix);
    if (rows() == 0) {
      m_analysisIsOk = true;
      m_info = Success;
      return;
    }
    setMatrix();
    pastixInit(&m_pastixdata, MPI_COMM_WORLD, m_iparm.data(), m_dparm.data());
    const int error = pastix_task_analyze(m_pastixdata, &m_spm);
    m_analysisIsOk = error == PASTIX_SUCCESS;
    m_info = m_analysisIsOk ? Success : NumericalIssue;
  }

  void factorize(ColSpMatrix &matrix) {
    eigen_assert(m_analysisIsOk && "The analysis phase should be called before the factorization phase");
    eigen_assert(matrix.rows() == rows() && matrix.cols() == cols());
    if (&matrix != &m_matrix) m_matrix.swap(matrix);
    if (rows() == 0) {
      m_isInitialized = true;
      m_info = Success;
      return;
    }
    setMatrix();
    const int error = pastix_task_numfact(m_pastixdata, &m_spm);
    m_isInitialized = error == PASTIX_SUCCESS;
    m_info = m_isInitialized ? Success : NumericalIssue;
  }

  Array<pastix_int_t, IPARM_SIZE, 1> m_iparm;
  Array<double, DPARM_SIZE, 1> m_dparm;

 private:
  void setMatrix() {
    // Eigen owns the CSC arrays: do not pass this descriptor to spmExit().
    m_iparm(IPARM_FREE_CSCUSER) = 0;
    spmInit(&m_spm);
    m_spm.mtxtype = static_cast<spm_mtxtype_t>(m_iparm(IPARM_MTX_TYPE));
    m_spm.flttype = NumTraits<Scalar>::IsComplex
                        ? (std::is_same<RealScalar, float>::value ? SpmComplex32 : SpmComplex64)
                        : (std::is_same<Scalar, float>::value ? SpmFloat : SpmDouble);
    m_spm.fmttype = SpmCSC;
    m_spm.baseval = 0;
    m_spm.n = internal::convert_index<spm_int_t>(m_matrix.rows());
    m_spm.nnz = internal::convert_index<spm_int_t>(m_matrix.nonZeros());
    m_spm.dof = 1;
    m_spm.colptr = m_matrix.outerIndexPtr();
    m_spm.rowptr = m_matrix.innerIndexPtr();
    m_spm.values = m_matrix.valuePtr();
    spmUpdateComputedFields(&m_spm);
  }

  ColSpMatrix m_matrix;
  spmatrix_t m_spm;
  pastix_data_t *m_pastixdata = nullptr;
  bool m_analysisIsOk = false;
  mutable ComputationInfo m_info = InvalidInput;
};

/** \ingroup PaStiXSupport_Module
 * \class PastixLU
 * \brief Sparse direct LU solver based on PaStiX library
 *
 * This class is used to solve the linear systems A.X = B with a supernodal LU
 * factorization in the PaStiX library. The matrix A should be squared and nonsingular
 * PaStiX requires that the matrix A has a symmetric structural pattern.
 * This interface can symmetrize the input matrix otherwise.
 * The vectors or matrices X and B can be either dense or sparse.
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam IsStrSym Indicates if the input matrix has a symmetric pattern, default is false
 * NOTE : Note that if the analysis and factorization phase are called separately,
 * the input matrix will be symmetrized at each call, hence it is advised to
 * symmetrize the matrix in a end-user program and set \p IsStrSym to true
 *
 * \implsparsesolverconcept
 *
 * \sa \ref TutorialSparseSolverConcept, class SparseLU
 *
 */
template <typename MatrixType_, bool IsStrSym>
class PastixLU : public PastixBase<PastixLU<MatrixType_> > {
 public:
  using MatrixType = MatrixType_;
  using Base = PastixBase<PastixLU<MatrixType>>;
  using ColSpMatrix = typename Base::ColSpMatrix;
  using StorageIndex = typename MatrixType::StorageIndex;

 public:
  PastixLU() : Base() { init(); }

  explicit PastixLU(const MatrixType &matrix) : Base() {
    init();
    compute(matrix);
  }
  /** Compute the LU supernodal factorization of \p matrix.
   * iparm and dparm can be used to tune the PaStiX parameters.
   * see the PaStiX user's manual
   * \sa analyzePattern() factorize()
   */
  void compute(const MatrixType &matrix) {
    m_structureIsUptodate = false;
    ColSpMatrix temp;
    grabMatrix(matrix, temp);
    Base::compute(temp);
  }
  /** Compute the LU symbolic factorization of \p matrix using its sparsity pattern.
   * Several ordering methods can be used at this step. See the PaStiX user's manual.
   * The result of this operation can be used with successive matrices having the same pattern as \p matrix
   * \sa factorize()
   */
  void analyzePattern(const MatrixType &matrix) {
    m_structureIsUptodate = false;
    ColSpMatrix temp;
    grabMatrix(matrix, temp);
    Base::analyzePattern(temp);
  }

  /** Compute the LU supernodal factorization of \p matrix
   * WARNING The matrix \p matrix should have the same structural pattern
   * as the same used in the analysis phase.
   * \sa analyzePattern()
   */
  void factorize(const MatrixType &matrix) {
    ColSpMatrix temp;
    grabMatrix(matrix, temp);
    Base::factorize(temp);
  }

 protected:
  void init() {
    m_structureIsUptodate = false;
    m_iparm(IPARM_MTX_TYPE) = PastixGeneral;
    m_iparm(IPARM_FACTORIZATION) = PastixFactGETRF;
  }

  void grabMatrix(const MatrixType &matrix, ColSpMatrix &out) {
    EIGEN_IF_CONSTEXPR (IsStrSym)
      out = matrix;
    else {
      if (!m_structureIsUptodate) {
        // update the transposed structure
        m_transposedStructure = matrix.transpose();

        // Set the elements of the matrix to zero
        for (Index j = 0; j < m_transposedStructure.outerSize(); ++j)
          for (typename ColSpMatrix::InnerIterator it(m_transposedStructure, j); it; ++it) it.valueRef() = 0.0;

        m_structureIsUptodate = true;
      }

      const ColSpMatrix &converted = matrix;
      out = m_transposedStructure + converted;
    }
    out.makeCompressed();
  }

  using Base::m_dparm;
  using Base::m_iparm;

  ColSpMatrix m_transposedStructure;
  bool m_structureIsUptodate;
};

/** \ingroup PaStiXSupport_Module
 * \class PastixLLT
 * \brief A sparse direct supernodal Cholesky (LLT) factorization and solver based on the PaStiX library
 *
 * This class solves A.X = B via a supernodal LL^T (real) or LL^* (complex) Cholesky factorization.
 * The matrix A must be symmetric (real) or Hermitian (complex) and positive definite.
 * \warning PaStiX can terminate the process on non-positive-definite input rather than return an error.
 * The vectors or matrices X and B can be either dense or sparse
 *
 * \tparam MatrixType the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam UpLo The part of the matrix to use : Lower or Upper. This parameter has no default: either part is accepted
 *              and copied to the lower part required by PaStiX.
 *
 * \implsparsesolverconcept
 *
 * \sa \ref TutorialSparseSolverConcept, class SimplicialLLT
 */
template <typename MatrixType_, int UpLo_>
class PastixLLT : public PastixBase<PastixLLT<MatrixType_, UpLo_> > {
 public:
  using MatrixType = MatrixType_;
  using Base = PastixBase<PastixLLT<MatrixType, UpLo_>>;
  using ColSpMatrix = typename Base::ColSpMatrix;

 public:
  enum { UpLo = UpLo_ };
  PastixLLT() : Base() { init(); }

  explicit PastixLLT(const MatrixType &matrix) : Base() {
    init();
    compute(matrix);
  }

  /** Compute the L factor of the LL^T supernodal factorization of \p matrix
   * \sa analyzePattern() factorize()
   */
  void compute(const MatrixType &matrix) {
    ColSpMatrix temp;
    grabMatrix(matrix, temp);
    Base::compute(temp);
  }

  /** Compute the LL^T symbolic factorization of \p matrix using its sparsity pattern
   * The result of this operation can be used with successive matrices having the same pattern as \p matrix
   * \sa factorize()
   */
  void analyzePattern(const MatrixType &matrix) {
    ColSpMatrix temp;
    grabMatrix(matrix, temp);
    Base::analyzePattern(temp);
  }
  /** Compute the LL^T supernodal numerical factorization of \p matrix
   * \sa analyzePattern()
   */
  void factorize(const MatrixType &matrix) {
    ColSpMatrix temp;
    grabMatrix(matrix, temp);
    Base::factorize(temp);
  }

 protected:
  using Base::m_iparm;

  void init() {
    m_iparm(IPARM_MTX_TYPE) = NumTraits<typename Base::Scalar>::IsComplex ? PastixHermitian : PastixSymmetric;
    m_iparm(IPARM_FACTORIZATION) = PastixFactPOTRF;
  }

  void grabMatrix(const MatrixType &matrix, ColSpMatrix &out) {
    out.resize(matrix.rows(), matrix.cols());
    // Pastix supports only lower, column-major matrices
    const ColSpMatrix &converted = matrix;
    out.template selfadjointView<Lower>() = converted.template selfadjointView<UpLo>();
    out.makeCompressed();
  }
};

/** \ingroup PaStiXSupport_Module
 * \class PastixLDLT
 * \brief A sparse direct supernodal LDLT factorization and solver based on the PaStiX library
 *
 * This class solves A.X = B via a supernodal LDL^T (real) or LDL^* (complex) factorization.
 * The matrix A should be symmetric (real) or Hermitian (complex) and positive definite.
 * The vectors or matrices X and B can be either dense or sparse
 *
 * \tparam MatrixType the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam UpLo The part of the matrix to use : Lower or Upper. This parameter has no default: either part is accepted
 *              and copied to the lower part required by PaStiX.
 *
 * \implsparsesolverconcept
 *
 * \sa \ref TutorialSparseSolverConcept, class SimplicialLDLT
 */
template <typename MatrixType_, int UpLo_>
class PastixLDLT : public PastixBase<PastixLDLT<MatrixType_, UpLo_> > {
 public:
  using MatrixType = MatrixType_;
  using Base = PastixBase<PastixLDLT<MatrixType, UpLo_>>;
  using ColSpMatrix = typename Base::ColSpMatrix;

 public:
  enum { UpLo = UpLo_ };
  PastixLDLT() : Base() { init(); }

  explicit PastixLDLT(const MatrixType &matrix) : Base() {
    init();
    compute(matrix);
  }

  /** Compute the L and D factors of the LDL^T factorization of \p matrix
   * \sa analyzePattern() factorize()
   */
  void compute(const MatrixType &matrix) {
    ColSpMatrix temp;
    grabMatrix(matrix, temp);
    Base::compute(temp);
  }

  /** Compute the LDL^T symbolic factorization of \p matrix using its sparsity pattern
   * The result of this operation can be used with successive matrices having the same pattern as \p matrix
   * \sa factorize()
   */
  void analyzePattern(const MatrixType &matrix) {
    ColSpMatrix temp;
    grabMatrix(matrix, temp);
    Base::analyzePattern(temp);
  }
  /** Compute the LDL^T supernodal numerical factorization of \p matrix
   *
   */
  void factorize(const MatrixType &matrix) {
    ColSpMatrix temp;
    grabMatrix(matrix, temp);
    Base::factorize(temp);
  }

 protected:
  using Base::m_iparm;

  void init() {
    m_iparm(IPARM_MTX_TYPE) = NumTraits<typename Base::Scalar>::IsComplex ? PastixHermitian : PastixSymmetric;
    m_iparm(IPARM_FACTORIZATION) = NumTraits<typename Base::Scalar>::IsComplex ? PastixFactHETRF : PastixFactSYTRF;
  }

  void grabMatrix(const MatrixType &matrix, ColSpMatrix &out) {
    // Pastix supports only lower, column-major matrices
    out.resize(matrix.rows(), matrix.cols());
    const ColSpMatrix &converted = matrix;
    out.template selfadjointView<Lower>() = converted.template selfadjointView<UpLo>();
    out.makeCompressed();
  }
};

}  // end namespace Eigen

#endif
