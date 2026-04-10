// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// GPU sparse LU factorization via cuDSS.
//
// For general (non-symmetric) sparse matrices. Uses pivoting.
// Same three-phase workflow as GpuSparseLLT.
//
// Usage:
//   GpuSparseLU<double> lu(A);          // analyze + factorize
//   VectorXd x = lu.solve(b);           // solve

#ifndef EIGEN_GPU_SPARSE_LU_H
#define EIGEN_GPU_SPARSE_LU_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./GpuSparseSolverBase.h"

namespace Eigen {

/** GPU sparse LU factorization (general matrices).
 *
 * Wraps cuDSS with CUDSS_MTYPE_GENERAL and CUDSS_MVIEW_FULL.
 * Accepts ColMajor SparseMatrix (CSC); internally converts to RowMajor
 * CSR since cuDSS requires CSR input.
 *
 * \tparam Scalar_  float, double, complex<float>, or complex<double>
 */
template <typename Scalar_>
class GpuSparseLU : public internal::GpuSparseSolverBase<Scalar_, GpuSparseLU<Scalar_>> {
  using Base = internal::GpuSparseSolverBase<Scalar_, GpuSparseLU>;
  friend Base;

 public:
  using Scalar = Scalar_;

  GpuSparseLU() = default;

  template <typename InputType>
  explicit GpuSparseLU(const SparseMatrixBase<InputType>& A) {
    this->compute(A);
  }

  static constexpr bool needs_csr_conversion() { return true; }
  static constexpr cudssMatrixType_t cudss_matrix_type() { return CUDSS_MTYPE_GENERAL; }
  static constexpr cudssMatrixViewType_t cudss_matrix_view() { return CUDSS_MVIEW_FULL; }
};

}  // namespace Eigen

#endif  // EIGEN_GPU_SPARSE_LU_H
