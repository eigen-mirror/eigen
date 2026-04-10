// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Solver expression types for DeviceMatrix.
//
// Each expression maps 1:1 to cuSOLVER library calls:
//   LltSolveExpr  → cusolverDnXpotrf + cusolverDnXpotrs
//   LuSolveExpr   → cusolverDnXgetrf + cusolverDnXgetrs
//
// Usage:
//   d_X = d_A.llt().solve(d_B);              // Cholesky solve
//   d_X.device(ctx) = d_A.lu().solve(d_B);   // LU solve on explicit stream

#ifndef EIGEN_GPU_DEVICE_SOLVER_EXPR_H
#define EIGEN_GPU_DEVICE_SOLVER_EXPR_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

// Forward declarations.
template <typename Scalar_>
class DeviceMatrix;
class GpuContext;

// ---- LLT solve expression ---------------------------------------------------
// d_A.llt().solve(d_B) → LltSolveExpr → cusolverDnXpotrf + cusolverDnXpotrs

template <typename Scalar_, int UpLo_ = Lower>
class LltSolveExpr {
 public:
  using Scalar = Scalar_;
  enum { UpLo = UpLo_ };

  LltSolveExpr(const DeviceMatrix<Scalar>& A, const DeviceMatrix<Scalar>& B) : A_(A), B_(B) {}
  const DeviceMatrix<Scalar>& matrix() const { return A_; }
  const DeviceMatrix<Scalar>& rhs() const { return B_; }

 private:
  const DeviceMatrix<Scalar>& A_;
  const DeviceMatrix<Scalar>& B_;
};

// ---- LU solve expression ----------------------------------------------------
// d_A.lu().solve(d_B) → LuSolveExpr → cusolverDnXgetrf + cusolverDnXgetrs

template <typename Scalar_>
class LuSolveExpr {
 public:
  using Scalar = Scalar_;

  LuSolveExpr(const DeviceMatrix<Scalar>& A, const DeviceMatrix<Scalar>& B) : A_(A), B_(B) {}
  const DeviceMatrix<Scalar>& matrix() const { return A_; }
  const DeviceMatrix<Scalar>& rhs() const { return B_; }

 private:
  const DeviceMatrix<Scalar>& A_;
  const DeviceMatrix<Scalar>& B_;
};

// ---- DeviceLLTView: d_A.llt() → view with .solve() and .device() -----------

template <typename Scalar_, int UpLo_ = Lower>
class DeviceLLTView {
 public:
  using Scalar = Scalar_;

  explicit DeviceLLTView(const DeviceMatrix<Scalar>& m) : mat_(m) {}

  /** Build a solve expression: d_A.llt().solve(d_B).
   * The expression is evaluated when assigned to a DeviceMatrix. */
  LltSolveExpr<Scalar, UpLo_> solve(const DeviceMatrix<Scalar>& rhs) const { return {mat_, rhs}; }

  // For cached factorizations, use the explicit GpuLLT API directly:
  //   GpuLLT<double> llt;
  //   llt.compute(d_A);
  //   auto d_X1 = llt.solve(d_B1);
  //   auto d_X2 = llt.solve(d_B2);

 private:
  const DeviceMatrix<Scalar>& mat_;
};

// ---- DeviceLUView: d_A.lu() → view with .solve() and .device() -------------

template <typename Scalar_>
class DeviceLUView {
 public:
  using Scalar = Scalar_;

  explicit DeviceLUView(const DeviceMatrix<Scalar>& m) : mat_(m) {}

  /** Build a solve expression: d_A.lu().solve(d_B). */
  LuSolveExpr<Scalar> solve(const DeviceMatrix<Scalar>& rhs) const { return {mat_, rhs}; }

  // For cached factorizations, use the explicit GpuLU API directly:
  //   GpuLU<double> lu;
  //   lu.compute(d_A);
  //   auto d_X1 = lu.solve(d_B1);
  //   auto d_X2 = lu.solve(d_B2);

 private:
  const DeviceMatrix<Scalar>& mat_;
};

}  // namespace Eigen

#endif  // EIGEN_GPU_DEVICE_SOLVER_EXPR_H
