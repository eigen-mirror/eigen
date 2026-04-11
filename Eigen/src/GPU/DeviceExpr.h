// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Lightweight expression types for DeviceMatrix operations.
//
// These are NOT Eigen expression templates. Each type maps 1:1 to a single
// NVIDIA library call (cuBLAS or cuSOLVER). There is no coefficient-level
// evaluation, no lazy fusion, no packet operations.
//
// Expression types:
//   DeviceAdjointView<S>  — d_A.adjoint()  → marks ConjTrans for GEMM
//   DeviceTransposeView<S> — d_A.transpose() → marks Trans for GEMM
//   DeviceScaled<Expr>    — alpha * expr    → carries scalar factor
//   GemmExpr<Lhs, Rhs>   — lhs * rhs       → dispatches to cublasXgemm

#ifndef EIGEN_GPU_DEVICE_EXPR_H
#define EIGEN_GPU_DEVICE_EXPR_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "./CuBlasSupport.h"

namespace Eigen {

// Forward declaration.
template <typename Scalar_>
class DeviceMatrix;

namespace internal {

// ---- Traits: extract operation info from expression types -------------------

// Default: a DeviceMatrix is NoTrans.
template <typename T>
struct device_expr_traits {
  static constexpr bool is_device_expr = false;
};

template <typename Scalar>
struct device_expr_traits<DeviceMatrix<Scalar>> {
  using scalar_type = Scalar;
  static constexpr GpuOp op = GpuOp::NoTrans;
  static constexpr bool is_device_expr = true;
  static const DeviceMatrix<Scalar>& matrix(const DeviceMatrix<Scalar>& x) { return x; }
  static Scalar alpha(const DeviceMatrix<Scalar>&) { return Scalar(1); }
};

}  // namespace internal

// ---- DeviceAdjointView: marks ConjTrans ------------------------------------
// Returned by DeviceMatrix::adjoint(). Maps to cublasXgemm transA/B = C.

template <typename Scalar_>
class DeviceAdjointView {
 public:
  using Scalar = Scalar_;
  explicit DeviceAdjointView(const DeviceMatrix<Scalar>& m) : mat_(m) {}
  const DeviceMatrix<Scalar>& matrix() const { return mat_; }

 private:
  const DeviceMatrix<Scalar>& mat_;
};

namespace internal {
template <typename Scalar>
struct device_expr_traits<DeviceAdjointView<Scalar>> {
  using scalar_type = Scalar;
  static constexpr GpuOp op = GpuOp::ConjTrans;
  static constexpr bool is_device_expr = true;
  static const DeviceMatrix<Scalar>& matrix(const DeviceAdjointView<Scalar>& x) { return x.matrix(); }
  static Scalar alpha(const DeviceAdjointView<Scalar>&) { return Scalar(1); }
};
}  // namespace internal

// ---- DeviceTransposeView: marks Trans --------------------------------------
// Returned by DeviceMatrix::transpose(). Maps to cublasXgemm transA/B = T.

template <typename Scalar_>
class DeviceTransposeView {
 public:
  using Scalar = Scalar_;
  explicit DeviceTransposeView(const DeviceMatrix<Scalar>& m) : mat_(m) {}
  const DeviceMatrix<Scalar>& matrix() const { return mat_; }

 private:
  const DeviceMatrix<Scalar>& mat_;
};

namespace internal {
template <typename Scalar>
struct device_expr_traits<DeviceTransposeView<Scalar>> {
  using scalar_type = Scalar;
  static constexpr GpuOp op = GpuOp::Trans;
  static constexpr bool is_device_expr = true;
  static const DeviceMatrix<Scalar>& matrix(const DeviceTransposeView<Scalar>& x) { return x.matrix(); }
  static Scalar alpha(const DeviceTransposeView<Scalar>&) { return Scalar(1); }
};
}  // namespace internal

// ---- DeviceScaled: alpha * expr --------------------------------------------
// Returned by operator*(Scalar, DeviceMatrix/View). Carries the scalar factor.

template <typename Inner>
class DeviceScaled {
 public:
  using Scalar = typename internal::device_expr_traits<Inner>::scalar_type;
  DeviceScaled(Scalar alpha, const Inner& inner) : alpha_(alpha), inner_(inner) {}
  Scalar scalar() const { return alpha_; }
  const Inner& inner() const { return inner_; }

 private:
  Scalar alpha_;
  const Inner& inner_;
};

namespace internal {
template <typename Inner>
struct device_expr_traits<DeviceScaled<Inner>> {
  using scalar_type = typename device_expr_traits<Inner>::scalar_type;
  static constexpr GpuOp op = device_expr_traits<Inner>::op;
  static constexpr bool is_device_expr = true;
  static const DeviceMatrix<scalar_type>& matrix(const DeviceScaled<Inner>& x) {
    return device_expr_traits<Inner>::matrix(x.inner());
  }
  static scalar_type alpha(const DeviceScaled<Inner>& x) {
    return x.scalar() * device_expr_traits<Inner>::alpha(x.inner());
  }
};
}  // namespace internal

// ---- GemmExpr: lhs * rhs → cublasXgemm ------------------------------------
// Returned by operator*(lhs_expr, rhs_expr). Dispatches to cuBLAS GEMM.

template <typename Lhs, typename Rhs>
class GemmExpr {
 public:
  using Scalar = typename internal::device_expr_traits<Lhs>::scalar_type;
  static_assert(std::is_same<Scalar, typename internal::device_expr_traits<Rhs>::scalar_type>::value,
                "DeviceMatrix GEMM: LHS and RHS must have the same scalar type");

  GemmExpr(const Lhs& lhs, const Rhs& rhs) : lhs_(lhs), rhs_(rhs) {}
  const Lhs& lhs() const { return lhs_; }
  const Rhs& rhs() const { return rhs_; }

 private:
  // Stored by reference — like Eigen's CPU expression templates, these must
  // not be captured with auto (the references will dangle). Use .eval() or
  // assign to a DeviceMatrix immediately.
  const Lhs& lhs_;
  const Rhs& rhs_;
};

// ---- Free operator* overloads that produce GemmExpr ------------------------
// These cover: DM*DM, Adj*DM, DM*Adj, Trans*DM, DM*Trans, Scaled*DM, etc.

// DeviceMatrix * DeviceMatrix
template <typename S>
GemmExpr<DeviceMatrix<S>, DeviceMatrix<S>> operator*(const DeviceMatrix<S>& a, const DeviceMatrix<S>& b) {
  return {a, b};
}

// AdjointView * DeviceMatrix
template <typename S>
GemmExpr<DeviceAdjointView<S>, DeviceMatrix<S>> operator*(const DeviceAdjointView<S>& a, const DeviceMatrix<S>& b) {
  return {a, b};
}

// DeviceMatrix * AdjointView
template <typename S>
GemmExpr<DeviceMatrix<S>, DeviceAdjointView<S>> operator*(const DeviceMatrix<S>& a, const DeviceAdjointView<S>& b) {
  return {a, b};
}

// TransposeView * DeviceMatrix
template <typename S>
GemmExpr<DeviceTransposeView<S>, DeviceMatrix<S>> operator*(const DeviceTransposeView<S>& a, const DeviceMatrix<S>& b) {
  return {a, b};
}

// DeviceMatrix * TransposeView
template <typename S>
GemmExpr<DeviceMatrix<S>, DeviceTransposeView<S>> operator*(const DeviceMatrix<S>& a, const DeviceTransposeView<S>& b) {
  return {a, b};
}

// Scaled * DeviceMatrix
template <typename Inner, typename S>
GemmExpr<DeviceScaled<Inner>, DeviceMatrix<S>> operator*(const DeviceScaled<Inner>& a, const DeviceMatrix<S>& b) {
  return {a, b};
}

// DeviceMatrix * Scaled
template <typename S, typename Inner>
GemmExpr<DeviceMatrix<S>, DeviceScaled<Inner>> operator*(const DeviceMatrix<S>& a, const DeviceScaled<Inner>& b) {
  return {a, b};
}

// ---- Scalar * DeviceMatrix / View → DeviceScaled ---------------------------

template <typename S>
DeviceScaled<DeviceMatrix<S>> operator*(S alpha, const DeviceMatrix<S>& m) {
  return {alpha, m};
}

template <typename S>
DeviceScaled<DeviceAdjointView<S>> operator*(S alpha, const DeviceAdjointView<S>& m) {
  return {alpha, m};
}

template <typename S>
DeviceScaled<DeviceTransposeView<S>> operator*(S alpha, const DeviceTransposeView<S>& m) {
  return {alpha, m};
}

}  // namespace Eigen

#endif  // EIGEN_GPU_DEVICE_EXPR_H
