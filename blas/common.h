// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_BLAS_COMMON_H
#define EIGEN_BLAS_COMMON_H

#include "../Eigen/Core"
#include "../Eigen/Jacobi"

#include <complex>

#ifndef SCALAR
#error the token SCALAR must be defined to compile this file
#endif

#include "blas.h"

constexpr size_t kBlasNameLength = 6;

namespace Eigen {
namespace internal {
// Optionally wraps an expression in a conjugation, used by the product kernels
// below. This alias used to live in Eigen's core products headers but was
// removed there as dead code; the BLAS kernels are its only remaining users.
template <bool Cond, typename T>
using conj_expr_if =
    std::conditional_t<!Cond, const T&, CwiseUnaryOp<scalar_conjugate_op<typename traits<T>::Scalar>, T>>;
}  // namespace internal
}  // namespace Eigen

#include "BandTriangularSolver.h"
#include "GeneralRank1Update.h"
#include "PackedSelfadjointProduct.h"
#include "PackedTriangularMatrixVector.h"
#include "PackedTriangularSolverVector.h"
#include "Rank2Update.h"

#define NOTR 0
#define TR 1
#define ADJ 2

#define LEFT 0
#define RIGHT 1

#define UP 0
#define LO 1

#define NUNIT 0
#define UNIT 1

#define INVALID 0xff

#define OP(X) \
  (((X) == 'N' || (X) == 'n') ? NOTR : ((X) == 'T' || (X) == 't') ? TR : ((X) == 'C' || (X) == 'c') ? ADJ : INVALID)

#define SIDE(X) (((X) == 'L' || (X) == 'l') ? LEFT : ((X) == 'R' || (X) == 'r') ? RIGHT : INVALID)

#define UPLO(X) (((X) == 'U' || (X) == 'u') ? UP : ((X) == 'L' || (X) == 'l') ? LO : INVALID)

#define DIAG(X) (((X) == 'N' || (X) == 'n') ? NUNIT : ((X) == 'U' || (X) == 'u') ? UNIT : INVALID)

using Scalar = SCALAR;
using RealScalar = Eigen::NumTraits<Scalar>::Real;
using Complex = std::complex<RealScalar>;

enum { IsComplex = Eigen::NumTraits<SCALAR>::IsComplex, Conj = IsComplex };

using PlainMatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using MatrixType =
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<>>;
using ConstMatrixType =
    Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<>>;
using StridedVectorType = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>, 0, Eigen::InnerStride<Eigen::Dynamic>>;
using CompactVectorType = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>;

template <typename T>
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<>> matrix(
    T* data, Eigen::Index rows, Eigen::Index cols, Eigen::Index stride) {
  return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<>>(
      data, rows, cols, Eigen::OuterStride<>(stride));
}

template <typename T>
Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<>> matrix(
    const T* data, Eigen::Index rows, Eigen::Index cols, Eigen::Index stride) {
  return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0, Eigen::OuterStride<>>(
      data, rows, cols, Eigen::OuterStride<>(stride));
}

template <typename T>
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>, 0, Eigen::InnerStride<Eigen::Dynamic>> make_vector(T* data,
                                                                                                   Eigen::Index size,
                                                                                                   Eigen::Index incr) {
  return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>, 0, Eigen::InnerStride<Eigen::Dynamic>>(
      data, size, Eigen::InnerStride<Eigen::Dynamic>(incr));
}

template <typename T>
Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, 0, Eigen::InnerStride<Eigen::Dynamic>> make_vector(
    const T* data, Eigen::Index size, Eigen::Index incr) {
  return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, 0, Eigen::InnerStride<Eigen::Dynamic>>(
      data, size, Eigen::InnerStride<Eigen::Dynamic>(incr));
}

template <typename T>
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> make_vector(T* data, Eigen::Index size) {
  return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(data, size);
}

template <typename T>
Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> make_vector(const T* data, Eigen::Index size) {
  return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(data, size);
}

// Calls f(vx, vy) on the BLAS vectors (x, n, incx) and (y, n, incy): element i of a vector with a negative increment
// lies at offset (n - 1 - i) |inc|, and a zero increment repeats one element. Pairing two reversed vectors is the
// same as pairing them in storage order.
template <typename X, typename Y, typename Func>
decltype(auto) apply_blas_vectors(Eigen::Index n, X* x, Eigen::Index incx, Y* y, Eigen::Index incy, Func f) {
  if (incx == 1 && incy == 1) return f(make_vector(x, n), make_vector(y, n));
  if ((incx < 0) == (incy < 0)) return f(make_vector(x, n, std::abs(incx)), make_vector(y, n, std::abs(incy)));
  if (incx < 0) return f(make_vector(x, n, -incx).reverse(), make_vector(y, n, incy));
  return f(make_vector(x, n, incx), make_vector(y, n, -incy).reverse());
}

// x, y := c x + s y, c y - s x on real BLAS vectors.
template <typename R>
void apply_real_rotation(Eigen::Index n, R* x, Eigen::Index incx, R* y, Eigen::Index incy, R c, R s) {
  apply_blas_vectors(n, x, incx, y, incy, [&](auto vx, auto vy) {
    Eigen::internal::apply_rotation_in_the_plane(vx, vy, Eigen::JacobiRotation<R>(c, s));
  });
}

// The 1-based index of the first largest coefficient of v, skipping NaN unless v(0) is NaN, as Reference BLAS 3.12
// does. Eigen's default propagation picks an ISA-dependent index when v holds NaN.
template <typename V>
EIGEN_BLAS_INT blas_index_of_max(const V& v) {
  if ((Eigen::numext::isnan)(v.coeff(0))) return 1;
  Eigen::Index index;
  v.template maxCoeff<Eigen::PropagateNumbers>(&index);
  return EIGEN_BLAS_INT(index + 1);
}
template <typename V>
EIGEN_BLAS_INT blas_index_of_min(const V& v) {
  if ((Eigen::numext::isnan)(v.coeff(0))) return 1;
  Eigen::Index index;
  v.template minCoeff<Eigen::PropagateNumbers>(&index);
  return EIGEN_BLAS_INT(index + 1);
}

// y := beta y for a unit-stride y of length n; beta == 0 overwrites y, so that NaN in y does not survive.
template <typename T>
void scale_vector(T* y, Eigen::Index n, const T& beta) {
  if (beta == T(1)) return;
  if (beta == T(0))
    make_vector(y, n).setZero();
  else
    make_vector(y, n) *= beta;
}

// Unit-stride view of the BLAS vector (x, n, incx): x itself when incx == 1, otherwise a copy in BLAS order that is
// written back on destruction unless T is const.
template <typename T>
class compact_vector {
  using Plain = std::remove_const_t<T>;

 public:
  compact_vector(T* x, Eigen::Index n, Eigen::Index incx) : x_(x), n_(n), incx_(incx), data_(x) {
    if (incx == 1) return;
    copy_.resize(n);
    apply_blas_vectors(n, x, incx, copy_.data(), 1, [](auto vx, auto vc) { vc = vx; });
    data_ = copy_.data();
  }
  ~compact_vector() { write_back(std::is_const<T>()); }
  compact_vector(const compact_vector&) = delete;
  compact_vector& operator=(const compact_vector&) = delete;

  T* data() const { return data_; }

 private:
  void write_back(std::true_type) {}
  void write_back(std::false_type) {
    if (data_ != x_) apply_blas_vectors(n_, copy_.data(), 1, x_, incx_, [](auto vc, auto vx) { vx = vc; });
  }

  T* x_;
  Eigen::Index n_, incx_;
  T* data_;
  Eigen::Matrix<Plain, Eigen::Dynamic, 1> copy_;
};

#ifndef EIGEN_BLAS_FUNC_SUFFIX
#define EIGEN_BLAS_FUNC_SUFFIX _
#endif

#define EIGEN_BLAS_FUNC_NAME(X) EIGEN_CAT(SCALAR_SUFFIX, EIGEN_CAT(X, EIGEN_BLAS_FUNC_SUFFIX))
#define EIGEN_BLAS_FUNC(X) extern "C" void EIGEN_BLAS_FUNC_NAME(X)

#endif  // EIGEN_BLAS_COMMON_H
