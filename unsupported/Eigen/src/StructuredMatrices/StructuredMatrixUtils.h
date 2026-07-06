// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_STRUCTURED_MATRIX_UTILS_H
#define EIGEN_STRUCTURED_MATRIX_UTILS_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// Evaluator shape of the structured operator types. Keying the operators on their
// own shape (instead of DenseShape) routes dense assignment through the
// EigenBase2EigenBase path (i.e. evalTo/addTo/subTo) and products through a single
// generic_product_impl partial specialization covering every product tag.
struct StructuredShape {};

// Below this dimension the FFT setup costs more than a plain O(n^2) evaluation,
// so the structured operators fall back to a direct segment-based product.
constexpr Index structured_direct_threshold() { return 32; }

// Below this dimension even the segment-based direct product loses to a plain
// scalar loop: the per-segment setup dominates when the average segment holds
// fewer than a couple of packets (measured crossover on AVX2 hardware).
constexpr Index structured_scalar_threshold() { return 16; }

/** \internal
 * \returns the smallest integer >= \a n whose only prime factors are 2, 3 and 5.
 *
 * Such "5-smooth" sizes keep the FFT fast and sidestep the default kissfft
 * backend's poor handling of sizes with large prime factors. Used to pad the
 * circulant embedding of a Toeplitz matrix. The {2,3,5}-smooth numbers are dense
 * enough that the linear search returns after only a handful of steps.
 */
inline Index fft_next_good_size(Index n) {
  if (n < 1) return 1;
  for (Index m = n;; ++m) {
    Index r = m;
    while (r % 2 == 0) r /= 2;
    while (r % 3 == 0) r /= 3;
    while (r % 5 == 0) r /= 5;
    if (r == 1) return m;
  }
}

/** \internal View a complex-valued expression as \a Scalar: the expression itself
 * when \a Scalar is complex, its real part when \a Scalar is real (the imaginary
 * part then only holds numerically negligible roundoff). A single dispatch struct
 * keeps the number of instantiated helpers down to one per scalar type. */
template <typename Scalar, bool IsComplex = NumTraits<Scalar>::IsComplex>
struct structured_scalar_part_impl {
  template <typename Xpr>
  static const Xpr& run(const Xpr& xpr) {
    return xpr;
  }
};

template <typename Scalar>
struct structured_scalar_part_impl<Scalar, false> {
  template <typename Xpr>
  static typename Xpr::RealReturnType run(const Xpr& xpr) {
    return xpr.real();
  }
};

/** \internal
 * Computes \c dst.col(k) += alpha * ifft( symbol .* fft(rhs.col(k)) ) for every
 * column of \a rhs, i.e. applies the circulant operator whose eigenvalues are
 * \a symbol. The leading \a outSize entries of each back-transform form the
 * corresponding output column. All workspace is allocated once outside the
 * per-column loop; right-hand sides shorter than the transform length are
 * zero-padded into the preallocated buffer so the FFT never re-allocates.
 */
template <typename Scalar, typename Dest, typename Rhs>
void structured_fft_apply(Dest& dst, const Matrix<std::complex<typename NumTraits<Scalar>::Real>, Dynamic, 1>& symbol,
                          Index outSize, const Rhs& rhs, const Scalar& alpha) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Complex = std::complex<RealScalar>;
  using ComplexVector = Matrix<Complex, Dynamic, 1>;

  const Index p = symbol.size();
  eigen_assert(rhs.rows() <= p && outSize <= p);

  if (p == 1) {
    // Degenerate 1x1 operator: the length-1 transform is the identity (and is not
    // supported by the kissfft backend anyway).
    dst.row(0) +=
        alpha * structured_scalar_part_impl<Scalar>::run(symbol.coeff(0) * rhs.row(0).template cast<Complex>());
    return;
  }

  FFT<RealScalar> fft;
  ComplexVector xt = ComplexVector::Zero(p);  // the zero padding beyond rhs.rows() is never overwritten
  ComplexVector xf(p), yt(p);
  for (Index k = 0; k < rhs.cols(); ++k) {
    xt.head(rhs.rows()) = rhs.col(k).template cast<Complex>();
    fft.fwd(xf, xt, p);
    xf.array() *= symbol.array();
    fft.inv(yt, xf, p);
    dst.col(k) += alpha * structured_scalar_part_impl<Scalar>::run(yt.head(outSize));
  }
}

/** \internal Shared product implementation for the structured operator types.
 * Forwards to the operator's \c addProduct member, which performs the fast
 * matrix-vector product. The same body serves every dense product dispatch tag. */
template <typename Op, typename Rhs>
struct structured_product_impl : generic_product_impl_base<Op, Rhs, structured_product_impl<Op, Rhs>> {
  using Scalar = typename Product<Op, Rhs>::Scalar;
  template <typename Dest>
  static void scaleAndAddTo(Dest& dst, const Op& lhs, const Rhs& rhs, const Scalar& alpha) {
    lhs.addProduct(dst, rhs, alpha);
  }
};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_STRUCTURED_MATRIX_UTILS_H
