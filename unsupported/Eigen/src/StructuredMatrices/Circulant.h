// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_STRUCTURED_CIRCULANT_H
#define EIGEN_STRUCTURED_CIRCULANT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename Scalar_, int Size_ = Dynamic>
class Circulant;

namespace internal {

template <typename Scalar_, int Size_>
struct traits<Circulant<Scalar_, Size_>> {
  using Scalar = Scalar_;
  using StorageKind = Dense;
  using XprKind = MatrixXpr;
  using StorageIndex = int;
  static constexpr int RowsAtCompileTime = Size_;
  static constexpr int ColsAtCompileTime = Size_;
  static constexpr int MaxRowsAtCompileTime = Size_;
  static constexpr int MaxColsAtCompileTime = Size_;
  static constexpr int Flags = NestByRefBit;
};

template <typename Scalar_, int Size_>
struct evaluator_traits<Circulant<Scalar_, Size_>> {
  using Kind = IndexBased;
  using Shape = StructuredShape;
};

}  // namespace internal

/** \ingroup StructuredMatrices_Module
 * \class Circulant
 * \brief An \c n x \c n circulant matrix represented by its first column.
 *
 * A circulant matrix is the square matrix whose entry \c (i,j) equals
 * \c c[(i-j) mod n], where \c c is its first column. It is diagonalized by the
 * discrete Fourier transform: its eigenvalues -- the operator's \em symbol,
 * computed once at construction and reused by every product -- are the DFT of
 * \c c. This yields an O(n log n) matrix-vector product (\c operator*) and an
 * O(n log n) direct solve (\ref solve).
 *
 * The operator stores its own copy of the generating column and derives from
 * \c EigenBase. Because \c operator* returns an Eigen product expression, a
 * \c Circulant also drops into the matrix-free iterative solvers, and it can be
 * assigned to a dense matrix when an explicit representation is needed.
 *
 * \tparam Scalar_ the scalar type, real or complex.
 * \tparam Size_ the dimension at compile time, or \c Dynamic (the default).
 *
 * \sa class Toeplitz, makeCirculant()
 */
template <typename Scalar_, int Size_>
class Circulant : public EigenBase<Circulant<Scalar_, Size_>> {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using StorageIndex = int;
  using Complex = std::complex<RealScalar>;
  using GeneratorType = Matrix<Scalar, Size_, 1>;
  using ComplexVector = Matrix<Complex, Dynamic, 1>;

  static constexpr int RowsAtCompileTime = Size_;
  static constexpr int ColsAtCompileTime = Size_;
  static constexpr int MaxRowsAtCompileTime = Size_;
  static constexpr int MaxColsAtCompileTime = Size_;
  static constexpr int SizeAtCompileTime = internal::size_at_compile_time(Size_, Size_);
  static constexpr int MaxSizeAtCompileTime = SizeAtCompileTime;
  static constexpr bool IsRowMajor = false;
  // Deliberately no IsVectorAtCompileTime: Ref<const Circulant>'s default StrideType
  // argument reads it, so its absence makes internal::is_ref_compatible SFINAE to
  // false and keeps the iterative solvers on their matrix-free path.

  /** Builds a circulant matrix from its first column \a col.
   *
   * When the matrix is large enough for products to take the FFT path, the DFT of
   * \a col -- the eigenvalues of the matrix -- is computed here, once, and reused
   * by every subsequent product and solve. */
  template <typename Derived>
  explicit Circulant(const MatrixBase<Derived>& col) : m_col(col) {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
    eigen_assert(m_col.size() > 0 && "Circulant generator must be non-empty");
    if (m_col.size() > internal::structured_direct_threshold()) m_symbol = computeSymbol();
  }

  EIGEN_DEVICE_FUNC Index rows() const { return m_col.size(); }
  EIGEN_DEVICE_FUNC Index cols() const { return m_col.size(); }

  /** \returns the generating first column. */
  const GeneratorType& column() const { return m_col; }

  /** \returns the symbol of the operator, i.e. the DFT of the generating column.
   * Its entries are the eigenvalues of the matrix. Cached when the operator is
   * large enough for products to take the FFT path, computed on the fly for small
   * operators. */
  ComplexVector symbol() const { return m_symbol.size() > 0 ? m_symbol : computeSymbol(); }

  /** \returns the coefficient at row \a row and column \a col. */
  Scalar coeff(Index row, Index col) const {
    Index k = row - col;
    if (k < 0) k += rows();
    return m_col.coeff(k);
  }

  /** \internal Writes the dense representation into \a dst; column \c j is the
   * generator rotated downwards by \c j, so only contiguous segment copies are
   * involved. Invoked through \c dense = circulant; */
  template <typename Dest>
  void evalTo(Dest& dst) const {
    const Index n = rows();
    for (Index j = 0; j < n; ++j) {
      dst.col(j).head(j) = m_col.tail(j);
      dst.col(j).tail(n - j) = m_col.head(n - j);
    }
  }

  /** \internal Computes \c dst += (*this), see evalTo(). */
  template <typename Dest>
  void addTo(Dest& dst) const {
    const Index n = rows();
    for (Index j = 0; j < n; ++j) {
      dst.col(j).head(j) += m_col.tail(j);
      dst.col(j).tail(n - j) += m_col.head(n - j);
    }
  }

  /** \internal Computes \c dst -= (*this), see evalTo(). */
  template <typename Dest>
  void subTo(Dest& dst) const {
    const Index n = rows();
    for (Index j = 0; j < n; ++j) {
      dst.col(j).head(j) -= m_col.tail(j);
      dst.col(j).tail(n - j) -= m_col.head(n - j);
    }
  }

  /** \returns the product expression \c (*this) * \a x, evaluated through a fast
   * FFT-based matrix-vector product. */
  template <typename Rhs>
  Product<Circulant, Rhs, AliasFreeProduct> operator*(const MatrixBase<Rhs>& x) const {
    return Product<Circulant, Rhs, AliasFreeProduct>(*this, x.derived());
  }

  /** \returns the solution \c x of \c (*this) * x = b, computed directly in the
   * Fourier domain: \c x = ifft( fft(b) ./ fft(c) ).
   * \warning The circulant matrix is assumed to be non-singular. */
  template <typename Rhs>
  Matrix<Scalar, Size_, Rhs::ColsAtCompileTime> solve(const MatrixBase<Rhs>& b) const {
    const Index n = rows();
    eigen_assert(b.rows() == n && "right-hand side has the wrong number of rows");
    Matrix<Scalar, Size_, Rhs::ColsAtCompileTime> x(n, b.cols());
    x.setZero();
    // Applying the circulant operator whose symbol is the inverse of ours is
    // exactly the inverse operator.
    internal::structured_fft_apply(x, symbol().cwiseInverse().eval(), n, b.derived(), Scalar(1));
    return x;
  }

  /** \internal Computes \c dst += alpha * (*this) * rhs. */
  template <typename Dest, typename Rhs>
  void addProduct(Dest& dst, const Rhs& rhs, const Scalar& alpha) const {
    const Index n = rows();
    eigen_assert(rhs.rows() == n && "invalid product: dimensions do not match");

    if (n <= internal::structured_scalar_threshold()) {
      // Tiny sizes: a plain scalar loop beats the segment-based path below, whose
      // per-segment setup dominates when segments hold only a few entries.
      for (Index k = 0; k < rhs.cols(); ++k)
        for (Index i = 0; i < n; ++i) {
          Scalar acc(0);
          for (Index j = 0; j < n; ++j) acc += coeff(i, j) * rhs.coeff(j, k);
          dst.coeffRef(i, k) += alpha * acc;
        }
      return;
    }

    if (n <= internal::structured_direct_threshold()) {
      // Direct path: accumulate x_j times the j-th column of the operator, which
      // is the generator rotated downwards by j. Only contiguous, forward segment
      // operations are involved, so everything vectorizes.
      for (Index k = 0; k < rhs.cols(); ++k) {
        auto dstCol = dst.col(k);
        for (Index j = 0; j < n; ++j) {
          const Scalar xj = alpha * rhs.coeff(j, k);
          dstCol.head(j) += xj * m_col.tail(j);
          dstCol.tail(n - j) += xj * m_col.head(n - j);
        }
      }
      return;
    }

    internal::structured_fft_apply(dst, m_symbol, n, rhs, alpha);
  }

 private:
  /** \internal \returns the DFT of the generating column. */
  ComplexVector computeSymbol() const {
    const Index n = m_col.size();
    const ComplexVector cc = m_col.template cast<Complex>();
    if (n == 1) return cc;  // the DFT of a single sample is the identity
    ComplexVector symbol(n);
    FFT<RealScalar> fft;
    fft.fwd(symbol, cc, n);
    return symbol;
  }

  GeneratorType m_col;
  ComplexVector m_symbol;
};

/** \ingroup StructuredMatrices_Module
 * \returns a \ref Circulant operator with first column \a col. The compile-time
 * size of the operator is deduced from \a col. */
template <typename Derived>
Circulant<typename Derived::Scalar, Derived::SizeAtCompileTime> makeCirculant(const MatrixBase<Derived>& col) {
  return Circulant<typename Derived::Scalar, Derived::SizeAtCompileTime>(col);
}

namespace internal {

// The StructuredShape key makes this single partial specialization cover every
// product tag (GEMV, GEMM, coeff-based, ...) without ambiguity against the stock
// dense specializations, so products of any size and fixedness reach the fast path.
template <typename Scalar_, int Size_, typename Rhs, int ProductTag>
struct generic_product_impl<Circulant<Scalar_, Size_>, Rhs, StructuredShape, DenseShape, ProductTag>
    : structured_product_impl<Circulant<Scalar_, Size_>, Rhs> {};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_STRUCTURED_CIRCULANT_H
