// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_STRUCTURED_TOEPLITZ_H
#define EIGEN_STRUCTURED_TOEPLITZ_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename Scalar_, int Rows_ = Dynamic, int Cols_ = Dynamic>
class Toeplitz;

namespace internal {

template <typename Scalar_, int Rows_, int Cols_>
struct traits<Toeplitz<Scalar_, Rows_, Cols_>> {
  using Scalar = Scalar_;
  using StorageKind = Dense;
  using XprKind = MatrixXpr;
  using StorageIndex = int;
  static constexpr int RowsAtCompileTime = Rows_;
  static constexpr int ColsAtCompileTime = Cols_;
  static constexpr int MaxRowsAtCompileTime = Rows_;
  static constexpr int MaxColsAtCompileTime = Cols_;
  static constexpr int Flags = NestByRefBit;
};

template <typename Scalar_, int Rows_, int Cols_>
struct evaluator_traits<Toeplitz<Scalar_, Rows_, Cols_>> {
  using Kind = IndexBased;
  using Shape = StructuredShape;
};

}  // namespace internal

/** \ingroup StructuredMatrices_Module
 * \class Toeplitz
 * \brief An \c m x \c n Toeplitz matrix represented by its first column and row.
 *
 * A Toeplitz matrix has constant diagonals: entry \c (i,j) equals \c c[i-j] when
 * \c i>=j and \c r[j-i] otherwise, where \c c is its first column and \c r its
 * first row. The diagonal entry is taken from \c c, so \c r[0] is ignored.
 *
 * The matrix-vector product (\c operator*) is evaluated in O(n log n) by
 * embedding the Toeplitz matrix in a larger circulant matrix, whose DFT symbol is
 * computed once at construction. As with \ref Circulant, \c operator* returns an
 * Eigen product expression, so a \c Toeplitz also plugs into the matrix-free
 * iterative solvers, and it can be assigned to a dense matrix when an explicit
 * representation is needed.
 *
 * \tparam Scalar_ the scalar type, real or complex.
 * \tparam Rows_ the number of rows at compile time, or \c Dynamic (the default).
 * \tparam Cols_ the number of columns at compile time, or \c Dynamic (the default).
 *
 * \sa class Circulant, makeToeplitz()
 */
template <typename Scalar_, int Rows_, int Cols_>
class Toeplitz : public EigenBase<Toeplitz<Scalar_, Rows_, Cols_>> {
 public:
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using StorageIndex = int;
  using Complex = std::complex<RealScalar>;
  using ColGeneratorType = Matrix<Scalar, Rows_, 1>;
  using RowGeneratorType = Matrix<Scalar, Cols_, 1>;
  using ComplexVector = Matrix<Complex, Dynamic, 1>;

  static constexpr int RowsAtCompileTime = Rows_;
  static constexpr int ColsAtCompileTime = Cols_;
  static constexpr int MaxRowsAtCompileTime = Rows_;
  static constexpr int MaxColsAtCompileTime = Cols_;
  static constexpr int SizeAtCompileTime = internal::size_at_compile_time(Rows_, Cols_);
  static constexpr int MaxSizeAtCompileTime = SizeAtCompileTime;
  static constexpr bool IsRowMajor = false;
  // Deliberately no IsVectorAtCompileTime: Ref<const Toeplitz>'s default StrideType
  // argument reads it, so its absence makes internal::is_ref_compatible SFINAE to
  // false and keeps the iterative solvers on their matrix-free path.

  /** Builds a Toeplitz matrix from its first column \a col and first row \a row.
   * The diagonal is taken from \a col, hence \c row[0] is ignored.
   *
   * Unless the matrix is small enough for products to always take the direct
   * path, it is embedded into a circulant matrix of 5-smooth size p >= m+n-1
   * whose first column is \c [c; 0...0; r[n-1], ..., r[1]]; multiplying that
   * circulant by \c [x; 0] reproduces the Toeplitz product in its leading \c m
   * entries. The circulant's DFT symbol is computed here, once, and reused by
   * every subsequent product. */
  template <typename ColDerived, typename RowDerived>
  Toeplitz(const MatrixBase<ColDerived>& col, const MatrixBase<RowDerived>& row) : m_col(col), m_row(row) {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(ColDerived)
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(RowDerived)
    eigen_assert(m_col.size() > 0 && m_row.size() > 0 && "Toeplitz generators must be non-empty");
    // Below the threshold every product takes the direct path and the symbol has
    // no consumer, so skip the embedding altogether.
    if (rows() > internal::structured_direct_threshold() || cols() > internal::structured_direct_threshold())
      m_symbol = computeSymbol();
  }

  EIGEN_DEVICE_FUNC Index rows() const { return m_col.size(); }
  EIGEN_DEVICE_FUNC Index cols() const { return m_row.size(); }

  /** \returns the generating first column. */
  const ColGeneratorType& column() const { return m_col; }
  /** \returns the generating first row. */
  const RowGeneratorType& row() const { return m_row; }

  /** \returns the symbol of the circulant embedding, i.e. the DFT of the first
   * column of the size-p circulant matrix the operator is embedded in. Cached
   * when the operator is large enough for products to take the FFT path,
   * computed on the fly for small operators. */
  ComplexVector symbol() const { return m_symbol.size() > 0 ? m_symbol : computeSymbol(); }

  /** \returns the coefficient at row \a row and column \a col. */
  Scalar coeff(Index row, Index col) const {
    Index k = row - col;
    return k >= 0 ? m_col.coeff(k) : m_row.coeff(-k);
  }

  /** \internal Writes the dense representation into \a dst; the head of column
   * \c j is a reversed slice of the row generator (entry \c i holds \c r[j-i])
   * and its tail the leading part of the column generator. Invoked through
   * \c dense = toeplitz; */
  template <typename Dest>
  void evalTo(Dest& dst) const {
    const Index m = rows(), n = cols();
    for (Index j = 0; j < n; ++j) {
      const Index h = numext::mini(j, m);
      dst.col(j).head(h) = m_row.segment(j - h + 1, h).reverse();
      if (j < m) dst.col(j).tail(m - j) = m_col.head(m - j);
    }
  }

  /** \internal Computes \c dst += (*this), see evalTo(). */
  template <typename Dest>
  void addTo(Dest& dst) const {
    const Index m = rows(), n = cols();
    for (Index j = 0; j < n; ++j) {
      const Index h = numext::mini(j, m);
      dst.col(j).head(h) += m_row.segment(j - h + 1, h).reverse();
      if (j < m) dst.col(j).tail(m - j) += m_col.head(m - j);
    }
  }

  /** \internal Computes \c dst -= (*this), see evalTo(). */
  template <typename Dest>
  void subTo(Dest& dst) const {
    const Index m = rows(), n = cols();
    for (Index j = 0; j < n; ++j) {
      const Index h = numext::mini(j, m);
      dst.col(j).head(h) -= m_row.segment(j - h + 1, h).reverse();
      if (j < m) dst.col(j).tail(m - j) -= m_col.head(m - j);
    }
  }

  /** \returns the product expression \c (*this) * \a x, evaluated through a fast
   * FFT-based matrix-vector product (circulant embedding). */
  template <typename Rhs>
  Product<Toeplitz, Rhs, AliasFreeProduct> operator*(const MatrixBase<Rhs>& x) const {
    return Product<Toeplitz, Rhs, AliasFreeProduct>(*this, x.derived());
  }

  /** \internal Computes \c dst += alpha * (*this) * rhs. */
  template <typename Dest, typename Rhs>
  void addProduct(Dest& dst, const Rhs& rhs, const Scalar& alpha) const {
    const Index m = rows(), n = cols();
    eigen_assert(rhs.rows() == n && "invalid product: dimensions do not match");

    if (m <= internal::structured_scalar_threshold() && n <= internal::structured_scalar_threshold()) {
      // Tiny sizes: a plain scalar loop beats the segment-based path below, whose
      // per-segment setup dominates when segments hold only a few entries.
      for (Index k = 0; k < rhs.cols(); ++k)
        for (Index i = 0; i < m; ++i) {
          Scalar acc(0);
          for (Index j = 0; j < n; ++j) acc += coeff(i, j) * rhs.coeff(j, k);
          dst.coeffRef(i, k) += alpha * acc;
        }
      return;
    }

    if (m <= internal::structured_direct_threshold() && n <= internal::structured_direct_threshold()) {
      // Direct path: accumulate x_j times the j-th column of the operator. Its
      // head is a reversed slice of the row generator (entry i holds r[j-i]) and
      // its tail the leading part of the column generator; both are segment
      // operations that vectorize.
      for (Index k = 0; k < rhs.cols(); ++k) {
        auto dstCol = dst.col(k);
        for (Index j = 0; j < n; ++j) {
          const Scalar xj = alpha * rhs.coeff(j, k);
          const Index h = numext::mini(j, m);
          dstCol.head(h) += xj * m_row.segment(j - h + 1, h).reverse();
          if (j < m) dstCol.tail(m - j) += xj * m_col.head(m - j);
        }
      }
      return;
    }

    internal::structured_fft_apply(dst, m_symbol, m, rhs, alpha);
  }

 private:
  /** \internal \returns the DFT of the first column of the circulant embedding. */
  ComplexVector computeSymbol() const {
    const Index m = rows(), n = cols();
    const Index p = internal::fft_next_good_size(m + n - 1);
    ComplexVector embedding = ComplexVector::Zero(p);
    embedding.head(m) = m_col.template cast<Complex>();
    embedding.tail(n - 1) = m_row.tail(n - 1).reverse().template cast<Complex>();
    if (p == 1) return embedding;  // the DFT of a single sample is the identity
    ComplexVector symbol(p);
    FFT<RealScalar> fft;
    fft.fwd(symbol, embedding, p);
    return symbol;
  }

  ColGeneratorType m_col;
  RowGeneratorType m_row;
  ComplexVector m_symbol;
};

/** \ingroup StructuredMatrices_Module
 * \returns a \ref Toeplitz operator with first column \a col and first row \a row.
 * The compile-time dimensions of the operator are deduced from the generators. */
template <typename ColDerived, typename RowDerived>
Toeplitz<typename ColDerived::Scalar, ColDerived::SizeAtCompileTime, RowDerived::SizeAtCompileTime> makeToeplitz(
    const MatrixBase<ColDerived>& col, const MatrixBase<RowDerived>& row) {
  return Toeplitz<typename ColDerived::Scalar, ColDerived::SizeAtCompileTime, RowDerived::SizeAtCompileTime>(col, row);
}

namespace internal {

// Single product specialization covering every product tag; see the note in
// Circulant.h.
template <typename Scalar_, int Rows_, int Cols_, typename Rhs, int ProductTag>
struct generic_product_impl<Toeplitz<Scalar_, Rows_, Cols_>, Rhs, StructuredShape, DenseShape, ProductTag>
    : structured_product_impl<Toeplitz<Scalar_, Rows_, Cols_>, Rhs> {};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_STRUCTURED_TOEPLITZ_H
