// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SCALEDPERMUTATIONMATRIX_H
#define EIGEN_SCALEDPERMUTATIONMATRIX_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename Scalar_, int SizeAtCompileTime, int MaxSizeAtCompileTime, typename StorageIndex_>
struct traits<ScaledPermutationMatrix<Scalar_, SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex_>> {
  using Scalar = Scalar_;
  using StorageIndex = StorageIndex_;
  // Dense, so that products and sums with dense operands promote with the existing rules; the O(n) kernels are
  // selected by the shape (ScaledPermutationShape), as for TriangularView.
  using StorageKind = Dense;
  using XprKind = MatrixXpr;
  using PermutationType = PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex_>;
  using ScalesType = Matrix<Scalar_, SizeAtCompileTime, 1, 0, MaxSizeAtCompileTime, 1>;
  static constexpr int RowsAtCompileTime = SizeAtCompileTime;
  static constexpr int ColsAtCompileTime = SizeAtCompileTime;
  static constexpr int MaxRowsAtCompileTime = MaxSizeAtCompileTime;
  static constexpr int MaxColsAtCompileTime = MaxSizeAtCompileTime;
  static constexpr unsigned int Flags = NestByRefBit;
};

/** \internal The scaled permutation produced by a permutation-like and a diagonal-like operand. */
template <typename Scalar, typename Lhs, typename Rhs, typename StorageIndex>
struct scaled_permutation_result {
  static constexpr int Size = size_prefer_fixed(traits<Lhs>::RowsAtCompileTime, traits<Rhs>::RowsAtCompileTime);
  static constexpr int MaxSize =
      min_size_prefer_fixed(traits<Lhs>::MaxRowsAtCompileTime, traits<Rhs>::MaxRowsAtCompileTime);
  using type = ScaledPermutationMatrix<Scalar, Size, MaxSize, StorageIndex>;
};

template <typename Lhs, typename Rhs, typename StorageIndex>
struct scaled_permutation_product_result {
  using LhsScalar = typename traits<Lhs>::Scalar;
  using RhsScalar = typename traits<Rhs>::Scalar;
  using Scalar =
      typename ScalarBinaryOpTraits<LhsScalar, RhsScalar, scalar_product_op<LhsScalar, RhsScalar>>::ReturnType;
  using type = typename scaled_permutation_result<Scalar, Lhs, Rhs, StorageIndex>::type;
};

/** \internal indices of the inverse permutation: result(indices(k)) = k. */
template <typename IndicesType, typename Result>
EIGEN_DEVICE_FUNC void invert_permutation_indices(const IndicesType& indices, Result& result) {
  using StorageIndex = typename Result::Scalar;
  for (Index k = 0; k < indices.size(); ++k) result.coeffRef(indices.coeff(k)) = StorageIndex(k);
}

}  // namespace internal

/** \class ScaledPermutationBase
 * \ingroup Core_Module
 *
 * \brief Base class for scaled permutation matrices
 *
 * A scaled permutation is the product $ P D $ of a permutation matrix and a diagonal matrix: a square matrix with
 * exactly one nonzero per row and per column. It is stored as the permutation's indices and one scale per
 * column, so that coefficient (indices()(k), k) equals scales()(k).
 *
 * The set is closed under products with permutations, diagonal matrices, scalars and other scaled permutations,
 * and under inversion and transposition; all of these cost $ O(n) $ and return a plain ScaledPermutationMatrix.
 * Products with a dense matrix scale and permute its rows or columns in $ O(nm) $, and assigning to a dense matrix
 * scatters the scales, so writing `P * D * x` or `P * D * P.inverse()` never forms an $ n \times n $ intermediate.
 * Sums with a dense matrix are lazy expressions.
 *
 * \sa class ScaledPermutationMatrix, class PermutationBase, class DiagonalBase
 */
template <typename Derived>
class ScaledPermutationBase : public EigenBase<Derived> {
  using Traits = internal::traits<Derived>;
  using Base = EigenBase<Derived>;

 public:
#ifndef EIGEN_PARSED_BY_DOXYGEN
  using Scalar = typename Traits::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using StorageIndex = typename Traits::StorageIndex;
  using StorageKind = typename Traits::StorageKind;
  using PermutationType = typename Traits::PermutationType;
  using ScalesType = typename Traits::ScalesType;
  using IndicesType = typename PermutationType::IndicesType;
  static constexpr int RowsAtCompileTime = Traits::RowsAtCompileTime;
  static constexpr int ColsAtCompileTime = Traits::ColsAtCompileTime;
  static constexpr int MaxRowsAtCompileTime = Traits::MaxRowsAtCompileTime;
  static constexpr int MaxColsAtCompileTime = Traits::MaxColsAtCompileTime;
  static constexpr int SizeAtCompileTime = internal::size_at_compile_time(RowsAtCompileTime, ColsAtCompileTime);
  static constexpr int MaxSizeAtCompileTime =
      internal::size_at_compile_time(MaxRowsAtCompileTime, MaxColsAtCompileTime);
  static constexpr bool IsVectorAtCompileTime = false;
  static constexpr unsigned int Flags = Traits::Flags;
  using DenseMatrixType =
      Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, 0, MaxRowsAtCompileTime, MaxColsAtCompileTime>;
  using PlainObject = ScaledPermutationMatrix<Scalar, RowsAtCompileTime, MaxRowsAtCompileTime, StorageIndex>;
  using Nested = const Derived&;
  using Base::derived;
#endif

  /** \returns the number of rows */
  EIGEN_DEVICE_FUNC constexpr Index rows() const { return indices().size(); }
  /** \returns the number of columns */
  EIGEN_DEVICE_FUNC constexpr Index cols() const { return indices().size(); }

  /** \returns the permutation factor $ P $ */
  EIGEN_DEVICE_FUNC const PermutationType& permutation() const { return derived().permutation(); }
  /** \returns the permutation's indices: column \c k has its nonzero in row `indices()(k)` */
  EIGEN_DEVICE_FUNC constexpr const IndicesType& indices() const { return derived().indices(); }
  /** \returns the scales: column \c k's nonzero equals `scales()(k)` */
  EIGEN_DEVICE_FUNC const ScalesType& scales() const { return derived().scales(); }

  /** \returns the coefficient at (\a row, \a col) of the represented matrix */
  EIGEN_DEVICE_FUNC Scalar coeff(Index row, Index col) const {
    eigen_assert(row >= 0 && col >= 0 && row < rows() && col < cols());
    return Index(indices().coeff(col)) == row ? scales().coeff(col) : Scalar(0);
  }

  /** \returns the represented matrix as a plain dense matrix */
  EIGEN_DEVICE_FUNC DenseMatrixType toDenseMatrix() const { return derived(); }

#ifndef EIGEN_PARSED_BY_DOXYGEN
  template <typename DenseDerived>
  EIGEN_DEVICE_FUNC void evalTo(MatrixBase<DenseDerived>& other) const {
    other.setZero();
    for (Index k = 0; k < rows(); ++k) other.coeffRef(indices().coeff(k), k) = scales().coeff(k);
  }
#endif

  /** \returns the inverse, $ (PD)^{-1} = D^{-1} P^{-1} $, which has $ 1 / \text{scales}(k) $ at
   * (\c k, `indices()(k)`). The scales must be nonzero. */
  PlainObject inverse() const {
    PlainObject result(rows());
    for (Index k = 0; k < rows(); ++k) {
      const Index image = indices().coeff(k);
      result.indices().coeffRef(image) = StorageIndex(k);
      result.scales().coeffRef(image) = Scalar(1) / scales().coeff(k);
    }
    return result;
  }

  /** \returns the transpose, which has `scales()(k)` at (\c k, `indices()(k)`) */
  PlainObject transpose() const {
    PlainObject result(rows());
    for (Index k = 0; k < rows(); ++k) {
      const Index image = indices().coeff(k);
      result.indices().coeffRef(image) = StorageIndex(k);
      result.scales().coeffRef(image) = scales().coeff(k);
    }
    return result;
  }

  /** \returns the adjoint, the transpose with conjugated scales */
  PlainObject adjoint() const {
    PlainObject result = transpose();
    result.scales() = result.scales().conjugate();
    return result;
  }

  /** \returns the determinant, the permutation's sign times the product of the scales */
  Scalar determinant() const { return Scalar(RealScalar(permutation().determinant())) * scales().prod(); }

  /** \returns the dense matrix product of \c *this by the dense matrix \a other: rows of \a other are scaled and
   * permuted. */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC const Product<Derived, OtherDerived, DefaultProduct> operator*(
      const MatrixBase<OtherDerived>& other) const {
    return Product<Derived, OtherDerived, DefaultProduct>(derived(), other.derived());
  }

  /** \returns the dense matrix product of the dense matrix \a other by \a scaled: columns of \a other are scaled
   * and permuted. */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC friend const Product<OtherDerived, Derived, DefaultProduct> operator*(
      const MatrixBase<OtherDerived>& other, const ScaledPermutationBase& scaled) {
    return Product<OtherDerived, Derived, DefaultProduct>(other.derived(), scaled.derived());
  }

  /** \returns the dense matrix product of \c *this by the triangular or self-adjoint view \a other */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC const Product<Derived, OtherDerived, DefaultProduct> operator*(
      const TriangularBase<OtherDerived>& other) const {
    return Product<Derived, OtherDerived, DefaultProduct>(derived(), other.derived());
  }

  /** \returns the dense matrix product of the triangular or self-adjoint view \a other by \a scaled */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC friend const Product<OtherDerived, Derived, DefaultProduct> operator*(
      const TriangularBase<OtherDerived>& other, const ScaledPermutationBase& scaled) {
    return Product<OtherDerived, Derived, DefaultProduct>(other.derived(), scaled.derived());
  }

  // Closed-form products, all O(n). With S e_c = v_c e_{tau(c)} the rules are:
  //   S D: scale v_c d_c            D S: scale d_{tau(c)} v_c
  //   S P: (tau(sigma(c)), v_{sigma(c)})   P S: (sigma(tau(c)), v_c)
  //   S T: (tau(tau2(c)), v_{tau2(c)} v2_c)

  /** \returns the scaled permutation \c *this times the diagonal matrix \a other */
  template <typename OtherDerived>
  typename internal::scaled_permutation_product_result<Derived, OtherDerived, StorageIndex>::type operator*(
      const DiagonalBase<OtherDerived>& other) const {
    using Result = typename internal::scaled_permutation_product_result<Derived, OtherDerived, StorageIndex>::type;
    eigen_assert(rows() == other.rows());
    return Result(indices(), scales().cwiseProduct(other.diagonal()));
  }

  /** \returns the scaled permutation diagonal matrix \a other times \a scaled */
  template <typename OtherDerived>
  friend typename internal::scaled_permutation_product_result<OtherDerived, Derived, StorageIndex>::type operator*(
      const DiagonalBase<OtherDerived>& other, const ScaledPermutationBase& scaled) {
    using Result = typename internal::scaled_permutation_product_result<OtherDerived, Derived, StorageIndex>::type;
    eigen_assert(other.rows() == scaled.rows());
    Result result(scaled.rows());
    result.indices() = scaled.indices();
    for (Index k = 0; k < scaled.rows(); ++k)
      result.scales().coeffRef(k) = other.diagonal().coeff(scaled.indices().coeff(k)) * scaled.scales().coeff(k);
    return result;
  }

  /** \returns the scaled permutation \c *this times the permutation \a other */
  template <typename OtherDerived>
  PlainObject operator*(const PermutationBase<OtherDerived>& other) const {
    eigen_assert(rows() == other.rows());
    PlainObject result(rows());
    for (Index k = 0; k < rows(); ++k) {
      const Index source = other.indices().coeff(k);
      result.indices().coeffRef(k) = indices().coeff(source);
      result.scales().coeffRef(k) = scales().coeff(source);
    }
    return result;
  }

  /** \returns the scaled permutation permutation \a other times \a scaled */
  template <typename OtherDerived>
  friend PlainObject operator*(const PermutationBase<OtherDerived>& other, const ScaledPermutationBase& scaled) {
    eigen_assert(other.rows() == scaled.rows());
    PlainObject result(scaled.rows());
    for (Index k = 0; k < scaled.rows(); ++k)
      result.indices().coeffRef(k) = StorageIndex(other.indices().coeff(scaled.indices().coeff(k)));
    result.scales() = scaled.scales();
    return result;
  }

  /** \returns the scaled permutation \c *this times the inverse permutation \a other */
  template <typename OtherDerived>
  PlainObject operator*(const InverseImpl<OtherDerived, PermutationStorage>& other) const {
    eigen_assert(rows() == other.rows());
    IndicesType inverse(rows());
    internal::invert_permutation_indices(other.derived().nestedExpression().indices(), inverse);
    PlainObject result(rows());
    for (Index k = 0; k < rows(); ++k) {
      const Index source = inverse.coeff(k);
      result.indices().coeffRef(k) = indices().coeff(source);
      result.scales().coeffRef(k) = scales().coeff(source);
    }
    return result;
  }

  /** \returns the scaled permutation inverse permutation \a other times \a scaled */
  template <typename OtherDerived>
  friend PlainObject operator*(const InverseImpl<OtherDerived, PermutationStorage>& other,
                               const ScaledPermutationBase& scaled) {
    eigen_assert(other.rows() == scaled.rows());
    IndicesType inverse(scaled.rows());
    internal::invert_permutation_indices(other.derived().nestedExpression().indices(), inverse);
    PlainObject result(scaled.rows());
    for (Index k = 0; k < scaled.rows(); ++k) result.indices().coeffRef(k) = inverse.coeff(scaled.indices().coeff(k));
    result.scales() = scaled.scales();
    return result;
  }

  /** \returns the scaled permutation \c *this times the scaled permutation \a other */
  template <typename OtherDerived>
  typename internal::scaled_permutation_product_result<Derived, OtherDerived, StorageIndex>::type operator*(
      const ScaledPermutationBase<OtherDerived>& other) const {
    using Result = typename internal::scaled_permutation_product_result<Derived, OtherDerived, StorageIndex>::type;
    eigen_assert(rows() == other.rows());
    Result result(rows());
    for (Index k = 0; k < rows(); ++k) {
      const Index source = other.indices().coeff(k);
      result.indices().coeffRef(k) = indices().coeff(source);
      result.scales().coeffRef(k) = scales().coeff(source) * other.scales().coeff(k);
    }
    return result;
  }

  /** \returns \c *this with its scales multiplied by \a alpha */
  PlainObject operator*(const Scalar& alpha) const { return PlainObject(indices(), scales() * alpha); }

  /** \returns \a scaled with its scales multiplied by \a alpha */
  friend PlainObject operator*(const Scalar& alpha, const ScaledPermutationBase& scaled) {
    return PlainObject(scaled.indices(), alpha * scaled.scales());
  }

  /** \returns \c *this with its scales negated */
  PlainObject operator-() const { return PlainObject(indices(), -scales()); }

  // Sums with a dense matrix are lazy: the scaled permutation is read through its index-based evaluator.

  /** \returns the lazy sum of the dense matrix \a lhs and the scaled permutation \a rhs */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC friend const EIGEN_CWISE_BINARY_RETURN_TYPE(OtherDerived, Derived, internal::scalar_sum_op)
  operator+(const MatrixBase<OtherDerived>& lhs, const ScaledPermutationBase & rhs) {
    return EIGEN_CWISE_BINARY_RETURN_TYPE(OtherDerived, Derived, internal::scalar_sum_op)(lhs.derived(), rhs.derived());
  }

  /** \returns the lazy sum of the scaled permutation \a lhs and the dense matrix \a rhs */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC friend const EIGEN_CWISE_BINARY_RETURN_TYPE(Derived, OtherDerived, internal::scalar_sum_op)
  operator+(const ScaledPermutationBase & lhs, const MatrixBase<OtherDerived>& rhs) {
    return EIGEN_CWISE_BINARY_RETURN_TYPE(Derived, OtherDerived, internal::scalar_sum_op)(lhs.derived(), rhs.derived());
  }

  /** \returns the lazy difference of the dense matrix \a lhs and the scaled permutation \a rhs */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC friend const EIGEN_CWISE_BINARY_RETURN_TYPE(OtherDerived, Derived, internal::scalar_difference_op)
  operator-(const MatrixBase<OtherDerived>& lhs, const ScaledPermutationBase & rhs) {
    return EIGEN_CWISE_BINARY_RETURN_TYPE(OtherDerived, Derived, internal::scalar_difference_op)(lhs.derived(),
                                                                                                 rhs.derived());
  }

  /** \returns the lazy difference of the scaled permutation \a lhs and the dense matrix \a rhs */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC friend const EIGEN_CWISE_BINARY_RETURN_TYPE(Derived, OtherDerived, internal::scalar_difference_op)
  operator-(const ScaledPermutationBase & lhs, const MatrixBase<OtherDerived>& rhs) {
    return EIGEN_CWISE_BINARY_RETURN_TYPE(Derived, OtherDerived, internal::scalar_difference_op)(lhs.derived(),
                                                                                                 rhs.derived());
  }
};

/** \class ScaledPermutationMatrix
 * \ingroup Core_Module
 *
 * \brief Scaled permutation matrix $ P D $ with its storage
 *
 * \tparam Scalar_ the type of the scales
 * \tparam SizeAtCompileTime the number of rows/cols, or Dynamic
 * \tparam MaxSizeAtCompileTime the maximum number of rows/cols, or Dynamic; defaults to SizeAtCompileTime
 * \tparam StorageIndex_ the integer type of the permutation's indices
 *
 * This is the type returned by the products of a permutation (or its inverse) and a diagonal matrix, in either
 * order. See ScaledPermutationBase for the operations.
 *
 * \sa class ScaledPermutationBase, class PermutationMatrix, class DiagonalMatrix
 */
template <typename Scalar_, int SizeAtCompileTime, int MaxSizeAtCompileTime, typename StorageIndex_>
class ScaledPermutationMatrix
    : public ScaledPermutationBase<
          ScaledPermutationMatrix<Scalar_, SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex_>> {
  using Base = ScaledPermutationBase<ScaledPermutationMatrix>;
  using Traits = internal::traits<ScaledPermutationMatrix>;

 public:
#ifndef EIGEN_PARSED_BY_DOXYGEN
  using Scalar = Scalar_;
  using StorageIndex = StorageIndex_;
  using PermutationType = typename Traits::PermutationType;
  using ScalesType = typename Traits::ScalesType;
  using IndicesType = typename PermutationType::IndicesType;
#endif

  /** Default constructor; the matrix is uninitialized. */
  EIGEN_DEVICE_FUNC ScaledPermutationMatrix() = default;

  /** Constructs an uninitialized scaled permutation of the given size. */
  EIGEN_DEVICE_FUNC explicit ScaledPermutationMatrix(Index size) : m_permutation(size), m_scales(size) {}

  /** Constructs the product \a permutation times \a diagonal. */
  template <typename PermutationDerived, typename DiagonalDerived>
  EIGEN_DEVICE_FUNC ScaledPermutationMatrix(const PermutationBase<PermutationDerived>& permutation,
                                            const DiagonalBase<DiagonalDerived>& diagonal)
      : m_permutation(permutation), m_scales(diagonal.diagonal()) {
    eigen_assert(permutation.rows() == diagonal.rows());
  }

  /** Constructs the scaled permutation with the given \a indices and \a scales. */
  template <typename IndicesDerived, typename ScalesDerived>
  EIGEN_DEVICE_FUNC ScaledPermutationMatrix(const MatrixBase<IndicesDerived>& indices,
                                            const MatrixBase<ScalesDerived>& scales)
      : m_permutation(indices), m_scales(scales) {
    eigen_assert(indices.size() == scales.size());
  }

  /** Converts the permutation \a permutation, all scales being one. */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC explicit ScaledPermutationMatrix(const PermutationBase<OtherDerived>& permutation)
      : m_permutation(permutation), m_scales(ScalesType::Ones(permutation.rows())) {}

  /** Converts the diagonal matrix \a diagonal, the permutation being the identity. */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC explicit ScaledPermutationMatrix(const DiagonalBase<OtherDerived>& diagonal)
      : m_permutation(diagonal.rows()), m_scales(diagonal.diagonal()) {
    m_permutation.setIdentity();
  }

  /** Copies the scaled permutation \a other. */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC ScaledPermutationMatrix(const ScaledPermutationBase<OtherDerived>& other)
      : m_permutation(other.permutation()), m_scales(other.scales()) {}

  /** Copies the scaled permutation \a other. */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC ScaledPermutationMatrix& operator=(const ScaledPermutationBase<OtherDerived>& other) {
    m_permutation = other.permutation();
    m_scales = other.scales();
    return *this;
  }

  /** \returns the permutation factor */
  EIGEN_DEVICE_FUNC const PermutationType& permutation() const { return m_permutation; }
  /** \returns a reference to the permutation factor */
  EIGEN_DEVICE_FUNC PermutationType& permutation() { return m_permutation; }
  /** \returns the permutation's indices */
  EIGEN_DEVICE_FUNC constexpr const IndicesType& indices() const { return m_permutation.indices(); }
  /** \returns a reference to the permutation's indices */
  EIGEN_DEVICE_FUNC constexpr IndicesType& indices() { return m_permutation.indices(); }
  /** \returns the scales */
  EIGEN_DEVICE_FUNC const ScalesType& scales() const { return m_scales; }
  /** \returns a reference to the scales */
  EIGEN_DEVICE_FUNC ScalesType& scales() { return m_scales; }

  /** Resizes to the given size, leaving the coefficients uninitialized. */
  EIGEN_DEVICE_FUNC void resize(Index size) {
    m_permutation.resize(size);
    m_scales.resize(size);
  }

  /** Sets \c *this to the identity matrix. */
  EIGEN_DEVICE_FUNC void setIdentity() {
    m_permutation.setIdentity();
    m_scales.setOnes();
  }

  /** Sets \c *this to the identity matrix of the given size. */
  EIGEN_DEVICE_FUNC void setIdentity(Index size) {
    resize(size);
    setIdentity();
  }

 protected:
  PermutationType m_permutation;
  ScalesType m_scales;
};

// The products of a permutation (or its inverse) and a diagonal matrix. With P e_c = e_{sigma(c)}:
//   P D:   (sigma(c), d_c)          D P:   (sigma(c), d_{sigma(c)})
//   P^-1 D: (sigma^-1(c), d_c)      D P^-1: (sigma^-1(c), d_{sigma^-1(c)})

/** \returns the scaled permutation \a permutation times \a diagonal
 * \relates ScaledPermutationMatrix */
template <typename PermutationDerived, typename DiagonalDerived>
typename internal::scaled_permutation_result<typename DiagonalDerived::Scalar, PermutationDerived, DiagonalDerived,
                                             typename PermutationDerived::StorageIndex>::type
operator*(const PermutationBase<PermutationDerived>& permutation, const DiagonalBase<DiagonalDerived>& diagonal) {
  using Result =
      typename internal::scaled_permutation_result<typename DiagonalDerived::Scalar, PermutationDerived,
                                                   DiagonalDerived, typename PermutationDerived::StorageIndex>::type;
  eigen_assert(permutation.rows() == diagonal.rows());
  return Result(permutation.indices(), diagonal.diagonal());
}

/** \returns the scaled permutation \a diagonal times \a permutation
 * \relates ScaledPermutationMatrix */
template <typename DiagonalDerived, typename PermutationDerived>
typename internal::scaled_permutation_result<typename DiagonalDerived::Scalar, PermutationDerived, DiagonalDerived,
                                             typename PermutationDerived::StorageIndex>::type
operator*(const DiagonalBase<DiagonalDerived>& diagonal, const PermutationBase<PermutationDerived>& permutation) {
  using Result =
      typename internal::scaled_permutation_result<typename DiagonalDerived::Scalar, PermutationDerived,
                                                   DiagonalDerived, typename PermutationDerived::StorageIndex>::type;
  eigen_assert(permutation.rows() == diagonal.rows());
  Result result(permutation.rows());
  result.indices() = permutation.indices();
  for (Index k = 0; k < permutation.rows(); ++k)
    result.scales().coeffRef(k) = diagonal.diagonal().coeff(permutation.indices().coeff(k));
  return result;
}

/** \returns the scaled permutation \a inverse (an inverse permutation) times \a diagonal
 * \relates ScaledPermutationMatrix */
template <typename PermutationType, typename DiagonalDerived>
typename internal::scaled_permutation_result<typename DiagonalDerived::Scalar, PermutationType, DiagonalDerived,
                                             typename PermutationType::StorageIndex>::type
operator*(const InverseImpl<PermutationType, PermutationStorage>& inverse,
          const DiagonalBase<DiagonalDerived>& diagonal) {
  using Result =
      typename internal::scaled_permutation_result<typename DiagonalDerived::Scalar, PermutationType, DiagonalDerived,
                                                   typename PermutationType::StorageIndex>::type;
  eigen_assert(inverse.rows() == diagonal.rows());
  Result result(inverse.rows());
  internal::invert_permutation_indices(inverse.derived().nestedExpression().indices(), result.indices());
  result.scales() = diagonal.diagonal();
  return result;
}

/** \returns the scaled permutation \a diagonal times \a inverse (an inverse permutation)
 * \relates ScaledPermutationMatrix */
template <typename DiagonalDerived, typename PermutationType>
typename internal::scaled_permutation_result<typename DiagonalDerived::Scalar, PermutationType, DiagonalDerived,
                                             typename PermutationType::StorageIndex>::type
operator*(const DiagonalBase<DiagonalDerived>& diagonal,
          const InverseImpl<PermutationType, PermutationStorage>& inverse) {
  using Result =
      typename internal::scaled_permutation_result<typename DiagonalDerived::Scalar, PermutationType, DiagonalDerived,
                                                   typename PermutationType::StorageIndex>::type;
  eigen_assert(inverse.rows() == diagonal.rows());
  Result result(inverse.rows());
  internal::invert_permutation_indices(inverse.derived().nestedExpression().indices(), result.indices());
  for (Index k = 0; k < inverse.rows(); ++k)
    result.scales().coeffRef(k) = diagonal.diagonal().coeff(result.indices().coeff(k));
  return result;
}

namespace internal {

template <typename Scalar_, int SizeAtCompileTime, int MaxSizeAtCompileTime, typename StorageIndex_>
struct evaluator_traits<ScaledPermutationMatrix<Scalar_, SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex_>> {
  using Kind = IndexBased;
  using Shape = ScaledPermutationShape;
};

/** \internal Index-based evaluator, for the lazy sums with a dense matrix. Every coefficient is synthesized, so there
 * is no linear, packet or direct access. */
template <typename Scalar_, int SizeAtCompileTime, int MaxSizeAtCompileTime, typename StorageIndex_>
struct evaluator<ScaledPermutationMatrix<Scalar_, SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex_>>
    : evaluator_base<ScaledPermutationMatrix<Scalar_, SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex_>> {
  using XprType = ScaledPermutationMatrix<Scalar_, SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex_>;
  using Scalar = Scalar_;
  using CoeffReturnType = Scalar;

  static constexpr int CoeffReadCost =
      int(NumTraits<StorageIndex_>::ReadCost) + int(NumTraits<Scalar>::ReadCost) + int(NumTraits<Scalar>::AddCost);
  static constexpr unsigned int Flags = 0;
  static constexpr int Alignment = 0;

  EIGEN_DEVICE_FUNC explicit evaluator(const XprType& xpr) : m_indices(xpr.indices()), m_scales(xpr.scales()) {}

  EIGEN_DEVICE_FUNC Scalar coeff(Index row, Index col) const {
    return Index(m_indices.coeff(col)) == row ? m_scales.coeff(col) : Scalar(0);
  }

  // Linear access is requested only for vector-shaped operands (inner products), i.e. a 1x1 matrix.
  EIGEN_DEVICE_FUNC Scalar coeff(Index index) const {
    eigen_assert(index == 0);
    return m_scales.coeff(index);
  }

 protected:
  evaluator<typename XprType::IndicesType> m_indices;
  evaluator<typename XprType::ScalesType> m_scales;
};

struct ScaledPermutation2Dense {};

template <>
struct AssignmentKind<DenseShape, ScaledPermutationShape> {
  using Kind = ScaledPermutation2Dense;
};

// Scaled permutation to dense assignment: zero fill plus one scatter of n coefficients.
template <typename DstXprType, typename SrcXprType, typename Functor>
struct Assignment<DstXprType, SrcXprType, Functor, ScaledPermutation2Dense> {
  static EIGEN_DEVICE_FUNC void run(
      DstXprType& dst, const SrcXprType& src,
      const internal::assign_op<typename DstXprType::Scalar, typename SrcXprType::Scalar>&) {
    if (dst.rows() != src.rows() || dst.cols() != src.cols()) dst.resize(src.rows(), src.cols());
    dst.setZero();
    for (Index k = 0; k < src.rows(); ++k) dst.coeffRef(src.indices().coeff(k), k) = src.scales().coeff(k);
  }

  static EIGEN_DEVICE_FUNC void run(
      DstXprType& dst, const SrcXprType& src,
      const internal::add_assign_op<typename DstXprType::Scalar, typename SrcXprType::Scalar>&) {
    eigen_assert(dst.rows() == src.rows() && dst.cols() == src.cols());
    for (Index k = 0; k < src.rows(); ++k) dst.coeffRef(src.indices().coeff(k), k) += src.scales().coeff(k);
  }

  static EIGEN_DEVICE_FUNC void run(
      DstXprType& dst, const SrcXprType& src,
      const internal::sub_assign_op<typename DstXprType::Scalar, typename SrcXprType::Scalar>&) {
    eigen_assert(dst.rows() == src.rows() && dst.cols() == src.cols());
    for (Index k = 0; k < src.rows(); ++k) dst.coeffRef(src.indices().coeff(k), k) -= src.scales().coeff(k);
  }
};

/***************************************************************************
 * Products with a dense matrix: rows (S * X) or columns (X * S) are scaled and permuted, O(n m).
 * evalTo reuses the dense permutation kernel, which handles X aliasing dst, and scales in place afterwards;
 * the accumulating form works on the nested (evaluated once) operand.
 * scaleAndAddTo evaluates dst += alpha * (lhs * rhs) with alpha a left factor of the product. Scalar
 * multiplication need not commute, so both kernels keep that operand order: S * X regroups it as
 * (alpha * v_c) * X(c, j) by associativity; X * S keeps alpha * (X(i, tau(c)) * v_c), as folding alpha into
 * v_c would move it past the dense coefficient.
 ***************************************************************************/

template <typename Lhs, typename Rhs, int ProductTag>
struct generic_product_impl<Lhs, Rhs, ScaledPermutationShape, DenseShape, ProductTag>
    : generic_product_impl_base<Lhs, Rhs,
                                generic_product_impl<Lhs, Rhs, ScaledPermutationShape, DenseShape, ProductTag>> {
  using Scalar = typename Product<Lhs, Rhs>::Scalar;

  template <typename Dest>
  static EIGEN_DEVICE_FUNC void evalTo(Dest& dst, const Lhs& lhs, const Rhs& rhs) {
    permutation_matrix_product<Rhs, OnTheLeft, false, DenseShape>::run(dst, lhs.permutation(), rhs);
    for (Index k = 0; k < lhs.rows(); ++k) {
      auto row = dst.row(lhs.indices().coeff(k));
      row = lhs.scales().coeff(k) * row;
    }
  }

  template <typename Dest>
  static EIGEN_DEVICE_FUNC void scaleAndAddTo(Dest& dst, const Lhs& lhs, const Rhs& rhs, const Scalar& alpha) {
    typename nested_eval<Rhs, 1>::type rhsNested(rhs);
    for (Index k = 0; k < lhs.rows(); ++k)
      dst.row(lhs.indices().coeff(k)) += (alpha * lhs.scales().coeff(k)) * rhsNested.row(k);
  }
};

template <typename Lhs, typename Rhs, int ProductTag>
struct generic_product_impl<Lhs, Rhs, DenseShape, ScaledPermutationShape, ProductTag>
    : generic_product_impl_base<Lhs, Rhs,
                                generic_product_impl<Lhs, Rhs, DenseShape, ScaledPermutationShape, ProductTag>> {
  using Scalar = typename Product<Lhs, Rhs>::Scalar;

  template <typename Dest>
  static EIGEN_DEVICE_FUNC void evalTo(Dest& dst, const Lhs& lhs, const Rhs& rhs) {
    permutation_matrix_product<Lhs, OnTheRight, false, DenseShape>::run(dst, rhs.permutation(), lhs);
    for (Index k = 0; k < rhs.rows(); ++k) dst.col(k) *= rhs.scales().coeff(k);
  }

  template <typename Dest>
  static EIGEN_DEVICE_FUNC void scaleAndAddTo(Dest& dst, const Lhs& lhs, const Rhs& rhs, const Scalar& alpha) {
    typename nested_eval<Lhs, 1>::type lhsNested(lhs);
    for (Index k = 0; k < rhs.rows(); ++k)
      dst.col(k) += alpha * (lhsNested.col(rhs.indices().coeff(k)) * rhs.scales().coeff(k));
  }
};

// A triangular or self-adjoint operand is evaluated into a plain matrix first, as toDenseMatrix() does.
template <typename Lhs, typename Rhs, int ProductTag, bool ViewOnTheRight>
struct scaled_permutation_view_product_impl;

template <typename Lhs, typename Rhs, int ProductTag>
struct scaled_permutation_view_product_impl<Lhs, Rhs, ProductTag, true>
    : generic_product_impl_base<Lhs, Rhs, scaled_permutation_view_product_impl<Lhs, Rhs, ProductTag, true>> {
  using Scalar = typename Product<Lhs, Rhs>::Scalar;
  using DenseType = typename Rhs::DenseMatrixType;
  using DenseImpl = generic_product_impl<Lhs, DenseType, ScaledPermutationShape, DenseShape, ProductTag>;

  template <typename Dest>
  static EIGEN_DEVICE_FUNC void evalTo(Dest& dst, const Lhs& lhs, const Rhs& rhs) {
    const DenseType dense(rhs);
    DenseImpl::evalTo(dst, lhs, dense);
  }
  template <typename Dest>
  static EIGEN_DEVICE_FUNC void scaleAndAddTo(Dest& dst, const Lhs& lhs, const Rhs& rhs, const Scalar& alpha) {
    const DenseType dense(rhs);
    DenseImpl::scaleAndAddTo(dst, lhs, dense, alpha);
  }
};

template <typename Lhs, typename Rhs, int ProductTag>
struct scaled_permutation_view_product_impl<Lhs, Rhs, ProductTag, false>
    : generic_product_impl_base<Lhs, Rhs, scaled_permutation_view_product_impl<Lhs, Rhs, ProductTag, false>> {
  using Scalar = typename Product<Lhs, Rhs>::Scalar;
  using DenseType = typename Lhs::DenseMatrixType;
  using DenseImpl = generic_product_impl<DenseType, Rhs, DenseShape, ScaledPermutationShape, ProductTag>;

  template <typename Dest>
  static EIGEN_DEVICE_FUNC void evalTo(Dest& dst, const Lhs& lhs, const Rhs& rhs) {
    const DenseType dense(lhs);
    DenseImpl::evalTo(dst, dense, rhs);
  }
  template <typename Dest>
  static EIGEN_DEVICE_FUNC void scaleAndAddTo(Dest& dst, const Lhs& lhs, const Rhs& rhs, const Scalar& alpha) {
    const DenseType dense(lhs);
    DenseImpl::scaleAndAddTo(dst, dense, rhs, alpha);
  }
};

template <typename Lhs, typename Rhs, int ProductTag>
struct generic_product_impl<Lhs, Rhs, ScaledPermutationShape, TriangularShape, ProductTag>
    : scaled_permutation_view_product_impl<Lhs, Rhs, ProductTag, true> {};
template <typename Lhs, typename Rhs, int ProductTag>
struct generic_product_impl<Lhs, Rhs, ScaledPermutationShape, SelfAdjointShape, ProductTag>
    : scaled_permutation_view_product_impl<Lhs, Rhs, ProductTag, true> {};
template <typename Lhs, typename Rhs, int ProductTag>
struct generic_product_impl<Lhs, Rhs, TriangularShape, ScaledPermutationShape, ProductTag>
    : scaled_permutation_view_product_impl<Lhs, Rhs, ProductTag, false> {};
template <typename Lhs, typename Rhs, int ProductTag>
struct generic_product_impl<Lhs, Rhs, SelfAdjointShape, ScaledPermutationShape, ProductTag>
    : scaled_permutation_view_product_impl<Lhs, Rhs, ProductTag, false> {};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_SCALEDPERMUTATIONMATRIX_H
