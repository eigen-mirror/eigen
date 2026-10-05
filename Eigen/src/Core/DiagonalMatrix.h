// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2007-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_DIAGONALMATRIX_H
#define EIGEN_DIAGONALMATRIX_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \class DiagonalBase
 * \ingroup Core_Module
 *
 * \brief Base class for diagonal matrices and expressions
 *
 * This is the base class that is inherited by diagonal matrix and related expression
 * types, which internally use a vector for storing the diagonal entries. Diagonal
 * types always represent square matrices.
 *
 * \tparam Derived is the derived type, a DiagonalMatrix or DiagonalWrapper.
 *
 * \sa class DiagonalMatrix, class DiagonalWrapper
 */
template <typename Derived>
class DiagonalBase : public EigenBase<Derived> {
 public:
  using DiagonalVectorType = typename internal::traits<Derived>::DiagonalVectorType;
  using Scalar = typename DiagonalVectorType::Scalar;
  using RealScalar = typename DiagonalVectorType::RealScalar;
  using StorageKind = typename internal::traits<Derived>::StorageKind;
  using StorageIndex = typename internal::traits<Derived>::StorageIndex;

  enum {
    RowsAtCompileTime = DiagonalVectorType::SizeAtCompileTime,
    ColsAtCompileTime = DiagonalVectorType::SizeAtCompileTime,
    MaxRowsAtCompileTime = DiagonalVectorType::MaxSizeAtCompileTime,
    MaxColsAtCompileTime = DiagonalVectorType::MaxSizeAtCompileTime,
    SizeAtCompileTime = internal::size_at_compile_time(RowsAtCompileTime, ColsAtCompileTime),
    MaxSizeAtCompileTime = internal::size_at_compile_time(MaxRowsAtCompileTime, MaxColsAtCompileTime),
    IsVectorAtCompileTime = 0,
    Flags = NoPreferredStorageOrderBit
  };

  using DenseMatrixType =
      Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, 0, MaxRowsAtCompileTime, MaxColsAtCompileTime>;
  using DenseType = DenseMatrixType;
  using PlainObject =
      DiagonalMatrix<Scalar, DiagonalVectorType::SizeAtCompileTime, DiagonalVectorType::MaxSizeAtCompileTime>;

  /** \returns a const reference to the derived object. */
  EIGEN_DEVICE_FUNC inline const Derived& derived() const { return *static_cast<const Derived*>(this); }
  /** \returns a reference to the derived object. */
  EIGEN_DEVICE_FUNC inline Derived& derived() { return *static_cast<Derived*>(this); }

  /**
   * Constructs a dense matrix from \c *this. Note, this directly returns a dense matrix type,
   * not an expression.
   * \returns A dense matrix, with its diagonal entries set from the derived object. */
  EIGEN_DEVICE_FUNC DenseMatrixType toDenseMatrix() const { return derived(); }

  /** \returns a const reference to the derived object's vector of diagonal coefficients. */
  EIGEN_DEVICE_FUNC inline const DiagonalVectorType& diagonal() const { return derived().diagonal(); }
  /** \returns a reference to the derived object's vector of diagonal coefficients. */
  EIGEN_DEVICE_FUNC inline DiagonalVectorType& diagonal() { return derived().diagonal(); }

  /** \returns the value of the coefficient as if \c *this was a dense matrix. */
  EIGEN_DEVICE_FUNC inline Scalar coeff(Index row, Index col) const {
    eigen_assert(row >= 0 && col >= 0 && row < rows() && col < cols());
    return row == col ? diagonal().coeff(row) : Scalar(0);
  }

  /** \returns the number of rows. */
  EIGEN_DEVICE_FUNC constexpr Index rows() const { return diagonal().size(); }
  /** \returns the number of columns. */
  EIGEN_DEVICE_FUNC constexpr Index cols() const { return diagonal().size(); }

  /** \returns the diagonal matrix product of \c *this by the dense matrix, \a matrix */
  template <typename MatrixDerived>
  EIGEN_DEVICE_FUNC const Product<Derived, MatrixDerived, LazyProduct> operator*(
      const MatrixBase<MatrixDerived>& matrix) const {
    return Product<Derived, MatrixDerived, LazyProduct>(derived(), matrix.derived());
  }

  template <typename OtherDerived>
  using DiagonalProductReturnType = DiagonalWrapper<const EIGEN_CWISE_BINARY_RETURN_TYPE(
      DiagonalVectorType, typename OtherDerived::DiagonalVectorType, internal::scalar_product_op)>;

  /** \returns the diagonal matrix product of \c *this by the diagonal matrix \a other */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC const DiagonalProductReturnType<OtherDerived> operator*(
      const DiagonalBase<OtherDerived>& other) const {
    return diagonal().cwiseProduct(other.diagonal()).asDiagonal();
  }

  using DiagonalInverseReturnType =
      DiagonalWrapper<const CwiseUnaryOp<internal::scalar_inverse_op<Scalar>, const DiagonalVectorType>>;

  /** \returns the inverse of \c *this. Computed as the coefficient-wise inverse of the diagonal. */
  EIGEN_DEVICE_FUNC inline const DiagonalInverseReturnType inverse() const {
    return diagonal().cwiseInverse().asDiagonal();
  }

  using TransposeReturnType = DiagonalWrapper<const DiagonalVectorType>;

  /** \returns the transpose of \c *this, a diagonal matrix with the same diagonal.
   *
   * \sa conjugate(), adjoint() */
  EIGEN_DEVICE_FUNC const TransposeReturnType transpose() const { return diagonal().asDiagonal(); }

  using ConjugateReturnType = std::conditional_t<
      NumTraits<Scalar>::IsComplex,
      DiagonalWrapper<const CwiseUnaryOp<internal::scalar_conjugate_op<Scalar>, const DiagonalVectorType>>,
      TransposeReturnType>;

  /** \returns the complex conjugate of \c *this; for real scalars, a diagonal matrix with the same diagonal.
   *
   * \sa transpose(), adjoint() */
  EIGEN_DEVICE_FUNC const ConjugateReturnType conjugate() const { return diagonal().conjugate().asDiagonal(); }

  using AdjointReturnType = ConjugateReturnType;

  /** \returns the adjoint (conjugate transpose) of \c *this. A diagonal matrix equals its transpose, so this is
   * conjugate().
   *
   * \sa transpose(), conjugate() */
  EIGEN_DEVICE_FUNC const AdjointReturnType adjoint() const { return conjugate(); }

  using DiagonalScaleReturnType = DiagonalWrapper<const EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(
      DiagonalVectorType, Scalar, internal::scalar_product_op)>;

  /** \returns the product of \c *this by the scalar \a scalar */
  EIGEN_DEVICE_FUNC inline const DiagonalScaleReturnType operator*(const Scalar& scalar) const {
    return (diagonal() * scalar).asDiagonal();
  }

  using ScaleDiagonalReturnType = DiagonalWrapper<const EIGEN_SCALAR_BINARYOP_EXPR_RETURN_TYPE(
      Scalar, DiagonalVectorType, internal::scalar_product_op)>;

  /** \returns the product of a scalar and the diagonal matrix \a other */
  EIGEN_DEVICE_FUNC friend inline const ScaleDiagonalReturnType operator*(const Scalar& scalar,
                                                                          const DiagonalBase& other) {
    return (scalar * other.diagonal()).asDiagonal();
  }

  template <typename OtherDerived>
  using DiagonalSumReturnType = DiagonalWrapper<const EIGEN_CWISE_BINARY_RETURN_TYPE(
      DiagonalVectorType, typename OtherDerived::DiagonalVectorType, internal::scalar_sum_op)>;

  /** \returns the sum of \c *this and the diagonal matrix \a other */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC inline const DiagonalSumReturnType<OtherDerived> operator+(
      const DiagonalBase<OtherDerived>& other) const {
    return (diagonal() + other.diagonal()).asDiagonal();
  }

  template <typename OtherDerived>
  using DiagonalDifferenceReturnType = DiagonalWrapper<const EIGEN_CWISE_BINARY_RETURN_TYPE(
      DiagonalVectorType, typename OtherDerived::DiagonalVectorType, internal::scalar_difference_op)>;

  /** \returns the difference of \c *this and the diagonal matrix \a other */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC inline const DiagonalDifferenceReturnType<OtherDerived> operator-(
      const DiagonalBase<OtherDerived>& other) const {
    return (diagonal() - other.diagonal()).asDiagonal();
  }

  // Sums with a dense matrix are lazy: the diagonal is read through its index-based evaluator and no dense
  // copy of it is formed, so `A + D` composes with the enclosing expression like `A + B` does.

  /** \returns the lazy sum of the dense matrix \a lhs and the diagonal matrix \a rhs */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC friend const EIGEN_CWISE_BINARY_RETURN_TYPE(OtherDerived, Derived, internal::scalar_sum_op)
  operator+(const MatrixBase<OtherDerived>& lhs, const DiagonalBase & rhs) {
    return EIGEN_CWISE_BINARY_RETURN_TYPE(OtherDerived, Derived, internal::scalar_sum_op)(lhs.derived(), rhs.derived());
  }

  /** \returns the lazy sum of the diagonal matrix \a lhs and the dense matrix \a rhs */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC friend const EIGEN_CWISE_BINARY_RETURN_TYPE(Derived, OtherDerived, internal::scalar_sum_op)
  operator+(const DiagonalBase & lhs, const MatrixBase<OtherDerived>& rhs) {
    return EIGEN_CWISE_BINARY_RETURN_TYPE(Derived, OtherDerived, internal::scalar_sum_op)(lhs.derived(), rhs.derived());
  }

  /** \returns the lazy difference of the dense matrix \a lhs and the diagonal matrix \a rhs */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC friend const EIGEN_CWISE_BINARY_RETURN_TYPE(OtherDerived, Derived, internal::scalar_difference_op)
  operator-(const MatrixBase<OtherDerived>& lhs, const DiagonalBase & rhs) {
    return EIGEN_CWISE_BINARY_RETURN_TYPE(OtherDerived, Derived, internal::scalar_difference_op)(lhs.derived(),
                                                                                                 rhs.derived());
  }

  /** \returns the lazy difference of the diagonal matrix \a lhs and the dense matrix \a rhs */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC friend const EIGEN_CWISE_BINARY_RETURN_TYPE(Derived, OtherDerived, internal::scalar_difference_op)
  operator-(const DiagonalBase & lhs, const MatrixBase<OtherDerived>& rhs) {
    return EIGEN_CWISE_BINARY_RETURN_TYPE(Derived, OtherDerived, internal::scalar_difference_op)(lhs.derived(),
                                                                                                 rhs.derived());
  }
};

/** \class DiagonalMatrix
 * \ingroup Core_Module
 *
 * \brief Represents a diagonal matrix with its storage
 *
 * \tparam Scalar_ the type of coefficients
 * \tparam SizeAtCompileTime the dimension of the matrix, or Dynamic
 * \tparam MaxSizeAtCompileTime the dimension of the matrix, or Dynamic. This parameter is optional and defaults
 *        to SizeAtCompileTime. Most of the time, you do not need to specify it.
 *
 * \sa class DiagonalBase, class DiagonalWrapper
 */

namespace internal {
template <typename Scalar_, int SizeAtCompileTime, int MaxSizeAtCompileTime>
struct traits<DiagonalMatrix<Scalar_, SizeAtCompileTime, MaxSizeAtCompileTime>>
    : traits<Matrix<Scalar_, SizeAtCompileTime, SizeAtCompileTime, 0, MaxSizeAtCompileTime, MaxSizeAtCompileTime>> {
  using DiagonalVectorType = Matrix<Scalar_, SizeAtCompileTime, 1, 0, MaxSizeAtCompileTime, 1>;
  using StorageKind = DiagonalShape;
  enum { Flags = LvalueBit | NoPreferredStorageOrderBit | NestByRefBit };
};
}  // namespace internal
template <typename Scalar_, int SizeAtCompileTime, int MaxSizeAtCompileTime>
class DiagonalMatrix : public DiagonalBase<DiagonalMatrix<Scalar_, SizeAtCompileTime, MaxSizeAtCompileTime>> {
 public:
#ifndef EIGEN_PARSED_BY_DOXYGEN
  using DiagonalVectorType = typename internal::traits<DiagonalMatrix>::DiagonalVectorType;
  using Nested = const DiagonalMatrix&;
  using Scalar = Scalar_;
  using StorageKind = typename internal::traits<DiagonalMatrix>::StorageKind;
  using StorageIndex = typename internal::traits<DiagonalMatrix>::StorageIndex;
#endif

 protected:
  DiagonalVectorType m_diagonal;

 public:
  /** const version of diagonal(). */
  EIGEN_DEVICE_FUNC constexpr inline const DiagonalVectorType& diagonal() const { return m_diagonal; }
  /** \returns a reference to the stored vector of diagonal coefficients. */
  EIGEN_DEVICE_FUNC constexpr inline DiagonalVectorType& diagonal() { return m_diagonal; }

  /** Default constructor without initialization */
  EIGEN_DEVICE_FUNC constexpr inline DiagonalMatrix() {}

  /** Constructs a diagonal matrix with given dimension  */
  EIGEN_DEVICE_FUNC constexpr explicit inline DiagonalMatrix(Index dim) : m_diagonal(dim) {}

  /** 2D constructor. */
  EIGEN_DEVICE_FUNC constexpr inline DiagonalMatrix(const Scalar& x, const Scalar& y) : m_diagonal(x, y) {}

  /** 3D constructor. */
  EIGEN_DEVICE_FUNC constexpr inline DiagonalMatrix(const Scalar& x, const Scalar& y, const Scalar& z)
      : m_diagonal(x, y, z) {}

  /** \brief Construct a diagonal matrix with fixed size from an arbitrary number of coefficients.
   *
   * \warning To construct a diagonal matrix of fixed size, the number of values passed to this
   * constructor must match the fixed dimension of \c *this.
   *
   * \sa DiagonalMatrix(const Scalar&, const Scalar&)
   * \sa DiagonalMatrix(const Scalar&, const Scalar&, const Scalar&)
   */
  template <typename... ArgTypes>
  EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE DiagonalMatrix(const Scalar& a0, const Scalar& a1, const Scalar& a2,
                                                                 const ArgTypes&... args)
      : m_diagonal(a0, a1, a2, args...) {}

  /** \brief Constructs a DiagonalMatrix and initializes it by elements given by an initializer list of initializer
   * lists
   */
  EIGEN_DEVICE_FUNC explicit EIGEN_STRONG_INLINE DiagonalMatrix(
      const std::initializer_list<std::initializer_list<Scalar>>& list)
      : m_diagonal(list) {}

  /** \brief Constructs a DiagonalMatrix from an r-value diagonal vector type */
  EIGEN_DEVICE_FUNC constexpr explicit inline DiagonalMatrix(DiagonalVectorType&& diag) : m_diagonal(std::move(diag)) {}

  /** Copy constructor. */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC constexpr inline DiagonalMatrix(const DiagonalBase<OtherDerived>& other)
      : m_diagonal(other.diagonal()) {}

#ifndef EIGEN_PARSED_BY_DOXYGEN
  /** copy constructor. prevent a default copy constructor from hiding the other templated constructor */
  inline DiagonalMatrix(const DiagonalMatrix& other) : m_diagonal(other.diagonal()) {}
#endif

  /** Move constructor. Moves the stored diagonal vector. */
  EIGEN_DEVICE_FUNC constexpr DiagonalMatrix(DiagonalMatrix&&) = default;

  /** generic constructor from expression of the diagonal coefficients */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC constexpr explicit inline DiagonalMatrix(const MatrixBase<OtherDerived>& other)
      : m_diagonal(other) {}

  /** Copy operator. */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC DiagonalMatrix& operator=(const DiagonalBase<OtherDerived>& other) {
    m_diagonal = other.diagonal();
    return *this;
  }

#ifndef EIGEN_PARSED_BY_DOXYGEN
  /** This is a special case of the templated operator=. Its purpose is to
   * prevent a default operator= from hiding the templated operator=.
   */
  EIGEN_DEVICE_FUNC DiagonalMatrix& operator=(const DiagonalMatrix& other) {
    m_diagonal = other.diagonal();
    return *this;
  }
#endif

  /** Move assignment operator. Transfers dynamic storage and copies inline storage. */
  EIGEN_DEVICE_FUNC constexpr DiagonalMatrix& operator=(DiagonalMatrix&& other) noexcept(
      DiagonalVectorType::MaxSizeAtCompileTime == Dynamic &&
      std::is_nothrow_move_assignable<DiagonalVectorType>::value) {
    EIGEN_IF_CONSTEXPR (DiagonalVectorType::MaxSizeAtCompileTime == Dynamic) {
      m_diagonal = std::move(other.m_diagonal);
    } else {
      // Preserve the vectorized assignment path for inline storage.
      m_diagonal = other.m_diagonal;
    }
    return *this;
  }

  using InitializeReturnType =
      DiagonalWrapper<const CwiseNullaryOp<internal::scalar_constant_op<Scalar>, DiagonalVectorType>>;

  using ZeroInitializeReturnType =
      DiagonalWrapper<const CwiseNullaryOp<internal::scalar_zero_op<Scalar>, DiagonalVectorType>>;

  /** Initializes a diagonal matrix of size SizeAtCompileTime with coefficients set to zero */
  EIGEN_DEVICE_FUNC static const ZeroInitializeReturnType Zero() { return DiagonalVectorType::Zero().asDiagonal(); }
  /** Initializes a diagonal matrix of size dim with coefficients set to zero */
  EIGEN_DEVICE_FUNC static const ZeroInitializeReturnType Zero(Index size) {
    return DiagonalVectorType::Zero(size).asDiagonal();
  }
  /** Initializes an identity matrix of size SizeAtCompileTime */
  EIGEN_DEVICE_FUNC static const InitializeReturnType Identity() { return DiagonalVectorType::Ones().asDiagonal(); }
  /** Initializes an identity matrix of size dim */
  EIGEN_DEVICE_FUNC static const InitializeReturnType Identity(Index size) {
    return DiagonalVectorType::Ones(size).asDiagonal();
  }

  /** Resizes to given size. */
  EIGEN_DEVICE_FUNC inline void resize(Index size) { m_diagonal.resize(size); }
  /** Sets all coefficients to zero. */
  EIGEN_DEVICE_FUNC inline void setZero() { m_diagonal.setZero(); }
  /** Resizes and sets all coefficients to zero. */
  EIGEN_DEVICE_FUNC inline void setZero(Index size) { m_diagonal.setZero(size); }
  /** Sets this matrix to be the identity matrix of the current size. */
  EIGEN_DEVICE_FUNC inline void setIdentity() { m_diagonal.setOnes(); }
  /** Sets this matrix to be the identity matrix of the given size. */
  EIGEN_DEVICE_FUNC inline void setIdentity(Index size) { m_diagonal.setOnes(size); }
};

/** \class DiagonalWrapper
 * \ingroup Core_Module
 *
 * \brief Expression of a diagonal matrix
 *
 * \tparam DiagonalVectorType_ the type of the vector of diagonal coefficients
 *
 * This class is an expression of a diagonal matrix, but not storing its own vector of diagonal coefficients,
 * instead wrapping an existing vector expression. It is the return type of MatrixBase::asDiagonal()
 * and most of the time this is the only way that it is used.
 *
 * \sa class DiagonalMatrix, class DiagonalBase, MatrixBase::asDiagonal()
 */

namespace internal {
template <typename DiagonalVectorType_>
struct traits<DiagonalWrapper<DiagonalVectorType_>> {
  using DiagonalVectorType = DiagonalVectorType_;
  using Scalar = typename DiagonalVectorType::Scalar;
  using StorageIndex = typename DiagonalVectorType::StorageIndex;
  using StorageKind = DiagonalShape;
  using XprKind = typename traits<DiagonalVectorType>::XprKind;
  enum {
    RowsAtCompileTime = DiagonalVectorType::SizeAtCompileTime,
    ColsAtCompileTime = DiagonalVectorType::SizeAtCompileTime,
    MaxRowsAtCompileTime = DiagonalVectorType::MaxSizeAtCompileTime,
    MaxColsAtCompileTime = DiagonalVectorType::MaxSizeAtCompileTime,
    Flags = (traits<DiagonalVectorType>::Flags & LvalueBit) | NoPreferredStorageOrderBit
  };
};
}  // namespace internal

template <typename DiagonalVectorType_>
class DiagonalWrapper : public DiagonalBase<DiagonalWrapper<DiagonalVectorType_>>, internal::no_assignment_operator {
 public:
#ifndef EIGEN_PARSED_BY_DOXYGEN
  using DiagonalVectorType = DiagonalVectorType_;
  using Nested = DiagonalWrapper;
#endif

  /** Constructor from expression of diagonal coefficients to wrap. */
  EIGEN_DEVICE_FUNC constexpr explicit inline DiagonalWrapper(DiagonalVectorType& a_diagonal)
      : m_diagonal(a_diagonal) {}

  /** \returns a const reference to the wrapped expression of diagonal coefficients. */
  EIGEN_DEVICE_FUNC constexpr const DiagonalVectorType& diagonal() const { return m_diagonal; }

 protected:
  typename DiagonalVectorType::Nested m_diagonal;
};

/** \returns a pseudo-expression of a diagonal matrix with *this as vector of diagonal coefficients
 *
 * \only_for_vectors
 *
 * Example: \include MatrixBase_asDiagonal.cpp
 * Output: \verbinclude MatrixBase_asDiagonal.out
 *
 * \sa class DiagonalWrapper, class DiagonalMatrix, diagonal(), isDiagonal()
 **/
template <typename Derived>
EIGEN_DEVICE_FUNC constexpr const DiagonalWrapper<const Derived> MatrixBase<Derived>::asDiagonal() const {
  return DiagonalWrapper<const Derived>(derived());
}

/** \returns true if *this is approximately equal to a diagonal matrix,
 *          within the precision given by \a prec.
 *
 * Example: \include MatrixBase_isDiagonal.cpp
 * Output: \verbinclude MatrixBase_isDiagonal.out
 *
 * \sa asDiagonal()
 */
template <typename Derived>
bool MatrixBase<Derived>::isDiagonal(const RealScalar& prec) const {
  if (cols() != rows()) return false;
  RealScalar maxAbsOnDiagonal = static_cast<RealScalar>(-1);
  for (Index j = 0; j < cols(); ++j) {
    RealScalar absOnDiagonal = numext::abs(coeff(j, j));
    if (absOnDiagonal > maxAbsOnDiagonal) maxAbsOnDiagonal = absOnDiagonal;
  }
  for (Index j = 0; j < cols(); ++j)
    for (Index i = 0; i < j; ++i) {
      if (!internal::isMuchSmallerThan(coeff(i, j), maxAbsOnDiagonal, prec)) return false;
      if (!internal::isMuchSmallerThan(coeff(j, i), maxAbsOnDiagonal, prec)) return false;
    }
  return true;
}

/** \returns DiagonalWrapper.
 *
 * Example: \include MatrixBase_diagonalView.cpp
 * Output: \verbinclude MatrixBase_diagonalView.out
 *
 * \sa diagonalView()
 */

/** This is the non-const version of diagonalView() with DiagIndex_ . */
template <typename Derived>
template <int DiagIndex_>
EIGEN_DEVICE_FUNC constexpr DiagonalWrapper<Diagonal<Derived, DiagIndex_>> MatrixBase<Derived>::diagonalView() {
  using DiagType = Diagonal<Derived, DiagIndex_>;
  using ReturnType = DiagonalWrapper<DiagType>;
  DiagType diag(this->derived());
  return ReturnType(diag);
}

/** This is the const version of diagonalView() with DiagIndex_ . */
template <typename Derived>
template <int DiagIndex_>
EIGEN_DEVICE_FUNC constexpr DiagonalWrapper<Diagonal<const Derived, DiagIndex_>> MatrixBase<Derived>::diagonalView()
    const {
  using DiagType = Diagonal<const Derived, DiagIndex_>;
  using ReturnType = DiagonalWrapper<DiagType>;
  DiagType diag(this->derived());
  return ReturnType(diag);
}

/** This is the non-const version of diagonalView() with dynamic index. */
template <typename Derived>
EIGEN_DEVICE_FUNC constexpr DiagonalWrapper<Diagonal<Derived, DynamicIndex>> MatrixBase<Derived>::diagonalView(
    Index index) {
  using DiagType = Diagonal<Derived, DynamicIndex>;
  using ReturnType = DiagonalWrapper<DiagType>;
  DiagType diag(this->derived(), index);
  return ReturnType(diag);
}

/** This is the const version of diagonalView() with dynamic index. */
template <typename Derived>
EIGEN_DEVICE_FUNC constexpr DiagonalWrapper<Diagonal<const Derived, DynamicIndex>> MatrixBase<Derived>::diagonalView(
    Index index) const {
  using DiagType = Diagonal<const Derived, DynamicIndex>;
  using ReturnType = DiagonalWrapper<DiagType>;
  DiagType diag(this->derived(), index);
  return ReturnType(diag);
}

namespace internal {

template <>
struct storage_kind_to_shape<DiagonalShape> {
  using Shape = DiagonalShape;
};

/** \internal
 * Index-based evaluator of a diagonal matrix, for coefficient-wise expressions that mix it with a dense
 * operand. Off-diagonal coefficients are synthesized, so there is no linear, packet or direct access.
 * Products with a diagonal matrix have dedicated evaluators and do not use this one.
 */
template <typename XprType>
struct diagonal_matrix_evaluator : evaluator_base<XprType> {
  using DiagonalVectorType = typename XprType::DiagonalVectorType;
  using Scalar = typename XprType::Scalar;
  using CoeffReturnType = Scalar;

  static constexpr int CoeffReadCost =
      int(evaluator<DiagonalVectorType>::CoeffReadCost) + int(NumTraits<Scalar>::AddCost);
  static constexpr unsigned int Flags = 0;
  static constexpr int Alignment = 0;

  EIGEN_DEVICE_FUNC explicit diagonal_matrix_evaluator(const XprType& xpr) : m_diagonal(xpr.diagonal()) {
    EIGEN_INTERNAL_CHECK_COST_VALUE(CoeffReadCost);
  }

  EIGEN_DEVICE_FUNC Scalar coeff(Index row, Index col) const { return row == col ? m_diagonal.coeff(row) : Scalar(0); }

  // Linear access is requested only for vector-shaped operands (inner products), i.e. a 1x1 diagonal.
  EIGEN_DEVICE_FUNC Scalar coeff(Index index) const {
    eigen_assert(index == 0);
    return m_diagonal.coeff(index);
  }

 protected:
  evaluator<DiagonalVectorType> m_diagonal;
};

template <typename Scalar_, int SizeAtCompileTime, int MaxSizeAtCompileTime>
struct evaluator<DiagonalMatrix<Scalar_, SizeAtCompileTime, MaxSizeAtCompileTime>>
    : diagonal_matrix_evaluator<DiagonalMatrix<Scalar_, SizeAtCompileTime, MaxSizeAtCompileTime>> {
  using XprType = DiagonalMatrix<Scalar_, SizeAtCompileTime, MaxSizeAtCompileTime>;
  EIGEN_DEVICE_FUNC explicit evaluator(const XprType& xpr) : diagonal_matrix_evaluator<XprType>(xpr) {}
};

template <typename DiagonalVectorType_>
struct evaluator<DiagonalWrapper<DiagonalVectorType_>>
    : diagonal_matrix_evaluator<DiagonalWrapper<DiagonalVectorType_>> {
  using XprType = DiagonalWrapper<DiagonalVectorType_>;
  EIGEN_DEVICE_FUNC explicit evaluator(const XprType& xpr) : diagonal_matrix_evaluator<XprType>(xpr) {}
};

struct Diagonal2Dense {};

template <>
struct AssignmentKind<DenseShape, DiagonalShape> {
  using Kind = Diagonal2Dense;
};

// Diagonal matrix to Dense assignment
template <typename DstXprType, typename SrcXprType, typename Functor>
struct Assignment<DstXprType, SrcXprType, Functor, Diagonal2Dense> {
  static EIGEN_DEVICE_FUNC void run(
      DstXprType& dst, const SrcXprType& src,
      const internal::assign_op<typename DstXprType::Scalar, typename SrcXprType::Scalar>& /*func*/) {
    Index dstRows = src.rows();
    Index dstCols = src.cols();
    if ((dst.rows() != dstRows) || (dst.cols() != dstCols)) dst.resize(dstRows, dstCols);

    dst.setZero();
    dst.diagonal() = src.diagonal();
  }

  static EIGEN_DEVICE_FUNC void run(
      DstXprType& dst, const SrcXprType& src,
      const internal::add_assign_op<typename DstXprType::Scalar, typename SrcXprType::Scalar>& /*func*/) {
    dst.diagonal() += src.diagonal();
  }

  static EIGEN_DEVICE_FUNC void run(
      DstXprType& dst, const SrcXprType& src,
      const internal::sub_assign_op<typename DstXprType::Scalar, typename SrcXprType::Scalar>& /*func*/) {
    dst.diagonal() -= src.diagonal();
  }
};

/***************************************************************************
 * Dense ?= lhs +/- rhs, where one operand is dense and the other has one structural nonzero in each column of the
 * destination: the lazy CwiseBinaryOp keeps its type, but a direct assignment writes each column as two vectorized
 * segments of the dense operand around that nonzero, instead of evaluating the sum coefficient by coefficient
 * without packets.
 ***************************************************************************/

// The three functors such an assignment can carry, and the sign with which they add the source.
template <typename Functor>
struct additive_assign_sign : std::integral_constant<int, 0> {};
template <typename Scalar>
struct additive_assign_sign<assign_op<Scalar, Scalar>> : std::integral_constant<int, 1> {};
template <typename Scalar>
struct additive_assign_sign<add_assign_op<Scalar, Scalar>> : std::integral_constant<int, 1> {};
template <typename Scalar>
struct additive_assign_sign<sub_assign_op<Scalar, Scalar>> : std::integral_constant<int, -1> {};

template <typename Functor>
struct is_plain_assign : std::false_type {};
template <typename Scalar>
struct is_plain_assign<assign_op<Scalar, Scalar>> : std::true_type {};

template <typename BinaryOp>
struct is_additive_binary_op : std::false_type {};
template <typename Scalar>
struct is_additive_binary_op<scalar_sum_op<Scalar, Scalar>> : std::true_type {};
template <typename Scalar>
struct is_additive_binary_op<scalar_difference_op<Scalar, Scalar>> : std::true_type {};

template <typename T>
struct is_dense_shape : std::is_same<typename evaluator_traits<T>::Shape, DenseShape> {};
template <typename T>
struct is_diagonal_shape : std::is_same<typename evaluator_traits<T>::Shape, DiagonalShape> {};
template <typename T>
struct is_default_product : std::false_type {};
template <typename Lhs, typename Rhs>
struct is_default_product<Product<Lhs, Rhs, DefaultProduct>> : std::true_type {};
// Specialized in PermutationMatrix.h: a permutation's dense expression counts as the structured operand of a sum
// with a diagonal matrix, not as its dense one.
template <typename T>
struct is_permutation_dense_xpr : std::false_type {};

template <typename BinaryOp>
struct is_difference_op : std::false_type {};
template <typename Scalar>
struct is_difference_op<scalar_difference_op<Scalar, Scalar>> : std::true_type {};

// The block pass below also forms dst ?= +/-dense at the structured coefficients, then overwrites them. For signed
// integers that intermediate can overflow where the coefficient-wise dst ?= op(lhs, rhs) does not (-INT_MIN, or
// INT_MIN - 1 in INT_MIN - (1 - 1)), so they take the fast path only when the block pass is a copy.
template <typename Scalar, typename Functor, bool NegateDense>
struct dense_block_pass_cannot_overflow
    : bool_constant<!(NumTraits<Scalar>::IsInteger && std::numeric_limits<Scalar>::is_signed) ||
                    (is_plain_assign<Functor>::value && !NegateDense)> {};

// Mixed scalar types and functors other than =, += and -= stay on the generic coefficient-wise path.
template <typename Dst, typename DenseXpr, typename DiagonalXpr, typename Functor, bool NegateDense>
struct dense_diagonal_sum_fast_path
    : bool_constant<is_dense_shape<DenseXpr>::value && !is_permutation_dense_xpr<DenseXpr>::value &&
                    is_diagonal_shape<DiagonalXpr>::value && additive_assign_sign<Functor>::value != 0 &&
                    std::is_same<typename Dst::Scalar, typename DenseXpr::Scalar>::value &&
                    std::is_same<typename DenseXpr::Scalar, typename DiagonalXpr::Scalar>::value &&
                    dense_block_pass_cannot_overflow<typename Dst::Scalar, Functor, NegateDense>::value> {};

// block ?= +/-dense block as one of =, = -, += and -=, chosen at compile time: bool has no negation or subtraction,
// so no other operator may be instantiated.
template <bool Assign, int Sign>
struct dense_block_update;
template <>
struct dense_block_update<true, 1> {
  template <typename Block, typename DenseBlock>
  EIGEN_DEVICE_FUNC static void run(Block&& block, const DenseBlock& dense) {
    block = dense;
  }
};
template <>
struct dense_block_update<true, -1> {
  template <typename Block, typename DenseBlock>
  EIGEN_DEVICE_FUNC static void run(Block&& block, const DenseBlock& dense) {
    block = -dense;
  }
};
template <>
struct dense_block_update<false, 1> {
  template <typename Block, typename DenseBlock>
  EIGEN_DEVICE_FUNC static void run(Block&& block, const DenseBlock& dense) {
    block += dense;
  }
};
template <>
struct dense_block_update<false, -1> {
  template <typename Block, typename DenseBlock>
  EIGEN_DEVICE_FUNC static void run(Block&& block, const DenseBlock& dense) {
    block -= dense;
  }
};

// A diagonal as the structured operand: its nonzero in column k is at row k.
template <typename DiagonalXpr>
struct diagonal_column_nonzeros {
  using Scalar = typename DiagonalXpr::Scalar;
  EIGEN_DEVICE_FUNC explicit diagonal_column_nonzeros(const DiagonalXpr& diagonal) : m_diagonal(diagonal) {}
  EIGEN_DEVICE_FUNC Index row(Index k) const { return k; }
  EIGEN_DEVICE_FUNC Scalar value(Index k) const { return m_diagonal.coeff(k, k); }
  evaluator<DiagonalXpr> m_diagonal;
};

// Column-major views of a row-major destination and its dense operand; a structured operand is only paired with a
// destination along whose columns it has one nonzero each, so after transposition it needs no view of its own.
template <bool Transpose>
struct column_major_view {
  template <typename Xpr>
  EIGEN_DEVICE_FUNC static Xpr& run(Xpr& xpr) {
    return xpr;
  }
};
template <>
struct column_major_view<true> {
  template <typename Xpr>
  EIGEN_DEVICE_FUNC static auto run(Xpr& xpr) {
    return xpr.transpose();
  }
};

// dst ?= op(lhs, rhs) in blocks of columns. For each column k of a block, with the structured operand's nonzero at
// row r, v(k) = func(dst(r, k), op(lhs(r, k), rhs(r, k))) is formed from coefficients read before the block is
// written; then block ?= +/-dense block, and dst(r, k) = v(k). So each structured coefficient gets the
// coefficient-wise value and grouping, also when the diagonal or the dense operand is read from dst; structural
// zeros are not added.
template <bool DenseOnLeft>
struct dense_structured_sum_assignment {
  template <typename Dst, typename DenseXpr, typename Structured, typename BinaryOp, typename Functor>
  EIGEN_DEVICE_FUNC static void run(Dst& dst, const DenseXpr& dense, const Structured& structured, const BinaryOp& op,
                                    const Functor& func) {
    using Scalar = typename Dst::Scalar;
    constexpr bool kNegateDense = !DenseOnLeft && std::is_same<BinaryOp, scalar_difference_op<Scalar, Scalar>>::value;
    constexpr int kSign = additive_assign_sign<Functor>::value * (kNegateDense ? -1 : 1);
    constexpr Index kBlockColumns = 32;
    using Update = dense_block_update<is_plain_assign<Functor>::value, kSign>;
    const evaluator<DenseXpr> denseEval(dense);
    Scalar values[kBlockColumns];
    for (Index j = 0; j < dst.cols(); j += kBlockColumns) {
      const Index columns = numext::mini(Index(kBlockColumns), dst.cols() - j);
      for (Index k = 0; k < columns; ++k) {
        const Index r = structured.row(j + k);
        const Scalar s = structured.value(j + k);
        const Scalar d = denseEval.coeff(r, j + k);
        EIGEN_IF_CONSTEXPR (!is_plain_assign<Functor>::value) {
          values[k] = dst.coeff(r, j + k);
        }
        func.assignCoeff(values[k], DenseOnLeft ? op(d, s) : op(s, d));
      }
      Update::run(dst.middleCols(j, columns), dense.middleCols(j, columns));
      for (Index k = 0; k < columns; ++k) {
        dst.coeffRef(structured.row(j + k), j + k) = values[k];
      }
    }
  }

  template <typename Dst, typename SrcXprType, typename DenseXpr, typename Structured, typename Functor>
  EIGEN_DEVICE_FUNC static void assign(Dst& dst, const SrcXprType& src, const DenseXpr& denseXpr,
                                       const Structured& structured, const Functor& func) {
    // Evaluates a product operand once, before dst is resized or written.
    const typename nested_eval<DenseXpr, 1>::type dense(denseXpr);
    resize_if_allowed(dst, src, func);
    auto&& dstView = column_major_view<bool(Dst::IsRowMajor)>::run(dst);
    run(dstView, column_major_view<bool(Dst::IsRowMajor)>::run(dense), structured, src.functor(), func);
  }
};

template <typename DstXprType, typename BinaryOp, typename Lhs, typename Rhs, typename Functor>
struct Assignment<DstXprType, CwiseBinaryOp<BinaryOp, const Lhs, const Rhs>, Functor, Dense2Dense,
                  std::enable_if_t<is_additive_binary_op<BinaryOp>::value &&
                                   dense_diagonal_sum_fast_path<DstXprType, Lhs, Rhs, Functor, false>::value>> {
  using SrcXprType = CwiseBinaryOp<BinaryOp, const Lhs, const Rhs>;
  EIGEN_DEVICE_FUNC static void run(DstXprType& dst, const SrcXprType& src, const Functor& func) {
    dense_structured_sum_assignment<true>::assign(dst, src, src.lhs(), diagonal_column_nonzeros<Rhs>(src.rhs()), func);
  }
};

// A dense product on the right is left to the "xpr + product" rule above.
template <typename DstXprType, typename BinaryOp, typename Lhs, typename Rhs, typename Functor>
struct Assignment<DstXprType, CwiseBinaryOp<BinaryOp, const Lhs, const Rhs>, Functor, Dense2Dense,
                  std::enable_if_t<is_additive_binary_op<BinaryOp>::value &&
                                   dense_diagonal_sum_fast_path<DstXprType, Rhs, Lhs, Functor,
                                                                is_difference_op<BinaryOp>::value>::value &&
                                   !is_default_product<Rhs>::value>> {
  using SrcXprType = CwiseBinaryOp<BinaryOp, const Lhs, const Rhs>;
  EIGEN_DEVICE_FUNC static void run(DstXprType& dst, const SrcXprType& src, const Functor& func) {
    dense_structured_sum_assignment<false>::assign(dst, src, src.rhs(), diagonal_column_nonzeros<Lhs>(src.lhs()), func);
  }
};

}  // namespace internal

}  // end namespace Eigen

#endif  // EIGEN_DIAGONALMATRIX_H
