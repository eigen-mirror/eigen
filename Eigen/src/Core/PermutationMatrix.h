// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2009-2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_PERMUTATIONMATRIX_H
#define EIGEN_PERMUTATIONMATRIX_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

enum PermPermProduct_t { PermPermProduct };

/** \internal
 * Nullary functor reading a permutation as a dense matrix of Scalar: P has its ones at (indices(k), k),
 * and its inverse (Transposed) at (k, indices(k)). It backs the lazy sums of a dense or diagonal matrix
 * with a permutation, whose scalar type the permutation borrows.
 */
template <typename Scalar, typename IndicesType, bool Transposed>
struct permutation_dense_op {
  EIGEN_DEVICE_FUNC explicit permutation_dense_op(const IndicesType& indices) : m_indices(indices) {}

  template <typename IndexType>
  EIGEN_DEVICE_FUNC Scalar operator()(IndexType row, IndexType col) const {
    const Index k = Transposed ? Index(row) : Index(col);
    const Index image = Transposed ? Index(col) : Index(row);
    return Index(m_indices.coeff(k)) == image ? Scalar(1) : Scalar(0);
  }

  EIGEN_DEVICE_FUNC const remove_all_t<typename IndicesType::Nested>& indices() const { return m_indices; }

  typename IndicesType::Nested m_indices;
};

template <typename Scalar, typename IndicesType, bool Transposed>
struct functor_traits<permutation_dense_op<Scalar, IndicesType, Transposed>> {
  static constexpr int Cost = int(NumTraits<typename IndicesType::Scalar>::ReadCost) + int(NumTraits<Scalar>::AddCost);
  static constexpr bool PacketAccess = false;
  static constexpr bool IsRepeatable = true;
};

template <typename Scalar, bool Transposed, typename PermutationType>
struct permutation_dense_expression {
  using IndicesType = remove_all_t<typename PermutationType::IndicesType>;
  using PlainObject = Matrix<Scalar, PermutationType::RowsAtCompileTime, PermutationType::ColsAtCompileTime, 0,
                             PermutationType::MaxRowsAtCompileTime, PermutationType::MaxColsAtCompileTime>;
  using type = CwiseNullaryOp<permutation_dense_op<Scalar, IndicesType, Transposed>, PlainObject>;

  static EIGEN_DEVICE_FUNC type run(const PermutationType& permutation) {
    return type(permutation.rows(), permutation.cols(),
                permutation_dense_op<Scalar, IndicesType, Transposed>(permutation.indices()));
  }
};

}  // end namespace internal

/** \class PermutationBase
 * \ingroup Core_Module
 *
 * \brief Base class for permutations
 *
 * \tparam Derived the derived class
 *
 * This class is the base class for all expressions representing a permutation matrix,
 * internally stored as a vector of integers.
 * The convention followed here is that if \f$ \sigma \f$ is a permutation, the corresponding permutation matrix
 * \f$ P_\sigma \f$ is such that if \f$ (e_1,\ldots,e_p) \f$ is the canonical basis, we have:
 *  \f[ P_\sigma(e_i) = e_{\sigma(i)}. \f]
 * This convention ensures that for any two permutations \f$ \sigma, \tau \f$, we have:
 *  \f[ P_{\sigma\circ\tau} = P_\sigma P_\tau. \f]
 *
 * Permutation matrices are square and invertible.
 *
 * Notice that in addition to the member functions and operators listed here, there also are non-member
 * operator* to multiply any kind of permutation object with any kind of matrix expression (MatrixBase)
 * on either side, and with a diagonal matrix (DiagonalBase) on either side, which yields a
 * ScaledPermutationMatrix.
 *
 * \sa class PermutationMatrix, class PermutationWrapper
 */
template <typename Derived>
class PermutationBase : public EigenBase<Derived> {
  using Traits = internal::traits<Derived>;
  using Base = EigenBase<Derived>;

 public:
#ifndef EIGEN_PARSED_BY_DOXYGEN
  using IndicesType = typename Traits::IndicesType;
  enum {
    Flags = Traits::Flags,
    RowsAtCompileTime = Traits::RowsAtCompileTime,
    ColsAtCompileTime = Traits::ColsAtCompileTime,
    MaxRowsAtCompileTime = Traits::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = Traits::MaxColsAtCompileTime
  };
  using StorageIndex = typename Traits::StorageIndex;
  using DenseMatrixType =
      Matrix<StorageIndex, RowsAtCompileTime, ColsAtCompileTime, 0, MaxRowsAtCompileTime, MaxColsAtCompileTime>;
  using PlainPermutationType =
      PermutationMatrix<IndicesType::SizeAtCompileTime, IndicesType::MaxSizeAtCompileTime, StorageIndex>;
  using PlainObject = PlainPermutationType;
  using Base::derived;
  using InverseReturnType = Inverse<Derived>;
  using Scalar = void;
#endif

  /** Copies the other permutation into *this */
  template <typename OtherDerived>
  Derived& operator=(const PermutationBase<OtherDerived>& other) {
    indices() = other.indices();
    return derived();
  }

  /** Assignment from the Transpositions \a tr */
  template <typename OtherDerived>
  Derived& operator=(const TranspositionsBase<OtherDerived>& tr) {
    setIdentity(tr.size());
    for (Index k = size() - 1; k >= 0; --k) applyTranspositionOnTheRight(k, tr.coeff(k));
    return derived();
  }

  /** \returns the number of rows */
  inline EIGEN_DEVICE_FUNC Index rows() const { return Index(indices().size()); }

  /** \returns the number of columns */
  inline EIGEN_DEVICE_FUNC Index cols() const { return Index(indices().size()); }

  /** \returns the size of a side of the respective square matrix, i.e., the number of indices */
  inline EIGEN_DEVICE_FUNC Index size() const { return Index(indices().size()); }

#ifndef EIGEN_PARSED_BY_DOXYGEN
  template <typename DenseDerived>
  void evalTo(MatrixBase<DenseDerived>& other) const {
    other.setZero();
    for (Index i = 0; i < rows(); ++i) other.coeffRef(indices().coeff(i), i) = typename DenseDerived::Scalar(1);
  }
#endif

  /** \returns a Matrix object initialized from this permutation matrix. Notice that it
   * is inefficient to return this Matrix object by value. For efficiency, favor using
   * the Matrix constructor taking EigenBase objects.
   */
  DenseMatrixType toDenseMatrix() const { return derived(); }

  /** \returns the plain matrix representation of the permutation. */
  DenseMatrixType eval() const { return toDenseMatrix(); }

  /** const version of indices(). */
  EIGEN_DEVICE_FUNC constexpr const IndicesType& indices() const { return derived().indices(); }
  /** \returns a reference to the stored array representing the permutation. */
  EIGEN_DEVICE_FUNC constexpr IndicesType& indices() { return derived().indices(); }

  /** Resizes to given size.
   */
  EIGEN_DEVICE_FUNC void resize(Index newSize) { indices().resize(newSize); }

  /** Sets *this to be the identity permutation matrix */
  EIGEN_DEVICE_FUNC void setIdentity() {
    StorageIndex n = StorageIndex(size());
    for (StorageIndex i = 0; i < n; ++i) indices().coeffRef(i) = i;
  }

  /** Sets *this to be the identity permutation matrix of given size.
   */
  EIGEN_DEVICE_FUNC void setIdentity(Index newSize) {
    resize(newSize);
    setIdentity();
  }

  /** Multiplies *this by the transposition \f$(ij)\f$ on the left.
   *
   * \returns a reference to *this.
   *
   * \warning This is much slower than applyTranspositionOnTheRight(Index,Index):
   * this has linear complexity.
   *
   * \sa applyTranspositionOnTheRight(Index,Index)
   */
  Derived& applyTranspositionOnTheLeft(Index i, Index j) {
    eigen_assert(i >= 0 && j >= 0 && i < size() && j < size());
    if (i == j) return derived();
    EIGEN_IF_CONSTEXPR ((internal::evaluator<IndicesType>::Flags & PacketAccessBit) &&
                        internal::packet_traits<StorageIndex>::HasCmp) {
      // Amortize packet setup over at least two packets.
      if (size() >= 2 * internal::packet_traits<StorageIndex>::size) {
        const StorageIndex first = StorageIndex(i), second = StorageIndex(j);
        indices() =
            indices().cwiseTypedEqual(first).select(second, indices().cwiseTypedEqual(second).select(first, indices()));
        return derived();
      }
    }
    for (Index k = 0; k < size(); ++k) {
      if (indices().coeff(k) == i)
        indices().coeffRef(k) = StorageIndex(j);
      else if (indices().coeff(k) == j)
        indices().coeffRef(k) = StorageIndex(i);
    }
    return derived();
  }

  /** Multiplies *this by the transposition \f$(ij)\f$ on the right.
   *
   * \returns a reference to *this.
   *
   * This is a fast operation, it only consists in swapping two indices.
   *
   * \sa applyTranspositionOnTheLeft(Index,Index)
   */
  Derived& applyTranspositionOnTheRight(Index i, Index j) {
    eigen_assert(i >= 0 && j >= 0 && i < size() && j < size());
    std::swap(indices().coeffRef(i), indices().coeffRef(j));
    return derived();
  }

  /** \returns the inverse permutation matrix.
   *
   * \note \blank \note_try_to_help_rvo
   */
  inline InverseReturnType inverse() const { return InverseReturnType(derived()); }
  /** \returns the transpose permutation matrix.
   *
   * \note \blank \note_try_to_help_rvo
   */
  inline InverseReturnType transpose() const { return InverseReturnType(derived()); }
  /** \returns the adjoint of the permutation matrix. Its entries are real and it is orthogonal, so this equals
   * transpose() and inverse().
   *
   * \note \blank \note_try_to_help_rvo
   */
  InverseReturnType adjoint() const { return InverseReturnType(derived()); }

  /**** multiplication helpers to hopefully get RVO ****/

#ifndef EIGEN_PARSED_BY_DOXYGEN
 protected:
  template <typename OtherDerived>
  void assignTranspose(const PermutationBase<OtherDerived>& other) {
    for (Index i = 0; i < rows(); ++i) indices().coeffRef(other.indices().coeff(i)) = StorageIndex(i);
  }
  template <typename Lhs, typename Rhs>
  void assignProduct(const Lhs& lhs, const Rhs& rhs) {
    eigen_assert(lhs.cols() == rhs.rows());
    for (Index i = 0; i < rows(); ++i) indices().coeffRef(i) = lhs.indices().coeff(rhs.indices().coeff(i));
  }
#endif

 public:
  /** \returns the product permutation matrix.
   *
   * \note \blank \note_try_to_help_rvo
   */
  template <typename Other>
  inline PlainPermutationType operator*(const PermutationBase<Other>& other) const {
    return PlainPermutationType(internal::PermPermProduct, derived(), other.derived());
  }

  /** \returns the product of a permutation with another inverse permutation.
   *
   * \note \blank \note_try_to_help_rvo
   */
  template <typename Other>
  inline PlainPermutationType operator*(const InverseImpl<Other, PermutationStorage>& other) const {
    const auto& rhs = other.derived().nestedExpression();
    eigen_assert(size() == rhs.size());
    PlainPermutationType result(size());
    // (P * Q.inverse())(Q(i)) = P(i).
    for (Index i = 0; i < size(); ++i) result.indices().coeffRef(rhs.indices().coeff(i)) = indices().coeff(i);
    return result;
  }

  /** \returns the product of an inverse permutation with another permutation.
   *
   * \note \blank \note_try_to_help_rvo
   */
  template <typename Other>
  friend inline PlainPermutationType operator*(const InverseImpl<Other, PermutationStorage>& other,
                                               const PermutationBase& perm) {
    return PlainPermutationType(internal::PermPermProduct, other.eval(), perm);
  }

  /** \returns the determinant of the permutation matrix, which is either 1 or -1 depending on the parity of the
   * permutation.
   *
   * This function is O(\c n) procedure allocating a buffer of \c n booleans.
   */
  Index determinant() const {
    Index res = 1;
    Index n = size();
    Matrix<bool, RowsAtCompileTime, 1, 0, MaxRowsAtCompileTime> mask(n);
    mask.fill(false);
    Index r = 0;
    while (r < n) {
      // search for the next seed
      while (r < n && mask[r]) r++;
      if (r >= n) break;
      // we got one, let's follow it until we are back to the seed
      Index k0 = r++;
      mask.coeffRef(k0) = true;
      for (Index k = indices().coeff(k0); k != k0; k = indices().coeff(k)) {
        mask.coeffRef(k) = true;
        res = -res;
      }
    }
    return res;
  }
};

namespace internal {
template <int SizeAtCompileTime, int MaxSizeAtCompileTime, typename StorageIndex_>
struct traits<PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex_> >
    : traits<
          Matrix<StorageIndex_, SizeAtCompileTime, SizeAtCompileTime, 0, MaxSizeAtCompileTime, MaxSizeAtCompileTime> > {
  using StorageKind = PermutationStorage;
  using IndicesType = Matrix<StorageIndex_, SizeAtCompileTime, 1, 0, MaxSizeAtCompileTime, 1>;
  using StorageIndex = StorageIndex_;
  using Scalar = void;
};
}  // namespace internal

/** \class PermutationMatrix
 * \ingroup Core_Module
 *
 * \brief Permutation matrix
 *
 * \tparam SizeAtCompileTime the number of rows/cols, or Dynamic
 * \tparam MaxSizeAtCompileTime the maximum number of rows/cols, or Dynamic. This optional parameter defaults to
 * SizeAtCompileTime. Most of the time, you should not have to specify it. \tparam StorageIndex_ the integer type of the
 * indices
 *
 * This class represents a permutation matrix, internally stored as a vector of integers.
 *
 * \sa class PermutationBase, class PermutationWrapper, class DiagonalMatrix
 */
template <int SizeAtCompileTime, int MaxSizeAtCompileTime, typename StorageIndex_>
class PermutationMatrix
    : public PermutationBase<PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex_> > {
  using Base = PermutationBase<PermutationMatrix>;
  using Traits = internal::traits<PermutationMatrix>;

 public:
  using Nested = const PermutationMatrix&;

#ifndef EIGEN_PARSED_BY_DOXYGEN
  using IndicesType = typename Traits::IndicesType;
  using StorageIndex = typename Traits::StorageIndex;
#endif

  EIGEN_DEVICE_FUNC PermutationMatrix() = default;

  /** Constructs an uninitialized permutation matrix of given size.
   */
  EIGEN_DEVICE_FUNC explicit PermutationMatrix(Index size) : m_indices(size) {
    eigen_internal_assert(size <= NumTraits<StorageIndex>::highest());
  }

  /** Copy constructor. */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC PermutationMatrix(const PermutationBase<OtherDerived>& other) : m_indices(other.indices()) {}

  /** Generic constructor from expression of the indices. The indices
   * array has the meaning that the permutations sends each integer i to indices[i].
   *
   * \warning It is your responsibility to check that the indices array that you passes actually
   * describes a permutation, i.e., each value between 0 and n-1 occurs exactly once, where n is the
   * array's size.
   */
  template <typename Other>
  EIGEN_DEVICE_FUNC explicit PermutationMatrix(const MatrixBase<Other>& indices) : m_indices(indices) {}

  /** Convert the Transpositions \a tr to a permutation matrix */
  template <typename Other>
  explicit PermutationMatrix(const TranspositionsBase<Other>& tr) : m_indices(tr.size()) {
    *this = tr;
  }

  /** Copies the other permutation into *this */
  template <typename Other>
  EIGEN_DEVICE_FUNC PermutationMatrix& operator=(const PermutationBase<Other>& other) {
    m_indices = other.indices();
    return *this;
  }

  /** Assignment from the Transpositions \a tr */
  template <typename Other>
  PermutationMatrix& operator=(const TranspositionsBase<Other>& tr) {
    return Base::operator=(tr.derived());
  }

  /** const version of indices(). */
  EIGEN_DEVICE_FUNC constexpr const IndicesType& indices() const { return m_indices; }
  /** \returns a reference to the stored array representing the permutation. */
  EIGEN_DEVICE_FUNC constexpr IndicesType& indices() { return m_indices; }

  /**** multiplication helpers to hopefully get RVO ****/

#ifndef EIGEN_PARSED_BY_DOXYGEN
  template <typename Other>
  PermutationMatrix(const InverseImpl<Other, PermutationStorage>& other)
      : m_indices(other.derived().nestedExpression().size()) {
    eigen_internal_assert(m_indices.size() <= NumTraits<StorageIndex>::highest());
    Base::assignTranspose(other.derived().nestedExpression());
  }
  template <typename Lhs, typename Rhs>
  PermutationMatrix(internal::PermPermProduct_t, const Lhs& lhs, const Rhs& rhs) : m_indices(lhs.indices().size()) {
    Base::assignProduct(lhs, rhs);
  }
#endif

 protected:
  IndicesType m_indices;
};

namespace internal {
template <int SizeAtCompileTime, int MaxSizeAtCompileTime, typename StorageIndex_, int PacketAccess_>
struct traits<Map<PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex_>, PacketAccess_> >
    : traits<
          Matrix<StorageIndex_, SizeAtCompileTime, SizeAtCompileTime, 0, MaxSizeAtCompileTime, MaxSizeAtCompileTime> > {
  using StorageKind = PermutationStorage;
  using IndicesType = Map<const Matrix<StorageIndex_, SizeAtCompileTime, 1, 0, MaxSizeAtCompileTime, 1>, PacketAccess_>;
  using StorageIndex = StorageIndex_;
  using Scalar = void;
};
}  // namespace internal

template <int SizeAtCompileTime, int MaxSizeAtCompileTime, typename StorageIndex_, int PacketAccess_>
class Map<PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex_>, PacketAccess_>
    : public PermutationBase<
          Map<PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex_>, PacketAccess_> > {
  using Base = PermutationBase<Map>;
  using Traits = internal::traits<Map>;

 public:
#ifndef EIGEN_PARSED_BY_DOXYGEN
  using IndicesType = typename Traits::IndicesType;
  using StorageIndex = typename IndicesType::Scalar;
#endif

  inline Map(const StorageIndex* indicesPtr) : m_indices(indicesPtr) {}

  inline Map(const StorageIndex* indicesPtr, Index size) : m_indices(indicesPtr, size) {}

  /** Copies the other permutation into *this */
  template <typename Other>
  Map& operator=(const PermutationBase<Other>& other) {
    return Base::operator=(other.derived());
  }

  /** Assignment from the Transpositions \a tr */
  template <typename Other>
  Map& operator=(const TranspositionsBase<Other>& tr) {
    return Base::operator=(tr.derived());
  }

#ifndef EIGEN_PARSED_BY_DOXYGEN
  /** This is a special case of the templated operator=. Its purpose is to
   * prevent a default operator= from hiding the templated operator=.
   */
  Map& operator=(const Map& other) {
    m_indices = other.m_indices;
    return *this;
  }
#endif

  /** const version of indices(). */
  const IndicesType& indices() const { return m_indices; }
  /** \returns a reference to the stored array representing the permutation. */
  IndicesType& indices() { return m_indices; }

 protected:
  IndicesType m_indices;
};

namespace internal {
template <typename IndicesType_>
struct traits<PermutationWrapper<IndicesType_> > {
  using StorageKind = PermutationStorage;
  using Scalar = void;
  using StorageIndex = typename IndicesType_::Scalar;
  using IndicesType = IndicesType_;
  enum {
    RowsAtCompileTime = IndicesType_::SizeAtCompileTime,
    ColsAtCompileTime = IndicesType_::SizeAtCompileTime,
    MaxRowsAtCompileTime = IndicesType::MaxSizeAtCompileTime,
    MaxColsAtCompileTime = IndicesType::MaxSizeAtCompileTime,
    Flags = 0
  };
};
}  // namespace internal

/** \class PermutationWrapper
 * \ingroup Core_Module
 *
 * \brief Class to view a vector of integers as a permutation matrix
 *
 * \tparam IndicesType_ the type of the vector of integer (can be any compatible expression)
 *
 * This class allows to view any vector expression of integers as a permutation matrix.
 *
 * \sa class PermutationBase, class PermutationMatrix
 */
template <typename IndicesType_>
class PermutationWrapper : public PermutationBase<PermutationWrapper<IndicesType_> > {
  using Base = PermutationBase<PermutationWrapper>;
  using Traits = internal::traits<PermutationWrapper>;

 public:
#ifndef EIGEN_PARSED_BY_DOXYGEN
  using IndicesType = typename Traits::IndicesType;
#endif

  inline PermutationWrapper(const IndicesType& indices) : m_indices(indices) {}

  /** const version of indices(). */
  const internal::remove_all_t<typename IndicesType::Nested>& indices() const { return m_indices; }

 protected:
  typename IndicesType::Nested m_indices;
};

/** \returns the matrix with the permutation applied to the columns.
 */
template <typename MatrixDerived, typename PermutationDerived>
EIGEN_DEVICE_FUNC const Product<MatrixDerived, PermutationDerived, DefaultProduct> operator*(
    const MatrixBase<MatrixDerived>& matrix, const PermutationBase<PermutationDerived>& permutation) {
  return Product<MatrixDerived, PermutationDerived, DefaultProduct>(matrix.derived(), permutation.derived());
}

/** \returns the matrix with the permutation applied to the rows.
 */
template <typename PermutationDerived, typename MatrixDerived>
EIGEN_DEVICE_FUNC const Product<PermutationDerived, MatrixDerived, DefaultProduct> operator*(
    const PermutationBase<PermutationDerived>& permutation, const MatrixBase<MatrixDerived>& matrix) {
  return Product<PermutationDerived, MatrixDerived, DefaultProduct>(permutation.derived(), matrix.derived());
}

// Sums with a permutation are lazy dense expressions: the permutation is read as a 0/1 matrix with the scalar
// type of the other operand, so no dense copy of it is formed.

/** \returns the lazy sum of the dense matrix \a matrix and the permutation matrix \a permutation */
template <typename MatrixDerived, typename PermutationDerived>
EIGEN_DEVICE_FUNC auto operator+(const MatrixBase<MatrixDerived>& matrix,
                                 const PermutationBase<PermutationDerived>& permutation) {
  return matrix.derived() +
         internal::permutation_dense_expression<typename MatrixDerived::Scalar, false, PermutationDerived>::run(
             permutation.derived());
}

/** \returns the lazy sum of the permutation matrix \a permutation and the dense matrix \a matrix */
template <typename PermutationDerived, typename MatrixDerived>
EIGEN_DEVICE_FUNC auto operator+(const PermutationBase<PermutationDerived>& permutation,
                                 const MatrixBase<MatrixDerived>& matrix) {
  return internal::permutation_dense_expression<typename MatrixDerived::Scalar, false, PermutationDerived>::run(
             permutation.derived()) +
         matrix.derived();
}

/** \returns the lazy difference of the dense matrix \a matrix and the permutation matrix \a permutation */
template <typename MatrixDerived, typename PermutationDerived>
EIGEN_DEVICE_FUNC auto operator-(const MatrixBase<MatrixDerived>& matrix,
                                 const PermutationBase<PermutationDerived>& permutation) {
  return matrix.derived() -
         internal::permutation_dense_expression<typename MatrixDerived::Scalar, false, PermutationDerived>::run(
             permutation.derived());
}

/** \returns the lazy difference of the permutation matrix \a permutation and the dense matrix \a matrix */
template <typename PermutationDerived, typename MatrixDerived>
EIGEN_DEVICE_FUNC auto operator-(const PermutationBase<PermutationDerived>& permutation,
                                 const MatrixBase<MatrixDerived>& matrix) {
  return internal::permutation_dense_expression<typename MatrixDerived::Scalar, false, PermutationDerived>::run(
             permutation.derived()) -
         matrix.derived();
}

/** \returns the lazy sum of the diagonal matrix \a diagonal and the permutation matrix \a permutation */
template <typename DiagonalDerived, typename PermutationDerived>
EIGEN_DEVICE_FUNC auto operator+(const DiagonalBase<DiagonalDerived>& diagonal,
                                 const PermutationBase<PermutationDerived>& permutation) {
  return diagonal.derived() +
         internal::permutation_dense_expression<typename DiagonalDerived::Scalar, false, PermutationDerived>::run(
             permutation.derived());
}

/** \returns the lazy sum of the permutation matrix \a permutation and the diagonal matrix \a diagonal */
template <typename PermutationDerived, typename DiagonalDerived>
EIGEN_DEVICE_FUNC auto operator+(const PermutationBase<PermutationDerived>& permutation,
                                 const DiagonalBase<DiagonalDerived>& diagonal) {
  return internal::permutation_dense_expression<typename DiagonalDerived::Scalar, false, PermutationDerived>::run(
             permutation.derived()) +
         diagonal.derived();
}

/** \returns the lazy difference of the diagonal matrix \a diagonal and the permutation matrix \a permutation */
template <typename DiagonalDerived, typename PermutationDerived>
EIGEN_DEVICE_FUNC auto operator-(const DiagonalBase<DiagonalDerived>& diagonal,
                                 const PermutationBase<PermutationDerived>& permutation) {
  return diagonal.derived() -
         internal::permutation_dense_expression<typename DiagonalDerived::Scalar, false, PermutationDerived>::run(
             permutation.derived());
}

/** \returns the lazy difference of the permutation matrix \a permutation and the diagonal matrix \a diagonal */
template <typename PermutationDerived, typename DiagonalDerived>
EIGEN_DEVICE_FUNC auto operator-(const PermutationBase<PermutationDerived>& permutation,
                                 const DiagonalBase<DiagonalDerived>& diagonal) {
  return internal::permutation_dense_expression<typename DiagonalDerived::Scalar, false, PermutationDerived>::run(
             permutation.derived()) -
         diagonal.derived();
}

template <typename PermutationType>
class InverseImpl<PermutationType, PermutationStorage> : public EigenBase<Inverse<PermutationType> > {
  using PlainPermutationType = typename PermutationType::PlainPermutationType;
  using PermTraits = internal::traits<PermutationType>;

 protected:
  InverseImpl() = default;

 public:
  using InverseType = Inverse<PermutationType>;
  using EigenBase<Inverse<PermutationType> >::derived;

#ifndef EIGEN_PARSED_BY_DOXYGEN
  using DenseMatrixType = typename PermutationType::DenseMatrixType;
  enum {
    RowsAtCompileTime = PermTraits::RowsAtCompileTime,
    ColsAtCompileTime = PermTraits::ColsAtCompileTime,
    MaxRowsAtCompileTime = PermTraits::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = PermTraits::MaxColsAtCompileTime
  };
#endif

#ifndef EIGEN_PARSED_BY_DOXYGEN
  template <typename DenseDerived>
  void evalTo(MatrixBase<DenseDerived>& other) const {
    other.setZero();
    for (Index i = 0; i < derived().rows(); ++i)
      other.coeffRef(i, derived().nestedExpression().indices().coeff(i)) = typename DenseDerived::Scalar(1);
  }
#endif

  /** \return the equivalent permutation matrix */
  PlainPermutationType eval() const { return derived(); }

  DenseMatrixType toDenseMatrix() const { return derived(); }

  /** \returns the matrix with the inverse permutation applied to the columns.
   */
  template <typename OtherDerived>
  friend const Product<OtherDerived, InverseType, DefaultProduct> operator*(const MatrixBase<OtherDerived>& matrix,
                                                                            const InverseType& trPerm) {
    return Product<OtherDerived, InverseType, DefaultProduct>(matrix.derived(), trPerm.derived());
  }

  /** \returns the matrix with the inverse permutation applied to the rows.
   */
  template <typename OtherDerived>
  const Product<InverseType, OtherDerived, DefaultProduct> operator*(const MatrixBase<OtherDerived>& matrix) const {
    return Product<InverseType, OtherDerived, DefaultProduct>(derived(), matrix.derived());
  }

  // Lazy sums with a dense or diagonal matrix, as for PermutationBase; the inverse is read as the transposed 0/1
  // matrix.
  /** \returns the lazy sum of the inverse permutation and the dense matrix \a matrix */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC auto operator+(const MatrixBase<OtherDerived>& matrix) const {
    return denseExpression<typename OtherDerived::Scalar>() + matrix.derived();
  }
  /** \returns the lazy sum of the dense matrix \a matrix and the inverse permutation \a inverse */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC friend auto operator+(const MatrixBase<OtherDerived>& matrix, const InverseType& inverse) {
    return matrix.derived() + inverse.template denseExpression<typename OtherDerived::Scalar>();
  }
  /** \returns the lazy difference of the inverse permutation and the dense matrix \a matrix */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC auto operator-(const MatrixBase<OtherDerived>& matrix) const {
    return denseExpression<typename OtherDerived::Scalar>() - matrix.derived();
  }
  /** \returns the lazy difference of the dense matrix \a matrix and the inverse permutation \a inverse */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC friend auto operator-(const MatrixBase<OtherDerived>& matrix, const InverseType& inverse) {
    return matrix.derived() - inverse.template denseExpression<typename OtherDerived::Scalar>();
  }
  /** \returns the lazy sum of the inverse permutation and the diagonal matrix \a diagonal */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC auto operator+(const DiagonalBase<OtherDerived>& diagonal) const {
    return denseExpression<typename OtherDerived::Scalar>() + diagonal.derived();
  }
  /** \returns the lazy sum of the diagonal matrix \a diagonal and the inverse permutation \a inverse */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC friend auto operator+(const DiagonalBase<OtherDerived>& diagonal, const InverseType& inverse) {
    return diagonal.derived() + inverse.template denseExpression<typename OtherDerived::Scalar>();
  }
  /** \returns the lazy difference of the inverse permutation and the diagonal matrix \a diagonal */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC auto operator-(const DiagonalBase<OtherDerived>& diagonal) const {
    return denseExpression<typename OtherDerived::Scalar>() - diagonal.derived();
  }
  /** \returns the lazy difference of the diagonal matrix \a diagonal and the inverse permutation \a inverse */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC friend auto operator-(const DiagonalBase<OtherDerived>& diagonal, const InverseType& inverse) {
    return diagonal.derived() - inverse.template denseExpression<typename OtherDerived::Scalar>();
  }

 private:
  template <typename Scalar>
  EIGEN_DEVICE_FUNC auto denseExpression() const {
    using Nested = internal::remove_all_t<typename InverseType::XprTypeNestedCleaned>;
    return internal::permutation_dense_expression<Scalar, true, Nested>::run(derived().nestedExpression());
  }
};

template <typename Derived>
const PermutationWrapper<const Derived> MatrixBase<Derived>::asPermutation() const {
  return derived();
}

namespace internal {

template <>
struct AssignmentKind<DenseShape, PermutationShape> {
  using Kind = EigenBase2EigenBase;
};

// Dense ?= (dense or diagonal) +/- permutation, in either order. The permutation's one in column k is at row
// indices(k), and at column indices(k) of row k for its transpose; the column kernels below need one nonzero per
// destination column, so P pairs with a column-major and P^T with a row-major destination, and the other pairings
// stay on the generic path.
template <typename Scalar, typename IndicesType>
struct permutation_column_nonzeros {
  EIGEN_DEVICE_FUNC explicit permutation_column_nonzeros(const IndicesType& indices) : m_indices(indices) {}
  EIGEN_DEVICE_FUNC Index row(Index k) const { return Index(m_indices.coeff(k)); }
  EIGEN_DEVICE_FUNC Scalar value(Index) const { return Scalar(1); }
  const IndicesType& m_indices;
};

template <typename Scalar, typename IndicesType, bool Transposed, typename Plain>
using permutation_dense_xpr = CwiseNullaryOp<permutation_dense_op<Scalar, IndicesType, Transposed>, Plain>;

template <typename Scalar, typename IndicesType, bool Transposed, typename Plain>
struct is_permutation_dense_xpr<permutation_dense_xpr<Scalar, IndicesType, Transposed, Plain>> : std::true_type {};

// A diagonal operand never goes through the block pass, so only a dense one is subject to
// dense_block_pass_cannot_overflow.
template <typename Dst, typename OtherXpr, bool Transposed, typename Functor, bool NegateOther>
struct dense_permutation_sum_fast_path
    : bool_constant<(is_dense_shape<OtherXpr>::value || is_diagonal_shape<OtherXpr>::value) &&
                    !is_permutation_dense_xpr<OtherXpr>::value && bool(Dst::IsRowMajor) == Transposed &&
                    additive_assign_sign<Functor>::value != 0 &&
                    std::is_same<typename Dst::Scalar, typename OtherXpr::Scalar>::value &&
                    (is_diagonal_shape<OtherXpr>::value ||
                     dense_block_pass_cannot_overflow<typename Dst::Scalar, Functor, NegateOther>::value)> {};

// dst ?= op(lhs, rhs) for a diagonal d and a permutation whose one in column k is at row r: op(d(k), [r == k]) at
// (k, k), op(0, 1) or op(1, 0) at (r, k) when r != k, and structural zeros, which only = writes. A block's d(k) are
// read before the block is written.
template <bool DiagonalOnLeft>
struct diagonal_permutation_sum_assignment {
  template <typename Dst, typename Diagonal, typename Permutation, typename BinaryOp, typename Functor>
  EIGEN_DEVICE_FUNC static void run(Dst& dst, const Diagonal& diagonal, const Permutation& permutation,
                                    const BinaryOp& op, const Functor& func) {
    using Scalar = typename Dst::Scalar;
    constexpr Index kBlockColumns = 32;
    const Scalar offDiagonalOne = DiagonalOnLeft ? op(Scalar(0), Scalar(1)) : op(Scalar(1), Scalar(0));
    Scalar atDiagonal[kBlockColumns];
    for (Index j = 0; j < dst.cols(); j += kBlockColumns) {
      const Index columns = numext::mini(Index(kBlockColumns), dst.cols() - j);
      for (Index k = 0; k < columns; ++k) {
        const Scalar d = diagonal.value(j + k);
        const Scalar p = permutation.row(j + k) == j + k ? Scalar(1) : Scalar(0);
        atDiagonal[k] = DiagonalOnLeft ? op(d, p) : op(p, d);
      }
      EIGEN_IF_CONSTEXPR (is_plain_assign<Functor>::value) {
        dst.middleCols(j, columns).setZero();
      }
      for (Index k = 0; k < columns; ++k) {
        const Index r = permutation.row(j + k);
        func.assignCoeff(dst.coeffRef(j + k, j + k), atDiagonal[k]);
        if (r != j + k) {
          func.assignCoeff(dst.coeffRef(r, j + k), offDiagonalOne);
        }
      }
    }
  }
};

template <bool OtherIsDiagonal, bool OtherOnLeft>
struct permutation_sum_assignment {
  template <typename Dst, typename SrcXprType, typename OtherXpr, typename Nonzeros, typename Functor>
  EIGEN_DEVICE_FUNC static void run(Dst& dst, const SrcXprType& src, const OtherXpr& other, const Nonzeros& ones,
                                    const Functor& func) {
    dense_structured_sum_assignment<OtherOnLeft>::assign(dst, src, other, ones, func);
  }
};

template <bool OtherOnLeft>
struct permutation_sum_assignment<true, OtherOnLeft> {
  template <typename Dst, typename SrcXprType, typename OtherXpr, typename Nonzeros, typename Functor>
  EIGEN_DEVICE_FUNC static void run(Dst& dst, const SrcXprType& src, const OtherXpr& other, const Nonzeros& ones,
                                    const Functor& func) {
    const diagonal_column_nonzeros<OtherXpr> diagonal(other);
    resize_if_allowed(dst, src, func);
    auto&& dstView = column_major_view<bool(Dst::IsRowMajor)>::run(dst);
    diagonal_permutation_sum_assignment<OtherOnLeft>::run(dstView, diagonal, ones, src.functor(), func);
  }
};

// other +/- permutation
template <typename DstXprType, typename BinaryOp, typename OtherXpr, typename Scalar, typename IndicesType,
          bool Transposed, typename Plain, typename Functor>
struct Assignment<
    DstXprType,
    CwiseBinaryOp<BinaryOp, const OtherXpr, const permutation_dense_xpr<Scalar, IndicesType, Transposed, Plain>>,
    Functor, Dense2Dense,
    std::enable_if_t<is_additive_binary_op<BinaryOp>::value &&
                     dense_permutation_sum_fast_path<DstXprType, OtherXpr, Transposed, Functor, false>::value>> {
  using SrcXprType =
      CwiseBinaryOp<BinaryOp, const OtherXpr, const permutation_dense_xpr<Scalar, IndicesType, Transposed, Plain>>;
  EIGEN_DEVICE_FUNC static void run(DstXprType& dst, const SrcXprType& src, const Functor& func) {
    const auto& indices = src.rhs().functor().indices();
    permutation_sum_assignment<is_diagonal_shape<OtherXpr>::value, true>::run(
        dst, src, src.lhs(), permutation_column_nonzeros<Scalar, remove_all_t<decltype(indices)>>(indices), func);
  }
};

// permutation +/- other; a dense product on the right is left to the "xpr + product" rule.
template <typename DstXprType, typename BinaryOp, typename OtherXpr, typename Scalar, typename IndicesType,
          bool Transposed, typename Plain, typename Functor>
struct Assignment<
    DstXprType,
    CwiseBinaryOp<BinaryOp, const permutation_dense_xpr<Scalar, IndicesType, Transposed, Plain>, const OtherXpr>,
    Functor, Dense2Dense,
    std::enable_if_t<is_additive_binary_op<BinaryOp>::value &&
                     dense_permutation_sum_fast_path<DstXprType, OtherXpr, Transposed, Functor,
                                                     is_difference_op<BinaryOp>::value>::value &&
                     !is_default_product<OtherXpr>::value>> {
  using SrcXprType =
      CwiseBinaryOp<BinaryOp, const permutation_dense_xpr<Scalar, IndicesType, Transposed, Plain>, const OtherXpr>;
  EIGEN_DEVICE_FUNC static void run(DstXprType& dst, const SrcXprType& src, const Functor& func) {
    const auto& indices = src.lhs().functor().indices();
    permutation_sum_assignment<is_diagonal_shape<OtherXpr>::value, false>::run(
        dst, src, src.rhs(), permutation_column_nonzeros<Scalar, remove_all_t<decltype(indices)>>(indices), func);
  }
};

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_PERMUTATIONMATRIX_H
