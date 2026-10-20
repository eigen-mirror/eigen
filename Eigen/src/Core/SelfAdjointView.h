// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SELFADJOINTMATRIX_H
#define EIGEN_SELFADJOINTMATRIX_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \class SelfAdjointView
 * \ingroup Core_Module
 *
 *
 * \brief Expression of a selfadjoint matrix from a triangular part of a dense matrix
 *
 * \tparam MatrixType the type of the dense matrix storing the coefficients
 * \tparam TriangularPart can be either \c #Lower or \c #Upper
 *
 * This class is an expression of a selfadjoint matrix from a triangular part of a matrix
 * with given dense storage of the coefficients. It is the return type of MatrixBase::selfadjointView()
 * and most of the time this is the only way that it is used.
 *
 * \sa class TriangularBase, MatrixBase::selfadjointView()
 */

namespace internal {

// Requires 0 <= i < j < matrix.rows(). The row prefixes may contain already-computed factors.
template <typename MatrixType>
EIGEN_DEVICE_FUNC void selfadjoint_swap_lower(MatrixType& matrix, Index i, Index j) {
  using Scalar = typename MatrixType::Scalar;
  const Index size = matrix.rows();
  eigen_internal_assert(matrix.cols() == size && i >= 0 && i < j && j < size);
  matrix.template block<1, Dynamic>(i, 0, 1, i).swap(matrix.template block<1, Dynamic>(j, 0, 1, i));
  for (Index k = i + 1; k < j; ++k) {
    const Scalar value = matrix.coeff(k, i);
    matrix.coeffRef(k, i) = numext::conj(matrix.coeff(j, k));
    matrix.coeffRef(j, k) = numext::conj(value);
  }
  matrix.template block<Dynamic, 1>(j + 1, i, size - j - 1, 1)
      .swap(matrix.template block<Dynamic, 1>(j + 1, j, size - j - 1, 1));
  EIGEN_IF_CONSTEXPR (NumTraits<Scalar>::IsComplex) {
    matrix.coeffRef(j, i) = numext::conj(matrix.coeff(j, i));
  }
  numext::swap(matrix.coeffRef(i, i), matrix.coeffRef(j, j));
}

// Consumes an owning copy of the permutation, disjoint from matrix.
template <typename MatrixType, typename PermutationType>
EIGEN_DEVICE_FUNC void selfadjoint_apply_permutation_lower(MatrixType& matrix, PermutationType& remaining) {
  const Index size = matrix.rows();
  for (Index i = 0; i < size; ++i) {
    while (Index(remaining.indices().coeff(i)) != i) {
      const Index j = convert_index<Index>(remaining.indices().coeff(i));
      // Earlier indices are fixed, so j > i. Each swap removes one edge from the permutation cycle.
      // If P(j) == j as well, index j repeats and this loop would not terminate.
      eigen_assert(i < j && j < size && Index(remaining.indices().coeff(j)) != j && "indices are not a permutation");
      selfadjoint_swap_lower(matrix, i, j);
      numext::swap(remaining.indices().coeffRef(i), remaining.indices().coeffRef(j));
    }
  }
}

template <typename MatrixType, unsigned int UpLo>
struct traits<SelfAdjointView<MatrixType, UpLo>> : traits<MatrixType> {
  using MatrixTypeNested = typename ref_selector<MatrixType>::non_const_type;
  using MatrixTypeNestedCleaned = remove_all_t<MatrixTypeNested>;
  using ExpressionType = MatrixType;
  using FullMatrixType = typename MatrixType::PlainObject;
  enum {
    Mode = UpLo | SelfAdjoint,
    FlagsLvalueBit = is_lvalue<MatrixType>::value ? LvalueBit : 0,
    Flags = MatrixTypeNestedCleaned::Flags & (HereditaryBits | FlagsLvalueBit) &
            (~(PacketAccessBit | DirectAccessBit | LinearAccessBit))  // FIXME these flags should be preserved
  };
};

}  // namespace internal

template <typename MatrixType_, unsigned int UpLo>
class SelfAdjointView : public TriangularBase<SelfAdjointView<MatrixType_, UpLo> > {
 public:
  EIGEN_STATIC_ASSERT(UpLo == Lower || UpLo == Upper, SELFADJOINTVIEW_ACCEPTS_UPPER_AND_LOWER_MODE_ONLY)

  using MatrixType = MatrixType_;
  using Base = TriangularBase<SelfAdjointView>;
  using MatrixTypeNested = typename internal::traits<SelfAdjointView>::MatrixTypeNested;
  using MatrixTypeNestedCleaned = typename internal::traits<SelfAdjointView>::MatrixTypeNestedCleaned;
  using NestedExpression = MatrixTypeNestedCleaned;

  /** \brief The type of coefficients in this matrix */
  using Scalar = typename internal::traits<SelfAdjointView>::Scalar;
  /** Real part of #Scalar */
  using RealScalar = typename NumTraits<Scalar>::Real;
  using StorageIndex = typename MatrixType::StorageIndex;

  enum {
    Mode = internal::traits<SelfAdjointView>::Mode,
    Flags = internal::traits<SelfAdjointView>::Flags,
    TransposeMode = ((int(Mode) & int(Upper)) ? Lower : 0) | ((int(Mode) & int(Lower)) ? Upper : 0)
  };
  using PlainObject = typename MatrixType::PlainObject;

  EIGEN_DEVICE_FUNC explicit inline SelfAdjointView(MatrixType& matrix) : m_matrix(matrix) {}
  using Base::operator*;
  EIGEN_DEFAULT_COPY_CONSTRUCTOR(SelfAdjointView)

  /** Assigns a matrix expression to the referenced triangular part of the selfadjoint matrix. */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC SelfAdjointView& operator=(const MatrixBase<OtherDerived>& other) {
    m_matrix.template triangularView<UpLo>() = other;
    return *this;
  }

  /** Assigns a triangular or selfadjoint expression without materializing a dense temporary. */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC SelfAdjointView& operator=(const TriangularBase<OtherDerived>& other) {
    other.evalToLazy(m_matrix);
    return *this;
  }

  EIGEN_DEVICE_FUNC SelfAdjointView& operator=(const SelfAdjointView& other) {
    return *this = static_cast<const Base&>(other);
  }

  /** \sa MatrixBase::operator+=() */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC SelfAdjointView& operator+=(const DenseBase<OtherDerived>& other) {
    m_matrix.template triangularView<UpLo>() += other;
    return *this;
  }

  /** \sa MatrixBase::operator-=() */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC SelfAdjointView& operator-=(const DenseBase<OtherDerived>& other) {
    m_matrix.template triangularView<UpLo>() -= other;
    return *this;
  }

  /** \sa MatrixBase::operator*=() */
  EIGEN_DEVICE_FUNC SelfAdjointView& operator*=(const Scalar& other) {
    eigen_assert(numext::imag(other) == typename NumTraits<Scalar>::Real(0) &&
                 "SelfAdjointView in-place scaling requires a real scalar; "
                 "scaling only the stored triangle by a non-real scalar would "
                 "leave conj(other) on the unstored half.");
    m_matrix.template triangularView<UpLo>() *= other;
    return *this;
  }

  /** \sa DenseBase::operator/=() */
  EIGEN_DEVICE_FUNC SelfAdjointView& operator/=(const Scalar& other) {
    eigen_assert(numext::imag(other) == typename NumTraits<Scalar>::Real(0) &&
                 "SelfAdjointView in-place division requires a real scalar; "
                 "dividing only the stored triangle by a non-real scalar would "
                 "leave conj(other) on the unstored half.");
    m_matrix.template triangularView<UpLo>() /= other;
    return *this;
  }

  /** Replaces this view's matrix \f$ S \f$ by \f$ P S P^T \f$ in place, where \a permutation represents \f$ P \f$.
   * Only the referenced triangle is read or written; the opposite triangle is unchanged. Complex coefficients
   * crossing the diagonal are conjugated. The matrix must be square and the permutation must have matching size.
   * The operation takes \f$ O(n^2) \f$ time and \f$ O(n) \f$ auxiliary storage for a copy of the permutation.
   *
   * \returns a reference to \c *this, retaining its self-adjoint structure.
   *
   * \code
   * A.selfadjointView<Lower>().applyPermutation(P);
   * \endcode
   */
  template <typename PermutationDerived>
  EIGEN_DEVICE_FUNC SelfAdjointView& applyPermutation(const PermutationBase<PermutationDerived>& permutation) {
    typename PermutationDerived::PlainPermutationType remaining(permutation);
    return applyPermutationInPlace(remaining);
  }

  /** Applies an inverse (or transposed) permutation in place. \sa applyPermutation() */
  template <typename PermutationDerived>
  EIGEN_DEVICE_FUNC SelfAdjointView& applyPermutation(
      const InverseImpl<PermutationDerived, PermutationStorage>& permutation) {
    typename PermutationDerived::PlainPermutationType remaining(permutation);
    return applyPermutationInPlace(remaining);
  }

  /** Applies the permutation represented by a sequence of transpositions in place.
   * The indices are copied before any matrix writes, including when they alias the matrix.
   * The operation takes \f$ O(n^2) \f$ time and \f$ O(n) \f$ auxiliary storage.
   * \sa applyPermutation(), applyTransposition()
   */
  template <typename TranspositionsDerived>
  EIGEN_DEVICE_FUNC SelfAdjointView& applyPermutation(const TranspositionsBase<TranspositionsDerived>& transpositions) {
    return applyTranspositions<false>(transpositions);
  }

  /** Applies an inverse (or transposed) sequence of transpositions in place. \sa applyPermutation() */
  template <typename TranspositionsDerived>
  EIGEN_DEVICE_FUNC SelfAdjointView& applyPermutation(
      const Transpose<TranspositionsBase<TranspositionsDerived>>& transpositions) {
    return applyTranspositions<true>(transpositions.nestedExpression());
  }

  /** Interchanges rows and columns \a i and \a j of this self-adjoint view in place.
   * Only the referenced triangle is read or written; complex coefficients crossing the diagonal are conjugated.
   * The matrix must be square and both indices must lie in [0, rows()). Equal indices leave the view unchanged.
   * The operation takes \f$ O(n) \f$ time and \f$ O(1) \f$ auxiliary storage, without heap allocation.
   * \returns a reference to \c *this.
   * \sa applyPermutation()
   */
  EIGEN_DEVICE_FUNC SelfAdjointView& applyTransposition(Index i, Index j) {
    EIGEN_STATIC_ASSERT_LVALUE(MatrixType);
    eigen_assert(this->rows() == this->cols() && i >= 0 && j >= 0 && i < this->rows() && j < this->rows());
    if (i == j) return *this;
    if (i > j) numext::swap(i, j);
    EIGEN_IF_CONSTEXPR (UpLo == Lower) {
      internal::selfadjoint_swap_lower(m_matrix, i, j);
    } else {
      auto transposed = m_matrix.transpose();
      internal::selfadjoint_swap_lower(transposed, i, j);
    }
    return *this;
  }

  /** \internal */
  EIGEN_DEVICE_FUNC constexpr const MatrixTypeNestedCleaned& _expression() const noexcept { return m_matrix; }

  EIGEN_DEVICE_FUNC constexpr const MatrixTypeNestedCleaned& nestedExpression() const noexcept { return m_matrix; }
  EIGEN_DEVICE_FUNC constexpr std::remove_reference_t<MatrixTypeNested>& nestedExpression() noexcept {
    return m_matrix;
  }

  EIGEN_DEVICE_FUNC const SelfAdjointView<
      const EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(MatrixType, Scalar, internal::scalar_product_op), UpLo>
  operator*(const Scalar& s) const {
    return (nestedExpression() * s).template selfadjointView<UpLo>();
  }

  friend EIGEN_DEVICE_FUNC const SelfAdjointView<
      const EIGEN_SCALAR_BINARYOP_EXPR_RETURN_TYPE(Scalar, MatrixType, internal::scalar_product_op), UpLo>
  operator*(const Scalar& s, const SelfAdjointView& mat) {
    return (s * mat.nestedExpression()).template selfadjointView<UpLo>();
  }

  /** Perform a symmetric rank 2 update of the selfadjoint matrix \c *this:
   * \f$ this = this + \alpha u v^* + conj(\alpha) v u^* \f$
   * \returns a reference to \c *this
   *
   * The vectors \a u and \c v \b must be column vectors, however they can be
   * an adjoint expression without any overhead. Only the meaningful triangular
   * part of the matrix is updated, the rest is left unchanged.
   *
   * \sa rankUpdate(const MatrixBase<DerivedU>&, Scalar)
   */
  template <typename DerivedU, typename DerivedV>
  EIGEN_DEVICE_FUNC SelfAdjointView& rankUpdate(const MatrixBase<DerivedU>& u, const MatrixBase<DerivedV>& v,
                                                const Scalar& alpha = Scalar(1));

  /** Perform a symmetric rank K update of the selfadjoint matrix \c *this:
   * \f$ this = this + \alpha ( u u^* ) \f$ where \a u is a vector or matrix.
   *
   * \returns a reference to \c *this
   *
   * Note that to perform \f$ this = this + \alpha ( u^* u ) \f$ you can simply
   * call this function with u.adjoint().
   *
   * \sa rankUpdate(const MatrixBase<DerivedU>&, const MatrixBase<DerivedV>&, Scalar)
   */
  template <typename DerivedU>
  EIGEN_DEVICE_FUNC SelfAdjointView& rankUpdate(const MatrixBase<DerivedU>& u, const Scalar& alpha = Scalar(1));

  /** \returns an expression of a triangular view extracted from the current selfadjoint view of a given triangular part
   *
   * The parameter \a TriMode can have the following values: \c #Upper, \c #StrictlyUpper, \c #UnitUpper,
   * \c #Lower, \c #StrictlyLower, \c #UnitLower.
   *
   * If \c TriMode references the same triangular part than \c *this, then this method simply return a \c TriangularView
   * of the nested expression, otherwise, the nested expression is first transposed, thus returning a \c
   * TriangularView<Transpose<MatrixType>> object.
   *
   * \sa MatrixBase::triangularView(), class TriangularView
   */
  template <unsigned int TriMode>
  EIGEN_DEVICE_FUNC
      std::conditional_t<(TriMode & (Upper | Lower)) == (UpLo & (Upper | Lower)), TriangularView<MatrixType, TriMode>,
                         TriangularView<typename MatrixType::AdjointReturnType, TriMode> >
      triangularView() const {
    std::conditional_t<(TriMode & (Upper | Lower)) == (UpLo & (Upper | Lower)), MatrixType&,
                       typename MatrixType::ConstTransposeReturnType>
        tmp1(m_matrix);
    std::conditional_t<(TriMode & (Upper | Lower)) == (UpLo & (Upper | Lower)), MatrixType&,
                       typename MatrixType::AdjointReturnType>
        tmp2(tmp1);
    return std::conditional_t<(TriMode & (Upper | Lower)) == (UpLo & (Upper | Lower)),
                              TriangularView<MatrixType, TriMode>,
                              TriangularView<typename MatrixType::AdjointReturnType, TriMode> >(tmp2);
  }

  /** \returns a const expression of the main diagonal of the matrix \c *this
   *
   * This method simply returns the diagonal of the nested expression, thus by-passing the SelfAdjointView decorator.
   *
   * \sa MatrixBase::diagonal(), class Diagonal */
  EIGEN_DEVICE_FUNC typename MatrixType::ConstDiagonalReturnType diagonal() const {
    return typename MatrixType::ConstDiagonalReturnType(m_matrix);
  }

  /** \returns the matrix 1-norm (maximum absolute column sum) of the implicit
   * full self-adjoint matrix, reading only the stored triangle. For Hermitian
   * (complex) scalars the unstored entries are conjugates of stored ones, and
   * since |conj(x)| = |x| the result matches the L1 norm of the full matrix.
   */
  EIGEN_DEVICE_FUNC RealScalar l1Norm() const { return internal::selfadjoint_l1_norm<UpLo>(m_matrix); }

  /////////// Cholesky module ///////////

  LLT<PlainObject, UpLo> llt() const;
  LDLT<PlainObject, UpLo> ldlt() const;
  BunchKaufman<PlainObject, UpLo> bunchKaufman() const;

  /////////// Eigenvalue module ///////////

  /** Return type of eigenvalues() */
  using EigenvaluesReturnType = Matrix<RealScalar, internal::traits<MatrixType>::ColsAtCompileTime, 1>;

  EIGEN_DEVICE_FUNC EigenvaluesReturnType eigenvalues() const;
  EIGEN_DEVICE_FUNC RealScalar operatorNorm() const;

 private:
  template <typename PermutationType>
  EIGEN_DEVICE_FUNC SelfAdjointView& applyPermutationInPlace(PermutationType& remaining) {
    EIGEN_STATIC_ASSERT_LVALUE(MatrixType);
    eigen_assert(this->rows() == this->cols() && this->rows() == remaining.size());
    EIGEN_IF_CONSTEXPR (UpLo == Lower) {
      internal::selfadjoint_apply_permutation_lower(m_matrix, remaining);
    } else {
      auto transposed = m_matrix.transpose();
      internal::selfadjoint_apply_permutation_lower(transposed, remaining);
    }
    return *this;
  }

  template <bool Reverse, typename TranspositionsDerived>
  EIGEN_DEVICE_FUNC SelfAdjointView& applyTranspositions(
      const TranspositionsBase<TranspositionsDerived>& transpositions) {
    EIGEN_STATIC_ASSERT_LVALUE(MatrixType);
    eigen_assert(this->rows() == this->cols() && this->rows() == transpositions.size());
    const typename TranspositionsDerived::IndicesType::PlainObject indices(transpositions.indices());
    const Index size = this->rows();
    for (Index k = 0; k < size; ++k) {
      const Index i = Reverse ? size - 1 - k : k;
      applyTransposition(i, internal::convert_index<Index>(indices.coeff(i)));
    }
    return *this;
  }

 protected:
  MatrixTypeNested m_matrix;
};

// selfadjoint to dense matrix

namespace internal {

// TODO currently a selfadjoint expression has the form SelfAdjointView<.,.>
//      in the future selfadjoint-ness should be defined by the expression traits
//      such that Transpose<SelfAdjointView<.,.> > is valid. (currently TriangularBase::transpose() is overloaded to
//      make it work)
template <typename MatrixType, unsigned int Mode>
struct evaluator_traits<SelfAdjointView<MatrixType, Mode> > {
  using Kind = typename storage_kind_to_evaluator_kind<typename MatrixType::StorageKind>::Kind;
  using Shape = SelfAdjointShape;
};

template <int UpLo, int SetOpposite, typename DstEvaluatorTypeT, typename SrcEvaluatorTypeT, typename Functor,
          int Version>
class triangular_dense_assignment_kernel<UpLo, SelfAdjoint, SetOpposite, DstEvaluatorTypeT, SrcEvaluatorTypeT, Functor,
                                         Version>
    : public generic_dense_assignment_kernel<DstEvaluatorTypeT, SrcEvaluatorTypeT, Functor, Version> {
 protected:
  using Base = generic_dense_assignment_kernel<DstEvaluatorTypeT, SrcEvaluatorTypeT, Functor, Version>;
  using DstXprType = typename Base::DstXprType;
  using SrcXprType = typename Base::SrcXprType;
  using Base::m_dst;
  using Base::m_functor;
  using Base::m_src;

 public:
  using DstEvaluatorType = typename Base::DstEvaluatorType;
  using SrcEvaluatorType = typename Base::SrcEvaluatorType;
  using Scalar = typename Base::Scalar;
  using AssignmentTraits = typename Base::AssignmentTraits;

  EIGEN_DEVICE_FUNC triangular_dense_assignment_kernel(DstEvaluatorType& dst, const SrcEvaluatorType& src,
                                                       const Functor& func, DstXprType& dstExpr)
      : Base(dst, src, func, dstExpr) {}

  EIGEN_DEVICE_FUNC void assignCoeff(Index row, Index col) {
    eigen_internal_assert(row != col);
    Scalar tmp = m_src.coeff(row, col);
    m_functor.assignCoeff(m_dst.coeffRef(row, col), tmp);
    m_functor.assignCoeff(m_dst.coeffRef(col, row), numext::conj(tmp));
  }

  // Override to ensure the SelfAdjoint assignCoeff (which mirrors conjugates) is called,
  // not the base class version (which is a plain copy).
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void assignCoeffByOuterInner(Index outer, Index inner) {
    Index row = Base::rowIndexByOuterInner(outer, inner);
    Index col = Base::colIndexByOuterInner(outer, inner);
    assignCoeff(row, col);
  }

  EIGEN_DEVICE_FUNC void assignDiagonalCoeff(Index id) { Base::assignCoeff(id, id); }

  EIGEN_DEVICE_FUNC void assignOppositeCoeff(Index, Index) { eigen_internal_assert(false && "should never be called"); }
};

}  // end namespace internal

/***************************************************************************
 * Implementation of MatrixBase methods
 ***************************************************************************/

/** This is the const version of MatrixBase::selfadjointView() */
template <typename Derived>
template <unsigned int UpLo>
EIGEN_DEVICE_FUNC constexpr typename MatrixBase<Derived>::template ConstSelfAdjointViewReturnType<UpLo>::Type
MatrixBase<Derived>::selfadjointView() const {
  return typename ConstSelfAdjointViewReturnType<UpLo>::Type(derived());
}

/** \returns an expression of a symmetric/self-adjoint view extracted from the upper or lower triangular part of the
 * current matrix
 *
 * The parameter \a UpLo can be either \c #Upper or \c #Lower
 *
 * Example: \include MatrixBase_selfadjointView.cpp
 * Output: \verbinclude MatrixBase_selfadjointView.out
 *
 * \sa class SelfAdjointView
 */
template <typename Derived>
template <unsigned int UpLo>
EIGEN_DEVICE_FUNC constexpr typename MatrixBase<Derived>::template SelfAdjointViewReturnType<UpLo>::Type
MatrixBase<Derived>::selfadjointView() {
  return typename SelfAdjointViewReturnType<UpLo>::Type(derived());
}

}  // end namespace Eigen

#endif  // EIGEN_SELFADJOINTMATRIX_H
