// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_STRUCTURED_KRONECKER_SPARSE_VIEW_H
#define EIGEN_STRUCTURED_KRONECKER_SPARSE_VIEW_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/** \internal The widest StorageIndex of the sparse leaves of \a Factor, \c int
 * when it has none. */
template <typename Factor, int Kind = kron_factor_kind<Factor>()>
struct kron_view_storage_index {
  using type = int;
};
template <typename Scalar, int Options, typename StorageIndex>
struct kron_view_storage_index<SparseMatrix<Scalar, Options, StorageIndex>, kKronSparseFactor> {
  using type = StorageIndex;
};
template <typename LhsMatrix, typename RhsMatrix>
struct kron_view_storage_index<KroneckerOperator<LhsMatrix, RhsMatrix>, kKronKroneckerFactor>
    : promote_index_type<typename kron_view_storage_index<LhsMatrix>::type,
                         typename kron_view_storage_index<RhsMatrix>::type> {};
template <typename LhsMatrix, typename RhsMatrix>
struct kron_view_storage_index<KroneckerSum<LhsMatrix, RhsMatrix>, kKronSumFactor>
    : promote_index_type<typename kron_view_storage_index<LhsMatrix>::type,
                         typename kron_view_storage_index<RhsMatrix>::type> {};

template <typename OperatorType, int Options>
struct traits<KroneckerSparseView<OperatorType, Options>> {
  using Scalar = typename OperatorType::Scalar;
  using StorageKind = Sparse;
  using XprKind = MatrixXpr;
  // The index type of the plain matrices Eigen evaluates the view into (eval(),
  // product temporaries), which the user does not choose.
  using StorageIndex = typename kron_view_storage_index<OperatorType>::type;
  static constexpr int RowsAtCompileTime = OperatorType::RowsAtCompileTime;
  static constexpr int ColsAtCompileTime = OperatorType::ColsAtCompileTime;
  static constexpr int MaxRowsAtCompileTime = OperatorType::MaxRowsAtCompileTime;
  static constexpr int MaxColsAtCompileTime = OperatorType::MaxColsAtCompileTime;
  // Nested by reference like a SparseMatrix: the view owns a copy of the
  // operator, which a by-value nesting would copy into every expression.
  static constexpr unsigned int Flags = (unsigned(Options) & RowMajorBit) | NestByRefBit;
};

/** \internal The factor type the view stores for \a Factor: sparse leaves in
 * the view's storage order, so that every inner vector is a contiguous
 * iteration; every other kind as it is. */
template <typename Factor, int Options, int Kind = kron_factor_kind<Factor>()>
struct kron_view_factor {
  using type = Factor;
  static const Factor& convert(const Factor& f) { return f; }
};
template <typename Scalar, int FactorOptions, typename StorageIndex, int Options>
struct kron_view_factor<SparseMatrix<Scalar, FactorOptions, StorageIndex>, Options, kKronSparseFactor> {
  using type = SparseMatrix<Scalar, Options & RowMajorBit ? RowMajor : ColMajor, StorageIndex>;
  // A same-order copy keeps the factor's entry order, and a compressed matrix
  // may hold unsorted inner vectors; the iterators need them sorted.
  static type convert(const SparseMatrix<Scalar, FactorOptions, StorageIndex>& f) {
    type g(f);
    if (g.innerIndicesAreSorted() != g.outerSize()) g.sortInnerIndices();
    return g;
  }
};
template <typename LhsMatrix, typename RhsMatrix, int Options>
struct kron_view_factor<KroneckerOperator<LhsMatrix, RhsMatrix>, Options, kKronKroneckerFactor> {
  using Lhs = kron_view_factor<LhsMatrix, Options>;
  using Rhs = kron_view_factor<RhsMatrix, Options>;
  using type = KroneckerOperator<typename Lhs::type, typename Rhs::type>;
  static type convert(const KroneckerOperator<LhsMatrix, RhsMatrix>& f) {
    return type(Lhs::convert(f.lhs()), Rhs::convert(f.rhs()));
  }
};
template <typename LhsMatrix, typename RhsMatrix, int Options>
struct kron_view_factor<KroneckerSum<LhsMatrix, RhsMatrix>, Options, kKronSumFactor> {
  using Lhs = kron_view_factor<LhsMatrix, Options>;
  using Rhs = kron_view_factor<RhsMatrix, Options>;
  using type = KroneckerSum<typename Lhs::type, typename Rhs::type>;
  static type convert(const KroneckerSum<LhsMatrix, RhsMatrix>& f) {
    return type(Lhs::convert(f.lhs()), Rhs::convert(f.rhs()));
  }
};

/** \internal Iterates the stored entries of inner vector \a outer of a factor --
 * column \a outer when \a RowMajor is false, row \a outer otherwise -- in
 * increasing inner index. Dense factor: every entry. */
template <typename Factor, bool RowMajor, int Kind = kron_factor_kind<Factor>()>
class kron_view_iterator {
 public:
  using Scalar = typename Factor::Scalar;
  kron_view_iterator(const Factor& f, Index outer)
      : m_f(&f), m_outer(outer), m_inner(0), m_end(RowMajor ? f.cols() : f.rows()) {}
  explicit operator bool() const { return m_inner < m_end; }
  kron_view_iterator& operator++() {
    ++m_inner;
    return *this;
  }
  Index index() const { return m_inner; }
  Scalar value() const { return RowMajor ? m_f->coeff(m_outer, m_inner) : m_f->coeff(m_inner, m_outer); }

 private:
  const Factor* m_f;
  Index m_outer, m_inner, m_end;
};

// Diagonal and identity factors: at most the diagonal entry.
template <typename Factor, bool RowMajor>
class kron_view_iterator<Factor, RowMajor, kKronDiagonalFactor> {
 public:
  using Scalar = typename Factor::Scalar;
  kron_view_iterator(const Factor& f, Index outer) : m_f(&f), m_outer(outer), m_valid(true) {}
  explicit operator bool() const { return m_valid; }
  kron_view_iterator& operator++() {
    m_valid = false;
    return *this;
  }
  Index index() const { return m_outer; }
  Scalar value() const { return m_f->diagonal().coeff(m_outer); }

 private:
  const Factor* m_f;
  Index m_outer;
  bool m_valid;
};

template <typename Factor, bool RowMajor>
class kron_view_iterator<Factor, RowMajor, kKronIdentityFactor> {
 public:
  using Scalar = typename Factor::Scalar;
  kron_view_iterator(const Factor& f, Index outer)
      : m_outer(outer), m_valid(outer < numext::mini(f.rows(), f.cols())) {}
  explicit operator bool() const { return m_valid; }
  kron_view_iterator& operator++() {
    m_valid = false;
    return *this;
  }
  Index index() const { return m_outer; }
  Scalar value() const { return Scalar(1); }

 private:
  Index m_outer;
  bool m_valid;
};

// Sparse factor, compressed and stored in the view's order (see
// kron_view_factor). Raw pointers rather than the matrix's InnerIterator keep
// the iterator assignable, which the Kronecker iterator needs to restart it.
template <typename Factor, bool RowMajor>
class kron_view_iterator<Factor, RowMajor, kKronSparseFactor> {
 public:
  using Scalar = typename Factor::Scalar;
  static_assert(bool(Factor::IsRowMajor) == RowMajor, "the view converts sparse factors to its storage order");
  kron_view_iterator(const Factor& f, Index outer)
      : m_values(f.valuePtr()),
        m_indices(f.innerIndexPtr()),
        m_pos(f.outerIndexPtr()[outer]),
        m_end(f.outerIndexPtr()[outer + 1]) {
    eigen_internal_assert(f.isCompressed());
  }
  explicit operator bool() const { return m_pos < m_end; }
  kron_view_iterator& operator++() {
    ++m_pos;
    return *this;
  }
  Index index() const { return m_indices[m_pos]; }
  Scalar value() const { return m_values[m_pos]; }

 private:
  const Scalar* m_values;
  const typename Factor::StorageIndex* m_indices;
  Index m_pos, m_end;
};

/** \internal Kronecker factor K = L (x) R: inner vector kL * outR + kR of K, with
 * outR and inR the outer and inner sizes of R, is the product of inner vectors
 * kL of L and kR of R; their entries meet at inner index iL * inR + iR,
 * increasing when the loop over L is outermost. */
template <typename LhsMatrix, typename RhsMatrix, bool RowMajor>
class kron_view_iterator<KroneckerOperator<LhsMatrix, RhsMatrix>, RowMajor, kKronKroneckerFactor> {
 public:
  using Factor = KroneckerOperator<LhsMatrix, RhsMatrix>;
  using Scalar = typename Factor::Scalar;
  kron_view_iterator(const Factor& f, Index outer)
      : m_rhs(&f.rhs()),
        m_outerR(outer % outerSize(f.rhs())),
        m_innerR(RowMajor ? f.rhs().cols() : f.rhs().rows()),
        m_itL(f.lhs(), outer / outerSize(f.rhs())),
        m_itR(f.rhs(), m_outerR),
        m_nonEmptyR(bool(m_itR)) {}
  explicit operator bool() const { return m_nonEmptyR && bool(m_itL); }
  kron_view_iterator& operator++() {
    ++m_itR;
    if (!m_itR) {
      ++m_itL;
      if (m_itL) m_itR = RhsIterator(*m_rhs, m_outerR);
    }
    return *this;
  }
  Index index() const { return m_itL.index() * m_innerR + m_itR.index(); }
  Scalar value() const { return m_itL.value() * m_itR.value(); }

 private:
  using LhsIterator = kron_view_iterator<LhsMatrix, RowMajor>;
  using RhsIterator = kron_view_iterator<RhsMatrix, RowMajor>;
  template <typename F>
  static Index outerSize(const F& f) {
    return RowMajor ? f.rows() : f.cols();
  }

  const RhsMatrix* m_rhs;
  Index m_outerR, m_innerR;
  LhsIterator m_itL;
  RhsIterator m_itR;
  bool m_nonEmptyR;
};

/** \internal Kronecker sum S = L (+) R with n2 the size of R: inner vector
 * kL * n2 + kR merges the entries of L (x) I, at iL * n2 + kR for the stored iL
 * of inner vector kL of L, with those of I (x) R, at kL * n2 + iR for the stored
 * iR of inner vector kR of R; the two streams meet on the diagonal. */
template <typename LhsMatrix, typename RhsMatrix, bool RowMajor>
class kron_view_iterator<KroneckerSum<LhsMatrix, RhsMatrix>, RowMajor, kKronSumFactor> {
 public:
  using Factor = KroneckerSum<LhsMatrix, RhsMatrix>;
  using Scalar = typename Factor::Scalar;
  kron_view_iterator(const Factor& f, Index outer)
      : m_n2(f.rhs().rows()),
        m_outerL(outer / m_n2),
        m_outerR(outer % m_n2),
        m_itL(f.lhs(), m_outerL),
        m_itR(f.rhs(), m_outerR) {}
  explicit operator bool() const { return bool(m_itL) || bool(m_itR); }
  kron_view_iterator& operator++() {
    const Index current = index();
    if (indexL() == current) ++m_itL;
    if (indexR() == current) ++m_itR;
    return *this;
  }
  Index index() const { return numext::mini(indexL(), indexR()); }
  // The values of the materialization (A (x) I) + (I (x) B), signed zeros
  // included: a + b on the diagonal, a + 0 and 0 + b off it.
  Scalar value() const {
    const Index l = indexL(), r = indexR();
    if (l == r) return m_itL.value() + m_itR.value();
    return l < r ? m_itL.value() + Scalar(0) : Scalar(0) + m_itR.value();
  }

 private:
  using LhsIterator = kron_view_iterator<LhsMatrix, RowMajor>;
  using RhsIterator = kron_view_iterator<RhsMatrix, RowMajor>;
  Index indexL() const { return m_itL ? m_itL.index() * m_n2 + m_outerR : NumTraits<Index>::highest(); }
  Index indexR() const { return m_itR ? m_outerL * m_n2 + m_itR.index() : NumTraits<Index>::highest(); }

  Index m_n2, m_outerL, m_outerR;
  LhsIterator m_itL;
  RhsIterator m_itR;
};

/** \internal The stored-entry count of the view: exact for Kronecker products,
 * an upper bound for Kronecker sums (whose two terms share the diagonal). */
template <typename Factor, int Kind = kron_factor_kind<Factor>()>
struct kron_view_nonzeros {
  static Index run(const Factor& f) { return kron_factor_ops<Factor>::innerNonZeros(f, false).sum(); }
};
template <typename LhsMatrix, typename RhsMatrix>
struct kron_view_nonzeros<KroneckerOperator<LhsMatrix, RhsMatrix>, kKronKroneckerFactor> {
  static Index run(const KroneckerOperator<LhsMatrix, RhsMatrix>& f) {
    return kron_view_nonzeros<LhsMatrix>::run(f.lhs()) * kron_view_nonzeros<RhsMatrix>::run(f.rhs());
  }
};
template <typename LhsMatrix, typename RhsMatrix>
struct kron_view_nonzeros<KroneckerSum<LhsMatrix, RhsMatrix>, kKronSumFactor> {
  static Index run(const KroneckerSum<LhsMatrix, RhsMatrix>& f) {
    return kron_view_nonzeros<LhsMatrix>::run(f.lhs()) * f.rhs().rows() +
           f.lhs().rows() * kron_view_nonzeros<RhsMatrix>::run(f.rhs());
  }
};

}  // namespace internal

/** \ingroup StructuredMatrices_Module
 * \class KroneckerSparseView
 * \brief A \ref KroneckerOperator or \ref KroneckerSum as a lazy sparse
 * expression.
 *
 * The view is a \c SparseMatrixBase expression whose inner vectors are
 * iterated straight from the factors, so the operator takes part in sparse
 * expressions, products and assignments without being materialized first.
 * A product follows Eigen's cost model for its operands like any sparse
 * expression: a sparse-sparse product, or one with a row-major dense
 * right-hand side, revisits every inner vector of the view and evaluates it
 * into a temporary first when an entry costs more to recompute than to read
 * (complex scalars, whose multiply is dearer than a read).
 * \code
 * auto L = makeKroneckerSum(Dy, Dx);                      // 2-D Laplacian
 * SparseMatrix<double> M = Id - tau * L.sparseView();     // assembled once, no temporary for L
 * SparseMatrix<double> P = L.sparseView() * S;            // sparse-sparse product
 * \endcode
 * Inner vector \c kA*n2' + kB of \f$ A \otimes B \f$ (\c n2' the outer size of
 * \c B) is the product of inner vectors \c kA of \c A and \c kB of \c B, so an
 * iterator walks the stored entries of two factor inner vectors; for
 * \f$ A \oplus B \f$ it merges the entries of \f$ A \otimes I \f$ and
 * \f$ I \otimes B \f$. The view stores exactly the entries a sparse
 * materialization would: every stored entry of a factor, explicit zeros
 * included, and nothing is pruned. It is the operand for sparse expressions
 * and assembly; a product with a dense right-hand side is faster through the
 * operator itself, which applies whole factors instead of one entry at a time.
 *
 * The view owns a copy of the operator, with sparse factors converted to its
 * storage order \a Options (\c ColMajor or \c RowMajor, which must match the
 * other operands of a sparse binary expression). Like a \c SparseMatrix it is
 * nested by reference in the expressions built from it, so an expression must
 * not outlive a temporary view. Its \c StorageIndex, the index type of the
 * temporaries Eigen evaluates it into, is the widest of its sparse factors'
 * (\c int without any) and must hold its dimensions and stored-entry count;
 * 64-bit sparse factors give a 64-bit view. Its \c InnerIterator is also the
 * interface the coefficient-reading preconditioners take, e.g.
 * \c DiagonalPreconditioner::compute(view). The view is not a matrix type for
 * the iterative solvers themselves, which bind theirs through \c Ref; give them
 * the operator (matrix-free, \c IdentityPreconditioner) or a materialized
 * \c SparseMatrix.
 *
 * \tparam OperatorType the \ref KroneckerOperator or \ref KroneckerSum type.
 * \tparam Options \c ColMajor or \c RowMajor, the storage order of the view.
 *
 * \sa KroneckerOperator::sparseView(), KroneckerSum::sparseView()
 */
template <typename OperatorType, int Options>
class KroneckerSparseView : public SparseMatrixBase<KroneckerSparseView<OperatorType, Options>> {
  using ViewFactor = internal::kron_view_factor<OperatorType, Options>;

 public:
  using Base = SparseMatrixBase<KroneckerSparseView>;
  EIGEN_SPARSE_PUBLIC_INTERFACE(KroneckerSparseView)
  using ViewOperator = typename ViewFactor::type;
  static constexpr bool RowMajorView = (unsigned(Options) & RowMajorBit) != 0;

  explicit KroneckerSparseView(const OperatorType& op) : m_op(ViewFactor::convert(op)) {}

  Index rows() const { return m_op.rows(); }
  Index cols() const { return m_op.cols(); }

  /** \returns the operator the view iterates, with its sparse factors in the
   * view's storage order. */
  const ViewOperator& nestedOperator() const { return m_op; }

  /** Iterates the stored entries of an inner vector in increasing inner index. */
  class InnerIterator {
   public:
    InnerIterator(const KroneckerSparseView& view, Index outer) : m_it(view.m_op, outer), m_outer(outer) {}
    InnerIterator& operator++() {
      ++m_it;
      return *this;
    }
    operator bool() const { return bool(m_it); }
    Index index() const { return m_it.index(); }
    Index outer() const { return m_outer; }
    Index row() const { return RowMajorView ? m_outer : index(); }
    Index col() const { return RowMajorView ? index() : m_outer; }
    Scalar value() const { return m_it.value(); }

   private:
    internal::kron_view_iterator<ViewOperator, RowMajorView> m_it;
    Index m_outer;
  };

 private:
  ViewOperator m_op;
};

namespace internal {

template <typename OperatorType, int Options>
struct evaluator<KroneckerSparseView<OperatorType, Options>>
    : evaluator_base<KroneckerSparseView<OperatorType, Options>> {
  using XprType = KroneckerSparseView<OperatorType, Options>;
  using Scalar = typename XprType::Scalar;
  static constexpr int CoeffReadCost = NumTraits<Scalar>::MulCost;
  static constexpr unsigned int Flags = traits<XprType>::Flags;

  class InnerIterator : public XprType::InnerIterator {
   public:
    InnerIterator(const evaluator& eval, Index outer) : XprType::InnerIterator(eval.m_view, outer) {}
  };

  explicit evaluator(const XprType& xpr) : m_view(xpr) {}
  Index nonZerosEstimate() const {
    return kron_view_nonzeros<typename XprType::ViewOperator>::run(m_view.nestedOperator());
  }

  const XprType& m_view;
};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_STRUCTURED_KRONECKER_SPARSE_VIEW_H
