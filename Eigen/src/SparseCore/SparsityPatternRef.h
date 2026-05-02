// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSITY_PATTERN_REF_H
#define EIGEN_SPARSITY_PATTERN_REF_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

/** \internal
 * \ingroup SparseCore_Module
 *
 * Non-owning view of a column-major sparsity pattern. The pattern is encoded
 * as CSC index arrays:
 *  - \c outer (size \c outerSize+1 when compressed) gives the start of each
 *    column's row-index range in \c inner.
 *  - \c inner holds the row indices.
 *  - \c innerNonZero is non-null only for an uncompressed source: in that
 *    case column \c j's row indices live in
 *    [outer[j], outer[j] + innerNonZero[j]) instead of [outer[j], outer[j+1]).
 *
 * Use the helper \ref nonZeros to iterate uniformly over both layouts:
 * \code
 *   const Index nz = pattern.nonZeros(j);
 *   for (Index k = pattern.outer[j], end = k + nz; k < end; ++k) {
 *     Index row = pattern.inner[k];
 *   }
 * \endcode
 *
 * Construct via \ref make_col_major_pattern_ref, which shares the source
 * storage when possible (already column-major SparseMatrix) and otherwise
 * materializes the pattern into caller-supplied scratch buffers without ever
 * reading the source's scalar values.
 */
template <typename StorageIndex>
struct SparsityPatternRef {
  const StorageIndex* outer = nullptr;
  const StorageIndex* inner = nullptr;
  // null ⇒ source is compressed: column j is [outer[j], outer[j+1]).
  // otherwise: column j is [outer[j], outer[j] + innerNonZero[j]).
  const StorageIndex* innerNonZero = nullptr;
  Index outerSize = 0;  // number of columns
  Index innerSize = 0;  // number of rows

  EIGEN_DEVICE_FUNC inline bool isCompressed() const { return innerNonZero == nullptr; }

  EIGEN_DEVICE_FUNC inline Index nonZeros(Index j) const {
    return isCompressed() ? Index(outer[j + 1] - outer[j]) : Index(innerNonZero[j]);
  }
};

/** \internal
 * Build a column-major sparsity-pattern view of \c amat without copying any
 * scalar values. The returned view aliases \c amat's index storage when
 * \c amat is already a column-major \c SparseMatrix; otherwise the pattern is
 * materialized into the caller-supplied \c outer_buf / \c inner_buf and the
 * returned view points into them.
 *
 * The buffers must outlive every use of the returned view.
 *
 * Specialization for column-major \c SparseMatrix (the no-copy fast path).
 */
template <typename Scalar, typename StorageIndex>
SparsityPatternRef<StorageIndex> make_col_major_pattern_ref(const SparseMatrix<Scalar, ColMajor, StorageIndex>& amat,
                                                            Matrix<StorageIndex, Dynamic, 1>& /*outer_buf*/,
                                                            Matrix<StorageIndex, Dynamic, 1>& /*inner_buf*/) {
  SparsityPatternRef<StorageIndex> p;
  p.outer = amat.outerIndexPtr();
  p.inner = amat.innerIndexPtr();
  p.innerNonZero = amat.isCompressed() ? nullptr : amat.innerNonZeroPtr();
  p.outerSize = amat.cols();
  p.innerSize = amat.rows();
  return p;
}

/** \internal
 * Generic fallback for any other sparse expression (row-major, products,
 * permutations, etc.). The pattern is materialized into \c outer_buf and
 * \c inner_buf via a value-free counting transpose driven by the expression
 * evaluator. Some evaluators may materialize internally.
 */
template <typename Derived>
SparsityPatternRef<typename Derived::StorageIndex> make_col_major_pattern_ref(
    const SparseMatrixBase<Derived>& amat_base, Matrix<typename Derived::StorageIndex, Dynamic, 1>& outer_buf,
    Matrix<typename Derived::StorageIndex, Dynamic, 1>& inner_buf) {
  typedef typename Derived::StorageIndex StorageIndex;
  const Derived& amat = amat_base.derived();
  internal::evaluator<Derived> amat_eval(amat);
  const Index n_cols = amat.cols();
  const Index n_outer = amat.outerSize();

  outer_buf.setZero(n_cols + 1);
  for (Index i = 0; i < n_outer; ++i)
    for (typename internal::evaluator<Derived>::InnerIterator it(amat_eval, i); it; ++it) ++outer_buf(it.col() + 1);
  for (Index j = 0; j < n_cols; ++j) outer_buf(j + 1) += outer_buf(j);
  inner_buf.resize(outer_buf(n_cols));
  Matrix<StorageIndex, Dynamic, 1> head = outer_buf.head(n_cols);
  for (Index i = 0; i < n_outer; ++i)
    for (typename internal::evaluator<Derived>::InnerIterator it(amat_eval, i); it; ++it)
      inner_buf(head(it.col())++) = convert_index<StorageIndex>(it.row());

  SparsityPatternRef<StorageIndex> p;
  p.outer = outer_buf.data();
  p.inner = inner_buf.data();
  p.innerNonZero = nullptr;
  p.outerSize = n_cols;
  p.innerSize = amat.rows();
  return p;
}

/** \internal
 * Materialize a column-major sparsity pattern view as a compressed
 * \c SparseMatrix with a 1-byte placeholder \c Scalar. Values are filled with
 * a fixed nonzero sentinel — downstream pattern consumers (\c transpose(),
 * sparse \c operator+, \c selfadjointView expansion) read coefficients even
 * when the algorithm only cares about the pattern, so leaving them
 * uninitialized would be UB.
 *
 * Intended for fill-reducing ordering algorithms (AMD, COLAMD) that read only
 * the pattern. \b Precondition: \c out's storage must not alias \c pat —
 * \c resize / \c resizeNonZeros may reallocate and invalidate the view.
 *
 * If \c row_perm is non-null, each inner row index is remapped through it
 * (entry \c (i, j) of \c pat becomes \c (row_perm[i], j) in the result),
 * implementing a left multiplication by a row permutation without copying any
 * scalar values from the source.
 *
 * The result preserves SparseMatrix's invariant that inner indices are sorted
 * within each column.
 */
template <typename StorageIndex>
void materialize_col_major_pattern(const SparsityPatternRef<StorageIndex>& pat, const StorageIndex* row_perm,
                                   SparseMatrix<signed char, ColMajor, StorageIndex>& out) {
  const Index n_outer = pat.outerSize;
  const Index n_inner = pat.innerSize;
  out.resize(n_inner, n_outer);

  Index total = 0;
  for (Index j = 0; j < n_outer; ++j) total += pat.nonZeros(j);
  out.resizeNonZeros(total);

  StorageIndex* o_outer = out.outerIndexPtr();
  StorageIndex* o_inner = out.innerIndexPtr();
  o_outer[0] = 0;
  for (Index j = 0; j < n_outer; ++j) {
    const Index nz = pat.nonZeros(j);
    o_outer[j + 1] = convert_index<StorageIndex>(o_outer[j] + nz);
    const StorageIndex* src = pat.inner + pat.outer[j];
    StorageIndex* dst = o_inner + o_outer[j];
    if (row_perm == nullptr) {
      std::copy(src, src + nz, dst);
    } else {
      for (Index k = 0; k < nz; ++k) dst[k] = row_perm[src[k]];
      std::sort(dst, dst + nz);
    }
  }
  // Fill values with a fixed nonzero sentinel so downstream consumers that
  // read coefficients (transpose, sparse +, selfadjointView expansion) do not
  // observe uninitialized memory.
  std::fill_n(out.valuePtr(), total, static_cast<signed char>(1));
}

/** \internal Convenience overload — identity row permutation. */
template <typename StorageIndex>
void materialize_col_major_pattern(const SparsityPatternRef<StorageIndex>& pat,
                                   SparseMatrix<signed char, ColMajor, StorageIndex>& out) {
  materialize_col_major_pattern(pat, static_cast<const StorageIndex*>(nullptr), out);
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_SPARSITY_PATTERN_REF_H
