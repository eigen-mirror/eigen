// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#if !defined(EIGEN_PARSED_BY_DOXYGEN)

public:
// define some aliases to ease readability

template <typename Indices>
using IvcRowType = typename internal::IndexedViewHelperIndicesWrapper<Indices, RowsAtCompileTime>::type;

template <typename Indices>
using IvcColType = typename internal::IndexedViewHelperIndicesWrapper<Indices, ColsAtCompileTime>::type;

template <typename Indices>
using IvcSizeType = typename internal::IndexedViewHelperIndicesWrapper<Indices, SizeAtCompileTime>::type;

template <typename Indices>
inline IvcRowType<Indices> ivcRow(const Indices& indices) const {
  return internal::IndexedViewHelperIndicesWrapper<Indices, RowsAtCompileTime>::CreateIndexSequence(indices,
                                                                                                    derived().rows());
}

template <typename Indices>
inline IvcColType<Indices> ivcCol(const Indices& indices) const {
  return internal::IndexedViewHelperIndicesWrapper<Indices, ColsAtCompileTime>::CreateIndexSequence(indices,
                                                                                                    derived().cols());
}

template <typename Indices>
inline IvcSizeType<Indices> ivcSize(const Indices& indices) const {
  return internal::IndexedViewHelperIndicesWrapper<Indices, SizeAtCompileTime>::CreateIndexSequence(indices,
                                                                                                    derived().size());
  ;
}

// this helper class assumes internal::valid_indexed_view_overload<RowIndices, ColIndices>::value == true
template <typename RowIndices, typename ColIndices, typename EnableIf = void>
struct IndexedViewSelector;

// Generic
template <typename RowIndices, typename ColIndices>
struct IndexedViewSelector<
    RowIndices, ColIndices,
    std::enable_if_t<
        internal::traits<IndexedView<Derived, IvcRowType<RowIndices>, IvcColType<ColIndices>>>::ReturnAsIndexedView>> {
  using ReturnType = IndexedView<Derived, IvcRowType<RowIndices>, IvcColType<ColIndices>>;
  using ConstReturnType = IndexedView<const Derived, IvcRowType<RowIndices>, IvcColType<ColIndices>>;

  static inline ReturnType run(Derived& derived, const RowIndices& rowIndices, const ColIndices& colIndices) {
    return ReturnType(derived, derived.ivcRow(rowIndices), derived.ivcCol(colIndices));
  }
  static inline ConstReturnType run(const Derived& derived, const RowIndices& rowIndices,
                                    const ColIndices& colIndices) {
    return ConstReturnType(derived, derived.ivcRow(rowIndices), derived.ivcCol(colIndices));
  }
};

// Block
template <typename RowIndices, typename ColIndices>
struct IndexedViewSelector<RowIndices, ColIndices,
                           std::enable_if_t<internal::traits<
                               IndexedView<Derived, IvcRowType<RowIndices>, IvcColType<ColIndices>>>::ReturnAsBlock>> {
  using ActualRowIndices = IvcRowType<RowIndices>;
  using ActualColIndices = IvcColType<ColIndices>;
  using IndexedViewType = IndexedView<Derived, ActualRowIndices, ActualColIndices>;
  using ConstIndexedViewType = IndexedView<const Derived, ActualRowIndices, ActualColIndices>;
  using ReturnType = typename internal::traits<IndexedViewType>::BlockType;
  using ConstReturnType = typename internal::traits<ConstIndexedViewType>::BlockType;
  using RowHelper = internal::IndexedViewHelper<ActualRowIndices>;
  using ColHelper = internal::IndexedViewHelper<ActualColIndices>;

  static inline ReturnType run(Derived& derived, const RowIndices& rowIndices, const ColIndices& colIndices) {
    auto actualRowIndices = derived.ivcRow(rowIndices);
    auto actualColIndices = derived.ivcCol(colIndices);
    return ReturnType(derived, RowHelper::first(actualRowIndices), ColHelper::first(actualColIndices),
                      RowHelper::size(actualRowIndices), ColHelper::size(actualColIndices));
  }
  static inline ConstReturnType run(const Derived& derived, const RowIndices& rowIndices,
                                    const ColIndices& colIndices) {
    auto actualRowIndices = derived.ivcRow(rowIndices);
    auto actualColIndices = derived.ivcCol(colIndices);
    return ConstReturnType(derived, RowHelper::first(actualRowIndices), ColHelper::first(actualColIndices),
                           RowHelper::size(actualRowIndices), ColHelper::size(actualColIndices));
  }
};

// Scalar
template <typename RowIndices, typename ColIndices>
struct IndexedViewSelector<RowIndices, ColIndices,
                           std::enable_if_t<internal::traits<
                               IndexedView<Derived, IvcRowType<RowIndices>, IvcColType<ColIndices>>>::ReturnAsScalar>> {
  using ReturnType = typename DenseBase<Derived>::Scalar&;
  using ConstReturnType = typename DenseBase<Derived>::CoeffReturnType;
  using ActualRowIndices = IvcRowType<RowIndices>;
  using ActualColIndices = IvcColType<ColIndices>;
  using RowHelper = internal::IndexedViewHelper<ActualRowIndices>;
  using ColHelper = internal::IndexedViewHelper<ActualColIndices>;
  static inline ReturnType run(Derived& derived, const RowIndices& rowIndices, const ColIndices& colIndices) {
    auto actualRowIndices = derived.ivcRow(rowIndices);
    auto actualColIndices = derived.ivcCol(colIndices);
    return derived(RowHelper::first(actualRowIndices), ColHelper::first(actualColIndices));
  }
  static inline ConstReturnType run(const Derived& derived, const RowIndices& rowIndices,
                                    const ColIndices& colIndices) {
    auto actualRowIndices = derived.ivcRow(rowIndices);
    auto actualColIndices = derived.ivcCol(colIndices);
    return derived(RowHelper::first(actualRowIndices), ColHelper::first(actualColIndices));
  }
};

// this helper class assumes internal::is_valid_index_type<Indices>::value == false
template <typename Indices, typename EnableIf = void>
struct VectorIndexedViewSelector;

// Generic
template <typename Indices>
struct VectorIndexedViewSelector<
    Indices, std::enable_if_t<!internal::is_single_range<IvcSizeType<Indices>>::value &&
                              internal::IndexedViewHelper<IvcSizeType<Indices>>::IncrAtCompileTime != 1>> {
  static constexpr bool IsRowMajor = DenseBase<Derived>::IsRowMajor;
  using ZeroIndex = internal::SingleRange<Index(0)>;
  using RowMajorReturnType = IndexedView<Derived, ZeroIndex, IvcSizeType<Indices>>;
  using ConstRowMajorReturnType = IndexedView<const Derived, ZeroIndex, IvcSizeType<Indices>>;

  using ColMajorReturnType = IndexedView<Derived, IvcSizeType<Indices>, ZeroIndex>;
  using ConstColMajorReturnType = IndexedView<const Derived, IvcSizeType<Indices>, ZeroIndex>;

  using ReturnType = typename internal::conditional<IsRowMajor, RowMajorReturnType, ColMajorReturnType>::type;
  using ConstReturnType =
      typename internal::conditional<IsRowMajor, ConstRowMajorReturnType, ConstColMajorReturnType>::type;

  template <bool UseRowMajor = IsRowMajor, std::enable_if_t<UseRowMajor, bool> = true>
  static inline RowMajorReturnType run(Derived& derived, const Indices& indices) {
    return RowMajorReturnType(derived, ZeroIndex(0), derived.ivcCol(indices));
  }
  template <bool UseRowMajor = IsRowMajor, std::enable_if_t<UseRowMajor, bool> = true>
  static inline ConstRowMajorReturnType run(const Derived& derived, const Indices& indices) {
    return ConstRowMajorReturnType(derived, ZeroIndex(0), derived.ivcCol(indices));
  }
  template <bool UseRowMajor = IsRowMajor, std::enable_if_t<!UseRowMajor, bool> = true>
  static inline ColMajorReturnType run(Derived& derived, const Indices& indices) {
    return ColMajorReturnType(derived, derived.ivcRow(indices), ZeroIndex(0));
  }
  template <bool UseRowMajor = IsRowMajor, std::enable_if_t<!UseRowMajor, bool> = true>
  static inline ConstColMajorReturnType run(const Derived& derived, const Indices& indices) {
    return ConstColMajorReturnType(derived, derived.ivcRow(indices), ZeroIndex(0));
  }
};

// Block
template <typename Indices>
struct VectorIndexedViewSelector<
    Indices, std::enable_if_t<!internal::is_single_range<IvcSizeType<Indices>>::value &&
                              internal::IndexedViewHelper<IvcSizeType<Indices>>::IncrAtCompileTime == 1>> {
  using Helper = internal::IndexedViewHelper<IvcSizeType<Indices>>;
  using ReturnType = VectorBlock<Derived, Helper::SizeAtCompileTime>;
  using ConstReturnType = VectorBlock<const Derived, Helper::SizeAtCompileTime>;
  static inline ReturnType run(Derived& derived, const Indices& indices) {
    auto actualIndices = derived.ivcSize(indices);
    return ReturnType(derived, Helper::first(actualIndices), Helper::size(actualIndices));
  }
  static inline ConstReturnType run(const Derived& derived, const Indices& indices) {
    auto actualIndices = derived.ivcSize(indices);
    return ConstReturnType(derived, Helper::first(actualIndices), Helper::size(actualIndices));
  }
};

// Symbolic
template <typename Indices>
struct VectorIndexedViewSelector<Indices, std::enable_if_t<internal::is_single_range<IvcSizeType<Indices>>::value>> {
  using ReturnType = typename DenseBase<Derived>::Scalar&;
  using ConstReturnType = typename DenseBase<Derived>::CoeffReturnType;
  using Helper = internal::IndexedViewHelper<IvcSizeType<Indices>>;
  static inline ReturnType run(Derived& derived, const Indices& indices) {
    auto actualIndices = derived.ivcSize(indices);
    return derived(Helper::first(actualIndices));
  }
  static inline ConstReturnType run(const Derived& derived, const Indices& indices) {
    auto actualIndices = derived.ivcSize(indices);
    return derived(Helper::first(actualIndices));
  }
};

// SFINAE dummy types

template <typename RowIndices, typename ColIndices>
using EnableOverload = std::enable_if_t<
    internal::valid_indexed_view_overload<RowIndices, ColIndices>::value && internal::is_lvalue<Derived>::value, bool>;

template <typename RowIndices, typename ColIndices>
using EnableConstOverload =
    std::enable_if_t<internal::valid_indexed_view_overload<RowIndices, ColIndices>::value, bool>;

template <typename Indices>
using EnableVectorOverload =
    std::enable_if_t<!internal::is_valid_index_type<Indices>::value && internal::is_lvalue<Derived>::value, bool>;

template <typename Indices>
using EnableConstVectorOverload = std::enable_if_t<!internal::is_valid_index_type<Indices>::value, bool>;

public:
// Public API for 2D matrices/arrays

// non-const versions

template <typename RowIndices, typename ColIndices>
using IndexedViewType = typename IndexedViewSelector<RowIndices, ColIndices>::ReturnType;

template <typename RowIndices, typename ColIndices, EnableOverload<RowIndices, ColIndices> = true>
IndexedViewType<RowIndices, ColIndices> operator()(const RowIndices& rowIndices, const ColIndices& colIndices) {
  return IndexedViewSelector<RowIndices, ColIndices>::run(derived(), rowIndices, colIndices);
}

template <typename RowType, size_t RowSize, typename ColIndices, typename RowIndices = Array<RowType, RowSize, 1>,
          EnableOverload<RowIndices, ColIndices> = true>
IndexedViewType<RowIndices, ColIndices> operator()(const RowType (&rowIndices)[RowSize], const ColIndices& colIndices) {
  return IndexedViewSelector<RowIndices, ColIndices>::run(derived(), RowIndices{rowIndices}, colIndices);
}

template <typename RowIndices, typename ColType, size_t ColSize, typename ColIndices = Array<ColType, ColSize, 1>,
          EnableOverload<RowIndices, ColIndices> = true>
IndexedViewType<RowIndices, ColIndices> operator()(const RowIndices& rowIndices, const ColType (&colIndices)[ColSize]) {
  return IndexedViewSelector<RowIndices, ColIndices>::run(derived(), rowIndices, ColIndices{colIndices});
}

template <typename RowType, size_t RowSize, typename ColType, size_t ColSize,
          typename RowIndices = Array<RowType, RowSize, 1>, typename ColIndices = Array<ColType, ColSize, 1>,
          EnableOverload<RowIndices, ColIndices> = true>
IndexedViewType<RowIndices, ColIndices> operator()(const RowType (&rowIndices)[RowSize],
                                                   const ColType (&colIndices)[ColSize]) {
  return IndexedViewSelector<RowIndices, ColIndices>::run(derived(), RowIndices{rowIndices}, ColIndices{colIndices});
}

// const versions

template <typename RowIndices, typename ColIndices>
using ConstIndexedViewType = typename IndexedViewSelector<RowIndices, ColIndices>::ConstReturnType;

template <typename RowIndices, typename ColIndices, EnableConstOverload<RowIndices, ColIndices> = true>
ConstIndexedViewType<RowIndices, ColIndices> operator()(const RowIndices& rowIndices,
                                                        const ColIndices& colIndices) const {
  return IndexedViewSelector<RowIndices, ColIndices>::run(derived(), rowIndices, colIndices);
}

template <typename RowType, size_t RowSize, typename ColIndices, typename RowIndices = Array<RowType, RowSize, 1>,
          EnableConstOverload<RowIndices, ColIndices> = true>
ConstIndexedViewType<RowIndices, ColIndices> operator()(const RowType (&rowIndices)[RowSize],
                                                        const ColIndices& colIndices) const {
  return IndexedViewSelector<RowIndices, ColIndices>::run(derived(), RowIndices{rowIndices}, colIndices);
}

template <typename RowIndices, typename ColType, size_t ColSize, typename ColIndices = Array<ColType, ColSize, 1>,
          EnableConstOverload<RowIndices, ColIndices> = true>
ConstIndexedViewType<RowIndices, ColIndices> operator()(const RowIndices& rowIndices,
                                                        const ColType (&colIndices)[ColSize]) const {
  return IndexedViewSelector<RowIndices, ColIndices>::run(derived(), rowIndices, ColIndices{colIndices});
}

template <typename RowType, size_t RowSize, typename ColType, size_t ColSize,
          typename RowIndices = Array<RowType, RowSize, 1>, typename ColIndices = Array<ColType, ColSize, 1>,
          EnableConstOverload<RowIndices, ColIndices> = true>
ConstIndexedViewType<RowIndices, ColIndices> operator()(const RowType (&rowIndices)[RowSize],
                                                        const ColType (&colIndices)[ColSize]) const {
  return IndexedViewSelector<RowIndices, ColIndices>::run(derived(), RowIndices{rowIndices}, ColIndices{colIndices});
}

// Public API for 1D vectors/arrays

// non-const versions

template <typename Indices>
using VectorIndexedViewType = typename VectorIndexedViewSelector<Indices>::ReturnType;

template <typename Indices, EnableVectorOverload<Indices> = true>
VectorIndexedViewType<Indices> operator()(const Indices& indices) {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return VectorIndexedViewSelector<Indices>::run(derived(), indices);
}

template <typename IndexType, size_t Size, typename Indices = Array<IndexType, Size, 1>,
          EnableVectorOverload<Indices> = true>
VectorIndexedViewType<Indices> operator()(const IndexType (&indices)[Size]) {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return VectorIndexedViewSelector<Indices>::run(derived(), Indices{indices});
}

// const versions

template <typename Indices>
using ConstVectorIndexedViewType = typename VectorIndexedViewSelector<Indices>::ConstReturnType;

template <typename Indices, EnableConstVectorOverload<Indices> = true>
ConstVectorIndexedViewType<Indices> operator()(const Indices& indices) const {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return VectorIndexedViewSelector<Indices>::run(derived(), indices);
}

template <typename IndexType, size_t Size, typename Indices = Array<IndexType, Size, 1>,
          EnableConstVectorOverload<Indices> = true>
ConstVectorIndexedViewType<Indices> operator()(const IndexType (&indices)[Size]) const {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return VectorIndexedViewSelector<Indices>::run(derived(), Indices{indices});
}

#else  // EIGEN_PARSED_BY_DOXYGEN

/**
 * \returns a generic submatrix view defined by the rows and columns indexed \a rowIndices and \a colIndices
 * respectively.
 *
 * Each parameter must either be:
 *  - An integer indexing a single row or column
 *  - Eigen::placeholders::all indexing the full set of respective rows or columns in increasing order
 *  - An ArithmeticSequence as returned by the Eigen::seq and Eigen::seqN functions
 *  - Any %Eigen's vector/array of integers or expressions
 *  - Plain C arrays: \c int[N]
 *  - And more generally any type exposing the following two member functions:
 * \code
 * <integral type> operator[](<integral type>) const;
 * <integral type> size() const;
 * \endcode
 * where \c <integral \c type>  stands for any integer type compatible with Eigen::Index (i.e. \c std::ptrdiff_t).
 *
 * The last statement implies compatibility with \c std::vector, \c std::valarray, \c std::array, many of the Range-v3's
 * ranges, etc.
 *
 * If the submatrix can be represented using a starting position \c (i,j) and positive sizes \c (rows,columns), then
 * this method will returns a Block object after extraction of the relevant information from the passed arguments. This
 * is the case when all arguments are either:
 *  - An integer
 *  - Eigen::placeholders::all
 *  - An ArithmeticSequence with compile-time increment strictly equal to 1, as returned by Eigen::seq(a,b), and
 * Eigen::seqN(a,N).
 *
 * Otherwise a more general IndexedView<Derived,RowIndices',ColIndices'> object will be returned, after conversion of
 * the inputs to more suitable types \c RowIndices' and \c ColIndices'.
 *
 * For 1D vectors and arrays, you better use the operator()(const Indices&) overload, which behave the same way but
 * taking a single parameter.
 *
 * See also this <a
 * href="https://stackoverflow.com/questions/46110917/eigen-replicate-items-along-one-dimension-without-useless-allocations">question</a>
 * and its answer for an example of how to duplicate coefficients.
 *
 * \sa operator()(const Indices&), class Block, class IndexedView, DenseBase::block(Index,Index,Index,Index)
 */
template <typename RowIndices, typename ColIndices>
IndexedView_or_Block operator()(const RowIndices& rowIndices, const ColIndices& colIndices);

/** This is an overload of operator()(const RowIndices&, const ColIndices&) for 1D vectors or arrays
 *
 * \only_for_vectors
 */
template <typename Indices>
IndexedView_or_VectorBlock operator()(const Indices& indices);

#endif  // EIGEN_PARSED_BY_DOXYGEN
