// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SHAPE_CHECKS_H
#define EIGEN_SHAPE_CHECKS_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

template <typename T, typename = void>
struct shape_expression {
  using type = T;
};
template <typename T>
struct shape_expression<T, void_t<decltype(std::declval<const T&>().derived())>> {
  using type = remove_all_t<decltype(std::declval<const T&>().derived())>;
};

// EigenBase extensions may omit either compile-time dimension.
template <typename T, typename = void>
struct shape_rows : std::integral_constant<int, Dynamic> {};
template <typename T>
struct shape_rows<T, void_t<decltype(traits<remove_all_t<T>>::RowsAtCompileTime)>>
    : std::integral_constant<int, traits<remove_all_t<T>>::RowsAtCompileTime> {};

template <typename T, typename = void>
struct shape_cols : std::integral_constant<int, Dynamic> {};
template <typename T>
struct shape_cols<T, void_t<decltype(traits<remove_all_t<T>>::ColsAtCompileTime)>>
    : std::integral_constant<int, traits<remove_all_t<T>>::ColsAtCompileTime> {};

template <typename T>
struct compile_time_shape {
  static constexpr int Rows = shape_rows<T>::value;
  static constexpr int Cols = shape_cols<T>::value;
  static constexpr int Size = size_at_compile_time(Rows, Cols);
};

template <int Left, int Right>
struct dimension_relation {
  static constexpr bool Known = Left != Dynamic && Right != Dynamic;
  static constexpr bool Compatible = !Known || Left == Right;
  static constexpr bool Proven = Known && Compatible;
};

template <typename Left, typename Right>
using matching_rows = dimension_relation<compile_time_shape<Left>::Rows, compile_time_shape<Right>::Rows>;
template <typename Left, typename Right>
using matching_cols = dimension_relation<compile_time_shape<Left>::Cols, compile_time_shape<Right>::Cols>;
template <typename Left, typename Right>
using matching_sizes = dimension_relation<compile_time_shape<Left>::Size, compile_time_shape<Right>::Size>;
template <typename SizeExpr, typename MatrixExpr, bool RowAxis>
using size_matches_axis =
    dimension_relation<compile_time_shape<SizeExpr>::Size,
                       RowAxis ? compile_time_shape<MatrixExpr>::Rows : compile_time_shape<MatrixExpr>::Cols>;
template <typename Left, typename Right>
struct matching_shapes {
  using Rows = matching_rows<Left, Right>;
  using Cols = matching_cols<Left, Right>;
  static constexpr bool Known = Rows::Known && Cols::Known;
  // Preserve the empty-expression exception of EIGEN_PREDICATE_SAME_MATRIX_SIZE.
  static constexpr bool Compatible = (compile_time_shape<Left>::Size == 0 && compile_time_shape<Right>::Size == 0) ||
                                     (Rows::Compatible && Cols::Compatible);
  static constexpr bool Proven = Rows::Proven && Cols::Proven;
};
template <typename Left, typename Right>
using product_dimensions = dimension_relation<compile_time_shape<Left>::Cols, compile_time_shape<Right>::Rows>;
template <typename T>
using square_dimensions = dimension_relation<compile_time_shape<T>::Rows, compile_time_shape<T>::Cols>;

template <typename T, int Rows, int Cols>
using reshape_dimensions = dimension_relation<size_at_compile_time(Rows, Cols), compile_time_shape<T>::Size>;

template <typename Solver, typename Rhs, bool Transposed = false>
using solver_rhs_dimensions =
    dimension_relation<Transposed ? compile_time_shape<Solver>::Cols : compile_time_shape<Solver>::Rows,
                       compile_time_shape<Rhs>::Rows>;

template <typename T>
struct vector_dimensions {
  static constexpr bool Proven = compile_time_shape<T>::Rows == 1 || compile_time_shape<T>::Cols == 1;
  static constexpr bool Known =
      Proven || (compile_time_shape<T>::Rows != Dynamic && compile_time_shape<T>::Cols != Dynamic);
  static constexpr bool Compatible = !Known || Proven;
};

template <int Extent>
struct nonempty_extent {
  static constexpr bool Known = Extent != Dynamic;
  static constexpr bool Compatible = !Known || Extent > 0;
  static constexpr bool Proven = Known && Compatible;
};

template <typename T>
using nonempty_dimensions = nonempty_extent<compile_time_shape<T>::Size>;

template <typename T, int Rows, int Cols>
struct fixed_dimensions {
  using RowRelation = dimension_relation<compile_time_shape<T>::Rows, Rows>;
  using ColRelation = dimension_relation<compile_time_shape<T>::Cols, Cols>;
  static constexpr bool Known = RowRelation::Known && ColRelation::Known;
  static constexpr bool Compatible = RowRelation::Compatible && ColRelation::Compatible;
  static constexpr bool Proven = RowRelation::Proven && ColRelation::Proven;
};

}  // namespace internal
}  // namespace Eigen

// These macros are private implementation helpers, not a public assertion API.
// Only EIGEN_INTERNAL_DEBUG_SHAPE_ASSERT uses internal-debug assertion semantics.
// Keep the runtime expression lazy and preserve assertion overrides.
// In particular, a proven check must not call dimension accessors even at -O0.
// An unevaluated use prevents unused-parameter warnings when assertions are disabled.
#define EIGEN_INTERNAL_SHAPE_ASSERT(Proven, Condition) \
  do {                                                 \
    EIGEN_IF_CONSTEXPR (!(Proven)) {                   \
      (void)sizeof(static_cast<bool>(Condition));      \
      eigen_assert(Condition);                         \
    }                                                  \
  } while (false)

#define EIGEN_INTERNAL_DEBUG_SHAPE_ASSERT(Proven, Condition) \
  do {                                                       \
    EIGEN_IF_CONSTEXPR (!(Proven)) {                         \
      (void)sizeof(static_cast<bool>(Condition));            \
      eigen_internal_assert(Condition);                      \
    }                                                        \
  } while (false)

// Infer the concrete expression without calling derived() or dimension accessors.
#define EIGEN_INTERNAL_SHAPE_TYPE(Xpr) \
  typename Eigen::internal::shape_expression<Eigen::internal::remove_all_t<decltype(Xpr)>>::type

// The relation is shared by the static diagnostic and the lazy debug check.
#define EIGEN_INTERNAL_CHECK_SHAPE(Condition, Message, ...) \
  do {                                                      \
    using ShapeRelation = __VA_ARGS__;                      \
    EIGEN_STATIC_ASSERT(ShapeRelation::Compatible, Message) \
    EIGEN_IF_CONSTEXPR (!ShapeRelation::Proven) {           \
      (void)sizeof(static_cast<bool>(Condition));           \
      eigen_assert(Condition);                              \
    }                                                       \
  } while (false)

#define EIGEN_INTERNAL_ASSERT_SAME_ROWS_MSG(Lhs, Rhs, Message)                          \
  EIGEN_INTERNAL_CHECK_SHAPE(                                                           \
      (Lhs).rows() == (Rhs).rows() && (Message), YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES, \
      Eigen::internal::matching_rows<EIGEN_INTERNAL_SHAPE_TYPE(Lhs), EIGEN_INTERNAL_SHAPE_TYPE(Rhs)>)
#define EIGEN_INTERNAL_ASSERT_SAME_ROWS(Lhs, Rhs) \
  EIGEN_INTERNAL_ASSERT_SAME_ROWS_MSG(Lhs, Rhs, "Both expressions must have the same number of rows")

#define EIGEN_INTERNAL_ASSERT_SAME_SHAPE_MSG(Lhs, Rhs, Message)                                           \
  do {                                                                                                    \
    using ShapeRelation =                                                                                 \
        Eigen::internal::matching_shapes<EIGEN_INTERNAL_SHAPE_TYPE(Lhs), EIGEN_INTERNAL_SHAPE_TYPE(Rhs)>; \
    EIGEN_STATIC_ASSERT(ShapeRelation::Compatible, YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES)                 \
    EIGEN_INTERNAL_SHAPE_ASSERT(ShapeRelation::Rows::Proven, (Lhs).rows() == (Rhs).rows() && (Message));  \
    EIGEN_INTERNAL_SHAPE_ASSERT(ShapeRelation::Cols::Proven, (Lhs).cols() == (Rhs).cols() && (Message));  \
  } while (false)
#define EIGEN_INTERNAL_ASSERT_SAME_SHAPE(Lhs, Rhs) \
  EIGEN_INTERNAL_ASSERT_SAME_SHAPE_MSG(Lhs, Rhs, "Both expressions must have the same dimensions")

#define EIGEN_INTERNAL_ASSERT_SAME_COLS_MSG(Lhs, Rhs, Message)                          \
  EIGEN_INTERNAL_CHECK_SHAPE(                                                           \
      (Lhs).cols() == (Rhs).cols() && (Message), YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES, \
      Eigen::internal::matching_cols<EIGEN_INTERNAL_SHAPE_TYPE(Lhs), EIGEN_INTERNAL_SHAPE_TYPE(Rhs)>)
#define EIGEN_INTERNAL_ASSERT_SAME_COLS(Lhs, Rhs) \
  EIGEN_INTERNAL_ASSERT_SAME_COLS_MSG(Lhs, Rhs, "Both expressions must have the same number of columns")

#define EIGEN_INTERNAL_ASSERT_SAME_SIZE_MSG(Lhs, Rhs, Message)                         \
  EIGEN_INTERNAL_CHECK_SHAPE(                                                          \
      (Lhs).size() == (Rhs).size() && (Message), YOU_MIXED_VECTORS_OF_DIFFERENT_SIZES, \
      Eigen::internal::matching_sizes<EIGEN_INTERNAL_SHAPE_TYPE(Lhs), EIGEN_INTERNAL_SHAPE_TYPE(Rhs)>)
#define EIGEN_INTERNAL_ASSERT_SAME_SIZE(Lhs, Rhs) \
  EIGEN_INTERNAL_ASSERT_SAME_SIZE_MSG(Lhs, Rhs, "Both expressions must have the same number of coefficients")

#define EIGEN_INTERNAL_ASSERT_PRODUCT_SHAPE_MSG(Lhs, Rhs, Message)       \
  EIGEN_INTERNAL_CHECK_SHAPE(                                            \
      (Lhs).cols() == (Rhs).rows() && (Message), INVALID_MATRIX_PRODUCT, \
      Eigen::internal::product_dimensions<EIGEN_INTERNAL_SHAPE_TYPE(Lhs), EIGEN_INTERNAL_SHAPE_TYPE(Rhs)>)
#define EIGEN_INTERNAL_ASSERT_PRODUCT_SHAPE(Lhs, Rhs) \
  EIGEN_INTERNAL_ASSERT_PRODUCT_SHAPE_MSG(            \
      Lhs, Rhs,                                       \
      "invalid matrix product; "                      \
      "if you wanted a coeff-wise or a dot product use the respective explicit functions")

// SizeExpr must use coefficient-count size semantics (rows * cols), not permutation/transposition order.
// Select the axis in the relation so C++14 never instantiates a check for the unused dimension.
#define EIGEN_INTERNAL_ASSERT_SIZE_MATCHES_AXIS_MSG(SizeExpr, MatrixExpr, RowAxis, Message)                          \
  EIGEN_INTERNAL_CHECK_SHAPE(                                                                                        \
      (SizeExpr).size() == ((RowAxis) ? (MatrixExpr).rows() : (MatrixExpr).cols()) && (Message),                     \
      YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES,                                                                         \
      Eigen::internal::size_matches_axis<EIGEN_INTERNAL_SHAPE_TYPE(SizeExpr), EIGEN_INTERNAL_SHAPE_TYPE(MatrixExpr), \
                                         RowAxis>)
#define EIGEN_INTERNAL_ASSERT_SIZE_MATCHES_AXIS(SizeExpr, MatrixExpr, RowAxis) \
  EIGEN_INTERNAL_ASSERT_SIZE_MATCHES_AXIS_MSG(SizeExpr, MatrixExpr, RowAxis,   \
                                              "Coefficient count must match the selected matrix dimension")
#define EIGEN_INTERNAL_ASSERT_SIZE_MATCHES_ROWS_MSG(SizeExpr, MatrixExpr, Message) \
  EIGEN_INTERNAL_ASSERT_SIZE_MATCHES_AXIS_MSG(SizeExpr, MatrixExpr, true, Message)
#define EIGEN_INTERNAL_ASSERT_SIZE_MATCHES_COLS_MSG(SizeExpr, MatrixExpr, Message) \
  EIGEN_INTERNAL_ASSERT_SIZE_MATCHES_AXIS_MSG(SizeExpr, MatrixExpr, false, Message)
#define EIGEN_INTERNAL_ASSERT_SIZE_MATCHES_ROWS(SizeExpr, MatrixExpr) \
  EIGEN_INTERNAL_ASSERT_SIZE_MATCHES_AXIS(SizeExpr, MatrixExpr, true)
#define EIGEN_INTERNAL_ASSERT_SIZE_MATCHES_COLS(SizeExpr, MatrixExpr) \
  EIGEN_INTERNAL_ASSERT_SIZE_MATCHES_AXIS(SizeExpr, MatrixExpr, false)

#define EIGEN_INTERNAL_ASSERT_SQUARE_MSG(Xpr, Message)                                                      \
  EIGEN_INTERNAL_SHAPE_ASSERT((Eigen::internal::square_dimensions<EIGEN_INTERNAL_SHAPE_TYPE(Xpr)>::Proven), \
                              (Xpr).rows() == (Xpr).cols() && (Message))
#define EIGEN_INTERNAL_ASSERT_SQUARE(Xpr) \
  EIGEN_INTERNAL_ASSERT_SQUARE_MSG(Xpr, "This operation requires a square matrix")

// TODO(C++17): guard the vector check with if constexpr (Enabled) instead of selecting a dummy relation.
#define EIGEN_INTERNAL_ASSERT_VECTOR_IF(Enabled, Xpr)                                                 \
  EIGEN_INTERNAL_CHECK_SHAPE(                                                                         \
      ((Xpr).rows() == 1 || (Xpr).cols() == 1) && "This operation requires a vector",                 \
      YOU_TRIED_CALLING_A_VECTOR_METHOD_ON_A_MATRIX,                                                  \
      std::conditional_t<Enabled, Eigen::internal::vector_dimensions<EIGEN_INTERNAL_SHAPE_TYPE(Xpr)>, \
                         Eigen::internal::dimension_relation<1, 1>>)
#define EIGEN_INTERNAL_ASSERT_VECTOR(Xpr) EIGEN_INTERNAL_ASSERT_VECTOR_IF(true, Xpr)

#define EIGEN_INTERNAL_ASSERT_SCALAR(Xpr)                                                        \
  EIGEN_INTERNAL_CHECK_SHAPE(                                                                    \
      ((Xpr).rows() == 1 && (Xpr).cols() == 1) && "This operation requires a 1-by-1 expression", \
      THIS_METHOD_IS_ONLY_FOR_1x1_EXPRESSIONS,                                                   \
      Eigen::internal::fixed_dimensions<EIGEN_INTERNAL_SHAPE_TYPE(Xpr), 1, 1>)

#define EIGEN_INTERNAL_ASSERT_NONEMPTY_MSG(Xpr, Message)                                                      \
  EIGEN_INTERNAL_SHAPE_ASSERT((Eigen::internal::nonempty_dimensions<EIGEN_INTERNAL_SHAPE_TYPE(Xpr)>::Proven), \
                              (Xpr).rows() > 0 && (Xpr).cols() > 0 && (Message))
#define EIGEN_INTERNAL_ASSERT_NONEMPTY(Xpr) EIGEN_INTERNAL_ASSERT_NONEMPTY_MSG(Xpr, "you are using an empty matrix")

#define EIGEN_INTERNAL_ASSERT_NONEMPTY_AXIS(Xpr, Vertical)                                                          \
  do {                                                                                                              \
    using ShapeRelation = Eigen::internal::nonempty_extent<                                                         \
        (Vertical) ? Eigen::internal::compile_time_shape<EIGEN_INTERNAL_SHAPE_TYPE(Xpr)>::Rows                      \
                   : Eigen::internal::compile_time_shape<EIGEN_INTERNAL_SHAPE_TYPE(Xpr)>::Cols>;                    \
    EIGEN_INTERNAL_SHAPE_ASSERT(ShapeRelation::Proven,                                                              \
                                ((Vertical) ? (Xpr).rows() : (Xpr).cols()) > 0 && "you are using an empty matrix"); \
  } while (false)

// Size must be the cached coefficient count of Xpr, not an allocation capacity.
#define EIGEN_INTERNAL_ASSERT_NONEMPTY_SIZE_MSG(Xpr, Size, Message)                                           \
  EIGEN_INTERNAL_SHAPE_ASSERT((Eigen::internal::nonempty_dimensions<EIGEN_INTERNAL_SHAPE_TYPE(Xpr)>::Proven), \
                              (Size) > 0 && (Message))
#define EIGEN_INTERNAL_ASSERT_NONEMPTY_SIZE(Xpr, Size) \
  EIGEN_INTERNAL_ASSERT_NONEMPTY_SIZE_MSG(Xpr, Size, "you are using an empty matrix")

#define EIGEN_INTERNAL_ASSERT_RESHAPE_SIZE(Xpr, Rows, Cols)                                                   \
  EIGEN_INTERNAL_CHECK_SHAPE(                                                                                 \
      (Rows) * (Cols) == (Xpr).rows() * (Xpr).cols() && "Reshaping must preserve the number of coefficients", \
      YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES,                                                                  \
      Eigen::internal::reshape_dimensions<EIGEN_INTERNAL_SHAPE_TYPE(Xpr), Rows, Cols>)

#define EIGEN_INTERNAL_ASSERT_SOLVER_RHS_MSG(Solver, Rhs, Transposed, Message)                                \
  EIGEN_INTERNAL_CHECK_SHAPE(((Transposed) ? (Solver).cols() : (Solver).rows()) == (Rhs).rows() && (Message), \
                             YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES,                                           \
                             Eigen::internal::solver_rhs_dimensions<EIGEN_INTERNAL_SHAPE_TYPE(Solver),        \
                                                                    EIGEN_INTERNAL_SHAPE_TYPE(Rhs), Transposed>)
#define EIGEN_INTERNAL_ASSERT_SOLVER_RHS(Solver, Rhs, Transposed) \
  EIGEN_INTERNAL_ASSERT_SOLVER_RHS_MSG(Solver, Rhs, Transposed,   \
                                       "solve(): invalid number of rows of the right hand side")

#endif  // EIGEN_SHAPE_CHECKS_H
