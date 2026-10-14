// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

template <int Rows, int Cols>
struct AxisShapeProbe;
namespace Eigen {
namespace internal {
template <int Rows, int Cols>
struct traits<AxisShapeProbe<Rows, Cols>> {
  static constexpr int RowsAtCompileTime = Rows;
  static constexpr int ColsAtCompileTime = Cols;
};
}  // namespace internal
}  // namespace Eigen

template <int Rows, int Cols>
struct AxisShapeProbe {
  Index row_count, col_count;
  mutable int size_reads = 0, row_reads = 0, col_reads = 0;
  Index size() const {
    ++size_reads;
    return row_count * col_count;
  }
  Index rows() const {
    ++row_reads;
    return row_count;
  }
  Index cols() const {
    ++col_reads;
    return col_count;
  }
};

struct UnknownAxisShape {
  Index size() const { return 3; }
  Index rows() const { return 3; }
  Index cols() const { return 4; }
};

EIGEN_DECLARE_TEST(shape_size_axis) {
  using namespace Eigen::internal;
  STATIC_CHECK((size_matches_axis<RowVector3d, Matrix<double, 3, 4>, true>::Proven));
  STATIC_CHECK((size_matches_axis<Vector4d, Matrix<double, 3, 4>, false>::Proven));
  STATIC_CHECK((!size_matches_axis<Vector3d, Matrix<double, 3, 4>, false>::Compatible));
  STATIC_CHECK((!size_matches_axis<Matrix<double, Dynamic, 1, 0, 4, 1>, Matrix4d, true>::Known));
  STATIC_CHECK((!size_matches_axis<UnknownAxisShape, Matrix3d, true>::Known));

  AxisShapeProbe<1, 3> row_vector{1, 3};
  AxisShapeProbe<4, 1> col_vector{4, 1};
  AxisShapeProbe<3, 4> fixed{3, 4};
  EIGEN_INTERNAL_ASSERT_SIZE_MATCHES_ROWS(row_vector, fixed);
  EIGEN_INTERNAL_ASSERT_SIZE_MATCHES_COLS(col_vector, fixed);
  VERIFY_IS_EQUAL(row_vector.size_reads + col_vector.size_reads + fixed.row_reads + fixed.col_reads, 0);

  AxisShapeProbe<Dynamic, 4> partial{3, 4};
  EIGEN_INTERNAL_ASSERT_SIZE_MATCHES_COLS(col_vector, partial);
  VERIFY_IS_EQUAL(col_vector.size_reads + partial.col_reads, 0);
  EIGEN_INTERNAL_ASSERT_SIZE_MATCHES_ROWS(row_vector, partial);
#ifndef EIGEN_NO_DEBUG
  VERIFY_IS_EQUAL(row_vector.size_reads, 1);
  VERIFY_IS_EQUAL(partial.row_reads, 1);
#endif
  VERIFY_IS_EQUAL(partial.col_reads, 0);

  const UnknownAxisShape unknown;
  EIGEN_INTERNAL_ASSERT_SIZE_MATCHES_ROWS(unknown, fixed);
  VERIFY_RAISES_ASSERT(EIGEN_INTERNAL_ASSERT_SIZE_MATCHES_COLS(unknown, fixed));
  AxisShapeProbe<Dynamic, Dynamic> wrong{2, 4};
  VERIFY_RAISES_ASSERT(EIGEN_INTERNAL_ASSERT_SIZE_MATCHES_ROWS(row_vector, wrong));
#ifdef EIGEN_NO_STATIC_ASSERT
  VERIFY_RAISES_ASSERT(EIGEN_INTERNAL_ASSERT_SIZE_MATCHES_COLS(row_vector, fixed));
#endif
  const Matrix<double, 0, 4> empty;
  const Matrix<double, 0, 1> empty_vector;
  EIGEN_INTERNAL_ASSERT_SIZE_MATCHES_ROWS(empty_vector, empty);
  const Matrix<double, 3, 4> matrix = Matrix<double, 3, 4>::Zero();
  const Vector3d vector = Vector3d::Ones();
  const MatrixBase<Vector3d>& vector_base = vector;
  const MatrixBase<Matrix<double, 3, 4>>& matrix_base = matrix;
  EIGEN_INTERNAL_ASSERT_SIZE_MATCHES_ROWS(vector_base, matrix_base);
}
