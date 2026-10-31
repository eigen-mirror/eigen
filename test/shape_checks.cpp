// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/Dense>
#include <contrib/Eigen/MatrixFunctions>

struct MissingShape {};
struct PartialShape {};
template <int Rows, int Cols>
struct ShapeProbe;
namespace Eigen {
namespace internal {
template <int Rows, int Cols>
struct traits<ShapeProbe<Rows, Cols>> : traits<Matrix<double, Rows, Cols>> {};
template <>
struct traits<PartialShape> {
  static constexpr int RowsAtCompileTime = 3;
};
}  // namespace internal
}  // namespace Eigen

template <int Rows, int Cols>
struct ShapeProbe : MatrixBase<ShapeProbe<Rows, Cols>> {
  using Nested = const ShapeProbe&;
  mutable int row_reads = 0;
  mutable int col_reads = 0;
  Index rows() const {
    ++row_reads;
    return 2;
  }
  Index cols() const {
    ++col_reads;
    return 2;
  }
};

void rank_update_shape_checks() {
  Matrix2d b = Matrix2d::Constant(2);
  Matrix2d updated = Matrix2d::Zero();
  updated.selfadjointView<Lower>().rankUpdate(b);
  VERIFY_IS_EQUAL(Matrix2d(updated.selfadjointView<Lower>()), (b * b.transpose()).eval());
  MatrixXd wrong = MatrixXd::Ones(3, 2);
  VERIFY_RAISES_ASSERT(updated.selfadjointView<Lower>().rankUpdate(wrong));
#ifdef EIGEN_NO_STATIC_ASSERT
  Matrix<double, 3, 2> fixed_wrong = wrong;
  VERIFY_RAISES_ASSERT(updated.selfadjointView<Lower>().rankUpdate(fixed_wrong));
#endif
}

void product_coefficient_shape_checks() {
  Matrix2d a = Matrix2d::Identity(), b = Matrix2d::Constant(2);
  VERIFY_IS_EQUAL(a.lazyProduct(b).coeff(0, 1), 2);
  Vector2d v = Vector2d::Ones();
  VERIFY_IS_EQUAL((v.transpose() * v).coeff(0, 0), 2);
  VERIFY_IS_EQUAL((v.transpose() * v).coeff(0), 2);
  MatrixXd dynamic_a = a, dynamic_b = b;
  VERIFY_RAISES_ASSERT((dynamic_a * dynamic_b).coeff(0, 0));
  VERIFY_RAISES_ASSERT((dynamic_a * dynamic_b).coeff(0));
  MatrixXd scalar = MatrixXd::Constant(1, 1, 3);
  VERIFY_IS_EQUAL((scalar * scalar).coeff(0), 9);
}

void transpose_shape_checks() {
  Matrix<double, 2, 3> rectangle;
  rectangle << 1, 2, 3, 4, 5, 6;
  MatrixXd resizable = rectangle;
  resizable.transposeInPlace();
  VERIFY_IS_EQUAL(resizable, rectangle.transpose());
  Matrix<double, 2, Dynamic> partial = Matrix2d::Identity();
  partial.transposeInPlace();
  VERIFY_IS_EQUAL(partial, Matrix2d::Identity());
  partial = rectangle;
  VERIFY_RAISES_ASSERT(partial.transposeInPlace());
#ifdef EIGEN_NO_STATIC_ASSERT
  VERIFY_RAISES_ASSERT(rectangle.transposeInPlace());
#endif
}

void product_shape_checks() {
  transpose_shape_checks();
  product_coefficient_shape_checks();
  rank_update_shape_checks();
  ShapeProbe<2, 2> a, b;
  const auto product = a * b;
  EIGEN_UNUSED_VARIABLE(product);
  VERIFY_IS_EQUAL(a.col_reads + b.row_reads, 0);
  const MatrixBase<ShapeProbe<2, 2>>& base_a = a;
  const MatrixBase<ShapeProbe<2, 2>>& base_b = b;
  EIGEN_INTERNAL_ASSERT_PRODUCT_SHAPE(base_a, base_b);
  VERIFY_IS_EQUAL(a.col_reads + b.row_reads, 0);
  Matrix2d fixed = Matrix2d::Identity();
  MatrixXd wrong(3, 2);
  VERIFY_RAISES_ASSERT((fixed * wrong));
#ifdef EIGEN_NO_STATIC_ASSERT
  Matrix<double, 3, 2> fixed_wrong;
  VERIFY_RAISES_ASSERT((Product<Matrix2d, decltype(fixed_wrong)>(fixed, fixed_wrong)));
#endif
  Transpositions<2> swaps;
  swaps.setIdentity();
  VERIFY_IS_EQUAL((swaps * fixed).eval(), fixed);
  VERIFY_IS_EQUAL((fixed * swaps.transpose()).eval(), fixed);
}

template <int Size, int Order>
void product_destination_shape_checks() {
  using MatrixType = Matrix<double, Size, Size, Order>;
  const MatrixType identity = MatrixType::Identity(3, 3);
  const MatrixType values = MatrixType::Constant(3, 3, 2);
  MatrixType result = MatrixType::Zero(3, 3);
  result.noalias() += identity * values;
  VERIFY_IS_EQUAL(result, values);
  result.noalias() -= identity * values;
  VERIFY(result.isZero());
  result.noalias() += identity.template selfadjointView<Lower>() * values;
  VERIFY_IS_EQUAL(result, values);
  result.noalias() -= identity.template triangularView<Upper>() * values;
  VERIFY(result.isZero());

  PermutationMatrix<Size> permutation(3);
  permutation.setIdentity();
  DiagonalMatrix<double, Size> diagonal(3);
  diagonal.diagonal().setConstant(2);
  result.noalias() = identity.template triangularView<Lower>() * diagonal;
  VERIFY_IS_EQUAL(result, 2 * identity);
  result.noalias() = diagonal * identity.template triangularView<Upper>();
  VERIFY_IS_EQUAL(result, 2 * identity);
  result.noalias() = identity.template selfadjointView<Lower>() * diagonal;
  VERIFY_IS_EQUAL(result, 2 * identity);
  result.setZero();
  const auto scaled = permutation * diagonal;
  result += scaled;
  VERIFY_IS_EQUAL(result, 2 * identity);
  result -= scaled;
  VERIFY(result.isZero());
  MatrixXd wrong_destination = MatrixXd::Zero(2, 3);
  VERIFY_RAISES_ASSERT(wrong_destination.noalias() += identity * values);
  VERIFY_RAISES_ASSERT(wrong_destination.noalias() -= identity * values);
  VERIFY_RAISES_ASSERT(wrong_destination += scaled);
  VERIFY_RAISES_ASSERT(wrong_destination -= scaled);
  result.noalias() += scaled * identity;
  VERIFY_IS_EQUAL(result, 2 * identity);
  result.noalias() -= scaled * identity;
  VERIFY(result.isZero());
  VERIFY_IS_EQUAL((scaled * permutation).toDenseMatrix(), 2 * identity);
  VERIFY_IS_EQUAL((permutation * scaled).toDenseMatrix(), 2 * identity);
  VERIFY_IS_EQUAL((permutation * permutation.inverse()).toDenseMatrix().template cast<double>(), identity);
  const ScaledPermutationMatrix<double, Size> from_coefficients(permutation.indices(), diagonal.diagonal());
  VERIFY_IS_EQUAL(from_coefficients.toDenseMatrix(), 2 * identity);
  DiagonalMatrix<double, Dynamic> wrong(2);
  VERIFY_RAISES_ASSERT(permutation * wrong);
}

EIGEN_DECLARE_TEST(shape_checks) {
  ShapeProbe<2, 2> fixed_a, fixed_b;
  const auto sum = fixed_a + fixed_b;
  EIGEN_UNUSED_VARIABLE(sum);
  internal::resize_if_allowed(fixed_a, fixed_b, internal::add_assign_op<double, double>());
  VERIFY_IS_EQUAL(fixed_a.row_reads + fixed_b.row_reads + fixed_a.col_reads + fixed_b.col_reads, 0);
  ShapeProbe<2, Dynamic> partial_a, partial_b;
  const auto partial_sum = partial_a + partial_b;
  EIGEN_UNUSED_VARIABLE(partial_sum);
  VERIFY_IS_EQUAL(partial_a.row_reads + partial_b.row_reads, 0);
#ifndef EIGEN_NO_DEBUG
  VERIFY(partial_a.col_reads > 0 && partial_b.col_reads > 0);
#endif
  Matrix2d matrix = Matrix2d::Identity();
  MatrixXd bad(3, 2);
  VERIFY_RAISES_ASSERT((matrix + bad));
  VERIFY_RAISES_ASSERT(matrix += bad);
  VERIFY_IS_EQUAL(hcat(matrix, matrix).cols(), 4);
  VERIFY_IS_EQUAL(vcat(matrix, matrix).rows(), 4);
  Matrix<double, 2, 3> concat_wide = Matrix<double, 2, 3>::Ones();
  VERIFY_IS_EQUAL(hcat(matrix, concat_wide).cols(), 5);
  VERIFY_IS_EQUAL(vcat(matrix, concat_wide.transpose()).rows(), 5);
#ifdef EIGEN_NO_STATIC_ASSERT
  VERIFY_RAISES_ASSERT(hcat(matrix, concat_wide.transpose()));
  VERIFY_RAISES_ASSERT(vcat(matrix, concat_wide));
#endif
  Matrix<double, 2, 3> wide = Matrix<double, 2, 3>::Ones();
  const DiagonalMatrix<double, 2> left_diagonal(Vector2d::Constant(2));
  const DiagonalMatrix<double, 3> right_diagonal(Vector3d::Constant(3));
  Matrix<double, 2, 3> triangular_product = left_diagonal * wide.triangularView<Upper>();
  VERIFY_IS_EQUAL(triangular_product, 2 * wide.triangularView<Upper>().toDenseMatrix());
  triangular_product = wide.triangularView<Upper>() * right_diagonal;
  VERIFY_IS_EQUAL(triangular_product, 3 * wide.triangularView<Upper>().toDenseMatrix());
  Vector3d vector = Vector3d::Ones();
  RowVector3d row = RowVector3d::Ones();
  VERIFY_IS_EQUAL(vector.dot(row), 3);
  VERIFY_IS_EQUAL(vector.reshaped(fix<1>, fix<3>).sum(), 3);
  VectorXd dynamic_reshape = vector;
  VERIFY_RAISES_ASSERT(dynamic_reshape.reshaped(fix<2>, fix<2>));
#ifdef EIGEN_NO_STATIC_ASSERT
  VERIFY_RAISES_ASSERT(vector.reshaped(fix<2>, fix<2>));
  Vector4d mismatched_vector = Vector4d::Ones();
  VERIFY_RAISES_ASSERT(vector.dot(mismatched_vector));
#endif
  VectorXd bad_vector(2);
  VERIFY_RAISES_ASSERT(vector.dot(bad_vector));
  VectorXd resized;
  resized.resizeLike(row);
  VERIFY_IS_EQUAL(resized.size(), 3);
  MatrixXd resize_source = matrix;
  VERIFY_RAISES_ASSERT(resized.resizeLike(resize_source));
#ifdef EIGEN_NO_STATIC_ASSERT
  VERIFY_RAISES_ASSERT(resized.resizeLike(matrix));
#endif
  Ref<VectorXd> ref(vector);
  Ref<Matrix2d> matrix_ref(matrix);
  VERIFY_IS_EQUAL(matrix_ref.rows(), 2);
  Matrix2d resized_matrix;
  resized_matrix.resizeLike(matrix);
#ifdef EIGEN_NO_STATIC_ASSERT
  VERIFY_RAISES_ASSERT(EIGEN_INTERNAL_ASSERT_SCALAR(matrix));
#endif
  VERIFY_IS_EQUAL(ref.size(), 3);
  MatrixXd dynamic_matrix = matrix;
  VERIFY_RAISES_ASSERT((Ref<VectorXd>(dynamic_matrix)));
  VERIFY_IS_EQUAL(matrix.determinant(), 1);
  VERIFY_IS_EQUAL(matrix.householderQr().absDeterminant(), 1);
  VERIFY_IS_EQUAL(matrix.fullPivLu().determinant(), 1);
  VERIFY_IS_EQUAL(matrix.fullPivLu().rcond(), 1);
  Matrix3d transpose_source = Matrix3d::Random();
  const Matrix3d transposed = transpose_source.transpose();
  transpose_source.transposeInPlace();
  VERIFY_IS_EQUAL(transpose_source, transposed);
  MatrixXd rectangular_transpose = MatrixXd::Random(2, 3);
  const MatrixXd rectangular_expected = rectangular_transpose.transpose();
  rectangular_transpose.transposeInPlace();
  VERIFY_IS_EQUAL(rectangular_transpose, rectangular_expected);
  VERIFY(matrix.llt().info() == Success);
  Vector2d rhs = Vector2d::Ones();
  const auto lu = matrix.fullPivLu();
  VERIFY_IS_EQUAL(lu.solve(rhs).eval(), rhs);
  VERIFY_IS_EQUAL(lu.transpose().solve(rhs).eval(), rhs);
  Matrix2d nonsymmetric;
  nonsymmetric << 2, 1, 0, 3;
  const auto partial_lu = nonsymmetric.partialPivLu();
  const Vector2d transpose_solution = partial_lu.transpose().solve(rhs);
  VERIFY_IS_APPROX(nonsymmetric.transpose() * transpose_solution, rhs);
  VERIFY_RAISES_ASSERT(partial_lu.transpose().solve(VectorXd::Ones(3)));
  VERIFY_RAISES_ASSERT(lu.solve(VectorXd::Ones(3)));
#ifdef EIGEN_NO_STATIC_ASSERT
  VERIFY_RAISES_ASSERT(lu.solve(Vector3d::Ones()));
#endif
  VERIFY_IS_APPROX(matrix.sqrt(), matrix);
  VERIFY(SelfAdjointEigenSolver<Matrix2d>(matrix).info() == Success);
  MatrixXd rectangle = MatrixXd::Ones(2, 3);
  VERIFY_RAISES_ASSERT(rectangle.determinant());
  VERIFY_RAISES_ASSERT(rectangle.householderQr().absDeterminant());
  VERIFY_RAISES_ASSERT(rectangle.fullPivLu().determinant());
  VERIFY_RAISES_ASSERT(rectangle.llt());
  VERIFY_RAISES_ASSERT(wide.fullPivLu().determinant());
  MatrixXd empty(0, 0);
  VERIFY_IS_EQUAL(empty.sum(), 0);
  const Matrix<double, 0, 0> fixed_empty;
  VERIFY_IS_EQUAL(fixed_empty.sum(), 0);
  VERIFY_IS_EQUAL(fixed_empty.prod(), 1);
  Matrix<double, 3, 0> empty_columns;
  Matrix<double, 0, 3> empty_rows;
  VERIFY_IS_EQUAL(empty_columns.colwise().maxCoeff().eval().size(), 0);
  VERIFY_IS_EQUAL(empty_rows.rowwise().minCoeff().eval().size(), 0);
  VERIFY_RAISES_ASSERT(fixed_empty.maxCoeff());
  VERIFY_RAISES_ASSERT(empty_columns.rowwise().maxCoeff());
  VERIFY_RAISES_ASSERT(empty_rows.colwise().minCoeff());
  VERIFY_RAISES_ASSERT(empty.maxCoeff());
  VERIFY_IS_EQUAL(matrix.maxCoeff(), 1);
  Index max_row, max_col;
  VERIFY_IS_EQUAL(matrix.maxCoeff(&max_row, &max_col), 1);
  CALL_SUBTEST(product_shape_checks());
  CALL_SUBTEST((product_destination_shape_checks<3, ColMajor>()));
  CALL_SUBTEST((product_destination_shape_checks<3, RowMajor>()));
  CALL_SUBTEST((product_destination_shape_checks<Dynamic, ColMajor>()));
  CALL_SUBTEST((product_destination_shape_checks<Dynamic, RowMajor>()));
  using namespace Eigen::internal;
  STATIC_CHECK((dimension_relation<3, 3>::Proven));
  STATIC_CHECK((!dimension_relation<3, 2>::Compatible));
  STATIC_CHECK((dimension_relation<Dynamic, 3>::Compatible));
  STATIC_CHECK((!dimension_relation<Dynamic, 3>::Known));
  STATIC_CHECK((!dimension_relation<3, Dynamic, 3, 2>::Compatible));
  STATIC_CHECK((!dimension_relation<Dynamic, 3, 2, 3>::Compatible));
  STATIC_CHECK((dimension_relation<Dynamic, 3, 4, 3>::Compatible));
  STATIC_CHECK((!dimension_relation<Dynamic, 3, 4, 3>::Proven));
  STATIC_CHECK((compile_time_shape<const Matrix<double, Dynamic, Dynamic, 0, 2, 3>&>::MaxRows == 2));
  STATIC_CHECK((compile_time_shape<const Matrix<double, Dynamic, Dynamic, 0, 2, 3>&>::MaxCols == 3));
  STATIC_CHECK((compile_time_shape<MissingShape>::Rows == Dynamic));
  STATIC_CHECK((compile_time_shape<PartialShape>::Rows == 3));
  STATIC_CHECK((compile_time_shape<PartialShape>::Cols == Dynamic));
  STATIC_CHECK((compile_time_shape<Matrix<double, Dynamic, 1, 0, 3, 1>>::Size == Dynamic));
  STATIC_CHECK((matching_sizes<Matrix<double, 2, 3>, Matrix<double, 3, 2>>::Proven));
  STATIC_CHECK((matching_shapes<Matrix<double, 1, 0>, Matrix<double, 0, 1>>::Compatible));
  const Matrix<double, 1, 0> empty_row;
  const Matrix<double, 0, 1> empty_column;
  VERIFY_RAISES_ASSERT(empty_row + empty_column);
  VERIFY_RAISES_ASSERT(internal::resize_if_allowed(empty_row, empty_column, internal::add_assign_op<double, double>()));
  STATIC_CHECK((square_dimensions<const Matrix3d&>::Proven));
  int reads = 0;
  struct RuntimeDimensions {
    mutable int accesses = 0;
    Index rows() const {
      ++accesses;
      return 2;
    }
    Index cols() const {
      ++accesses;
      return 3;
    }
  } runtime_a, runtime_b;
  EIGEN_INTERNAL_ASSERT_SAME_ROWS(runtime_a, runtime_b);
  EIGEN_INTERNAL_ASSERT_SAME_COLS(runtime_a, runtime_b);
#ifndef EIGEN_NO_DEBUG
  VERIFY_IS_EQUAL(runtime_a.accesses, 2);
  VERIFY_IS_EQUAL(runtime_b.accesses, 2);
#endif
  EIGEN_INTERNAL_SHAPE_ASSERT((dimension_relation<3, 3>::Proven), ++reads == 1);  // NOLINT(bugprone-assert-side-effect)
  VERIFY_IS_EQUAL(reads, 0);
  EIGEN_INTERNAL_ASSERT_NONEMPTY_SIZE(fixed_a, ++reads);  // NOLINT(bugprone-assert-side-effect)
  VERIFY_IS_EQUAL(reads, 0);
#ifndef EIGEN_NO_DEBUG
  // NOLINTBEGIN(bugprone-assert-side-effect)
  EIGEN_INTERNAL_SHAPE_ASSERT((dimension_relation<Dynamic, 3>::Proven), ++reads == 1);
  // NOLINTEND(bugprone-assert-side-effect)
  VERIFY_IS_EQUAL(reads, 1);
  VERIFY_RAISES_ASSERT([] { EIGEN_INTERNAL_SHAPE_ASSERT(false, false); }());
#endif
}
