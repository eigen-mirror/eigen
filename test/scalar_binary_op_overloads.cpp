// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#define EIGEN_TEST_ANNOYING_SCALAR_DONT_THROW
#include "AnnoyingScalar.h"
#include <Eigen/Geometry>
#include <Eigen/SparseCore>

template <int Dimension>
struct QuantityScalar {
  double value = 0;
  QuantityScalar() = default;
  QuantityScalar(double x) : value(x) {}
};

template <int A, int B>
QuantityScalar<A + B> operator*(QuantityScalar<A> lhs, QuantityScalar<B> rhs) {
  return lhs.value * rhs.value;
}
template <int A>
QuantityScalar<A> operator+(QuantityScalar<A> lhs, QuantityScalar<A> rhs) {
  return lhs.value + rhs.value;
}
template <int A>
QuantityScalar<A>& operator+=(QuantityScalar<A>& lhs, QuantityScalar<A> rhs) {
  lhs.value += rhs.value;
  return lhs;
}

namespace Eigen {
template <int A>
struct NumTraits<QuantityScalar<A>> : NumTraits<double> {
  using Real = QuantityScalar<A>;
  using NonInteger = QuantityScalar<A>;
  using Nested = QuantityScalar<A>;
  using Literal = QuantityScalar<A>;
  static constexpr bool RequireInitialization = true;
};
template <int A, typename B>
struct ScalarBinaryOpTraits<QuantityScalar<A>, B, internal::scalar_product_op<QuantityScalar<A>, B>> {
  using ReturnType = decltype(std::declval<QuantityScalar<A>>() * std::declval<B>());
};
template <typename A, int B>
struct ScalarBinaryOpTraits<A, QuantityScalar<B>, internal::scalar_product_op<A, QuantityScalar<B>>> {
  using ReturnType = decltype(std::declval<A>() * std::declval<QuantityScalar<B>>());
};
template <int A, int B>
struct ScalarBinaryOpTraits<QuantityScalar<A>, QuantityScalar<B>,
                            internal::scalar_product_op<QuantityScalar<A>, QuantityScalar<B>>> {
  using ReturnType = QuantityScalar<A + B>;
};
}  // namespace Eigen

template <int Size, int Order, int RhsDimension>
void quantity_scalar_product() {
  Matrix<QuantityScalar<1>, Size, Size, Order> lhs(2, 2);
  Matrix<QuantityScalar<RhsDimension>, Size, Size, Order> rhs(2, 2);
  for (Index j = 0; j < 2; ++j)
    for (Index i = 0; i < 2; ++i) {
      lhs(i, j) = double(i + 2 * j + 1);
      rhs(i, j) = double(2 * i - j + 1);
    }
  const Matrix<QuantityScalar<1 + RhsDimension>, Size, Size, Order> result = lhs * rhs;
  const Matrix<QuantityScalar<1>, Dynamic, Dynamic, Order> dynamic_lhs = lhs;
  const Matrix<QuantityScalar<RhsDimension>, Dynamic, Dynamic, Order> dynamic_rhs = rhs;
  const Matrix<QuantityScalar<1 + RhsDimension>, Dynamic, Dynamic, Order> lazy_result =
      dynamic_lhs.lazyProduct(dynamic_rhs);
  for (Index j = 0; j < 2; ++j)
    for (Index i = 0; i < 2; ++i) {
      double expected = 0;
      for (Index k = 0; k < 2; ++k) expected += lhs(i, k).value * rhs(k, j).value;
      VERIFY_IS_EQUAL(result(i, j).value, expected);
      VERIFY_IS_EQUAL(lazy_result(i, j).value, expected);
    }
}

template <typename Kind>
struct EigenBaseScalar;
namespace Eigen {
namespace internal {
template <typename Kind>
struct traits<EigenBaseScalar<Kind>> : traits<Matrix2d> {
  using XprKind = Kind;
};
}  // namespace internal
}  // namespace Eigen

template <typename Kind>
struct EigenBaseScalar : Eigen::EigenBase<EigenBaseScalar<Kind>> {
  double value = 3;
  Eigen::Index rows() const { return 2; }
  Eigen::Index cols() const { return 2; }
};

template <typename Kind>
double operator*(double lhs, const EigenBaseScalar<Kind>& rhs) {
  return lhs * rhs.value;
}
template <typename Kind>
double operator*(const EigenBaseScalar<Kind>& lhs, double rhs) {
  return lhs.value * rhs;
}

struct PrivateMatrixScalar : private Eigen::Matrix2d {
  PrivateMatrixScalar() = default;
  explicit PrivateMatrixScalar(double value) : value(value) {}
  using Eigen::Matrix2d::derived;
  double value = 0;
};

using BoundedRows = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::ColMajor, 2, 2>;
using BoundedCols = Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::ColMajor, 2, 2>;
using BoundedArray = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 2, 2>;
using ScalarMatrix = Eigen::Matrix<double, 1, 1>;
using ScalarArray = Eigen::Array<double, 1, 1>;

struct BaseReferenceScalar final {
  BaseReferenceScalar(double value = 0) : value(value) {}
  double value;
};
using BaseReferenceMatrix = Eigen::Matrix<BaseReferenceScalar, 1, 1>;
using BaseReferenceArray = Eigen::Array<BaseReferenceScalar, 1, 1>;

BaseReferenceScalar operator*(BaseReferenceScalar lhs, double rhs) { return lhs.value * rhs; }
BaseReferenceScalar operator*(double lhs, BaseReferenceScalar rhs) { return lhs * rhs.value; }
BaseReferenceScalar operator+(BaseReferenceScalar lhs, BaseReferenceScalar rhs) { return lhs.value + rhs.value; }
BaseReferenceScalar operator*(BaseReferenceScalar lhs, const ScalarMatrix& rhs) { return lhs.value + rhs(0); }
BaseReferenceScalar operator*(BaseReferenceScalar lhs, const ScalarArray& rhs) { return lhs.value + rhs(0); }

struct DiagonalOperandScalar final {
  DiagonalOperandScalar(double value = 0) : value(value) {}
  double value;
};
using DiagonalOperandMatrix = Eigen::Matrix<DiagonalOperandScalar, 1, 1>;
using ScalarDiagonal = Eigen::DiagonalMatrix<double, 1>;

DiagonalOperandScalar operator*(const DiagonalOperandScalar& lhs, double rhs) { return lhs.value * rhs; }
DiagonalOperandScalar operator*(double lhs, const DiagonalOperandScalar& rhs) { return lhs * rhs.value; }
template <typename Derived>
DiagonalOperandScalar operator*(const DiagonalOperandScalar& lhs, const Eigen::DiagonalBase<Derived>& rhs) {
  return lhs.value + rhs.diagonal()(0);
}
template <typename Derived>
DiagonalOperandScalar operator*(const Eigen::DiagonalBase<Derived>& lhs, const DiagonalOperandScalar& rhs) {
  return lhs.diagonal()(0) + rhs.value;
}

double operator*(double lhs, const PrivateMatrixScalar& rhs) { return lhs * rhs.value; }
double operator*(const PrivateMatrixScalar& lhs, double rhs) { return lhs.value * rhs; }
double operator/(const PrivateMatrixScalar& lhs, const Eigen::Matrix2d& rhs) { return lhs.value / rhs.sum(); }
Eigen::Matrix2d operator*(const PrivateMatrixScalar& lhs, const Eigen::Matrix2d& rhs) { return lhs.value * rhs; }
Eigen::Matrix2d operator*(const PrivateMatrixScalar& lhs, const BoundedRows& rhs) { return lhs.value * rhs; }
Eigen::Matrix2d operator*(const BoundedCols& lhs, const PrivateMatrixScalar& rhs) { return lhs * rhs.value; }
Eigen::Array22d operator*(const PrivateMatrixScalar& lhs, const BoundedArray& rhs) { return lhs.value * rhs; }
Eigen::Array22d operator*(const BoundedArray& lhs, const PrivateMatrixScalar& rhs) { return lhs * rhs.value; }
double operator*(const PrivateMatrixScalar& lhs, const ScalarMatrix& rhs) { return lhs.value + rhs(0); }
double operator*(const ScalarMatrix& lhs, const PrivateMatrixScalar& rhs) { return lhs(0) + rhs.value; }
double operator*(const PrivateMatrixScalar& lhs, const ScalarArray& rhs) { return lhs.value + rhs(0); }
double operator*(const ScalarArray& lhs, const PrivateMatrixScalar& rhs) { return lhs(0) + rhs.value; }
double operator+(const PrivateMatrixScalar& lhs, const ScalarArray& rhs) { return 2 * lhs.value + rhs(0); }
double operator+(const ScalarArray& lhs, const PrivateMatrixScalar& rhs) { return lhs(0) + 2 * rhs.value; }
double pow(const ScalarArray& base, const PrivateMatrixScalar& exponent) { return base(0) + exponent.value; }
double pow(const Eigen::Matrix2d& base, const PrivateMatrixScalar& exponent) {
  return std::pow(base.sum(), exponent.value);
}

AnnoyingScalar operator*(int scalar, const AnnoyingScalar& value) { return AnnoyingScalar(scalar) * value; }
AnnoyingScalar pow(const AnnoyingScalar& base, const AnnoyingScalar& exponent) {
  return AnnoyingScalar(std::pow(*base.v, *exponent.v));
}

namespace Eigen {

template <>
struct NumTraits<BaseReferenceScalar> : NumTraits<double> {
  static constexpr bool RequireInitialization = true;
};
template <typename BinaryOp>
struct ScalarBinaryOpTraits<BaseReferenceScalar, double, BinaryOp> {
  using ReturnType = BaseReferenceScalar;
};
template <typename BinaryOp>
struct ScalarBinaryOpTraits<double, BaseReferenceScalar, BinaryOp> {
  using ReturnType = BaseReferenceScalar;
};
template <typename BinaryOp>
struct ScalarBinaryOpTraits<BaseReferenceScalar, ScalarMatrix, BinaryOp> {
  using ReturnType = BaseReferenceScalar;
};
template <typename BinaryOp>
struct ScalarBinaryOpTraits<BaseReferenceScalar, ScalarArray, BinaryOp> {
  using ReturnType = BaseReferenceScalar;
};
template <typename BinaryOp>
struct ScalarBinaryOpTraits<BaseReferenceMatrix, double, BinaryOp> {
  using ReturnType = BaseReferenceMatrix;
};
template <typename BinaryOp>
struct ScalarBinaryOpTraits<MatrixBase<BaseReferenceMatrix>, double, BinaryOp> {
  using ReturnType = BaseReferenceMatrix;
};
template <typename BinaryOp>
struct ScalarBinaryOpTraits<BaseReferenceArray, double, BinaryOp> {
  using ReturnType = BaseReferenceArray;
};
template <typename BinaryOp>
struct ScalarBinaryOpTraits<ArrayBase<BaseReferenceArray>, double, BinaryOp> {
  using ReturnType = BaseReferenceArray;
};

template <>
struct NumTraits<DiagonalOperandScalar> : NumTraits<double> {};
template <typename BinaryOp>
struct ScalarBinaryOpTraits<DiagonalOperandScalar, double, BinaryOp> {
  using ReturnType = DiagonalOperandScalar;
};
template <typename BinaryOp>
struct ScalarBinaryOpTraits<double, DiagonalOperandScalar, BinaryOp> {
  using ReturnType = DiagonalOperandScalar;
};
template <typename BinaryOp>
struct ScalarBinaryOpTraits<DiagonalOperandScalar, ScalarDiagonal, BinaryOp> {
  using ReturnType = DiagonalOperandScalar;
};
template <typename BinaryOp>
struct ScalarBinaryOpTraits<ScalarDiagonal, DiagonalOperandScalar, BinaryOp> {
  using ReturnType = DiagonalOperandScalar;
};
template <typename BinaryOp>
struct ScalarBinaryOpTraits<DiagonalOperandMatrix, double, BinaryOp> {
  using ReturnType = DiagonalOperandMatrix;
};
template <typename BinaryOp>
struct ScalarBinaryOpTraits<double, DiagonalOperandMatrix, BinaryOp> {
  using ReturnType = DiagonalOperandMatrix;
};

template <>
struct ScalarBinaryOpTraits<PrivateMatrixScalar, ScalarMatrix,
                            internal::scalar_product_op<PrivateMatrixScalar, ScalarMatrix>> {
  using ReturnType = double;
};
template <>
struct ScalarBinaryOpTraits<ScalarMatrix, PrivateMatrixScalar,
                            internal::scalar_product_op<ScalarMatrix, PrivateMatrixScalar>> {
  using ReturnType = double;
};
template <>
struct ScalarBinaryOpTraits<PrivateMatrixScalar, ScalarArray,
                            internal::scalar_product_op<PrivateMatrixScalar, ScalarArray>> {
  using ReturnType = double;
};
template <>
struct ScalarBinaryOpTraits<ScalarArray, PrivateMatrixScalar,
                            internal::scalar_product_op<ScalarArray, PrivateMatrixScalar>> {
  using ReturnType = double;
};
template <>
struct ScalarBinaryOpTraits<PrivateMatrixScalar, ScalarArray,
                            internal::scalar_sum_op<PrivateMatrixScalar, ScalarArray>> {
  using ReturnType = double;
};
template <>
struct ScalarBinaryOpTraits<ScalarArray, PrivateMatrixScalar,
                            internal::scalar_sum_op<ScalarArray, PrivateMatrixScalar>> {
  using ReturnType = double;
};
template <>
struct ScalarBinaryOpTraits<ScalarArray, PrivateMatrixScalar,
                            internal::scalar_pow_op<ScalarArray, PrivateMatrixScalar>> {
  using ReturnType = double;
};

template <typename Kind>
struct ScalarBinaryOpTraits<double, EigenBaseScalar<Kind>, internal::scalar_product_op<double, EigenBaseScalar<Kind>>> {
  using ReturnType = double;
};
template <typename Kind>
struct ScalarBinaryOpTraits<EigenBaseScalar<Kind>, double, internal::scalar_product_op<EigenBaseScalar<Kind>, double>> {
  using ReturnType = double;
};

template <>
struct ScalarBinaryOpTraits<PrivateMatrixScalar, BoundedRows,
                            internal::scalar_product_op<PrivateMatrixScalar, BoundedRows>> {
  using ReturnType = Matrix2d;
};

template <>
struct ScalarBinaryOpTraits<BoundedCols, PrivateMatrixScalar,
                            internal::scalar_product_op<BoundedCols, PrivateMatrixScalar>> {
  using ReturnType = Matrix2d;
};

template <>
struct ScalarBinaryOpTraits<PrivateMatrixScalar, BoundedArray,
                            internal::scalar_product_op<PrivateMatrixScalar, BoundedArray>> {
  using ReturnType = Array22d;
};

template <>
struct ScalarBinaryOpTraits<BoundedArray, PrivateMatrixScalar,
                            internal::scalar_product_op<BoundedArray, PrivateMatrixScalar>> {
  using ReturnType = Array22d;
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<double, PrivateMatrixScalar, BinaryOp> {
  using ReturnType = double;
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<PrivateMatrixScalar, double, BinaryOp> {
  using ReturnType = double;
};

template <>
struct ScalarBinaryOpTraits<PrivateMatrixScalar, Matrix2d,
                            internal::scalar_quotient_op<PrivateMatrixScalar, Matrix2d>> {
  using ReturnType = double;
};

template <>
struct ScalarBinaryOpTraits<PrivateMatrixScalar, Matrix2d, internal::scalar_product_op<PrivateMatrixScalar, Matrix2d>> {
  using ReturnType = Matrix2d;
};

template <>
struct ScalarBinaryOpTraits<Matrix2d, PrivateMatrixScalar, internal::scalar_pow_op<Matrix2d, PrivateMatrixScalar>> {
  using ReturnType = double;
};

template <>
struct ScalarBinaryOpTraits<Quaterniond, Matrix3d, internal::scalar_product_op<Quaterniond, Matrix3d>> {
  using ReturnType = Matrix3d;
};

// Broad traits must not make both generic scalar overloads compete with matrix/array operations.
template <typename T, typename BinaryOp>
struct ScalarBinaryOpTraits<AnnoyingScalar, T, BinaryOp> {
  using ReturnType = AnnoyingScalar;
};

template <typename T, typename BinaryOp>
struct ScalarBinaryOpTraits<T, AnnoyingScalar, BinaryOp> {
  using ReturnType = AnnoyingScalar;
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<AnnoyingScalar, AnnoyingScalar, BinaryOp> {
  using ReturnType = AnnoyingScalar;
};

}  // namespace Eigen

template <typename MatrixType>
void scalar_binary_op_overloads() {
  using Scalar = typename MatrixType::Scalar;
  MatrixType a(3, 3), b(3, 3);
  Matrix3f af, bf;
  for (Index j = 0; j < 3; ++j) {
    for (Index i = 0; i < 3; ++i) {
      af(i, j) = float(i + 2 * j + 1);
      bf(i, j) = float(2 * i - j + 1);
      a(i, j) = Scalar(af(i, j));
      b(i, j) = Scalar(bf(i, j));
    }
  }

  Matrix<Scalar, Dynamic, Dynamic, MatrixType::Options> result = a * b;
  VERIFY_IS_APPROX(result.template cast<float>(), af * bf);
  result = a.transpose() * b.template leftCols<2>();
  VERIFY_IS_APPROX(result.template cast<float>(), af.transpose() * bf.template leftCols<2>());
  result = a * b.col(0);
  VERIFY_IS_APPROX(result.template cast<float>(), af * bf.col(0));
  const SparseMatrix<Scalar, MatrixType::Options> sparse_a = a.sparseView();
  result = sparse_a * b;
  VERIFY_IS_APPROX(result.template cast<float>(), af * bf);
  result = a * sparse_a;
  VERIFY_IS_APPROX(result.template cast<float>(), af * af);
  result = sparse_a * sparse_a;
  VERIFY_IS_APPROX(result.template cast<float>(), af * af);

  result = a * Scalar(2);
  VERIFY_IS_APPROX(result.template cast<float>(), af * 2);
  result = Scalar(2) * a;
  VERIFY_IS_APPROX(result.template cast<float>(), 2 * af);
  result = a * 2;
  VERIFY_IS_APPROX(result.template cast<float>(), af * 2);
  result = 2 * a;
  VERIFY_IS_APPROX(result.template cast<float>(), 2 * af);

  const auto aa = a.array();
  const auto ba = b.array();
  VERIFY_IS_APPROX((aa * ba).template cast<float>(), af.array() * bf.array());
  VERIFY_IS_APPROX((aa + ba).template cast<float>(), af.array() + bf.array());
  VERIFY_IS_APPROX((aa - ba).template cast<float>(), af.array() - bf.array());
  VERIFY_IS_APPROX((aa / (ba + Scalar(10))).template cast<float>(), af.array() / (bf.array() + 10));
  VERIFY_IS_APPROX((aa + Scalar(2)).template cast<float>(), af.array() + 2);
  VERIFY_IS_APPROX((Scalar(2) + aa).template cast<float>(), 2 + af.array());
  VERIFY_IS_APPROX(Eigen::pow(aa, ba).template cast<float>(), af.array().pow(bf.array()));
  VERIFY_IS_APPROX(Eigen::pow(af.array(), ba).template cast<float>(), af.array().pow(bf.array()));
  VERIFY_IS_APPROX(Eigen::pow(aa, bf.array()).template cast<float>(), af.array().pow(bf.array()));
  VERIFY_IS_APPROX(Eigen::pow(Scalar(2), ba).template cast<float>(), Eigen::pow(2.0f, bf.array()));
  VERIFY_IS_APPROX(Eigen::pow(af.array(), 2).eval(), af.array().square());

  struct DerivedMatrix : MatrixType {
    explicit DerivedMatrix(const MatrixType& matrix) : MatrixType(matrix) {}
  };
  DerivedMatrix derived_a(a), derived_b(b);
  result = derived_a * derived_b;
  VERIFY_IS_APPROX(result.template cast<float>(), af * bf);
}

template <int Rows, int Cols>
void bounded_array_scalar_binary_op_overloads() {
  Array<PrivateMatrixScalar, Rows, Cols> coefficients;
  for (Index i = 0; i < coefficients.size(); ++i) coefficients(i) = PrivateMatrixScalar(i + 1);
  const BoundedArray operand = Array22d::Random();
  const Array<Array22d, Rows, Cols> right = coefficients * operand;
  const Array<Array22d, Rows, Cols> left = operand * coefficients;
  for (Index i = 0; i < coefficients.size(); ++i) {
    VERIFY_IS_APPROX(right(i), (i + 1) * operand);
    VERIFY_IS_APPROX(left(i), (i + 1) * operand);
  }
}

EIGEN_DECLARE_TEST(scalar_binary_op_overloads) {
  quantity_scalar_product<2, ColMajor, 2>();
  quantity_scalar_product<2, RowMajor, 2>();
  BaseReferenceMatrix base_matrix;
  base_matrix(0) = BaseReferenceScalar(2);
  const MatrixBase<BaseReferenceMatrix>& matrix_base = base_matrix;
  const ScalarMatrix base_matrix_operand = ScalarMatrix::Constant(3);
  const BaseReferenceMatrix base_matrix_result = matrix_base * base_matrix_operand;
  VERIFY_IS_EQUAL(base_matrix_result(0).value, 5);
  BaseReferenceArray base_array;
  base_array(0) = BaseReferenceScalar(2);
  const ArrayBase<BaseReferenceArray>& array_base = base_array;
  const ScalarArray base_array_operand = ScalarArray::Constant(3);
  const BaseReferenceArray base_array_result = array_base * base_array_operand;
  VERIFY_IS_EQUAL(base_array_result(0).value, 5);
  DiagonalOperandMatrix diagonal_coefficients;
  diagonal_coefficients(0) = DiagonalOperandScalar(2);
  ScalarDiagonal diagonal_operand;
  diagonal_operand.diagonal()(0) = 3;
  const DiagonalOperandMatrix diagonal_right = diagonal_coefficients * diagonal_operand;
  const DiagonalOperandMatrix diagonal_left = diagonal_operand * diagonal_coefficients;
  VERIFY_IS_EQUAL(diagonal_right(0).value, 5);
  VERIFY_IS_EQUAL(diagonal_left(0).value, 5);
  Matrix<PrivateMatrixScalar, 1, 1> registered_matrix;
  registered_matrix(0) = PrivateMatrixScalar(2);
  const ScalarMatrix matrix_operand = ScalarMatrix::Constant(3);
  const ScalarMatrix matrix_right = registered_matrix * matrix_operand;
  const ScalarMatrix matrix_left = matrix_operand * registered_matrix;
  VERIFY_IS_EQUAL(matrix_right(0), 5);
  VERIFY_IS_EQUAL(matrix_left(0), 5);
  const Array<PrivateMatrixScalar, 1, 1> registered_array = registered_matrix.array();
  const ScalarArray array_operand = ScalarArray::Constant(3);
  const ScalarArray array_right = registered_array * array_operand;
  const ScalarArray array_left = array_operand * registered_array;
  const ScalarArray sum_right = registered_array + array_operand;
  const ScalarArray sum_left = array_operand + registered_array;
  const ScalarArray power = Eigen::pow(array_operand, registered_array);
  VERIFY_IS_EQUAL(array_right(0), 5);
  VERIFY_IS_EQUAL(array_left(0), 5);
  VERIFY_IS_EQUAL(sum_right(0), 7);
  VERIFY_IS_EQUAL(sum_left(0), 7);
  VERIFY_IS_EQUAL(power(0), 5);
  const EigenBaseScalar<MatrixXpr> matrix_scale;
  const Matrix2d matrix = Matrix2d::Ones();
  VERIFY_IS_EQUAL((matrix * matrix_scale).eval(), 3 * matrix);
  VERIFY_IS_EQUAL((matrix_scale * matrix).eval(), 3 * matrix);
  const EigenBaseScalar<ArrayXpr> array_scale;
  const Array22d array = Array22d::Ones();
  VERIFY(((array * array_scale) == 3 * array).all());
  VERIFY(((array_scale * array) == 3 * array).all());

  using SparseMatrixType = SparseMatrix<double>;
  using SparseView = SparseSelfAdjointView<SparseMatrixType, Lower>;
  using DenseView = SelfAdjointView<Matrix2d, Lower>;
  static_assert(internal::has_generic_scalar_binary_ops<SparseMatrixType, SparseMatrixType, MatrixXpr>::value,
                "sparse generic scalar operators");
  static_assert(internal::has_generic_scalar_binary_ops<Matrix2d, Matrix2d, MatrixXpr>::value,
                "dense generic scalar operators");
  static_assert(internal::has_generic_scalar_binary_ops<Array22d, Array22d, ArrayXpr>::value,
                "array generic scalar operators");
  static_assert(!internal::has_generic_scalar_binary_ops<SparseView, SparseView, MatrixXpr>::value,
                "sparse self-adjoint scaling uses fixed scalar overloads");
  static_assert(!internal::has_generic_scalar_binary_ops<DenseView, DenseView, MatrixXpr>::value,
                "dense self-adjoint scaling uses fixed scalar overloads");
  static_assert(!internal::has_generic_scalar_binary_ops<ScalarDiagonal, ScalarDiagonal, MatrixXpr>::value,
                "diagonal scaling uses fixed scalar overloads");

  struct PrivateMatrix : private Matrix3f {};
  struct UnrelatedDerived {
    const UnrelatedDerived& derived() const;
  };
  static_assert(internal::is_eigen_expression<const Matrix3f&>::value, "cv-qualified expressions");
  static_assert(internal::is_eigen_expression<MatrixBase<Matrix3f>>::value, "expression base references");
  static_assert(!internal::is_eigen_expression<AnnoyingScalar>::value, "custom scalars");
  static_assert(!internal::is_eigen_expression<PrivateMatrix>::value, "private implementation bases");
  static_assert(!internal::is_eigen_expression<PrivateMatrixScalar>::value, "public derived with private bases");
  static_assert(!internal::is_eigen_expression<UnrelatedDerived>::value, "unrelated derived methods");
  static_assert(internal::shape_max_dimensions<UnrelatedDerived>::Rows == Dynamic, "missing maximum metadata");
  static_assert(internal::dimension_relation<Dynamic, Dynamic, 2, 3>::Compatible,
                "unequal maxima do not imply unequal dimensions");
  static_assert(internal::dimension_relation<2, Dynamic, 2, 2>::Compatible, "matching at the capacity limit");
  static_assert(!internal::dimension_relation<3, Dynamic, 3, 2>::Compatible, "fixed above dynamic maximum");
  static_assert(!internal::dimension_relation<Dynamic, 3, 2, 3>::Compatible, "dynamic maximum below fixed");
  static_assert(!internal::has_competing_expression_binary_op<Matrix<PrivateMatrixScalar, 1, 2>, BoundedRows,
                                                              internal::scalar_product_op, false>::value,
                "registered scalar products remain valid even when dimensions also match");
  scalar_binary_op_overloads<Matrix<AnnoyingScalar, 3, 3, ColMajor>>();
  scalar_binary_op_overloads<Matrix<AnnoyingScalar, 3, 3, RowMajor>>();
  scalar_binary_op_overloads<Matrix<AnnoyingScalar, Dynamic, Dynamic, ColMajor>>();
  scalar_binary_op_overloads<Matrix<AnnoyingScalar, Dynamic, Dynamic, RowMajor>>();
  VERIFY_IS_EQUAL(AnnoyingScalar::instances, 0);

  Matrix<PrivateMatrixScalar, 1, 3> bounded_coefficients;
  for (Index i = 0; i < 3; ++i) bounded_coefficients(i) = PrivateMatrixScalar(i + 1);
  const BoundedRows bounded_rows = Matrix2d::Random();
  const BoundedCols bounded_cols = Matrix2d::Random();
  const Matrix<Matrix2d, 1, 3> bounded_right = bounded_coefficients * bounded_rows;
  const Matrix<PrivateMatrixScalar, 3, 1> bounded_column = bounded_coefficients.transpose();
  const Matrix<Matrix2d, 3, 1> bounded_left = bounded_cols * bounded_column;
  for (Index i = 0; i < 3; ++i) {
    VERIFY_IS_APPROX(bounded_right(i), (i + 1) * bounded_rows);
    VERIFY_IS_APPROX(bounded_left(i), (i + 1) * bounded_cols);
  }
  bounded_array_scalar_binary_op_overloads<3, 1>();
  bounded_array_scalar_binary_op_overloads<1, 3>();

  const PrivateMatrixScalar private_scale(3);
  const Matrix2d private_right = Matrix2d::Ones() * private_scale;
  const Matrix2d private_left = private_scale * Matrix2d::Ones();
  VERIFY_IS_APPROX(private_right, 3 * Matrix2d::Ones());
  VERIFY_IS_APPROX(private_left, 3 * Matrix2d::Ones());

  Array<Quaterniond, 2, 1> rotations;
  rotations(0) = Quaterniond::Identity();
  rotations(1) = Quaterniond(AngleAxisd(0.5, Vector3d::UnitZ()));
  const Matrix3d operand = Matrix3d::Random();
  const Array<Matrix3d, 2, 1> rotated = rotations * operand;
  for (Index i = 0; i < rotations.size(); ++i) {
    VERIFY_IS_APPROX(rotated(i), rotations(i).toRotationMatrix() * operand);
  }
  const Matrix<Quaterniond, 2, 1> matrix_rotations = rotations.matrix();
  const Matrix<Matrix3d, 2, 1> matrix_rotated = matrix_rotations * operand;
  for (Index i = 0; i < rotations.size(); ++i) {
    VERIFY_IS_APPROX(matrix_rotated(i), rotations(i).toRotationMatrix() * operand);
  }

  Matrix<PrivateMatrixScalar, 2, 1> dividends;
  dividends(0) = PrivateMatrixScalar(4);
  dividends(1) = PrivateMatrixScalar(8);
  const Matrix2d divisor = Matrix2d::Ones();
  const Vector2d quotients = dividends / divisor;
  VERIFY_IS_EQUAL(quotients(0), 1);
  VERIFY_IS_EQUAL(quotients(1), 2);
  const Matrix<Matrix2d, 2, 1> multiplied = dividends * divisor;
  VERIFY_IS_APPROX(multiplied(0), 4 * divisor);
  VERIFY_IS_APPROX(multiplied(1), 8 * divisor);
  const Array<PrivateMatrixScalar, 2, 1> exponents = dividends.array();
  const Array2d powers = Eigen::pow(divisor, exponents);
  VERIFY_IS_EQUAL(powers(0), 256);
  VERIFY_IS_EQUAL(powers(1), 65536);

  Array<Matrix2f, 2, 1> matrices;
  matrices(0) = Matrix2f::Identity();
  matrices(1) = Matrix2f::Ones();
  const Matrix2f scale = 2 * Matrix2f::Identity();
  const Array<Matrix2f, 2, 1> scaled = matrices * scale;
  VERIFY_IS_APPROX(scaled(0), scale);
  VERIFY_IS_APPROX(scaled(1), 2 * Matrix2f::Ones());
  const Array<Matrix2f, 2, 1> left_scaled = scale * matrices;
  VERIFY_IS_APPROX(left_scaled(0), scale);
  VERIFY_IS_APPROX(left_scaled(1), 2 * Matrix2f::Ones());
}
