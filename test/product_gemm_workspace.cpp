// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_RUNTIME_NO_MALLOC
#include "main.h"
#include "NonCommutativeScalar.h"

namespace unannotated_product {
static Index multiplications = 0;
struct Scalar {
  double value;
  Scalar(double v = 0) : value(v) {}
  Scalar operator+(const Scalar& other) const { return value + other.value; }
  Scalar operator-(const Scalar& other) const { return value - other.value; }
  Scalar operator-() const { return -value; }
  Scalar operator*(const Scalar& other) const {
    ++multiplications;
    return value * other.value;
  }
  Scalar& operator+=(const Scalar& other) { return *this = *this + other; }
  Scalar& operator-=(const Scalar& other) { return *this = *this - other; }
  Scalar& operator*=(const Scalar& other) { return *this = *this * other; }
  bool operator==(const Scalar& other) const { return value == other.value; }
  bool operator!=(const Scalar& other) const { return !(*this == other); }
};
}  // namespace unannotated_product

namespace Eigen {
template <>
struct NumTraits<unannotated_product::Scalar> : GenericNumTraits<unannotated_product::Scalar> {
  static constexpr bool RequireInitialization = true;
};
}  // namespace Eigen

template <typename Scalar, int Order, bool Bounded = false>
void fixed_gemm_workspace() {
  using MatrixType = Matrix<Scalar, Bounded ? Dynamic : 16, Bounded ? Dynamic : 16, Order, 16, 16>;
  STATIC_CHECK((!internal::functor_is_commutative<internal::scalar_product_op<Scalar>>::value));
  STATIC_CHECK((internal::product_type<MatrixType, MatrixType>::value == GemmProduct));
  const MatrixType a = MatrixType::Constant(16, 16, Scalar(2)), b = MatrixType::Constant(16, 16, Scalar(3));
  MatrixType result(16, 16);
  const Scalar scale(2);
  const auto check = [&](double expected) {
    for (Index i = 0; i < result.size(); ++i) VERIFY(result(i) == Scalar(expected));
  };
  internal::set_is_malloc_allowed(false);
  result.noalias() = a * b;
  check(96);
  result.noalias() += a * b;
  check(192);
  result.noalias() -= a * b;
  check(96);
  result.noalias() = a * (b * scale);
  check(192);
  result.noalias() = (scale * a) * b;
  check(192);
  result.noalias() = scale * (a * b);
  check(192);
  using Vector = Matrix<Scalar, Bounded ? Dynamic : 16, 1, ColMajor, 16, 1>;
  const Vector x = Vector::Constant(16, Scalar(3));
  Vector y(16);
  y.noalias() = a * x;
  for (Index i = 0; i < y.size(); ++i) VERIFY(y(i) == Scalar(96));
  y.noalias() -= a * x;
  for (Index i = 0; i < y.size(); ++i) VERIFY(y(i) == Scalar(0));
  internal::set_is_malloc_allowed(true);
}

template <int Order>
void quaternion_unit_scale() {
  using Q = noncommutative_scalar::Quaternion;
  using MatrixType = Matrix<Q, Dynamic, Dynamic, Order>;
  MatrixType a = MatrixType::Zero(3, 3), b = MatrixType::Zero(3, 3);
  a(0, 0) = Q(NumTraits<double>::infinity());
  b(0, 0) = Q(1, 1, 1, 1);
  const MatrixType result = a * b;
  const Q value = result(0, 0);
  for (double component : {value.r, value.i, value.j, value.k}) {
    VERIFY((numext::isinf)(component));
    VERIFY(component > 0);
  }
}

template <int Order, bool Left>
void ordered_gemv_scale_cost() {
  using Scalar = unannotated_product::Scalar;
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using Vector = Matrix<Scalar, Dynamic, 1>;
  using Result = Matrix<Scalar, Left ? 1 : Dynamic, Left ? Dynamic : 1, Left ? RowMajor : ColMajor>;
  const Index size = 17, depth = 9;
  const MatrixType matrix = MatrixType::Constant(Left ? depth : size, Left ? size : depth, Scalar(2));
  const Vector vector = Vector::Constant(depth, Scalar(3));
  Result result(Left ? 1 : size, Left ? size : 1);
  internal::set_is_malloc_allowed(false);
  unannotated_product::multiplications = 0;
  if (Left)
    result.noalias() = Scalar(0.5) * (vector.transpose() * matrix);
  else
    result.noalias() = Scalar(0.5) * (matrix * vector);
  VERIFY_IS_EQUAL(unannotated_product::multiplications, size * depth + size);
  for (Index i = 0; i < size; ++i) VERIFY(result(i) == Scalar(27));
  internal::set_is_malloc_allowed(true);
}

EIGEN_DECLARE_TEST(product_gemm_workspace) {
  CALL_SUBTEST((ordered_gemv_scale_cost<ColMajor, false>()));
  CALL_SUBTEST((ordered_gemv_scale_cost<RowMajor, false>()));
  CALL_SUBTEST((ordered_gemv_scale_cost<ColMajor, true>()));
  CALL_SUBTEST((ordered_gemv_scale_cost<RowMajor, true>()));
  CALL_SUBTEST((fixed_gemm_workspace<unannotated_product::Scalar, ColMajor>()));
  CALL_SUBTEST((fixed_gemm_workspace<unannotated_product::Scalar, RowMajor>()));
  CALL_SUBTEST((fixed_gemm_workspace<noncommutative_scalar::Quaternion, ColMajor>()));
  CALL_SUBTEST((fixed_gemm_workspace<noncommutative_scalar::Quaternion, RowMajor>()));
  CALL_SUBTEST((fixed_gemm_workspace<unannotated_product::Scalar, ColMajor, true>()));
  CALL_SUBTEST((fixed_gemm_workspace<unannotated_product::Scalar, RowMajor, true>()));
  CALL_SUBTEST((fixed_gemm_workspace<noncommutative_scalar::Quaternion, ColMajor, true>()));
  CALL_SUBTEST((fixed_gemm_workspace<noncommutative_scalar::Quaternion, RowMajor, true>()));
  CALL_SUBTEST(quaternion_unit_scale<ColMajor>());
  CALL_SUBTEST(quaternion_unit_scale<RowMajor>());
}
