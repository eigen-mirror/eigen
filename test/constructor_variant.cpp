// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <variant>

EIGEN_DECLARE_TEST(constructor_variant) {
  PermutationMatrix<3> permutation;
  permutation.setIdentity();
  std::variant<Matrix3d, Array<double, 3, 3>> permutation_variant(permutation);
  VERIFY_IS_EQUAL(permutation_variant.index(), std::size_t(0));
  VERIFY_IS_EQUAL(std::get<Matrix3d>(permutation_variant), Matrix3d::Identity());
  std::variant<Array<double, 3, 3>, Matrix3d> permutation_reversed(permutation);
  VERIFY_IS_EQUAL(permutation_reversed.index(), std::size_t(1));
  permutation.indices() << 2, 0, 1;
  permutation_variant = permutation;
  VERIFY_IS_EQUAL(std::get<Matrix3d>(permutation_variant), Matrix3d(permutation));

  const Array3d array_source(1, 2, 3);
  const Array3i array_integers(1, 2, 3);
  static_assert(!std::is_convertible<Array3d, Array3i>::value, "Mixed scalars require an explicit cast");
  static_assert(!std::is_convertible<Array3cd, Array3d>::value, "Complex-to-real requires an explicit cast");
  std::variant<Array3d, Array3i> array_variant(array_source);
  VERIFY_IS_EQUAL(array_variant.index(), std::size_t(0));
  VERIFY((std::get<Array3d>(array_variant) == array_source).all());
  array_variant = array_integers;
  VERIFY((std::get<Array3i>(array_variant) == array_integers).all());
  std::variant<Array3i, Array3d> array_reversed(array_source);
  VERIFY((std::get<Array3d>(array_reversed) == array_source).all());
  std::variant<ArrayXd, ArrayXi> array_dynamic(array_source + array_source);
  VERIFY((std::get<ArrayXd>(array_dynamic) == 2 * array_source).all());
  const Array3cd array_complex = array_source.cast<std::complex<double>>() * std::complex<double>(1, 2);
  std::variant<Array3d, Array3cd> array_complex_variant(array_complex);
  VERIFY((std::get<Array3cd>(array_complex_variant) == array_complex).all());
  array_complex_variant = array_source;
  VERIFY((std::get<Array3d>(array_complex_variant) == array_source).all());
  std::variant<Array3cd, Array3d> array_complex_reversed(array_complex);
  VERIFY((std::get<Array3cd>(array_complex_reversed) == array_complex).all());
  std::variant<ArrayXd, ArrayXcd> array_complex_expression(array_complex + array_complex);
  VERIFY((std::get<ArrayXcd>(array_complex_expression) == 2 * array_complex).all());
  array_complex_expression = array_complex + array_complex;
  VERIFY((std::get<ArrayXcd>(array_complex_expression) == 2 * array_complex).all());

  const Vector3d source(1, 2, 3);
  const Vector3i integers(1, 2, 3);
  std::variant<Vector3d, Vector3i> variant(source);
  VERIFY_IS_EQUAL(variant.index(), std::size_t(0));
  VERIFY_IS_EQUAL(std::get<Vector3d>(variant), source);
  variant = integers;
  VERIFY_IS_EQUAL(std::get<Vector3i>(variant), integers);
  std::variant<Vector3i, Vector3d> reversed(source);
  VERIFY_IS_EQUAL(std::get<Vector3d>(reversed), source);
  std::variant<VectorXd, VectorXi> dynamic(source + source);
  VERIFY_IS_EQUAL(std::get<VectorXd>(dynamic), 2 * source);

  const Vector3cd complex = source.cast<std::complex<double>>() * std::complex<double>(1, 2);
  std::variant<Vector3d, Vector3cd> complex_variant(complex);
  VERIFY_IS_EQUAL(std::get<Vector3cd>(complex_variant), complex);
  complex_variant = source;
  VERIFY_IS_EQUAL(std::get<Vector3d>(complex_variant), source);
  complex_variant = complex;
  VERIFY_IS_EQUAL(std::get<Vector3cd>(complex_variant), complex);
  std::variant<Vector3cd, Vector3d> complex_reversed(complex);
  VERIFY_IS_EQUAL(std::get<Vector3cd>(complex_reversed), complex);
  std::variant<VectorXd, VectorXcd> complex_expression(complex + complex);
  VERIFY_IS_EQUAL(std::get<VectorXcd>(complex_expression), 2 * complex);
  complex_expression = complex + complex;
  VERIFY_IS_EQUAL(std::get<VectorXcd>(complex_expression), 2 * complex);
}
