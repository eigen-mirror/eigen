// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include "main.h"
#include <contrib/Eigen/Tensor>

EIGEN_DECLARE_TEST(tensor_shape_checks) {
  using namespace Eigen;
  STATIC_CHECK((internal::tensor_dimension_relation<Sizes<2, 3>, Sizes<2, 3>>::Proven));
  STATIC_CHECK((!internal::tensor_dimension_relation<Sizes<2, 3>, Sizes<3, 2>>::Compatible));
  STATIC_CHECK((!internal::tensor_dimension_relation<DSizes<Index, 2>, DSizes<Index, 2>>::Proven));
  STATIC_CHECK((!internal::tensor_dimension_relation<Sizes<2, 3>, DSizes<Index, 2>>::Proven));
  STATIC_CHECK((!internal::tensor_dimension_relation<DSizes<Index, 2>, DSizes<Index, 3>>::Compatible));
  TensorFixedSize<double, Sizes<2, 3>> a, b, result;
  a.setConstant(2);
  b.setConstant(3);
  result = a + b;
  for (Index i = 0; i < result.size(); ++i) VERIFY_IS_EQUAL(result.data()[i], 5);
  Tensor<double, 2> dynamic(2, 3), wrong(3, 2);
  dynamic = a;
  result = a + dynamic;
  for (Index i = 0; i < result.size(); ++i) VERIFY_IS_EQUAL(result.data()[i], 4);
  VERIFY_RAISES_ASSERT(result = a + wrong);
#ifdef EIGEN_NO_STATIC_ASSERT
  TensorFixedSize<double, Sizes<3, 2>> wrong_fixed;
  VERIFY_RAISES_ASSERT(result = a + wrong_fixed);
  Tensor<double, 3> wrong_rank(2, 3, 1);
  VERIFY_RAISES_ASSERT(EIGEN_INTERNAL_ASSERT_TENSOR_SHAPE(a, wrong_rank));
#endif
}
