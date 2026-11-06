// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/Geometry>

template <typename Scalar, int Size>
void homogeneous(void) {
  /* this test covers the following files:
     Homogeneous.h
  */

  using MatrixType = Matrix<Scalar, Size, Size>;
  using VectorType = Matrix<Scalar, Size, 1, ColMajor>;

  using HMatrixType = Matrix<Scalar, Size + 1, Size>;
  using HVectorType = Matrix<Scalar, Size + 1, 1>;

  using T1MatrixType = Matrix<Scalar, Size, Size + 1>;
  using T2MatrixType = Matrix<Scalar, Size + 1, Size + 1>;
  using T3MatrixType = Matrix<Scalar, Size + 1, Size>;

  VectorType v0 = VectorType::Random(), ones = VectorType::Ones();

  HVectorType hv0 = HVectorType::Random();

  MatrixType m0 = MatrixType::Random();

  HMatrixType hm0 = HMatrixType::Random();

  hv0 << v0, 1;
  VERIFY_IS_APPROX(v0.homogeneous(), hv0);
  VERIFY_IS_APPROX(v0, hv0.hnormalized());

  VERIFY_IS_APPROX(v0.homogeneous().sum(), hv0.sum());
  VERIFY_IS_APPROX(v0.homogeneous().prod(), hv0.prod());
  VERIFY_IS_APPROX(v0.homogeneous().minCoeff(), hv0.minCoeff());
  VERIFY_IS_APPROX(v0.homogeneous().maxCoeff(), hv0.maxCoeff());

  hm0 << m0, ones.transpose();
  VERIFY_IS_APPROX(m0.colwise().homogeneous(), hm0);
  VERIFY_IS_APPROX(m0.colwise().homogeneous().sum(), hm0.sum());
  VERIFY_IS_APPROX(m0.transpose().rowwise().homogeneous().sum(), hm0.sum());
  VERIFY_IS_APPROX(m0, hm0.colwise().hnormalized());
  hm0.row(Size - 1).setRandom();
  for (int j = 0; j < Size; ++j) m0.col(j) = hm0.col(j).head(Size) / hm0(Size, j);
  VERIFY_IS_APPROX(m0, hm0.colwise().hnormalized());

  T1MatrixType t1 = T1MatrixType::Random();
  VERIFY_IS_APPROX(t1 * (v0.homogeneous().eval()), t1 * v0.homogeneous());
  VERIFY_IS_APPROX(t1 * (m0.colwise().homogeneous().eval()), t1 * m0.colwise().homogeneous());

  T2MatrixType t2 = T2MatrixType::Random();
  VERIFY_IS_APPROX(t2 * (v0.homogeneous().eval()), t2 * v0.homogeneous());
  VERIFY_IS_APPROX(t2 * (m0.colwise().homogeneous().eval()), t2 * m0.colwise().homogeneous());
  VERIFY_IS_APPROX(t2 * (v0.homogeneous().asDiagonal()), t2 * hv0.asDiagonal());
  VERIFY_IS_APPROX((v0.homogeneous().asDiagonal()) * t2, hv0.asDiagonal() * t2);

  VERIFY_IS_APPROX((v0.transpose().rowwise().homogeneous().eval()) * t2, v0.transpose().rowwise().homogeneous() * t2);
  VERIFY_IS_APPROX((m0.transpose().rowwise().homogeneous().eval()) * t2, m0.transpose().rowwise().homogeneous() * t2);

  T3MatrixType t3 = T3MatrixType::Random();
  VERIFY_IS_APPROX((v0.transpose().rowwise().homogeneous().eval()) * t3, v0.transpose().rowwise().homogeneous() * t3);
  VERIFY_IS_APPROX((m0.transpose().rowwise().homogeneous().eval()) * t3, m0.transpose().rowwise().homogeneous() * t3);

  // test product with a Transform object
  Transform<Scalar, Size, Affine> aff;
  Transform<Scalar, Size, AffineCompact> caff;
  Transform<Scalar, Size, Projective> proj;
  Matrix<Scalar, Size, Dynamic> pts;
  Matrix<Scalar, Size + 1, Dynamic> pts1, pts2;

  aff.affine().setRandom();
  proj = caff = aff;
  pts.setRandom(Size, internal::random<int>(1, 20));

  pts1 = pts.colwise().homogeneous();
  VERIFY_IS_APPROX(aff * pts.colwise().homogeneous(), (aff * pts1).colwise().hnormalized());
  VERIFY_IS_APPROX(caff * pts.colwise().homogeneous(), (caff * pts1).colwise().hnormalized());
  VERIFY_IS_APPROX(proj * pts.colwise().homogeneous(), (proj * pts1));

  VERIFY_IS_APPROX((aff * pts1).colwise().hnormalized(), aff * pts);
  VERIFY_IS_APPROX((caff * pts1).colwise().hnormalized(), caff * pts);

  pts2 = pts1;
  pts2.row(Size).setRandom();
  VERIFY_IS_APPROX((aff * pts2).colwise().hnormalized(), aff * pts2.colwise().hnormalized());
  VERIFY_IS_APPROX((caff * pts2).colwise().hnormalized(), caff * pts2.colwise().hnormalized());
  VERIFY_IS_APPROX((proj * pts2).colwise().hnormalized(),
                   (proj * pts2.colwise().hnormalized().colwise().homogeneous()).colwise().hnormalized());

  // Test combination of homogeneous

  VERIFY_IS_APPROX((t2 * v0.homogeneous()).hnormalized(),
                   (t2.template topLeftCorner<Size, Size>() * v0 + t2.template topRightCorner<Size, 1>()) /
                       ((t2.template bottomLeftCorner<1, Size>() * v0).value() + t2(Size, Size)));

  {
    using Wide = std::conditional_t<std::is_same<Scalar, float>::value, double, long double>;
    const Matrix<Scalar, Size, Dynamic> normalized[] = {
        (t2 * pts.colwise().homogeneous()).colwise().hnormalized(),
        Matrix<Scalar, Size + 1, Dynamic>(t2 * pts1).colwise().hnormalized(),
        (t2.lazyProduct(pts.colwise().homogeneous())).colwise().hnormalized(),
        (pts.transpose().rowwise().homogeneous().lazyProduct(t2)).rowwise().hnormalized().transpose()};
    // For q = fl(n/d), test |q*d-n| <= gamma*(sum|n terms| + |q|*sum|d terms|).
    // 2*(Size+1) product/sum roundings plus division; widened reference roundoff is included.
    const Wide rounding = Wide(2 * (Size + 1) + 1) * (Wide(NumTraits<Scalar>::epsilon()) + NumTraits<Wide>::epsilon());
    VERIFY(rounding < Wide(1));
    const Wide gamma = rounding / (Wide(1) - rounding);
    for (Index path = 0; path < 4; ++path) {
      for (Index j = 0; j < pts.cols(); ++j) {
        Wide reference[Size + 1] = {}, magnitude[Size + 1] = {};
        for (Index row = 0; row <= Size; ++row) {
          for (Index k = 0; k <= Size; ++k) {
            const Wide term = Wide(path == 3 ? t2(k, row) : t2(row, k)) * (k == Size ? Wide(1) : Wide(pts(k, j)));
            reference[row] += term;
            magnitude[row] += numext::abs(term);
          }
        }
        for (Index row = 0; row < Size; ++row) {
          const Wide q = Wide(normalized[path](row, j));
          const Wide bound = gamma * (magnitude[row] + numext::abs(q) * magnitude[Size]);
          VERIFY((numext::isfinite)(q));
          VERIFY((numext::isfinite)(bound));
          VERIFY(numext::abs(q * reference[Size] - reference[row]) <= bound);
        }
      }
    }
  }

  VERIFY_IS_APPROX((t2.lazyProduct(v0.homogeneous())).hnormalized(), (t2 * v0.homogeneous()).hnormalized());

  VERIFY_IS_APPROX((v0.transpose().homogeneous().lazyProduct(t2)).hnormalized(),
                   (v0.transpose().homogeneous() * t2).hnormalized());

  VERIFY_IS_APPROX((t2.template triangularView<Lower>() * v0.homogeneous()).eval(),
                   (t2.template triangularView<Lower>() * hv0));

  {
    const MatrixType points = MatrixType::Random();
    const VectorType center = VectorType::Random();

    const auto pts3 = points.rowwise() - center.transpose();
    const auto pts_xy1 = pts3.template leftCols<Size - 1>().rowwise().homogeneous();
    const auto pts_xy2 = pts3.template topRows<Size - 1>().colwise().homogeneous();

    VERIFY_IS_APPROX(pts_xy1.transpose() * pts_xy1, pts_xy1.transpose() * pts_xy1.eval());
    VERIFY_IS_APPROX(pts_xy2 * pts_xy2.transpose(), pts_xy2.eval() * pts_xy2.transpose());
  }

  {
    PermutationMatrix<Size> P{Vector<int, Size>::EqualSpaced(0, 1).reverse()};
    auto right = Vector<Scalar, Size - 1>::Random().eval().nestByValue().homogeneous();
    auto left = RowVector<Scalar, Size - 1>::Random().eval().nestByValue().homogeneous();

    VERIFY_IS_APPROX(P * right, right.reverse());
    VERIFY_IS_APPROX(left * P, left.reverse());
  }
}

template <typename Expression>
void check_empty_homogeneous_reductions(const Expression& expression, Index appended_ones) {
  using Scalar = typename Expression::Scalar;
  VERIFY_IS_EQUAL(expression.size(), appended_ones);
  VERIFY_IS_EQUAL(expression.sum(), Scalar(appended_ones));
  VERIFY_IS_EQUAL(expression.prod(), Scalar(1));
  if (appended_ones > 0) {
    VERIFY_IS_EQUAL(expression.minCoeff(), Scalar(1));
    VERIFY_IS_EQUAL(expression.maxCoeff(), Scalar(1));
    // This associative functor has identity -1, not a guessed sum/product identity.
    VERIFY_IS_EQUAL(expression.redux([](Scalar a, Scalar b) { return a + b + Scalar(1); }),
                    Scalar(2 * appended_ones - 1));
  } else {
    VERIFY_RAISES_ASSERT(expression.minCoeff());
    VERIFY_RAISES_ASSERT(expression.maxCoeff());
  }
}

template <typename Scalar>
void homogeneous_empty_reductions() {
  using DynamicMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  check_empty_homogeneous_reductions(Matrix<Scalar, 0, 1>().homogeneous(), 1);
  check_empty_homogeneous_reductions(Matrix<Scalar, 1, 0>().homogeneous(), 1);
  check_empty_homogeneous_reductions(Matrix<Scalar, Dynamic, 1>(0).homogeneous(), 1);
  check_empty_homogeneous_reductions(Matrix<Scalar, 1, Dynamic>(0).homogeneous(), 1);
  check_empty_homogeneous_reductions(Matrix<Scalar, 0, 3>().colwise().homogeneous(), 3);
  check_empty_homogeneous_reductions(Matrix<Scalar, 0, 3>().rowwise().homogeneous(), 0);
  check_empty_homogeneous_reductions(Matrix<Scalar, 3, 0>().colwise().homogeneous(), 0);
  check_empty_homogeneous_reductions(Matrix<Scalar, 3, 0>().rowwise().homogeneous(), 3);
  check_empty_homogeneous_reductions(Matrix<Scalar, 0, 0>().colwise().homogeneous(), 0);
  check_empty_homogeneous_reductions(Matrix<Scalar, 0, 0>().rowwise().homogeneous(), 0);
  check_empty_homogeneous_reductions(DynamicMatrix(0, 3).colwise().homogeneous(), 3);
  check_empty_homogeneous_reductions(DynamicMatrix(0, 3).rowwise().homogeneous(), 0);
  check_empty_homogeneous_reductions(DynamicMatrix(3, 0).colwise().homogeneous(), 0);
  check_empty_homogeneous_reductions(DynamicMatrix(3, 0).rowwise().homogeneous(), 3);
  check_empty_homogeneous_reductions(Matrix<Scalar, Dynamic, Dynamic, 0, 0, 0>().colwise().homogeneous(), 0);
  check_empty_homogeneous_reductions(Matrix<Scalar, Dynamic, Dynamic, 0, 0, 0>().rowwise().homogeneous(), 0);
}

template <typename Scalar>
void homogeneous_empty_complex_reductions() {
  VERIFY_IS_EQUAL((Matrix<Scalar, 0, 1>().homogeneous().sum()), Scalar(1));
  VERIFY_IS_EQUAL((Matrix<Scalar, 1, 0>().homogeneous().prod()), Scalar(1));
  VERIFY_IS_EQUAL((Matrix<Scalar, Dynamic, 1>(0).homogeneous().prod()), Scalar(1));
  VERIFY_IS_EQUAL((Matrix<Scalar, 1, Dynamic>(0).homogeneous().sum()), Scalar(1));
  VERIFY_IS_EQUAL((Matrix<Scalar, 0, 3>().colwise().homogeneous().sum()), Scalar(3));
  VERIFY_IS_EQUAL((Matrix<Scalar, 3, 0>().rowwise().homogeneous().sum()), Scalar(3));
}

EIGEN_DECLARE_TEST(geo_homogeneous) {
  CALL_SUBTEST_1(homogeneous_empty_reductions<float>());
  CALL_SUBTEST_2(homogeneous_empty_reductions<double>());
  CALL_SUBTEST_2(homogeneous_empty_complex_reductions<std::complex<double>>());
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1((homogeneous<float, 1>()));
    CALL_SUBTEST_2((homogeneous<double, 3>()));
    CALL_SUBTEST_3((homogeneous<double, 8>()));
  }
}
