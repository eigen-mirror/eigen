// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

template <typename Scalar>
void special_numbers() {
  typedef Matrix<Scalar, Dynamic, Dynamic> MatType;
  int rows = internal::random<int>(1, 300);
  int cols = internal::random<int>(1, 300);

  Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
  Scalar inf = std::numeric_limits<Scalar>::infinity();
  Scalar s1 = internal::random<Scalar>();

  MatType m1 = MatType::Random(rows, cols), mnan = MatType::Random(rows, cols), minf = MatType::Random(rows, cols),
          mboth = MatType::Random(rows, cols);

  int n = internal::random<int>(1, 10);
  for (int k = 0; k < n; ++k) {
    mnan(internal::random<int>(0, rows - 1), internal::random<int>(0, cols - 1)) = nan;
    minf(internal::random<int>(0, rows - 1), internal::random<int>(0, cols - 1)) = inf;
  }
  mboth = mnan + minf;

  VERIFY(!m1.hasNaN());
  VERIFY(m1.allFinite());

  VERIFY(mnan.hasNaN());
  VERIFY((s1 * mnan).hasNaN());
  VERIFY(!minf.hasNaN());
  VERIFY(!(2 * minf).hasNaN());
  VERIFY(mboth.hasNaN());
  VERIFY(mboth.array().hasNaN());

  VERIFY(!mnan.allFinite());
  VERIFY(!minf.allFinite());
  VERIFY(!(minf - mboth).allFinite());
  VERIFY(!mboth.allFinite());
  VERIFY(!mboth.array().allFinite());
}

// allFinite() over every position of a single non-finite coefficient, so the first packet, interior packets and the
// scalar tail are all reached, plus finite edge values that must not register.
template <typename Scalar>
void all_finite_positions() {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  RealScalar inf = std::numeric_limits<RealScalar>::infinity();
  RealScalar nan = std::numeric_limits<RealScalar>::quiet_NaN();
  RealScalar finite_edges[] = {RealScalar(0),
                               -RealScalar(0),
                               std::numeric_limits<RealScalar>::denorm_min(),
                               -std::numeric_limits<RealScalar>::denorm_min(),
                               NumTraits<RealScalar>::highest(),
                               NumTraits<RealScalar>::lowest()};
  RealScalar non_finite[] = {inf, -inf, nan};
  for (Index size : {Index(1), Index(3), Index(7), Index(16), Index(37)}) {
    Vec v = Vec::Ones(size);
    for (Index i = 0; i < size; ++i) v(i) = Scalar(finite_edges[i % 6]);
    VERIFY(v.allFinite());
    for (Index i = 0; i < size; ++i) {
      for (RealScalar x : non_finite) {
        Vec w = v;
        w(i) = Scalar(x);
        VERIFY(!w.allFinite());
        VERIFY(!w.array().allFinite());
      }
    }
  }
  // Row-major storage and an inner-strided map take the non-linear and strided visitor paths.
  Matrix<Scalar, 5, 7, RowMajor> m = Matrix<Scalar, 5, 7, RowMajor>::Ones();
  VERIFY(m.allFinite());
  m(4, 6) = Scalar(nan);
  VERIFY(!m.allFinite());
  VERIFY((m.template topLeftCorner<4, 6>().allFinite()));
  Vec buffer = Vec::Ones(40);
  Map<Vec, 0, InnerStride<2>> strided(buffer.data(), 20);
  buffer(39) = Scalar(inf);
  VERIFY(strided.allFinite());
  buffer(38) = Scalar(-inf);
  VERIFY(!strided.allFinite());
}

template <typename Scalar>
void all_finite_complex() {
  using RealScalar = typename NumTraits<Scalar>::Real;
  RealScalar inf = std::numeric_limits<RealScalar>::infinity();
  RealScalar nan = std::numeric_limits<RealScalar>::quiet_NaN();
  Matrix<Scalar, Dynamic, 1> v = Matrix<Scalar, Dynamic, 1>::Ones(9);
  VERIFY(v.allFinite());
  for (Index i = 0; i < v.size(); ++i) {
    for (Scalar x : {Scalar(inf, 0), Scalar(0, -inf), Scalar(nan, 0), Scalar(0, nan)}) {
      Matrix<Scalar, Dynamic, 1> w = v;
      w(i) = x;
      VERIFY(!w.allFinite());
    }
  }
}

EIGEN_DECLARE_TEST(special_numbers) {
  for (int i = 0; i < 10 * g_repeat; i++) {
    CALL_SUBTEST_1(special_numbers<float>());
    CALL_SUBTEST_1(special_numbers<double>());
  }
  CALL_SUBTEST_2(all_finite_positions<float>());
  CALL_SUBTEST_2(all_finite_positions<double>());
  CALL_SUBTEST_3(all_finite_positions<half>());
  CALL_SUBTEST_3(all_finite_positions<bfloat16>());
  CALL_SUBTEST_4(all_finite_complex<std::complex<float>>());
  CALL_SUBTEST_4(all_finite_complex<std::complex<double>>());
  CALL_SUBTEST_4(VERIFY((Matrix<int, Dynamic, 1>::Constant(17, (std::numeric_limits<int>::max)()).allFinite())));
}
