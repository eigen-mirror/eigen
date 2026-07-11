// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_RUNTIME_NO_MALLOC

#include "main.h"

template <typename T>
EIGEN_DONT_INLINE T copy(const T& x) {
  return x;
}

struct StableNormCountingOp {
  explicit StableNormCountingOp(Index* counter) : count(counter) {}

  EIGEN_DONT_INLINE double operator()(Index index) const {
    ++*count;
    return double(index + 1);
  }

  Index* count;
};

template <typename MatrixType>
void stable_norm(const MatrixType& m) {
  /* this test covers the following files:
     StableNorm.h
  */
  using std::abs;
  using std::sqrt;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  bool complex_real_product_ok = true;

  // Check the basic machine-dependent constants.
  {
    int ibeta, it, iemin, iemax;

    ibeta = std::numeric_limits<RealScalar>::radix;         // base for floating-point numbers
    it = std::numeric_limits<RealScalar>::digits;           // number of base-beta digits in mantissa
    iemin = std::numeric_limits<RealScalar>::min_exponent;  // minimum exponent
    iemax = std::numeric_limits<RealScalar>::max_exponent;  // maximum exponent

    VERIFY((!(iemin > 1 - 2 * it || 1 + it > iemax || (it == 2 && ibeta < 5) || (it <= 4 && ibeta <= 3) || it < 2)) &&
           "the stable norm algorithm cannot be guaranteed on this computer");

    Scalar inf = std::numeric_limits<RealScalar>::infinity();
    if (NumTraits<Scalar>::IsComplex && (numext::isnan)(inf * RealScalar(1))) {
      complex_real_product_ok = false;
      static bool first = true;
      if (first)
        std::cerr << "WARNING: compiler mess up complex*real product, " << inf << " * " << 1.0 << " = "
                  << inf * RealScalar(1) << std::endl;
      first = false;
    }
  }

  Index rows = m.rows();
  Index cols = m.cols();

  // Get a random factor bounded away from zero: |factor| >= 0.1.
  Scalar factor = internal::random<Scalar>(Scalar(RealScalar(0.1)), Scalar(RealScalar(1)));
  Scalar big = factor * ((std::numeric_limits<RealScalar>::max)() * RealScalar(1e-4));

  factor = internal::random<Scalar>(Scalar(RealScalar(0.1)), Scalar(RealScalar(1)));
  Scalar small = factor * ((std::numeric_limits<RealScalar>::min)() * RealScalar(1e4));

  Scalar one(1);

  MatrixType vzero = MatrixType::Zero(rows, cols), vrand = MatrixType::Random(rows, cols), vbig(rows, cols),
             vsmall(rows, cols);

  vbig.fill(big);
  vsmall.fill(small);

  VERIFY_IS_MUCH_SMALLER_THAN(vzero.norm(), static_cast<RealScalar>(1));
  VERIFY_IS_APPROX(vrand.stableNorm(), vrand.norm());
  VERIFY_IS_APPROX(vrand.blueNorm(), vrand.norm());
  VERIFY_IS_APPROX(vrand.hypotNorm(), vrand.norm());

  // test with expressions as input
  VERIFY_IS_APPROX((one * vrand).stableNorm(), vrand.norm());
  VERIFY_IS_APPROX((one * vrand).blueNorm(), vrand.norm());
  VERIFY_IS_APPROX((one * vrand).hypotNorm(), vrand.norm());
  VERIFY_IS_APPROX((one * vrand + one * vrand - one * vrand).stableNorm(), vrand.norm());
  VERIFY_IS_APPROX((one * vrand + one * vrand - one * vrand).blueNorm(), vrand.norm());
  VERIFY_IS_APPROX((one * vrand + one * vrand - one * vrand).hypotNorm(), vrand.norm());

  RealScalar size = static_cast<RealScalar>(m.size());

  // test numext::isfinite
  VERIFY(!(numext::isfinite)(std::numeric_limits<RealScalar>::infinity()));
  VERIFY(!(numext::isfinite)(sqrt(-abs(big))));

  // test overflow
  VERIFY((numext::isfinite)(sqrt(size) * abs(big)));
  VERIFY_IS_NOT_APPROX(sqrt(copy(vbig.squaredNorm())), abs(sqrt(size) * big));  // here the default norm must fail
  VERIFY_IS_APPROX(vbig.stableNorm(), sqrt(size) * abs(big));
  VERIFY_IS_APPROX(vbig.blueNorm(), sqrt(size) * abs(big));
  VERIFY_IS_APPROX(vbig.hypotNorm(), sqrt(size) * abs(big));

  // test underflow
  VERIFY((numext::isfinite)(sqrt(size) * abs(small)));
  VERIFY_IS_NOT_APPROX(sqrt(copy(vsmall.squaredNorm())), abs(sqrt(size) * small));  // here the default norm must fail
  VERIFY_IS_APPROX(vsmall.stableNorm(), sqrt(size) * abs(small));
  VERIFY_IS_APPROX(vsmall.blueNorm(), sqrt(size) * abs(small));
  VERIFY_IS_APPROX(vsmall.hypotNorm(), sqrt(size) * abs(small));

  // Test compilation of cwise() version
  VERIFY_IS_APPROX(vrand.colwise().stableNorm(), vrand.colwise().norm());
  VERIFY_IS_APPROX(vrand.colwise().blueNorm(), vrand.colwise().norm());
  VERIFY_IS_APPROX(vrand.colwise().hypotNorm(), vrand.colwise().norm());
  VERIFY_IS_APPROX(vrand.rowwise().stableNorm(), vrand.rowwise().norm());
  VERIFY_IS_APPROX(vrand.rowwise().blueNorm(), vrand.rowwise().norm());
  VERIFY_IS_APPROX(vrand.rowwise().hypotNorm(), vrand.rowwise().norm());

  // test NaN, +inf, -inf
  MatrixType v;
  Index i = internal::random<Index>(0, rows - 1);
  Index j = internal::random<Index>(0, cols - 1);

  // NaN
  {
    v = vrand;
    v(i, j) = std::numeric_limits<RealScalar>::quiet_NaN();
    VERIFY(!(numext::isfinite)(v.squaredNorm()));
    VERIFY((numext::isnan)(v.squaredNorm()));
    VERIFY(!(numext::isfinite)(v.norm()));
    VERIFY((numext::isnan)(v.norm()));
    VERIFY(!(numext::isfinite)(v.stableNorm()));
    VERIFY((numext::isnan)(v.stableNorm()));
    VERIFY(!(numext::isfinite)(v.blueNorm()));
    VERIFY((numext::isnan)(v.blueNorm()));
    VERIFY(!(numext::isfinite)(v.hypotNorm()));
    VERIFY((numext::isnan)(v.hypotNorm()));
  }

  // +inf
  {
    v = vrand;
    v(i, j) = std::numeric_limits<RealScalar>::infinity();
    VERIFY(!(numext::isfinite)(v.squaredNorm()));
    VERIFY(isPlusInf(v.squaredNorm()));
    VERIFY(!(numext::isfinite)(v.norm()));
    VERIFY(isPlusInf(v.norm()));
    VERIFY(!(numext::isfinite)(v.stableNorm()));
    if (complex_real_product_ok) {
      VERIFY(isPlusInf(v.stableNorm()));
    }
    VERIFY(!(numext::isfinite)(v.blueNorm()));
    VERIFY(isPlusInf(v.blueNorm()));
    VERIFY(!(numext::isfinite)(v.hypotNorm()));
    VERIFY(isPlusInf(v.hypotNorm()));
  }

  // -inf
  {
    v = vrand;
    v(i, j) = -std::numeric_limits<RealScalar>::infinity();
    VERIFY(!(numext::isfinite)(v.squaredNorm()));
    VERIFY(isPlusInf(v.squaredNorm()));
    VERIFY(!(numext::isfinite)(v.norm()));
    VERIFY(isPlusInf(v.norm()));
    VERIFY(!(numext::isfinite)(v.stableNorm()));
    if (complex_real_product_ok) {
      VERIFY(isPlusInf(v.stableNorm()));
    }
    VERIFY(!(numext::isfinite)(v.blueNorm()));
    VERIFY(isPlusInf(v.blueNorm()));
    VERIFY(!(numext::isfinite)(v.hypotNorm()));
    VERIFY(isPlusInf(v.hypotNorm()));
  }

  // mix
  {
    Index i2 = internal::random<Index>(0, rows - 1);
    Index j2 = internal::random<Index>(0, cols - 1);
    v = vrand;
    v(i, j) = -std::numeric_limits<RealScalar>::infinity();
    v(i2, j2) = std::numeric_limits<RealScalar>::quiet_NaN();
    VERIFY(!(numext::isfinite)(v.squaredNorm()));
    VERIFY((numext::isnan)(v.squaredNorm()));
    VERIFY(!(numext::isfinite)(v.norm()));
    VERIFY((numext::isnan)(v.norm()));
    VERIFY(!(numext::isfinite)(v.stableNorm()));
    VERIFY((numext::isnan)(v.stableNorm()));
    VERIFY(!(numext::isfinite)(v.blueNorm()));
    VERIFY((numext::isnan)(v.blueNorm()));
    if (i2 != i || j2 != j) {
      // hypot propagates inf over NaN.
      VERIFY(!(numext::isfinite)(v.hypotNorm()));
      VERIFY((numext::isinf)(v.hypotNorm()));
    } else {
      // inf is overwritten by NaN, expect norm to be NaN.
      VERIFY(!(numext::isfinite)(v.hypotNorm()));
      VERIFY((numext::isnan)(v.hypotNorm()));
    }
  }

  // stableNormalize[d]
  {
    VERIFY_IS_APPROX(vrand.stableNormalized(), vrand.normalized());
    MatrixType vcopy(vrand);
    vcopy.stableNormalize();
    VERIFY_IS_APPROX(vcopy, vrand.normalized());
    VERIFY_IS_APPROX((vrand.stableNormalized()).norm(), RealScalar(1));
    VERIFY_IS_APPROX(vcopy.norm(), RealScalar(1));
    VERIFY_IS_APPROX((vbig.stableNormalized()).norm(), RealScalar(1));
    VERIFY_IS_APPROX((vsmall.stableNormalized()).norm(), RealScalar(1));
    RealScalar big_scaling = ((std::numeric_limits<RealScalar>::max)() * RealScalar(1e-4));
    VERIFY_IS_APPROX(vbig / big_scaling, (vbig.stableNorm() * vbig.stableNormalized()).eval() / big_scaling);
    VERIFY_IS_APPROX(vsmall, vsmall.stableNorm() * vsmall.stableNormalized());
  }
}

void test_empty() {
  Eigen::VectorXf empty(0);
  VERIFY_IS_EQUAL(empty.stableNorm(), 0.0f);
}

template <typename RealScalar>
void stable_normalize_extremes() {
  typedef Matrix<RealScalar, 2, 1> Vector2;
  typedef Matrix<RealScalar, Dynamic, 1> VectorX;
  typedef Matrix<RealScalar, Dynamic, Dynamic> MatrixX;
  using std::signbit;
  using std::sqrt;

  const RealScalar highest = (std::numeric_limits<RealScalar>::max)();
  const RealScalar denorm = std::numeric_limits<RealScalar>::denorm_min();
  const RealScalar infinity = std::numeric_limits<RealScalar>::infinity();
  const RealScalar nan = std::numeric_limits<RealScalar>::quiet_NaN();
  const RealScalar inv_sqrt_two = RealScalar(1) / sqrt(RealScalar(2));

  {
    const Vector2 input = Vector2::Constant(highest);
    const Vector2 expected = Vector2::Constant(inv_sqrt_two);
    VERIFY_IS_APPROX(input.stableNormalized(), expected);
    Vector2 actual = input;
    actual.stableNormalize();
    VERIFY_IS_APPROX(actual, expected);
    VERIFY_IS_APPROX(actual.norm(), RealScalar(1));
  }

  {
    Vector2 input;
    input << highest, highest / RealScalar(2);
    Vector2 expected;
    expected << RealScalar(1), RealScalar(0.5);
    expected.normalize();
    VERIFY_IS_APPROX(input.stableNormalized(), expected);
    input.stableNormalize();
    VERIFY_IS_APPROX(input, expected);
  }

  if (std::numeric_limits<RealScalar>::has_denorm == std::denorm_present && denorm > RealScalar(0)) {
    const Vector2 input = Vector2::Constant(denorm);
    const Vector2 expected = Vector2::Constant(inv_sqrt_two);
    VERIFY_IS_APPROX(input.stableNormalized(), expected);
    Vector2 actual = input;
    actual.stableNormalize();
    VERIFY_IS_APPROX(actual, expected);
    VERIFY_IS_APPROX(actual.norm(), RealScalar(1));
  }

  {
    Vector2 zero;
    zero << RealScalar(0), -RealScalar(0);
    const Vector2 normalized = zero.stableNormalized();
    VERIFY_IS_EQUAL(normalized(0), RealScalar(0));
    VERIFY_IS_EQUAL(normalized(1), -RealScalar(0));
    VERIFY(!signbit(normalized(0)));
    VERIFY(signbit(normalized(1)));
    zero.stableNormalize();
    VERIFY(!signbit(zero(0)));
    VERIFY(signbit(zero(1)));
  }

  {
    Vector2 input;
    input << infinity, RealScalar(1);
    const Vector2 normalized = input.stableNormalized();
    VERIFY(isPlusInf(normalized(0)));
    VERIFY_IS_EQUAL(normalized(1), RealScalar(1));
    input.stableNormalize();
    VERIFY(isPlusInf(input(0)));
    VERIFY_IS_EQUAL(input(1), RealScalar(1));
  }

  {
    Vector2 input;
    input << nan, RealScalar(1);
    const Vector2 normalized = input.stableNormalized();
    VERIFY((numext::isnan)(normalized(0)));
    VERIFY_IS_EQUAL(normalized(1), RealScalar(1));
    input.stableNormalize();
    VERIFY((numext::isnan)(input(0)));
    VERIFY_IS_EQUAL(input(1), RealScalar(1));
  }

  {
    Vector2 input;
    input << RealScalar(1), nan;
    const Vector2 normalized = input.stableNormalized();
    VERIFY_IS_EQUAL(normalized(0), RealScalar(1));
    VERIFY((numext::isnan)(normalized(1)));
    input.stableNormalize();
    VERIFY_IS_EQUAL(input(0), RealScalar(1));
    VERIFY((numext::isnan)(input(1)));
  }

  {
    VectorX empty_vector(0);
    VERIFY_IS_EQUAL(empty_vector.stableNormalized().size(), 0);
    empty_vector.stableNormalize();
    VERIFY_IS_EQUAL(empty_vector.size(), 0);

    MatrixX empty_rows(0, 3);
    MatrixX empty_cols(3, 0);
    VERIFY_IS_EQUAL(empty_rows.stableNormalized().size(), 0);
    VERIFY_IS_EQUAL(empty_cols.stableNormalized().size(), 0);
    empty_rows.stableNormalize();
    empty_cols.stableNormalize();
    VERIFY_IS_EQUAL(empty_rows.rows(), 0);
    VERIFY_IS_EQUAL(empty_rows.cols(), 3);
    VERIFY_IS_EQUAL(empty_cols.rows(), 3);
    VERIFY_IS_EQUAL(empty_cols.cols(), 0);
    VERIFY_IS_EQUAL(empty_rows.blueNorm(), RealScalar(0));
    VERIFY_IS_EQUAL(empty_cols.blueNorm(), RealScalar(0));
    VERIFY_IS_EQUAL(empty_rows.hypotNorm(), RealScalar(0));
    VERIFY_IS_EQUAL(empty_cols.hypotNorm(), RealScalar(0));
  }
}

template <typename RealScalar>
void stable_normalize_complex_extremes() {
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<Complex, Dynamic, 1> VectorX;
  using std::sqrt;

  const RealScalar highest = (std::numeric_limits<RealScalar>::max)();
  const RealScalar denorm = std::numeric_limits<RealScalar>::denorm_min();
  const RealScalar inv_sqrt_two = RealScalar(1) / sqrt(RealScalar(2));

  {
    VectorX input(1);
    input(0) = Complex(highest, highest);
    const Complex expected(inv_sqrt_two, inv_sqrt_two);
    const VectorX normalized = input.stableNormalized();
    VERIFY_IS_APPROX(normalized(0), expected);
    input.stableNormalize();
    VERIFY_IS_APPROX(input(0), expected);
    VERIFY_IS_APPROX(input.norm(), RealScalar(1));
  }

  if (std::numeric_limits<RealScalar>::has_denorm == std::denorm_present && denorm > RealScalar(0)) {
    VectorX input(1);
    input(0) = Complex(denorm, -denorm);
    const Complex expected(inv_sqrt_two, -inv_sqrt_two);
    VERIFY_IS_APPROX(input.stableNormalized()(0), expected);
    input.stableNormalize();
    VERIFY_IS_APPROX(input(0), expected);
  }
}

template <typename RealScalar>
void stable_norm_extreme_cross_product() {
  typedef Matrix<RealScalar, 2, 1> Vector2;
  using std::sqrt;

  const RealScalar denorm = std::numeric_limits<RealScalar>::denorm_min();
  const RealScalar smallest = (std::numeric_limits<RealScalar>::min)();
  const RealScalar highest = (std::numeric_limits<RealScalar>::max)();
  const RealScalar epsilon = NumTraits<RealScalar>::epsilon();
  const RealScalar values[] = {RealScalar(0),
                               denorm,
                               smallest,
                               sqrt(smallest),
                               epsilon,
                               RealScalar(1),
                               RealScalar(1) / epsilon,
                               sqrt(highest) / RealScalar(2),
                               highest / RealScalar(2),
                               highest};
  const int value_count = int(sizeof(values) / sizeof(values[0]));

  for (int i = 0; i < value_count; ++i) {
    for (int j = 0; j < value_count; ++j) {
      Vector2 input;
      input << values[i], values[j];
      const RealScalar reference = numext::hypot(values[i], values[j]);
      if ((numext::isinf)(reference)) {
        VERIFY(isPlusInf(input.stableNorm()));
        VERIFY(isPlusInf(input.blueNorm()));
        VERIFY(isPlusInf(input.hypotNorm()));
      } else {
        VERIFY_IS_APPROX(input.stableNorm(), reference);
        VERIFY_IS_APPROX(input.blueNorm(), reference);
        VERIFY_IS_APPROX(input.hypotNorm(), reference);
      }
    }
  }
}

template <typename RealScalar>
void stable_norm_mixed_underflow() {
  typedef Matrix<RealScalar, Dynamic, 1> VectorX;
  using std::abs;
  using std::sqrt;

  if (std::numeric_limits<RealScalar>::has_denorm != std::denorm_present) return;

  const Index size = 4096;
  const RealScalar large = sqrt((std::numeric_limits<RealScalar>::min)());
  const RealScalar small = sqrt(std::numeric_limits<RealScalar>::denorm_min()) * RealScalar(0.5);
  VectorX input = VectorX::Constant(size, small);
  input(0) = large;

  const RealScalar reference = numext::hypot(large, sqrt(RealScalar(size - 1)) * small);
  const RealScalar relative_error = abs(input.stableNorm() - reference) / reference;
  // Leave room for the SIMD reduction order while still detecting the roughly
  // 512-epsilon loss caused by squaring this block without scaling.
  const RealScalar tolerance = RealScalar(128) * NumTraits<RealScalar>::epsilon();
  VERIFY(relative_error <= tolerance);
}

template <typename RealScalar>
void stable_norm_denormal_rounding() {
  typedef Matrix<RealScalar, 2, 1> Vector2;
  const RealScalar denorm = std::numeric_limits<RealScalar>::denorm_min();
  if (std::numeric_limits<RealScalar>::has_denorm != std::denorm_present || !(denorm > RealScalar(0))) return;

  // sqrt(2) * denorm_min rounds back to denorm_min.  An approximate check at
  // this scale can accept zero because its own error calculation underflows.
  const Vector2 input = Vector2::Constant(denorm);
  VERIFY_IS_EQUAL(input.stableNorm(), denorm);
  VERIFY_IS_EQUAL(input.blueNorm(), denorm);
  VERIFY_IS_EQUAL(input.hypotNorm(), denorm);
}

template <typename Scalar>
void stable_norm_low_precision() {
  typedef Matrix<Scalar, Dynamic, 1> VectorX;
  using std::abs;
  using std::sqrt;

  const Index size = 65536;
  const Scalar value(0.001f);
  const float value_as_float = static_cast<float>(value);
  const float reference = sqrt(static_cast<float>(size)) * abs(value_as_float);
  const float relative_tolerance = 8.0f * static_cast<float>(NumTraits<Scalar>::epsilon());
  VectorX input = VectorX::Constant(size, value);

  const float stable_norm = static_cast<float>(input.stableNorm());
  const float blue_norm = static_cast<float>(input.blueNorm());
  const float hypot_norm = static_cast<float>(input.hypotNorm());
  VERIFY(abs(stable_norm - reference) <= relative_tolerance * reference);
  VERIFY(abs(blue_norm - reference) <= relative_tolerance * reference);
  VERIFY(abs(hypot_norm - reference) <= relative_tolerance * reference);

  const VectorX normalized = input.stableNormalized();
  const float promoted_norm = normalized.template cast<float>().norm();
  VERIFY(abs(promoted_norm - 1.0f) <= relative_tolerance);
  input.stableNormalize();
  const float promoted_in_place_norm = input.template cast<float>().norm();
  VERIFY(abs(promoted_in_place_norm - 1.0f) <= relative_tolerance);
}

template <typename RealScalar>
void stable_norm_complex_low_precision() {
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<Complex, 1, 1> Vector1;
  using std::abs;

  Vector1 input;
  input(0) = Complex(RealScalar(3), RealScalar(4));
  const float tolerance = 8.0f * static_cast<float>(NumTraits<RealScalar>::epsilon());
  const Complex normalized = input.stableNormalized()(0);
  VERIFY(abs(static_cast<float>(normalized.real()) - 0.6f) <= tolerance);
  VERIFY(abs(static_cast<float>(normalized.imag()) - 0.8f) <= tolerance);
  input.stableNormalize();
  VERIFY(abs(static_cast<float>(input(0).real()) - 0.6f) <= tolerance);
  VERIFY(abs(static_cast<float>(input(0).imag()) - 0.8f) <= tolerance);
}

void stable_normalize_promoted_factor() {
  typedef Matrix<half, Dynamic, 1> VectorX;
  using std::abs;

  // The combined normalization factor is just below half's first subnormal,
  // while every final coefficient is representable.  Applying the factor in
  // float before converting each result must therefore not produce zeros.
  const Index size = 4194305;
  const half value(16384.0f);
  const float tolerance = 8.0f * static_cast<float>(NumTraits<half>::epsilon());
  VectorX input = VectorX::Constant(size, value);

  const VectorX normalized = input.stableNormalized();
  VERIFY(static_cast<float>(normalized(0)) > 0.0f);
  VERIFY(abs(normalized.template cast<float>().norm() - 1.0f) <= tolerance);

  input.stableNormalize();
  VERIFY(static_cast<float>(input(0)) > 0.0f);
  VERIFY(abs(input.template cast<float>().norm() - 1.0f) <= tolerance);
}

void stable_normalize_no_malloc() {
  VectorXd input = VectorXd::Constant(2, (std::numeric_limits<double>::max)());
  const Vector2d expected = Vector2d::Constant(1.0 / std::sqrt(2.0));

  internal::set_is_malloc_allowed(false);
  input.stableNormalize();
  internal::set_is_malloc_allowed(true);
  VERIFY_IS_APPROX(input, expected);
}

void stable_norm_expression_and_stride() {
  const Index size = 31;
  Index evaluation_count = 0;
  const auto expression = VectorXd::NullaryExpr(size, StableNormCountingOp(&evaluation_count));
  const double expected = std::sqrt(double(size) * double(size + 1) * double(2 * size + 1) / 6.0);
  VERIFY_IS_APPROX(expression.stableNorm(), expected);
  VERIFY_IS_EQUAL(evaluation_count, size);
  evaluation_count = 0;
  const VectorXd expression_normalized = expression.stableNormalized();
  VERIFY_IS_APPROX(expression_normalized.norm(), 1.0);
  VERIFY_IS_EQUAL(evaluation_count, size);

  VectorXd storage = VectorXd::Zero(2 * size);
  for (Index i = 0; i < size; ++i) storage(2 * i) = double(i + 1);
  typedef InnerStride<Dynamic> VectorStride;
  Map<VectorXd, Unaligned, VectorStride> strided(storage.data(), size, VectorStride(2));
  VERIFY_IS_APPROX(strided.stableNorm(), expected);
  strided.stableNormalize();
  VERIFY_IS_APPROX(strided.norm(), 1.0);
  for (Index i = 0; i < size; ++i) VERIFY_IS_EQUAL(storage(2 * i + 1), 0.0);

  RowVectorXd packed_storage = RowVectorXd::LinSpaced(size, 1.0, double(size));
  Map<RowVectorXd, Unaligned, VectorStride> runtime_packed(packed_storage.data(), size, VectorStride(1));
  VERIFY_IS_APPROX(runtime_packed.stableNorm(), expected);
  const RowVectorXd normalized = runtime_packed.stableNormalized();
  VERIFY_IS_EQUAL(normalized.rows(), 1);
  VERIFY_IS_EQUAL(normalized.cols(), size);
  VERIFY_IS_APPROX(normalized.norm(), 1.0);
  runtime_packed.stableNormalize();
  VERIFY_IS_APPROX(runtime_packed.norm(), 1.0);

  // Flattening must preserve total size for fixed rows and multi-column matrices.
  typedef Matrix<double, 1, 4> FixedRowVector;
  FixedRowVector fixed_row_storage;
  fixed_row_storage << 1.0, 2.0, 3.0, 4.0;
  Map<FixedRowVector, Unaligned, VectorStride> fixed_row(fixed_row_storage.data(), VectorStride(1));
  const double fixed_row_norm = std::sqrt(30.0);
  const FixedRowVector expected_fixed_row = fixed_row_storage / fixed_row_norm;
  VERIFY_IS_APPROX(fixed_row.stableNorm(), fixed_row_norm);
  const FixedRowVector fixed_row_normalized = fixed_row.stableNormalized();
  VERIFY_IS_APPROX(fixed_row_normalized, expected_fixed_row);
  VERIFY_IS_APPROX(fixed_row_normalized.norm(), 1.0);
  fixed_row.stableNormalize();
  VERIFY_IS_APPROX(fixed_row, expected_fixed_row);
  VERIFY_IS_APPROX(fixed_row.norm(), 1.0);

  typedef Stride<Dynamic, Dynamic> MatrixStride;
  typedef Matrix<double, 2, 3> FixedMatrix;
  FixedMatrix fixed_matrix_storage;
  fixed_matrix_storage << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
  Map<FixedMatrix, Unaligned, MatrixStride> fixed_matrix(fixed_matrix_storage.data(),
                                                         MatrixStride(FixedMatrix::RowsAtCompileTime, 1));
  const double fixed_matrix_norm = std::sqrt(91.0);
  const FixedMatrix expected_fixed_matrix = fixed_matrix_storage / fixed_matrix_norm;
  VERIFY_IS_APPROX(fixed_matrix.stableNorm(), fixed_matrix_norm);
  const FixedMatrix fixed_matrix_normalized = fixed_matrix.stableNormalized();
  VERIFY_IS_APPROX(fixed_matrix_normalized, expected_fixed_matrix);
  VERIFY_IS_APPROX(fixed_matrix_normalized.norm(), 1.0);
  fixed_matrix.stableNormalize();
  VERIFY_IS_APPROX(fixed_matrix, expected_fixed_matrix);
  VERIFY_IS_APPROX(fixed_matrix.norm(), 1.0);

  const double padding_value = 42.0;
  VectorXd matrix_storage = VectorXd::Constant(20, padding_value);
  Map<Matrix<double, Dynamic, Dynamic>, Unaligned, MatrixStride> gapped_matrix(matrix_storage.data(), 3, 4,
                                                                               MatrixStride(5, 1));
  gapped_matrix.setRandom();
  VERIFY_IS_APPROX(gapped_matrix.stableNorm(), gapped_matrix.norm());
  MatrixXd packed_matrix = gapped_matrix;
  VERIFY_IS_APPROX(gapped_matrix.stableNormalized(), packed_matrix.stableNormalized());
  gapped_matrix.stableNormalize();
  packed_matrix.stableNormalize();
  VERIFY_IS_APPROX(gapped_matrix, packed_matrix);
  for (Index outer = 0; outer < gapped_matrix.outerSize(); ++outer) {
    for (Index inner = gapped_matrix.innerSize(); inner < gapped_matrix.outerStride(); ++inner) {
      VERIFY_IS_EQUAL(matrix_storage(outer * gapped_matrix.outerStride() + inner), padding_value);
    }
  }
}

template <typename Scalar>
void test_hypot() {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  // Get a random factor bounded away from zero: |factor| >= 0.1.
  Scalar factor = internal::random<Scalar>(Scalar(RealScalar(0.1)), Scalar(RealScalar(1)));
  Scalar big = factor * ((std::numeric_limits<RealScalar>::max)() * RealScalar(1e-4));

  factor = internal::random<Scalar>(Scalar(RealScalar(0.1)), Scalar(RealScalar(1)));
  Scalar small = factor * ((std::numeric_limits<RealScalar>::min)() * RealScalar(1e4));

  Scalar one(1), zero(0), sqrt2(std::sqrt(2)), nan(std::numeric_limits<RealScalar>::quiet_NaN());

  Scalar a = internal::random<Scalar>(-1, 1);
  Scalar b = internal::random<Scalar>(-1, 1);
  VERIFY_IS_APPROX(numext::hypot(a, b), std::sqrt(numext::abs2(a) + numext::abs2(b)));
  VERIFY_IS_EQUAL(numext::hypot(zero, zero), zero);
  VERIFY_IS_APPROX(numext::hypot(one, one), sqrt2);
  VERIFY_IS_APPROX(numext::hypot(big, big), sqrt2 * numext::abs(big));
  VERIFY_IS_APPROX(numext::hypot(small, small), sqrt2 * numext::abs(small));
  VERIFY_IS_APPROX(numext::hypot(small, big), numext::abs(big));
  VERIFY((numext::isnan)(numext::hypot(nan, a)));
  VERIFY((numext::isnan)(numext::hypot(a, nan)));
}

template <typename Scalar>
void stable_norm_complex_infinity() {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> VecType;

  const RealScalar inf = std::numeric_limits<RealScalar>::infinity();
  const Scalar finite(RealScalar(3), RealScalar(-4));
  const Scalar both_inf(inf, inf);
  const Scalar real_inf(inf, RealScalar(1));
  const Scalar imag_inf(RealScalar(1), -inf);

  VERIFY(isPlusInf(numext::abs(both_inf)));
  VERIFY(isPlusInf(numext::abs(real_inf)));
  VERIFY(isPlusInf(numext::abs(imag_inf)));
  VERIFY(isPlusInf(numext::hypot(both_inf, finite)));

  VecType v(4);
  v << both_inf, finite, real_inf, imag_inf;

  VERIFY(isPlusInf(v.cwiseAbs().maxCoeff()));
  VERIFY(isPlusInf(v.stableNorm()));
  VERIFY(isPlusInf(v.blueNorm()));
  VERIFY(isPlusInf(v.hypotNorm()));
}

// Test stableNorm at the 4096-element block boundary.
// stable_norm_impl_inner_step processes vectors in blocks of 4096.
// Sizes near this boundary exercise the transition between full blocks
// and the remainder tail, including scale propagation across blocks.
template <typename Scalar>
void stable_norm_block_boundary() {
  using std::abs;
  using std::sqrt;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> VecType;

  // Test sizes around the 4096 block boundary.
  const Index sizes[] = {4095, 4096, 4097, 8191, 8192, 8193, 12288};
  for (int si = 0; si < 7; ++si) {
    Index n = sizes[si];
    VecType v = VecType::Random(n);
    VERIFY_IS_APPROX(v.stableNorm(), v.norm());
    VERIFY_IS_APPROX(v.blueNorm(), v.norm());
  }

  // Test scale transitions across blocks: first block has tiny values,
  // second block has huge values. This exercises the scale/invScale
  // update logic when maxCoeff > scale in stable_norm_kernel.
  {
    RealScalar tiny = (std::numeric_limits<RealScalar>::min)() * RealScalar(1e4);
    RealScalar huge_val = (std::numeric_limits<RealScalar>::max)() * RealScalar(1e-4);
    Index n = 8192;
    VecType v(n);
    // First 4096 elements: tiny. Second 4096 elements: huge.
    v.head(4096).setConstant(Scalar(tiny));
    v.tail(4096).setConstant(Scalar(huge_val));
    // The huge part dominates, so the expected norm is sqrt(4096)*huge_val.
    RealScalar expected = sqrt(RealScalar(4096)) * abs(huge_val);
    VERIFY_IS_APPROX(v.stableNorm(), expected);
    VERIFY_IS_APPROX(v.blueNorm(), expected);
  }

  // Reverse: first block huge, second block tiny.
  {
    RealScalar tiny = (std::numeric_limits<RealScalar>::min)() * RealScalar(1e4);
    RealScalar huge_val = (std::numeric_limits<RealScalar>::max)() * RealScalar(1e-4);
    Index n = 8192;
    VecType v(n);
    v.head(4096).setConstant(Scalar(huge_val));
    v.tail(4096).setConstant(Scalar(tiny));
    RealScalar expected = sqrt(RealScalar(4096)) * abs(huge_val);
    VERIFY_IS_APPROX(v.stableNorm(), expected);
    VERIFY_IS_APPROX(v.blueNorm(), expected);
  }

  // Matrix version: columns with different magnitudes.
  // Scale must propagate correctly across columns.
  {
    RealScalar tiny = (std::numeric_limits<RealScalar>::min)() * RealScalar(1e4);
    RealScalar huge_val = (std::numeric_limits<RealScalar>::max)() * RealScalar(1e-4);
    typedef Matrix<Scalar, Dynamic, Dynamic> MatType;
    MatType m(100, 2);
    m.col(0).setConstant(Scalar(tiny));
    m.col(1).setConstant(Scalar(huge_val));
    RealScalar expected = sqrt(RealScalar(100)) * abs(huge_val);
    VERIFY_IS_APPROX(m.stableNorm(), expected);
    VERIFY_IS_APPROX(m.blueNorm(), expected);
  }
}

EIGEN_DECLARE_TEST(stable_norm) {
  CALL_SUBTEST_1(test_empty());

  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_3(test_hypot<double>());
    CALL_SUBTEST_4(test_hypot<float>());
    CALL_SUBTEST_5(test_hypot<std::complex<double> >());
    CALL_SUBTEST_6(test_hypot<std::complex<float> >());

    CALL_SUBTEST_1(stable_norm(Matrix<float, 1, 1>()));
    CALL_SUBTEST_2(stable_norm(Vector4d()));
    CALL_SUBTEST_3(stable_norm(VectorXd(internal::random<int>(10, 2000))));
    CALL_SUBTEST_3(stable_norm(MatrixXd(internal::random<int>(10, 200), internal::random<int>(10, 200))));
    CALL_SUBTEST_4(stable_norm(VectorXf(internal::random<int>(10, 2000))));
    CALL_SUBTEST_5(stable_norm(VectorXcd(internal::random<int>(10, 2000))));
    CALL_SUBTEST_6(stable_norm(VectorXcf(internal::random<int>(10, 2000))));
  }

  // Block boundary and scale transition tests (deterministic, outside g_repeat).
  CALL_SUBTEST_7(stable_norm_block_boundary<float>());
  CALL_SUBTEST_7(stable_norm_block_boundary<double>());
  CALL_SUBTEST_8(stable_norm_complex_infinity<std::complex<float> >());
  CALL_SUBTEST_8(stable_norm_complex_infinity<std::complex<double> >());
  CALL_SUBTEST_9(stable_normalize_extremes<float>());
  CALL_SUBTEST_9(stable_normalize_extremes<double>());
  CALL_SUBTEST_10(stable_normalize_complex_extremes<float>());
  CALL_SUBTEST_10(stable_normalize_complex_extremes<double>());
  CALL_SUBTEST_11(stable_norm_extreme_cross_product<float>());
  CALL_SUBTEST_11(stable_norm_extreme_cross_product<double>());
  CALL_SUBTEST_11(stable_norm_mixed_underflow<float>());
  CALL_SUBTEST_11(stable_norm_mixed_underflow<double>());
  CALL_SUBTEST_11(stable_norm_denormal_rounding<float>());
  CALL_SUBTEST_11(stable_norm_denormal_rounding<double>());
  CALL_SUBTEST_12(stable_norm_low_precision<half>());
  CALL_SUBTEST_12(stable_norm_low_precision<bfloat16>());
  CALL_SUBTEST_12(stable_norm_complex_low_precision<half>());
  CALL_SUBTEST_12(stable_norm_complex_low_precision<bfloat16>());
  CALL_SUBTEST_12(stable_normalize_promoted_factor());
  CALL_SUBTEST_13(stable_norm_expression_and_stride());
  CALL_SUBTEST_13(stable_normalize_no_malloc());
}
