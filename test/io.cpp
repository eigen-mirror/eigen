// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2019 Joel Holdsworth <joel.holdsworth@vcatechnology.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include <sstream>

#include "main.h"

namespace stream_precision_test {
struct Rational {
  int numerator;
  int denominator;
  Rational(int n = 0, int d = 1) : numerator(n), denominator(d) {}
};

struct BinaryScalar {
  double value;
};
double log2(const BinaryScalar& value) { return std::log2(value.value); }

struct IncompleteScalar {};
struct UnsupportedLog {};
UnsupportedLog log2(const IncompleteScalar&) { return {}; }

struct ExplicitPrecision {};

struct MetadataReal {};
struct MetadataScalar {
  double value;
  MetadataScalar(double x = 0) : value(x) {}
};
std::ostream& operator<<(std::ostream& stream, const MetadataScalar& value) { return stream << value.value; }

std::ostream& operator<<(std::ostream& stream, const Rational& value) {
  return stream << (std::to_string(value.numerator) + '/' + std::to_string(value.denominator));
}
}  // namespace stream_precision_test

namespace std {
template <>
class numeric_limits<stream_precision_test::MetadataReal> : public numeric_limits<double> {};
}  // namespace std

namespace Eigen {
template <>
struct NumTraits<stream_precision_test::Rational> : GenericNumTraits<stream_precision_test::Rational> {};
template <>
struct NumTraits<stream_precision_test::BinaryScalar> : GenericNumTraits<stream_precision_test::BinaryScalar> {
  static stream_precision_test::BinaryScalar epsilon() { return {1.0 / 1024}; }
};
template <>
struct NumTraits<stream_precision_test::ExplicitPrecision>
    : GenericNumTraits<stream_precision_test::ExplicitPrecision> {
  static constexpr int digits() { return 11; }
  static constexpr int digits10() { return 3; }
  static constexpr int max_digits10() { return 5; }
};
template <>
struct NumTraits<stream_precision_test::MetadataScalar> : GenericNumTraits<stream_precision_test::MetadataScalar> {
  using Real = stream_precision_test::MetadataReal;
};
}  // namespace Eigen

void check_custom_scalar_precision() {
  using MetadataScalar = stream_precision_test::MetadataScalar;
  VERIFY_IS_EQUAL(NumTraits<MetadataScalar>::digits10(), 15);
  VERIFY_IS_EQUAL(NumTraits<MetadataScalar>::max_digits10(), 17);
  Matrix<MetadataScalar, 1, 1> decimal;
  decimal(0) = MetadataScalar(1.2345678901234567);
  std::ostringstream full_precision;
  full_precision << decimal.format(IOFormat(FullPrecision));
  VERIFY(full_precision.str() == "1.2345678901234567");
  VERIFY_IS_EQUAL(NumTraits<stream_precision_test::BinaryScalar>::digits(), 11);
  VERIFY_IS_EQUAL(NumTraits<stream_precision_test::BinaryScalar>::digits10(), 3);
  VERIFY_IS_EQUAL(NumTraits<stream_precision_test::BinaryScalar>::max_digits10(), 5);
  VERIFY_IS_EQUAL(NumTraits<stream_precision_test::IncompleteScalar>::digits10(), 0);
  VERIFY_IS_EQUAL(NumTraits<stream_precision_test::IncompleteScalar>::max_digits10(), 0);
  VERIFY_IS_EQUAL(NumTraits<stream_precision_test::ExplicitPrecision>::digits(), 11);
  VERIFY_IS_EQUAL(NumTraits<stream_precision_test::ExplicitPrecision>::digits10(), 3);
  VERIFY_IS_EQUAL(NumTraits<stream_precision_test::ExplicitPrecision>::max_digits10(), 5);
  VERIFY_IS_EQUAL(NumTraits<double>::max_digits10(), std::numeric_limits<double>::max_digits10);
  using Scalar = stream_precision_test::Rational;
  VERIFY_IS_EQUAL(NumTraits<Scalar>::digits10(), 0);
  VERIFY_IS_EQUAL(NumTraits<Scalar>::max_digits10(), 0);
  Matrix<Scalar, 2, 1> values;
  values << Scalar(1, 2), Scalar(3, 4);
  const int precisions[] = {StreamPrecision, FullPrecision, 4};
  for (int precision : precisions) {
    std::ostringstream stream;
    stream.precision(7);
    stream << values.format(IOFormat(precision));
    VERIFY(stream.str() == "1/2\n3/4");
    VERIFY_IS_EQUAL(stream.precision(), 7);
  }
  std::ostringstream stream;
  stream << values;
  VERIFY(stream.str() == "1/2\n3/4");
}

template <typename Scalar>
struct check_ostream_impl {
  static void run() {
    const Array<Scalar, 1, 1> array(123);
    std::ostringstream ss;
    ss << array;
    VERIFY(ss.str() == "123");

    check_ostream_impl<std::complex<Scalar> >::run();
  };
};

template <>
struct check_ostream_impl<bool> {
  static void run() {
    const Array<bool, 1, 2> array(1, 0);
    std::ostringstream ss;
    ss << array;
    VERIFY(ss.str() == "1  0");
  };
};

template <typename Scalar>
struct check_ostream_impl<std::complex<Scalar> > {
  static void run() {
    const Array<std::complex<Scalar>, 1, 1> array(std::complex<Scalar>(12, 34));
    std::ostringstream ss;
    ss << array;
    VERIFY(ss.str() == "(12,34)");
  };
};

template <typename Scalar>
static void check_ostream() {
  check_ostream_impl<Scalar>::run();
}

EIGEN_DECLARE_TEST(io) {
  CALL_SUBTEST(check_custom_scalar_precision());
  CALL_SUBTEST(check_ostream<bool>());
  CALL_SUBTEST(check_ostream<float>());
  CALL_SUBTEST(check_ostream<double>());
  CALL_SUBTEST(check_ostream<Eigen::numext::int8_t>());
  CALL_SUBTEST(check_ostream<Eigen::numext::uint8_t>());
  CALL_SUBTEST(check_ostream<Eigen::numext::int16_t>());
  CALL_SUBTEST(check_ostream<Eigen::numext::uint16_t>());
  CALL_SUBTEST(check_ostream<Eigen::numext::int32_t>());
  CALL_SUBTEST(check_ostream<Eigen::numext::uint32_t>());
  CALL_SUBTEST(check_ostream<Eigen::numext::int64_t>());
  CALL_SUBTEST(check_ostream<Eigen::numext::uint64_t>());
}
