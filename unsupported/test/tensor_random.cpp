// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include <Eigen/Tensor>

template <typename Scalar>
static void test_default() {
  Tensor<Scalar, 1> vec(6);

  // Fixme: we should check that the generated numbers follow a uniform
  // distribution instead.
  // For low-precision types (half, bfloat16), the RNG has limited distinct
  // values (e.g. 128 for bfloat16), so adjacent collisions are possible.
  // Retry a few times to avoid spurious failures.
  bool all_distinct = false;
  for (int attempt = 0; attempt < 10 && !all_distinct; ++attempt) {
    vec.setRandom();
    all_distinct = true;
    for (int i = 1; i < 6; ++i) {
      if (vec(i) == vec(i - 1)) {
        all_distinct = false;
        break;
      }
    }
  }
  VERIFY(all_distinct);
}

template <typename Scalar>
static void test_normal() {
  Tensor<Scalar, 1> vec(6);

  // Fixme: we should check that the generated numbers follow a gaussian
  // distribution instead.
  bool all_distinct = false;
  for (int attempt = 0; attempt < 10 && !all_distinct; ++attempt) {
    vec.template setRandom<Eigen::internal::NormalRandomGenerator<Scalar>>();
    all_distinct = true;
    for (int i = 1; i < 6; ++i) {
      if (vec(i) == vec(i - 1)) {
        all_distinct = false;
        break;
      }
    }
  }
  VERIFY(all_distinct);
}

template <typename Scalar>
static void test_normal_all_finite(Eigen::Index size) {
  // Regression test: the 16-bit uniform draw is exactly 0 with probability
  // 2^-10 (half) / 2^-7 (bfloat16). Running the ratio-of-uniforms rejection
  // in 16-bit arithmetic let log(0) = -inf poison the acceptance test and
  // returned v / 0 = +/-inf (or 0/0 = NaN) at measurable rates.
  Tensor<Scalar, 1> vec(size);
  vec.template setRandom<Eigen::internal::NormalRandomGenerator<Scalar>>();
  Eigen::Index num_not_finite = 0;
  for (Eigen::Index i = 0; i < size; ++i) {
    if (!(numext::isfinite)(vec(i))) ++num_not_finite;
  }
  VERIFY_IS_EQUAL(num_not_finite, Eigen::Index(0));
}

template <typename Scalar>
static void test_uniform_range(Eigen::Index size) {
  // All uniform draws must lie in [0, 1).
  Tensor<Scalar, 1> vec(size);
  vec.setRandom();
  Eigen::Index num_out_of_range = 0;
  for (Eigen::Index i = 0; i < size; ++i) {
    if (!(vec(i) >= Scalar(0.0f) && vec(i) < Scalar(1.0f))) ++num_out_of_range;
  }
  VERIFY_IS_EQUAL(num_out_of_range, Eigen::Index(0));
}

struct MyGenerator {
  MyGenerator() {}
  MyGenerator(const MyGenerator&) {}

  // Return a random value to be used.  "element_location" is the
  // location of the entry to set in the tensor, it can typically
  // be ignored.
  int operator()(Eigen::DenseIndex element_location, Eigen::DenseIndex /*unused*/ = 0) const {
    return static_cast<int>(3 * element_location);
  }

  // Same as above but generates several numbers at a time.
  internal::packet_traits<int>::type packetOp(Eigen::DenseIndex packet_location,
                                              Eigen::DenseIndex /*unused*/ = 0) const {
    const int packetSize = internal::packet_traits<int>::size;
    EIGEN_ALIGN_MAX int values[packetSize];
    for (int i = 0; i < packetSize; ++i) {
      values[i] = static_cast<int>(3 * (packet_location + i));
    }
    return internal::pload<typename internal::packet_traits<int>::type>(values);
  }
};

static void test_custom() {
  Tensor<int, 1> vec(6);
  vec.setRandom<MyGenerator>();

  for (int i = 0; i < 6; ++i) {
    VERIFY_IS_EQUAL(vec(i), 3 * i);
  }
}

EIGEN_DECLARE_TEST(tensor_random) {
  CALL_SUBTEST((test_default<float>()));
  CALL_SUBTEST((test_normal<float>()));
  CALL_SUBTEST((test_default<double>()));
  CALL_SUBTEST((test_normal<double>()));
  CALL_SUBTEST((test_default<Eigen::half>()));
  CALL_SUBTEST((test_normal<Eigen::half>()));
  CALL_SUBTEST((test_default<Eigen::bfloat16>()));
  CALL_SUBTEST((test_normal<Eigen::bfloat16>()));
  CALL_SUBTEST((test_normal_all_finite<Eigen::half>(Eigen::Index(1) << 21)));
  CALL_SUBTEST((test_normal_all_finite<Eigen::bfloat16>(Eigen::Index(1) << 18)));
  CALL_SUBTEST((test_uniform_range<Eigen::half>(Eigen::Index(1) << 16)));
  CALL_SUBTEST((test_uniform_range<Eigen::bfloat16>(Eigen::Index(1) << 16)));
  CALL_SUBTEST(test_custom());
}
