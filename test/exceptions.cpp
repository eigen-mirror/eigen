// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Various sanity tests with exceptions and non trivially copyable scalar type.
//  - no memory leak when a custom scalar type throw an exceptions
//  - todo: complete the list of tests!

#define EIGEN_STACK_ALLOCATION_LIMIT 100000000

#include "main.h"
#include "AnnoyingScalar.h"

#define CHECK_MEMLEAK(OP)                                                                                             \
  {                                                                                                                   \
    AnnoyingScalar::countdown = 100;                                                                                  \
    int before = AnnoyingScalar::instances;                                                                           \
    bool exception_thrown = false;                                                                                    \
    EIGEN_TRY { OP; }                                                                                                 \
    EIGEN_CATCH(my_exception) {                                                                                       \
      exception_thrown = true;                                                                                        \
      VERIFY(AnnoyingScalar::instances == before && "memory leak detected in " && EIGEN_MAKESTRING(OP));              \
    }                                                                                                                 \
    VERIFY((AnnoyingScalar::dont_throw) || (exception_thrown && " no exception thrown in " && EIGEN_MAKESTRING(OP))); \
  }

#ifdef EIGEN_EXCEPTIONS
struct assign_exception {};

// Scalar whose copy assignment throws once `budget` assignments have been made. `live` counts the constructed objects.
struct ThrowingAssignScalar {
  static int live;
  static int budget;
  double v;
  ThrowingAssignScalar() : v(0) { ++live; }
  ThrowingAssignScalar(const ThrowingAssignScalar& other) : v(other.v) { ++live; }
  ~ThrowingAssignScalar() { --live; }
  ThrowingAssignScalar& operator=(const ThrowingAssignScalar& other) {
    if (budget >= 0) {
      const bool exhausted = budget == 0;
      --budget;
      if (exhausted) throw assign_exception();
    }
    v = other.v;
    return *this;
  }
};
int ThrowingAssignScalar::live = 0;
int ThrowingAssignScalar::budget = -1;

namespace Eigen {
template <>
struct NumTraits<ThrowingAssignScalar> : NumTraits<double> {
  static constexpr int RequireInitialization = 1;
};
}  // namespace Eigen

// A copy-assignment that throws partway through must destroy every element and free the buffer: the copy constructor
// of the dynamic DenseStorage specializations, and the (exception-safe) copy assignment, for each of their layouts.
template <typename MatrixType>
void check_throwing_assign(Index rows, Index cols) {
  VERIFY_IS_EQUAL(ThrowingAssignScalar::live, 0);
  {
    MatrixType src(rows, cols);
    for (int budget = 0; budget < rows * cols; budget += 3) {
      bool thrown = false;
      ThrowingAssignScalar::budget = budget;
      // NOLINTNEXTLINE(performance-unnecessary-copy-initialization): the copy is the operation under test.
      EIGEN_TRY { MatrixType copy(src); }
      EIGEN_CATCH(assign_exception) { thrown = true; }
      ThrowingAssignScalar::budget = -1;
      VERIFY(thrown);
      VERIFY_IS_EQUAL(ThrowingAssignScalar::live, static_cast<int>(src.size()));

      // A different size, so that the assignment reallocates.
      MatrixType dst(MatrixType::RowsAtCompileTime == Dynamic ? rows + 1 : rows,
                     MatrixType::RowsAtCompileTime != Dynamic ? cols + 1 : cols);
      thrown = false;
      ThrowingAssignScalar::budget = budget;
      EIGEN_TRY { dst = src; }
      EIGEN_CATCH(assign_exception) { thrown = true; }
      ThrowingAssignScalar::budget = -1;
      VERIFY(thrown);
      VERIFY_IS_EQUAL(ThrowingAssignScalar::live, static_cast<int>(src.size() + dst.size()));
    }
  }
  VERIFY_IS_EQUAL(ThrowingAssignScalar::live, 0);
}
#endif

EIGEN_DECLARE_TEST(exceptions) {
  using VectorType = Eigen::Matrix<AnnoyingScalar, Dynamic, 1>;
  using MatrixType = Eigen::Matrix<AnnoyingScalar, Dynamic, Dynamic>;

  {
#if defined(EIGEN_EXCEPTIONS) && !defined(EIGEN_TEST_ANNOYING_SCALAR_DONT_THROW)
    AnnoyingScalar::dont_throw = false;
#else
    AnnoyingScalar::dont_throw = true;
#endif
    int n = 50;
    VectorType v0(n), v1(n);
    MatrixType m0(n, n), m1(n, n), m2(n, n);
    v0.setOnes();
    v1.setOnes();
    m0.setOnes();
    m1.setOnes();
    m2.setOnes();
    CHECK_MEMLEAK(v0 = m0 * m1 * v1);
    CHECK_MEMLEAK(m2 = m0 * m1 * m2);
    CHECK_MEMLEAK((v0 + v1).dot(v0 + v1));
  }
#ifdef EIGEN_EXCEPTIONS
  check_throwing_assign<Matrix<ThrowingAssignScalar, Dynamic, Dynamic>>(4, 4);
  check_throwing_assign<Matrix<ThrowingAssignScalar, Dynamic, 3>>(5, 3);
  check_throwing_assign<Matrix<ThrowingAssignScalar, 3, Dynamic>>(3, 5);
  check_throwing_assign<Matrix<ThrowingAssignScalar, Dynamic, 1>>(7, 1);
  check_throwing_assign<Matrix<ThrowingAssignScalar, Dynamic, Dynamic, RowMajor>>(3, 6);
#endif
  VERIFY(AnnoyingScalar::instances == 0 && "global memory leak detected in " && EIGEN_MAKESTRING(OP));
}
