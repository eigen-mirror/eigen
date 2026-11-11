// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include <Eigen/src/Core/util/MaxSizeVector.h>
#include "lifetime_scalar.h"

using Foo = lifetime_scalar::LifetimeScalar<128>;
using lifetime_scalar::Failure;

EIGEN_DECLARE_TEST(maxsizevector) {
  using VectorX = MaxSizeVector<Foo>;
  // A copy would share, and then free twice, the owned buffer.
  VERIFY(!std::is_copy_constructible<VectorX>::value);
  VERIFY(!std::is_copy_assignable<VectorX>::value);
  for (int r = 0; r < g_repeat; r++) {
    Index rows = internal::random<Index>(3, 30);
    Foo::max_live = internal::random<Index>(0, rows - 2) + 1;
#ifdef EIGEN_EXCEPTIONS
    bool exception_raised = false;
    try {
      VectorX vect(rows);
      for (int i = 0; i < rows; ++i) vect.push_back(Foo());
      VERIFY(false);  // not reached if exceptions are enabled
    } catch (const Failure&) {
      exception_raised = true;
    }
    VERIFY(exception_raised);
#endif
    VERIFY_IS_EQUAL(Index(0), Foo::live);

    {
      Foo::max_live = rows + 2;
      VectorX vect2(rows, Foo());
      VERIFY_IS_EQUAL(Foo::live, rows);
    }
    VERIFY_IS_EQUAL(Index(0), Foo::live);
#ifdef EIGEN_EXCEPTIONS
    // A copy throwing partway through the fill destroys exactly the copies already made.
    exception_raised = false;
    Foo::max_live_on_copy = rows / 2 + 1;
    try {
      VectorX vect3(rows, Foo());
    } catch (const Failure&) {
      exception_raised = true;
    }
    Foo::max_live_on_copy = Foo::kUnlimited;
    VERIFY(exception_raised);
    VERIFY_IS_EQUAL(Index(0), Foo::live);

    // A throwing push_back leaves the vector unchanged.
    {
      VectorX vect4(rows);
      vect4.push_back(Foo());
      exception_raised = false;
      Foo::max_live_on_copy = 1;
      try {
        vect4.push_back(Foo());
      } catch (const Failure&) {
        exception_raised = true;
      }
      Foo::max_live_on_copy = Foo::kUnlimited;
      VERIFY(exception_raised);
      VERIFY_IS_EQUAL(vect4.size(), size_t(1));
    }
    VERIFY_IS_EQUAL(Index(0), Foo::live);
#endif
  }
}
