// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include <Eigen/src/Core/util/MaxSizeVector.h>
#include "lifetime_scalar.h"

using Foo = lifetime_scalar::LifetimeScalar<128>;
using lifetime_scalar::Failure;

#ifdef EIGEN_EXCEPTIONS
template <typename T>
void check_throws_bad_alloc(size_t n) {
  bool threw = false;
  try {
    MaxSizeVector<T> vect(n);
  } catch (const std::bad_alloc&) {
    threw = true;
  }
  VERIFY(threw);
  threw = false;
  try {
    MaxSizeVector<T> vect(n, T());
  } catch (const std::bad_alloc&) {
    threw = true;
  }
  VERIFY(threw);
}

// Sizes that cannot be reserved must throw std::bad_alloc rather than wrap around or return a null buffer.
void check_oversized_allocation() {
  const size_t max_size = static_cast<size_t>((std::numeric_limits<std::ptrdiff_t>::max)());
  // n * sizeof(int) wraps around to 0 for n = 2^(bits - 2).
  check_throws_bad_alloc<int>(static_cast<size_t>(1) << (std::numeric_limits<size_t>::digits - 2));
  check_throws_bad_alloc<double>(max_size / sizeof(double) + 1);
  // PTRDIFF_MAX bytes leave no room for the alignment padding, so handmade_aligned_malloc rejects them on every target.
  check_throws_bad_alloc<char>(max_size);
}
#endif

EIGEN_DECLARE_TEST(maxsizevector) {
  using VectorX = MaxSizeVector<Foo>;
#ifdef EIGEN_EXCEPTIONS
  check_oversized_allocation();
#endif
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
