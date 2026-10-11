// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#ifdef EIGEN_EXCEPTIONS
#include <exception>  // std::exception
#endif

#include <Eigen/src/Core/util/MaxSizeVector.h>

struct Foo {
  static Index object_count;
  static Index object_limit;
  static Index copy_limit;
  EIGEN_ALIGN_TO_BOUNDARY(128) int dummy;

  Foo(int x = 0) : dummy(x) {
#ifdef EIGEN_EXCEPTIONS
    // TODO: Is this the correct way to handle this?
    if (Foo::object_count > Foo::object_limit) {
      std::cout << "\nThrow!\n";
      throw Foo::Fail();
    }
#endif
    std::cout << '+';
    ++Foo::object_count;
    eigen_assert((std::uintptr_t(this) & (127)) == 0);
  }
  Foo(const Foo&) {
#ifdef EIGEN_EXCEPTIONS
    if (Foo::object_count > Foo::copy_limit) throw Foo::Fail();
#endif
    std::cout << 'c';
    ++Foo::object_count;
    eigen_assert((std::uintptr_t(this) & (127)) == 0);
  }

  ~Foo() {
    std::cout << '~';
    --Foo::object_count;
  }
#ifdef EIGEN_EXCEPTIONS
  class Fail : public std::exception {};
#endif
};

Index Foo::object_count = 0;
Index Foo::object_limit = 0;
Index Foo::copy_limit = NumTraits<Index>::highest();

EIGEN_DECLARE_TEST(maxsizevector) {
  typedef MaxSizeVector<Foo> VectorX;
  Foo::object_count = 0;
  for (int r = 0; r < g_repeat; r++) {
    Index rows = internal::random<Index>(3, 30);
    Foo::object_limit = internal::random<Index>(0, rows - 2);
    std::cout << "object_limit = " << Foo::object_limit << std::endl;
#ifdef EIGEN_EXCEPTIONS
    bool exception_raised = false;
    try {
      std::cout << "\nVectorX m(" << rows << ");\n";
      VectorX vect(rows);
      for (int i = 0; i < rows; ++i) vect.push_back(Foo());
      VERIFY(false);  // not reached if exceptions are enabled
    } catch (const Foo::Fail&) {
      exception_raised = true;
    }
    VERIFY(exception_raised);
#endif
    VERIFY_IS_EQUAL(Index(0), Foo::object_count);

    {
      Foo::object_limit = rows + 1;
      VectorX vect2(rows, Foo());
      VERIFY_IS_EQUAL(Foo::object_count, rows);
    }
    VERIFY_IS_EQUAL(Index(0), Foo::object_count);
#ifdef EIGEN_EXCEPTIONS
    // A copy throwing partway through the fill destroys exactly the copies already made.
    exception_raised = false;
    Foo::copy_limit = rows / 2;
    try {
      VectorX vect3(rows, Foo());
    } catch (const Foo::Fail&) {
      exception_raised = true;
    }
    Foo::copy_limit = NumTraits<Index>::highest();
    VERIFY(exception_raised);
    VERIFY_IS_EQUAL(Index(0), Foo::object_count);

    // A throwing push_back leaves the vector unchanged.
    {
      VectorX vect4(rows);
      vect4.push_back(Foo());
      exception_raised = false;
      Foo::copy_limit = 0;
      try {
        vect4.push_back(Foo());
      } catch (const Foo::Fail&) {
        exception_raised = true;
      }
      Foo::copy_limit = NumTraits<Index>::highest();
      VERIFY(exception_raised);
      VERIFY_IS_EQUAL(vect4.size(), size_t(1));
    }
    VERIFY_IS_EQUAL(Index(0), Foo::object_count);
#endif
    std::cout << '\n';
  }
}
