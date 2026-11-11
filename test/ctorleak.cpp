// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include "lifetime_scalar.h"

using Foo = lifetime_scalar::LifetimeScalar<>;
using lifetime_scalar::Failure;

#undef EIGEN_TEST_MAX_SIZE
#define EIGEN_TEST_MAX_SIZE 3

EIGEN_DECLARE_TEST(ctorleak) {
  using MatrixX = Matrix<Foo, Dynamic, Dynamic>;
  using VectorX = Matrix<Foo, Dynamic, 1>;

  for (int i = 0; i < g_repeat; i++) {
    Index rows = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE),
          cols = internal::random<Index>(2, EIGEN_TEST_MAX_SIZE);
    Foo::max_live = rows * cols + 1;
    {
      MatrixX r(rows, cols);
      Foo::max_live = r.size() + internal::random<Index>(0, rows * cols - 2) + 1;
#ifdef EIGEN_EXCEPTIONS
      try {
#endif
        if (internal::random<bool>()) {
          std::cout << "\nMatrixX m(" << rows << ", " << cols << ");\n";
          MatrixX m(rows, cols);
        } else {
          std::cout << "\nMatrixX m(r);\n";
          MatrixX m(r);
        }
#ifdef EIGEN_EXCEPTIONS
        VERIFY(false);             // not reached if exceptions are enabled
      } catch (const Failure&) {
        // The elements constructed before the throw are destroyed by then; only r's remain.
        VERIFY_IS_EQUAL(r.size(), Foo::live);
      }
#endif
    }
    VERIFY_IS_EQUAL(Index(0), Foo::live);

    {
      Foo::max_live = (rows + 1) * (cols + 1) + 1;
      MatrixX A(rows, cols);
      VERIFY_IS_EQUAL(Foo::live, rows * cols);
      VectorX v = A.row(0);
      VERIFY_IS_EQUAL(Foo::live, (rows + 1) * cols);
      v = A.col(0);
      VERIFY_IS_EQUAL(Foo::live, rows * (cols + 1));
    }
    VERIFY_IS_EQUAL(Index(0), Foo::live);
  }
  std::cout << "\n";
}
