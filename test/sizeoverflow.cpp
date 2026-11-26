// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#ifdef EIGEN_EXCEPTIONS
#define VERIFY_THROWS_BADALLOC(a)                         \
  {                                                       \
    bool threw = false;                                   \
    try {                                                 \
      a;                                                  \
    } catch (std::bad_alloc&) {                           \
      threw = true;                                       \
    }                                                     \
    VERIFY(threw && "should have thrown bad_alloc: " #a); \
  }
#else
// No way to catch a bad alloc - program terminates.
#define VERIFY_THROWS_BADALLOC(a)
#endif

template <typename MatrixType>
void triggerMatrixBadAlloc(Index rows, Index cols) {
  VERIFY_THROWS_BADALLOC(MatrixType m(rows, cols));
  VERIFY_THROWS_BADALLOC(MatrixType m; m.resize(rows, cols));
  VERIFY_THROWS_BADALLOC(MatrixType m; m.conservativeResize(rows, cols));
  // The failed resize of a non-empty object must leave it destructible.
  VERIFY_THROWS_BADALLOC(MatrixType m(3, 3); m.resize(rows, cols));
  VERIFY_THROWS_BADALLOC(MatrixType m(3, 3); m.conservativeResize(rows, cols));
}

template <typename VectorType>
void triggerVectorBadAlloc(Index size) {
  VERIFY_THROWS_BADALLOC(VectorType v(size));
  VERIFY_THROWS_BADALLOC(VectorType v; v.resize(size));
  VERIFY_THROWS_BADALLOC(VectorType v; v.conservativeResize(size));
  VERIFY_THROWS_BADALLOC(VectorType v(3); v.resize(size));
  VERIFY_THROWS_BADALLOC(VectorType v(3); v.conservativeResize(size));
}

EIGEN_DECLARE_TEST(sizeoverflow) {
  // there are 2 levels of overflow checking. first in PlainObjectBase.h we check for overflow in rows*cols
  // computations. this is tested in tests of the form times_itself_gives_0 * times_itself_gives_0 Then in Memory.h we
  // check for overflow in size * sizeof(T) computations. this is tested in tests of the form times_4_gives_0 *
  // sizeof(float), which only run when Index is as wide as size_t.

  using UIndex = std::make_unsigned_t<Index>;
  const Index times_itself_gives_0 = Index(1) << (8 * sizeof(Index) / 2);
  VERIFY(UIndex(UIndex(times_itself_gives_0) * UIndex(times_itself_gives_0)) == 0);

  triggerMatrixBadAlloc<MatrixXf>(times_itself_gives_0, times_itself_gives_0);
  triggerMatrixBadAlloc<MatrixXd>(times_itself_gives_0, times_itself_gives_0);

  // The sizes below fit Index, and their byte counts overflow size_t only when it is no wider than Index. With a
  // narrower Index, as for EIGEN_DEFAULT_DENSE_INDEX_TYPE=int on a 64-bit target, the requests are representable and
  // need not fail, so the second level is not tested there.
  if (sizeof(Index) == sizeof(size_t)) {
    const Index times_4_gives_0 = Index(1) << (8 * sizeof(Index) - 2);
    VERIFY(size_t(times_4_gives_0) * 4 == 0);

    const Index times_8_gives_0 = Index(1) << (8 * sizeof(Index) - 3);
    VERIFY(size_t(times_8_gives_0) * 8 == 0);

    triggerMatrixBadAlloc<MatrixXf>(times_itself_gives_0 / 4, times_itself_gives_0);
    triggerMatrixBadAlloc<MatrixXf>(times_4_gives_0, 1);

    triggerMatrixBadAlloc<MatrixXd>(times_itself_gives_0 / 8, times_itself_gives_0);
    triggerMatrixBadAlloc<MatrixXd>(times_8_gives_0, 1);

    triggerVectorBadAlloc<VectorXf>(times_4_gives_0);

    triggerVectorBadAlloc<VectorXd>(times_8_gives_0);
  }

  // A reallocation that fails throws for unaligned storage too, leaving the vector as it was. The request is
  // representable but not satisfiable; ASan aborts on such requests instead of failing them.
#if defined(EIGEN_EXCEPTIONS) && !defined(__SANITIZE_ADDRESS__) && !__has_feature(address_sanitizer)
  if (sizeof(Index) >= 8) {
    using UnalignedVector = Matrix<float, Dynamic, 1, DontAlign>;
    UnalignedVector v = UnalignedVector::Ones(3);
    VERIFY_THROWS_BADALLOC(v.conservativeResize(Index(1) << (8 * sizeof(Index) - 4)));
    VERIFY_IS_EQUAL(v, UnalignedVector::Ones(3));
  }
#endif
}
