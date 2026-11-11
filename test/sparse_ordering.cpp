// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "sparse.h"
#include <Eigen/OrderingMethods>

template <typename Idx>
void verify_is_permutation(const PermutationMatrix<Dynamic, Dynamic, Idx>& perm) {
  using Vec = Matrix<Idx, Dynamic, 1>;
  Vec sorted = perm.indices();
  std::sort(sorted.data(), sorted.data() + sorted.size());
  Vec expected = Vec::LinSpaced(perm.size(), 0, Idx(perm.size() - 1));
  VERIFY_IS_EQUAL(sorted, expected);
}

template <typename Scalar>
void test_colamd_uncompressed() {
  using Idx = int;
  using SpMat = SparseMatrix<Scalar, ColMajor, Idx>;
  const int n = 32;
  SpMat A(n, n);
  A.reserve(VectorXi::Constant(n, 5));
  for (int j = 0; j < n; ++j) {
    A.insert(j, j) = Scalar(2) + Scalar(j);
    if (j > 0) A.insert(j - 1, j) = Scalar(-1);
    if (j + 1 < n) A.insert(j + 1, j) = Scalar(-1);
  }
  A.insert(0, n - 1) = Scalar(0.5);
  A.insert(n - 1, 0) = Scalar(0.25);
  VERIFY(!A.isCompressed());

  // Regression for #649: COLAMD on uncompressed input used to silently corrupt
  // the permutation in release builds.
  PermutationMatrix<Dynamic, Dynamic, Idx> perm_u, perm_c;
  COLAMDOrdering<Idx> ord;
  ord(A, perm_u);
  verify_is_permutation(perm_u);

  SpMat B = A;
  B.makeCompressed();
  ord(B, perm_c);
  verify_is_permutation(perm_c);

  VERIFY_IS_EQUAL(perm_u.indices(), perm_c.indices());
}

void test_colamd_mixed_index_types() {
  using SpMat = SparseMatrix<double, ColMajor, int>;
  const int n = 8;
  SpMat A(n, n);
  A.reserve(VectorXi::Constant(n, 3));
  for (int j = 0; j < n; ++j) {
    A.insert(j, j) = 2.0 + j;
    if (j > 0) A.insert(j - 1, j) = -1.0;
    if (j + 1 < n) A.insert(j + 1, j) = -1.0;
  }
  A.makeCompressed();

  // COLAMDOrdering<StorageIndex> is a public API choice independent of the
  // source matrix's StorageIndex.  Keep accepting mixed index types.
  PermutationMatrix<Dynamic, Dynamic, long> perm;
  COLAMDOrdering<long> ord;
  ord(A, perm);
  verify_is_permutation(perm);
}

template <typename Scalar>
void test_amd_uncompressed() {
  using Idx = int;
  using SpMat = SparseMatrix<Scalar, ColMajor, Idx>;
  const int n = 24;
  SpMat A(n, n);
  A.reserve(VectorXi::Constant(n, 4));
  for (int j = 0; j < n; ++j) {
    A.insert(j, j) = Scalar(2) + Scalar(j);
    if (j > 0) {
      A.insert(j - 1, j) = Scalar(-1);
      A.insert(j, j - 1) = Scalar(-1);
    }
  }
  VERIFY(!A.isCompressed());

  PermutationMatrix<Dynamic, Dynamic, Idx> perm;
  AMDOrdering<Idx> ord;
  ord(A, perm);
  verify_is_permutation(perm);
}

// Node 0 is adjacent to the `degree` highest-numbered nodes, every other node only to itself.
template <typename Idx>
PermutationMatrix<Dynamic, Dynamic, Idx> amd_of_star(Idx n, Idx degree) {
  std::vector<Triplet<double, Idx> > triplets;
  triplets.reserve(n + 2 * degree);
  for (Idx j = 0; j < n; ++j) triplets.emplace_back(j, j, 1.0);
  for (Idx j = n - degree; j < n; ++j) {
    triplets.emplace_back(0, j, 1.0);
    triplets.emplace_back(j, 0, 1.0);
  }
  SparseMatrix<double, ColMajor, Idx> A(n, n);
  A.setFromTriplets(triplets.begin(), triplets.end());
  PermutationMatrix<Dynamic, Dynamic, Idx> perm;
  AMDOrdering<Idx> ord;
  ord(A, perm);
  return perm;
}

void test_amd_hash_overflow() {
  // Node 0 stays below AMD's dense threshold 10*sqrt(n) (its column also holds the diagonal), and its supervariable
  // hash, the sum of its neighbours' indices, exceeds INT_MAX. With a signed int hash this overflowed, h % n was
  // negative and hhead[h] was written out of bounds. The ordering does not depend on the hash values, so the int and
  // the 64-bit orderings must agree.
  const int n = 400000;
  const int degree = int(10 * std::sqrt(double(n))) - 2;
  VERIFY(double(degree) * double(n - degree) > double(NumTraits<int>::highest()));
  PermutationMatrix<Dynamic, Dynamic, int> perm = amd_of_star<int>(n, degree);
  verify_is_permutation(perm);
  PermutationMatrix<Dynamic, Dynamic, std::int64_t> perm64 = amd_of_star<std::int64_t>(n, degree);
  VERIFY_IS_EQUAL(perm.indices().cast<std::int64_t>().eval(), perm64.indices());
}

EIGEN_DECLARE_TEST(sparse_ordering) {
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1(test_colamd_uncompressed<double>());
    CALL_SUBTEST_2(test_colamd_uncompressed<std::complex<double> >());
    CALL_SUBTEST_3(test_amd_uncompressed<double>());
    CALL_SUBTEST_4(test_colamd_mixed_index_types());
  }
  CALL_SUBTEST_5(test_amd_hash_overflow());
}
