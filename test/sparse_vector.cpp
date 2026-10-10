// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "sparse.h"

template <typename Scalar, typename StorageIndex>
void sparse_vector(int rows, int cols) {
  double densityMat = (std::max)(8. / (rows * cols), 0.01);
  double densityVec = (std::max)(8. / (rows), 0.1);
  typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrix;
  typedef Matrix<Scalar, Dynamic, 1> DenseVector;
  typedef Matrix<DenseIndex, Dynamic, 1> DenseIndexVector;
  typedef SparseVector<Scalar, 0, StorageIndex> SparseVectorType;
  typedef SparseMatrix<Scalar, 0, StorageIndex> SparseMatrixType;
  Scalar eps = 1e-6;

  SparseMatrixType m1(rows, rows);
  SparseVectorType v1(rows), v2(rows), v3(rows);
  DenseMatrix refM1 = DenseMatrix::Zero(rows, rows);
  DenseVector refV1 = DenseVector::Random(rows), refV2 = DenseVector::Random(rows), refV3 = DenseVector::Random(rows);

  std::vector<int> zerocoords, nonzerocoords;
  // Retry until at least one nonzero is produced: code below indexes nonzerocoords[0].
  do {
    zerocoords.clear();
    nonzerocoords.clear();
    initSparse<Scalar>(densityVec, refV1, v1, &zerocoords, &nonzerocoords);
  } while (nonzerocoords.empty());
  initSparse<Scalar>(densityMat, refM1, m1);

  initSparse<Scalar>(densityVec, refV2, v2);
  initSparse<Scalar>(densityVec, refV3, v3);

  Scalar s1 = internal::random<Scalar>();

  // test coeff and coeffRef
  for (unsigned int i = 0; i < zerocoords.size(); ++i) {
    VERIFY_IS_MUCH_SMALLER_THAN(v1.coeff(zerocoords[i]), eps);
    // VERIFY_RAISES_ASSERT( v1.coeffRef(zerocoords[i]) = 5 );
  }
  {
    VERIFY(int(nonzerocoords.size()) == v1.nonZeros());
    int j = 0;
    for (typename SparseVectorType::InnerIterator it(v1); it; ++it, ++j) {
      VERIFY(nonzerocoords[j] == it.index());
      VERIFY_IS_EQUAL(it.value(), v1.coeff(it.index()));
      VERIFY_IS_EQUAL(it.value(), refV1.coeff(it.index()));
    }
  }
  VERIFY_IS_APPROX(v1, refV1);

  // test coeffRef with reallocation
  {
    SparseVectorType v4(rows);
    DenseVector v5 = DenseVector::Zero(rows);
    for (int k = 0; k < rows; ++k) {
      int i = internal::random<int>(0, rows - 1);
      Scalar v = internal::random<Scalar>();
      v4.coeffRef(i) += v;
      v5.coeffRef(i) += v;
    }
    VERIFY_IS_APPROX(v4, v5);
  }

  v1.coeffRef(nonzerocoords[0]) = Scalar(5);
  refV1.coeffRef(nonzerocoords[0]) = Scalar(5);
  VERIFY_IS_APPROX(v1, refV1);

  VERIFY_IS_APPROX(v1 + v2, refV1 + refV2);
  VERIFY_IS_APPROX(v1 + v2 + v3, refV1 + refV2 + refV3);

  VERIFY_IS_APPROX(v1 * s1 - v2, refV1 * s1 - refV2);

  VERIFY_IS_APPROX(v1 *= s1, refV1 *= s1);
  VERIFY_IS_APPROX(v1 /= s1, refV1 /= s1);

  VERIFY_IS_APPROX(v1 += v2, refV1 += refV2);
  VERIFY_IS_APPROX(v1 -= v2, refV1 -= refV2);

  VERIFY_IS_APPROX(v1.dot(v2), refV1.dot(refV2));
  VERIFY_IS_APPROX(v1.dot(refV2), refV1.dot(refV2));

  VERIFY_IS_APPROX(m1 * v2, refM1 * refV2);
  VERIFY_IS_APPROX(v1.dot(m1 * v2), refV1.dot(refM1 * refV2));
  {
    int i = internal::random<int>(0, rows - 1);
    VERIFY_IS_APPROX(v1.dot(m1.col(i)), refV1.dot(refM1.col(i)));
  }

  VERIFY_IS_APPROX(v1.squaredNorm(), refV1.squaredNorm());

  VERIFY_IS_APPROX(v1.blueNorm(), refV1.blueNorm());
  VERIFY_IS_APPROX((v1 * s1).blueNorm(), (refV1 * s1).blueNorm());

  // test aliasing
  VERIFY_IS_APPROX((v1 = -v1), (refV1 = -refV1));
  VERIFY_IS_APPROX((v1 = v1.transpose()), (refV1 = refV1.transpose().eval()));
  VERIFY_IS_APPROX((v1 += -v1), (refV1 += -refV1));

  // sparse matrix to sparse vector
  SparseMatrixType mv1;
  VERIFY_IS_APPROX((mv1 = v1), v1);
  VERIFY_IS_APPROX(mv1, (v1 = mv1));
  VERIFY_IS_APPROX(mv1, (v1 = mv1.transpose()));

  // check copy to dense vector with transpose
  refV3.resize(0);
  VERIFY_IS_APPROX(refV3 = v1.transpose(), v1.toDense());
  VERIFY_IS_APPROX(DenseVector(v1), v1.toDense());

  // test move
  {
    SparseVectorType tmp(std::move(v1));
    VERIFY_IS_APPROX(tmp, refV1);
    v1 = tmp;
  }

  {
    SparseVectorType tmp;
    tmp = std::move(v1);
    VERIFY_IS_APPROX(tmp, refV1);
    v1 = tmp;
  }

  {
    SparseVectorType tmp(std::move(mv1));
    VERIFY_IS_APPROX(tmp, refV1);
    mv1 = tmp;
  }

  {
    SparseVectorType tmp;
    tmp = std::move(mv1);
    VERIFY_IS_APPROX(tmp, refV1);
    mv1 = tmp;
  }

  // test conservative resize
  {
    std::vector<StorageIndex> inc;
    if (rows > 3) inc.push_back(-3);
    inc.push_back(0);
    inc.push_back(3);
    inc.push_back(1);
    inc.push_back(10);

    for (std::size_t i = 0; i < inc.size(); i++) {
      StorageIndex incRows = inc[i];
      SparseVectorType vec1(rows);
      DenseVector refVec1 = DenseVector::Zero(rows);
      initSparse<Scalar>(densityVec, refVec1, vec1);

      vec1.conservativeResize(rows + incRows);
      refVec1.conservativeResize(rows + incRows);
      if (incRows > 0) refVec1.tail(incRows).setZero();

      VERIFY_IS_APPROX(vec1, refVec1);

      // Insert new values
      if (incRows > 0) vec1.insert(vec1.rows() - 1) = refVec1(refVec1.rows() - 1) = 1;

      VERIFY_IS_APPROX(vec1, refVec1);
    }
  }

  // test sort
  if (rows > 1) {
    SparseVectorType vec1(rows);
    DenseVector refVec1 = DenseVector::Zero(rows);
    DenseIndexVector innerIndices(rows);
    innerIndices.setLinSpaced(0, rows - 1);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(innerIndices.begin(), innerIndices.end(), g);
    Index nz = internal::random<Index>(2, rows / 2);
    for (Index k = 0; k < nz; k++) {
      Index i = innerIndices[k];
      Scalar val = internal::random<Scalar>();
      refVec1.coeffRef(i) = val;
      vec1.insert(i) = val;
    }

    vec1.template sortInnerIndices<std::greater<>>();
    VERIFY_IS_APPROX(vec1, refVec1);
    VERIFY_IS_EQUAL(vec1.template innerIndicesAreSorted<std::greater<>>(), 1);
    VERIFY_IS_EQUAL(vec1.template innerIndicesAreSorted<std::less<>>(), 0);
    vec1.template sortInnerIndices<std::less<>>();
    VERIFY_IS_APPROX(vec1, refVec1);
    VERIFY_IS_EQUAL(vec1.template innerIndicesAreSorted<std::greater<>>(), 0);
    VERIFY_IS_EQUAL(vec1.template innerIndicesAreSorted<std::less<>>(), 1);
  }
}
void test_pruning() {
  using SparseVectorType = SparseVector<double, 0, int>;

  SparseVectorType vec;
  auto init_vec = [&]() {
    ;
    vec.resize(10);
    vec.insert(3) = 0.1;
    vec.insert(5) = 1.0;
    vec.insert(8) = -0.1;
    vec.insert(9) = -0.2;
  };
  init_vec();

  VERIFY_IS_EQUAL(vec.nonZeros(), 4);
  VERIFY_IS_EQUAL(vec.prune(0.1, 1.0), 2);
  VERIFY_IS_EQUAL(vec.nonZeros(), 2);
  VERIFY_IS_EQUAL(vec.coeff(5), 1.0);
  VERIFY_IS_EQUAL(vec.coeff(9), -0.2);

  init_vec();
  VERIFY_IS_EQUAL(vec.prune([](double v) { return v >= 0; }), 2);
  VERIFY_IS_EQUAL(vec.nonZeros(), 2);
  VERIFY_IS_EQUAL(vec.coeff(3), 0.1);
  VERIFY_IS_EQUAL(vec.coeff(5), 1.0);
}

// Reductions of empty sparse objects are zero, as for dense ones.
void test_empty_reductions() {
  SparseMatrix<double> m(0, 3), m2(0, 3);
  SparseVector<double> v(0);
  VERIFY_IS_EQUAL(m.sum(), 0.0);
  VERIFY_IS_EQUAL(m.norm(), 0.0);
  VERIFY(m.isApprox(m2));
  VERIFY_IS_EQUAL(v.squaredNorm(), 0.0);
  VERIFY_IS_EQUAL(v.dot(VectorXd(0)), 0.0);

  // Empty blocks that cross the outer vectors of a matrix with no outer vectors.
  SparseMatrix<double> c(3, 0);
  SparseMatrix<double, RowMajor> r(0, 3);
  VERIFY_IS_EQUAL(c.row(1).sum(), 0.0);
  VERIFY_IS_EQUAL(c.row(1).norm(), 0.0);
  VERIFY_IS_EQUAL(c.row(1).dot(RowVectorXd(0)), 0.0);
  VERIFY_IS_EQUAL(c.row(1).dot(SparseVector<double, RowMajor>(0)), 0.0);
  VERIFY_IS_EQUAL(r.col(1).sum(), 0.0);
}

// An empty block that crosses the outer vectors must not open an iterator on one. The outer index array here is
// followed by a nonzero count, and the index and value pointers are null, so such an iterator would dereference null.
void test_empty_outer_vector_block_opens_no_iterator() {
  const int outer[2] = {0, 5};
  Map<const SparseMatrix<double, ColMajor, int>> c(3, 0, 5, outer, static_cast<const int*>(nullptr),
                                                   static_cast<const double*>(nullptr));
  MatrixXd d = c.row(1);
  VERIFY_IS_EQUAL(d.size(), 0);
  SparseVector<double, RowMajor> x = c.row(1);
  VERIFY_IS_EQUAL(x.size(), 0);
  VERIFY_IS_EQUAL(x.nonZeros(), 0);
}

void test_swap_with_matrix() {
  // The matrix may be uncompressed and holds a different number of nonzeros than the vector.
  SparseMatrix<double> m(5, 1);
  m.reserve(VectorXi::Constant(1, 4));
  m.insert(1, 0) = 1;
  SparseVector<double> v(5);
  v.insert(0) = 2;
  v.insert(4) = 3;
  const VectorXd refM = m.toDense(), refV = v.toDense();
  v.swap(m);
  VERIFY_IS_EQUAL(m.nonZeros(), 2);
  VERIFY_IS_EQUAL(VectorXd(m.toDense()), refV);
  VERIFY_IS_EQUAL(VectorXd(v.toDense()), refM);

  SparseMatrix<double, RowMajor> mr(1, 5);
  mr.insert(0, 1) = 1;
  SparseVector<double, RowMajor> vr(5);
  vr.insert(3) = 2;
  vr.swap(mr);
  VERIFY_IS_EQUAL(mr.coeff(0, 3), 2.0);
  VERIFY_IS_EQUAL(vr.coeff(1), 1.0);
}

EIGEN_DECLARE_TEST(sparse_vector) {
  for (int i = 0; i < g_repeat; i++) {
    int r = Eigen::internal::random<int>(1, 500), c = Eigen::internal::random<int>(1, 500);
    if (Eigen::internal::random<int>(0, 4) == 0) {
      r = c;  // check square matrices in 25% of tries
    }
    EIGEN_UNUSED_VARIABLE(r + c);

    CALL_SUBTEST_1((sparse_vector<double, int>(8, 8)));
    CALL_SUBTEST_2((sparse_vector<std::complex<double>, int>(r, c)));
    CALL_SUBTEST_1((sparse_vector<double, long int>(r, c)));
    CALL_SUBTEST_1((sparse_vector<double, short>(r, c)));
  }

  CALL_SUBTEST_1(test_pruning());
  CALL_SUBTEST_1(test_swap_with_matrix());
  CALL_SUBTEST_1(test_empty_reductions());
  CALL_SUBTEST_1(test_empty_outer_vector_block_opens_no_iterator());
}
