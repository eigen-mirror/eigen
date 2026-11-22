// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include <cstdlib>
#include <string>

#include "sparse.h"

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

#include <Eigen/SparseExtra>

// Read from an environment variable TEST_TMPDIR, if available,
// and append the provided filename. Defaults to local directory.
std::string GetTestTempFilename(const char* filename) {
  const char* test_tmpdir = std::getenv("TEST_TMPDIR");
  if (test_tmpdir == nullptr) {
    return std::string(filename);
  }
  return std::string(test_tmpdir) + std::string("/") + std::string(filename);
}

template <typename SetterType, typename DenseType, typename Scalar, int Options>
bool test_random_setter(SparseMatrix<Scalar, Options>& sm, const DenseType& ref,
                        const std::vector<Vector2i>& nonzeroCoords) {
  {
    sm.setZero();
    SetterType w(sm);
    std::vector<Vector2i> remaining = nonzeroCoords;
    while (!remaining.empty()) {
      int i = internal::random<int>(0, static_cast<int>(remaining.size()) - 1);
      w(remaining[i].x(), remaining[i].y()) = ref.coeff(remaining[i].x(), remaining[i].y());
      remaining[i] = remaining.back();
      remaining.pop_back();
    }
  }
  return sm.isApprox(ref);
}

template <typename SparseMatrixType>
void sparse_extra(const SparseMatrixType& ref) {
  const Index rows = ref.rows();
  const Index cols = ref.cols();
  using Scalar = typename SparseMatrixType::Scalar;

  double density = (std::max)(8. / (rows * cols), 0.01);
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;

  SparseMatrixType m(rows, cols);
  DenseMatrix refMat = DenseMatrix::Zero(rows, cols);

  std::vector<Vector2i> zeroCoords;
  std::vector<Vector2i> nonzeroCoords;
  initSparse<Scalar>(density, refMat, m, 0, &zeroCoords, &nonzeroCoords);

  if (zeroCoords.empty() || nonzeroCoords.empty()) return;

  // random setter
  //   {
  //     m.setZero();
  //     VERIFY_IS_NOT_APPROX(m, refMat);
  //     SparseSetter<SparseMatrixType, RandomAccessPattern> w(m);
  //     std::vector<Vector2i> remaining = nonzeroCoords;
  //     while(!remaining.empty())
  //     {
  //       int i = internal::random<int>(0,remaining.size()-1);
  //       w->coeffRef(remaining[i].x(),remaining[i].y()) = refMat.coeff(remaining[i].x(),remaining[i].y());
  //       remaining[i] = remaining.back();
  //       remaining.pop_back();
  //     }
  //   }
  //   VERIFY_IS_APPROX(m, refMat);

  VERIFY((test_random_setter<RandomSetter<SparseMatrixType, StdMapTraits> >(m, refMat, nonzeroCoords)));
  VERIFY((test_random_setter<RandomSetter<SparseMatrixType, StdUnorderedMapTraits> >(m, refMat, nonzeroCoords)));
#ifdef EIGEN_GOOGLEHASH_SUPPORT
  VERIFY((test_random_setter<RandomSetter<SparseMatrixType, GoogleDenseHashMapTraits> >(m, refMat, nonzeroCoords)));
  VERIFY((test_random_setter<RandomSetter<SparseMatrixType, GoogleSparseHashMapTraits> >(m, refMat, nonzeroCoords)));
#endif

  // test RandomSetter
  /*{
    SparseMatrixType m1(rows,cols), m2(rows,cols);
    DenseMatrix refM1 = DenseMatrix::Zero(rows, rows);
    initSparse<Scalar>(density, refM1, m1);
    {
      Eigen::RandomSetter<SparseMatrixType > setter(m2);
      for (int j=0; j<m1.outerSize(); ++j)
        for (typename SparseMatrixType::InnerIterator i(m1,j); i; ++i)
          setter(i.index(), j) = i.value();
    }
    VERIFY_IS_APPROX(m1, m2);
  }*/
}

// The setter imports the target's coefficients and writes the result back, so the target may be in either storage
// mode, with or without coefficients, when the setter is constructed.
template <typename SetterType, typename SparseMatrixType>
void check_random_setter_target_modes() {
  using Scalar = typename SparseMatrixType::Scalar;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  const Index n = 70;
  for (int prefill = 0; prefill < 2; ++prefill) {
    for (int uncompressed = 0; uncompressed < 2; ++uncompressed) {
      SparseMatrixType m(n, n);
      DenseMatrix ref = DenseMatrix::Zero(n, n);
      if (prefill) {
        for (Index i = 0; i < n; i += 3) {
          m.insert(i, (i * 7) % n) = Scalar(1 + i);
          ref(i, (i * 7) % n) = Scalar(1 + i);
        }
        if (!uncompressed) m.makeCompressed();
      } else if (uncompressed) {
        m.reserve(Matrix<int, Dynamic, 1>::Constant(n, 4));
      }
      VERIFY_IS_EQUAL(m.isCompressed(), !uncompressed);
      {
        SetterType w(m);
        for (Index k = 0; k < 200; ++k) {
          const Index i = (k * 37) % n, j = (k * 11 + 5) % n;
          w(i, j) += Scalar(k + 1);
          ref(i, j) += Scalar(k + 1);
        }
      }
      VERIFY(m.isCompressed());
      VERIFY(m.innerIndicesAreSorted());
      VERIFY_IS_EQUAL(m.nonZeros(), (ref.array() != Scalar(0)).count());
      VERIFY_IS_EQUAL(DenseMatrix(m), ref);
    }
  }
}

template <typename SparseMatrixType>
void check_random_setter_target_modes() {
  check_random_setter_target_modes<RandomSetter<SparseMatrixType, StdMapTraits>, SparseMatrixType>();
  check_random_setter_target_modes<RandomSetter<SparseMatrixType, StdUnorderedMapTraits>, SparseMatrixType>();
  check_random_setter_target_modes<RandomSetter<SparseMatrixType>, SparseMatrixType>();
#ifdef EIGEN_GOOGLEHASH_SUPPORT
  check_random_setter_target_modes<RandomSetter<SparseMatrixType, GoogleDenseHashMapTraits>, SparseMatrixType>();
  check_random_setter_target_modes<RandomSetter<SparseMatrixType, GoogleSparseHashMapTraits>, SparseMatrixType>();
#endif
}

template <typename SparseMatrixType>
void check_marketio() {
  using DenseMatrix = Matrix<typename SparseMatrixType::Scalar, Dynamic, Dynamic>;
  Index rows = internal::random<Index>(1, 100);
  Index cols = internal::random<Index>(1, 100);
  SparseMatrixType m1, m2;
  m1 = DenseMatrix::Random(rows, cols).sparseView();
  std::string filename = GetTestTempFilename("sparse_extra.mtx");
  saveMarket(m1, filename);
  loadMarket(m2, filename);
  VERIFY_IS_EQUAL(DenseMatrix(m1), DenseMatrix(m2));
}

template <typename VectorType>
void check_marketio_vector() {
  Index size = internal::random<Index>(1, 100);
  VectorType v1, v2;
  v1 = VectorType::Random(size);
  std::string filename = GetTestTempFilename("vector_extra.mtx");
  saveMarketVector(v1, filename);
  loadMarketVector(v2, filename);
  VERIFY_IS_EQUAL(v1, v2);
}

template <typename DenseMatrixType>
void check_marketio_dense() {
  Index rows = DenseMatrixType::MaxRowsAtCompileTime;
  if (DenseMatrixType::MaxRowsAtCompileTime == Dynamic) {
    rows = internal::random<Index>(1, 100);
  } else if (DenseMatrixType::RowsAtCompileTime == Dynamic) {
    rows = internal::random<Index>(1, DenseMatrixType::MaxRowsAtCompileTime);
  }

  Index cols = DenseMatrixType::MaxColsAtCompileTime;
  if (DenseMatrixType::MaxColsAtCompileTime == Dynamic) {
    cols = internal::random<Index>(1, 100);
  } else if (DenseMatrixType::ColsAtCompileTime == Dynamic) {
    cols = internal::random<Index>(1, DenseMatrixType::MaxColsAtCompileTime);
  }

  DenseMatrixType m1, m2;
  m1 = DenseMatrixType::Random(rows, cols);
  std::string filename = GetTestTempFilename("dense_extra.mtx");
  saveMarketDense(m1, filename);
  loadMarketDense(m2, filename);
  VERIFY_IS_EQUAL(m1, m2);
}

template <typename Scalar>
void check_sparse_inverse() {
  using MatrixType = SparseMatrix<Scalar>;
  using RowMatrixType = SparseMatrix<Scalar, RowMajor>;

  Matrix<Scalar, -1, -1> A;
  A.resize(1000, 1000);
  A.fill(0);
  A.setIdentity();
  A.col(0).array() += 1;
  A.row(0).array() += 2;
  A.col(2).array() += 3;
  A.row(7).array() += 3;
  A.col(9).array() += 3;
  A.block(3, 4, 4, 2).array() += 9;
  A.middleRows(10, 50).array() += 3;
  A.middleCols(50, 50).array() += 40;
  A.block(500, 300, 40, 20).array() += 10;
  A.transposeInPlace();

  Eigen::SparseLU<MatrixType> slu;
  slu.compute(A.sparseView());
  Matrix<Scalar, -1, -1> Id(A.rows(), A.cols());
  Id.setIdentity();
  Matrix<Scalar, -1, -1> inv = slu.solve(Id);

  const MatrixType sparseInv = Eigen::SparseInverse<Scalar>().compute(A.sparseView()).inverse();

  // The two inverses are computed independently (SparseLU solve against the identity vs the Takahashi
  // recurrence), so entrywise and summed deviations scale with n * eps for this 1000x1000 matrix. The input is
  // deterministic, but the deviation still moves by a factor of three with the accumulation order that the compiler,
  // the ISA and the buffer alignment happen to produce, so these factors are the smallest powers of two at or above
  // the long-standing 1e-11 and 1e-10 bounds rather than a tight fit to a single host.
  Scalar sumdiff = 0;  // Check the diff only of the non-zero elements
  for (Eigen::Index j = 0; j < A.cols(); j++) {
    for (typename MatrixType::InnerIterator iter(sparseInv, j); iter; ++iter) {
      const Scalar diff = std::abs(inv(iter.row(), iter.col()) - iter.value());
      VERIFY_IS_APPROX_OR_LESS_THAN(diff, Scalar(65536) * NumTraits<Scalar>::epsilon());

      if (iter.value() != 0) {
        sumdiff += diff;
      }
    }
  }

  VERIFY_IS_APPROX_OR_LESS_THAN(sumdiff, Scalar(524288) * NumTraits<Scalar>::epsilon());

  RowMatrixType DU = slu.matrixU().toSparse();
  Matrix<Scalar, Dynamic, 1> invD = DU.diagonal().cwiseInverse();
  RowMatrixType scaled_before_view = (invD.asDiagonal() * DU).template triangularView<StrictlyUpper>();
  RowMatrixType view_before_scaled = invD.asDiagonal() * DU.template triangularView<StrictlyUpper>();
  VERIFY_IS_APPROX(scaled_before_view, view_before_scaled);
}

// The Takahashi recurrence uses bilinear (unconjugated) sums, so this needs entries with nonzero imaginary parts to
// tell a conjugated dot product from the right one.
template <typename Scalar>
void check_sparse_inverse_random_entries() {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using MatrixType = SparseMatrix<Scalar>;
  const Index n = 40;

  std::vector<Triplet<Scalar> > triplets;
  for (Index i = 0; i < n; ++i) {
    triplets.emplace_back(i, i, Scalar(RealScalar(8)) + internal::random<Scalar>());
    for (int k = 0; k < 4; ++k) {
      const Index j = internal::random<Index>(0, n - 1);
      if (j != i) triplets.emplace_back(i, j, internal::random<Scalar>());
    }
    if (i + 1 < n) triplets.emplace_back(i, i + 1, internal::random<Scalar>());
  }
  MatrixType A(n, n);
  A.setFromTriplets(triplets.begin(), triplets.end());

  const MatrixType sparseInv = SparseInverse<Scalar>().compute(A).inverse();
  const Matrix<Scalar, Dynamic, Dynamic> inv = Matrix<Scalar, Dynamic, Dynamic>(A).inverse();

  VERIFY_IS_EQUAL(sparseInv.rows(), n);
  VERIFY(sparseInv.nonZeros() >= A.nonZeros());
  const RealScalar tolerance =
      RealScalar(64) * RealScalar(n) * NumTraits<Scalar>::epsilon() * inv.cwiseAbs().maxCoeff();
  for (Index j = 0; j < sparseInv.outerSize(); ++j) {
    for (typename MatrixType::InnerIterator it(sparseInv, j); it; ++it) {
      VERIFY(numext::abs(it.value() - inv(it.row(), it.col())) <= tolerance);
    }
  }
}

EIGEN_DECLARE_TEST(sparse_extra) {
  for (int i = 0; i < g_repeat; i++) {
    int s = Eigen::internal::random<int>(1, 50);
    CALL_SUBTEST_1(sparse_extra(SparseMatrix<double>(8, 8)));
    CALL_SUBTEST_2(sparse_extra(SparseMatrix<std::complex<double> >(s, s)));
    CALL_SUBTEST_1(sparse_extra(SparseMatrix<double>(s, s)));

    CALL_SUBTEST_3((check_marketio<SparseMatrix<float, ColMajor, int> >()));
    CALL_SUBTEST_3((check_marketio<SparseMatrix<double, ColMajor, int> >()));
    CALL_SUBTEST_3((check_marketio<SparseMatrix<std::complex<float>, ColMajor, int> >()));
    CALL_SUBTEST_3((check_marketio<SparseMatrix<std::complex<double>, ColMajor, int> >()));
    CALL_SUBTEST_3((check_marketio<SparseMatrix<float, ColMajor, long int> >()));
    CALL_SUBTEST_3((check_marketio<SparseMatrix<double, ColMajor, long int> >()));
    CALL_SUBTEST_3((check_marketio<SparseMatrix<std::complex<float>, ColMajor, long int> >()));
    CALL_SUBTEST_3((check_marketio<SparseMatrix<std::complex<double>, ColMajor, long int> >()));

    CALL_SUBTEST_4((check_marketio_dense<Matrix<float, Dynamic, Dynamic> >()));
    CALL_SUBTEST_4((check_marketio_dense<Matrix<float, Dynamic, Dynamic, RowMajor> >()));
    CALL_SUBTEST_4((check_marketio_dense<Matrix<double, Dynamic, Dynamic> >()));
    CALL_SUBTEST_4((check_marketio_dense<Matrix<std::complex<float>, Dynamic, Dynamic> >()));
    CALL_SUBTEST_4((check_marketio_dense<Matrix<std::complex<double>, Dynamic, Dynamic> >()));
    CALL_SUBTEST_4((check_marketio_dense<Matrix<float, Dynamic, 3> >()));
    CALL_SUBTEST_4((check_marketio_dense<Matrix<double, 3, Dynamic> >()));
    CALL_SUBTEST_4((check_marketio_dense<Matrix<double, 3, 4> >()));
    CALL_SUBTEST_4((check_marketio_dense<Matrix<double, Dynamic, Dynamic, ColMajor, 5, 5> >()));

    CALL_SUBTEST_5((check_marketio_vector<Matrix<float, 1, Dynamic> >()));
    CALL_SUBTEST_5((check_marketio_vector<Matrix<double, 1, Dynamic> >()));
    CALL_SUBTEST_5((check_marketio_vector<Matrix<std::complex<float>, 1, Dynamic> >()));
    CALL_SUBTEST_5((check_marketio_vector<Matrix<std::complex<double>, 1, Dynamic> >()));
    CALL_SUBTEST_5((check_marketio_vector<Matrix<float, Dynamic, 1> >()));
    CALL_SUBTEST_5((check_marketio_vector<Matrix<double, Dynamic, 1> >()));
    CALL_SUBTEST_5((check_marketio_vector<Matrix<std::complex<float>, Dynamic, 1> >()));
    CALL_SUBTEST_5((check_marketio_vector<Matrix<std::complex<double>, Dynamic, 1> >()));

    CALL_SUBTEST_6((check_sparse_inverse<double>()));
    CALL_SUBTEST_6((check_sparse_inverse_random_entries<double>()));
    CALL_SUBTEST_6((check_sparse_inverse_random_entries<std::complex<double> >()));
    CALL_SUBTEST_6((check_sparse_inverse_random_entries<std::complex<float> >()));

    CALL_SUBTEST_7((check_random_setter_target_modes<SparseMatrix<double, ColMajor, int> >()));
    CALL_SUBTEST_7((check_random_setter_target_modes<SparseMatrix<double, RowMajor, int> >()));
    CALL_SUBTEST_7((check_random_setter_target_modes<SparseMatrix<std::complex<float>, ColMajor, long int> >()));
    CALL_SUBTEST_7((check_random_setter_target_modes<SparseMatrix<float, RowMajor, long int> >()));

    TEST_SET_BUT_UNUSED_VARIABLE(s);
  }
}
