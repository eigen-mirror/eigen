// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2008 Daniel Gomez Ferro <dgomezferro@gmail.com>
// Copyright (C) 2013 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPARSE_TEST_INCLUDED_FROM_SPARSE_EXTRA
static long g_realloc_count = 0;
#define EIGEN_SPARSE_COMPRESSED_STORAGE_REALLOCATE_PLUGIN g_realloc_count++;

static long g_dense_op_sparse_count = 0;
#define EIGEN_SPARSE_ASSIGNMENT_FROM_DENSE_OP_SPARSE_PLUGIN g_dense_op_sparse_count++;
#define EIGEN_SPARSE_ASSIGNMENT_FROM_SPARSE_ADD_DENSE_PLUGIN g_dense_op_sparse_count += 10;
#define EIGEN_SPARSE_ASSIGNMENT_FROM_SPARSE_SUB_DENSE_PLUGIN g_dense_op_sparse_count += 20;
#endif

#include "sparse.h"
#include "sparse_test_helpers.h"
#include "MovableScalar.h"

template <typename SparseMatrixType>
void sparse_basic(const SparseMatrixType& ref) {
  using StorageIndex = typename SparseMatrixType::StorageIndex;
  using Vector2 = Matrix<StorageIndex, 2, 1>;

  const Index rows = ref.rows();
  const Index cols = ref.cols();
  const Index inner = ref.innerSize();
  const Index outer = ref.outerSize();

  using Scalar = typename SparseMatrixType::Scalar;
  using RealScalar = typename SparseMatrixType::RealScalar;
  enum { Flags = SparseMatrixType::Flags };

  double density = (std::max)(8. / (rows * cols), 0.01);
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;
  using CompatibleDenseMatrix = Matrix<Scalar, Dynamic, Dynamic, SparseMatrixType::IsRowMajor ? RowMajor : ColMajor>;
  Scalar eps = Scalar(1e-6);

  Scalar s1 = internal::random<Scalar>();
  {
    SparseMatrixType m(rows, cols);
    DenseMatrix refMat = DenseMatrix::Zero(rows, cols);
    DenseVector vec1 = DenseVector::Random(rows);

    std::vector<Vector2> zeroCoords;
    std::vector<Vector2> nonzeroCoords;
    initSparse<Scalar>(density, refMat, m, 0, &zeroCoords, &nonzeroCoords);

    // test coeff and coeffRef
    for (std::size_t i = 0; i < zeroCoords.size(); ++i) {
      VERIFY_IS_MUCH_SMALLER_THAN(m.coeff(zeroCoords[i].x(), zeroCoords[i].y()), eps);
      if (std::is_same<SparseMatrixType, SparseMatrix<Scalar, Flags>>::value)
        VERIFY_RAISES_ASSERT(m.coeffRef(zeroCoords[i].x(), zeroCoords[i].y()) = 5);
    }
    VERIFY_IS_APPROX(m, refMat);

    if (!nonzeroCoords.empty()) {
      m.coeffRef(nonzeroCoords[0].x(), nonzeroCoords[0].y()) = Scalar(5);
      refMat.coeffRef(nonzeroCoords[0].x(), nonzeroCoords[0].y()) = Scalar(5);
    }

    VERIFY_IS_APPROX(m, refMat);

    // test assertion
    VERIFY_RAISES_ASSERT(m.coeffRef(-1, 1) = 0);
    VERIFY_RAISES_ASSERT(m.coeffRef(0, m.cols()) = 0);
  }

  // test insert (inner random)
  {
    DenseMatrix m1(rows, cols);
    m1.setZero();
    SparseMatrixType m2(rows, cols);
    bool call_reserve = internal::random<int>() % 2;
    Index nnz = internal::random<int>(1, int(rows) / 2);
    if (call_reserve) {
      if (internal::random<int>() % 2)
        m2.reserve(VectorXi::Constant(m2.outerSize(), int(nnz)));
      else
        m2.reserve(m2.outerSize() * nnz);
    }
    g_realloc_count = 0;
    for (Index j = 0; j < cols; ++j) {
      for (Index k = 0; k < nnz; ++k) {
        Index i = internal::random<Index>(0, rows - 1);
        if (m1.coeff(i, j) == Scalar(0)) {
          Scalar v = internal::random<Scalar>();
          if (v == Scalar(0)) v = Scalar(1);
          m1(i, j) = v;
          m2.insert(i, j) = v;
        }
      }
    }

    if (call_reserve && !SparseMatrixType::IsRowMajor) {
      VERIFY(g_realloc_count == 0);
    }

    VERIFY_IS_APPROX(m2, m1);
  }

  // test insert (fully random)
  {
    DenseMatrix m1(rows, cols);
    m1.setZero();
    SparseMatrixType m2(rows, cols);
    if (internal::random<int>() % 2) m2.reserve(VectorXi::Constant(m2.outerSize(), 2));
    for (int k = 0; k < rows * cols; ++k) {
      Index i = internal::random<Index>(0, rows - 1);
      Index j = internal::random<Index>(0, cols - 1);
      if ((m1.coeff(i, j) == Scalar(0)) && (internal::random<int>() % 2)) {
        Scalar v = internal::random<Scalar>();
        if (v == Scalar(0)) v = Scalar(1);
        m1(i, j) = v;
        m2.insert(i, j) = v;
      } else {
        Scalar v = internal::random<Scalar>();
        if (v == Scalar(0)) v = Scalar(1);
        m1(i, j) = v;
        m2.coeffRef(i, j) = v;
      }
    }
    VERIFY_IS_APPROX(m2, m1);
  }

  // test insert (un-compressed)
  for (int mode = 0; mode < 4; ++mode) {
    DenseMatrix m1(rows, cols);
    m1.setZero();
    SparseMatrixType m2(rows, cols);
    VectorXi r(VectorXi::Constant(m2.outerSize(),
                                  ((mode % 2) == 0) ? int(m2.innerSize()) : std::max<int>(1, int(m2.innerSize()) / 8)));
    m2.reserve(r);
    for (Index k = 0; k < rows * cols; ++k) {
      Index i = internal::random<Index>(0, rows - 1);
      Index j = internal::random<Index>(0, cols - 1);
      if (m1.coeff(i, j) == Scalar(0)) {
        Scalar v = internal::random<Scalar>();
        if (v == Scalar(0)) v = Scalar(1);
        m1(i, j) = v;
        m2.insert(i, j) = v;
      }
      if (mode == 3) m2.reserve(r);
    }
    if (internal::random<int>() % 2) m2.makeCompressed();
    VERIFY_IS_APPROX(m2, m1);
  }

  // test removeOuterVectors / insertEmptyOuterVectors
  {
    for (int mode = 0; mode < 4; mode++) {
      CompatibleDenseMatrix m1(rows, cols);
      m1.setZero();
      SparseMatrixType m2(rows, cols);
      Vector<Index, Dynamic> reserveSizes(outer);
      for (Index j = 0; j < outer; j++) reserveSizes(j) = internal::random<Index>(1, inner - 1);
      m2.reserve(reserveSizes);
      for (Index j = 0; j < outer; j++) {
        Index i = internal::random<Index>(0, inner - 1);
        Scalar val = internal::random<Scalar>();
        m1.coeffRefByOuterInner(j, i) = val;
        m2.insertByOuterInner(j, i) = val;
      }
      if (mode % 2 == 0) m2.makeCompressed();

      if (mode < 2) {
        Index num = internal::random<Index>(0, outer - 1);
        Index start = internal::random<Index>(0, outer - num);

        Index newRows = SparseMatrixType::IsRowMajor ? rows - num : rows;
        Index newCols = SparseMatrixType::IsRowMajor ? cols : cols - num;

        CompatibleDenseMatrix m3(newRows, newCols);
        m3.setConstant(Scalar(NumTraits<RealScalar>::quiet_NaN()));

        if (SparseMatrixType::IsRowMajor) {
          m3.topRows(start) = m1.topRows(start);
          m3.bottomRows(newRows - start) = m1.bottomRows(newRows - start);
        } else {
          m3.leftCols(start) = m1.leftCols(start);
          m3.rightCols(newCols - start) = m1.rightCols(newCols - start);
        }

        SparseMatrixType m4 = m2;
        m4.removeOuterVectors(start, num);

        VERIFY_IS_CWISE_EQUAL(m3, m4.toDense());

        SparseMatrixType m5 = m2;
        m5.removeOuterVectors(0, outer);
        VERIFY_IS_EQUAL(m5.outerSize(), 0);
        VERIFY_IS_EQUAL(m5.nonZeros(), 0);
      } else {
        Index num = internal::random<Index>(0, outer - 1);
        Index start = internal::random<Index>(0, outer - 1);

        Index newRows = SparseMatrixType::IsRowMajor ? rows + num : rows;
        Index newCols = SparseMatrixType::IsRowMajor ? cols : cols + num;

        CompatibleDenseMatrix m3(newRows, newCols);
        m3.setConstant(Scalar(NumTraits<RealScalar>::quiet_NaN()));

        if (SparseMatrixType::IsRowMajor) {
          m3.topRows(start) = m1.topRows(start);
          m3.middleRows(start, num).setZero();
          m3.bottomRows(rows - start) = m1.bottomRows(rows - start);
        } else {
          m3.leftCols(start) = m1.leftCols(start);
          m3.middleCols(start, num).setZero();
          m3.rightCols(cols - start) = m1.rightCols(cols - start);
        }

        SparseMatrixType m4 = m2;
        m4.insertEmptyOuterVectors(start, num);

        VERIFY_IS_CWISE_EQUAL(m3, m4.toDense());
      }
    }
  }

  // test sort
  if (inner > 1) {
    bool StorageOrdersMatch = int(DenseMatrix::IsRowMajor) == int(SparseMatrixType::IsRowMajor);
    DenseMatrix m1(rows, cols);
    m1.setZero();
    SparseMatrixType m2(rows, cols);
    // generate random inner indices with no repeats
    Vector<Index, Dynamic> innerIndices(inner);
    innerIndices.setLinSpaced(inner, 0, inner - 1);
    std::random_device rd;
    std::mt19937 g(rd());
    for (Index j = 0; j < outer; j++) {
      std::shuffle(innerIndices.begin(), innerIndices.end(), g);
      Index nzj = internal::random<Index>(2, inner / 2);
      for (Index k = 0; k < nzj; k++) {
        Index i = innerIndices[k];
        Scalar val = internal::random<Scalar>();
        m1.coeffRefByOuterInner(StorageOrdersMatch ? j : i, StorageOrdersMatch ? i : j) = val;
        m2.insertByOuterInner(j, i) = val;
      }
    }

    VERIFY_IS_APPROX(m2, m1);
    // sort wrt greater
    m2.template sortInnerIndices<std::greater<>>();
    // verify that all inner vectors are not sorted wrt less
    VERIFY_IS_EQUAL(m2.template innerIndicesAreSorted<std::less<>>(), 0);
    // verify that all inner vectors are sorted wrt greater
    VERIFY_IS_EQUAL(m2.template innerIndicesAreSorted<std::greater<>>(), m2.outerSize());
    // verify that sort does not change evaluation
    VERIFY_IS_APPROX(m2, m1);
    // sort wrt less
    m2.template sortInnerIndices<std::less<>>();
    // verify that all inner vectors are sorted wrt less
    VERIFY_IS_EQUAL(m2.template innerIndicesAreSorted<std::less<>>(), m2.outerSize());
    // verify that all inner vectors are not sorted wrt greater
    VERIFY_IS_EQUAL(m2.template innerIndicesAreSorted<std::greater<>>(), 0);
    // verify that sort does not change evaluation
    VERIFY_IS_APPROX(m2, m1);

    m2.makeCompressed();
    // sort wrt greater
    m2.template sortInnerIndices<std::greater<>>();
    // verify that all inner vectors are not sorted wrt less
    VERIFY_IS_EQUAL(m2.template innerIndicesAreSorted<std::less<>>(), 0);
    // verify that all inner vectors are sorted wrt greater
    VERIFY_IS_EQUAL(m2.template innerIndicesAreSorted<std::greater<>>(), m2.outerSize());
    // verify that sort does not change evaluation
    VERIFY_IS_APPROX(m2, m1);
    // sort wrt less
    m2.template sortInnerIndices<std::less<>>();
    // verify that all inner vectors are sorted wrt less
    VERIFY_IS_EQUAL(m2.template innerIndicesAreSorted<std::less<>>(), m2.outerSize());
    // verify that all inner vectors are not sorted wrt greater
    VERIFY_IS_EQUAL(m2.template innerIndicesAreSorted<std::greater<>>(), 0);
    // verify that sort does not change evaluation
    VERIFY_IS_APPROX(m2, m1);
  }

  // test basic computations
  {
    DenseMatrix refM1 = DenseMatrix::Zero(rows, cols);
    DenseMatrix refM2 = DenseMatrix::Zero(rows, cols);
    DenseMatrix refM3 = DenseMatrix::Zero(rows, cols);
    DenseMatrix refM4 = DenseMatrix::Zero(rows, cols);
    SparseMatrixType m1(rows, cols);
    SparseMatrixType m2(rows, cols);
    SparseMatrixType m3(rows, cols);
    SparseMatrixType m4(rows, cols);
    initSparse<Scalar>(density, refM1, m1);
    initSparse<Scalar>(density, refM2, m2);
    initSparse<Scalar>(density, refM3, m3);
    initSparse<Scalar>(density, refM4, m4);

    if (internal::random<bool>()) m1.makeCompressed();

    Index m1_nnz = m1.nonZeros();

    VERIFY_IS_APPROX(m1 * s1, refM1 * s1);
    VERIFY_IS_APPROX(m1 + m2, refM1 + refM2);
    VERIFY_IS_APPROX(m1 + m2 + m3, refM1 + refM2 + refM3);
    VERIFY_IS_APPROX(m3.cwiseProduct(m1 + m2), refM3.cwiseProduct(refM1 + refM2));
    VERIFY_IS_APPROX(m1 * s1 - m2, refM1 * s1 - refM2);
    VERIFY_IS_APPROX(m4 = m1 / s1, refM1 / s1);
    VERIFY_IS_EQUAL(m4.nonZeros(), m1_nnz);

    if (SparseMatrixType::IsRowMajor)
      VERIFY_IS_APPROX(m1.innerVector(0).dot(refM2.row(0)), refM1.row(0).dot(refM2.row(0)));
    else
      VERIFY_IS_APPROX(m1.innerVector(0).dot(refM2.col(0)), refM1.col(0).dot(refM2.col(0)));

    DenseVector rv = DenseVector::Random(m1.cols());
    DenseVector cv = DenseVector::Random(m1.rows());
    Index r = internal::random<Index>(0, m1.rows() - 2);
    Index c = internal::random<Index>(0, m1.cols() - 1);
    VERIFY_IS_APPROX((m1.template block<1, Dynamic>(r, 0, 1, m1.cols()).dot(rv)), refM1.row(r).dot(rv));
    VERIFY_IS_APPROX(m1.row(r).dot(rv), refM1.row(r).dot(rv));
    VERIFY_IS_APPROX(m1.col(c).dot(cv), refM1.col(c).dot(cv));

    VERIFY_IS_APPROX(m1.conjugate(), refM1.conjugate());
    VERIFY_IS_APPROX(m1.real(), refM1.real());

    refM4.setRandom();
    // sparse cwise* dense
    VERIFY_IS_APPROX(m3.cwiseProduct(refM4), refM3.cwiseProduct(refM4));
    // dense cwise* sparse
    VERIFY_IS_APPROX(refM4.cwiseProduct(m3), refM4.cwiseProduct(refM3));
    //     VERIFY_IS_APPROX(m3.cwise()/refM4, refM3.cwise()/refM4);

    // mixed sparse-dense
    VERIFY_IS_APPROX(refM4 + m3, refM4 + refM3);
    VERIFY_IS_APPROX(m3 + refM4, refM3 + refM4);
    VERIFY_IS_APPROX(refM4 - m3, refM4 - refM3);
    VERIFY_IS_APPROX(m3 - refM4, refM3 - refM4);
    VERIFY_IS_APPROX((RealScalar(0.5) * refM4 + RealScalar(0.5) * m3).eval(),
                     RealScalar(0.5) * refM4 + RealScalar(0.5) * refM3);
    VERIFY_IS_APPROX((RealScalar(0.5) * refM4 + m3 * RealScalar(0.5)).eval(),
                     RealScalar(0.5) * refM4 + RealScalar(0.5) * refM3);
    VERIFY_IS_APPROX((RealScalar(0.5) * refM4 + m3.cwiseProduct(m3)).eval(),
                     RealScalar(0.5) * refM4 + refM3.cwiseProduct(refM3));

    VERIFY_IS_APPROX((RealScalar(0.5) * refM4 + (m3 + m3)).eval(), RealScalar(0.5) * refM4 + (refM3 + refM3));
    VERIFY_IS_APPROX(((refM3 + m3) + RealScalar(0.5) * m3).eval(), RealScalar(0.5) * refM3 + (refM3 + refM3));
    VERIFY_IS_APPROX((RealScalar(0.5) * refM4 + (refM3 + m3)).eval(), RealScalar(0.5) * refM4 + (refM3 + refM3));
    VERIFY_IS_APPROX((RealScalar(0.5) * refM4 + (m3 + refM3)).eval(), RealScalar(0.5) * refM4 + (refM3 + refM3));

    VERIFY_IS_APPROX(m1.sum(), refM1.sum());

    m4 = m1;
    refM4 = m4;

    VERIFY_IS_APPROX(m1 *= s1, refM1 *= s1);
    VERIFY_IS_EQUAL(m1.nonZeros(), m1_nnz);
    VERIFY_IS_APPROX(m1 /= s1, refM1 /= s1);
    VERIFY_IS_EQUAL(m1.nonZeros(), m1_nnz);

    VERIFY_IS_APPROX(m1 += m2, refM1 += refM2);
    VERIFY_IS_APPROX(m1 -= m2, refM1 -= refM2);

    refM3 = refM1;

    VERIFY_IS_APPROX(refM1 += m2, refM3 += refM2);
    VERIFY_IS_APPROX(refM1 -= m2, refM3 -= refM2);

    g_dense_op_sparse_count = 0;
    VERIFY_IS_APPROX(refM1 = m2 + refM4, refM3 = refM2 + refM4);
    VERIFY_IS_EQUAL(g_dense_op_sparse_count, 10);
    g_dense_op_sparse_count = 0;
    VERIFY_IS_APPROX(refM1 += m2 + refM4, refM3 += refM2 + refM4);
    VERIFY_IS_EQUAL(g_dense_op_sparse_count, 1);
    g_dense_op_sparse_count = 0;
    VERIFY_IS_APPROX(refM1 -= m2 + refM4, refM3 -= refM2 + refM4);
    VERIFY_IS_EQUAL(g_dense_op_sparse_count, 1);
    g_dense_op_sparse_count = 0;
    VERIFY_IS_APPROX(refM1 = refM4 + m2, refM3 = refM2 + refM4);
    VERIFY_IS_EQUAL(g_dense_op_sparse_count, 1);
    g_dense_op_sparse_count = 0;
    VERIFY_IS_APPROX(refM1 += refM4 + m2, refM3 += refM2 + refM4);
    VERIFY_IS_EQUAL(g_dense_op_sparse_count, 1);
    g_dense_op_sparse_count = 0;
    VERIFY_IS_APPROX(refM1 -= refM4 + m2, refM3 -= refM2 + refM4);
    VERIFY_IS_EQUAL(g_dense_op_sparse_count, 1);

    g_dense_op_sparse_count = 0;
    VERIFY_IS_APPROX(refM1 = m2 - refM4, refM3 = refM2 - refM4);
    VERIFY_IS_EQUAL(g_dense_op_sparse_count, 20);
    g_dense_op_sparse_count = 0;
    VERIFY_IS_APPROX(refM1 += m2 - refM4, refM3 += refM2 - refM4);
    VERIFY_IS_EQUAL(g_dense_op_sparse_count, 1);
    g_dense_op_sparse_count = 0;
    VERIFY_IS_APPROX(refM1 -= m2 - refM4, refM3 -= refM2 - refM4);
    VERIFY_IS_EQUAL(g_dense_op_sparse_count, 1);
    g_dense_op_sparse_count = 0;
    VERIFY_IS_APPROX(refM1 = refM4 - m2, refM3 = refM4 - refM2);
    VERIFY_IS_EQUAL(g_dense_op_sparse_count, 1);
    g_dense_op_sparse_count = 0;
    VERIFY_IS_APPROX(refM1 += refM4 - m2, refM3 += refM4 - refM2);
    VERIFY_IS_EQUAL(g_dense_op_sparse_count, 1);
    g_dense_op_sparse_count = 0;
    VERIFY_IS_APPROX(refM1 -= refM4 - m2, refM3 -= refM4 - refM2);
    VERIFY_IS_EQUAL(g_dense_op_sparse_count, 1);
    refM3 = m3;

    if (rows >= 2 && cols >= 2) {
      VERIFY_RAISES_ASSERT(m1 += m1.innerVector(0));
      VERIFY_RAISES_ASSERT(m1 -= m1.innerVector(0));
      VERIFY_RAISES_ASSERT(refM1 -= m1.innerVector(0));
      VERIFY_RAISES_ASSERT(refM1 += m1.innerVector(0));
    }
    m1 = m4;
    refM1 = refM4;

    // test aliasing
    VERIFY_IS_APPROX((m1 = -m1), (refM1 = -refM1));
    VERIFY_IS_EQUAL(m1.nonZeros(), m1_nnz);
    m1 = m4;
    refM1 = refM4;
    VERIFY_IS_APPROX((m1 = m1.transpose()), (refM1 = refM1.transpose().eval()));
    VERIFY_IS_EQUAL(m1.nonZeros(), m1_nnz);
    m1 = m4;
    refM1 = refM4;
    VERIFY_IS_APPROX((m1 = -m1.transpose()), (refM1 = -refM1.transpose().eval()));
    VERIFY_IS_EQUAL(m1.nonZeros(), m1_nnz);
    m1 = m4;
    refM1 = refM4;
    VERIFY_IS_APPROX((m1 += -m1), (refM1 += -refM1));
    VERIFY_IS_EQUAL(m1.nonZeros(), m1_nnz);
    m1 = m4;
    refM1 = refM4;

    if (m1.isCompressed()) {
      VERIFY_IS_APPROX(m1.coeffs().sum(), m1.sum());
      m1.coeffs() += s1;
      for (Index j = 0; j < m1.outerSize(); ++j)
        for (typename SparseMatrixType::InnerIterator it(m1, j); it; ++it) refM1(it.row(), it.col()) += s1;
      VERIFY_IS_APPROX(m1, refM1);
    }

    // and/or
    {
      using SpBool = SparseMatrix<bool, SparseMatrixType::Options, typename SparseMatrixType::StorageIndex>;
      SpBool mb1 = m1.real().template cast<bool>();
      SpBool mb2 = m2.real().template cast<bool>();
      VERIFY_IS_EQUAL(mb1.template cast<int>().sum(), refM1.real().template cast<bool>().count());
      VERIFY_IS_EQUAL((mb1 && mb2).template cast<int>().sum(),
                      (refM1.real().template cast<bool>() && refM2.real().template cast<bool>()).count());
      VERIFY_IS_EQUAL((mb1 || mb2).template cast<int>().sum(),
                      (refM1.real().template cast<bool>() || refM2.real().template cast<bool>()).count());
      SpBool mb3 = mb1 && mb2;
      if (mb1.coeffs().all() && mb2.coeffs().all()) {
        VERIFY_IS_EQUAL(mb3.nonZeros(),
                        (refM1.real().template cast<bool>() && refM2.real().template cast<bool>()).count());
      }
    }
  }

  // test reverse iterators
  {
    DenseMatrix refMat2 = DenseMatrix::Zero(rows, cols);
    SparseMatrixType m2(rows, cols);
    initSparse<Scalar>(density, refMat2, m2);
    std::vector<Scalar> ref_value(m2.innerSize());
    std::vector<Index> ref_index(m2.innerSize());
    if (internal::random<bool>()) m2.makeCompressed();
    for (Index j = 0; j < m2.outerSize(); ++j) {
      Index count_forward = 0;

      for (typename SparseMatrixType::InnerIterator it(m2, j); it; ++it) {
        ref_value[ref_value.size() - 1 - count_forward] = it.value();
        ref_index[ref_index.size() - 1 - count_forward] = it.index();
        count_forward++;
      }
      Index count_reverse = 0;
      for (typename SparseMatrixType::ReverseInnerIterator it(m2, j); it; --it) {
        VERIFY_IS_APPROX(std::abs(ref_value[ref_value.size() - count_forward + count_reverse]) + 1,
                         std::abs(it.value()) + 1);
        VERIFY_IS_EQUAL(ref_index[ref_index.size() - count_forward + count_reverse], it.index());
        count_reverse++;
      }
      VERIFY_IS_EQUAL(count_forward, count_reverse);
    }
  }

  // test transpose
  {
    DenseMatrix refMat2 = DenseMatrix::Zero(rows, cols);
    SparseMatrixType m2(rows, cols);
    initSparse<Scalar>(density, refMat2, m2);
    VERIFY_IS_APPROX(m2.transpose().eval(), refMat2.transpose().eval());
    VERIFY_IS_APPROX(m2.transpose(), refMat2.transpose());

    VERIFY_IS_APPROX(SparseMatrixType(m2.adjoint()), refMat2.adjoint());

    // check isApprox handles opposite storage order
    typename Transpose<SparseMatrixType>::PlainObject m3(m2);
    VERIFY(m2.isApprox(m3));
  }

  // test prune
  {
    SparseMatrixType m2(rows, cols);
    DenseMatrix refM2(rows, cols);
    refM2.setZero();
    int countFalseNonZero = 0;
    int countTrueNonZero = 0;
    m2.reserve(VectorXi::Constant(m2.outerSize(), int(m2.innerSize())));
    for (Index j = 0; j < m2.cols(); ++j) {
      for (Index i = 0; i < m2.rows(); ++i) {
        float x = internal::random<float>(0, 1);
        if (x < 0.1f) {
          // do nothing
        } else if (x < 0.5f) {
          countFalseNonZero++;
          m2.insert(i, j) = Scalar(0);
        } else {
          countTrueNonZero++;
          m2.insert(i, j) = Scalar(1);
          refM2(i, j) = Scalar(1);
        }
      }
    }
    if (internal::random<bool>()) m2.makeCompressed();
    VERIFY(countFalseNonZero + countTrueNonZero == m2.nonZeros());
    if (countTrueNonZero > 0) VERIFY_IS_APPROX(m2, refM2);
    m2.prune(Scalar(1));
    VERIFY(countTrueNonZero == m2.nonZeros());
    VERIFY_IS_APPROX(m2, refM2);
  }

  // test setFromTriplets / insertFromTriplets
  {
    using TripletType = Triplet<Scalar, StorageIndex>;
    Index ntriplets = rows * cols;

    std::vector<TripletType> triplets;

    triplets.reserve(ntriplets);
    DenseMatrix refMat_sum = DenseMatrix::Zero(rows, cols);

    for (Index i = 0; i < ntriplets; ++i) {
      StorageIndex r = internal::random<StorageIndex>(0, StorageIndex(rows - 1));
      StorageIndex c = internal::random<StorageIndex>(0, StorageIndex(cols - 1));
      Scalar v = internal::random<Scalar>();
      triplets.push_back(TripletType(r, c, v));
      refMat_sum(r, c) += v;
    }
    triplets.emplace_back(0, 0, Scalar(0));
    triplets.emplace_back(0, 0, Scalar(2));
    refMat_sum(0, 0) += Scalar(2);

    std::vector<TripletType> moreTriplets;
    moreTriplets.reserve(ntriplets);
    DenseMatrix refMat_sum_more = refMat_sum;

    for (Index i = 0; i < ntriplets; ++i) {
      StorageIndex r = internal::random<StorageIndex>(0, StorageIndex(rows - 1));
      StorageIndex c = internal::random<StorageIndex>(0, StorageIndex(cols - 1));
      Scalar v = internal::random<Scalar>();
      moreTriplets.push_back(TripletType(r, c, v));
      refMat_sum_more(r, c) += v;
    }
    moreTriplets.emplace_back(0, 0, Scalar(0));
    moreTriplets.emplace_back(0, 0, Scalar(3));
    refMat_sum_more(0, 0) += Scalar(3);

    // setFromTriplets sorts internally by (outer, inner), so non-commutative
    // reductions (std::multiplies, "last wins") depend on sorted order.
    // Compute these references from sorted triplets to match.
    struct triplet_comp {
      inline bool operator()(const TripletType& a, const TripletType& b) {
        return SparseMatrixType::IsRowMajor ? ((a.row() != b.row()) ? (a.row() < b.row()) : (a.col() < b.col()))
                                            : ((a.col() != b.col()) ? (a.col() < b.col()) : (a.row() < b.row()));
      }
    };

    DenseMatrix refMat_prod = DenseMatrix::Zero(rows, cols);
    DenseMatrix refMat_last = DenseMatrix::Zero(rows, cols);
    // A stored zero still participates in duplicate reductions.
    Matrix<bool, Dynamic, Dynamic> visited = Matrix<bool, Dynamic, Dynamic>::Constant(rows, cols, false);
    {
      auto sorted = triplets;
      std::stable_sort(sorted.begin(), sorted.end(), triplet_comp());
      for (const auto& t : sorted) {
        if (!visited(t.row(), t.col()))
          refMat_prod(t.row(), t.col()) = t.value();
        else
          refMat_prod(t.row(), t.col()) *= t.value();
        visited(t.row(), t.col()) = true;
        refMat_last(t.row(), t.col()) = t.value();
      }
    }

    DenseMatrix refMat_prod_more = refMat_prod;
    DenseMatrix refMat_last_more = refMat_last;
    {
      auto sorted = moreTriplets;
      std::stable_sort(sorted.begin(), sorted.end(), triplet_comp());
      for (const auto& t : sorted) {
        if (!visited(t.row(), t.col()))
          refMat_prod_more(t.row(), t.col()) = t.value();
        else
          refMat_prod_more(t.row(), t.col()) *= t.value();
        visited(t.row(), t.col()) = true;
        refMat_last_more(t.row(), t.col()) = t.value();
      }
    }

    SparseMatrixType m(rows, cols);

    // test setFromTriplets / insertFromTriplets

    m.setFromTriplets(triplets.begin(), triplets.end());
    VERIFY_IS_APPROX(m, refMat_sum);
    VERIFY_IS_EQUAL(m.innerIndicesAreSorted(), m.outerSize());
    VERIFY(m.isCompressed());
    m.insertFromTriplets(moreTriplets.begin(), moreTriplets.end());
    VERIFY_IS_APPROX(m, refMat_sum_more);
    VERIFY_IS_EQUAL(m.innerIndicesAreSorted(), m.outerSize());

    m.setFromTriplets(triplets.begin(), triplets.end(), std::multiplies<Scalar>());
    VERIFY_IS_APPROX(m, refMat_prod);
    VERIFY_IS_EQUAL(m.innerIndicesAreSorted(), m.outerSize());
    VERIFY(m.isCompressed());
    m.insertFromTriplets(moreTriplets.begin(), moreTriplets.end(), std::multiplies<Scalar>());
    VERIFY_IS_APPROX(m, refMat_prod_more);
    VERIFY_IS_EQUAL(m.innerIndicesAreSorted(), m.outerSize());

    m.setFromTriplets(triplets.begin(), triplets.end(), [](Scalar, Scalar b) { return b; });
    VERIFY_IS_APPROX(m, refMat_last);
    VERIFY_IS_EQUAL(m.innerIndicesAreSorted(), m.outerSize());
    m.setFromTriplets(triplets.begin(), triplets.end(), [](Scalar, Scalar b) { return b; });
    VERIFY(m.isCompressed());
    m.insertFromTriplets(moreTriplets.begin(), moreTriplets.end(), [](Scalar, Scalar b) { return b; });
    VERIFY_IS_APPROX(m, refMat_last_more);
    VERIFY_IS_EQUAL(m.innerIndicesAreSorted(), m.outerSize());

    // insert into an uncompressed matrix

    VectorXi reserveSizes(m.outerSize());
    for (Index i = 0; i < m.outerSize(); i++) reserveSizes[i] = internal::random<int>(1, 7);

    m.setFromTriplets(triplets.begin(), triplets.end());
    m.reserve(reserveSizes);
    VERIFY(!m.isCompressed());
    m.insertFromTriplets(moreTriplets.begin(), moreTriplets.end());
    VERIFY_IS_APPROX(m, refMat_sum_more);
    VERIFY_IS_EQUAL(m.innerIndicesAreSorted(), m.outerSize());

    m.setFromTriplets(triplets.begin(), triplets.end(), std::multiplies<Scalar>());
    m.reserve(reserveSizes);
    VERIFY(!m.isCompressed());
    m.insertFromTriplets(moreTriplets.begin(), moreTriplets.end(), std::multiplies<Scalar>());
    VERIFY_IS_APPROX(m, refMat_prod_more);
    VERIFY_IS_EQUAL(m.innerIndicesAreSorted(), m.outerSize());

    m.setFromTriplets(triplets.begin(), triplets.end(), [](Scalar, Scalar b) { return b; });
    m.reserve(reserveSizes);
    VERIFY(!m.isCompressed());
    m.insertFromTriplets(moreTriplets.begin(), moreTriplets.end(), [](Scalar, Scalar b) { return b; });
    VERIFY_IS_APPROX(m, refMat_last_more);
    VERIFY_IS_EQUAL(m.innerIndicesAreSorted(), m.outerSize());

    // test setFromSortedTriplets / insertFromSortedTriplets

    // stable_sort is only necessary when the reduction functor is dependent on the order of the triplets
    // this is the case with refMat_last
    // for most cases, std::sort is sufficient and preferred

    std::stable_sort(triplets.begin(), triplets.end(), triplet_comp());
    std::stable_sort(moreTriplets.begin(), moreTriplets.end(), triplet_comp());

    m.setFromSortedTriplets(triplets.begin(), triplets.end());
    VERIFY_IS_APPROX(m, refMat_sum);
    VERIFY_IS_EQUAL(m.innerIndicesAreSorted(), m.outerSize());
    VERIFY(m.isCompressed());
    m.insertFromSortedTriplets(moreTriplets.begin(), moreTriplets.end());
    VERIFY_IS_APPROX(m, refMat_sum_more);
    VERIFY_IS_EQUAL(m.innerIndicesAreSorted(), m.outerSize());

    m.setFromSortedTriplets(triplets.begin(), triplets.end(), std::multiplies<Scalar>());
    VERIFY_IS_APPROX(m, refMat_prod);
    VERIFY_IS_EQUAL(m.innerIndicesAreSorted(), m.outerSize());
    VERIFY(m.isCompressed());
    m.insertFromSortedTriplets(moreTriplets.begin(), moreTriplets.end(), std::multiplies<Scalar>());
    VERIFY_IS_APPROX(m, refMat_prod_more);
    VERIFY_IS_EQUAL(m.innerIndicesAreSorted(), m.outerSize());

    m.setFromSortedTriplets(triplets.begin(), triplets.end(), [](Scalar, Scalar b) { return b; });
    VERIFY_IS_APPROX(m, refMat_last);
    VERIFY_IS_EQUAL(m.innerIndicesAreSorted(), m.outerSize());
    VERIFY(m.isCompressed());
    m.insertFromSortedTriplets(moreTriplets.begin(), moreTriplets.end(), [](Scalar, Scalar b) { return b; });
    VERIFY_IS_APPROX(m, refMat_last_more);
    VERIFY_IS_EQUAL(m.innerIndicesAreSorted(), m.outerSize());

    // insert into an uncompressed matrix

    m.setFromSortedTriplets(triplets.begin(), triplets.end());
    m.reserve(reserveSizes);
    VERIFY(!m.isCompressed());
    m.insertFromSortedTriplets(moreTriplets.begin(), moreTriplets.end());
    VERIFY_IS_APPROX(m, refMat_sum_more);
    VERIFY_IS_EQUAL(m.innerIndicesAreSorted(), m.outerSize());

    m.setFromSortedTriplets(triplets.begin(), triplets.end(), std::multiplies<Scalar>());
    m.reserve(reserveSizes);
    VERIFY(!m.isCompressed());
    m.insertFromSortedTriplets(moreTriplets.begin(), moreTriplets.end(), std::multiplies<Scalar>());
    VERIFY_IS_APPROX(m, refMat_prod_more);
    VERIFY_IS_EQUAL(m.innerIndicesAreSorted(), m.outerSize());

    m.setFromSortedTriplets(triplets.begin(), triplets.end(), [](Scalar, Scalar b) { return b; });
    m.reserve(reserveSizes);
    VERIFY(!m.isCompressed());
    m.insertFromSortedTriplets(moreTriplets.begin(), moreTriplets.end(), [](Scalar, Scalar b) { return b; });
    VERIFY_IS_APPROX(m, refMat_last_more);
    VERIFY_IS_EQUAL(m.innerIndicesAreSorted(), m.outerSize());
  }

  // test Map
  {
    DenseMatrix refMat2(rows, cols), refMat3(rows, cols);
    SparseMatrixType m2(rows, cols), m3(rows, cols);
    initSparse<Scalar>(density, refMat2, m2);
    initSparse<Scalar>(density, refMat3, m3);
    {
      auto mapMat2 = make_map(m2);
      auto mapMat3 = make_map(m3);
      VERIFY_IS_APPROX(mapMat2 + mapMat3, refMat2 + refMat3);
      VERIFY_IS_APPROX(mapMat2 + mapMat3, refMat2 + refMat3);
    }

    Index i = internal::random<Index>(0, rows - 1);
    Index j = internal::random<Index>(0, cols - 1);
    m2.coeffRef(i, j) = 123;
    if (internal::random<bool>()) m2.makeCompressed();
    auto mapMat2 = make_map(m2);
    VERIFY_IS_EQUAL(m2.coeff(i, j), Scalar(123));
    VERIFY_IS_EQUAL(mapMat2.coeff(i, j), Scalar(123));
    mapMat2.coeffRef(i, j) = -123;
    VERIFY_IS_EQUAL(m2.coeff(i, j), Scalar(-123));
  }

  // test triangularView
  {
    DenseMatrix refMat2(rows, cols), refMat3(rows, cols);
    SparseMatrixType m2(rows, cols), m3(rows, cols);
    initSparse<Scalar>(density, refMat2, m2);
    refMat3 = refMat2.template triangularView<Lower>();
    m3 = m2.template triangularView<Lower>();
    VERIFY_IS_APPROX(m3, refMat3);

    refMat3 = refMat2.template triangularView<Upper>();
    m3 = m2.template triangularView<Upper>();
    VERIFY_IS_APPROX(m3, refMat3);

    {
      refMat3 = refMat2.template triangularView<UnitUpper>();
      m3 = m2.template triangularView<UnitUpper>();
      VERIFY_IS_APPROX(m3, refMat3);

      refMat3 = refMat2.template triangularView<UnitLower>();
      m3 = m2.template triangularView<UnitLower>();
      VERIFY_IS_APPROX(m3, refMat3);
    }

    refMat3 = refMat2.template triangularView<StrictlyUpper>();
    m3 = m2.template triangularView<StrictlyUpper>();
    VERIFY_IS_APPROX(m3, refMat3);

    refMat3 = refMat2.template triangularView<StrictlyLower>();
    m3 = m2.template triangularView<StrictlyLower>();
    VERIFY_IS_APPROX(m3, refMat3);

    // check sparse-triangular to dense
    refMat3 = m2.template triangularView<StrictlyUpper>();
    VERIFY_IS_APPROX(refMat3, DenseMatrix(refMat2.template triangularView<StrictlyUpper>()));

    // check sparse triangular view iteration-based evaluation
    m2.setZero();
    VERIFY_IS_CWISE_EQUAL(m2.template triangularView<UnitLower>().toDense(), DenseMatrix::Identity(rows, cols));
    VERIFY_IS_CWISE_EQUAL(m2.template triangularView<UnitUpper>().toDense(), DenseMatrix::Identity(rows, cols));
  }

  // test selfadjointView
  if (!SparseMatrixType::IsRowMajor) {
    DenseMatrix refMat2(rows, rows), refMat3(rows, rows);
    SparseMatrixType m2(rows, rows), m3(rows, rows);
    initSparse<Scalar>(density, refMat2, m2);
    refMat3 = refMat2.template selfadjointView<Lower>();
    m3 = m2.template selfadjointView<Lower>();
    VERIFY_IS_APPROX(m3, refMat3);

    refMat3 += refMat2.template selfadjointView<Lower>();
    m3 += m2.template selfadjointView<Lower>();
    VERIFY_IS_APPROX(m3, refMat3);

    refMat3 -= refMat2.template selfadjointView<Lower>();
    m3 -= m2.template selfadjointView<Lower>();
    VERIFY_IS_APPROX(m3, refMat3);

    Scalar s2 = internal::random<Scalar>();
    refMat3 = DenseMatrix(refMat2.template selfadjointView<Upper>());
    refMat3 *= s2;
    SparseMatrixType m4 = s2 * m2.template selfadjointView<Upper>();
    VERIFY_IS_APPROX(m4, refMat3);
    refMat3 = DenseMatrix(refMat2.template selfadjointView<Upper>());
    refMat3 *= s2;
    m4 = m2.template selfadjointView<Upper>() * s2;
    VERIFY_IS_APPROX(m4, refMat3);
    refMat3 = DenseMatrix(refMat2.template selfadjointView<Lower>());
    refMat3 *= s2;
    m4 = s2 * m2.template selfadjointView<Lower>();
    VERIFY_IS_APPROX(m4, refMat3);
    refMat3 = DenseMatrix(refMat2.template selfadjointView<Lower>());
    refMat3 *= s2;
    m4 = m2.template selfadjointView<Lower>() * s2;
    VERIFY_IS_APPROX(m4, refMat3);

    // selfadjointView only works for square matrices:
    SparseMatrixType m5(rows, rows + 1);
    VERIFY_RAISES_ASSERT(m5.template selfadjointView<Lower>());
    VERIFY_RAISES_ASSERT(m5.template selfadjointView<Upper>());
  }

  // test sparseView
  {
    DenseMatrix refMat2 = DenseMatrix::Zero(rows, rows);
    SparseMatrixType m2(rows, rows);
    initSparse<Scalar>(density, refMat2, m2);
    VERIFY_IS_APPROX(m2.eval(), refMat2.sparseView().eval());

    // sparse view on expressions:
    VERIFY_IS_APPROX((s1 * m2).eval(), (s1 * refMat2).sparseView().eval());
    VERIFY_IS_APPROX((m2 + m2).eval(), (refMat2 + refMat2).sparseView().eval());
    VERIFY_IS_APPROX((m2 * m2).eval(), (refMat2.lazyProduct(refMat2)).sparseView().eval());
    VERIFY_IS_APPROX((m2 * m2).eval(), (refMat2 * refMat2).sparseView().eval());
  }

  // test diagonal
  {
    DenseMatrix refMat2 = DenseMatrix::Zero(rows, cols);
    SparseMatrixType m2(rows, cols);
    initSparse<Scalar>(density, refMat2, m2);
    VERIFY_IS_APPROX(m2.diagonal(), refMat2.diagonal().eval());
    DenseVector d = m2.diagonal();
    VERIFY_IS_APPROX(d, refMat2.diagonal().eval());
    d = m2.diagonal().array();
    VERIFY_IS_APPROX(d, refMat2.diagonal().eval());
    VERIFY_IS_APPROX(const_cast<const SparseMatrixType&>(m2).diagonal(), refMat2.diagonal().eval());

    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag);
    m2.diagonal() += refMat2.diagonal();
    refMat2.diagonal() += refMat2.diagonal();
    VERIFY_IS_APPROX(m2, refMat2);

    // traits<Transpose<D>> and traits<CwiseUnaryView<Op, D>> derive from traits<D> for D = Diagonal<SparseMatrix>.
    VERIFY_IS_APPROX(m2.diagonal().real(), refMat2.diagonal().real().eval());
    VERIFY_IS_APPROX(m2.diagonal().transpose(), refMat2.diagonal().transpose().eval());
    m2.diagonal().real() *= RealScalar(2);
    refMat2.diagonal().real() *= RealScalar(2);
    VERIFY_IS_APPROX(m2, refMat2);
  }

  // test diagonal to sparse
  {
    DenseVector d = DenseVector::Random(rows);
    DenseMatrix refMat2 = d.asDiagonal();
    SparseMatrixType m2;
    m2 = d.asDiagonal();
    VERIFY_IS_APPROX(m2, refMat2);
    SparseMatrixType m3(d.asDiagonal());
    VERIFY_IS_APPROX(m3, refMat2);
    refMat2 += d.asDiagonal();
    m2 += d.asDiagonal();
    VERIFY_IS_APPROX(m2, refMat2);
    m2.setZero();
    m2 += d.asDiagonal();
    refMat2.setZero();
    refMat2 += d.asDiagonal();
    VERIFY_IS_APPROX(m2, refMat2);
    m2.setZero();
    m2 -= d.asDiagonal();
    refMat2.setZero();
    refMat2 -= d.asDiagonal();
    VERIFY_IS_APPROX(m2, refMat2);

    initSparse<Scalar>(density, refMat2, m2);
    m2.makeCompressed();
    m2 += d.asDiagonal();
    refMat2 += d.asDiagonal();
    VERIFY_IS_APPROX(m2, refMat2);

    initSparse<Scalar>(density, refMat2, m2);
    m2.makeCompressed();
    VectorXi res(rows);
    setRandomInRange(res, 0, 3);
    m2.reserve(res);
    m2 -= d.asDiagonal();
    refMat2 -= d.asDiagonal();
    VERIFY_IS_APPROX(m2, refMat2);

    // zero-length diagonal
    d = DenseVector(0);
    SparseMatrixType m4(d.asDiagonal());
  }

  // test conservative resize
  {
    std::vector<std::pair<StorageIndex, StorageIndex>> inc;
    if (rows > 3 && cols > 2) inc.push_back(std::pair<StorageIndex, StorageIndex>(-3, -2));
    inc.push_back(std::pair<StorageIndex, StorageIndex>(0, 0));
    inc.push_back(std::pair<StorageIndex, StorageIndex>(3, 2));
    inc.push_back(std::pair<StorageIndex, StorageIndex>(3, 0));
    inc.push_back(std::pair<StorageIndex, StorageIndex>(0, 3));
    inc.push_back(std::pair<StorageIndex, StorageIndex>(0, -1));
    inc.push_back(std::pair<StorageIndex, StorageIndex>(-1, 0));
    inc.push_back(std::pair<StorageIndex, StorageIndex>(-1, -1));

    for (size_t i = 0; i < inc.size(); i++) {
      StorageIndex incRows = inc[i].first;
      StorageIndex incCols = inc[i].second;
      SparseMatrixType m1(rows, cols);
      DenseMatrix refMat1 = DenseMatrix::Zero(rows, cols);
      initSparse<Scalar>(density, refMat1, m1);

      SparseMatrixType m2 = m1;
      m2.makeCompressed();

      m1.conservativeResize(rows + incRows, cols + incCols);
      m2.conservativeResize(rows + incRows, cols + incCols);
      refMat1.conservativeResize(rows + incRows, cols + incCols);
      if (incRows > 0) refMat1.bottomRows(incRows).setZero();
      if (incCols > 0) refMat1.rightCols(incCols).setZero();

      VERIFY_IS_APPROX(m1, refMat1);
      VERIFY_IS_APPROX(m2, refMat1);

      // Insert new values
      if (incRows > 0) m1.insert(m1.rows() - 1, 0) = refMat1(refMat1.rows() - 1, 0) = 1;
      if (incCols > 0) m1.insert(0, m1.cols() - 1) = refMat1(0, refMat1.cols() - 1) = 1;

      VERIFY_IS_APPROX(m1, refMat1);
    }
  }

  // test resize / conservativeResize with Eigen::NoChange (bug #656)
  {
    SparseMatrixType m1(rows, cols);
    DenseMatrix refMat1 = DenseMatrix::Zero(rows, cols);
    initSparse<Scalar>(density, refMat1, m1);

    SparseMatrixType m2 = m1;

    m1.conservativeResize(NoChange, cols + 2);
    refMat1.conservativeResize(NoChange, cols + 2);
    refMat1.rightCols(2).setZero();
    VERIFY_IS_APPROX(m1, refMat1);

    m1.conservativeResize(rows + 1, NoChange);
    refMat1.conservativeResize(rows + 1, NoChange);
    refMat1.bottomRows(1).setZero();
    VERIFY_IS_APPROX(m1, refMat1);

    m2.resize(NoChange, cols + 4);
    VERIFY(m2.rows() == rows && m2.cols() == cols + 4);
    VERIFY(m2.nonZeros() == 0);

    m2.resize(rows + 2, NoChange);
    VERIFY(m2.rows() == rows + 2 && m2.cols() == cols + 4);

    SparseVector<Scalar, ColMajor, StorageIndex> v(rows);
    v.resize(NoChange, 1);
    VERIFY(v.size() == rows);
    v.resize(rows + 3, NoChange);
    VERIFY(v.size() == rows + 3);
  }

  // test Identity matrix
  {
    DenseMatrix refMat1 = DenseMatrix::Identity(rows, rows);
    SparseMatrixType m1(rows, rows);
    m1.setIdentity();
    VERIFY_IS_APPROX(m1, refMat1);
    for (int k = 0; k < rows * rows / 4; ++k) {
      Index i = internal::random<Index>(0, rows - 1);
      Index j = internal::random<Index>(0, rows - 1);
      Scalar v = internal::random<Scalar>();
      m1.coeffRef(i, j) = v;
      refMat1.coeffRef(i, j) = v;
      VERIFY_IS_APPROX(m1, refMat1);
      if (internal::random<Index>(0, 10) < 2) m1.makeCompressed();
    }
    m1.setIdentity();
    refMat1.setIdentity();
    VERIFY_IS_APPROX(m1, refMat1);
  }

  // test array/vector of InnerIterator
  {
    using IteratorType = typename SparseMatrixType::InnerIterator;

    DenseMatrix refMat2 = DenseMatrix::Zero(rows, cols);
    SparseMatrixType m2(rows, cols);
    initSparse<Scalar>(density, refMat2, m2);
    IteratorType static_array[2];
    static_array[0] = IteratorType(m2, 0);
    static_array[1] = IteratorType(m2, m2.outerSize() - 1);
    VERIFY(static_array[0] || m2.innerVector(static_array[0].outer()).nonZeros() == 0);
    VERIFY(static_array[1] || m2.innerVector(static_array[1].outer()).nonZeros() == 0);
    if (static_array[0] && static_array[1]) {
      ++(static_array[1]);
      static_array[1] = IteratorType(m2, 0);
      VERIFY(static_array[1]);
      VERIFY(static_array[1].index() == static_array[0].index());
      VERIFY(static_array[1].outer() == static_array[0].outer());
      VERIFY(static_array[1].value() == static_array[0].value());
    }

    std::vector<IteratorType> iters(2);
    iters[0] = IteratorType(m2, 0);
    iters[1] = IteratorType(m2, m2.outerSize() - 1);
  }

  // test InnerIterator equality (bug #1192)
  {
    using IteratorType = typename SparseMatrixType::InnerIterator;
    SparseMatrixType m2(rows, cols);
    DenseMatrix refMat2 = DenseMatrix::Zero(rows, cols);
    initSparse<Scalar>(density, refMat2, m2);
    Index outer_with_two = -1;
    for (Index o = 0; o < m2.outerSize(); ++o)
      if (m2.innerVector(o).nonZeros() >= 2) {
        outer_with_two = o;
        break;
      }
    if (outer_with_two >= 0) {
      IteratorType a(m2, outer_with_two), b(m2, outer_with_two);
      VERIFY(a == b);
      ++b;
      VERIFY(a != b);
      VERIFY(!(a == b));
      ++a;
      VERIFY(a == b);
      while (a) ++a;
      while (b) ++b;
      VERIFY(a == b);
    }
  }

  // test reserve with empty rows/columns
  {
    SparseMatrixType m1(0, cols);
    m1.reserve(ArrayXi::Constant(m1.outerSize(), 1));
    SparseMatrixType m2(rows, 0);
    m2.reserve(ArrayXi::Constant(m2.outerSize(), 1));
  }

  // test move
  {
    using TransposedType = SparseMatrix<Scalar, SparseMatrixType::IsRowMajor ? ColMajor : RowMajor,
                                        typename SparseMatrixType::StorageIndex>;
    DenseMatrix refMat1 = DenseMatrix::Random(rows, cols);
    SparseMatrixType m1(rows, cols);
    initSparse<Scalar>(density, refMat1, m1);
    // test move ctor
    SparseMatrixType m2(std::move(m1));
    VERIFY_IS_APPROX(m2, refMat1);
    // test move assignment
    m1 = std::move(m2);
    VERIFY_IS_APPROX(m1, refMat1);
    // test move ctor (SparseMatrixBase)
    TransposedType m3(std::move(m1.transpose()));
    VERIFY_IS_APPROX(m3, refMat1.transpose());
    // test move assignment (SparseMatrixBase)
    m2 = std::move(m3.transpose());
    VERIFY_IS_APPROX(m2, refMat1);
  }
}

template <typename SparseMatrixType>
void big_sparse_triplet(Index rows, Index cols, double density) {
  using StorageIndex = typename SparseMatrixType::StorageIndex;
  using Scalar = typename SparseMatrixType::Scalar;
  using TripletType = Triplet<Scalar, Index>;
  std::vector<TripletType> triplets;
  double nelements = density * static_cast<double>(rows * cols);
  VERIFY(nelements >= 0 && nelements < static_cast<double>(NumTraits<StorageIndex>::highest()));
  Index ntriplets = Index(nelements);
  triplets.reserve(ntriplets);
  Scalar sum = Scalar(0);
  for (Index i = 0; i < ntriplets; ++i) {
    Index r = internal::random<Index>(0, rows - 1);
    Index c = internal::random<Index>(0, cols - 1);
    // use positive values to prevent numerical cancellation errors in sum
    Scalar v = numext::abs(internal::random<Scalar>());
    triplets.push_back(TripletType(r, c, v));
    sum += v;
  }
  SparseMatrixType m(rows, cols);
  m.setFromTriplets(triplets.begin(), triplets.end());
  VERIFY(m.nonZeros() <= ntriplets);
  VERIFY_IS_APPROX(sum, m.sum());
}

// Regression test: SparseMatrixBase::operator-=(const EigenBase&) must subtract, not overwrite.
// SparseSelfAdjointView inherits EigenBase (not SparseMatrixBase), so it exercises this overload.
template <int>
void sparse_sub_assign_eigenbase() {
  Index n = 10;
  SparseMatrix<double> A(n, n), B(n, n);
  A.setIdentity();
  B.setIdentity();
  B *= 3.0;

  // Make B symmetric so selfadjointView is valid
  MatrixXd refA = MatrixXd(A);
  MatrixXd refB = MatrixXd(B);

  // operator-= going through the EigenBase<OtherDerived> overload
  A -= B.selfadjointView<Lower>();
  refA -= refB.selfadjointView<Lower>();
  VERIFY_IS_APPROX(MatrixXd(A), refA);

  // Also test operator+= for symmetry
  A += B.selfadjointView<Lower>();
  refA += refB.selfadjointView<Lower>();
  VERIFY_IS_APPROX(MatrixXd(A), refA);
}

// Regression test: AmbiVector::coeff() must return the correct linked-list element.
template <int>
void ambivector_coeff() {
  using namespace Eigen::internal;
  const Index size = 20;
  AmbiVector<double, int> vec(size);
  vec.setBounds(0, size);

  // Test in sparse (linked-list) mode
  vec.init(IsSparse);
  vec.restart();
  // Insert elements out of order to stress linked-list traversal:
  // Insert at indices 2, 5, 10, 15 with distinct values
  vec.coeffRef(2) = 2.0;
  vec.coeffRef(5) = 5.0;
  vec.coeffRef(10) = 10.0;
  vec.coeffRef(15) = 15.0;

  // coeff() should return the correct value for each inserted element
  VERIFY_IS_APPROX(vec.coeff(2), 2.0);
  VERIFY_IS_APPROX(vec.coeff(5), 5.0);
  VERIFY_IS_APPROX(vec.coeff(10), 10.0);
  VERIFY_IS_APPROX(vec.coeff(15), 15.0);

  // coeff() should return zero for non-inserted elements
  VERIFY_IS_EQUAL(vec.coeff(0), 0.0);
  VERIFY_IS_EQUAL(vec.coeff(3), 0.0);
  VERIFY_IS_EQUAL(vec.coeff(7), 0.0);
  VERIFY_IS_EQUAL(vec.coeff(18), 0.0);

  // Verify coeff() still works after coeffRef() advances m_llCurrent.
  // This is the specific scenario that triggered the bug: coeffRef(15)
  // leaves m_llCurrent pointing at element 15, then coeff(2) should
  // still return 2.0, not 15.0.
  vec.restart();
  vec.coeffRef(15) += 0.0;  // advances m_llCurrent to element at index 15
  VERIFY_IS_APPROX(vec.coeff(2), 2.0);
  VERIFY_IS_APPROX(vec.coeff(5), 5.0);

  // Test in dense mode as well (simpler, but for completeness)
  vec.init(IsDense);
  vec.setZero();
  vec.coeffRef(3) = 3.0;
  vec.coeffRef(7) = 7.0;
  VERIFY_IS_APPROX(vec.coeff(3), 3.0);
  VERIFY_IS_APPROX(vec.coeff(7), 7.0);
  VERIFY_IS_EQUAL(vec.coeff(0), 0.0);
}

template <typename Pattern, typename SparseMatrixType>
void verify_sparsity_pattern_ref_matches(const Pattern& pattern, const SparseMatrixType& expected) {
  VERIFY_IS_EQUAL(pattern.outerSize, expected.cols());
  VERIFY_IS_EQUAL(pattern.innerSize, expected.rows());

  Index pattern_nonzeros = 0;
  for (Index j = 0; j < pattern.outerSize; ++j) {
    const Index nz = pattern.nonZeros(j);
    pattern_nonzeros += nz;
    for (Index k = pattern.outer[j], end = k + nz; k < end; ++k) {
      VERIFY(expected.coeff(pattern.inner[k], j) != typename SparseMatrixType::Scalar(0));
    }
  }
  VERIFY_IS_EQUAL(pattern_nonzeros, expected.nonZeros());
}

template <int>
void sparsity_pattern_ref_sparse_expressions() {
  using Scalar = std::complex<double>;
  using SparseMatrixType = SparseMatrix<Scalar, ColMajor, int>;
  using RealSparseMatrixType = SparseMatrix<double, ColMajor, int>;
  using RowMajorSparseMatrixType = SparseMatrix<double, RowMajor, int>;
  using VectorI = Matrix<int, Dynamic, 1>;

  SparseMatrixType a(3, 3);
  std::vector<Triplet<Scalar, int>> triplets;
  triplets.emplace_back(0, 0, Scalar(1.0, 1.0));
  triplets.emplace_back(1, 0, Scalar(2.0, -1.0));
  triplets.emplace_back(2, 1, Scalar(3.0, 2.0));
  triplets.emplace_back(2, 2, Scalar(4.0, -3.0));
  a.setFromTriplets(triplets.begin(), triplets.end());

  PermutationMatrix<Dynamic, Dynamic, int> p(3);
  p.indices() << 1, 2, 0;

  VectorI outer, inner;
  SparseMatrixType permuted = p * a;
  const internal::SparsityPatternRef<int> permuted_pattern = internal::make_col_major_pattern_ref(p * a, outer, inner);
  verify_sparsity_pattern_ref_matches(permuted_pattern, permuted);

  RealSparseMatrixType real_part = a.real();
  const internal::SparsityPatternRef<int> real_pattern = internal::make_col_major_pattern_ref(a.real(), outer, inner);
  verify_sparsity_pattern_ref_matches(real_pattern, real_part);

  RowMajorSparseMatrixType row_major = real_part;
  const internal::SparsityPatternRef<int> row_major_pattern =
      internal::make_col_major_pattern_ref(row_major, outer, inner);
  verify_sparsity_pattern_ref_matches(row_major_pattern, real_part);
}

// Verify that two compressed column-major sparse matrices have the same
// sparsity pattern (rows, cols, and per-column inner indices).
template <typename SparseA, typename SparseB>
void verify_same_pattern(const SparseA& got, const SparseB& expected) {
  VERIFY_IS_EQUAL(got.rows(), expected.rows());
  VERIFY_IS_EQUAL(got.cols(), expected.cols());
  for (Index j = 0; j < expected.cols(); ++j) {
    std::vector<Index> got_rows, expected_rows;
    for (typename SparseA::InnerIterator it(got, j); it; ++it) got_rows.push_back(it.row());
    for (typename SparseB::InnerIterator it(expected, j); it; ++it) expected_rows.push_back(it.row());
    std::sort(got_rows.begin(), got_rows.end());
    std::sort(expected_rows.begin(), expected_rows.end());
    VERIFY_IS_EQUAL(Index(got_rows.size()), Index(expected_rows.size()));
    for (size_t k = 0; k < got_rows.size(); ++k) VERIFY_IS_EQUAL(got_rows[k], expected_rows[k]);
  }
}

template <int>
void materialize_at_plus_a_pattern_basic() {
  using SparseMatrixType = SparseMatrix<double, ColMajor, int>;
  using PatternMatrixType = SparseMatrix<signed char, ColMajor, int>;
  using VectorI = Matrix<int, Dynamic, 1>;

  // Asymmetric pattern: (0,0), (1,0), (2,1), (0,2), (2,2). Its A^T+A pattern
  // adds (0,1), (1,2), (2,0).
  SparseMatrixType a(3, 3);
  std::vector<Triplet<double, int>> triplets;
  triplets.emplace_back(0, 0, 1.0);
  triplets.emplace_back(1, 0, 2.0);
  triplets.emplace_back(2, 1, 3.0);
  triplets.emplace_back(0, 2, 4.0);
  triplets.emplace_back(2, 2, 5.0);
  a.setFromTriplets(triplets.begin(), triplets.end());

  VectorI outer, inner;
  internal::SparsityPatternRef<int> pat = internal::make_col_major_pattern_ref(a, outer, inner);

  PatternMatrixType got;
  internal::materialize_at_plus_a_pattern(pat, got);

  // Reference: pattern of (a + a.transpose()), keeping structural nonzeros.
  SparseMatrixType expected = a + SparseMatrixType(a.transpose());
  verify_same_pattern(got, expected);
  // Output values must be the placeholder sentinel.
  const signed char one = 1;
  for (Index j = 0; j < got.cols(); ++j) {
    for (PatternMatrixType::InnerIterator it(got, j); it; ++it) {
      VERIFY_IS_EQUAL(it.value(), one);
    }
  }
}

template <int>
void materialize_at_plus_a_pattern_random() {
  using SparseMatrixType = SparseMatrix<double, ColMajor, int>;
  using PatternMatrixType = SparseMatrix<signed char, ColMajor, int>;
  using DenseMatrixType = Matrix<double, Dynamic, Dynamic, ColMajor>;
  using VectorI = Matrix<int, Dynamic, 1>;

  const int n = internal::random<int>(8, 64);
  DenseMatrixType ref(n, n);
  SparseMatrixType a(n, n);
  initSparse<double>(0.4, ref, a, ForceNonZeroDiag);

  VectorI outer, inner;
  internal::SparsityPatternRef<int> pat = internal::make_col_major_pattern_ref(a, outer, inner);
  PatternMatrixType got;
  internal::materialize_at_plus_a_pattern(pat, got);

  SparseMatrixType expected = a + SparseMatrixType(a.transpose());
  verify_same_pattern(got, expected);
}

template <unsigned int UpLo>
void materialize_selfadjoint_pattern_random_impl() {
  using SparseMatrixType = SparseMatrix<double, ColMajor, int>;
  using PatternMatrixType = SparseMatrix<signed char, ColMajor, int>;
  using DenseMatrixType = Matrix<double, Dynamic, Dynamic, ColMajor>;
  using VectorI = Matrix<int, Dynamic, 1>;

  // Build a random sparse matrix that may have entries on both triangles; the
  // selfadjoint pattern must filter to only the requested triangle.
  const int n = internal::random<int>(8, 64);
  DenseMatrixType ref(n, n);
  SparseMatrixType a(n, n);
  initSparse<double>(0.4, ref, a, ForceNonZeroDiag);
  a.makeCompressed();

  VectorI outer, inner;
  internal::SparsityPatternRef<int> pat = internal::make_col_major_pattern_ref(a, outer, inner);
  PatternMatrixType got;
  internal::materialize_selfadjoint_pattern<UpLo>(pat, got);

  // Reference: explicit symmetrization of the requested triangle.
  SparseMatrixType expected(n, n);
  expected = a.template selfadjointView<UpLo>();
  verify_same_pattern(got, expected);
}

template <int>
void materialize_selfadjoint_pattern_lower_upper() {
  materialize_selfadjoint_pattern_random_impl<Lower>();
  materialize_selfadjoint_pattern_random_impl<Upper>();
}

// isApprox() compared squared norms, which overflow to inf <= inf or underflow to 0 <= 0, so a.isApprox(3 * a) held for
// tiny and huge matrices, and a.isApprox(-a) where a - (-a) overflows.
template <typename Scalar, int Options>
void sparse_isapprox_extreme_scales() {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using SpMat = SparseMatrix<Scalar, Options>;
  using OtherSpMat = SparseMatrix<Scalar, Options ^ RowMajor>;
  const RealScalar highest = NumTraits<RealScalar>::highest();
  const RealScalar min = (std::numeric_limits<RealScalar>::min)();
  for (const RealScalar s : extreme_scales<RealScalar>()) {
    SpMat a(5, 4);
    a.insert(0, 0) = Scalar(s);
    a.insert(2, 1) = Scalar(-2 * s);
    a.insert(4, 3) = Scalar(s / 2);
    a.makeCompressed();
    const SpMat b = Scalar(3) * a;
    SpMat uncompressed = a;
    uncompressed.uncompress();
    const OtherSpMat other = a;

    VERIFY(a.isApprox(a));
    VERIFY(!a.isApprox(b));
    VERIFY(!b.isApprox(a));
    VERIFY(!a.isApprox(SpMat(-a)));
    VERIFY(!a.isApprox(SpMat(a.rows(), a.cols())));
    VERIFY(uncompressed.isApprox(a));
    VERIFY(!uncompressed.isApprox(b));
    VERIFY(other.isApprox(a));
    VERIFY(!a.isApprox(Scalar(3) * other));

    SparseVector<Scalar> v(6);
    v.insert(1) = Scalar(s);
    v.insert(4) = Scalar(-s);
    VERIFY(v.isApprox(v));
    VERIFY(!v.isApprox(SparseVector<Scalar>(Scalar(3) * v)));

    // A tight prec: |a - c| = 2^-10 |a| exactly where the values stay normal.
    if (s >= min * RealScalar(16) && s <= highest / RealScalar(8)) {
      const SpMat c = a * Scalar(RealScalar(1) + RealScalar(1) / RealScalar(1024));
      VERIFY(a.isApprox(c, RealScalar(1) / RealScalar(512)));
      VERIFY(!a.isApprox(c, RealScalar(1) / RealScalar(2048)));
    }
  }
}

template <int>
void bug1105() {
  // Regression test for bug 1105
  int n = Eigen::internal::random<int>(200, 600);
  SparseMatrix<std::complex<double>, 0, long> mat(n, n);
  std::complex<double> val;

  for (int i = 0; i < n; ++i) {
    mat.coeffRef(i, i % (n / 10)) = val;
    VERIFY(mat.data().allocatedSize() < 20 * n);
  }
}

// A short StorageIndex bounds the number of stored entries, not the outer dimension, which exceeds its range here.
template <int Options>
void sparse_insert_narrow_index() {
  using SpMat = SparseMatrix<double, Options, short>;
  const Index highest = NumTraits<short>::highest();
  const Index n = highest + 1000;
  SpMat m(SpMat::IsRowMajor ? n : 4, SpMat::IsRowMajor ? 4 : n);
  // The first insertion goes into an empty matrix; the rest go below and above it.
  const Index outers[] = {n - 1, 0, n / 2, highest + 1, 1};
  for (Index k = 0; k < 5; ++k) {
    const Index outer = outers[k], inner = k % 4;
    insert_by_outer(m, outer, inner) = double(k + 1);
  }
  // The storage grows geometrically: five entries must not reserve the whole index range.
  VERIFY(m.data().allocatedSize() < highest);
  for (int compressed = 0; compressed < 2; ++compressed) {
    if (compressed) m.makeCompressed();
    VERIFY_IS_EQUAL(m.nonZeros(), 5);
    for (Index k = 0; k < 5; ++k) {
      const Index outer = outers[k], inner = k % 4;
      VERIFY_IS_EQUAL(coeff_by_outer(m, outer, inner), double(k + 1));
    }
  }

  // setZero() keeps the allocation, which must not count as stored entries.
  SpMat refill(m.rows(), m.cols());
  insert_by_outer(refill, n - 1, 1) = 1.0;
  refill.setZero();
  insert_by_outer(refill, n - 1, 1) = 2.0;
  VERIFY_IS_EQUAL(refill.nonZeros(), 1);
  VERIFY_IS_EQUAL(coeff_by_outer(refill, n - 1, 1), 2.0);
}

// insert() can store highest() entries, however much reserve() or an earlier fill allocated, and throws past that.
template <int Options>
void sparse_insert_full_index_range() {
  using SpMat = SparseMatrix<double, Options, short>;
  const Index highest = NumTraits<short>::highest();
  const Index n = 256;
  const auto insert = [&](SpMat& m, Index k) -> double& { return insert_by_outer(m, k / n, k % n); };
  const auto coeff = [&](const SpMat& m, Index k) { return coeff_by_outer(m, k / n, k % n); };

  SpMat m(n, n);
  m.reserve(highest);
  insert(m, 1) = 1.0;
  VERIFY_IS_EQUAL(m.nonZeros(), 1);
  VERIFY_IS_EQUAL(coeff(m, 1), 1.0);

  for (int fill = 0; fill < 2; ++fill) {
    m.setZero();
    for (Index k = 0; k < highest; ++k) insert(m, k) = double(k + 1);
    VERIFY_IS_EQUAL(m.nonZeros(), highest);
#ifdef EIGEN_EXCEPTIONS
    bool threw = false;
    try {
      insert(m, highest) = 0.0;
    } catch (const std::bad_alloc&) {
      threw = true;
    }
    VERIFY(threw);
    VERIFY_IS_EQUAL(m.nonZeros(), highest);
#endif
    VERIFY_IS_EQUAL(coeff(m, 0), 1.0);
    VERIFY_IS_EQUAL(coeff(m, highest - 1), double(highest));
  }
}

// Reads `m` through its iterators into a dense matrix, so that the inner indices are compared along with the values.
template <typename SparseType>
void verify_movable_entries(const SparseType& m, const MatrixXd& ref) {
  MatrixXd dense = MatrixXd::Zero(m.rows(), m.cols());
  Index entries = 0;
  for (Index j = 0; j < m.outerSize(); ++j)
    for (typename SparseType::InnerIterator it(m, j); it; ++it, ++entries)
      dense(it.row(), it.col()) = double(it.value());
  VERIFY_IS_EQUAL(entries, m.nonZeros());
  VERIFY_IS_CWISE_EQUAL(dense, ref);
}

// The operation since the last MovableScalarCounters::reset() shifted stored entries: it moved them, copied exactly
// `COPIES` values, and move-assigned no entry to itself.
#define VERIFY_SHIFT_COUNTS(COPIES)                                      \
  do {                                                                   \
    VERIFY(MovableScalarCounters::get().moves > 0);                      \
    VERIFY_IS_EQUAL(MovableScalarCounters::get().copies, Index(COPIES)); \
    VERIFY_IS_EQUAL(MovableScalarCounters::get().self_moves, Index(0));  \
  } while (false)

// Every shift, compaction and relocation of stored entries inside SparseCore, for a scalar that owns heap memory.
template <int Options>
void sparse_shift_moves() {
  using Scalar = MovableScalar<double>;
  using SpMat = SparseMatrix<Scalar, Options>;
  const Index n = 8;
  auto coeff = [](MatrixXd& dense, Index outer, Index inner) -> double& {
    return SpMat::IsRowMajor ? dense(outer, inner) : dense(inner, outer);
  };

  // Inner vector j holds the indices {1, 4, 6} \ {j}, so no diagonal entry is stored.
  SpMat base(n, n);
  MatrixXd baseRef = MatrixXd::Zero(n, n);
  double value = 1;
  for (Index j = 0; j < n; ++j) {
    for (Index i : {1, 4, 6}) {
      if (i == j) continue;
      base.insertByOuterInner(j, i) = Scalar(value);
      coeff(baseRef, j, i) = value++;
    }
  }
  base.makeCompressed();
  verify_movable_entries(base, baseRef);

  auto insert = [&](SpMat& m, MatrixXd& ref, Index outer, Index inner) {
    MovableScalarCounters::reset();
    m.insertByOuterInner(outer, inner) = Scalar(value);
    coeff(ref, outer, inner) = value++;
  };

  {
    SpMat m = base;
    MatrixXd ref = baseRef;

    // reserve() on a compressed matrix: vectors 0 to 2 have nothing reserved before them and stay in place.
    VectorXi sizes(n);
    sizes << 0, 0, 2, 0, 1, 0, 3, 0;
    MovableScalarCounters::reset();
    m.reserve(sizes);
    VERIFY_SHIFT_COUNTS(0);
    VERIFY(!m.isCompressed());
    verify_movable_entries(m, ref);

    // reserve() on an uncompressed matrix: vectors 0 to 3 stay in place.
    sizes << 0, 0, 2, 1, 1, 0, 3, 2;
    MovableScalarCounters::reset();
    m.reserve(sizes);
    VERIFY_SHIFT_COUNTS(0);
    verify_movable_entries(m, ref);

    // Vector 2 has capacity: its entries shift right.
    insert(m, ref, 2, 0);
    VERIFY_SHIFT_COUNTS(0);
    verify_movable_entries(m, ref);

    // Vectors 1 and 0 are full, vector 2 has capacity: the tail of vector 1 and all of vector 2 shift right.
    insert(m, ref, 1, 5);
    VERIFY_SHIFT_COUNTS(0);
    verify_movable_entries(m, ref);

    // Vector 5 is full, its left neighbor has capacity: the head of vector 5 shifts left.
    insert(m, ref, 5, 5);
    VERIFY_SHIFT_COUNTS(0);
    verify_movable_entries(m, ref);

    // makeCompressed(): vectors 0 and 1 are contiguous and stay in place, the rest shift left.
    MovableScalarCounters::reset();
    m.makeCompressed();
    VERIFY_SHIFT_COUNTS(0);
    VERIFY(m.isCompressed());
    verify_movable_entries(m, ref);

    // No vector has capacity: the insertion reserves one slot per vector, then shifts within vector 3.
    m.uncompress();
    insert(m, ref, 3, 0);
    VERIFY_SHIFT_COUNTS(0);
    verify_movable_entries(m, ref);
  }

  // removeOuterVectors(): the new first vector shifts left to the start of the buffer.
  {
    SpMat m = base;
    MovableScalarCounters::reset();
    m.removeOuterVectors(0);
    VERIFY_SHIFT_COUNTS(0);
    MatrixXd ref = SpMat::IsRowMajor ? MatrixXd(baseRef.bottomRows(n - 1)) : MatrixXd(baseRef.rightCols(n - 1));
    verify_movable_entries(m, ref);
  }

  // Adding a diagonal inserts into every vector. With no capacity everything is shifted; with capacity in every
  // vector each chunk stays in place until its own insertion; the mixed layout shifts some of them early.
  for (int layout = 0; layout < 3; ++layout) {
    SpMat m = base;
    if (layout == 1) m.reserve(VectorXi::Constant(n, 1));
    if (layout == 2) m.reserve((VectorXi(n) << 1, 0, 0, 1, 0, 1, 1, 0).finished());
    Matrix<Scalar, Dynamic, 1> diagonal(n);
    MatrixXd ref = baseRef;
    for (Index j = 0; j < n; ++j) {
      diagonal(j) = Scalar(100.0 + double(j));
      ref(j, j) = 100.0 + double(j);
    }
    MovableScalarCounters::reset();
    m += diagonal.asDiagonal();
    const MovableScalarCounters inserted = MovableScalarCounters::get();
    verify_movable_entries(m, ref);
    // A second addition updates the stored diagonal in place, so its copies are those of reading `diagonal`.
    MovableScalarCounters::reset();
    m += diagonal.asDiagonal();
    VERIFY_IS_EQUAL(MovableScalarCounters::get().moves, Index(0));
    VERIFY(inserted.moves > 0);
    VERIFY_IS_EQUAL(inserted.copies, MovableScalarCounters::get().copies);
    VERIFY_IS_EQUAL(inserted.self_moves, Index(0));
    ref.diagonal() *= 2;
    verify_movable_entries(m, ref);
  }

  // Assigning an inner vector of a compressed matrix shifts the entries behind it: left when the vector shrinks,
  // right when it grows. In an uncompressed matrix a vector that outgrows its room reallocates the storage, and the
  // entries in front of it and behind it move to the new buffer. Only the `nnz` assigned values are copied.
  struct InnerVectorCase {
    bool compressed;
    Index nnz;
  };
  for (InnerVectorCase c : {InnerVectorCase{true, 1}, InnerVectorCase{true, 6}, InnerVectorCase{false, 6}}) {
    const Index nnz = c.nnz;
    SpMat m = base;
    if (c.compressed)
      m.data().reserve(n);
    else
      m.uncompress();
    MatrixXd ref = baseRef;
    SpMat source(SpMat::IsRowMajor ? 1 : n, SpMat::IsRowMajor ? n : 1);
    for (Index i = 0; i < n; ++i) coeff(ref, 2, i) = 0;
    for (Index i = 0; i < nnz; ++i) {
      source.insertByOuterInner(0, i) = Scalar(value);
      coeff(ref, 2, i) = value++;
    }
    source.makeCompressed();
    MovableScalarCounters::reset();
    m.innerVector(2) = source;
    VERIFY_SHIFT_COUNTS(nnz);
    VERIFY_IS_EQUAL(m.isCompressed(), c.compressed);
    verify_movable_entries(m, ref);
  }

  // SparseVector::coeffRef() inserts in sorted order; its one copy is the default value of the new entry.
  {
    SparseVector<Scalar> v(n);
    MatrixXd ref = MatrixXd::Zero(n, 1);
    for (Index i : {2, 5, 7}) {
      v.coeffRef(i) = Scalar(value);
      ref(i) = value++;
    }
    MovableScalarCounters::reset();
    v.coeffRef(0) = Scalar(value);
    ref(0) = value++;
    VERIFY_SHIFT_COUNTS(1);
    verify_movable_entries(v, ref);
  }

  // SparseVector::insert() shifts the entries behind the new one.
  {
    SparseVector<Scalar> v(n);
    MatrixXd ref = MatrixXd::Zero(n, 1);
    for (Index i : {2, 5, 7}) {
      v.insert(i) = Scalar(value);
      ref(i) = value++;
    }
    MovableScalarCounters::reset();
    v.insert(0) = Scalar(value);
    ref(0) = value++;
    VERIFY_SHIFT_COUNTS(0);
    verify_movable_entries(v, ref);
  }

  // SparseVector::prune() packs the kept entries; the one in front of the dropped entry stays in place.
  {
    SparseVector<Scalar> v(n);
    MatrixXd ref = MatrixXd::Zero(n, 1);
    for (Index i : {1, 3, 4, 6}) {
      v.insert(i) = Scalar(value);
      ref(i) = value++;
    }
    const double dropped = ref(3);
    ref(3) = 0;
    MovableScalarCounters::reset();
    v.prune([dropped](const Scalar& x) { return double(x) != dropped; });
    VERIFY_SHIFT_COUNTS(0);
    verify_movable_entries(v, ref);
  }

  // prune() packs the kept entries across the vectors of a compressed matrix and within each vector of an
  // uncompressed one. Dropping inner index 4 leaves the entries in front of the first dropped one in place.
  for (bool compressed : {true, false}) {
    SpMat m = base;
    if (!compressed) m.uncompress();
    MatrixXd ref = baseRef;
    for (Index j = 0; j < n; ++j) coeff(ref, j, 4) = 0;
    MovableScalarCounters::reset();
    m.prune([](Index row, Index col, const Scalar&) { return (SpMat::IsRowMajor ? col : row) != 4; });
    VERIFY_SHIFT_COUNTS(0);
    VERIFY_IS_EQUAL(m.isCompressed(), compressed);
    verify_movable_entries(m, ref);
  }

  // setFromTriplets() sums duplicate entries and packs the first entry of each coefficient in front. Its copies are
  // those of reading the triplets and of the transposing assignment that sorts the result.
  {
    std::vector<Triplet<Scalar>> triplets;
    MatrixXd ref = MatrixXd::Zero(n, n);
    for (Index j = 0; j < n; ++j) {
      for (Index i : {1, 4, 6, 4}) {
        const int outer = int(j), inner = int(i);
        triplets.emplace_back(SpMat::IsRowMajor ? outer : inner, SpMat::IsRowMajor ? inner : outer, Scalar(value));
        coeff(ref, j, i) += value++;
      }
    }
    SpMat m(n, n);
    MovableScalarCounters::reset();
    m.setFromTriplets(triplets.begin(), triplets.end(),
                      [](const Scalar& a, const Scalar& b) { return Scalar(double(a) + double(b)); });
    VERIFY_SHIFT_COUNTS(Index(triplets.size()) + m.nonZeros());
    verify_movable_entries(m, ref);
  }
}

// User-provided copy operations, which make a class built on it not trivially copyable.
template <bool TriviallyCopyable>
struct CopyOnlyBase {
  double value = 0;
  CopyOnlyBase() = default;
  CopyOnlyBase(const CopyOnlyBase& other) : value(other.value) {}
  CopyOnlyBase& operator=(const CopyOnlyBase& other) {
    value = other.value;
    return *this;
  }
};

template <>
struct CopyOnlyBase<true> {
  double value = 0;
};

// A copy assignment and a deleted move assignment.
template <bool TriviallyCopyable, int RequireInitialization>
struct CopyOnlyScalar : CopyOnlyBase<TriviallyCopyable> {
  CopyOnlyScalar() = default;
  CopyOnlyScalar(double v) { this->value = v; }
  CopyOnlyScalar(const CopyOnlyScalar&) = default;
  CopyOnlyScalar& operator=(const CopyOnlyScalar&) = default;
  CopyOnlyScalar& operator=(CopyOnlyScalar&&) = delete;
  explicit operator double() const { return this->value; }
};

namespace Eigen {
template <bool TriviallyCopyable, int RequireInitialization_>
struct NumTraits<CopyOnlyScalar<TriviallyCopyable, RequireInitialization_>> : NumTraits<double> {
  static constexpr int RequireInitialization = RequireInitialization_;
};
}  // namespace Eigen

// The shifts copy a scalar that cannot be move-assigned rather than reject it. With RequireInitialization == 0,
// growing the storage move-constructs the stored entries of a scalar that is not trivially copyable, and reallocates
// the bytes of one that is.
template <bool TriviallyCopyable, int RequireInitialization, int Options>
void sparse_shift_copy_only() {
  using Scalar = CopyOnlyScalar<TriviallyCopyable, RequireInitialization>;
  using SpMat = SparseMatrix<Scalar, Options>;
  STATIC_CHECK(std::is_trivially_copyable<Scalar>::value == TriviallyCopyable);
  STATIC_CHECK(!std::is_move_assignable<Scalar>::value);
  const Index n = 6;
  auto coeff = [](MatrixXd& dense, Index outer, Index inner) -> double& {
    return SpMat::IsRowMajor ? dense(outer, inner) : dense(inner, outer);
  };
  // insert() assigns a temporary, which this scalar forbids, so the arrays are filled directly. Outer vector j
  // holds `perVector` entries at the inner indices j % 2, j % 2 + 2, ...
  auto fill = [&coeff](SpMat& m, MatrixXd& ref, Index perVector, double value) {
    m.resizeNonZeros(perVector * m.outerSize());
    for (Index j = 0; j < m.outerSize(); ++j) {
      m.outerIndexPtr()[j] = int(perVector * j);
      for (Index k = 0; k < perVector; ++k, ++value) {
        const Scalar entry(value);
        m.innerIndexPtr()[perVector * j + k] = int(j % 2 + 2 * k);
        m.valuePtr()[perVector * j + k] = entry;
        coeff(ref, j, j % 2 + 2 * k) = value;
      }
    }
    m.outerIndexPtr()[m.outerSize()] = int(perVector * m.outerSize());
  };
  SpMat m(n, n);
  MatrixXd ref = MatrixXd::Zero(n, n);
  fill(m, ref, 2, 1.0);

  m.reserve(VectorXi::Constant(n, 2));
  VERIFY(!m.isCompressed());
  verify_movable_entries(m, ref);

  m.makeCompressed();
  verify_movable_entries(m, ref);

  m.prune([](Index, Index, const Scalar& x) { return int(x.value) % 3 != 0; });
  ref = (ref.array() != 0 && ref.array().cast<int>() % 3 == 0).select(0.0, ref);
  verify_movable_entries(m, ref);

  // The first inner vector outgrows its room, so the entries behind it shift right.
  SpMat source(SpMat::IsRowMajor ? 1 : n, SpMat::IsRowMajor ? n : 1);
  MatrixXd sourceRef = MatrixXd::Zero(source.rows(), source.cols());
  fill(source, sourceRef, 3, 100.0);
  m.innerVector(0) = source;
  for (Index i = 0; i < n; ++i) coeff(ref, 0, i) = coeff(sourceRef, 0, i);
  verify_movable_entries(m, ref);

  // Removing the vectors in front shifts the next one to the start of the storage.
  m.removeOuterVectors(0, 2);
  const MatrixXd remaining = SpMat::IsRowMajor ? MatrixXd(ref.bottomRows(n - 2)) : MatrixXd(ref.rightCols(n - 2));
  verify_movable_entries(m, remaining);
}

#ifndef EIGEN_SPARSE_TEST_INCLUDED_FROM_SPARSE_EXTRA

EIGEN_DECLARE_TEST(sparse_basic) {
  g_dense_op_sparse_count = 0;  // Suppresses compiler warning.
  for (int i = 0; i < g_repeat; i++) {
    int r = Eigen::internal::random<int>(1, 200), c = Eigen::internal::random<int>(1, 200);
    if (Eigen::internal::random<int>(0, 3) == 0) {
      r = c;  // check square matrices in 25% of tries
    }
    EIGEN_UNUSED_VARIABLE(r + c);
    CALL_SUBTEST_1((sparse_basic(SparseMatrix<double>(1, 1))));
    CALL_SUBTEST_1((sparse_basic(SparseMatrix<double>(8, 8))));
    CALL_SUBTEST_2((sparse_basic(SparseMatrix<std::complex<double>, ColMajor>(r, c))));
    CALL_SUBTEST_2((sparse_basic(SparseMatrix<std::complex<double>, RowMajor>(r, c))));
    CALL_SUBTEST_2((sparse_basic(SparseMatrix<float, RowMajor>(r, c))));
    CALL_SUBTEST_2((sparse_basic(SparseMatrix<float, ColMajor>(r, c))));
    CALL_SUBTEST_3((sparse_basic(SparseMatrix<double, ColMajor>(r, c))));
    CALL_SUBTEST_3((sparse_basic(SparseMatrix<double, RowMajor>(r, c))));
    CALL_SUBTEST_4((sparse_basic(SparseMatrix<double, ColMajor, long int>(r, c))));
    CALL_SUBTEST_4((sparse_basic(SparseMatrix<double, RowMajor, long int>(r, c))));

    r = Eigen::internal::random<int>(1, 100);
    c = Eigen::internal::random<int>(1, 100);
    if (Eigen::internal::random<int>(0, 3) == 0) {
      r = c;  // check square matrices in 25% of tries
    }

    CALL_SUBTEST_5((sparse_basic(SparseMatrix<double, ColMajor, short int>(short(r), short(c)))));
    CALL_SUBTEST_5((sparse_basic(SparseMatrix<double, RowMajor, short int>(short(r), short(c)))));
  }

  // Regression test for bug 900: (manually insert higher values here, if you have enough RAM):
  CALL_SUBTEST_5((big_sparse_triplet<SparseMatrix<float, RowMajor, int>>(10000, 10000, 0.125)));
  CALL_SUBTEST_5((big_sparse_triplet<SparseMatrix<double, ColMajor, long int>>(10000, 10000, 0.125)));

  CALL_SUBTEST_5(bug1105<0>());
  CALL_SUBTEST_2((sparse_insert_narrow_index<ColMajor>()));
  CALL_SUBTEST_2((sparse_insert_narrow_index<RowMajor>()));
  CALL_SUBTEST_2((sparse_insert_full_index_range<ColMajor>()));
  CALL_SUBTEST_2((sparse_insert_full_index_range<RowMajor>()));
  CALL_SUBTEST_1((sparse_shift_moves<ColMajor>()));
  CALL_SUBTEST_1((sparse_shift_moves<RowMajor>()));
  CALL_SUBTEST_1((sparse_shift_copy_only<false, 1, ColMajor>()));
  CALL_SUBTEST_1((sparse_shift_copy_only<false, 1, RowMajor>()));
  CALL_SUBTEST_1((sparse_shift_copy_only<false, 0, ColMajor>()));
  CALL_SUBTEST_1((sparse_shift_copy_only<false, 0, RowMajor>()));
  CALL_SUBTEST_1((sparse_shift_copy_only<true, 0, ColMajor>()));
  CALL_SUBTEST_1((sparse_shift_copy_only<true, 0, RowMajor>()));
  CALL_SUBTEST_1((sparse_isapprox_extreme_scales<float, ColMajor>()));
  CALL_SUBTEST_1((sparse_isapprox_extreme_scales<double, RowMajor>()));
  CALL_SUBTEST_1((sparse_isapprox_extreme_scales<std::complex<double>, ColMajor>()));
  CALL_SUBTEST_1(sparse_sub_assign_eigenbase<0>());
  CALL_SUBTEST_1(ambivector_coeff<0>());
  CALL_SUBTEST_1(sparsity_pattern_ref_sparse_expressions<0>());
  CALL_SUBTEST_1(materialize_at_plus_a_pattern_basic<0>());
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1(materialize_at_plus_a_pattern_random<0>());
    CALL_SUBTEST_1(materialize_selfadjoint_pattern_lower_upper<0>());
  }
}
#endif
