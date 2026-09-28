// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

template <typename MatrixType>
void matrixVisitor_impl(MatrixType& m) {
  typedef typename MatrixType::Scalar Scalar;

  Index rows = m.rows();
  Index cols = m.cols();

  Scalar minc = Scalar(1000), maxc = Scalar(-1000);
  Index minrow = 0, mincol = 0, maxrow = 0, maxcol = 0;
  for (Index j = 0; j < cols; j++)
    for (Index i = 0; i < rows; i++) {
      if (m(i, j) < minc) {
        minc = m(i, j);
        minrow = i;
        mincol = j;
      }
      if (m(i, j) > maxc) {
        maxc = m(i, j);
        maxrow = i;
        maxcol = j;
      }
    }
  Index eigen_minrow, eigen_mincol, eigen_maxrow, eigen_maxcol;
  Scalar eigen_minc, eigen_maxc;
  eigen_minc = m.minCoeff(&eigen_minrow, &eigen_mincol);
  eigen_maxc = m.maxCoeff(&eigen_maxrow, &eigen_maxcol);
  VERIFY(minrow == eigen_minrow);
  VERIFY(maxrow == eigen_maxrow);
  VERIFY(mincol == eigen_mincol);
  VERIFY(maxcol == eigen_maxcol);
  VERIFY_IS_APPROX(minc, eigen_minc);
  VERIFY_IS_APPROX(maxc, eigen_maxc);
  VERIFY_IS_APPROX(minc, m.minCoeff());
  VERIFY_IS_APPROX(maxc, m.maxCoeff());

  eigen_maxc = (m.adjoint() * m).maxCoeff(&eigen_maxrow, &eigen_maxcol);
  Index maxrow2 = 0, maxcol2 = 0;
  eigen_maxc = (m.adjoint() * m).eval().maxCoeff(&maxrow2, &maxcol2);
  VERIFY(maxrow2 == eigen_maxrow);
  VERIFY(maxcol2 == eigen_maxcol);

  if (!NumTraits<Scalar>::IsInteger && m.size() > 2) {
    // Test NaN propagation by replacing an element with NaN.
    bool stop = false;
    for (Index j = 0; j < cols && !stop; ++j) {
      for (Index i = 0; i < rows && !stop; ++i) {
        if (!(j == mincol && i == minrow) && !(j == maxcol && i == maxrow)) {
          m(i, j) = NumTraits<Scalar>::quiet_NaN();
          stop = true;
          break;
        }
      }
    }

    eigen_minc = m.template minCoeff<PropagateNumbers>(&eigen_minrow, &eigen_mincol);
    eigen_maxc = m.template maxCoeff<PropagateNumbers>(&eigen_maxrow, &eigen_maxcol);
    VERIFY(minrow == eigen_minrow);
    VERIFY(maxrow == eigen_maxrow);
    VERIFY(mincol == eigen_mincol);
    VERIFY(maxcol == eigen_maxcol);
    VERIFY_IS_APPROX(minc, eigen_minc);
    VERIFY_IS_APPROX(maxc, eigen_maxc);
    VERIFY_IS_APPROX(minc, m.template minCoeff<PropagateNumbers>());
    VERIFY_IS_APPROX(maxc, m.template maxCoeff<PropagateNumbers>());

    eigen_minc = m.template minCoeff<PropagateNaN>(&eigen_minrow, &eigen_mincol);
    eigen_maxc = m.template maxCoeff<PropagateNaN>(&eigen_maxrow, &eigen_maxcol);
    VERIFY(minrow != eigen_minrow || mincol != eigen_mincol);
    VERIFY(maxrow != eigen_maxrow || maxcol != eigen_maxcol);
    VERIFY((numext::isnan)(eigen_minc));
    VERIFY((numext::isnan)(eigen_maxc));

    // Test matrix of all NaNs.
    m.fill(NumTraits<Scalar>::quiet_NaN());
    eigen_minc = m.template minCoeff<PropagateNumbers>(&eigen_minrow, &eigen_mincol);
    eigen_maxc = m.template maxCoeff<PropagateNumbers>(&eigen_maxrow, &eigen_maxcol);
    VERIFY(eigen_minrow == 0);
    VERIFY(eigen_maxrow == 0);
    VERIFY(eigen_mincol == 0);
    VERIFY(eigen_maxcol == 0);
    VERIFY((numext::isnan)(eigen_minc));
    VERIFY((numext::isnan)(eigen_maxc));

    eigen_minc = m.template minCoeff<PropagateNaN>(&eigen_minrow, &eigen_mincol);
    eigen_maxc = m.template maxCoeff<PropagateNaN>(&eigen_maxrow, &eigen_maxcol);
    VERIFY(eigen_minrow == 0);
    VERIFY(eigen_maxrow == 0);
    VERIFY(eigen_mincol == 0);
    VERIFY(eigen_maxcol == 0);
    VERIFY((numext::isnan)(eigen_minc));
    VERIFY((numext::isnan)(eigen_maxc));

    eigen_minc = m.template minCoeff<PropagateFast>(&eigen_minrow, &eigen_mincol);
    eigen_maxc = m.template maxCoeff<PropagateFast>(&eigen_maxrow, &eigen_maxcol);
    VERIFY(eigen_minrow == 0);
    VERIFY(eigen_maxrow == 0);
    VERIFY(eigen_mincol == 0);
    VERIFY(eigen_maxcol == 0);
    VERIFY((numext::isnan)(eigen_minc));
    VERIFY((numext::isnan)(eigen_maxc));
  }
}
template <typename MatrixType>
void matrixVisitor(const MatrixType& p) {
  MatrixType m(p.rows(), p.cols());
  // construct a random matrix where all coefficients are different
  m.setRandom();
  for (Index i = 0; i < m.size(); i++)
    for (Index i2 = 0; i2 < i; i2++)
      while (numext::equal_strict(m(i), m(i2)))  // yes, strict equality
        m(i) = internal::random<typename DenseBase<MatrixType>::Scalar>();
  MatrixType n = m;
  matrixVisitor_impl(m);
  // force outer-inner access pattern
  using BlockType = Block<MatrixType, Dynamic, Dynamic>;
  BlockType m_block = n.block(0, 0, n.rows(), n.cols());
  matrixVisitor_impl(m_block);
}

template <typename VectorType>
void vectorVisitor(const VectorType& w) {
  typedef typename VectorType::Scalar Scalar;

  Index size = w.size();

  // construct a random vector where all coefficients are different
  VectorType v;
  v = VectorType::Random(size);
  for (Index i = 0; i < size; i++)
    for (Index i2 = 0; i2 < i; i2++)
      while (v(i) == v(i2))  // yes, ==
        v(i) = internal::random<Scalar>();

  Scalar minc = v(0), maxc = v(0);
  Index minidx = 0, maxidx = 0;
  for (Index i = 0; i < size; i++) {
    if (v(i) < minc) {
      minc = v(i);
      minidx = i;
    }
    if (v(i) > maxc) {
      maxc = v(i);
      maxidx = i;
    }
  }
  Index eigen_minidx, eigen_maxidx;
  Scalar eigen_minc, eigen_maxc;
  eigen_minc = v.minCoeff(&eigen_minidx);
  eigen_maxc = v.maxCoeff(&eigen_maxidx);
  VERIFY(minidx == eigen_minidx);
  VERIFY(maxidx == eigen_maxidx);
  VERIFY_IS_APPROX(minc, eigen_minc);
  VERIFY_IS_APPROX(maxc, eigen_maxc);
  VERIFY_IS_APPROX(minc, v.minCoeff());
  VERIFY_IS_APPROX(maxc, v.maxCoeff());

  Index idx0 = internal::random<Index>(0, size - 1);
  Index idx1 = eigen_minidx;
  Index idx2 = eigen_maxidx;
  VectorType v1(v), v2(v);
  v1(idx0) = v1(idx1);
  v2(idx0) = v2(idx2);
  v1.minCoeff(&eigen_minidx);
  v2.maxCoeff(&eigen_maxidx);
  VERIFY(eigen_minidx == (std::min)(idx0, idx1));
  VERIFY(eigen_maxidx == (std::min)(idx0, idx2));

  if (!NumTraits<Scalar>::IsInteger && size > 2) {
    // Test NaN propagation by replacing an element with NaN.
    for (Index i = 0; i < size; ++i) {
      if (i != minidx && i != maxidx) {
        v(i) = NumTraits<Scalar>::quiet_NaN();
        break;
      }
    }
    eigen_minc = v.template minCoeff<PropagateNumbers>(&eigen_minidx);
    eigen_maxc = v.template maxCoeff<PropagateNumbers>(&eigen_maxidx);
    VERIFY(minidx == eigen_minidx);
    VERIFY(maxidx == eigen_maxidx);
    VERIFY_IS_APPROX(minc, eigen_minc);
    VERIFY_IS_APPROX(maxc, eigen_maxc);
    VERIFY_IS_APPROX(minc, v.template minCoeff<PropagateNumbers>());
    VERIFY_IS_APPROX(maxc, v.template maxCoeff<PropagateNumbers>());

    eigen_minc = v.template minCoeff<PropagateNaN>(&eigen_minidx);
    eigen_maxc = v.template maxCoeff<PropagateNaN>(&eigen_maxidx);
    VERIFY(minidx != eigen_minidx);
    VERIFY(maxidx != eigen_maxidx);
    VERIFY((numext::isnan)(eigen_minc));
    VERIFY((numext::isnan)(eigen_maxc));
  }
}

template <typename Derived, bool Vectorizable, bool Linear = false>
struct TrackedVisitor {
  using Scalar = typename DenseBase<Derived>::Scalar;
  static constexpr int PacketSize = Eigen::internal::packet_traits<Scalar>::size;
  static constexpr bool RowMajor = Derived::IsRowMajor;

  explicit TrackedVisitor(const Derived& input) : matrix(input) {}

  void init(Scalar v, Index i, Index j) {
    ++initCalls;
    this->operator()(v, i, j);
  }
  template <typename Packet>
  void initpacket(Packet p, Index i, Index j) {
    ++initCalls;
    this->packet(p, i, j);
  }
  void operator()(Scalar v, Index i, Index j) {
    VERIFY_IS_EQUAL(v, matrix.coeff(i, j));
    visited.emplace_back(i, j);
    scalarOps++;
  }

  template <typename Packet>
  void packet(Packet p, Index i, Index j) {
    Scalar values[PacketSize];
    internal::pstoreu(values, p);
    for (int k = 0; k < PacketSize; k++) {
      VERIFY_IS_EQUAL(values[k], matrix.coeff(RowMajor ? i : i + k, RowMajor ? j + k : j));
      if (RowMajor)
        visited.emplace_back(i, j + k);
      else
        visited.emplace_back(i + k, j);
    }
    vectorOps++;
  }

  void init(Scalar v, Index index) {
    ++initCalls;
    this->operator()(v, index);
  }
  void operator()(Scalar v, Index index) {
    VERIFY_IS_EQUAL(v, matrix.coeff(index));
    recordIndex(index);
    ++scalarOps;
  }
  template <typename Packet>
  void initpacket(Packet p, Index index) {
    ++initCalls;
    this->packet(p, index);
  }
  template <typename Packet>
  void packet(Packet p, Index index) {
    Scalar values[PacketSize];
    internal::pstoreu(values, p);
    for (int k = 0; k < PacketSize; ++k) {
      VERIFY_IS_EQUAL(values[k], matrix.coeff(index + k));
      recordIndex(index + k);
    }
    ++vectorOps;
  }
  void recordIndex(Index index) {
    const Index innerSize = RowMajor ? matrix.cols() : matrix.rows();
    const Index inner = index % innerSize;
    const Index outer = index / innerSize;
    visited.emplace_back(RowMajor ? outer : inner, RowMajor ? inner : outer);
  }
  const Derived& matrix;
  std::vector<std::pair<Index, Index>> visited;
  Index initCalls = 0;
  Index scalarOps = 0;
  Index vectorOps = 0;
};

namespace Eigen {
namespace internal {

template <typename T, bool Vectorizable>
struct functor_traits<TrackedVisitor<T, Vectorizable>> {
  static constexpr bool PacketAccess = Vectorizable;
  static constexpr int Cost = 1;
};

template <typename T>
struct functor_traits<TrackedVisitor<T, false>> {
  static constexpr bool PacketAccess = false;
  static constexpr bool LinearAccess = false;
  static constexpr int Cost = 1;
};

template <typename T, bool Vectorizable>
struct functor_traits<TrackedVisitor<T, Vectorizable, true>> {
  static constexpr bool PacketAccess = Vectorizable;
  static constexpr bool LinearAccess = true;
  static constexpr int Cost = 1;
};

}  // namespace internal
}  // namespace Eigen

template <typename Derived, bool Vectorized, bool Linear = false>
void checkTraversal(const DenseBase<Derived>& mat) {
  using Scalar = typename DenseBase<Derived>::Scalar;
  static constexpr int PacketSize = Eigen::internal::packet_traits<Scalar>::size;
  static constexpr bool RowMajor = Derived::IsRowMajor;
  const Derived& X = mat.derived();
  using Visitor = TrackedVisitor<Derived, Vectorized, Linear>;
  using VisitImpl = Eigen::internal::visit_impl<Derived, Visitor, false>;
  STATIC_CHECK((internal::visitor_has_linear_access<Visitor>::value == Linear));
  STATIC_CHECK((VisitImpl::LinearAccess == (Linear && bool(internal::evaluator<Derived>::Flags & LinearAccessBit))));
  Visitor visitor(X);
  visitor.visited.reserve(X.size());
  X.visit(visitor);
  VERIFY_IS_EQUAL(visitor.visited.size(), static_cast<std::size_t>(X.size()));
  VERIFY_IS_EQUAL(visitor.initCalls, X.size() == 0 ? 0 : 1);
  Index count = 0;
  for (Index j = 0; j < X.outerSize(); ++j) {
    for (Index i = 0; i < X.innerSize(); ++i) {
      Index r = RowMajor ? j : i;
      Index c = RowMajor ? i : j;
      VERIFY_IS_EQUAL(visitor.visited[count].first, r);
      VERIFY_IS_EQUAL(visitor.visited[count].second, c);
      ++count;
    }
  }
  Index vectorOps = VisitImpl::Vectorize ? (VisitImpl::LinearAccess ? X.size() / PacketSize
                                                                    : (X.innerSize() / PacketSize) * X.outerSize())
                                         : 0;
  Index scalarOps = X.size() - (vectorOps * PacketSize);
  VERIFY_IS_EQUAL(vectorOps, visitor.vectorOps);
  VERIFY_IS_EQUAL(scalarOps, visitor.scalarOps);
}

template <typename Derived, bool Vectorized, bool Linear = false>
void checkOptimalTraversal_impl(const DenseBase<Derived>& mat) {
  Derived matrix = Derived::Random(mat.rows(), mat.cols());
  checkTraversal<Derived, Vectorized, Linear>(matrix);
}

template <int Options>
void checkLinearTraversal() {
  using MatrixType = Matrix<float, Dynamic, Dynamic, Options>;
  constexpr int PacketSize = internal::packet_traits<float>::size;
  using FixedMatrix =
      Matrix<float, Options == RowMajor ? 2 : PacketSize + 1, Options == RowMajor ? PacketSize + 1 : 2, Options>;
  using FixedVisitor = TrackedVisitor<FixedMatrix, true, true>;
  STATIC_CHECK((internal::visit_impl<FixedMatrix, FixedVisitor, false>::Unroll));
  checkOptimalTraversal_impl<FixedMatrix, false, true>(FixedMatrix());
  checkOptimalTraversal_impl<FixedMatrix, true, true>(FixedMatrix());
  checkOptimalTraversal_impl<Matrix<float, 0, 0, Options>, true, true>(Matrix<float, 0, 0, Options>());

  const Index sizes[] = {0, 1, PacketSize - 1, PacketSize, PacketSize + 1, 2 * PacketSize + 1};
  for (Index inner : sizes) {
    const Index rows = Options == RowMajor ? 3 : inner;
    const Index cols = Options == RowMajor ? inner : 3;
    MatrixType matrix = MatrixType::Random(rows, cols);
    checkTraversal<MatrixType, false, true>(matrix);
    checkTraversal<MatrixType, true, true>(matrix);

    MatrixType storage = MatrixType::Random(rows + 2, cols + 2);
    auto block = storage.block(1, 1, rows, cols);
    STATIC_CHECK(!(internal::evaluator<decltype(block)>::Flags & LinearAccessBit));
    checkTraversal<decltype(block), true, true>(block);
    checkTraversal<decltype(block), false, false>(block);

    using UnalignedMap = Map<MatrixType, Unaligned>;
    UnalignedMap unaligned(storage.data() + 1, rows, cols);
    checkTraversal<UnalignedMap, true, true>(unaligned);
  }

  VectorXf storage = VectorXf::Random(6 * PacketSize + 4);
  using StridedMap = Map<VectorXf, Unaligned, InnerStride<2>>;
  StridedMap strided(storage.data(), 3 * PacketSize + 2);
  checkTraversal<StridedMap, true, true>(strided);
}

template <typename Derived>
void checkBooleanVisitors(const DenseBase<Derived>& matrix) {
  using Scalar = typename Derived::Scalar;
  Index count = 0;
  for (Index col = 0; col < matrix.cols(); ++col)
    for (Index row = 0; row < matrix.rows(); ++row) count += matrix(row, col) != Scalar(0);
  VERIFY_IS_EQUAL(matrix.count(), count);
  VERIFY_IS_EQUAL(matrix.all(), count == matrix.size());
  VERIFY_IS_EQUAL(matrix.any(), count != 0);
}

template <typename Scalar, int Options>
void checkBooleanVisitorTraversal() {
  STATIC_CHECK((internal::visitor_has_linear_access<internal::all_visitor<Scalar>>::value));
  STATIC_CHECK((internal::visitor_has_linear_access<internal::any_visitor<Scalar>>::value));
  STATIC_CHECK((internal::visitor_has_linear_access<internal::count_visitor<Scalar>>::value));
  constexpr int PacketSize = internal::packet_traits<Scalar>::size;
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Options>;
  const Index sizes[] = {0, 1, PacketSize - 1, PacketSize, PacketSize + 1, 2 * PacketSize + 1};
  for (Index inner : sizes) {
    MatrixType matrix(Options == RowMajor ? 3 : inner, Options == RowMajor ? inner : 3);
    for (int value = 0; value <= 1; ++value) {
      matrix.setConstant(Scalar(value));
      checkBooleanVisitors(matrix);
      for (Index index = 0; index < matrix.size(); ++index) {
        matrix(index) = Scalar(!value);
        checkBooleanVisitors(matrix);
        checkBooleanVisitors(matrix.block(0, 0, matrix.rows(), matrix.cols()));
        checkBooleanVisitors(matrix.array() != Scalar(0));
        matrix(index) = Scalar(value);
      }
    }
  }
  Matrix<Scalar, PacketSize + 1, 2, Options> fixed;
  fixed.setOnes();
  checkBooleanVisitors(fixed);
  fixed(fixed.size() - 1) = Scalar(0);
  checkBooleanVisitors(fixed);
}

template <bool Vectorize = false>
struct CountVisitorReads {
  Index* reads;
  float operator()(float value) const {
    ++*reads;
    return value;
  }
  template <typename Packet>
  Packet packetOp(const Packet& value) const {
    *reads += internal::unpacket_traits<Packet>::size;
    return value;
  }
};

namespace Eigen {
namespace internal {
template <bool Vectorize>
struct functor_traits<CountVisitorReads<Vectorize>> {
  static constexpr int Cost = 1;
  static constexpr bool PacketAccess = Vectorize;
};
}  // namespace internal
}  // namespace Eigen

template <int Options, bool Vectorize = false>
void checkVisitorShortCircuit() {
  using MatrixType = Matrix<float, Dynamic, Dynamic, Options>;
  const Index inner = Vectorize ? 1 : 7;
  const Index outer = 35;
  MatrixType matrix(Options == RowMajor ? outer : inner, Options == RowMajor ? inner : outer);
  for (Index index : {Index(0), Index(1), matrix.size() - 1}) {
    Index reads = 0;
    auto counted = matrix.unaryExpr(CountVisitorReads<Vectorize>{&reads});
    for (int value = 0; value <= 1; ++value) {
      matrix.setConstant(float(value));
      matrix(index) = float(!value);
      reads = 0;
      VERIFY_IS_EQUAL(value ? counted.all() : counted.any(), !value);
      VERIFY_IS_EQUAL(reads, index + 1);
      reads = 0;
      auto block = counted.block(0, 0, matrix.rows(), matrix.cols());
      VERIFY_IS_EQUAL(value ? block.all() : block.any(), !value);
      VERIFY_IS_EQUAL(reads, index + 1);
    }
  }
}

void checkOptimalTraversal() {
  using Scalar = float;
  constexpr int PacketSize = Eigen::internal::packet_traits<Scalar>::size;
  // use sizes that mix vector and scalar ops
  constexpr int Rows = PacketSize + 1;
  constexpr int Cols = 2;
  int rows = internal::random(PacketSize + 1, EIGEN_TEST_MAX_SIZE);
  int cols = internal::random(PacketSize + 1, EIGEN_TEST_MAX_SIZE);

  using UnrollColMajor = Matrix<Scalar, Rows, Cols, ColMajor>;
  using UnrollRowMajor = Matrix<Scalar, Cols, Rows, RowMajor>;
  using DynamicColMajor = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using DynamicRowMajor = Matrix<Scalar, Dynamic, Dynamic, RowMajor>;

  // Scalar-only visitors
  checkOptimalTraversal_impl<UnrollColMajor, false>(UnrollColMajor(Rows, Cols));
  checkOptimalTraversal_impl<UnrollRowMajor, false>(UnrollRowMajor(Cols, Rows));
  checkOptimalTraversal_impl<DynamicColMajor, false>(DynamicColMajor(rows, cols));
  checkOptimalTraversal_impl<DynamicRowMajor, false>(DynamicRowMajor(rows, cols));

  // Vectorized visitors
  checkOptimalTraversal_impl<UnrollColMajor, true>(UnrollColMajor(Rows, Cols));
  checkOptimalTraversal_impl<UnrollRowMajor, true>(UnrollRowMajor(Cols, Rows));
  checkOptimalTraversal_impl<DynamicColMajor, true>(DynamicColMajor(rows, cols));
  checkOptimalTraversal_impl<DynamicRowMajor, true>(DynamicRowMajor(rows, cols));

  const Eigen::Array<bool, Eigen::Dynamic, 1> a = Eigen::Array<bool, 2, 1>{false, true};
  Eigen::Index i = -1;

  VERIFY(!a.minCoeff(&i));
  VERIFY(i == 0);

  VERIFY(!(!a).minCoeff(&i));
  VERIFY(i == 1);

  Eigen::Index j = -1;

  VERIFY(a.maxCoeff(&j));
  VERIFY(j == 1);

  VERIFY((!a).maxCoeff(&j));
  VERIFY(j == 0);
}

// Test minCoeff/maxCoeff at vectorization boundary sizes.
// Visitor uses LinearVectorizedTraversal with packet-based min/max,
// so we test at sizes around packet multiples.
template <typename Scalar>
void visitor_vec_boundary() {
  const Index PS = internal::packet_traits<Scalar>::size;
  const Index sizes[] = {1, 2, 3, PS - 1, PS, PS + 1, 2 * PS - 1, 2 * PS, 2 * PS + 1, 4 * PS, 4 * PS + 1};
  for (int si = 0; si < 11; ++si) {
    const Index n = sizes[si];
    if (n <= 0) continue;
    typedef Matrix<Scalar, Dynamic, 1> Vec;
    Vec v = Vec::Random(n);
    // Ensure all elements are distinct.
    for (Index i = 0; i < n; ++i)
      for (Index j = 0; j < i; ++j)
        while (numext::equal_strict(v(i), v(j))) v(i) = internal::random<Scalar>();
    // Reference
    Scalar ref_min = v(0), ref_max = v(0);
    Index ref_minidx = 0, ref_maxidx = 0;
    for (Index k = 0; k < n; ++k) {
      if (v(k) < ref_min) {
        ref_min = v(k);
        ref_minidx = k;
      }
      if (v(k) > ref_max) {
        ref_max = v(k);
        ref_maxidx = k;
      }
    }
    Index eigen_minidx, eigen_maxidx;
    VERIFY_IS_APPROX(v.minCoeff(&eigen_minidx), ref_min);
    VERIFY_IS_APPROX(v.maxCoeff(&eigen_maxidx), ref_max);
    VERIFY(eigen_minidx == ref_minidx);
    VERIFY(eigen_maxidx == ref_maxidx);

    // Also test matrix form at this size (exercises different inner/outer sizes).
    if (n >= 2) {
      typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
      // Test as n×1 and 1×n (different inner sizes for visitor traversal).
      Mat mc = v;
      Mat mr = v.transpose();
      Index ri, ci;
      VERIFY_IS_APPROX(mc.minCoeff(&ri, &ci), ref_min);
      VERIFY(ri == ref_minidx && ci == 0);
      VERIFY_IS_APPROX(mr.minCoeff(&ri, &ci), ref_min);
      VERIFY(ri == 0 && ci == ref_minidx);
    }
  }
}

EIGEN_DECLARE_TEST(visitor) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(matrixVisitor(Matrix<float, 1, 1>()));
    CALL_SUBTEST_2(matrixVisitor(Matrix2f()));
    CALL_SUBTEST_3(matrixVisitor(Matrix4d()));
    CALL_SUBTEST_4(matrixVisitor(MatrixXd(8, 12)));
    CALL_SUBTEST_5(matrixVisitor(Matrix<double, Dynamic, Dynamic, RowMajor>(20, 20)));
    CALL_SUBTEST_6(matrixVisitor(MatrixXi(8, 12)));
  }
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_7(vectorVisitor(Vector4f()));
    CALL_SUBTEST_7(vectorVisitor(Matrix<int, 12, 1>()));
    CALL_SUBTEST_8(vectorVisitor(VectorXd(10)));
    CALL_SUBTEST_9(vectorVisitor(RowVectorXd(10)));
    CALL_SUBTEST_10(vectorVisitor(VectorXf(33)));
  }
  CALL_SUBTEST_11(checkOptimalTraversal());

  // Vectorization boundary sizes — deterministic, run once.
  CALL_SUBTEST_12(visitor_vec_boundary<float>());
  CALL_SUBTEST_12(visitor_vec_boundary<double>());
  CALL_SUBTEST_12(visitor_vec_boundary<int>());

  CALL_SUBTEST_13(checkLinearTraversal<ColMajor>());
  CALL_SUBTEST_13(checkLinearTraversal<RowMajor>());
  CALL_SUBTEST_14((checkBooleanVisitorTraversal<bool, ColMajor>()));
  CALL_SUBTEST_14((checkBooleanVisitorTraversal<bool, RowMajor>()));
  CALL_SUBTEST_14((checkBooleanVisitorTraversal<float, ColMajor>()));
  CALL_SUBTEST_14((checkBooleanVisitorTraversal<double, RowMajor>()));
  CALL_SUBTEST_14((checkBooleanVisitorTraversal<int, ColMajor>()));
  CALL_SUBTEST_14((checkBooleanVisitorTraversal<std::complex<float>, RowMajor>()));
  CALL_SUBTEST_14(checkVisitorShortCircuit<ColMajor>());
  CALL_SUBTEST_14(checkVisitorShortCircuit<RowMajor>());
  CALL_SUBTEST_14((checkVisitorShortCircuit<ColMajor, true>()));
  CALL_SUBTEST_14((checkVisitorShortCircuit<RowMajor, true>()));
}
