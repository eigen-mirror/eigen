// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include "fp_control.h"

struct ZeroOnlyScalar {
  double value;
  ZeroOnlyScalar(double x = 0) : value(x) {}
  friend ZeroOnlyScalar abs(const ZeroOnlyScalar& x) { return numext::abs(x.value); }
  friend ZeroOnlyScalar operator*(const ZeroOnlyScalar& x, const ZeroOnlyScalar& y) { return x.value * y.value; }
  friend bool operator<=(const ZeroOnlyScalar& x, const ZeroOnlyScalar& y) { return x.value <= y.value; }
};

struct ApproxOnlyScalar {
  double value;
  ApproxOnlyScalar(double x = 0) : value(x) {}
};

namespace Eigen {
template <>
struct NumTraits<ZeroOnlyScalar> : GenericNumTraits<ZeroOnlyScalar> {};
template <>
struct NumTraits<ApproxOnlyScalar> : GenericNumTraits<ApproxOnlyScalar> {};
namespace internal {
template <>
struct scalar_fuzzy_impl<ApproxOnlyScalar> {
  static bool isApprox(const ApproxOnlyScalar& x, const ApproxOnlyScalar& y, const ApproxOnlyScalar& precision) {
    return internal::isApprox(x.value, y.value, precision.value);
  }
};
}  // namespace internal
}  // namespace Eigen

void check_fuzzy_custom_scalars() {
  // Neither operation may instantiate the other scalar predicate, including in C++14.
  Matrix<ZeroOnlyScalar, 2, 1> zero;
  zero(0) = 0;
  zero(1) = 0;
  VERIFY(zero.isZero(ZeroOnlyScalar(1e-6)));
  zero(1) = 1;
  VERIFY(!zero.isZero(ZeroOnlyScalar(1e-6)));
  Matrix<ZeroOnlyScalar, Dynamic, 1> dynamicZero = zero;
  VERIFY(!dynamicZero.isZero(ZeroOnlyScalar(1e-6)));
  dynamicZero.resize(0);
  VERIFY(dynamicZero.isZero(ZeroOnlyScalar(1e-6)));

  Matrix<ApproxOnlyScalar, 2, 1> constant;
  constant(0) = 0.5;
  constant(1) = 0.5;
  VERIFY(constant.isApproxToConstant(ApproxOnlyScalar(0.5), ApproxOnlyScalar(1e-6)));
  constant(0) = 1;
  VERIFY(!constant.isApproxToConstant(ApproxOnlyScalar(0.5), ApproxOnlyScalar(1e-6)));
  Matrix<ApproxOnlyScalar, Dynamic, 1> dynamicConstant = constant;
  VERIFY(!dynamicConstant.isApproxToConstant(ApproxOnlyScalar(0.5), ApproxOnlyScalar(1e-6)));
  dynamicConstant.resize(0);
  VERIFY(dynamicConstant.isApproxToConstant(ApproxOnlyScalar(0.5), ApproxOnlyScalar(1e-6)));
}

template <typename Scalar, int Order>
void check_fuzzy_loops() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, Order>;
  const Scalar nan = NumTraits<Scalar>::quiet_NaN();
  for (Index rows : {0, 1, 3, 17, 31, 32, 33}) {
    for (Index cols : {0, 1, 5, 19, 31, 32, 33}) {
      Mat a = Mat::Zero(rows, cols);
      for (int variant = 0; variant < 6; ++variant) {
        if (variant == 1) a.setIdentity();
        if (variant == 2) a.setConstant(Scalar(0.5));
        if (variant == 3 && a.size()) a(a.rows() - 1, a.cols() - 1) = nan;
        if (variant == 4) a.setConstant(nan);
        if (variant == 5) a.setConstant(NumTraits<Scalar>::infinity());
        for (Scalar precision : {Scalar(0), Scalar(0.1), Scalar(-0.1), nan, NumTraits<Scalar>::infinity()}) {
          bool zero = true, constant = true, upper = true, lower = true;
          Scalar upperMax = Scalar(-1), lowerMax = Scalar(-1);
          for (Index j = 0; j < cols; ++j) {
            for (Index i = 0; i < rows; ++i) {
              zero &= internal::isMuchSmallerThan(a(i, j), Scalar(1), precision);
              constant &= internal::isApprox(a(i, j), Scalar(0.5), precision);
              const Scalar value = numext::abs(a(i, j));
              if (i <= j && value > upperMax) upperMax = value;
              if (i >= j && value > lowerMax) lowerMax = value;
            }
          }
          for (Index j = 0; j < cols; ++j)
            for (Index i = 0; i < rows; ++i) {
              if (i > j && numext::abs(a(i, j)) > upperMax * precision) upper = false;
              if (i < j && numext::abs(a(i, j)) > lowerMax * precision) lower = false;
            }
          VERIFY_IS_EQUAL(a.isZero(precision), zero);
          VERIFY_IS_EQUAL(a.isApproxToConstant(Scalar(0.5), precision), constant);
          VERIFY_IS_EQUAL(a.isUpperTriangular(precision), upper);
          VERIFY_IS_EQUAL(a.isLowerTriangular(precision), lower);
        }
      }
    }
  }
}

template <typename Scalar, int Order>
void check_fuzzy_boundaries() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using Visitor = internal::fuzzy_constant_visitor<Scalar, false>;
  STATIC_CHECK((internal::visit_impl<Mat, Visitor, true>::LinearAccess));
  STATIC_CHECK((!internal::visit_impl<Block<Mat>, Visitor, true>::LinearAccess));
  constexpr int packetSize = internal::packet_traits<Scalar>::size;
  for (Index size :
       {Index(1), Index(packetSize - 1), Index(packetSize), Index(packetSize + 1), Index(2 * packetSize + 1)}) {
    // Unaligned linear access, including a packet crossing a column boundary.
    std::vector<Scalar> data(3 * size + 1);
    Map<Mat, Unaligned> matrix(data.data() + 1, 3, size);
    for (Index k = 0; k < matrix.size(); ++k) {
      matrix.setZero();
      VERIFY(matrix.isZero());
      matrix(k) = NumTraits<Scalar>::quiet_NaN();
      VERIFY(!matrix.isZero());
      matrix.setConstant(Scalar(0.5));
      VERIFY(matrix.isApproxToConstant(Scalar(0.5)));
      matrix(k) = Scalar(1);
      VERIFY(!matrix.isApproxToConstant(Scalar(0.5)));
    }
  }
  for (Index inner : {Index(1), Index(packetSize), Index(packetSize + 1), Index(2 * packetSize + 1)}) {
    Mat storage = Mat::Zero(inner + 2, inner + 2);
    auto block = storage.block(1, 1, inner, inner);
    for (Index k = 0; k < inner * inner; ++k) {
      block.setZero();
      VERIFY(block.isZero());
      block(k / inner, k % inner) = Scalar(1);
      VERIFY(!block.isZero());
      block.setConstant(Scalar(0.5));
      VERIFY(block.isApproxToConstant(Scalar(0.5)));
      block(k / inner, k % inner) = Scalar(1);
      VERIFY(!block.isApproxToConstant(Scalar(0.5)));
    }
  }
  Matrix<Scalar, 3, 3, Order> fixed = Matrix<Scalar, 3, 3, Order>::Zero();
  VERIFY(fixed.isZero());
  fixed(2, 2) = Scalar(1);
  VERIFY(!fixed.isZero());
  for (Index k = 0; k < fixed.size(); ++k) {
    fixed.setZero();
    fixed(k) = Scalar(1);
    VERIFY(!fixed.isZero());
    fixed.setConstant(Scalar(0.5));
    fixed(k) = Scalar(1);
    VERIFY(!fixed.isApproxToConstant(Scalar(0.5)));
  }
  Matrix<Scalar, packetSize + 1, 1> packetTail;
  for (Index k = 0; k < packetTail.size(); ++k) {
    packetTail.setZero();
    packetTail(k) = Scalar(1);
    VERIFY(!packetTail.isZero());
    packetTail.setConstant(Scalar(0.5));
    packetTail(k) = Scalar(1);
    VERIFY(!packetTail.isApproxToConstant(Scalar(0.5)));
  }
  Scalar data[18] = {};
  Map<Matrix<Scalar, Dynamic, 1>, Unaligned, InnerStride<2>> strided(data, 9);
  VERIFY(strided.isZero());
  strided(8) = Scalar(1);
  VERIFY(!strided.isZero());
}

template <typename Scalar, int Order>
void check_fuzzy_subnormals() {
  const ScopedFlushToZero gradual(FlushToZeroMode::None);
  // Require gradual scalar arithmetic, even if the packet unit flushes (ARMv7 NEON).
  if (subnormalInputProbe<Scalar>() == Scalar(0) || underflowProbe<Scalar>() == Scalar(0)) return;
  const Scalar subnormal = (std::numeric_limits<Scalar>::min)() / Scalar(4);
  using Mat = Matrix<Scalar, Dynamic, Dynamic, Order>;
  using Visitor = internal::fuzzy_constant_visitor<Scalar, false>;
  STATIC_CHECK(
      (!EIGEN_ARCH_ARM || !std::is_same<Scalar, float>::value || !internal::functor_traits<Visitor>::PacketAccess));
  for (Index size : {1, 2, 4, 8, 9, 17}) {
    Matrix<Scalar, Dynamic, 1> v = Matrix<Scalar, Dynamic, 1>::Zero(size);
    for (Index k = 0; k < size; ++k) {
      for (Scalar sign : {Scalar(1), Scalar(-1)}) {
        v.setZero();
        v(k) = sign * subnormal;
        VERIFY(!v.isZero(Scalar(0)));
        VERIFY(!v.isConstant(Scalar(0)));
        VERIFY(v.isZero(subnormal));
        VERIFY(!v.isZero(subnormal / Scalar(2)));
      }
    }
  }
  Mat c = Mat::Constant(4, 4, Scalar(2) * subnormal);
  VERIFY(!c.isApproxToConstant(subnormal, Scalar(0)));
  VERIFY(!c.isZero(subnormal));
  VERIFY(c.isApproxToConstant(Scalar(2) * subnormal, Scalar(0)));
  VERIFY(c.isZero(Scalar(2) * subnormal));
  VERIFY(!c.block(0, 0, 3, 3).isZero(subnormal));
  VERIFY(!c.block(0, 0, 3, 3).isApproxToConstant(subnormal, Scalar(0)));
  Matrix<Scalar, 3, 3, Order> fixed = c.template topLeftCorner<3, 3>();
  VERIFY(!fixed.isZero(subnormal));
  VERIFY(!fixed.isApproxToConstant(subnormal, Scalar(0)));

  // Normal operands can produce a subnormal difference in the approximate comparison.
  const Scalar normal = (std::numeric_limits<Scalar>::min)();
  c.setConstant(normal + subnormal);
  VERIFY(!c.isApproxToConstant(normal, Scalar(0)));
  VERIFY(c.isApproxToConstant(normal + subnormal, Scalar(0)));
}

EIGEN_DECLARE_TEST(fuzzy_matrix_checks) {
  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    CALL_SUBTEST_1((check_fuzzy_boundaries<float, ColMajor>()));
    CALL_SUBTEST_1((check_fuzzy_loops<float, ColMajor>()));
    CALL_SUBTEST_1((check_fuzzy_subnormals<float, ColMajor>()));
    CALL_SUBTEST_2((check_fuzzy_boundaries<double, ColMajor>()));
    CALL_SUBTEST_2((check_fuzzy_loops<double, ColMajor>()));
    CALL_SUBTEST_2((check_fuzzy_subnormals<double, ColMajor>()));
    CALL_SUBTEST_3((check_fuzzy_boundaries<float, RowMajor>()));
    CALL_SUBTEST_3((check_fuzzy_loops<float, RowMajor>()));
    CALL_SUBTEST_3((check_fuzzy_subnormals<float, RowMajor>()));
    CALL_SUBTEST_4((check_fuzzy_boundaries<double, RowMajor>()));
    CALL_SUBTEST_4((check_fuzzy_loops<double, RowMajor>()));
    CALL_SUBTEST_4((check_fuzzy_subnormals<double, RowMajor>()));
    CALL_SUBTEST_5(check_fuzzy_custom_scalars());
  }
}
