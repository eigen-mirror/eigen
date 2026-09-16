// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_RUNTIME_NO_MALLOC
#include "main.h"
#include <Eigen/Core>

template <unsigned int UpLo, bool Hessenberg, typename Derived>
void check_structured_abs_sum(const MatrixBase<Derived>& matrix) {
  using RealScalar = typename Derived::RealScalar;
  long double expected = 0;
  for (Index row = 0; row < matrix.rows(); ++row) {
    for (Index col = 0; col < matrix.cols(); ++col) {
      if (UpLo == Upper ? row - col <= Index(Hessenberg) : col - row <= Index(Hessenberg)) {
        const long double real = static_cast<long double>(numext::real(matrix(row, col)));
        const long double imag = static_cast<long double>(numext::imag(matrix(row, col)));
        expected += std::sqrt(real * real + imag * imag);
      }
    }
  }
  internal::set_is_malloc_allowed(false);
  const RealScalar actual =
      Hessenberg ? internal::hessenberg_abs_sum<UpLo>(matrix) : internal::triangular_abs_sum<UpLo>(matrix);
  internal::set_is_malloc_allowed(true);
  // Each magnitude and addition contributes O(eps); the reference accumulates in long double.
  const long double bound = 8 * static_cast<long double>(matrix.size()) *
                            static_cast<long double>(NumTraits<RealScalar>::epsilon()) * expected;
  VERIFY((numext::isfinite)(actual));
  VERIFY(std::abs(static_cast<long double>(actual) - expected) <= bound);
}

template <typename Derived>
void check_structured_abs_sums(const MatrixBase<Derived>& matrix) {
  check_structured_abs_sum<Upper, false>(matrix);
  check_structured_abs_sum<Lower, false>(matrix);
  check_structured_abs_sum<Upper, true>(matrix);
  check_structured_abs_sum<Lower, true>(matrix);
}

template <typename Scalar, int StorageOrder>
void structured_abs_sums() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, StorageOrder>;
  const Matrix<Scalar, 3, 5, StorageOrder> fixed = Matrix<Scalar, 3, 5, StorageOrder>::Random();
  check_structured_abs_sums(fixed);
  check_structured_abs_sums(fixed.transpose());
  for (Index rows : {0, 1, 2, 7, 16, 17, 33}) {
    for (Index cols : {0, 1, 2, 7, 16, 17, 33}) {
      Mat matrix = Mat::Random(rows, cols);
      check_structured_abs_sums(matrix);
      check_structured_abs_sums(matrix.transpose());
      check_structured_abs_sums(matrix * Scalar(2));
      Mat storage = Mat::Random(rows + 2, cols + 3);
      check_structured_abs_sums(storage.block(1, 2, rows, cols));
      using StridedMap = Map<const Mat, Unaligned, Stride<Dynamic, 2>>;
      Mat stridedStorage = Mat::Random(2 * (rows + 1), 2 * (cols + 1));
      StridedMap strided(stridedStorage.data(), rows, cols, Stride<Dynamic, 2>(stridedStorage.outerStride(), 2));
      check_structured_abs_sums(strided);
    }
  }

  using RealScalar = typename NumTraits<Scalar>::Real;
  const Scalar nan((std::numeric_limits<RealScalar>::quiet_NaN)());
  for (Index extra : {0, 1}) {
    Mat matrix = Mat::Random(17, 17);
    for (Index row = 0; row < matrix.rows(); ++row)
      for (Index col = 0; col + extra < row; ++col) matrix(row, col) = nan;
    if (extra == 0) {
      check_structured_abs_sum<Upper, false>(matrix);
      check_structured_abs_sum<Lower, false>(matrix.transpose());
    } else {
      check_structured_abs_sum<Upper, true>(matrix);
      check_structured_abs_sum<Lower, true>(matrix.transpose());
    }
  }

  Mat matrix = Mat::Zero(17, 17);
  matrix(0, 0) = nan;
  VERIFY((numext::isnan)(internal::triangular_abs_sum<Upper>(matrix)));
  VERIFY((numext::isnan)(internal::hessenberg_abs_sum<Lower>(matrix)));
  matrix(0, 0) = Scalar((std::numeric_limits<RealScalar>::infinity)());
  VERIFY((numext::isinf)(internal::triangular_abs_sum<Lower>(matrix)));
  VERIFY((numext::isinf)(internal::hessenberg_abs_sum<Upper>(matrix)));
}

EIGEN_DECLARE_TEST(structured_norm) {
  CALL_SUBTEST_1((structured_abs_sums<float, ColMajor>()));
  CALL_SUBTEST_1((structured_abs_sums<float, RowMajor>()));
  CALL_SUBTEST_2((structured_abs_sums<double, ColMajor>()));
  CALL_SUBTEST_2((structured_abs_sums<double, RowMajor>()));
  CALL_SUBTEST_3((structured_abs_sums<std::complex<float>, ColMajor>()));
  CALL_SUBTEST_3((structured_abs_sums<std::complex<float>, RowMajor>()));
  CALL_SUBTEST_4((structured_abs_sums<std::complex<double>, ColMajor>()));
  CALL_SUBTEST_4((structured_abs_sums<std::complex<double>, RowMajor>()));
  CALL_SUBTEST_5((structured_abs_sums<long double, ColMajor>()));
  CALL_SUBTEST_5((structured_abs_sums<long double, RowMajor>()));
}
