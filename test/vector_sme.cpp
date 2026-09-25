// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#include "main.h"
#include <Eigen/Core>
#include <cfenv>

#ifndef EIGEN_VECTORIZE_SME
#error "vector_sme requires an SME2 target"
#endif

template <typename Lhs, typename Rhs>
void sme_check_dot(const Lhs& lhs, const Rhs& rhs) {
  using Scalar = typename Lhs::Scalar;
  long double expected = 0, magnitude = 0;
  for (Index i = 0; i < lhs.size(); ++i) {
    const long double term = static_cast<long double>(lhs[i]) * static_cast<long double>(rhs[i]);
    expected += term;
    magnitude += numext::abs(term);
  }
  // gamma_(2*n+4) bounds multiplication and accumulation in both the kernel and reference.
  const long double rounding =
      (2 * static_cast<long double>(lhs.size()) + 4) * static_cast<long double>(NumTraits<Scalar>::epsilon());
  const long double bound = rounding / (1 - rounding) * magnitude;
  const Scalar actual = lhs.dot(rhs);
  VERIFY(rounding < 1 && (numext::isfinite)(bound));
  VERIFY((numext::isfinite)(actual));
  VERIFY(numext::abs(static_cast<long double>(actual) - expected) <= bound);
}

template <typename Scalar>
void sme_vectors() {
  using Vec = Vector<Scalar, Dynamic>;
  using Row = Matrix<Scalar, 1, Dynamic>;
  using Strided = Map<Vec, 0, InnerStride<Dynamic>>;
  STATIC_CHECK((internal::sme_dot_supported<Vec, Row>::value));
  STATIC_CHECK((internal::sme_dot_supported<Strided, Vec>::value));
  STATIC_CHECK((!internal::sme_dot_supported<decltype(std::declval<Vec>() + std::declval<Vec>()), Vec>::value));
  STATIC_CHECK((!internal::sme_vector_scalar<std::complex<Scalar>>::value));
  STATIC_CHECK((!internal::sme_vector_scalar<int>::value));

  std::vector<Index> sizes;
  for (Index n = 0; n <= 260; ++n) {
    sizes.push_back(n);
    sizes.push_back(32768 + n);
  }
  for (Index n : {511,  512,  513,  1023, 1024,  1025,  2047,  2048,  2049,  4095, 4096,
                  4097, 8191, 8192, 8193, 16383, 16384, 16385, 32767, 32768, 32769})
    sizes.push_back(n);
  for (Index n : sizes) {
    Vec x = Vec::Ones(n), storage = Vec::Ones(n + 2);
    VERIFY_IS_EQUAL(x.dot(storage.head(n)), Scalar(n));
    x.setRandom();
    storage.setRandom();
    const Vec before = storage;
    auto y = storage.segment(1, n);
    sme_check_dot(x, y);
    sme_check_dot(x.transpose(), y);
    for (int side = 0; side < 2; ++side) {
      storage = before;
      if (side == 0)
        y += Scalar(0.75) * x;
      else
        y += x * Scalar(0.75);
      for (Index i = 0; i < n; ++i) {
        const long double expected = static_cast<long double>(before[i + 1]) + 0.75L * static_cast<long double>(x[i]);
        const long double magnitude =
            numext::abs(static_cast<long double>(before[i + 1])) + 0.75L * numext::abs(static_cast<long double>(x[i]));
        VERIFY(numext::abs(static_cast<long double>(y[i]) - expected) <=
               4 * static_cast<long double>(NumTraits<Scalar>::epsilon()) * magnitude);
      }
      VERIFY_IS_EQUAL(storage[0], before[0]);
      VERIFY_IS_EQUAL(storage[n + 1], before[n + 1]);
    }
    Vec alias = x;
    alias += Scalar(0.75) * alias;
    for (Index i = 0; i < n; ++i) {
      const long double expected = static_cast<long double>(x[i]) * 1.75L;
      VERIFY(numext::abs(static_cast<long double>(alias[i]) - expected) <=
             4 * static_cast<long double>(NumTraits<Scalar>::epsilon()) * numext::abs(expected));
    }
    Strided contiguous(x.data(), n, InnerStride<Dynamic>(1));
    sme_check_dot(contiguous, y);
  }
  Vec x = Vec::Random(65538), y = Vec::Random(32769), expected = x;
  Strided strided(x.data(), y.size(), InnerStride<Dynamic>(2));
  Strided negative(x.data() + x.size() - 2, y.size(), InnerStride<Dynamic>(-2));
  sme_check_dot(strided, y);
  sme_check_dot(negative, y);
  for (Index i = 0; i < y.size(); ++i) expected[2 * i] += Scalar(0.75) * y[i];
  strided += Scalar(0.75) * y;
  VERIFY_IS_APPROX(x, expected);
  sme_check_dot(-y, strided);
  sme_check_dot(y + y, strided);
  sme_check_dot(y.reverse(), strided);
  VERIFY_RAISES_ASSERT(y.dot(x));

  for (int pattern = 0; pattern < 3; ++pattern) {
    Vec a = Vec::Random(65537), b = Vec::Random(65537);
    for (Index i = 0; i < a.size(); ++i) {
      if (pattern == 0) b[i] = (i & 1) ? -a[i] : a[i];
      if (pattern == 1) a[i] = numext::ldexp(a[i], int(i % 81) - 40);
      if (pattern == 2) {
        a[i] = Scalar(1);
        b[i] = i % 3 == 0   ? Scalar(1) / NumTraits<Scalar>::epsilon()
               : i % 3 == 1 ? Scalar(1)
                            : -Scalar(1) / NumTraits<Scalar>::epsilon();
      }
    }
    sme_check_dot(a, b);
  }
  for (Index n : {524287, 524288, 524289, 1048575, 1048576, 1048577}) {
    Vec a = Vec::Ones(n), b = Vec::Ones(n);
    VERIFY_IS_EQUAL(a.dot(b), Scalar(n));
    b += Scalar(0.75) * a;
    VERIFY((b.array() == Scalar(1.75)).all());
  }
}

template <typename Scalar>
void sme_vector_cache_sizes() {
  using Vec = Vector<Scalar, Dynamic>;
  const std::ptrdiff_t old_l1 = l1CacheSize(), old_l2 = l2CacheSize(), old_l3 = l3CacheSize();
  const Index n = 32769;
  const std::ptrdiff_t bytes = n * sizeof(Scalar);
  const Vec x = Vec::Ones(n);
  Vec y(n);
  for (std::ptrdiff_t l2 : {bytes, bytes - 1, std::ptrdiff_t(0), std::ptrdiff_t(-1), bytes + 1}) {
    setCpuCacheSizes(old_l1, l2, old_l3);
    VERIFY_IS_EQUAL(internal::sme_vector_size_suitable<Scalar>(n), l2 >= bytes);
    VERIFY(!internal::sme_vector_size_suitable<Scalar>((std::numeric_limits<Index>::max)()));
    VERIFY_IS_EQUAL(x.dot(x), Scalar(n));
    y.setOnes();
    y += Scalar(0.75) * x;
    VERIFY((y.array() == Scalar(1.75)).all());
    y.setOnes();
    y += x * Scalar(0.75);
    VERIFY((y.array() == Scalar(1.75)).all());
  }
  for (std::ptrdiff_t l1 : {bytes, bytes - 1, std::ptrdiff_t(0), std::ptrdiff_t(-1), bytes + 1}) {
    setCpuCacheSizes(l1, bytes, old_l3);
    const bool suitable = l1 > 0 && l1 <= bytes;
    VERIFY_IS_EQUAL(internal::sme_vector_size_suitable<Scalar>(n), suitable);
    y.setOnes();
    VERIFY_IS_EQUAL(internal::sme_try_axpy(y, x, Scalar(0.75)), suitable);
    VERIFY((y.array() == (suitable ? Scalar(1.75) : Scalar(1))).all());
    VERIFY_IS_EQUAL(x.dot(x), Scalar(n));
  }
  for (std::ptrdiff_t l1 : {2 * bytes - 1, 2 * bytes, 2 * bytes + 1}) {
    setCpuCacheSizes(l1, bytes, old_l3);
    VERIFY_IS_EQUAL((internal::sme_vector_size_suitable<Scalar, 2>(n)), l1 <= 2 * bytes);
    VERIFY_IS_EQUAL(x.dot(x), Scalar(n));
  }
#ifndef EIGEN_USE_BLAS
  const std::ptrdiff_t matrix_bytes = 128 * 4 * sizeof(Scalar);
  for (std::ptrdiff_t l1 : {matrix_bytes - 1, matrix_bytes, matrix_bytes + 1, std::ptrdiff_t(0)}) {
    setCpuCacheSizes(l1, old_l2, old_l3);
    VERIFY_IS_EQUAL(internal::sme_gemv_size_suitable<Scalar>(Index(128), Index(4)), l1 > 0 && l1 <= matrix_bytes);
    VERIFY(!internal::sme_gemv_size_suitable<Scalar>(Index(0), Index(4)));
  }
#endif
  setCpuCacheSizes(old_l1, old_l2, old_l3);
}

template <typename Scalar>
void sme_vector_scalar_factors() {
  using Vec = Vector<Scalar, Dynamic>;
  const Index n = 32769;
  Vec x = Vec::Constant(n, Scalar(0.25)), y = Vec::Zero(n);
  y += x.cwiseProduct(x);
  VERIFY((y.array() == Scalar(0.0625)).all());
  y.setZero();
  y += Vec::Constant(n, Scalar(0.75)).cwiseProduct(x);
  VERIFY((y.array() == Scalar(0.1875)).all());

  // Folding the two scale factors would overflow even though every output is finite.
  const Scalar large = (std::numeric_limits<Scalar>::max)() / Scalar(2);
  y.setZero();
  y += large * (Scalar(4) * x);
  VERIFY((y.array() == large).all());
  y.setZero();
  y += (x * Scalar(4)) * large;
  VERIFY((y.array() == large).all());

  Array<Scalar, Dynamic, 1> a = x.array(), b = Array<Scalar, Dynamic, 1>::Zero(n);
  b += Scalar(0.75) * a;
  VERIFY((b == Scalar(0.1875)).all());
  b.setZero();
  b += a * Scalar(0.75);
  VERIFY((b == Scalar(0.1875)).all());
}

template <typename Scalar>
void sme_matrix_vectors() {
  using Vec = Vector<Scalar, Dynamic>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  for (Index rows : {127, 128, 129, 255, 256, 257, 513}) {
    for (Index cols : {3, 4, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129}) {
      for (Index padding : {0, 3}) {
        Mat storage = Mat::Random(rows + padding, cols);
        auto a = storage.topRows(rows);
        Vec x = Vec::Random(cols), original = Vec::Random(rows), result(rows);
        for (Scalar alpha : {Scalar(0), Scalar(0.75), Scalar(-1)}) {
          result = original;
          result.noalias() += alpha * a * x;
          for (Index i = 0; i < rows; ++i) {
            long double expected = static_cast<long double>(original[i]), magnitude = numext::abs(expected);
            for (Index j = 0; j < cols; ++j) {
              const long double term =
                  static_cast<long double>(alpha) * static_cast<long double>(a(i, j)) * static_cast<long double>(x[j]);
              expected += term;
              magnitude += numext::abs(term);
            }
            const long double bound =
                8 * (cols + 1) * static_cast<long double>(NumTraits<Scalar>::epsilon()) * magnitude;
            VERIFY((numext::isfinite)(bound));
            VERIFY(numext::abs(static_cast<long double>(result[i]) - expected) <= bound);
          }
        }
        Vec strided_storage = Vec::Random(2 * rows), untouched = strided_storage;
        Map<Vec, 0, InnerStride<2>> strided(strided_storage.data(), rows);
        Vec reference = strided;
        reference.noalias() += a * x;
        strided.noalias() += a * x;
        VERIFY_IS_APPROX(strided, reference);
        for (Index i = 0; i < rows; ++i) VERIFY_IS_EQUAL(strided_storage[2 * i + 1], untouched[2 * i + 1]);
      }
    }
  }
}

template <typename Scalar>
void sme_vector_alignment() {
  using Vec = Vector<Scalar, Dynamic>;
  const Index padding = 64 / sizeof(Scalar);
  for (Index tail : {0, 1, 15, 31, 63, 127, 255, 256}) {
    const Index n = 32768 + tail;
    Vec x_storage = Vec::Ones(n + padding), y_storage(n + 2 * padding);
    for (Index offset = 0; offset < padding; ++offset) {
      y_storage.setConstant(Scalar(2));
      const auto x = x_storage.segment(offset, n);
      auto y = y_storage.segment(padding + offset, n);
      VERIFY(internal::sme_try_axpy(y, x, Scalar(0.75)));
      VERIFY((y.array() == Scalar(2.75)).all());
      VERIFY((y_storage.head(padding + offset).array() == Scalar(2)).all());
      VERIFY((y_storage.tail(padding - offset).array() == Scalar(2)).all());
      VERIFY_IS_EQUAL(x.dot(y), Scalar(2.75) * Scalar(n));
    }
  }
}

template <typename Scalar>
void sme_gemv_tails() {
  using Vec = Vector<Scalar, Dynamic>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  // Every predicated chunk pattern of a row block, including at the largest architectural SVL.
  for (Index rows = 128; rows <= 1100; ++rows) {
    const Mat a = Mat::Ones(rows, 65);
    const Vec x = Vec::Ones(65);
    Vec storage = Vec::Ones(rows + 2);
    auto y = storage.segment(1, rows);
    y.noalias() += Scalar(0.75) * a * x;
    VERIFY((y.array() == Scalar(49.75)).all());
    VERIFY_IS_EQUAL(storage[0], Scalar(1));
    VERIFY_IS_EQUAL(storage[rows + 1], Scalar(1));
  }
}

template <typename Scalar>
void sme_vector_special_values() {
  using Vec = Vector<Scalar, Dynamic>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  const Scalar inf = NumTraits<Scalar>::infinity(), nan = NumTraits<Scalar>::quiet_NaN();
  const Scalar tiny = (std::numeric_limits<Scalar>::denorm_min)();
  for (Index pos : {0, 7, 31, 63, 127, 255, 256, 32768}) {
    Vec x = Vec::Zero(32769), y = Vec::Ones(32769), result(32769);
    for (Scalar special : {inf, -inf, nan, tiny, -tiny, Scalar(-0.0)}) {
      x[pos] = special;
      const Scalar actual = x.dot(y);
      if ((numext::isnan)(special))
        VERIFY((numext::isnan)(actual));
      else
        VERIFY_IS_EQUAL(actual, special);
      result.setZero();
      result += Scalar(1) * x;
      for (Index i = 0; i < result.size(); ++i) {
        const Scalar expected = x[i] + Scalar(0);
        if ((numext::isnan)(expected))
          VERIFY((numext::isnan)(result[i]));
        else {
          VERIFY_IS_EQUAL(result[i], expected);
          VERIFY_IS_EQUAL(std::signbit(result[i]), std::signbit(expected));
        }
      }
    }
  }
  for (Scalar special : {inf, -inf, nan, tiny, -tiny}) {
    Mat a = Mat::Zero(257, 65);
    Vec x = Vec::Ones(65), y = Vec::Zero(257);
    a(0, 0) = a(256, 64) = special;
    y.noalias() += a * x;
    for (Index i = 0; i < y.size(); ++i) {
      const Scalar expected = i == 0 || i == 256 ? special : Scalar(0);
      if ((numext::isnan)(expected))
        VERIFY((numext::isnan)(y[i]));
      else
        VERIFY_IS_EQUAL(y[i], expected);
    }
    y.setOnes();
    y.noalias() += Scalar(0) * a * x;
    VERIFY((y.array() == Scalar(1)).all());
  }
  Vec x = Vec::Zero(32769), y = Vec::Zero(32769);
  x[0] = inf;
  VERIFY((numext::isnan)(x.dot(y)));
}

template <typename Scalar>
void sme_gemv_scaled_range() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using Vec = Vector<Scalar, Dynamic>;
  const Scalar largest = (std::numeric_limits<Scalar>::max)();
  const std::ptrdiff_t l1 = l1CacheSize(), l2 = l2CacheSize(), l3 = l3CacheSize();
  for (std::ptrdiff_t cache : {l1, std::ptrdiff_t(0)}) {
    setCpuCacheSizes(cache, l2, l3);
    for (Index rows : {256, 257}) {
      for (Index cols : {128, 129}) {
        Mat matrix = Mat::Constant(rows, cols, largest / Scalar(64));
        Vec rhs = Vec::Ones(cols), result = Vec::Zero(rows);
        result.noalias() += Scalar(0.25) * matrix * rhs;
        const Scalar expected = (largest / Scalar(256)) * Scalar(cols);
        const Scalar bound = Scalar(4 * cols) * NumTraits<Scalar>::epsilon();
        for (Index i = 0; i < rows; ++i) {
          VERIFY((numext::isfinite)(result[i]));
          VERIFY(numext::abs(result[i] / expected - Scalar(1)) <= bound);
        }
      }
    }
  }
  setCpuCacheSizes(l1, l2, l3);
}

template <typename Scalar>
void sme_dot_signed_zero() {
  using Vec = Vector<Scalar, Dynamic>;
  const Scalar tiny = (std::numeric_limits<Scalar>::denorm_min)();
  for (Index n : {32768, 32769, 32799, 33791}) {
    Vec x = Vec::Constant(n, Scalar(-0.0)), y = Vec::Ones(n);
    const Scalar negative = x.dot(y);
    VERIFY_IS_EQUAL(negative, Scalar(0));
    VERIFY((numext::signbit)(negative));
    x.setZero();
    const Scalar positive = x.dot(y);
    VERIFY_IS_EQUAL(positive, Scalar(0));
    VERIFY(!(numext::signbit)(positive));
    // A single positive-zero product, in the first group or the last element, gives +0.
    for (Index pos : {Index(0), n - 1}) {
      x.setConstant(Scalar(-0.0));
      x[pos] = Scalar(0);
      const Scalar mixed = x.dot(y);
      VERIFY_IS_EQUAL(mixed, Scalar(0));
      VERIFY(!(numext::signbit)(mixed));
    }
    // Exact cancellation, across lanes (adjacent terms) or within a fused lane (halves), rounds to +0.
    for (int within_lane = 0; within_lane < 2; ++within_lane) {
      x.setConstant(Scalar(-0.0));
      const Index half = n / 2;
      for (Index i = 0; i < half; ++i) {
        x[within_lane ? i : 2 * i] = Scalar(1);
        x[within_lane ? half + i : 2 * i + 1] = Scalar(-1);
      }
      const Scalar cancelled = x.dot(y);
      VERIFY_IS_EQUAL(cancelled, Scalar(0));
      VERIFY(!(numext::signbit)(cancelled));
    }
    // Products that underflow to -0 keep the sign of every fused partial sum.
    x.setConstant(-tiny);
    const Scalar underflow = x.dot(Vec::Constant(n, Scalar(0.25)));
    VERIFY_IS_EQUAL(underflow, Scalar(0));
    VERIFY((numext::signbit)(underflow));
  }
}

template <typename Scalar>
void sme_dot_directed_rounding() {
#if EIGEN_COMP_CLANG
#pragma STDC FENV_ACCESS ON
#endif
  using Vec = Vector<Scalar, Dynamic>;
  const int saved = std::fegetround();
  const Index n = 32769;
  const Vec y = Vec::Ones(n);
  for (int mode : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    VERIFY_IS_EQUAL(std::fesetround(mode), 0);
    // IEEE 754 6.3: x + x keeps the sign of x; an exact zero sum of opposite signs is -0 only when rounding down.
    Vec x = Vec::Zero(n);
    const Scalar positive = x.dot(y);
    x.setConstant(Scalar(-0.0));
    const Scalar negative = x.dot(y);
    x[0] = Scalar(0);
    const Scalar mixed = x.dot(y);
    std::fesetround(saved);
    VERIFY(positive == Scalar(0) && !std::signbit(positive));
    VERIFY(negative == Scalar(0) && std::signbit(negative));
    VERIFY(mixed == Scalar(0) && std::signbit(mixed) == (mode == FE_DOWNWARD));
  }
}

template <typename Scalar>
void sme_vector_exception_state() {
#if EIGEN_COMP_CLANG
#pragma STDC FENV_ACCESS ON
#endif
  using Vec = Vector<Scalar, Dynamic>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  std::fenv_t saved;
  std::fegetenv(&saved);
  const Scalar inf = NumTraits<Scalar>::infinity(), largest = (std::numeric_limits<Scalar>::max)();
  // The kernels leave the caller's flags as they were, whether or not an SVE tail or row remainder follows the
  // ZA loop: exact inputs, then inf * 0 (invalid), then max * 2 (overflow).
  for (Index n : {32768, 32769}) {
    Vec x(n), y(n);
    Mat matrix(n == 32768 ? 256 : 257, 128);
    Vec rhs(128), result(matrix.rows());
    for (int input = 0; input < 3; ++input) {
      const Scalar value = input == 0 ? Scalar(1) : input == 1 ? inf : largest;
      const Scalar factor = input == 0 ? Scalar(1) : input == 1 ? Scalar(0) : Scalar(2);
      for (int prior : {0, FE_DIVBYZERO}) {
        for (int operation = 0; operation < 3; ++operation) {
          x.setConstant(value);
          y.setConstant(factor);
          matrix.setConstant(value);
          rhs.setConstant(factor);
          result.setZero();
          std::feclearexcept(FE_ALL_EXCEPT);
          std::feraiseexcept(prior);
          Scalar actual;
          if (operation == 0) {
            actual = x.dot(y);
          } else if (operation == 1) {
            y += factor * x;
            actual = y[n - 1];
          } else {
            result.noalias() += matrix * rhs;
            actual = result[result.size() - 1];
          }
          const int flags = std::fetestexcept(FE_ALL_EXCEPT);
          VERIFY_IS_EQUAL(flags, prior);
          if (input == 1) {
            VERIFY((numext::isnan)(actual));
          } else if (input == 2) {
            VERIFY_IS_EQUAL(actual, inf);
          } else {
            VERIFY_IS_EQUAL(actual, operation == 0 ? Scalar(n) : operation == 1 ? Scalar(2) : Scalar(128));
          }
        }
      }
    }
  }
  // Longer calls are more likely to be preempted while streaming.
  Vec x = Vec::Ones(Index(1) << 20), y = Vec::Ones(Index(1) << 20);
  for (int repetition = 0; repetition < 64; ++repetition) {
    std::feclearexcept(FE_ALL_EXCEPT);
    const Scalar dot = x.dot(y);
    y += Scalar(repetition % 2 ? -0.75 : 0.75) * x;
    VERIFY_IS_EQUAL(std::fetestexcept(FE_ALL_EXCEPT), 0);
    VERIFY_IS_EQUAL(dot, Scalar(repetition % 2 ? 1.75 : 1) * Scalar(x.size()));
  }
  std::fesetenv(&saved);
}

EIGEN_DECLARE_TEST(vector_sme) {
  const std::ptrdiff_t l1 = l1CacheSize(), l2 = l2CacheSize(), l3 = l3CacheSize();
  setCpuCacheSizes(128 * 1024, 4 * 1024 * 1024, l3);
  CALL_SUBTEST_1(sme_vector_cache_sizes<float>());
  CALL_SUBTEST_1(sme_vector_scalar_factors<float>());
  CALL_SUBTEST_1(sme_vectors<float>());
  CALL_SUBTEST_1(sme_matrix_vectors<float>());
  CALL_SUBTEST_1(sme_vector_alignment<float>());
  CALL_SUBTEST_1(sme_gemv_tails<float>());
  CALL_SUBTEST_1(sme_vector_special_values<float>());
  CALL_SUBTEST_3(sme_gemv_scaled_range<float>());
  CALL_SUBTEST_3(sme_dot_signed_zero<float>());
  CALL_SUBTEST_3(sme_dot_directed_rounding<float>());
  CALL_SUBTEST_3(sme_vector_exception_state<float>());
#ifdef EIGEN_VECTORIZE_SME_F64F64
  CALL_SUBTEST_2(sme_vector_cache_sizes<double>());
  CALL_SUBTEST_2(sme_vector_scalar_factors<double>());
  CALL_SUBTEST_2(sme_vectors<double>());
  CALL_SUBTEST_2(sme_matrix_vectors<double>());
  CALL_SUBTEST_2(sme_vector_alignment<double>());
  CALL_SUBTEST_2(sme_gemv_tails<double>());
  CALL_SUBTEST_2(sme_vector_special_values<double>());
  CALL_SUBTEST_4(sme_gemv_scaled_range<double>());
  CALL_SUBTEST_4(sme_dot_signed_zero<double>());
  CALL_SUBTEST_4(sme_dot_directed_rounding<double>());
  CALL_SUBTEST_4(sme_vector_exception_state<double>());
#endif
  setCpuCacheSizes(l1, l2, l3);
}
