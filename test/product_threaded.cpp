// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2023 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_GEMM_THREADPOOL
#include "main.h"

// Eigen::setGemmThreadPool(nullptr) is the *getter*: Parallelizer.h only stores the pointer when
// it is non-null, so once any test registers a pool it stays registered for the rest of the
// process.  A "serial" reference computed after that point is really a threaded one, and the
// comparison degenerates into threaded-against-threaded.
//
// parallelize_gemm decides on min(nbThreads(), work-derived bound) <= 1 before it ever reaches the
// pool, so forcing the thread count is what actually pins the serial path.  It also makes the
// stale registration harmless.
struct ScopedSerialGemm {
  ScopedSerialGemm() : saved_threads_(Eigen::nbThreads()) { Eigen::setNbThreads(1); }
  // In a thread-pool build nbThreads() reports the raw -1 that means "never set", and
  // setNbThreads rejects a negative count.  0 is its documented "back to the pool's count".
  ~ScopedSerialGemm() { Eigen::setNbThreads(saved_threads_ > 0 ? saved_threads_ : 0); }
  ScopedSerialGemm(const ScopedSerialGemm&) = delete;
  ScopedSerialGemm& operator=(const ScopedSerialGemm&) = delete;

 private:
  int saved_threads_;
};

void test_parallelize_gemm() {
  constexpr int n = 1024;
  constexpr int num_threads = 4;
  MatrixXf a = MatrixXf::Random(n, n);
  MatrixXf b = MatrixXf::Random(n, n);
  MatrixXf c = MatrixXf::Random(n, n);
  {
    ScopedSerialGemm serial;
    c.noalias() = a * b;
  }

  // static: setGemmThreadPool cannot unregister, so a pool with narrower lifetime than the process
  // would leave a dangling pointer registered for whatever runs next.
  static ThreadPool pool(num_threads);
  Eigen::setGemmThreadPool(&pool);
  MatrixXf c_threaded(n, n);
  c_threaded.noalias() = a * b;

  VERIFY_IS_APPROX(c, c_threaded);
}

void test_parallelize_gemm_varied() {
  constexpr int num_threads = 4;
  static ThreadPool pool(num_threads);

  // Non-square float
  {
    MatrixXf a = MatrixXf::Random(512, 2048);
    MatrixXf b = MatrixXf::Random(2048, 256);
    MatrixXf c_serial(512, 256);
    {
      ScopedSerialGemm serial;
      c_serial.noalias() = a * b;
    }
    Eigen::setGemmThreadPool(&pool);
    MatrixXf c_threaded(512, 256);
    c_threaded.noalias() = a * b;
    VERIFY_IS_APPROX(c_serial, c_threaded);
  }

  // Double
  {
    MatrixXd a = MatrixXd::Random(512, 512);
    MatrixXd b = MatrixXd::Random(512, 512);
    MatrixXd c_serial(512, 512);
    {
      ScopedSerialGemm serial;
      c_serial.noalias() = a * b;
    }
    Eigen::setGemmThreadPool(&pool);
    MatrixXd c_threaded(512, 512);
    c_threaded.noalias() = a * b;
    VERIFY_IS_APPROX(c_serial, c_threaded);
  }

  // Complex double
  {
    MatrixXcd a = MatrixXcd::Random(256, 256);
    MatrixXcd b = MatrixXcd::Random(256, 256);
    MatrixXcd c_serial(256, 256);
    {
      ScopedSerialGemm serial;
      c_serial.noalias() = a * b;
    }
    Eigen::setGemmThreadPool(&pool);
    MatrixXcd c_threaded(256, 256);
    c_threaded.noalias() = a * b;
    VERIFY_IS_APPROX(c_serial, c_threaded);
  }
}

void test_balanced_gemm_range() {
  static const Index totals[] = {0, 1, 3, 4, 7, 12, 63, 100, 4096, 8192, 9216};
  static const Index part_counts[] = {1, 2, 3, 7, 8, 32, 64, 72};
  static const Index grains[] = {1, 4, 6, 8, 12};
  for (Index total : totals) {
    for (Index parts : part_counts) {
      for (Index grain : grains) {
        Index expected_start = 0;
        Index min_chunks = NumTraits<Index>::highest();
        Index max_chunks = 0;
        for (Index part = 0; part < parts; ++part) {
          Index start = -1, length = -1;
          internal::balanced_gemm_range<Index>(total, parts, grain, part, start, length);
          // Gaps or overlaps corrupt the packed-lhs handoff, which indexes blockA by lhs_start.
          VERIFY(start == expected_start);
          VERIFY(length >= 0);
          expected_start = start + length;
          const Index chunks = numext::div_ceil(length, grain);
          min_chunks = numext::mini(min_chunks, chunks);
          max_chunks = numext::maxi(max_chunks, chunks);
        }
        VERIFY(expected_start == total);
        // A single oversized range throttles every other thread.
        VERIFY(max_chunks - min_chunks <= Index(1));
      }
    }
  }
}

template <typename MatrixType>
void verify_threaded_product(ThreadPool& pool, Index rows, Index depth, Index cols) {
  MatrixType a = MatrixType::Random(rows, depth), b = MatrixType::Random(depth, cols);
  MatrixType c_serial;
  {
    ScopedSerialGemm serial;
    c_serial = a * b;
  }
  Eigen::setGemmThreadPool(&pool);
  MatrixType c_threaded = a * b;
  VERIFY_IS_APPROX(c_serial, c_threaded);
}

void test_parallelize_gemm_indivisible() {
  // Shapes deliberately not divisible by the thread count, where the split has to spread the
  // remainder rather than append it to the last thread.
  static ThreadPool pool(8);
  verify_threaded_product<MatrixXf>(pool, 517, 331, 523);
  verify_threaded_product<MatrixXf>(pool, 1021, 331, 259);
  verify_threaded_product<MatrixXf>(pool, 64, 331, 4099);
  verify_threaded_product<MatrixXf>(pool, 4099, 331, 64);
}

// A scalar whose constructors set a magic value that operator= checks: the shared blocking buffers of a threaded
// product must be constructed before the packers assign into them (NumTraits::RequireInitialization).
struct TrackedScalar {
  static constexpr unsigned kMagic = 0x5EEDu;
  static int unconstructed_assignments;
  double v;
  unsigned magic;
  TrackedScalar() : v(0), magic(kMagic) {}
  TrackedScalar(double d) : v(d), magic(kMagic) {}
  TrackedScalar(const TrackedScalar& o) : v(o.v), magic(kMagic) {}
  TrackedScalar& operator=(const TrackedScalar& o) {
    if (magic != kMagic) ++unconstructed_assignments;
    v = o.v;
    return *this;
  }
  TrackedScalar operator+(const TrackedScalar& o) const { return TrackedScalar(v + o.v); }
  TrackedScalar operator-(const TrackedScalar& o) const { return TrackedScalar(v - o.v); }
  TrackedScalar operator*(const TrackedScalar& o) const { return TrackedScalar(v * o.v); }
  TrackedScalar operator-() const { return TrackedScalar(-v); }
  TrackedScalar& operator+=(const TrackedScalar& o) { return *this = *this + o; }
  TrackedScalar& operator*=(const TrackedScalar& o) { return *this = *this * o; }
  bool operator==(const TrackedScalar& o) const { return v == o.v; }
  bool operator!=(const TrackedScalar& o) const { return v != o.v; }
  bool operator<(const TrackedScalar& o) const { return v < o.v; }
};
int TrackedScalar::unconstructed_assignments = 0;

namespace Eigen {
template <>
struct NumTraits<TrackedScalar> : GenericNumTraits<TrackedScalar> {
  using Real = TrackedScalar;
  using NonInteger = TrackedScalar;
  using Nested = TrackedScalar;
  static constexpr int IsComplex = 0;
  static constexpr int IsInteger = 0;
  static constexpr int IsSigned = 1;
  static constexpr int RequireInitialization = 1;
  static constexpr int ReadCost = 1;
  static constexpr int AddCost = 1;
  static constexpr int MulCost = 1;
};
}  // namespace Eigen

void test_parallelize_gemm_require_initialization() {
  constexpr int n = 160;
  static ThreadPool pool(4);
  Eigen::setGemmThreadPool(&pool);
  using Mat = Matrix<TrackedScalar, Dynamic, Dynamic>;
  const MatrixXd ad = MatrixXd::Random(n, n), bd = MatrixXd::Random(n, n);
  const Mat a = ad.cast<TrackedScalar>(), b = bd.cast<TrackedScalar>();
  Mat c(n, n);
  TrackedScalar::unconstructed_assignments = 0;
  c.noalias() = a * b;
  VERIFY_IS_EQUAL(TrackedScalar::unconstructed_assignments, 0);
  const MatrixXd cd = c.unaryExpr([](const TrackedScalar& x) { return x.v; });
  VERIFY_IS_APPROX(cd, ad * bd);
}

// Tiny results and thin column ranges on a pool: every thread of a parallel session must take the same path.
template <typename Scalar>
void test_parallelize_gemm_tiny() {
  static ThreadPool pool(4);
  Eigen::setGemmThreadPool(&pool);
  Eigen::setNbThreads(4);
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  for (Index m : {2, 4, 8})
    for (Index n : {3, 8, 17, 36})
      for (Index k : {16, 2000, 4096}) {
        const Mat a = Mat::Random(m, k), b = Mat::Random(k, n);
        Mat c(m, n);
        c.noalias() = a * b;
        VERIFY_IS_APPROX(c, a.lazyProduct(b));
      }
  Eigen::setNbThreads(0);
}

EIGEN_DECLARE_TEST(product_threaded) {
  CALL_SUBTEST_6(test_parallelize_gemm_tiny<float>());
  CALL_SUBTEST_6(test_parallelize_gemm_tiny<double>());
  CALL_SUBTEST_1(test_parallelize_gemm());
  CALL_SUBTEST_2(test_parallelize_gemm_varied());
  CALL_SUBTEST_3(test_balanced_gemm_range());
  CALL_SUBTEST_4(test_parallelize_gemm_indivisible());
  CALL_SUBTEST_5(test_parallelize_gemm_require_initialization());
}
