// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include "../blas.h"

static_assert(sizeof(EIGEN_BLAS_INT) == 8, "EIGEN_BLAS_INT must be 64-bit in blas_ilp64");
static_assert(sizeof(Eigen::BlasIndex) == 8, "Eigen::BlasIndex must be 64-bit in blas_ilp64");

namespace {

EIGEN_BLAS_INT g_last_xerbla_info = 0;

}  // namespace

extern "C" void BLASFUNC(xerbla)(const char* /*msg*/, EIGEN_BLAS_INT* info, size_t /*len*/) {
  g_last_xerbla_info = *info;
}

namespace {

// Low 32 bits equal 1, full 64-bit value is negative (-4294967295).
// A routine reading 32-bit int* sees 1 and executes; a 64-bit routine rejects it via xerbla_.
constexpr EIGEN_BLAS_INT kNegativeWithPositiveLowWord =
    static_cast<EIGEN_BLAS_INT>(std::uint64_t(1) | (std::uint64_t(0xFFFFFFFF) << 32));

void test_ilp64_upper_word_sensitivity() {
  EIGEN_BLAS_INT one = 1;
  EIGEN_BLAS_INT bad = kNegativeWithPositiveLowWord;
  double alpha = 1.0;
  double beta = 0.0;
  double a = 3.0;
  double b = 5.0;
  double c = -999.0;

  g_last_xerbla_info = 0;
  BLASFUNC(dgemm)("N", "N", &bad, &one, &one, &alpha, &a, &one, &b, &one, &beta, &c, &one);
  VERIFY_IS_EQUAL(g_last_xerbla_info, EIGEN_BLAS_INT(3));
  VERIFY_IS_EQUAL(c, -999.0);

  g_last_xerbla_info = 0;
  BLASFUNC(dgemv)("N", &bad, &one, &alpha, &a, &one, &b, &one, &beta, &c, &one);
  VERIFY_IS_EQUAL(g_last_xerbla_info, EIGEN_BLAS_INT(2));
  VERIFY_IS_EQUAL(c, -999.0);

  g_last_xerbla_info = 0;
  char uplo = 'U';
  BLASFUNC(dsymv)(&uplo, &bad, &alpha, &a, &one, &b, &one, &beta, &c, &one);
  VERIFY_IS_EQUAL(g_last_xerbla_info, EIGEN_BLAS_INT(2));
  VERIFY_IS_EQUAL(c, -999.0);

  g_last_xerbla_info = 0;
  BLASFUNC(dsyrk)(&uplo, "N", &bad, &one, &alpha, &a, &one, &beta, &c, &one);
  VERIFY_IS_EQUAL(g_last_xerbla_info, EIGEN_BLAS_INT(3));
  VERIFY_IS_EQUAL(c, -999.0);
}

void test_ilp64_level1() {
  EIGEN_BLAS_INT n = 5;
  EIGEN_BLAS_INT inc_pos = 1;
  EIGEN_BLAS_INT inc_neg = -1;
  EIGEN_BLAS_INT inc_zero = 0;

  double x[5] = {1.0, -4.0, 2.0, 9.0, -3.0};
  double y[5] = {10.0, 20.0, 30.0, 40.0, 50.0};

  VERIFY_IS_EQUAL(BLASFUNC(idamax)(&n, x, &inc_pos), EIGEN_BLAS_INT(4));
  VERIFY_IS_EQUAL(BLASFUNC(idamin)(&n, x, &inc_pos), EIGEN_BLAS_INT(1));

  // Reverse stride (-1 has 0xFFFFFFFF in the upper 32 bits).
  double d = BLASFUNC(ddot)(&n, x, &inc_neg, y, &inc_pos);
  double expected_d = x[4] * y[0] + x[3] * y[1] + x[2] * y[2] + x[1] * y[3] + x[0] * y[4];
  VERIFY_IS_APPROX(d, expected_d);

  // Broadcast copy (incx == 0).
  double src = 7.5;
  double dst[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  BLASFUNC(dcopy)(&n, &src, &inc_zero, dst, &inc_pos);
  for (EIGEN_BLAS_INT i = 0; i < n; ++i) {
    VERIFY_IS_EQUAL(dst[i], src);
  }
}

template <typename Scalar>
void test_ilp64_eigen_blas_dispatch() {
  using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  const Eigen::Index m = 19;
  const Eigen::Index n = 23;
  const Eigen::Index k = 17;

  Mat A = Mat::Random(m, k);
  Mat B = Mat::Random(k, n);
  Mat C = Mat::Zero(m, n);
  Mat C_ref = A.lazyProduct(B);
  C.noalias() = A * B;
  VERIFY_IS_APPROX(C, C_ref);

  Vec v = Vec::Random(k);
  Vec y = A * v;
  Vec y_ref = A.lazyProduct(v);
  VERIFY_IS_APPROX(y, y_ref);

  Mat S = Mat::Random(m, m);
  S = S + S.adjoint().eval();
  Mat B_sym = Mat::Random(m, n);
  Mat C_sym = S.template selfadjointView<Eigen::Upper>() * B_sym;
  Mat S_full = S.template selfadjointView<Eigen::Upper>();
  VERIFY_IS_APPROX(C_sym, S_full.lazyProduct(B_sym));

  Mat T = Mat::Random(m, m);
  T.diagonal().array() += Scalar(m);
  Mat X = T.template triangularView<Eigen::Upper>().solve(B_sym);
  Mat T_upper = T.template triangularView<Eigen::Upper>();
  VERIFY_IS_APPROX(T_upper * X, B_sym);
}

}  // namespace

EIGEN_DECLARE_TEST(blas_ilp64) {
  CALL_SUBTEST(test_ilp64_upper_word_sensitivity());
  CALL_SUBTEST(test_ilp64_level1());
  CALL_SUBTEST(test_ilp64_eigen_blas_dispatch<float>());
  CALL_SUBTEST(test_ilp64_eigen_blas_dispatch<double>());
  CALL_SUBTEST(test_ilp64_eigen_blas_dispatch<std::complex<float>>());
  CALL_SUBTEST(test_ilp64_eigen_blas_dispatch<std::complex<double>>());
}
