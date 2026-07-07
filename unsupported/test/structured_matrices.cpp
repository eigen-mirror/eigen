// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include <unsupported/Eigen/StructuredMatrices>

using namespace Eigen;

// Reference dense circulant built straight from the generating column, so that the
// fast (FFT) product is validated against an independent construction of the matrix.
template <typename Scalar>
Matrix<Scalar, Dynamic, Dynamic> reference_circulant(const Matrix<Scalar, Dynamic, 1>& c) {
  const Index n = c.size();
  Matrix<Scalar, Dynamic, Dynamic> dense(n, n);
  for (Index j = 0; j < n; ++j)
    for (Index i = 0; i < n; ++i) {
      Index k = i - j;
      if (k < 0) k += n;
      dense(i, j) = c[k];
    }
  return dense;
}

template <typename Scalar>
Matrix<Scalar, Dynamic, Dynamic> reference_toeplitz(const Matrix<Scalar, Dynamic, 1>& c,
                                                    const Matrix<Scalar, Dynamic, 1>& r) {
  const Index m = c.size(), n = r.size();
  Matrix<Scalar, Dynamic, Dynamic> dense(m, n);
  for (Index j = 0; j < n; ++j)
    for (Index i = 0; i < m; ++i) dense(i, j) = (i >= j) ? c[i - j] : r[j - i];
  return dense;
}

template <typename Scalar>
void test_circulant_product(Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec c = Vec::Random(n);
  Circulant<Scalar> C(c);
  Mat dense = reference_circulant<Scalar>(c);

  // The operator agrees with the independently-built dense matrix, both through
  // coeff access and assigned to a dense matrix via its evaluator.
  Mat Cd = C;
  VERIFY_IS_APPROX(Cd, dense);
  for (Index t = 0; t < (std::min)(n, Index(5)); ++t) {
    Index i = internal::random<Index>(0, n - 1), j = internal::random<Index>(0, n - 1);
    VERIFY_IS_APPROX(C.coeff(i, j), dense(i, j));
  }

  // Fast matrix-vector and matrix-matrix products.
  Vec x = Vec::Random(n);
  VERIFY_IS_APPROX((C * x).eval(), (dense * x).eval());

  Mat X = Mat::Random(n, 3);
  VERIFY_IS_APPROX((C * X).eval(), (dense * X).eval());

  // Accumulation forms exercised by the iterative solvers.
  Vec y = Vec::Random(n);
  Vec y0 = y;
  y.noalias() += C * x;
  VERIFY_IS_APPROX(y, (y0 + dense * x).eval());
}

template <typename Scalar>
void test_circulant_solve(Index n) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  // Diagonally dominant => well conditioned, so the direct FFT solve is accurate.
  Vec c = Vec::Random(n);
  c[0] += Scalar(RealScalar(2 * n));
  Circulant<Scalar> C(c);
  Mat dense = reference_circulant<Scalar>(c);

  Vec b = Vec::Random(n);
  Vec x = C.solve(b);
  VERIFY_IS_APPROX((dense * x).eval(), b);

  // Multiple right-hand sides at once.
  Mat B = Mat::Random(n, 4);
  Mat Xs = C.solve(B);
  VERIFY_IS_APPROX((dense * Xs).eval(), B);
}

// The precomputed symbol holds the eigenvalues of the circulant matrix:
// C * f_k = symbol[k] * f_k, with f_k the k-th column of the inverse DFT matrix.
template <typename Scalar>
void test_circulant_symbol(Index n) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Complex, Dynamic, 1> ComplexVec;

  Vec c = Vec::Random(n);
  Circulant<Scalar> C(c);
  VERIFY_IS_EQUAL(C.symbol().size(), n);
  Matrix<Complex, Dynamic, Dynamic> denseC = reference_circulant<Scalar>(c).template cast<Complex>();

  const Index step = numext::maxi<Index>(n / 4, 1);
  for (Index k = 0; k < n; k += step) {
    ComplexVec f(n);
    for (Index j = 0; j < n; ++j) f[j] = std::polar(RealScalar(1), RealScalar(2 * EIGEN_PI * j * k) / RealScalar(n));
    VERIFY_IS_APPROX((denseC * f).eval(), (C.symbol()[k] * f).eval());
  }
}

template <typename Scalar>
void test_toeplitz_product(Index m, Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec c = Vec::Random(m), r = Vec::Random(n);
  r[0] = c[0];  // diagonal entry; r[0] is ignored anyway
  Toeplitz<Scalar> T(c, r);
  Mat dense = reference_toeplitz<Scalar>(c, r);

  Mat Td = T;
  VERIFY_IS_APPROX(Td, dense);

  Vec x = Vec::Random(n);
  VERIFY_IS_APPROX((T * x).eval(), (dense * x).eval());

  Mat X = Mat::Random(n, 3);
  VERIFY_IS_APPROX((T * X).eval(), (dense * X).eval());
}

// Fixed-size operators: generators are stored in fixed-size vectors, products and
// solves return fixed-size results, and small sizes go through the coeff-based
// product dispatch.
template <typename Scalar, int N>
void test_circulant_fixed() {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, N, 1> VecN;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, N, N> MatN;

  VecN c = VecN::Random();
  c[0] += Scalar(RealScalar(2 * N));  // well conditioned for the solve below
  Circulant<Scalar, N> C(c);
  STATIC_CHECK((Circulant<Scalar, N>::RowsAtCompileTime == N));
  STATIC_CHECK((internal::remove_all_t<decltype(makeCirculant(c))>::RowsAtCompileTime == N));

  MatN dense = C;
  VERIFY_IS_APPROX(dense, MatN(reference_circulant<Scalar>(Vec(c))));

  VecN x = VecN::Random();
  VecN y = C * x;
  VERIFY_IS_APPROX(y, (dense * x).eval());

  VecN b = VecN::Random();
  VecN xs = C.solve(b);
  VERIFY_IS_APPROX((dense * xs).eval(), b);
}

template <typename Scalar, int M, int N>
void test_toeplitz_fixed() {
  typedef Matrix<Scalar, M, 1> ColVec;
  typedef Matrix<Scalar, N, 1> RowVec;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, M, N> MatMN;

  ColVec c = ColVec::Random();
  RowVec r = RowVec::Random();
  r[0] = c[0];
  Toeplitz<Scalar, M, N> T(c, r);
  STATIC_CHECK((Toeplitz<Scalar, M, N>::RowsAtCompileTime == M));
  STATIC_CHECK((Toeplitz<Scalar, M, N>::ColsAtCompileTime == N));
  STATIC_CHECK((internal::remove_all_t<decltype(makeToeplitz(c, r))>::ColsAtCompileTime == N));

  MatMN dense = T;
  VERIFY_IS_APPROX(dense, MatMN(reference_toeplitz<Scalar>(Vec(c), Vec(r))));

  RowVec x = RowVec::Random();
  Matrix<Scalar, M, 1> y = T * x;
  VERIFY_IS_APPROX(y, (dense * x).eval());
}

template <typename Scalar>
void test_matrix_free_cg(Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  // Symmetric, strongly diagonally dominant circulant => SPD.
  Vec c = Vec::Zero(n);
  c[0] = Scalar(4);
  c[1] = Scalar(-1);
  c[n - 1] = Scalar(-1);
  Circulant<Scalar> C(c);
  Mat dense = reference_circulant<Scalar>(c);

  Vec b = Vec::Random(n);
  ConjugateGradient<Circulant<Scalar>, Lower | Upper, IdentityPreconditioner> cg;
  cg.compute(C);
  Vec x = cg.solve(b);
  VERIFY(cg.info() == Success);
  VERIFY_IS_APPROX((dense * x).eval(), b);
}

template <typename Scalar>
void test_matrix_free_gmres(Index n) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  // Strongly diagonally dominant (nonsymmetric) Toeplitz.
  Vec c = Vec::Random(n) * Scalar(RealScalar(0.1));
  Vec r = Vec::Random(n) * Scalar(RealScalar(0.1));
  c[0] = Scalar(3);
  r[0] = Scalar(3);
  Toeplitz<Scalar> T(c, r);
  Mat dense = reference_toeplitz<Scalar>(c, r);

  Vec b = Vec::Random(n);
  GMRES<Toeplitz<Scalar>, IdentityPreconditioner> gmres;
  gmres.compute(T);
  Vec x = gmres.solve(b);
  VERIFY(gmres.info() == Success);
  VERIFY_IS_APPROX((dense * x).eval(), b);
}

// Diagonally dominant (well-conditioned) Toeplitz: the look-ahead solver must agree
// with a dense LU solve.
template <typename Scalar>
void test_levinson_wellcond(Index n) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec c = Vec::Random(n), r = Vec::Random(n);
  c[0] = r[0] = Scalar(RealScalar(2 * n));
  Toeplitz<Scalar> T(c, r);
  Mat dense = T;

  Vec b = Vec::Random(n);
  LookAheadLevinson<Scalar> lev(T);
  VERIFY(lev.info() == Success);
  Vec x = lev.solve(b);
  VERIFY_IS_APPROX(x, dense.fullPivLu().solve(b).eval());

  // Multiple right-hand sides.
  Mat B = Mat::Random(n, 3);
  VERIFY_IS_APPROX(lev.solve(B), dense.fullPivLu().solve(B).eval());
}

// Indefinite / ill-conditioned matrices that force look-ahead block steps. The
// generators and required block sizes are from Chan & Hansen's test set; the true
// solution is the all-ones vector.
void test_levinson_lookahead() {
  typedef Matrix<double, Dynamic, 1> Vec;
  auto check = [](const Vec& c, const Vec& r, Index pmax) {
    Toeplitz<double> T(c, r);
    Vec xt = Vec::Ones(c.size());
    Vec b = T * xt;
    LookAheadLevinson<double> lev;
    lev.setMaxBlockSize(pmax).compute(T);
    VERIFY(lev.info() == Success);
    Vec x = lev.solve(b);
    VERIFY((x - xt).norm() <= 1e-9 * xt.norm());
  };

  Vec c1(6), r1(6);  // Sweet-1 (block size 2)
  c1 << 4, 6, 71.0 / 15 + 5e-8, 5, 3, 1;
  r1 << 4, 8, 1, 6, 2, 3;
  check(c1, r1, 3);

  Vec c2(6), r2(6);  // Sweet-2 (block size 2)
  c2 << 8, 4, -34 + 5e-13, 5, 3, 1;
  r2 << 8, 4, 1, 6, 2, 3;
  check(c2, r2, 3);

  Vec c3(13), r3(13);  // Sweet-3 (block size 6)
  c3 << 5, 1, -3, 12.755, -19.656, 28.361, -7, -1, 2, 1, -6, 1, -0.5;
  r3 << 5, -1, 6, 2, 5.697, 5.850, 3, -5, -2, -7, 1, 10, -15;
  check(c3, r3, 6);

  // shifted KMS: leading submatrices T_k with k = 1,4,7,... are singular, so the
  // full order n must be a multiple of 3 for T_n itself to be non-singular.
  for (Index n : {15, 30, 60}) {
    Vec c(n), r(n);
    c[0] = r[0] = 1e-14;
    for (Index i = 1; i < n; ++i) c[i] = r[i] = std::pow(0.5, double(i - 1));
    check(c, r, 3);
  }
}

// Fixed-size Toeplitz operators also feed the solver.
void test_levinson_fixed() {
  typedef Matrix<double, 12, 1> Vec12;
  Vec12 c = Vec12::Random(), r = Vec12::Random();
  c[0] = r[0] = 24.0;
  Toeplitz<double, 12, 12> T(c, r);
  Matrix<double, 12, 12> dense = T;
  Vec12 b = Vec12::Random();
  LookAheadLevinson<double> lev(T);
  VERIFY(lev.info() == Success);
  VERIFY_IS_APPROX(lev.solve(b), dense.fullPivLu().solve(b).eval());
}

// A numerically singular Toeplitz must be reported through info().
void test_levinson_singular() {
  typedef Matrix<double, Dynamic, 1> Vec;
  for (Index n : {4, 9}) {
    Vec c = Vec::Ones(n), r = Vec::Ones(n);  // all-ones Toeplitz is rank 1 for n >= 2
    LookAheadLevinson<double> lev(Toeplitz<double>(c, r));
    VERIFY(lev.info() == NumericalIssue);
  }
}

EIGEN_DECLARE_TEST(structured_matrices) {
  for (int i = 0; i < g_repeat; ++i) {
    // Circulant: direct path (small), FFT path (composite and prime sizes), edge cases.
    CALL_SUBTEST_1((test_circulant_product<double>(1)));
    CALL_SUBTEST_1((test_circulant_product<double>(2)));
    CALL_SUBTEST_1((test_circulant_product<double>(8)));
    CALL_SUBTEST_1((test_circulant_product<double>(64)));
    CALL_SUBTEST_1((test_circulant_product<double>(97)));  // prime, FFT path
    CALL_SUBTEST_1((test_circulant_product<float>(48)));
    CALL_SUBTEST_1((test_circulant_product<std::complex<double>>(7)));  // direct path, complex
    CALL_SUBTEST_1((test_circulant_product<std::complex<double>>(50)));
    CALL_SUBTEST_1((test_circulant_product<std::complex<float>>(40)));
    CALL_SUBTEST_1((test_circulant_solve<double>(1)));  // degenerate 1x1 solve
    CALL_SUBTEST_1((test_circulant_solve<double>(8)));
    CALL_SUBTEST_1((test_circulant_solve<double>(50)));
    CALL_SUBTEST_1((test_circulant_solve<std::complex<double>>(40)));
    CALL_SUBTEST_1((test_circulant_solve<float>(32)));
    CALL_SUBTEST_1((test_circulant_symbol<double>(16)));
    CALL_SUBTEST_1((test_circulant_symbol<std::complex<double>>(12)));

    // Toeplitz: square, tall, wide, small (direct), single row/column, real and complex.
    CALL_SUBTEST_2((test_toeplitz_product<double>(1, 1)));
    CALL_SUBTEST_2((test_toeplitz_product<double>(2, 2)));
    CALL_SUBTEST_2((test_toeplitz_product<double>(10, 10)));
    CALL_SUBTEST_2((test_toeplitz_product<double>(64, 64)));
    CALL_SUBTEST_2((test_toeplitz_product<double>(96, 48)));
    CALL_SUBTEST_2((test_toeplitz_product<double>(48, 96)));
    CALL_SUBTEST_2((test_toeplitz_product<double>(1, 40)));  // single row, FFT path
    CALL_SUBTEST_2((test_toeplitz_product<double>(40, 1)));  // single column, FFT path
    CALL_SUBTEST_2((test_toeplitz_product<float>(50, 50)));
    CALL_SUBTEST_2((test_toeplitz_product<std::complex<double>>(5, 7)));  // direct path, complex
    CALL_SUBTEST_2((test_toeplitz_product<std::complex<double>>(48, 64)));
    CALL_SUBTEST_2((test_toeplitz_product<std::complex<float>>(40, 40)));

    // Matrix-free iterative solves through the existing solvers.
    CALL_SUBTEST_3((test_matrix_free_cg<double>(80)));
    CALL_SUBTEST_3((test_matrix_free_gmres<double>(80)));

    // Fixed-size operators: small (coeff-based dispatch) and above the FFT threshold.
    CALL_SUBTEST_4((test_circulant_fixed<double, 4>()));
    CALL_SUBTEST_4((test_circulant_fixed<std::complex<float>, 4>()));
    CALL_SUBTEST_4((test_circulant_fixed<double, 48>()));
    CALL_SUBTEST_4((test_toeplitz_fixed<double, 4, 6>()));
    CALL_SUBTEST_4((test_toeplitz_fixed<double, 40, 24>()));
    CALL_SUBTEST_4((test_toeplitz_fixed<std::complex<float>, 6, 4>()));

    // Look-ahead Levinson direct Toeplitz solver.
    CALL_SUBTEST_5((test_levinson_wellcond<double>(1)));
    CALL_SUBTEST_5((test_levinson_wellcond<double>(2)));
    CALL_SUBTEST_5((test_levinson_wellcond<double>(20)));
    CALL_SUBTEST_5((test_levinson_wellcond<double>(60)));
    CALL_SUBTEST_5((test_levinson_wellcond<float>(40)));
    CALL_SUBTEST_5((test_levinson_wellcond<std::complex<double>>(30)));
    CALL_SUBTEST_5((test_levinson_wellcond<std::complex<float>>(24)));
    CALL_SUBTEST_5(test_levinson_lookahead());
    CALL_SUBTEST_5(test_levinson_fixed());
    CALL_SUBTEST_5(test_levinson_singular());
  }
}
