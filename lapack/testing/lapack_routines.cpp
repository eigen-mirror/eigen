// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Checks of Eigen's LAPACK routines against the reference LAPACK contract that need no Fortran compiler.

#include "main.h"
#include "../lapack.h"

namespace {

using std::complex;

// The routines take column-major matrices, so pin the layout that EIGEN_DEFAULT_TO_ROW_MAJOR would change.
template <typename Scalar>
using ColMat = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;

void larfb(const char* side, const char* trans, const char* direct, const char* storev, int* m, int* n, int* k,
           double* v, int* ldv, double* t, int* ldt, double* c, int* ldc, double* work, int* ldwork) {
  BLASFUNC(dlarfb)(side, trans, direct, storev, m, n, k, v, ldv, t, ldt, c, ldc, work, ldwork);
}
void larfb(const char* side, const char* trans, const char* direct, const char* storev, int* m, int* n, int* k,
           complex<double>* v, int* ldv, complex<double>* t, int* ldt, complex<double>* c, int* ldc,
           complex<double>* work, int* ldwork) {
  BLASFUNC(zlarfb)
  (side, trans, direct, storev, m, n, k, reinterpret_cast<double*>(v), ldv, reinterpret_cast<double*>(t), ldt,
   reinterpret_cast<double*>(c), ldc, reinterpret_cast<double*>(work), ldwork);
}

// xLARFB applies H = I - V T V^H, or H^H, where V has an implicit unit triangle: in its first k rows (DIRECT = 'F') or
// its last k rows (DIRECT = 'B'). The diagonal of that triangle and the cells beyond it are not referenced. This stores
// `unreferenced` there and compares with H formed explicitly. Zeros there, as QL or RQ leaves them for a trivial
// reflector, once made xLARFB misplace the triangle; a NaN there shows any read of those cells.
template <typename Scalar>
void check_larfb(char side, char trans, char direct, char storev, const Scalar& unreferenced) {
  using Mat = ColMat<Scalar>;
  int k = internal::random<int>(1, 4);
  int len = k + internal::random<int>(0, 5);
  int other = internal::random<int>(1, 5);
  int m = side == 'L' ? len : other, n = side == 'L' ? other : len;
  const bool forward = direct == 'F', colwise = storev == 'C';

  Mat V_eff = Mat::Random(len, k);  // V as a len x k matrix of columns
  for (int j = 0; j < k; ++j) {
    const int pivot = forward ? j : len - k + j;
    V_eff(pivot, j) = Scalar(1);
    for (int i = forward ? 0 : pivot + 1; i < (forward ? pivot : len); ++i) V_eff(i, j) = Scalar(0);
  }
  Mat V = colwise ? Mat(V_eff) : Mat(V_eff.adjoint());
  for (int j = 0; j < k; ++j) {
    const int pivot = forward ? j : len - k + j;
    for (int i = forward ? 0 : pivot; i <= (forward ? pivot : len - 1); ++i)
      (colwise ? V(i, j) : V(j, i)) = unreferenced;
  }
  Mat T = Mat::Random(k, k), C = Mat::Random(m, n);
  const Mat T_eff = forward ? Mat(T.template triangularView<Upper>()) : Mat(T.template triangularView<Lower>());
  const Mat H = Mat::Identity(len, len) - V_eff * T_eff * V_eff.adjoint();
  const Mat op = trans == 'N' ? H : Mat(H.adjoint());
  const Mat expected = side == 'L' ? Mat(op * C) : Mat(C * op);

  int ldv = int(V.rows()), ldt = k, ldc = m, ldwork = (std::max)(m, n);
  Mat work(ldwork, k);
  larfb(&side, &trans, &direct, &storev, &m, &n, &k, V.data(), &ldv, T.data(), &ldt, C.data(), &ldc, work.data(),
        &ldwork);
  VERIFY_IS_APPROX(C, expected);
}

template <typename Scalar>
void test_larfb() {
  const char trans_adj = NumTraits<Scalar>::IsComplex ? 'C' : 'T';
  for (const Scalar unreferenced : {Scalar(0), Scalar(std::numeric_limits<double>::quiet_NaN())})
    for (char side : {'L', 'R'})
      for (char trans : {'N', trans_adj})
        for (char direct : {'F', 'B'})
          for (char storev : {'C', 'R'}) check_larfb<Scalar>(side, trans, direct, storev, unreferenced);
}

// Like the reference, xLARFB applies H to all of C: a NaN in a row of C that V does not reach still spreads through
// W = C^H V to the whole column.
void test_larfb_nan_propagation() {
  int m = 4, n = 2, k = 1, ldv = 4, ldt = 1, ldc = 4, ldwork = 2;
  ColMat<double> V(4, 1), T(1, 1), C = ColMat<double>::Ones(4, 2), work(2, 1);
  V << 1, 0.5, 0.5, 0;
  T << 1;
  C(3, 0) = std::numeric_limits<double>::quiet_NaN();
  BLASFUNC(dlarfb)
  ("L", "N", "F", "C", &m, &n, &k, V.data(), &ldv, T.data(), &ldt, C.data(), &ldc, work.data(), &ldwork);
  VERIFY(C.col(0).array().isNaN().all());
  VERIFY(C.col(1).allFinite());
}

complex<float> ladiv(float a, float b, float c, float d) {
  float p, q;
  BLASFUNC(sladiv)(&a, &b, &c, &d, &p, &q);
  return complex<float>(p, q);
}
complex<double> ladiv(double a, double b, double c, double d) {
  double p, q;
  BLASFUNC(dladiv)(&a, &b, &c, &d, &p, &q);
  return complex<double>(p, q);
}
complex<float> ladiv(const complex<float>& x, const complex<float>& y) {
  const eigen_lapack_complex_float z =
      BLASFUNC(cladiv)(reinterpret_cast<const float*>(&x), reinterpret_cast<const float*>(&y));
  return complex<float>(z.r, z.i);
}
complex<double> ladiv(const complex<double>& x, const complex<double>& y) {
  const eigen_lapack_complex_double z =
      BLASFUNC(zladiv)(reinterpret_cast<const double*>(&x), reinterpret_cast<const double*>(&y));
  return complex<double>(z.r, z.i);
}

// Checks (a + ib) / (c + id) = p + iq through xLADIV and its complex counterpart, componentwise: the robust branches
// exist to get the smaller component right, which a normwise check cannot see. A subnormal result and its stored exact
// value each lie within denorm_min / 2 of the true quotient. An exact zero must come out as +0, as in the reference.
template <typename Real>
void check_ladiv(Real a, Real b, Real c, Real d, Real p, Real q) {
  const Real eps = NumTraits<Real>::epsilon(), denorm_min = std::numeric_limits<Real>::denorm_min();
  const auto matches = [&](Real x, Real exact) {
    if (exact == 0) return x == 0 && !(std::signbit)(x);
    return numext::abs(x - exact) <= 4 * eps * numext::abs(exact) + denorm_min;
  };
  for (const complex<Real>& z : {ladiv(a, b, c, d), ladiv(complex<Real>(a, b), complex<Real>(c, d))}) {
    VERIFY(matches(z.real(), p));
    VERIFY(matches(z.imag(), q));
  }
}

void test_ladiv() {
  const double eps = NumTraits<double>::epsilon();
  // Smith's algorithm without scaling overflows to (NaN, 0) here, since c + d (d / c) = 2e308, and underflows to
  // (0, 0) for (1 + i) / (c + ic) = 1 / c, which is subnormal.
  const double big = 1e308, c = 1.5e308;
  const float bigf = 3e38f;
  check_ladiv(big, big, big, big, 1.0, 0.0);
  check_ladiv(1.0, 1.0, c, c, 1 / c, 0.0);
  check_ladiv(bigf, bigf, bigf, bigf, 1.0f, 0.0f);

  // The hard cases of M. Baudin and R. L. Smith, "A Robust Complex Division in Scilab", arXiv:1210.4539 (2012), with
  // their exact quotients: (a + ib) / (c + id) = p + iq. They reach the scaling, the swap for |d| > |c|, and the
  // branches of reference DLADIV2 for an underflowing r = d / c or b r, where DLADIV2's b is -a for q.
  const double hard[][6] = {
      {1, 1, 1, std::ldexp(1.0, 1023), std::ldexp(1.0, -1023), -std::ldexp(1.0, -1023)},
      {1, 1, std::ldexp(1.0, -1023), std::ldexp(1.0, -1023), std::ldexp(1.0, 1023), 0},
      {std::ldexp(1.0, 1023), std::ldexp(1.0, -1023), std::ldexp(1.0, 677), std::ldexp(1.0, -677), std::ldexp(1.0, 346),
       -std::ldexp(1.0, -1008)},
      {std::ldexp(1.0, 1023), std::ldexp(1.0, 1023), 1, 1, std::ldexp(1.0, 1023), 0},
      {std::ldexp(1.0, 1020), std::ldexp(1.0, -844), std::ldexp(1.0, 656), std::ldexp(1.0, -780), std::ldexp(1.0, 364),
       -std::ldexp(1.0, -1072)},
      {std::ldexp(1.0, -71), std::ldexp(1.0, 1021), std::ldexp(1.0, 1001), std::ldexp(1.0, -323),
       std::ldexp(1.0, -1072), std::ldexp(1.0, 20)},
      {std::ldexp(1.0, -347), std::ldexp(1.0, -54), std::ldexp(1.0, -1037), std::ldexp(1.0, -1058),
       3.898125604559113300e289, 8.174961907852353577e295},
      {std::ldexp(1.0, -1074), std::ldexp(1.0, -1074), std::ldexp(1.0, -1073), std::ldexp(1.0, -1074), 0.6, 0.2},
      {std::ldexp(1.0, 1015), std::ldexp(1.0, -989), std::ldexp(1.0, 1023), std::ldexp(1.0, 1023), 0.001953125,
       -0.001953125},
      {std::ldexp(1.0, -622), std::ldexp(1.0, -1071), std::ldexp(1.0, -343), std::ldexp(1.0, -798),
       1.02951151789360578e-84, 6.97145987515076231e-220},
  };
  for (const auto& h : hard) check_ladiv(h[0], h[1], h[2], h[3], h[4], h[5]);

  // Single-precision cases built the same way for float's thresholds. The third and fifth reach r = d / c = 0, with a
  // scaled numerator and with a subnormal quotient; the sixth reaches an underflowing b r for q; the seventh needs the
  // scaling of the divisor. Each quotient follows from the powers of two, e.g. the sixth's
  // q = 2^-109 (1 - 2^-6) / (1 + 2^-120).
  const float hard_float[][6] = {
      {1, 1, 1, std::ldexp(1.0f, 127), std::ldexp(1.0f, -127), -std::ldexp(1.0f, -127)},
      {1, 1, std::ldexp(1.0f, -127), std::ldexp(1.0f, -127), std::ldexp(1.0f, 127), 0},
      {std::ldexp(1.0f, 127), std::ldexp(1.0f, -127), std::ldexp(1.0f, 84), std::ldexp(1.0f, -84), std::ldexp(1.0f, 43),
       -std::ldexp(1.0f, -125)},
      {std::ldexp(1.0f, 127), std::ldexp(1.0f, 127), 1, 1, std::ldexp(1.0f, 127), 0},
      {std::ldexp(1.0f, 124), std::ldexp(1.0f, -100), std::ldexp(1.0f, 80), std::ldexp(1.0f, -95), std::ldexp(1.0f, 44),
       -std::ldexp(1.0f, -131)},
      {std::ldexp(1.0f, -95), std::ldexp(1.0f, -149), std::ldexp(1.0f, -40), std::ldexp(1.0f, -100),
       std::ldexp(1.0f, -55), std::ldexp(63.0f, -115)},
      {std::ldexp(1.0f, -149), std::ldexp(1.0f, -149), std::ldexp(1.0f, -148), std::ldexp(1.0f, -149), 0.6f, 0.2f},
  };
  for (const auto& h : hard_float) check_ladiv(h[0], h[1], h[2], h[3], h[4], h[5]);

  // For operands of moderate size, both this and std::complex division err by a few eps normwise.
  for (int i = 0; i < 1000; ++i) {
    const complex<double> x = internal::random<complex<double> >(), y = internal::random<complex<double> >();
    for (const complex<double>& z : {ladiv(x.real(), x.imag(), y.real(), y.imag()), ladiv(x, y)})
      VERIFY(numext::abs(z - x / y) <= 8 * eps * numext::abs(x / y));
  }
}

template <typename Real>
void test_ladiv_special() {
  const Real nan = std::numeric_limits<Real>::quiet_NaN(), inf = std::numeric_limits<Real>::infinity();
  const Real denorm_min = std::numeric_limits<Real>::denorm_min();
  const auto both = [](Real a, Real b, Real c, Real d) {
    return std::vector<complex<Real> >{ladiv(a, b, c, d), ladiv(complex<Real>(a, b), complex<Real>(c, d))};
  };
  // As in the reference, a NaN operand or a zero divisor (r = 0 / 0) gives NaN in both parts.
  const Real nan_quotient[][4] = {{nan, 2, 3, 4}, {1, nan, 3, 4}, {1, 2, nan, 4}, {1, 2, 3, nan}, {1, 2, 0, 0}};
  for (const auto& o : nan_quotient)
    for (const complex<Real>& z : both(o[0], o[1], o[2], o[3]))
      VERIFY((numext::isnan)(z.real()) && (numext::isnan)(z.imag()));
  // A finite numerator over an infinite divisor gives zeros with the reference's signs, the last two entries.
  const Real zero_quotient[][6] = {{1, 2, inf, 0, 1, 1}, {1, 2, 0, inf, 1, -1}, {1, 2, -inf, 3, -1, -1}};
  for (const auto& o : zero_quotient)
    for (const complex<Real>& z : both(o[0], o[1], o[2], o[3]))
      VERIFY(z.real() == 0 && (std::signbit)(z.real()) == (o[4] < 0) && z.imag() == 0 &&
             (std::signbit)(z.imag()) == (o[5] < 0));
  // Only the numerator is scaled here: (dm + i dm) / (2 + i) = (0.6 dm, 0.2 dm) rounds to (dm, +0) for dm =
  // denorm_min. Unscaled, b r = dm / 2 and a t = 0.4 dm both round to 0, so p = 0, which check_ladiv's denorm_min slack
  // would accept.
  for (const complex<Real>& z : both(denorm_min, denorm_min, 2, 1))
    VERIFY(z.real() == denorm_min && z.imag() == 0 && !(std::signbit)(z.imag()));
}

float lapy2(float x, float y) { return BLASFUNC(slapy2)(&x, &y); }
double lapy2(double x, double y) { return BLASFUNC(dlapy2)(&x, &y); }

template <typename Real>
void test_lapy2() {
  const Real inf = std::numeric_limits<Real>::infinity(), nan = std::numeric_limits<Real>::quiet_NaN();
  const Real eps = NumTraits<Real>::epsilon();
  const Real big = (std::numeric_limits<Real>::max)() / 4;  // big^2 overflows
  const Real big_hypot = big * numext::sqrt(Real(2));
  VERIFY_IS_EQUAL(lapy2(Real(3), Real(4)), Real(5));
  VERIFY(numext::abs(lapy2(big, big) - big_hypot) <= 4 * eps * big_hypot);
  VERIFY((numext::isinf)(lapy2(inf, Real(3))));
  // Unlike hypot, xLAPY2 propagates a NaN even when the other argument is infinite.
  VERIFY((numext::isnan)(lapy2(inf, nan)));
  VERIFY((numext::isnan)(lapy2(nan, inf)));
}

void potrf(const char* uplo, int* n, double* a, int* lda, int* info) { BLASFUNC(dpotrf)(uplo, n, a, lda, info); }
void potrf(const char* uplo, int* n, complex<double>* a, int* lda, int* info) {
  BLASFUNC(zpotrf)(uplo, n, reinterpret_cast<double*>(a), lda, info);
}

// xPOTRF returns INFO = j for the first pivot j that is <= 0 or NaN.
template <typename Scalar>
void test_potrf_info() {
  using Mat = ColMat<Scalar>;
  const Scalar nan(std::numeric_limits<double>::quiet_NaN());
  Mat spd(3, 3);
  spd << Scalar(4), Scalar(1), Scalar(1), Scalar(1), Scalar(5), Scalar(1), Scalar(1), Scalar(1), Scalar(6);
  // spd with entries (i, j) and (j, i) set to value.
  const auto modified = [&](int i, int j, const Scalar& value) {
    Mat a = spd;
    a(i, j) = a(j, i) = value;
    return a;
  };
  struct Case {
    Mat a;
    int info;
  };
  const Case cases[] = {
      {spd, 0},
      {modified(0, 0, nan), 1},
      {modified(1, 1, nan), 2},
      {modified(2, 0, nan), 3},           // a NaN off the diagonal first makes pivot 3 NaN
      {modified(1, 1, Scalar(0.25)), 2},  // 0.25 - 1/4 = 0: not positive definite
  };
  for (const Case& cs : cases) {
    for (char uplo : {'L', 'U'}) {
      Mat factor = cs.a;
      int n = 3, lda = 3, info = -1;
      potrf(&uplo, &n, factor.data(), &lda, &info);
      VERIFY_IS_EQUAL(info, cs.info);
    }
  }
}

// The same in a matrix large enough for the blocked kernel, with the NaN pivot in the first or a later block, and with
// a later non-positive pivot that must not mask it. In the diagonal matrix the block solve and the trailing update
// multiply the NaN only by zeros, so the NaN reaches no later pivot if a kernel skips zero operands, as reference BLAS
// does.
template <typename Scalar>
void test_potrf_info_blocked() {
  using Mat = ColMat<Scalar>;
  const int size = 100;
  const Scalar nan(std::numeric_limits<double>::quiet_NaN());
  const Mat dense = Mat::Constant(size, size, Scalar(1)) + Scalar(size) * Mat::Identity(size, size);
  const Mat diagonal = Scalar(size) * Mat::Identity(size, size);
  struct Case {
    const Mat* spd;
    int nan_pivot, negative_pivot;  // -1 for none
    int info;
  };
  const Case cases[] = {
      {&dense, -1, -1, 0}, {&dense, 5, 80, 6}, {&dense, 64, -1, 65}, {&dense, -1, 80, 81}, {&diagonal, 5, 80, 6}};
  for (const Case& cs : cases) {
    for (char uplo : {'L', 'U'}) {
      Mat a = *cs.spd;
      if (cs.nan_pivot >= 0) a(cs.nan_pivot, cs.nan_pivot) = nan;
      if (cs.negative_pivot >= 0) a(cs.negative_pivot, cs.negative_pivot) = Scalar(-1000);
      int n = size, lda = size, info = -1;
      potrf(&uplo, &n, a.data(), &lda, &info);
      VERIFY_IS_EQUAL(info, cs.info);
    }
  }
}

// xPOTRF neither reads nor writes the triangle that UPLO excludes, in the unblocked or the blocked kernel. NaN fills
// that strict triangle, including its parts inside the diagonal blocks.
template <typename Scalar>
void test_potrf_unreferenced_nan() {
  using Mat = ColMat<Scalar>;
  const Scalar nan(std::numeric_limits<double>::quiet_NaN());
  for (int size : {3, 100}) {
    for (char uplo : {'L', 'U'}) {
      const auto excluded = [uplo](int i, int j) { return uplo == 'L' ? i < j : i > j; };
      Mat a = Mat::Constant(size, size, Scalar(1)) + Scalar(size) * Mat::Identity(size, size);
      for (int j = 0; j < size; ++j)
        for (int i = 0; i < size; ++i)
          if (excluded(i, j)) a(i, j) = nan;
      int n = size, lda = size, info = -1;
      potrf(&uplo, &n, a.data(), &lda, &info);
      VERIFY_IS_EQUAL(info, 0);
      bool nan_exactly_where_excluded = true;
      for (int j = 0; j < size; ++j)
        for (int i = 0; i < size; ++i)
          nan_exactly_where_excluded &= excluded(i, j) == (numext::isnan)(numext::real(a(i, j)));
      VERIFY(nan_exactly_where_excluded);
    }
  }
}

// ZPOTRF uses only the real part of a diagonal entry, so a NaN imaginary part there is not a NaN pivot.
void test_potrf_imaginary_nan() {
  using Mat = ColMat<complex<double> >;
  const int size = 100;
  Mat a = Mat::Constant(size, size, 1) + double(size) * Mat::Identity(size, size);
  a(2, 2) = complex<double>(size + 1, std::numeric_limits<double>::quiet_NaN());
  a(70, 70) = complex<double>(size + 1, std::numeric_limits<double>::quiet_NaN());
  for (char uplo : {'L', 'U'}) {
    Mat factor = a;
    int n = size, lda = size, info = -1;
    potrf(&uplo, &n, factor.data(), &lda, &info);
    VERIFY_IS_EQUAL(info, 0);
  }
}

// Reference xGESDD documents INFO = -4 for a NaN in A.
void test_gesdd_nan() {
  const char jobz = 'A';
  int m = 3, n = 2, lda = 3, ldu = 3, ldvt = 2, lwork = 64, info = 0;
  std::vector<double> s(2), work(128), rwork(128);
  std::vector<int> iwork(16);

  ColMat<double> A = ColMat<double>::Ones(3, 2), U(3, 3), VT(2, 2);
  A(1, 1) = std::numeric_limits<double>::quiet_NaN();
  BLASFUNC(dgesdd)
  (&jobz, &m, &n, A.data(), &lda, s.data(), U.data(), &ldu, VT.data(), &ldvt, work.data(), &lwork, iwork.data(), &info);
  VERIFY_IS_EQUAL(info, -4);

  ColMat<complex<double> > Z = ColMat<complex<double> >::Ones(3, 2), ZU(3, 3), ZVT(2, 2);
  Z(2, 0) = complex<double>(1, std::numeric_limits<double>::quiet_NaN());
  info = 0;
  BLASFUNC(zgesdd)
  (&jobz, &m, &n, reinterpret_cast<double*>(Z.data()), &lda, s.data(), reinterpret_cast<double*>(ZU.data()), &ldu,
   reinterpret_cast<double*>(ZVT.data()), &ldvt, work.data(), &lwork, rwork.data(), iwork.data(), &info);
  VERIFY_IS_EQUAL(info, -4);
}

// The drivers allocate their own workspace and read WORK only for a workspace query, so one element is enough.
void gesdd(char jobz, ColMat<double>& a, std::vector<double>& s, ColMat<double>& u, ColMat<double>& vt, int& info) {
  int m = int(a.rows()), n = int(a.cols()), lda = m, ldu = m, ldvt = n, lwork = 1;
  double work[1];
  std::vector<int> iwork(8 * s.size());
  BLASFUNC(dgesdd)
  (&jobz, &m, &n, a.data(), &lda, s.data(), u.data(), &ldu, vt.data(), &ldvt, work, &lwork, iwork.data(), &info);
}
void gesdd(char jobz, ColMat<complex<double> >& a, std::vector<double>& s, ColMat<complex<double> >& u,
           ColMat<complex<double> >& vt, int& info) {
  int m = int(a.rows()), n = int(a.cols()), lda = m, ldu = m, ldvt = n, lwork = 1;
  double work[2], rwork[1];
  std::vector<int> iwork(8 * s.size());
  BLASFUNC(zgesdd)
  (&jobz, &m, &n, reinterpret_cast<double*>(a.data()), &lda, s.data(), reinterpret_cast<double*>(u.data()), &ldu,
   reinterpret_cast<double*>(vt.data()), &ldvt, work, &lwork, rwork, iwork.data(), &info);
}
void gesvd(char jobu, char jobvt, ColMat<double>& a, std::vector<double>& s, ColMat<double>& u, ColMat<double>& vt,
           int& info) {
  int m = int(a.rows()), n = int(a.cols()), lda = m, ldu = m, ldvt = n, lwork = 1;
  double work[1];
  BLASFUNC(dgesvd)
  (&jobu, &jobvt, &m, &n, a.data(), &lda, s.data(), u.data(), &ldu, vt.data(), &ldvt, work, &lwork, &info);
}
void gesvd(char jobu, char jobvt, ColMat<complex<double> >& a, std::vector<double>& s, ColMat<complex<double> >& u,
           ColMat<complex<double> >& vt, int& info) {
  int m = int(a.rows()), n = int(a.cols()), lda = m, ldu = m, ldvt = n, lwork = 1;
  double work[2], rwork[1];
  BLASFUNC(zgesvd)
  (&jobu, &jobvt, &m, &n, reinterpret_cast<double*>(a.data()), &lda, s.data(), reinterpret_cast<double*>(u.data()),
   &ldu, reinterpret_cast<double*>(vt.data()), &ldvt, work, &lwork, rwork, &info);
}

// Reference xGESDD and xGESVD turn an Inf in A into NaN singular values with INFO = 0; Eigen's write NaN to S and to
// each requested U, V^H or overwritten A. Reference xGESVD's INFO for a NaN depends on the size (0 at 3 x 2, 17 at
// 20 x 18), so Eigen's treats a NaN as an Inf. n >= 16 reaches BDCSVD's divide and conquer; xGESDD takes
// JOBZ = jobu = jobvt.
template <typename Scalar>
void check_svd_nonfinite(bool divide_and_conquer, char jobu, char jobvt, int m, int n, double bad) {
  using Mat = ColMat<Scalar>;
  const int k = (std::min)(m, n);
  Mat a = Mat::Ones(m, n), u = Mat::Zero(m, m), vt = Mat::Zero(n, n);
  a(1, 1) = Scalar(bad);
  std::vector<double> s(k, 0);
  int info = -1;
  if (divide_and_conquer)
    gesdd(jobu, a, s, u, vt, info);
  else
    gesvd(jobu, jobvt, a, s, u, vt, info);
  VERIFY_IS_EQUAL(info, 0);
  VERIFY(Map<VectorXd>(s.data(), k).array().isNaN().all());
  const int u_cols = jobu == 'A' ? m : jobu == 'N' ? 0 : k;
  const int vt_rows = jobvt == 'A' ? n : jobvt == 'N' ? 0 : k;
  // JOBZ = 'O' overwrites A with U if m >= n and with V^H otherwise.
  const bool u_in_a = jobu == 'O' && (!divide_and_conquer || m >= n);
  const bool vt_in_a = jobvt == 'O' && (!divide_and_conquer || m < n);
  VERIFY((u_in_a ? a : u).leftCols(u_cols).array().isNaN().all());
  VERIFY((vt_in_a ? a : vt).topRows(vt_rows).array().isNaN().all());
  if (!u_in_a) VERIFY(u.rightCols(m - u_cols).isZero(0));
  if (!vt_in_a) VERIFY(vt.bottomRows(n - vt_rows).isZero(0));
}

template <typename Scalar>
void test_svd_nonfinite() {
  const double inf = std::numeric_limits<double>::infinity(), nan = std::numeric_limits<double>::quiet_NaN();
  const char gesvd_jobs[][2] = {{'A', 'A'}, {'S', 'S'}, {'N', 'N'}, {'O', 'S'}, {'S', 'O'}, {'N', 'O'}};
  for (int size : {2, 18}) {
    for (int wide = 0; wide < 2; ++wide) {
      const int m = wide ? size : size + 1, n = wide ? size + 1 : size;
      for (char jobz : {'A', 'S', 'N', 'O'}) check_svd_nonfinite<Scalar>(true, jobz, jobz, m, n, inf);
      for (const auto& jobs : gesvd_jobs)
        for (double bad : {inf, nan}) check_svd_nonfinite<Scalar>(false, jobs[0], jobs[1], m, n, bad);
    }
  }
}

void test_lacgv() {
  complex<double> x[2] = {complex<double>(1, 2), complex<double>(3, 4)};
  int n = 2, inc = 1;
  BLASFUNC(zlacgv)(&n, reinterpret_cast<double*>(x), &inc);
  VERIFY_IS_EQUAL(x[1], complex<double>(3, -4));
  n = -1;  // the reference loop runs zero times
  BLASFUNC(zlacgv)(&n, reinterpret_cast<double*>(x), &inc);
  VERIFY_IS_EQUAL(x[0], complex<double>(1, -2));
}

}  // namespace

EIGEN_DECLARE_TEST(lapack_routines) {
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST(test_larfb<double>());
    CALL_SUBTEST(test_larfb<complex<double> >());
  }
  CALL_SUBTEST(test_larfb_nan_propagation());
  CALL_SUBTEST(test_ladiv());
  CALL_SUBTEST(test_ladiv_special<float>());
  CALL_SUBTEST(test_ladiv_special<double>());
  CALL_SUBTEST(test_lapy2<float>());
  CALL_SUBTEST(test_lapy2<double>());
  CALL_SUBTEST(test_potrf_info<double>());
  CALL_SUBTEST(test_potrf_info<complex<double> >());
  CALL_SUBTEST(test_potrf_info_blocked<double>());
  CALL_SUBTEST(test_potrf_info_blocked<complex<double> >());
  CALL_SUBTEST(test_potrf_unreferenced_nan<double>());
  CALL_SUBTEST(test_potrf_unreferenced_nan<complex<double> >());
  CALL_SUBTEST(test_potrf_imaginary_nan());
  CALL_SUBTEST(test_gesdd_nan());
  CALL_SUBTEST(test_svd_nonfinite<double>());
  CALL_SUBTEST(test_svd_nonfinite<complex<double> >());
  CALL_SUBTEST(test_lacgv());
}
