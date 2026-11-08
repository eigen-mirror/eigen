// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Checks of the LAPACK routines that Eigen's library defines on top of its own decompositions, none of which needs a
// Fortran compiler: the contract at the edges (rank deficiency, singular and non-finite input, untouched arguments,
// XERBLA, workspace queries), the pivots reference LAPACK chooses, and the packed factors, rebuilt by their documented
// convention rather than through the library's own routines. lapack_wrappers_ilp64 builds them with 64-bit integers.
//
// Error model of the residual checks, with n the order, eps = epsilon<Real>, and norms Frobenius: a factorization
// computed by Householder QR, Cholesky or Bunch-Kaufman with partial pivoting has a backward error
// ||dA|| <= c n eps ||A|| (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., Thms 10.3, 19.4); the
// factor c at each site is set from the measured errors of the four scalar types and carries a margin of at least 4.

#include "main.h"
#include "../lapack.h"

namespace {

using std::complex;
// LAPACK's INTEGER: 64-bit when this file is built as lapack_wrappers_ilp64.
using Int = EIGEN_BLAS_INT;

template <typename Scalar>
using RealOf = typename NumTraits<Scalar>::Real;
// Column-major whatever EIGEN_DEFAULT_TO_ROW_MAJOR says: the routines read data() with a leading dimension.
template <typename Scalar>
using Mat = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
template <typename Scalar>
using Vec = Matrix<Scalar, Dynamic, 1>;

// The routines of the four scalar types by one name: pstrf, bktrf and bktrs for x{SY,HE}TRF/S, qrgen and qrmul for
// xORGQR/xUNGQR and xORMQR/xUNMQR. The arguments are those of the Fortran routine, with the complex ones as pointers
// to the real type (see rp()).
template <typename Scalar>
struct Lapack;

#define LW_MEMBER(NAME, FULL)      \
  template <typename... Args>      \
  static void NAME(Args... args) { \
    BLASFUNC(FULL)(args...);       \
  }
#define LW_LAPACK(SCALAR, P, SY, OR) \
  template <>                        \
  struct Lapack<SCALAR> {            \
    LW_MEMBER(potrf, P##potrf)       \
    LW_MEMBER(pstrf, P##pstrf)       \
    LW_MEMBER(bktrf, P##SY##trf)     \
    LW_MEMBER(bktrs, P##SY##trs)     \
    LW_MEMBER(trtri, P##trtri)       \
    LW_MEMBER(trtrs, P##trtrs)       \
    LW_MEMBER(lauum, P##lauum)       \
    LW_MEMBER(potri, P##potri)       \
    LW_MEMBER(getrf, P##getrf)       \
    LW_MEMBER(getri, P##getri)       \
    LW_MEMBER(geqrf, P##geqrf)       \
    LW_MEMBER(qrgen, P##OR##gqr)     \
    LW_MEMBER(qrmul, P##OR##mqr)     \
  };
LW_LAPACK(float, s, sy, or)
LW_LAPACK(double, d, sy, or)
LW_LAPACK(complex<float>, c, he, un)
LW_LAPACK(complex<double>, z, he, un)
#undef LW_LAPACK
#undef LW_MEMBER

template <typename Scalar>
RealOf<Scalar>* rp(Scalar* p) {
  return reinterpret_cast<RealOf<Scalar>*>(p);
}

void ev(const char* jobz, const char* uplo, Int* n, float* a, Int* lda, float* w, float* work, Int* lwork, Int* info) {
  BLASFUNC(ssyev)(jobz, uplo, n, a, lda, w, work, lwork, info);
}
void ev(const char* jobz, const char* uplo, Int* n, double* a, Int* lda, double* w, double* work, Int* lwork,
        Int* info) {
  BLASFUNC(dsyev)(jobz, uplo, n, a, lda, w, work, lwork, info);
}
void ev(const char* jobz, const char* uplo, Int* n, complex<float>* a, Int* lda, float* w, complex<float>* work,
        Int* lwork, Int* info) {
  std::vector<float> rwork((std::max)(Int(1), 3 * *n - 2));
  BLASFUNC(cheev)
  (jobz, uplo, n, reinterpret_cast<float*>(a), lda, w, reinterpret_cast<float*>(work), lwork, rwork.data(), info);
}
void ev(const char* jobz, const char* uplo, Int* n, complex<double>* a, Int* lda, double* w, complex<double>* work,
        Int* lwork, Int* info) {
  std::vector<double> rwork((std::max)(Int(1), 3 * *n - 2));
  BLASFUNC(zheev)
  (jobz, uplo, n, reinterpret_cast<double*>(a), lda, w, reinterpret_cast<double*>(work), lwork, rwork.data(), info);
}

// What the library reports to XERBLA. Eigen's BLAS defines xerbla_ weakly, so this definition replaces its printf.
struct XerblaRecord {
  Int calls = 0;
  Int info = 0;
  std::string name;
};
XerblaRecord g_xerbla;

void reset_xerbla() { g_xerbla = XerblaRecord(); }

// Exactly one call, with this routine name and positive argument number; consumes it.
void verify_xerbla(const std::string& name, Int info) {
  VERIFY_IS_EQUAL(g_xerbla.calls, 1);
  VERIFY_IS_EQUAL(g_xerbla.info, info);
  VERIFY(g_xerbla.name == name);
  reset_xerbla();
}

// The name that the routine of Scalar's precision passes to XERBLA, given the name without its precision prefix for
// real and for complex Scalar.
template <typename Scalar>
std::string xerbla_name(const char* real_name, const char* complex_name) {
  const bool single = sizeof(RealOf<Scalar>) == sizeof(float);
  if (NumTraits<Scalar>::IsComplex) return std::string(single ? "C" : "Z") + complex_name;
  return std::string(single ? "S" : "D") + real_name;
}

}  // namespace

void BLASFUNC(xerbla)(const char* msg, EIGEN_BLAS_INT* info, size_t len) {
  ++g_xerbla.calls;
  g_xerbla.info = *info;
  g_xerbla.name.assign(msg, len);
  while (!g_xerbla.name.empty() && g_xerbla.name.back() == ' ') g_xerbla.name.pop_back();
}

namespace {

template <typename Scalar>
Scalar make_nan() {
  return Scalar(std::numeric_limits<RealOf<Scalar> >::quiet_NaN());
}

template <typename Real>
void set_imag(Real&, Real) {}
template <typename Real>
void set_imag(complex<Real>& x, Real imag) {
  x = complex<Real>(x.real(), imag);
}

// Bitwise equality, which tells signed zeros and NaN payloads apart.
template <typename Real>
bool same_bits(const Real& a, const Real& b) {
  using Bits = typename numext::get_integer_by_size<sizeof(Real)>::unsigned_type;
  return numext::bit_cast<Bits>(a) == numext::bit_cast<Bits>(b);
}

template <typename Real>
bool same_bits(const complex<Real>& a, const complex<Real>& b) {
  return same_bits(a.real(), b.real()) && same_bits(a.imag(), b.imag());
}

template <typename Scalar>
bool same_bits(const Mat<Scalar>& a, const Mat<Scalar>& b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
         a.binaryExpr(b, [](const Scalar& x, const Scalar& y) { return same_bits(x, y); }).all();
}

// NaN in the strict triangle of A that UPLO does not reference.
template <typename Scalar>
void poison_opposite(Mat<Scalar>& A, char uplo) {
  if (uplo == 'L')
    A.template triangularView<StrictlyUpper>().setConstant(make_nan<Scalar>());
  else
    A.template triangularView<StrictlyLower>().setConstant(make_nan<Scalar>());
}

template <typename Scalar>
bool opposite_untouched(const Mat<Scalar>& A, const Mat<Scalar>& before, char uplo) {
  for (Index j = 0; j < A.cols(); ++j)
    for (Index i = 0; i < A.rows(); ++i)
      if ((uplo == 'L' ? i < j : i > j) && !same_bits(A(i, j), before(i, j))) return false;
  return true;
}

// The UPLO triangle of A, as a full matrix, with a unit diagonal if UNIT.
template <typename Scalar>
Mat<Scalar> triangle(const Mat<Scalar>& A, char uplo, bool unit) {
  Mat<Scalar> T = Mat<Scalar>::Zero(A.rows(), A.cols());
  if (uplo == 'L')
    T.template triangularView<Lower>() = A;
  else
    T.template triangularView<Upper>() = A;
  if (unit) T.diagonal().setOnes();
  return T;
}

template <typename Scalar>
Mat<Scalar> random_hermitian(Int n) {
  const Mat<Scalar> M = Mat<Scalar>::Random(n, n);
  return Mat<Scalar>((M + M.adjoint()) * RealOf<Scalar>(0.5));  // exactly Hermitian, real diagonal
}

// A = G G^H, G n x r: positive semidefinite of rank r.
template <typename Scalar>
Mat<Scalar> random_psd(Int n, Int r) {
  const Mat<Scalar> G = Mat<Scalar>::Random(n, r);
  const Mat<Scalar> A = G * G.adjoint();
  return Mat<Scalar>((A + A.adjoint()) * RealOf<Scalar>(0.5));
}

// Well-conditioned triangular: off-diagonal entries O(1/n), diagonal entries of modulus >= 1.
template <typename Scalar>
Mat<Scalar> random_triangular(Int n, char uplo) {
  const Mat<Scalar> R = Mat<Scalar>::Random(n, n);
  Mat<Scalar> T = triangle<Scalar>(Mat<Scalar>(R / RealOf<Scalar>(n)), uplo, false);
  for (Int i = 0; i < n; ++i) T(i, i) = Scalar(RealOf<Scalar>(1.5)) + Scalar(RealOf<Scalar>(0.5)) * R(i, i);
  return T;
}

// ---------------------------------------------------------------------------------------------------------------
// xPSTRF

template <typename Scalar>
Int call_pstrf(char uplo, Mat<Scalar>& A, std::vector<Int>& piv, Int& rank, RealOf<Scalar> tol) {
  Int n = Int(A.rows()), lda = n, info = -99;
  piv.assign(n, 0);
  std::vector<RealOf<Scalar> > work(2 * size_t(n));
  Lapack<Scalar>::pstrf(&uplo, &n, rp(A.data()), &lda, piv.data(), &rank, &tol, work.data(), &info);
  return info;
}

// PIV is a permutation of 1..n, and P^T A P = F F^H for the leading RANK columns F of the factor (rows U of the
// factor for UPLO = 'U', F = U^H). The factor is a Cholesky factor, so the backward error is that of Cholesky; the
// trailing Schur complement dropped at the stopping pivot adds at most n times the stopping tolerance.
template <typename Scalar>
void check_pstrf_factor(char uplo, const Mat<Scalar>& A0, const Mat<Scalar>& A, const std::vector<Int>& piv, Int rank) {
  using Real = RealOf<Scalar>;
  const Int n = Int(A0.rows());
  std::vector<Int> sorted(piv);
  std::sort(sorted.begin(), sorted.end());
  for (Int i = 0; i < n; ++i) VERIFY_IS_EQUAL(sorted[i], i + 1);

  Mat<Scalar> permuted(n, n);
  for (Int j = 0; j < n; ++j)
    for (Int i = 0; i < n; ++i) permuted(i, j) = A0(piv[i] - 1, piv[j] - 1);
  Mat<Scalar> F = Mat<Scalar>::Zero(n, rank);
  if (uplo == 'L') {
    F.template triangularView<Lower>() = A.leftCols(rank);
  } else {
    Mat<Scalar> U = Mat<Scalar>::Zero(rank, n);
    U.template triangularView<Upper>() = A.topRows(rank);
    F = U.adjoint();
  }
  const Real eps = NumTraits<Real>::epsilon();
  const Real err = (permuted - F * F.adjoint()).norm();
  VERIFY(err <= 8 * n * eps * A0.norm());
}

template <typename Scalar>
void test_pstrf() {
  using Real = RealOf<Scalar>;
  const Real eps = NumTraits<Real>::epsilon();
  const struct {
    Int n, r;
  } deficient[] = {{20, 5}, {150, 1}, {150, 16}, {150, 40}};  // 150 > 96 takes the blocked LDLT panels
  for (char uplo : {'L', 'U'}) {
    for (const auto& c : deficient) {
      const Mat<Scalar> A0 = random_psd<Scalar>(c.n, c.r);
      Mat<Scalar> A = A0;
      poison_opposite(A, uplo);
      const Mat<Scalar> before = A;
      std::vector<Int> piv;
      Int rank = -1;
      VERIFY_IS_EQUAL(call_pstrf(uplo, A, piv, rank, Real(-1)), 1);
      VERIFY_IS_EQUAL(rank, c.r);
      check_pstrf_factor(uplo, A0, A, piv, rank);
      VERIFY(opposite_untouched(A, before, uplo));
    }

    for (Int n : {1, 5, 97, 150}) {
      const Mat<Scalar> A0 = Mat<Scalar>(random_psd<Scalar>(n, n) + Scalar(Real(n)) * Mat<Scalar>::Identity(n, n));
      Mat<Scalar> A = A0;
      poison_opposite(A, uplo);
      std::vector<Int> piv;
      Int rank = -1;
      VERIFY_IS_EQUAL(call_pstrf(uplo, A, piv, rank, Real(-1)), 0);
      VERIFY_IS_EQUAL(rank, n);
      check_pstrf_factor(uplo, A0, A, piv, rank);
    }

    // The largest diagonal entry is <= 0, whether negative or exactly zero: nothing is factored.
    for (const Real top : {Real(-1), Real(0)}) {
      for (Int n : {1, 4, 100}) {
        Mat<Scalar> A = Mat<Scalar>::Random(n, n);
        for (Int i = 0; i < n; ++i) A(i, i) = Scalar(i == n / 2 ? top : Real(-1) - Real(i));
        const Mat<Scalar> before = A;
        std::vector<Int> piv;
        Int rank = -1;
        VERIFY_IS_EQUAL(call_pstrf(uplo, A, piv, rank, Real(-1)), 1);
        VERIFY_IS_EQUAL(rank, 0);
        VERIFY(same_bits(A, before));
        for (Int i = 0; i < n; ++i) VERIFY_IS_EQUAL(piv[i], i + 1);
      }
    }

    // A NaN TOL never satisfies d <= TOL. The default tolerance stops at the pivot 4t <= 3 eps/2, TOL = 0 and a NaN
    // TOL continue to the last: P^T A P = diag(1, 4t, t) with pivots in that order.
    {
      const Real t = eps / 4;
      Mat<Scalar> A0 = Mat<Scalar>::Zero(3, 3);
      A0.diagonal() << Scalar(t), Scalar(1), Scalar(4 * t);
      for (const Real tol : {Real(-1), Real(0), std::numeric_limits<Real>::quiet_NaN()}) {
        Mat<Scalar> A = A0;
        std::vector<Int> piv;
        Int rank = -1;
        const Int info = call_pstrf(uplo, A, piv, rank, tol);
        if (tol < 0) {
          VERIFY_IS_EQUAL(info, 1);
          VERIFY_IS_EQUAL(rank, 1);
        } else {
          VERIFY_IS_EQUAL(info, 0);
          VERIFY_IS_EQUAL(rank, 3);
          check_pstrf_factor(uplo, A0, A, piv, rank);
          VERIFY_IS_EQUAL(piv[0], 2);
          VERIFY_IS_EQUAL(piv[1], 3);
          VERIFY_IS_EQUAL(piv[2], 1);
        }
      }
    }

    // Indefinite: the pivot -2 of diag(1, -2) is <= the tolerance, so xPSTRF stops after the first pivot.
    {
      Mat<Scalar> A = Mat<Scalar>::Zero(2, 2);
      A.diagonal() << Scalar(1), Scalar(-2);
      std::vector<Int> piv;
      Int rank = -1;
      VERIFY_IS_EQUAL(call_pstrf(uplo, A, piv, rank, Real(-1)), 1);
      VERIFY_IS_EQUAL(rank, 1);
      VERIFY_IS_EQUAL(piv[0], 1);
      VERIFY_IS_EQUAL(A(0, 0), Scalar(1));
    }

    // NaN diagonal entries. The pivot search skips them, as MAXLOC does: diag(4, NaN, 1) takes 4 and 1 and stops at
    // the NaN. A NaN in A(1,1) stops the real routines before anything is factored (their first scan starts there),
    // while the complex ones' MAXLOC skips it.
    {
      const Real nan = std::numeric_limits<Real>::quiet_NaN();
      for (Int nan_first = 0; nan_first < 2; ++nan_first) {
        Mat<Scalar> A = Mat<Scalar>::Zero(3, 3);
        if (nan_first)
          A.diagonal() << Scalar(nan), Scalar(4), Scalar(1);
        else
          A.diagonal() << Scalar(4), Scalar(nan), Scalar(1);
        std::vector<Int> piv;
        Int rank = -1;
        VERIFY_IS_EQUAL(call_pstrf(uplo, A, piv, rank, Real(-1)), 1);
        const Int expected_rank = (nan_first && !NumTraits<Scalar>::IsComplex) ? 0 : 2;
        VERIFY_IS_EQUAL(rank, expected_rank);
        if (expected_rank == 2) {
          VERIFY_IS_EQUAL(numext::real(A(0, 0)), Real(2));
          VERIFY_IS_EQUAL(numext::real(A(1, 1)), Real(1));
        }
      }
      // The vectorized search: a NaN at n/2 just before the largest entry, with smaller ones on both sides.
      for (Int n : {4, 9, 20}) {
        Mat<Scalar> A = Mat<Scalar>::Zero(n, n);
        for (Int i = 0; i < n; ++i) A(i, i) = Scalar(Real(1 + i));
        A(n / 2, n / 2) = Scalar(nan);
        A(n / 2 + 1, n / 2 + 1) = Scalar(Real(4 * n));
        std::vector<Int> piv;
        Int rank = -1;
        VERIFY_IS_EQUAL(call_pstrf(uplo, A, piv, rank, Real(-1)), 1);
        VERIFY_IS_EQUAL(rank, n - 1);
        VERIFY_IS_EQUAL(piv[0], n / 2 + 2);
        VERIFY_IS_EQUAL(piv[n - 1], n / 2 + 1);
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// xSYTRF/xSYTRS and xHETRF/xHETRS

// Query, then factor in place. Returns INFO.
template <typename Scalar>
Int call_bktrf(char uplo, Mat<Scalar>& A, std::vector<Int>& ipiv) {
  Int n = Int(A.rows()), lda = n, lwork = -1, info = -99;
  ipiv.assign(n, 0);
  Scalar query(-1);
  const Mat<Scalar> before = A;
  Lapack<Scalar>::bktrf(&uplo, &n, rp(A.data()), &lda, ipiv.data(), rp(&query), &lwork, &info);
  VERIFY_IS_EQUAL(info, 0);
  VERIFY(numext::real(query) >= 1);
  VERIFY(same_bits(A, before));
  lwork = Int(numext::real(query));
  std::vector<Scalar> work(lwork);
  Lapack<Scalar>::bktrf(&uplo, &n, rp(A.data()), &lda, ipiv.data(), rp(work.data()), &lwork, &info);
  return info;
}

// Blocks of IPIV pair up as in LAPACK: IPIV(i) > 0 is a 1x1 block interchanged with row IPIV(i), which is >= i for
// UPLO = 'L' (columns are eliminated first to last) and <= i for 'U' (last to first); a 2x2 block takes two equal
// negative entries.
void check_ipiv(char uplo, const std::vector<Int>& ipiv) {
  const Int n = Int(ipiv.size());
  for (Int i = 0; i < n;) {
    const Int p = ipiv[i];
    VERIFY(p != 0 && std::abs(p) <= n);
    VERIFY(uplo == 'L' ? std::abs(p) >= i + 1 : std::abs(p) <= i + 1);
    if (p > 0) {
      ++i;
    } else {
      VERIFY(i + 1 < n && ipiv[i + 1] == p);
      i += 2;
    }
  }
}

// A = M D M^H rebuilt from F and IPIV by xSYTRF's documented storage, not through the library's xSYTRS: M = P(1) L(1)
// P(2) L(2) ... for 'L', P(n) U(n) P(n-1) U(n-1) ... for 'U', P(k) swapping |IPIV(k)| with k (k+1 for 'L', k-1 for 'U'
// if D(k) is 2x2), L(k) and U(k) unit triangular with block k's multipliers. Bunch-Kaufman's backward error is
// ||A - M D M^H|| <= c n eps (||A|| + || |M| |D| |M|^T ||) (Higham, Thm 11.3); xHETRF writes D's diagonal real.
template <typename Scalar>
void check_bk_factor(char uplo, const Mat<Scalar>& A0, const Mat<Scalar>& F, const std::vector<Int>& ipiv) {
  using Real = RealOf<Scalar>;
  const Index n = A0.rows();
  Mat<Scalar> M = Mat<Scalar>::Identity(n, n), D = Mat<Scalar>::Zero(n, n);
  for (Index step = 0; step < n;) {
    const bool lower = uplo == 'L';
    const Index k = lower ? step : n - 1 - step;
    const Index s = ipiv[k] > 0 ? 1 : 2;
    const Index first = lower ? k : k - s + 1;  // the block is rows and columns first .. first + s - 1
    M.col(lower ? first + s - 1 : first).swap(M.col(std::abs(ipiv[k]) - 1));
    if (lower)
      M.middleCols(first, s) += M.rightCols(n - first - s) * F.block(first + s, first, n - first - s, s);
    else
      M.middleCols(first, s) += M.leftCols(first) * F.block(0, first, first, s);
    D(first, first) = F(first, first);
    if (s == 2) {
      const Scalar d21 = lower ? F(first + 1, first) : numext::conj(F(first, first + 1));
      D(first + 1, first) = d21;
      D(first, first + 1) = numext::conj(d21);
      D(first + 1, first + 1) = F(first + 1, first + 1);
    }
    step += s;
  }
  for (Index i = 0; i < n; ++i) VERIFY(numext::imag(D(i, i)) == Real(0));
  VERIFY(M.allFinite() && D.allFinite());  // the bound below scales with them
  const Real err = (A0 - M * D * M.adjoint()).norm();
  const Real scale = A0.norm() + (M.cwiseAbs() * D.cwiseAbs() * M.cwiseAbs().transpose()).norm();
  VERIFY(err <= 8 * Real(n) * NumTraits<Real>::epsilon() * scale);
}

// Solves A X = B with the factor F and checks ||A X - B|| <= c n eps ||A|| ||X||, the normwise backward error of
// a solve with a stable factorization. A and the factor are left alone.
template <typename Scalar>
void check_bktrs(char uplo, const Mat<Scalar>& A0, const Mat<Scalar>& F, const std::vector<Int>& ipiv) {
  using Real = RealOf<Scalar>;
  Int n = Int(A0.rows()), lda = n, nrhs = 3, ldb = n, info = -99;
  const Mat<Scalar> B = Mat<Scalar>::Random(n, nrhs);
  Mat<Scalar> X = B;
  Mat<Scalar> factor = F;
  std::vector<Int> pivots = ipiv;
  Lapack<Scalar>::bktrs(&uplo, &n, &nrhs, rp(factor.data()), &lda, pivots.data(), rp(X.data()), &ldb, &info);
  VERIFY_IS_EQUAL(info, 0);
  VERIFY(same_bits(factor, F) && pivots == ipiv);
  VERIFY(X.allFinite());  // the bound below scales with ||X||
  const Real err = (A0 * X - B).norm();
  VERIFY(err <= 8 * n * NumTraits<Real>::epsilon() * A0.norm() * X.norm());
}

template <typename Scalar>
void test_bunch_kaufman() {
  for (char uplo : {'L', 'U'}) {
    for (Int n : {1, 2, 5, 65, 129}) {  // 65 and 129 take the blocked panels
      for (bool zero_diagonal : {false, true}) {
        if (zero_diagonal && n == 1) continue;  // singular
        Mat<Scalar> A0 = random_hermitian<Scalar>(n);
        if (zero_diagonal) A0.diagonal().setZero();  // forces 2x2 pivots
        Mat<Scalar> A = A0;
        poison_opposite(A, uplo);
        const Mat<Scalar> before = A;
        std::vector<Int> ipiv;
        VERIFY_IS_EQUAL(call_bktrf(uplo, A, ipiv), 0);
        check_ipiv(uplo, ipiv);
        VERIFY(opposite_untouched(A, before, uplo));
        check_bk_factor(uplo, A0, A, ipiv);
        check_bktrs(uplo, A0, A, ipiv);
      }
    }
  }
}

// The reference xHETRF reads only the real part of the diagonal.
template <typename Scalar>
void test_hetrf_ignores_imaginary_diagonal() {
  using Real = RealOf<Scalar>;
  if (!NumTraits<Scalar>::IsComplex) return;
  for (char uplo : {'L', 'U'}) {
    for (Int n : {2, 5, 129}) {
      const Mat<Scalar> A0 = random_hermitian<Scalar>(n);
      Mat<Scalar> clean = A0;
      poison_opposite(clean, uplo);
      std::vector<Int> clean_ipiv;
      VERIFY_IS_EQUAL(call_bktrf(uplo, clean, clean_ipiv), 0);
      for (const Real imag : {Real(7), std::numeric_limits<Real>::quiet_NaN(), std::numeric_limits<Real>::infinity()}) {
        Mat<Scalar> dirty = A0;
        for (Int i = 0; i < n; ++i) set_imag(dirty(i, i), imag);
        poison_opposite(dirty, uplo);
        std::vector<Int> dirty_ipiv;
        VERIFY_IS_EQUAL(call_bktrf(uplo, dirty, dirty_ipiv), 0);
        VERIFY(dirty_ipiv == clean_ipiv);
        for (Int j = 0; j < n; ++j) {
          for (Int i = 0; i < n; ++i) {
            if (i == j)
              VERIFY(numext::real(dirty(i, j)) == numext::real(clean(i, j)));
            else
              VERIFY(same_bits(dirty(i, j), clean(i, j)));
          }
        }
        check_bk_factor(uplo, A0, dirty, dirty_ipiv);
        check_bktrs(uplo, A0, dirty, dirty_ipiv);
      }
    }
  }
}

// The pivots reference LAPACK 3.12.1 takes on inputs that pin down its search: IxAMAX
// measures a complex candidate by |re| + |im|, which prefers 2+2i to 3, and takes the first of equal candidates,
// which is the last in the reversed order that UPLO = 'U' is factored in.
template <typename Scalar>
void test_bunch_kaufman_reference_pivots() {
  using Real = RealOf<Scalar>;
  Scalar c(Real(2));  // 2+2i, or 2 for real Scalar
  set_imag(c, Real(2));
  for (char uplo : {'L', 'U'}) {
    Mat<Scalar> A0(3, 3);
    std::vector<Int> expected;
    if (!NumTraits<Scalar>::IsComplex) {
      A0 << Scalar(0), Scalar(1), Scalar(1), Scalar(1), Scalar(0), Scalar(1), Scalar(1), Scalar(1), Scalar(0);
      expected = uplo == 'L' ? std::vector<Int>{-2, -2, 3} : std::vector<Int>{1, -1, -1};
    } else if (uplo == 'L') {  // column 1 below the diagonal: 3, 2+2i
      A0 << Scalar(0), Scalar(3), numext::conj(c), Scalar(3), Scalar(0), Scalar(1), c, Scalar(1), Scalar(0);
      expected = {-3, -3, 3};
    } else {  // column 3 above the diagonal: 3, 2+2i
      A0 << Scalar(0), Scalar(1), Scalar(3), Scalar(1), Scalar(0), c, Scalar(3), numext::conj(c), Scalar(0);
      expected = {1, -2, -2};
    }
    Mat<Scalar> A = A0;
    std::vector<Int> ipiv;
    VERIFY_IS_EQUAL(call_bktrf(uplo, A, ipiv), 0);
    VERIFY(ipiv == expected);
    check_bk_factor(uplo, A0, A, ipiv);
  }
}

// A 2x2 block [0 s; s 0] whose determinant -s^2 overflows or underflows. xSYTRS and xHETRS solve with the block
// scaled by its off-diagonal entry, so A [2; 1] = [s; 2s] solves exactly.
template <typename Scalar>
void test_bunch_kaufman_scaled_2x2() {
  using Real = RealOf<Scalar>;
  const Real big = Real(4) * std::sqrt((std::numeric_limits<Real>::max)());
  const Real tiny = std::sqrt((std::numeric_limits<Real>::min)()) / Real(4);
  for (char uplo : {'L', 'U'}) {
    for (const Real s : {big, tiny}) {
      Mat<Scalar> A0(2, 2);
      A0 << Scalar(0), Scalar(s), Scalar(s), Scalar(0);
      Mat<Scalar> A = A0;
      std::vector<Int> ipiv;
      VERIFY_IS_EQUAL(call_bktrf(uplo, A, ipiv), 0);
      VERIFY(ipiv[0] < 0 && ipiv[1] == ipiv[0]);
      Int n = 2, nrhs = 1, lda = 2, ldb = 2, info = -99;
      Vec<Scalar> x(2);
      x << Scalar(s), Scalar(2 * s);
      Lapack<Scalar>::bktrs(&uplo, &n, &nrhs, rp(A.data()), &lda, ipiv.data(), rp(x.data()), &ldb, &info);
      VERIFY_IS_EQUAL(info, 0);
      VERIFY(x.allFinite());
      VERIFY((x - Vec<Scalar>::Constant(2, Scalar(1)) - Vec<Scalar>::Unit(2, 0)).norm() <=
             4 * NumTraits<Real>::epsilon());
    }
  }
}

// A zero row and column makes a zero pivot D(j, j), reported as INFO = j + 1 for the first one met: the smallest j
// for UPLO = 'L', which eliminates first to last, and the largest for 'U'. Diagonal dominance keeps every pivot 1x1
// and unswapped, so the zero row stays at its position j.
template <typename Scalar>
void test_bunch_kaufman_singular() {
  using Real = RealOf<Scalar>;
  for (char uplo : {'L', 'U'}) {
    for (Int n : {1, 2, 5, 65, 129}) {
      std::vector<std::vector<Int> > zero_rows = {{0}, {n / 2}, {n - 1}};
      if (n >= 5) zero_rows.push_back({1, n - 2});
      for (const auto& zeros : zero_rows) {
        Mat<Scalar> A = random_hermitian<Scalar>(n);
        for (Int i = 0; i < n; ++i) A(i, i) = Scalar(Real(i % 2 ? -1 : 1) * Real(2 * n + 2));
        for (Int j : zeros) {
          A.row(j).setZero();
          A.col(j).setZero();
        }
        std::vector<Int> ipiv;
        VERIFY_IS_EQUAL(call_bktrf(uplo, A, ipiv), (uplo == 'L' ? zeros.front() : zeros.back()) + 1);
      }
    }
  }
  // LWORK = 0 is below the minimum of 1.
  Mat<Scalar> A = Mat<Scalar>::Identity(3, 3);
  Int n = 3, lda = 3, lwork = 0, info = 0, ipiv[3];
  Scalar work(0);
  reset_xerbla();
  Lapack<Scalar>::bktrf("L", &n, rp(A.data()), &lda, ipiv, rp(&work), &lwork, &info);
  VERIFY_IS_EQUAL(info, -7);
  verify_xerbla(xerbla_name<Scalar>("SYTRF", "HETRF"), 7);
}

// ---------------------------------------------------------------------------------------------------------------
// xTRTRI, xTRTRS, xLAUUM, xPOTRI, xGETRI

template <typename Scalar>
void test_trtri() {
  using Real = RealOf<Scalar>;
  const Real eps = NumTraits<Real>::epsilon();
  for (char uplo : {'L', 'U'}) {
    for (char diag : {'N', 'U'}) {
      const bool unit = diag == 'U';
      for (Int n : {1, 2, 5, 70, 130}) {
        const Mat<Scalar> T = random_triangular<Scalar>(n, uplo);
        Mat<Scalar> A = T;
        poison_opposite(A, uplo);
        if (unit) A.diagonal().setConstant(make_nan<Scalar>());  // not referenced
        const Mat<Scalar> before = A;
        Int nn = n, lda = n, info = -99;
        Lapack<Scalar>::trtri(&uplo, &diag, &nn, rp(A.data()), &lda, &info);
        VERIFY_IS_EQUAL(info, 0);
        VERIFY(opposite_untouched(A, before, uplo));
        if (unit) VERIFY(same_bits(Mat<Scalar>(A.diagonal()), Mat<Scalar>(before.diagonal())));
        // Right residual ||X T - I||_F <= c n eps ||X|| ||T||, as in the reference's xTRT01.
        const Mat<Scalar> Teff = triangle(T, uplo, unit), X = triangle(A, uplo, unit);
        VERIFY(X.allFinite());
        const Real err = (X * Teff - Mat<Scalar>::Identity(n, n)).norm();
        VERIFY(err <= 8 * n * eps * X.norm() * Teff.norm());
      }
    }
  }
}

// Exact zeros at i = 3 and 5 of a 7 x 7 triangle: INFO = 3, and no argument is modified.
template <typename Scalar>
void test_triangular_singular() {
  const Int n = 7;
  for (char uplo : {'L', 'U'}) {
    for (char diag : {'N', 'U'}) {
      Mat<Scalar> A = random_triangular<Scalar>(n, uplo);
      A(2, 2) = Scalar(0);
      A(4, 4) = Scalar(0);
      poison_opposite(A, uplo);
      const Mat<Scalar> A_before = A;
      const Int expected = diag == 'N' ? 3 : 0;
      {
        Int nn = n, lda = n, info = -99;
        Lapack<Scalar>::trtri(&uplo, &diag, &nn, rp(A.data()), &lda, &info);
        VERIFY_IS_EQUAL(info, expected);
        if (expected > 0) VERIFY(same_bits(A, A_before));
        A = A_before;
      }
      for (char trans : {'N', 'T', 'C'}) {
        Mat<Scalar> B = Mat<Scalar>::Random(n, 2);
        const Mat<Scalar> B_before = B;
        Int nn = n, lda = n, nrhs = 2, ldb = n, info = -99;
        Lapack<Scalar>::trtrs(&uplo, &trans, &diag, &nn, &nrhs, rp(A.data()), &lda, rp(B.data()), &ldb, &info);
        VERIFY_IS_EQUAL(info, expected);
        VERIFY(same_bits(A, A_before));
        if (expected > 0) VERIFY(same_bits(B, B_before));
      }
      if (diag == 'N') {  // xPOTRI is xTRTRI then xLAUUM, and stops at the same zero
        Int nn = n, lda = n, info = -99;
        Lapack<Scalar>::potri(&uplo, &nn, rp(A.data()), &lda, &info);
        VERIFY_IS_EQUAL(info, 3);
        VERIFY(same_bits(A, A_before));
      }
    }
  }
}

template <typename Scalar>
void test_trtrs() {
  using Real = RealOf<Scalar>;
  const Real eps = NumTraits<Real>::epsilon();
  for (char uplo : {'L', 'U'}) {
    for (char diag : {'N', 'U'}) {
      const bool unit = diag == 'U';
      for (char trans : {'N', 'T', 'C'}) {
        for (Int n : {1, 5, 70}) {
          const Mat<Scalar> T = random_triangular<Scalar>(n, uplo);
          Mat<Scalar> A = T;
          poison_opposite(A, uplo);
          if (unit) A.diagonal().setConstant(make_nan<Scalar>());
          const Mat<Scalar> A_before = A;
          const Mat<Scalar> B = Mat<Scalar>::Random(n, 3);
          Mat<Scalar> X = B;
          Int nn = n, lda = n, nrhs = 3, ldb = n, info = -99;
          Lapack<Scalar>::trtrs(&uplo, &trans, &diag, &nn, &nrhs, rp(A.data()), &lda, rp(X.data()), &ldb, &info);
          VERIFY_IS_EQUAL(info, 0);
          VERIFY(same_bits(A, A_before));
          // ||op(T) X - B|| <= c n eps ||T|| ||X||: a triangular solve is backward stable, componentwise.
          const Mat<Scalar> Teff = triangle(T, uplo, unit);
          Mat<Scalar> op = Teff;
          if (trans == 'T')
            op = Teff.transpose();
          else if (trans == 'C')
            op = Teff.adjoint();
          VERIFY(X.allFinite());
          const Real err = (op * X - B).norm();
          VERIFY(err <= 8 * n * eps * Teff.norm() * X.norm());
        }
      }
    }
  }
}

template <typename Scalar>
void test_lauum() {
  using Real = RealOf<Scalar>;
  const Real eps = NumTraits<Real>::epsilon();
  for (char uplo : {'L', 'U'}) {
    for (Int n : {1, 4, 70, 130}) {
      const Mat<Scalar> T = random_triangular<Scalar>(n, uplo);
      Mat<Scalar> A = T;
      poison_opposite(A, uplo);
      const Mat<Scalar> before = A;
      Int nn = n, lda = n, info = -99;
      Lapack<Scalar>::lauum(&uplo, &nn, rp(A.data()), &lda, &info);
      VERIFY_IS_EQUAL(info, 0);
      VERIFY(opposite_untouched(A, before, uplo));
      for (Int i = 0; i < n; ++i) VERIFY(numext::imag(A(i, i)) == Real(0));
      // U U^H or L^H L, compared on the stored triangle: the entries are dot products of length <= n.
      const Mat<Scalar> E = uplo == 'U' ? Mat<Scalar>(T * T.adjoint()) : Mat<Scalar>(T.adjoint() * T);
      const Real err = (triangle(A, uplo, false) - triangle(E, uplo, false)).norm();
      VERIFY(err <= 8 * n * eps * T.norm() * T.norm());
    }
  }
}

template <typename Scalar>
void test_potri() {
  using Real = RealOf<Scalar>;
  const Real eps = NumTraits<Real>::epsilon();
  for (char uplo : {'L', 'U'}) {
    for (Int n : {1, 5, 70, 130}) {
      const Mat<Scalar> A0 = Mat<Scalar>(random_psd<Scalar>(n, n) + Scalar(Real(n)) * Mat<Scalar>::Identity(n, n));
      Mat<Scalar> A = A0;
      poison_opposite(A, uplo);
      const Mat<Scalar> before_factor = A;
      Int nn = n, lda = n, info = -99;
      Lapack<Scalar>::potrf(&uplo, &nn, rp(A.data()), &lda, &info);
      VERIFY_IS_EQUAL(info, 0);
      const Mat<Scalar> factored = A;
      Lapack<Scalar>::potri(&uplo, &nn, rp(A.data()), &lda, &info);
      VERIFY_IS_EQUAL(info, 0);
      VERIFY(opposite_untouched(A, before_factor, uplo));
      for (Int i = 0; i < n; ++i) VERIFY(numext::imag(A(i, i)) == Real(0));
      Mat<Scalar> X = uplo == 'L' ? Mat<Scalar>(A.template selfadjointView<Lower>())
                                  : Mat<Scalar>(A.template selfadjointView<Upper>());
      const Mat<Scalar> inverse = A0.partialPivLu().inverse();
      // ||X - A^-1|| <= c n eps cond(A) ||A^-1||, the forward error of a backward stable inverse.
      const Real err = (X - inverse).norm();
      VERIFY(err <= 16 * n * eps * A0.norm() * inverse.norm() * inverse.norm());
    }
  }
}

template <typename Scalar>
void test_getri() {
  using Real = RealOf<Scalar>;
  const Real eps = NumTraits<Real>::epsilon();
  for (Int n : {1, 5, 40, 130}) {  // 130 > 64: the blocked solve, if LWORK >= 2 N
    const Mat<Scalar> A0 = Mat<Scalar>::Random(n, n);
    Mat<Scalar> lu = A0;
    std::vector<Int> ipiv(n);
    Int nn = n, lda = n, info = -99;
    Lapack<Scalar>::getrf(&nn, &nn, rp(lu.data()), &lda, ipiv.data(), &info);
    VERIFY_IS_EQUAL(info, 0);

    Scalar query(-1);
    Int lwork = -1;
    const Mat<Scalar> lu_before = lu;
    Lapack<Scalar>::getri(&nn, rp(lu.data()), &lda, ipiv.data(), rp(&query), &lwork, &info);
    VERIFY_IS_EQUAL(info, 0);
    VERIFY(same_bits(lu, lu_before));
    VERIFY(numext::real(query) >= n);
    const Int optimal = Int(numext::real(query));

    for (Int w : {n, 2 * n, 64 * n, optimal}) {
      Mat<Scalar> X = lu_before;
      std::vector<Scalar> work(w);
      lwork = w;
      Lapack<Scalar>::getri(&nn, rp(X.data()), &lda, ipiv.data(), rp(work.data()), &lwork, &info);
      VERIFY_IS_EQUAL(info, 0);
      VERIFY(X.allFinite());
      // ||I - A inv(A)|| <= c n eps ||A|| ||inv(A)||, as in the reference's xGET03.
      const Real err = (Mat<Scalar>::Identity(n, n) - A0 * X).norm();
      VERIFY(err <= 8 * n * eps * A0.norm() * X.norm());
    }

    // LWORK < N is an error: XERBLA argument 6.
    if (n > 1) {
      Mat<Scalar> X = lu_before;
      std::vector<Scalar> work(n);
      lwork = n - 1;
      reset_xerbla();
      Lapack<Scalar>::getri(&nn, rp(X.data()), &lda, ipiv.data(), rp(work.data()), &lwork, &info);
      VERIFY_IS_EQUAL(info, -6);
      verify_xerbla(xerbla_name<Scalar>("GETRI", "GETRI"), 6);
      VERIFY(same_bits(X, lu_before));
    }
  }

  // A zero U(i, i) is found before anything is written: INFO = i of the first one.
  const Int n = 6;
  Mat<Scalar> lu = Mat<Scalar>::Random(n, n);
  lu(2, 2) = Scalar(0);
  lu(4, 4) = Scalar(0);
  const Mat<Scalar> before = lu;
  std::vector<Int> ipiv = {1, 2, 3, 4, 5, 6};
  std::vector<Scalar> work(n);
  Int nn = n, lda = n, lwork = n, info = -99;
  Lapack<Scalar>::getri(&nn, rp(lu.data()), &lda, ipiv.data(), rp(work.data()), &lwork, &info);
  VERIFY_IS_EQUAL(info, 3);
  VERIFY(same_bits(lu, before));
}

// ---------------------------------------------------------------------------------------------------------------
// xSYEV and xHEEV

template <typename Scalar>
constexpr Int ev_workspace_factor() {
  return NumTraits<Scalar>::IsComplex ? 2 : 3;  // minimum LWORK is factor * N - 1
}

// Query, then JOBZ = 'N' or 'V' on a symmetric or Hermitian A, which holds the eigenvectors after 'V'. Returns INFO.
template <typename Scalar>
Int call_ev(char jobz, char uplo, Mat<Scalar>& A, RealOf<Scalar>* w) {
  Int n = Int(A.rows()), lda = n, lwork = -1, info = -99;
  Scalar query(-1);
  ev(&jobz, &uplo, &n, A.data(), &lda, w, &query, &lwork, &info);
  VERIFY_IS_EQUAL(info, 0);
  VERIFY(Int(numext::real(query)) >= (std::max<Int>)(1, ev_workspace_factor<Scalar>() * n - 1));
  lwork = Int(numext::real(query));
  std::vector<Scalar> work(lwork);
  ev(&jobz, &uplo, &n, A.data(), &lda, w, work.data(), &lwork, &info);
  return info;
}

template <typename Scalar>
void test_ev() {
  using Real = RealOf<Scalar>;
  const Real eps = NumTraits<Real>::epsilon();
  for (char uplo : {'L', 'U'}) {
    for (Int n : {2, 5, 40}) {
      const Mat<Scalar> A0 = random_hermitian<Scalar>(n);
      Vec<Real> w_values(n);
      for (char jobz : {'N', 'V'}) {
        Mat<Scalar> A = A0;
        poison_opposite(A, uplo);
        Vec<Real> w(n);
        VERIFY_IS_EQUAL(call_ev(jobz, uplo, A, w.data()), 0);
        for (Int i = 1; i < n; ++i) VERIFY(w(i - 1) <= w(i));
        if (jobz == 'N') {
          w_values = w;
        } else {
          // A V = V W and V^H V = I, to c n eps ||A||.
          const Real residual = (A0 * A - A * w.template cast<Scalar>().asDiagonal()).norm();
          VERIFY(residual <= 8 * n * eps * A0.norm());
          const Real orthogonality = (A.adjoint() * A - Mat<Scalar>::Identity(n, n)).norm();
          VERIFY(orthogonality <= 16 * n * eps);
          const Real dw = (w - w_values).norm();
          VERIFY(dw <= 8 * n * eps * A0.norm());
        }
      }
    }
  }
}

// xHEEV scales by max|A(i,j)| over the whole matrix, so a diagonal imaginary part of NaN, Inf or huge size must be
// dropped first, as xHETRD does.
template <typename Scalar>
void test_heev_imaginary_diagonal() {
  using Real = RealOf<Scalar>;
  if (!NumTraits<Scalar>::IsComplex) return;
  const Real eps = NumTraits<Real>::epsilon();
  for (const Real imag : {std::numeric_limits<Real>::quiet_NaN(), std::numeric_limits<Real>::infinity(),
                          (std::numeric_limits<Real>::max)()}) {
    for (char uplo : {'L', 'U'}) {
      for (char jobz : {'N', 'V'}) {
        Mat<Scalar> M(2, 2);
        M << Scalar(1), Scalar(Real(0.5)), Scalar(Real(0.5)), Scalar(1);
        Mat<Scalar> A = M;
        set_imag(A(0, 0), imag);
        poison_opposite(A, uplo);
        Vec<Real> w(2);
        VERIFY_IS_EQUAL(call_ev(jobz, uplo, A, w.data()), 0);
        // Eigenvalues 1/2 and 3/2 of [1 1/2; 1/2 1], each to c eps ||M||_2.
        const Real error = (w - (Vec<Real>(2) << Real(0.5), Real(1.5)).finished()).norm();
        VERIFY(error <= 8 * eps);
        if (jobz == 'V') {
          const Real residual = (M * A - A * w.template cast<Scalar>().asDiagonal()).norm();
          VERIFY(residual <= 8 * eps);
        }
      }
    }
  }
}

// N = 1: W(1) = real(A(1,1)) with INFO = 0, for any value, and A(1,1) = 1 for JOBZ = 'V'.
template <typename Scalar>
void test_ev_order_one() {
  using Real = RealOf<Scalar>;
  const Real nan = std::numeric_limits<Real>::quiet_NaN(), inf = std::numeric_limits<Real>::infinity();
  for (const Real re : {nan, inf, -inf, Real(3)}) {
    for (const Real im : {Real(0), nan, inf, Real(2)}) {
      if (!NumTraits<Scalar>::IsComplex && im != Real(0)) continue;
      for (char uplo : {'L', 'U'}) {
        for (char jobz : {'N', 'V'}) {
          Mat<Scalar> A(1, 1);
          A(0, 0) = Scalar(re);
          set_imag(A(0, 0), im);
          Real w = Real(-5);
          VERIFY_IS_EQUAL(call_ev(jobz, uplo, A, &w), 0);
          VERIFY((numext::isnan)(re) ? (numext::isnan)(w) : w == re);
          if (jobz == 'V') VERIFY(A(0, 0) == Scalar(1));
        }
      }
    }
  }
}

// WORK(1) is written by a call that solves and by one that fails on LWORK, which XERBLA reports as argument 8.
template <typename Scalar>
void test_ev_work_one() {
  using Real = RealOf<Scalar>;
  const Int n = 5, min_lwork = ev_workspace_factor<Scalar>() * n - 1;
  const std::string name = xerbla_name<Scalar>("SYEV", "HEEV");
  Vec<Scalar> work = Vec<Scalar>::Constant(2 * min_lwork, Scalar(-7));
  Vec<Real> w(n);
  char jobz = 'V', uplo = 'L';
  for (Int lwork : {min_lwork, min_lwork - 1}) {
    Mat<Scalar> A = random_hermitian<Scalar>(n);
    Int nn = n, lda = n, info = -99;
    work(0) = Scalar(-7);
    reset_xerbla();
    ev(&jobz, &uplo, &nn, A.data(), &lda, w.data(), work.data(), &lwork, &info);
    VERIFY_IS_EQUAL(info, lwork == min_lwork ? 0 : -8);
    VERIFY(numext::real(work(0)) >= min_lwork);
    if (info != 0) verify_xerbla(name, 8);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// xGEQRF, xORGQR/xUNGQR, xORMQR/xUNMQR

// Q = H_0 ... H_{k-1}, H_i = I - tau_i v_i v_i^H with v_i = (0, ..., 0, 1, V(i+1:, i)), formed one reflector at a time.
template <typename Scalar>
Mat<Scalar> explicit_q(const Mat<Scalar>& V, const Vec<Scalar>& tau, Int k) {
  const Int nq = Int(V.rows());
  Mat<Scalar> Q = Mat<Scalar>::Identity(nq, nq);
  for (Int i = 0; i < k; ++i) {
    Vec<Scalar> v = Vec<Scalar>::Zero(nq);
    v(i) = Scalar(1);
    v.tail(nq - i - 1) = V.col(i).tail(nq - i - 1);
    Q -= tau(i) * (Q * v) * v.adjoint();
  }
  return Q;
}

// Reflectors of V(:, 0:k) as input to ORGQR/ORMQR: the entries those routines must not read are NaN.
template <typename Scalar>
Mat<Scalar> reflector_input(const Mat<Scalar>& factored, Int ld, Int cols, Int k) {
  Mat<Scalar> V = Mat<Scalar>::Constant(ld, cols, make_nan<Scalar>());
  V.topLeftCorner(factored.rows(), k) = factored.leftCols(k);
  for (Int j = 0; j < k; ++j) V.col(j).head((std::min<Int>)(j + 1, Int(V.rows()))).setConstant(make_nan<Scalar>());
  return V;
}

template <typename Scalar>
void geqrf(Mat<Scalar>& A, Vec<Scalar>& tau, Int lwork) {
  Int m = Int(A.rows()), n = Int(A.cols()), lda = m, info = -99;
  std::vector<Scalar> work((std::max<Int>)(lwork, 1));
  tau.resize((std::min)(m, n));
  Lapack<Scalar>::geqrf(&m, &n, rp(A.data()), &lda, rp(tau.data()), rp(work.data()), &lwork, &info);
  VERIFY_IS_EQUAL(info, 0);
  VERIFY(numext::real(work[0]) >= 1);
}

template <typename Scalar>
void test_qr_factor(Int m, Int n) {
  using Real = RealOf<Scalar>;
  const Real eps = NumTraits<Real>::epsilon();
  const Int k = (std::min)(m, n);
  const Mat<Scalar> A0 = Mat<Scalar>::Random(m, n);

  // The minimal LWORK = N, after a query that reports at least that.
  Mat<Scalar> A = A0;
  {
    Int lwork = -1, info = -99, lda = m, mm = m, nn = n;
    Scalar query(-1);
    Lapack<Scalar>::geqrf(&mm, &nn, rp(A.data()), &lda, rp(&query), rp(&query), &lwork, &info);
    VERIFY_IS_EQUAL(info, 0);
    VERIFY(numext::real(query) >= n);
    VERIFY(same_bits(A, A0));
  }
  Vec<Scalar> tau;
  geqrf(A, tau, n);

  Mat<Scalar> R = Mat<Scalar>::Zero(m, n);
  R.template triangularView<Upper>() = A;
  const Mat<Scalar> Qe = explicit_q(A, tau, k);

  // Q from ORGQR, full (m x m) and thin (m x k), against the product of the reflectors.
  for (Int cols : {m, k}) {
    Mat<Scalar> Q = reflector_input(A, m, cols, k);
    Int mm = m, nn = cols, kk = k, lda = m, lwork = cols, info = -99;
    std::vector<Scalar> work((std::max<Int>)(cols, 1));
    Lapack<Scalar>::qrgen(&mm, &nn, &kk, rp(Q.data()), &lda, rp(tau.data()), rp(work.data()), &lwork, &info);
    VERIFY_IS_EQUAL(info, 0);
    const Real orthogonality = (Q.adjoint() * Q - Mat<Scalar>::Identity(cols, cols)).norm();
    VERIFY(orthogonality <= 16 * m * eps);
    const Real convention = (Q - Qe.leftCols(cols)).norm();
    VERIFY(convention <= 8 * m * eps * Qe.norm());
    if (cols == m) {
      const Real reconstruction = (A0 - Q * R).norm();
      VERIFY(reconstruction <= 16 * m * eps * A0.norm());
    }
  }
}

// K < N reflectors generate Q = H_0 ... H_{K-1} applied to the first N columns of I: for K = 0 exactly those columns.
template <typename Scalar>
void test_orgqr_fewer_reflectors(Int m, Int n) {
  using Real = RealOf<Scalar>;
  const Real eps = NumTraits<Real>::epsilon();
  Mat<Scalar> A = Mat<Scalar>::Random(m, n);
  Vec<Scalar> tau;
  geqrf(A, tau, n);
  for (Int k : {Int(0), Int(1), (std::min)(m, n) / 2}) {
    Mat<Scalar> Q = reflector_input(A, m, n, k);
    Int mm = m, nn = n, kk = k, lda = m, lwork = n, info = -99;
    std::vector<Scalar> work(n);
    Lapack<Scalar>::qrgen(&mm, &nn, &kk, rp(Q.data()), &lda, rp(tau.data()), rp(work.data()), &lwork, &info);
    VERIFY_IS_EQUAL(info, 0);
    if (k == 0) {
      VERIFY(same_bits(Q, Mat<Scalar>(Mat<Scalar>::Identity(m, n))));
    } else {
      const Real err = (Q - explicit_q(A, tau, k).leftCols(n)).norm();
      VERIFY(err <= 8 * m * eps * Real(std::sqrt(double(m))));
    }
  }
}

// op(Q) C and C op(Q) for every SIDE and TRANS against the explicit Q, for all, some and no reflectors, with the
// leading dimensions larger than the matrices and NaN in the padding.
template <typename Scalar>
void test_ormqr(Int nq) {
  using Real = RealOf<Scalar>;
  const Real eps = NumTraits<Real>::epsilon();
  const char adjoint = NumTraits<Scalar>::IsComplex ? 'C' : 'T';
  for (Int cols_of_a : {nq, nq / 2 + 1}) {
    Mat<Scalar> A = Mat<Scalar>::Random(nq, cols_of_a);
    Vec<Scalar> tau;
    geqrf(A, tau, cols_of_a);
    const Int k_max = (std::min)(nq, cols_of_a);
    for (Int k : {k_max, (k_max + 1) / 2, Int(0)}) {
      const Mat<Scalar> Qe = explicit_q(A, tau, k);
      const Mat<Scalar> V = reflector_input(A, nq + 3, (std::max<Int>)(k, 1), k);
      for (char side : {'L', 'R'}) {
        for (char trans : {'N', adjoint}) {
          const Int rows = side == 'L' ? nq : 7, cols = side == 'L' ? 7 : nq;
          const Mat<Scalar> C0 = Mat<Scalar>::Random(rows, cols);
          Mat<Scalar> C = Mat<Scalar>::Constant(rows + 2, cols, make_nan<Scalar>());
          C.topRows(rows) = C0;
          const Mat<Scalar> C_before = C;
          const Mat<Scalar> op = trans == 'N' ? Qe : Mat<Scalar>(Qe.adjoint());
          const Mat<Scalar> expected = side == 'L' ? Mat<Scalar>(op * C0) : Mat<Scalar>(C0 * op);

          Int mm = rows, nn = cols, kk = k, lda = nq + 3, ldc = rows + 2, lwork = side == 'L' ? cols : rows, info = -99;
          std::vector<Scalar> work(lwork);
          Lapack<Scalar>::qrmul(&side, &trans, &mm, &nn, &kk, rp(const_cast<Scalar*>(V.data())), &lda, rp(tau.data()),
                                rp(C.data()), &ldc, rp(work.data()), &lwork, &info);
          VERIFY_IS_EQUAL(info, 0);
          const Real err = (C.topRows(rows) - expected).norm();
          VERIFY(err <= 8 * nq * eps * C0.norm());
          VERIFY(same_bits(Mat<Scalar>(C.bottomRows(2)), Mat<Scalar>(C_before.bottomRows(2))));
        }
      }
    }
  }
}

// TRANS = 'T' for complex and 'C' for real are not in the contract of xUNMQR and xORMQR: INFO = -2 through XERBLA,
// with C untouched.
template <typename Scalar>
void test_ormqr_wrong_trans() {
  const char wrong = NumTraits<Scalar>::IsComplex ? 'T' : 'C';
  Mat<Scalar> A = Mat<Scalar>::Random(5, 3);
  Vec<Scalar> tau;
  geqrf(A, tau, 3);
  for (char side : {'L', 'R'}) {
    Mat<Scalar> C = Mat<Scalar>::Random(5, 5);
    const Mat<Scalar> before = C;
    Int m = 5, n = 5, k = 3, lda = 5, ldc = 5, lwork = 5, info = 0;
    std::vector<Scalar> work(5);
    reset_xerbla();
    Lapack<Scalar>::qrmul(&side, &wrong, &m, &n, &k, rp(A.data()), &lda, rp(tau.data()), rp(C.data()), &ldc,
                          rp(work.data()), &lwork, &info);
    VERIFY_IS_EQUAL(info, -2);
    verify_xerbla(xerbla_name<Scalar>("ORMQR", "UNMQR"), 2);
    VERIFY(same_bits(C, before));
  }
}

template <typename Scalar>
void test_qr_all() {
  const struct {
    Int m, n;
  } shapes[] = {{1, 1}, {6, 4}, {4, 6}, {5, 5}, {1, 5}, {5, 1}, {150, 100}, {100, 150}, {130, 130}};
  for (const auto& s : shapes) {
    test_qr_factor<Scalar>(s.m, s.n);
    test_orgqr_fewer_reflectors<Scalar>(s.m, (std::min)(s.m, s.n));
  }
  for (Int nq : {5, 40, 150}) test_ormqr<Scalar>(nq);
  test_ormqr_wrong_trans<Scalar>();
}

// ---------------------------------------------------------------------------------------------------------------
// Single-precision workspace queries

// WORK(1) of a query is the LWORK to pass, and a float holds integers above 2^24 only if even: it must round up,
// or an INT(WORK(1)) allocation falls short of the routine's own minimum.
template <typename Real>
long long as_integer(Real work) {
  return static_cast<long long>(work);
}

void test_workspace_query_rounding() {
  const Int big = (1 << 24) + 1;
  Int lwork = -1, info = -99;
  float a = 0, tau = 0, w = 0, dummy_w = 0;
  float work2[2] = {0, 0};
  Int ipiv = 0;

  {  // xGEQRF with M = 1: minimum LWORK = N.
    Int m = 1, n = big, lda = 1;
    BLASFUNC(sgeqrf)(&m, &n, &a, &lda, &tau, &w, &lwork, &info);
    VERIFY_IS_EQUAL(info, 0);
    VERIFY(as_integer(w) >= n);
    work2[0] = 0;
    BLASFUNC(cgeqrf)(&m, &n, &a, &lda, &tau, work2, &lwork, &info);
    VERIFY_IS_EQUAL(info, 0);
    VERIFY(as_integer(work2[0]) >= n);
  }
  {  // xORGQR and xUNGQR: LWORK >= N; xORMQR and xUNMQR: LWORK >= N for SIDE = 'L'.
    Int m = big, n = big, k = 0, lda = big;
    w = 0;
    BLASFUNC(sorgqr)(&m, &n, &k, &a, &lda, &tau, &w, &lwork, &info);
    VERIFY_IS_EQUAL(info, 0);
    VERIFY(as_integer(w) >= n);
    work2[0] = 0;
    BLASFUNC(cungqr)(&m, &n, &k, &a, &lda, &tau, work2, &lwork, &info);
    VERIFY_IS_EQUAL(info, 0);
    VERIFY(as_integer(work2[0]) >= n);

    Int mc = 1, nc = big, kc = 1, lda_q = 1, ldc = 1;
    w = 0;
    BLASFUNC(sormqr)("L", "N", &mc, &nc, &kc, &a, &lda_q, &tau, &dummy_w, &ldc, &w, &lwork, &info);
    VERIFY_IS_EQUAL(info, 0);
    VERIFY(as_integer(w) >= nc);
    work2[0] = 0;
    BLASFUNC(cunmqr)("L", "N", &mc, &nc, &kc, &a, &lda_q, &tau, &dummy_w, &ldc, work2, &lwork, &info);
    VERIFY_IS_EQUAL(info, 0);
    VERIFY(as_integer(work2[0]) >= nc);
  }
  {  // xSYEV: LWORK >= 3 N - 1 = 2^24 + 1. xHEEV: LWORK >= 2 N - 1 = 2^24 + 1.
    Int n = (big + 1 + 2) / 3, lda = n;
    VERIFY_IS_EQUAL(3LL * n - 1, big);
    float wev = 0;
    w = 0;
    BLASFUNC(ssyev)("N", "L", &n, &a, &lda, &wev, &w, &lwork, &info);
    VERIFY_IS_EQUAL(info, 0);
    VERIFY(as_integer(w) >= 3LL * n - 1);

    n = (1 << 23) + 1;
    lda = n;
    VERIFY_IS_EQUAL(2LL * n - 1, big);
    float rwork = 0;
    work2[0] = 0;
    BLASFUNC(cheev)("N", "L", &n, &a, &lda, &wev, work2, &lwork, &rwork, &info);
    VERIFY_IS_EQUAL(info, 0);
    VERIFY(as_integer(work2[0]) >= 2LL * n - 1);
  }
  {  // xGETRI: 64 N, the block size times N.
    Int n = 300000, lda = n;
    w = 0;
    BLASFUNC(sgetri)(&n, &a, &lda, &ipiv, &w, &lwork, &info);
    VERIFY_IS_EQUAL(info, 0);
    VERIFY(as_integer(w) >= 64LL * n);
    work2[0] = 0;
    BLASFUNC(cgetri)(&n, &a, &lda, &ipiv, work2, &lwork, &info);
    VERIFY_IS_EQUAL(info, 0);
    VERIFY(as_integer(work2[0]) >= 64LL * n);
  }
}

// In the ILP64 build, arguments whose upper 32 bits matter: N = 1 - 2^32 is negative but has low word 1, and
// LDA = 2^32 is valid for N = 1 but has low word 0.
void test_ilp64_arguments() {
  if (sizeof(Int) < 8) return;
  const Int lda_high = Int(1) << (4 * sizeof(Int)), negative_low_one = 1 - lda_high;  // 2^32 when Int is 64-bit
  double a = 4, work = 0, tau = 0;
  Int n = negative_low_one, lda = 1, lwork = 1, info = -99, ipiv = 1;
  reset_xerbla();
  BLASFUNC(dgetri)(&n, &a, &lda, &ipiv, &work, &lwork, &info);
  VERIFY_IS_EQUAL(info, -1);
  verify_xerbla("DGETRI", 1);
  BLASFUNC(dsytrf)("L", &n, &a, &lda, &ipiv, &work, &lwork, &info);
  VERIFY_IS_EQUAL(info, -2);
  verify_xerbla("DSYTRF", 2);
  n = 1;
  lda = lda_high;
  BLASFUNC(dpotrf)("L", &n, &a, &lda, &info);
  VERIFY_IS_EQUAL(info, 0);
  VERIFY_IS_EQUAL(a, 2.0);
  Int rank = -1;
  double tol = -1, pwork[2] = {0, 0};
  a = 9;
  BLASFUNC(dpstrf)("L", &n, &a, &lda, &ipiv, &rank, &tol, pwork, &info);
  VERIFY_IS_EQUAL(info, 0);
  VERIFY_IS_EQUAL(rank, 1);
  VERIFY_IS_EQUAL(ipiv, 1);
  VERIFY_IS_EQUAL(a, 3.0);
  // A workspace query for N > 2^32.
  Int m = 1, big = lda_high + 5;
  lda = 1;
  lwork = -1;
  work = 0;
  BLASFUNC(dgeqrf)(&m, &big, &a, &lda, &tau, &work, &lwork, &info);
  VERIFY_IS_EQUAL(info, 0);
  VERIFY(work >= double(big));
}

void test_no_stray_xerbla() { VERIFY_IS_EQUAL(g_xerbla.calls, 0); }

}  // namespace

EIGEN_DECLARE_TEST(lapack_wrappers) {
// FUNC names a function template, so it cannot be parenthesized.
// NOLINTBEGIN(bugprone-macro-parentheses)
#define LW_FOR_ALL_SCALARS(FUNC)         \
  CALL_SUBTEST(FUNC<float>());           \
  CALL_SUBTEST(FUNC<double>());          \
  CALL_SUBTEST(FUNC<complex<float> >()); \
  CALL_SUBTEST(FUNC<complex<double> >())
  // NOLINTEND(bugprone-macro-parentheses)
  for (Int i = 0; i < g_repeat; ++i) {
    LW_FOR_ALL_SCALARS(test_pstrf);
    LW_FOR_ALL_SCALARS(test_bunch_kaufman);
    LW_FOR_ALL_SCALARS(test_hetrf_ignores_imaginary_diagonal);
    LW_FOR_ALL_SCALARS(test_bunch_kaufman_reference_pivots);
    LW_FOR_ALL_SCALARS(test_bunch_kaufman_scaled_2x2);
    LW_FOR_ALL_SCALARS(test_bunch_kaufman_singular);
    LW_FOR_ALL_SCALARS(test_trtri);
    LW_FOR_ALL_SCALARS(test_triangular_singular);
    LW_FOR_ALL_SCALARS(test_trtrs);
    LW_FOR_ALL_SCALARS(test_lauum);
    LW_FOR_ALL_SCALARS(test_potri);
    LW_FOR_ALL_SCALARS(test_getri);
    LW_FOR_ALL_SCALARS(test_ev);
    LW_FOR_ALL_SCALARS(test_heev_imaginary_diagonal);
    LW_FOR_ALL_SCALARS(test_ev_order_one);
    LW_FOR_ALL_SCALARS(test_ev_work_one);
    LW_FOR_ALL_SCALARS(test_qr_all);
  }
#undef LW_FOR_ALL_SCALARS
  CALL_SUBTEST(test_workspace_query_rounding());
  CALL_SUBTEST(test_ilp64_arguments());
  CALL_SUBTEST(test_no_stray_xerbla());
}
