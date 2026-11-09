// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Compares every routine of Eigen's BLAS with netlib's reference BLAS, which is built without the trailing
// underscore and declared by including blas.h again with an empty suffix. Inputs hold NaN wherever BLAS promises not
// to read: padding, the other triangle, a unit diagonal, the imaginary part of a Hermitian diagonal, operands scaled by
// alpha == 0, and outputs scaled by beta == 0.

#include <map>
#include <sstream>

#include "main.h"
#include "../blas.h"

#undef BLAS_H
#undef BLAS_FUNC_SUFFIX
#define BLAS_FUNC_SUFFIX
#include "../blas.h"

namespace {

using BlasInt = EIGEN_BLAS_INT;

struct complex_float_result {
  float re, im;
};
struct complex_double_result {
  double re, im;
};

}  // namespace

extern "C" {
complex_float_result cdotc_(BlasInt*, float*, BlasInt*, float*, BlasInt*);
complex_float_result cdotu_(BlasInt*, float*, BlasInt*, float*, BlasInt*);
complex_double_result zdotc_(BlasInt*, double*, BlasInt*, double*, BlasInt*);
complex_double_result zdotu_(BlasInt*, double*, BlasInt*, double*, BlasInt*);
complex_float_result cdotc(BlasInt*, float*, BlasInt*, float*, BlasInt*);
complex_float_result cdotu(BlasInt*, float*, BlasInt*, float*, BlasInt*);
complex_double_result zdotc(BlasInt*, double*, BlasInt*, double*, BlasInt*);
complex_double_result zdotu(BlasInt*, double*, BlasInt*, double*, BlasInt*);
}

namespace {

template <typename T>
using Real = typename NumTraits<T>::Real;

template <typename F>
struct Pair {
  F eig, ref;
};
template <typename F>
Pair<F> pair(F eig, F ref) {
  return {eig, ref};
}

template <typename T>
Real<T>* rp(T* p) {
  return reinterpret_cast<Real<T>*>(p);
}
template <typename T>
Real<T>* rp(std::vector<T>& v) {
  return rp(v.data());
}

template <typename T>
T nan_value() {
  return T(std::numeric_limits<Real<T>>::quiet_NaN());
}

template <typename T>
std::enable_if_t<!NumTraits<T>::IsComplex, T> random_value() {
  return internal::random<T>(T(-1), T(1));
}
template <typename T>
std::enable_if_t<NumTraits<T>::IsComplex, T> random_value() {
  return T(random_value<Real<T>>(), random_value<Real<T>>());
}

// A scalar of type S (T or Real<T>) from a T.
template <typename S, typename T>
std::enable_if_t<NumTraits<S>::IsComplex, S> convert(const T& a) {
  return S(a);
}
template <typename S, typename T>
std::enable_if_t<!NumTraits<S>::IsComplex, S> convert(const T& a) {
  return S(numext::real(a));
}

template <typename T>
std::vector<T> random_vector(Index size) {
  std::vector<T> v(size);
  for (T& x : v) x = random_value<T>();
  return v;
}

// The scalars alpha and beta range over: zero and one select BLAS quick paths, the third does not.
template <typename T>
std::vector<T> scalar_values() {
  return {T(0), T(1), T(-0.7)};
}
template <>
std::vector<std::complex<float>> scalar_values() {
  return {0.f, 1.f, {0.6f, -0.8f}};
}
template <>
std::vector<std::complex<double>> scalar_values() {
  return {0., 1., {0.6, -0.8}};
}

template <typename R>
R with_nan_imag(const R& x) {
  return x;
}
template <typename R>
std::complex<R> with_nan_imag(const std::complex<R>& x) {
  return {x.real(), std::numeric_limits<R>::quiet_NaN()};
}

template <typename T>
bool is_nan(const T& x) {
  return (numext::isnan)(numext::real(x)) || (numext::isnan)(numext::imag(x));
}

// Same NaN pattern and |e - r| <= tol * max(|r|, floor), tol = 64 eps (depth + 2). A floor of zero makes the test
// purely relative, which the scaling tests of the level-1 routines need.
template <typename T>
bool same_value(const T& e, const T& r, Index depth, Real<T> floor = Real<T>(1)) {
  if (is_nan(r) || is_nan(e)) return is_nan(r) && is_nan(e);
  if (!(numext::isfinite)(numext::abs(r)) || !(numext::isfinite)(numext::abs(e))) return e == r;
  const Real<T> tol = Real<T>(64) * NumTraits<Real<T>>::epsilon() * Real<T>(depth + 2);
  return numext::abs(e - r) <= tol * numext::maxi(numext::abs(r), floor);
}

std::string describe(std::initializer_list<std::pair<const char*, std::string>> args) {
  std::ostringstream os;
  for (const auto& a : args) os << ' ' << a.first << '=' << a.second;
  return os.str();
}
template <typename T>
std::string str(const T& x) {
  std::ostringstream os;
  os << x;
  return os.str();
}
std::string str(char c) { return std::string(1, c); }

#define BLAS_ARGS(...) describe({__VA_ARGS__})
#define BLAS_ARG(x) \
  { #x, str(x) }

// Mismatches per routine; the first few of each are printed, and the test fails at the end if any occurred.
std::map<std::string, int>& mismatches() {
  static std::map<std::string, int> counts;
  return counts;
}

template <typename T>
void verify_same(const std::vector<T>& e, const std::vector<T>& r, Index depth, const char* routine,
                 const std::string& args, Real<T> floor = Real<T>(1)) {
  VERIFY_IS_EQUAL(e.size(), r.size());
  for (size_t i = 0; i < r.size(); ++i) {
    if (!same_value(e[i], r[i], depth, floor)) {
      if (mismatches()[routine]++ < 3)
        std::cerr << routine << args << ": element " << i << " is " << e[i] << ", reference " << r[i] << "\n";
      return;
    }
  }
}
template <typename T>
void verify_same(const T& e, const T& r, Index depth, const char* routine, const std::string& args,
                 Real<T> floor = Real<T>(1)) {
  verify_same(std::vector<T>{e}, std::vector<T>{r}, depth, routine, args, floor);
}

// Storage for a BLAS vector of n elements with increment inc; untouched elements hold NaN.
template <typename T>
std::vector<T> strided_vector(Index n, Index inc) {
  const Index len = n == 0 ? 1 : 1 + (n - 1) * std::abs(inc);
  std::vector<T> v(len, nan_value<T>());
  for (Index i = 0; i < n; ++i) v[i * std::abs(inc)] = random_value<T>();
  return v;
}

// An output vector of n elements with increment inc; all NaN when beta == 0, where BLAS must not read it.
template <typename T>
std::vector<T> output_vector(Index n, Index inc, const T& beta) {
  std::vector<T> y = strided_vector<T>(n, inc);
  if (beta == T(0)) std::fill(y.begin(), y.end(), nan_value<T>());
  return y;
}

const Index kSizes1[] = {0, 1, 2, 5, 17};
const Index kIncs1[] = {1, 2, -1, -3, 0};

// ---------------------------------------------------------------------------------------------------------------
// Level 1

template <typename T, typename F>
void check_axpy(Pair<F> f) {
  for (Index n : kSizes1)
    for (Index incx : kIncs1)
      for (Index incy : {1, -2, 3})
        for (T alpha : scalar_values<T>()) {
          std::vector<T> x = strided_vector<T>(n, incx);
          if (alpha == T(0)) std::fill(x.begin(), x.end(), nan_value<T>());
          std::vector<T> ye = strided_vector<T>(n, incy), yr = ye;
          BlasInt bn = n, bx = incx, by = incy;
          f.eig(&bn, rp(&alpha), rp(x), &bx, rp(ye), &by);
          f.ref(&bn, rp(&alpha), rp(x), &bx, rp(yr), &by);
          verify_same(ye, yr, 1, "axpy", BLAS_ARGS(BLAS_ARG(n), BLAS_ARG(incx), BLAS_ARG(incy), BLAS_ARG(alpha)));
        }
}

template <typename T, typename F>
void check_copy_swap(Pair<F> copy, Pair<F> swap) {
  for (Index n : kSizes1)
    for (Index incx : kIncs1)
      for (Index incy : {1, -2, 0}) {
        BlasInt bn = n, bx = incx, by = incy;
        std::vector<T> x = strided_vector<T>(n, incx), ye = strided_vector<T>(n, incy), yr = ye;
        copy.eig(&bn, rp(x), &bx, rp(ye), &by);
        copy.ref(&bn, rp(x), &bx, rp(yr), &by);
        verify_same(ye, yr, 0, "copy", BLAS_ARGS(BLAS_ARG(n), BLAS_ARG(incx), BLAS_ARG(incy)));
        if (incx == 0 || incy == 0) continue;
        std::vector<T> xe = x, xr = x;
        swap.eig(&bn, rp(xe), &bx, rp(ye), &by);
        swap.ref(&bn, rp(xr), &bx, rp(yr), &by);
        verify_same(xe, xr, 0, "swap", BLAS_ARGS(BLAS_ARG(n), BLAS_ARG(incx), BLAS_ARG(incy)));
        verify_same(ye, yr, 0, "swap", BLAS_ARGS(BLAS_ARG(n), BLAS_ARG(incx), BLAS_ARG(incy)));
      }
}

template <typename T, typename S, typename F>
void check_scal(Pair<F> f) {
  for (Index n : kSizes1)
    for (Index incx : kIncs1)
      for (T a : scalar_values<T>()) {
        S alpha = convert<S>(a);
        BlasInt bn = n, bx = incx;
        std::vector<T> xe = strided_vector<T>(n, incx), xr = xe;
        f.eig(&bn, rp(&alpha), rp(xe), &bx);
        f.ref(&bn, rp(&alpha), rp(xr), &bx);
        verify_same(xe, xr, 0, "scal", BLAS_ARGS(BLAS_ARG(n), BLAS_ARG(incx), BLAS_ARG(alpha)));
      }
}

template <typename T, typename R, typename F>
void check_dot(Pair<F> f, const char* routine) {
  for (Index n : kSizes1)
    for (Index incx : kIncs1)
      for (Index incy : {1, -2, 3}) {
        BlasInt bn = n, bx = incx, by = incy;
        std::vector<T> x = strided_vector<T>(n, incx), y = strided_vector<T>(n, incy);
        R e = f.eig(&bn, rp(x), &bx, rp(y), &by);
        R r = f.ref(&bn, rp(x), &bx, rp(y), &by);
        verify_same(T(e.re, e.im), T(r.re, r.im), n, routine, BLAS_ARGS(BLAS_ARG(n), BLAS_ARG(incx), BLAS_ARG(incy)));
      }
}

template <typename T, typename F>
void check_real_dot(Pair<F> f, const char* routine) {
  for (Index n : kSizes1)
    for (Index incx : kIncs1)
      for (Index incy : {1, -2, 3}) {
        BlasInt bn = n, bx = incx, by = incy;
        std::vector<float> x = strided_vector<float>(n, incx), y = strided_vector<float>(n, incy);
        T e = f.eig(&bn, x.data(), &bx, y.data(), &by);
        T r = f.ref(&bn, x.data(), &bx, y.data(), &by);
        verify_same(e, r, n, routine, BLAS_ARGS(BLAS_ARG(n), BLAS_ARG(incx), BLAS_ARG(incy)));
      }
}

void check_sdsdot(Pair<float (*)(BlasInt*, float*, float*, BlasInt*, float*, BlasInt*)> f) {
  for (Index n : kSizes1)
    for (Index incx : kIncs1) {
      BlasInt bn = n, bx = incx, by = -1;
      float sb = 0.25f;
      std::vector<float> x = strided_vector<float>(n, incx), y = strided_vector<float>(n, -1);
      verify_same(f.eig(&bn, &sb, x.data(), &bx, y.data(), &by), f.ref(&bn, &sb, x.data(), &bx, y.data(), &by), n,
                  "sdsdot", BLAS_ARGS(BLAS_ARG(n), BLAS_ARG(incx)));
    }
}

// nrm2 and asum, with entries scaled near overflow and underflow.
template <typename T, typename F>
void check_norms(Pair<F> nrm2, Pair<F> asum) {
  using R = Real<T>;
  const R scales[] = {R(1),
                      R(1e-3) * std::sqrt((std::numeric_limits<R>::max)()),
                      R(1e3) * std::sqrt((std::numeric_limits<R>::min)()),
                      (std::numeric_limits<R>::max)() / R(8),
                      (std::numeric_limits<R>::min)() * R(4),
                      std::numeric_limits<R>::denorm_min() * R(16)};
  for (Index n : kSizes1)
    for (Index incx : kIncs1)
      for (R scale : scales) {
        BlasInt bn = n, bx = incx;
        std::vector<T> x = strided_vector<T>(n, incx);
        for (T& v : x) v *= scale;
        // Denormal inputs lose relative accuracy in the inputs themselves.
        const R floor = scale < (std::numeric_limits<R>::min)() ? scale : R(0);
        verify_same(nrm2.eig(&bn, rp(x), &bx), nrm2.ref(&bn, rp(x), &bx), n, "nrm2",
                    BLAS_ARGS(BLAS_ARG(n), BLAS_ARG(incx), BLAS_ARG(scale)), floor);
        if (scale > R(1) && n > 1) continue;  // asum overflows by design
        verify_same(asum.eig(&bn, rp(x), &bx), asum.ref(&bn, rp(x), &bx), n, "asum",
                    BLAS_ARGS(BLAS_ARG(n), BLAS_ARG(incx), BLAS_ARG(scale)), floor);
      }
}

template <typename T, typename F>
void check_iamax(Pair<F> f) {
  for (Index n : kSizes1)
    for (Index incx : kIncs1)
      for (int variant = 0; variant < 4; ++variant) {
        BlasInt bn = n, bx = incx;
        std::vector<T> x = strided_vector<T>(n, incx);
        // 1: equal |re| + |im| everywhere but every third entry, so the first index wins; 2 and 3: a NaN in the
        // middle or in front, which 3.12.1 skips unless it comes first.
        if (variant == 1)
          for (Index i = 0; i < n; ++i) x[i * std::abs(incx)] = T(i % 3 == 1 ? 0.25 : 0.5) * (i % 2 ? T(-1) : T(1));
        if (variant == 2 && n > 2) x[(n / 2) * std::abs(incx)] = nan_value<T>();
        if (variant == 3 && n > 0) x[0] = nan_value<T>();
        verify_same(Index(f.eig(&bn, rp(x), &bx)), Index(f.ref(&bn, rp(x), &bx)), 0, "iamax",
                    BLAS_ARGS(BLAS_ARG(n), BLAS_ARG(incx), BLAS_ARG(variant)));
      }
}

template <typename T, typename F>
void check_rot(Pair<F> f) {
  using R = Real<T>;
  for (Index n : kSizes1)
    for (Index incx : {1, 2, -1, -3})
      for (Index incy : {1, -2}) {
        const R angle = internal::random<R>(R(-3), R(3));
        R c = std::cos(angle), s = std::sin(angle);
        BlasInt bn = n, bx = incx, by = incy;
        std::vector<T> xe = strided_vector<T>(n, incx), xr = xe, ye = strided_vector<T>(n, incy), yr = ye;
        f.eig(&bn, rp(xe), &bx, rp(ye), &by, &c, &s);
        f.ref(&bn, rp(xr), &bx, rp(yr), &by, &c, &s);
        verify_same(xe, xr, 2, "rot", BLAS_ARGS(BLAS_ARG(n), BLAS_ARG(incx), BLAS_ARG(incy)));
        verify_same(ye, yr, 2, "rot", BLAS_ARGS(BLAS_ARG(n), BLAS_ARG(incx), BLAS_ARG(incy)));
      }
}

template <typename T, typename F>
void check_rotg(Pair<F> f) {
  using R = Real<T>;
  const R big = (std::numeric_limits<R>::max)() / R(4), small = (std::numeric_limits<R>::min)() * R(4);
  std::vector<T> values = {T(0), T(1), T(-2), T(R(0.5)), T(big), T(-big), T(small)};
  if (NumTraits<T>::IsComplex) {
    values.push_back(numext::sqrt(T(-1)) * T(big));
    values.push_back(numext::sqrt(T(-1)) * T(R(-3)));
  }
  for (int i = 0; i < 4; ++i) values.push_back(random_value<T>());
  for (T a : values)
    for (T b : values) {
      T ae = a, be = b, ar = a, br = b, se, sr;
      R ce, cr;
      f.eig(rp(&ae), rp(&be), &ce, rp(&se));
      f.ref(rp(&ar), rp(&br), &cr, rp(&sr));
      const std::string args = BLAS_ARGS(BLAS_ARG(a), BLAS_ARG(b));
      verify_same(ae, ar, 4, "rotg(r)", args, R(0));
      verify_same(ce, cr, 4, "rotg(c)", args, R(0));
      verify_same(se, sr, 4, "rotg(s)", args, R(0));
      if (!NumTraits<T>::IsComplex) verify_same(be, br, 4, "rotg(z)", args, R(0));
    }
}

template <typename T, typename F>
void check_rotm(Pair<F> f) {
  for (Index n : kSizes1)
    for (Index incx : {1, 2, -1, -3})
      for (Index incy : {1, -2})
        for (T flag : {T(-2), T(-1), T(0), T(1)}) {
          T param[5] = {flag, random_value<T>(), random_value<T>(), random_value<T>(), random_value<T>()};
          BlasInt bn = n, bx = incx, by = incy;
          std::vector<T> xe = strided_vector<T>(n, incx), xr = xe, ye = strided_vector<T>(n, incy), yr = ye;
          f.eig(&bn, xe.data(), &bx, ye.data(), &by, param);
          f.ref(&bn, xr.data(), &bx, yr.data(), &by, param);
          const std::string args = BLAS_ARGS(BLAS_ARG(n), BLAS_ARG(incx), BLAS_ARG(incy), BLAS_ARG(flag));
          verify_same(xe, xr, 2, "rotm", args);
          verify_same(ye, yr, 2, "rotm", args);
        }
}

template <typename T, typename F>
void check_rotmg(Pair<F> f) {
  // {d1, d2, x1, y1}: the regular branches, the zeroing branches, and the rescaling loops (gam = 4096). No input
  // rescales both d1 and d2: there 3.12.1 overwrites h12 and h21, fixed after the release (Reference-LAPACK
  // 6dc1a78ea, "Fix DROTMG/SROTMG rescaling overwriting DH12").
  const T cases[][4] = {{2, 3, 1, 0.5},    {0.5, 3, 0.25, 2}, {-1, 2, 1, 1},     {2, 0, 1, 1},   {2, 3, 1, 0},
                        {1, -2, 1, 2},     {1, -2, 4, 0.5},   {1e-9, 2, 1, 0.5}, {1e9, 2, 1, 1}, {2, 1e-9, 1, 1e3},
                        {2, 1e9, 1e-3, 1}, {0, 1, 1, 1},      {1, 1, 0, 1}};
  for (const auto& c : cases) {
    T de1 = c[0], de2 = c[1], xe1 = c[2], dr1 = c[0], dr2 = c[1], xr1 = c[2], y1 = c[3];
    T pe[5] = {7, 7, 7, 7, 7}, pr[5] = {7, 7, 7, 7, 7};
    f.eig(&de1, &de2, &xe1, &y1, pe);
    f.ref(&dr1, &dr2, &xr1, &y1, pr);
    const std::string args = BLAS_ARGS({"d1", str(c[0])}, {"d2", str(c[1])}, {"x1", str(c[2])}, {"y1", str(c[3])});
    verify_same(std::vector<T>{de1, de2, xe1}, std::vector<T>{dr1, dr2, xr1}, 4, "rotmg", args, T(0));
    // param[0] is the flag; only the entries it selects are defined.
    verify_same(pe[0], pr[0], 0, "rotmg(flag)", args);
    const int flag = int(pr[0]);
    for (int i = 1; i <= 4; ++i) {
      const bool used = flag == -1 || (flag == 0 && (i == 2 || i == 3)) || (flag == 1 && (i == 1 || i == 4));
      if (used) verify_same(pe[i], pr[i], 4, "rotmg(param)", args, T(0));
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Level 2 and 3 storage

// An m-by-n column-major matrix with leading dimension m + 2; the padding holds NaN.
template <typename T>
struct Dense {
  Index rows, cols, ld;
  std::vector<T> data;
  Dense(Index m, Index n) : rows(m), cols(n), ld(m + 2), data(std::max<Index>(1, ld * n), nan_value<T>()) {
    for (Index j = 0; j < n; ++j)
      for (Index i = 0; i < m; ++i) at(i, j) = random_value<T>();
  }
  T& at(Index i, Index j) { return data[i + j * ld]; }
  // Keeps the uplo triangle (with a real diagonal when hermitian); the rest becomes NaN.
  void triangle(char uplo, char diag, bool hermitian) {
    for (Index j = 0; j < cols; ++j)
      for (Index i = 0; i < rows; ++i) {
        const bool kept = uplo == 'U' ? i <= j : i >= j;
        if (!kept)
          at(i, j) = nan_value<T>();
        else if (diag == 'U' && i != j)
          at(i, j) /= T(Real<T>(rows));  // well conditioned, for the solvers
      }
    for (Index i = 0; i < (std::min)(rows, cols); ++i) {
      if (diag == 'U')
        at(i, i) = nan_value<T>();
      else if (diag == 'D')
        at(i, i) += T(Real<T>(rows + 2));  // diagonally dominant, for the solvers
      if (hermitian) at(i, i) = with_nan_imag(at(i, i));
    }
  }
  void fill_nan() { std::fill(data.begin(), data.end(), nan_value<T>()); }
};

// The uplo triangle of an n-by-n matrix in packed storage.
template <typename T>
std::vector<T> packed(Dense<T>& a, char uplo) {
  std::vector<T> p;
  for (Index j = 0; j < a.cols; ++j)
    for (Index i = uplo == 'U' ? 0 : j; i <= (uplo == 'U' ? j : a.rows - 1); ++i) p.push_back(a.at(i, j));
  if (p.empty()) p.push_back(nan_value<T>());
  return p;
}

// Band storage of the kl sub- and ku super-diagonals, with leading dimension kl + ku + 2; the rest holds NaN.
template <typename T>
std::vector<T> band(Dense<T>& a, Index kl, Index ku, Index& ld) {
  ld = kl + ku + 2;
  std::vector<T> b(std::max<Index>(1, ld * a.cols), nan_value<T>());
  for (Index j = 0; j < a.cols; ++j)
    for (Index i = (std::max)(Index(0), j - ku); i <= (std::min)(a.rows - 1, j + kl); ++i)
      b[ku + i - j + j * ld] = a.at(i, j);
  return b;
}

const Index kSizes2[] = {0, 1, 3, 8};
const char kTrans[] = {'N', 'T', 'C'};
const char kUplo[] = {'U', 'L'};

// ---------------------------------------------------------------------------------------------------------------
// Level 2

template <typename T, typename Fgemv, typename Fgbmv>
void check_gemv_gbmv(Pair<Fgemv> gemv, Pair<Fgbmv> gbmv) {
  for (Index m : kSizes2)
    for (Index n : kSizes2)
      for (char trans : kTrans)
        for (Index incx : {1, -2})
          for (Index incy : {1, 3})
            for (T alpha : scalar_values<T>())
              for (T beta : scalar_values<T>()) {
                const Index lx = trans == 'N' ? n : m, ly = trans == 'N' ? m : n;
                Dense<T> a(m, n);
                if (alpha == T(0)) a.fill_nan();
                std::vector<T> x = strided_vector<T>(lx, incx), ye = output_vector(ly, incy, beta);
                if (alpha == T(0)) std::fill(x.begin(), x.end(), nan_value<T>());
                std::vector<T> yr = ye;
                BlasInt bm = m, bn = n, ld = a.ld, bx = incx, by = incy;
                const std::string args = BLAS_ARGS(BLAS_ARG(trans), BLAS_ARG(m), BLAS_ARG(n), BLAS_ARG(incx),
                                                   BLAS_ARG(incy), BLAS_ARG(alpha), BLAS_ARG(beta));
                gemv.eig(&trans, &bm, &bn, rp(&alpha), rp(a.data), &ld, rp(x), &bx, rp(&beta), rp(ye), &by);
                gemv.ref(&trans, &bm, &bn, rp(&alpha), rp(a.data), &ld, rp(x), &bx, rp(&beta), rp(yr), &by);
                verify_same(ye, yr, lx, "gemv", args);

                for (Index kl : {0, 2})
                  for (Index ku : {1, 3}) {
                    Index lb;
                    std::vector<T> ab = band(a, kl, ku, lb), ybe = output_vector(ly, incy, beta);
                    std::vector<T> ybr = ybe;
                    BlasInt bkl = kl, bku = ku, bld = lb;
                    gbmv.eig(&trans, &bm, &bn, &bkl, &bku, rp(&alpha), rp(ab), &bld, rp(x), &bx, rp(&beta), rp(ybe),
                             &by);
                    gbmv.ref(&trans, &bm, &bn, &bkl, &bku, rp(&alpha), rp(ab), &bld, rp(x), &bx, rp(&beta), rp(ybr),
                             &by);
                    verify_same(ybe, ybr, kl + ku + 1, "gbmv", args + BLAS_ARGS(BLAS_ARG(kl), BLAS_ARG(ku)));
                  }
              }
}

// symv/hemv, sbmv/hbmv and spmv/hpmv.
template <typename T, typename Fsy, typename Fsb, typename Fsp>
void check_symmetric_mv(Pair<Fsy> symv, Pair<Fsb> sbmv, Pair<Fsp> spmv, bool hermitian) {
  for (Index n : kSizes2)
    for (char uplo : kUplo)
      for (Index incx : {1, -2})
        for (Index incy : {1, -3})
          for (T alpha : scalar_values<T>())
            for (T beta : scalar_values<T>()) {
              Dense<T> a(n, n);
              a.triangle(uplo, 'N', hermitian);
              if (alpha == T(0)) a.fill_nan();
              std::vector<T> x = strided_vector<T>(n, incx);
              if (alpha == T(0)) std::fill(x.begin(), x.end(), nan_value<T>());
              BlasInt bn = n, ld = a.ld, bx = incx, by = incy;
              const std::string args = BLAS_ARGS(BLAS_ARG(uplo), BLAS_ARG(n), BLAS_ARG(incx), BLAS_ARG(incy),
                                                 BLAS_ARG(alpha), BLAS_ARG(beta));
              std::vector<T> ye = output_vector(n, incy, beta), yr = ye;
              symv.eig(&uplo, &bn, rp(&alpha), rp(a.data), &ld, rp(x), &bx, rp(&beta), rp(ye), &by);
              symv.ref(&uplo, &bn, rp(&alpha), rp(a.data), &ld, rp(x), &bx, rp(&beta), rp(yr), &by);
              verify_same(ye, yr, n, hermitian ? "hemv" : "symv", args);

              std::vector<T> ap = packed(a, uplo);
              ye = output_vector(n, incy, beta), yr = ye;
              spmv.eig(&uplo, &bn, rp(&alpha), rp(ap), rp(x), &bx, rp(&beta), rp(ye), &by);
              spmv.ref(&uplo, &bn, rp(&alpha), rp(ap), rp(x), &bx, rp(&beta), rp(yr), &by);
              verify_same(ye, yr, n, hermitian ? "hpmv" : "spmv", args);

              for (Index k : {0, 1, 2, 9}) {
                Index lb;
                std::vector<T> ab = band(a, uplo == 'U' ? 0 : k, uplo == 'U' ? k : 0, lb);
                BlasInt bk = k, bld = lb;
                ye = output_vector(n, incy, beta), yr = ye;
                sbmv.eig(&uplo, &bn, &bk, rp(&alpha), rp(ab), &bld, rp(x), &bx, rp(&beta), rp(ye), &by);
                sbmv.ref(&uplo, &bn, &bk, rp(&alpha), rp(ab), &bld, rp(x), &bx, rp(&beta), rp(yr), &by);
                verify_same(ye, yr, 2 * k + 1, hermitian ? "hbmv" : "sbmv", args + BLAS_ARGS(BLAS_ARG(k)));
              }
            }
}

// trmv/trsv, tbmv/tbsv and tpmv/tpsv.
template <typename T, typename Ftr, typename Ftb, typename Ftp>
void check_triangular_mv(Pair<Ftr> trmv, Pair<Ftr> trsv, Pair<Ftb> tbmv, Pair<Ftb> tbsv, Pair<Ftp> tpmv,
                         Pair<Ftp> tpsv) {
  for (Index n : kSizes2)
    for (char uplo : kUplo)
      for (char trans : kTrans)
        for (char diag : {'N', 'U'})
          for (Index incx : {1, -2}) {
            Dense<T> a(n, n);
            a.triangle(uplo, diag == 'U' ? 'U' : 'D', false);
            BlasInt bn = n, ld = a.ld, bx = incx;
            const std::string args =
                BLAS_ARGS(BLAS_ARG(uplo), BLAS_ARG(trans), BLAS_ARG(diag), BLAS_ARG(n), BLAS_ARG(incx));
            std::vector<T> xe = strided_vector<T>(n, incx), xr = xe;
            trmv.eig(&uplo, &trans, &diag, &bn, rp(a.data), &ld, rp(xe), &bx);
            trmv.ref(&uplo, &trans, &diag, &bn, rp(a.data), &ld, rp(xr), &bx);
            verify_same(xe, xr, n, "trmv", args);
            trsv.eig(&uplo, &trans, &diag, &bn, rp(a.data), &ld, rp(xe), &bx);
            trsv.ref(&uplo, &trans, &diag, &bn, rp(a.data), &ld, rp(xr), &bx);
            verify_same(xe, xr, 2 * n, "trsv", args);

            std::vector<T> ap = packed(a, uplo);
            tpmv.eig(&uplo, &trans, &diag, &bn, rp(ap), rp(xe), &bx);
            tpmv.ref(&uplo, &trans, &diag, &bn, rp(ap), rp(xr), &bx);
            verify_same(xe, xr, n, "tpmv", args);
            tpsv.eig(&uplo, &trans, &diag, &bn, rp(ap), rp(xe), &bx);
            tpsv.ref(&uplo, &trans, &diag, &bn, rp(ap), rp(xr), &bx);
            verify_same(xe, xr, 2 * n, "tpsv", args);

            for (Index k : {0, 1, 2, 9}) {
              Index lb;
              std::vector<T> ab = band(a, uplo == 'U' ? 0 : k, uplo == 'U' ? k : 0, lb);
              BlasInt bk = k, bld = lb;
              const std::string bargs = args + BLAS_ARGS(BLAS_ARG(k));
              tbmv.eig(&uplo, &trans, &diag, &bn, &bk, rp(ab), &bld, rp(xe), &bx);
              tbmv.ref(&uplo, &trans, &diag, &bn, &bk, rp(ab), &bld, rp(xr), &bx);
              verify_same(xe, xr, k + 1, "tbmv", bargs);
              tbsv.eig(&uplo, &trans, &diag, &bn, &bk, rp(ab), &bld, rp(xe), &bx);
              tbsv.ref(&uplo, &trans, &diag, &bn, &bk, rp(ab), &bld, rp(xr), &bx);
              verify_same(xe, xr, 2 * k + 2, "tbsv", bargs);
            }
          }
}

template <typename T, typename F>
void check_ger(Pair<F> f, const char* routine) {
  for (Index m : kSizes2)
    for (Index n : kSizes2)
      for (Index incx : {1, -2})
        for (Index incy : {1, 3})
          for (T alpha : scalar_values<T>()) {
            Dense<T> ae(m, n);
            Dense<T> ar = ae;
            std::vector<T> x = strided_vector<T>(m, incx), y = strided_vector<T>(n, incy);
            if (alpha == T(0)) std::fill(x.begin(), x.end(), nan_value<T>());
            BlasInt bm = m, bn = n, ld = ae.ld, bx = incx, by = incy;
            f.eig(&bm, &bn, rp(&alpha), rp(x), &bx, rp(y), &by, rp(ae.data), &ld);
            f.ref(&bm, &bn, rp(&alpha), rp(x), &bx, rp(y), &by, rp(ar.data), &ld);
            verify_same(ae.data, ar.data, 1, routine,
                        BLAS_ARGS(BLAS_ARG(m), BLAS_ARG(n), BLAS_ARG(incx), BLAS_ARG(incy), BLAS_ARG(alpha)));
          }
}

// syr/her and spr/hpr (real alpha for the hermitian ones), syr2/her2 and spr2/hpr2.
template <typename T, typename S, typename Fr1, typename Fp1, typename Fr2, typename Fp2>
void check_symmetric_updates(Pair<Fr1> syr, Pair<Fp1> spr, Pair<Fr2> syr2, Pair<Fp2> spr2, bool hermitian) {
  for (Index n : kSizes2)
    for (char uplo : kUplo)
      for (Index incx : {1, -2})
        for (Index incy : {1, 3})
          for (T a2 : scalar_values<T>()) {
            S alpha = convert<S>(a2);
            Dense<T> ae(n, n);
            ae.triangle(uplo, 'N', hermitian);
            if (hermitian)  // the update writes a real diagonal; an untouched one keeps its NaN imaginary part
              for (Index i = 0; i < n; ++i) ae.at(i, i) = T(numext::real(ae.at(i, i)));
            std::vector<T> x = strided_vector<T>(n, incx), y = strided_vector<T>(n, incy);
            BlasInt bn = n, ld = ae.ld, bx = incx, by = incy;
            const std::string args = BLAS_ARGS(BLAS_ARG(uplo), BLAS_ARG(n), BLAS_ARG(incx), BLAS_ARG(alpha));
            Dense<T> ar = ae;
            std::vector<T> pe = packed(ae, uplo), pr = pe;
            syr.eig(&uplo, &bn, rp(&alpha), rp(x), &bx, rp(ae.data), &ld);
            syr.ref(&uplo, &bn, rp(&alpha), rp(x), &bx, rp(ar.data), &ld);
            verify_same(ae.data, ar.data, 1, hermitian ? "her" : "syr", args);
            spr.eig(&uplo, &bn, rp(&alpha), rp(x), &bx, rp(pe));
            spr.ref(&uplo, &bn, rp(&alpha), rp(x), &bx, rp(pr));
            verify_same(pe, pr, 1, hermitian ? "hpr" : "spr", args);

            T alpha2 = a2;
            syr2.eig(&uplo, &bn, rp(&alpha2), rp(x), &bx, rp(y), &by, rp(ae.data), &ld);
            syr2.ref(&uplo, &bn, rp(&alpha2), rp(x), &bx, rp(y), &by, rp(ar.data), &ld);
            verify_same(ae.data, ar.data, 2, hermitian ? "her2" : "syr2", args + BLAS_ARGS(BLAS_ARG(incy)));
            spr2.eig(&uplo, &bn, rp(&alpha2), rp(x), &bx, rp(y), &by, rp(pe));
            spr2.ref(&uplo, &bn, rp(&alpha2), rp(x), &bx, rp(y), &by, rp(pr));
            verify_same(pe, pr, 2, hermitian ? "hpr2" : "spr2", args + BLAS_ARGS(BLAS_ARG(incy)));
          }
}

// ---------------------------------------------------------------------------------------------------------------
// Level 3

const Index kSizes3[] = {0, 1, 4, 9, 33};

// An m-by-n output matrix; all NaN when beta == 0, where BLAS must not read it.
template <typename T>
Dense<T> output_matrix(Index m, Index n, const T& beta) {
  Dense<T> c(m, n);
  if (beta == T(0)) c.fill_nan();
  return c;
}

template <typename T, typename F>
void check_gemm(Pair<F> f) {
  for (Index m : kSizes3)
    for (Index n : kSizes3)
      for (Index k : {0, 1, 9})
        for (char ta : kTrans)
          for (char tb : kTrans)
            for (T alpha : scalar_values<T>())
              for (T beta : scalar_values<T>()) {
                if (ta == 'C' && tb == 'C' && m > 9) continue;
                Dense<T> a(ta == 'N' ? m : k, ta == 'N' ? k : m), b(tb == 'N' ? k : n, tb == 'N' ? n : k);
                if (alpha == T(0)) a.fill_nan(), b.fill_nan();
                Dense<T> ce = output_matrix(m, n, beta);
                Dense<T> cr = ce;
                BlasInt bm = m, bn = n, bk = k, lda = a.ld, ldb = b.ld, ldc = ce.ld;
                f.eig(&ta, &tb, &bm, &bn, &bk, rp(&alpha), rp(a.data), &lda, rp(b.data), &ldb, rp(&beta), rp(ce.data),
                      &ldc);
                f.ref(&ta, &tb, &bm, &bn, &bk, rp(&alpha), rp(a.data), &lda, rp(b.data), &ldb, rp(&beta), rp(cr.data),
                      &ldc);
                verify_same(ce.data, cr.data, k, "gemm",
                            BLAS_ARGS(BLAS_ARG(ta), BLAS_ARG(tb), BLAS_ARG(m), BLAS_ARG(n), BLAS_ARG(k),
                                      BLAS_ARG(alpha), BLAS_ARG(beta)));
              }
}

template <typename T, typename F>
void check_gemmtr(Pair<F> f) {
  for (Index n : kSizes3)
    for (Index k : {0, 1, 9})
      for (char uplo : kUplo)
        for (char ta : kTrans)
          for (char tb : kTrans)
            for (T alpha : scalar_values<T>())
              for (T beta : scalar_values<T>()) {
                Dense<T> a(ta == 'N' ? n : k, ta == 'N' ? k : n), b(tb == 'N' ? k : n, tb == 'N' ? n : k);
                if (alpha == T(0)) a.fill_nan(), b.fill_nan();
                Dense<T> ce = output_matrix(n, n, beta);
                Dense<T> cr = ce;
                BlasInt bn = n, bk = k, lda = a.ld, ldb = b.ld, ldc = ce.ld;
                f.eig(&uplo, &ta, &tb, &bn, &bk, rp(&alpha), rp(a.data), &lda, rp(b.data), &ldb, rp(&beta), rp(ce.data),
                      &ldc);
                f.ref(&uplo, &ta, &tb, &bn, &bk, rp(&alpha), rp(a.data), &lda, rp(b.data), &ldb, rp(&beta), rp(cr.data),
                      &ldc);
                verify_same(ce.data, cr.data, k, "gemmtr",
                            BLAS_ARGS(BLAS_ARG(uplo), BLAS_ARG(ta), BLAS_ARG(tb), BLAS_ARG(n), BLAS_ARG(k),
                                      BLAS_ARG(alpha), BLAS_ARG(beta)));
              }
}

// symm, or hemm when hermitian.
template <typename T, typename F>
void check_symm(Pair<F> f, bool hermitian) {
  for (Index m : kSizes3)
    for (Index n : kSizes3)
      for (char side : {'L', 'R'})
        for (char uplo : kUplo)
          for (T alpha : scalar_values<T>())
            for (T beta : scalar_values<T>()) {
              const Index ka = side == 'L' ? m : n;
              Dense<T> a(ka, ka), b(m, n);
              a.triangle(uplo, 'N', hermitian);
              if (alpha == T(0)) a.fill_nan(), b.fill_nan();
              Dense<T> ce = output_matrix(m, n, beta);
              Dense<T> cr = ce;
              BlasInt bm = m, bn = n, lda = a.ld, ldb = b.ld, ldc = ce.ld;
              f.eig(&side, &uplo, &bm, &bn, rp(&alpha), rp(a.data), &lda, rp(b.data), &ldb, rp(&beta), rp(ce.data),
                    &ldc);
              f.ref(&side, &uplo, &bm, &bn, rp(&alpha), rp(a.data), &lda, rp(b.data), &ldb, rp(&beta), rp(cr.data),
                    &ldc);
              verify_same(
                  ce.data, cr.data, ka, hermitian ? "hemm" : "symm",
                  BLAS_ARGS(BLAS_ARG(side), BLAS_ARG(uplo), BLAS_ARG(m), BLAS_ARG(n), BLAS_ARG(alpha), BLAS_ARG(beta)));
            }
}

// syrk and syr2k, or herk and her2k (real alpha and beta for herk, real beta for her2k) when hermitian.
template <typename T, typename S, typename Frk, typename Fr2k>
void check_rank_k(Pair<Frk> rk, Pair<Fr2k> r2k, bool hermitian) {
  std::vector<char> transes = {'N', 'T', 'C'};
  if (hermitian)
    transes = {'N', 'C'};
  else if (NumTraits<T>::IsComplex)
    transes = {'N', 'T'};
  for (Index n : kSizes3)
    for (Index k : {0, 1, 9})
      for (char uplo : kUplo)
        for (char trans : transes)
          for (T a2 : scalar_values<T>())
            for (T b2 : scalar_values<T>()) {
              S alpha = convert<S>(a2), beta = convert<S>(b2);
              T alpha2 = a2;
              Dense<T> a(trans == 'N' ? n : k, trans == 'N' ? k : n), b(a.rows, a.cols);
              if (a2 == T(0)) a.fill_nan(), b.fill_nan();
              Dense<T> ce = output_matrix(n, n, T(beta));
              ce.triangle(uplo, 'N', hermitian);
              Dense<T> cr = ce;
              BlasInt bn = n, bk = k, lda = a.ld, ldb = b.ld, ldc = ce.ld;
              const std::string args =
                  BLAS_ARGS(BLAS_ARG(uplo), BLAS_ARG(trans), BLAS_ARG(n), BLAS_ARG(k), BLAS_ARG(alpha), BLAS_ARG(beta));
              rk.eig(&uplo, &trans, &bn, &bk, rp(&alpha), rp(a.data), &lda, rp(&beta), rp(ce.data), &ldc);
              rk.ref(&uplo, &trans, &bn, &bk, rp(&alpha), rp(a.data), &lda, rp(&beta), rp(cr.data), &ldc);
              verify_same(ce.data, cr.data, k, hermitian ? "herk" : "syrk", args);
              r2k.eig(&uplo, &trans, &bn, &bk, rp(&alpha2), rp(a.data), &lda, rp(b.data), &ldb, rp(&beta), rp(ce.data),
                      &ldc);
              r2k.ref(&uplo, &trans, &bn, &bk, rp(&alpha2), rp(a.data), &lda, rp(b.data), &ldb, rp(&beta), rp(cr.data),
                      &ldc);
              verify_same(ce.data, cr.data, 2 * k, hermitian ? "her2k" : "syr2k", args);
            }
}

// trmm and trsm.
template <typename T, typename F>
void check_triangular_mm(Pair<F> trmm, Pair<F> trsm) {
  for (Index m : kSizes3)
    for (Index n : kSizes3)
      for (char side : {'L', 'R'})
        for (char uplo : kUplo)
          for (char trans : kTrans)
            for (char diag : {'N', 'U'})
              for (T alpha : scalar_values<T>()) {
                const Index ka = side == 'L' ? m : n;
                Dense<T> a(ka, ka);
                a.triangle(uplo, diag == 'U' ? 'U' : 'D', false);
                if (alpha == T(0)) a.fill_nan();
                Dense<T> be = output_matrix(m, n, alpha);
                Dense<T> br = be;
                BlasInt bm = m, bn = n, lda = a.ld, ldb = be.ld;
                const std::string args = BLAS_ARGS(BLAS_ARG(side), BLAS_ARG(uplo), BLAS_ARG(trans), BLAS_ARG(diag),
                                                   BLAS_ARG(m), BLAS_ARG(n), BLAS_ARG(alpha));
                trmm.eig(&side, &uplo, &trans, &diag, &bm, &bn, rp(&alpha), rp(a.data), &lda, rp(be.data), &ldb);
                trmm.ref(&side, &uplo, &trans, &diag, &bm, &bn, rp(&alpha), rp(a.data), &lda, rp(br.data), &ldb);
                verify_same(be.data, br.data, ka, "trmm", args);
                if (alpha != T(0)) {
                  trsm.eig(&side, &uplo, &trans, &diag, &bm, &bn, rp(&alpha), rp(a.data), &lda, rp(be.data), &ldb);
                  trsm.ref(&side, &uplo, &trans, &diag, &bm, &bn, rp(&alpha), rp(a.data), &lda, rp(br.data), &ldb);
                  verify_same(be.data, br.data, 2 * ka, "trsm", args);
                }
              }
}

// trsm applies alpha before the solve: B = 1e200, A = 1e-200, alpha = 1e-200 gives 1e200, not Inf.
template <typename F>
void check_trsm_scaling(Pair<F> f) {
  for (char side : {'L', 'R'}) {
    char uplo = 'U', trans = 'N', diag = 'N';
    double a = 1e-200, alpha = 1e-200, be = 1e200, br = 1e200;
    BlasInt one = 1;
    f.eig(&side, &uplo, &trans, &diag, &one, &one, &alpha, &a, &one, &be, &one);
    f.ref(&side, &uplo, &trans, &diag, &one, &one, &alpha, &a, &one, &br, &one);
    verify_same(be, br, 2, "trsm", BLAS_ARGS(BLAS_ARG(side), BLAS_ARG(alpha)));
  }
}

// A real factor scales each component of a complex number, so an infinite real part leaves the imaginary part finite.
// Netlib built with gfortran promotes the factor to complex and returns NaN here, so Eigen is checked on its own.
template <typename T, typename Frot, typename Fherk>
void check_real_factors(Frot rot, Fherk herk) {
  using R = Real<T>;
  const R inf = std::numeric_limits<R>::infinity();
  BlasInt one = 1, two = 2, zero = 0;
  R c = R(0.6), s = R(0.8);
  T x(inf, 1), y(0, 0);
  rot(&one, rp(&x), &one, rp(&y), &one, &c, &s);
  VERIFY_IS_EQUAL(numext::imag(x), c);
  VERIFY_IS_EQUAL(numext::imag(y), -s);

  char uplo = 'U', trans = 'N';
  R alpha = 1, beta = R(0.5);
  T a(0, 0), cmat[4] = {T(1, 0), T(0, 0), T(inf, 1), T(2, 0)};
  herk(&uplo, &trans, &two, &zero, &alpha, rp(&a), &two, &beta, rp(cmat), &two);
  VERIFY_IS_EQUAL(cmat[2], T(inf, beta));
}

#define BOTH(f) pair(f##_, f)

}  // namespace

EIGEN_DECLARE_TEST(blas_reference) {
  using cf = std::complex<float>;
  using cd = std::complex<double>;

  CALL_SUBTEST((check_axpy<float>(BOTH(saxpy)), check_axpy<double>(BOTH(daxpy)), check_axpy<cf>(BOTH(caxpy)),
                check_axpy<cd>(BOTH(zaxpy))));
  CALL_SUBTEST((check_copy_swap<float>(BOTH(scopy), BOTH(sswap)), check_copy_swap<double>(BOTH(dcopy), BOTH(dswap)),
                check_copy_swap<cf>(BOTH(ccopy), BOTH(cswap)), check_copy_swap<cd>(BOTH(zcopy), BOTH(zswap))));
  CALL_SUBTEST((check_scal<float, float>(BOTH(sscal)), check_scal<double, double>(BOTH(dscal)),
                check_scal<cf, cf>(BOTH(cscal)), check_scal<cd, cd>(BOTH(zscal)), check_scal<cf, float>(BOTH(csscal)),
                check_scal<cd, double>(BOTH(zdscal))));
  CALL_SUBTEST((check_real_dot<float>(BOTH(sdot), "sdot"), check_real_dot<double>(BOTH(dsdot), "dsdot"),
                check_sdsdot(BOTH(sdsdot))));
  CALL_SUBTEST((check_dot<cf, complex_float_result>(BOTH(cdotc), "cdotc"),
                check_dot<cf, complex_float_result>(BOTH(cdotu), "cdotu"),
                check_dot<cd, complex_double_result>(BOTH(zdotc), "zdotc"),
                check_dot<cd, complex_double_result>(BOTH(zdotu), "zdotu")));
  CALL_SUBTEST((check_norms<float>(BOTH(snrm2), BOTH(sasum)), check_norms<double>(BOTH(dnrm2), BOTH(dasum)),
                check_norms<cf>(BOTH(scnrm2), BOTH(scasum)), check_norms<cd>(BOTH(dznrm2), BOTH(dzasum))));
  CALL_SUBTEST((check_iamax<float>(BOTH(isamax)), check_iamax<double>(BOTH(idamax)), check_iamax<cf>(BOTH(icamax)),
                check_iamax<cd>(BOTH(izamax))));
  CALL_SUBTEST((check_rot<float>(BOTH(srot)), check_rot<double>(BOTH(drot)), check_rot<cf>(BOTH(csrot)),
                check_rot<cd>(BOTH(zdrot))));
  CALL_SUBTEST((check_rotg<float>(BOTH(srotg)), check_rotg<double>(BOTH(drotg)), check_rotg<cf>(BOTH(crotg)),
                check_rotg<cd>(BOTH(zrotg))));
  CALL_SUBTEST((check_rotm<float>(BOTH(srotm)), check_rotm<double>(BOTH(drotm)), check_rotmg<float>(BOTH(srotmg)),
                check_rotmg<double>(BOTH(drotmg))));

  CALL_SUBTEST((check_gemv_gbmv<float>(BOTH(sgemv), BOTH(sgbmv)), check_gemv_gbmv<double>(BOTH(dgemv), BOTH(dgbmv)),
                check_gemv_gbmv<cf>(BOTH(cgemv), BOTH(cgbmv)), check_gemv_gbmv<cd>(BOTH(zgemv), BOTH(zgbmv))));
  CALL_SUBTEST((check_symmetric_mv<float>(BOTH(ssymv), BOTH(ssbmv), BOTH(sspmv), false),
                check_symmetric_mv<double>(BOTH(dsymv), BOTH(dsbmv), BOTH(dspmv), false),
                check_symmetric_mv<cf>(BOTH(chemv), BOTH(chbmv), BOTH(chpmv), true),
                check_symmetric_mv<cd>(BOTH(zhemv), BOTH(zhbmv), BOTH(zhpmv), true)));
  CALL_SUBTEST(
      (check_triangular_mv<float>(BOTH(strmv), BOTH(strsv), BOTH(stbmv), BOTH(stbsv), BOTH(stpmv), BOTH(stpsv)),
       check_triangular_mv<double>(BOTH(dtrmv), BOTH(dtrsv), BOTH(dtbmv), BOTH(dtbsv), BOTH(dtpmv), BOTH(dtpsv)),
       check_triangular_mv<cf>(BOTH(ctrmv), BOTH(ctrsv), BOTH(ctbmv), BOTH(ctbsv), BOTH(ctpmv), BOTH(ctpsv)),
       check_triangular_mv<cd>(BOTH(ztrmv), BOTH(ztrsv), BOTH(ztbmv), BOTH(ztbsv), BOTH(ztpmv), BOTH(ztpsv))));
  CALL_SUBTEST((check_ger<float>(BOTH(sger), "sger"), check_ger<double>(BOTH(dger), "dger"),
                check_ger<cf>(BOTH(cgeru), "cgeru"), check_ger<cf>(BOTH(cgerc), "cgerc"),
                check_ger<cd>(BOTH(zgeru), "zgeru"), check_ger<cd>(BOTH(zgerc), "zgerc")));
  CALL_SUBTEST((check_symmetric_updates<float, float>(BOTH(ssyr), BOTH(sspr), BOTH(ssyr2), BOTH(sspr2), false),
                check_symmetric_updates<double, double>(BOTH(dsyr), BOTH(dspr), BOTH(dsyr2), BOTH(dspr2), false),
                check_symmetric_updates<cf, float>(BOTH(cher), BOTH(chpr), BOTH(cher2), BOTH(chpr2), true),
                check_symmetric_updates<cd, double>(BOTH(zher), BOTH(zhpr), BOTH(zher2), BOTH(zhpr2), true)));

  CALL_SUBTEST((check_gemm<float>(BOTH(sgemm)), check_gemm<double>(BOTH(dgemm)), check_gemm<cf>(BOTH(cgemm)),
                check_gemm<cd>(BOTH(zgemm))));
  CALL_SUBTEST((check_gemmtr<float>(BOTH(sgemmtr)), check_gemmtr<double>(BOTH(dgemmtr)),
                check_gemmtr<cf>(BOTH(cgemmtr)), check_gemmtr<cd>(BOTH(zgemmtr))));
  CALL_SUBTEST((check_symm<float>(BOTH(ssymm), false), check_symm<double>(BOTH(dsymm), false),
                check_symm<cf>(BOTH(csymm), false), check_symm<cd>(BOTH(zsymm), false),
                check_symm<cf>(BOTH(chemm), true), check_symm<cd>(BOTH(zhemm), true)));
  CALL_SUBTEST((check_rank_k<float, float>(BOTH(ssyrk), BOTH(ssyr2k), false),
                check_rank_k<double, double>(BOTH(dsyrk), BOTH(dsyr2k), false),
                check_rank_k<cf, cf>(BOTH(csyrk), BOTH(csyr2k), false),
                check_rank_k<cd, cd>(BOTH(zsyrk), BOTH(zsyr2k), false),
                check_rank_k<cf, float>(BOTH(cherk), BOTH(cher2k), true),
                check_rank_k<cd, double>(BOTH(zherk), BOTH(zher2k), true)));
  CALL_SUBTEST(check_trsm_scaling(BOTH(dtrsm)));
  CALL_SUBTEST((check_real_factors<cf>(csrot_, cherk_), check_real_factors<cd>(zdrot_, zherk_)));
  CALL_SUBTEST((check_triangular_mm<float>(BOTH(strmm), BOTH(strsm)),
                check_triangular_mm<double>(BOTH(dtrmm), BOTH(dtrsm)),
                check_triangular_mm<cf>(BOTH(ctrmm), BOTH(ctrsm)), check_triangular_mm<cd>(BOTH(ztrmm), BOTH(ztrsm))));

  for (const auto& m : mismatches()) std::cerr << m.first << ": " << m.second << " mismatching calls\n";
  VERIFY(mismatches().empty());
}
