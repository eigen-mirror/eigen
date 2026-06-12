// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include "tridiag_test_matrices.h"
#include <limits>
#include <Eigen/Eigenvalues>

// Test TridiagonalEigenSolver (SIMD Sturm-sequence spectral bisection) on the full
// structured-tridiagonal catalog: compare against the implicit-QR path of
// SelfAdjointEigenSolver::computeFromTridiagonal(), exercise index/value range subset
// selection, and verify absolute accuracy against matrices with known spectra.
template <typename RealScalar>
void tridiagonal_eigensolver_bisection() {
  typedef Matrix<RealScalar, Dynamic, Dynamic> MatrixType;
  typedef Matrix<RealScalar, Dynamic, 1> VectorType;
  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  const RealScalar tiny = (std::numeric_limits<RealScalar>::min)();

  // (a) Compare bisection against the QR path over every structured matrix, and exercise
  // index- and value-range subset selection.
  test::for_all_symmetric_tridiag_test_matrices<RealScalar>([&](const VectorType& diag, const VectorType& offdiag) {
    const Index n = diag.size();

    SelfAdjointEigenSolver<MatrixType> qr;
    qr.computeFromTridiagonal(diag, offdiag, EigenvaluesOnly);
    if (qr.info() != Success) return;  // skip pathological inputs the QR path itself rejects
    const VectorType w = qr.eigenvalues();

    TridiagonalEigenSolver<RealScalar> bz;
    bz.computeEigenvalues(diag, offdiag);
    VERIFY_IS_EQUAL(bz.info(), Success);
    VERIFY_IS_EQUAL(bz.eigenvalues().size(), n);
    for (Index i = 1; i < n; ++i) VERIFY(bz.eigenvalues()(i) >= bz.eigenvalues()(i - 1));

    // Bisection has an absolute accuracy floor of ~pivmin (the smallest safe pivot),
    // so the tolerance carries an absolute term in addition to the eps*||T|| term.
    const RealScalar radius = w.cwiseAbs().maxCoeff();
    const RealScalar tol = RealScalar(64) * RealScalar(n) * (eps * radius + tiny);
    VERIFY((bz.eigenvalues() - w).cwiseAbs().maxCoeff() <= tol);

    // Index-range subsets: the k smallest and the k largest.
    if (n >= 4) {
      const Index k = n / 3;
      TridiagonalEigenSolver<RealScalar> lo, hi;
      lo.computeEigenvalues(diag, offdiag, EigenvalueRange::indices(0, k));
      VERIFY_IS_EQUAL(lo.eigenvalues().size(), k);
      VERIFY((lo.eigenvalues() - w.head(k)).cwiseAbs().maxCoeff() <= tol);
      hi.computeEigenvalues(diag, offdiag, EigenvalueRange::indices(n - k, n));
      VERIFY_IS_EQUAL(hi.eigenvalues().size(), k);
      VERIFY((hi.eigenvalues() - w.tail(k)).cwiseAbs().maxCoeff() <= tol);
    }

    // Value-range subset [vl, vu) about the spectrum center. The count is compared strictly
    // only when no eigenvalue sits within tol of a boundary (otherwise it is legitimately
    // ambiguous which side the boundary eigenvalue falls on).
    if (n >= 2 && radius > tiny) {
      const RealScalar vl = -RealScalar(0.3) * radius, vu = RealScalar(0.3) * radius;
      TridiagonalEigenSolver<RealScalar> bv;
      bv.computeEigenvalues(diag, offdiag, EigenvalueRange::values(double(vl), double(vu)));
      for (Index i = 0; i < bv.eigenvalues().size(); ++i)
        VERIFY(bv.eigenvalues()(i) >= vl - tol && bv.eigenvalues()(i) < vu + tol);
      const bool clean = (w.array() - vl).abs().minCoeff() > tol && (w.array() - vu).abs().minCoeff() > tol;
      if (clean) {
        const Index expected = (w.array() >= vl && w.array() < vu).count();
        VERIFY_IS_EQUAL(bv.eigenvalues().size(), expected);
        if (expected > 0) {
          Index first = 0;
          while (first < n && w(first) < vl) ++first;
          VERIFY((bv.eigenvalues() - w.segment(first, expected)).cwiseAbs().maxCoeff() <= tol);
        }
      }
    }
  });

  // (a2) Half-open value-range convention, fixed-size inputs, and input validation.
  {
    // values(vl, vu) selects the eigenvalues in the lower-closed, upper-open interval [vl, vu) (like
    // indices()): an eigenvalue exactly equal to vl is included, one exactly equal to vu is not. Use a
    // diagonal matrix (zero off-diagonal) with the exact integer spectrum {1, 2, 3} so the endpoints
    // land precisely on eigenvalues.
    VectorType d(3), e(2);
    d << RealScalar(1), RealScalar(2), RealScalar(3);
    e.setZero();
    TridiagonalEigenSolver<RealScalar> bmid;
    bmid.computeEigenvalues(d, e, EigenvalueRange::values(1.0L, 3.0L));
    VERIFY_IS_EQUAL(bmid.eigenvalues().size(), Index(2));  // [1, 3) = {1, 2}: lower end closed, upper end open
    VERIFY_IS_APPROX(bmid.eigenvalues()(0), RealScalar(1));
    VERIFY_IS_APPROX(bmid.eigenvalues()(1), RealScalar(2));
    TridiagonalEigenSolver<RealScalar> bopen;
    bopen.computeEigenvalues(d, e, EigenvalueRange::values(0.0L, 1.0L));
    VERIFY_IS_EQUAL(bopen.eigenvalues().size(), Index(0));  // [0, 1) excludes the eigenvalue at the open end 1
    TridiagonalEigenSolver<RealScalar> blow;
    blow.computeEigenvalues(d, e, EigenvalueRange::values(1.0L, 2.0L));
    VERIFY_IS_EQUAL(blow.eigenvalues().size(), Index(1));  // [1, 2) = {1}: includes the eigenvalue at the closed end 1
    VERIFY_IS_APPROX(blow.eigenvalues()(0), RealScalar(1));

    // Fixed-size input vectors work, including subset selection (the solver's own storage is
    // dynamic, so a subset shorter than the input is not a problem).
    Matrix<RealScalar, 3, 1> fd;
    fd << RealScalar(1), RealScalar(2), RealScalar(3);
    Matrix<RealScalar, 2, 1> fe;
    fe.setZero();
    TridiagonalEigenSolver<RealScalar> fz;
    fz.computeEigenvalues(fd, fe);
    VERIFY_IS_EQUAL(fz.eigenvalues().size(), Index(3));
    fz.computeEigenvalues(fd, fe, EigenvalueRange::indices(0, 1));
    VERIFY_IS_EQUAL(fz.eigenvalues().size(), Index(1));
    VERIFY_IS_APPROX(fz.eigenvalues()(0), RealScalar(1));

    // Querying an uninitialized solver is a usage error.
    TridiagonalEigenSolver<RealScalar> uninit;
    VERIFY_RAISES_ASSERT(uninit.eigenvalues());
    VERIFY_RAISES_ASSERT(uninit.info());

    // A non-finite input (NaN or Inf, diagonal or sub-diagonal) is rejected up front and
    // reported via info().
    VectorType dbad = d;
    dbad(1) = std::numeric_limits<RealScalar>::quiet_NaN();
    TridiagonalEigenSolver<RealScalar> bnan;
    bnan.computeEigenvalues(dbad, e);
    VERIFY_IS_EQUAL(bnan.info(), NoConvergence);
    dbad(1) = std::numeric_limits<RealScalar>::infinity();
    bnan.computeEigenvalues(dbad, e);
    VERIFY_IS_EQUAL(bnan.info(), NoConvergence);
    VectorType ebad = e;
    ebad(0) = std::numeric_limits<RealScalar>::quiet_NaN();
    bnan.computeEigenvalues(d, ebad);
    VERIFY_IS_EQUAL(bnan.info(), NoConvergence);
  }

  // (b) Absolute-accuracy checks against matrices with known spectra (independent of QR).
  // A subnormal-magnitude matrix exercises the normalization: dividing entries by the (subnormal) max
  // magnitude keeps the scaled matrix O(1), whereas multiplying by 1/scale would overflow to infinity
  // and return wrong eigenvalues. Compare against the same matrix in a normal range, scaled back down.
  {
    const Index nn = 7;
    VectorType bd(nn), be(nn - 1);
    test::tridiag_1_2_1(bd, be);
    const RealScalar sub_scale = (std::numeric_limits<RealScalar>::min)() / RealScalar(64);  // subnormal
    TridiagonalEigenSolver<RealScalar> bref, bsub;
    bref.computeEigenvalues(bd, be);
    bsub.computeEigenvalues(VectorType(bd * sub_scale), VectorType(be * sub_scale));
    VERIFY_IS_EQUAL(bsub.info(), Success);
    // The subnormal eigenvalues, scaled back to O(1), agree with the reference to the subnormal
    // granularity at this magnitude (denorm_min / sub_scale) plus the usual bisection error.
    const RealScalar radius = bref.eigenvalues().cwiseAbs().maxCoeff();
    const RealScalar gran = (std::numeric_limits<RealScalar>::denorm_min)() / sub_scale;
    VERIFY((bsub.eigenvalues() / sub_scale - bref.eigenvalues()).cwiseAbs().maxCoeff() <=
           RealScalar(16) * (RealScalar(nn) * eps + gran) * radius);
  }
  const double pi = 3.14159265358979323846;
  for (Index n : {7, 16, 33, 64}) {
    VectorType d(n), e(n - 1), wexact(n);

    // 1-2-1 Toeplitz: lambda_k = 2 - 2 cos(k*pi/(n+1)), ascending for k = 1..n.
    test::tridiag_1_2_1(d, e);
    TridiagonalEigenSolver<RealScalar> b121;
    b121.computeEigenvalues(d, e);
    for (Index k = 0; k < n; ++k) wexact(k) = RealScalar(2.0 - 2.0 * std::cos(double(k + 1) * pi / double(n + 1)));
    VERIFY((b121.eigenvalues() - wexact).cwiseAbs().maxCoeff() <= RealScalar(64) * RealScalar(n) * eps * RealScalar(4));

    // Clement / Kac: integer spectrum -(n-1), -(n-3), ..., (n-1).
    test::tridiag_clement(d, e);
    TridiagonalEigenSolver<RealScalar> bcl;
    bcl.computeEigenvalues(d, e);
    for (Index k = 0; k < n; ++k) wexact(k) = RealScalar(-(n - 1) + 2 * k);
    VERIFY((bcl.eigenvalues() - wexact).cwiseAbs().maxCoeff() <= RealScalar(64) * RealScalar(n) * eps * RealScalar(n));

    // Spectra exactly symmetric about 0: signed Wilkinson, Hermite, Legendre, [1,0,1] Toeplitz.
    auto check_symmetric = [&](const VectorType& dd, const VectorType& ee) {
      TridiagonalEigenSolver<RealScalar> bs;
      bs.computeEigenvalues(dd, ee);
      const VectorType s = bs.eigenvalues();
      const RealScalar rad = s.cwiseAbs().maxCoeff();
      const RealScalar tol = RealScalar(64) * RealScalar(n) * (eps * rad + tiny);
      for (Index i = 0; i < n; ++i) VERIFY(numext::abs(s(i) + s(n - 1 - i)) <= tol);
    };
    test::tridiag_wilkinson_signed(d, e);
    check_symmetric(d, e);
    test::tridiag_hermite(d, e);
    check_symmetric(d, e);
    test::tridiag_legendre(d, e);
    check_symmetric(d, e);
    test::tridiag_toeplitz(d, e);  // default a = 1, b = 0
    check_symmetric(d, e);
  }
}

EIGEN_DECLARE_TEST(tridiagonal_eigensolver) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(tridiagonal_eigensolver_bisection<double>());
    CALL_SUBTEST_2(tridiagonal_eigensolver_bisection<float>());
  }
}
