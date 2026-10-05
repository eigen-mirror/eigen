// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include "structured_test_helpers.h"

#include <contrib/Eigen/StructuredMatrices>

using namespace Eigen;

// Dense inverse DFT matrix, used to synthesize generators independently of the
// FFT implementation under test.
template <typename RealScalar>
Matrix<std::complex<RealScalar>, Dynamic, Dynamic> inverse_dft_matrix(Index n) {
  typedef std::complex<RealScalar> Complex;
  Matrix<Complex, Dynamic, Dynamic> F(n, n);
  for (Index a = 0; a < n; ++a)
    for (Index c = 0; c < n; ++c)
      F(a, c) =
          std::polar(RealScalar(1) / RealScalar(n), RealScalar(2 * EIGEN_PI) * RealScalar((a * c) % n) / RealScalar(n));
  return F;
}

// Reference dense BCCB built entry-wise from the generating array, independently
// of the operator under test: entry (i,j) with i = b1*n2+i2, j = c1*n2+j2 is
// G((i2-j2) mod n2, (b1-c1) mod n1).
template <typename Scalar>
Matrix<Scalar, Dynamic, Dynamic> reference_bccb(const Matrix<Scalar, Dynamic, Dynamic>& G) {
  const Index n2 = G.rows(), n1 = G.cols(), N = n1 * n2;
  Matrix<Scalar, Dynamic, Dynamic> dense(N, N);
  for (Index j = 0; j < N; ++j)
    for (Index i = 0; i < N; ++i) {
      Index k2 = i % n2 - j % n2;
      if (k2 < 0) k2 += n2;
      Index k1 = i / n2 - j / n2;
      if (k1 < 0) k1 += n1;
      dense(i, j) = G(k2, k1);
    }
  return dense;
}

template <typename Scalar>
void test_bccb_product(Index n2, Index n1) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Mat G = Mat::Random(n2, n1);
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  const Index N = n1 * n2;
  VERIFY_IS_EQUAL(C.rows(), N);

  Mat Cd = C;
  VERIFY_IS_APPROX(Cd, dense);
  Mat accumulated = Mat::Random(N, N);
  const Mat initial = accumulated;
  accumulated += C;
  VERIFY_IS_APPROX(accumulated, (initial + dense).eval());
  accumulated = initial;
  accumulated -= C;
  VERIFY_IS_APPROX(accumulated, (initial - dense).eval());
  for (Index t = 0; t < 5; ++t) {
    Index i = internal::random<Index>(0, N - 1), j = internal::random<Index>(0, N - 1);
    VERIFY_IS_APPROX(C.coeff(i, j), dense(i, j));
  }

  Vec x = Vec::Random(N);
  VERIFY_IS_APPROX((C * x).eval(), (dense * x).eval());

  Mat X = Mat::Random(N, 3);
  VERIFY_IS_APPROX((C * X).eval(), (dense * X).eval());

  // Accumulation form exercised by the iterative solvers.
  Vec y = Vec::Random(N);
  Vec y0 = y;
  y.noalias() += C * x;
  VERIFY_IS_APPROX(y, (y0 + dense * x).eval());
}

template <typename Scalar>
void test_bccb_transpose(Index n2, Index n1) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Mat G = Mat::Random(n2, n1);
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  const Index N = n1 * n2;

  Mat Td = C.transpose();
  VERIFY_IS_APPROX(Td, Mat(dense.transpose()));
  Mat Ad = C.adjoint();
  VERIFY_IS_APPROX(Ad, Mat(dense.adjoint()));
  Mat Kd = C.conjugate();
  VERIFY_IS_APPROX(Kd, Mat(dense.conjugate()));

  Vec x = Vec::Random(N);
  VERIFY_IS_APPROX((C.transpose() * x).eval(), (dense.transpose() * x).eval());
  VERIFY_IS_APPROX((C.adjoint() * x).eval(), (dense.adjoint() * x).eval());

  // Exact round trips: generators and symbols are pure permutations/conjugations.
  Bccb<Scalar> Ctt = C.transpose().transpose();
  VERIFY_IS_EQUAL(Ctt.generator(), G);
  VERIFY_IS_EQUAL(Ctt.symbol(), C.symbol());
  Bccb<Scalar> Caa = C.adjoint().adjoint();
  VERIFY_IS_EQUAL(Caa.generator(), G);
  VERIFY_IS_EQUAL(Caa.symbol(), C.symbol());
}

// transpose()/conjugate()/adjoint() return owning temporaries, so the product
// expression must nest the structured operand by value: a delayed-evaluated
// expression has to outlive the temporary operator it was built from. The static
// check pins the value nesting; the behavioral check would read freed memory if
// the product held a reference instead.
template <typename Scalar>
void test_bccb_delayed_product(Index n2, Index n1) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  STATIC_CHECK(!std::is_reference<typename internal::ref_selector<Bccb<Scalar>>::type>::value);

  Mat G = Mat::Random(n2, n1);
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  const Index N = n1 * n2;
  Vec x = Vec::Random(N);

  auto expr = C.adjoint() * x;        // the adjoint temporary dies with the full expression
  Vec scribble = Vec::Random(2 * N);  // reuses the temporary's freed heap storage
  Vec y = expr;
  VERIFY_IS_APPROX(y, (dense.adjoint() * x).eval());
  VERIFY_IS_EQUAL(scribble.size(), 2 * N);  // keep the scribble alive across the evaluation
}

// The products carry the default product tag, so plain assignment materializes
// a temporary exactly like a dense product: x = C * x and x += C * x get the
// ordinary dense-product aliasing semantics.
template <typename Scalar>
void test_bccb_aliased_product(Index n2, Index n1) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Mat G = Mat::Random(n2, n1);
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  const Index N = n1 * n2;

  Vec x = Vec::Random(N);
  Vec y = x;
  y = C * y;
  VERIFY_IS_APPROX(y, (dense * x).eval());

  y = x;
  y += C * y;
  VERIFY_IS_APPROX(y, (x + dense * x).eval());

  y = x;
  y -= C * y;
  VERIFY_IS_APPROX(y, (x - dense * x).eval());

  Mat X = Mat::Random(N, 3);
  Mat Y = X;
  Y = C * Y;
  VERIFY_IS_APPROX(Y, (dense * X).eval());
}

// Aliasing beyond the same-object case: the default-product temporary must also
// resolve right-hand-side expressions that reference the destination and
// overlapping views of one buffer, neither of which is_same_dense can see.
template <typename Scalar>
void test_bccb_aliased_expression(Index n2, Index n1) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Mat G = Mat::Random(n2, n1);
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  const Index N = n1 * n2;

  // Right-hand-side expression referencing the destination.
  Vec x = Vec::Random(N), x0 = x;
  x = C * (x + Vec::Ones(N));
  VERIFY_IS_APPROX(x, (dense * (x0 + Vec::Ones(N))).eval());

  // Overlapping (shifted) segments of one buffer.
  Vec buf = Vec::Random(N + 1);
  Vec expected = dense * buf.tail(N);
  buf.head(N) = C * buf.tail(N);
  VERIFY_IS_APPROX(buf.head(N).eval(), expected);
}

// Mixed-scalar products: a real operator applied to a complex right-hand side
// (and a complex operator applied to a real one) promotes to the complex product
// scalar, so alpha and the accumulation must run in the promoted type rather than
// the operator scalar.
template <typename RealScalar>
void test_bccb_mixed_scalar(Index n2, Index n1) {
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<RealScalar, Dynamic, 1> RVec;
  typedef Matrix<RealScalar, Dynamic, Dynamic> RMat;
  typedef Matrix<Complex, Dynamic, 1> CVec;
  typedef Matrix<Complex, Dynamic, Dynamic> CMat;

  RMat G = RMat::Random(n2, n1);
  Bccb<RealScalar> C(G);
  CMat dense = reference_bccb<RealScalar>(G).template cast<Complex>();
  const Index N = n1 * n2;

  CVec x = CVec::Random(N);
  CVec y = C * x;
  VERIFY_IS_APPROX(y, (dense * x).eval());

  CVec y0 = CVec::Random(N);
  y = y0;
  y.noalias() += C * x;
  VERIFY_IS_APPROX(y, (y0 + dense * x).eval());

  CMat Gc = CMat::Random(n2, n1);
  Bccb<Complex> Cc(Gc);
  CMat denseC = reference_bccb<Complex>(Gc);
  RVec xr = RVec::Random(N);
  CVec z = Cc * xr;
  VERIFY_IS_APPROX(z, (denseC * xr).eval());
}

// Non-5-smooth block dimensions: products transform on the padded per-axis
// circulant-embedding grid (kissfft's generic butterfly is quadratic in prime
// factors), while the spectral operations keep the exact-size symbol. Cover
// each padded combination -- n2 awkward, n1 awkward, both -- against the dense
// reference, including the transposition family (which reuses the padded
// symbol through the reversal/conjugation rules), solve, and inverse (which
// rebuilds the padded symbol for the inverse operator).
template <typename Scalar>
void test_bccb_prime_dimensions(Index n2, Index n1) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
  const Index N = n1 * n2;

  Mat G = Mat::Random(n2, n1);
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);

  Vec x = Vec::Random(N);
  VERIFY_IS_APPROX((C * x).eval(), (dense * x).eval());
  Mat X = Mat::Random(N, 3);
  VERIFY_IS_APPROX((C * X).eval(), (dense * X).eval());

  VERIFY_IS_APPROX((C.transpose() * x).eval(), (dense.transpose() * x).eval());
  VERIFY_IS_APPROX((C.adjoint() * x).eval(), (dense.adjoint() * x).eval());
  VERIFY_IS_APPROX((C.conjugate() * x).eval(), (dense.conjugate() * x).eval());

  // Exact symbol round trip through the transposition family: both cached
  // symbols are pure permutations/conjugations of the originals.
  Bccb<Scalar> Ctt = C.transpose().transpose();
  VERIFY_IS_EQUAL(Ctt.generator(), G);
  VERIFY_IS_EQUAL(Ctt.symbol(), C.symbol());

  // Solve (exact-size spectral path) and inverse (whose product rebuilds the
  // padded symbol): diagonal dominance keeps the symbol away from zero.
  Mat Gd = Mat::Random(n2, n1);
  Gd(0, 0) += Scalar(RealScalar(2 * N));
  Bccb<Scalar> Cd(Gd);
  Mat densed = reference_bccb<Scalar>(Gd);
  Vec b = Vec::Random(N);
  Vec xs = Cd.solve(b);
  VERIFY_IS_APPROX((densed * xs).eval(), b);
  VERIFY_IS_APPROX((Cd.inverse() * b).eval(), xs);
}

// Dynamic dimension mismatches must trip the runtime assertion when the product
// or solve expression is built. (Incompatible *fixed* sizes are rejected at
// compile time by a static assertion in operator* / solve, which a runtime test
// cannot exercise.)
template <typename = void>
void test_bccb_dimension_asserts() {
  typedef Matrix<double, Dynamic, 1> Vec;
  typedef Matrix<double, Dynamic, Dynamic> Mat;

  Mat G = Mat::Random(3, 4);
  Bccb<double> C(G);
  Vec bad = Vec::Random(11);
  VERIFY_RAISES_ASSERT(C * bad);
  VERIFY_RAISES_ASSERT(C.solve(bad));
}

template <typename Scalar>
void test_bccb_solve(Index n2, Index n1) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  // Diagonal dominance (through the (0,0) generator entry) keeps the symbol away
  // from zero, so the direct 2-D FFT solve is exact up to roundoff.
  Mat G = Mat::Random(n2, n1);
  G(0, 0) += Scalar(RealScalar(2 * n1 * n2));
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  const Index N = n1 * n2;

  Vec b = Vec::Random(N);
  Vec x = C.solve(b);
  VERIFY_IS_APPROX((dense * x).eval(), b);

  Mat B = Mat::Random(N, 3);
  Mat Xs = C.solve(B);
  VERIFY_IS_APPROX((dense * Xs).eval(), B);
}

// Rank-deficient BCCB synthesized by zeroing 2-D symbol entries: the rank counts
// the surviving entries and solve() matches the SVD pseudo-inverse.
template <typename Scalar>
void test_bccb_rank_deficient(Index n2, Index n1, Index defect) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
  typedef Matrix<Complex, Dynamic, Dynamic> CMat;

  const Index N = n1 * n2;
  CMat S = CMat::Random(n2, n1);
  S.array() += Complex(2);  // keep the surviving moduli away from the threshold
  for (Index k = 0; k < defect; ++k) S((2 * k + 1) % n2, (3 * k) % n1) = Complex(0);

  const CMat F2 = inverse_dft_matrix<RealScalar>(n2), F1 = inverse_dft_matrix<RealScalar>(n1);
  Mat G = (F2 * S * F1.transpose()).eval();  // Scalar is complex here
  Bccb<Scalar> C(G);
  VERIFY_IS_EQUAL(C.rank(), N - defect);

  Mat dense = reference_bccb<Scalar>(G);
  JacobiSVD<Mat> svd(dense, ComputeThinU | ComputeThinV);
  Vec b = Vec::Random(N);
  VERIFY_IS_APPROX(C.solve(b), svd.solve(b).eval());
}

template <typename Scalar>
void test_bccb_zero(Index n2, Index n1) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Bccb<Scalar> C(Mat(Mat::Zero(n2, n1)));
  VERIFY_IS_EQUAL(C.rank(), 0);
  VERIFY(C.singularValues().isZero());
  Vec b = Vec::Random(n1 * n2);
  VERIFY(C.solve(b).isZero());
}

template <typename = void>
void test_bccb_nan_propagation(Index n2, Index n1) {
  typedef Matrix<double, Dynamic, 1> Vec;
  typedef Matrix<double, Dynamic, Dynamic> Mat;

  Mat G = Mat::Random(n2, n1);
  G(n2 / 2, n1 / 2) = std::numeric_limits<double>::quiet_NaN();
  Bccb<double> C(G);
  VERIFY_IS_EQUAL(C.rank(), n1 * n2);
  Vec b = Vec::Random(n1 * n2);
  Vec x = C.solve(b);
  VERIFY(!(x.array() == x.array()).all());
}

// An Inf or NaN in the data is never lost and does not leak into other columns
// (see nonfinite_covers_reference()).
template <typename Scalar>
void test_bccb_nonfinite_product(Index n2, Index n1) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
  const RealScalar inf = std::numeric_limits<RealScalar>::infinity();
  const RealScalar nan = std::numeric_limits<RealScalar>::quiet_NaN();
  const Index N = n1 * n2;

  Mat G = Mat::Random(n2, n1);
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);

  Vec x = Vec::Random(N);
  x[N / 2] = Scalar(inf);
  VERIFY(nonfinite_covers_reference((C * x).eval(), reference_product_ieee(dense, x)));
  Vec xn = Vec::Random(N);
  xn[N - 1] = Scalar(nan);
  VERIFY(nonfinite_covers_reference((C * xn).eval(), reference_product_ieee(dense, xn)));
  Vec xz = Vec::Zero(N);
  xz[0] = Scalar(nan);
  VERIFY(nonfinite_covers_reference((C * xz).eval(), reference_product_ieee(dense, xz)));

  Mat Xm(N, 2);
  Xm.col(0) = Vec::Random(N);
  Xm.col(1) = x;
  Mat Ym = C * Xm;
  VERIFY_IS_APPROX(Ym.col(0).eval(), (dense * Xm.col(0)).eval());
  VERIFY(nonfinite_covers_reference(Ym.col(1).eval(), reference_product_ieee(dense, Vec(Xm.col(1)))));

  Mat G2 = Mat::Random(n2, n1);
  G2(n2 / 2, n1 / 2) = Scalar(-inf);
  Bccb<Scalar> C2(G2);
  Mat dense2 = reference_bccb<Scalar>(G2);
  Vec x2 = Vec::Random(N);
  VERIFY(nonfinite_covers_reference((C2 * x2).eval(), reference_product_ieee(dense2, x2)));

  Mat Gd = Mat::Random(n2, n1);
  Gd(0, 0) += Scalar(RealScalar(2 * N));
  Bccb<Scalar> Cd(Gd);
  Mat B(N, 2);
  B.col(0) = Vec::Random(N);
  B.col(1) = x;
  Mat X = Cd.solve(B);
  VERIFY_IS_APPROX((reference_bccb<Scalar>(Gd) * X.col(0)).eval(), Vec(B.col(0)));
  VERIFY(!X.col(1).allFinite());
}

// An all-zero right-hand side does not produce an exact zero when the operator
// holds non-finite data: every row of a BCCB touches every generator
// entry, so each result entry is a dot product with an Inf coefficient times
// zero -- NaN under IEEE, not zero.
template <typename = void>
void test_bccb_nonfinite_zero_rhs(Index n2, Index n1) {
  typedef Matrix<double, Dynamic, 1> Vec;
  typedef Matrix<double, Dynamic, Dynamic> Mat;
  const Index N = n1 * n2;

  Mat G = Mat::Random(n2, n1);
  G(n2 / 2, n1 / 2) = std::numeric_limits<double>::infinity();
  Bccb<double> C(G);
  Mat dense = reference_bccb<double>(G);

  const Vec z = Vec::Zero(N);
  Vec y = C * z;
  VERIFY(y.hasNaN());
  VERIFY(nonfinite_covers_reference(y, reference_product_ieee(dense, z)));

  // The same holds for solve(): the symbol of a non-finite operator holds NaN,
  // so the pseudo-inverse applied to a zero right-hand side is NaN, not zero.
  Vec xs = C.solve(z);
  VERIFY(xs.hasNaN());
}

// Spectra with a wide dynamic range: the determinant is representable, but a
// plain product of the eigenvalues in FFT order overflows to infinity (or
// underflows to an exact zero) partway through. Pins the balanced accumulation
// in determinant().
template <typename = void>
void test_bccb_determinant_scaled() {
  typedef std::complex<double> Complex;
  typedef Matrix<Complex, Dynamic, Dynamic> CMat;

  const Index n2 = 25, n1 = 40;  // N = 1000

  // Symbol magnitudes `lead` on the first nLead column-major-flattened indices
  // (the accumulation order of determinant()) and `rest` elsewhere, so the
  // naive running product leaves the representable range partway through while
  // the true determinant lead^nLead * rest^(N - nLead) is representable. The
  // generator is recovered through the dense inverse 2-D DFT matrices,
  // independently of the implementation under test.
  auto makeOperator = [n2, n1](double lead, double rest) {
    constexpr Index nLead = 653;
    CMat S(n2, n1);
    for (Index k1 = 0; k1 < n1; ++k1)
      for (Index k2 = 0; k2 < n2; ++k2) S(k2, k1) = Complex(k1 * n2 + k2 < nLead ? lead : rest);
    const CMat F2 = inverse_dft_matrix<double>(n2), F1 = inverse_dft_matrix<double>(n1);
    return Bccb<Complex>((F2 * S * F1.transpose()).eval());
  };

  // The spectrum is only reproduced up to the FFT round trip's forward error,
  // and the determinant multiplies ~1e3 such factors.
  const double kSpectrumRoundTripTol = 1e8 * NumTraits<double>::epsilon();
  {
    // det = 10^653 * 10^-347 = 1e306; the naive partial product reaches 1e327.
    Bccb<Complex> C = makeOperator(10.0, 0.1);
    const Complex det = C.determinant();
    VERIFY((numext::isfinite)(numext::abs(det)));
    VERIFY(numext::abs(det / 1e306 - Complex(1)) <= kSpectrumRoundTripTol);
  }
  {
    // det = 10^-653 * 10^347 = 1e-306; the naive partial product reaches
    // 1e-327, well below the smallest subnormal, and flushes to an exact zero.
    Bccb<Complex> C = makeOperator(0.1, 10.0);
    const Complex det = C.determinant();
    VERIFY(det != Complex(0));
    VERIFY(numext::abs(det / 1e-306 - Complex(1)) <= kSpectrumRoundTripTol);
  }
  {
    // A genuinely overflowing determinant must still saturate to infinity.
    typedef Matrix<double, Dynamic, Dynamic> Mat;
    Mat G = Mat::Zero(6, 6);
    G(0, 0) = 1e308;  // the symbol is 1e308 everywhere: det = 10^11088
    Bccb<double> C(G);
    VERIFY((numext::isinf)(C.determinant()));
  }
}

// The rank decision at the clamped threshold (the smallest normal number): a
// smallest-normal symbol entry is still inverted -- the comparison is strict,
// matching SVDBase::rank(), which likewise reports rank one -- a subnormal entry
// is treated as an exact zero, and non-finite entries count as non-zero.
template <typename = void>
void test_bccb_rank_boundaries() {
  typedef Matrix<double, Dynamic, 1> Vec;
  typedef Matrix<double, Dynamic, Dynamic> Mat;

  const double mn = (std::numeric_limits<double>::min)();
  const Vec b = Vec::Ones(1);
  {
    Bccb<double> C(Mat(Mat::Constant(1, 1, mn)));
    VERIFY_IS_EQUAL(C.rank(), 1);
    Vec x = C.solve(b);
    VERIFY((numext::isfinite)(x[0]));
    VERIFY_IS_APPROX(x[0], 1.0 / mn);
  }
  {
    Bccb<double> C(Mat(Mat::Constant(1, 1, mn / 2)));  // subnormal
    VERIFY_IS_EQUAL(C.rank(), 0);
    VERIFY(C.solve(b).isZero());
  }
  {
    Bccb<double> C(Mat(Mat::Constant(1, 1, std::numeric_limits<double>::infinity())));
    VERIFY_IS_EQUAL(C.rank(), 1);
    VERIFY((numext::isinf)(C.determinant()));
  }
}

template <typename Scalar>
void test_bccb_eigen(Index n2, Index n1) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
  typedef Matrix<Complex, Dynamic, Dynamic> CMat;

  Mat G = Mat::Random(n2, n1);
  Bccb<Scalar> C(G);
  CMat denseC = reference_bccb<Scalar>(G).template cast<Complex>();
  const Index N = n1 * n2;

  Matrix<Complex, Dynamic, 1> lam = C.eigenvalues();
  CMat V = C.eigenvectors();
  VERIFY_IS_APPROX((denseC * V).eval(), (V * lam.asDiagonal()).eval());
  VERIFY_IS_APPROX((V.adjoint() * V).eval(), CMat(CMat::Identity(N, N)));
}

template <typename Scalar>
void test_bccb_svd(Index n2, Index n1) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
  typedef Matrix<Complex, Dynamic, Dynamic> CMat;

  Mat G = Mat::Random(n2, n1);
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  const Index N = n1 * n2;

  Matrix<RealScalar, Dynamic, 1> sv = C.singularValues();
  JacobiSVD<Mat> svd(dense);
  VERIFY_IS_APPROX(sv, svd.singularValues());

  CMat U = C.matrixU(), V = C.matrixV();
  VERIFY_IS_APPROX((U * sv.template cast<Complex>().asDiagonal() * V.adjoint()).eval(),
                   CMat(dense.template cast<Complex>()));
  VERIFY_IS_APPROX((U.adjoint() * U).eval(), CMat(CMat::Identity(N, N)));
  VERIFY_IS_APPROX((V.adjoint() * V).eval(), CMat(CMat::Identity(N, N)));
}

template <typename Scalar>
void test_bccb_inverse(Index n2, Index n1) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Mat G = Mat::Random(n2, n1);
  G(0, 0) += Scalar(RealScalar(2 * n1 * n2));  // safely invertible, tame determinant scale
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  const Index N = n1 * n2;

  Mat inv = C.inverse();
  VERIFY_IS_APPROX((inv * dense).eval(), Mat(Mat::Identity(N, N)));

  Vec b = Vec::Random(N);
  VERIFY_IS_APPROX((C.inverse() * b).eval(), C.solve(b));
}

template <typename Scalar>
void test_bccb_determinant(Index n2, Index n1) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Mat G = Mat::Random(n2, n1);
  G(0, 0) += Scalar(RealScalar(2 * n1 * n2));
  Bccb<Scalar> C(G);
  Mat dense = reference_bccb<Scalar>(G);
  VERIFY_IS_APPROX(C.determinant(), dense.determinant());
}

template <typename Scalar>
void test_bccb_matrix_free_cg(Index n2, Index n1) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  // Symmetrizing the generator under the two-dimensional index reversal makes
  // the BCCB matrix symmetric; diagonal dominance then makes it positive
  // definite.
  Mat G = Mat::Random(n2, n1);
  Mat Gs = G;
  for (Index k1 = 0; k1 < n1; ++k1)
    for (Index k2 = 0; k2 < n2; ++k2) Gs(k2, k1) = (G(k2, k1) + G((n2 - k2) % n2, (n1 - k1) % n1)) / Scalar(2);
  Gs(0, 0) += Scalar(RealScalar(2 * n1 * n2));
  Bccb<Scalar> C(Gs);
  Mat dense = reference_bccb<Scalar>(Gs);
  VERIFY_IS_APPROX(dense, Mat(dense.transpose()));

  const Index N = n1 * n2;
  Vec b = Vec::Random(N);
  ConjugateGradient<Bccb<Scalar>, Lower | Upper, IdentityPreconditioner> cg;
  cg.compute(C);
  Vec x = cg.solve(b);
  VERIFY(cg.info() == Success);
  VERIFY_IS_APPROX((dense * x).eval(), b);
}

template <typename Scalar, int N2, int N1>
void test_bccb_fixed() {
  typedef Matrix<Scalar, N2, N1> GenMat;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  GenMat G = GenMat::Random();
  Bccb<Scalar, N2, N1> C(G);
  STATIC_CHECK((Bccb<Scalar, N2, N1>::RowsAtCompileTime == N1 * N2));
  STATIC_CHECK((internal::remove_all_t<decltype(makeBccb(G))>::RowsAtCompileTime == N1 * N2));

  Mat dense = reference_bccb<Scalar>(Mat(G));
  Matrix<Scalar, N1 * N2, N1 * N2> Cd = C;
  VERIFY_IS_APPROX(Mat(Cd), dense);

  Matrix<Scalar, N1 * N2, 1> x = Matrix<Scalar, N1 * N2, 1>::Random();
  Matrix<Scalar, N1 * N2, 1> y = C * x;
  VERIFY_IS_APPROX(y, (dense * x).eval());
}

template <typename RealScalar>
void test_constant_symbol_solves() {
  using Complex = std::complex<RealScalar>;
  using CMatrix = Matrix<Complex, Dynamic, Dynamic>;
  CMatrix generator = CMatrix::Zero(3, 5);
  generator(0, 0) = Complex(3, 4);  // A constant non-real symbol.
  const Bccb<Complex> op(generator);
  const CMatrix expected = CMatrix::Random(15, 2);
  const CMatrix rhs = generator(0, 0) * expected;
  CMatrix actual = op.solve(rhs);
  VERIFY(actual.allFinite());
  // Forward/inverse 3-by-5 transforms and one reciprocal contribute rounding.
  VERIFY((actual - expected).norm() <= RealScalar(64) * NumTraits<RealScalar>::epsilon() * expected.norm());
  Matrix<Complex, Dynamic, 1> column = Matrix<Complex, Dynamic, 1>::Zero(15);
  column(0) = generator(0, 0);
  const Circulant<Complex> circulant(column);
  actual = circulant.solve(rhs);
  VERIFY(actual.allFinite());
  VERIFY((actual - expected).norm() <= RealScalar(64) * NumTraits<RealScalar>::epsilon() * expected.norm());
  actual = circulant.inverse() * rhs;
  VERIFY(actual.allFinite());
  VERIFY((actual - expected).norm() <= RealScalar(64) * NumTraits<RealScalar>::epsilon() * expected.norm());
}

template <typename RealScalar>
void test_circulant_inverse_modes() {
  using Complex = std::complex<RealScalar>;
  using CVector = Matrix<Complex, Dynamic, 1>;
  const RealScalar epsilon = NumTraits<RealScalar>::epsilon();
  // The small mode is representable but below the solve's rank threshold.
  CVector column(2);
  column << Complex(1), Complex(1 - epsilon);
  const Circulant<Complex> circulant(column);
  VERIFY_IS_EQUAL(circulant.rank(), 1);
  const Complex circulant_mode = circulant.inverse().symbol()(1);
  VERIFY(numext::abs(circulant_mode * epsilon - Complex(1)) <= RealScalar(8) * epsilon);
}

EIGEN_DECLARE_TEST(structured_bccb) {
  for (int i = 0; i < g_repeat; ++i) {
    // Products, dense assignment, coefficient access: scalar tier (N <= 32) and
    // 2-D FFT tier, including single-row/column-of-blocks degenerate shapes.
    CALL_SUBTEST_1((test_bccb_product<double>(1, 1)));
    CALL_SUBTEST_1((test_bccb_product<double>(3, 4)));
    CALL_SUBTEST_1((test_bccb_product<double>(5, 5)));
    CALL_SUBTEST_1((test_bccb_product<double>(8, 6)));
    CALL_SUBTEST_1((test_bccb_product<double>(7, 5)));  // odd prime dimensions
    CALL_SUBTEST_1((test_bccb_product<double>(1, 40)));
    CALL_SUBTEST_1((test_bccb_product<double>(40, 1)));
    CALL_SUBTEST_1((test_bccb_product<float>(6, 8)));
    CALL_SUBTEST_1((test_bccb_product<std::complex<double>>(4, 3)));
    CALL_SUBTEST_1((test_bccb_product<std::complex<double>>(9, 6)));
    CALL_SUBTEST_1((test_bccb_product<std::complex<float>>(6, 7)));

    // Transposition family with exact symbol round trips.
    CALL_SUBTEST_2((test_bccb_transpose<double>(3, 4)));
    CALL_SUBTEST_2((test_bccb_transpose<double>(8, 6)));
    CALL_SUBTEST_2((test_bccb_transpose<double>(1, 12)));
    CALL_SUBTEST_2((test_bccb_transpose<std::complex<double>>(6, 8)));
    CALL_SUBTEST_2((test_bccb_transpose<std::complex<float>>(5, 7)));

    // Direct and pseudo-inverse solves, degenerate operators.
    CALL_SUBTEST_3((test_bccb_solve<double>(1, 1)));
    CALL_SUBTEST_3((test_bccb_solve<double>(6, 8)));
    CALL_SUBTEST_3((test_bccb_solve<float>(5, 6)));
    CALL_SUBTEST_3((test_bccb_solve<std::complex<double>>(6, 6)));
    CALL_SUBTEST_3((test_bccb_rank_deficient<std::complex<double>>(6, 8, 4)));
    CALL_SUBTEST_3((test_bccb_rank_deficient<std::complex<float>>(4, 5, 2)));
    CALL_SUBTEST_3((test_bccb_zero<double>(4, 5)));
    CALL_SUBTEST_3(test_bccb_nan_propagation<>(6, 7));

    // Closed-form eigendecomposition, SVD, inverse, determinant.
    CALL_SUBTEST_4((test_bccb_eigen<double>(4, 5)));
    CALL_SUBTEST_4((test_bccb_eigen<double>(1, 8)));
    CALL_SUBTEST_4((test_bccb_eigen<std::complex<double>>(3, 5)));
    CALL_SUBTEST_4((test_bccb_svd<double>(4, 5)));
    CALL_SUBTEST_4((test_bccb_svd<std::complex<double>>(4, 4)));
    CALL_SUBTEST_4((test_bccb_svd<float>(3, 4)));
    CALL_SUBTEST_4((test_bccb_inverse<double>(6, 8)));
    CALL_SUBTEST_4((test_bccb_inverse<std::complex<double>>(5, 5)));
    CALL_SUBTEST_4((test_bccb_determinant<double>(3, 4)));
    CALL_SUBTEST_4((test_bccb_determinant<std::complex<double>>(3, 3)));

    // Matrix-free iterative solve and fixed-size generators.
    CALL_SUBTEST_5((test_bccb_matrix_free_cg<double>(8, 10)));
    CALL_SUBTEST_5((test_bccb_fixed<double, 3, 4>()));
    CALL_SUBTEST_5((test_bccb_fixed<std::complex<float>, 4, 3>()));

    // Non-5-smooth (prime) block dimensions: the padded product-embedding grid,
    // in each axis combination (n2 awkward, n1 awkward, both).
    CALL_SUBTEST_5((test_bccb_prime_dimensions<double>(7, 6)));
    CALL_SUBTEST_5((test_bccb_prime_dimensions<double>(6, 7)));
    CALL_SUBTEST_5((test_bccb_prime_dimensions<double>(7, 7)));
    CALL_SUBTEST_5((test_bccb_prime_dimensions<std::complex<double>>(11, 7)));
    CALL_SUBTEST_5((test_bccb_prime_dimensions<float>(7, 6)));

    // Product lifetime, aliasing, mixed-scalar promotion, dimension mismatches,
    // across the scalar-loop and FFT dispatch tiers.
    CALL_SUBTEST_6((test_bccb_delayed_product<double>(4, 5)));
    CALL_SUBTEST_6((test_bccb_delayed_product<std::complex<double>>(6, 8)));
    CALL_SUBTEST_6((test_bccb_aliased_product<double>(3, 4)));
    CALL_SUBTEST_6((test_bccb_aliased_product<double>(8, 6)));
    CALL_SUBTEST_6((test_bccb_aliased_product<std::complex<double>>(6, 8)));
    CALL_SUBTEST_6((test_bccb_aliased_expression<double>(3, 4)));
    CALL_SUBTEST_6((test_bccb_aliased_expression<double>(6, 8)));
    CALL_SUBTEST_6((test_bccb_aliased_expression<std::complex<double>>(6, 8)));
    CALL_SUBTEST_6((test_bccb_mixed_scalar<double>(3, 4)));
    CALL_SUBTEST_6((test_bccb_mixed_scalar<double>(6, 8)));
    CALL_SUBTEST_6((test_bccb_mixed_scalar<float>(8, 8)));
    CALL_SUBTEST_6(test_bccb_dimension_asserts<>());

    // Balanced determinant accumulation and the rank threshold boundary.
    CALL_SUBTEST_7(test_bccb_determinant_scaled<>());
    CALL_SUBTEST_7(test_bccb_rank_boundaries<>());

    // Inf/NaN propagation on the direct and FFT tiers. A zero right-hand side
    // under a non-finite operator yields NaN, not zero.
    CALL_SUBTEST_7((test_bccb_nonfinite_product<double>(6, 8)));
    CALL_SUBTEST_7((test_bccb_nonfinite_product<double>(3, 4)));
    CALL_SUBTEST_7((test_bccb_nonfinite_product<std::complex<double>>(6, 8)));
    CALL_SUBTEST_7(test_bccb_nonfinite_zero_rhs<>(6, 8));

    CALL_SUBTEST_8(test_constant_symbol_solves<float>());
    CALL_SUBTEST_8(test_constant_symbol_solves<double>());
    CALL_SUBTEST_8(test_circulant_inverse_modes<float>());
    CALL_SUBTEST_8(test_circulant_inverse_modes<double>());
  }
}
