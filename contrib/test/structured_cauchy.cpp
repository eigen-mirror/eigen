// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include <contrib/Eigen/StructuredMatrices>

using namespace Eigen;

// Reference dense Cauchy built entry-wise from the node vectors.
template <typename Scalar>
Matrix<Scalar, Dynamic, Dynamic> reference_cauchy(const Matrix<Scalar, Dynamic, 1>& x,
                                                  const Matrix<Scalar, Dynamic, 1>& y) {
  Matrix<Scalar, Dynamic, Dynamic> dense(x.size(), y.size());
  for (Index j = 0; j < y.size(); ++j)
    for (Index i = 0; i < x.size(); ++i) dense(i, j) = Scalar(1) / (x[i] - y[j]);
  return dense;
}

// Separated node sets: x in [2,3], y in [0,1], so all denominators are in [1,3].
template <typename Scalar>
void separated_nodes(Index m, Index n, Matrix<Scalar, Dynamic, 1>& x, Matrix<Scalar, Dynamic, 1>& y) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  x = Matrix<Scalar, Dynamic, 1>::Random(m);
  y = Matrix<Scalar, Dynamic, 1>::Random(n);
  x = (x * Scalar(RealScalar(0.5))).array() + Scalar(RealScalar(2.5));
  y = (y * Scalar(RealScalar(0.5))).array() + Scalar(RealScalar(0.5));
}

template <typename Scalar>
void test_cauchy_product(Index m, Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec x, y;
  separated_nodes<Scalar>(m, n, x, y);
  Cauchy<Scalar> C(x, y);
  Mat dense = reference_cauchy<Scalar>(x, y);

  Mat Cd = C;
  VERIFY_IS_APPROX(Cd, dense);
  Mat accumulated = Mat::Random(m, n);
  const Mat initial = accumulated;
  accumulated += C;
  VERIFY_IS_APPROX(accumulated, initial + dense);
  accumulated -= C;
  VERIFY_IS_APPROX(accumulated, initial);
  for (Index t = 0; t < 5; ++t) {
    Index i = internal::random<Index>(0, m - 1), j = internal::random<Index>(0, n - 1);
    VERIFY_IS_APPROX(C.coeff(i, j), dense(i, j));
  }

  Vec v = Vec::Random(n);
  VERIFY_IS_APPROX((C * v).eval(), (dense * v).eval());

  Mat V = Mat::Random(n, 3);
  VERIFY_IS_APPROX((C * V).eval(), (dense * V).eval());

  // Accumulation form exercised by the iterative solvers.
  Vec w = Vec::Random(m);
  Vec w0 = w;
  w.noalias() += C * v;
  VERIFY_IS_APPROX(w, (w0 + dense * v).eval());

  // Several right-hand sides share each reciprocal column; -= passes alpha = -1.
  Mat W = Mat::Random(m, 3);
  const Mat W0 = W;
  W.noalias() -= C * V;
  VERIFY_IS_APPROX(W, (W0 - dense * V).eval());
}

// With 100 right-hand sides, 1100 rows exceed the 1024-row block floor of the
// multi-column product, so the rows split into a full and a ragged block.
template <typename Scalar>
void test_cauchy_product_row_blocks() {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;
  const Index m = 1100, n = 8, r = 100;
  Vec x, y;
  separated_nodes<Scalar>(m, n, x, y);
  Cauchy<Scalar> C(x, y);
  const Mat dense = reference_cauchy<Scalar>(x, y);
  const Mat V = Mat::Random(n, r);
  Mat W = Mat::Random(m, r);
  const Mat W0 = W;
  W.noalias() -= C * V;
  VERIFY_IS_APPROX(W, (W0 - dense * V).eval());
}

template <typename Scalar>
void test_cauchy_transpose(Index m, Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec x, y;
  separated_nodes<Scalar>(m, n, x, y);
  Cauchy<Scalar> C(x, y);
  Mat dense = reference_cauchy<Scalar>(x, y);

  Mat Td = C.transpose();
  VERIFY_IS_APPROX(Td, Mat(dense.transpose()));
  Mat Ad = C.adjoint();
  VERIFY_IS_APPROX(Ad, Mat(dense.adjoint()));
  Mat Kd = C.conjugate();
  VERIFY_IS_APPROX(Kd, Mat(dense.conjugate()));

  Vec w = Vec::Random(m);
  VERIFY_IS_APPROX((C.transpose() * w).eval(), (dense.transpose() * w).eval());
  VERIFY_IS_APPROX((C.adjoint() * w).eval(), (dense.adjoint() * w).eval());
}

// GKO solve on separated random nodes, verified through residuals (Cauchy
// matrices are exponentially ill-conditioned, so forward-error comparisons
// between different algorithms would not be meaningful).
template <typename Scalar>
void test_cauchy_lu(Index n) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec x, y;
  separated_nodes<Scalar>(n, n, x, y);
  Cauchy<Scalar> C(x, y);
  Mat dense = reference_cauchy<Scalar>(x, y);

  CauchyLU<Scalar> lu(C);
  VERIFY(lu.info() == Success);

  // Backward-stability-style residual bound, scaled by the scalar's epsilon (the
  // separated node sets bound the GKO generators by 1, so no growth term).
  const RealScalar tol = RealScalar(1000) * RealScalar(n) * NumTraits<RealScalar>::epsilon();

  Vec b = Vec::Random(n);
  Vec u = lu.solve(b);
  VERIFY((dense * u - b).norm() <= tol * (dense.norm() * u.norm() + b.norm()));

  // Multiple right-hand sides.
  Mat B = Mat::Random(n, 3);
  Mat U = lu.solve(B);
  VERIFY((dense * U - B).norm() <= tol * (dense.norm() * U.norm() + B.norm()));

  // Transposed and adjoint systems reuse the same factorization.
  Vec vt = lu.transpose().solve(b);
  VERIFY((dense.transpose() * vt - b).norm() <= tol * (dense.norm() * vt.norm() + b.norm()));
  Vec va = lu.adjoint().solve(b);
  VERIFY((dense.adjoint() * va - b).norm() <= tol * (dense.norm() * va.norm() + b.norm()));
}

// The Hilbert matrix as a Cauchy matrix (x_i = i+1, y_j = -j): deterministic
// residual bounds far below what the astronomical conditioning would allow a
// forward-error test, plus the closed-form determinant against the dense LU one.
template <typename = void>
void test_cauchy_hilbert() {
  typedef Matrix<double, Dynamic, 1> Vec;
  typedef Matrix<double, Dynamic, Dynamic> Mat;

  auto residual = [](Index n) {
    Vec x(n), y(n);
    for (Index i = 0; i < n; ++i) {
      x[i] = double(i + 1);
      y[i] = -double(i);
    }
    Cauchy<double> C(x, y);
    Mat dense = reference_cauchy<double>(x, y);
    Vec b = dense * Vec::Ones(n);
    CauchyLU<double> lu(C);
    VERIFY(lu.info() == Success);
    Vec u = lu.solve(b);
    return (dense * u - b).norm() / b.norm();
  };
  // GKO partial pivoting is backward stable, so the residual stays a small
  // multiple of epsilon even as the Hilbert conditioning explodes with n.
  const double eps = NumTraits<double>::epsilon();
  VERIFY(residual(8) <= 5e5 * eps);   // ~1e-10
  VERIFY(residual(12) <= 5e7 * eps);  // ~1e-8

  Vec x(5), y(5);
  for (Index i = 0; i < 5; ++i) {
    x[i] = double(i + 1);
    y[i] = -double(i);
  }
  Cauchy<double> H5(x, y);
  Mat dense = reference_cauchy<double>(x, y);
  VERIFY_IS_APPROX(H5.determinant(), dense.determinant());
}

// Clustered row nodes make leading minors nearly singular: partial pivoting must
// keep the factorization backward stable (residual check).
template <typename = void>
void test_cauchy_lu_pivoting(Index n) {
  typedef Matrix<double, Dynamic, 1> Vec;
  typedef Matrix<double, Dynamic, Dynamic> Mat;

  Vec x(n), y(n);
  for (Index i = 0; i < n; ++i) {
    x[i] = 2.0 + 1e-13 * double(i * i);  // tight cluster
    y[i] = double(i) / double(n);        // spread out
  }
  Cauchy<double> C(x, y);
  Mat dense = reference_cauchy<double>(x, y);
  CauchyLU<double> lu(C);
  VERIFY(lu.info() == Success);
  Vec b = Vec::Random(n);
  Vec u = lu.solve(b);
  const double tol = 5e7 * NumTraits<double>::epsilon();  // ~1e-8
  VERIFY((dense * u - b).norm() <= tol * (dense.norm() * u.norm() + b.norm()));
}

// A duplicated row node makes two rows identical, hence the matrix exactly
// singular; a zero pivot must survive partial pivoting and be reported.
template <typename = void>
void test_cauchy_lu_singular() {
  typedef Matrix<double, Dynamic, 1> Vec;
  Vec x(5), y(5);
  x << 2.0, 2.5, 2.0, 2.75, 2.25;  // x[2] duplicates x[0]
  y << 0.1, 0.3, 0.5, 0.7, 0.9;
  Cauchy<double> C(x, y);
  CauchyLU<double> lu(C);
  VERIFY(lu.info() == NumericalIssue);
}

// Exactly singular and non-finite pivots are reported through info().
template <typename Scalar>
void test_cauchy_lu_issues() {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;

  Vec xs(2), y(2);
  xs << Scalar(0), Scalar(0);
  y << Scalar(1), Scalar(-0.5);
  CauchyLU<Scalar> singularLu(Cauchy<Scalar>(xs, y));
  VERIFY(singularLu.info() == NumericalIssue);

  Vec xi(1), yi(1);
  xi << Scalar(0);
  yi << Scalar(-std::numeric_limits<RealScalar>::denorm_min());
  CauchyLU<Scalar> nonfiniteLu(Cauchy<Scalar>(xi, yi));
  VERIFY(nonfiniteLu.info() == NumericalIssue);
}

// y = -x gives the symmetric generalized Hilbert matrix 1/(x_i + x_j): verify
// the symmetry closure of the transpose through the node identity C^T = C(-y,-x).
template <typename Scalar>
void test_cauchy_symmetric(Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec x(n);
  for (Index i = 0; i < n; ++i) x[i] = Scalar(1) + Scalar(i);
  Vec negx = -x;
  Cauchy<Scalar> C(x, negx);
  Mat dense = reference_cauchy<Scalar>(x, negx);
  VERIFY_IS_APPROX(dense, Mat(dense.transpose()));
  Cauchy<Scalar> Ct = C.transpose();
  VERIFY_IS_EQUAL(Ct.rowNodes(), Vec(x));  // -(-x) round-trips exactly
  Mat Ctd = Ct;
  VERIFY_IS_APPROX(Ctd, dense);
}

// The closed form is a product of exact node differences (relative error
// O(n^2 eps)); the dense LU determinant it is compared against carries a
// cond(C)*eps error, so keep n small and the tolerance loose: the reference is
// the less accurate side of this comparison.
template <typename Scalar>
void test_cauchy_determinant(Index n) {
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec x, y;
  separated_nodes<Scalar>(n, n, x, y);
  Cauchy<Scalar> C(x, y);
  Mat dense = reference_cauchy<Scalar>(x, y);
  // The closed form is accurate to O(n^2 eps); the dense LU reference is the less
  // accurate side, its determinant carrying a relative error that grows like
  // cond(C)*eps (empirically ~0.1*cond*eps). Scale the bound by the SVD condition
  // number so the test is robust across random node draws.
  JacobiSVD<Mat> svd(dense);
  const RealScalar cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
  const RealScalar tol = RealScalar(100) * cond * NumTraits<RealScalar>::epsilon();
  VERIFY(numext::abs(C.determinant() - dense.determinant()) <= tol * numext::abs(dense.determinant()));
}

// The products carry the default product tag, so assignment materializes a
// temporary exactly like a dense product: x = C * x and x += C * x must see the
// pre-assignment right-hand side. Without the temporary, x = C * x would read a
// zeroed right-hand side and x += C * x would interleave destination writes
// with right-hand-side reads.
template <typename Scalar>
void test_cauchy_aliased_product(Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec x, y;
  separated_nodes<Scalar>(n, n, x, y);
  Cauchy<Scalar> C(x, y);
  Mat dense = reference_cauchy<Scalar>(x, y);

  Vec v = Vec::Random(n);
  Vec w = v;
  w = C * w;
  VERIFY_IS_APPROX(w, (dense * v).eval());

  w = v;
  w += C * w;
  VERIFY_IS_APPROX(w, (v + dense * v).eval());

  w = v;
  w -= C * w;
  VERIFY_IS_APPROX(w, (v - dense * v).eval());

  Mat V = Mat::Random(n, 3);
  Mat W = V;
  W = C * W;
  VERIFY_IS_APPROX(W, (dense * V).eval());
}

// Aliasing beyond the same-object case: the default-product temporary must also
// resolve right-hand-side expressions that reference the destination,
// overlapping views of one buffer, and rectangular self-assignment where the
// destination is resized by the assignment.
template <typename Scalar>
void test_cauchy_aliased_expression(Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  Vec xn, yn;
  separated_nodes<Scalar>(n, n, xn, yn);
  Cauchy<Scalar> C(xn, yn);
  Mat dense = reference_cauchy<Scalar>(xn, yn);

  // Right-hand-side expression referencing the destination.
  Vec v = Vec::Random(n), v0 = v;
  v = C * (v + Vec::Ones(n));
  VERIFY_IS_APPROX(v, (dense * (v0 + Vec::Ones(n))).eval());

  // Overlapping (shifted) segments of one buffer.
  Vec buf = Vec::Random(n + 1);
  Vec expected = dense * buf.tail(n);
  buf.head(n) = C * buf.tail(n);
  VERIFY_IS_APPROX(buf.head(n).eval(), expected);

  // Rectangular self-assignment: z = R * z resizes the destination from n to m,
  // so the product must be captured before the destination storage is touched.
  const Index m = n + 3;
  Vec xr, yr;
  separated_nodes<Scalar>(m, n, xr, yr);
  Cauchy<Scalar> R(xr, yr);
  Mat denseR = reference_cauchy<Scalar>(xr, yr);
  Vec z = Vec::Random(n), z0 = z;
  z = R * z;
  VERIFY_IS_EQUAL(z.size(), m);
  VERIFY_IS_APPROX(z, (denseR * z0).eval());
}

// transpose()/conjugate()/adjoint() return owning temporaries, so the product
// expression must nest the structured operand by value: a delayed-evaluated
// expression has to outlive the temporary operator it was built from. The static
// check pins the value nesting; the behavioral check would read freed memory if
// the product held a reference instead.
template <typename Scalar>
void test_cauchy_delayed_product(Index m, Index n) {
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, Dynamic, Dynamic> Mat;

  STATIC_CHECK(!std::is_reference<typename internal::ref_selector<Cauchy<Scalar>>::type>::value);

  Vec x, y;
  separated_nodes<Scalar>(m, n, x, y);
  Cauchy<Scalar> C(x, y);
  Mat dense = reference_cauchy<Scalar>(x, y);

  Vec w = Vec::Random(m);
  auto expr = C.adjoint() * w;        // the adjoint temporary dies with the full expression
  Vec scribble = Vec::Random(m + n);  // reuses the temporary's freed heap storage
  Vec u = expr;
  VERIFY_IS_APPROX(u, (dense.adjoint() * w).eval());
  VERIFY_IS_EQUAL(scribble.size(), m + n);  // keep the scribble alive across the evaluation
}

// Mixed-scalar products: a real operator applied to a complex right-hand side
// (and a complex operator applied to a real one) promotes to the complex product
// scalar, so alpha and the accumulation must run in the promoted type rather than
// the operator scalar.
template <typename RealScalar>
void test_cauchy_mixed_scalar(Index m, Index n) {
  typedef std::complex<RealScalar> Complex;
  typedef Matrix<RealScalar, Dynamic, 1> RVec;
  typedef Matrix<Complex, Dynamic, 1> CVec;
  typedef Matrix<Complex, Dynamic, Dynamic> CMat;

  RVec x, y;
  separated_nodes<RealScalar>(m, n, x, y);
  Cauchy<RealScalar> C(x, y);
  CMat dense = reference_cauchy<RealScalar>(x, y).template cast<Complex>();

  CVec v = CVec::Random(n);
  CVec w = C * v;
  VERIFY_IS_APPROX(w, (dense * v).eval());

  CVec w0 = CVec::Random(m);
  w = w0;
  w.noalias() += C * v;
  VERIFY_IS_APPROX(w, (w0 + dense * v).eval());

  CMat V = CMat::Random(n, 3);
  CMat W = CMat::Random(m, 3);
  const CMat W0 = W;
  W.noalias() -= C * V;
  VERIFY_IS_APPROX(W, (W0 - dense * V).eval());

  CVec xc, yc;
  separated_nodes<Complex>(m, n, xc, yc);
  Cauchy<Complex> Cc(xc, yc);
  CMat denseC = reference_cauchy<Complex>(xc, yc);
  RVec vr = RVec::Random(n);
  CVec z = Cc * vr;
  VERIFY_IS_APPROX(z, (denseC * vr).eval());
}

// Wide-dynamic-range determinants: the closed-form factors overflow or underflow
// individually while the determinant itself is representable, so the balanced
// m * 2^e accumulation must carry the exponent past the intermediate extremes.
template <typename = void>
void test_cauchy_determinant_range() {
  typedef Matrix<double, Dynamic, 1> Vec;
  typedef Matrix<double, Dynamic, Dynamic> Mat;

  // Reviewer repro: x = [2s, 3s], y = [0, s] with s = 1e160. The determinant is
  // -1/(12 s^2) ~ -8.35e-322, a subnormal, while the naive numerator product
  // -s^2 = -1e320 already overflows. Both the closed form and the dense LU
  // reference round once into the subnormal range (spacing denorm_min), so the
  // comparison needs an absolute term of a few subnormal spacings on top of a
  // relative term; the relative term alone would be far below denorm_min.
  {
    const double s = 1e160;
    Vec x(2), y(2);
    x << 2 * s, 3 * s;
    y << 0.0, s;
    Cauchy<double> C(x, y);
    Mat dense = reference_cauchy<double>(x, y);
    const double det = C.determinant();
    const double ref = dense.partialPivLu().determinant();
    const double kRelTol = 16 * NumTraits<double>::epsilon();
    const double kAbsTol = 4 * std::numeric_limits<double>::denorm_min();
    VERIFY((numext::isfinite)(det));
    VERIFY(numext::abs(det - ref) <= kRelTol * numext::abs(ref) + kAbsTol);
  }

  // Overflow-side analogue: x = [2s, 3s, 4s], y = [0, s/2, s] with s = 1e-60.
  // The determinant is -1/(3780 s^3) ~ -2.65e176, huge but representable, while
  // the naive numerator product -s^6/2 = -5e-361 underflows to zero. The dense
  // LU reference is the less accurate side (its error carries the conditioning
  // of the matrix, measured ~64 eps here); the deterministic nodes make 1000 eps
  // comfortable headroom.
  {
    const double s = 1e-60;
    Vec x(3), y(3);
    x << 2 * s, 3 * s, 4 * s;
    y << 0.0, s / 2, s;
    Cauchy<double> C(x, y);
    Mat dense = reference_cauchy<double>(x, y);
    const double det = C.determinant();
    const double ref = dense.partialPivLu().determinant();
    const double kRelTol = 1000 * NumTraits<double>::epsilon();
    VERIFY((numext::isfinite)(det));
    VERIFY(numext::abs(det - ref) <= kRelTol * numext::abs(ref));
  }

  // Genuinely infinite determinant: x[0] == y[0] makes entry (0,0) infinite (a
  // zero denominator factor with a non-zero numerator); the signed limit as
  // x[0] -> y[0] from above is -inf, and division by the exact zero factor must
  // produce it rather than a balanced-away finite value.
  {
    Vec x(2), y(2);
    x << 1.0, 2.0;
    y << 1.0, 3.0;
    Cauchy<double> C(x, y);
    const double det = C.determinant();
    VERIFY((numext::isinf)(det));
    VERIFY(det < 0.0);
  }

  // Coincident row nodes: two identical rows, so the matrix is exactly singular
  // and the zero numerator factor must propagate to an exact zero.
  {
    Vec x(3), y(3);
    x << 2.0, 3.0, 2.0;  // x[2] duplicates x[0]
    y << 0.0, 0.5, 1.0;
    Cauchy<double> C(x, y);
    VERIFY_IS_EQUAL(C.determinant(), 0.0);
  }
}

// Genuine overflow: clustered tiny nodes push the determinant past the
// representable range while every factor stays finite; the accumulated
// exponent must saturate to a correctly signed infinity, in both signs.
template <typename = void>
void test_cauchy_determinant_saturation() {
  typedef Matrix<double, Dynamic, 1> Vec;
  const double t = std::ldexp(1.0, -537);
  {
    Vec x(2), y(2);
    x << 0.0, 3.0 * t;
    y << t, 2.0 * t;  // det = -3 / (4 t^2) ~ -1.6e323
    const double det = Cauchy<double>(x, y).determinant();
    VERIFY((numext::isinf)(det) && std::signbit(det));
  }
  {
    Vec x(2), y(2);
    x << 0.0, 3.0 * t;
    y << 2.0 * t, t;  // swapped y nodes: det = +3 / (4 t^2)
    const double det = Cauchy<double>(x, y).determinant();
    VERIFY((numext::isinf)(det) && !std::signbit(det));
  }
}

template <typename Scalar, int M, int N>
void test_cauchy_fixed() {
  typedef Matrix<Scalar, M, 1> XVec;
  typedef Matrix<Scalar, N, 1> YVec;
  typedef Matrix<Scalar, Dynamic, 1> Vec;
  typedef Matrix<Scalar, M, N> MatMN;

  Vec xd, yd;
  separated_nodes<Scalar>(M, N, xd, yd);
  XVec x = xd;
  YVec y = yd;
  typedef Cauchy<Scalar, M, N> CauchyType;
  CauchyType C(x, y);
  STATIC_CHECK((Cauchy<Scalar, M, N>::RowsAtCompileTime == M));
  STATIC_CHECK((Cauchy<Scalar, M, N>::ColsAtCompileTime == N));
  STATIC_CHECK((internal::remove_all_t<decltype(makeCauchy(x, y))>::ColsAtCompileTime == N));
  STATIC_CHECK((internal::remove_all_t<decltype(C.transpose())>::RowsAtCompileTime == N));
  STATIC_CHECK((internal::remove_all_t<decltype(C.transpose())>::ColsAtCompileTime == M));

  MatMN dense = C;
  VERIFY_IS_APPROX(dense, MatMN(reference_cauchy<Scalar>(xd, yd)));

  CauchyType* heapC = new CauchyType(x, y);
  VERIFY(std::uintptr_t(heapC) % std::alignment_of<CauchyType>::value == 0);
  MatMN heapDense = *heapC;
  delete heapC;
  VERIFY_IS_APPROX(heapDense, dense);

#if EIGEN_MAX_ALIGN_BYTES > 0 && !EIGEN_HAS_CXX17_OVERALIGN
  void* raw = (CauchyType::operator new)(sizeof(CauchyType));
  VERIFY(std::uintptr_t(raw) % std::alignment_of<CauchyType>::value == 0);
  (CauchyType::operator delete)(raw);
#endif

  YVec v = YVec::Random();
  Matrix<Scalar, M, 1> w = C * v;
  VERIFY_IS_APPROX(w, (dense * v).eval());

  // .noalias() keeps the direct (temporary-free) path of the default product tag;
  // the matching fixed dimensions also pin the compile-time product check.
  Matrix<Scalar, M, 1> w2;
  w2.noalias() = C * v;
  VERIFY_IS_APPROX(w2, (dense * v).eval());

  // Several right-hand sides take the fixed-size reciprocal workspace.
  const Matrix<Scalar, N, 3> V = Matrix<Scalar, N, 3>::Random();
  Matrix<Scalar, M, 3> W;
  W.noalias() = C * V;
  VERIFY_IS_APPROX(W, (dense * V).eval());
}

EIGEN_DECLARE_TEST(structured_cauchy) {
  for (int i = 0; i < g_repeat; ++i) {
    // Products, dense assignment, coefficient access.
    CALL_SUBTEST_1((test_cauchy_product<double>(1, 1)));
    CALL_SUBTEST_1((test_cauchy_product<double>(8, 8)));
    CALL_SUBTEST_1((test_cauchy_product<double>(20, 12)));  // tall
    CALL_SUBTEST_1((test_cauchy_product<double>(12, 20)));  // wide
    CALL_SUBTEST_1((test_cauchy_product<float>(10, 10)));
    CALL_SUBTEST_1((test_cauchy_product<std::complex<double>>(9, 7)));
    CALL_SUBTEST_1((test_cauchy_product<std::complex<float>>(7, 9)));
    CALL_SUBTEST_1(test_cauchy_product_row_blocks<double>());
    CALL_SUBTEST_1(test_cauchy_product_row_blocks<std::complex<double>>());
    CALL_SUBTEST_1((test_cauchy_transpose<double>(10, 14)));
    CALL_SUBTEST_1((test_cauchy_transpose<std::complex<double>>(8, 6)));

    // GKO pivoted LU solves.
    CALL_SUBTEST_2((test_cauchy_lu<double>(1)));
    CALL_SUBTEST_2((test_cauchy_lu<double>(2)));
    CALL_SUBTEST_2((test_cauchy_lu<double>(12)));
    CALL_SUBTEST_2((test_cauchy_lu<double>(30)));
    CALL_SUBTEST_2((test_cauchy_lu<std::complex<double>>(16)));
    CALL_SUBTEST_2((test_cauchy_lu<float>(10)));
    CALL_SUBTEST_2(test_cauchy_hilbert<>());
    CALL_SUBTEST_2(test_cauchy_lu_pivoting<>(20));
    CALL_SUBTEST_2(test_cauchy_lu_singular<>());
    CALL_SUBTEST_2((test_cauchy_lu_issues<double>()));
    CALL_SUBTEST_2((test_cauchy_lu_issues<std::complex<double>>()));

    // Closed-form determinant, symmetric generalized Hilbert, fixed sizes.
    CALL_SUBTEST_3((test_cauchy_determinant<double>(4)));
    CALL_SUBTEST_3((test_cauchy_determinant<std::complex<double>>(4)));
    CALL_SUBTEST_3((test_cauchy_symmetric<double>(9)));
    CALL_SUBTEST_3((test_cauchy_fixed<double, 6, 4>()));
    CALL_SUBTEST_3((test_cauchy_fixed<std::complex<float>, 4, 5>()));

    // Numerical and lifetime boundaries: aliased and value-nested (owning)
    // delayed products, mixed-scalar products, wide-dynamic-range determinants.
    CALL_SUBTEST_4((test_cauchy_aliased_product<double>(11)));
    CALL_SUBTEST_4((test_cauchy_aliased_product<std::complex<double>>(8)));
    CALL_SUBTEST_4((test_cauchy_aliased_expression<double>(11)));
    CALL_SUBTEST_4((test_cauchy_aliased_expression<std::complex<double>>(8)));
    CALL_SUBTEST_4((test_cauchy_delayed_product<double>(12, 9)));
    CALL_SUBTEST_4((test_cauchy_delayed_product<std::complex<double>>(7, 10)));
    CALL_SUBTEST_4((test_cauchy_mixed_scalar<double>(10, 13)));
    CALL_SUBTEST_4((test_cauchy_mixed_scalar<float>(9, 6)));
    CALL_SUBTEST_4(test_cauchy_determinant_range<>());
    CALL_SUBTEST_4(test_cauchy_determinant_saturation<>());
  }
}
