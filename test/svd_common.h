// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef SVD_DEFAULT
#error a macro SVD_DEFAULT(MatrixType) must be defined prior to including svd_common.h
#endif

#ifndef SVD_FOR_MIN_NORM
#error a macro SVD_FOR_MIN_NORM(MatrixType) must be defined prior to including svd_common.h
#endif

#ifndef SVD_STATIC_OPTIONS
#error a macro SVD_STATIC_OPTIONS(MatrixType, Options) must be defined prior to including svd_common.h
#endif

#include "svd_fill.h"
#include "solverbase.h"

// U S V^H reconstructs m to the working precision plus the quantization of singular values stored in the subnormal
// range, checked 2^k above m so that FTZ/DAZ cannot flush the check itself:
//   |U (2^k S) V^H - 2^k m|_max <= 64 n eps |2^k m|_max + n 2^k denorm_min,
// the second term because each singular value was rounded onto the subnormal grid, and its column carries that error
// into at most n entries with unit weights. The scaling by 2^k goes through the representation and is exact.
template <typename SvdType, typename MatrixType>
void svd_check_scaled_residual(const MatrixType& m, const SvdType& svd, int k) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using RealVector = Matrix<RealScalar, Dynamic, 1>;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  const Index diagSize = (std::min)(m.rows(), m.cols());
  const RealScalar n = RealScalar((std::max)(m.rows(), m.cols()));
  const DenseMatrix scaledInput = m.unaryExpr(internal::scale_by_exponent_op<RealScalar>(k));
  const RealVector scaledSigma = svd.singularValues().unaryExpr(internal::scale_by_exponent_op<RealScalar>(k));
  const RealScalar granularity = numext::ldexp(
      RealScalar(1), std::numeric_limits<RealScalar>::min_exponent - std::numeric_limits<RealScalar>::digits + k);
  const DenseMatrix reconstruction =
      svd.matrixU().leftCols(diagSize) * scaledSigma.asDiagonal() * svd.matrixV().leftCols(diagSize).adjoint();
  const RealScalar residual = (reconstruction - scaledInput).cwiseAbs().template maxCoeff<PropagateNaN>();
  const RealScalar tolerance = RealScalar(64) * n * eps * scaledInput.cwiseAbs().maxCoeff() + n * granularity;
  VERIFY((numext::isfinite)(tolerance));
  VERIFY(residual <= tolerance);
}

// Check that the matrix m is properly reconstructed and that the U and V factors are unitary
// The SVD must have already been computed.
template <typename SvdType, typename MatrixType>
void svd_check_full(const MatrixType& m, const SvdType& svd) {
  Index rows = m.rows();
  Index cols = m.cols();

  enum { RowsAtCompileTime = MatrixType::RowsAtCompileTime, ColsAtCompileTime = MatrixType::ColsAtCompileTime };

  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> MatrixUType;
  typedef Matrix<Scalar, ColsAtCompileTime, ColsAtCompileTime> MatrixVType;

  MatrixType sigma = MatrixType::Zero(rows, cols);
  sigma.diagonal() = svd.singularValues().template cast<Scalar>();
  MatrixUType u = svd.matrixU();
  MatrixVType v = svd.matrixV();
  RealScalar scaling = m.cwiseAbs().maxCoeff();
  if (scaling < (std::numeric_limits<RealScalar>::min)()) {
    // A subnormal or zero matrix: 2^(digits + 2) brings every nonzero entry into the normal range.
    svd_check_scaled_residual(m, svd, std::numeric_limits<RealScalar>::digits + 2);
  } else {
    VERIFY_IS_APPROX(m / scaling, u * (sigma / scaling) * v.adjoint());
  }
  VERIFY_IS_UNITARY(u);
  VERIFY_IS_UNITARY(v);
}

template <typename MatrixType, typename SvdType, typename RhsType, typename SolutionType>
bool svd_check_normal_equation(const MatrixType& m, const SvdType& svd, const RhsType& rhs, const SolutionType& x) {
  using RealScalar = typename MatrixType::RealScalar;
  const RealScalar matrix_norm = m.stableNorm();
  const RealScalar rhs_norm = rhs.stableNorm();
  // Truncation gives A^H (A X - B) = -sum_{i >= rank} sigma_i v_i u_i^H B.
  const RealScalar truncated =
      svd.rank() < svd.singularValues().size() ? svd.singularValues()(svd.rank()) * rhs_norm : RealScalar(0);
  const typename SolutionType::PlainObject normal_lhs = m.adjoint() * (m * x);
  const typename SolutionType::PlainObject normal_rhs = m.adjoint() * rhs;
  const RealScalar normal_error = (normal_lhs - normal_rhs).stableNorm();
  // For C = A^H A and D = A^H B, perturbations bounded by eta*||A||^2 and eta*||A||*||B|| give
  // ||C X - D|| <= eta*||A||*(||A||*||X|| + ||B||). This includes cancellation in A*X.
  // Use eta = 8*(rows + cols)*eps as the backward-error budget for the decomposition,
  // solve, and checking products (including complex arithmetic).
  const RealScalar roundoff = 8 * RealScalar(m.rows() + m.cols()) * NumTraits<RealScalar>::epsilon();
  const RealScalar normal_tolerance = (roundoff * matrix_norm) * (matrix_norm * x.stableNorm() + rhs_norm) + truncated;
  return (numext::isfinite)(normal_tolerance) && normal_error <= normal_tolerance;
}

template <typename MatrixType, typename SvdType>
void svd_least_square(const MatrixType& m, SvdType& svd) {
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  Index rows = m.rows();
  Index cols = m.cols();

  enum { RowsAtCompileTime = MatrixType::RowsAtCompileTime, ColsAtCompileTime = MatrixType::ColsAtCompileTime };

  typedef Matrix<Scalar, RowsAtCompileTime, Dynamic> RhsType;
  typedef Matrix<Scalar, ColsAtCompileTime, Dynamic> SolutionType;

  RhsType rhs = RhsType::Random(rows, internal::random<Index>(1, cols));

  if (std::is_same<RealScalar, double>::value)
    svd.setThreshold(RealScalar(1e-8));
  else if (std::is_same<RealScalar, float>::value)
    svd.setThreshold(RealScalar(2e-4));

  SolutionType x = svd.solve(rhs);

  RealScalar residual = (m * x - rhs).norm();
  RealScalar rhs_norm = rhs.norm();
  if (!test_isMuchSmallerThan(residual, rhs.norm())) {
    // ^^^ If the residual is very small, then we have an exact solution, so we are already good.

    // evaluate normal equation which works also for least-squares solutions
    if (std::is_same<RealScalar, double>::value || svd.rank() == m.diagonal().size()) {
      if (std::is_same<RealScalar, float>::value) ++g_test_level;

      VERIFY(svd_check_normal_equation(m, svd, rhs, x));

      if (std::is_same<RealScalar, float>::value) --g_test_level;
    }

    // Check that there is no significantly better solution in the neighborhood of x
    for (Index k = 0; k < x.rows(); ++k) {
      using std::abs;

      SolutionType y(x);
      y.row(k) = (RealScalar(1) + 2 * NumTraits<RealScalar>::epsilon()) * x.row(k);
      RealScalar residual_y = (m * y - rhs).norm();
      VERIFY(test_isMuchSmallerThan(abs(residual_y - residual), rhs_norm) || residual < residual_y);
      if (std::is_same<RealScalar, float>::value) ++g_test_level;
      VERIFY(test_isApprox(residual_y, residual) || residual < residual_y);
      if (std::is_same<RealScalar, float>::value) --g_test_level;

      y.row(k) = (RealScalar(1) - 2 * NumTraits<RealScalar>::epsilon()) * x.row(k);
      residual_y = (m * y - rhs).norm();
      VERIFY(test_isMuchSmallerThan(abs(residual_y - residual), rhs_norm) || residual < residual_y);
      if (std::is_same<RealScalar, float>::value) ++g_test_level;
      VERIFY(test_isApprox(residual_y, residual) || residual < residual_y);
      if (std::is_same<RealScalar, float>::value) --g_test_level;
    }
  }
  svd.setThreshold(Default);
}

template <typename Scalar, int StorageOrder>
void svd_normal_equation_roundoff() {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, StorageOrder>;
  const RealScalar delta = std::is_same<RealScalar, float>::value ? RealScalar(1) / 32 : RealScalar(1) / 1048576;
  const RealScalar large_scale = numext::sqrt((std::numeric_limits<RealScalar>::max)()) * RealScalar(16);
  const RealScalar scales[] = {RealScalar(1), large_scale, RealScalar(1) / large_scale};
  MatrixType m(3, 2), rhs(3, 2), x(2, 2), exact(2, 2);
  rhs << 0, 0, 1, -2, 1, 3;
  // [3 7; delta -delta; 0 0] has x = [0.7; -0.3]/delta for b = [0; 1; 1].
  // Rounding x perturbs the cancelling first row by O(eps/delta).
  exact << RealScalar(0.7) / delta, -RealScalar(1.4) / delta, -RealScalar(0.3) / delta, RealScalar(0.6) / delta;
  Scalar phase(1);
  maybe_set_imag_part<Scalar>::run(phase, RealScalar(1));
  for (RealScalar scale : scales) {
    m << 3, 7, delta, -delta, 0, 0;
    m.col(0) *= phase;
    m *= scale;
    SVD_STATIC_OPTIONS(MatrixType, ComputeFullU | ComputeFullV) svd(m);
    VERIFY_IS_EQUAL(svd.info(), Success);
    VERIFY_IS_EQUAL(svd.rank(), 2);
    x = svd.solve(rhs);
    VERIFY(svd_check_normal_equation(m, svd, rhs, x));
    x = exact / scale;
    x.row(0) /= phase;
    VERIFY(svd_check_normal_equation(m, svd, rhs, x));
    x.setZero();
    VERIFY(!svd_check_normal_equation(m, svd, rhs, x));
  }

  // Sharpness: C = D = 1, delta_C = -eta, delta_D = eta gives
  // X = (1 + eta)/(1 - eta), |X - 1| = eta*(|X| + 1).
  // Here eta = 16*eps: the rounded X is accepted, but one more ULP is rejected.
  m = MatrixType::Ones(1, 1);
  rhs = MatrixType::Ones(1, 1);
  x.resize(1, 1);
  SVD_STATIC_OPTIONS(MatrixType, ComputeFullU | ComputeFullV) scalar_svd(m);
  x(0, 0) = Scalar(RealScalar(1) + 32 * NumTraits<RealScalar>::epsilon());
  VERIFY(svd_check_normal_equation(m, scalar_svd, rhs, x));
  x(0, 0) = Scalar(RealScalar(1) + 33 * NumTraits<RealScalar>::epsilon());
  VERIFY(!svd_check_normal_equation(m, scalar_svd, rhs, x));

  // A discarded singular direction must contribute its truncation allowance.
  m.setZero(3, 2);
  rhs.resize(3, 2);
  m(0, 0) = Scalar(1);
  m(1, 1) = Scalar(numext::sqrt(NumTraits<RealScalar>::epsilon()));
  rhs << 1, -2, 1, -2, 1, 3;
  SVD_STATIC_OPTIONS(MatrixType, ComputeFullU | ComputeFullV) svd(m);
  svd.setThreshold(2 * numext::sqrt(NumTraits<RealScalar>::epsilon()));
  VERIFY_IS_EQUAL(svd.rank(), 1);
  x = svd.solve(rhs);
  VERIFY(svd_check_normal_equation(m, svd, rhs, x));
  x.setZero();
  VERIFY(!svd_check_normal_equation(m, svd, rhs, x));

  x.setConstant(Scalar(std::numeric_limits<RealScalar>::infinity()));
  VERIFY(!svd_check_normal_equation(m, svd, rhs, x));
  x.setConstant(Scalar(std::numeric_limits<RealScalar>::quiet_NaN()));
  VERIFY(!svd_check_normal_equation(m, svd, rhs, x));
}

// check minimal norm solutions, the input matrix m is only used to recover problem size
template <typename MatrixType, int Options>
void svd_min_norm(const MatrixType& m) {
  typedef typename MatrixType::Scalar Scalar;
  Index cols = m.cols();

  enum { ColsAtCompileTime = MatrixType::ColsAtCompileTime };

  typedef Matrix<Scalar, ColsAtCompileTime, Dynamic> SolutionType;

  // Generate a full-rank m x n problem with m<n. Keep fixed columns from the
  // caller, but do not encode generated row counts into the helper matrix types.
  // Otherwise fixed-column callers instantiate extra fully fixed SVD shapes.
  constexpr int RankAtCompileTime2 = Dynamic;
  constexpr int RowsAtCompileTime3 = Dynamic;
  typedef Matrix<Scalar, RankAtCompileTime2, ColsAtCompileTime> MatrixType2;
  typedef Matrix<Scalar, RankAtCompileTime2, 1> RhsType2;
  typedef Matrix<Scalar, ColsAtCompileTime, RankAtCompileTime2> MatrixType2T;
  Index rank = ColsAtCompileTime == Dynamic ? internal::random<Index>(1, cols) : Index(ColsAtCompileTime / 2 + 1);
  MatrixType2 m2(rank, cols);
  m2.setRandom();
  if (SVD_FOR_MIN_NORM(MatrixType2)(m2).setThreshold(test_precision<Scalar>()).rank() != rank) {
    // Ensure full row rank by making the leading square block diagonally dominant.
    for (Index i = 0; i < rank; ++i) m2(i, i) += Scalar(1);
  }

  RhsType2 rhs2 = RhsType2::Random(rank);
  // use QR to find a reference minimal norm solution
  HouseholderQR<MatrixType2T> qr(m2.adjoint());
  Matrix<Scalar, Dynamic, 1> tmp =
      qr.matrixQR().topLeftCorner(rank, rank).template triangularView<Upper>().adjoint().solve(rhs2);
  tmp.conservativeResize(cols);
  tmp.tail(cols - rank).setZero();
  SolutionType x21 = qr.householderQ() * tmp;
  // now check with SVD
  SVD_STATIC_OPTIONS(MatrixType2, Options) svd2(m2);
  SolutionType x22 = svd2.solve(rhs2);
  VERIFY_IS_APPROX(m2 * x21, rhs2);
  VERIFY_IS_APPROX(m2 * x22, rhs2);
  VERIFY_IS_APPROX(x21, x22);

  // Now check with a rank deficient matrix
  typedef Matrix<Scalar, RowsAtCompileTime3, ColsAtCompileTime> MatrixType3;
  typedef Matrix<Scalar, RowsAtCompileTime3, 1> RhsType3;
  Index rows3 =
      ColsAtCompileTime == Dynamic ? internal::random<Index>(rank + 1, 2 * cols) : Index(ColsAtCompileTime + 1);
  Matrix<Scalar, RowsAtCompileTime3, Dynamic> C = Matrix<Scalar, RowsAtCompileTime3, Dynamic>::Random(rows3, rank);
  MatrixType3 m3 = C * m2;
  RhsType3 rhs3 = C * rhs2;
  SVD_STATIC_OPTIONS(MatrixType3, Options) svd3(m3);
  SolutionType x3 = svd3.solve(rhs3);
  VERIFY_IS_APPROX(m3 * x3, rhs3);
  VERIFY_IS_APPROX(m3 * x21, rhs3);
  VERIFY_IS_APPROX(m2 * x3, rhs2);
  VERIFY_IS_APPROX(x21, x3);
}

template <typename MatrixType, typename SolverType>
void svd_test_solvers(const MatrixType& m, const SolverType& solver) {
  Index rows, cols, cols2;

  rows = m.rows();
  cols = m.cols();

  if (MatrixType::ColsAtCompileTime == Dynamic) {
    cols2 = internal::random<int>(2, EIGEN_TEST_MAX_SIZE);
  } else {
    cols2 = cols;
  }
  typedef Matrix<typename MatrixType::Scalar, MatrixType::ColsAtCompileTime, MatrixType::ColsAtCompileTime> CMatrixType;
  check_solverbase<CMatrixType, MatrixType>(m, solver, rows, cols, cols2);
}

template <typename MatrixType, int Options,
          bool HasQrPreconditioner = (Options & internal::QRPreconditionerBits) != NoQRPreconditioner>
struct svd_min_norm_if {
  static void run(const MatrixType& m) { svd_min_norm<MatrixType, Options>(m); }
};

template <typename MatrixType, int Options>
struct svd_min_norm_if<MatrixType, Options, false> {
  static void run(const MatrixType&) {}
};

template <typename MatrixType, int Options, typename SVDType,
          bool ComputesBothUnitaries =
              (Options & (ComputeThinU | ComputeFullU)) != 0 && (Options & (ComputeThinV | ComputeFullV)) != 0>
struct svd_solver_checks_if {
  static void run(const MatrixType&, SVDType&) {}
};

template <typename MatrixType, int Options, typename SVDType>
struct svd_solver_checks_if<MatrixType, Options, SVDType, true> {
  static void run(const MatrixType& m, SVDType& staticSvd) {
    svd_test_solvers(m, staticSvd);
    svd_least_square(m, staticSvd);
    // svd_min_norm generates non-square matrices so it can't be used with NoQRPreconditioner.
    svd_min_norm_if<MatrixType, Options>::run(m);
  }
};

// This function verifies we don't iterate infinitely on nan/inf values,
// and that info() returns InvalidInput.
template <typename MatrixType>
void svd_inf_nan() {
  SVD_STATIC_OPTIONS(MatrixType, ComputeFullU | ComputeFullV) svd;
  typedef typename MatrixType::Scalar Scalar;
  const Scalar some_inf = (std::numeric_limits<Scalar>::infinity)();
  VERIFY((numext::isinf)(some_inf));
  svd.compute(MatrixType::Constant(10, 10, some_inf));
  VERIFY(svd.info() == InvalidInput);

  Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
  VERIFY(nan != nan);
  svd.compute(MatrixType::Constant(10, 10, nan));
  VERIFY(svd.info() == InvalidInput);

  MatrixType m = MatrixType::Zero(10, 10);
  m(internal::random<int>(0, 9), internal::random<int>(0, 9)) = some_inf;
  svd.compute(m);
  VERIFY(svd.info() == InvalidInput);

  m = MatrixType::Zero(10, 10);
  m(internal::random<int>(0, 9), internal::random<int>(0, 9)) = nan;
  svd.compute(m);
  VERIFY(svd.info() == InvalidInput);

  // regression test for bug 791
  m.resize(3, 3);
  m << 0, 2 * NumTraits<Scalar>::epsilon(), 0.5, 0, -0.5, 0, nan, 0, 0;
  svd.compute(m);
  VERIFY(svd.info() == InvalidInput);

  Scalar min = (std::numeric_limits<Scalar>::min)();
  m.resize(4, 4);
  m << 1, 0, 0, 0, 0, 3, 1, min, 1, 0, 1, nan, 0, nan, nan, 0;
  svd.compute(m);
  VERIFY(svd.info() == InvalidInput);
}

template <typename Scalar>
struct svd_subnormal_entry {
  template <typename RealScalar>
  static Scalar run(RealScalar real, RealScalar) {
    return real;
  }
};

template <typename RealScalar>
struct svd_subnormal_entry<std::complex<RealScalar>> {
  static std::complex<RealScalar> run(RealScalar real, RealScalar imag) { return std::complex<RealScalar>(real, imag); }
};

// A rows-by-cols matrix whose entries are all subnormal: signed significands of digits - 2 bits times denorm_min,
// assembled without floating-point arithmetic so that FTZ/DAZ hardware cannot flush the fixture before the solver
// sees it.
template <typename MatrixType>
MatrixType svd_subnormal_fixture(Index rows, Index cols) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using Binary = internal::binary_floating_point_traits<RealScalar>;
  using Bits = typename Binary::Bits;
  const Bits top = Bits(1) << (std::numeric_limits<RealScalar>::digits - 3);
  const auto component = [&]() {
    const Bits sign = internal::random<bool>() ? Binary::kSignBit : Bits(0);
    return numext::bit_cast<RealScalar>(sign | (top + internal::random<Bits>(Bits(0), top - Bits(1))));
  };
  MatrixType m(rows, cols);
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i) m(i, j) = svd_subnormal_entry<Scalar>::run(component(), component());
  return m;
}

// svd decomposed the all-subnormal matrix m, scaled (computed beforehand, under gradual underflow) its power-of-two
// multiple ms = m * 2^k. Both are normalized by a power of two, so they run the same iteration on work matrices a
// power of two apart and their singular values agree bit for bit up to the rounding of svd's onto the subnormal grid:
//   |2^k sigma_i - sigma_i(ms)| <= 2^k denorm_min / 2.
// U and V then reconstruct m to the scaled residual bound.
template <typename SvdType, typename MatrixType>
void svd_check_flushed_subnormal(const SvdType& svd, const SvdType& scaled, const MatrixType& m, int k) {
  using RealScalar = typename MatrixType::RealScalar;
  using RealVector = Matrix<RealScalar, Dynamic, 1>;
  VERIFY_IS_EQUAL(svd.info(), Success);
  VERIFY_IS_EQUAL(scaled.info(), Success);
  VERIFY_IS_EQUAL(svd.nonzeroSingularValues(), scaled.nonzeroSingularValues());
  const RealVector sigmaUp = svd.singularValues().unaryExpr(internal::scale_by_exponent_op<RealScalar>(k));
  VERIFY(scaled.singularValues()(0) > RealScalar(0));
  const RealScalar halfGranularity = numext::ldexp(
      RealScalar(1), std::numeric_limits<RealScalar>::min_exponent - std::numeric_limits<RealScalar>::digits + k - 1);
  VERIFY((sigmaUp - scaled.singularValues()).cwiseAbs().template maxCoeff<PropagateNaN>() <= halfGranularity);
  VERIFY_IS_UNITARY(svd.matrixU());
  VERIFY_IS_UNITARY(svd.matrixV());
  svd_check_scaled_residual(m, svd, k);
}

// Regression test for bug 286: JacobiSVD loops indefinitely with some
// matrices containing denormal numbers.
template <typename>
void svd_underoverflow() {
#if defined __INTEL_COMPILER
// shut up warning #239: floating point underflow
#pragma warning push
#pragma warning disable 239
#endif
  Matrix2d M;
  M << -7.90884e-313, -4.94e-324, 0, 5.60844e-313;
  SVD_STATIC_OPTIONS(Matrix2d, ComputeFullU | ComputeFullV) svd;
  svd.compute(M);
  CALL_SUBTEST(svd_check_full(M, svd));

  // Check all 2x2 matrices made with the following coefficients:
  VectorXd value_set(9);
  value_set << 0, 1, -1, 5.60844e-313, -5.60844e-313, 4.94e-324, -4.94e-324, -4.94e-223, 4.94e-223;
  Array4i id(0, 0, 0, 0);
  int k = 0;
  do {
    M << value_set(id(0)), value_set(id(1)), value_set(id(2)), value_set(id(3));
    svd.compute(M);
    CALL_SUBTEST(svd_check_full(M, svd));

    id(k)++;
    if (id(k) >= value_set.size()) {
      while (k < 3 && id(k) >= value_set.size()) id(++k)++;
      id.head(k).setZero();
      k = 0;
    }

  } while ((id < int(value_set.size())).all());

#if defined __INTEL_COMPILER
#pragma warning pop
#endif

  // Check for overflow:
  Matrix3d M3;
  M3 << 4.4331978442502944e+307, -5.8585363752028680e+307, 6.4527017443412964e+307, 3.7841695601406358e+307,
      2.4331702789740617e+306, -3.5235707140272905e+307, -8.7190887618028355e+307, -7.3453213709232193e+307,
      -2.4367363684472105e+307;

  SVD_STATIC_OPTIONS(Matrix3d, ComputeFullU | ComputeFullV) svd3;
  svd3.compute(M3);  // just check we don't loop indefinitely
  CALL_SUBTEST(svd_check_full(M3, svd3));
}

template <typename>
void svd_preallocate() {
  Vector3f v(3.f, 2.f, 1.f);
  MatrixXf m = v.asDiagonal();

  internal::set_is_malloc_allowed(false);
  VERIFY_RAISES_ASSERT(VectorXf tmp(10);)
  SVD_DEFAULT(MatrixXf) svd;
  internal::set_is_malloc_allowed(true);
  svd.compute(m);
  VERIFY_IS_APPROX(svd.singularValues(), v);
  VERIFY_RAISES_ASSERT(svd.matrixU());
  VERIFY_RAISES_ASSERT(svd.matrixV());

  SVD_STATIC_OPTIONS(MatrixXf, ComputeFullU | ComputeFullV) svd2(3, 3);
  internal::set_is_malloc_allowed(false);
  svd2.compute(m);
  internal::set_is_malloc_allowed(true);
  VERIFY_IS_APPROX(svd2.singularValues(), v);
  VERIFY_IS_APPROX(svd2.matrixU(), Matrix3f::Identity());
  VERIFY_IS_APPROX(svd2.matrixV(), Matrix3f::Identity());
  internal::set_is_malloc_allowed(false);
  svd2.compute(m);
  internal::set_is_malloc_allowed(true);

  MatrixXf tall = MatrixXf::Random(4, 3);
  MatrixXf wide = MatrixXf::Random(3, 4);
  SVD_STATIC_OPTIONS(MatrixXf, ComputeThinU | ComputeThinV) tallSvd(4, 3);
  SVD_STATIC_OPTIONS(MatrixXf, ComputeThinU | ComputeThinV) wideSvd(3, 4);
  internal::set_is_malloc_allowed(false);
  tallSvd.compute(tall);
  wideSvd.compute(wide);
  internal::set_is_malloc_allowed(true);
  VERIFY_IS_APPROX(tall, tallSvd.matrixU() * tallSvd.singularValues().asDiagonal() * tallSvd.matrixV().adjoint());
  VERIFY_IS_APPROX(wide, wideSvd.matrixU() * wideSvd.singularValues().asDiagonal() * wideSvd.matrixV().adjoint());
}

template <typename MatrixType, int QRPreconditioner = 0>
void svd_verify_assert_full_only(const MatrixType& m) {
  enum { RowsAtCompileTime = MatrixType::RowsAtCompileTime };

  typedef Matrix<typename MatrixType::Scalar, RowsAtCompileTime, 1> RhsType;
  RhsType rhs = RhsType::Zero(m.rows());
  EIGEN_UNUSED_VARIABLE(rhs);  // Only used if asserts are enabled.

  SVD_STATIC_OPTIONS(MatrixType, QRPreconditioner) svd0;
  VERIFY_RAISES_ASSERT((svd0.matrixU()));
  VERIFY_RAISES_ASSERT((svd0.singularValues()));
  VERIFY_RAISES_ASSERT((svd0.matrixV()));
  VERIFY_RAISES_ASSERT((svd0.solve(rhs)));
  VERIFY_RAISES_ASSERT((svd0.transpose().solve(rhs)));
  VERIFY_RAISES_ASSERT((svd0.adjoint().solve(rhs)));

  SVD_STATIC_OPTIONS(MatrixType, QRPreconditioner) svd1(m);
  VERIFY_RAISES_ASSERT((svd1.matrixU()));
  VERIFY_RAISES_ASSERT((svd1.matrixV()));
  VERIFY_RAISES_ASSERT((svd1.solve(rhs)));

  SVD_STATIC_OPTIONS(MatrixType, QRPreconditioner | ComputeFullU) svdFullU(m);
  VERIFY_RAISES_ASSERT((svdFullU.matrixV()));
  VERIFY_RAISES_ASSERT((svdFullU.solve(rhs)));
  SVD_STATIC_OPTIONS(MatrixType, QRPreconditioner | ComputeFullV) svdFullV(m);
  VERIFY_RAISES_ASSERT((svdFullV.matrixU()));
  VERIFY_RAISES_ASSERT((svdFullV.solve(rhs)));
}

template <typename MatrixType, int QRPreconditioner = 0>
void svd_verify_assert(const MatrixType& m) {
  enum { RowsAtCompileTime = MatrixType::RowsAtCompileTime };
  typedef Matrix<typename MatrixType::Scalar, RowsAtCompileTime, 1> RhsType;
  RhsType rhs = RhsType::Zero(m.rows());
  EIGEN_UNUSED_VARIABLE(rhs);  // Only used if asserts are enabled.

  SVD_STATIC_OPTIONS(MatrixType, QRPreconditioner | ComputeThinU) svdThinU(m);
  VERIFY_RAISES_ASSERT((svdThinU.matrixV()));
  VERIFY_RAISES_ASSERT((svdThinU.solve(rhs)));
  SVD_STATIC_OPTIONS(MatrixType, QRPreconditioner | ComputeThinV) svdThinV(m);
  VERIFY_RAISES_ASSERT((svdThinV.matrixU()));
  VERIFY_RAISES_ASSERT((svdThinV.solve(rhs)));

  svd_verify_assert_full_only<MatrixType, QRPreconditioner>(m);
}

template <typename MatrixType, int Options>
void svd_compute_checks(const MatrixType& m) {
  typedef SVD_STATIC_OPTIONS(MatrixType, Options) SVDType;

  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    DiagAtCompileTime = internal::min_size_prefer_dynamic(RowsAtCompileTime, ColsAtCompileTime),
    MatrixURowsAtCompileTime = SVDType::MatrixUType::RowsAtCompileTime,
    MatrixUColsAtCompileTime = SVDType::MatrixUType::ColsAtCompileTime,
    MatrixVRowsAtCompileTime = SVDType::MatrixVType::RowsAtCompileTime,
    MatrixVColsAtCompileTime = SVDType::MatrixVType::ColsAtCompileTime
  };

  SVDType staticSvd(m);

  VERIFY(MatrixURowsAtCompileTime == RowsAtCompileTime);
  VERIFY(MatrixVRowsAtCompileTime == ColsAtCompileTime);
  if (Options & ComputeThinU) VERIFY(MatrixUColsAtCompileTime == DiagAtCompileTime);
  if (Options & ComputeFullU) VERIFY(MatrixUColsAtCompileTime == RowsAtCompileTime);
  if (Options & ComputeThinV) VERIFY(MatrixVColsAtCompileTime == DiagAtCompileTime);
  if (Options & ComputeFullV) VERIFY(MatrixVColsAtCompileTime == ColsAtCompileTime);

  if (Options & (ComputeThinU | ComputeFullU))
    VERIFY(staticSvd.computeU());
  else
    VERIFY(!staticSvd.computeU());
  if (Options & (ComputeThinV | ComputeFullV))
    VERIFY(staticSvd.computeV());
  else
    VERIFY(!staticSvd.computeV());

  if (staticSvd.computeU()) VERIFY(staticSvd.matrixU().isUnitary());
  if (staticSvd.computeV()) VERIFY(staticSvd.matrixV().isUnitary());

  svd_solver_checks_if<MatrixType, Options, SVDType>::run(m, staticSvd);
}

template <typename MatrixType, int QRPreconditioner = 0>
void svd_full_option_checks(const MatrixType& m) {
  svd_compute_checks<MatrixType, QRPreconditioner | ComputeFullU>(m);
  svd_compute_checks<MatrixType, QRPreconditioner | ComputeFullV>(m);
  svd_compute_checks<MatrixType, QRPreconditioner | ComputeFullU | ComputeFullV>(m);

  SVD_STATIC_OPTIONS(MatrixType, QRPreconditioner | ComputeFullU | ComputeFullV) fullSvd(m);
  svd_check_full(m, fullSvd);
}

template <typename MatrixType, int QRPreconditioner = 0>
void svd_thin_full_option_checks(const MatrixType& input) {
  MatrixType m(input.rows(), input.cols());
  svd_fill_random(m);

  svd_verify_assert<MatrixType, QRPreconditioner>(m);

  svd_compute_checks<MatrixType, QRPreconditioner>(m);
  svd_compute_checks<MatrixType, QRPreconditioner | ComputeThinU>(m);
  svd_compute_checks<MatrixType, QRPreconditioner | ComputeThinV>(m);
  svd_compute_checks<MatrixType, QRPreconditioner | ComputeThinU | ComputeThinV>(m);

  svd_compute_checks<MatrixType, QRPreconditioner | ComputeThinU | ComputeFullV>(m);
  svd_compute_checks<MatrixType, QRPreconditioner | ComputeFullU | ComputeThinV>(m);

  svd_full_option_checks<MatrixType, QRPreconditioner>(m);
}

template <typename MatrixType, int QRPreconditioner = 0>
void svd_option_checks_full_only(const MatrixType& input) {
  MatrixType m(input.rows(), input.cols());
  svd_fill_random(m);
  svd_verify_assert_full_only<MatrixType, QRPreconditioner>(m);
  svd_full_option_checks<MatrixType, QRPreconditioner>(m);
}

template <typename MatrixType, int QRPreconditioner = 0>
void svd_check_max_size_matrix(int initialRows, int initialCols) {
  enum {
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
  };

  int rows = MaxRowsAtCompileTime == Dynamic ? initialRows : (std::min)(initialRows, (int)MaxRowsAtCompileTime);
  int cols = MaxColsAtCompileTime == Dynamic ? initialCols : (std::min)(initialCols, (int)MaxColsAtCompileTime);

  MatrixType m(rows, cols);
  svd_fill_random(m);
  SVD_STATIC_OPTIONS(MatrixType, QRPreconditioner | ComputeThinU | ComputeThinV) thinSvd(m);
  SVD_STATIC_OPTIONS(MatrixType, QRPreconditioner | ComputeThinU | ComputeFullV) mixedSvd1(m);
  SVD_STATIC_OPTIONS(MatrixType, QRPreconditioner | ComputeFullU | ComputeThinV) mixedSvd2(m);
  SVD_STATIC_OPTIONS(MatrixType, QRPreconditioner | ComputeFullU | ComputeFullV) fullSvd(m);

  MatrixType n(MaxRowsAtCompileTime, MaxColsAtCompileTime);
  svd_fill_random(n);
  thinSvd.compute(n);
  mixedSvd1.compute(n);
  mixedSvd2.compute(n);
  fullSvd.compute(n);

  MatrixX<typename MatrixType::Scalar> dynamicMatrix(MaxRowsAtCompileTime + 1, MaxColsAtCompileTime + 1);

  VERIFY_RAISES_ASSERT(thinSvd.compute(dynamicMatrix));
  VERIFY_RAISES_ASSERT(mixedSvd1.compute(dynamicMatrix));
  VERIFY_RAISES_ASSERT(mixedSvd2.compute(dynamicMatrix));
  VERIFY_RAISES_ASSERT(fullSvd.compute(dynamicMatrix));
}

#undef SVD_DEFAULT
#undef SVD_FOR_MIN_NORM
#undef SVD_STATIC_OPTIONS
