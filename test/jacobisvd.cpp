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

// discard stack allocation as that too bypasses malloc
#define EIGEN_STACK_ALLOCATION_LIMIT 0
#define EIGEN_RUNTIME_NO_MALLOC
#include "main.h"
#include <Eigen/SVD>
#include <cfenv>
#include <random>
#include "fp_control.h"

#define SVD_DEFAULT(M) JacobiSVD<M>
#define SVD_FOR_MIN_NORM(M) JacobiSVD<M, ColPivHouseholderQRPreconditioner>
#define SVD_STATIC_OPTIONS(M, O) JacobiSVD<M, O>
#include "svd_common.h"

template <typename MatrixType>
void jacobisvd_method() {
  enum { Size = MatrixType::RowsAtCompileTime };
  using RealScalar = typename MatrixType::RealScalar;
  using RealVecType = Matrix<RealScalar, Size, 1>;
  MatrixType m = MatrixType::Identity();
  VERIFY_IS_APPROX(m.jacobiSvd().singularValues(), RealVecType::Ones());
  VERIFY_RAISES_ASSERT(m.jacobiSvd().matrixU());
  VERIFY_RAISES_ASSERT(m.jacobiSvd().matrixV());
  VERIFY_IS_APPROX(m.template jacobiSvd<ComputeFullU | ComputeFullV>().solve(m), m);
  VERIFY_IS_APPROX(m.template jacobiSvd<ComputeFullU | ComputeFullV>().transpose().solve(m), m);
  VERIFY_IS_APPROX(m.template jacobiSvd<ComputeFullU | ComputeFullV>().adjoint().solve(m), m);
}

template <typename MatrixType>
void jacobisvd_thin_full_options_colpiv(const MatrixType& input) {
  svd_thin_full_option_checks<MatrixType, 0>(input);
  svd_thin_full_option_checks<MatrixType, PreconditionSquareMatrix>(input);
}

template <typename MatrixType>
void jacobisvd_thin_full_options_householder(const MatrixType& input) {
  svd_thin_full_option_checks<MatrixType, HouseholderQRPreconditioner>(input);
  svd_thin_full_option_checks<MatrixType, HouseholderQRPreconditioner | PreconditionSquareMatrix>(input);
}

template <typename MatrixType>
void jacobisvd_thin_full_options_fullpiv(const MatrixType& input) {
  // FullPiv only used when computing full unitaries
  svd_option_checks_full_only<MatrixType, FullPivHouseholderQRPreconditioner>(input);
  svd_option_checks_full_only<MatrixType, FullPivHouseholderQRPreconditioner | PreconditionSquareMatrix>(input);
}

template <typename MatrixType>
void jacobisvd_thin_full_options(const MatrixType& input = MatrixType()) {
  jacobisvd_thin_full_options_colpiv(input);
  jacobisvd_thin_full_options_householder(input);
  jacobisvd_thin_full_options_fullpiv(input);
}

template <typename MatrixType>
void jacobisvd_vector_asserts(const MatrixType& input = MatrixType()) {
  MatrixType m(input.rows(), input.cols());
  svd_fill_random(m);

  svd_verify_assert<MatrixType>(m);
  svd_verify_assert<MatrixType, HouseholderQRPreconditioner>(m);
  svd_verify_assert_full_only<MatrixType, FullPivHouseholderQRPreconditioner>(m);
}

template <typename MatrixType>
void jacobisvd_verify_inputs(const MatrixType& input = MatrixType()) {
  // check defaults
  using DefaultSVD = JacobiSVD<MatrixType>;
  MatrixType m(input.rows(), input.cols());
  svd_fill_random(m);
  DefaultSVD defaultSvd(m);
  VERIFY((int)DefaultSVD::QRPreconditioner == (int)ColPivHouseholderQRPreconditioner);
  VERIFY(!defaultSvd.computeU());
  VERIFY(!defaultSvd.computeV());

  // ColPivHouseholderQR is always default in presence of other options.
  VERIFY(((int)JacobiSVD<MatrixType, ComputeThinU>::QRPreconditioner == (int)ColPivHouseholderQRPreconditioner));
  VERIFY(((int)JacobiSVD<MatrixType, ComputeThinV>::QRPreconditioner == (int)ColPivHouseholderQRPreconditioner));
  VERIFY(((int)JacobiSVD<MatrixType, ComputeThinU | ComputeThinV>::QRPreconditioner ==
          (int)ColPivHouseholderQRPreconditioner));
  VERIFY(((int)JacobiSVD<MatrixType, ComputeFullU | ComputeFullV>::QRPreconditioner ==
          (int)ColPivHouseholderQRPreconditioner));
  VERIFY(((int)JacobiSVD<MatrixType, ComputeThinU | ComputeFullV>::QRPreconditioner ==
          (int)ColPivHouseholderQRPreconditioner));
  VERIFY(((int)JacobiSVD<MatrixType, ComputeFullU | ComputeThinV>::QRPreconditioner ==
          (int)ColPivHouseholderQRPreconditioner));
}

template <typename MatrixType>
void svd_triangular_matrix(const MatrixType& input = MatrixType()) {
  MatrixType matrix(input.rows(), input.cols());
  svd_fill_random(matrix);
  // Make sure that we only consider the 'Lower' part of the matrix.
  MatrixType matrix_self_adj = matrix.template selfadjointView<Lower>().toDenseMatrix();

  JacobiSVD<MatrixType, ComputeFullV> svd_triangular(matrix.template selfadjointView<Lower>());
  JacobiSVD<MatrixType, ComputeFullV> svd_full(matrix_self_adj);

  VERIFY_IS_APPROX(svd_triangular.singularValues(), svd_full.singularValues());
}

namespace Foo {
class Bar {
 public:
  Bar() {}
};
bool operator<(const Bar&, const Bar&) { return true; }
}  // namespace Foo
// regression test for a very strange MSVC issue for which simply
// including SVDBase.h messes up with std::max and custom scalar type
template <typename = void>
void msvc_workaround() {
  const Foo::Bar a;
  const Foo::Bar b;
  const Foo::Bar c = std::max EIGEN_NOT_A_MACRO(a, b);
  EIGEN_UNUSED_VARIABLE(c);
}

template <typename = void>
void jacobisvd_mixed_option_enum_regression() {
  using NoQrFullSVD = JacobiSVD<MatrixXd, NoQRPreconditioner | ComputeFullU | ComputeFullV>;
  using ReversedMixedSVD = JacobiSVD<MatrixXd, ComputeThinU | HouseholderQRPreconditioner | ComputeFullV>;

  STATIC_CHECK((int(NoQrFullSVD::QRPreconditioner) == int(NoQRPreconditioner)));
  STATIC_CHECK(((int(NoQrFullSVD::Options) & ComputeFullU) != 0));
  STATIC_CHECK(((int(NoQrFullSVD::Options) & ComputeFullV) != 0));
  STATIC_CHECK(((int(NoQrFullSVD::Options) & ComputeThinU) == 0));
  STATIC_CHECK(((int(NoQrFullSVD::Options) & ComputeThinV) == 0));

  STATIC_CHECK((int(ReversedMixedSVD::QRPreconditioner) == int(HouseholderQRPreconditioner)));
  STATIC_CHECK(((int(ReversedMixedSVD::Options) & ComputeThinU) != 0));
  STATIC_CHECK(((int(ReversedMixedSVD::Options) & ComputeFullV) != 0));
  STATIC_CHECK(((int(ReversedMixedSVD::Options) & ComputeFullU) == 0));
  STATIC_CHECK(((int(ReversedMixedSVD::Options) & ComputeThinV) == 0));
}

EIGEN_DIAGNOSTICS(push)
EIGEN_DISABLE_DEPRECATED_WARNING
// Thin unitaries requested through the deprecated runtime options must be rejected with FullPiv, as they are
// statically through Options.
template <int Options>
void jacobisvd_fullpiv_runtime_thin_asserts(Index rows, Index cols) {
  using SVD = JacobiSVD<MatrixXd, FullPivHouseholderQRPreconditioner | Options>;
  const MatrixXd m = MatrixXd::Random(rows, cols);
  for (unsigned int thin : {unsigned(ComputeThinU), unsigned(ComputeThinV), unsigned(ComputeThinU | ComputeThinV)}) {
    VERIFY_RAISES_ASSERT(SVD(m, thin));
    VERIFY_RAISES_ASSERT(SVD(rows, cols, thin));
    SVD svd;
    VERIFY_RAISES_ASSERT(svd.compute(m, thin));
  }
  SVD full(m, ComputeFullU | ComputeFullV);
  svd_check_full(m, full);
}
EIGEN_DIAGNOSTICS(pop)

template <typename = void>
void jacobisvd_large_tau_regression() {
  Matrix3f m;
  m << 3.7855173218304116745e-07f, 0.0f, 500.0f, -4.9999995231628417969f, -0.0f, -1.9106853686029490191e-12f,
      -8.6602544784545898438f, 0.0f, 2.1855694285477511585f;

  JacobiSVD<Matrix3f, ComputeFullU | ComputeFullV> svd;
#ifdef FE_OVERFLOW
  const int overflow = fp_exceptions_raised_by(FE_OVERFLOW, [&] { svd.compute(m); });
  if (overflow >= 0) VERIFY_IS_EQUAL(overflow, 0);
#else
  std::cout << "SKIP: JacobiSVD overflow flag check: FE_OVERFLOW is unavailable.\n";
  svd.compute(m);
#endif
  svd_check_full(m, svd);
}

// An all-subnormal matrix, which a SIMD unit that flushes subnormal inputs (ARMv7 NEON, Arm FZ, DAZ) reads as zero in
// the maxCoeff() selecting the scale: recovering the maximum from the representation and scaling through integer
// significands keeps the decomposition, and the singular values come back through the subnormal range exactly.
template <typename MatrixType>
void jacobisvd_flushed_subnormal_matrix(Index rows, Index cols) {
  using RealScalar = typename MatrixType::RealScalar;
  const MatrixType m = svd_subnormal_fixture<MatrixType>(rows, cols);
  const int k = 60;
  const MatrixType ms = m.unaryExpr(internal::scale_by_exponent_op<RealScalar>(k));
  const JacobiSVD<MatrixType, ComputeFullU | ComputeFullV> scaled(ms);
  forEachFlushToZeroMode([&](FlushToZeroMode) {
    const JacobiSVD<MatrixType, ComputeFullU | ComputeFullV> svd(m);
    svd_check_flushed_subnormal(svd, scaled, m, k);
  });
}

// The Jacobi sweep leaves a diagonal matrix untouched, so with distinct magnitudes |d_i| the singular values are
// those magnitudes bit for bit, descending, U the permutation carrying the signs of d and V the permutation, in every
// flush-to-zero mode. Signed subnormal entries with and without a normal one reach the sort of a subnormal tail, the
// extraction of magnitude and sign, and the nonzero count in the subnormal range.
template <typename RealScalar>
void jacobisvd_subnormal_diagonal(const Matrix<RealScalar, Dynamic, 1>& diagonal) {
  using Binary = internal::binary_floating_point_traits<RealScalar>;
  using Bits = typename Binary::Bits;
  using MatrixType = Matrix<RealScalar, Dynamic, Dynamic>;
  const Index n = diagonal.size();
  const MatrixType m = diagonal.asDiagonal();
  std::vector<Bits> magnitudes;
  std::vector<Index> order;
  Index nonzero = 0;
  for (Index i = 0; i < n; ++i) {
    magnitudes.push_back(Binary::magnitude(diagonal(i)));
    order.push_back(i);
    if (magnitudes.back() != 0) ++nonzero;
  }
  std::stable_sort(order.begin(), order.end(), [&](Index a, Index b) { return magnitudes[a] > magnitudes[b]; });
  for (Index j = 1; j < nonzero; ++j) VERIFY(magnitudes[order[j - 1]] != magnitudes[order[j]]);

  const auto checkValues = [&](const auto& svd) {
    VERIFY_IS_EQUAL(svd.info(), Success);
    VERIFY_IS_EQUAL(svd.nonzeroSingularValues(), nonzero);
    for (Index j = 0; j < n; ++j) VERIFY_IS_EQUAL(Binary::bits(svd.singularValues()(j)), magnitudes[order[j]]);
  };
  forEachFlushToZeroMode([&](FlushToZeroMode) {
    const JacobiSVD<MatrixType> valuesOnly(m);
    checkValues(valuesOnly);
    const JacobiSVD<MatrixType, ComputeFullU | ComputeFullV> full(m);
    checkValues(full);
    VERIFY_IS_UNITARY(full.matrixU());
    VERIFY_IS_UNITARY(full.matrixV());
    for (Index j = 0; j < nonzero; ++j) {
      const Index p = order[j];
      const bool negative = (Binary::bits(diagonal(p)) & Binary::kSignBit) != 0;
      VERIFY_IS_EQUAL(full.matrixU().col(j).cwiseAbs().sum(), RealScalar(1));
      VERIFY_IS_EQUAL(full.matrixU()(p, j), negative ? RealScalar(-1) : RealScalar(1));
      VERIFY_IS_EQUAL(full.matrixV().col(j).cwiseAbs().sum(), RealScalar(1));
      VERIFY_IS_EQUAL(full.matrixV()(p, j), RealScalar(1));
    }
  });
}

template <typename RealScalar>
void jacobisvd_subnormal_diagonals() {
  using Binary = internal::binary_floating_point_traits<RealScalar>;
  using Bits = typename Binary::Bits;
  using VectorType = Matrix<RealScalar, Dynamic, 1>;
  constexpr int digits = std::numeric_limits<RealScalar>::digits;
  const auto subnormal = [](Bits significand, bool negative) {
    return numext::bit_cast<RealScalar>((negative ? Binary::kSignBit : Bits(0)) | significand);
  };
  const RealScalar denormMin = subnormal(Bits(1), false);
  const RealScalar small = subnormal(Bits(1) << (digits - 3), true);
  const RealScalar large = subnormal(Bits(1) << (digits - 2), false);
  const RealScalar top = subnormal(Binary::kFractionMask, true);
  // Signed subnormal entries only, with zeros and a signed zero in the tail.
  VectorType allSubnormal(7);
  allSubnormal << small, RealScalar(0), top, denormMin, large, subnormal(Bits(3), true), -RealScalar(0);
  jacobisvd_subnormal_diagonal(allSubnormal);
  // A normal entry in [1, 2): the identity scaling, and the subnormal tail sorted from the representation.
  VectorType normalLeading(6);
  normalLeading << RealScalar(1), RealScalar(0), small, large, RealScalar(-1.5), denormMin;
  jacobisvd_subnormal_diagonal(normalLeading);
  VectorType tail(2);
  tail << RealScalar(1.5), denormMin;
  jacobisvd_subnormal_diagonal(tail);
  // A normal entry below min / eps: the integer scaling into and out of the subnormal range.
  VectorType belowRecovery(5);
  belowRecovery << subnormal(Bits(5) << (digits - 6), false), RealScalar(0),
      -numext::ldexp(RealScalar(1), internal::safe_scaling<RealScalar>::subnormal_recovery_exponent() - 20), small, top;
  jacobisvd_subnormal_diagonal(belowRecovery);
}

template <typename = void>
void jacobisvd_power_of_two_scaling() {
  // Reciprocal scaling rounds the smaller singular value up by one ULP.
  Matrix2f matrix = Matrix2f::Zero();
  matrix(0, 0) = numext::bit_cast<float>(numext::uint32_t(0x58f6aaed));
  matrix(1, 1) = numext::bit_cast<float>(numext::uint32_t(0x537dcf0e));

  const JacobiSVD<Matrix2f> svd(matrix);
  VERIFY_IS_EQUAL(svd.singularValues()(0), matrix(0, 0));
  VERIFY_IS_EQUAL(svd.singularValues()(1), matrix(1, 1));

  Matrix<float, 3, 2> rectangular = Matrix<float, 3, 2>::Zero();
  rectangular.template topRows<2>() = matrix;
  const JacobiSVD<Matrix<float, 3, 2>> rectangularSvd(rectangular);
  VERIFY_IS_EQUAL(rectangularSvd.singularValues(), matrix.diagonal());

  const auto checkSubnormalScaling = [] {
    volatile float normalMinInput = (std::numeric_limits<float>::min)();
    const float inputScale = 256.0f * normalMinInput;
    Matrix2cf normalized = Matrix2cf::Identity();
    normalized(1, 0) = std::complex<float>(1.0f / 512.0f, -1.0f / 1024.0f);
    Matrix2cf tiny = Matrix2cf::Identity() * inputScale;
    // Narrowing an arithmetic product to float would flush the fixture before the solver sees it.
    tiny(1, 0) = std::complex<float>(numext::bit_cast<float>(numext::uint32_t(0x00400000)),
                                     numext::bit_cast<float>(numext::uint32_t(0x80200000)));
    const JacobiSVD<Matrix2cf> normalizedSvd(normalized);
    const JacobiSVD<Matrix2cf> tinySvd(tiny);
    // These well-conditioned, nearly identity problems should agree to a few rounding errors.
    const float tolerance = 8 * NumTraits<float>::epsilon();
    VERIFY((tinySvd.singularValues() / inputScale - normalizedSvd.singularValues()).norm() <= tolerance);

    Matrix<std::complex<float>, 3, 2> tinyTall = Matrix<std::complex<float>, 3, 2>::Zero();
    tinyTall.template topRows<2>() = tiny;
    const JacobiSVD<Matrix<std::complex<float>, 3, 2>> tinyTallSvd(tinyTall);
    VERIFY((tinyTallSvd.singularValues() / inputScale - normalizedSvd.singularValues()).norm() <= tolerance);

    const Matrix<std::complex<float>, 2, 3> tinyWide = tinyTall.adjoint();
    const JacobiSVD<Matrix<std::complex<float>, 2, 3>> tinyWideSvd(tinyWide);
    VERIFY((tinyWideSvd.singularValues() / inputScale - normalizedSvd.singularValues()).norm() <= tolerance);
  };
  forEachFlushToZeroMode([&](FlushToZeroMode) { checkSubnormalScaling(); });
}

// Backward error and unitarity of a full decomposition, each bounded by 64 n eps (times |A|_max for the residual). The
// maxima propagate NaN, so a NaN in a factor fails the bound.
template <typename MatrixType, int Options>
void jacobisvd_check_backward_error(const MatrixType& m) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  const Index n = (std::max)(m.rows(), m.cols());
  const RealScalar tolerance = RealScalar(64 * n) * NumTraits<RealScalar>::epsilon();
  const JacobiSVD<MatrixType, Options | ComputeFullU | ComputeFullV> svd(m);
  VERIFY_IS_EQUAL(svd.info(), Success);
  DenseMatrix sigma = DenseMatrix::Zero(m.rows(), m.cols());
  sigma.diagonal() = svd.singularValues().template cast<Scalar>();
  const RealScalar scale = m.cwiseAbs().maxCoeff();
  const RealScalar residual =
      (svd.matrixU() * sigma * svd.matrixV().adjoint() - m).cwiseAbs().template maxCoeff<PropagateNaN>();
  VERIFY_LE(residual, tolerance * scale);
  const DenseMatrix iu = DenseMatrix::Identity(m.rows(), m.rows());
  const DenseMatrix iv = DenseMatrix::Identity(m.cols(), m.cols());
  VERIFY_LE((svd.matrixU().adjoint() * svd.matrixU() - iu).cwiseAbs().template maxCoeff<PropagateNaN>(), tolerance);
  VERIFY_LE((svd.matrixV().adjoint() * svd.matrixV() - iv).cwiseAbs().template maxCoeff<PropagateNaN>(), tolerance);
}

// Runs the backward-error check on each matrix, under every flush-to-zero mode the host selects exactly, for the QR
// preconditioners that reach the complex pivot code: NoQR on the square matrices, and the others with the square
// preconditioner so that they also decompose them.
template <typename MatrixType>
void jacobisvd_check_backward_error_all(const std::vector<MatrixType>& matrices) {
  forEachFlushToZeroMode([&](FlushToZeroMode) {
    for (const MatrixType& m : matrices) {
      jacobisvd_check_backward_error<MatrixType, NoQRPreconditioner>(m);
      jacobisvd_check_backward_error<MatrixType, ColPivHouseholderQRPreconditioner | PreconditionSquareMatrix>(m);
      jacobisvd_check_backward_error<MatrixType, HouseholderQRPreconditioner | PreconditionSquareMatrix>(m);
      jacobisvd_check_backward_error<MatrixType, FullPivHouseholderQRPreconditioner | PreconditionSquareMatrix>(m);
    }
  });
}

// Pivot scales s: the normal decades from 1e-10 down, then subnormal ones, k denorm_min and min / 2^j. Computed here,
// in the default floating-point mode, so that a flush-to-zero mode does not flush the fixtures.
template <typename RealScalar>
std::vector<RealScalar> jacobisvd_pivot_scales() {
  std::vector<RealScalar> scales;
  const int exponent = std::numeric_limits<RealScalar>::min_exponent10;  // -37 (float), -307 (double)
  for (int e = 10; e < -exponent; ++e) scales.push_back(std::pow(RealScalar(10), -RealScalar(e)));
  const RealScalar denormMin = std::numeric_limits<RealScalar>::denorm_min();
  const RealScalar normalMin = (std::numeric_limits<RealScalar>::min)();
  for (const int k : {1, 3, 5, 1 << 10}) scales.push_back(RealScalar(k) * denormMin);
  for (const int j : {4, 8, 20}) scales.push_back(std::ldexp(normalMin, -j));
  return scales;
}

// The 2x2 pivot block of the complex preconditioning has the first column (w_pp, w_qp), and its norm
// n = |(w_pp, w_qp)| gives the rotation c = conj(w_pp) / n, s = w_qp / n, which is unitary only if n is accurate.
// With n = sqrt(|w_pp|^2 + |w_qp|^2), the squares are subnormal once |w| < sqrt(min) (about 1e-19 for float, 1e-154
// for double) and n loses its relative accuracy, so U and V stop being unitary. The pivot column is (m(0,1), m(1,1)),
// at the scales of jacobisvd_pivot_scales().
template <typename RealScalar>
void jacobisvd_complex_tiny_pivot() {
  using ComplexScalar = std::complex<RealScalar>;
  using MatrixType = Matrix<ComplexScalar, Dynamic, Dynamic>;
  std::vector<MatrixType> matrices;
  for (const RealScalar scale : jacobisvd_pivot_scales<RealScalar>()) {
    const RealScalar a = -scale;
    for (int variant = 0; variant < 6; ++variant) {
      MatrixType m = MatrixType::Zero(2, 2);
      m(0, 0) = RealScalar(-4.75);
      m(1, 0) = ComplexScalar(RealScalar(1.3), RealScalar(1.7));
      if (variant == 0) m(1, 1) = ComplexScalar(a);
      if (variant == 1) m(1, 1) = ComplexScalar(a, -a);
      if (variant == 2) {
        // Tiny entries in the first column, as the QR factor of a rank-deficient matrix has.
        m(0, 0) = ComplexScalar(a, RealScalar(2) * a);
        m(1, 0) = ComplexScalar(RealScalar(-3) * a, a);
        m(0, 1) = ComplexScalar(RealScalar(1.5), RealScalar(-0.5));
        m(1, 1) = ComplexScalar(RealScalar(0.25), RealScalar(2));
      }
      if (variant == 3) {
        m(1, 1) = ComplexScalar(a, a);
        m(0, 1) = ComplexScalar(a);
      }
      if (variant == 4) {
        m(1, 1) = ComplexScalar(RealScalar(3) * a, RealScalar(3) * a);
        m(0, 1) = ComplexScalar(0, RealScalar(5) * a);
      }
      if (variant == 5) {
        // The pivot column of the reproducer: [[1, s], [0.5 + 0.3i, s (1 + i)]].
        m(0, 0) = RealScalar(1);
        m(1, 0) = ComplexScalar(RealScalar(0.5), RealScalar(0.3));
        m(0, 1) = ComplexScalar(scale);
        m(1, 1) = ComplexScalar(scale, scale);
      }
      matrices.push_back(m);
    }
  }
  jacobisvd_check_backward_error_all(matrices);
}

// jacobi_unit_phase(w) = conj(w) / |w| must be unit-modulus to rounding and point along conj(w) for w just above the
// smallest normal number, in every flush-to-zero mode. There the term im + (re / im) re of Smith's complex division
// is subnormal for |re| < |im| and FTZ or DAZ drops it, which leaves |z| = |w|^2 / im^2, up to 2. The reference is
// formed from the entries scaled by a power of two, which is exact and leaves them normal and O(1).
template <typename RealScalar>
void jacobisvd_unit_phase_near_minimum() {
  using ComplexScalar = std::complex<RealScalar>;
  const RealScalar normalMin = (std::numeric_limits<RealScalar>::min)();
  const int shift = 2 - std::numeric_limits<RealScalar>::min_exponent;
  std::vector<ComplexScalar> values, references;
  // Multiples of min, including the ones where re^2 / im is subnormal, with both orderings and all four signs.
  for (const double a : {1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 16.0, 1024.0}) {
    for (const double b : {1.0, 1.5, 2.0, 5.0, 64.0}) {
      for (const int signs : {0, 1, 2, 3}) {
        const RealScalar re = RealScalar(a) * normalMin * ((signs & 1) != 0 ? RealScalar(-1) : RealScalar(1));
        const RealScalar im = RealScalar(b) * normalMin * ((signs & 2) != 0 ? RealScalar(-1) : RealScalar(1));
        values.push_back(ComplexScalar(re, im));
        const long double rs = std::ldexp(static_cast<long double>(re), shift);
        const long double is = std::ldexp(static_cast<long double>(im), shift);
        const long double n = std::sqrt(rs * rs + is * is);
        references.push_back(ComplexScalar(RealScalar(rs / n), RealScalar(-is / n)));
      }
    }
  }
  const RealScalar tolerance = RealScalar(8) * NumTraits<RealScalar>::epsilon();
  forEachFlushToZeroMode([&](FlushToZeroMode) {
    for (std::size_t k = 0; k < values.size(); ++k) {
      const ComplexScalar z = internal::jacobi_unit_phase(values[k]);
      VERIFY_LE(numext::abs(numext::abs2(z) - RealScalar(1)), tolerance);
      VERIFY_LE(numext::abs(numext::real(z) - numext::real(references[k])), tolerance);
      VERIFY_LE(numext::abs(numext::imag(z) - numext::imag(references[k])), tolerance);
    }
  });
}

// A square matrix whose diagonal and strict upper triangle, the pivot columns of the sweeps, are scale times
// a random complex matrix, and whose strict lower triangle is of order 1: the first column of every 2x2 pivot block
// starts at the pivot scale. It reaches the non-blocking sweep for small n and the blocked sweep for large n.
template <typename RealScalar>
void jacobisvd_complex_tiny_pivot_sweeps(Index n) {
  using ComplexScalar = std::complex<RealScalar>;
  using MatrixType = Matrix<ComplexScalar, Dynamic, Dynamic>;
  std::mt19937 gen(777u + unsigned(n));
  std::normal_distribution<double> dist;
  std::vector<MatrixType> matrices;
  for (const RealScalar scale : jacobisvd_pivot_scales<RealScalar>()) {
    MatrixType m(n, n);
    for (Index j = 0; j < n; ++j) {
      for (Index i = 0; i < n; ++i) {
        const RealScalar re = RealScalar(dist(gen));
        const ComplexScalar value(re, RealScalar(dist(gen)));
        m(i, j) = i <= j ? value * scale : value;
      }
    }
    matrices.push_back(m);
  }
  jacobisvd_check_backward_error_all(matrices);
}

// A rank-one matrix u 1^T: the QR factor R has a diagonal falling like eps^k, so the pivot blocks the sweeps see
// pass through the subnormal-square range, as in the 4x20 float example below.
template <typename RealScalar, int Options>
void jacobisvd_rank_deficient_family(Index rows, Index cols) {
  using Scalar = std::complex<RealScalar>;
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  std::mt19937 gen(12345u + unsigned(rows * 131 + cols));
  std::normal_distribution<double> dist;
  for (int seed = 0; seed < 100; ++seed) {
    MatrixType u(rows, 1);
    for (Index i = 0; i < rows; ++i) {
      const RealScalar re = RealScalar(dist(gen));
      u(i) = Scalar(re, RealScalar(dist(gen)));
    }
    const MatrixType m = u * MatrixType::Ones(1, cols);
    jacobisvd_check_backward_error<MatrixType, Options>(m);
  }
}

EIGEN_DECLARE_TEST(jacobisvd) {
  CALL_SUBTEST_60((svd_normal_equation_roundoff<float, ColMajor>()));
  CALL_SUBTEST_60((svd_normal_equation_roundoff<double, RowMajor>()));
  CALL_SUBTEST_61((svd_normal_equation_roundoff<std::complex<float>, RowMajor>()));
  CALL_SUBTEST_61((svd_normal_equation_roundoff<std::complex<double>, ColMajor>()));

  CALL_SUBTEST_1((jacobisvd_verify_inputs<Matrix4d>()));
  CALL_SUBTEST_2((jacobisvd_verify_inputs(Matrix<float, 5, Dynamic>(5, 6))));
  CALL_SUBTEST_3((jacobisvd_verify_inputs<Matrix<std::complex<double>, 7, 5>>()));
  CALL_SUBTEST_4((jacobisvd_mixed_option_enum_regression<>()));
  CALL_SUBTEST_4((jacobisvd_large_tau_regression<>()));

  CALL_SUBTEST_11((jacobisvd_thin_full_options<Matrix2cd>()));
  CALL_SUBTEST_12((jacobisvd_thin_full_options<Matrix2d>()));

  for (int i = 0; i < g_repeat; i++) {
    int r = internal::random<int>(1, 30), c = internal::random<int>(1, 30);

    TEST_SET_BUT_UNUSED_VARIABLE(r);
    TEST_SET_BUT_UNUSED_VARIABLE(c);

    CALL_SUBTEST_13((jacobisvd_thin_full_options<Matrix3f>()));
    CALL_SUBTEST_15((jacobisvd_thin_full_options<Matrix4d>()));
    CALL_SUBTEST_17((jacobisvd_thin_full_options<Matrix<float, 2, 3>>()));
    CALL_SUBTEST_19((jacobisvd_thin_full_options<Matrix<double, 4, 7>>()));
    CALL_SUBTEST_21((jacobisvd_thin_full_options<Matrix<double, 7, 4>>()));
    CALL_SUBTEST_23((jacobisvd_thin_full_options<Matrix<double, Dynamic, 5>>(Matrix<double, Dynamic, 5>(r, 5))));
    CALL_SUBTEST_25((jacobisvd_thin_full_options<Matrix<double, 5, Dynamic>>(Matrix<double, 5, Dynamic>(5, c))));
    CALL_SUBTEST_27((jacobisvd_thin_full_options<MatrixXf>(MatrixXf(r, c))));
    CALL_SUBTEST_29((jacobisvd_thin_full_options<MatrixXcd>(MatrixXcd(r, c))));
    CALL_SUBTEST_31((jacobisvd_thin_full_options<MatrixXd>(MatrixXd(r, c))));
    CALL_SUBTEST_33((jacobisvd_thin_full_options<Matrix<double, 5, 7, RowMajor>>()));
    CALL_SUBTEST_35((jacobisvd_thin_full_options<Matrix<double, 7, 5, RowMajor>>()));

    CALL_SUBTEST_62((svd_precondition_square_matrix_accuracy<Matrix2f>(2)));
    CALL_SUBTEST_62((svd_precondition_square_matrix_accuracy<Matrix4d>(4)));
    CALL_SUBTEST_62((svd_precondition_square_matrix_accuracy<MatrixXf>(12)));
    CALL_SUBTEST_62((svd_precondition_square_matrix_accuracy<MatrixXd>(12)));
    CALL_SUBTEST_62((svd_precondition_square_matrix_accuracy<Matrix<double, Dynamic, Dynamic, RowMajor>>(12)));
    CALL_SUBTEST_63((svd_precondition_square_matrix_accuracy<Matrix2cd>(2)));
    CALL_SUBTEST_63((svd_precondition_square_matrix_accuracy<Matrix4cf>(4)));
    CALL_SUBTEST_63((svd_precondition_square_matrix_accuracy<MatrixXcf>(12)));
    CALL_SUBTEST_63((svd_precondition_square_matrix_accuracy<MatrixXcd>(12)));
    CALL_SUBTEST_64((svd_precondition_square_matrix_runtime_options<MatrixXf>(12)));
    CALL_SUBTEST_64((svd_precondition_square_matrix_runtime_options<MatrixXcd>(r)));
    CALL_SUBTEST_68((jacobisvd_fullpiv_runtime_thin_asserts<0>(8, 5)));
    CALL_SUBTEST_68((jacobisvd_fullpiv_runtime_thin_asserts<PreconditionSquareMatrix>(6, 6)));
    CALL_SUBTEST_69((jacobisvd_thin_full_options<Matrix<double, Dynamic, Dynamic, RowMajor>>(
        Matrix<double, Dynamic, Dynamic, RowMajor>(8, 8))));

    MatrixXcd noQRTest = MatrixXcd(r, r);
    CALL_SUBTEST_37((svd_thin_full_option_checks<MatrixXcd, NoQRPreconditioner>(noQRTest)));

    CALL_SUBTEST_38((
        svd_check_max_size_matrix<Matrix<float, Dynamic, Dynamic, ColMajor, 13, 15>, ColPivHouseholderQRPreconditioner>(
            r, c)));
    CALL_SUBTEST_39(
        (svd_check_max_size_matrix<Matrix<float, Dynamic, Dynamic, ColMajor, 15, 13>, HouseholderQRPreconditioner>(r,
                                                                                                                   c)));
    CALL_SUBTEST_40((
        svd_check_max_size_matrix<Matrix<float, Dynamic, Dynamic, RowMajor, 13, 15>, ColPivHouseholderQRPreconditioner>(
            r, c)));
    CALL_SUBTEST_41(
        (svd_check_max_size_matrix<Matrix<float, Dynamic, Dynamic, RowMajor, 15, 13>, HouseholderQRPreconditioner>(r,
                                                                                                                   c)));

    // Test on inf/nan matrix
    CALL_SUBTEST_42((svd_inf_nan<MatrixXf>()));
    CALL_SUBTEST_43((svd_inf_nan<MatrixXd>()));

    CALL_SUBTEST_44((jacobisvd_vector_asserts<Matrix<double, 6, 1>>()));
    CALL_SUBTEST_45((jacobisvd_vector_asserts<Matrix<double, 1, 6>>()));
    CALL_SUBTEST_46((jacobisvd_vector_asserts<Matrix<double, Dynamic, 1>>(Matrix<double, Dynamic, 1>(r))));
    CALL_SUBTEST_47((jacobisvd_vector_asserts<Matrix<double, 1, Dynamic>>(Matrix<double, 1, Dynamic>(c))));
  }

  CALL_SUBTEST_48((jacobisvd_thin_full_options<MatrixXd>(
      MatrixXd(internal::random<int>(EIGEN_TEST_MAX_SIZE / 4, EIGEN_TEST_MAX_SIZE / 2),
               internal::random<int>(EIGEN_TEST_MAX_SIZE / 4, EIGEN_TEST_MAX_SIZE / 2)))));
  // Three parts, because under QEMU's SVE and SME emulation the six preconditioners together reach CI's 3000 s
  // per-test timeout.
  const int cd_rows = internal::random<int>(EIGEN_TEST_MAX_SIZE / 4, EIGEN_TEST_MAX_SIZE / 3);
  const int cd_cols = internal::random<int>(EIGEN_TEST_MAX_SIZE / 4, EIGEN_TEST_MAX_SIZE / 3);
  TEST_SET_BUT_UNUSED_VARIABLE(cd_rows);
  TEST_SET_BUT_UNUSED_VARIABLE(cd_cols);
  CALL_SUBTEST_49((jacobisvd_thin_full_options_colpiv<MatrixXcd>(MatrixXcd(cd_rows, cd_cols))));
  CALL_SUBTEST_50((jacobisvd_thin_full_options_householder<MatrixXcd>(MatrixXcd(cd_rows, cd_cols))));
  CALL_SUBTEST_51((jacobisvd_thin_full_options_fullpiv<MatrixXcd>(MatrixXcd(cd_rows, cd_cols))));

  // test matrixbase method
  CALL_SUBTEST_52((jacobisvd_method<Matrix2cd>()));
  CALL_SUBTEST_53((jacobisvd_method<Matrix3f>()));

  // Test problem size constructors
  CALL_SUBTEST_54(JacobiSVD<MatrixXf>(10, 10));

  // Check that preallocation avoids subsequent mallocs
  CALL_SUBTEST_55(svd_preallocate<void>());

  CALL_SUBTEST_56(svd_underoverflow<void>());
  CALL_SUBTEST_56((svd_overflowing_modulus<MatrixXcf>(6)));
  CALL_SUBTEST_56((svd_overflowing_modulus<MatrixXcd>(6)));
  CALL_SUBTEST_56(jacobisvd_power_of_two_scaling<>());
  CALL_SUBTEST_65((jacobisvd_flushed_subnormal_matrix<MatrixXf>(5, 4)));
  CALL_SUBTEST_65((jacobisvd_flushed_subnormal_matrix<MatrixXd>(4, 5)));
  CALL_SUBTEST_65((jacobisvd_flushed_subnormal_matrix<MatrixXcf>(4, 4)));
  CALL_SUBTEST_65((jacobisvd_flushed_subnormal_matrix<MatrixXcd>(3, 5)));
  CALL_SUBTEST_66(jacobisvd_subnormal_diagonals<float>());
  CALL_SUBTEST_66(jacobisvd_subnormal_diagonals<double>());
  CALL_SUBTEST_70(jacobisvd_complex_tiny_pivot<float>());
  CALL_SUBTEST_70(jacobisvd_complex_tiny_pivot<double>());
  CALL_SUBTEST_70(jacobisvd_unit_phase_near_minimum<float>());
  CALL_SUBTEST_70(jacobisvd_unit_phase_near_minimum<double>());
  CALL_SUBTEST_73(jacobisvd_complex_tiny_pivot_sweeps<float>(6));
  CALL_SUBTEST_73(jacobisvd_complex_tiny_pivot_sweeps<double>(6));
  CALL_SUBTEST_71((jacobisvd_rank_deficient_family<float, ColPivHouseholderQRPreconditioner>(4, 20)));
  CALL_SUBTEST_71((jacobisvd_rank_deficient_family<float, HouseholderQRPreconditioner>(4, 20)));
  CALL_SUBTEST_71((jacobisvd_rank_deficient_family<float, FullPivHouseholderQRPreconditioner>(4, 20)));
  CALL_SUBTEST_71((jacobisvd_rank_deficient_family<float, HouseholderQRPreconditioner>(20, 4)));
  CALL_SUBTEST_71((jacobisvd_rank_deficient_family<float, NoQRPreconditioner>(8, 8)));
  CALL_SUBTEST_72((jacobisvd_rank_deficient_family<double, ColPivHouseholderQRPreconditioner>(4, 20)));
  CALL_SUBTEST_72((jacobisvd_rank_deficient_family<double, HouseholderQRPreconditioner>(8, 40)));
  CALL_SUBTEST_72((jacobisvd_rank_deficient_family<double, FullPivHouseholderQRPreconditioner>(4, 20)));
  CALL_SUBTEST_72((jacobisvd_rank_deficient_family<double, NoQRPreconditioner>(8, 8)));
  CALL_SUBTEST_67((svd_zero_matrix_solvers<Matrix3d>(3, 3)));
  CALL_SUBTEST_67((svd_zero_matrix_solvers<Matrix<float, 2, 5>>(2, 5)));
  CALL_SUBTEST_67((svd_zero_matrix_solvers<MatrixXcf>(5, 3)));
  CALL_SUBTEST_67((svd_zero_matrix_solvers<MatrixXd>(0, 0)));
  CALL_SUBTEST_67((svd_zero_matrix_solvers<MatrixXd>(0, 3)));
  CALL_SUBTEST_67((svd_zero_matrix_solvers<MatrixXd>(3, 0)));

  // Check that the TriangularBase constructor works
  CALL_SUBTEST_57((svd_triangular_matrix<Matrix3d>()));
  CALL_SUBTEST_58((svd_triangular_matrix<Matrix4f>()));
  CALL_SUBTEST_59((svd_triangular_matrix<Matrix<double, 10, 10>>()));

  msvc_workaround<>();
}
