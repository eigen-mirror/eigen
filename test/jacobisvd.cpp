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
#include "fp_control.h"

#define SVD_DEFAULT(M) JacobiSVD<M>
#define SVD_FOR_MIN_NORM(M) JacobiSVD<M, ColPivHouseholderQRPreconditioner>
#define SVD_STATIC_OPTIONS(M, O) JacobiSVD<M, O>
#include "svd_common.h"

template <typename MatrixType>
void jacobisvd_method() {
  enum { Size = MatrixType::RowsAtCompileTime };
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<RealScalar, Size, 1> RealVecType;
  MatrixType m = MatrixType::Identity();
  VERIFY_IS_APPROX(m.jacobiSvd().singularValues(), RealVecType::Ones());
  VERIFY_RAISES_ASSERT(m.jacobiSvd().matrixU());
  VERIFY_RAISES_ASSERT(m.jacobiSvd().matrixV());
  VERIFY_IS_APPROX(m.template jacobiSvd<ComputeFullU | ComputeFullV>().solve(m), m);
  VERIFY_IS_APPROX(m.template jacobiSvd<ComputeFullU | ComputeFullV>().transpose().solve(m), m);
  VERIFY_IS_APPROX(m.template jacobiSvd<ComputeFullU | ComputeFullV>().adjoint().solve(m), m);
}

template <typename MatrixType>
void jacobisvd_thin_full_options(const MatrixType& input = MatrixType()) {
  svd_thin_full_option_checks<MatrixType, 0>(input);
  svd_thin_full_option_checks<MatrixType, HouseholderQRPreconditioner>(input);
  svd_option_checks_full_only<MatrixType, FullPivHouseholderQRPreconditioner>(
      input);  // FullPiv only used when computing full unitaries
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
  typedef JacobiSVD<MatrixType> DefaultSVD;
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
void msvc_workaround() {
  const Foo::Bar a;
  const Foo::Bar b;
  const Foo::Bar c = std::max EIGEN_NOT_A_MACRO(a, b);
  EIGEN_UNUSED_VARIABLE(c);
}

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

void jacobisvd_large_tau_regression() {
  Matrix3f m;
  m << 3.7855173218304116745e-07f, 0.0f, 500.0f, -4.9999995231628417969f, -0.0f, -1.9106853686029490191e-12f,
      -8.6602544784545898438f, 0.0f, 2.1855694285477511585f;

#ifdef FE_OVERFLOW
  std::fenv_t environment;
  const bool check_overflow = std::feholdexcept(&environment) == 0;
  if (!check_overflow) std::cout << "SKIP: JacobiSVD overflow flag check: feholdexcept failed.\n";
#else
  std::cout << "SKIP: JacobiSVD overflow flag check: FE_OVERFLOW is unavailable.\n";
#endif
  JacobiSVD<Matrix3f> svd(m, ComputeFullU | ComputeFullV);
#ifdef FE_OVERFLOW
  if (check_overflow) {
    const int overflow = std::fetestexcept(FE_OVERFLOW);
    const int restored = std::fesetenv(&environment);
    VERIFY_IS_EQUAL(restored, 0);
    VERIFY_IS_EQUAL(overflow, 0);
  }
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

EIGEN_DECLARE_TEST(jacobisvd) {
  CALL_SUBTEST_60((svd_normal_equation_roundoff<float, ColMajor>()));
  CALL_SUBTEST_60((svd_normal_equation_roundoff<double, RowMajor>()));
  CALL_SUBTEST_61((svd_normal_equation_roundoff<std::complex<float>, RowMajor>()));
  CALL_SUBTEST_61((svd_normal_equation_roundoff<std::complex<double>, ColMajor>()));

  CALL_SUBTEST_1((jacobisvd_verify_inputs<Matrix4d>()));
  CALL_SUBTEST_2((jacobisvd_verify_inputs(Matrix<float, 5, Dynamic>(5, 6))));
  CALL_SUBTEST_3((jacobisvd_verify_inputs<Matrix<std::complex<double>, 7, 5>>()));
  CALL_SUBTEST_4((jacobisvd_mixed_option_enum_regression()));
  CALL_SUBTEST_4((jacobisvd_large_tau_regression()));

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
  CALL_SUBTEST_50((jacobisvd_thin_full_options<MatrixXcd>(
      MatrixXcd(internal::random<int>(EIGEN_TEST_MAX_SIZE / 4, EIGEN_TEST_MAX_SIZE / 3),
                internal::random<int>(EIGEN_TEST_MAX_SIZE / 4, EIGEN_TEST_MAX_SIZE / 3)))));

  // test matrixbase method
  CALL_SUBTEST_52((jacobisvd_method<Matrix2cd>()));
  CALL_SUBTEST_53((jacobisvd_method<Matrix3f>()));

  // Test problem size constructors
  CALL_SUBTEST_54(JacobiSVD<MatrixXf>(10, 10));

  // Check that preallocation avoids subsequent mallocs
  CALL_SUBTEST_55(svd_preallocate<void>());

  CALL_SUBTEST_56(svd_underoverflow<void>());
  CALL_SUBTEST_56(jacobisvd_power_of_two_scaling());
  CALL_SUBTEST_56((jacobisvd_flushed_subnormal_matrix<MatrixXf>(5, 4)));
  CALL_SUBTEST_56((jacobisvd_flushed_subnormal_matrix<MatrixXd>(4, 5)));
  CALL_SUBTEST_56((jacobisvd_flushed_subnormal_matrix<MatrixXcf>(4, 4)));
  CALL_SUBTEST_56((jacobisvd_flushed_subnormal_matrix<MatrixXcd>(3, 5)));
  CALL_SUBTEST_56(jacobisvd_subnormal_diagonals<float>());
  CALL_SUBTEST_56(jacobisvd_subnormal_diagonals<double>());
  CALL_SUBTEST_56((svd_zero_matrix_solvers<Matrix3d>(3, 3)));
  CALL_SUBTEST_56((svd_zero_matrix_solvers<Matrix<float, 2, 5>>(2, 5)));
  CALL_SUBTEST_56((svd_zero_matrix_solvers<MatrixXcf>(5, 3)));

  // Check that the TriangularBase constructor works
  CALL_SUBTEST_57((svd_triangular_matrix<Matrix3d>()));
  CALL_SUBTEST_58((svd_triangular_matrix<Matrix4f>()));
  CALL_SUBTEST_59((svd_triangular_matrix<Matrix<double, 10, 10>>()));

  msvc_workaround();
}
