// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Gauthier Brun <brun.gauthier@gmail.com>
// Copyright (C) 2013 Nicolas Carre <nicolas.carre@ensimag.fr>
// Copyright (C) 2013 Jean Ceccato <jean.ceccato@ensimag.fr>
// Copyright (C) 2013 Pierre Zoppitelli <pierre.zoppitelli@ensimag.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// discard stack allocation as that too bypasses malloc
#define EIGEN_STACK_ALLOCATION_LIMIT 0
#define EIGEN_RUNTIME_NO_MALLOC

#include "main.h"
#include <Eigen/SVD>

#define SVD_DEFAULT(M) BDCSVD<M>
#define SVD_FOR_MIN_NORM(M) BDCSVD<M>
#define SVD_STATIC_OPTIONS(M, O) BDCSVD<M, O>
#include "svd_common.h"

template <typename MatrixType>
void bdcsvd_method() {
  enum { Size = MatrixType::RowsAtCompileTime };
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<RealScalar, Size, 1> RealVecType;
  MatrixType m = MatrixType::Identity();
  VERIFY_IS_APPROX(m.bdcSvd().singularValues(), RealVecType::Ones());
  VERIFY_RAISES_ASSERT(m.bdcSvd().matrixU());
  VERIFY_RAISES_ASSERT(m.bdcSvd().matrixV());
}

// compare the Singular values returned with Jacobi and Bdc
template <typename MatrixType>
void compare_bdc_jacobi(const MatrixType& a = MatrixType(), int algoswap = 16, bool random = true) {
  MatrixType m = random ? MatrixType::Random(a.rows(), a.cols()) : a;

  BDCSVD<MatrixType> bdc_svd(m.rows(), m.cols());
  bdc_svd.setSwitchSize(algoswap);
  bdc_svd.compute(m);

  JacobiSVD<MatrixType> jacobi_svd(m);
  VERIFY_IS_APPROX(bdc_svd.singularValues(), jacobi_svd.singularValues());
}

// Verifies total deflation is **not** triggered.
void compare_bdc_jacobi_instance(bool structure_as_m, int algoswap = 16) {
  MatrixXd m(4, 3);
  if (structure_as_m) {
    // The first 3 rows are the reduced form of Matrix 1 as shown below, and it
    // has nonzero elements in the first column and diagonals only.
    m << 1.056293, 0, 0, -0.336468, 0.907359, 0, -1.566245, 0, 0.149150, -0.1, 0, 0;
  } else {
    // Matrix 1.
    m << 0.882336, 18.3914, -26.7921, -5.58135, 17.1931, -24.0892, -20.794, 8.68496, -4.83103, -8.4981, -10.5451,
        23.9072;
  }
  compare_bdc_jacobi(m, algoswap, false);
}

template <typename MatrixType>
void bdcsvd_thin_options(const MatrixType& input = MatrixType()) {
  svd_thin_option_checks<MatrixType, 0>(input);
}

template <typename MatrixType>
void bdcsvd_full_options(const MatrixType& input = MatrixType()) {
  svd_option_checks_full_only<MatrixType, 0>(input);
}

template <typename MatrixType>
void bdcsvd_verify_assert(const MatrixType& input = MatrixType()) {
  svd_verify_assert<MatrixType>(input);
  svd_verify_constructor_options_assert<BDCSVD<MatrixType>>(input);
}

template <typename MatrixType>
void bdcsvd_check_convergence(const MatrixType& input) {
  BDCSVD<MatrixType, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(input);
  VERIFY(svd.info() == Eigen::Success);
  MatrixType D = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
  VERIFY_IS_APPROX(input, D);
}

// Verify SVD of bidiagonal matrix given as diagonal + superdiagonal vectors.
template <typename RealScalar>
void verify_bidiagonal_svd(const Matrix<RealScalar, Dynamic, 1>& diag,
                           const Matrix<RealScalar, Dynamic, 1>& superdiag) {
  typedef Matrix<RealScalar, Dynamic, Dynamic> MatrixXr;
  typedef Matrix<RealScalar, Dynamic, 1> VectorXr;
  const Index n = diag.size();

  BDCSVD<MatrixXr, ComputeFullU | ComputeFullV> bdcsvd(diag, superdiag);
  VERIFY(bdcsvd.info() == Success);

  const VectorXr& sv = bdcsvd.singularValues();

  // Singular values must be non-negative.
  for (Index i = 0; i < sv.size(); ++i) {
    VERIFY(sv(i) >= RealScalar(0));
  }

  // Singular values must be sorted descending.
  for (Index i = 1; i < sv.size(); ++i) {
    VERIFY(sv(i - 1) >= sv(i));
  }

  // Orthogonality of U and V.
  VERIFY_IS_APPROX(bdcsvd.matrixU().transpose() * bdcsvd.matrixU(), MatrixXr::Identity(n, n));
  VERIFY_IS_APPROX(bdcsvd.matrixV().transpose() * bdcsvd.matrixV(), MatrixXr::Identity(n, n));

  // Reconstruction: U * S * V^T should equal the original bidiagonal.
  MatrixXr B = MatrixXr::Zero(n, n);
  B.diagonal() = diag;
  if (n > 1) B.diagonal(1) = superdiag;
  MatrixXr recon = bdcsvd.matrixU() * sv.asDiagonal() * bdcsvd.matrixV().transpose();
  VERIFY_IS_APPROX(recon, B);

  // Cross-validate singular values against JacobiSVD.
  JacobiSVD<MatrixXr> jacobi(B);
  VERIFY_IS_APPROX(sv, jacobi.singularValues());
}

// Verify that bidiagonal API and matrix API produce matching singular values.
template <typename RealScalar>
void verify_bidiagonal_vs_matrix_svd(const Matrix<RealScalar, Dynamic, 1>& diag,
                                     const Matrix<RealScalar, Dynamic, 1>& superdiag) {
  typedef Matrix<RealScalar, Dynamic, Dynamic> MatrixXr;
  const Index n = diag.size();

  // Build dense bidiagonal matrix.
  MatrixXr B = MatrixXr::Zero(n, n);
  B.diagonal() = diag;
  if (n > 1) B.diagonal(1) = superdiag;

  BDCSVD<MatrixXr> bidiag_svd(diag, superdiag);
  BDCSVD<MatrixXr> matrix_svd(B);

  VERIFY(bidiag_svd.info() == Success);
  VERIFY(matrix_svd.info() == Success);
  VERIFY_IS_APPROX(bidiag_svd.singularValues(), matrix_svd.singularValues());
}

template <typename RealScalar>
void bdcsvd_bidiagonal_hard_cases() {
  using std::abs;
  using std::cos;
  using std::pow;
  using std::sin;
  typedef Matrix<RealScalar, Dynamic, 1> VectorXr;

  Eigen::internal::set_is_malloc_allowed(true);

  const RealScalar eps = NumTraits<RealScalar>::epsilon();

  // Test sizes: cover n=1, very small, below/above algoSwap (16), and larger.
  const int sizes[] = {1, 2, 3, 5, 10, 16, 20, 50, 100};
  const int numSizes = sizeof(sizes) / sizeof(sizes[0]);

  for (int si = 0; si < numSizes; ++si) {
    const Index n = sizes[si];
    VectorXr diag(n), superdiag(n > 1 ? n - 1 : 0);

    // 1. Identity: d=[1,...,1], e=[0,...,0]
    diag.setOnes();
    superdiag.setZero();
    verify_bidiagonal_svd<RealScalar>(diag, superdiag);
    verify_bidiagonal_vs_matrix_svd<RealScalar>(diag, superdiag);

    // 2. Zero: d=[0,...,0], e=[0,...,0]
    diag.setZero();
    superdiag.setZero();
    verify_bidiagonal_svd<RealScalar>(diag, superdiag);

    // 3. Scalar (only meaningful for n=1, but runs for all)
    if (n == 1) {
      diag(0) = RealScalar(3.14);
      verify_bidiagonal_svd<RealScalar>(diag, superdiag);
    }

    // 4. Golub-Kahan: d=[1,...,1], e=[1,...,1]
    diag.setOnes();
    if (n > 1) superdiag.setOnes();
    verify_bidiagonal_svd<RealScalar>(diag, superdiag);

    // 5. Kahan matrix: d_i = s^(i-1), e_i = -c*s^(i-1)
    // Clamp exponents so condition number stays bounded by 1/eps.
    {
      const RealScalar theta = RealScalar(0.3);
      const RealScalar s = sin(theta);
      const RealScalar c = cos(theta);
      using std::log;
      const RealScalar maxPower = -log(eps) / (-log(s));
      for (Index i = 0; i < n; ++i) diag(i) = pow(s, numext::mini(RealScalar(i), maxPower));
      for (Index i = 0; i < n - 1; ++i) superdiag(i) = -c * pow(s, numext::mini(RealScalar(i), maxPower));
      verify_bidiagonal_svd<RealScalar>(diag, superdiag);
    }

    // 6. Geometric decay diagonal: d_i = 0.5^i, e=[0,...,0]
    // Clamp so condition number stays bounded by 1/eps.
    {
      using std::log;
      const RealScalar base = RealScalar(0.5);
      const RealScalar maxPower = -log(eps) / (-log(base));
      for (Index i = 0; i < n; ++i) diag(i) = pow(base, numext::mini(RealScalar(i), maxPower));
      superdiag.setZero();
      verify_bidiagonal_svd<RealScalar>(diag, superdiag);
    }

    // 7. Geometric decay superdiagonal: d=[1,...,1], e_i = 0.5^i
    diag.setOnes();
    for (Index i = 0; i < n - 1; ++i) superdiag(i) = pow(RealScalar(0.5), RealScalar(i));
    verify_bidiagonal_svd<RealScalar>(diag, superdiag);

    // 8. Clustered at 1: d_i = 1 + i*eps, e=[0,...,0]
    for (Index i = 0; i < n; ++i) diag(i) = RealScalar(1) + RealScalar(i) * eps;
    superdiag.setZero();
    verify_bidiagonal_svd<RealScalar>(diag, superdiag);

    // 9. Two clusters: half ≈ 1, half ≈ eps
    for (Index i = 0; i < n; ++i) diag(i) = (i < n / 2) ? RealScalar(1) : eps;
    superdiag.setZero();
    verify_bidiagonal_svd<RealScalar>(diag, superdiag);

    // 10. Single tiny singular value: d=[1,...,1,eps], e=[eps^2,...]
    diag.setOnes();
    diag(n - 1) = eps;
    for (Index i = 0; i < n - 1; ++i) superdiag(i) = eps * eps;
    verify_bidiagonal_svd<RealScalar>(diag, superdiag);

    // 11. Graded: d_i = 10^(-i), e_i = 10^(-i)
    for (Index i = 0; i < n; ++i) diag(i) = pow(RealScalar(10), -RealScalar(i));
    for (Index i = 0; i < n - 1; ++i) superdiag(i) = pow(RealScalar(10), -RealScalar(i));
    verify_bidiagonal_svd<RealScalar>(diag, superdiag);

    // 12. Nearly diagonal: random diag, eps * random superdiag
    diag = VectorXr::Random(n).cwiseAbs() + VectorXr::Constant(n, RealScalar(0.1));
    for (Index i = 0; i < n - 1; ++i) superdiag(i) = eps * (RealScalar(0.5) + abs(internal::random<RealScalar>()));
    verify_bidiagonal_svd<RealScalar>(diag, superdiag);

    // 13. All equal: d=[c,...,c], e=[c,...,c]
    diag.setConstant(RealScalar(2.5));
    if (n > 1) superdiag.setConstant(RealScalar(2.5));
    verify_bidiagonal_svd<RealScalar>(diag, superdiag);

    // 14. Wilkinson: d_i = |n/2 - i|, e=[1,...,1]
    for (Index i = 0; i < n; ++i) diag(i) = abs(RealScalar(n / 2) - RealScalar(i));
    if (n > 1) superdiag.setOnes();
    verify_bidiagonal_svd<RealScalar>(diag, superdiag);

    // 15. Overflow/underflow: alternating big/tiny diagonal, tiny/big superdiagonal
    {
      const RealScalar big = (std::numeric_limits<RealScalar>::max)() / RealScalar(1000);
      const RealScalar tiny = (std::numeric_limits<RealScalar>::min)() * RealScalar(1000);
      for (Index i = 0; i < n; ++i) diag(i) = (i % 2 == 0) ? big : tiny;
      for (Index i = 0; i < n - 1; ++i) superdiag(i) = (i % 2 == 0) ? tiny : big;
      verify_bidiagonal_svd<RealScalar>(diag, superdiag);
    }

    // 16. Prescribed condition number: d_i = kappa^(-i/(n-1)), e_i = eps * random
    if (n > 1) {
      const RealScalar kappa = RealScalar(1) / eps;
      for (Index i = 0; i < n; ++i) diag(i) = pow(kappa, -RealScalar(i) / RealScalar(n - 1));
      for (Index i = 0; i < n - 1; ++i) superdiag(i) = eps * abs(internal::random<RealScalar>());
      verify_bidiagonal_svd<RealScalar>(diag, superdiag);
    }

    // 17. Rank-deficient: d=[1,..,0,..,0,..,1], e=[0,...,0]
    for (Index i = 0; i < n; ++i) diag(i) = (i < n / 3 || i >= 2 * n / 3) ? RealScalar(1) : RealScalar(0);
    superdiag.setZero();
    verify_bidiagonal_svd<RealScalar>(diag, superdiag);

    // 18. Arrowhead stress: d_i = linspace(1, n), e_i = 1/(i+1)
    for (Index i = 0; i < n; ++i) diag(i) = RealScalar(1) + RealScalar(i);
    for (Index i = 0; i < n - 1; ++i) superdiag(i) = RealScalar(1) / RealScalar(i + 1);
    verify_bidiagonal_svd<RealScalar>(diag, superdiag);

    // 19. Repeated singular values: d=[1,2,3,1,2,3,...], e=[0,...,0]
    for (Index i = 0; i < n; ++i) diag(i) = RealScalar((i % 3) + 1);
    superdiag.setZero();
    verify_bidiagonal_svd<RealScalar>(diag, superdiag);

    // 20. Glued identity: d=[1,...,1], e=0 except e[n/2-1]=eps
    diag.setOnes();
    superdiag.setZero();
    if (n > 2) superdiag(n / 2 - 1) = eps;
    verify_bidiagonal_svd<RealScalar>(diag, superdiag);
  }
}

EIGEN_DECLARE_TEST(bdcsvd) {
  CALL_SUBTEST_1((bdcsvd_verify_assert<Matrix3f>()));
  CALL_SUBTEST_2((bdcsvd_verify_assert<Matrix4d>()));
  CALL_SUBTEST_3((bdcsvd_verify_assert<Matrix<float, 10, 7>>()));
  CALL_SUBTEST_4((bdcsvd_verify_assert<Matrix<float, 7, 10>>()));
  CALL_SUBTEST_5((bdcsvd_verify_assert<Matrix<std::complex<double>, 6, 9>>()));

  CALL_SUBTEST_6((svd_all_trivial_2x2(bdcsvd_thin_options<Matrix2cd>)));
  CALL_SUBTEST_7((svd_all_trivial_2x2(bdcsvd_full_options<Matrix2cd>)));
  CALL_SUBTEST_8((svd_all_trivial_2x2(bdcsvd_thin_options<Matrix2d>)));
  CALL_SUBTEST_9((svd_all_trivial_2x2(bdcsvd_full_options<Matrix2d>)));

  for (int i = 0; i < g_repeat; i++) {
    int r = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2), c = internal::random<int>(1, EIGEN_TEST_MAX_SIZE / 2);

    TEST_SET_BUT_UNUSED_VARIABLE(r);
    TEST_SET_BUT_UNUSED_VARIABLE(c);

    CALL_SUBTEST_10((compare_bdc_jacobi<MatrixXf>(MatrixXf(r, c))));
    CALL_SUBTEST_11((compare_bdc_jacobi<MatrixXd>(MatrixXd(r, c))));
    CALL_SUBTEST_12((compare_bdc_jacobi<MatrixXcd>(MatrixXcd(r, c))));
    // Test on inf/nan matrix
    CALL_SUBTEST_13((svd_inf_nan<MatrixXf>()));
    CALL_SUBTEST_14((svd_inf_nan<MatrixXd>()));

    // Verify some computations using all combinations of the Options template parameter.
    CALL_SUBTEST_15((bdcsvd_thin_options<Matrix3f>()));
    CALL_SUBTEST_16((bdcsvd_full_options<Matrix3f>()));
    CALL_SUBTEST_17((bdcsvd_thin_options<Matrix<float, 2, 3>>()));
    CALL_SUBTEST_18((bdcsvd_full_options<Matrix<float, 2, 3>>()));
    CALL_SUBTEST_19((bdcsvd_thin_options<MatrixXd>(MatrixXd(20, 17))));
    CALL_SUBTEST_20((bdcsvd_full_options<MatrixXd>(MatrixXd(20, 17))));
    CALL_SUBTEST_21((bdcsvd_thin_options<MatrixXd>(MatrixXd(17, 20))));
    CALL_SUBTEST_22((bdcsvd_full_options<MatrixXd>(MatrixXd(17, 20))));
    CALL_SUBTEST_23((bdcsvd_thin_options<Matrix<double, Dynamic, 15>>(Matrix<double, Dynamic, 15>(r, 15))));
    CALL_SUBTEST_24((bdcsvd_full_options<Matrix<double, Dynamic, 15>>(Matrix<double, Dynamic, 15>(r, 15))));
    CALL_SUBTEST_25((bdcsvd_thin_options<Matrix<double, 13, Dynamic>>(Matrix<double, 13, Dynamic>(13, c))));
    CALL_SUBTEST_26((bdcsvd_full_options<Matrix<double, 13, Dynamic>>(Matrix<double, 13, Dynamic>(13, c))));
    CALL_SUBTEST_27((bdcsvd_thin_options<MatrixXf>(MatrixXf(r, c))));
    CALL_SUBTEST_28((bdcsvd_full_options<MatrixXf>(MatrixXf(r, c))));
    CALL_SUBTEST_29((bdcsvd_thin_options<MatrixXcd>(MatrixXcd(r, c))));
    CALL_SUBTEST_30((bdcsvd_full_options<MatrixXcd>(MatrixXcd(r, c))));
    CALL_SUBTEST_31((bdcsvd_thin_options<MatrixXd>(MatrixXd(r, c))));
    CALL_SUBTEST_32((bdcsvd_full_options<MatrixXd>(MatrixXd(r, c))));
    CALL_SUBTEST_33((bdcsvd_thin_options<Matrix<double, Dynamic, Dynamic, RowMajor>>(
        Matrix<double, Dynamic, Dynamic, RowMajor>(20, 27))));
    CALL_SUBTEST_34((bdcsvd_full_options<Matrix<double, Dynamic, Dynamic, RowMajor>>(
        Matrix<double, Dynamic, Dynamic, RowMajor>(20, 27))));
    CALL_SUBTEST_35((bdcsvd_thin_options<Matrix<double, Dynamic, Dynamic, RowMajor>>(
        Matrix<double, Dynamic, Dynamic, RowMajor>(27, 20))));
    CALL_SUBTEST_36((bdcsvd_full_options<Matrix<double, Dynamic, Dynamic, RowMajor>>(
        Matrix<double, Dynamic, Dynamic, RowMajor>(27, 20))));
    CALL_SUBTEST_37((
        svd_check_max_size_matrix<Matrix<float, Dynamic, Dynamic, ColMajor, 20, 35>, ColPivHouseholderQRPreconditioner>(
            r, c)));
    CALL_SUBTEST_38(
        (svd_check_max_size_matrix<Matrix<float, Dynamic, Dynamic, ColMajor, 35, 20>, HouseholderQRPreconditioner>(r,
                                                                                                                   c)));
    CALL_SUBTEST_39((
        svd_check_max_size_matrix<Matrix<float, Dynamic, Dynamic, RowMajor, 20, 35>, ColPivHouseholderQRPreconditioner>(
            r, c)));
    CALL_SUBTEST_40(
        (svd_check_max_size_matrix<Matrix<float, Dynamic, Dynamic, RowMajor, 35, 20>, HouseholderQRPreconditioner>(r,
                                                                                                                   c)));
  }

  // test matrixbase method
  CALL_SUBTEST_41((bdcsvd_method<Matrix2cd>()));
  CALL_SUBTEST_42((bdcsvd_method<Matrix3f>()));

  // Test problem size constructors
  CALL_SUBTEST_43(BDCSVD<MatrixXf>(10, 10));

  // Check that preallocation avoids subsequent mallocs
  // Disabled because not supported by BDCSVD
  // CALL_SUBTEST_9( svd_preallocate<void>() );

  CALL_SUBTEST_44(svd_underoverflow<void>());

  // Without total deflation issues.
  CALL_SUBTEST_45((compare_bdc_jacobi_instance(true)));
  CALL_SUBTEST_46((compare_bdc_jacobi_instance(false)));

  // With total deflation issues before, when it shouldn't be triggered.
  CALL_SUBTEST_47((compare_bdc_jacobi_instance(true, 3)));
  CALL_SUBTEST_48((compare_bdc_jacobi_instance(false, 3)));

  // Convergence for large constant matrix (https://gitlab.com/libeigen/eigen/-/issues/2491)
  CALL_SUBTEST_49(bdcsvd_check_convergence<MatrixXf>(MatrixXf::Constant(500, 500, 1)));

  // Bidiagonal SVD hard test cases
  CALL_SUBTEST_50((bdcsvd_bidiagonal_hard_cases<float>()));
  CALL_SUBTEST_51((bdcsvd_bidiagonal_hard_cases<double>()));
}
