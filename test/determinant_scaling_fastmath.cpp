// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/LU>

// Under -ffinite-math-only the compiler assumes that no product overflows, so a rescue that tests the partial product
// for an infinity is compiled away. The pivots here overflow together before the small ones bring the product back.
template <typename Scalar>
void determinant_scaling_fastmath() {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  const RealScalar big = std::ldexp(RealScalar(1), std::numeric_limits<RealScalar>::max_exponent / 2 + 2);
  const RealScalar small = RealScalar(1) / big;
  Matrix<Scalar, Dynamic, 1> diagonal(6);
  diagonal << Scalar(big), Scalar(big), Scalar(3), Scalar(small), Scalar(small), Scalar(-5);
  const MatrixType A = diagonal.asDiagonal();

  VERIFY_IS_EQUAL(internal::scaled_prod(diagonal).value(), Scalar(-15));
  VERIFY_IS_EQUAL(A.template triangularView<Upper>().determinant(), Scalar(-15));
  const PartialPivLU<MatrixType> lu(A);
  VERIFY_IS_EQUAL(lu.determinant(), Scalar(-15));
  // |z| of a complex scalar takes a square root, which clang 14 compiles under -ffast-math on x86-64 to rsqrtss and
  // one Newton step (clang 15 emits sqrtss): relative error ~(3/2) (1.5 2^-12)^2 < 2 eps, plus a rounding per step.
  VERIFY(numext::abs(lu.absDeterminant() - RealScalar(15)) <=
         RealScalar(4) * NumTraits<RealScalar>::epsilon() * RealScalar(15));
  // log(mantissa), exponent log(2) and their sum each round once.
  VERIFY(numext::abs(lu.logAbsDeterminant() - std::log(RealScalar(15))) <=
         RealScalar(4) * NumTraits<RealScalar>::epsilon() * std::log(RealScalar(15)));
}

EIGEN_DECLARE_TEST(determinant_scaling_fastmath) {
  CALL_SUBTEST(determinant_scaling_fastmath<float>());
  CALL_SUBTEST(determinant_scaling_fastmath<double>());
  CALL_SUBTEST(determinant_scaling_fastmath<std::complex<float>>());
  CALL_SUBTEST(determinant_scaling_fastmath<std::complex<double>>());
}
