// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Defining only the run-time threshold must leave the compile-time fixed-size one at its default (#3119). 30 lies
// between the two defaults, so a bound derived from it as 2 * 30 would move 14 x 14 x 14 to the coeff-based path.
#define EIGEN_GEMM_TO_COEFFBASED_THRESHOLD 30
#include "main.h"

static_assert(EIGEN_FIXED_SIZE_GEMM_TO_COEFFBASED_THRESHOLD == 40,
              "the fixed-size default must not follow the runtime one");

template <typename Scalar>
void fixed_size_threshold_is_independent() {
#ifdef EIGEN_VECTORIZE_SME
  constexpr int kThreshold = internal::sme_has_gebp_kernel<Scalar, Scalar>::value
                                 ? EIGEN_SME_FIXED_SIZE_GEMM_TO_COEFFBASED_THRESHOLD
                                 : EIGEN_FIXED_SIZE_GEMM_TO_COEFFBASED_THRESHOLD;
#else
  constexpr int kThreshold = EIGEN_FIXED_SIZE_GEMM_TO_COEFFBASED_THRESHOLD;
#endif
  constexpr int kGemmSide = (kThreshold + 2) / 3;
  using Below = Matrix<Scalar, kGemmSide - 1, kGemmSide - 1>;
  using At = Matrix<Scalar, kGemmSide, kGemmSide>;
  STATIC_CHECK((internal::product_type<At, At>::FixedSizeThreshold == kThreshold));
  STATIC_CHECK((internal::product_type<Below, Below>::value == CoeffBasedProductMode));
  STATIC_CHECK((internal::product_type<At, At>::value == GemmProduct));

  const Below a = Below::Random(), b = Below::Random();
  Below c;
  c.noalias() = a * b;
  VERIFY_IS_APPROX(c, a.lazyProduct(b));
  // 3 * kGemmSide >= 30: the run-time bound keeps it on the GEMM path.
  const At d = At::Random(), e = At::Random();
  At f;
  f.noalias() = d * e;
  VERIFY_IS_APPROX(f, d.lazyProduct(e));
}

EIGEN_DECLARE_TEST(product_threshold) {
  CALL_SUBTEST(fixed_size_threshold_is_independent<float>());
  CALL_SUBTEST(fixed_size_threshold_is_independent<double>());
  CALL_SUBTEST(fixed_size_threshold_is_independent<std::complex<float>>());
  CALL_SUBTEST(fixed_size_threshold_is_independent<std::complex<double>>());
}
