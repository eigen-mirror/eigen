// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_RUNTIME_NO_MALLOC
#include "main.h"
#include <Eigen/Cholesky>

template <typename MatrixType, typename Factorization>
void check_cholesky_reconstruction(const MatrixType& input, const Factorization& factorization) {
  using RealScalar = typename MatrixType::RealScalar;
  VERIFY_IS_EQUAL(factorization.info(), Success);
  const MatrixType reconstructed = factorization.reconstructedMatrix();
  // Two triangular products and the factorization each accumulate O(n) rounding errors per coefficient.
  const RealScalar tolerance =
      RealScalar(64) * RealScalar((std::max)(Index(1), input.rows())) * NumTraits<RealScalar>::epsilon() * input.norm();
  VERIFY(reconstructed.allFinite());
  VERIFY((reconstructed - input).norm() <= tolerance);
  VERIFY_IS_EQUAL(MatrixType(reconstructed.template triangularView<StrictlyUpper>()),
                  MatrixType(reconstructed.adjoint().template triangularView<StrictlyUpper>()));
}

template <typename Scalar, int Order, int UpLo>
void cholesky_reconstruction() {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, Order>;
  for (Index size : {0, 1, 2, 5, 17, 63, 64, 65}) {
    const MatrixType random = MatrixType::Random(size, size);
    MatrixType input = random * random.adjoint();
    input.diagonal().array() += Scalar(1);
    check_cholesky_reconstruction(input, LDLT<MatrixType, UpLo>(input));
    input = random + random.adjoint();
    check_cholesky_reconstruction(input, BunchKaufman<MatrixType, UpLo>(input));
  }
}

template <typename Scalar, int Order, int UpLo>
void cholesky_reconstruction_fixed() {
  using MatrixType = Matrix<Scalar, 4, 4, Order>;
  MatrixType input = MatrixType::Zero();
  input.diagonal() << Scalar(1), Scalar(4), Scalar(2), Scalar(3);
  LDLT<MatrixType, UpLo> ldlt(input);
  VERIFY_IS_EQUAL(ldlt.transpositionsP().coeff(0), 1);
  internal::set_is_malloc_allowed(false);
  check_cholesky_reconstruction(input, ldlt);
  internal::set_is_malloc_allowed(true);

  // The off-diagonal entry forces a 2x2 pivot and a non-adjacent symmetric interchange.
  const Scalar offDiagonal = Scalar(2) + Scalar(0.5) * internal::random<Scalar>();
  input.setZero();
  input(1, 1) = Scalar(2);
  input(2, 2) = Scalar(-3);
  input(3, 0) = offDiagonal;
  input(0, 3) = numext::conj(offDiagonal);
  BunchKaufman<MatrixType, UpLo> bk(input);
  VERIFY_IS_EQUAL(bk.transpositionsP().coeff(1), 3);
  VERIFY(!numext::is_exactly_zero(bk.subDiagonal().coeff(0)));
  internal::set_is_malloc_allowed(false);
  check_cholesky_reconstruction(input, bk);
  internal::set_is_malloc_allowed(true);
}

EIGEN_DECLARE_TEST(cholesky_reconstruction) {
  CALL_SUBTEST_1((cholesky_reconstruction<double, ColMajor, Lower>()));
  CALL_SUBTEST_1((cholesky_reconstruction<double, ColMajor, Upper>()));
  CALL_SUBTEST_2((cholesky_reconstruction<double, RowMajor, Lower>()));
  CALL_SUBTEST_2((cholesky_reconstruction<double, RowMajor, Upper>()));
  CALL_SUBTEST_3((cholesky_reconstruction<std::complex<double>, ColMajor, Lower>()));
  CALL_SUBTEST_3((cholesky_reconstruction<std::complex<double>, ColMajor, Upper>()));
  CALL_SUBTEST_4((cholesky_reconstruction<std::complex<double>, RowMajor, Lower>()));
  CALL_SUBTEST_4((cholesky_reconstruction<std::complex<double>, RowMajor, Upper>()));
  CALL_SUBTEST_5((cholesky_reconstruction<float, ColMajor, Lower>()));
  CALL_SUBTEST_5((cholesky_reconstruction<std::complex<float>, RowMajor, Upper>()));
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1((cholesky_reconstruction_fixed<double, ColMajor, Lower>()));
    CALL_SUBTEST_2((cholesky_reconstruction_fixed<double, RowMajor, Upper>()));
    CALL_SUBTEST_3((cholesky_reconstruction_fixed<std::complex<double>, ColMajor, Upper>()));
    CALL_SUBTEST_4((cholesky_reconstruction_fixed<std::complex<double>, RowMajor, Lower>()));
  }
}
