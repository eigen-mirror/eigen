// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_USE_LAPACKE
#include "main.h"
#include <Eigen/Eigenvalues>

namespace {
int real_schur_calls = 0;
int complex_schur_calls = 0;
int selfadjoint_calls = 0;
}  // namespace

// Dispatch probes return a failure code before Eigen can consume any unwritten output.
extern "C" lapack_int LAPACKE_dgees(int, char, char, LAPACK_D_SELECT2, lapack_int, double*, lapack_int, lapack_int*,
                                    double*, double*, double*, lapack_int) {
  ++real_schur_calls;
  return 1;
}
extern "C" lapack_int LAPACKE_zgees(int, char, char, LAPACK_Z_SELECT1, lapack_int, lapack_complex_double*, lapack_int,
                                    lapack_int*, lapack_complex_double*, lapack_complex_double*, lapack_int) {
  ++complex_schur_calls;
  return 1;
}
extern "C" lapack_int LAPACKE_dsyev(int, char, char, lapack_int, double*, lapack_int, double*) {
  ++selfadjoint_calls;
  return 1;
}

template <int Options>
void inplace_plain_lapacke_dispatch() {
  using RealMatrix = Matrix<double, Dynamic, Dynamic, Options>;
  using ComplexMatrix = Matrix<std::complex<double>, Dynamic, Dynamic, Options>;
  RealMatrix real = RealMatrix::Identity(3, 3);
  ComplexMatrix complex = ComplexMatrix::Identity(3, 3);
  const RealMatrix constReal = RealMatrix::Identity(3, 3);
  const ComplexMatrix constComplex = ComplexMatrix::Identity(3, 3);
  const int realBefore = real_schur_calls, complexBefore = complex_schur_calls, selfadjointBefore = selfadjoint_calls;
  RealSchur<RealMatrix> mutableRealSchur(real), constRealSchur(constReal);
  ComplexSchur<ComplexMatrix> mutableComplexSchur(complex), constComplexSchur(constComplex);
  SelfAdjointEigenSolver<RealMatrix> mutableSelfAdjoint(real), constSelfAdjoint(constReal);
  VERIFY_IS_EQUAL(real_schur_calls, realBefore + 2);
  VERIFY_IS_EQUAL(complex_schur_calls, complexBefore + 2);
  VERIFY_IS_EQUAL(selfadjoint_calls, selfadjointBefore + 2);
  VERIFY_IS_EQUAL(mutableRealSchur.info(), NoConvergence);
  VERIFY_IS_EQUAL(constRealSchur.info(), NoConvergence);
  VERIFY_IS_EQUAL(mutableComplexSchur.info(), NoConvergence);
  VERIFY_IS_EQUAL(constComplexSchur.info(), NoConvergence);
  VERIFY_IS_EQUAL(mutableSelfAdjoint.info(), NoConvergence);
  VERIFY_IS_EQUAL(constSelfAdjoint.info(), NoConvergence);
  VERIFY_IS_EQUAL(real, constReal);
  VERIFY_IS_EQUAL(complex, constComplex);
}

EIGEN_DECLARE_TEST(inplace_decomposition_lapacke) {
  inplace_plain_lapacke_dispatch<ColMajor>();
  inplace_plain_lapacke_dispatch<RowMajor>();
}
