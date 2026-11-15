// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_USE_LAPACKE
#include "main.h"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>

// A system <lapacke.h> included after Eigen must still declare the full LAPACKE API. Eigen's bundled subset header
// must neither claim the system header's include guards nor conflict with its declarations. The guard check needs no
// system header, so it runs everywhere; the conflict check below needs one. Only a declaration is needed, so the test
// does not link against LAPACKE.
#if !defined(EIGEN_USE_MKL) && !defined(EIGEN_LAPACKE_SYSTEM) && (defined(_LAPACKE_H_) || defined(_MKL_LAPACKE_H_))
#error "the bundled lapacke.h must not define the include guard of the system header"
#endif

#if !defined(EIGEN_USE_MKL) && !defined(EIGEN_LAPACKE_SYSTEM) && defined(__has_include)
#if __has_include(<lapacke.h>)
#include <lapacke.h>
static_assert(std::is_function<std::remove_pointer<decltype(&LAPACKE_dgesv)>::type>::value,
              "LAPACKE_dgesv is not declared after Eigen");
#endif
#endif

namespace {
int real_schur_calls = 0;
int complex_schur_calls = 0;
int selfadjoint_calls = 0;
int potrf_calls = 0;
int getrf_calls = 0;
int geqp3_calls = 0;
const double* factor_data = nullptr;
lapack_int factor_lda = 0;
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

// Record the matrix handed to the factorization and report success, leaving it unchanged.
extern "C" lapack_int LAPACKE_dpotrf(int, char, lapack_int, double* a, lapack_int lda) {
  ++potrf_calls;
  factor_data = a;
  factor_lda = lda;
  return 0;
}
extern "C" lapack_int LAPACKE_dgetrf(int, lapack_int m, lapack_int n, double* a, lapack_int lda, lapack_int* ipiv) {
  ++getrf_calls;
  factor_data = a;
  factor_lda = lda;
  for (lapack_int i = 0; i < (std::min)(m, n); ++i) ipiv[i] = i + 1;
  return 0;
}

// LAPACKE rejects a NaN input with info = -4 before touching the matrix. The mock does the same for any input, so the
// binding has to fall back to Eigen's own algorithm and still produce a valid decomposition.
extern "C" lapack_int LAPACKE_dgeqp3(int, lapack_int, lapack_int, double*, lapack_int, lapack_int*, double*) {
  ++geqp3_calls;
  return -4;
}

void colpiv_qr_lapacke_rejected_input() {
  const MatrixXd A = MatrixXd::Random(6, 4);
  const VectorXd b = A * VectorXd::Random(4);
  const int geqp3Before = geqp3_calls;
  ColPivHouseholderQR<MatrixXd> qr(A);
  VERIFY_IS_EQUAL(geqp3_calls, geqp3Before + 1);
  VERIFY_IS_EQUAL(qr.rank(), Index(4));
  const MatrixXd R = qr.matrixR().triangularView<Upper>();
  VERIFY_IS_APPROX(qr.householderQ() * R, A * qr.colsPermutation());
  VERIFY_IS_APPROX(A * qr.solve(b), b);
}

// LAPACK addresses a matrix by its leading dimension only, so an in-place factorization of a Ref with an inner stride
// other than 1 must hand it a contiguous copy, and a Ref with a unit inner stride its own storage.
template <int Options>
void inplace_lapacke_inner_stride() {
  using MatrixType = Matrix<double, Dynamic, Dynamic, Options>;
  using StridedRef = Ref<MatrixType, 0, Stride<Dynamic, Dynamic>>;
  const Index size = 4;
  for (Index inner : {Index(1), Index(2), Index(3)}) {
    const Stride<Dynamic, Dynamic> stride(inner * size + 2, inner);
    VectorXd storage = VectorXd::Random(stride.outer() * size);
    const VectorXd saved = storage;
    Map<MatrixType, 0, Stride<Dynamic, Dynamic>> a(storage.data(), size, size, stride);
    StridedRef ref(a);
    const lapack_int expected_lda = inner == 1 ? lapack_int(stride.outer()) : lapack_int(size);

    const int potrfBefore = potrf_calls;
    LLT<StridedRef> llt(ref);
    VERIFY_IS_EQUAL(potrf_calls, potrfBefore + 1);
    VERIFY_IS_EQUAL(llt.info(), Success);
    VERIFY_IS_EQUAL(factor_lda, expected_lda);
    VERIFY((factor_data == storage.data()) == (inner == 1));
    VERIFY_IS_EQUAL(storage, saved);

    const int getrfBefore = getrf_calls;
    PartialPivLU<StridedRef> lu(ref);
    VERIFY_IS_EQUAL(getrf_calls, getrfBefore + 1);
    VERIFY_IS_EQUAL(factor_lda, expected_lda);
    VERIFY((factor_data == storage.data()) == (inner == 1));
    VERIFY_IS_EQUAL(storage, saved);
  }
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
  colpiv_qr_lapacke_rejected_input();
  inplace_lapacke_inner_stride<ColMajor>();
  inplace_lapacke_inner_stride<RowMajor>();
}
