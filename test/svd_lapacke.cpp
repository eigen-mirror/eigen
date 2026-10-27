// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_USE_LAPACKE
#include "main.h"
#include <Eigen/SVD>

namespace {
// The plain copy of the matrix being decomposed, of the scalar type under test.
const void* expected_a = nullptr;
int svd_calls = 0;
bool args_match = false;

// Columns of U, or rows of V^T, that a LAPACK job references; 0 when it references none.
lapack_int job_extent(char job, lapack_int full, lapack_int thin) { return job == 'A' ? full : job == 'S' ? thin : 0; }

// Minimum leading dimension of a rows x cols array, or 1 for an array the job does not reference.
lapack_int min_ld(bool col_major, lapack_int rows, lapack_int cols) {
  return rows == 0 || cols == 0 ? 1 : col_major ? rows : cols;
}

// Whether the arguments describe the matrix being decomposed: a holds it in the given layout, and u and vt have
// leading dimensions the LAPACKE contract allows.
template <typename Scalar>
bool matches_expected(int layout, char jobu, char jobvt, lapack_int m, lapack_int n, const Scalar* a, lapack_int lda,
                      lapack_int ldu, lapack_int ldvt) {
  const auto& expected = *static_cast<const Matrix<Scalar, Dynamic, Dynamic>*>(expected_a);
  const bool col_major = layout == LAPACK_COL_MAJOR;
  const lapack_int k = numext::mini(m, n);
  // The binding decomposes a fresh compact copy. Requiring its exact leading dimension also keeps the reads below
  // inside the copy.
  if (m != expected.rows() || n != expected.cols() || lda != numext::maxi<lapack_int>(1, col_major ? m : n)) {
    return false;
  }
  if (ldu < min_ld(col_major, m, job_extent(jobu, m, k)) || ldvt < min_ld(col_major, job_extent(jobvt, n, k), n)) {
    return false;
  }
  for (lapack_int i = 0; i < m; ++i) {
    for (lapack_int j = 0; j < n; ++j) {
      if (a[col_major ? i + j * lda : i * lda + j] != expected(i, j)) return false;
    }
  }
  return true;
}

// Probes return a failure code, so Eigen reads no output but the singular values, which it checks for finiteness.
template <typename Scalar, typename RealScalar>
lapack_int svd_probe(int layout, char jobu, char jobvt, lapack_int m, lapack_int n, const Scalar* a, lapack_int lda,
                     RealScalar* s, lapack_int ldu, lapack_int ldvt) {
  ++svd_calls;
  args_match = expected_a != nullptr && matches_expected(layout, jobu, jobvt, m, n, a, lda, ldu, ldvt);
  std::fill_n(s, numext::mini(m, n), RealScalar(0));
  return 1;
}
}  // namespace

// The bindings take the address of every scalar type's routine, so an unoptimized build needs all of them.
// TYPE and RTYPE form declarations, not expressions.
// NOLINTBEGIN(bugprone-macro-parentheses)
#define EIGEN_SVD_PROBES(PREFIX, TYPE, RTYPE)                                                                    \
  extern "C" lapack_int LAPACKE_##PREFIX##gesvd(int layout, char jobu, char jobvt, lapack_int m, lapack_int n,   \
                                                TYPE* a, lapack_int lda, RTYPE* s, TYPE*, lapack_int ldu, TYPE*, \
                                                lapack_int ldvt, RTYPE*) {                                       \
    return svd_probe(layout, jobu, jobvt, m, n, a, lda, s, ldu, ldvt);                                           \
  }                                                                                                              \
  extern "C" lapack_int LAPACKE_##PREFIX##gesdd(int layout, char jobz, lapack_int m, lapack_int n, TYPE* a,      \
                                                lapack_int lda, RTYPE* s, TYPE*, lapack_int ldu, TYPE*,          \
                                                lapack_int ldvt) {                                               \
    return svd_probe(layout, jobz, jobz, m, n, a, lda, s, ldu, ldvt);                                            \
  }
// NOLINTEND(bugprone-macro-parentheses)
EIGEN_SVD_PROBES(s, float, float)
EIGEN_SVD_PROBES(d, double, double)
EIGEN_SVD_PROBES(c, lapack_complex_float, float)
EIGEN_SVD_PROBES(z, lapack_complex_double, double)
#undef EIGEN_SVD_PROBES

template <typename SVD, typename Input>
void check_lapacke_input(const Input& input) {
  const Matrix<typename SVD::Scalar, Dynamic, Dynamic> expected = input;
  expected_a = &expected;
  args_match = false;
  const int calls_before = svd_calls;
  SVD svd(input);
  expected_a = nullptr;
  VERIFY_IS_EQUAL(svd_calls, calls_before + 1);
  VERIFY(args_match);
  VERIFY_IS_EQUAL(svd.info(), NoConvergence);
}

// LAPACK rejects the zero leading dimensions of an empty input, and given valid ones it leaves U and VT unwritten. The
// bindings must not call it, and must return the native path's identity unitaries.
template <typename SVD>
void check_lapacke_empty(Index rows, Index cols) {
  using MatrixType = typename SVD::MatrixType;
  const int calls_before = svd_calls;
  SVD svd(MatrixType(rows, cols));
  VERIFY_IS_EQUAL(svd_calls, calls_before);
  VERIFY_IS_EQUAL(svd.info(), Success);
  VERIFY_IS_EQUAL(svd.singularValues().size(), 0);
  if (svd.computeU()) VERIFY_IS_EQUAL(svd.matrixU(), MatrixType::Identity(rows, svd.matrixU().cols()));
  if (svd.computeV()) VERIFY_IS_EQUAL(svd.matrixV(), MatrixType::Identity(cols, svd.matrixV().cols()));
}

// The bindings hand LAPACK a copy in MatrixType's layout (#3217): an input whose own layout or outer stride differs
// must not leak into the arguments that describe the copy.
template <typename SVD>
void svd_lapacke_input_layout() {
  using Scalar = typename SVD::Scalar;
  using ColMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  using RowMatrix = Matrix<Scalar, Dynamic, Dynamic, RowMajor>;
  const Index m = internal::random<Index>(4, 12), n = internal::random<Index>(1, m - 1);
  const ColMatrix tall = ColMatrix::Random(m, n), wide = ColMatrix::Random(n, m), square = ColMatrix::Random(m, m);
  const ColMatrix big = ColMatrix::Random(m + 3, m + 2);
  const RowMatrix row_major = tall, big_row_major = big;
  check_lapacke_input<SVD>(tall);
  check_lapacke_input<SVD>(row_major);
  check_lapacke_input<SVD>(tall.transpose());
  check_lapacke_input<SVD>(wide.transpose());
  check_lapacke_input<SVD>(square.transpose());
  check_lapacke_input<SVD>(big.block(1, 2, m, n));
  check_lapacke_input<SVD>(big.block(1, 2, n, m).transpose());
  check_lapacke_input<SVD>(big_row_major.block(1, 2, m, n));
  check_lapacke_input<SVD>(Scalar(2) * tall);
  check_lapacke_empty<SVD>(0, 0);
  check_lapacke_empty<SVD>(0, n);
  check_lapacke_empty<SVD>(m, 0);
}

template <typename Scalar, int Options>
void svd_lapacke_options() {
  using ColMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  using RowMatrix = Matrix<Scalar, Dynamic, Dynamic, RowMajor>;
  svd_lapacke_input_layout<JacobiSVD<ColMatrix, Options>>();
  svd_lapacke_input_layout<JacobiSVD<RowMatrix, Options>>();
  svd_lapacke_input_layout<BDCSVD<ColMatrix, Options>>();
  svd_lapacke_input_layout<BDCSVD<RowMatrix, Options>>();
}

// ThinU | FullV is the case where the BDCSVD binding computes U in a local buffer.
template <typename Scalar>
void svd_lapacke() {
  svd_lapacke_options<Scalar, 0>();
  svd_lapacke_options<Scalar, ComputeThinU | ComputeThinV>();
  svd_lapacke_options<Scalar, ComputeFullU | ComputeFullV>();
  svd_lapacke_options<Scalar, ComputeThinU | ComputeFullV>();
}

EIGEN_DECLARE_TEST(svd_lapacke) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(svd_lapacke<double>());
    CALL_SUBTEST_2(svd_lapacke<float>());
    CALL_SUBTEST_3(svd_lapacke<std::complex<double>>());
    CALL_SUBTEST_4(svd_lapacke<std::complex<float>>());
  }
}
