// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Tests the dispatch of coefficient-wise assignments to the AOCL vector math library (Assign_AOCL.h). Without
// AOCL, its entry points are stubbed with their documented semantics, dst[i] = f(src[i]), so that what is tested
// is which assignments Eigen dispatches and how.
#if defined(__has_include)
#if !__has_include("amdlibm.h")
#define EIGEN_TEST_AOCL_STUBS
#endif
#endif

#ifdef EIGEN_TEST_AOCL_STUBS
#include <cmath>
#define EIGEN_TEST_AOCL_UNARY_STUB(NAME, FUNC)     \
  static void NAME(int n, double* x, double* y) {  \
    for (int i = 0; i < n; ++i) y[i] = FUNC(x[i]); \
  }
#define EIGEN_TEST_AOCL_BINARY_STUB(NAME, FUNC)              \
  static void NAME(int n, double* x, double* z, double* y) { \
    for (int i = 0; i < n; ++i) y[i] = FUNC(x[i], z[i]);     \
  }
static double aocl_stub_add(double a, double b) { return a + b; }
EIGEN_TEST_AOCL_UNARY_STUB(amd_vrda_exp2, std::exp2)
EIGEN_TEST_AOCL_UNARY_STUB(amd_vrda_exp, std::exp)
EIGEN_TEST_AOCL_UNARY_STUB(amd_vrda_sin, std::sin)
EIGEN_TEST_AOCL_UNARY_STUB(amd_vrda_cos, std::cos)
EIGEN_TEST_AOCL_UNARY_STUB(amd_vrda_sqrt, std::sqrt)
EIGEN_TEST_AOCL_UNARY_STUB(amd_vrda_cbrt, std::cbrt)
EIGEN_TEST_AOCL_UNARY_STUB(amd_vrda_fabs, std::fabs)
EIGEN_TEST_AOCL_UNARY_STUB(amd_vrda_log, std::log)
EIGEN_TEST_AOCL_UNARY_STUB(amd_vrda_log10, std::log10)
EIGEN_TEST_AOCL_UNARY_STUB(amd_vrda_log2, std::log2)
EIGEN_TEST_AOCL_BINARY_STUB(amd_vrda_add, aocl_stub_add)
EIGEN_TEST_AOCL_BINARY_STUB(amd_vrda_pow, std::pow)
EIGEN_TEST_AOCL_BINARY_STUB(amd_vrda_fmax, std::fmax)
EIGEN_TEST_AOCL_BINARY_STUB(amd_vrda_fmin, std::fmin)
#endif

#define EIGEN_USE_AOCL_ALL
#include "main.h"

// Assignment to a plain object resizes it, as without AOCL.
template <int>
void aocl_vml_resize() {
  const Index n = 2 * EIGEN_AOCL_VML_THRESHOLD + 1;
  const VectorXd x = VectorXd::Random(n), y = VectorXd::Random(n);
  VectorXd r;
  r = x.array().exp();
  VERIFY_IS_EQUAL(r.size(), n);
  VERIFY_IS_APPROX(r, x.unaryExpr([](double v) { return std::exp(v); }));
  r.resize(0);
  r = x + y;
  VERIFY_IS_EQUAL(r.size(), n);
  VERIFY_IS_APPROX(r, x.binaryExpr(y, [](double a, double b) { return a + b; }));
}

// Operands that are not one contiguous array: blocks of a larger matrix, strided maps, and block destinations,
// whose coefficients outside the block must stay untouched.
template <int>
void aocl_vml_non_contiguous() {
  const Index n = 2 * EIGEN_AOCL_VML_THRESHOLD + 3, m = n - 7;
  const MatrixXd a = MatrixXd::Random(n, n), b = MatrixXd::Random(n, n);
  const MatrixXd aBlock = a.block(1, 2, m, m), bBlock = b.block(3, 0, m, m);
  const auto exp = [](double v) { return std::exp(v); };
  const auto sum = [](double x, double y) { return x + y; };

  MatrixXd r = a.block(1, 2, m, m).array().exp();
  VERIFY_IS_APPROX(r, aBlock.unaryExpr(exp));
  r = a.block(1, 2, m, m) + b.block(3, 0, m, m);
  VERIFY_IS_APPROX(r, aBlock.binaryExpr(bBlock, sum));

  MatrixXd d = MatrixXd::Zero(n, n);
  d.block(2, 1, m, m).array() = aBlock.array().exp();
  VERIFY_IS_APPROX(MatrixXd(d.block(2, 1, m, m)), aBlock.unaryExpr(exp));
  d.block(2, 1, m, m).setZero();
  VERIFY(d.isZero());
  d.block(2, 1, m, m) = aBlock + bBlock;
  VERIFY_IS_APPROX(MatrixXd(d.block(2, 1, m, m)), aBlock.binaryExpr(bBlock, sum));
  d.block(2, 1, m, m).setZero();
  VERIFY(d.isZero());

  const Map<const VectorXd, 0, InnerStride<2> > strided(a.data(), n);
  const VectorXd stridedCopy = strided;
  VectorXd v = strided.array().exp();
  VERIFY_IS_APPROX(v, stridedCopy.unaryExpr(exp));
}

EIGEN_DECLARE_TEST(aocl_vml) {
  CALL_SUBTEST_1(aocl_vml_resize<0>());
  CALL_SUBTEST_2(aocl_vml_non_contiguous<0>());
}
