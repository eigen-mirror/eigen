// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// GEMM assignment, C = A * B, which kernels that can overwrite (gebp_kernel_overwrites) compute without zeroing C
// first. Every overwritten coefficient is alpha * acc + 0, so the result must be bitwise the one that zeroing C and
// accumulating gives; C starts out NaN, so a coefficient not written, or computed from C's old contents, fails the
// comparison. Small blocking budgets split the products over several depth, row and column blocks, of which only
// the first depth block may overwrite.

#define EIGEN_DEBUG_SMALL_PRODUCT_BLOCKS
#include "main.h"
#include "product_test_helpers.h"

#include <limits>

template <typename Scalar>
Scalar nan_value() {
  return Scalar(std::numeric_limits<typename NumTraits<Scalar>::Real>::quiet_NaN());
}

// Bitwise equality of every coefficient, so that +0 and -0 differ.
template <typename Real>
bool bits_equal(const Real& x, const Real& y) {
  using Bits = typename numext::get_integer_by_size<sizeof(Real)>::unsigned_type;
  return numext::bit_cast<Bits>(x) == numext::bit_cast<Bits>(y);
}
template <typename Real>
bool bits_equal(const std::complex<Real>& x, const std::complex<Real>& y) {
  return bits_equal(x.real(), y.real()) && bits_equal(x.imag(), y.imag());
}
template <typename A, typename B>
bool bitwise_equal(const DenseBase<A>& a, const DenseBase<B>& b) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
  for (Index j = 0; j < a.cols(); ++j)
    for (Index i = 0; i < a.rows(); ++i)
      if (!bits_equal(a.derived().coeff(i, j), b.derived().coeff(i, j))) return false;
  return true;
}

// dst = scale * a * b through assignment, from a NaN-filled destination, against the zero-then-accumulate result and
// the coefficient-based product. The two must be bitwise equal where the product runs GEMM. The coefficient-based
// product and the GEMV fallbacks compile `=` and `+=` separately, and those may round differently.
template <typename Dst, typename Lhs, typename Rhs, typename Scalar>
void check_assign(Dst& dst, const Lhs& a, const Rhs& b, const Scalar& scale) {
  using ScaledLhs = internal::remove_all_t<decltype(scale * a)>;
  using Impl = internal::generic_product_impl<ScaledLhs, Rhs, DenseShape, DenseShape, GemmProduct>;
  const bool gemm = dst.rows() > 1 && dst.cols() > 1 && !Impl::useRuntimeCoeffBasedProduct(dst, b);
  // Accumulated into dst itself, so that it takes the same kernel (or external BLAS) as the assignment: a row-major or
  // strided destination may take another, which rounds differently.
  using Plain = typename Dst::PlainObject;
  dst.setZero();
  dst.noalias() += scale * a * b;
  const Plain expected = dst;

  dst.setConstant(nan_value<typename Dst::Scalar>());
  dst.noalias() = scale * a * b;
  VERIFY(dst.allFinite());
  if (gemm)
    VERIFY(bitwise_equal(dst, expected));
  else
    VERIFY(verifyProduct(dst, expected, scale * a, b));

  // Assignment without noalias goes through a temporary, whose storage order is the product's.
  dst.setConstant(nan_value<typename Dst::Scalar>());
  dst = scale * a * b;
  VERIFY(dst.allFinite());
  VERIFY(verifyProduct(dst, expected, scale * a, b));

  if (a.cols() > 0) VERIFY(verifyProduct(expected, Plain((scale * a).lazyProduct(b)), scale * a, b));
}

// Sizes around the packet widths and the register and cache blocks of the kernels, at depths below, at and above
// the small blocking's depth block.
template <typename Scalar, int DstOrder>
void assign_shapes() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using DstMat = Matrix<Scalar, Dynamic, Dynamic, DstOrder>;
  const Index sizes[] = {1, 3, 16, 17, 32, 33, 48, 65, 129};
  const Index depths[] = {1, 25, 64, 130, 300};
  for (Index m : sizes)
    for (Index n : sizes)
      for (Index k : depths) {
        if (m * n * k > 1000000) continue;
        const Mat a = Mat::Random(m, k), b = Mat::Random(k, n);
        DstMat c(m, n);
        check_assign(c, a, b, Scalar(1));
        if (m > 1 && n > 1) check_assign(c, a, b, Scalar(2));
      }

  // Transposed operands, and a column-major result read back through an outer stride with sentinel padding.
  const Index m = 97, n = 70, k = 300;
  const Mat at = Mat::Random(k, m), bt = Mat::Random(n, k);
  DstMat c(m, n);
  check_assign(c, at.transpose(), bt.transpose(), Scalar(1));

  const Mat a = Mat::Random(m, k), b = Mat::Random(k, n);
  Mat storage = Mat::Constant(m + 5, n, Scalar(7));
  Map<Mat, 0, OuterStride<>> mapped(storage.data(), m, n, OuterStride<>(m + 5));
  check_assign(mapped, a, b, Scalar(1));
  VERIFY((storage.bottomRows(5).array() == Scalar(7)).all());
}

// Outer stride 1 and inner stride n: a row-major layout seen through column-major strides, so the kernel's C has a
// unit column stride.
template <typename Scalar>
void assign_unit_col_stride() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  for (Index m : {Index(17), Index(65), Index(129)})
    for (Index n : {Index(9), Index(40), Index(129)})
      for (Index k : {Index(25), Index(300)}) {
        const Mat a = Mat::Random(m, k), b = Mat::Random(k, n);
        Matrix<Scalar, Dynamic, 1> storage = Matrix<Scalar, Dynamic, 1>::Constant(m * n + 3, Scalar(7));
        Map<Mat, 0, Stride<Dynamic, Dynamic>> mapped(storage.data(), m, n, Stride<Dynamic, Dynamic>(1, n));
        check_assign(mapped, a, b, Scalar(1));
        for (Index i = m * n; i < storage.size(); ++i) VERIFY_IS_EQUAL(storage(i), Scalar(7));
      }
}

// A depth past the blocking's depth block: only the first pass overwrites, the later ones accumulate.
template <typename Scalar>
void assign_depth_passes() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using Blocking = internal::gemm_blocking_space<ColMajor, Scalar, Scalar, Dynamic, Dynamic, Dynamic>;
  for (Index m : {Index(33), Index(129)}) {
    const Index n = 40;
    Index k = 2 * Blocking(m, n, 1 << 20, 1, true).kc() + 7;
    VERIFY(Blocking(m, n, k, 1, true).kc() < k);
    const Mat a = Mat::Random(m, k), b = Mat::Random(k, n);
    Mat c(m, n);
    check_assign(c, a, b, Scalar(1));
  }
}

// An inner stride of two: the result coefficients are every other scalar, and the ones in between stay untouched.
template <typename Scalar>
void assign_inner_stride() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  const Index m = 65, n = 40, k = 150;
  const Mat a = Mat::Random(m, k), b = Mat::Random(k, n);
  Matrix<Scalar, Dynamic, 1> storage = Matrix<Scalar, Dynamic, 1>::Constant(2 * m * n, Scalar(7));
  Map<Mat, 0, Stride<Dynamic, 2>> mapped(storage.data(), m, n, Stride<Dynamic, 2>(2 * m, 2));
  check_assign(mapped, a, b, Scalar(1));
  for (Index i = 1; i < storage.size(); i += 2) VERIFY_IS_EQUAL(storage(i), Scalar(7));
}

// Conjugated operands and scalar factors on either side, and a real operand on either side of a complex product.
template <typename Scalar>
void assign_conjugate() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using RealMat = Matrix<RealScalar, Dynamic, Dynamic, ColMajor>;
  const Index m = 70, n = 45, k = 140;
  const Mat a = Mat::Random(m, k), b = Mat::Random(k, n), at = Mat::Random(k, m);
  Mat c(m, n);
  const Scalar s = internal::random<Scalar>();
  check_assign(c, a.conjugate(), b, s);
  check_assign(c, at.adjoint(), b, s);
  check_assign(c, a, (s * b).conjugate(), Scalar(1));
  const RealMat ra = RealMat::Random(m, k), rb = RealMat::Random(k, n);
  check_assign(c, ra, b, RealScalar(1));
  check_assign(c, a, rb, RealScalar(1));
}

// Accumulation is unchanged: += and -= onto a nonzero result.
template <typename Scalar>
void accumulate_unchanged() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  const Index m = 66, n = 33, k = 200;
  const Mat a = Mat::Random(m, k), b = Mat::Random(k, n), c0 = Mat::Random(m, n);
  Mat c = c0;
  c.noalias() += a * b;
  VERIFY(verifyProduct(Mat(c - c0), Mat(a.lazyProduct(b)), a, b));
  c = c0;
  c.noalias() -= a * b;
  VERIFY(verifyProduct(Mat(c0 - c), Mat(a.lazyProduct(b)), a, b));
}

// alpha * acc is -0 for alpha = -1 and an all-zero product, and overwriting must keep the sign that accumulating into
// a zeroed result gives: +0 for real scalars (0 + (-0)); for complex ones it depends on the packet pmadd (the AVX and
// SSE fmaddsub forms give a -0 real part). Odd sizes reach the scalar row and column tails as well as the packet
// stores.
template <typename Scalar>
void assign_signed_zero() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  for (Index size : {64, 67}) {
    const Mat a = Mat::Ones(size, size), b = Mat::Zero(size, size);
    Mat expected = Mat::Zero(size, size);
    expected.noalias() += Scalar(-1) * a * b;
    Mat c = Mat::Constant(size, size, nan_value<Scalar>());
    c.noalias() = Scalar(-1) * a * b;
    VERIFY(bitwise_equal(c, expected));
    for (Index j = 0; j < size; ++j)
      for (Index i = 0; i < size; ++i)
        VERIFY(c(i, j) == Scalar(0) && (NumTraits<Scalar>::IsComplex || !(std::signbit)(numext::real(c(i, j)))));
  }
}

// The driver's own contract: overwriting assigns, so a zero factor or an empty depth zeroes the result. The result is
// a padded column-major block whatever the default storage order, and the padding must stay untouched. Through
// gemm_run_overwrite, which takes run_overwrite where the driver has one (not the external-BLAS one).
template <typename Scalar>
void driver_overwrite() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using Gemm =
      internal::general_matrix_matrix_product<Index, Scalar, ColMajor, false, Scalar, ColMajor, false, ColMajor, 1>;
  using Blocking = internal::gemm_blocking_space<ColMajor, Scalar, Scalar, Dynamic, Dynamic, Dynamic>;
  const Index m = 40, n = 30, k = 20, ldc = m + 5;
  const Mat a = Mat::Random(m, k), b = Mat::Random(k, n);
  Mat storage(ldc, n);
  internal::GemmParallelInfo<Index>* no_info = nullptr;
  for (Index depth : {Index(0), k}) {
    for (Scalar alpha : {Scalar(0), Scalar(1)}) {
      Blocking blocking(m, n, depth, 1, true);
      storage.setConstant(Scalar(7));
      storage.topRows(m).setConstant(nan_value<Scalar>());
      internal::gemm_run_overwrite<Gemm>(m, n, depth, a.data(), m, b.data(), k, storage.data(), Index(1), ldc, alpha,
                                         blocking, no_info);
      const Mat expected = (depth == 0 || alpha == Scalar(0)) ? Mat(Mat::Zero(m, n)) : Mat(a.lazyProduct(b));
      VERIFY(storage.topRows(m).allFinite());
      VERIFY(verifyProduct(Mat(storage.topRows(m)), expected, a, b));
      VERIFY((storage.bottomRows(ldc - m).array() == Scalar(7)).all());
    }
  }
  // An empty result is left alone.
  Blocking blocking(m, 0, k, 1, true);
  internal::gemm_run_overwrite<Gemm>(m, Index(0), k, a.data(), m, b.data(), k, storage.data(), Index(1), ldc, Scalar(1),
                                     blocking, no_info);
}

// A kernel from before the overwrite contract: only the untagged call, which accumulates.
struct accumulate_only_kernel {
  int* calls;
  template <typename... Args>
  void operator()(const Args&...) const {
    ++*calls;
  }
};

// A general_matrix_matrix_product specialization from before the contract: no run_overwrite and no CanOverwrite.
struct accumulate_only_gemm {
  static int calls;
  template <typename Scalar, typename Blocking>
  static void run(Index rows, Index cols, Index, const Scalar*, Index, const Scalar*, Index, Scalar* res, Index,
                  Index resStride, Scalar, Blocking&, internal::GemmParallelInfo<Index>*) {
    ++calls;
    for (Index j = 0; j < cols; ++j)
      for (Index i = 0; i < rows; ++i) res[i + j * resStride] += Scalar(1);
  }
};
int accumulate_only_gemm::calls = 0;

// A scalar from before the overwrite contract whose product is an expression: it supports `dst += a * c` but not
// adding that expression to a scalar, so accumulating stores must ask for nothing more.
struct et_scalar;
struct et_product {
  double v;
};
struct et_scalar {
  double v;
  et_scalar& operator+=(const et_product& p) {
    v += p.v;
    return *this;
  }
};
et_product operator*(const et_scalar& a, const et_scalar& b) { return et_product{a.v * b.v}; }

// Kernels with and without the profitability hook.
struct pays_never {
  static bool overwrite_pays(Index, Index, Index) { return false; }
};
struct no_pays_hook {};

#ifdef EIGEN_VECTORIZE_SME
// A kernel from before the contract that reads its LHS in place: only the untagged run_direct_lhs.
struct accumulate_only_direct_kernel {
  int* calls;
  template <typename... Args>
  void run_direct_lhs(const Args&...) const {
    ++*calls;
  }
};
#endif

// Old interfaces keep working: the kernel only ever gets the untagged call, and a driver without run_overwrite is
// overwritten by zeroing the result and accumulating into it.
void old_interfaces() {
  et_scalar dst{1}, alpha{2}, acc{3};
  internal::gebp_store_scalar<false>(true, dst, alpha, acc);
  VERIFY_IS_EQUAL(dst.v, 7.0);

  VERIFY((internal::gebp_overwrite_pays<no_pays_hook>(Index(64), Index(64), Index(64))));
  VERIFY(!(internal::gebp_overwrite_pays<pays_never>(Index(64), Index(64), Index(64))));

#ifdef EIGEN_VECTORIZE_SME
  int direct_calls = 0;
  accumulate_only_direct_kernel direct{&direct_calls};
  internal::sme_direct_lhs_call(std::false_type(), direct, true, 1, 2, 3);
  VERIFY_IS_EQUAL(direct_calls, 1);
#endif

  int calls = 0;
  accumulate_only_kernel kernel{&calls};
  internal::gebp_run(std::false_type(), kernel, true, 1, 2, 3);
  VERIFY_IS_EQUAL(calls, 1);
  STATIC_CHECK(!(internal::gebp_kernel_overwrites<accumulate_only_kernel>::value));

  STATIC_CHECK(!(internal::gemm_can_overwrite<accumulate_only_gemm>::value));
  const Index m = 6, n = 4, k = 3, ldc = 8;
  Matrix<float, Dynamic, 1> res = Matrix<float, Dynamic, 1>::Constant(ldc * n, nan_value<float>());
  internal::gemm_blocking_space<ColMajor, float, float, Dynamic, Dynamic, Dynamic> blocking(m, n, k, 1, true);
  const float* none = nullptr;
  internal::GemmParallelInfo<Index>* no_info = nullptr;
  accumulate_only_gemm::calls = 0;
  internal::gemm_run_overwrite<accumulate_only_gemm>(m, n, k, none, m, none, k, res.data(), Index(1), ldc, 1.f,
                                                     blocking, no_info);
  VERIFY_IS_EQUAL(accumulate_only_gemm::calls, 1);
  for (Index j = 0; j < n; ++j)
    for (Index i = 0; i < ldc; ++i) VERIFY(i < m ? res(i + j * ldc) == 1.f : (numext::isnan)(res(i + j * ldc)));
}

// A complex-like custom scalar over a built-in real type.
struct custom_complex {
  double re, im;
};

// int and half have no architecture kernel, so they take the generic one: built-in arithmetic opts in, half and
// custom scalars do not (assigning them is not known to match accumulating onto zero). Architecture kernels opt in
// where they implement it.
template <typename Lhs, typename Rhs = Lhs>
using kernel_for =
    internal::gebp_kernel<Lhs, Rhs, Index, internal::blas_data_mapper<Lhs, Index, ColMajor>,
                          internal::gebp_traits<Lhs, Rhs>::mr, internal::gebp_traits<Lhs, Rhs>::nr, false, false>;
void capability_trait() {
  STATIC_CHECK((internal::gebp_kernel_overwrites<kernel_for<int>>::value));
  STATIC_CHECK(!(internal::gebp_kernel_overwrites<kernel_for<half>>::value));
  STATIC_CHECK((internal::gebp_overwritable_scalar<std::complex<double>>::value));
  STATIC_CHECK(!(internal::gebp_overwritable_scalar<custom_complex>::value));
#ifdef EIGEN_USE_BLAS
  // With beta = 0, ?gemm may form alpha * t and return -0 where zero-then-accumulate gives +0 (reference BLAS does
  // in its transposed branches), so the external-BLAS driver keeps zeroing first.
  STATIC_CHECK(
      !(internal::gemm_can_overwrite<internal::general_matrix_matrix_product<Index, float, ColMajor, false, float,
                                                                             ColMajor, false, ColMajor, 1>>::value));
  STATIC_CHECK(
      !(internal::gemm_can_overwrite<internal::general_matrix_matrix_product<Index, double, RowMajor, false, double,
                                                                             ColMajor, false, ColMajor, 1>>::value));
#endif
#ifdef EIGEN_VECTORIZE_SME
  STATIC_CHECK((internal::gebp_kernel_overwrites<kernel_for<float>>::value));
  STATIC_CHECK((internal::gebp_kernel_overwrites<kernel_for<std::complex<float>>>::value));
  // Complex results overwrite up to 128 KB.
  VERIFY((kernel_for<std::complex<float>>::overwrite_pays(128, 128, 64)));
  VERIFY(!(kernel_for<std::complex<float>>::overwrite_pays(128, 129, 64)));
#ifdef EIGEN_VECTORIZE_SME_F64F64
  // 64-bit results overwrite up to 128 KB; a 32-bit Index must not overflow forming the byte count.
  using DoubleKernel = internal::gebp_kernel<double, double, int, internal::blas_data_mapper<double, int, ColMajor>,
                                             internal::gebp_traits<double, double>::mr,
                                             internal::gebp_traits<double, double>::nr, false, false>;
  VERIFY(DoubleKernel::overwrite_pays(128, 128, 64));
  VERIFY(!DoubleKernel::overwrite_pays(128, 129, 64));
  VERIFY(!DoubleKernel::overwrite_pays(16384, 16384, 64));
  VERIFY((kernel_for<std::complex<double>>::overwrite_pays(128, 64, 64)));
  VERIFY(!(kernel_for<std::complex<double>>::overwrite_pays(128, 65, 64)));
#endif
#endif
}

EIGEN_DECLARE_TEST(product_assign) {
  CALL_SUBTEST_1(capability_trait());
  CALL_SUBTEST_1(old_interfaces());
  CALL_SUBTEST_1((assign_shapes<float, ColMajor>()));
  CALL_SUBTEST_1((assign_shapes<float, RowMajor>()));
  CALL_SUBTEST_1(assign_inner_stride<float>());
  CALL_SUBTEST_1(assign_unit_col_stride<float>());
  CALL_SUBTEST_1(assign_depth_passes<float>());
  CALL_SUBTEST_1(assign_signed_zero<float>());
  CALL_SUBTEST_1(accumulate_unchanged<float>());
  CALL_SUBTEST_1(driver_overwrite<float>());
  CALL_SUBTEST_2((assign_shapes<double, ColMajor>()));
  CALL_SUBTEST_2((assign_shapes<double, RowMajor>()));
  CALL_SUBTEST_2(assign_inner_stride<double>());
  CALL_SUBTEST_2(assign_unit_col_stride<double>());
  CALL_SUBTEST_2(assign_depth_passes<double>());
  CALL_SUBTEST_2(assign_signed_zero<double>());
  CALL_SUBTEST_2(driver_overwrite<double>());
  CALL_SUBTEST_3((assign_shapes<std::complex<float>, ColMajor>()));
  CALL_SUBTEST_3((assign_shapes<std::complex<float>, RowMajor>()));
  CALL_SUBTEST_3(assign_conjugate<std::complex<float>>());
  CALL_SUBTEST_3(assign_inner_stride<std::complex<float>>());
  CALL_SUBTEST_3(assign_unit_col_stride<std::complex<float>>());
  CALL_SUBTEST_3(assign_depth_passes<std::complex<float>>());
  CALL_SUBTEST_3(assign_signed_zero<std::complex<float>>());
  CALL_SUBTEST_3(accumulate_unchanged<std::complex<float>>());
  CALL_SUBTEST_3(driver_overwrite<std::complex<float>>());
  CALL_SUBTEST_4((assign_shapes<std::complex<double>, ColMajor>()));
  CALL_SUBTEST_4((assign_shapes<std::complex<double>, RowMajor>()));
  CALL_SUBTEST_4(assign_conjugate<std::complex<double>>());
  CALL_SUBTEST_4(assign_inner_stride<std::complex<double>>());
  CALL_SUBTEST_4(assign_unit_col_stride<std::complex<double>>());
  CALL_SUBTEST_4(assign_depth_passes<std::complex<double>>());
  CALL_SUBTEST_4(assign_signed_zero<std::complex<double>>());
  CALL_SUBTEST_4(driver_overwrite<std::complex<double>>());
}
