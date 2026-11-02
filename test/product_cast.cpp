// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define TEST_ENABLE_TEMPORARY_TRACKING
#define EIGEN_RUNTIME_NO_MALLOC
#define TEST_IGNORE_STACK_ALLOCATED_TEMPORARY

#include "main.h"
#include "product_test_helpers.h"

// GEMM operands of the form A.cast<T>() are packed straight from A, and half and bfloat16 GEMMs accumulate in float.

// Temporaries per cast operand: none, or one with EIGEN_USE_BLAS or on SME, which materialize the cast.
#if !defined(EIGEN_USE_BLAS) && !defined(EIGEN_VECTORIZE_SME)
constexpr int kCastTemporaries = 0;
#else
constexpr int kCastTemporaries = 1;
#endif

template <typename Src, int Order>
Matrix<Src, Dynamic, Dynamic, Order> random_small_integers(Index rows, Index cols, int bound) {
  Matrix<Src, Dynamic, Dynamic, Order> m(rows, cols);
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i) m(i, j) = Src(internal::random<int>(-bound, bound));
  return m;
}

// Integers in [-bound, bound], or for a floating-point Src, fractions there that a narrowing cast rounds.
template <typename Src, int Order>
Matrix<Src, Dynamic, Dynamic, Order> random_operand(Index rows, Index cols, int bound) {
  if (NumTraits<Src>::IsInteger) return random_small_integers<Src, Order>(rows, cols, bound);
  return Matrix<Src, Dynamic, Dynamic, Order>::Random(rows, cols) * Src(bound);
}

// The lazily cast operands pack to the same blocks as the materialized casts, so the products agree bit for bit.
template <typename Src, typename Dst, int LhsOrder, int RhsOrder, int DstOrder>
void lazy_cast_product(Index rows, Index cols, Index depth) {
  using DstMat = Matrix<Dst, Dynamic, Dynamic, DstOrder>;
  using PlainDst = Matrix<Dst, Dynamic, Dynamic>;
  Matrix<Src, Dynamic, Dynamic, LhsOrder> a = random_operand<Src, LhsOrder>(rows + 3, depth + 2, 8);
  Matrix<Src, Dynamic, Dynamic, RhsOrder> b = random_operand<Src, RhsOrder>(depth + 1, cols + 4, 8);
  // Strided operands: blocks of larger matrices.
  auto A = a.block(1, 2, rows, depth);
  auto B = b.block(1, 3, depth, cols);
  Matrix<Dst, Dynamic, Dynamic, LhsOrder> Ad = A.template cast<Dst>();
  Matrix<Dst, Dynamic, Dynamic, RhsOrder> Bd = B.template cast<Dst>();

  DstMat ref(rows, cols), dst(rows, cols);
  ref.noalias() = Ad * Bd;
  dst.setRandom();
  VERIFY_EVALUATION_COUNT(dst.noalias() = A.template cast<Dst>() * B.template cast<Dst>(), 2 * kCastTemporaries);
  VERIFY_IS_EQUAL(dst, ref);
  DstMat ref_update = ref;
  ref_update.noalias() += Ad * Bd;
  VERIFY_EVALUATION_COUNT(dst.noalias() += A.template cast<Dst>() * B.template cast<Dst>(), 2 * kCastTemporaries);
  VERIFY_IS_EQUAL(dst, ref_update);
  ref_update.noalias() -= Ad * Bd;
  VERIFY_EVALUATION_COUNT(dst.noalias() -= A.template cast<Dst>() * B.template cast<Dst>(), 2 * kCastTemporaries);
  VERIFY_IS_EQUAL(dst, ref_update);

  // One side already in the kernel's type.
  ref.noalias() = Ad * Bd;
  VERIFY_EVALUATION_COUNT(dst.noalias() = A.template cast<Dst>() * Bd, kCastTemporaries);
  VERIFY_IS_EQUAL(dst, ref);
  VERIFY_EVALUATION_COUNT(dst.noalias() = Ad * B.template cast<Dst>(), kCastTemporaries);
  VERIFY_IS_EQUAL(dst, ref);
  // A transposed operand below the cast.
  DstMat dst_t(cols, rows), ref_t(cols, rows);
  ref_t.noalias() = Bd.transpose() * Ad.transpose();
  VERIFY_EVALUATION_COUNT(dst_t.noalias() = B.transpose().template cast<Dst>() * A.transpose().template cast<Dst>(),
                          2 * kCastTemporaries);
  VERIFY_IS_EQUAL(dst_t, ref_t);

  // A scalar factor below the cast is not hoisted out of it, and the operand is materialized as before.
  ref.noalias() = (Src(2) * A).template cast<Dst>().eval() * Bd;
  dst.noalias() = (Src(2) * A).template cast<Dst>() * B.template cast<Dst>();
  VERIFY_IS_EQUAL(dst, ref);
  // A scalar factor above the cast multiplies the result.
  ref.noalias() = (Dst(3) * Ad) * Bd;
  dst.noalias() = (Dst(3) * A.template cast<Dst>()) * B.template cast<Dst>();
  VERIFY_IS_EQUAL(dst, ref);
  // Not a GEMM: the GEMV fallback for a single result column, which sums in another order than the GEMV kernel.
  PlainDst col(rows, 1);
  col.noalias() = A.template cast<Dst>() * B.col(0).template cast<Dst>();
  VERIFY_IS_APPROX(col, (Ad * Bd.col(0)).eval());
}

template <typename Src, typename Dst>
void lazy_cast_products() {
  const Index rows = internal::random<Index>(20, 300);
  const Index cols = internal::random<Index>(20, 300);
  const Index depth = internal::random<Index>(20, 300);
  CALL_SUBTEST((lazy_cast_product<Src, Dst, ColMajor, ColMajor, ColMajor>(rows, cols, depth)));
  CALL_SUBTEST((lazy_cast_product<Src, Dst, RowMajor, ColMajor, ColMajor>(rows, cols, depth)));
  CALL_SUBTEST((lazy_cast_product<Src, Dst, ColMajor, RowMajor, RowMajor>(rows, cols, depth)));
  CALL_SUBTEST((lazy_cast_product<Src, Dst, RowMajor, RowMajor, RowMajor>(rows, cols, depth)));
  // Sizes that are not a multiple of the packet or the register block.
  CALL_SUBTEST((lazy_cast_product<Src, Dst, ColMajor, RowMajor, ColMajor>(17, 13, 9)));
  CALL_SUBTEST((lazy_cast_product<Src, Dst, RowMajor, ColMajor, RowMajor>(31, 7, 23)));
}

// A half or bfloat16 GEMM sums in float and rounds each coefficient once.
template <typename T, int LhsOrder, int RhsOrder, int DstOrder>
void float_accumulation(Index rows, Index cols, Index depth) {
  using Mat = Matrix<T, Dynamic, Dynamic, DstOrder>;
  // |sum| <= 256 * depth < 2^24 stays an integer that float holds exactly, so the reference does not depend on the
  // order. Its spread of about 16^2 / 3 * sqrt(depth) exceeds half's and bfloat16's exact integers, so a sum in T
  // rounds.
  Matrix<T, Dynamic, Dynamic, LhsOrder> A = random_small_integers<T, LhsOrder>(rows, depth, 16);
  Matrix<T, Dynamic, Dynamic, RhsOrder> B = random_small_integers<T, RhsOrder>(depth, cols, 16);
  MatrixXf exact = A.template cast<float>().eval() * B.template cast<float>().eval();

  Mat C(rows, cols);
  C.setConstant(T(1000));
  C.noalias() = A * B;
  VERIFY_IS_EQUAL(C, exact.cast<T>().eval());
  C = A * B;
  VERIFY_IS_EQUAL(C, exact.cast<T>().eval());

  Mat C0 = random_small_integers<T, DstOrder>(rows, cols, 64);
  C = C0;
  C.noalias() += A * B;
  VERIFY_IS_EQUAL(C, (C0.template cast<float>() + exact).template cast<T>().eval());
  C = C0;
  C.noalias() -= A * B;
  VERIFY_IS_EQUAL(C, (C0.template cast<float>() - exact).template cast<T>().eval());

  // Strided destination and operands.
  Mat big = Mat::Constant(rows + 2, cols + 3, T(7));
  big.block(1, 2, rows, cols).noalias() = A.leftCols(depth) * B.topRows(depth);
  VERIFY_IS_EQUAL(big.block(1, 2, rows, cols), exact.cast<T>().eval());
  VERIFY_IS_EQUAL(big.row(0), Mat::Constant(1, cols + 3, T(7)));
  VERIFY_IS_EQUAL(big.col(0), Mat::Constant(rows + 2, 1, T(7)));
  VERIFY_IS_EQUAL(big.rightCols(1), Mat::Constant(rows + 2, 1, T(7)));
  VERIFY_IS_EQUAL(big.bottomRows(1), Mat::Constant(1, cols + 3, T(7)));
}

template <typename T>
void float_accumulations() {
  const Index rows = internal::random<Index>(20, 200);
  const Index cols = internal::random<Index>(20, 200);
  const Index depth = internal::random<Index>(20, 600);
  CALL_SUBTEST((float_accumulation<T, ColMajor, ColMajor, ColMajor>(rows, cols, depth)));
  CALL_SUBTEST((float_accumulation<T, RowMajor, ColMajor, RowMajor>(rows, cols, depth)));
  CALL_SUBTEST((float_accumulation<T, RowMajor, RowMajor, ColMajor>(rows, cols, depth)));
  CALL_SUBTEST((float_accumulation<T, ColMajor, RowMajor, RowMajor>(13, 9, 31)));
}

template <typename T>
void float_accumulation_depth() {
  // Summed in T, a run of ones stalls where T's spacing reaches 2: at 2048 for half and 256 for bfloat16.
  const Index depth = std::is_same<T, half>::value ? 4096 : 1024;
  using Mat = Matrix<T, Dynamic, Dynamic>;
  Mat A = Mat::Ones(24, depth), B = Mat::Ones(depth, 21);
  Mat C = A * B;
  VERIFY_IS_EQUAL(C, Mat::Constant(24, 21, T(float(depth))));
  // Nonnegative operands, where no cancellation hides the growth of an error summed in T: here it stays one
  // rounding to T at any depth.
  Mat X = Mat::Random(24, depth).cwiseAbs(), Y = Mat::Random(depth, 21).cwiseAbs();
  C.noalias() = X * Y;
  VERIFY(verifyProduct<float>(C.template cast<double>(),
                              (X.template cast<double>().eval() * Y.template cast<double>().eval()).eval(), X, Y, 1));

  // The scalar factors combine in float: s * t = 2^-28 is below half's and above bfloat16's smallest subnormal, and
  // depth * s * t = 2^-16 (half) or 2^-18 (bfloat16) is representable in both.
  const T s = T(std::ldexp(1.0f, -14));
  C.noalias() = (s * A) * (B * s);
  VERIFY_IS_EQUAL(C, Mat::Constant(24, 21, T(std::ldexp(float(depth), -28))));
  // Where s * s underflows in T, a product small enough for the coefficient-based path would evaluate 0 * inf there,
  // so it takes the float GEMM as well.
  for (Index n : {5, 24}) {
    Mat P = Mat::Ones(n, n), Q = Mat::Ones(n, n), R(n, n);
    P(0, 0) = NumTraits<T>::infinity();
    R.noalias() = (s * P) * (Q * s);
    VERIFY(R.row(0).array().isInf().all());
    VERIFY_IS_EQUAL(R.bottomRows(n - 1), Mat::Constant(n - 1, n, T(std::ldexp(float(n), -28))));
    R.setOnes();
    R.noalias() += (s * P) * (Q * s);
    VERIFY(R.row(0).array().isInf().all());
    VERIFY_IS_EQUAL(R.bottomRows(n - 1), Mat::Constant(n - 1, n, T(1.0f + std::ldexp(float(n), -28))));
  }
}

// A cast to T is materialized in T, as before, rather than packed from the source and summed in float: the product
// sees T's overflow, its rounding of the source, and inf * 0.
template <typename T, typename Src>
void cast_to_float_accumulated() {
  using SrcMat = Matrix<Src, Dynamic, Dynamic>;
  using Mat = Matrix<T, Dynamic, Dynamic>;
  const Index n = 32;
  SrcMat A = random_small_integers<Src, ColMajor>(n, n, 8), B = random_small_integers<Src, ColMajor>(n, n, 8);
  // Out of half's range, and above bfloat16's spacing of 2^10 at 2^17.
  A(0, 0) = Src(std::is_same<T, half>::value ? 70000 : 205001);
  // 2051 rounds to 2052 in half and 2048 in bfloat16.
  A(1, 0) = Src(2051);
  B.row(0).setConstant(Src(3));
  B(0, 1) = Src(0);
  Mat ref = A.template cast<T>().eval() * B.template cast<T>().eval();
  Mat C = A.template cast<T>() * B.template cast<T>();
  // inf and NaN coefficients compare unequal, so compare their classification and the finite rest.
  VERIFY((C.array().isNaN() == ref.array().isNaN()).all());
  VERIFY((C.array().isInf() == ref.array().isInf()).all());
  VERIFY_IS_EQUAL(C.array().isFinite().select(C, T(0)).eval(), ref.array().isFinite().select(ref, T(0)).eval());
  VERIFY_IS_EQUAL(C.array().isInf().any(), (std::is_same<T, half>::value));
}

// The float workspace goes where the packed blocks do: a fixed-size product above the stack limit compiles and runs,
// and a preallocated product below it does not allocate.
template <typename T>
void float_accumulation_workspace() {
  using Fixed = Matrix<T, 192, 192>;
  std::unique_ptr<Fixed> A(new Fixed), B(new Fixed), C(new Fixed);
  *A = random_small_integers<T, ColMajor>(192, 192, 4);
  *B = random_small_integers<T, ColMajor>(192, 192, 4);
  C->noalias() = *A * *B;
  using Mat = Matrix<T, Dynamic, Dynamic>;
  MatrixXf Af = A->template cast<float>(), Bf = B->template cast<float>();
  VERIFY_IS_EQUAL(Mat(*C), (Af * Bf).template cast<T>().eval());

  // With EIGEN_USE_BLAS or on SME the float product materializes both casts.
#if !defined(EIGEN_USE_BLAS) && !defined(EIGEN_VECTORIZE_SME)
  Mat X = random_small_integers<T, ColMajor>(48, 40, 4), Y = random_small_integers<T, ColMajor>(40, 36, 4);
  MatrixXf exact = X.template cast<float>() * Y.template cast<float>();
  Mat Z(48, 36);
  internal::set_is_malloc_allowed(false);
  Z.noalias() = X * Y;
  internal::set_is_malloc_allowed(true);
  VERIFY_IS_EQUAL(Z, exact.cast<T>().eval());
  MatrixXf twice = Z.template cast<float>() + exact;
  internal::set_is_malloc_allowed(false);
  Z.noalias() += X * Y;
  internal::set_is_malloc_allowed(true);
  VERIFY_IS_EQUAL(Z, twice.cast<T>().eval());
#endif

  // A runtime vector result takes GEMV on T, which does not allocate for a row-major matrix by a column or a row by a
  // column-major matrix. Entries in [-1, 1] at depth 40 keep every partial sum exact in T.
  Matrix<T, Dynamic, Dynamic, RowMajor> R = random_small_integers<T, RowMajor>(48, 40, 1);
  Mat v = random_small_integers<T, ColMajor>(40, 1, 1), w = random_small_integers<T, ColMajor>(1, 40, 1);
  Mat V = random_small_integers<T, ColMajor>(40, 36, 1), y(48, 1), z(1, 36);
  internal::set_is_malloc_allowed(false);
  y.noalias() = R * v;
  z.noalias() = w * V;
  internal::set_is_malloc_allowed(true);
  VERIFY_IS_EQUAL(y, (R.template cast<float>() * v.template cast<float>()).template cast<T>().eval());
  VERIFY_IS_EQUAL(z, (w.template cast<float>() * V.template cast<float>()).template cast<T>().eval());

  // An empty product neither allocates nor rewrites dst, which keeps a NaN payload that a round trip through float
  // would quieten.
  Mat E0(200, 0), E1(0, 200), W = Mat::Ones(200, 200);
  W(0, 0) = numext::bit_cast<T>(std::is_same<T, half>::value ? uint16_t(0x7d01) : uint16_t(0x7fa1));
  const Mat W0 = W;
  internal::set_is_malloc_allowed(false);
  W.noalias() += E0 * E1;
  internal::set_is_malloc_allowed(true);
  VERIFY_IS_EQUAL(numext::bit_cast<uint16_t>(W(0, 0)), numext::bit_cast<uint16_t>(W0(0, 0)));
  VERIFY_IS_EQUAL(W.bottomRows(199), W0.bottomRows(199));
  W.noalias() = E0 * E1;
  VERIFY_IS_EQUAL(W, Mat::Zero(200, 200));
}

EIGEN_DECLARE_TEST(product_cast) {
  using MatrixXhalf = Matrix<half, Dynamic, Dynamic>;
  using internal::gemm_accumulates_in_float;
  using internal::gemm_operand_traits;
#if !defined(EIGEN_USE_BLAS) && !defined(EIGEN_VECTORIZE_SME)
  STATIC_CHECK((std::is_same<gemm_operand_traits<decltype(MatrixXhalf().cast<float>())>::StorageScalar, half>::value));
  STATIC_CHECK((
      std::is_same<gemm_operand_traits<decltype(MatrixXf().transpose().cast<double>())>::StorageScalar, float>::value));
#endif
  STATIC_CHECK(
      (std::is_same<gemm_operand_traits<decltype((MatrixXhalf() + MatrixXhalf()).cast<float>())>::StorageScalar,
                    float>::value));
  STATIC_CHECK((std::is_same<gemm_operand_traits<decltype((half(2) * MatrixXhalf()).cast<float>())>::StorageScalar,
                             float>::value));
  STATIC_CHECK((std::is_same<gemm_operand_traits<decltype(MatrixXf().cast<std::complex<float>>())>::StorageScalar,
                             std::complex<float>>::value));
  STATIC_CHECK((std::is_same<gemm_operand_traits<decltype(MatrixXf().cast<half>())>::StorageScalar, half>::value));
  STATIC_CHECK((gemm_accumulates_in_float<half, half>::value));
#if !(defined(EIGEN_USE_BLAS) && defined(EIGEN_USE_OPENBLAS_BFLOAT16) && EIGEN_USE_OPENBLAS_BFLOAT16)
  STATIC_CHECK((gemm_accumulates_in_float<bfloat16, bfloat16>::value));
#endif
  STATIC_CHECK((!gemm_accumulates_in_float<half, bfloat16>::value));
  STATIC_CHECK((!gemm_accumulates_in_float<float, float>::value));
  STATIC_CHECK((!gemm_accumulates_in_float<double, double>::value));

  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1((lazy_cast_products<half, float>()));
    CALL_SUBTEST_2((lazy_cast_products<bfloat16, float>()));
    CALL_SUBTEST_3((lazy_cast_products<float, double>()));
    CALL_SUBTEST_3((lazy_cast_products<int, float>()));
    CALL_SUBTEST_3((lazy_cast_products<double, float>()));
    CALL_SUBTEST_4((lazy_cast_products<int, double>()));
    CALL_SUBTEST_4((lazy_cast_products<half, double>()));
  }
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_5(float_accumulations<half>());
    CALL_SUBTEST_6(float_accumulations<bfloat16>());
  }
  CALL_SUBTEST_5(float_accumulation_depth<half>());
  CALL_SUBTEST_6(float_accumulation_depth<bfloat16>());
  CALL_SUBTEST_5((cast_to_float_accumulated<half, float>()));
  CALL_SUBTEST_5((cast_to_float_accumulated<half, double>()));
  CALL_SUBTEST_6((cast_to_float_accumulated<bfloat16, float>()));
  CALL_SUBTEST_6((cast_to_float_accumulated<bfloat16, double>()));
  CALL_SUBTEST_5(float_accumulation_workspace<half>());
  CALL_SUBTEST_6(float_accumulation_workspace<bfloat16>());
}
