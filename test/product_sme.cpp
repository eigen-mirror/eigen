// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// SME GEMM kernel tests.
// Requires compiler flags: -march=armv9.2-a+sme2 -msve-vector-bits=512
// and the define: -DEIGEN_ARM64_USE_SME

#include "product.h"

// Without the right -march flags, __ARM_FEATURE_SME is undefined and
// EIGEN_VECTORIZE_SME never fires - the test would silently compile
// against the NEON GEBP kernel and pass, making this a useless no-op.
// Fail the build instead.
#if !defined(EIGEN_VECTORIZE_SME)
#error \
    "product_sme requires the SME backend.  Build with -march=armv9.2-a+sme2 " \
    "-msve-vector-bits=512 -DEIGEN_ARM64_USE_SME (see -DEIGEN_TEST_SME=ON in " \
    "test/CMakeLists.txt for the typical CMake invocation)."
#endif

using SmeColMajorMatF = Matrix<float, Dynamic, Dynamic, ColMajor>;
using SmeRowMajorMatF = Matrix<float, Dynamic, Dynamic, RowMajor>;
using SmeColMajorStridedMatF = Map<SmeColMajorMatF, 0, Stride<Dynamic, Dynamic>>;
using SmeRowMajorStridedMatF = Map<SmeRowMajorMatF, 0, Stride<Dynamic, Dynamic>>;

template <typename InputMat, typename ResultMat, typename ResultMap>
static void verify_strided_result(int n, ResultMat& storage, const Stride<Dynamic, Dynamic>& stride) {
  InputMat A = InputMat::Random(n, n);
  InputMat B = InputMat::Random(n, n);
  ResultMap C(storage.data(), n, n, stride);
  C = ResultMat::Random(n, n);
  ResultMat c_before = C.eval();

  C.noalias() += A * B;

  ResultMat ref = c_before + (A.lazyProduct(B)).eval();
  ResultMat got = C;
  VERIFY_IS_APPROX(got, ref);
}

template <typename InputMat>
static void test_general_strided_result(int n) {
  // General-stride C path: InputMat selects the source packers, while both C
  // strides are non-unit so sme_store_za_tile uses scalar scatter.
  SmeColMajorMatF storage = SmeColMajorMatF::Zero(2 * n, n);
  verify_strided_result<InputMat, SmeColMajorMatF, SmeColMajorStridedMatF>(
      n, storage, Stride<Dynamic, Dynamic>(/*outer=*/2 * n, /*inner=*/2));

  // Padding rows skipped by the strided Map should not be touched.
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      VERIFY(storage(2 * i + 1, j) == float(0));
    }
  }
}

static void test_rowmajor_strided_result(int n) {
  // RowMajor C path: inner stride is one, with padded columns after the Map.
  SmeRowMajorMatF storage = SmeRowMajorMatF::Zero(n, 2 * n);
  verify_strided_result<SmeRowMajorMatF, SmeRowMajorMatF, SmeRowMajorStridedMatF>(
      n, storage, Stride<Dynamic, Dynamic>(/*outer=*/2 * n, /*inner=*/1));

  // Padding columns skipped by the strided Map should not be touched.
  for (int i = 0; i < n; ++i) {
    for (int j = n; j < 2 * n; ++j) {
      VERIFY(storage(i, j) == float(0));
    }
  }
}

static void test_deep_k_split() {
  constexpr int rows = 64;
  constexpr int depth = 2050;
  constexpr int cols = 64;
  SmeColMajorMatF A = SmeColMajorMatF::Random(rows, depth);
  SmeColMajorMatF B = SmeColMajorMatF::Random(depth, cols);
  SmeColMajorMatF C = SmeColMajorMatF::Random(rows, cols);
  SmeColMajorMatF c_before = C;

  C.noalias() += A * B;

  VERIFY_IS_APPROX(C, c_before + (A.lazyProduct(B)).eval());
}

EIGEN_DECLARE_TEST(product_sme) {
  // Square edge cases around the 2x2 tile-grid boundaries (MR=NR=32,
  // ZA tiles are 16x16 fp32; sizes near 16, 17, 31, 32, 33, 64, 65
  // exercise the VL-wide intra-tile splits and the MR/NR tails).
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(1, 1)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(15, 15)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(16, 16)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(17, 17)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(31, 31)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(33, 33)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(63, 63)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(64, 64)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(65, 65)));

  // Thin / wide rectangular cases (M x 1, 1 x N)
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(32, 1)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(1, 32)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(1, 64)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(64, 1)));

  // Non-float scalar smoke tests: SME only specializes fp32, so these prove
  // unsupported scalar types still route through the generic product path.
  CALL_SUBTEST_2(product(Matrix<double, Dynamic, Dynamic>(33, 17)));
  CALL_SUBTEST_3(product(Matrix<std::complex<float>, Dynamic, Dynamic>(33, 17)));

  // Non-square cases that exercise tail paths for both M and N
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(17, 65)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(65, 17)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(15, 63)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(33, 7)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(7, 33)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(128, 3)));
  CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(3, 128)));

  // Exercise the kc split path just above the SME heuristic's 2048 depth cap.
  test_deep_k_split();

  // Random sizes
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(product(Matrix<float, Dynamic, Dynamic>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                                           internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
  }

  // Exercise the RowMajor packers and RowMajor result path.  When the input
  // MatrixType is RowMajor, product() instantiates m1/m2/m3/res in RowMajor,
  // so every matrix-matrix product in the suite flows through:
  //   - the RowMajor LHS packer (gemm_pack_lhs<..., RowMajor>)
  //   - the RowMajor RHS packer (gemm_pack_rhs<..., RowMajor>)
  //   - the RowMajor-C dispatch in GeneralMatrixMatrix.h (which transposes
  //     the computation: C^T = B^T * A^T).
  CALL_SUBTEST_1(product(SmeRowMajorMatF(15, 15)));
  CALL_SUBTEST_1(product(SmeRowMajorMatF(16, 16)));
  CALL_SUBTEST_1(product(SmeRowMajorMatF(17, 17)));
  CALL_SUBTEST_1(product(SmeRowMajorMatF(31, 31)));
  CALL_SUBTEST_1(product(SmeRowMajorMatF(32, 32)));
  CALL_SUBTEST_1(product(SmeRowMajorMatF(33, 33)));
  CALL_SUBTEST_1(product(SmeRowMajorMatF(64, 64)));
  CALL_SUBTEST_1(product(SmeRowMajorMatF(65, 65)));
  CALL_SUBTEST_1(product(SmeRowMajorMatF(17, 65)));
  CALL_SUBTEST_1(product(SmeRowMajorMatF(65, 17)));
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(product(
        SmeRowMajorMatF(internal::random<int>(1, EIGEN_TEST_MAX_SIZE), internal::random<int>(1, EIGEN_TEST_MAX_SIZE))));
  }

  // Exercise the general-stride branch of sme_store_za_tile: fires when both
  // C_stride_row != 1 and C_stride_col != 1, e.g. a Map<Matrix> with an
  // explicit non-unit inner stride.  product.h never builds such a result, so
  // without this subtest the scalar-scatter path is effectively untested.
  for (int n : {15, 16, 17, 31, 32, 33, 63, 64, 65}) {
    test_general_strided_result<SmeColMajorMatF>(n);
    test_general_strided_result<SmeRowMajorMatF>(n);
    test_rowmajor_strided_result(n);
  }

  // Row-LHS x Row-RHS -> Col-C: the one LHS/RHS/C storage combination that
  // product.h's transpose-style expressions never build directly (it always
  // flips one side of the multiplication).  The code paths are the same as
  // other combinations via Eigen's dispatch, but exercise them explicitly.
  for (int n : {15, 16, 17, 31, 32, 33, 63, 64, 65}) {
    Matrix<float, Dynamic, Dynamic, RowMajor> A = Matrix<float, Dynamic, Dynamic, RowMajor>::Random(n, n);
    Matrix<float, Dynamic, Dynamic, RowMajor> B = Matrix<float, Dynamic, Dynamic, RowMajor>::Random(n, n);
    SmeColMajorMatF C = SmeColMajorMatF::Zero(n, n);
    C.noalias() += A * B;
    VERIFY_IS_APPROX(C, (A.lazyProduct(B)).eval());
  }
}
