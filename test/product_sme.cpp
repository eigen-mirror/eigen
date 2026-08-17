// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// SME GEMM kernel tests.
// Requires compiler flags: -march=armv9.2-a+sme2 and -DEIGEN_ARM64_USE_SME.

#include "product.h"

// Without the right -march flags, __ARM_FEATURE_SME is undefined and
// EIGEN_VECTORIZE_SME never fires - the test would silently compile
// against the NEON GEBP kernel and pass, making this a useless no-op.
// Fail the build instead.
#if !defined(EIGEN_VECTORIZE_SME)
#error \
    "product_sme requires the SME backend.  Build with -march=armv9.2-a+sme2 " \
    "-DEIGEN_ARM64_USE_SME (see -DEIGEN_TEST_SME=ON in test/CMakeLists.txt for " \
    "the typical CMake invocation)."
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

// ---------------------------------------------------------------------------
// Raw packed-buffer tests.
//
// The product tests above validate the packers only transitively through a full
// product, where the gebp_kernel can mask a mispack, or where a bug only shows
// at a specific SVL/region boundary.  The tests below call the SME packers
// directly and compare the packed buffer exactly against a scalar reference
// (equality is exact).  This pins every region -- in particular a dropped
// row-group in the two-pass trailing transpose, and a SYMM packer that copies
// the unused triangle instead of mirroring -- at whatever SVL the run uses.
// ---------------------------------------------------------------------------

// A distinctive marker for buffer cells the packer must leave untouched, and
// for the unused triangle of a lower-triangular operand. Random values live in
// [-1, 1], so it never collides with a real packed value.
static constexpr float kPackSentinel = 98765.0f;

// Lower-triangular n x n operand plus the dense symmetric reference the packer
// must emit. The unused triangle is filled with kPackSentinel so a packer that
// copies the dense matrix and never mirrors fails VERIFY_IS_EQUAL.
// product_selfadjoint_matrix stores the valid triangle where row >= col
// (after the Upper/RowMajor xor), so the packer must read stored(row,col)
// below the diagonal and stored(col,row) above it.
template <int StorageOrder>
static void make_lower_stored_symmetric(Index n, Matrix<float, Dynamic, Dynamic, StorageOrder>& stored,
                                        Matrix<float, Dynamic, Dynamic, StorageOrder>& full) {
  full = Matrix<float, Dynamic, Dynamic, StorageOrder>::Random(n, n);
  full = ((full + full.transpose()) * 0.5f).eval();
  stored = Matrix<float, Dynamic, Dynamic, StorageOrder>::Constant(n, n, kPackSentinel);
  for (Index i = 0; i < n; ++i)
    for (Index j = 0; j <= i; ++j) stored(i, j) = full(i, j);
}

// LHS SYMM packer: a square selfadjoint diagonal block of size n, packed into
// uniform kSmeMr-wide depth-major panels.  Reference: full(i+r, k).
template <int StorageOrder>
static void verify_symm_pack_lhs(Index n) {
  const Index MR = internal::kSmeMr;
  Matrix<float, Dynamic, Dynamic, StorageOrder> stored, full;
  make_lower_stored_symmetric<StorageOrder>(n, stored, full);

  VectorXf packed = VectorXf::Constant(n * n, kPackSentinel);
  VectorXf ref = VectorXf::Constant(n * n, kPackSentinel);
  for (Index i = 0; i < n; i += MR) {
    const Index w = numext::mini(MR, n - i);
    for (Index k = 0; k < n; ++k)
      for (Index r = 0; r < w; ++r) ref[i * n + k * w + r] = full(i + r, k);
  }

  internal::symm_pack_lhs<float, Index, internal::kSmeMr, 1, StorageOrder> pack;
  pack(packed.data(), stored.data(), stored.outerStride(), /*cols(depth)=*/n, /*rows=*/n);
  VERIFY_IS_EQUAL(packed, ref);
}

// RHS SYMM packer: a depth block [k2, k2 + depth) x cols columns of an N x N
// selfadjoint matrix, packed into kSmeNr-wide depth-major panels.  Reference:
// full(k2 + k, j + c).  A k2 > 0 offset makes the transposed region non-empty,
// so partial-width panels reach the two-pass transpose.
template <int StorageOrder>
static void verify_symm_pack_rhs(Index N, Index depth, Index cols, Index k2) {
  eigen_assert(k2 + depth <= N && cols <= N);
  const Index NR = internal::kSmeNr;
  Matrix<float, Dynamic, Dynamic, StorageOrder> stored, full;
  make_lower_stored_symmetric<StorageOrder>(N, stored, full);

  VectorXf packed = VectorXf::Constant(cols * depth, kPackSentinel);
  VectorXf ref = VectorXf::Constant(cols * depth, kPackSentinel);
  for (Index j = 0; j < cols; j += NR) {
    const Index w = numext::mini(NR, cols - j);
    for (Index k = 0; k < depth; ++k)
      for (Index c = 0; c < w; ++c) ref[j * depth + k * w + c] = full(k2 + k, j + c);
  }

  internal::symm_pack_rhs<float, Index, internal::kSmeNr, StorageOrder> pack;
  pack(packed.data(), stored.data(), stored.outerStride(), /*rows(depth)=*/depth, /*cols=*/cols, k2);
  VERIFY_IS_EQUAL(packed, ref);
}

static void test_symm_pack() {
  // The last panel width sweeps a range of partial widths; at each SVL the
  // two-pass trailing transpose (the if->loop fix) fires when a partial width
  // leaves a trailing row-group remainder in (svlw, 2*svlw).  The spread below
  // hits that for svlw in {4, 8, 16, 32, 64} (SVL 128..2048).
  const int sizes[] = {1, 5, 7, 17, 31, 32, 33, 37, 39, 45, 48, 49, 55, 57, 63, 64, 65, 79, 96, 97};
  for (int n : sizes) {
    verify_symm_pack_lhs<ColMajor>(n);
    verify_symm_pack_lhs<RowMajor>(n);
    // RHS, single depth block anchored at the diagonal (k2 == 0).
    verify_symm_pack_rhs<ColMajor>(n, n, n, 0);
    verify_symm_pack_rhs<RowMajor>(n, n, n, 0);
  }

  // RHS depth blocks offset from the diagonal (k2 > 0): the transposed region is
  // non-empty, so the RowMajor operand drives partial-width panels through the
  // two-pass transpose and the ColMajor operand through the partial copy.
  struct RhsCase {
    int N, depth, cols, k2;
  };
  const RhsCase rhs_cases[] = {
      {100, 32, 39, 16}, {100, 24, 39, 32}, {100, 40, 64, 8}, {100, 39, 39, 33}, {128, 57, 57, 40}, {128, 33, 45, 60},
  };
  for (const RhsCase& c : rhs_cases) {
    verify_symm_pack_rhs<ColMajor>(c.N, c.depth, c.cols, c.k2);
    verify_symm_pack_rhs<RowMajor>(c.N, c.depth, c.cols, c.k2);
  }
}

// ---------------------------------------------------------------------------
// Mapper-based packing fallback (sme_pack_{lhs,rhs}_fallback).
//
// Taken by by-value tensor sub-mappers and inner-strided blas mappers, which
// the raw pointer + stride packers cannot walk.  The product suite reaches these
// only through tensor contractions (nightly, SVL=512) and TriangularSolver
// (random sizes), so here we drive the fallback directly with both mapper
// families and compare the packed buffer against a scalar reference.
// ---------------------------------------------------------------------------

// Minimal stand-ins for by-value sub-mappers. ColMajor packets advance the
// first index, while RowMajor packets follow the normal storage-inner second
// index. operator() returns by value so both take the no-direct-access dispatch.
struct ByValueColMajorLhsMapper {
  const float* data;
  Index stride;  // element(i, k) = data[i + k * stride], contiguous in i
  float operator()(Index i, Index k) const { return data[i + k * stride]; }
  template <typename Packet>
  EIGEN_ALWAYS_INLINE Packet loadPacket(Index i, Index k) const {
    return internal::ploadu<Packet>(data + i + k * stride);
  }
};

struct ByValueRowMajorLhsMapper {
  const float* data;
  Index stride;  // element(i, k) = data[i * stride + k], contiguous in k
  float operator()(Index i, Index k) const { return data[i * stride + k]; }
  template <typename Packet>
  EIGEN_ALWAYS_INLINE Packet loadPacket(Index i, Index k) const {
    return internal::ploadu<Packet>(data + i * stride + k);
  }
};

struct ByValueColMajorRhsMapper {
  struct LinearMapper {
    const float* p;  // &element(0, col); contiguous in depth
    float operator()(Index k) const { return p[k]; }
    template <typename Packet>
    EIGEN_ALWAYS_INLINE Packet loadPacket(Index k) const {
      return internal::ploadu<Packet>(p + k);
    }
  };
  const float* data;
  Index stride;  // element(k, col) = data[k + col * stride], contiguous in k
  float operator()(Index k, Index col) const { return data[k + col * stride]; }
  LinearMapper getLinearMapper(Index k, Index col) const { return LinearMapper{data + k + col * stride}; }
};

struct ByValueRowMajorRhsMapper {
  struct LinearMapper {
    const float* p;  // &element(row, col); packet offsets advance columns
    float operator()(Index offset) const { return p[offset]; }
    template <typename Packet>
    EIGEN_ALWAYS_INLINE Packet loadPacket(Index offset) const {
      return internal::ploadu<Packet>(p + offset);
    }
  };
  const float* data;
  Index stride;  // element(k, col) = data[k * stride + col], contiguous in col
  float operator()(Index k, Index col) const { return data[k * stride + col]; }
  LinearMapper getLinearMapper(Index k, Index col) const { return LinearMapper{data + k * stride + col}; }
};

// Length of the packed LHS buffer for the given panel-mode layout, matching the
// dst_panel formula in sme_pack_lhs_fallback.
template <bool PanelMode>
static Index packed_len(Index outer, Index depth, Index unit, Index dst_stride, Index dst_offset) {
  // `outer` is rows (LHS) or cols (RHS); `unit` is kSmeMr or kSmeNr.
  if (!PanelMode) return outer * depth;
  Index end = 0;
  for (Index i = 0; i < outer; i += unit) {
    const Index w = numext::mini(unit, outer - i);
    end = numext::maxi(end, i * dst_stride + dst_offset * w + depth * w);
  }
  return end;
}

template <bool PanelMode, typename MatrixType>
static void fill_lhs_ref(VectorXf& ref, const MatrixType& V, Index rows, Index depth, Index dst_stride,
                         Index dst_offset) {
  const Index MR = internal::kSmeMr;
  ref.setConstant(kPackSentinel);
  for (Index i = 0; i < rows; i += MR) {
    const Index w = numext::mini(MR, rows - i);
    const Index base = PanelMode ? i * dst_stride + dst_offset * w : i * depth;
    for (Index k = 0; k < depth; ++k)
      for (Index r = 0; r < w; ++r) ref[base + k * w + r] = V(i + r, k);
  }
}

template <bool PanelMode, typename MatrixType>
static void fill_rhs_ref(VectorXf& ref, const MatrixType& V, Index cols, Index depth, Index dst_stride,
                         Index dst_offset) {
  const Index NR = internal::kSmeNr;
  ref.setConstant(kPackSentinel);
  for (Index j = 0; j < cols; j += NR) {
    const Index w = numext::mini(NR, cols - j);
    const Index base = PanelMode ? j * dst_stride + dst_offset * w : j * depth;
    for (Index k = 0; k < depth; ++k)
      for (Index c = 0; c < w; ++c) ref[base + k * w + c] = V(k, j + c);
  }
}

// Inner-strided blas mapper LHS: element(i, k) laid out with inner stride
// `incr`.  ColMajor takes the vectorised gather path; RowMajor takes the scalar
// path (its packets would run along depth, not rows).
template <int StorageOrder, bool PanelMode>
static void verify_fallback_lhs_strided(Index rows, Index depth, Index incr) {
  typedef internal::blas_data_mapper<float, Index, StorageOrder, Unaligned, Dynamic> Mapper;
  MatrixXf V = MatrixXf::Random(rows, depth);
  const Index mstride = (StorageOrder == ColMajor ? rows : depth) * incr;
  VectorXf buf = VectorXf::Zero((StorageOrder == ColMajor ? depth : rows) * mstride + incr);
  for (Index k = 0; k < depth; ++k)
    for (Index i = 0; i < rows; ++i)
      buf[StorageOrder == ColMajor ? i * incr + k * mstride : k * incr + i * mstride] = V(i, k);
  Mapper mapper(buf.data(), mstride, incr);

  const Index dst_stride = PanelMode ? depth + 5 : 0;
  const Index dst_offset = PanelMode ? 3 : 0;
  const Index len = packed_len<PanelMode>(rows, depth, internal::kSmeMr, dst_stride, dst_offset);
  VectorXf packed = VectorXf::Constant(len, kPackSentinel);
  VectorXf ref(len);
  fill_lhs_ref<PanelMode>(ref, V, rows, depth, dst_stride, dst_offset);

  internal::gemm_pack_lhs<float, Index, Mapper, internal::kSmeMr, 1, typename internal::packet_traits<float>::type,
                          StorageOrder, false, PanelMode>
      pack;
  pack(packed.data(), mapper, depth, rows, dst_stride, dst_offset);
  VERIFY_IS_EQUAL(packed, ref);
}

// By-value LHS mappers exercise both packet directions. RowMajor must stay
// scalar because its packets advance depth rather than rows.
template <int StorageOrder, bool PanelMode>
static void verify_fallback_lhs_byvalue(Index rows, Index depth) {
  typedef Matrix<float, Dynamic, Dynamic, StorageOrder> MatrixType;
  typedef typename std::conditional<StorageOrder == ColMajor, ByValueColMajorLhsMapper, ByValueRowMajorLhsMapper>::type
      Mapper;
  MatrixType V = MatrixType::Random(rows, depth);
  Mapper mapper{V.data(), V.outerStride()};

  const Index dst_stride = PanelMode ? depth + 5 : 0;
  const Index dst_offset = PanelMode ? 3 : 0;
  const Index len = packed_len<PanelMode>(rows, depth, internal::kSmeMr, dst_stride, dst_offset);
  VectorXf packed = VectorXf::Constant(len, kPackSentinel);
  VectorXf ref(len);
  fill_lhs_ref<PanelMode>(ref, V, rows, depth, dst_stride, dst_offset);

  internal::gemm_pack_lhs<float, Index, Mapper, internal::kSmeMr, 1, typename internal::packet_traits<float>::type,
                          StorageOrder, false, PanelMode>
      pack;
  pack(packed.data(), mapper, depth, rows, dst_stride, dst_offset);
  VERIFY_IS_EQUAL(packed, ref);
}

// Inner-strided blas mapper RHS: element(k, col) with inner stride `incr`.
// ColMajor takes the vectorised transpose path; RowMajor takes the scalar path.
template <int StorageOrder, bool PanelMode>
static void verify_fallback_rhs_strided(Index depth, Index cols, Index incr) {
  typedef internal::blas_data_mapper<float, Index, StorageOrder, Unaligned, Dynamic> Mapper;
  MatrixXf V = MatrixXf::Random(depth, cols);
  const Index mstride = (StorageOrder == ColMajor ? depth : cols) * incr;
  VectorXf buf = VectorXf::Zero((StorageOrder == ColMajor ? cols : depth) * mstride + incr);
  for (Index col = 0; col < cols; ++col)
    for (Index k = 0; k < depth; ++k)
      buf[StorageOrder == ColMajor ? k * incr + col * mstride : col * incr + k * mstride] = V(k, col);
  Mapper mapper(buf.data(), mstride, incr);

  const Index dst_stride = PanelMode ? depth + 5 : 0;
  const Index dst_offset = PanelMode ? 3 : 0;
  const Index len = packed_len<PanelMode>(cols, depth, internal::kSmeNr, dst_stride, dst_offset);
  VectorXf packed = VectorXf::Constant(len, kPackSentinel);
  VectorXf ref(len);
  fill_rhs_ref<PanelMode>(ref, V, cols, depth, dst_stride, dst_offset);

  internal::gemm_pack_rhs<float, Index, Mapper, internal::kSmeNr, StorageOrder, false, PanelMode> pack;
  pack(packed.data(), mapper, depth, cols, dst_stride, dst_offset);
  VERIFY_IS_EQUAL(packed, ref);
}

// By-value RHS mappers likewise cover both packet directions. RowMajor packets
// advance columns, so the depth-oriented transpose fallback must stay scalar.
template <int StorageOrder, bool PanelMode>
static void verify_fallback_rhs_byvalue(Index depth, Index cols) {
  typedef Matrix<float, Dynamic, Dynamic, StorageOrder> MatrixType;
  typedef typename std::conditional<StorageOrder == ColMajor, ByValueColMajorRhsMapper, ByValueRowMajorRhsMapper>::type
      Mapper;
  MatrixType V = MatrixType::Random(depth, cols);
  Mapper mapper{V.data(), V.outerStride()};

  const Index dst_stride = PanelMode ? depth + 5 : 0;
  const Index dst_offset = PanelMode ? 3 : 0;
  const Index len = packed_len<PanelMode>(cols, depth, internal::kSmeNr, dst_stride, dst_offset);
  VectorXf packed = VectorXf::Constant(len, kPackSentinel);
  VectorXf ref(len);
  fill_rhs_ref<PanelMode>(ref, V, cols, depth, dst_stride, dst_offset);

  internal::gemm_pack_rhs<float, Index, Mapper, internal::kSmeNr, StorageOrder, false, PanelMode> pack;
  pack(packed.data(), mapper, depth, cols, dst_stride, dst_offset);
  VERIFY_IS_EQUAL(packed, ref);
}

static void test_mapper_fallback() {
  const int widths[] = {4, 5, 32, 33, 65};  // rows/cols around 4 and 32
  const int depths[] = {1, 3, 8, 35};       // depth remainders 1..3 and larger
  for (int n : widths) {
    for (int d : depths) {
      for (int incr : {2, 3}) {
        verify_fallback_lhs_strided<ColMajor, false>(n, d, incr);
        verify_fallback_lhs_strided<ColMajor, true>(n, d, incr);
        verify_fallback_lhs_strided<RowMajor, false>(n, d, incr);  // scalar path
        verify_fallback_lhs_strided<RowMajor, true>(n, d, incr);
        verify_fallback_rhs_strided<ColMajor, false>(d, n, incr);
        verify_fallback_rhs_strided<ColMajor, true>(d, n, incr);
        verify_fallback_rhs_strided<RowMajor, false>(d, n, incr);  // scalar path
        verify_fallback_rhs_strided<RowMajor, true>(d, n, incr);
      }
      verify_fallback_lhs_byvalue<ColMajor, false>(n, d);
      verify_fallback_lhs_byvalue<ColMajor, true>(n, d);
      verify_fallback_lhs_byvalue<RowMajor, false>(n, d);
      verify_fallback_lhs_byvalue<RowMajor, true>(n, d);
      verify_fallback_rhs_byvalue<ColMajor, false>(d, n);
      verify_fallback_rhs_byvalue<ColMajor, true>(d, n);
      verify_fallback_rhs_byvalue<RowMajor, false>(d, n);
      verify_fallback_rhs_byvalue<RowMajor, true>(d, n);
    }
  }
}

EIGEN_DECLARE_TEST(product_sme) {
  // Square edge cases around the block and tile boundaries (the block is
  // kSmeMr x kSmeNr and a ZA tile is svlw x svlw, so the sizes below land
  // just on/off the intra-tile splits and the block tails at SVL=512).
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

  // Exercise the kc split path just above the SME blocking heuristic's depth
  // cap (sme_max_kc in GeneralBlockPanelKernel.h).
  test_deep_k_split();
  test_symm_pack();
  test_mapper_fallback();

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
