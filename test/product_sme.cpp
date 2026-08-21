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
// Double precision additionally needs FEAT_SME_F64F64 (+sme-f64f64, or a -mcpu
// that implies it); without it EIGEN_VECTORIZE_SME_F64F64 is undefined and
// double keeps the generic kernel, so the double subtest packs its cases
// through that path instead.

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

template <typename Scalar>
using SmeColMajorMat = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
template <typename Scalar>
using SmeRowMajorMat = Matrix<Scalar, Dynamic, Dynamic, RowMajor>;
template <typename Scalar>
using SmeVector = Matrix<Scalar, Dynamic, 1>;
template <typename Scalar>
using SmeColMajorStridedMat = Map<SmeColMajorMat<Scalar>, 0, Stride<Dynamic, Dynamic>>;
template <typename Scalar>
using SmeRowMajorStridedMat = Map<SmeRowMajorMat<Scalar>, 0, Stride<Dynamic, Dynamic>>;

// The logical micro-kernel block width for Scalar: kSmeMr for float, kSmeMrD
// for double.  Sizes below are expressed in terms of it so each scalar sweeps
// its own block and ZA-tile boundaries.
template <typename Scalar>
static constexpr int sme_mr() {
  return internal::sme_block<Scalar>::mr;
}
template <typename Scalar>
static constexpr int sme_nr() {
  return internal::sme_block<Scalar>::nr;
}

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

template <typename Scalar, typename InputMat>
static void test_general_strided_result(int n) {
  // General-stride C path: InputMat selects the source packers, while both C
  // strides are non-unit so sme_store_za_tile uses scalar scatter.
  SmeColMajorMat<Scalar> storage = SmeColMajorMat<Scalar>::Zero(2 * n, n);
  verify_strided_result<InputMat, SmeColMajorMat<Scalar>, SmeColMajorStridedMat<Scalar>>(
      n, storage, Stride<Dynamic, Dynamic>(/*outer=*/2 * n, /*inner=*/2));

  // Padding rows skipped by the strided Map should not be touched.
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      VERIFY(storage(2 * i + 1, j) == Scalar(0));
    }
  }
}

template <typename Scalar>
static void test_rowmajor_strided_result(int n) {
  // RowMajor C path: inner stride is one, with padded columns after the Map.
  SmeRowMajorMat<Scalar> storage = SmeRowMajorMat<Scalar>::Zero(n, 2 * n);
  verify_strided_result<SmeRowMajorMat<Scalar>, SmeRowMajorMat<Scalar>, SmeRowMajorStridedMat<Scalar>>(
      n, storage, Stride<Dynamic, Dynamic>(/*outer=*/2 * n, /*inner=*/1));

  // Padding columns skipped by the strided Map should not be touched.
  for (int i = 0; i < n; ++i) {
    for (int j = n; j < 2 * n; ++j) {
      VERIFY(storage(i, j) == Scalar(0));
    }
  }
}

// Exercise the kc split path just above the SME blocking heuristic's depth cap
// (sme_max_kc in GeneralBlockPanelKernel.h, scaled by the scalar width).
template <typename Scalar>
static void test_deep_k_split() {
  constexpr int rows = 64;
  const int depth = int(2 * (Index(EIGEN_SME_MAX_KC) * Index(sizeof(float)) / Index(sizeof(Scalar)))) + 2;
  constexpr int cols = 64;
  SmeColMajorMat<Scalar> A = SmeColMajorMat<Scalar>::Random(rows, depth);
  SmeColMajorMat<Scalar> B = SmeColMajorMat<Scalar>::Random(depth, cols);
  SmeColMajorMat<Scalar> C = SmeColMajorMat<Scalar>::Random(rows, cols);
  SmeColMajorMat<Scalar> c_before = C;

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
template <typename Scalar>
static Scalar pack_sentinel() {
  return Scalar(98765);
}

// Lower-triangular n x n operand plus the dense symmetric reference the packer
// must emit. The unused triangle is filled with the sentinel so a packer that
// copies the dense matrix and never mirrors fails VERIFY_IS_EQUAL.
// product_selfadjoint_matrix stores the valid triangle where row >= col
// (after the Upper/RowMajor xor), so the packer must read stored(row,col)
// below the diagonal and stored(col,row) above it.
template <typename Scalar, int StorageOrder>
static void make_lower_stored_symmetric(Index n, Matrix<Scalar, Dynamic, Dynamic, StorageOrder>& stored,
                                        Matrix<Scalar, Dynamic, Dynamic, StorageOrder>& full) {
  full = Matrix<Scalar, Dynamic, Dynamic, StorageOrder>::Random(n, n);
  full = ((full + full.transpose()) * Scalar(0.5)).eval();
  stored = Matrix<Scalar, Dynamic, Dynamic, StorageOrder>::Constant(n, n, pack_sentinel<Scalar>());
  for (Index i = 0; i < n; ++i)
    for (Index j = 0; j <= i; ++j) stored(i, j) = full(i, j);
}

// LHS SYMM packer: a square selfadjoint diagonal block of size n, packed into
// uniform mr-wide depth-major panels.  Reference: full(i+r, k).
template <typename Scalar, int StorageOrder>
static void verify_symm_pack_lhs(Index n) {
  const Index MR = sme_mr<Scalar>();
  Matrix<Scalar, Dynamic, Dynamic, StorageOrder> stored, full;
  make_lower_stored_symmetric<Scalar, StorageOrder>(n, stored, full);

  SmeVector<Scalar> packed = SmeVector<Scalar>::Constant(n * n, pack_sentinel<Scalar>());
  SmeVector<Scalar> ref = SmeVector<Scalar>::Constant(n * n, pack_sentinel<Scalar>());
  for (Index i = 0; i < n; i += MR) {
    const Index w = numext::mini(MR, n - i);
    for (Index k = 0; k < n; ++k)
      for (Index r = 0; r < w; ++r) ref[i * n + k * w + r] = full(i + r, k);
  }

  internal::symm_pack_lhs<Scalar, Index, sme_mr<Scalar>(), 1, StorageOrder> pack;
  pack(packed.data(), stored.data(), stored.outerStride(), /*cols(depth)=*/n, /*rows=*/n);
  VERIFY_IS_EQUAL(packed, ref);
}

// RHS SYMM packer: a depth block [k2, k2 + depth) x cols columns of an N x N
// selfadjoint matrix, packed into nr-wide depth-major panels.  Reference:
// full(k2 + k, j + c).  A k2 > 0 offset makes the transposed region non-empty,
// so partial-width panels reach the two-pass transpose.
template <typename Scalar, int StorageOrder>
static void verify_symm_pack_rhs(Index N, Index depth, Index cols, Index k2) {
  eigen_assert(k2 + depth <= N && cols <= N);
  const Index NR = sme_nr<Scalar>();
  Matrix<Scalar, Dynamic, Dynamic, StorageOrder> stored, full;
  make_lower_stored_symmetric<Scalar, StorageOrder>(N, stored, full);

  SmeVector<Scalar> packed = SmeVector<Scalar>::Constant(cols * depth, pack_sentinel<Scalar>());
  SmeVector<Scalar> ref = SmeVector<Scalar>::Constant(cols * depth, pack_sentinel<Scalar>());
  for (Index j = 0; j < cols; j += NR) {
    const Index w = numext::mini(NR, cols - j);
    for (Index k = 0; k < depth; ++k)
      for (Index c = 0; c < w; ++c) ref[j * depth + k * w + c] = full(k2 + k, j + c);
  }

  internal::symm_pack_rhs<Scalar, Index, sme_nr<Scalar>(), StorageOrder> pack;
  pack(packed.data(), stored.data(), stored.outerStride(), /*rows(depth)=*/depth, /*cols=*/cols, k2);
  VERIFY_IS_EQUAL(packed, ref);
}

template <typename Scalar>
static void test_symm_pack() {
  // The last panel width sweeps a range of partial widths; at each SVL the
  // two-pass trailing transpose (the if->loop fix) fires when a partial width
  // leaves a trailing row-group remainder in (svl, 2*svl).  The spread below
  // hits that for svl in {2, 4, 8, 16, 32, 64} -- fp32 SVL 128..2048 and the
  // fp64 lane counts, which are half of those.
  const int sizes[] = {1, 5, 7, 17, 31, 32, 33, 37, 39, 45, 48, 49, 55, 57, 63, 64, 65, 79, 96, 97};
  for (int n : sizes) {
    verify_symm_pack_lhs<Scalar, ColMajor>(n);
    verify_symm_pack_lhs<Scalar, RowMajor>(n);
    // RHS, single depth block anchored at the diagonal (k2 == 0).
    verify_symm_pack_rhs<Scalar, ColMajor>(n, n, n, 0);
    verify_symm_pack_rhs<Scalar, RowMajor>(n, n, n, 0);
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
    verify_symm_pack_rhs<Scalar, ColMajor>(c.N, c.depth, c.cols, c.k2);
    verify_symm_pack_rhs<Scalar, RowMajor>(c.N, c.depth, c.cols, c.k2);
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
template <typename Scalar>
struct ByValueColMajorLhsMapper {
  const Scalar* data;
  Index stride;  // element(i, k) = data[i + k * stride], contiguous in i
  Scalar operator()(Index i, Index k) const { return data[i + k * stride]; }
  template <typename Packet>
  EIGEN_ALWAYS_INLINE Packet loadPacket(Index i, Index k) const {
    return internal::ploadu<Packet>(data + i + k * stride);
  }
};

template <typename Scalar>
struct ByValueRowMajorLhsMapper {
  const Scalar* data;
  Index stride;  // element(i, k) = data[i * stride + k], contiguous in k
  Scalar operator()(Index i, Index k) const { return data[i * stride + k]; }
  template <typename Packet>
  EIGEN_ALWAYS_INLINE Packet loadPacket(Index i, Index k) const {
    return internal::ploadu<Packet>(data + i * stride + k);
  }
};

template <typename Scalar>
struct ByValueColMajorRhsMapper {
  struct LinearMapper {
    const Scalar* p;  // &element(0, col); contiguous in depth
    Scalar operator()(Index k) const { return p[k]; }
    template <typename Packet>
    EIGEN_ALWAYS_INLINE Packet loadPacket(Index k) const {
      return internal::ploadu<Packet>(p + k);
    }
  };
  const Scalar* data;
  Index stride;  // element(k, col) = data[k + col * stride], contiguous in k
  Scalar operator()(Index k, Index col) const { return data[k + col * stride]; }
  LinearMapper getLinearMapper(Index k, Index col) const { return LinearMapper{data + k + col * stride}; }
};

template <typename Scalar>
struct ByValueRowMajorRhsMapper {
  struct LinearMapper {
    const Scalar* p;  // &element(row, col); packet offsets advance columns
    Scalar operator()(Index offset) const { return p[offset]; }
    template <typename Packet>
    EIGEN_ALWAYS_INLINE Packet loadPacket(Index offset) const {
      return internal::ploadu<Packet>(p + offset);
    }
  };
  const Scalar* data;
  Index stride;  // element(k, col) = data[k * stride + col], contiguous in col
  Scalar operator()(Index k, Index col) const { return data[k * stride + col]; }
  LinearMapper getLinearMapper(Index k, Index col) const { return LinearMapper{data + k * stride + col}; }
};

// Length of the packed LHS buffer for the given panel-mode layout, matching the
// dst_panel formula in sme_pack_lhs_fallback.
template <bool PanelMode>
static Index packed_len(Index outer, Index depth, Index unit, Index dst_stride, Index dst_offset) {
  // `outer` is rows (LHS) or cols (RHS); `unit` is the panel width mr or nr.
  if (!PanelMode) return outer * depth;
  Index end = 0;
  for (Index i = 0; i < outer; i += unit) {
    const Index w = numext::mini(unit, outer - i);
    end = numext::maxi(end, i * dst_stride + dst_offset * w + depth * w);
  }
  return end;
}

template <typename Scalar, bool PanelMode, typename MatrixType>
static void fill_lhs_ref(SmeVector<Scalar>& ref, const MatrixType& V, Index rows, Index depth, Index dst_stride,
                         Index dst_offset) {
  const Index MR = sme_mr<Scalar>();
  ref.setConstant(pack_sentinel<Scalar>());
  for (Index i = 0; i < rows; i += MR) {
    const Index w = numext::mini(MR, rows - i);
    const Index base = PanelMode ? i * dst_stride + dst_offset * w : i * depth;
    for (Index k = 0; k < depth; ++k)
      for (Index r = 0; r < w; ++r) ref[base + k * w + r] = V(i + r, k);
  }
}

template <typename Scalar, bool PanelMode, typename MatrixType>
static void fill_rhs_ref(SmeVector<Scalar>& ref, const MatrixType& V, Index cols, Index depth, Index dst_stride,
                         Index dst_offset) {
  const Index NR = sme_nr<Scalar>();
  ref.setConstant(pack_sentinel<Scalar>());
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
template <typename Scalar, int StorageOrder, bool PanelMode>
static void verify_fallback_lhs_strided(Index rows, Index depth, Index incr) {
  using Mapper = internal::blas_data_mapper<Scalar, Index, StorageOrder, Unaligned, Dynamic>;
  Matrix<Scalar, Dynamic, Dynamic> V = Matrix<Scalar, Dynamic, Dynamic>::Random(rows, depth);
  const Index mstride = (StorageOrder == ColMajor ? rows : depth) * incr;
  SmeVector<Scalar> buf = SmeVector<Scalar>::Zero((StorageOrder == ColMajor ? depth : rows) * mstride + incr);
  for (Index k = 0; k < depth; ++k)
    for (Index i = 0; i < rows; ++i)
      buf[StorageOrder == ColMajor ? i * incr + k * mstride : k * incr + i * mstride] = V(i, k);
  Mapper mapper(buf.data(), mstride, incr);

  const Index dst_stride = PanelMode ? depth + 5 : 0;
  const Index dst_offset = PanelMode ? 3 : 0;
  const Index len = packed_len<PanelMode>(rows, depth, sme_mr<Scalar>(), dst_stride, dst_offset);
  SmeVector<Scalar> packed = SmeVector<Scalar>::Constant(len, pack_sentinel<Scalar>());
  SmeVector<Scalar> ref(len);
  fill_lhs_ref<Scalar, PanelMode>(ref, V, rows, depth, dst_stride, dst_offset);

  internal::gemm_pack_lhs<Scalar, Index, Mapper, sme_mr<Scalar>(), 1, typename internal::packet_traits<Scalar>::type,
                          StorageOrder, false, PanelMode>
      pack;
  pack(packed.data(), mapper, depth, rows, dst_stride, dst_offset);
  VERIFY_IS_EQUAL(packed, ref);
}

// By-value LHS mappers exercise both packet directions. RowMajor must stay
// scalar because its packets advance depth rather than rows.
template <typename Scalar, int StorageOrder, bool PanelMode>
static void verify_fallback_lhs_byvalue(Index rows, Index depth) {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, StorageOrder>;
  using Mapper = typename std::conditional<StorageOrder == ColMajor, ByValueColMajorLhsMapper<Scalar>,
                                           ByValueRowMajorLhsMapper<Scalar>>::type;
  MatrixType V = MatrixType::Random(rows, depth);
  Mapper mapper{V.data(), V.outerStride()};

  const Index dst_stride = PanelMode ? depth + 5 : 0;
  const Index dst_offset = PanelMode ? 3 : 0;
  const Index len = packed_len<PanelMode>(rows, depth, sme_mr<Scalar>(), dst_stride, dst_offset);
  SmeVector<Scalar> packed = SmeVector<Scalar>::Constant(len, pack_sentinel<Scalar>());
  SmeVector<Scalar> ref(len);
  fill_lhs_ref<Scalar, PanelMode>(ref, V, rows, depth, dst_stride, dst_offset);

  internal::gemm_pack_lhs<Scalar, Index, Mapper, sme_mr<Scalar>(), 1, typename internal::packet_traits<Scalar>::type,
                          StorageOrder, false, PanelMode>
      pack;
  pack(packed.data(), mapper, depth, rows, dst_stride, dst_offset);
  VERIFY_IS_EQUAL(packed, ref);
}

// Inner-strided blas mapper RHS: element(k, col) with inner stride `incr`.
// ColMajor takes the vectorised transpose path; RowMajor takes the scalar path.
template <typename Scalar, int StorageOrder, bool PanelMode>
static void verify_fallback_rhs_strided(Index depth, Index cols, Index incr) {
  using Mapper = internal::blas_data_mapper<Scalar, Index, StorageOrder, Unaligned, Dynamic>;
  Matrix<Scalar, Dynamic, Dynamic> V = Matrix<Scalar, Dynamic, Dynamic>::Random(depth, cols);
  const Index mstride = (StorageOrder == ColMajor ? depth : cols) * incr;
  SmeVector<Scalar> buf = SmeVector<Scalar>::Zero((StorageOrder == ColMajor ? cols : depth) * mstride + incr);
  for (Index col = 0; col < cols; ++col)
    for (Index k = 0; k < depth; ++k)
      buf[StorageOrder == ColMajor ? k * incr + col * mstride : col * incr + k * mstride] = V(k, col);
  Mapper mapper(buf.data(), mstride, incr);

  const Index dst_stride = PanelMode ? depth + 5 : 0;
  const Index dst_offset = PanelMode ? 3 : 0;
  const Index len = packed_len<PanelMode>(cols, depth, sme_nr<Scalar>(), dst_stride, dst_offset);
  SmeVector<Scalar> packed = SmeVector<Scalar>::Constant(len, pack_sentinel<Scalar>());
  SmeVector<Scalar> ref(len);
  fill_rhs_ref<Scalar, PanelMode>(ref, V, cols, depth, dst_stride, dst_offset);

  internal::gemm_pack_rhs<Scalar, Index, Mapper, sme_nr<Scalar>(), StorageOrder, false, PanelMode> pack;
  pack(packed.data(), mapper, depth, cols, dst_stride, dst_offset);
  VERIFY_IS_EQUAL(packed, ref);
}

// By-value RHS mappers likewise cover both packet directions. RowMajor packets
// advance columns, so the depth-oriented transpose fallback must stay scalar.
template <typename Scalar, int StorageOrder, bool PanelMode>
static void verify_fallback_rhs_byvalue(Index depth, Index cols) {
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic, StorageOrder>;
  using Mapper = typename std::conditional<StorageOrder == ColMajor, ByValueColMajorRhsMapper<Scalar>,
                                           ByValueRowMajorRhsMapper<Scalar>>::type;
  MatrixType V = MatrixType::Random(depth, cols);
  Mapper mapper{V.data(), V.outerStride()};

  const Index dst_stride = PanelMode ? depth + 5 : 0;
  const Index dst_offset = PanelMode ? 3 : 0;
  const Index len = packed_len<PanelMode>(cols, depth, sme_nr<Scalar>(), dst_stride, dst_offset);
  SmeVector<Scalar> packed = SmeVector<Scalar>::Constant(len, pack_sentinel<Scalar>());
  SmeVector<Scalar> ref(len);
  fill_rhs_ref<Scalar, PanelMode>(ref, V, cols, depth, dst_stride, dst_offset);

  internal::gemm_pack_rhs<Scalar, Index, Mapper, sme_nr<Scalar>(), StorageOrder, false, PanelMode> pack;
  pack(packed.data(), mapper, depth, cols, dst_stride, dst_offset);
  VERIFY_IS_EQUAL(packed, ref);
}

template <typename Scalar>
static void test_mapper_fallback() {
  const int MR = sme_mr<Scalar>();
  const int widths[] = {4, 5, MR, MR + 1, 2 * MR + 1};  // rows/cols around 4 and the panel width
  const int depths[] = {1, 3, 8, 35};                   // depth remainders 1..3 and larger
  for (int n : widths) {
    for (int d : depths) {
      for (int incr : {2, 3}) {
        verify_fallback_lhs_strided<Scalar, ColMajor, false>(n, d, incr);
        verify_fallback_lhs_strided<Scalar, ColMajor, true>(n, d, incr);
        verify_fallback_lhs_strided<Scalar, RowMajor, false>(n, d, incr);  // scalar path
        verify_fallback_lhs_strided<Scalar, RowMajor, true>(n, d, incr);
        verify_fallback_rhs_strided<Scalar, ColMajor, false>(d, n, incr);
        verify_fallback_rhs_strided<Scalar, ColMajor, true>(d, n, incr);
        verify_fallback_rhs_strided<Scalar, RowMajor, false>(d, n, incr);  // scalar path
        verify_fallback_rhs_strided<Scalar, RowMajor, true>(d, n, incr);
      }
      verify_fallback_lhs_byvalue<Scalar, ColMajor, false>(n, d);
      verify_fallback_lhs_byvalue<Scalar, ColMajor, true>(n, d);
      verify_fallback_lhs_byvalue<Scalar, RowMajor, false>(n, d);
      verify_fallback_lhs_byvalue<Scalar, RowMajor, true>(n, d);
      verify_fallback_rhs_byvalue<Scalar, ColMajor, false>(d, n);
      verify_fallback_rhs_byvalue<Scalar, ColMajor, true>(d, n);
      verify_fallback_rhs_byvalue<Scalar, RowMajor, false>(d, n);
      verify_fallback_rhs_byvalue<Scalar, RowMajor, true>(d, n);
    }
  }
}

// ---------------------------------------------------------------------------
// Product-level coverage, swept relative to the scalar's own block width.
// ---------------------------------------------------------------------------

// Sizes that land just on and off the block tails and the intra-block ZA-tile
// splits.  MR/2 is the tile side at the SVL=512 design point.
template <typename Scalar>
static std::vector<int> sme_edge_sizes() {
  const int MR = sme_mr<Scalar>();
  return {1, MR / 2 - 1, MR / 2, MR / 2 + 1, MR - 1, MR, MR + 1, 2 * MR - 1, 2 * MR, 2 * MR + 1};
}

template <typename Scalar>
static void test_products() {
  const int MR = sme_mr<Scalar>();

  // Square edge cases around the block and tile boundaries.
  for (int n : sme_edge_sizes<Scalar>()) product(SmeColMajorMat<Scalar>(n, n));

  // Thin / wide rectangular cases (M x 1, 1 x N) and non-square cases that
  // exercise tail paths for both M and N.
  product(SmeColMajorMat<Scalar>(MR, 1));
  product(SmeColMajorMat<Scalar>(1, MR));
  product(SmeColMajorMat<Scalar>(1, 2 * MR));
  product(SmeColMajorMat<Scalar>(2 * MR, 1));
  product(SmeColMajorMat<Scalar>(MR + 1, 2 * MR + 1));
  product(SmeColMajorMat<Scalar>(2 * MR + 1, MR + 1));
  product(SmeColMajorMat<Scalar>(MR - 1, 2 * MR - 1));
  product(SmeColMajorMat<Scalar>(MR + 1, 7));
  product(SmeColMajorMat<Scalar>(7, MR + 1));
  product(SmeColMajorMat<Scalar>(4 * MR, 3));
  product(SmeColMajorMat<Scalar>(3, 4 * MR));

  test_deep_k_split<Scalar>();

  // Random sizes
  for (int i = 0; i < g_repeat; i++) {
    product(SmeColMajorMat<Scalar>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                   internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }

  // Exercise the RowMajor packers and RowMajor result path.  When the input
  // MatrixType is RowMajor, product() instantiates m1/m2/m3/res in RowMajor,
  // so every matrix-matrix product in the suite flows through:
  //   - the RowMajor LHS packer (gemm_pack_lhs<..., RowMajor>)
  //   - the RowMajor RHS packer (gemm_pack_rhs<..., RowMajor>)
  //   - the RowMajor-C dispatch in GeneralMatrixMatrix.h (which transposes
  //     the computation: C^T = B^T * A^T).
  for (int n : sme_edge_sizes<Scalar>()) {
    if (n > 1) product(SmeRowMajorMat<Scalar>(n, n));
  }
  product(SmeRowMajorMat<Scalar>(MR + 1, 2 * MR + 1));
  product(SmeRowMajorMat<Scalar>(2 * MR + 1, MR + 1));
  for (int i = 0; i < g_repeat; i++) {
    product(SmeRowMajorMat<Scalar>(internal::random<int>(1, EIGEN_TEST_MAX_SIZE),
                                   internal::random<int>(1, EIGEN_TEST_MAX_SIZE)));
  }

  // Exercise the general-stride branch of sme_store_za_tile: fires when both
  // C_stride_row != 1 and C_stride_col != 1, e.g. a Map<Matrix> with an
  // explicit non-unit inner stride.  product.h never builds such a result, so
  // without this subtest the scalar-scatter path is effectively untested.
  for (int n : sme_edge_sizes<Scalar>()) {
    if (n < 2) continue;
    test_general_strided_result<Scalar, SmeColMajorMat<Scalar>>(n);
    test_general_strided_result<Scalar, SmeRowMajorMat<Scalar>>(n);
    test_rowmajor_strided_result<Scalar>(n);
  }

  // Row-LHS x Row-RHS -> Col-C: the one LHS/RHS/C storage combination that
  // product.h's transpose-style expressions never build directly (it always
  // flips one side of the multiplication).  The code paths are the same as
  // other combinations via Eigen's dispatch, but exercise them explicitly.
  for (int n : sme_edge_sizes<Scalar>()) {
    if (n < 2) continue;
    SmeRowMajorMat<Scalar> A = SmeRowMajorMat<Scalar>::Random(n, n);
    SmeRowMajorMat<Scalar> B = SmeRowMajorMat<Scalar>::Random(n, n);
    SmeColMajorMat<Scalar> C = SmeColMajorMat<Scalar>::Zero(n, n);
    C.noalias() += A * B;
    VERIFY_IS_APPROX(C, (A.lazyProduct(B)).eval());
  }
}

EIGEN_DECLARE_TEST(product_sme) {
  CALL_SUBTEST_1(test_products<float>());
  CALL_SUBTEST_1(test_symm_pack<float>());
  CALL_SUBTEST_1(test_mapper_fallback<float>());

  // double only reaches the SME kernel and packers with FEAT_SME_F64F64; the
  // product sweep is meaningful either way, but the packed-layout tests name
  // specializations that only exist when it is available.
  CALL_SUBTEST_2(test_products<double>());
#ifdef EIGEN_VECTORIZE_SME_F64F64
  CALL_SUBTEST_2(test_symm_pack<double>());
  CALL_SUBTEST_2(test_mapper_fallback<double>());
#endif

  // Scalar types SME does not specialize: these prove they still route through
  // the generic product path.
  CALL_SUBTEST_3(product(Matrix<std::complex<float>, Dynamic, Dynamic>(33, 17)));
  CALL_SUBTEST_3(product(Matrix<std::complex<double>, Dynamic, Dynamic>(33, 17)));
}
