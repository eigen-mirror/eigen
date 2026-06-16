// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Benchmark: SparseMatrix vs BlockSparseMatrix, real and complex scalars.

#include <benchmark/benchmark.h>
#include <Eigen/Sparse>

#include <complex>
#include <random>
#include <set>

using namespace Eigen;
using cd = std::complex<double>;

// ---------------------------------------------------------------------------
// Random-value helper — one call per real scalar, two for complex.
// ---------------------------------------------------------------------------
template <typename Scalar>
static typename std::enable_if<!std::is_same<Scalar, cd>::value, Scalar>::type
randVal(std::mt19937& rng, std::normal_distribution<double>& d) { return Scalar(d(rng)); }

template <typename Scalar>
static typename std::enable_if<std::is_same<Scalar, cd>::value, Scalar>::type
randVal(std::mt19937& rng, std::normal_distribution<double>& d) { return cd{d(rng), d(rng)}; }

// ---------------------------------------------------------------------------
// Build a general (full) block-sparse pair.
// ---------------------------------------------------------------------------
template <typename Scalar, int B>
static void buildPair(int nB, int nnzPerCol, unsigned seed,
                      BlockSparseMatrix<Scalar, ColMajor, B, B>& bsm,
                      SparseMatrix<Scalar>& sm) {
  using BSM     = BlockSparseMatrix<Scalar, ColMajor, B, B>;
  using BT      = typename BSM::BlockType;
  using Triplet = typename BSM::TripletType;

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> rowDist(0, nB - 1);
  std::normal_distribution<double>   vd;

  std::vector<Triplet> triplets;
  triplets.reserve(nB * nnzPerCol);
  for (int j = 0; j < nB; ++j) {
    std::set<int> rows;
    rows.insert(j % nB);
    while ((int)rows.size() < std::min(nnzPerCol, nB)) rows.insert(rowDist(rng));
    for (int bi : rows) {
      BT blk;
      for (int r = 0; r < B; ++r)
        for (int c = 0; c < B; ++c)
          blk(r, c) = randVal<Scalar>(rng, vd);
      triplets.emplace_back(bi, j, blk);
    }
  }
  bsm = BSM(nB, nB);
  bsm.setFromTriplets(triplets.begin(), triplets.end());
  sm = bsm.toSparse();
}

// ---------------------------------------------------------------------------
// Build an upper-triangular block-sparse pair.
// forSolve=true → diagonal blocks diagonally dominant (well-conditioned).
// ---------------------------------------------------------------------------
template <typename Scalar, int B>
static void buildUpperTriPair(int nB, int nnzPerCol, unsigned seed,
                              BlockSparseMatrix<Scalar, ColMajor, B, B>& bsm,
                              SparseMatrix<Scalar>& sm,
                              bool forSolve = false) {
  using BSM     = BlockSparseMatrix<Scalar, ColMajor, B, B>;
  using BT      = typename BSM::BlockType;
  using Triplet = typename BSM::TripletType;

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> rowDist(0, nB - 1);
  std::normal_distribution<double>   vd;

  std::vector<Triplet> triplets;
  triplets.reserve(nB * nnzPerCol);
  for (int j = 0; j < nB; ++j) {
    BT diag;
    for (int r = 0; r < B; ++r)
      for (int c = 0; c < B; ++c)
        diag(r, c) = randVal<Scalar>(rng, vd);
    if (forSolve) {
      diag *= Scalar(0.1);
      for (int d = 0; d < B; ++d) diag(d, d) += Scalar(double(B));
    }
    triplets.emplace_back(j, j, diag);

    std::set<int> rows;
    while ((int)rows.size() < std::min(nnzPerCol - 1, j))
      rows.insert(rowDist(rng) % j);
    for (int bi : rows) {
      BT blk;
      for (int r = 0; r < B; ++r)
        for (int c = 0; c < B; ++c)
          blk(r, c) = randVal<Scalar>(rng, vd);
      triplets.emplace_back(bi, j, blk);
    }
  }
  bsm = BSM(nB, nB);
  bsm.setFromTriplets(triplets.begin(), triplets.end());
  sm = bsm.toSparse();
}

// ---------------------------------------------------------------------------
// Build an upper-triangular pair with Hermitian diagonal blocks.
// Makes DiagIsSelfAdjoint=true semantically correct.
// ---------------------------------------------------------------------------
template <typename Scalar, int B>
static void buildHermDiagUpperTriPair(int nB, int nnzPerCol, unsigned seed,
                                      BlockSparseMatrix<Scalar, ColMajor, B, B>& bsm,
                                      SparseMatrix<Scalar>& sm) {
  using BSM     = BlockSparseMatrix<Scalar, ColMajor, B, B>;
  using BT      = typename BSM::BlockType;
  using Triplet = typename BSM::TripletType;

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> rowDist(0, nB - 1);
  std::normal_distribution<double>   vd;

  std::vector<Triplet> triplets;
  triplets.reserve(nB * nnzPerCol);
  for (int j = 0; j < nB; ++j) {
    BT raw;
    for (int r = 0; r < B; ++r)
      for (int c = 0; c < B; ++c)
        raw(r, c) = randVal<Scalar>(rng, vd);
    BT diag = (raw + raw.adjoint()) / Scalar(2);  // Hermitian for complex, symmetric for real
    triplets.emplace_back(j, j, diag);

    std::set<int> rows;
    while ((int)rows.size() < std::min(nnzPerCol - 1, j))
      rows.insert(rowDist(rng) % j);
    for (int bi : rows) {
      BT blk;
      for (int r = 0; r < B; ++r)
        for (int c = 0; c < B; ++c)
          blk(r, c) = randVal<Scalar>(rng, vd);
      triplets.emplace_back(bi, j, blk);
    }
  }
  bsm = BSM(nB, nB);
  bsm.setFromTriplets(triplets.begin(), triplets.end());
  sm = bsm.toSparse();
}

// ---------------------------------------------------------------------------
// Addition
// ---------------------------------------------------------------------------
template <typename Scalar, int B>
static void BM_SparseAdd(benchmark::State& state) {
  int nB = state.range(0), nnz = state.range(1);
  SparseMatrix<Scalar> smA, smB, smC;
  BlockSparseMatrix<Scalar, ColMajor, B, B> tmp;
  buildPair<Scalar, B>(nB, nnz, 1, tmp, smA);
  buildPair<Scalar, B>(nB, nnz, 2, tmp, smB);
  for (auto _ : state) { smC = smA + smB; benchmark::DoNotOptimize(smC.valuePtr()); }
  state.counters["n"] = smA.rows();
}

template <typename Scalar, int B>
static void BM_BlockSparseAdd(benchmark::State& state) {
  int nB = state.range(0), nnz = state.range(1);
  using BSM = BlockSparseMatrix<Scalar, ColMajor, B, B>;
  BSM bsmA, bsmB, bsmC; SparseMatrix<Scalar> smTmp;
  buildPair<Scalar, B>(nB, nnz, 1, bsmA, smTmp);
  buildPair<Scalar, B>(nB, nnz, 2, bsmB, smTmp);
  for (auto _ : state) { bsmC = bsmA + bsmB; benchmark::DoNotOptimize(bsmC.valuePtr()); }
  state.counters["n"] = bsmA.rows();
}

// ---------------------------------------------------------------------------
// SpGEMV
// ---------------------------------------------------------------------------
template <typename Scalar, int B>
static void BM_SparseGEMV(benchmark::State& state) {
  int nB = state.range(0), nnz = state.range(1);
  SparseMatrix<Scalar> sm; BlockSparseMatrix<Scalar, ColMajor, B, B> tmp;
  buildPair<Scalar, B>(nB, nnz, 1, tmp, sm);
  Matrix<Scalar, Dynamic, 1> x = Matrix<Scalar, Dynamic, 1>::Random(sm.cols());
  Matrix<Scalar, Dynamic, 1> y(sm.rows());
  for (auto _ : state) { y.noalias() = sm * x; benchmark::DoNotOptimize(y.data()); }
  state.counters["n"] = sm.rows();
}

template <typename Scalar, int B>
static void BM_BlockSparseGEMV(benchmark::State& state) {
  int nB = state.range(0), nnz = state.range(1);
  using BSM = BlockSparseMatrix<Scalar, ColMajor, B, B>;
  BSM bsm; SparseMatrix<Scalar> smTmp;
  buildPair<Scalar, B>(nB, nnz, 1, bsm, smTmp);
  Matrix<Scalar, Dynamic, 1> x = Matrix<Scalar, Dynamic, 1>::Random(bsm.cols());
  Matrix<Scalar, Dynamic, 1> y(bsm.rows());
  for (auto _ : state) { y.noalias() = bsm * x; benchmark::DoNotOptimize(y.data()); }
  state.counters["n"] = bsm.rows();
}

// ---------------------------------------------------------------------------
// Triangular SpMV
// ---------------------------------------------------------------------------
template <typename Scalar, int B>
static void BM_SparseTriMV(benchmark::State& state) {
  int nB = state.range(0), nnz = state.range(1);
  SparseMatrix<Scalar> sm; BlockSparseMatrix<Scalar, ColMajor, B, B> tmp;
  buildUpperTriPair<Scalar, B>(nB, nnz, 1, tmp, sm);
  Matrix<Scalar, Dynamic, 1> x = Matrix<Scalar, Dynamic, 1>::Random(sm.cols());
  Matrix<Scalar, Dynamic, 1> y(sm.rows());
  for (auto _ : state) { y.noalias() = sm.template triangularView<Upper>() * x; benchmark::DoNotOptimize(y.data()); }
  state.counters["n"] = sm.rows();
}

template <typename Scalar, int B>
static void BM_BlockSparseTriMV(benchmark::State& state) {
  int nB = state.range(0), nnz = state.range(1);
  using BSM = BlockSparseMatrix<Scalar, ColMajor, B, B>;
  BSM bsm; SparseMatrix<Scalar> smTmp;
  buildUpperTriPair<Scalar, B>(nB, nnz, 1, bsm, smTmp);
  Matrix<Scalar, Dynamic, 1> x = Matrix<Scalar, Dynamic, 1>::Random(bsm.cols());
  Matrix<Scalar, Dynamic, 1> y(bsm.rows());
  for (auto _ : state) { y.noalias() = bsm.template triangularView<Upper>() * x; benchmark::DoNotOptimize(y.data()); }
  state.counters["n"] = bsm.rows();
}

// ---------------------------------------------------------------------------
// Selfadjoint SpMV  (DiagIsSelfAdjoint=false, general diagonal blocks)
// ---------------------------------------------------------------------------
template <typename Scalar, int B>
static void BM_SparseSymmMV(benchmark::State& state) {
  int nB = state.range(0), nnz = state.range(1);
  SparseMatrix<Scalar> sm; BlockSparseMatrix<Scalar, ColMajor, B, B> tmp;
  buildUpperTriPair<Scalar, B>(nB, nnz, 1, tmp, sm);
  Matrix<Scalar, Dynamic, 1> x = Matrix<Scalar, Dynamic, 1>::Random(sm.cols());
  Matrix<Scalar, Dynamic, 1> y(sm.rows());
  for (auto _ : state) { y.noalias() = sm.template selfadjointView<Upper>() * x; benchmark::DoNotOptimize(y.data()); }
  state.counters["n"] = sm.rows();
}

template <typename Scalar, int B>
static void BM_BlockSparseSymmMV(benchmark::State& state) {
  int nB = state.range(0), nnz = state.range(1);
  using BSM = BlockSparseMatrix<Scalar, ColMajor, B, B>;
  BSM bsm; SparseMatrix<Scalar> smTmp;
  buildUpperTriPair<Scalar, B>(nB, nnz, 1, bsm, smTmp);
  Matrix<Scalar, Dynamic, 1> x = Matrix<Scalar, Dynamic, 1>::Random(bsm.cols());
  Matrix<Scalar, Dynamic, 1> y(bsm.rows());
  for (auto _ : state) { y.noalias() = bsm.template selfadjointView<Upper>() * x; benchmark::DoNotOptimize(y.data()); }
  state.counters["n"] = bsm.rows();
}

// ---------------------------------------------------------------------------
// Selfadjoint SpMV — Hermitian diagonal blocks, DiagIsSelfAdjoint=false/true
// ---------------------------------------------------------------------------
template <typename Scalar, int B>
static void BM_BlockSparseSymmMV_DiagNSA(benchmark::State& state) {
  int nB = state.range(0), nnz = state.range(1);
  using BSM = BlockSparseMatrix<Scalar, ColMajor, B, B>;
  BSM bsm; SparseMatrix<Scalar> smTmp;
  buildHermDiagUpperTriPair<Scalar, B>(nB, nnz, 1, bsm, smTmp);
  Matrix<Scalar, Dynamic, 1> x = Matrix<Scalar, Dynamic, 1>::Random(bsm.cols());
  Matrix<Scalar, Dynamic, 1> y(bsm.rows());
  for (auto _ : state) { y.noalias() = bsm.template selfadjointView<Upper, false>() * x; benchmark::DoNotOptimize(y.data()); }
  state.counters["n"] = bsm.rows();
}

template <typename Scalar, int B>
static void BM_BlockSparseSymmMV_DiagSA(benchmark::State& state) {
  int nB = state.range(0), nnz = state.range(1);
  using BSM = BlockSparseMatrix<Scalar, ColMajor, B, B>;
  BSM bsm; SparseMatrix<Scalar> smTmp;
  buildHermDiagUpperTriPair<Scalar, B>(nB, nnz, 1, bsm, smTmp);
  Matrix<Scalar, Dynamic, 1> x = Matrix<Scalar, Dynamic, 1>::Random(bsm.cols());
  Matrix<Scalar, Dynamic, 1> y(bsm.rows());
  for (auto _ : state) { y.noalias() = bsm.template selfadjointView<Upper, true>() * x; benchmark::DoNotOptimize(y.data()); }
  state.counters["n"] = bsm.rows();
}

// ---------------------------------------------------------------------------
// Triangular solve
// ---------------------------------------------------------------------------
template <typename Scalar, int B>
static void BM_SparseTriSolve(benchmark::State& state) {
  int nB = state.range(0), nnz = state.range(1);
  SparseMatrix<Scalar> sm; BlockSparseMatrix<Scalar, ColMajor, B, B> tmp;
  buildUpperTriPair<Scalar, B>(nB, nnz, 1, tmp, sm, true);
  Matrix<Scalar, Dynamic, 1> rhs = Matrix<Scalar, Dynamic, 1>::Random(sm.cols());
  Matrix<Scalar, Dynamic, 1> x(sm.cols());
  for (auto _ : state) {
    x = rhs;
    sm.template triangularView<Upper>().solveInPlace(x);
    benchmark::DoNotOptimize(x.data());
  }
  state.counters["n"] = sm.rows();
}

template <typename Scalar, int B>
static void BM_BlockSparseTriSolve(benchmark::State& state) {
  int nB = state.range(0), nnz = state.range(1);
  using BSM = BlockSparseMatrix<Scalar, ColMajor, B, B>;
  BSM bsm; SparseMatrix<Scalar> smTmp;
  buildUpperTriPair<Scalar, B>(nB, nnz, 1, bsm, smTmp, true);
  Matrix<Scalar, Dynamic, 1> rhs = Matrix<Scalar, Dynamic, 1>::Random(bsm.cols());
  Matrix<Scalar, Dynamic, 1> x(bsm.cols());
  for (auto _ : state) {
    x = rhs;
    bsm.template triangularView<Upper>().solveInPlace(x);
    benchmark::DoNotOptimize(x.data());
  }
  state.counters["n"] = bsm.rows();
}

// ---------------------------------------------------------------------------
// SpGEMM
// ---------------------------------------------------------------------------
template <typename Scalar, int B>
static void BM_SparseMul(benchmark::State& state) {
  int nB = state.range(0), nnz = state.range(1);
  SparseMatrix<Scalar> smA, smB, smC;
  BlockSparseMatrix<Scalar, ColMajor, B, B> tmp;
  buildPair<Scalar, B>(nB, nnz, 1, tmp, smA);
  buildPair<Scalar, B>(nB, nnz, 2, tmp, smB);
  for (auto _ : state) { smC = smA * smB; benchmark::DoNotOptimize(smC.valuePtr()); }
  state.counters["n"] = smA.rows();
}

template <typename Scalar, int B>
static void BM_BlockSparseMul(benchmark::State& state) {
  int nB = state.range(0), nnz = state.range(1);
  using BSM = BlockSparseMatrix<Scalar, ColMajor, B, B>;
  BSM bsmA, bsmB, bsmC; SparseMatrix<Scalar> smTmp;
  buildPair<Scalar, B>(nB, nnz, 1, bsmA, smTmp);
  buildPair<Scalar, B>(nB, nnz, 2, bsmB, smTmp);
  for (auto _ : state) { bsmC = bsmA * bsmB; benchmark::DoNotOptimize(bsmC.valuePtr()); }
  state.counters["n"] = bsmA.rows();
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

#define NS  benchmark::kNanosecond
#define US  benchmark::kMicrosecond
#define REG(fn, S, B) BENCHMARK(fn<S,B>)->Args({100,10})->Args({200,10})

#define BENCH_TYPE(S, B) \
  REG(BM_SparseAdd,                S, B)->Unit(US); \
  REG(BM_BlockSparseAdd,           S, B)->Unit(US); \
  REG(BM_SparseGEMV,               S, B)->Unit(NS); \
  REG(BM_BlockSparseGEMV,          S, B)->Unit(NS); \
  REG(BM_SparseTriMV,              S, B)->Unit(NS); \
  REG(BM_BlockSparseTriMV,         S, B)->Unit(NS); \
  REG(BM_SparseSymmMV,             S, B)->Unit(NS); \
  REG(BM_BlockSparseSymmMV,        S, B)->Unit(NS); \
  REG(BM_BlockSparseSymmMV_DiagNSA,S, B)->Unit(NS); \
  REG(BM_BlockSparseSymmMV_DiagSA, S, B)->Unit(NS); \
  REG(BM_SparseTriSolve,           S, B)->Unit(NS); \
  REG(BM_BlockSparseTriSolve,      S, B)->Unit(NS); \
  REG(BM_SparseMul,                S, B)->Unit(US); \
  REG(BM_BlockSparseMul,           S, B)->Unit(US);

BENCH_TYPE(double, 2)  BENCH_TYPE(cd, 2)
BENCH_TYPE(double, 3)  BENCH_TYPE(cd, 3)
BENCH_TYPE(double, 4)  BENCH_TYPE(cd, 4)
