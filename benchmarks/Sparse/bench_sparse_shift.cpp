// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Operations that shift, pack or relocate stored entries inside a SparseMatrix or a SparseVector: sorted insertion,
// reserve(), makeCompressed(), adding a diagonal, assigning an inner vector, prune(), and setFromTriplets() with
// duplicates. Each runs for double, where a shift is a memmove, and for a scalar that owns heap memory, where it is
// one assignment per entry. A last case copies a matrix of a scalar whose user-provided copy only copies a field.

#include <benchmark/benchmark.h>
#include <Eigen/SparseCore>

#include <algorithm>
#include <memory>
#include <random>
#include <utility>
#include <vector>

using namespace Eigen;

namespace {

// Owns heap memory, as multi-precision and automatic-differentiation scalars do: a copy duplicates kLimbs doubles,
// a move exchanges a pointer.
class HeapScalar {
 public:
  static constexpr int kLimbs = 8;
  HeapScalar() = default;
  HeapScalar(double value) : limbs_(kLimbs, value) {}
  HeapScalar& operator+=(const HeapScalar& other) {
    if (other.limbs_.empty()) return *this;
    if (limbs_.empty()) limbs_.assign(kLimbs, 0.0);
    for (int i = 0; i < kLimbs; ++i) limbs_[i] += other.limbs_[i];
    return *this;
  }
  friend HeapScalar operator+(HeapScalar a, const HeapScalar& b) { return a += b; }
  double value() const { return limbs_.empty() ? 0.0 : limbs_[0]; }

 private:
  std::vector<double> limbs_;
};

// Not trivially copyable, though its copy only copies a field: smart_copy assigns it entry by entry, and the compiler
// may turn that loop into memcpy once it knows that the ranges do not overlap.
struct FieldCopyScalar {
  double value = 0;
  FieldCopyScalar() = default;
  FieldCopyScalar(double x) : value(x) {}
  FieldCopyScalar(const FieldCopyScalar& other) noexcept : value(other.value) {}
  FieldCopyScalar& operator=(const FieldCopyScalar& other) noexcept {
    value = other.value;
    return *this;
  }
};

double toDouble(double x) { return x; }
double toDouble(const HeapScalar& x) { return x.value(); }
double toDouble(const FieldCopyScalar& x) { return x.value; }

template <typename SparseType>
double sumOfEntries(const SparseType& m) {
  double sum = 0;
  for (Index j = 0; j < m.outerSize(); ++j)
    for (typename SparseType::InnerIterator it(m, j); it; ++it) sum += toDouble(it.value());
  return sum;
}

// A compressed size x size matrix with nnzPerVector entries in each column, none on the diagonal.
template <typename Scalar>
SparseMatrix<Scalar> makeOffDiagonal(Index size, Index nnzPerVector) {
  SparseMatrix<Scalar> m(size, size);
  m.reserve(VectorXi::Constant(size, int(nnzPerVector)));
  for (Index j = 0; j < size; ++j)
    for (Index k = 0; k < nnzPerVector; ++k) m.insert((j + 1 + 7 * k) % size, j) = Scalar(1.0);
  m.makeCompressed();
  return m;
}

}  // namespace

namespace Eigen {
template <>
struct NumTraits<HeapScalar> : NumTraits<double> {
  static constexpr int RequireInitialization = 1;
};
template <>
struct NumTraits<FieldCopyScalar> : NumTraits<double> {
  static constexpr int RequireInitialization = 0;
};
}  // namespace Eigen

// insert() into reserved room in descending inner order: every insertion shifts the entries already in its vector.
template <typename Scalar>
static void BM_InsertDescending(benchmark::State& state) {
  const Index nnzPerVector = state.range(0);
  const Index outerSize = 256;
  SparseMatrix<Scalar> m(4 * nnzPerVector, outerSize);
  const VectorXi sizes = VectorXi::Constant(outerSize, int(nnzPerVector));
  for (auto _ : state) {
    m.setZero();
    m.reserve(sizes);
    for (Index j = 0; j < outerSize; ++j)
      for (Index k = nnzPerVector - 1; k >= 0; --k) m.insert(3 * k, j) = Scalar(1.0);
    benchmark::DoNotOptimize(m.valuePtr());
    benchmark::ClobberMemory();
  }
  if (m.nonZeros() != outerSize * nnzPerVector || sumOfEntries(m) != double(outerSize * nnzPerVector))
    state.SkipWithError("insertion lost or changed coefficients");
  state.SetItemsProcessed(state.iterations() * outerSize * nnzPerVector);
}

// insert() in random order without reserve(): a vector that is full takes room from a neighbor or makes the matrix
// reserve one more slot per vector, and both shift entries across vectors.
template <typename Scalar>
static void BM_InsertRandomNoReserve(benchmark::State& state) {
  const Index nnzPerVector = state.range(0);
  const Index outerSize = 256;
  const Index innerSize = 4 * nnzPerVector;
  std::vector<std::pair<Index, Index>> coords;
  for (Index j = 0; j < outerSize; ++j)
    for (Index k = 0; k < nnzPerVector; ++k) coords.emplace_back(3 * k, j);
  std::shuffle(coords.begin(), coords.end(), std::mt19937(42));
  SparseMatrix<Scalar> m(innerSize, outerSize);
  for (auto _ : state) {
    m.setZero();
    for (const std::pair<Index, Index>& c : coords) m.insert(c.first, c.second) = Scalar(1.0);
    benchmark::DoNotOptimize(m.valuePtr());
    benchmark::ClobberMemory();
  }
  if (m.nonZeros() != outerSize * nnzPerVector || sumOfEntries(m) != double(outerSize * nnzPerVector))
    state.SkipWithError("insertion lost or changed coefficients");
  state.SetItemsProcessed(state.iterations() * outerSize * nnzPerVector);
}

// reserve() spreads the vectors of a compressed matrix apart, makeCompressed() packs them again.
template <typename Scalar>
static void BM_ReserveMakeCompressed(benchmark::State& state) {
  const Index nnzPerVector = state.range(0);
  const Index size = 1024;
  SparseMatrix<Scalar> m = makeOffDiagonal<Scalar>(size, nnzPerVector);
  const VectorXi sizes = VectorXi::Constant(size, 2);
  for (auto _ : state) {
    m.reserve(sizes);
    m.makeCompressed();
    benchmark::DoNotOptimize(m.valuePtr());
    benchmark::ClobberMemory();
  }
  if (m.nonZeros() != size * nnzPerVector || sumOfEntries(m) != double(size * nnzPerVector))
    state.SkipWithError("reserve() and makeCompressed() lost or changed coefficients");
  state.SetItemsProcessed(state.iterations() * size * nnzPerVector);
}

// m += diagonal where m stores no diagonal entry: one sorted insertion per vector. range(1) selects a compressed
// matrix, which shifts every entry behind an insertion, or one free slot per vector, which shifts within the vector.
template <typename Scalar>
static void BM_AddDiagonal(benchmark::State& state) {
  const Index nnzPerVector = state.range(0);
  const bool withCapacity = state.range(1) != 0;
  const Index size = 4096;
  const SparseMatrix<Scalar> base = makeOffDiagonal<Scalar>(size, nnzPerVector);
  const Matrix<Scalar, Dynamic, 1> diagonal = Matrix<Scalar, Dynamic, 1>::Constant(size, Scalar(1.0));
  const VectorXi sizes = VectorXi::Constant(size, 1);
  SparseMatrix<Scalar> m;
  for (auto _ : state) {
    state.PauseTiming();
    m = base;
    if (withCapacity) m.reserve(sizes);
    state.ResumeTiming();
    m += diagonal.asDiagonal();
    benchmark::DoNotOptimize(m.valuePtr());
    benchmark::ClobberMemory();
  }
  if (m.nonZeros() != size * (nnzPerVector + 1) || sumOfEntries(m) != double(size * (nnzPerVector + 1)))
    state.SkipWithError("adding the diagonal lost or changed coefficients");
  state.SetItemsProcessed(state.iterations() * size * nnzPerVector);
}

// Assigning the first inner vector of a compressed matrix shifts every entry behind it: right when the vector
// grows, left when it shrinks.
template <typename Scalar>
static void BM_AssignInnerVector(benchmark::State& state) {
  const Index nnzPerVector = state.range(0);
  const Index size = 1024;
  SparseMatrix<Scalar> m = makeOffDiagonal<Scalar>(size, nnzPerVector);
  SparseMatrix<Scalar> longer(size, 1), shorter(size, 1);
  for (Index k = 0; k < 2 * nnzPerVector; ++k) longer.insert(k, 0) = Scalar(1.0);
  shorter.insert(0, 0) = Scalar(1.0);
  longer.makeCompressed();
  shorter.makeCompressed();
  for (auto _ : state) {
    m.col(0) = longer;
    m.col(0) = shorter;
    benchmark::DoNotOptimize(m.valuePtr());
    benchmark::ClobberMemory();
  }
  if (m.nonZeros() != (size - 1) * nnzPerVector + 1 || sumOfEntries(m) != double((size - 1) * nnzPerVector + 1))
    state.SkipWithError("assigning an inner vector lost or changed coefficients");
  state.SetItemsProcessed(state.iterations() * 2 * (size - 1) * nnzPerVector);
}

// SparseVector::coeffRef() in descending index order: every insertion shifts all the stored entries.
template <typename Scalar>
static void BM_VectorCoeffRefDescending(benchmark::State& state) {
  const Index nnz = state.range(0);
  SparseVector<Scalar> v(4 * nnz);
  for (auto _ : state) {
    v.setZero();
    for (Index k = nnz - 1; k >= 0; --k) v.coeffRef(3 * k) = Scalar(1.0);
    benchmark::DoNotOptimize(v.valuePtr());
    benchmark::ClobberMemory();
  }
  if (v.nonZeros() != nnz || sumOfEntries(v) != double(nnz))
    state.SkipWithError("insertion lost or changed coefficients");
  state.SetItemsProcessed(state.iterations() * nnz);
}

// SparseVector::insert() in descending index order: every insertion shifts all the stored entries.
template <typename Scalar>
static void BM_VectorInsertDescending(benchmark::State& state) {
  const Index nnz = state.range(0);
  SparseVector<Scalar> v(4 * nnz);
  for (auto _ : state) {
    v.setZero();
    for (Index k = nnz - 1; k >= 0; --k) v.insert(3 * k) = Scalar(1.0);
    benchmark::DoNotOptimize(v.valuePtr());
    benchmark::ClobberMemory();
  }
  if (v.nonZeros() != nnz || sumOfEntries(v) != double(nnz))
    state.SkipWithError("insertion lost or changed coefficients");
  state.SetItemsProcessed(state.iterations() * nnz);
}

// prune() keeping the even inner indices of a compressed matrix: about half the entries move to the front. The
// timing includes releasing the matrix: a move into a dropped entry frees its memory during prune(), where a copy
// reuses it and leaves the release to the destructor.
template <typename Scalar>
static void BM_Prune(benchmark::State& state) {
  const Index nnzPerVector = state.range(0);
  const Index size = 1024;
  const SparseMatrix<Scalar> base = makeOffDiagonal<Scalar>(size, nnzPerVector);
  auto keep = [](Index row, Index, const Scalar&) { return row % 2 == 0; };
  for (auto _ : state) {
    state.PauseTiming();
    std::unique_ptr<SparseMatrix<Scalar>> m(new SparseMatrix<Scalar>(base));
    state.ResumeTiming();
    m->prune(keep);
    benchmark::DoNotOptimize(m->valuePtr());
    m.reset();
    benchmark::ClobberMemory();
  }
  Index kept = 0;
  for (Index j = 0; j < size; ++j)
    for (typename SparseMatrix<Scalar>::InnerIterator it(base, j); it; ++it) kept += it.row() % 2 == 0;
  SparseMatrix<Scalar> m = base;
  m.prune(keep);
  if (m.nonZeros() != kept || sumOfEntries(m) != double(kept)) state.SkipWithError("prune() kept the wrong entries");
  state.SetItemsProcessed(state.iterations() * size * nnzPerVector);
}

// setFromTriplets() with every coefficient given twice, in random order: summing the duplicates packs the first
// entry of each coefficient to the front of its vector.
template <typename Scalar>
static void BM_SetFromTripletsDuplicates(benchmark::State& state) {
  const Index nnzPerVector = state.range(0);
  const Index size = 1024;
  std::vector<Triplet<Scalar>> triplets;
  for (Index j = 0; j < size; ++j)
    for (Index k = 0; k < nnzPerVector; ++k)
      for (int copy = 0; copy < 2; ++copy) triplets.emplace_back(int((j + 1 + 7 * k) % size), int(j), Scalar(1.0));
  std::shuffle(triplets.begin(), triplets.end(), std::mt19937(42));
  SparseMatrix<Scalar> m(size, size);
  for (auto _ : state) {
    m.setFromTriplets(triplets.begin(), triplets.end());
    benchmark::DoNotOptimize(m.valuePtr());
    benchmark::ClobberMemory();
  }
  if (m.nonZeros() != size * nnzPerVector || sumOfEntries(m) != 2.0 * double(size * nnzPerVector))
    state.SkipWithError("setFromTriplets() lost or changed coefficients");
  state.SetItemsProcessed(state.iterations() * Index(triplets.size()));
}

// Assigning a longer first inner vector to an uncompressed matrix with no free room reallocates the storage, and
// every other entry goes to the new buffer.
template <typename Scalar>
static void BM_AssignInnerVectorUncompressed(benchmark::State& state) {
  const Index nnzPerVector = state.range(0);
  const Index size = 1024;
  const SparseMatrix<Scalar> base = makeOffDiagonal<Scalar>(size, nnzPerVector);
  SparseMatrix<Scalar> longer(size, 1);
  for (Index k = 0; k < 2 * nnzPerVector; ++k) longer.insert(k, 0) = Scalar(1.0);
  longer.makeCompressed();
  SparseMatrix<Scalar> m;
  for (auto _ : state) {
    state.PauseTiming();
    m = SparseMatrix<Scalar>(base);  // a fresh copy, so the storage has no free room
    m.uncompress();
    state.ResumeTiming();
    m.col(0) = longer;
    benchmark::DoNotOptimize(m.valuePtr());
    benchmark::ClobberMemory();
  }
  if (m.nonZeros() != (size + 1) * nnzPerVector || sumOfEntries(m) != double((size + 1) * nnzPerVector))
    state.SkipWithError("assigning an inner vector lost or changed coefficients");
  state.SetItemsProcessed(state.iterations() * size * nnzPerVector);
}

// Copy assignment into storage that is already large enough copies the values with smart_copy.
template <typename Scalar>
static void BM_CopyAssign(benchmark::State& state) {
  const Index nnzPerVector = state.range(0);
  const Index size = 256;
  const SparseMatrix<Scalar> source = makeOffDiagonal<Scalar>(size, nnzPerVector);
  SparseMatrix<Scalar> m = source;
  for (auto _ : state) {
    m = source;
    benchmark::DoNotOptimize(m.valuePtr());
    benchmark::ClobberMemory();
  }
  if (m.nonZeros() != size * nnzPerVector || sumOfEntries(m) != double(size * nnzPerVector))
    state.SkipWithError("copy assignment lost or changed coefficients");
  state.SetItemsProcessed(state.iterations() * size * nnzPerVector);
}

BENCHMARK_TEMPLATE(BM_InsertDescending, double)->Arg(4)->Arg(16)->Arg(64);
BENCHMARK_TEMPLATE(BM_InsertDescending, HeapScalar)->Arg(4)->Arg(16)->Arg(64);
BENCHMARK_TEMPLATE(BM_InsertRandomNoReserve, double)->Arg(4)->Arg(16);
BENCHMARK_TEMPLATE(BM_InsertRandomNoReserve, HeapScalar)->Arg(4)->Arg(16);
BENCHMARK_TEMPLATE(BM_ReserveMakeCompressed, double)->Arg(4)->Arg(16)->Arg(64);
BENCHMARK_TEMPLATE(BM_ReserveMakeCompressed, HeapScalar)->Arg(4)->Arg(16)->Arg(64);
BENCHMARK_TEMPLATE(BM_AddDiagonal, double)->ArgsProduct({{4, 16, 64}, {0, 1}});
BENCHMARK_TEMPLATE(BM_AddDiagonal, HeapScalar)->ArgsProduct({{4, 16, 64}, {0, 1}});
BENCHMARK_TEMPLATE(BM_AssignInnerVector, double)->Arg(4)->Arg(16);
BENCHMARK_TEMPLATE(BM_AssignInnerVector, HeapScalar)->Arg(4)->Arg(16);
BENCHMARK_TEMPLATE(BM_VectorCoeffRefDescending, double)->Arg(16)->Arg(128)->Arg(1024);
BENCHMARK_TEMPLATE(BM_VectorCoeffRefDescending, HeapScalar)->Arg(16)->Arg(128)->Arg(1024);
BENCHMARK_TEMPLATE(BM_VectorInsertDescending, double)->Arg(16)->Arg(128)->Arg(1024);
BENCHMARK_TEMPLATE(BM_VectorInsertDescending, HeapScalar)->Arg(16)->Arg(128)->Arg(1024);
BENCHMARK_TEMPLATE(BM_Prune, double)->Arg(4)->Arg(16)->Arg(64);
BENCHMARK_TEMPLATE(BM_Prune, HeapScalar)->Arg(4)->Arg(16)->Arg(64);
BENCHMARK_TEMPLATE(BM_SetFromTripletsDuplicates, double)->Arg(4)->Arg(16);
BENCHMARK_TEMPLATE(BM_SetFromTripletsDuplicates, HeapScalar)->Arg(4)->Arg(16);
BENCHMARK_TEMPLATE(BM_AssignInnerVectorUncompressed, double)->Arg(4)->Arg(16);
BENCHMARK_TEMPLATE(BM_AssignInnerVectorUncompressed, HeapScalar)->Arg(4)->Arg(16);
BENCHMARK_TEMPLATE(BM_CopyAssign, FieldCopyScalar)->Arg(4)->Arg(16)->Arg(64);
