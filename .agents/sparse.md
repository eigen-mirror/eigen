# Sparse Matrices And Solvers

Use this guide for [`Eigen/src/SparseCore`](../Eigen/src/SparseCore), the sparse decompositions in `SparseCholesky`,
`SparseLU`, `SparseQR`, and `OrderingMethods`, the external `*Support` backend wrappers, and their tests.
`IterativeLinearSolvers` shares the solver contract below; [`numerics.md`](numerics.md) governs accuracy expectations
for all of them. The checked-out headers are authoritative:

- [`SparseMatrix.h`](../Eigen/src/SparseCore/SparseMatrix.h) owns both storage modes, assembly, and resizing.
- [`SparseCompressedBase.h`](../Eigen/src/SparseCore/SparseCompressedBase.h) exposes the raw arrays, `InnerIterator`,
  and the inner-index sorting API.
- [`SparseRef.h`](../Eigen/src/SparseCore/SparseRef.h) and
  [`SparsityPatternRef.h`](../Eigen/src/SparseCore/SparsityPatternRef.h) define the non-owning views.
- [`SparseSolverBase.h`](../Eigen/src/SparseCore/SparseSolverBase.h) defines the `solve()` plumbing shared by direct
  solvers.
- [`test/sparse.h`](../test/sparse.h) and [`test/sparse_solver.h`](../test/sparse_solver.h) define the shared sparse
  test helpers.

## Compressed And Uncompressed Storage

A `SparseMatrix` is in one of two storage modes, and most bugs in this module come from a path that silently assumes
one. `m_innerNonZeros == nullptr` means compressed; when it is non-null, inner vector `j` occupies
`[outerIndexPtr()[j], outerIndexPtr()[j] + innerNonZeroPtr()[j])` rather than running to `outerIndexPtr()[j + 1]`.

- `insert()` and `coeffRef()` turn a compressed matrix into uncompressed mode when they add an entry. A function that
  takes a `SparseMatrix&` and inserts is therefore free to change the caller's storage mode; `makeCompressed()`
  restores it.
- Do not derive an entry count or an iteration bound from consecutive `outerIndexPtr()` differences. Use `nonZeros()`,
  `innerNonZeroPtr()`, or the uniform loop that `SparsityPatternRef.h` documents, which is correct in both modes.
- `resize()` zeroes the matrix, drops to compressed mode, and keeps the allocation; `conservativeResize()` preserves
  contents. Neither is a way to change storage mode deliberately.
- `Ref<SparseMatrix>` accepts an uncompressed argument unless it is declared with `StandardCompressedFormat`. With that
  option a writable `Ref` asserts `isCompressed()`, while a `Ref<const SparseMatrix, StandardCompressedFormat>`
  silently materializes a compressed copy instead of failing. A `Ref` parameter is consequently not proof that no copy
  happened; state which form a new API takes and why.
- `InnerIterator` and every raw pointer obtained from the matrix are invalidated by an insertion. Finish iterating, or
  collect the coordinates first and mutate afterwards.
- A consumer may require compressed input outright rather than adapting: `SparseQR::analyzePattern` opens with an
  `eigen_assert(mat.isCompressed())`, while `SparseLU` branches on `isCompressed()` and falls back to copying the outer
  index array. Because that check is an `eigen_assert`, a release build does not diagnose the violation at all — call
  `makeCompressed()` before handing an assembled matrix to a direct solver rather than relying on the assertion.

## Sorted Inner Indices

Sorted inner indices within each inner vector are an invariant most of the API maintains and several parts require.
`coeff()` resolves an entry by binary search over the inner range, and the sparse-sparse coefficient-wise binary
evaluator merges two inner streams by advancing whichever index is smaller. Both return wrong values on unsorted
input rather than failing.

The invariant is not universal, and the exception is a trap. `SparseQR::matrixR()` hands back a reference to a stored
factor built with `insertBackByOuterInnerUnordered`, so it is compressed but **not** sorted, and the rank-deficient
path preserves that: right-multiplying a column-major matrix by the pivot permutation takes the outer-permutation
branch, which relocates whole inner vectors without reordering within them. Compressed is not sorted, and a factor
returned by reference has had no opportunity to become either.

A storage-order round-trip is the fix, and it works for a reason worth knowing: a cross-order sparse assignment is a
counting transpose that walks the source in outer order, so each destination inner vector receives its entries in
ascending order no matter how the source was ordered. Eigen relies on this internally — `SparseQR::_sort_matrix_Q()`
sorts the stored reflectors exactly that way before `matrixQ()` is materialized into a sparse destination, which is why
`Q` is not a hazard in the way `R` is. Verify with `innerIndicesAreSorted()` rather than assuming a decomposition's
output factor satisfies the invariant.

- `setFromTriplets()` accepts unsorted input with duplicates and produces a sorted, compressed matrix with duplicates
  summed. It destroys the previous contents and does not resize — construct or `resize()` the matrix first, since the
  dimensions are not inferred from the triplets.
- `setFromSortedTriplets()`, `insertFromTriplets()`, and `insertFromSortedTriplets()` are the other three corners: the
  `Sorted` variants promise pre-sorted input, and the `insertFrom` variants merge into existing entries rather than
  replacing them. All four take an optional duplicate functor; the default sums.
- `insert()` requires that the entry not already exist. Use `coeffRef()` when it may, and `reserve(const SizesType&)`
  before random-order insertion — the sequential fast path only holds for increasing outer indices.
- The sparse-sparse product selectors preserve sortedness deliberately, choosing between a sorted insertion and an
  unsorted pass followed by a transpose round-trip that sorts as a side effect
  ([`ConservativeSparseSparseProduct.h`](../Eigen/src/SparseCore/ConservativeSparseSparseProduct.h)). A new product,
  permutation, or assembly path must restore the invariant; `sortInnerIndices()` and `innerIndicesAreSorted()` on
  `SparseCompressedBase` are the tools, and the latter belongs in a test rather than only in reasoning.

## Products

`A * B` on two sparse operands uses the conservative product; `(A * B).pruned()` selects the pruning product in
[`SparseSparseProductWithPruning.h`](../Eigen/src/SparseCore/SparseSparseProductWithPruning.h) instead. The distinction
is semantic before it is a matter of performance: the conservative path stores every structurally generated entry,
including one whose accumulation cancels to exactly zero, while the pruning path drops completed values at or below its
tolerance. The two therefore produce different patterns from the same operands, and a test or benchmark written against
one does not transfer to the other. Neither reserves the exact result size up front — the conservative path starts from
the heuristic `nonZerosEstimate()` sum documented at its definition and grows or over-allocates from there.

Threaded SpMV is opt-in: [`Eigen/SparseCore`](../Eigen/SparseCore) includes
`ThreadedSparseProduct.h` and `Eigen/ThreadPool` only under `EIGEN_USE_THREADS`. Coverage lives in
`test/sparse_threaded_product.cpp`, and [`tensor-threadpool.md`](tensor-threadpool.md) applies to its threading.

## Solver Contract

Direct sparse solvers split pattern analysis from numerical work: `analyzePattern()`, then `factorize()`, with
`compute()` doing both. Re-solving with the same pattern and new values must reuse the analysis; a change that forces a
re-analysis is a performance regression even when results match.

- Sparse solvers report failure through `info()`, not exceptions, and `info()` lives on each concrete solver and on
  `IterativeSolverBase` — not on `SparseSolverBase`. Check it after `compute()`/`factorize()` and again after `solve()`
  where the solver documents doing so. A test that ignores `info()` can pass on a matrix the solver rejected.
- `solve()` asserts that the solver was initialized, so a missing `compute()` surfaces only in a debug build.
- Reordering is part of the result: a solver's permutation affects fill-in and the achievable accuracy, so an ordering
  change needs the fill-in or timing evidence [`benchmarking.md`](benchmarking.md) asks for, not only a residual check.

## Testing Sparse Changes

`initSparse()` in `test/sparse.h` fills a dense reference and a sparse matrix together, with `ForceNonZeroDiag`,
`MakeLowerTriangular`, `MakeUpperTriangular`, and `ForceRealDiag` for the shapes solvers require. `test/sparse_solver.h`
provides the `check_sparse_solving`, `check_sparse_spd_solving`, `check_sparse_nonhermitian_solving`, and determinant
harnesses; prefer them to a hand-rolled solve so a new solver inherits the established coverage.

Scale coverage to the axes this module actually branches on: both storage orders, **both storage modes**, a
non-default `StorageIndex` width, complex scalars where conjugation is not a no-op, and a matrix with an empty inner
vector. Comparing against a dense reference computed by Eigen is the standard technique; keep the tolerance a named
epsilon multiple scaled by dimension or conditioning as [`testing.md`](testing.md) requires.

External backend tests are registered conditionally in [`test/CMakeLists.txt`](../test/CMakeLists.txt), so a green local
run says nothing about any of them. The full set is `cholmod_support`, `umfpack_support`, `klu_support`,
`superlu_support`, `pastix_support`, `spqr_support`, `accelerate_support`, `metis_support` — an ordering backend rather
than a solver — and `pardiso_support`. Most are gated on a `find_package` result and register in
`EIGEN_MISSING_BACKENDS` when absent, several additionally require `EIGEN_BUILD_BLAS` or `EIGEN_BUILD_LAPACK`, and
`metis_support` and `pastix_support` depend on variables the PaStiX search sets when the `METIS` component is requested.

`pardiso_support` is the exception worth knowing: the tree contains no `find_package(PARDISO)` and no
`EIGEN_MISSING_BACKENDS` entry for it, so it is registered only when `PARDISO_FOUND` arrives from outside the project
and its absence is silent even in the missing-backend summary. Report which sparse backends were unavailable rather than
implying full coverage.
