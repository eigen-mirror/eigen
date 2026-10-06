# Sparse Matrices And Solvers

Use this guide for [`Eigen/src/SparseCore`](../Eigen/src/SparseCore), the sparse decompositions in `SparseCholesky`,
`SparseLU`, `SparseQR`, and `OrderingMethods`, the external `*Support` backend wrappers, and their tests.
`IterativeLinearSolvers` shares the solver contract below; [`numerics.md`](numerics.md) governs accuracy expectations
for all of them. The checked-out headers are authoritative:

- [`SparseMatrix.h`](../Eigen/src/SparseCore/SparseMatrix.h) implements both storage modes, assembly, and resizing.
- [`SparseCompressedBase.h`](../Eigen/src/SparseCore/SparseCompressedBase.h) exposes the raw arrays, `InnerIterator`,
  and the inner-index sorting API.
- [`SparseRef.h`](../Eigen/src/SparseCore/SparseRef.h) and
  [`SparsityPatternRef.h`](../Eigen/src/SparseCore/SparsityPatternRef.h) define the non-owning views.
- [`SparseSolverBase.h`](../Eigen/src/SparseCore/SparseSolverBase.h) defines the `solve()` plumbing shared by direct
  solvers.
- [`test/sparse.h`](../test/sparse.h) and [`test/sparse_solver.h`](../test/sparse_solver.h) define the shared sparse
  test helpers.

## Compressed And Uncompressed Storage

A `SparseMatrix` is in one of two storage modes, compressed or uncompressed, and most bugs in this module come from code
that silently assumes one of the two. The matrix is compressed when `m_innerNonZeros == nullptr`. When it is non-null,
inner vector `j` occupies `[outerIndexPtr()[j], outerIndexPtr()[j] + innerNonZeroPtr()[j])` rather than running to
`outerIndexPtr()[j + 1]`.

- `insert()` and `coeffRef()` turn a compressed matrix into uncompressed mode when they add an entry. A function that
  takes a `SparseMatrix&` and inserts is therefore free to change the caller's storage mode; `makeCompressed()`
  restores it.
- Do not derive an entry count or an iteration bound from consecutive `outerIndexPtr()` differences. Use `nonZeros()`,
  `innerNonZeroPtr()`, or the uniform loop that `SparsityPatternRef.h` documents, which is correct in both modes.
- `resize()` zeroes the matrix, drops to compressed mode, and keeps the allocation; `conservativeResize()` preserves
  contents. Neither is a way to change storage mode deliberately.
- `Ref<SparseMatrix>` accepts an uncompressed argument unless it is declared with `StandardCompressedFormat`. With that
  option a writable `Ref` asserts `isCompressed()`, while a `Ref<const SparseMatrix, StandardCompressedFormat>`
  silently makes a compressed copy instead of failing. So a `Ref` parameter does not prove that no copy was made. For a
  new API, state which form it takes and why.
- `InnerIterator` and every raw pointer obtained from the matrix are invalidated by an insertion. Finish iterating, or
  collect the coordinates first and mutate afterwards.
- Some consumers require compressed input instead of handling both modes. `SparseQR::analyzePattern` starts with
  `eigen_assert(mat.isCompressed())`, while `SparseLU` branches on `isCompressed()` and falls back to copying the outer
  index array. Because the `SparseQR` check is an `eigen_assert`, a release build does not report the violation at all.
  Call `makeCompressed()` before passing an assembled matrix to a direct solver rather than relying on the assertion.

## Sorted Inner Indices

Most of the API keeps the inner indices of each inner vector sorted, and several parts require it. `coeff()` finds an
entry by binary search over the inner range. The evaluator for a coefficient-wise operation on two sparse operands,
such as `A + B`, merges the two inner vectors by advancing whichever index is smaller. On unsorted input both return
wrong values instead of failing.

Sorted order is not guaranteed everywhere, and the exception is easy to miss. `SparseQR::matrixR()` returns a reference
to a stored factor built with `insertBackByOuterInnerUnordered`, so it is compressed but **not** sorted. The
rank-deficient path keeps it unsorted: it right-multiplies a column-major matrix by the pivot permutation, and that
product takes the outer-permutation branch, which moves whole inner vectors without reordering the entries inside them.
A compressed matrix is not necessarily sorted, and a factor returned by reference has not been through an assignment
that would compress or sort it.

To sort such a matrix, assign it to a matrix of the other storage order and back (a storage-order round-trip). This
works because an assignment between storage orders is a counting transpose: it walks the source in outer order, so
each destination inner vector receives its entries in ascending order however the source was ordered. Eigen relies on
this internally. `SparseQR::_sort_matrix_Q()` sorts the stored reflectors this way before `matrixQ()` is materialized
into a sparse destination, which is why `Q` is not a hazard in the way `R` is. Check a decomposition's output factor
with `innerIndicesAreSorted()` rather than assuming it is sorted.

- `setFromTriplets()` accepts unsorted input with duplicates and produces a sorted, compressed matrix with duplicates
  summed. It destroys the previous contents and does not resize — construct or `resize()` the matrix first, since the
  dimensions are not inferred from the triplets.
- `setFromSortedTriplets()`, `insertFromTriplets()`, and `insertFromSortedTriplets()` complete the set. The `Sorted`
  variants assume the input is already sorted, and the `insertFrom` variants merge into existing entries rather than
  replacing them. All four take an optional functor for combining duplicates; the default sums them.
- `insert()` requires that the entry not already exist. Use `coeffRef()` when it may exist. Before inserting in random
  order, call `reserve(const SizesType&)`: the sequential fast path applies only when outer indices increase.
- The sparse-sparse product selectors keep the result sorted on purpose. They choose between inserting in sorted order
  and an unsorted pass followed by a transpose round-trip, which sorts as a side effect
  ([`ConservativeSparseSparseProduct.h`](../Eigen/src/SparseCore/ConservativeSparseSparseProduct.h)). A new product,
  permutation, or assembly path must also leave the indices sorted. `sortInnerIndices()` and `innerIndicesAreSorted()`
  on `SparseCompressedBase` are the tools for this; check `innerIndicesAreSorted()` in a test, not only in reasoning.

## Products

`A * B` on two sparse operands uses the conservative product; `(A * B).pruned()` selects the pruning product in
[`SparseSparseProductWithPruning.h`](../Eigen/src/SparseCore/SparseSparseProductWithPruning.h) instead. The two
differ in results, not only in speed. The conservative product stores every entry that the operands' sparsity patterns
generate, even one whose sum cancels to exactly zero. The pruning product drops finished values at or below its
tolerance. So the two produce different patterns from the same operands, and a test or benchmark written against one
does not transfer to the other. Neither reserves the exact result size in advance. The conservative product starts
from the heuristic `nonZerosEstimate()` sum, documented at its definition, and grows or over-allocates from there.

Threaded SpMV is opt-in: [`Eigen/SparseCore`](../Eigen/SparseCore) includes `ThreadedSparseProduct.h` and
`Eigen/ThreadPool` only when `EIGEN_USE_THREADS` is defined. Its tests are in `test/sparse_threaded_product.cpp`, and
[`tensor-threadpool.md`](tensor-threadpool.md) applies to its threading.

## Solver Contract

Direct sparse solvers split pattern analysis from numerical work: `analyzePattern()`, then `factorize()`, with
`compute()` doing both. Re-solving with the same pattern and new values must reuse the analysis; a change that forces a
re-analysis is a performance regression even when results match.

- Sparse solvers report failure through `info()`, not exceptions. `info()` is a member of each concrete solver and of
  `IterativeSolverBase`, not of `SparseSolverBase`. Check it after `compute()`/`factorize()` and again after `solve()`
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

[`test/CMakeLists.txt`](../test/CMakeLists.txt) registers the external backend tests conditionally, so a green local
run says nothing about any of them. The full set is `cholmod_support`, `umfpack_support`, `klu_support`,
`superlu_support`, `pastix_support`, `spqr_support`, `accelerate_support`, `metis_support` (an ordering backend rather
than a solver), and `pardiso_support`. Most are registered only if a `find_package` call finds the library; otherwise
the backend is added to `EIGEN_MISSING_BACKENDS`. Several also require `EIGEN_BUILD_BLAS` or `EIGEN_BUILD_LAPACK`.
`metis_support` and `pastix_support` depend on variables that the PaStiX search sets when the `METIS` component is
requested.

`pardiso_support` is the exception to know about. The tree contains no `find_package(PARDISO)` and no
`EIGEN_MISSING_BACKENDS` entry for it. So it is registered only when `PARDISO_FOUND` is set from outside the project,
and nothing reports its absence, not even the missing-backend summary. Report which sparse backends were unavailable
rather than implying full coverage.
