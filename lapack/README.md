# Eigen's LAPACK library

This directory builds a library that implements a subset of LAPACK's Fortran 77
interface in C++, on top of Eigen's dense decompositions. A program that calls
only the routines listed below can link it in place of reference LAPACK.
Eigen's C++ API does not use it.

## Interface

- Symbols follow the Fortran 77 calling convention: a lowercase name with a
  trailing underscore (`dgetrf_`), every argument passed by pointer,
  column-major matrices with a leading dimension, and complex numbers stored as
  (real, imaginary) pairs. Dimensions, pivot indices and `INFO` are 32-bit
  `int`, or 64-bit in the ILP64 build (`eigen_lapack_ilp64`), which exports the
  same names and links Eigen's ILP64 BLAS library.
- In a routine with an `INFO` argument, an invalid `i`-th argument makes the
  routine call `XERBLA` and return `INFO = -i`, as in reference LAPACK. The
  auxiliary routines have no `INFO` argument and do not check their arguments.
- Eigen's BLAS library provides an `XERBLA` that prints the error and returns,
  where reference `XERBLA` stops the program. With GNU-compatible compilers
  other than MinGW and Cygwin it is a weak symbol, so a program can define its
  own.
- The driver routines allocate their own workspace. They use `WORK` only to
  answer a workspace query (`LWORK = -1`). `xSYEV` still requires
  `LWORK >= max(1, 3*N-1)`, and `xHEEV` `LWORK >= max(1, 2*N-1)`.

## Building

The library is enabled with `EIGEN_BUILD_LAPACK` and requires
`EIGEN_BUILD_BLAS`. Both default to `ON` when Eigen is the top-level CMake
project. The `lapack` target builds `eigen_lapack_static`, which links
`eigen_blas_static`. Where the platform supports shared libraries
(`EIGEN_BUILD_SHARED_LIBS`), it also builds `eigen_lapack`, which links
`eigen_blas`. With `EIGEN_BUILD_LAPACK_ILP64`, which defaults to
`EIGEN_BUILD_BLAS_ILP64`, it builds the same pair with 64-bit integers,
`eigen_lapack_ilp64_static` and `eigen_lapack_ilp64`, which link Eigen's ILP64
BLAS libraries. `EIGEN_INSTALL` installs them all.

## Routines

The tables follow the classification in Chapter 2 of the
[LAPACK Users' Guide](https://www.netlib.org/lapack/lug/), third edition:

- A driver routine solves a complete problem.
- A computational routine performs one step of a driver's work.
- An auxiliary routine is a building block of the other two.

An `x` in a routine name stands for a precision prefix. The Precisions column
lists the prefixes that exist:

- S: real single.
- D: real double.
- C: complex single.
- Z: complex double.

### Driver routines (Users' Guide §2.3)

| Problem | Routine | Precisions | Computes | Implemented with |
|---|---|---|---|---|
| Symmetric eigenproblem (SEP) | `xSYEV` | S, D | All eigenvalues and, optionally, the eigenvectors of a symmetric matrix | `SelfAdjointEigenSolver` |
| | `xHEEV` | C, Z | The same for a Hermitian matrix; the imaginary part of its diagonal is ignored | `SelfAdjointEigenSolver` |
| Singular value decomposition (SVD) | `xGESVD` | S, D, C, Z | `A = U Σ V^H`: the singular values and, optionally, the left and right singular vectors | `JacobiSVD` (two-sided Jacobi, not bidiagonal QR iteration) |
| | `xGESDD` | S, D, C, Z | The same, by divide and conquer | `BDCSVD` |

The library has no driver for any other problem class in §2.3:

- Linear equations (`xGESV`, `xPOSV`, ...).
- Linear least squares, and generalized linear least squares.
- Nonsymmetric eigenproblems.
- Generalized eigenvalue and singular value problems.

### Computational routines (Users' Guide §2.4)

| Problem | Routine | Precisions | Computes | Implemented with |
|---|---|---|---|---|
| Linear equations (§2.4.1), general matrix | `xGETRF` | S, D, C, Z | LU factorization with partial pivoting, `A = P L U` | `PartialPivLU`'s blocked kernel |
| | `xGETRS` | S, D, C, Z | Solves `A X = B`, `A^T X = B` or `A^H X = B` with the factors from `xGETRF` | Triangular solves |
| | `xGETRI` | S, D, C, Z | `inv(A) = inv(U) inv(L) P^T` from the factors of `xGETRF`, blocked over `WORK` | `xTRTRI` and triangular solves |
| Linear equations (§2.4.1), symmetric or Hermitian positive definite matrix | `xPOTRF` | S, D, C, Z | Cholesky factorization, `A = U^H U` or `A = L L^H` | `LLT`'s blocked kernel |
| | `xPOTRS` | S, D, C, Z | Solves `A X = B` with the factor from `xPOTRF` | Triangular solves |
| | `xPOTRI` | S, D, C, Z | `inv(A)` from the factor of `xPOTRF` | `xTRTRI`, then `xLAUUM` |
| Linear equations (§2.4.1), symmetric or Hermitian positive semidefinite matrix | `xPSTRF` | S, D, C, Z | Cholesky factorization with complete pivoting, `P^T A P = U^H U` or `L L^H`, stopping at the numerical rank | `LDLT`'s panel kernel, pivoting on the largest diagonal entry |
| Linear equations (§2.4.1), symmetric indefinite matrix | `xSYTRF` | S, D | Bunch-Kaufman factorization `A = U D U^T` or `L D L^T`, `D` with 1x1 and 2x2 blocks | `BunchKaufman`'s blocked kernel, with LAPACK's pivot search |
| | `xSYTRS` | S, D | Solves `A X = B` with the factorization from `xSYTRF` | One block of the factorization at a time, without workspace, as reference LAPACK does |
| Linear equations (§2.4.1), Hermitian indefinite matrix | `xHETRF` | C, Z | Bunch-Kaufman factorization `A = U D U^H` or `L D L^H` | `BunchKaufman`'s blocked kernel, with LAPACK's pivot search |
| | `xHETRS` | C, Z | Solves `A X = B` with the factorization from `xHETRF` | One block of the factorization at a time, without workspace, as reference LAPACK does |
| Linear equations (§2.4.1), triangular matrix | `xTRTRS` | S, D, C, Z | Solves `op(A) X = B`, checking for a zero diagonal entry | Triangular solves |
| | `xTRTRI` | S, D, C, Z | `inv(A)` in place | `TriangularView::inverseInPlace()` |
| Orthogonal factorizations (§2.4.2) | `xGEQRF` | S, D, C, Z | QR factorization `A = Q R`, `Q` as Householder reflectors | `HouseholderQR`'s blocked kernel |
| | `xORGQR`, `xUNGQR` | S, D (ORGQR); C, Z (UNGQR) | Forms the leading columns of `Q` | Block Householder kernels |
| | `xORMQR`, `xUNMQR` | S, D (ORMQR); C, Z (UNMQR) | `Q C`, `Q^H C`, `C Q` or `C Q^H` | Block Householder kernels |

Of §2.4.2 the library has only the QR factorization. It has no routines from
§2.4.3 through §2.4.9, which cover generalized orthogonal factorizations, the
symmetric, nonsymmetric and generalized eigenproblems, the SVD, and the
generalized SVD.

### Auxiliary routines (Users' Guide, Index of Auxiliary Routines)

The Users' Guide indexes auxiliary routines alphabetically. They are grouped
here by purpose.

| Purpose | Routine | Precisions | Computes |
|---|---|---|---|
| Householder reflectors | `xLARFG` | S, D, C, Z | An elementary reflector `H` with `H^H [α; x] = [β; 0]`, `β` real |
| | `xLARF` | S, D, C, Z | `H C` or `C H` for `H = I - τ v v^H` |
| | `xLARFT` | S, D, C, Z | The triangular factor `T` of a block reflector `H = I - V T V^H` |
| | `xLARFB` | S, D, C, Z | `H C`, `H^H C`, `C H` or `C H^H` for a block reflector `H` |
| Triangular products | `xLAUUM` | S, D, C, Z | `U U^H` or `L^H L` in place, keeping a complex diagonal real |
| Matrix scanning | `ILAxLR`, `ILAxLC` | S, D, C, Z | Index of the last nonzero row or column |
| Scalar arithmetic | `xLAPY2`, `xLAPY3` | S, D | `sqrt(x^2 + y^2)` and `sqrt(x^2 + y^2 + z^2)` without unnecessary overflow |
| | `xLADIV` | S, D, C, Z | Complex division without unnecessary overflow. S and D pass real and imaginary parts as separate arguments. |
| Vector operations | `xLACGV` | C, Z | Conjugates a vector |
| Machine parameters | `xLAMCH` | S, D | Floating-point parameters: machine epsilon, safe minimum, ... |
| | `xLAMC3` | S, D | `a + b`, forced through memory |
| Version | `ILAVER` | – | The reference LAPACK release that the routines follow, 3.12.1 |
| Timing | `SECOND`, `DSECND` | – | Process CPU time in seconds |

Except for `xLAUUM`, which shares its kernel with `LLT::inverse()`, these
routines are written directly to the reference LAPACK specification rather
than built on Eigen's decompositions. The Householder routines support every
combination of `SIDE`, `TRANS`, `DIRECT` and `STOREV`.

## Tests

`lapack_routines` and `lapack_wrappers`, built with Eigen's other tests
(`EIGEN_BUILD_TESTING`), check Eigen's routines directly from C++ and need no
Fortran compiler. `lapack_wrappers` adds what reference LAPACK's tests leave
out, such as NaN and infinite input, arguments left untouched when `INFO > 0`
and the pivots reference LAPACK chooses; `lapack_wrappers_ilp64` runs it
against the ILP64 library.

The optional numerical tests (`EIGEN_ENABLE_LAPACK_TESTS`) require a Fortran
compiler. CMake downloads the unmodified reference LAPACK release that
`CMakeLists.txt` pins by version and SHA-256 hash.

The tests run reference LAPACK's own drivers on every input that reference
LAPACK's `TESTING/CMakeLists.txt` registers. They also run
`testing/[sdcz]test_large.in`, which repeat the linear-equation paths that reach
Eigen's routines at sizes above Eigen's blocking thresholds; the reference inputs
stop at N = 50. The drivers link Eigen's BLAS and LAPACK libraries. Each routine
that `eigen_lapack.def` does not export comes from the reference sources, built
into the test-only `eigen_lapack_reference` library, which is never installed.
`LAPACK_Test_Symbols` checks the split: Eigen defines every routine the `.def`
file lists, defines no Fortran-callable routine it omits, and shares none with
the reference library. On ELF platforms it also checks that the shared library
exports every listed routine. `buildtests_lapack` builds the drivers and the
`lapack_*` C++ tests, and `buildtests_blas` the tests under `blas/testing`.

The tests and their summary run through CMake/CTest without Python.
`LAPACK_Test_Summary` waits for the drivers and combines their output files into
`TESTING/testing_results.txt` in the LAPACK build directory. For each output it
counts numerical errors and other errors (illegal-argument, error-exit and INFO
diagnostics). It fails when an output reports more of either than
`testing_allowed_failures.txt` allows, which is none for an output the file does
not list. It also fails when an expected output is missing, when no numerical
tests were analyzed, and when an entry of the file names no output.
`TESTING/testing_failures.txt` records the observed counts in the file's format.
Selecting `LAPACK_Test_Summary` alone also runs its drivers through a CTest
fixture. If a driver fails to execute successfully, CTest skips the summary.

Balancing tests contribute their example counts and fail on nonzero INFO/ILO
error counts or non-finite maximum errors. Their finite maximum errors remain
in the combined report; these drivers do not specify a pass/fail threshold.

The allowed counts are those of the `test:linux:lapack` CI job. Other compilers,
flags and architectures round differently and fail a different set, mostly the
eigenvalue tests that compare results computed with and without eigenvectors
bit for bit ([#3220](https://gitlab.com/libeigen/eigen/-/work_items/3220)). To
check such a configuration, set `EIGEN_LAPACK_TEST_ALLOWED_FAILURES` to a file of
its own, or to an empty string to allow no failures. A change that alters the
job's counts on purpose updates the file's entries from the job's
`testing_failures.txt` artifact.
