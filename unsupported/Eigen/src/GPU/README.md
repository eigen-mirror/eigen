# Eigen GPU Module (`unsupported/Eigen/GPU`)

GPU-accelerated dense linear algebra for Eigen users, dispatching to NVIDIA
CUDA Math Libraries (cuBLAS, cuSOLVER). Requires CUDA 11.4+. Header-only (link
against CUDA runtime, cuBLAS, and cuSOLVER).

## Why this module

Eigen is the linear algebra foundation for a large ecosystem of C++ projects
in robotics (ROS, Drake, MoveIt, Pinocchio), computer vision (OpenCV, COLMAP,
Open3D), scientific computing (Ceres, Stan), and beyond. Many of these
projects run on GPU-equipped hardware but cannot use GPUs for Eigen operations
without dropping down to raw CUDA library APIs. Third-party projects like
[EigenCuda](https://github.com/NLESC-JCER/EigenCuda) and
[cholespy](https://github.com/rgl-epfl/cholespy) exist specifically to fill
this gap, and downstream projects like
[Ceres](https://github.com/ceres-solver/ceres-solver/issues/1151) and
[COLMAP](https://github.com/colmap/colmap/issues/4018) have open requests for
GPU-accelerated solvers through Eigen.

The `unsupported/Eigen/GPU` module aims to close this gap: Existing Eigen users should be
able to move performance-critical dense linear algebra to the GPU with minimal
code changes and without learning CUDA library APIs directly.

## Design philosophy

**CPU and GPU coexist.** There is no global compile-time switch that replaces
CPU implementations (unlike `EIGEN_USE_LAPACKE`). Users choose GPU solvers
explicitly -- `gpu::LLT<double>` vs `LLT<MatrixXd>` -- and both coexist in
the same binary. This also lets users keep the factored matrix on device across
multiple solves, something impossible with compile-time replacement.

**Familiar syntax.** GPU operations use the same expression patterns as CPU
Eigen. Here is a side-by-side comparison:

```cpp
// ---- CPU (Eigen) ----               // ---- GPU (unsupported/Eigen/GPU) ----
#include <Eigen/Dense>                  #define EIGEN_USE_GPU
                                        #include <unsupported/Eigen/GPU>

MatrixXd A = ...;                       auto d_A = gpu::DeviceMatrix<double>::fromHost(A);
MatrixXd B = ...;                       auto d_B = gpu::DeviceMatrix<double>::fromHost(B);

MatrixXd C = A * B;                     gpu::DeviceMatrix<double> d_C = d_A * d_B;
MatrixXd X = A.llt().solve(B);          gpu::DeviceMatrix<double> d_X = d_A.llt().solve(d_B);

                                        MatrixXd X = d_X.toHost();
```

The GPU version reads like CPU Eigen with explicit upload/download.
`operator*` dispatches to cuBLAS GEMM, `.llt().solve()` dispatches to
cuSOLVER potrf + potrs. Unsupported expressions are compile errors.

**Standalone module.** `unsupported/Eigen/GPU` does not modify or depend on Eigen's Core
expression template system (`MatrixBase`, `CwiseBinaryOp`, etc.).
`DeviceMatrix` is not an Eigen expression type and does not inherit from
`MatrixBase`. The expression layer is a thin compile-time dispatch where every
supported expression maps to a single NVIDIA library call. There is no
coefficient-level evaluation, lazy fusion, or packet operations.

**Explicit over implicit.** Host-device transfers, stream management, and
library handle lifetimes are visible in the API. There are no hidden
allocations or synchronizations except where documented (e.g., `toHost()` must
synchronize to deliver data to the host).

## Key concepts

### `DeviceMatrix<Scalar>`

A typed RAII wrapper for a dense column-major matrix in GPU device memory.
This is the GPU counterpart of Eigen's `MatrixX<Scalar>`. A vector is simply
a `DeviceMatrix` with one column.

```cpp
// Upload from host
auto d_A = gpu::DeviceMatrix<double>::fromHost(A);

// Allocate uninitialized
gpu::DeviceMatrix<double> d_C(m, n);

// Download to host
MatrixXd C = d_C.toHost();

// Async download (returns a future)
auto transfer = d_C.toHostAsync();
// ... do other work ...
MatrixXd C = transfer.get();
```

`DeviceMatrix` supports expression methods that mirror Eigen's API:
`adjoint()`, `transpose()`, `triangularView<UpLo>()`,
`selfadjointView<UpLo>()`, `llt()`, `lu()`. These return lightweight
expression objects that are evaluated when assigned.

### `gpu::Context`

Every GPU operation needs a CUDA stream and library handles (cuBLAS,
cuSOLVER). `gpu::Context` bundles these together.

For simple usage, you don't need to create one -- a per-thread default context
is created lazily on first use:

```cpp
// These use the thread-local default context automatically
d_C = d_A * d_B;
d_X = d_A.llt().solve(d_B);
```

For concurrent multi-stream execution, create explicit contexts:

```cpp
gpu::Context ctx1, ctx2;
d_C1.device(ctx1) = d_A1 * d_B1;   // runs on stream 1
d_C2.device(ctx2) = d_A2 * d_B2;   // runs on stream 2 (concurrently)
```

## Usage

### Matrix operations (cuBLAS)

```cpp
auto d_A = gpu::DeviceMatrix<double>::fromHost(A);
auto d_B = gpu::DeviceMatrix<double>::fromHost(B);

// GEMM: C = A * B, C = A^H * B, C = A * B^T, ...
gpu::DeviceMatrix<double> d_C = d_A * d_B;
d_C = d_A.adjoint() * d_B;
d_C = d_A * d_B.transpose();

// Scaled and accumulated
d_C += 2.0 * d_A * d_B;             // alpha=2, beta=1
d_C.device(ctx) -= d_A * d_B;       // alpha=-1, beta=1 (requires explicit context)

// Triangular solve (TRSM)
d_X = d_A.triangularView<Lower>().solve(d_B);

// Symmetric/Hermitian multiply (SYMM/HEMM)
d_C = d_A.selfadjointView<Lower>() * d_B;

// Rank-k update (SYRK/HERK)
d_C.selfadjointView<Lower>().rankUpdate(d_A);  // C += A * A^H
```

### Dense solvers (cuSOLVER)

**One-shot expression syntax** -- Convenient, re-factorizes each time:

```cpp
// Cholesky solve (potrf + potrs)
d_X = d_A.llt().solve(d_B);

// LU solve (getrf + getrs)
d_Y = d_A.lu().solve(d_B);
```

**Cached factorization** -- Factor once, solve many times:

```cpp
gpu::LLT<double> llt;
llt.compute(d_A);                    // factorize (async)
if (llt.info() != Success) { ... }   // lazy sync on first info() call
auto d_X1 = llt.solve(d_B1);        // reuses factor (async)
auto d_X2 = llt.solve(d_B2);        // reuses factor (async)
MatrixXd X2 = d_X2.toHost();

// LU with transpose solve
gpu::LU<double> lu;
lu.compute(d_A);
auto d_Y = lu.solve(d_B, gpu::GpuOp::Trans);           // A^T Y = B

// QR solve (overdetermined least squares)
gpu::QR<double> qr(A);                // host matrix input
MatrixXd X = qr.solve(B);           // Q^H * B via ormqr, then trsm on R

// SVD
gpu::SVD<double> svd(A, ComputeThinU | ComputeThinV);
VectorXd S = svd.singularValues();
MatrixXd U = svd.matrixU();
MatrixXd VT = svd.matrixVT();
MatrixXd X = svd.solve(B);          // pseudoinverse solve

// Self-adjoint eigenvalue decomposition
gpu::SelfAdjointEigenSolver<double> es(A);
VectorXd eigenvals = es.eigenvalues();
MatrixXd eigenvecs = es.eigenvectors();
```

The cached API keeps the factored matrix on device, avoiding redundant
host-device transfers and re-factorizations.

### Precision control

GEMM dispatch uses `cublasXgemm` (type-specific Sgemm/Dgemm/Cgemm/Zgemm).
cuBLAS may internally use tensor cores depending on the GPU architecture,
matrix dimensions, and CUDA math mode settings. No Eigen-specific macros
control this; use the standard `CUDA_MATH_MODE` environment variable or
`cublasSetMathMode()` to configure tensor core behavior if needed.

### Stream control and async execution

Operations are asynchronous by default. The compute-solve chain runs without
host synchronization until you need a result on the host:

```text
fromHost(A) --sync-->  compute() --async-->  solve() --async-->  toHost()
   H2D                  potrf                 potrs                D2H
                                                                   sync
```

Mandatory sync points:
- `fromHost()` -- Synchronizes to complete the upload before returning
- `toHost()` / `HostTransfer::get()` -- Must deliver data to host
- `info()` -- Must read the factorization status

**Cross-stream safety** is automatic. `DeviceMatrix` tracks write completion
via CUDA events. When a matrix written on stream A is read on stream B, the
module automatically inserts `cudaStreamWaitEvent`. Same-stream operations
skip the wait (CUDA guarantees in-order execution within a stream).

**Lifetime of cached factorizations.** A `gpu::LLT` / `gpu::LU` object owns
its CUDA stream, library handle, and the cached factor on device. Destroying
the factorization object while a `solve()` it issued is still in flight is
*correct* but not actually async: `cudaStreamDestroy` returns immediately,
but the destructor of the cached factor calls `cudaFree`, which is fully
device-synchronous and stalls until the in-flight `potrs`/`getrs` retires.
For genuine async pipelining keep the factorization object alive until you
have drained its results (e.g. via `toHost()` or by binding consumption to
an explicit `gpu::Context` that outlives both producer and consumer):

```cpp
gpu::LLT<double> llt(d_A);             // factor stays on device
auto d_X = llt.solve(d_B);
auto h_x = d_X.toHostAsync(llt.stream());
h_x.get();                             // sync: factor + result complete
// llt may now be destroyed without stalling the device
```

## Reference

### Supported scalar types

`float`, `double`, `std::complex<float>`, `std::complex<double>`.

### Expression -> library call mapping

| DeviceMatrix expression | Library call | Parameters |
|---|---|---|
| `C = A * B` | `cublasGemmEx` | transA=N, transB=N, alpha=1, beta=0 |
| `C = A.adjoint() * B` | `cublasGemmEx` | transA=C, transB=N |
| `C = A.transpose() * B` | `cublasGemmEx` | transA=T, transB=N |
| `C = A * B.adjoint()` | `cublasGemmEx` | transA=N, transB=C |
| `C = A * B.transpose()` | `cublasGemmEx` | transA=N, transB=T |
| `C = alpha * A * B` | `cublasGemmEx` | alpha from LHS |
| `C = A * (alpha * B)` | `cublasGemmEx` | alpha from RHS |
| `C += A * B` | `cublasGemmEx` | alpha=1, beta=1 |
| `C.device(ctx) -= A * B` | `cublasGemmEx` | alpha=-1, beta=1 |
| `X = A.llt().solve(B)` | `cusolverDnXpotrf` + `Xpotrs` | uplo, n, nrhs |
| `X = A.llt<Upper>().solve(B)` | same | uplo=Upper |
| `X = A.lu().solve(B)` | `cusolverDnXgetrf` + `Xgetrs` | n, nrhs |
| `X = A.triangularView<L>().solve(B)` | `cublasXtrsm` | side=L, uplo, diag=NonUnit |
| `C = A.selfadjointView<L>() * B` | `cublasXsymm` / `cublasXhemm` | side=L, uplo |
| `C.selfadjointView<L>().rankUpdate(A)` | `cublasXsyrk` / `cublasXherk` | uplo, trans=N |

### `DeviceMatrix<Scalar>` API

| Method | Sync? | Description |
|--------|-------|-------------|
| `DeviceMatrix()` | -- | Empty (0x0) |
| `DeviceMatrix(rows, cols)` | -- | Allocate uninitialized |
| `fromHost(matrix, stream)` | yes | Upload from Eigen matrix |
| `fromHostAsync(ptr, rows, cols, stream)` | no | Async upload (caller manages lifetime) |
| `toHost(stream)` | yes | Synchronous download |
| `toHostAsync(stream)` | no | Returns `HostTransfer` future |
| `clone(stream)` | no | Device-to-device deep copy |
| `resize(rows, cols)` | -- | Discard contents, reallocate |
| `data()` | -- | Raw device pointer |
| `rows()`, `cols()` | -- | Dimensions |
| `sizeInBytes()` | -- | Total device allocation size in bytes |
| `empty()` | -- | True if 0x0 |
| `adjoint()` | -- | Adjoint view (GEMM ConjTrans) |
| `transpose()` | -- | Transpose view (GEMM Trans) |
| `llt()` / `llt<UpLo>()` | -- | Cholesky expression builder |
| `lu()` | -- | LU expression builder |
| `triangularView<UpLo>()` | -- | Triangular view (TRSM) |
| `selfadjointView<UpLo>()` | -- | Self-adjoint view (SYMM, rankUpdate) |
| `device(ctx)` | -- | Assignment proxy bound to context |

### `gpu::Context`

Unified GPU execution context owning a CUDA stream and library handles.

```cpp
gpu::Context()                                             // Creates dedicated stream + handles
static gpu::Context& threadLocal()                         // Per-thread default (lazy-created)

cudaStream_t       stream()
cublasHandle_t     cublasHandle()
cusolverDnHandle_t cusolverHandle()
```

Non-copyable, non-movable (owns library handles).

### `gpu::LLT<Scalar, UpLo>` API

GPU dense Cholesky (LL^T) via cuSOLVER. Caches factor on device.

| Method | Sync? | Description |
|--------|-------|-------------|
| `gpu::LLT(A)` | deferred | Construct and factorize from host matrix |
| `compute(host_matrix)` | deferred | Upload and factorize |
| `compute(DeviceMatrix)` | deferred | D2D copy and factorize |
| `compute(DeviceMatrix&&)` | deferred | Move-adopt and factorize (no copy) |
| `solve(host_matrix)` | yes | Solve, return host matrix |
| `solve(DeviceMatrix)` | no | Solve, return `DeviceMatrix` (async) |
| `info()` | lazy | Syncs stream on first call, returns `Success` or `NumericalIssue` |

### `gpu::LU<Scalar>` API

GPU dense partial-pivoting LU via cuSOLVER. Same pattern as `gpu::LLT`, plus a
`gpu::GpuOp` parameter on `solve()` (`NoTrans`, `Trans`, `ConjTrans`).

### `gpu::QR<Scalar>` API

GPU dense QR decomposition via cuSOLVER (`geqrf`). Solve uses `ormqr` (apply
Q^H) + `trsm` (back-substitute on R) -- Q is never formed explicitly.

| Method | Description |
|--------|-------------|
| `gpu::QR()` | Default construct, then call `compute()` |
| `gpu::QR(A)` | Construct and factorize from host matrix |
| `compute(A)` | Upload + factorize |
| `compute(DeviceMatrix)` | D2D copy + factorize |
| `solve(host_matrix)` | Solve, return host matrix (syncs) |
| `solve(DeviceMatrix)` | Solve, return `DeviceMatrix` (async) |
| `info()` | Lazy sync |
| `rows()`, `cols()`, `stream()` | Dimensions and CUDA stream |

### `gpu::SVD<Scalar>` API

GPU dense SVD via cuSOLVER (`gesvd`). Supports thin, full, and values-only
modes via Eigen's `ComputeThinU | ComputeThinV`, `ComputeFullU | ComputeFullV`,
or `0` (values only).

| Method | Description |
|--------|-------------|
| `gpu::SVD()` | Default construct, then call `compute()` |
| `gpu::SVD(A, options)` | Construct and compute (options default: `ComputeThinU \| ComputeThinV`) |
| `compute(A, options)` | Compute from host matrix |
| `compute(DeviceMatrix, options)` | Compute from device matrix |
| `singularValues()` | Download singular values to host |
| `matrixU()` | Download U to host (requires `ComputeThinU` or `ComputeFullU`) |
| `matrixVT()` | Download V^T to host (requires `ComputeThinV` or `ComputeFullV`) |
| `solve(B)` | Pseudoinverse solve (returns host matrix) |
| `solve(B, k)` | Truncated solve (top k singular triplets) |
| `solve(B, lambda)` | Tikhonov regularized solve |
| `rank(threshold)` | Count singular values above threshold |
| `info()` | Lazy sync |
| `rows()`, `cols()`, `stream()` | Dimensions and CUDA stream |

Wide matrices (m < n) are handled by internally transposing via cuBLAS `geam`.

### `gpu::SelfAdjointEigenSolver<Scalar>` API

GPU symmetric/Hermitian eigenvalue decomposition via cuSOLVER (`syevd`).

| Method | Description |
|--------|-------------|
| `gpu::SelfAdjointEigenSolver()` | Default construct, then call `compute()` |
| `gpu::SelfAdjointEigenSolver(A, mode)` | Construct and compute (mode default: `ComputeEigenvectors`) |
| `compute(A, mode)` | Compute from host matrix |
| `compute(DeviceMatrix, mode)` | Compute from device matrix |
| `eigenvalues()` | Download eigenvalues to host (ascending order) |
| `eigenvectors()` | Download eigenvectors to host (columns) |
| `info()` | Lazy sync |
| `rows()`, `cols()`, `stream()` | Dimensions and CUDA stream |

`ComputeMode`: `gpu::SelfAdjointEigenSolver::EigenvaluesOnly` or
`gpu::SelfAdjointEigenSolver::ComputeEigenvectors`.

### `HostTransfer<Scalar>` API

Future for async device-to-host transfer.

| Method | Description |
|--------|-------------|
| `get()` | Block until transfer completes, return host matrix reference. Idempotent. |
| `ready()` | Non-blocking poll |

### Aliasing

Unlike Eigen's `Matrix`, where omitting `.noalias()` triggers a copy to a
temporary, DeviceMatrix dispatches directly to NVIDIA library calls which have
no built-in aliasing protection. All operations are implicitly noalias.
The caller must ensure operands don't alias the destination for GEMM, TRSM,
SYMM/HEMM, and SYRK/HERK. Debug builds assert on these violations before
dispatching to cuBLAS.

## File layout

| File | Depends on | Contents |
|------|-----------|----------|
| `GpuSupport.h` | `<cuda_runtime.h>` | Error macro, `DeviceBuffer`, `PinnedHostBuffer`, `cuda_data_type<>` |
| `DeviceMatrix.h` | `GpuSupport.h` | `DeviceMatrix<>`, `HostTransfer<>` |
| `DeviceExpr.h` | `DeviceMatrix.h` | GEMM expression wrappers |
| `DeviceBlasExpr.h` | `DeviceMatrix.h` | TRSM, SYMM, SYRK expression wrappers |
| `DeviceSolverExpr.h` | `DeviceMatrix.h` | Solver expression wrappers (LLT, LU) |
| `DeviceDispatch.h` | all above | All dispatch functions + `Assignment` |
| `GpuContext.h` | `CuBlasSupport.h`, `CuSolverSupport.h` | `gpu::Context` |
| `CuBlasSupport.h` | `GpuSupport.h`, `<cublas_v2.h>` | cuBLAS error macro, op/compute type maps |
| `CuSolverSupport.h` | `GpuSupport.h`, `<cusolverDn.h>` | cuSOLVER params, fill-mode mapping |
| `GpuLLT.h` | `CuSolverSupport.h` | Cached dense Cholesky factorization |
| `GpuLU.h` | `CuSolverSupport.h` | Cached dense LU factorization |
| `GpuQR.h` | `CuSolverSupport.h`, `CuBlasSupport.h` | Dense QR decomposition |
| `GpuSVD.h` | `CuSolverSupport.h`, `CuBlasSupport.h` | Dense SVD decomposition |
| `GpuEigenSolver.h` | `CuSolverSupport.h` | Self-adjoint eigenvalue decomposition |

## Building and testing

```bash
cmake -G Ninja -B build -S . \
  -DEIGEN_TEST_CUDA=ON \
  -DEIGEN_CUDA_COMPUTE_ARCH="70"

cmake --build build --target cublas cusolver_llt cusolver_lu \
  cusolver_qr cusolver_svd cusolver_eigen device_matrix
ctest --test-dir build -L gpu --output-on-failure
```

`EIGEN_TEST_CUBLAS` and `EIGEN_TEST_CUSOLVER` default to ON when CUDA is enabled
(cuBLAS and cuSOLVER are part of the CUDA toolkit).
