# Eigen GPU Module (`Eigen/GPU`)

GPU-accelerated linear algebra for Eigen users, dispatching to NVIDIA CUDA
Math Libraries (cuBLAS, cuSOLVER, cuFFT, cuSPARSE, cuDSS). Requires CUDA
11.4+; cuDSS features require CUDA 12.0+ and a separate cuDSS install.
Header-only.

## Why this module

Eigen is the linear algebra foundation for a large ecosystem of C++ projects
in robotics (ROS, Drake, MoveIt, Pinocchio), computer vision (OpenCV, COLMAP,
Open3D), scientific computing (Ceres, Stan), and beyond. Many of these
projects run on GPU-equipped hardware but cannot use GPUs for Eigen operations
without dropping down to raw CUDA library APIs.

GPU sparse solvers are a particularly acute gap. Sparse factorization is the
bottleneck in SLAM, bundle adjustment, FEM, and nonlinear optimization --
exactly the workloads where GPU acceleration matters most. Downstream projects
like [Ceres](https://github.com/ceres-solver/ceres-solver/issues/1151) and
[COLMAP](https://github.com/colmap/colmap/issues/4018) have open requests for
GPU-accelerated sparse solvers, and third-party projects like
[cholespy](https://github.com/rgl-epfl/cholespy) exist specifically because
Eigen lacks them. The `Eigen/GPU` module provides GPU sparse Cholesky, LDL^T,
and LU factorization via cuDSS, alongside dense solvers (cuSOLVER), matrix
products (cuBLAS), FFT (cuFFT), and sparse matrix-vector products (cuSPARSE).

Existing Eigen users should be able to move performance-critical dense or
sparse linear algebra to the GPU with minimal code changes and without
learning CUDA library APIs directly.

## Design philosophy

**CPU and GPU coexist.** There is no global compile-time switch that replaces
CPU implementations (unlike `EIGEN_USE_LAPACKE`). Users choose GPU solvers
explicitly -- `gpu::LLT<double>` vs `LLT<MatrixXd>`, `gpu::SparseLLT<double>` vs
`SimplicialLLT<SparseMatrix<double>>` -- and both coexist in the same binary.
This also lets users keep the factored matrix on device across multiple solves,
something impossible with compile-time replacement.

**Familiar syntax.** GPU operations use the same expression patterns as CPU
Eigen. Here is a side-by-side comparison:

```cpp
// ---- CPU (Eigen) ----               // ---- GPU (Eigen/GPU) ----
#include <Eigen/Dense>                  #define EIGEN_USE_GPU
                                        #include <Eigen/GPU>

// Dense
MatrixXd A = ...;                       auto d_A = gpu::DeviceMatrix<double>::fromHost(A);
MatrixXd B = ...;                       auto d_B = gpu::DeviceMatrix<double>::fromHost(B);

MatrixXd C = A * B;                     gpu::DeviceMatrix<double> d_C = d_A * d_B;
MatrixXd X = A.llt().solve(B);          gpu::DeviceMatrix<double> d_X = d_A.llt().solve(d_B);

                                        MatrixXd X = d_X.toHost();

// Sparse (using SpMat = SparseMatrix<double>)
SimplicialLLT<SpMat> llt(A);            gpu::SparseLLT<double> llt(A);
VectorXd x = llt.solve(b);              VectorXd x = llt.solve(b);
```

The GPU version reads like CPU Eigen with explicit upload/download for dense
operations, and an almost identical API for sparse solvers. Unsupported
expressions are compile errors.

**Standalone module.** `Eigen/GPU` does not modify or depend on Eigen's Core
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
auto d_Y = lu.solve(d_B, gpu::LU<double>::Transpose);  // A^T Y = B

// QR solve (overdetermined least squares)
gpu::QR<double> qr;
qr.compute(d_A);                     // factorize on device (async)
auto d_X = qr.solve(d_B);           // Q^H * B via ormqr, then trsm on R
MatrixXd X = d_X.toHost();

// SVD (results downloaded on access)
gpu::SVD<double> svd;
svd.compute(d_A, ComputeThinU | ComputeThinV);
VectorXd S = svd.singularValues();   // downloads to host
MatrixXd U = svd.matrixU();          // downloads to host
MatrixXd VT = svd.matrixVT();        // V^T (matches cuSOLVER)

// Self-adjoint eigenvalue decomposition (results downloaded on access)
gpu::SelfAdjointEigenSolver<double> es;
es.compute(d_A);
VectorXd eigenvals = es.eigenvalues();    // downloads to host
MatrixXd eigenvecs = es.eigenvectors();   // downloads to host
```

The cached API keeps the factored matrix on device, avoiding redundant
host-device transfers and re-factorizations. All solvers also accept host
matrices directly as a convenience (e.g., `gpu::LLT<double> llt(A)` or
`qr.solve(B)`), which handles upload/download internally.

### Sparse direct solvers (cuDSS)

Requires cuDSS (separate install, CUDA 12.0+). Define `EIGEN_CUDSS` before
including `Eigen/GPU` and link with `-lcudss`.

```cpp
SparseMatrix<double> A = ...;  // symmetric positive definite
VectorXd b = ...;

// Sparse Cholesky -- one-liner
gpu::SparseLLT<double> llt(A);
VectorXd x = llt.solve(b);

// Three-phase workflow for repeated solves with the same sparsity pattern
gpu::SparseLLT<double> llt;
llt.analyzePattern(A);               // symbolic analysis (once)
llt.factorize(A);                    // numeric factorization
VectorXd x = llt.solve(b);
llt.factorize(A_new_values);         // refactorize (reuses symbolic analysis)
VectorXd x2 = llt.solve(b);

// Sparse LDL^T (symmetric indefinite)
gpu::SparseLDLT<double> ldlt(A);
VectorXd x = ldlt.solve(b);

// Sparse LU (general non-symmetric)
gpu::SparseLU<double> lu(A);
VectorXd x = lu.solve(b);
```

### FFT (cuFFT)

```cpp
gpu::FFT<float> fft;

// 1D complex-to-complex
VectorXcf X = fft.fwd(x);           // forward
VectorXcf y = fft.inv(X);           // inverse (scaled by 1/n)

// 1D real-to-complex / complex-to-real
VectorXcf R = fft.fwd(r);           // returns n/2+1 complex (half-spectrum)
VectorXf  s = fft.invReal(R, n);    // C2R inverse, caller specifies n

// 2D complex-to-complex
MatrixXcf B = fft.fwd2d(A);         // 2D forward
MatrixXcf C = fft.inv2d(B);         // 2D inverse (scaled by 1/(rows*cols))

// Plans are cached and reused across calls with the same size/type.
```

### Sparse matrix-vector multiply (cuSPARSE)

```cpp
SparseMatrix<double> A = ...;
VectorXd x = ...;

gpu::SparseContext<double> ctx;
VectorXd y = ctx.multiply(A, x);            // y = A * x
VectorXd z = ctx.multiplyT(A, x);           // z = A^T * x
ctx.multiply(A, x, y, 2.0, 1.0);            // y = 2*A*x + y

// Multiple RHS (SpMM)
MatrixXd Y = ctx.multiplyMat(A, X);         // Y = A * X
```

### Precision control

GEMM dispatch uses `cublasXgemm` (type-specific Sgemm/Dgemm/Cgemm/Zgemm).
cuBLAS may internally use tensor cores depending on the GPU architecture,
matrix dimensions, and CUDA math mode settings. No Eigen-specific macros
control this; use the standard `CUDA_MATH_MODE` environment variable or
`cublasSetMathMode()` to configure tensor core behavior if needed.

### Stream control and async execution

Operations are asynchronous by default. The compute-solve chain runs without
host synchronization until you need a result on the host:

```
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

## Reference

### Supported scalar types

`float`, `double`, `std::complex<float>`, `std::complex<double>` (unless
noted otherwise).

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

### `DeviceMatrix<Scalar>`

Typed RAII wrapper for a dense column-major matrix in GPU device memory.
Always dense (leading dimension = rows). A vector is a `DeviceMatrix` with
one column.

```cpp
// Construction
DeviceMatrix<Scalar>()                                   // Empty (0x0)
DeviceMatrix<Scalar>(rows, cols)                         // Allocate uninitialized

// Upload / download
static DeviceMatrix fromHost(matrix, stream=nullptr)           // -> DeviceMatrix (syncs)
static DeviceMatrix fromHostAsync(ptr, rows, cols, outerStride, s)  // -> DeviceMatrix (no sync, caller manages ptr lifetime)
PlainMatrix        toHost(stream=nullptr)                      // -> host Matrix (syncs)
HostTransfer       toHostAsync(stream=nullptr)                 // -> HostTransfer future (no sync)
DeviceMatrix       clone(stream=nullptr)                       // -> DeviceMatrix (D2D copy, async)

// Dimensions and access
Index   rows()
Index   cols()
size_t  sizeInBytes()
bool    empty()
Scalar* data()                                           // Raw device pointer
void    resize(Index rows, Index cols)                   // Discard contents, reallocate

// Expression builders (return lightweight views, evaluated on assignment)
AdjointView       adjoint()                              // GEMM with ConjTrans
TransposeView     transpose()                            // GEMM with Trans
LltExpr            llt() / llt<UpLo>()                   // -> .solve(d_B) -> DeviceMatrix
LuExpr             lu()                                  // -> .solve(d_B) -> DeviceMatrix
TriangularView     triangularView<UpLo>()                // -> .solve(d_B) -> DeviceMatrix (TRSM)
SelfAdjointView    selfadjointView<UpLo>()               // -> * d_B (SYMM), .rankUpdate(d_A) (SYRK)
Assignment   device(gpu::Context& ctx)                // Bind assignment to explicit stream
```

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

### `gpu::LLT<Scalar, UpLo>` -- Dense Cholesky (cuSOLVER)

Caches the Cholesky factor on device for repeated solves.

```cpp
gpu::LLT()                                                // Default construct, then call compute()
gpu::LLT(const EigenBase<D>& A)                           // Convenience: upload + factorize

gpu::LLT&            compute(const EigenBase<D>& A)       // Upload + factorize
gpu::LLT&            compute(const DeviceMatrix& d_A)     // D2D copy + factorize
gpu::LLT&            compute(DeviceMatrix&& d_A)          // Adopt + factorize (no copy)

PlainMatrix        solve(const MatrixBase<D>& B)         // -> host Matrix (syncs)
DeviceMatrix       solve(const DeviceMatrix& d_B)        // -> DeviceMatrix (async, stays on device)

ComputationInfo    info()                                // Lazy sync on first call: Success or NumericalIssue
Index              rows() / cols()
cudaStream_t       stream()
```

### `gpu::LU<Scalar>` -- Dense LU (cuSOLVER)

Same pattern as `gpu::LLT`. Adds `TransposeMode` parameter on `solve()`.

```cpp
PlainMatrix        solve(const MatrixBase<D>& B, TransposeMode m = NoTranspose)  // -> host Matrix
DeviceMatrix       solve(const DeviceMatrix& d_B, TransposeMode m = NoTranspose) // -> DeviceMatrix
```

`TransposeMode`: `NoTranspose`, `Transpose`, `ConjugateTranspose`.

### `gpu::QR<Scalar>` -- Dense QR (cuSOLVER)

QR factorization via `cusolverDnXgeqrf`. Solve uses ORMQR (apply Q^H) + TRSM
(back-substitute on R) -- Q is never formed explicitly.

```cpp
gpu::QR()                                                  // Default construct
gpu::QR(const EigenBase<D>& A)                             // Convenience: upload + factorize

gpu::QR&             compute(const EigenBase<D>& A)        // Upload + factorize
gpu::QR&             compute(const DeviceMatrix& d_A)      // D2D copy + factorize

PlainMatrix        solve(const MatrixBase<D>& B)         // -> host Matrix (syncs)
DeviceMatrix       solve(const DeviceMatrix& d_B)        // -> DeviceMatrix (async)

ComputationInfo    info()                                // Lazy sync
Index              rows() / cols()
cudaStream_t       stream()
```

### `gpu::SVD<Scalar>` -- Dense SVD (cuSOLVER)

SVD via `cusolverDnXgesvd`. Supports `ComputeThinU | ComputeThinV`,
`ComputeFullU | ComputeFullV`, or `0` (values only). Wide matrices (m < n)
handled by internal transpose.

```cpp
gpu::SVD()                                                 // Default construct, then call compute()
gpu::SVD(const EigenBase<D>& A, unsigned options = ComputeThinU | ComputeThinV)  // Convenience

gpu::SVD&            compute(const EigenBase<D>& A, unsigned options = ComputeThinU | ComputeThinV)
gpu::SVD&            compute(const DeviceMatrix& d_A, unsigned options = ComputeThinU | ComputeThinV)

RealVector         singularValues()                      // -> host vector (syncs, downloads)
PlainMatrix        matrixU()                             // -> host Matrix (syncs, downloads)
PlainMatrix        matrixVT()                            // -> host Matrix (syncs, downloads V^T)

PlainMatrix        solve(const MatrixBase<D>& B)         // -> host Matrix (pseudoinverse)
PlainMatrix        solve(const MatrixBase<D>& B, Index k)       // Truncated (top k triplets)
PlainMatrix        solve(const MatrixBase<D>& B, RealScalar l)  // Tikhonov regularized

Index              rank(RealScalar threshold = -1)
ComputationInfo    info()                                // Lazy sync
Index              rows() / cols()
cudaStream_t       stream()
```

**Note:** `singularValues()`, `matrixU()`, and `matrixVT()` download to host
on each call. Device-side accessors returning `DeviceMatrix` are planned but
not yet implemented.

### `gpu::SelfAdjointEigenSolver<Scalar>` -- Eigendecomposition (cuSOLVER)

Symmetric/Hermitian eigenvalue decomposition via `cusolverDnXsyevd`.
`ComputeMode` enum: `EigenvaluesOnly`, `ComputeEigenvectors`.

```cpp
gpu::SelfAdjointEigenSolver()                              // Default construct, then call compute()
gpu::SelfAdjointEigenSolver(const EigenBase<D>& A, ComputeMode mode = ComputeEigenvectors)  // Convenience

gpu::SelfAdjointEigenSolver& compute(const EigenBase<D>& A, ComputeMode mode = ComputeEigenvectors)
gpu::SelfAdjointEigenSolver& compute(const DeviceMatrix& d_A, ComputeMode mode = ComputeEigenvectors)

RealVector         eigenvalues()                         // -> host vector (syncs, downloads, ascending order)
PlainMatrix        eigenvectors()                        // -> host Matrix (syncs, downloads, columns)

ComputationInfo    info()                                // Lazy sync
Index              rows() / cols()
cudaStream_t       stream()
```

**Note:** `eigenvalues()` and `eigenvectors()` download to host on each call.
Device-side accessors returning `DeviceMatrix` are planned but not yet
implemented.

### `HostTransfer<Scalar>`

Future for async device-to-host transfer. Returned by
`DeviceMatrix::toHostAsync()`.

```cpp
PlainMatrix&       get()                                 // Block until complete, return host Matrix ref. Idempotent.
bool               ready()                               // Non-blocking poll
```

### `gpu::SparseLLT<Scalar, UpLo>` -- Sparse Cholesky (cuDSS)

Requires cuDSS (CUDA 12.0+, `#define EIGEN_CUDSS`). Three-phase workflow
with symbolic reuse. Accepts `SparseMatrix<Scalar, ColMajor, int>` (CSC).

```cpp
gpu::SparseLLT()                                           // Default construct
gpu::SparseLLT(const SparseMatrixBase<D>& A)               // Analyze + factorize

gpu::SparseLLT&      analyzePattern(const SparseMatrixBase<D>& A)  // Symbolic analysis (reusable)
gpu::SparseLLT&      factorize(const SparseMatrixBase<D>& A)       // Numeric factorization
gpu::SparseLLT&      compute(const SparseMatrixBase<D>& A)         // analyzePattern + factorize
void               setOrdering(GpuSparseOrdering ord)             // AMD (default), METIS, or RCM

DenseMatrix        solve(const MatrixBase<D>& B)         // -> host Matrix (syncs)

ComputationInfo    info()                                // Lazy sync
Index              rows() / cols()
cudaStream_t       stream()
```

### `gpu::SparseLDLT<Scalar, UpLo>` -- Sparse LDL^T (cuDSS)

Symmetric indefinite. Same API as `gpu::SparseLLT`.

### `gpu::SparseLU<Scalar>` -- Sparse LU (cuDSS)

General non-symmetric. Same API as `gpu::SparseLLT` (without `UpLo`).

### `gpu::FFT<Scalar>` -- FFT (cuFFT)

Plans cached by (size, type) and reused. Inverse transforms scaled so
`inv(fwd(x)) == x`. Supported scalars: `float`, `double`.

```cpp
// 1D transforms (host vectors in and out)
ComplexVector      fwd(const MatrixBase<D>& x)           // C2C forward (complex input)
ComplexVector      fwd(const MatrixBase<D>& x)           // R2C forward (real input, returns n/2+1)
ComplexVector      inv(const MatrixBase<D>& X)           // C2C inverse, scaled by 1/n
RealVector         invReal(const MatrixBase<D>& X, Index n)  // C2R inverse, scaled by 1/n

// 2D transforms (host matrices in and out)
ComplexMatrix      fwd2d(const MatrixBase<D>& A)         // 2D C2C forward
ComplexMatrix      inv2d(const MatrixBase<D>& A)         // 2D C2C inverse, scaled by 1/(rows*cols)

cudaStream_t       stream()
```

All FFT methods accept host data and return host data. Upload/download is
handled internally. The C2C and R2C overloads of `fwd()` are distinguished by
the input scalar type (complex vs real).

### `gpu::SparseContext<Scalar>` -- SpMV/SpMM (cuSPARSE)

Accepts `SparseMatrix<Scalar, ColMajor>`. All methods accept host data and
return host data.

```cpp
gpu::SparseContext()                                       // Creates own stream + cuSPARSE handle

DenseVector        multiply(A, x)                                       // y = A * x
void               multiply(A, x, y, alpha=1, beta=0,                   // y = alpha*op(A)*x + beta*y
                     op=CUSPARSE_OPERATION_NON_TRANSPOSE)
DenseVector        multiplyT(A, x)                                      // y = A^T * x
DenseMatrix        multiplyMat(A, X)                                    // Y = A * X (SpMM)

cudaStream_t       stream()
```

### Aliasing

Unlike Eigen's `Matrix`, where omitting `.noalias()` triggers a copy to a
temporary, DeviceMatrix dispatches directly to NVIDIA library calls which have
no built-in aliasing protection. All operations are implicitly noalias.
The caller must ensure operands don't alias the destination for GEMM and TRSM
(debug asserts catch violations).

## File layout

| File | Depends on | Contents |
|------|-----------|----------|
| `GpuSupport.h` | `<cuda_runtime.h>` | Error macro, `DeviceBuffer`, `cuda_data_type<>` |
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
| `CuFftSupport.h` | `GpuSupport.h`, `<cufft.h>` | cuFFT error macro, type-dispatch wrappers |
| `GpuFFT.h` | `CuFftSupport.h`, `CuBlasSupport.h` | 1D/2D FFT with plan caching |
| `CuSparseSupport.h` | `GpuSupport.h`, `<cusparse.h>` | cuSPARSE error macro |
| `GpuSparseContext.h` | `CuSparseSupport.h` | SpMV/SpMM via cuSPARSE |
| `CuDssSupport.h` | `GpuSupport.h`, `<cudss.h>` | cuDSS error macro, type traits (optional) |
| `GpuSparseSolverBase.h` | `CuDssSupport.h` | CRTP base for sparse solvers (optional) |
| `GpuSparseLLT.h` | `GpuSparseSolverBase.h` | Sparse Cholesky via cuDSS (optional) |
| `GpuSparseLDLT.h` | `GpuSparseSolverBase.h` | Sparse LDL^T via cuDSS (optional) |
| `GpuSparseLU.h` | `GpuSparseSolverBase.h` | Sparse LU via cuDSS (optional) |

## Building and testing

```bash
cmake -G Ninja -B build -S . \
  -DEIGEN_TEST_CUDA=ON \
  -DEIGEN_CUDA_COMPUTE_ARCH="70" \
  -DEIGEN_TEST_CUBLAS=ON \
  -DEIGEN_TEST_CUSOLVER=ON

cmake --build build --target gpu_cublas gpu_cusolver_llt gpu_cusolver_lu \
  gpu_cusolver_qr gpu_cusolver_svd gpu_cusolver_eigen \
  gpu_device_matrix gpu_cufft gpu_cusparse_spmv
ctest --test-dir build -R "gpu_" --output-on-failure

# Sparse solvers (cuDSS -- separate install required)
cmake -G Ninja -B build -S . \
  -DEIGEN_TEST_CUDA=ON \
  -DEIGEN_CUDA_COMPUTE_ARCH="70" \
  -DEIGEN_TEST_CUDSS=ON

cmake --build build --target gpu_cudss_llt gpu_cudss_ldlt gpu_cudss_lu
ctest --test-dir build -R gpu_cudss --output-on-failure
```

## Future work

- **Device-side accessors for decomposition results.** `gpu::SVD`,
  `gpu::SelfAdjointEigenSolver`, and `gpu::QR` currently download decomposition
  results to host on access (e.g., `svd.matrixU()` returns a host `MatrixXd`).
  Device-side accessors returning `DeviceMatrix` views of the internal buffers
  would allow chaining GPU operations (e.g., `svd.deviceU() * d_A`) without
  round-tripping through host memory.
- **Device-resident sparse matrix-vector products.** `gpu::SparseContext`
  currently operates on host vectors and matrices, uploading and downloading
  on each call. The key missing piece is a `DeviceSparseView` that holds a
  sparse matrix on device and supports operator syntax (`d_y = d_A * d_x`)
  with `DeviceMatrix` operands -- keeping the entire SpMV/SpMM pipeline on
  device. This is essential for iterative solvers and any workflow that chains
  sparse and dense operations without returning to the host.
