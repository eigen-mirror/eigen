# Eigen StructuredMatrices Module (`contrib/Eigen/StructuredMatrices`)

Fast operators and direct solvers for matrices determined by far fewer parameters than they have entries: circulant,
Toeplitz, Hankel, block-circulant, Kronecker products and sums, diagonal-plus-low-rank, Vandermonde and Cauchy
matrices, and the diagonal-plus-rank-one symmetric eigenproblem. Each operator stores only its generators or factors,
applies itself without forming the matrix, and solves, factors or inverts itself through closed forms or algorithms
that exploit the structure, for example in $`O(n \log n)`$ for circulant systems where a dense LU would take
$`O(n^3)`$.

```cpp
#include <contrib/Eigen/StructuredMatrices>  // or the legacy <unsupported/Eigen/StructuredMatrices>
```

The module needs the `FFT` module (included by the umbrella header) and nothing outside Eigen. The class reference is
the Doxygen block of each class; this file gives the overview, the mathematics, and the costs.

## Contents

| Class | Matrix | Stored as | Product | Direct solve | Also |
|---|---|---|---|---|---|
| [`Circulant`](#circulant) | $`C_{ij} = c_{(i-j) \bmod n}`$ | first column | $`O(n \log n)`$ | `solve()` (pseudo-inverse), $`O(n \log n)`$ | eigen, SVD, rank, inverse, det |
| [`Toeplitz`](#toeplitz-and-lookaheadlevinson) | $`T_{ij} = t_{i-j}`$ | first column, first row | $`O(p \log p)`$ | `LookAheadLevinson`, $`O(n^2)`$ | solver `conditionEstimate()` |
| [`Hankel`](#hankel) | $`H_{ij} = h_{i+j}`$ | $`h`$, length $`m+n-1`$ | $`O(p \log p)`$ | `solve()`, $`O(n^2)`$ | `toToeplitz()` |
| [`Bccb`](#bccb) | 2-D circular convolution | $`n_2 \times n_1`$ array | $`O(N \log N)`$ | `solve()` (pseudo-inverse), $`O(N \log N)`$ | eigen, SVD, rank, inverse, det |
| [`KroneckerOperator`](#kroneckeroperator) | $`A \otimes B`$ | $`A`$, $`B`$ | $`O(N (n_1 + n_2))`$ | `solve()`, `leastSquaresSolve()` | eigen, SVD, rank, inverse, det |
| [`KroneckerSum`](#kroneckersum-and-bartelsstewart) | $`A \oplus B`$ | $`A`$, $`B`$ | $`O(N (n_1 + n_2))`$ | `BartelsStewart`, $`O(N \sum_k n_k)`$ | eigen |
| [`DiagonalPlusLowRank`](#diagonalpluslowrank) | $`D + U V^H`$ | $`d`$, $`U`$, $`V`$ | $`O(nk)`$ | `solve()` (Woodbury), $`O(nk^2 + k^3)`$ | inverse, det |
| [`Vandermonde`](#vandermonde-and-bjorckpereyra) | $`V_{ij} = x_i^{\,j}`$ | nodes | $`O(mn)`$ | `BjorckPereyra`, $`O(n^2)`$ | det |
| [`Cauchy`](#cauchy-and-cauchylu) | $`C_{ij} = 1/(x_i - y_j)`$ | nodes | $`O(mn)`$ | `CauchyLU`, $`O(n^2)`$ | det |
| [`DPR1EigenSolver`](#dpr1eigensolver) | $`D + \rho z z^T`$ | — | — | — | eigen, $`O(n^2)`$ |

Sizes: $`n`$ for a square operator, $`m \times n`$ for a rectangular one, $`p`$ the 5-smooth embedding length
$`\ge m+n-1`$, $`N = n_1 n_2`$ for the two-level operators, $`k`$ the rank of a low-rank correction. Product costs are
per right-hand side; the Kronecker entries are for dense square factors.

## Common conventions

### Operators

Every class in the table but `DPR1EigenSolver` is an operator: it derives from `EigenBase`, stores its own copy of its
generators, and provides `rows()`, `cols()` and `coeff(i, j)`. It is built by its constructor or by a `make*` factory
that deduces the compile-time sizes from the arguments (`makeCirculant`, `makeToeplitz`, `makeHankel`, `makeBccb`,
`makeKroneckerOperator`, `makeKroneckerSum`, `makeDiagonalPlusLowRank`, `makeVandermonde`, `makeCauchy`).

- **Products.** `A * x` returns an Eigen `Product` expression for a vector or a multi-column `x`. Assigning it behaves
  like any dense product: a temporary resolves aliasing between the destination and `x`, and `.noalias()` skips it.
  A real operator applied to a complex right-hand side yields a complex result.
- **Materialization.** Assigning an operator to a dense matrix forms it explicitly (`MatrixXd M = A;`, and also
  `M += A;`, `M -= A;`). `KroneckerOperator` and `KroneckerSum` also materialize into a `SparseMatrix`
  (`SparseMatrix<double> S; S = K;`).
- **Transposition.** `transpose()`, `conjugate()` and `adjoint()` return an operator of the same kind for every class
  except `Vandermonde`, whose transpose has no fast product. The FFT-based operators transform their cached symbol
  (index reversal, conjugation, or a phase factor) instead of recomputing an FFT.
- **Iterative solvers.** Since `A * x` is an ordinary product expression, the operators run matrix-free in
  `ConjugateGradient`, `BiCGSTAB`, `GMRES`, `MINRES`, and, through `adjoint()`, the least-squares solvers `LSMR` and
  `LeastSquaresConjugateGradient`. They expose no coefficient storage, so the solver must be instantiated with
  `IdentityPreconditioner`:

  ```cpp
  Toeplitz<double> T(col, row);  // rectangular, m > n
  LSMR<Toeplitz<double>, IdentityPreconditioner> lsmr(T);
  VectorXd x = lsmr.solve(b);    // O(p log p) per iteration
  ```

### Solvers

`LookAheadLevinson`, `BjorckPereyra`, `CauchyLU` and `BartelsStewart` derive from `SolverBase` and follow the usual
decomposition style: `compute(op)` or the constructor, a lazy `solve(b)` with any number of right-hand sides,
`info()`, and `transpose().solve(b)` and `adjoint().solve(b)`, which reuse the same factorization at the same cost.
`Circulant`, `Bccb`, `Hankel`, `KroneckerOperator`, `KroneckerSum` and `DiagonalPlusLowRank` also have a `solve(b)`
member that returns a plain matrix. `Circulant` and `Bccb` solve with their cached symbol; the others redo their setup
on every call (a Levinson factorization, factor LUs, Schur forms, or the capacitance LU), so for repeated solves use
`LookAheadLevinson` on `toToeplitz()` or `BartelsStewart` directly.

### Numerics

- **Determinants** are accumulated in the balanced form $`\mu \cdot 2^e`$, renormalizing every factor and the running
  product to $`\max(|\Re\mu|, |\Im\mu|) \in [\tfrac12, 1)`$: the mantissa/exponent split of LINPACK's `xGEDI` [4],
  in base 2 rather than 10 so that the rescaling is exact [12]. A partial product therefore cannot overflow or
  underflow when the determinant itself is representable, whatever the ordering of the factors; zeros and non-finite
  factors propagate exactly.
- **Numerical rank** follows `SVDBase`: a singular value $`\sigma`$ counts as zero exactly when
  $`\sigma < \max(c\,\varepsilon\,\sigma_{\max},\ \text{smallest normal})`$ with $`c = \min(\text{rows}, \text{cols})`$.
  The pseudo-inverse solves of `Circulant` and `Bccb` invert exactly the symbol entries that `rank()` counts.
- **FFTs** go through the `FFT` module's default backend, kissfft unless `EIGEN_FFTW_DEFAULT`, `EIGEN_MKL_DEFAULT`,
  `EIGEN_POCKETFFT_DEFAULT` or `EIGEN_DUCCFFT_DEFAULT` selects another. kissfft is fast only for 2-, 3- and 5-smooth
  lengths, so products pad the transform to the next length $`2^a 3^b 5^c`$; spectral quantities (eigenvalues,
  solves, rank, determinant) use the exact length. Products use the FFT only when a dimension exceeds 32 (for `Bccb`,
  when $`N`$ does), and never for a single-row or single-column `Hankel`. Each thread reuses one FFT engine and its
  plan tables (a fresh engine per call under `EIGEN_AVOID_THREAD_LOCAL`).

## Circulant

`Circulant<Scalar, Size = Dynamic>` is the $`n \times n`$ matrix generated by its first column $`c`$:

```math
C_{ij} = c_{(i-j) \bmod n}, \qquad
C = F^H \operatorname{diag}(\hat c)\, F, \qquad
\hat c_k = \sum_{j=0}^{n-1} c_j\, \omega^{jk}, \quad \omega = e^{-2\pi i/n},
```

with $`F_{jk} = \omega^{jk}/\sqrt{n}`$ the unitary DFT matrix. The eigenvalues are the DFT $`\hat c`$ of the generator,
the operator's *symbol*, computed once at construction; the eigenvectors are the columns $`f_k`$ of $`F^H`$, with
entries $`e^{2\pi i jk/n}/\sqrt{n}`$, the same for every circulant matrix. Everything follows from this factorization:

| Member | Result | Cost |
|---|---|---|
| `C * x` | $`\mathcal{F}^{-1}(\hat c \odot \mathcal{F} x)`$ | $`O(n \log n)`$ |
| `solve(b)` | $`C^+ b = \mathcal{F}^{-1}(\hat c^{+} \odot \mathcal{F} b)`$, $`\hat c_k^{+} = 1/\hat c_k`$ for the entries `rank()` counts, else 0 | $`O(n \log n)`$ |
| `eigenvalues()`, `symbol()` | $`\hat c`$, in DFT order | precomputed for $`n > 32`$ |
| `eigenvectors()` | $`[f_0, \dots, f_{n-1}]`$, dense | $`O(n^2)`$ |
| `singularValues()` | $`\lvert \hat c_k \rvert`$, decreasing | $`O(n \log n)`$ |
| `matrixU()`, `matrixV()` | columns $`e^{i \arg \hat c_k} f_k`$ and $`f_k`$, in the order of `singularValues()`, dense | $`O(n^2)`$ |
| `rank()`, `determinant()` | count of retained $`\hat c_k`$; $`\prod_k \hat c_k`$ | $`O(n \log n)`$ |
| `inverse()` | the `Circulant` with column $`\mathcal{F}^{-1}(1/\hat c)`$ | $`O(n \log n)`$ |
| `transpose()`, `conjugate()`, `adjoint()` | `Circulant`, symbol $`\hat c_{-k}`$, $`\overline{\hat c_{-k}}`$, $`\overline{\hat c_k}`$ (indices mod $`n`$) | $`O(n)`$ |

For a real generator the symbol is conjugate-symmetric, $`\hat c_{n-k} = \overline{\hat c_k}`$, the two modes of each
pair carry identical singular values, and `determinant()` returns the real part of the product. `inverse()` requires
a nonsingular operator; `solve()` handles singular ones.

```cpp
VectorXd c = VectorXd::Random(n);
Circulant<double> C(c);                 // or makeCirculant(c)
VectorXd y = C * x;
VectorXd z = C.solve(y);                // minimum-norm least-squares solution
VectorXcd lambda = C.eigenvalues();
```

## Toeplitz and LookAheadLevinson

`Toeplitz<Scalar, Rows = Dynamic, Cols = Dynamic>` is the $`m \times n`$ matrix with constant diagonals, generated by
its first column $`c`$ and first row $`r`$ ($`r_0`$ is ignored; the diagonal comes from $`c`$):

```math
T_{ij} = \begin{cases} c_{i-j}, & i \ge j, \\ r_{j-i}, & i < j. \end{cases}
```

The product embeds $`T`$ in the circulant matrix of 5-smooth size $`p \ge m+n-1`$ with first column
$`[c_0, \dots, c_{m-1}, 0, \dots, 0, r_{n-1}, \dots, r_1]^T`$; $`Tx`$ is the leading $`m`$ entries of that circulant
applied to $`[x; 0]`$, at $`O(p \log p)`$ with the embedding symbol computed once at construction. `transpose()`
swaps the generators; it and `conjugate()` and `adjoint()` reuse the embedding symbol. `column()` and `row()` return the
generators.

`LookAheadLevinson<Scalar>` solves square systems $`Tx = b`$ in $`O(n^2)`$ by the look-ahead Levinson algorithm of
Chan and Hansen [1, 2]. The classical Levinson recursion breaks down at a singular or ill-conditioned leading principal
submatrix, which general (indefinite, nonsymmetric) Toeplitz matrices often have; the look-ahead variant steps over
up to $`p_{\max} - 1`$ consecutive ill-conditioned leading submatrices with a block step, and remains weakly stable
when there are no longer runs.

| Member | Meaning |
|---|---|
| `setMaxBlockSize(p)` | $`p_{\max}`$, default 4; call before `compute()` |
| `info()` | `Success`, or `NumericalIssue` when `conditionEstimate()` reaches $`1/\varepsilon`$; a run of ill-conditioned leading submatrices longer than the look-ahead is reported only through that estimate |
| `conditionEstimate()` | estimate of $`\kappa_2(T)`$, the "algorithm condition number" of [1] |
| `transpose().solve(b)`, `adjoint().solve(b)` | reuse the factorization through the persymmetry $`T^T = E T E`$ ($`E`$ the exchange matrix) |

```cpp
Toeplitz<double> T(col, row);           // square
LookAheadLevinson<double> levinson(T);
VectorXd x = levinson.solve(b);         // O(n^2)
if (levinson.info() != Success) { /* condition estimate reached 1/eps */ }
```

## Hankel

`Hankel<Scalar, Rows = Dynamic, Cols = Dynamic>` is the $`m \times n`$ matrix with constant anti-diagonals,
$`H_{ij} = h_{i+j}`$. It is built from its first column and last row (the shared corner entry comes from the column)
and stores the single sequence $`h`$ of length $`m+n-1`$ (`generator()`, `column()`, `lastRow()`). A real square
Hankel matrix is symmetric.

A Hankel matrix is a column-reversed Toeplitz matrix, $`H = TE`$ with $`T_{ij} = h_{i+n-1-j}`$ (`toToeplitz()`), so
its product is a convolution evaluated in $`O(p \log p)`$ like the Toeplitz one. Single-row and single-column
operators skip the FFT. `transpose()` keeps $`h`$ and swaps the dimensions; its symbol is the cached one multiplied by
the phases $`e^{2\pi i f (m-n)/p}`$, with no new FFT.

`solve(b)` solves a square system in $`O(n^2)`$ as $`T (Ex) = b`$ with `LookAheadLevinson`. It factorizes on every
call and `eigen_assert`s that the recursion did not break down; to reuse the factorization or check `info()` instead,
run `LookAheadLevinson` on `toToeplitz()` and reverse the rows of its solution.

## Bccb

`Bccb<Scalar, BlockSize = Dynamic, NumBlocks = Dynamic>` is a block circulant matrix with circulant blocks, the
matrix of a two-dimensional circular convolution. It is $`N \times N`$, $`N = n_1 n_2`$: an $`n_1 \times n_1`$ block
circulant whose blocks are $`n_2 \times n_2`$ circulants. It is generated by the $`n_2 \times n_1`$ array $`G`$ whose
column $`k`$ is the first column of block $`k`$; with $`i = i_1 n_2 + i_2`$ and $`j = j_1 n_2 + j_2`$,

```math
C_{ij} = G_{(i_2 - j_2) \bmod n_2,\ (i_1 - j_1) \bmod n_1}, \qquad
C \operatorname{vec}(X) = \operatorname{vec}(G \circledast X),
```

where $`\operatorname{vec}`$ stacks columns and $`\circledast`$ is 2-D circular convolution. BCCB matrices are
diagonalized by the 2-D DFT $`F_{n_1} \otimes F_{n_2}`$ [5, 6]: the symbol $`\hat G`$, the 2-D DFT of $`G`$, holds
the eigenvalues, and eigenvalue $`f_1 n_2 + f_2`$ is $`\hat G_{f_2 f_1}`$. The API mirrors `Circulant` (`solve`,
`eigenvalues`, `eigenvectors`, `singularValues`, `matrixU`, `matrixV`, `rank`, `inverse`, `determinant`, and the
transposition family) at $`O(N \log N)`$, with $`O(N^2)`$ for the dense vector matrices. `blockSize()` and
`numBlocks()` return $`n_2`$ and $`n_1`$.

BCCB matrices model blurring with periodic boundary conditions in image deblurring, and are the standard
preconditioners for two-level Toeplitz (BTTB) systems [6].

## KroneckerOperator

`KroneckerOperator<Lhs, Rhs>` is the Kronecker product $`A \otimes B`$, $`A`$ of size $`m_1 \times n_1`$ and $`B`$ of
size $`m_2 \times n_2`$, as an operator that is never formed. With $`\operatorname{vec}`$ stacking the columns of an
$`n_2 \times n_1`$ matrix $`X`$ (whatever the storage order) and $`\operatorname{mat}`$ its inverse,

```math
(A \otimes B) \operatorname{vec}(X) = \operatorname{vec}(B X A^T),
```

at $`O(m_2 n_1 (n_2 + m_1))`$ per right-hand side instead of $`O(m_1 m_2 n_1 n_2)`$; right-hand sides are processed in
cache-sized batches. Everything else factors through $`A`$ and $`B`$:

| Member | Identity | Notes |
|---|---|---|
| `solve(b)` | $`X = B^{-1} \operatorname{mat}(b) A^{-T}`$ | one LU per dense factor, `SparseLU` per sparse factor; factors square and invertible |
| `leastSquaresSolve(b)` | $`(A \otimes B)^+ = A^+ \otimes B^+`$ | one `CompleteOrthogonalDecomposition` per factor, rank decided per factor; rectangular or rank-deficient factors |
| `rank()` | number of $`\sigma_i(A)\,\sigma_j(B) \ge \min(\text{rows}, \text{cols})\,\varepsilon\,\sigma_{\max}(A)\,\sigma_{\max}(B)`$ | two SVDs; can be below $`\operatorname{rank}(A)\operatorname{rank}(B)`$ |
| `determinant()` | $`\det(A)^{n_2} \det(B)^{n_1}`$ | balanced accumulation of the factor LU pivots |
| `inverse()` | $`A^{-1} \otimes B^{-1}`$ | a `KroneckerOperator` |
| `eigenvalues()`, `eigenvectors()` | $`\lambda_i(A)\,\mu_j(B)`$ at index $`i n_2 + j`$; $`V_A \otimes V_B`$ | unsorted; the vectors are a `KroneckerOperator` |
| `singularValues()`, `matrixU()`, `matrixV()` | $`\sigma_i(A)\,\sigma_j(B)`$; $`U_A \otimes U_B`$; $`V_A \otimes V_B`$ | thin SVD, unsorted; the vectors are `KroneckerOperator`s |
| `transpose()`, `conjugate()`, `adjoint()` | $`A^T \otimes B^T`$, $`\bar A \otimes \bar B`$, $`A^H \otimes B^H`$ | |

The eigenvalues and singular values stay in Kronecker order because sorting them would destroy the Kronecker
structure of the vector matrices.

Each factor can be any of:

| Factor passed | Stored as | Its side of a product | Its solve |
|---|---|---|---|
| dense expression | `Matrix` | GEMM | `PartialPivLU` |
| `DiagonalMatrix` or `.asDiagonal()` expression | its diagonal, $`O(n)`$ | scaling | entrywise division |
| `SparseMatrix` | compressed `SparseMatrix` | sparse-dense product, $`O(\operatorname{nnz})`$ per column | `SparseLU`, factorized once |
| `MatrixXd::Identity(m, n)` | dimensions only | skipped when square; a rectangular one keeps the leading $`\min(m, n)`$ rows or columns | identity |
| `KroneckerOperator` | as is (nested) | its own vec identity | its own factor solves |
| `KroneckerSum` | as is (nested) | $`B X + X A^T`$ | `BartelsStewart` |

`makeKroneckerOperator(a, b, c, ...)` nests to the right, so products of three or more factors stay implicit. The
decomposition family (`eigenvalues` through `matrixV`, `rank`, `leastSquaresSolve`) works on dense copies of
diagonal, identity and sparse factors; `rank` and `leastSquaresSolve` also materialize a nested factor. A sparse
factor whose `SparseLU` fails solves and inverts to NaN.

```cpp
SparseMatrix<double> A = ...;           // n x n
auto K = makeKroneckerOperator(MatrixXd::Identity(p, p), A, MatrixXd::Identity(q, q));  // I_p (x) A (x) I_q
VectorXd y = K * x;                     // O(p q nnz(A)), nothing of size p n q formed
```

Unlike `kroneckerProduct()` from the `KroneckerProduct` module, an expression meant to be evaluated into a matrix,
`KroneckerOperator` is meant to be applied and solved with as it is.

## KroneckerSum and BartelsStewart

`KroneckerSum<Lhs, Rhs>` is the Kronecker sum of square factors, $`A`$ of size $`n_1`$ and $`B`$ of size $`n_2`$,

```math
A \oplus B = A \otimes I_{n_2} + I_{n_1} \otimes B, \qquad
(A \oplus B) \operatorname{vec}(X) = \operatorname{vec}(B X + X A^T),
```

the operator of separable discretizations on tensor-product grids: with $`D_x, D_y`$ the 1-D second-difference
matrices and $`x`$ the fast index of the unknown vector, the 2-D Laplacian is $`D_y \oplus D_x`$ and the 3-D one
$`D_z \oplus D_y \oplus D_x`$. The factors may be
of any kind `KroneckerOperator` accepts, including a `KroneckerSum` (`makeKroneckerSum(a, b, c, ...)` nests to the
right). The product costs one product with each factor, $`O(N (n_1 + n_2))`$ for dense factors and
$`O(n_1 \operatorname{nnz}(B) + n_2 \operatorname{nnz}(A))`$ for sparse ones, with no identity ever formed.
`eigenvalues()` returns the sums $`\lambda_i(A) + \mu_j(B)`$ at index $`i n_2 + j`$ and `eigenvectors()` the
`KroneckerOperator` $`V_A \otimes V_B`$.

`BartelsStewart<KroneckerSumType>` solves $`(A_1 \oplus \cdots \oplus A_d)\, x = b`$ for any number of factors. With
Schur forms $`A_k = Q_k T_k Q_k^H`$,

```math
A_1 \oplus \cdots \oplus A_d = Q\, (T_1 \oplus \cdots \oplus T_d)\, Q^H, \qquad Q = Q_1 \otimes \cdots \otimes Q_d,
```

so a solve applies $`Q^H`$, back-substitutes the (quasi-)upper-triangular middle factor, and applies $`Q`$; for two
factors this is the Bartels-Stewart algorithm [3] for the Sylvester equation $`B X + X A^T = \operatorname{mat}(b)`$.
The Schur forms depend on the factors:

- every factor exactly Hermitian (`isHermitian()`): eigendecompositions, so the middle factor is diagonal and the
  solve divides by the eigenvalue sums, the fast diagonalization method [7], in real arithmetic for real factors;
- real factors, not all symmetric, at most four of them: real Schur forms, $`Q_k`$ orthogonal and $`T_k`$
  quasi-upper-triangular with a $`2 \times 2`$ block per complex conjugate eigenvalue pair; the coupled block rows
  are solved jointly, ending in dense solves at most $`2^d`$ wide, as LAPACK's `xTRSYL` does for two factors;
- otherwise (complex factors, or more than four real ones): complex Schur forms, $`T_k`$ upper triangular.

Setup costs one $`O(n_k^3)`$ decomposition per factor (sparse factors are densified for it) and each solve
$`O(N \sum_k n_k)`$ per right-hand side, plus up to $`O(4^d N)`$ for the coupled solves of the real path.
`KroneckerSum::solve(b)` builds a `BartelsStewart` for that one call.

The system is singular exactly when some sum $`\lambda_{i_1}(A_1) + \cdots + \lambda_{i_d}(A_d)`$ vanishes, which, as
with `PartialPivLU`, nothing detects. The residual satisfies
$`\lVert b - Mx \rVert = O\big((\textstyle\sum_k n_k)\,\varepsilon\,(\sum_k \lVert A_k \rVert)\,\lVert x \rVert\big)`$,
relative to $`\sum_k \lVert A_k \rVert`$ rather than $`\lVert M \rVert`$, which matters when a shift is split across
the factors as in $`(A + cI) \oplus (B - cI)`$. `info()` reports `InvalidInput` for a non-finite factor and
`NoConvergence` when a Schur or eigenvalue iteration fails; the solve then returns NaN.

```cpp
// Implicit Euler for u' = (Dy (+) Dx) u: (I - tau L) u_new = u, with I (x) I = I absorbed into one factor.
SparseMatrix<double> Iy(ny, ny);
Iy.setIdentity();
auto M = makeKroneckerSum(Iy - tau * Dy, -tau * Dx);
BartelsStewart<decltype(M)> step(M);    // Schur forms, once
for (int k = 0; k < steps; ++k) u = step.solve(u);
```

## DiagonalPlusLowRank

`DiagonalPlusLowRank<Scalar, Size = Dynamic, Rank = Dynamic>` is $`A = D + U V^H`$ with $`D = \operatorname{diag}(d)`$
and $`U, V`$ of size $`n \times k`$ (`diagonal()`, `factorU()`, `factorV()`, `correctionRank()`; $`k = 0`$ is
allowed). Products cost $`O(nk)`$. The Woodbury identity [8] and the matrix determinant lemma [9] reduce everything
else to the $`k \times k`$ capacitance matrix $`S = I_k + V^H D^{-1} U`$ (`capacitance()`):

```math
A^{-1} = D^{-1} - D^{-1} U S^{-1} V^H D^{-1}, \qquad \det A = \det(D) \det(S).
```

`solve(b)` costs $`O(nk^2 + k^3)`$ to set up and $`O(nk)`$ per right-hand side; `determinant()` is $`O(nk^2 + k^3)`$;
`inverse()` returns the `DiagonalPlusLowRank` $`D^{-1} + \tilde U \tilde V^H`$ with $`\tilde U = -D^{-1} U S^{-1}`$ and
$`\tilde V = D^{-H} V`$; `transpose()` is $`D + \bar V \bar U^H`$ and `adjoint()` is $`\bar D + V U^H`$.

`solve` and `inverse` need $`D`$ and $`S`$ nonsingular and `determinant` needs $`D`$ nonsingular, all with finite
reciprocals $`1/d_i`$; this is not checked beyond NaN/Inf propagation. The splitting routes the solution through
$`D^{-1} b`$, so accuracy degrades when $`\max_i |1/d_i|`$ greatly exceeds $`\lVert A^{-1} \rVert`$, even for a
well-conditioned $`A`$ [10].

## Vandermonde and BjorckPereyra

`Vandermonde<Scalar, Rows = Dynamic, Cols = Dynamic>` is the $`m \times n`$ matrix $`V_{ij} = x_i^{\,j}`$ of the
nodes $`x`$ (`nodes()`); `Vandermonde(x)` is square and `Vandermonde(x, n)` has $`n`$ columns. `V * a` evaluates the
polynomial with ascending coefficients $`a`$ at every node by Horner's rule,

```math
(Va)_i = \sum_{j=0}^{n-1} a_j x_i^{\,j}: \qquad p_{n-1} = a_{n-1}, \quad p_j = a_j + x_i\, p_{j+1},
```

at $`O(mn)`$, the cost of a dense product but with $`O(m)`$ storage. `determinant()` uses
$`\det V = \prod_{i<j} (x_j - x_i)`$ in $`O(n^2)`$. The transpose has no fast product, so `Vandermonde` is not closed
under transposition; least-squares problems are best solved by a dense QR of the materialized matrix. Integer scalar
types are rejected.

`BjorckPereyra<Scalar>` solves square systems in $`O(n^2)`$ operations and $`O(n)`$ storage [11, §4.6]: $`Va = f`$ is
polynomial interpolation, solved by divided differences in the Newton basis followed by the change to the monomial
basis, and the dual (moment) system $`V^T w = b`$ runs the transposed recurrences through `transpose().solve(b)`.
There is no factorization: `compute()` stores the nodes and `info()` reports `NumericalIssue` for an exactly repeated
node or `InvalidInput` for a non-finite one. Genuinely complex nodes are put in Leja order [13] to limit growth in
the Newton representation; real nodes keep their input order.

Real-node Vandermonde matrices are exponentially ill-conditioned: $`\kappa_2(V)`$ grows at least like $`2^n`$ for every
real node set [14]. Björck-Pereyra solves are nevertheless often far more accurate than that suggests: for monotone
nodes and a sign-alternating right-hand side the forward error obeys a small relative bound independent of
$`\kappa(V)`$ [15, ch. 22]. Nodes on the unit circle are the well-conditioned case; for the $`n`$-th roots of unity
$`V/\sqrt{n}`$ is unitary.

## Cauchy and CauchyLU

`Cauchy<Scalar, Rows = Dynamic, Cols = Dynamic>` is $`C_{ij} = 1/(x_i - y_j)`$ for row nodes $`x`$ and column nodes
$`y`$ (`rowNodes()`, `colNodes()`), stored as the $`m + n`$ nodes. Every $`x_i`$ must differ from every $`y_j`$; this
is not checked. Products are evaluated directly at $`O(mn)`$ with $`O(m)`$ extra storage. The class is closed under
transposition, $`C(x, y)^T = C(-y, -x)`$ and $`C(x, y)^H = C(-\bar y, -\bar x)`$, and the determinant of a square
Cauchy matrix has the closed form

```math
\det C = \frac{\prod_{i<j} (x_j - x_i)(y_i - y_j)}{\prod_{i,j} (x_i - y_j)},
```

evaluated in $`O(n^2)`$. The Hilbert matrix is the Cauchy matrix with $`x_i = i + 1`$, $`y_j = -j`$.

`CauchyLU<Scalar>` factors $`PC = LU`$ with partial pivoting in $`O(n^2)`$ by the Gohberg-Kailath-Olshevsky
algorithm [16]. A Cauchy matrix satisfies the rank-one displacement equation
$`D_x C - C D_y = \mathbf{1}\mathbf{1}^T`$, which survives row permutations, so every Schur complement is again
Cauchy-like, $`S_{ij} = a_i b_j / (x_i - y_j)`$, and elimination step $`k`$ only updates the generators:

```math
a_i \leftarrow a_i \frac{x_i - x_k}{x_i - y_k}, \qquad b_j \leftarrow b_j \frac{y_j - y_k}{y_j - x_k}.
```

Each column of the Schur complement is generated from $`O(n)`$ data, so the pivot search costs nothing extra: this is
genuine partial pivoting at fast-algorithm cost, which the Levinson family for Toeplitz matrices lacks. The factors
are stored densely ($`O(n^2)`$ memory); `info()` reports `NumericalIssue` when a zero or non-finite entry prevents
the factorization.

## DPR1EigenSolver

`DPR1EigenSolver<RealScalar>` (`float`, `double` or `long double`) computes all eigenvalues and eigenvectors of the
real symmetric diagonal-plus-rank-one matrix $`A = D + \rho z z^T`$ in $`O(n^2)`$. It is the kernel at the heart of
divide-and-conquer symmetric eigensolvers (LAPACK's `xLAED2`/`xLAED3`/`xLAED4`), as a standalone class. After
deflation (entries with negligible $`|z_i|`$ are already eigenpairs, and nearly equal $`d_i`$ are merged by Givens
rotations whose dropped coupling is below a backward-stability threshold), the remaining eigenvalues are the roots of
the secular equation

```math
f(\lambda) = 1 + \rho \sum_i \frac{z_i^2}{d_i - \lambda} = 0,
```

one between each pair of consecutive poles and, for $`\rho > 0`$, one beyond the largest pole. Each root is bracketed
and bisected in coordinates shifted to its nearest pole, so every $`\lambda - d_i`$ is an exact data difference plus a
small offset. The eigenvectors are built from the Gu-Eisenstat vector [17]

```math
\hat z_i^2 = \frac{\prod_j (\lambda_j - d_i)}{\rho \prod_{j \ne i} (d_j - d_i)}, \qquad
v_j \propto (D - \lambda_j I)^{-1} \hat z,
```

for which the computed $`\lambda_j`$ are the exact eigenvalues of $`D + \rho \hat z \hat z^T`$; this makes the computed
$`V`$ numerically orthogonal without reorthogonalization. Either sign of $`\rho`$, $`\rho = 0`$, zero $`z`$, repeated
$`d_i`$ and any ordering of $`d`$ are supported; the problem is scaled internally by an exact power of two.

```cpp
DPR1EigenSolver<double> es(d, rho, z);  // or es.compute(d, rho, z, EigenvaluesOnly)
VectorXd lambda = es.eigenvalues();     // ascending
MatrixXd V = es.eigenvectors();         // orthogonal
```

`info()` reports `InvalidInput` for non-finite input, an overflowing $`\rho \lVert z \rVert^2`$ or a non-finite
eigenvalue, and `NoConvergence` when a secular root could not be resolved.

## Tests and benchmarks

The tests are `contrib/test/structured_*.cpp`: `structured_matrices` (Circulant, Toeplitz, LookAheadLevinson,
Hankel), `structured_bccb`, `structured_kronecker`, `structured_kronecker_sum`, `structured_dplr`,
`structured_vandermonde` (and `structured_vandermonde_int_index`), `structured_cauchy` and `structured_dpr1`.

```bash
cmake -G Ninja -S . -B build
cmake --build build --target structured_matrices structured_bccb structured_kronecker structured_kronecker_sum \
  structured_dplr structured_vandermonde structured_vandermonde_int_index structured_cauchy structured_dpr1
ctest --test-dir build -R '^structured_' --output-on-failure --no-tests=error
```

Benchmarks are in `contrib/benchmarks/StructuredMatrices` (`bench_structured_*`, one per operator and for the
solves, transposition, materialization and the Kronecker spectral paths) and `benchmarks/StructuredMatrices`
(FFT operators, Vandermonde and Cauchy, diagonal-plus-low-rank). Each tree is a standalone CMake project:

```bash
cmake -G Ninja -S contrib/benchmarks -B build-contrib-bench -DCMAKE_BUILD_TYPE=Release
cmake --build build-contrib-bench --target bench_structured_circulant
```

## References

1. T. F. Chan and P. C. Hansen, "A look-ahead Levinson algorithm for general Toeplitz systems," *IEEE Trans. Signal
   Process.* 40(5):1079–1090, 1992.
2. T. F. Chan and P. C. Hansen, "A look-ahead Levinson algorithm for indefinite Toeplitz systems," *SIAM J. Matrix
   Anal. Appl.* 13(2):490–506, 1992.
3. R. H. Bartels and G. W. Stewart, "Solution of the matrix equation AX + XB = C," *Comm. ACM* 15:820–826, 1972.
4. J. J. Dongarra, J. R. Bunch, C. B. Moler and G. W. Stewart, *LINPACK Users' Guide*, SIAM, 1979.
5. P. J. Davis, *Circulant Matrices*, Wiley, 1979.
6. R. H. Chan and X.-Q. Jin, *An Introduction to Iterative Toeplitz Solvers*, SIAM, 2007.
7. R. E. Lynch, J. R. Rice and D. H. Thomas, "Direct solution of partial difference equations by tensor product
   methods," *Numer. Math.* 6:185–199, 1964.
8. M. A. Woodbury, "Inverting modified matrices," Memorandum Report 42, Statistical Research Group, Princeton
   University, 1950.
9. W. W. Hager, "Updating the inverse of a matrix," *SIAM Review* 31(2):221–239, 1989.
10. E. L. Yip, "A note on the stability of solving a rank-p modification of a linear system by the
    Sherman-Morrison-Woodbury formula," *SIAM J. Sci. Stat. Comput.* 7(2):507–513, 1986.
11. G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins University Press, 2013.
12. P. H. Sterbenz, *Floating-Point Computation*, Prentice-Hall, 1974.
13. L. Reichel, "Newton interpolation at Leja points," *BIT* 30:332–346, 1990.
14. B. Beckermann, "The condition number of real Vandermonde, Krylov and positive definite Hankel matrices,"
    *Numer. Math.* 85:553–577, 2000.
15. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002.
16. I. Gohberg, T. Kailath and V. Olshevsky, "Fast Gaussian elimination with partial pivoting for matrices with
    displacement structure," *Math. Comp.* 64:1557–1576, 1995.
17. M. Gu and S. C. Eisenstat, "A stable and efficient algorithm for the rank-one modification of the symmetric
    eigenproblem," *SIAM J. Matrix Anal. Appl.* 15(4):1266–1276, 1994.
