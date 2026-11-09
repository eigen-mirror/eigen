
This directory contains a BLAS library built on top of Eigen.

The libraries are part of the default build when Eigen is the top-level
project (EIGEN_BUILD_BLAS); 'make blas' builds them alone.

On 64-bit platforms, 'make blas' builds both 32-bit integer (LP64: eigen_blas,
eigen_blas_static) and 64-bit integer (ILP64: eigen_blas_ilp64,
eigen_blas_ilp64_static) libraries by default (controlled by
EIGEN_BUILD_BLAS_ILP64).

Tests: 'make buildtests_blas' builds them and 'ctest -L blas' runs them. With
a Fortran compiler, these include netlib's reference BLAS test drivers. With
-DEIGEN_TEST_BLAS_REFERENCE=ON, blas_reference also compares every routine
with netlib's reference BLAS, which CMake downloads with Reference-LAPACK.
