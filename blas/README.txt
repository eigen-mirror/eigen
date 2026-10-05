
This directory contains a BLAS library built on top of Eigen.

This module is not built by default. In order to compile it, you need to
type 'make blas' from within your build dir.

On 64-bit platforms, 'make blas' builds both 32-bit integer (LP64: eigen_blas,
eigen_blas_static) and 64-bit integer (ILP64: eigen_blas_ilp64,
eigen_blas_ilp64_static) libraries by default (controlled by
EIGEN_BUILD_BLAS_ILP64).
