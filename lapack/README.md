The LAPACK compatibility library is enabled with `EIGEN_BUILD_LAPACK` and
requires `EIGEN_BUILD_BLAS`. Its optional numerical tests
(`EIGEN_ENABLE_LAPACK_TESTS`) require a Fortran compiler and the LAPACK test data.

The tests and their summary run through CMake/CTest without Python.
`LAPACK_Test_Summary` waits for the registered numerical tests and combines
only their expected output files into `TESTING/testing_results.txt` in the
LAPACK build directory. It fails when the output reports numerical errors,
illegal-argument or INFO diagnostics, when an expected output is missing, or
when no numerical tests were analyzed. Selecting `LAPACK_Test_Summary` alone also
runs its numerical tests through a CTest fixture. If a numerical driver fails to
execute successfully, CTest skips the summary.

Balancing tests contribute their example counts and fail on nonzero INFO/ILO
error counts or non-finite maximum errors. Their finite maximum errors remain
in the combined report; these drivers do not specify a pass/fail threshold.

The numerical tests use unmodified LAPACK 3.12.1 sources, downloaded with a pinned
SHA-256 hash. They exercise Eigen's BLAS/LAPACK implementations, using reference
LAPACK for the remaining routines. All test cases remain enabled; known numerical
and test-driver failures are reported by the summary.
