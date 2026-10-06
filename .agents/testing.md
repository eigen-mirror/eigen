# Testing Eigen Changes

Use this guide when adding or changing tests. The checked-out source is authoritative:

- [`test/main.h`](../test/main.h) configures and runs the test framework and aggregates the shared helpers.
- [`test/numerical_test_helpers.h`](../test/numerical_test_helpers.h) defines numerical comparison, assertion, and
  tolerance helpers.
- [`test/product_test_helpers.h`](../test/product_test_helpers.h) defines matrix-product error bounds.
- [`test/random_matrix_helper.h`](../test/random_matrix_helper.h) and
  [`test/type_test_helpers.h`](../test/type_test_helpers.h) define random-matrix and type utilities.
- [`cmake/EigenTesting.cmake`](../cmake/EigenTesting.cmake) defines test registration and splitting.
- [`test/CMakeLists.txt`](../test/CMakeLists.txt) and
  [`contrib/test/CMakeLists.txt`](../contrib/test/CMakeLists.txt) register the suites.
- [`cmake/EigenConfigureTesting.cmake`](../cmake/EigenConfigureTesting.cmake) defines aggregate build and check
  targets.

## Configure And Build

Configure a dedicated build directory. Unit tests are excluded from CMake's default `all` target, although a bare
build may still build enabled auxiliary libraries.

```bash
cmake -G Ninja -S . -B build
cmake --build build --target buildtests
ctest --test-dir build --parallel --output-on-failure --no-tests=error
```

Useful aggregate targets are `BuildOfficial`, `BuildContrib`, `buildsmoketests`, `buildtests_gpu`, `check`, and
`check_gpu`. Build and run one test explicitly when possible:

```bash
cmake --build build --target bdcsvd_3
ctest --test-dir build -R '^bdcsvd_3$' --output-on-failure --no-tests=error
```

Run the generated wrappers from the build directory because they invoke the configured build tool relative to their
working directory:

```bash
cd build
./buildtests.sh <regex>
./check.sh <regex>
```

They filter registered parent names such as `bdcsvd`, not generated part names such as `bdcsvd_3`; use the explicit
target recipe for one part.

Use a separate build directory for each materially different configuration. Do not rewrite one cache and describe
the result as a second test run.

```bash
cmake -G Ninja -S . -B build-row-major -DEIGEN_DEFAULT_TO_ROW_MAJOR=ON
cmake -G Ninja -S . -B build-no-vector -DEIGEN_TEST_NO_EXPLICIT_VECTORIZATION=ON
```

Consult the top-level [`CMakeLists.txt`](../CMakeLists.txt) and nearby test CMake files for current options instead of
copying an option inventory into documentation.

## Current Test Framework

Eigen currently uses its own framework, not GoogleTest:

1. Add `test/<name>.cpp` or `contrib/test/<name>.cpp`.
2. Include `main.h`, then the public umbrella header for tests of public behavior. A focused test of a private utility
   may include its implementation header only when that matches an established nearby pattern; never present such a
   path as a user include.
3. Use `VERIFY`, `VERIFY_IS_EQUAL`, `VERIFY_IS_APPROX`, and the other helpers exposed through `test/main.h`.
4. End with `EIGEN_DECLARE_TEST(<name>) { ... }`.
5. Register the source with `ei_add_test(<name>)` in the matching `CMakeLists.txt`, then reconfigure.

Keep `test/main.h` limited to framework configuration, registration, shared-helper aggregation, and the test driver.
Put reusable utilities in a narrowly named helper header; include it from `main.h` only when most tests need it.

For compile-failure coverage, use the established `failtest/` pattern. Its `_ok` target must compile and its `_ko`
target must fail with `EIGEN_SHOULD_FAIL_TO_BUILD` defined. Register it with `ei_add_failtest` above the closing
`ei_add_failtest_fixture()` call. The `buildfailtests` fixture builds the whole suite at once, and `_ko` passes when its
target did not build while `_ok` did. That rules out a broken toolchain, but not a different compile error than the one
intended, so keep the construct narrow.

## Split Tests

`ei_add_test` scans the source for `CALL_SUBTEST_N`, `EIGEN_TEST_PART_N`, and `EIGEN_SUFFIXES;...` markers.

- With `EIGEN_SPLIT_LARGE_TESTS=ON`, every discovered suffix becomes an executable `<name>_<N>` compiled with
  `EIGEN_TEST_PART_<N>=1`; the parent `<name>` target builds all parts.
- `EIGEN_SUFFIXES;...` supplies an explicit suffix list when ordinary source scanning cannot see macro-generated or
  conditional parts.
- With splitting off, tests containing only `CALL_SUBTEST_N` or `EIGEN_SUFFIXES` fold into one `<name>` executable
  compiled with `EIGEN_TEST_PART_ALL=1`.
- An explicit `EIGEN_TEST_PART_N` marker forces splitting even when the option is off. If any such marker is present,
  all suffixes discovered in that source are emitted.
- [`cmake/EigenTestPartGroups.cmake`](../cmake/EigenTestPartGroups.cmake) lists ranges of parts that compile together as
  one executable, named after the range's first part, e.g. `array_cwise_1` for `1-4`; `ctest -R '<name>'` still selects
  them. A range may only hold parts that differ in nothing but their `CALL_SUBTEST_N` calls and that are not listed
  individually in the smoke list. Its compile must stay under the 4 GiB peak RSS recorded in the file header. Adding or
  renumbering subtests inside a listed range makes that compile bigger, so re-measure it.

`ctest -R '^<name>$'` does not match split parts. Use `ctest -R '<name>'` for every part or anchor one generated name.

After changing subtest registration, reconfigure and read back the generated target list. Two failure modes are silent.
A subtest function whose `CALL_SUBTEST` call was dropped still compiles and looks like coverage. And under
`EIGEN_SPLIT_LARGE_TESTS=ON`, a part invoked only through a dispatch macro is not built unless an `EIGEN_SUFFIXES`
marker lists it.

## Coverage That Can Fail

A test that passes when the change is reverted is not coverage. Establish that it fails at the parent commit, or, when
that is impractical, that by construction it runs the new code.

- Test a new fast path through the public entry point that selects it, with inputs that actually take it, not only
  through a direct call to the new method. Where a flag or trait selects the fast path, pin the selection with a
  `STATIC_CHECK` on it in both directions: for a type that must opt in and for one that must stay out.
- Cover the branches the change adds, not just one convenient shape: sizes that are not a multiple of the packet or
  block dimension, complex scalars where conjugation is otherwise a no-op, both storage orders, and the uncompressed
  or strided variants of an input type.
- Verify the complete result against an independent reference; skipping coefficients the test setup did not write
  hides corruption in exactly those places.
- Exercise the customization points users are documented to have (custom scalars, functors without declared traits),
  not only the built-in specializations that happen to satisfy a new precondition.

## Build-System Tests

[`test/buildsystem`](../test/buildsystem) holds the coverage for Eigen's own CMake surface: what an install tree
contains, what `find_package(Eigen3)` and the version ranges in
[`cmake/Eigen3ConfigVersion.cmake.in`](../cmake/Eigen3ConfigVersion.cmake.in) accept, and how an embedding project
opts out of Eigen's install rules. They exist because those are claims
[`doc/TopicCMakeGuide.dox`](../doc/TopicCMakeGuide.dox) makes to users and nothing else checks; the blocking
documentation job only builds the docs, it does not run what they describe.

Not every scenario checks a documented claim. Some cover CMake behavior that nothing else exercises either: a find
module that has to survive a second configure of the same build tree, or the wiring that routes a compiler launcher into
a test's compile command.

```bash
cmake -G Ninja -S . -B build -DEIGEN_BUILD_TESTING=ON
cmake -E chdir build ctest -L buildsystem --output-on-failure --no-tests=error
```

`--no-tests=error` belongs on every `ctest` invocation in this guide, because CTest otherwise exits 0 when nothing
matched. Nothing matches with an anchored `-R '^name$'` against a split test, with a mistyped name, or with `--test-dir`
under CMake 3.17 to 3.19: those versions predate `--test-dir`, ignore it, and inspect the source directory instead.
`cmake -E chdir` is the spelling that also works there.

No target needs building first: each scenario runs its own nested configure, build, and install into the CTest
binary directory. Add a claim by dropping a scenario in `scenarios/` and naming it in the list in
`test/buildsystem/CMakeLists.txt`; the driver `run_scenario.cmake` supplies the assertion helpers.

Two hazards are specific to these tests. First, Eigen calls `export(PACKAGE Eigen3)`, so CMake's user package registry
names every Eigen build tree on the machine. A `find_package` scenario must therefore disable both the user and the
system package registries and assert that the package came from the prefix it installed; otherwise it passes without
reading that prefix at all. Second, CMake code registers these tests, so if a guard stops matching, CTest finds no tests
rather than reporting a failure. That is why the CI job runs `ctest` with `--no-tests=error`.

## Configurations The Test Suite Cannot See

- In the default host-test configuration, no test compiles an `EIGEN_NO_DEBUG` code path: `test/main.h` undefines
  `NDEBUG`, and `Macros.h` derives `EIGEN_NO_DEBUG` from it. (HIP/SYCL device compilation and an explicit
  `-DEIGEN_NO_DEBUG` define it independently.) Behavior that depends on the macro needs a dedicated `-DEIGEN_NO_DEBUG`
  test target or a standalone `-DNDEBUG` check. Conversely, an `eigen_assert` body is only type-checked where
  assertions are enabled, so it can call members its argument type does not have and still compile in every release
  build.
- Run an `EIGEN_DEFAULT_TO_ROW_MAJOR` build when layout is in play, and pin the layout explicitly where a test aliases
  one object's storage through a view whose default layout is fixed.
- Cover `EIGEN_TEST_NO_EXPLICIT_VECTORIZATION`, `EIGEN_UNALIGNED_VECTORIZE=0`, or a narrower
  `EIGEN_DEFAULT_DENSE_INDEX_TYPE` when the change reasons about packets, alignment, or index width.
- Tests build optimized (`CMAKE_BUILD_TYPE` defaults to Release) and no CI job builds Debug, so a `static constexpr`
  class-template member that is odr-used without its C++14 namespace-scope definition links in every CI build and fails
  only at -O0; see [`conventions.md`](conventions.md). Build one Debug tree when adding such constants.
- Compiler fast-math coverage is limited to targets registered with those flags in `test/CMakeLists.txt`.
  The smoke list includes `packetmath_fastmath`, `packetmath_fastmath_generic_16` where vector extensions are
  available, `bfloat16_classification_fastmath`, and parts of `fastmath`, `bdcsvd_fastmath`, and
  `stable_norm_fastmath`; these compile with `-ffast-math` where supported. Ordinary `packetmath` uses Eigen's
  `EIGEN_FAST_MATH=1` approximation switch, which does not enable the compiler flag. Add focused coverage when a
  changed path falls outside the existing fast-math tests; [`numerics.md`](numerics.md) records the special-value
  hazards.

## Numerical Assertions

`VERIFY_IS_APPROX` is a convenient broad comparison, not a machine-epsilon guarantee. `test_precision<T>()` uses
`NumTraits<T>::dummy_precision()` generically and currently specializes float to `1e-3` and double/long double to
`1e-6`. Do not use it alone to claim ULP accuracy, backward stability, or IEEE special-value conformance.

For numerical kernels, add explicit named bounds based on epsilon, dimension, conditioning, or a backward-error
model as appropriate. Check NaN, infinity, and signed zero explicitly when their distinction matters. Follow
[`numerics.md`](numerics.md) for solver, packet, and scalar-math coverage.

Write such a bound as `factor * NumTraits<RealScalar>::epsilon()` at the site, and explain `factor` by its error model.
Do not introduce tolerance wrapper helpers: the raw form is the established idiom across `test/` and `contrib/test/`,
and it shows the computation itself, so a bound like `10 * n * eps * A.norm()` reads as the formula it is. A bare
decimal literal is worse than opaque: `1e-9` demands impossible accuracy from a `float` instantiation.

Two kinds of comparison silently accept everything, and both have shipped here. The first is a tolerance computed by the
operation under test: a bound formed as `(A.cwiseAbs() * B.cwiseAbs())` goes through the product code being tested, so
accumulate it independently instead. The second is a comparison that admits non-finite values: `error <= tolerance`
holds for two infinities, and `if (error > bound)` never fires for a NaN error. Test for failure as the negation of the
passing condition, `!(error <= bound)`, and reject a non-finite tolerance.

When a numerical check fails for some seeds, find out whether the computation or the check is at fault before changing
the tolerance. Compare the results with a reference computed in higher precision (quad or MPFR). Measure the backward
error, and the forward error relative to the first-order condition bound, across enough seeds to see the tail:

- A result worse than its conditioning allows is an accuracy defect. Fix the algorithm; a wider tolerance would hide
  it.
- A result within that accuracy that still fails means the check asks for more than the working precision can
  deliver. Derive the tolerance from the conditioning rather than a flat factor.
- Solving the same inputs in the next wider type proves neither. The wider type resolves what the working precision
  cannot, such as a tight cluster of roots that the narrower type can only locate to within a wider set.

An accuracy defect and an over-strict check can both be present. In the `polynomialsolver` flake, the companion
eigenvalues had backward errors of 1.8e4 eps, yet once every root was below one eps, the flat 3.16% check kept failing
at the same rate. Land such a computation fix and test fix as independent merge requests. In the first, state whether it
removes the failure, backed by seed sweeps against the parent commit.

Run reproducible failures directly with a fixed seed and repeat count:

```bash
EIGEN_REPEAT=10 EIGEN_SEED=1 build/test/foo_3
build/test/foo_3 r10 s1
```

## External BLAS And Shim Libraries

`EIGEN_TEST_EXTERNAL_BLAS=ON` finds a system BLAS, defines `EIGEN_USE_BLAS`, and links that BLAS into applicable
official tests. With it off, ordinary tests exercise Eigen's normal implementation; they do not transparently use
the in-tree `eigen_blas` library. `EIGEN_BUILD_BLAS` and `EIGEN_BUILD_LAPACK` separately build Eigen's ABI shim
libraries, which are also used to satisfy some optional sparse-backend links. There is currently no
`EIGEN_TEST_EXTERNAL_LAPACK` option.

Report the exact targets, CTest regexes, configurations, compiler, and seeds run. Also report relevant hardware or
optional backends that were unavailable locally.
