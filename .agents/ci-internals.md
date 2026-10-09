# CI Internals

Use this guide when changing [`.gitlab-ci.yml`](../.gitlab-ci.yml), the job definitions under [`ci/`](../ci), the
test selector [`scripts/affected_tests.py`](../scripts/affected_tests.py), the pass cache
[`ci/scripts/test_cache.py`](../ci/scripts/test_cache.py), or the clang-tidy driver. [`ci.md`](ci.md) is the
consumer's view of the same machinery; the checked-out files are authoritative where the two disagree.

Both selector scripts, `affected_tests.py` and `test_cache.py`, fail closed: an error or an unknown case either fails
the job or widens the selection, never silently narrows it. A wrong answer, however, goes unnoticed: a job that skips
too many tests still reports success. Their unit tests are therefore blocking and run on every merge request in
`checkformat:lint`:

```bash
python3 scripts/test_affected_tests.py
python3 ci/scripts/test_test_cache.py
```

## Artifacts And Test Reports

Build jobs publish the configured build directory as an artifact. Their paired test jobs consume that artifact and
run CTest without rebuilding. When changing either side, keep the test job's `needs`, CTest label or filter, and the
corresponding build target consistent; otherwise CTest can discover tests whose executables are absent.

Publishing is opt-in per job rather than inherited: the bases `.common:linux:cross` and `.common:windows` have no
`artifacts:` key. To make a job publish, add `.artifacts:linux:builddir`, `.artifacts:windows:builddir` or
`.artifacts:test:results` as a second `extends:` parent. Give a test job the results template, not a build-directory
template. A test job links nothing, and the build directory it downloaded is already the build job's artifact. The
results template also registers `JUnitTestResults_*.xml` through `artifacts:reports:junit:`. That registration puts
failures in the job's Tests tab and in the merge request widget, not only in the log. A job that needs neither, such as
`test:linux:buildsystem`, extends the base alone and publishes nothing.

A job that failed and then passed on retry still reports the failed first attempt. The report is converted from the
dashboard file `Test.xml`, and the retry runs `--rerun-failed` without `-T test`, so it never rewrites that file. The
job exits 42 to mark the soft failure. The merge request widget can compare results only against a base-branch report
from a job with the same name. Default-branch pushes run only a small subset of jobs, so most jobs show a summary
without a comparison.

## The Pass Cache

In merge request pipelines, the Linux test jobs keep a content-addressed pass cache in `.testcache/`, with one GitLab
cache per job name. [`test.linux.script.sh`](../ci/scripts/test.linux.script.sh) skips a test when an earlier merge
request pipeline recorded a pass on the first attempt for the same executable, emulator, CTest definition, and
environment fingerprint. The fingerprint covers the image, the `lib*` package state, `ci/scripts/` and `ci/docker/`,
and variables that change behavior, such as `EIGEN_REPEAT` and `QEMU_CPU`. After the run, the script calls
[`test_cache.py`](../ci/scripts/test_cache.py) to record the tests that passed on the first attempt. It reads their
statuses from the dashboard run's `Test.xml`.

Scheduled and web pipelines always run every test their job selects. Each of their runs draws fresh random seeds from
the clock, and those seeds are part of what they test. Sharded jobs never skip, and `EIGEN_CI_TEST_CACHE: "off"` opts a
job out. Skipped tests are absent from that run's JUnit report.

The fingerprint hashes `ci/scripts/` and `ci/docker/` but not the `ci/*.gitlab-ci.yml` files. Every YAML setting that
can change a test's outcome already enters the key by value:

- job variables through `KEYED_ENV_PREFIXES`;
- the image through `CI_JOB_IMAGE`;
- compiler flags and the cross emulator through the digests of the files in the test's command;
- CTest timeouts through the properties hash.

If the key included the YAML files, any edit to them would discard all recorded passes. Leaving them out has two
consequences:

- **When you add a job variable that can change a test's outcome, add it to `KEYED_ENV_PREFIXES`.** Setting it in the
  YAML alone does not put it in the key.
- When you move a job to a runner pool whose CPU differs, you should also clear the cache, because the fingerprint does
  not see a job's `tags:`. The key cannot tell two hosts in the same tag pool apart either.

## Tier Rules

The `affected-tests` and `all-tests` labels each turn off the smoke jobs (`.rules:libeigen:smoketest`). Both tiers test
more deeply than the fixed smoke list, on the same native runners, so the smoke jobs would add cost but no coverage. The
smoke jobs are turned off only in the `libeigen` namespace, because the affected and full tiers have no jobs in a fork.
In merge request pipelines, `.rules:libeigen:nvhpc` runs the NVHPC build and test pair only when the `nvhpc-tests`
label is set. When the pair was part of the `all-tests` matrix, its compiler frontend used roughly a quarter of the
project's hosted-runner minutes.

Under `affected-tests`, either of two independent triggers adds a platform beyond the four unconditional jobs: a change
under the platform's backend directory, detected through `rules:changes:`, or a label, read from
`$CI_MERGE_REQUEST_LABELS`. GitLab ANDs `if:` with `changes:` within one rule entry, so each rule set has two entries,
one per trigger. Each rule set tests the whole label string on its own, so several labels select the union of their
platforms. The `*-tests` labels are **unscoped** so that a merge request can carry several of them. GitLab makes scoped
labels (`backend::NEON`) mutually exclusive, so it would allow only one.

The rule sets for `arch/SVE` and `arch/SME` also list `Eigen/src/Core/util/ConfigureVectorization.h` in `changes:`,
because that header decides whether either backend is compiled at all. SVE runs one build per vector length. Eigen takes
`EIGEN_ARM64_SVE_VL` from `__ARM_FEATURE_SVE_BITS`, and only `-msve-vector-bits` sets that macro.
`test/sve_vector_length` reads `RDVL`, so a binary run at another width fails instead of computing wrong answers.

`all-platforms` leaves out three rows of the platform table in `ci.md` because their jobs ignore the selection (the list
of affected tests that `select:tests` computes). The AVX512-FP16 pair and the SME build are compile-only, with no paired
test job, and the GPU jobs build `buildtests_gpu`. SME gets compile coverage rather than the selection. Its test jobs,
one per streaming vector length (SVL), already run a curated subset of targets through `EIGEN_CI_CTEST_REGEX`, and the
selection would conflict with that subset. The GPU row adds jobs from outside the tier through the `affected-tests`
entry in `.rules:libeigen:gpu`. `gpu-tests` already triggers those jobs on its own, so it combines with `affected-tests`
without a second rule entry. When adding a runner for a backend that has none (ZVector, MSA, HVX, HIP, SYCL), add its
trigger to these rule sets too.

## The Selector

`select:tests` writes `affected/targets.txt` and `affected/ctest_regex.txt`. The paired build and test jobs on Linux and
Windows read them through `EIGEN_CI_BUILD_TARGET_FILE` and `EIGEN_CI_CTEST_REGEX_FILE`. On Windows, the readers are
[`build.windows.script.ps1`](../ci/scripts/build.windows.script.ps1) and
[`test.windows.script.ps1`](../ci/scripts/test.windows.script.ps1).

The selector follows the textual `#include` graph and ignores preprocessor guards, so it selects a strict superset of
the tests that actually compile a changed file. Changes to CMake, `ci/scripts/`, `ci/docker/`, or the BLAS/LAPACK shims
force the full suite, because they invalidate the mapping itself. The `ci/*.gitlab-ci.yml` files only orchestrate jobs
and cannot change which test includes which header, so they select nothing. Neither do the clang-tidy and lint images
under `ci/tidy/` and `ci/lint/`. Git rename detection is disabled for the input diff, so the selector sees both the old
and the new path of a moved file. An old path absent from the current graph safely forces the full suite.

The selector maps sources to targets by reading the test registrations in the CMake files. The registrations include
executables built from several translation units, and the GPU tests. GPU test sources are `.cu`, because `ei_add_test`
takes the extension from `EIGEN_ADD_TEST_FILENAME_EXTENSION`. The selector treats a changed test source without a
registration as an error; it does not drop the source as an unconfigured target. The selector skips
`test/buildsystem/`. The consumer projects there are separate CMake projects, and only `test:linux:buildsystem`
configures them. An `add_executable` in them is not a test registration, and no test in the main project depends on
their sources.

Some targets need an optional dependency such as CHOLMOD, CUDA or SYCL and are absent from configurations without it.
The build script filters the selection against `ninja -t targets` after the CMake configure, because ninja aborts on an
unknown target. A selection of only such targets builds nothing and is not a failure. A missing selection artifact,
however, must fail the job rather than fall back to the default target, which would silently build everything. A job
reads a `NONE` selection before the toolchain setup and the configure step, so a merge request that affects no test
costs a checkout rather than a full configure. `rules:` cannot keep such a job out of the pipeline, because GitLab
evaluates them when it creates the pipeline, before `select:tests` runs. Only a child pipeline generated from the
selection could skip the job entirely.

The build script shuffles the remaining targets and builds them in batches. Before that, it expands each target through
ninja's phony edges into the targets it aggregates. Most selected names are aggregates: `buildtests`, and the parent
target of every split test, which builds all of that test's parts. Batching limits memory pressure by spreading the
targets it is given across batches. An unexpanded parent target would put a whole test family into one batch and
defeat that limit.

Two kinds of registration need special handling. First, `buildtests` aggregates only the `ei_add_test` targets, so in
full-suite mode the selector names a bare `add_executable`, such as the `bug1213` link regression, explicitly next to
`buildtests`. Second, the compile-failure suite under `failtest/` is `EXCLUDE_FROM_ALL` and is compiled at test time, so
the selector picks those tests by their CTest names, `<name>_ok` and `<name>_ko`, and never hands them to the build job.
Both matter because a `-R` filter silently drops whatever it does not name. The unfiltered runs in the other tiers
include them automatically.

The `buildfailtests` fixture compiles the failtests. Every `_ok` and `_ko` test requires that fixture, so CTest adds it
to any run that selects one of them, even if `-E` excludes it. The fixture builds every failtest target in one
`cmake --build` that keeps going after errors. Separate builds, one per test, would have to run serially: concurrent
builds in one binary directory collide whenever a CMake regeneration is pending. On the hosted runners, that serial
suite took 80-95% of an affected test job's wall time.

The tests themselves only check which executables exist. `_ok` passes when its own executable exists. `_ko` passes when
its own executable is missing and its `_ok` twin's exists. A missing compiler therefore fails both tests instead of
making `_ko` pass. When a job excludes the failtests by name from an "ALL" selection, it must exclude
`^buildfailtests$` as well, because an "ALL" selection applies no `-R`.

The RISC-V affected tier runs the `failtest` label on an amd64 job with the original cross compiler. Its native
runtime job excludes those compile tests and the nested `buildsystem` scenarios: the runtime image has neither Ninja
nor a compiler, and the cached compiler paths name amd64 executables. The separate `test:linux:buildsystem` job covers
the nested consumers when build-system files change.

## Clang-Tidy Compilation Database

For a source in the compilation database, the driver first narrows the database with
[`tidy_compile_db.py`](../scripts/tidy_compile_db.py). A split test has one database entry per `EIGEN_TEST_PART`, or per
range of parts that `cmake/EigenTestPartGroups.cmake` compiles together. clang-tidy parses the file once for each entry
that names it: 14 times for `test/array_cwise.cpp`, which alone uses up the job's timeout. The narrowed database keeps
one entry per distinct compiler configuration. Within a configuration that is split into parts, it keeps the entries
whose parts actually compile the added lines. A line inside a `CALL_SUBTEST_<n>(...)` call or an
`#if defined(EIGEN_TEST_PART_<n>)` guard needs part `<n>`; any other line can be checked under any part. The driver
prints what it left out next to the file name, so a run that hits the cap names the parts it did not check instead of
reporting the file clean.
