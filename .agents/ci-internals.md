# CI Internals

Use this guide when changing [`.gitlab-ci.yml`](../.gitlab-ci.yml), the job definitions under [`ci/`](../ci), the
test selector [`scripts/affected_tests.py`](../scripts/affected_tests.py), the pass cache
[`ci/scripts/test_cache.py`](../ci/scripts/test_cache.py), or the clang-tidy driver. [`ci.md`](ci.md) is the
consumer's view of the same machinery; the checked-out files are authoritative where the two disagree.

Both selector scripts fail closed, but a wrong answer is silent — a job that skips too much still reports success — so
their unit tests are blocking and run on every merge request in `checkformat:scripts`:

```bash
python3 scripts/test_affected_tests.py
python3 ci/scripts/test_test_cache.py
```

## Artifacts And Test Reports

Build jobs publish the configured build directory as an artifact. Their paired test jobs consume that artifact and
run CTest without rebuilding. When changing either side, keep the test job's `needs`, CTest label or filter, and the
corresponding build target consistent; otherwise CTest can discover tests whose executables are absent.

Publishing is opt-in per job rather than inherited: `.common:linux:cross` and `.common:windows` carry no `artifacts:`
key, and a job picks up `.artifacts:linux:builddir`, `.artifacts:windows:builddir` or `.artifacts:test:results` as a
second `extends:` parent. A test job takes the results template — it links nothing, so re-publishing the build
directory it just downloaded would only duplicate the build job's artifact — and that template also registers
`JUnitTestResults_*.xml` through `artifacts:reports:junit:`, which is what puts failures in the job's Tests tab and
the merge request widget rather than only in the log. A job that needs neither, such as `test:linux:buildsystem`,
extends the base alone and publishes nothing.

A job that failed and then passed on retry still reports the failed first attempt because the retry runs
`--rerun-failed` without `-T test`, so it never rewrites the dashboard `Test.xml` the report is converted from; the
job exits 42 to mark the soft failure. The widget's comparison needs a base-branch report for the same job name, and
default-branch pushes run only a small subset of jobs, so most jobs show a summary without one.

## The Pass Cache

In merge-request pipelines the Linux test jobs keep a content-addressed pass cache (a per-job-name GitLab cache
holding `.testcache/`): [`test.linux.script.sh`](../ci/scripts/test.linux.script.sh) skips tests whose executable,
emulator, CTest definition, and environment fingerprint (image, `lib*` package state, `ci/scripts/` and
`ci/docker/`, and behavior-affecting variables such as `EIGEN_REPEAT` and `QEMU_CPU`) match a first-attempt pass
recorded by an earlier MR pipeline, then records this run's first-attempt passes — taken from the dashboard run's
`Test.xml` statuses — via [`test_cache.py`](../ci/scripts/test_cache.py). Scheduled and web pipelines always run
their full selection (fresh clock-derived RNG seeds are part of their coverage), sharded jobs never skip, and
`EIGEN_CI_TEST_CACHE: "off"` opts a job out. Skipped tests are absent from that run's JUnit report.

The fingerprint covers `ci/scripts/` and `ci/docker/`, not the `ci/*.gitlab-ci.yml` files: everything in the YAML that
reaches a test's outcome already reaches the key by value — job variables through `KEYED_ENV_PREFIXES`, the image
through `CI_JOB_IMAGE`, compiler flags and the cross emulator through the digests of the files in the test's command,
CTest timeouts through the properties hash — so hashing the YAML as well only meant that every CI-maintenance merge
request discarded every job's manifest. Two consequences follow. **A new job variable that can change a test's outcome
must be added to `KEYED_ENV_PREFIXES`**; setting it in the YAML alone no longer keys it. And a job's `tags:` are
invisible to the fingerprint, so moving a job to a runner pool whose CPU differs should be paired with a cache clear —
though nothing distinguished two hosts within one tag pool before this either.

## Tier Rules

`affected-tests` and `all-tests` each suppress the smoke jobs (`.rules:libeigen:smoketest`), because both go deeper
than the fixed list on the same native runners and the smoke jobs would only pay for it twice. The suppression is
scoped to the `libeigen` namespace, since neither wider tier has any job in a fork. The NVHPC pair sits behind
`.rules:libeigen:scheduled-or-web`: its frontend accounted for roughly a quarter of the project's hosted-runner
minutes while it was in the `all-tests` matrix.

Under `affected-tests`, platforms beyond the four unconditional jobs are added on two independent triggers,
`rules:changes:` on the backend directory and `$CI_MERGE_REQUEST_LABELS`, as two entries per rule set because GitLab
ANDs `if:` with `changes:` within one entry. Each rule set matches the whole label string on its own, so several
labels select the union of their platforms. That is why the `*-tests` labels are **unscoped**: GitLab makes scoped
labels (`backend::NEON`) mutually exclusive, so a scoped axis could never express a union, which is the point of the
axis. Both `arch/SVE` and `arch/SME` also list `Eigen/src/Core/util/ConfigureVectorization.h` in `changes:`, since
that header decides whether either backend is compiled at all. SVE runs one build per vector length because
`EIGEN_ARM64_SVE_VL` comes from `__ARM_FEATURE_SVE_BITS`, which only `-msve-vector-bits` sets; `test/sve_vector_length`
reads `RDVL` so a binary run at another width fails instead of computing wrong answers.

The three `all-platforms` exclusions are the rows whose jobs ignore the selection: the AVX512-FP16 pair and the SME
build are compile-only with no paired test job, and the GPU jobs build `buildtests_gpu`. SME gets compile coverage
rather than a selection because its per-SVL test jobs already filter to a curated target subset through
`EIGEN_CI_CTEST_REGEX`, which a selection would fight with. The GPU row adds jobs outside the tier through the
`affected-tests` entry in `.rules:libeigen:gpu`; `gpu-tests` already triggers those jobs on its own, so it composes
with `affected-tests` without a second rule entry. When adding a runner for a backend that has none — ZVector, MSA,
HVX, HIP, SYCL — add its trigger to these rule sets too.

## The Selector

`select:tests` writes `affected/targets.txt` and `affected/ctest_regex.txt`, which the paired build and test jobs on
both Linux and Windows consume through `EIGEN_CI_BUILD_TARGET_FILE` and `EIGEN_CI_CTEST_REGEX_FILE`
([`build.windows.script.ps1`](../ci/scripts/build.windows.script.ps1) and
[`test.windows.script.ps1`](../ci/scripts/test.windows.script.ps1) on Windows).

Selection follows the textual `#include` graph, ignoring preprocessor guards, so it is a strict superset of the real
compile dependency. Changes to CMake, `ci/scripts/`, `ci/docker/`, or the BLAS/LAPACK shims force the full suite,
since they invalidate the mapping itself; the `ci/*.gitlab-ci.yml` files are orchestration and cannot change which
test includes which header, so they select nothing. Git rename detection is disabled for the input diff so both the
old and new path of a move are evaluated; an old path absent from the current graph safely forces the full suite.

The selector derives source-to-target mappings from test CMake registration, including multi-translation-unit
executables and the GPU tests, whose sources are `.cu` because `ei_add_test` takes the extension from
`EIGEN_ADD_TEST_FILENAME_EXTENSION`. A changed test source without a registration is an error rather than an
unconfigured target to drop. `test/buildsystem/` is skipped: its consumers are separate CMake projects that only
`test:linux:buildsystem` configures, so an `add_executable` there is not a registration and its sources reach no
test here. Targets absent from one configuration (optional dependencies such as CHOLMOD, CUDA or SYCL) are still
filtered against `ninja -t targets` after cmake configure, because ninja aborts on an unknown target; a selection
consisting only of such targets is a no-op, not a failure. A missing selection artifact must also fail the job rather
than fall through to the default target, which would silently build everything. A `NONE` selection is read before
the toolchain setup and the configure step, so a merge request that reaches no test costs a checkout rather than a
full configure. `rules:` cannot decline to schedule that job in the first place, because GitLab evaluates them when
the pipeline is created, before `select:tests` has run; only a child pipeline generated from the selection could.

The build script expands the surviving selection through ninja's phony edges before it shuffles and batches. Most
selected names are aggregates — `buildtests`, and the parent of every split test — and the batch loop can only spread
apart what it is handed, so an unexpanded parent would put a whole test family in one batch and undo the
memory-pressure protection the batching exists for.

Two registrations do not reduce to a build target. `buildtests` aggregates the `ei_add_test` targets only, so a bare
`add_executable` such as the `bug1213` link regression is named explicitly alongside `buildtests` in the full-suite
mode. The compile-failure suite under `failtest/` is `EXCLUDE_FROM_ALL` and each of its CTest tests builds its own
target as the test action, so those are selected as `<name>_ok` and `<name>_ko` CTest names and never handed to the
build job. Both matter because a `-R` filter silently drops whatever it does not name, while the unfiltered runs in
the other tiers pick them up for free.

Because that test action is a build in the shared binary directory, `ei_add_failtest` puts the whole suite behind one
`RESOURCE_LOCK`. Without it, `ctest --parallel` starts dozens of concurrent builds over one build system and they
collide whenever a regeneration is pending. The failure is not only noisy: `_ko` is `WILL_FAIL`, so a build system
that errors for an unrelated reason satisfies it just as well as the compile error it is supposed to assert.

The RISC-V affected tier runs the `failtest` label on an amd64 job with the original cross compiler. Its native
runtime job excludes those compile tests and the nested `buildsystem` scenarios: the runtime image has neither Ninja
nor a compiler, and the cached compiler paths name amd64 executables. The separate `test:linux:buildsystem` job covers
the nested consumers when build-system files change.

## Clang-Tidy Compilation Database

For a source in the compilation database the driver narrows that database first, through
[`tidy_compile_db.py`](../scripts/tidy_compile_db.py). A split test contributes one entry per `EIGEN_TEST_PART`, and
clang-tidy parses the file once per entry naming it — 41 times for `test/array_cwise.cpp` — which alone exhausts the
job's timeout. The reduction keeps one entry per distinct compiler configuration and, within a configuration split
into parts, the parts that actually compile the added lines: a line inside a `CALL_SUBTEST_<n>(...)` or an
`#if defined(EIGEN_TEST_PART_<n>)` guard needs part `<n>`, anything else needs no particular part. What that leaves
out is printed beside the file name, so a capped run names the parts it did not check rather than reporting the file
clean.
