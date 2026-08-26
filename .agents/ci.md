# Formatting And CI

Use the checked-out configuration as the source of truth. [`.gitlab-ci.yml`](../.gitlab-ci.yml) defines stages and
includes; [`ci/*.gitlab-ci.yml`](../ci) and [`ci/scripts/`](../ci/scripts) define the actual jobs. Default MR pipelines
run a limited smoke matrix; labels such as `affected-tests`, `all-tests` and `gpu-tests`, plus scheduled or manually
started pipelines, enable broader jobs, and `affected-tests` composes with the `*-tests` platform labels and
`all-platforms` to pick where it runs. A green default MR pipeline is not proof that every supported
configuration was exercised.

A pipeline is evidence only for the commit it ran on: after a push, amend, or rebase, check which SHA the pipeline and
the merge request point at before citing either — a green run on a superseded revision proves nothing about the
current head, and a reported failure should be reproduced at the current head too.

Build jobs publish the configured build directory as an artifact. Their paired test jobs consume that artifact and
run CTest without rebuilding. When changing either side, keep the test job's `needs`, CTest label or filter, and the
corresponding build target consistent; otherwise CTest can discover tests whose executables are absent.

In merge-request pipelines the Linux test jobs also keep a content-addressed pass cache (a per-job-name GitLab cache
holding `.testcache/`): [`test.linux.script.sh`](../ci/scripts/test.linux.script.sh) skips tests whose executable,
emulator, CTest definition, and environment fingerprint (image, `lib*` package state, `ci/scripts/` and
`ci/docker/`, and behavior-affecting variables such as `EIGEN_REPEAT` and `QEMU_CPU`) match a first-attempt pass
recorded by an earlier MR pipeline, then records this run's first-attempt passes — taken from the dashboard run's
`Test.xml` statuses — via [`test_cache.py`](../ci/scripts/test_cache.py). Scheduled and web pipelines always run
their full selection (fresh clock-derived RNG seeds are part of their coverage), sharded jobs never skip, and
`EIGEN_CI_TEST_CACHE: "off"` opts a job out. Skipped tests are absent from that run's JUnit report, and a test job
whose binaries all match cached passes legitimately reports "No tests were found".

The fingerprint covers `ci/scripts/` and `ci/docker/`, not the `ci/*.gitlab-ci.yml` files: everything in the YAML that
reaches a test's outcome already reaches the key by value — job variables through `KEYED_ENV_PREFIXES`, the image
through `CI_JOB_IMAGE`, compiler flags and the cross emulator through the digests of the files in the test's command,
CTest timeouts through the properties hash — so hashing the YAML as well only meant that every CI-maintenance merge
request discarded every job's manifest. Two consequences follow. **A new job variable that can change a test's outcome
must be added to `KEYED_ENV_PREFIXES`**; setting it in the YAML alone no longer keys it. And a job's `tags:` are now
invisible to the fingerprint, so moving a job to a runner pool whose CPU differs should be paired with a cache clear —
though nothing distinguished two hosts within one tag pool before this either.

## Test Tiers On Merge Requests

Three tiers, in increasing cost:

| Tier | Trigger | What runs |
|---|---|---|
| smoke | every MR with neither label below | the fixed list in [`cmake/EigenSmokeTestList.cmake`](../cmake/EigenSmokeTestList.cmake), usually one part per test, at baseline ISA on x86-64, aarch64 and riscv64, under gcc and clang |
| affected | `affected-tests` label | every test the diff can reach, all parts, on x86-64 (gcc AVX2, clang baseline) and aarch64 (gcc, clang), plus any platform the diff or a `*-tests` label selects |
| full | `all-tests` label | the whole suite across the entire compiler and ISA matrix, minus the schedule-only jobs below |

One configuration sits outside all three tiers and runs only on schedules and web pipelines: the NVHPC (`nvc++`) build
and test pair. Its frontend is slow enough that those two builds alone accounted for roughly a quarter of the project's
hosted-runner minutes while they were in the `all-tests` matrix. Start a web pipeline when a change plausibly affects
`nvc++` rather than waiting for the scheduled run to find it.

The affected tier exists because the smoke list samples: it is broad but shallow, so a change confined to one module
gets only the one part of each related test that the list happens to name. Reach for `affected-tests` when a change is
module-local and you want depth without paying for the full matrix.

The tiers do not stack: `affected-tests` and `all-tests` each suppress the smoke jobs
(`.rules:libeigen:smoketest`), because both go deeper than the fixed list on the same native runners and the smoke
jobs would only pay for it twice. The affected tier's four unconditional jobs therefore mirror the smoke matrix's
*compilers* — gcc-10 and clang-14 on x86-64 and aarch64. Two smoke *configurations* are still not reproduced, and
both need a label to get back: riscv64, whose native runner is a single scarce machine (`rvv-tests` or
`all-platforms`), and x86-64 gcc at baseline ISA, since the unconditional gcc job is AVX2 (`sse-tests` or
`all-platforms`). The suppression is scoped to the `libeigen` namespace, since neither wider tier has any job in a
fork.

[`scripts/affected_tests.py`](../scripts/affected_tests.py) computes the selection in the `select:tests` job and writes
`affected/targets.txt` and `affected/ctest_regex.txt`, which the paired build and test jobs on both Linux and Windows
consume through `EIGEN_CI_BUILD_TARGET_FILE` and `EIGEN_CI_CTEST_REGEX_FILE`. Run it locally the same way CI does:

```bash
python3 scripts/affected_tests.py --base-sha $(git merge-base origin/master HEAD)
python3 scripts/test_affected_tests.py     # unit tests, also run by checkformat:scripts
python3 ci/scripts/test_test_cache.py      # pass-cache unit tests, same job
```

`checkformat:scripts` runs both suites on every merge request and is blocking: both scripts fail closed, but a wrong
answer is silent — a job that skips too much still reports success.

Selection follows the textual `#include` graph, ignoring preprocessor guards, so it is a strict superset of the real
compile dependency and never drops an affected test. Because Eigen is header-only and the umbrella headers are hubs,
a change under `Eigen/src/Core` typically reaches every test and the selector degrades to the full suite — that is the
correct answer, not a failure. Changes to CMake, `ci/scripts/`, `ci/docker/`, or the BLAS/LAPACK shims also force the
full suite, since they invalidate the mapping itself; the `ci/*.gitlab-ci.yml` files are orchestration and cannot
change which test includes which header, so they select nothing. Git rename detection is disabled for the input diff so both the old and new path of a
move are evaluated; an old path absent from the current graph safely forces the full suite.

The selector derives source-to-target mappings from test CMake registration, including multi-translation-unit
executables and the GPU tests, whose sources are `.cu` because `ei_add_test` takes the extension from
`EIGEN_ADD_TEST_FILENAME_EXTENSION`. A changed test source without a registration is an error rather than an
unconfigured target to drop. `test/buildsystem/` is skipped: its consumers are separate CMake projects that only
`test:linux:buildsystem` configures, so an `add_executable` there is not a registration and its sources reach no
test here. Targets absent from one configuration (optional dependencies such as CHOLMOD, CUDA or
SYCL) are still filtered against `ninja -t targets` after cmake configure, because ninja aborts on an unknown target;
a selection consisting only of such targets is a no-op, not a failure. A missing selection artifact must also fail the
job rather than fall through to the default target, which would silently build everything. A `NONE` selection is read
before the toolchain setup and the configure step, so a merge request that reaches no test costs a checkout rather than
a full configure. `rules:` cannot decline to schedule that job in the first place, because GitLab evaluates them when
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

### Platform-Triggered Configurations

Every job in the default smoke matrix builds at baseline ISA, so a change under `Eigen/src/Core/arch/AVX512` gets no
AVX-512 compilation at all unless someone applies `all-tests`. Under the `affected-tests` label the tier adds
platforms beyond the four unconditional jobs on two independent triggers, either of which is enough:

- **the diff**, through `rules:changes:` on the backend directory — automatic, and the common case;
- **a label**, through `$CI_MERGE_REQUEST_LABELS` — the axis orthogonal to the include graph. The graph decides
  *which tests* run; the labels decide *where*. Use this to run the affected tests somewhere the diff does not
  point at: a `Core` change on ppc64le, a `Geometry` change on Windows.

| Backend directory | Label | Added configuration | In `all-platforms` |
|---|---|---|---|
| `arch/SSE` | `sse-tests` | x86-64 gcc-10 baseline, AVX, and AVX-512DQ | yes |
| `arch/AVX` | `avx-tests` | x86-64 gcc-10 AVX and AVX-512DQ | yes |
| `arch/AVX512` | `avx512-tests` | x86-64 gcc-10 AVX-512DQ | yes |
| `arch/AVX512/*FP16*` | `avx512-tests` | the split gcc-13 AVX512-FP16 compile builds | no |
| `arch/NEON` | `neon-tests` | 32-bit arm (aarch64 already runs unconditionally) | yes |
| `arch/AltiVec` | `altivec-tests` | ppc64le gcc-14, under qemu | yes |
| `arch/LSX` | `lsx-tests` | loongarch64 gcc-14, under qemu | yes |
| `arch/RVV10` | `rvv-tests` | riscv64 gcc-15, on the native runner | yes |
| `arch/SVE`, `arch/SME` | `sme-tests` | the full SME build, compile-only | no |
| — | `windows-tests` | MSVC 14.29 x64 baseline | yes |
| `arch/GPU`, `test/*.cu`, `test/gpu_common.h`, `unsupported/test/*.cu`, `unsupported/test/GPU/**` | `gpu-tests` | the CUDA build and test jobs | no |

Each rule set matches the whole label string on its own, so several labels select the union of their platforms —
`neon-tests` with `altivec-tests` runs 32-bit arm and ppc64le and nothing else. That is why these labels are
**unscoped**: GitLab makes scoped labels (`backend::NEON`) mutually exclusive, so a scoped axis could never express
a union, which is the point of the axis. Apart from `gpu-tests`, none of them does anything without
`affected-tests`.

`all-platforms` is a shorthand for every row that *runs the affected selection*. The three rows marked "no" are
excluded because their jobs ignore the selection and compile the whole suite instead — the AVX512-FP16 pair and the
SME build are compile-only with no paired test job, and the GPU jobs build `buildtests_gpu`. Reaching those means
naming their label, so `all-platforms` on a one-line change cannot silently buy hours of whole-suite compilation.

A wider x86 configuration compiles the narrower backends' headers, which is why SSE fans out to three builds. SVE and
SME get compile coverage rather than a selection because their per-SVL test jobs already filter to a curated target
subset through `EIGEN_CI_CTEST_REGEX`, which a selection would fight with.

Windows has no `changes:` trigger. What MSVC catches that the Linux jobs do not — template instantiation limits,
`EIGEN_STRONG_INLINE` behaviour, optimizer heap exhaustion — is whole-library rather than confined to a subtree a
diff could name, so there is nothing to key an automatic rule on and `windows-tests` is the only way in. The
selection is consumed by [`build.windows.script.ps1`](../ci/scripts/build.windows.script.ps1) and
[`test.windows.script.ps1`](../ci/scripts/test.windows.script.ps1). Only MSVC x64 at baseline ISA is wired up; the
32-bit, AVX2 and AVX-512DQ Windows configurations stay in `all-tests`.

AVX512-FP16 headers are guarded by `EIGEN_VECTORIZE_AVX512FP16`, so an AVX512DQ build does not parse them, and the
`*FP16*` row exists to compile them. Those jobs are compile-only because no current runner can execute AVX512-FP16
instructions.

The GPU row is the one entry that adds jobs outside the tier rather than an affected build and test pair, because no
affected-tier configuration enables CUDA, HIP or SYCL. In a host-only build there is no `gpu_basic`, `tensor_gpu`,
`cusolver_*` or `cudss_*` target at all, so a diff confined to the GPU test sources selects names that every affected
build reports as unconfigured and hands the test jobs a `-R` regex matching nothing: every step exits 0 and the tier
reads as green having compiled and run nothing. Those paths therefore add the existing CUDA jobs, through the
`affected-tests` entry in `.rules:libeigen:gpu`. They ignore the selection — `EIGEN_CI_BUILD_TARGET` is
`buildtests_gpu` and the test jobs filter on the `gpu` CTest label — so this is coverage of the whole GPU suite, not
of the affected subset. `gpu-tests` is the platform label for this row and already triggers those jobs on its own,
so it composes with `affected-tests` without a second rule entry.

`arch/ZVector`, `arch/MSA`, `arch/HVX` and the `arch/HIP` and `arch/SYCL` backends have no matching test
configuration, so a change there gets only the four unconditional jobs and the same hollow result; `gpu-tests` is no
help either, since the GPU jobs it gates are all CUDA. When adding a runner for one of these, add the trigger here
too.

## Worktree-Safe Formatting

Inspect `git status --short` before formatting and preserve unrelated changes. Eigen requires `clang-format-17`
exactly; the pin lives in [`ci/checkformat.gitlab-ci.yml`](../ci/checkformat.gitlab-ci.yml), which installs
`clang17-extra-tools`. Format only files owned by the task:

```bash
clang-format-17 -i path/to/file.cpp path/to/header.h
clang-format-17 --dry-run --Werror path/to/file.cpp path/to/header.h
git clang-format --binary clang-format-17 --diff <base-sha>
```

`.clang-format` intentionally disables include sorting and registers Eigen-specific macros and attributes. Do not
reorder includes or restyle those macros manually.

[`scripts/format.sh`](../scripts/format.sh) rewrites every matching file in the tree in parallel. Run it only when the
worktree is clean or every affected change is owned by the task. Review `git diff` afterward in either case.

## Local Checks

Run checks relevant to the changed files and report unavailable tools:

```bash
codespell --config setup.cfg path/to/changed-file
reuse lint
python3 scripts/check_style.py --diff <base-sha>
python3 scripts/clang_tidy_hook.py --diff <base-sha>   # needs clang-tidy
```

Both report only on the lines a change adds, and both are advisory. `check_style.py` covers the conventions
clang-tidy cannot state — comment verbosity, and the declaration forms still awaiting a `CustomChecks` query
(see the parked block in `.clang-tidy`). `clang_tidy_hook.py` runs clang-tidy itself, restricted to added lines
with `--line-filter`; it needs no build directory, generating a driver that includes the module umbrella and then
the edited `Eigen/src` header, the way `ci/scripts/run-clang-tidy.sh` does for merge requests. It skips silently when
clang-tidy is absent.

Claude Code sessions run both automatically through the hooks registered in `.claude/settings.json`.

The whole-tree codespell invocation used by CI can expose pre-existing findings. Do not modify unrelated files merely
to make a local broad scan clean. In the current CI configuration, clang-format, codespell, and clang-tidy jobs are
`allow_failure`; treat their diagnostics as review findings anyway. The REUSE job is blocking.

Source-like files normally carry an inline SPDX copyright and license header using the file type's comment syntax.
Files that should not carry inline comments need coverage in [`REUSE.toml`](../REUSE.toml). To process selected new
source files with the repository helper, pass them explicitly because its default scan considers tracked files:

```bash
python3 scripts/add_spdx_headers.py --paths path/to/new-file.cpp
```

## Documentation Builds

The documentation job is blocking and easy to miss. Unlike the clang-format, codespell, and clang-tidy jobs,
`build:linux:docs` in [`ci/build.linux.gitlab-ci.yml`](../ci/build.linux.gitlab-ci.yml) is not `allow_failure`, and
[`doc/Doxyfile.in`](../doc/Doxyfile.in) sets `WARN_AS_ERROR = FAIL_ON_WARNINGS_PRINT`, so one Doxygen warning fails it.
Its rules exclude the default merge-request pipeline: it runs on schedules, web pipelines, a merge request labeled
`all-tests`, and a push to the default branch. A malformed `\ref` therefore passes an entire review green and breaks the
pipeline on `master` after the merge. Apply the `all-tests` label to any merge request that touches Doxygen markup, a
cross-reference target, or a documented name.

The recurring authoring mistake is trailing punctuation absorbed into a cross-reference: a colon directly after
`\ref name` becomes part of the symbol Doxygen tries to resolve, so `\ref adjoint: the ...` fails while
`\ref adjoint. The ...` resolves. Separate a reference from following prose with a space, comma, or period. Punctuation
inside the name itself is fine — `\ref MatrixBase::cross()` is a qualified symbol, not a glued colon.

The `doc` target also compiles and runs the configured examples and snippets under [`doc/snippets`](../doc/snippets),
[`doc/examples`](../doc/examples), and their unsupported counterparts, by way of the `all_snippets` and `all_examples`
prerequisites in [`doc/CMakeLists.txt`](../doc/CMakeLists.txt). A renamed or removed public name breaks the
documentation build even when every comment is well formed, so search those directories before changing one.
"Configured" is the operative word: `unsupported/doc/examples/CMakeLists.txt` adds its `SYCL` subdirectory only under
`EIGEN_TEST_SYCL`, which `build:linux:docs` does not set, so a broken unsupported SYCL example leaves this target green.
Treat the target as coverage for the sets the configuration actually enables, and check the conditional before citing
it as coverage.

`EIGEN_BUILD_DOC` defaults on for a top-level, non-cross-compiling configuration, but `doc` is excluded from `all` and
must be named:

```bash
cmake --build build --target doc
```

Doxygen and graphviz must be installed. CI builds a pinned Doxygen from source
([`ci/scripts/build_and_install_doxygen.sh`](../ci/scripts/build_and_install_doxygen.sh)), so another local version can
diagnose a different set of warnings; report the version that produced a local result.

## Clang-Tidy

Use the CI driver rather than invoking clang-tidy directly on an implementation header; the driver routes such a
header through its public umbrella include.

```bash
cmake -G Ninja -S . -B .tidy-build \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DEIGEN_BUILD_TESTING=ON
ci/scripts/run-clang-tidy.sh <base-sha> .tidy-build
```

The driver examines files committed between `<base-sha>` and `HEAD`; uncommitted-only edits are not included. Eigen's
`.clang-tidy` policy is authoritative. Do not apply generic `modernize-*` or `cppcoreguidelines-*` campaigns.

A module that reaches a third-party header the machine does not install — `<cuda_runtime.h>` from
`unsupported/Eigen/src/GPU`, `<cholmod.h>` from `CholmodSupport` — is still checked, but clang parses a truncated
translation unit, so the driver marks the heading `— partial: <header> is not installed` and reports that file's
findings without failing the job. Installing the dependency gets the module checked in full; for CUDA the driver
looks under `CUDAToolkit_ROOT`, `CUDA_HOME`, `CUDA_PATH`, then `/usr/local/cuda`, and so does
`clang_tidy_hook.py`. An unresolved *in-tree* include is a defect in the change and stays a hard error.

A header under `arch/<ISA>/` other than `arch/Default/` is not forced into the driver: it parses only under the
`-march`/`-mcpu` that selects it, which this job does not pass. Such a header is linted only when the host target
selects the backend — SSE2 on the x86-64 runner — and the heading says which backend went unchecked. Validate a
change to one with a build that enables the ISA rather than relying on this job.

For a source in the compilation database the driver narrows that database first, through
[`tidy_compile_db.py`](../scripts/tidy_compile_db.py). A split test contributes one entry per `EIGEN_TEST_PART`, and
clang-tidy parses the file once per entry naming it — 41 times for `test/array_cwise.cpp` — which alone exhausts the
job's timeout. The reduction keeps one entry per distinct compiler configuration and, within a configuration split
into parts, the parts that actually compile the added lines: a line inside a `CALL_SUBTEST_<n>(...)` or an
`#if defined(EIGEN_TEST_PART_<n>)` guard needs part `<n>`, anything else needs no particular part. What that leaves
out is printed beside the file name, so a capped run names the parts it did not check rather than reporting the file
clean.

## Before Review

1. Inspect `git diff` and `git diff --check`.
2. Format the exact changed source files with clang-format-17.
3. Run the focused builds and tests documented in [`testing.md`](testing.md).
4. Run applicable spelling, REUSE, and clang-tidy checks.
5. Build the `doc` target when the change touches Doxygen markup, a documented name, or a snippet, and label the merge
   request `all-tests` so the blocking documentation job runs before the merge rather than after it.
6. State what ran, what did not run, and why. Do not claim coverage from jobs or hardware that were unavailable.
