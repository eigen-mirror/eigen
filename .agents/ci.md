# Formatting And CI

Use the checked-out configuration as the source of truth. [`.gitlab-ci.yml`](../.gitlab-ci.yml) defines stages and
includes; [`ci/*.gitlab-ci.yml`](../ci) and [`ci/scripts/`](../ci/scripts) define the actual jobs. This guide covers
what a change needs from CI and how to run the same checks locally; how the selection, pass cache, and artifact
plumbing work inside the jobs is in [`ci-internals.md`](ci-internals.md), and the blocking documentation job is in
[`docs.md`](docs.md).

Default MR pipelines run a limited smoke matrix; labels such as `affected-tests`, `all-tests` and `gpu-tests`, plus
scheduled or manually started pipelines, enable broader jobs, and `affected-tests` composes with the `*-tests` platform
labels and `all-platforms` to pick where it runs. A green default MR pipeline is not proof that every supported
configuration was exercised.

A pipeline is evidence only for the commit it ran on: after a push, amend, or rebase, check which SHA the pipeline and
the merge request point at before citing either — a green run on a superseded revision proves nothing about the
current head, and a reported failure should be reproduced at the current head too.

Three things to know when reading a test report. A job that failed and then passed on retry still reports the failed
first attempt, exits 42, and is a soft warning: a green pipeline showing a failed test is the flake being surfaced, not
a regression. The widget's *comparison* needs a base-branch report for the same job name, and default-branch pushes run
only a small subset of jobs, so most jobs show a summary without one. And in merge-request pipelines the Linux test
jobs skip tests whose binary and environment match a first-attempt pass recorded by an earlier MR pipeline, so a test
count that falls between pipelines, or a test job reporting "No tests were found", is expected rather than a
regression; scheduled and web pipelines never skip, and `EIGEN_CI_TEST_CACHE: "off"` opts a job out.

## Test Tiers On Merge Requests

Three tiers, in increasing cost:

| Tier | Trigger | What runs |
|---|---|---|
| smoke | every MR with neither label below | the fixed list in [`cmake/EigenSmokeTestList.cmake`](../cmake/EigenSmokeTestList.cmake), usually one part per test, at baseline ISA on x86-64, aarch64 and riscv64, under gcc and clang |
| affected | `affected-tests` label | every test the diff can reach, all parts, on x86-64 (gcc AVX2, clang baseline) and aarch64 (gcc, clang), plus any platform the diff or a `*-tests` label selects |
| full | `all-tests` label | the whole suite across the entire compiler and ISA matrix, minus the schedule-only jobs below |

One configuration sits outside all three tiers and runs only on schedules and web pipelines: the NVHPC (`nvc++`) build
and test pair, whose frontend is slow enough that those two builds alone once took roughly a quarter of the project's
hosted-runner minutes. Start a web pipeline when a change plausibly affects `nvc++` rather than waiting for the
scheduled run to find it.

The affected tier exists because the smoke list samples: it is broad but shallow, so a change confined to one module
gets only the one part of each related test that the list happens to name. Reach for `affected-tests` when a change is
module-local and you want depth without paying for the full matrix.

The tiers do not stack: `affected-tests` and `all-tests` each suppress the smoke jobs, and the affected tier's four
unconditional jobs use the smoke compilers, gcc-10 and clang-14 on x86-64 and aarch64. Two smoke configurations come
back only with a label: riscv64 (`rvv-tests` or `all-platforms`), and x86-64 gcc at baseline ISA (`sse-tests` or
`all-platforms`), since the unconditional gcc job is AVX2.

[`scripts/affected_tests.py`](../scripts/affected_tests.py) computes the selection in the `select:tests` job. Run it
locally the same way CI does:

```bash
python3 scripts/affected_tests.py --base-sha $(git merge-base origin/master HEAD)
```

Selection follows the textual `#include` graph, ignoring preprocessor guards, so it is a strict superset of the real
compile dependency and never drops an affected test. Because Eigen is header-only and the umbrella headers are hubs,
a change under `Eigen/src/Core` typically reaches every test and the selector degrades to the full suite — that is the
correct answer, not a failure. Changes to CMake, `ci/scripts/`, `ci/docker/`, or the BLAS/LAPACK shims also force the
full suite; the `ci/*.gitlab-ci.yml` files select nothing.

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
| `arch/SVE` | `sve-tests` | SVE cross builds and test runs at 128, 256 and 512 bits under qemu | yes |
| `arch/SME` | `sme-tests` | the full SME build, compile-only | no |
| — | `windows-tests` | MSVC 14.29 x64 baseline | yes |
| `arch/GPU`, `test/*.cu`, `test/gpu_common.h`, `unsupported/test/*.cu`, `unsupported/test/GPU/**` | `gpu-tests` | the CUDA build and test jobs | no |

Several labels select the union of their platforms — `neon-tests` with `altivec-tests` runs 32-bit arm and ppc64le and
nothing else. Apart from `gpu-tests`, none of them does anything without `affected-tests`. `all-platforms` is a
shorthand for every row that *runs the affected selection*; the three rows marked "no" ignore the selection and
compile the whole suite, so reaching them means naming their label, and `all-platforms` on a one-line change cannot
silently buy hours of whole-suite compilation.

Rows worth knowing before relying on them:

- A wider x86 configuration compiles the narrower backends' headers, which is why SSE fans out to three builds.
  AVX512-FP16 headers are guarded by `EIGEN_VECTORIZE_AVX512FP16`, so an AVX-512DQ build does not parse them, and the
  `*FP16*` row is compile-only because no current runner can execute those instructions.
- SVE is a fixed-length backend, so each vector length is a separate build with different fold counts and transpose
  networks, and `test/sve_vector_length` fails the run when a binary meets a different length than it was built for —
  a mismatch otherwise computes wrong answers while the suite passes.
- Windows has no `changes:` trigger: what MSVC catches — template instantiation limits, `EIGEN_STRONG_INLINE`
  behaviour, optimizer heap exhaustion — is whole-library, so `windows-tests` is the only way in, and only MSVC x64 at
  baseline ISA is wired up. The 32-bit, AVX2 and AVX-512DQ Windows configurations stay in `all-tests`.
- No affected-tier configuration enables CUDA, HIP or SYCL, so a diff confined to GPU test sources would select targets
  no host build has and read green having run nothing. Those paths add the existing CUDA jobs instead, which build
  `buildtests_gpu` and run the whole `gpu` label: coverage of the GPU suite, not of the affected subset.
- `arch/ZVector`, `arch/MSA`, `arch/HVX` and the `arch/SYCL` backend have no matching test configuration, so a change
  there gets only the four unconditional jobs and the same hollow result; `gpu-tests` is no help either, since the GPU
  jobs it gates are all CUDA or ROCm.

## Worktree-Safe Formatting

Inspect `git status --short` before formatting and preserve unrelated changes. Eigen requires `clang-format-17`
exactly; the pin lives in [`ci/checkformat.gitlab-ci.yml`](../ci/checkformat.gitlab-ci.yml), which installs
`clang17-extra-tools`. CI checks only the lines a merge request changes, and the tree is not uniformly
clang-format-17 clean (a whole-file pass rewrites `> >` closers in a couple of dozen headers), so format the diff:

```bash
git clang-format --binary clang-format-17 --force <base-sha> -- path/to/file.cpp path/to/header.h
git clang-format --binary clang-format-17 --diff <base-sha> -- path/to/file.cpp path/to/header.h
clang-format-17 -i path/to/new-file.h
clang-format-17 --dry-run --Werror path/to/new-file.h
```

Inspect the selected files' diffs first: every change being formatted must belong to the task. `--force` permits
unstaged edits; without it, files that need formatting must be staged or committed first. Untracked files are absent
from the Git diff, so the whole-file commands above cover task-created files. `git clang-format` exits 1 when it makes
or reports formatting changes; rerun the `--diff` check after applying them.

`.clang-format` intentionally disables include sorting and registers Eigen-specific macros and attributes. Do not
reorder includes or restyle those macros manually.

[`scripts/format.sh`](../scripts/format.sh) rewrites every matching file in the tree in parallel. Run it only when the
worktree is clean and a whole-tree pass is intentional. Review `git diff` afterward in either case.

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
clang-tidy is absent, and shows the user a non-blocking notice when a file's translation unit does not compile.

Claude Code sessions run both automatically through the hooks registered in `.claude/settings.json`. Their unit
tests, [`scripts/test_check_style.py`](../scripts/test_check_style.py) and
[`scripts/test_clang_tidy_hook.py`](../scripts/test_clang_tidy_hook.py), run in `checkformat:scripts`; run them after
changing either script.

The whole-tree codespell invocation used by CI can expose pre-existing findings. Do not modify unrelated files merely
to make a local broad scan clean. In the current CI configuration, clang-format, codespell, and clang-tidy jobs are
`allow_failure`; treat their diagnostics as review findings anyway. The REUSE job is blocking.

Source files carry the inline SPDX header [`conventions.md`](conventions.md) records; files that cannot need coverage
in [`REUSE.toml`](../REUSE.toml). To stamp selected new files with the repository helper, pass them explicitly because
its default scan considers tracked files:

```bash
python3 scripts/add_spdx_headers.py --paths path/to/new-file.cpp
```

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

For a split test the driver checks only the parts that compile the added lines and prints beside the file name the
parts it left out, so a capped run names what it did not check rather than reporting the file clean;
[`ci-internals.md`](ci-internals.md) has the reduction.

## Before Review

1. Inspect `git diff` and `git diff --check`.
2. Format and check the task's changed lines and new files using the Worktree-Safe Formatting recipes above.
3. Run the focused builds and tests documented in [`testing.md`](testing.md).
4. Run applicable spelling, REUSE, and clang-tidy checks.
5. Build the `doc` target when the change touches Doxygen markup, a documented name, or a snippet, and label the merge
   request `all-tests` so the blocking documentation job ([`docs.md`](docs.md)) runs before the merge rather than after
   it.
6. State what ran, what did not run, and why. Do not claim coverage from jobs or hardware that were unavailable.
