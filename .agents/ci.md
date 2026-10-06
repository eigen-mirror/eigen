# Formatting And CI

Use the checked-out configuration as the source of truth. [`.gitlab-ci.yml`](../.gitlab-ci.yml) defines stages and
includes; [`ci/*.gitlab-ci.yml`](../ci) and [`ci/scripts/`](../ci/scripts) define the actual jobs. This guide covers
what a change needs from CI and how to run the same checks locally. [`ci-internals.md`](ci-internals.md) explains how
the jobs work inside: test selection, the pass cache, and artifact handling. [`docs.md`](docs.md) covers the blocking
documentation job.

Default MR pipelines run a limited smoke matrix. Recommend `affected-tests` with the relevant `*-tests` platform labels,
or `affected-tests` with `all-platforms` when the change needs coverage on every platform that runs the affected tests.
`affected-tests` runs every test the diff can reach (the *affected tests*). GPU, SME, and AVX512-FP16 coverage needs
additional labels, listed in the platform table below. The `docs-build` label runs the blocking documentation job, which
no default MR pipeline runs. Do not add `all-tests` without the user's explicit permission for that label. Permission to
push, rebase, address review, or validate an MR does not authorize it. A green default MR pipeline is not proof that
every supported configuration was exercised.

A pipeline is evidence only for the commit it ran on. After a push, amend, or rebase, check which SHA the pipeline and
the merge request point at before citing either. A green run on a superseded revision proves nothing about the current
head, and a reported failure should be reproduced at the current head too.

Three things to know when reading a test report. First, a job that failed and then passed on retry still reports the
failed first attempt and exits 42, which shows as a soft warning. A green pipeline that shows a failed test is reporting
a flaky test, not a regression. Second, the merge request widget can show a *comparison* only if the base branch has a
report for the same job name. Default-branch pushes run only a small subset of jobs, so most jobs show a summary with no
comparison. Third, in merge-request pipelines the Linux test jobs skip a test when its binary and environment match a
first-attempt pass recorded by an earlier MR pipeline. A test count that falls between pipelines, or a test job that
reports "No tests were found", is therefore expected rather than a regression. Scheduled and web pipelines never skip
tests, and setting `EIGEN_CI_TEST_CACHE: "off"` in a job opts it out.

Build jobs keep a second GitLab cache: the ccache pool, which holds `.ccache/`. Its key is
`$EIGEN_CI_CCACHE_POOL$EIGEN_CI_CCACHE_SCOPE-ccache`. The pool is the job's own slug. The exception is a job that
compiles a subset of another job's targets with the same flags (the `:affected` tier), which uses its parent's slug. The
scope is `-mr<iid>` in a merge request pipeline, `-<ref slug>` in other pipelines off the default branch, and empty on
the default branch. Only default-branch builds therefore write the unscoped pool, which is the one `fallback_keys`
names. A restore replaces `.ccache/` rather than merging into it, so with one pool shared by every pipeline the last
writer wins. When merge requests on unrelated bases shared a pool, they overwrote each other's objects and compiled at a
0.35% hit rate. GitLab appends its clear-cache index and `-protected` or `-non_protected` to both the key and the
fallback key. A merge request pipeline started by a Developer therefore cannot read the pool that the scheduled builds
fill. Two tiers opt out of scoping by setting `EIGEN_CI_CCACHE_SCOPE` back to `""` in the job. The smoke tier
(`.smoketest:build`) runs only on merge request events, so if it were scoped, nothing would ever write its shared pool.
Windows opts out because its runner has no distributed cache, so per-merge-request archives would pile up on its disk
without limit. Any self-hosted runner without `[runners.cache]` keeps one archive per key forever.
[`prune_runner_cache.py`](../ci/scripts/prune_runner_cache.py) caps such a directory (`--max-gb`, LRU by mtime) and
drops superseded clear-cache generations (`--stale-index-below`). Its unit tests,
[`test_prune_runner_cache.py`](../ci/scripts/test_prune_runner_cache.py), run in `checkformat:lint`.

## Test Tiers On Merge Requests

Three tiers, in increasing cost:

| Tier | Trigger | What runs |
|---|---|---|
| smoke | every MR with neither label below | the fixed list in [`cmake/EigenSmokeTestList.cmake`](../cmake/EigenSmokeTestList.cmake), usually one part per test, at baseline ISA on x86-64, aarch64 and riscv64, under gcc and clang |
| affected | `affected-tests` label | every test the diff can reach, all parts, on x86-64 (gcc AVX2, clang baseline) and aarch64 (gcc, clang), plus any platform the diff or a `*-tests` label selects |
| full | `all-tests` label (requires explicit user permission) | the whole suite across the entire compiler and ISA matrix, minus the NVHPC pair below |

One configuration sits outside all three tiers: the NVHPC (`nvc++`) build and test jobs. The `nvc++` frontend is so slow
that the two NVHPC builds alone once took roughly a quarter of the project's hosted-runner minutes. They run on
schedules, web pipelines, and merge requests labeled `nvhpc-tests`. That label works without any other label, and the
smoke jobs still run alongside it. Apply `nvhpc-tests` when a change plausibly affects `nvc++` rather than waiting for
the scheduled run to find it. Like `all-tests`, it requires explicit user permission. A web pipeline is no substitute on
a merge request: it needs the branch in `libeigen/eigen` and runs the full tier as well.

The affected tier exists because the smoke list is only a sample. It is broad but shallow: a change confined to one
module gets only the one part of each related test that the list happens to name. Use `affected-tests` for depth, then
choose platform labels for the compilers and backends the change can affect. For a shared-header change needing broad
platform coverage, recommend `affected-tests` with `all-platforms`. The affected tier already expands to the full suite
when needed.

The tiers do not stack: `affected-tests` and `all-tests` each turn off the smoke jobs. The affected tier's four
unconditional jobs use the smoke compilers, gcc-10 and clang-14, on x86-64 and aarch64. Two smoke configurations come
back only with a label: riscv64 (`rvv-tests` or `all-platforms`), and x86-64 gcc at baseline ISA (`sse-tests` or
`all-platforms`), since the unconditional gcc job builds for AVX2.

[`scripts/affected_tests.py`](../scripts/affected_tests.py) chooses the affected tests in the `select:tests` job. Run it
locally the same way CI does:

```bash
python3 scripts/affected_tests.py --base-sha $(git merge-base origin/master HEAD)
```

The script follows the textual `#include` graph and ignores preprocessor guards. Its selection is therefore a strict
superset of the real compile dependencies, and it never drops an affected test. Eigen is header-only, and the umbrella
headers (public module headers such as `Eigen/Core`) are hubs of the include graph. A change under `Eigen/src/Core`
therefore typically reaches every test, and the script falls back to the full suite. That is the correct answer, not a
failure. Changes to CMake, `ci/scripts/`, `ci/docker/`, or the BLAS/LAPACK shims also force the full suite. Changes to
the `ci/*.gitlab-ci.yml` files, or to the clang-tidy and lint images under `ci/tidy/` and `ci/lint/`, select no tests.

### Platform-Triggered Configurations

Every job in the default smoke matrix builds at baseline ISA, so the smoke tier never compiles a change under
`Eigen/src/Core/arch/AVX512` with AVX-512 enabled. The `affected-tests` tier adds platforms beyond its four
unconditional jobs when either of two independent triggers fires:

- **the diff**, when a `rules:changes:` entry matches the backend directory. This is automatic and the common case.
- **a label**, read from `$CI_MERGE_REQUEST_LABELS`. The include graph decides *which tests* run; the labels decide
  *where* they run, independently of the graph. Use a label to run the affected tests on a platform the diff does not
  trigger: a `Core` change on ppc64le, or a `Geometry` change on Windows.

| Backend directory | Label | Added configuration | In `all-platforms` |
|---|---|---|---|
| `arch/SSE` | `sse-tests` | x86-64 gcc-10 baseline, AVX, and AVX-512DQ | yes |
| `arch/AVX` | `avx-tests` | x86-64 gcc-10 AVX and AVX-512DQ | yes |
| `arch/AVX512` | `avx512-tests` | x86-64 gcc-10 AVX-512DQ | yes |
| `arch/AVX512/*FP16*` | `avx512-tests` | the split clang-19 AVX512-FP16 compile builds | no |
| `arch/NEON` | `neon-tests` | 32-bit arm (aarch64 already runs unconditionally) | yes |
| `arch/AltiVec` | `altivec-tests` | ppc64le gcc-14, under qemu | yes |
| `arch/LSX` | `lsx-tests` | loongarch64 gcc-14, under qemu | yes |
| `arch/RVV10` | `rvv-tests` | riscv64 gcc-15, on the native runner | yes |
| `arch/SVE` | `sve-tests` | SVE cross builds and test runs at 128, 256 and 512 bits under qemu | yes |
| `arch/SME` | `sme-tests` | the full SME build, compile-only | no |
| — | `windows-tests` | MSVC 14.29 x64 baseline | yes |
| `arch/GPU`, the `Half.h`/`BFloat16.h` scalar headers, the `GpuHipCuda*.inc` alias files and `GpuRuntime.h`, `cmake/EigenTesting.cmake` and `cmake/EigenGpuTesting.cmake`, the `contrib/Eigen/GPU` module, the Tensor `*Gpu*.h` headers, the GPU tests and their harness headers (`.rules:libeigen:gpu` in [`ci/common.gitlab-ci.yml`](../ci/common.gitlab-ci.yml) has the exact list) | `gpu-tests` | the CUDA build and test jobs | no |

Several labels together select the union of their platforms: `neon-tests` with `altivec-tests` runs 32-bit arm and
ppc64le and nothing else. Apart from `gpu-tests`, none of them does anything without `affected-tests`. `all-platforms`
is shorthand for every row that *runs the affected tests*. The three rows marked "no" ignore which tests are affected
and compile the whole suite. `all-platforms` does not add them; name each row's own label. `all-platforms` on a one-line
change therefore cannot silently cost hours of whole-suite compilation.

Rows worth knowing before relying on them:

- A wider x86 configuration compiles the narrower backends' headers, which is why the SSE row adds three builds.
  AVX512-FP16 headers are guarded by `EIGEN_VECTORIZE_AVX512FP16`, so an AVX-512DQ build does not parse them. The
  `*FP16*` row is compile-only because no current runner can execute those instructions.
- SVE is a fixed-length backend, so each vector length is a separate build with different fold counts and transpose
  networks. `test/sve_vector_length` fails the run when a binary runs at a different length than it was built for.
  Without that test, such a mismatch would compute wrong answers while the suite passes.
- Windows has no `changes:` trigger. The problems MSVC catches, such as template instantiation limits,
  `EIGEN_STRONG_INLINE` behavior, and optimizer heap exhaustion, can come from anywhere in the library. A label
  (`windows-tests` or `all-platforms`) is therefore the only way to add Windows, and only MSVC x64 at baseline ISA is
  wired up. The 32-bit, AVX2 and AVX-512DQ Windows configurations remain in `all-tests`.
- No affected-tier configuration enables CUDA, HIP or SYCL. A diff confined to GPU test sources would therefore select
  targets that no host build defines, and the pipeline would show green having run nothing. Changes to those paths add
  the existing CUDA jobs instead. These build `buildtests_gpu` and run every test with the `gpu` CTest label, so they
  cover the whole GPU suite, not just the affected tests.
- `arch/ZVector`, `arch/MSA`, `arch/HVX` and the `arch/SYCL` backend have no matching test configuration. A change
  there gets only the four unconditional jobs and the same hollow result: green without testing the change.
  `gpu-tests` does not help either, because the GPU jobs it adds are all CUDA or ROCm.

The CUDA matrix has two parts. GitLab's SaaS T4 runners (sm_75) run CUDA 11.8 with gcc-10 and clang-14. The project's
L4 runner (sm_89) runs CUDA 12.6 with gcc-13 and clang-19, plus CUDA 13.3 with gcc-13. The ROCm job is build-only. The
`.cu` tests are compiled with CMake's CUDA language support, configured once in
[`cmake/EigenGpuTesting.cmake`](../cmake/EigenGpuTesting.cmake). `nvc++` and clang-as-CUDA-on-Windows cannot use that
language support, so they compile the tests as C++ instead. The Linux CUDA test jobs are `allow_failure: true`, so a
red GPU job shows as a warning, and a green pipeline is not evidence that the GPU tests passed. Hence the policy for a
merge request that touches any path in the GPU row: apply `gpu-tests` (and `affected-tests` when it also changes
shared headers), name the GPU jobs that ran and their status in the description, and re-run the L4 jobs after rebasing
onto another change that touches GPU paths. A scheduled pipeline with `EIGEN_CI_SCHEDULE_SCOPE` set to `gpu` runs only
these jobs. That is how a second, cheaper GPU schedule coexists with the weekly full run.


## Worktree-Safe Formatting

Inspect `git status --short` before formatting and preserve unrelated changes. Eigen requires exactly
`clang-format-17`. The version is pinned in [`ci/lint/Dockerfile`](../ci/lint/Dockerfile), which builds a static
clang-format 17.0.6 from the LLVM release. CI checks only the lines a merge request changes. The tree is not uniformly
clang-format-17 clean: formatting whole files rewrites `> >` closers in a couple of dozen headers. So format the diff:

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

The two Python scripts report only on the lines a change adds, and both are advisory. `check_style.py` covers the
conventions clang-tidy cannot express: comment verbosity, and the declaration forms whose `CustomChecks` queries are
not enabled yet (see the commented-out block in `.clang-tidy`). `clang_tidy_hook.py` runs clang-tidy itself, limited
to added lines with `--line-filter`. It needs no build directory. Instead it generates a driver source file that
includes the module's umbrella header and then the edited `Eigen/src` header, as `ci/scripts/run-clang-tidy.sh` does
for merge requests. It skips silently when clang-tidy is not installed, and shows the user a non-blocking notice when a
file's translation unit does not compile.

Claude Code sessions run both automatically through the hooks registered in `.claude/settings.json`. Their unit
tests, [`scripts/test_check_style.py`](../scripts/test_check_style.py) and
[`scripts/test_clang_tidy_hook.py`](../scripts/test_clang_tidy_hook.py), run in `checkformat:lint`; run them after
changing either script.

The whole-tree codespell invocation used by CI can expose pre-existing findings. Do not modify unrelated files merely to
make a local broad scan clean. `checkformat:lint` runs clang-format, codespell, REUSE, and the Python helper tests
through [`ci/lint/lint.sh`](../ci/lint/lint.sh), with `vermin` checking that the helpers still run on Python 3.12. REUSE
and the helper tests are blocking. A clang-format or codespell failure alone only marks the job as a warning, and any
failure in `checkformat:clangtidy` is also only a warning. Treat their diagnostics as review findings anyway.

Source files carry the inline SPDX header that [`conventions.md`](conventions.md) records. Files that cannot carry
one need coverage in [`REUSE.toml`](../REUSE.toml). To stamp selected new files with the repository helper, pass them
explicitly because its default scan considers tracked files:

```bash
python3 scripts/add_spdx_headers.py --paths path/to/new-file.cpp
```

## Clang-Tidy

Lint an implementation header with the CI driver below, not by running clang-tidy on it directly. The driver compiles
the header through its public umbrella header.

```bash
cmake -G Ninja -S . -B .tidy-build \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DEIGEN_BUILD_TESTING=ON
ci/scripts/run-clang-tidy.sh <base-sha> .tidy-build
```

`checkformat:clangtidy` runs clang-tidy 18 from [`ci/tidy/Dockerfile`](../ci/tidy/Dockerfile). The driver examines files
committed between `<base-sha>` and `HEAD`; a file whose only edits are uncommitted is not checked. Eigen's `.clang-tidy`
policy is authoritative. Do not apply generic `modernize-*` or `cppcoreguidelines-*` rewrites in bulk.

When a module includes a third-party header that the machine does not have installed, such as `<cuda_runtime.h>` in
`contrib/Eigen/src/GPU` or `<cholmod.h>` in `CholmodSupport`, the module is still checked, but clang parses a
truncated translation unit. The driver therefore marks the file's log heading `— partial: <header> is not installed`
and reports that file's findings without failing the job. Installing the dependency gets the module checked in full.
For CUDA, both the driver and `clang_tidy_hook.py` look under `CUDAToolkit_ROOT`, `CUDA_HOME`, `CUDA_PATH`, then
`/usr/local/cuda`. An unresolved *in-tree* include is a defect in the change and stays a hard error.

The driver does not include an edited header under `arch/<ISA>/`, other than `arch/Default/`, directly. Such a
header parses only under the `-march`/`-mcpu` flag that selects its backend, and this job does not pass one. It is
linted only when the host target already selects the backend, as the x86-64 runner does for SSE2, and the heading
says which backend went unchecked. Validate a change to such a header with a build that enables the ISA rather than
relying on this job.

For a split test the driver checks only the parts that compile the added lines and prints beside the file name the
parts it left out. When a cap on the number of parts leaves some out, the run names them rather than reporting the
file clean. [`ci-internals.md`](ci-internals.md) explains how the parts are chosen.

## Before Review

1. Inspect `git diff` and `git diff --check`.
2. Format and check the task's changed lines and new files using the Worktree-Safe Formatting recipes above.
3. Run the focused builds and tests documented in [`testing.md`](testing.md).
4. Run applicable spelling, REUSE, and clang-tidy checks.
5. Apply the `docs-build` label when the change touches Doxygen markup, a documented name, a module `README`, or a
   snippet. The recommended test labels do not trigger the documentation job; [`docs.md`](docs.md) records its
   coverage and validation requirements.
6. State what ran, what did not run, and why. Do not claim coverage from jobs or hardware that were unavailable.
