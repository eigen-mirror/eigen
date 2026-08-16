# Formatting And CI

Use the checked-out configuration as the source of truth. [`.gitlab-ci.yml`](../.gitlab-ci.yml) defines stages and
includes; [`ci/*.gitlab-ci.yml`](../ci) and [`ci/scripts/`](../ci/scripts) define the actual jobs. Default MR pipelines
run a limited smoke matrix; labels such as `all-tests` and `gpu-tests`, plus scheduled or manually started pipelines,
enable broader jobs. A green default MR pipeline is not proof that every supported configuration was exercised.

A pipeline is evidence only for the commit it ran on: after a push, amend, or rebase, check which SHA the pipeline and
the merge request point at before citing either — a green run on a superseded revision proves nothing about the
current head, and a reported failure should be reproduced at the current head too.

Build jobs publish the configured build directory as an artifact. Their paired test jobs consume that artifact and
run CTest without rebuilding. When changing either side, keep the test job's `needs`, CTest label or filter, and the
corresponding build target consistent; otherwise CTest can discover tests whose executables are absent.

In merge-request pipelines the Linux test jobs also keep a content-addressed pass cache (a per-job-name GitLab cache
holding `.testcache/`): [`test.linux.script.sh`](../ci/scripts/test.linux.script.sh) skips tests whose executable,
emulator, CTest definition, and environment fingerprint (image, `lib*` package state, the checked-in CI
configuration, and behavior-affecting variables such as `EIGEN_REPEAT` and `QEMU_CPU`) match a first-attempt pass
recorded by an earlier MR pipeline, then records this run's first-attempt passes — taken from the dashboard run's
`Test.xml` statuses — via [`test_cache.py`](../ci/scripts/test_cache.py). Scheduled and web pipelines always run
their full selection (fresh clock-derived RNG seeds are part of their coverage), sharded jobs never skip, and
`EIGEN_CI_TEST_CACHE: "off"` opts a job out. Skipped tests are absent from that run's JUnit report, and a test job
whose binaries all match cached passes legitimately reports "No tests were found".

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

## Before Review

1. Inspect `git diff` and `git diff --check`.
2. Format the exact changed source files with clang-format-17.
3. Run the focused builds and tests documented in [`testing.md`](testing.md).
4. Run applicable spelling, REUSE, and clang-tidy checks.
5. Build the `doc` target when the change touches Doxygen markup, a documented name, or a snippet, and label the merge
   request `all-tests` so the blocking documentation job runs before the merge rather than after it.
6. State what ran, what did not run, and why. Do not claim coverage from jobs or hardware that were unavailable.
