# Contributing to Eigen

Eigen is written and maintained by volunteers. Contributions — code, documentation, bug triage, tests on uncommon platforms, design feedback — are all welcome.

This document is where human contributors should start. [`AGENTS.md`](AGENTS.md) holds the guidance written for AI
coding agents, and it links to focused guides for coding conventions, testing, numerical and sparse work, performance,
SIMD/GPU, Tensor/ThreadPool, documentation, CI, and merge requests. Those guides are also useful to human contributors
working in the corresponding areas.

## Where Eigen lives

- **Upstream repository:** <https://gitlab.com/libeigen/eigen> — bugs, merge requests, and discussion all happen here. The GitHub mirror is read-only.
- **Issue tracker:** <https://gitlab.com/libeigen/eigen/-/issues>
- **CI pipelines:** <https://gitlab.com/libeigen/eigen/-/pipelines>
- **API documentation (nightly):** <https://libeigen.gitlab.io/eigen/docs-nightly>
- **Project site:** <https://libeigen.gitlab.io> (some of its pages are out of date; this file and the guides
  linked from `AGENTS.md` take precedence)
- **Chat:** [Discord](https://discord.com/channels/777904510169382942/777904791136370758) — the primary support and discussion channel.

## Ways to contribute

You don't have to write C++ to help.

- **Answer questions** on Discord. New users often ask things that point at gaps in our documentation — patches to fill those gaps are very welcome.
- **Triage bugs.** Reproduce reports, ask reporters for missing details (compiler, OS, instruction set, minimal example), and weigh in on issues labeled [`decision needed`](https://gitlab.com/libeigen/eigen/-/issues?label_name%5B%5D=decision+needed). Issues labeled [`contributions welcome`](https://gitlab.com/libeigen/eigen/-/issues?label_name%5B%5D=contributions+welcome) are good places to start.
- **Run the test suite** on rarely-tested compilers, OSes, or architectures and report failures. The CI pipelines page linked above shows which configurations the scheduled runs already cover.
- **Improve documentation** — the Doxygen reference (comments in headers) and the manual pages, snippets, and examples under `doc/`; see [`.agents/docs.md`](.agents/docs.md). The project site has its own repository, <https://gitlab.com/libeigen/libeigen.gitlab.io>.
- **Contribute code** — bug fixes, new features, new tests, new benchmarks. See below.

## Submitting a merge request

1. **Sign in to GitLab** and fork <https://gitlab.com/libeigen/eigen>.
2. **Clone your fork** and add `libeigen/eigen` as an upstream remote (use the `https://` clone URL if you don't have an SSH key set up):
   ```bash
   git clone git@gitlab.com:<your-username>/eigen.git    # or: https://gitlab.com/<your-username>/eigen.git
   cd eigen
   git remote add upstream https://gitlab.com/libeigen/eigen.git
   ```
3. **Create a topic branch** off `master` for your change:
   ```bash
   git fetch upstream
   git checkout -b my-topic upstream/master
   ```
4. **Make your changes**, with tests (and a benchmark if it's a performance-sensitive change — see [Tests and benchmarks](#tests-and-benchmarks)).
5. **Format** with `clang-format-17` (see [Coding standards](#coding-standards)).
6. **Commit** with an `Area: Short imperative subject` message (see [Commit messages](#commit-messages)). Stage files by path — don't use `git add -A`.
7. **Push** to your fork and open a merge request against `libeigen/eigen` `master`.
8. **Be patient and responsive.** Reviewers are volunteers; reviews can take a while. If your MR is being ignored for a long time, don't be shy about leaving a comment to ask for a review.

If you're planning a **large change** (new module, API redesign, significant refactor), open an issue or start a discussion first. Agreeing on the approach before you write code saves everyone work.

## Build and test

Eigen is **header-only**, so consumers don't need to build the library itself — but contributors do need to build the test suite. Configure CMake in a separate build directory. The tests are deliberately *not* part of the default `all` target, so build them through the named targets below.

```bash
cmake -S . -B build -G Ninja
cmake --build build --target buildtests        # build all unit tests
cmake --build build --target check             # build + run all tests
cmake --build build --target buildsmoketests   # the smoke tests an MR pipeline runs by default
ctest --test-dir build --parallel --output-on-failure --no-tests=error
```

Other useful targets: `BuildOfficial` (just `test/`), `BuildContrib` (just `contrib/test/`), `buildtests_gpu`, `check_gpu`. After configuring, the build directory also contains `./buildtests.sh <regex>` and `./check.sh <regex>`, which build or run only the tests whose names match.

[`.agents/testing.md`](.agents/testing.md) explains how to build a single test, register a new one, split a test into
parts, and choose assertions. [`.agents/simd-gpu.md`](.agents/simd-gpu.md) covers testing SIMD and GPU code. For the
current CMake options, read the CMake files themselves.

> **Migration in progress:** [MR 2159](https://gitlab.com/libeigen/eigen/-/merge_requests/2159) is migrating the test framework from Eigen's own `EIGEN_DECLARE_TEST` / `CALL_SUBTEST_N` macros to Google Test. Until it is merged, write new tests with the existing framework (`test/main.h`, `VERIFY_*`, `EIGEN_DECLARE_TEST`, `ei_add_test` in the matching `CMakeLists.txt`).

## Tests and benchmarks

- **Every new feature comes with tests.** Bug fixes should add a regression test that fails before the fix and passes after.
- **Performance-sensitive changes come with a benchmark** under `benchmarks/` or `contrib/benchmarks/`. Don't defer benchmarks to a follow-up MR.
- **Cover numerical corner cases**, not just typical inputs: ±0, ±∞, NaN, subnormals, domain boundaries, near-overflow / underflow values. Decomposition and solver tests should exercise ill-conditioned and structured matrices (Hilbert, Vandermonde, rank-deficient, near-singular, Jordan blocks, …); see [Quality bar](#quality-bar). Standard references: Higham's *Accuracy and Stability of Numerical Algorithms* / *Functions of Matrices*, Golub & van Loan's *Matrix Computations*.
- **New SIMD intrinsics need a `test/packetmath.cpp` entry** (or `contrib/test/special_packetmath.cpp` for special functions), and a specialization in every architecture backend that supports the type. `Eigen/src/Core/arch/Default/` holds generic SIMD implementations shared across backends; scalar fallbacks live in `Eigen/src/Core/GenericPacketMath.h` and `Eigen/src/Core/MathFunctions.h`.
- **Benchmarking discipline.** Benchmarks on a loaded system are meaningless. Before running one: check `uptime` shows a low load average, finish or cancel background builds, and never run two benchmark binaries in the same shell invocation (parallel or chained with `&&`). Run each in its own shell, take medians within a binary, and optionally alternate (A, B, A, B) across separate invocations to detect drift.

## Coding standards

A few rules trip up contributors more often than the rest. The full list is in
[`AGENTS.md`](AGENTS.md#non-negotiable-rules), which also links to the task guides. In short:

1. **Include only the umbrella headers.** User code, examples, and tests of public behavior include the umbrella headers (`Eigen/Core`, `Eigen/Dense`, `Eigen/SVD`, …), never files under `Eigen/src/...` or `contrib/Eigen/src/...`. Most of those files include `InternalHeaderCheck.h`, which raises a compile error when one is included without its module's umbrella header; the paths are private even where that guard is absent.
2. **C++14 baseline.** Supported headers must compile as C++14: no `if constexpr` (use `EIGEN_IF_CONSTEXPR`), structured bindings, fold expressions, or class template argument deduction. The exception is backend code that is compiled only behind a preprocessor guard and documents that it needs a newer standard. See [`.agents/conventions.md`](.agents/conventions.md).
3. **Preserve `EIGEN_DEVICE_FUNC`** on coefficient-level methods. Dropping it silently breaks CUDA / HIP / SYCL builds and rarely shows up in local testing.
4. **Format with `clang-format-17` exactly** — Google style, 120 columns, configured in `.clang-format`. Other versions
   of `clang-format` produce different output. CI checks only the lines a merge request changes
   (`git clang-format --diff --commit <base>`), because older code in the tree is not all clang-format-17 clean. The
   check is advisory: a failure shows as a warning on the `checkformat:lint` job instead of blocking the pipeline. Fix
   it anyway before asking for review. To format your changed lines, first check that the files contain no unrelated
   edits, then run `git clang-format --binary clang-format-17 --force <base-sha> -- <files>`; `--force` allows files
   with unstaged edits. Format new, untracked files with `clang-format-17 -i <files>`, because Git's diff omits them.
   [`.agents/ci.md`](.agents/ci.md) describes the checks. `scripts/format.sh` reformats the whole tree; run it only
   when that is the intent.
5. **Don't apply general C++ "modernize" cleanups.** No `modernize-*` / `cppcoreguidelines-*` clang-tidy fixes, no replacing `EIGEN_STRONG_INLINE` with `inline`, no reordering includes (`.clang-format` has `SortIncludes: false`). `.clang-format` lists Eigen's macros under `StatementMacros` and `AttributeMacros`, so the layout it produces around them is intended — don't "fix" it.
6. **Tests use `test/main.h`, not gtest** (for now; see the migration note above). `VERIFY_*` assertions, `CALL_SUBTEST_N` / `EIGEN_TEST_PART_N` for splitting, `EIGEN_DECLARE_TEST(<name>) { ... }` as the entry point.
7. **`auto` traps.** `auto x = A + B;` captures a lazy expression holding references that can dangle. Use `.eval()` or an explicit type. Similarly, `.noalias()` is a promise from the caller — only use it when the destination doesn't appear on the right side (so `mat.noalias() = mat * mat` is **wrong**).
8. **Stage commits explicitly.** Don't `git add -A` / `git add .` — the repo root accumulates `.vscode/`, `.idea/`, `build*/`, and other untracked files that must not enter commits. Add by path or with targeted globs.

Naming conventions:
- Classes: PascalCase (`Matrix`, `BDCSVD`).
- Methods: camelCase (`coeffRef`, `applyOnTheLeft`).
- Macros: `EIGEN_UPPER_CASE` (`EIGEN_DEVICE_FUNC`, `EIGEN_STRONG_INLINE`).
- Compile-time constants: `static constexpr` members, not `enum` blocks. `kCamelCase` is accepted alongside the older
  `snake_case` and `SCREAMING_CASE` spellings.
- Internal implementation lives in the `Eigen::internal` namespace; public API stays in `Eigen::` or module namespaces.
- Use `eigen_assert(cond)` for runtime preconditions (not raw `assert`) and `eigen_internal_assert` for internal
  invariants, which are checked only when `EIGEN_INTERNAL_DEBUGGING` is defined. Check compile-time API preconditions
  with the appropriate `EIGEN_STATIC_ASSERT_*` helper or `EIGEN_STATIC_ASSERT(cond, MSG_TOKEN)`, so that
  `EIGEN_NO_STATIC_ASSERT` and a user-provided `EIGEN_STATIC_ASSERT` still take effect. Use
  `static_assert(cond, "message")` for implementation invariants that must always be checked.

Recent additions such as `contrib/Eigen/src/GPU/DeviceMatrix.h` show the usual layout of a class and its header:

- A template parameter that the class also exposes as a public alias gets a trailing underscore: `template <typename Scalar_, ...>` with `using Scalar = Scalar_;`. Prefer `using` to `typedef` in new code.
- Member variables use an `m_` prefix (`m_matrix`, `m_isInitialized`).
- Implementation headers under `Eigen/src/...` carry `// IWYU pragma: private` and `#include "./InternalHeaderCheck.h"`, and their header guards follow `EIGEN_<NAME>_H`.
- Implementation details go in the nested namespace, `namespace Eigen { namespace internal { ... } }`.
- Public classes carry Doxygen blocks (`\class`, `\brief`, `\tparam`, `\sa`) and link to runnable snippets from `doc/snippets/` with `\include` / `\verbinclude`.

When in doubt, copy the style from a recent neighboring file rather than inventing a new convention.

### License and SPDX/REUSE headers

Every new source file must carry an inline copyright + license header. Without one, the `reuse lint` check in the `checkformat:lint` CI job fails the pipeline. Two header styles are in active use; pick whichever fits.

Attributed collectively (the most common form in the tree; use it for files written by multiple contributors, or where individual attribution isn't useful):

```cpp
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
```

Attributed to an individual contributor:

```cpp
// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) <year> <Your Name> <your.email@example.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0
```

Use the file's own comment syntax (`#` in CMake and shell). Markdown docs, generated files (`*.in`), and binary assets that can't carry inline headers are listed by path in `REUSE.toml` — add yours there if you create one.

### Provenance and attribution

This is non-negotiable.

- Eigen code must be **original**, or derived from publicly published **MPL-2.0-compatible** material.
- **Do not copy** — verbatim, paraphrased, or "translated" to another syntax — from incompatibly-licensed sources (proprietary, under NDA, internal to a previous employer, GPL/AGPL). A close paraphrase or a line-by-line translation can still infringe copyright, even when no line matches. The same applies to AI-suggested code: cite the source, or rewrite from a known reference and cite that, or drop it. Ideas aren't copyrightable; specific expressions are — learn from the source, write your own, and credit it.
- **Do not look inside proprietary software.** Proprietary libraries (Intel oneMKL, NVIDIA's cuBLAS, Arm Performance Libraries, Apple Accelerate, vendor binaries in general) are black boxes: use their documented interface and measure them as shipped. Copyright leaves ideas free, but looking inside can breach the license, and using what was learned can support a trade-secret claim, so code that uses an idea obtained that way is unacceptable even when nothing is copied. See [`.agents/provenance.md`](.agents/provenance.md).
- **Cite published references inline** when they inform an implementation: LAPACK and the LAPACK Working Notes, ACM TOMS / SIAM papers, Higham, Golub & van Loan, textbook algorithms, Boost components, vendor application notes — by name (author, year, identifier). A comment or Doxygen `\note` block is enough.
- When copying permissively-licensed code into an MPL-2.0 file, follow Mozilla's [Guidelines for Developers](https://www.mozilla.org/MPL/2.0/permissive-code-into-mpl.html).

**Show where a change comes from.** We encourage you to link, in the merge request description, the publicly available sources that support your code: the instruction-set or architecture manual, the vendor's optimization guide or intrinsics reference, the paper or standard an algorithm follows, the documented API a backend relies on. Original research is just as welcome — your own microbenchmarks of the hardware, measurements of other libraries through their public interfaces, numerical experiments, new algorithms and derivations — provided you have the right to contribute it; include its evidence (the reproducer, the method and results, the derivation) instead of a link. This helps most with hardware-specific optimizations and new features, where it lets a reviewer check the change against its sources and see that everything it relies on is public or published with it.

### Copyright and credit

- **No copyright assignment.** Everyone keeps copyright on their own contributions. There is no contributor license agreement (CLA).
- **Commit under your own identity.** Make sure `user.name` and `user.email` in your git config are set to credit you correctly.
- When contributing on behalf of your employer, **commit in your own name** where possible rather than your employer's.
- For substantial improvements to a source file, feel free to add yourself to its copyright line — but it isn't required; you get copyright implicitly on the code you wrote. The collective `The Eigen Authors` attribution is fine when individual credit isn't useful.

## Commit messages

Format: `Area: Short imperative subject`, with both the area and the subject capitalized.

Examples (from recent history):
- `Core: Bring the vectorized cbrt within 1 ULP`
- `SparseCore: Fix expressions on a sparse matrix diagonal under MSVC`
- `Tests: Split numext into parts to bound its compile memory`
- `CMake: Remove discontinued ComputeCpp support`

Keep the subject under ~72 characters; use the body for the *why*, not the *what* — the diff already shows what changed. Reference issues by `#<n>` when relevant.

## Quality bar

Eigen aims to be state of the art in two respects — **performance** and **numerical accuracy / IEEE-754 conformance** — which are separate goals and sometimes conflict.

- **Performance.** Eigen Core is mostly optimized for single-core throughput, through SIMD code for each architecture (the packet backends) and cache-aware blocking. Multithreading in Core is opt-in (OpenMP or `EIGEN_GEMM_THREADPOOL`) and covers a subset of operations. The Tensor module is designed for multithreading through `ThreadPoolDevice`.
- **Numerical accuracy.** Decompositions and solvers should match LAPACK in backward stability, pivoting strategy, and handling of ill-conditioned inputs. Don't trade their numerical robustness for speed without a maintainer's explicit agreement. Standard math functions on scalars should match the C++ standard library. For vectorized math functions on regular inputs, a few ULPs of error is the long-standing price of SIMD speed; larger deviations need explicit justification.
- **IEEE-754 / special values.** For special values (NaN, ±0, ±∞, subnormals, singularities, branch boundaries — e.g. `log(0)`, `pow(0, 0)`) the requirement is **exact conformance** to the IEEE 754, ISO C, and C++ specifications, as documented on cppreference. The few-ULPs allowance does **not** apply to special values.

## Widely used contrib modules

Some `contrib/` modules matter more than the directory name suggests:

- **`contrib/Eigen/Tensor`** and **`Eigen/ThreadPool`** together form TensorFlow's core compute backend. The `contrib/` location means *looser API-stability guarantees* — it does **not** mean rarely used or unimportant. Breaking changes to header layout, signatures, semantics, or the performance of the contraction and reduction kernels affect every TensorFlow build. Prefer additive changes, keep header paths stable, run the `contrib/test/tensor_*` tests before submitting, and call out any behavior change prominently in the MR description.

## Communication

- **Discord** is the primary chat channel ([link above](#where-eigen-lives)). Most informal design discussion happens there.
- **GitLab issues** are the place for bug reports, feature requests, and threaded design discussions tied to a specific topic.
- For larger contributions, **discuss the plan first** to avoid duplicated effort and to agree on the API before implementing it.

## Further reading

- [`AGENTS.md`](AGENTS.md) — guidance for AI coding agents: the repository-wide rules, the standard workflow, essential
  Eigen hazards, and links to the task guides.
- Task guides: [coding conventions](.agents/conventions.md), [testing](.agents/testing.md),
  [numerical code](.agents/numerics.md), [sparse](.agents/sparse.md), [benchmarking](.agents/benchmarking.md),
  [provenance](.agents/provenance.md), [SIMD/GPU](.agents/simd-gpu.md),
  [Tensor/ThreadPool](.agents/tensor-threadpool.md), [documentation](.agents/docs.md),
  [formatting/CI](.agents/ci.md), [merge request descriptions](.agents/merge-requests.md), and
  [answering review comments](.agents/review-response.md).
- [`README.md`](README.md) — high-level project description and pointers to the websites.
- [`CHANGELOG.md`](CHANGELOG.md) — release-by-release notes.
- API reference (nightly): <https://libeigen.gitlab.io/eigen/docs-nightly>.

Thank you for contributing to Eigen!
