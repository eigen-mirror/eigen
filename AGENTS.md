# AGENTS.md

Guidance for AI coding agents working in Eigen. Human contributors should start with
[`README.md`](README.md) and the project documentation it links to. Per-tool files such as `CLAUDE.md` should import
this file and contain only tool-specific additions.

## Scope and precedence

Follow the user's task, then the nearest applicable `AGENTS.md`, then repository documentation and established local
patterns. The checked-out source, tests, CMake files, and CI configuration are the authority on how things currently
work. If this guide disagrees with the tree, follow the tree, report the discrepancy, and update the guidance when
that is in scope.

Read this file for every task. Then read the guides in every row below that matches the work. Do not load unrelated
guides by default.

| Work area | Additional guidance |
|---|---|
| Any new or rewritten code | [`.agents/conventions.md`](.agents/conventions.md) |
| Tests and CMake test targets | [`.agents/testing.md`](.agents/testing.md) |
| Numerical kernels, decompositions, solvers, accuracy | [`.agents/numerics.md`](.agents/numerics.md) |
| Sparse matrices, sparse solvers, external sparse backends | [`.agents/sparse.md`](.agents/sparse.md) |
| Performance changes and benchmarks | [`.agents/benchmarking.md`](.agents/benchmarking.md) |
| Comparing against, benchmarking against, or integrating external libraries | [`.agents/provenance.md`](.agents/provenance.md) |
| Packet math, CUDA, HIP, SYCL, `contrib/Eigen/GPU` | [`.agents/simd-gpu.md`](.agents/simd-gpu.md) |
| Tensor, ThreadPool, and multithreading | [`.agents/tensor-threadpool.md`](.agents/tensor-threadpool.md) |
| Formatting, lint, and GitLab CI | [`.agents/ci.md`](.agents/ci.md) |
| Doxygen blocks, `doc/` pages, snippets, and examples | [`.agents/docs.md`](.agents/docs.md) |
| Changes under `ci/`, `.gitlab-ci.yml`, or the test-selection and cache scripts | [`.agents/ci-internals.md`](.agents/ci-internals.md) |
| Writing or updating a merge request description | [`.agents/merge-requests.md`](.agents/merge-requests.md) |
| Answering merge request review comments | [`.agents/review-response.md`](.agents/review-response.md) |
| Editing `AGENTS.md` or a guide under `.agents/` | [`.agents/writing-guidance.md`](.agents/writing-guidance.md) |
| Expression templates or evaluator internals | [`doc/TopicLazyEvaluation.dox`](doc/TopicLazyEvaluation.dox), [`doc/NewExpressionType.dox`](doc/NewExpressionType.dox), and [`doc/ClassHierarchy.dox`](doc/ClassHierarchy.dox) |

The wording of a rule sets its strength:

- *Must*, *must not*, *never*, and a plain imperative state a requirement. Depart from one only when the user tells you
  to, and report any requirement you could not meet.
- *Prefer*, *avoid*, *should*, and *usually* state a default. Depart from one when the task gives a concrete reason, and
  give that reason in your report.
- *May* grants permission. For possibility, the guides write *can* or *might*.

## Non-negotiable rules

1. **Preserve existing work.** Start with `git status --short`. Never discard, overwrite, reformat, or stage unrelated
   user changes. Do not use destructive Git commands unless the user explicitly requests that operation. Stage named
   paths, never `git add .` or `git add -A`.
2. **Keep provenance clean.** Code must be original or derived from source material whose license is compatible with
   Eigen's MPL-2.0 distribution. Do not copy, paraphrase, or translate code from proprietary, NDA-covered, internal, or
   incompatibly licensed sources. Published papers, standards, textbooks, and algorithm descriptions may inform an
   independent implementation. Cite each one inline where it materially informs the implementation. A citation does
   not make copied code permissible. The rule covers information as well as code. Treat proprietary software as a black
   box: use it only through its documented interface and measure it as shipped. Never disassemble, dump, debug into, or
   alter it; see [`.agents/provenance.md`](.agents/provenance.md). Never invent an attribution for AI-generated code. A
   `Co-Authored-By` trailer for the model that actually produced the change is accurate attribution, and is permitted.
3. **Respect the header-only and C++14 contracts.** Supported headers must compile as C++14, unless a backend behind a
   preprocessor guard has a documented newer requirement. In user code, examples, and public-behavior tests, include
   umbrella headers such as `Eigen/Core` or `Eigen/SVD`, not files below `Eigen/src/` or `contrib/Eigen/src/`. Where
   nearby tests establish the pattern, a focused test of a private utility may include the utility's header directly.
   Those paths remain private even where the header has no `InternalHeaderCheck.h` guard. Definitions in public headers
   must have linkage that is valid in a header and must not cause one-definition-rule (ODR) violations.
4. **Protect compatibility.** Treat supported public names, signatures, header paths, semantics, and ABI-affecting
   configuration as compatibility surfaces. Prefer additive changes and deprecation over removal. When you move a
   private implementation header, update the public umbrella header and delete the old private file rather than adding a
   forwarding shim at the old path. ABI-affecting Eigen macros must be consistent across translation units.
5. **Preserve Eigen annotations and style.** Do not drop `EIGEN_DEVICE_FUNC` from coefficient-level or device-callable
   functions. Do not replace `EIGEN_STRONG_INLINE` with `inline`, reorder includes, normalize Eigen macro layout, or
   apply broad `modernize-*` or `cppcoreguidelines-*` rewrites. The repository's conventions and `.clang-format` take
   precedence over generic C++ advice. This rule covers code your task does not otherwise change. Write new
   declarations in the current form that [`.agents/conventions.md`](.agents/conventions.md) describes, not in the
   superseded form around them.
6. **Enable a fast path only when the property it needs holds.** For a new specialization, capability flag, or enable
   condition, state the exact precondition it depends on and check for that. Do not check for an adjacent capability,
   for the existence of an overload, or for a property the built-in types merely happen to share. Give a new opt-in
   trait the conservative default. Extension points that users can specialize must stay correct when users leave them
   unannotated.
7. **Ship verification with behavior.** Add focused tests with new functionality. With a bug fix, add a regression test
   that fails without the fix when practical. With a performance-sensitive change, add an appropriate benchmark. Scale
   broader coverage to the affected scalar types, storage orders, backends, and public contracts. Confirm the new test
   fails at the parent commit when practical; otherwise show that the test runs the changed code by construction. See
   [`.agents/testing.md`](.agents/testing.md).
8. **Treat external writes as deliberate actions.** Push, open or update a merge request, comment on an issue, or write
   to any other external system only when the user has asked for that action. Otherwise, stop after the local commit and
   ask. Recommend the `affected-tests` label with the relevant platform labels, or with `all-platforms` for broader
   coverage; see [`.agents/ci.md`](.agents/ci.md). Do not add `all-tests` without the user's explicit permission for
   that label.

## Standard workflow

1. Inspect `git status --short`, the current branch, and the diff. Separate pre-existing work from the requested change.
2. As applicable, read the public header, implementation, nearby tests, registration in `CMakeLists.txt`, and relevant
   task guides before deciding on an implementation. Search with `rg` or `rg --files` (add `--hidden` to search
   `.agents/`). Before writing a helper, check `numext`, `NumTraits`, `MathFunctions.h`, `Meta.h`, `XprHelper.h`, and
   the `test/*_helpers.h` headers for an existing one. If one exists but lacks the hardening you need, fix it there
   rather than adding a local copy.
3. Keep the patch within the owning module and established patterns. Avoid opportunistic refactors and churn in
   generated files or metadata.
4. Add or update applicable tests and benchmarks in the same patch. Test public behavior through its umbrella header, so
   the test catches anything the umbrella fails to export. Follow nearby patterns for focused tests of private
   internals.
5. Inspect the diffs of the files you are about to format, so that you do not format unrelated changes. Then format the
   task's changed lines with `git clang-format --binary clang-format-17 --force <base-sha> -- <files>`; `--force` lets
   the command format files that have unstaged edits. Format task-created files with `clang-format-17 -i <files>`,
   because untracked files are absent from the diff. Formatting a whole existing file, or running `scripts/format.sh`,
   also rewrites older lines that are not clang-format-17 clean. Use them only when you intend that churn. See
   [`.agents/ci.md`](.agents/ci.md) for the matching check.
6. Build and run the narrowest relevant test first, then widen validation according to the change's risk. Use separate
   build directories for materially different CMake configurations.
7. Review `git diff --check`, `git diff`, and `git status --short`. Report the exact validation run and any unavailable
   compiler, ISA, GPU, dependency, or downstream coverage.
8. When review comments arrive, follow [`.agents/review-response.md`](.agents/review-response.md).

## Repository essentials

Eigen is a header-only expression-template library. Consumers include module headers under `Eigen/` or
`contrib/Eigen/`. The top-level CMake project builds tests, documentation, demos, and BLAS/LAPACK shims rather than
a core Eigen library; benchmarks use separate CMake projects. `Eigen/Dense` aggregates the dense modules, while
`Eigen/Eigen` includes `Dense` and `Sparse`. External backend support modules and `Eigen/ThreadPool` remain separate
includes. The upstream project is on GitLab; its GitHub repository is a read-only mirror.

The supported implementation is under `Eigen/src/`; tests are under `test/`. Modules with looser API-stability
guarantees are under `contrib/Eigen/`, with tests under `contrib/test/`. Legacy `unsupported/Eigen/...` include paths
remain valid: one-line forwarding shims under `unsupported/Eigen/` point at the `contrib/` headers and are installed
alongside them. "Contrib" does not imply low impact: Tensor is a foundational TensorFlow dependency. A module's public
umbrella header is the source of truth for which internals the module exports.

The `lapack/*.f` files are vendored copies of the netlib LAPACK reference sources and are read-only here. Do not edit
them ad hoc. Flag a merge request that changes one. The exception is an explicit refresh from a named netlib release;
check that diff against the release. `.git-blame-ignore-revs` lists the commits that ran
clang-format or added SPDX tags across the whole tree. To see the history beneath them, pass that file to `git blame`
with `--ignore-revs-file`.

Every new source file needs accurate REUSE metadata; [`.agents/conventions.md`](.agents/conventions.md) records the
required header form and the `REUSE.toml` rules for files that cannot carry an inline tag.

## Essential Eigen hazards

### Expressions, lifetimes, and aliasing

Eigen expressions are lazy and frequently retain references. Assignment, construction, coefficient access, reductions,
and `.eval()` can all consume an expression.

- `auto x = A + B;` stores a lazy expression whose references can dangle. When ownership is required, materialize it
  with `(A + B).eval()` or use an appropriate plain-object type such as `Matrix` or `Array`.
- `.noalias()` is a promise, not a runtime check. Use it only when the destination cannot appear in the right-hand side.
  Product evaluation makes `mat = mat * mat` safe; `mat.noalias() = mat * mat` is wrong.
- Prefer Eigen expressions when they express the operation clearly and avoid repeated evaluation. Keep a scalar loop
  when it represents control flow better, avoids an unnecessary temporary, or has measured performance benefits.
- Prefer block and view expressions when a uniform operation or existing Eigen method applies to a submatrix; for
  example, scale a 2-by-2 block or call its `determinant()` instead of spelling out its coefficients. When a block's
  size is known at compile time, preserve that compile-time size with a fixed-size accessor such as `block<Rows,
  Cols>(i, j)`. In dependent template code, write `m.template block<Rows, Cols>(i, j)`. Use runtime sizes only when they
  are genuinely dynamic. Use individual coefficient access when entries require different operations. Blocks remain
  lazy, non-owning views, so the lifetime and overlap rules above still apply.
- The two arms of `?:` must have a common C++ type; distinct Eigen expression types often do not. Use `if`/`else` when
  necessary.
- Declare dynamically sized matrix and vector workspaces outside the loop that fills them. A plain object declared
  inside the loop body allocates on every iteration. So does every subexpression in the loop body that materializes a
  temporary.

### Scalar, index, and storage genericity

Use `Eigen::Index` for dimensions and counts, but remember that its underlying type is configurable. Use `NumTraits` for
scalar properties, and Eigen's `numext` helpers when the code must support custom scalars or device code. Do not:

- store sizes or loop counts in `Scalar`;
- hard-code `float`/`double` without an API reason;
- narrow a value to the `int` a vendor API takes without checking the range.

Test the real, complex, integer, and narrow or custom scalar types that the operation's documented domain covers.
Commutativity, exactness, the tie behavior of `min`/`max`, and other algebraic properties of the built-in types need
not hold for every `Scalar`. Establish such a property for each scalar category, and leave custom scalars on the
conservative path.

Propagate storage-order and expression flags deliberately. `RowMajorBit`, fixed versus dynamic dimensions, alignment,
and vectorization eligibility affect evaluators and fast paths. Eigen alignment depends on configuration and
architecture; do not hard-code an assumed byte value. When configuration-sensitive behavior changes semantics or ABI,
cover it in tests.

### Public APIs and diagnostics

For generic APIs, accept the least restrictive established Eigen base (`EigenBase`, `DenseBase`, `MatrixBase`,
`ArrayBase`, or a suitable `Ref`) that preserves the intended semantics. When the function writes to an expression
argument, follow the established patterns nearby. Do not cast away constness from genuinely const storage. When you add
a non-template definition or object to a public header and an ODR regression is plausible, add a link test that
includes the header from multiple translation units.

Pass fixed-size vectorizable Eigen objects by reference rather than by value. C++17 passes over-aligned objects by value
safely, but the supported C++14 configurations cannot rely on that. See
[`doc/PassingByValue.dox`](doc/PassingByValue.dox).

Use `eigen_assert` for runtime preconditions that are part of Eigen's public debug behavior, and `eigen_internal_assert`
for internal invariants, which are checked only when `EIGEN_INTERNAL_DEBUGGING` is defined. Use the local compile-time
assertion style that gives the clearest diagnostic.

Comments should explain non-obvious mathematics, invariants, compatibility constraints, or provenance rather than
narrating the code. Keep comments concise and proportional to the code's complexity. Avoid tutorial-style prose,
section-by-section narration, and comments that restate identifiers or control flow. Longer comments are justified only
when their rationale cannot be expressed clearly in code. Reviewers here read mathematics and code faster than English:
where a formula, a recurrence, an error bound, or two lines of pseudo-code state the point more precisely than a
paragraph, write that instead. The same preference applies to merge request descriptions and review comments;
[`.agents/merge-requests.md`](.agents/merge-requests.md) records the KaTeX syntax GitLab renders.

## Quick build and test

By default, tests are not part of the `all` target, although that target can build configured auxiliary libraries. A
typical focused workflow is:

```bash
cmake -G Ninja -S . -B build
cmake --build build --target <test-name>
ctest --test-dir build -R '^<test-name>$' --output-on-failure --no-tests=error
```

For one part of a split test, such as `foo_3`, build that exact target and match it exactly with CTest. Keep
`--no-tests=error`, because without it a filter that matches nothing exits 0. The generated `buildtests.sh` and
`check.sh` wrappers accept regexes over source or test names and are useful for building all matching parts. Use
`buildtests`, `BuildOfficial`, `BuildContrib`, `buildsmoketests`, or `check` only when the requested validation warrants
that scope. See [`.agents/testing.md`](.agents/testing.md) for the current test framework, split rules, configuration
variants, and failure-test workflow.

## Completion checklist

Before declaring the task complete:

- The diff contains only intentional changes and preserves pre-existing work.
- New public implementation is reachable through the intended umbrella header.
- New files have correct REUSE metadata and no generated or local-tool files are staged.
- Changed source lines and task-created source files pass the clang-format-17 checks in `.agents/ci.md`;
  `git diff --check` is clean.
- Documentation of the changed behavior is updated with it: the Doxygen block above a changed declaration, the module
  `README`, and nearby comments that name a value or precondition the change moved.
- Focused regression tests pass, with broader tests or benchmarks run when the risk warrants them.
- Numerical, aliasing, scalar, storage-order, device, threading, and ABI implications have been considered where
  relevant.
- The final report names validation performed, residual risk, and anything that could not be tested locally.

Prefer commit subjects of the form `Category: Short description`, for example
`Core: Fix alias handling in product assignment`.
