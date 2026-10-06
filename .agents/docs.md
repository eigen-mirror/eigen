# Documentation

Use this guide when editing a Doxygen block, a page under [`doc/`](../doc), a snippet or example, or a documented public
name. The documentation consists of the Doxygen comments in the headers, the topic pages in `doc/*.dox`, and the
programs under [`doc/snippets`](../doc/snippets), [`doc/examples`](../doc/examples) and their `contrib/doc`
counterparts. The `doc` target compiles and runs those programs, and the pages embed their output. Keep the Doxygen
block above a changed declaration describing the current behavior, preconditions, and return value. When a module
`README` names a value that the change alters, update the `README` too.

## The Blocking Job

The documentation job is blocking and easy to miss. Unlike the clang-format, codespell, and clang-tidy jobs,
`build:linux:docs` in [`ci/build.linux.gitlab-ci.yml`](../ci/build.linux.gitlab-ci.yml) is not `allow_failure`.
[`doc/Doxyfile.in`](../doc/Doxyfile.in) sets `WARN_AS_ERROR = FAIL_ON_WARNINGS_PRINT`, so one Doxygen warning fails it.
The job does not run in the default merge-request pipeline. It runs on schedules, web pipelines, a merge request
labeled `docs-build` or `all-tests`, and a push to the default branch. A malformed `\ref` can therefore pass review
with green CI and then break the pipeline on `master` after the merge. For changes to Doxygen markup, a
cross-reference target, a documented name, a module `README`, or a snippet, apply `docs-build`. That label runs only
this job and leaves the test tier unchanged, so it can be combined with `affected-tests`. A local `doc` build is weaker
evidence, because local Doxygen versions resolve some references that CI's pinned version rejects. If you build
locally instead, use that pinned version and report the result.

Recommend `affected-tests` with the relevant platform labels, or `affected-tests` with `all-platforms`, for test
coverage as described in [`ci.md`](ci.md). Of the test labels, only `all-tests` also runs `build:linux:docs`. Do not
add it for that purpose, and do not add it at all without the user's explicit permission for that label.

The recurring authoring mistake is trailing punctuation that Doxygen reads as part of a cross-reference. A colon
directly after `\ref name` becomes part of the symbol Doxygen tries to resolve, so `\ref adjoint: the ...` fails while
`\ref adjoint. The ...` resolves. Separate a reference from following prose with a space, comma, or period. Punctuation
inside the name itself is fine: `\ref MatrixBase::cross()` is a qualified symbol, not a colon attached to a name.

A second way to break the job without editing a comment is to insert a declaration between a Doxygen block and the
entity it describes. A block without a structural command (`\class`, `\fn`, `\ingroup`, ...) documents whatever
declaration follows it. When that declaration is `namespace internal {`, the whole `Eigen::internal` namespace becomes
documented. Every internal doc block then enters the output, and any `\param` mismatch hidden in those blocks fails
the build far from the edit. For example, commit 8f8d4ed4c placed helper structs under the `Transform::rotate` block,
which exposed a stale `\param` in `GMRES.h`. After inserting code near a doc block, confirm the block still directly
precedes its declaration. If the Doxygen log prints `Generating docs for namespace Eigen::internal`, it does not.

The `doc` target also compiles and runs the configured examples and snippets, by way of the `all_snippets` and
`all_examples` prerequisites in [`doc/CMakeLists.txt`](../doc/CMakeLists.txt). A renamed or removed public name breaks
the documentation build even when every comment is well formed, so search those directories before changing one.
Only the *configured* programs are built. For example, `contrib/doc/examples/CMakeLists.txt` adds its `SYCL`
subdirectory only when `EIGEN_TEST_SYCL` is set, and `build:linux:docs` does not set it, so a broken contrib SYCL
example leaves this target green. Treat the target as coverage for the sets the configuration actually enables, and
check the CMake condition before citing it as coverage.

## Building Locally

`EIGEN_BUILD_DOC` defaults on for a top-level, non-cross-compiling configuration, but `doc` is excluded from `all` and
must be named:

```bash
cmake --build build --target doc
```

Doxygen and graphviz must be installed. CI builds a pinned Doxygen from source
([`ci/scripts/build_and_install_doxygen.sh`](../ci/scripts/build_and_install_doxygen.sh)), so another local version can
diagnose a different set of warnings; report the version that produced a local result.
