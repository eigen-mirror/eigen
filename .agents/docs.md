# Documentation

Use this guide when editing a Doxygen block, a page under [`doc/`](../doc), a snippet or example, or a documented public
name. The documentation is the Doxygen comments in the headers, the topic pages in `doc/*.dox`, and the programs under
[`doc/snippets`](../doc/snippets), [`doc/examples`](../doc/examples) and their `contrib/doc` counterparts, which the
`doc` target compiles and runs to produce the output the pages embed. Keep the Doxygen block above a changed
declaration describing the current behavior, preconditions, and return value, and give a module `README` that names a
moved value the same update.

## The Blocking Job

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

The `doc` target also compiles and runs the configured examples and snippets, by way of the `all_snippets` and
`all_examples` prerequisites in [`doc/CMakeLists.txt`](../doc/CMakeLists.txt). A renamed or removed public name breaks
the documentation build even when every comment is well formed, so search those directories before changing one.
"Configured" is the operative word: `contrib/doc/examples/CMakeLists.txt` adds its `SYCL` subdirectory only under
`EIGEN_TEST_SYCL`, which `build:linux:docs` does not set, so a broken contrib SYCL example leaves this target green.
Treat the target as coverage for the sets the configuration actually enables, and check the conditional before citing
it as coverage.

## Building Locally

`EIGEN_BUILD_DOC` defaults on for a top-level, non-cross-compiling configuration, but `doc` is excluded from `all` and
must be named:

```bash
cmake --build build --target doc
```

Doxygen and graphviz must be installed. CI builds a pinned Doxygen from source
([`ci/scripts/build_and_install_doxygen.sh`](../ci/scripts/build_and_install_doxygen.sh)), so another local version can
diagnose a different set of warnings; report the version that produced a local result.
