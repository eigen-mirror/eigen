# Local Conventions For New Code

Use this guide when writing new declarations anywhere in the tree. It records the forms reviewers ask for. It does not
authorize rewriting lines your task leaves alone; rule 5 in the repository-root `AGENTS.md` covers those. Eigen
predates most of these forms, so the most common form in the tree is not the convention. Write new code in the current
form. A file that you edit heavily should end up uniform rather than half converted.

## Declarations

- Declare trait and evaluator constants as `static constexpr` members, not `enum` blocks; `enum` constants are being
  phased out. Give each the type it is used as: `Flags` is `unsigned int` by convention, predicates are `bool`. In C++14
  the in-class declaration is not a definition. If code odr-uses such a member of a class template, the member also
  needs a namespace-scope definition, `template <...> constexpr T Cls<...>::kName;` (`arch/Default/Half.h` has the
  form). Without it, the code links at -O2 but fails to link at -O0. Binding the member to a `const T&` parameter, such
  as `numext::mini`'s, odr-uses it, and so does taking its address. Pass a prvalue (`+kName`, `Index(kName)`) or add the
  definition. The test suite builds optimized, so it will not catch the omission.
- Prefer `using` to `typedef`, `nullptr` to `NULL`, `= default` and default member initializers to empty constructor
  bodies that assign each member. The `using` rule applies everywhere, `test/` and `contrib/` included. The sweep that
  converted `Eigen/src` skipped those directories, so most aliases near new code there are still `typedef`. Copying the
  neighbors reproduces the form the sweep removed. Do not rely on CI to catch it: clang-tidy's `modernize-use-using`
  check does not report function-local typedefs, a gap recorded in
  [`scripts/check_style.py`](../scripts/check_style.py).
- `kCamelCase` is an accepted spelling for `static constexpr` and static constants, alongside the older `snake_case`
  and `SCREAMING_CASE` forms. It is not a review finding.
- Use `numext::` math functions rather than `std::` in library code. Use Eigen's metaprogramming aliases
  (`bool_constant`, `void_t`, `remove_all_t`; see `Eigen/src/Core/util/Meta.h`) rather than spelling out the standard
  forms. `internal::is_arithmetic` is not a spelling of `std::is_arithmetic`: it is deliberately specialized for
  packet and Eigen scalar types and differs on `long double` during GPU compilation, so use it only when Eigen's
  extended arithmetic category is specifically intended.
- Put SFINAE in a defaulted template parameter rather than the return type. When an overload set needs the negative
  case too, constrain both overloads: an exact-match overload next to an unconstrained one can bind a converted
  temporary and return a dangling reference.
- An in-class definition is already implicitly `inline`; a bare `inline` there is noise. Use `EIGEN_STRONG_INLINE` or
  `EIGEN_ALWAYS_INLINE` when inlining matters, and nothing otherwise.
- Check compile-time API preconditions with the `EIGEN_STATIC_ASSERT_*` helper that names them (`_VECTOR_ONLY`,
  `_SAME_MATRIX_SIZE`, ...) or with `EIGEN_STATIC_ASSERT(cond, TOKEN)`. These honor `EIGEN_NO_STATIC_ASSERT` and a
  user-provided `EIGEN_STATIC_ASSERT` override. Write unconditional implementation invariants as
  `static_assert(cond, "what must hold")`; a plain `static_assert` bypasses those configuration mechanisms.
- Deprecate, do not remove: mark the old declaration `EIGEN_DEPRECATED` or `EIGEN_DEPRECATED_WITH_REASON("use ...")`,
  keep it working by forwarding to the replacement, and name the replacement in its Doxygen block.
- Spell names out (`scratch`, not `scr`) and name traits for the property they assert.

## Scalar, index, and storage genericity

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

## Public APIs and diagnostics

For generic APIs, accept the least restrictive established Eigen base (`EigenBase`, `DenseBase`, `MatrixBase`,
`ArrayBase`, or a suitable `Ref`) that preserves the intended semantics. When the function writes to an expression
argument, follow the established patterns nearby. Do not cast away constness from genuinely const storage. When you add
a non-template definition or object to a public header and an ODR regression is plausible, add a link test that
includes the header from multiple translation units.

Pass fixed-size vectorizable Eigen objects by reference rather than by value. C++17 passes over-aligned objects by value
safely, but the supported C++14 configurations cannot rely on that. See
[`doc/PassingByValue.dox`](../doc/PassingByValue.dox).

Use `eigen_assert` for runtime preconditions that are part of Eigen's public debug behavior, and `eigen_internal_assert`
for internal invariants, which are checked only when `EIGEN_INTERNAL_DEBUGGING` is defined. Use the local compile-time
assertion style that gives the clearest diagnostic.

## The C++14 baseline

Supported headers compile as C++14, which rules out forms that review suggestions often reach for:

- `if constexpr` is C++17: use `EIGEN_IF_CONSTEXPR(...)` wherever the condition is compile-time constant. Under C++14
  the macro expands to a plain `if`, so the code must still be valid C++14.
- Designated initializers are C++20: use aggregate assignment with `/*name=*/` comments. When a field is derived from
  other fields of the same object, compute it from locals first: the braced temporary cannot read fields it is about to
  set.
- `std::span`, CTAD, fold expressions, `constinit`, and later library additions are unavailable outside guarded
  backends with a documented newer requirement (the SYCL configurations force C++17, for example).

## Hot paths

Code added to a hot inner loop makes the enclosing function bigger. The bigger function can fall out of the instruction
cache even when an `EIGEN_PREDICT_FALSE` guard keeps the new code from running. Watch for this when you add a check or
a fallback to such a loop. If a benchmark shows the cost, one way to recover it is to move the cold path into an
`EIGEN_DONT_INLINE` helper.

## Comments

Reviewers enforce the comment rules in the repository-root `AGENTS.md`, and breaking them is the most common style
finding here. Before publishing a diff, reread each added comment and delete the ones that narrate code or restate an
identifier. Keep the ones recording mathematics, invariants, compatibility constraints, provenance, or the reason a
slower or unusual form is deliberate. Put them at the code they explain, not in the merge request.

Prefer the most precise notation that fits. A recurrence, an error bound, an invariant written as an expression, or
two lines of pseudo-code usually carry more than a paragraph and are read faster by this audience:

```cpp
// Bad: the relative error in summing n elements this way is bounded by roughly twice the
// machine epsilon multiplied by the quantity log base two of n over B, plus B, where B is
// the number of elements summed sequentially in each leaf of the tree.

// Good: tree summation, relative error <= ~2*eps*(log2(n/B) + B) for leaf size B.
```

An exact bound, invariant, or identity beats prose that conveys it only in part (`m` kept in `[1, 2)` says more than
"balanced form"). A comment that names a theorem also states it. Reuse the symbols of the surrounding file and the
cited reference rather than inventing new ones for a single sentence. Notation that restates what the code shows is
the same defect as prose that does. Prose remains the tool for a *reason*: why this form and not the obvious one.
Comments are plain text, so write expressions the way the rest of the tree does, not in a markup language that does
not render.

## REUSE metadata for new files

Every new source file needs accurate REUSE metadata. Original Eigen code normally uses MPL-2.0; prefer the collective
form when an agent cannot truthfully attribute an individual author:

```cpp
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
```

Use the language's comment syntax. Documentation or assets that should not carry inline tags must be covered precisely
in [`REUSE.toml`](../REUSE.toml); do not add a broad annotation that hides unrelated files. Material adapted from a
compatibly licensed source can require a different license expression and attribution. Preserve them rather than
relabeling the material as MPL-2.0.
