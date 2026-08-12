# Local Conventions For New Code

Use this guide when writing new declarations anywhere in the tree. It records the forms review asks for. It does not
authorize rewriting code the task is not otherwise changing; rule 5 in the repository-root `AGENTS.md` governs
untouched lines. Eigen predates most of these forms, so a tree-wide count is not the convention: new code uses the
current form, and a file being edited heavily should come out uniform rather than half converted.

## Declarations

- Trait and evaluator constants are `static constexpr` members, not `enum` blocks; `enum` constants are being phased
  out. Give each the type it is used as: `Flags` is `unsigned int` by convention, predicates are `bool`.
- Prefer `using` to `typedef`, `nullptr` to `NULL`, `= default` and default member initializers to empty constructor
  bodies that assign each member.
- Use `numext::` math functions rather than `std::` in library code, and Eigen's metaprogramming aliases
  (`bool_constant`, `void_t`, `remove_all_t`; see `Eigen/src/Core/util/Meta.h`) rather than spelling out the standard
  forms. `internal::is_arithmetic` is not a spelling of `std::is_arithmetic`: it is deliberately specialized for
  packet and Eigen scalar types and differs on `long double` during GPU compilation, so use it only when Eigen's
  extended arithmetic category is specifically intended.
- Put SFINAE in a defaulted template parameter rather than the return type. When an overload set needs the negative
  case too, constrain both overloads: an exact-match overload next to an unconstrained one can bind a converted
  temporary and return a dangling reference.
- An in-class definition is already implicitly `inline`; a bare `inline` there is noise. Use `EIGEN_STRONG_INLINE` or
  `EIGEN_ALWAYS_INLINE` when inlining matters, and nothing otherwise.
- Spell names out (`scratch`, not `scr`) and name traits for the property they assert.

## The C++14 baseline

Supported headers compile as C++14, which rules out forms that review suggestions often reach for:

- `if constexpr` is C++17: use `EIGEN_IF_CONSTEXPR(...)` wherever the condition is compile-time constant. Note the
  condition must still be valid C++14 either way — the macro lowers to a plain `if` there.
- Designated initializers are C++20: use aggregate assignment with `/*name=*/` comments. Fields derived from other
  fields of the same object must be computed from locals first; the braced temporary cannot read fields it is about
  to set.
- `std::span`, CTAD, fold expressions, `constinit`, and later library additions are unavailable outside guarded
  backends with a documented newer requirement (the SYCL configurations force C++17, for example).

## Comments

The comment rules in the repository-root `AGENTS.md` are enforced in review and are the most repeated style finding
here. Before publishing a diff, reread each added comment and delete the ones that narrate code or restate an
identifier. Keep the ones recording mathematics, invariants, compatibility constraints, provenance, or the reason a
slower or unusual form is deliberate — stated at the construct, not in the merge request.
