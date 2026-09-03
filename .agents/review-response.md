# Responding To Review

Use this guide when answering merge request review comments.

A posted code suggestion is a sketch that has not been compiled; verify it like your own work before adopting it —
including the C++14 baseline, `Matrix`/`Array` and expression-type mismatches, and numerically deliberate groupings.
Reproduce a claimed defect before fixing it, and judge the suggested remedy separately from the finding: a real bug
often arrives with a fix that breaks cases the current code handles. Hold your own claims to the same standard —
a behavioral claim is established by reading the function body and the branch actually taken, never by a header's
own Doxygen, which can be stale or describe an adjacent case; confirming that a path or symbol exists proves nothing
about behavior, and universals need enumeration rather than inference from a few instances.
Address every thread: apply the suggestion or explain the deviation, naming the commit that resolved it. Keep the
response within the comment's scope; a defect it exposes in shared code belongs in its own commit or merge request.
After each round, re-verify that the merge request description and commit messages still describe the current head.

[`.coderabbit.yaml`](../.coderabbit.yaml) holds the path-specific instructions the CodeRabbit review bot applies when
it is enabled on the project; they are the bot's rendering of these conventions and predict what an automated review
will flag.

Typeset real mathematics in comments — bounds, recurrences, identities, error terms — the way
[`merge-requests.md`](merge-requests.md) prescribes for descriptions: KaTeX inline with dollar-backtick delimiters
(``$`h_j = \varepsilon\,\max(|x_j|, 1)`$``; bare `$...$` does not render on gitlab.com), display equations in a fenced
` ```math ` block, and identifiers that name actual code kept in code spans.
