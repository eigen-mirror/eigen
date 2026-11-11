# Responding To Review

Use this guide when answering merge request review comments.

A code suggestion posted in review is a sketch that has not been compiled. Verify it like your own work before adopting
it. In particular, check that it:

- compiles under the C++14 baseline;
- uses matching `Matrix`/`Array` and expression types;
- keeps any grouping of operations chosen deliberately for numerical reasons.

Reproduce a claimed defect before fixing it. Judge the suggested remedy separately from the finding: a real bug often
arrives with a fix that breaks cases the current code handles.

Hold your own replies to the same standard: rule 9 in [`AGENTS.md`](../AGENTS.md#non-negotiable-rules) covers every
claim about what the code does.

Address every thread: apply the suggestion or explain the deviation, and name the commit that resolved the thread. Keep
the response within the comment's scope. When the review exposes a defect in shared code, fix it in its own commit or
merge request. After each round, re-verify that the merge request description and commit messages still describe the
current head.

[`.coderabbit.yaml`](../.coderabbit.yaml) holds the path-specific instructions that the CodeRabbit review bot applies
when it is enabled on the project. They are the bot's version of these conventions and predict what an automated review
will flag.

Typeset real mathematics in comments — bounds, recurrences, identities, error terms — as KaTeX, in the form
[`merge-requests.md`](merge-requests.md) records for descriptions.
