# Responding To Review

Use this guide when answering merge request review comments.

A code suggestion posted in review is a sketch that has not been compiled. Verify it like your own work before adopting
it. In particular, check it against the C++14 baseline, that its `Matrix`/`Array` and expression types match, and that
it keeps any grouping of operations chosen deliberately for numerical reasons. Reproduce a claimed defect before fixing
it, and judge the suggested remedy separately from the finding: a real bug often arrives with a fix that breaks cases
the current code handles.

Hold your own claims to the same standard. To establish what code does, read the function body and the branch actually
taken. Never rely on a header's own Doxygen for this; it can be stale or describe an adjacent case. Confirming that a
path or symbol exists proves nothing about behavior. A claim about every case needs the cases enumerated, not inferred
from a few instances.

Address every thread: apply the suggestion or explain the deviation, naming the commit that resolved it. Keep the
response within the comment's scope. A defect in shared code that the review exposes belongs in its own commit or merge
request. After each round, re-verify that the merge request description and commit messages still describe the current
head.

[`.coderabbit.yaml`](../.coderabbit.yaml) holds the path-specific instructions that the CodeRabbit review bot applies
when it is enabled on the project. They are the bot's version of these conventions and predict what an automated review
will flag.

Typeset real mathematics in comments — bounds, recurrences, identities, error terms — as KaTeX, in the form
[`merge-requests.md`](merge-requests.md) records for descriptions.
