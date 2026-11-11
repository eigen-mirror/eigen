# Writing Agent Guidance

Use this guide when adding to or editing `AGENTS.md` or a guide under `.agents/`. Agents read these files literally
and apply them to cases their authors did not foresee. The rules below adapt a few rules of ASD-STE100, Simplified
Technical English, to that reader. Its controlled dictionary does not apply: a model already knows the vocabulary. It
misreads guidance when a sentence buries an instruction or a condition, not when it uses a rare word.

## Instructions

- Write one instruction per sentence. The sentence can also carry the instruction's condition and its reason. A reader
  tends to drop the second of two instructions in one sentence, so split the sentence or use a list.
- When a condition decides whether a rule applies, put the condition first: "When a change touches `Eigen/src/Core`,
  ...". A reader whose task does not match can then skip the rule.
- Write instructions in the imperative. An instruction phrased as a description ("the addition deserves a link test")
  leaves the reader to infer the action that "add a link test" states.
- Use a numbered list for steps whose order matters, and a bulleted list for three or more parallel items.
- Give each paragraph one topic. A paragraph of more than six sentences usually holds two.

## Strength And Reasons

- Set each rule's strength with the words that [`AGENTS.md`](../AGENTS.md#scope-and-precedence) defines. Before you
  write a requirement, ask what it would block on a day when someone had a good reason. If the answer is nothing anyone
  would want, write the requirement. Otherwise write the default and the cost of departing from it, so that a reader can
  tell when the cost does not apply.
- Put a rule's reason after the rule, and keep it. An agent extends a rule correctly to an unforeseen case only when it
  knows why the rule exists. A reason must not carry a new instruction.
- When a requirement exists because the failure already happened here, say so. A bare requirement reads as taste, and
  the next editor softens it.

## Words And Length

- Write each sentence so that a reader takes it in on one pass: a subject, a concrete verb, and at most one clause hung
  on either. "Hashing the YAML as well only meant that every merge request that touched the CI YAML discarded every
  job's recorded passes" needs a second reading. "If the key included the YAML files, any edit to them would discard
  all recorded passes" does not.
- Use one term for one concept across all the guides. A second term reads as a second concept.
- Reserve *may* for permission. Write *can* or *might* for possibility.
- Use the active voice when the actor matters: "CI runs the job" names the actor, and "the job is run" does not.
- Treat a sentence of more than about 25 words as a signal, not a limit. It usually holds two instructions or an inline
  list; split it, or move the list into bullets.

## What To Write, And Where

- Before you add a sentence, ask whether a capable model would get this wrong in this repository without it. Keep what
  the tree cannot tell the reader: a failure that already happened here, a constraint whose reason is not visible, a
  fact about CI or a downstream user. Leave out advice about C++, benchmarking, or numerical analysis that holds in
  any repository; the reader has it already, and it buries the sentences that matter.
- State each rule in one guide, the guide for the task that needs it. Elsewhere, link to that statement. A second copy
  drifts from the first, and a reader cannot tell which one is current.
- `AGENTS.md` is loaded for every task, so keep it to what every task needs. Put the rest in the topic guide for the
  tasks that need it. Keep `AGENTS.md` under 24 KiB: Codex stops loading project instructions at 32 KiB by default, and
  a user-level instruction file counts toward the same limit.
- When a procedure needs more than a few lines of shell, check it in as a script and name the script. A script can be
  run and tested. A recipe in prose is copied with mistakes, and nothing notices when it stops working.
- Name a file by its repository path, in a code span or a link.
  [`scripts/check_agent_guides.py`](../scripts/check_agent_guides.py) checks that those paths and link targets exist,
  and that `AGENTS.md` stays under its size limit. It cannot check a file that is named only in prose.

## Loading A Guide

A guide helps only the sessions that read it. Two mechanisms get it read:

- The work-area table in [`AGENTS.md`](../AGENTS.md#scope-and-precedence) is in every session's context. Every harness
  depends on it. When you add a guide, add its row.
- A pointer under `.claude/rules/` names, in its `paths:` list, the files that a guide governs. Claude Code shows the
  pointer to the model when it first reads a matching file. When you add a guide, add its pointer. Write the pointer
  as a link, never as an `@` import. Claude Code loads an import at session start even when the rule that holds it is
  path-scoped (measured with CLI 2.1.322), so an import would put the guide in every session's context.

Do not add an `AGENTS.md` below the repository root as a third mechanism. Codex reads one only when the session starts
in that directory or below it. Claude Code skips one when the checkout has a `CLAUDE.md` (measured with CLI 2.1.322).
Under `Eigen/`, the file would also be installed with the headers and read by Doxygen.
