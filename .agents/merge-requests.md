# Merge Request Descriptions

Use this guide when writing or updating a merge request description; [`review-response.md`](review-response.md)
covers the review round that follows. A description is read by a reviewer who knows the code and wants to judge the
change, not relive it.

Lead with two to four plain sentences before any heading: what the change does and the headline outcome or number.
The first sentence continues the title rather than restarting it. Opening with a heading is the most common defect.

Use `###` headings, numbered lists with one clause per distinct change, tables, and code blocks where they mark a real
division, not as decoration. The template headings (`### Reference issue`, `### What does this
implement/fix?`, `### Additional information`) fit when there is an issue to reference. Avoid a heading per paragraph
and sections that re-explain the diff line by line.

State costs as flatly as wins. Say without hedging what is left undone. When validation is incomplete, open the request
as a Draft and name what was not run and why.

Long accounts of approaches tried and dropped are discouraged but not banned. Two kinds are worth reporting, because
each saves a review round: a rejected alternative that a reviewer would otherwise propose, and a measurement that rules
out an obvious design. State the conclusion and the evidence for it, not the chronology. Make headline comparisons
against the target branch. Give numbers for a superseded variant only where they support such a conclusion.

Prefer notation to prose: a bound, a recurrence, an identity, or two lines of pseudo-code stated exactly beats the
paragraph that spells it out. When you name a theorem, give its statement. Do not invent a symbol for a single sentence.

GitLab renders KaTeX in descriptions and comments. Inline math takes dollar-backtick delimiters
(``$`\|AX - B\|_F \le c\,n\,\varepsilon\,\|A\|_F\,\|X\|_F`$``); bare `$...$` does not render on gitlab.com. Display
equations go in a fenced ` ```math ` block. Identifiers that name actual code (`eps`, `numext::maxi`, `nrhs`) stay in
code spans. Do not dress a code-level statement up in LaTeX.

Show headline before/after measurements in a clear table near the top of performance merge requests. Include the
measured cases, units, and speedup or change; report regressions as clearly as improvements.

Put bulk evidence, and only bulk evidence, in a collapsible appendix: benchmark tables, validation matrices, ULP
sweeps, exhaustive case enumerations. Keep the reasoning the reviewer needs in order to judge the change on the page.
Use tables for benchmark measurements in appendices too, never raw benchmark dumps. Generate tables from the recorded
data and retain available variability and statistical information. Raw artifacts may accompany the tables but not
replace them.

```markdown
<details>
<summary>Appendix A: AVX2 benchmark numbers</summary>

| Case | Before (ns) | After (ns) | Change |
|---|---:|---:|---:|
| ... | ... | ... | ... |

</details>
```

The blank line after `</summary>` is required for GitLab to render the inner markdown. If a description is long because
the change is too large to review, split the change rather than collapsing text into an appendix.

Give each number its provenance: the exact expression, operand types and sizes, compiler and flags, and the CPU as the
OS reports it. On Linux, use the `Model name:` line of `lscpu`, which decodes the Arm implementer and part codes that
`/proc/cpuinfo` shows raw. If an old util-linux prints no model name, give those codes instead. On macOS, use
`sysctl -n machdep.cpu.brand_string`. Disclose a virtualized host such as WSL2, where those commands report whatever the
hypervisor exposes.

Credit reporters and contributors by name or handle, and link the issue with `Closes #NNNN`. GitLab's closing pattern is
blind to negation, so "does not fix #NNNN" still closes the issue on merge. To reference an issue without closing it,
write "Related to #NNNN".

The body and its numbers always describe the current head. After a review round in which the code or its performance
changed materially, rewrite the body and its numbers rather than layering corrections on the old text. Keep the commit
messages current too. A short `Update:` paragraph or changelog at the end credits the reviewers and gives a brief
history of what changed.
