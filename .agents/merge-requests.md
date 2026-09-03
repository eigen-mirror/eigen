# Merge Request Descriptions

Use this guide when writing or updating a merge request description; [`review-response.md`](review-response.md)
covers the review round that follows. A description is read by a reviewer who knows the code and wants to judge the
change, not relive it.

Lead with two to four plain sentences before any heading: what the change does and the headline outcome or number.
The first sentence continues the title rather than restarting it. Opening with a heading is the most common defect.

Structure divides; it does not decorate. `###` headings, numbered lists with one clause per distinct change, tables,
and code blocks belong where they mark a real division, as the template headings (`### Reference issue`,
`### What does this implement/fix?`, `### Additional information`) do when there is an issue to reference. Avoid a
heading per paragraph and sections that re-explain the diff line by line. State costs as flatly as wins and say what
is left undone without hedging; open the request as a Draft when validation is incomplete, naming what was not run
and why.

Prefer notation to prose: a bound, a recurrence, an identity, or two lines of pseudo-code stated exactly beats the
paragraph that spells it out, and a named theorem comes with its statement. GitLab renders KaTeX in descriptions and
comments — inline math takes dollar-backtick delimiters (``$`\|AX - B\|_F \le c\,n\,\varepsilon\,\|A\|_F\,\|X\|_F`$``;
bare `$...$` does not render on gitlab.com), display equations go in a fenced ` ```math ` block. Identifiers that name
actual code (`eps`, `numext::maxi`, `nrhs`) stay in code spans; do not dress a code-level statement up in LaTeX, and do
not invent a symbol for a single sentence.

Bulk evidence goes in a collapsible appendix, and only bulk evidence: benchmark tables, validation matrices, ULP
sweeps, exhaustive case enumerations. Reasoning the reviewer needs in order to judge the change stays on the page.
Paste raw benchmark output in a fenced block, variance and p-value columns intact, rather than retyping the numbers.

```markdown
<details>
<summary>Appendix A: AVX2 benchmark numbers</summary>

...raw output...
</details>
```

The blank line after `</summary>` is required for GitLab to render the inner markdown. A description that is long
because the change is too large to review needs a split, not a fold.

Numbers carry their provenance: the exact expression, operand types and sizes, compiler and flags, and the CPU as the
OS reports it — `lscpu`'s `Model name:` on Linux, which decodes the Arm implementer/part codes `/proc/cpuinfo` leaves
raw (fall back to those codes when an old util-linux prints none), or `sysctl -n machdep.cpu.brand_string` on macOS.
Disclose a virtualized host such as WSL2, where those commands report whatever the hypervisor exposes.

Credit reporters and contributors by name or handle, and link the issue with `Closes #NNNN`. GitLab's closing pattern
is blind to negation, so "does not fix #NNNN" still closes the issue on merge; reference without closing as
"Related to #NNNN". After a review round, append an `Update:` paragraph crediting the reviewer instead of silently
rewriting the body, and keep the description and the commit messages describing the current head.
