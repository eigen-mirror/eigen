# Merge Request Descriptions

Use this guide when writing or updating a merge request description. Answering review comments is covered by
[`review-response.md`](review-response.md).

A description is read by a reviewer who knows the code and wants to judge the change, not relive it. Match this
shape:

1. **Lead with two to four plain sentences before any heading**: what the change does and the headline outcome or
   number. The first sentence continues the title rather than restarting it. Opening with a heading is the most
   common defect.
2. **Structure divides, it does not decorate.** `###` headings, numbered lists with one clause per distinct change,
   tables, and code blocks are welcome where they mark real divisions. The template headings (`### Reference issue`,
   `### What does this implement/fix?`, `### Additional information`) fit when there is an issue to reference. Avoid a
   heading per paragraph and sections that re-explain the diff line by line. State costs as flatly as wins and say
   what is left undone without hedging. Open the request as a Draft when validation is incomplete, and name what was
   not run and why.
3. **Prefer notation to prose.** Reviewers read mathematics and code faster than English: a bound, a recurrence, an
   identity, or two lines of pseudo-code stated exactly beats the paragraph that spells it out, and a named theorem
   comes with its statement. GitLab renders KaTeX in descriptions and comments: inline math uses dollar-backtick
   delimiters (``$`\|AX - B\|_F \le c\,n\,\varepsilon\,\|A\|_F\,\|X\|_F`$``; bare `$...$` does not render on
   gitlab.com), display equations go in a fenced ` ```math ` block. Identifiers that name actual code (`eps`,
   `numext::maxi`, `nrhs`) stay in code spans; do not dress a code-level statement up in LaTeX, and do not invent a
   symbol for a single sentence.
4. **Bulk evidence goes in a collapsible appendix, and only bulk evidence**: benchmark tables, validation matrices,
   ULP sweeps, exhaustive case enumerations. Reasoning the reviewer needs in order to judge the change stays on the
   page. Paste raw benchmark output in a fenced block with its variance and p-value columns intact rather than
   retyping the numbers.

   ```markdown
   <details>
   <summary>Appendix A: AVX2 benchmark numbers</summary>

   ...raw output...
   </details>
   ```

   The blank line after `</summary>` is required for GitLab to render the inner markdown. A description that is
   long because the change is too large to review needs a split, not a fold.

When numbers appear, name what was measured — the exact expression, operand types and sizes, compiler and flags —
and name the CPU as the OS reports it: the `Model name:` line of `lscpu` on Linux, which decodes the Arm
implementer/part codes that `/proc/cpuinfo` leaves raw (fall back to those codes when an old util-linux prints
none), or `sysctl -n machdep.cpu.brand_string` on macOS. Note a virtualized environment such as WSL2, which reports
whatever the hypervisor exposes. Credit
reporters and contributors by name or handle and link the issue with `Closes #NNNN`. GitLab's closing pattern has
no negation handling, so "does not fix #NNNN" still closes the issue on merge; reference without closing as
"Related to #NNNN". After a review round, append an `Update:` paragraph crediting the reviewer instead of silently
rewriting the body, and keep the description and the commit messages describing the current head.
