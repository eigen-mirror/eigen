# Provenance Of Information

Use this guide when comparing Eigen against, benchmarking against, or integrating with software Eigen does not own,
and whenever a task would involve looking at how such software works. [`AGENTS.md`](../AGENTS.md) rule 2 is the
contract; [`numerics.md`](numerics.md#provenance) covers citing the literature an implementation is built on.

Eigen aims to be the best library it can be, in speed, accuracy and everything else, by legal and ethical means, from
original research and publicly available information, and by no other. Copyright protects expression, not ideas, so
the concern here is how an idea was obtained: looking inside a vendor's software can breach its license, and using
what was learned can support a trade-secret claim even when nothing is copied. Eigen treats code informed that way as
tainted.

## Original Research Is Encouraged

Measure the hardware with your own microbenchmarks, measure other libraries from outside, run numerical experiments,
design new algorithms, and derive new bounds. The rest of this guide limits how you learn about software Eigen does not
own, not what you discover yourself. The condition is that you have the right to contribute the result under Eigen's
license: no NDA or other confidentiality terms bind it, and whoever owns the work, often an employer, has agreed.

## Proprietary Software Is A Black Box

Proprietary libraries, such as Intel oneMKL, NVIDIA's cuBLAS, Arm Performance Libraries, and Apple Accelerate, are
used through their documented interface and measured as shipped. Eigen draws its line at the black box whatever a
particular license or jurisdiction permits, and the line does not move when someone else has already published what is
inside: the vendor's documentation defines the supported interface, not a forum post. If a comparison cannot be
configured through documented means, report the limitation.

| Allowed: observe from outside | Not allowed: look inside |
|---|---|
| Link against the library, call its documented API, time it, compare its numerical results | Disassemble or decompile it (`objdump -d`, `otool -tv`, `cuobjdump`, Ghidra, IDA, radare2) |
| Read its published documentation, headers, release notes, application notes, and the vendor's papers and talks | Dump its strings or bytes (`strings`, `xxd`, `grep -a`) to find undocumented switches, kernel names, or dispatch tables |
| Set documented environment variables and verbose modes | Step a debugger through it, or `perf annotate` its code |
| Run it as shipped, configured only through its documented interface | Change how it runs by other means: override or interpose its symbols, patch it, or set switches its vendor does not document |
| Diagnose linking: `ldd`, `nm -D`, `otool -L`, `readelf -d`, `objdump -p` | Read confidential source or information: leaked, under NDA, or from a current or former employer |
| Profile at function granularity to see how much time is spent in the library as a whole | Use third-party write-ups you know or have reason to believe obtained their content by any of the above |

The left column is the most Eigen allows, not a license to do it. A vendor's terms can forbid more, for example by
granting use only for internal evaluation, by treating the software and information about it as confidential, or by
restricting published benchmark results. Those terms bind whoever accepted them; read them before publishing
measurements.

A proprietary library linked statically into one of Eigen's own benchmark binaries is still inside the box: restrict
disassembly and annotation to Eigen's symbols. Eigen's own functions stay open to inspection when the compiler has
inlined code from a vendor's headers or device libraries into them; analyze Eigen's code there, not the inlined vendor
code.

## Open-Source Software

Software published under an open-source license, such as OpenBLAS, BLIS, and reference LAPACK, may be read and
disassembled. Two cautions:

- Public is not the same as compatible. Reading GPL or LGPL code is allowed, but copying, paraphrasing, or translating it
  into Eigen is not; see rule 2 and [`CONTRIBUTING.md`](../CONTRIBUTING.md#provenance-and-attribution).
- A binary a vendor ships under its own license stays inside the box even when the vendor also publishes its source. To
  look inside, read or build the published source.

## Show Where The Change Comes From

A merge request description should link the publicly available documentation that supports the change: the ISA or
architecture manual, the vendor's optimization guide or intrinsics reference, the paper or standard an algorithm
follows, the documented API a backend relies on. Where the change rests on original research, include the evidence
instead: the reproducer, the method and results of a measurement, or the derivation. This helps most with
hardware-specific optimizations and new features, where it lets a reviewer check the change against its sources and
see that everything it relies on is public or published with it.

## If You Are Exposed Anyway

Stop, do not act on what you saw, and tell the user what was seen and how. Do not record it in code, comments, commit
messages, merge request descriptions, benchmark write-ups, or agent memory. If it has already been committed, pushed,
or published, say exactly where; the maintainers decide how far back to remove it and whether the affected work needs
a different author or a fresh start. If a vendor has raised the matter, preserve everything and get legal advice
before removing anything.
