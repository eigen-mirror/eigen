# Benchmarking

Use this guidance for performance-sensitive changes and benchmark reviews. Performance claims need a benchmark that
ships in the same merge request; correctness tests still ship separately and run before timing.

## Performance Hypothesis

Performance-critical changes should start from an explicit hypothesis about what limits performance and how the
proposed change reduces that cost. Ground the hypothesis in a cost model appropriate to the operation, considering
the relevant computer architecture and compiler code generation. Where useful, use roofline or speed-of-light
analysis to estimate the available improvement.

A lightweight model is often sufficient:

> "This loop is bandwidth-bound; eliminating this temporary removes one write and one read per coefficient."

For this example, assembly analysis can check whether those accesses disappear, while benchmarks test whether the
reduction improves performance at the relevant sizes. Check the model's assumptions, including whether bandwidth
limits the operation, whether the compiler already eliminates the temporary, and how the working set fits in cache.
Scale the analysis to the change; a formal model is not required for every contribution.

## Projects and Builds

The supported and contrib benchmark trees are separate, standalone CMake projects. They are not part of Eigen's
main test build and both require Google Benchmark:

```bash
cmake -G Ninja -S benchmarks -B build-bench -DCMAKE_BUILD_TYPE=Release
cmake --build build-bench --target <benchmark-target>

cmake -G Ninja -S contrib/benchmarks -B build-contrib-bench -DCMAKE_BUILD_TYPE=Release
cmake --build build-contrib-bench --target <benchmark-target>
```

The contrib parent project automatically adds its GPU subtree when it detects `CUDAToolkit`. That configuration
also requires a working CUDA compiler and architecture selection. On a host with only a partial toolkit installation,
configure CPU-only contrib benchmarks with `-DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE` or report the GPU
configuration as unavailable.

Consult [`benchmarks/CMakeLists.txt`](../benchmarks/CMakeLists.txt) and
[`contrib/benchmarks/CMakeLists.txt`](../contrib/benchmarks/CMakeLists.txt) for current targets and compile
settings. CUDA benchmarks also have a standalone project and instructions in
[`contrib/benchmarks/GPU/CMakeLists.txt`](../contrib/benchmarks/GPU/CMakeLists.txt). No CI job builds or runs
benchmarks, so the pipeline validates neither a benchmark's own compilation nor a performance claim: build and run
both locally, and report the measurement conditions this guide requires.

### GPU kernel benchmarks

`benchmarks/GPU/` times the kernels Eigen generates for `GpuDevice` (elementwise expressions, launch overhead,
reductions, contractions and allocation) against hand-written kernels and vendor baselines. It is part of the supported
benchmark project behind an option, so the CPU tree builds as before:

```bash
cmake -G Ninja -S benchmarks -B build-bench-gpu -DCMAKE_BUILD_TYPE=Release -DEIGEN_BENCH_CUDA=ON \
      -DEIGEN_BENCH_CPU=OFF -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-bench-gpu
./build-bench-gpu/GPU/bench_gpu_elementwise --benchmark_repetitions=10 --benchmark_report_aggregates_only=true
```

The subtree holds `bench_gpu_elementwise`, `bench_gpu_launch`, `bench_gpu_reduction` (against CUB),
`bench_gpu_contraction` (against cuBLAS) and `bench_gpu_alloc`.

`CMAKE_CUDA_ARCHITECTURES=native` needs CMake 3.24; name the architecture (`89`) otherwise. Device time is measured
with events around a batch of launches (`eigen_bench::timeLaunches`), using `UseManualTime()` to report time per
launch. Allocation benchmarks use `UseRealTime()` for the host API cost; `host_us_per_launch` is the host-side cost of enqueueing, `bytes_per_second` counts every
operand read once and the result written once. Results are checked against a reference outside the timed loop.
Quote the `GPU:` line the binary prints (also in the JSON context) with every number, and say whether clocks were
locked; a laptop under WSL2 cannot lock them.

## Adding A Benchmark

One family per translation unit, `bench_<topic>.cpp` under the module directory (`benchmarks/LU/bench_lu.cpp`),
registered beside it with `eigen_add_benchmark(<target> <source> [LIBRARIES ...] [DEFINITIONS ...])`, which links
`benchmark_main`, compiles at `-O3` with `NDEBUG`, and takes the include path from the tree. Do not merge families
into one file: code-layout shifts between combined and separate binaries have shown up as deltas of tens of percent
in kernels that did not change. Multi-threaded benchmarks call `UseRealTime()` on the registration to measure elapsed
time. The default CPU timer measures the main thread and omits internally spawned workers; `MeasureProcessCPUTime()`
includes those workers when total CPU consumption is also needed. See Google Benchmark's
[CPU timers](https://google.github.io/benchmark/user_guide.html#cpu-timers).

## Benchmark Design

- Benchmark the user-visible operation affected by the change, with representative scalar types, sizes, shapes,
  storage layouts, sparsity, and thread counts. Include transition sizes where a kernel or blocking strategy changes.
- Confirm the registered arguments actually reach the changed code — a size that falls off the fast path, or no case
  at all for the affected configuration, measures something else while looking green. Check hand-written
  `bytes_per_second`/items multipliers against the operation; a miscount silently rescales every reported rate.
- Keep allocation, input generation, validation, and unrelated setup outside the timed region. Prevent dead-code
  elimination with Google Benchmark's `DoNotOptimize` and `ClobberMemory` where appropriate.
- Validate results outside the measured loop. A faster incorrect kernel is not a useful result.
- Use enough work per iteration to dominate timer noise without hiding important small-problem behavior. Report
  meaningful rates or byte/operation counters when they improve interpretation.
- Compare the change against the relevant baseline with identical compiler, optimization, ISA, dependency, and
  benchmark arguments. Record the commit, hardware, compiler, flags, and command needed to reproduce the result.

## Argument Grids

Express static grids declaratively on the registration:

- `Args({a, b})` for individual points.
- `Range`, `DenseRange`, or `Ranges` for swept dimensions.
- `ArgsProduct({{...}, {...}})` for Cartesian products.

Do not use `Apply()`. Its callback is typed on `benchmark::internal::Benchmark*`, a library-internal name that
benchmark sources must not reference. A grid that appears to need it is expressible by enumerating the points in
`ArgsProduct` or `Args`, or by registering several benchmarks.

## Running Measurements

1. Check `uptime` and stop or finish competing builds and compute-heavy work. Run only one benchmark process at a
   time; concurrent benchmarks invalidate both measurements.
2. Keep the machine, CPU affinity, power/governor policy, thermal state, compiler, flags, ISA, and dependencies as
   constant as practical. Disclose anything that could not be controlled.
3. Use multiple repetitions, for example `--benchmark_repetitions=10`, and retain raw results. Compare medians plus a
   dispersion measure such as MAD, IQR, or standard deviation; do not select the best run.
4. For before/after binaries, alternate separate invocations (`A, B, A, B`) to expose thermal or background-load
   drift. Use the same benchmark filter and arguments for each pair.
5. Re-run suspicious or noisy cases. Treat changes smaller than the observed run-to-run variation as inconclusive,
   not as wins or regressions.

When the machine cannot be made quiet enough for the effect size, deterministic counters are the honest measurement:
Callgrind instruction counts, allocation counts (e.g. `-Wl,--wrap=malloc`), with identical result checksums across
both variants. Report them as counter measurements naming the tool, not as timings; that plus a statement that wall
clock was inconclusive is a complete performance claim, where an unqualified ratio from a loaded host is not.

Never infer a general speedup from one convenient size or one warm run. State the tested domain, include regressions
as well as improvements, and keep numerical accuracy results separate from performance measurements.

## Supporting Performance Evidence

A good merge request connects the hypothesis, the code change, and supporting evidence. Benchmark measurements
establish the observed performance effect; Callgrind counts and/or assembly analysis help test whether the change
realizes the predicted reduction in cost. For performance-critical changes, strongly prefer this complementary
evidence even when timings are stable. Choose the tools that test the hypothesis; neither Callgrind nor assembly
analysis is mandatory for every contribution.

- **Callgrind:** Compare before/after instruction counts (`Ir`) for the affected operation using identical inputs,
  compiler flags, ISA, and a fixed number of iterations. Isolate the operation from startup, unrelated allocation, input
  generation, and benchmark calibration; pausing the benchmark timer does not pause Callgrind collection. Report
  counts per operation and the measured region, rather than comparing whole-process totals from runs with different
  amounts of work. Use optimized builds with debug information for attribution, and retain the commands, tool version,
  and relevant `callgrind_annotate` output. See the
  [Callgrind manual](https://valgrind.org/docs/manual/cl-manual.html) for collection controls. Label optional cache
  and branch simulation results as simulated events.
- **Assembly:** Compare the generated code for the same representative instantiation before and after the change,
  using the benchmark's compiler, optimization flags, and target ISA. Inspect the hot loop in the benchmark binary
  or a small reproducer that evaluates the same Eigen expression and keeps its result observable. Include a short
  annotated excerpt or diff showing the relevant change, such as removed loads, stores, shuffles, branches, spills,
  or calls; vectorization; or changed loop dependencies. Record the build and disassembly commands and explain how
  the inspected code relates to the benchmark case. Inspect Eigen's code only: a proprietary library that the
  benchmark links is measured from outside and never disassembled or annotated; see [`provenance.md`](provenance.md).

Connect this evidence to the measured results in the contribution's performance summary. Instruction counts and
assembly explain mechanisms; fewer instructions alone do not establish a speedup on a particular CPU. Investigate
results that contradict the hypothesis or disagree with timings, and report unresolved uncertainty. State the
workloads and configurations over which the evidence supports the claim. If tooling cannot analyze the relevant ISA
or backend, state that limitation and use applicable evidence; do not silently change the target and attribute those
results to the original configuration.
