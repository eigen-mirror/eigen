# Benchmarking

Use this guidance for performance-sensitive changes and benchmark reviews. Performance claims need a benchmark that
ships in the same merge request. Correctness tests still ship separately from the benchmark. Run them before timing.

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

When CMake finds `CUDAToolkit`, the contrib benchmark project automatically adds its `GPU` subdirectory. That
subdirectory also needs a working CUDA compiler and a selected CUDA architecture. On a host with only a partial toolkit
installation, configure CPU-only contrib benchmarks with `-DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE` or report the
GPU configuration as unavailable.

Consult [`benchmarks/CMakeLists.txt`](../benchmarks/CMakeLists.txt) and
[`contrib/benchmarks/CMakeLists.txt`](../contrib/benchmarks/CMakeLists.txt) for current targets and compile
settings. CUDA benchmarks also have a standalone project and instructions in
[`contrib/benchmarks/GPU/CMakeLists.txt`](../contrib/benchmarks/GPU/CMakeLists.txt). No CI job builds or runs
benchmarks, so CI checks neither that a benchmark compiles nor that a performance claim holds. Build and run the
benchmark locally, and report the measurement conditions this guide requires.

### GPU kernel benchmarks

`benchmarks/GPU/` times the kernels Eigen generates for `GpuDevice` (elementwise expressions, launch overhead,
reductions, contractions and allocation) against hand-written kernels and vendor baselines. It is part of the supported
benchmark project, but only builds when the `EIGEN_BENCH_CUDA` option is on, so the CPU benchmarks build as before:

```bash
cmake -G Ninja -S benchmarks -B build-bench-gpu -DCMAKE_BUILD_TYPE=Release -DEIGEN_BENCH_CUDA=ON \
      -DEIGEN_BENCH_CPU=OFF -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-bench-gpu
./build-bench-gpu/GPU/bench_gpu_elementwise --benchmark_repetitions=10 --benchmark_report_aggregates_only=true
```

The subtree holds `bench_gpu_elementwise`, `bench_gpu_launch`, `bench_gpu_reduction` (against CUB),
`bench_gpu_contraction` (against cuBLAS) and `bench_gpu_alloc`.

`CMAKE_CUDA_ARCHITECTURES=native` needs CMake 3.24. With older CMake, name the architecture instead, for example
`89`. The benchmarks measure device time with CUDA events around a batch of launches (`eigen_bench::timeLaunches`)
and report the time per launch through `UseManualTime()`. Allocation benchmarks use `UseRealTime()` to measure the
cost of the host API. The `host_us_per_launch` counter is the host-side cost of enqueueing a launch. The
`bytes_per_second` counter counts every operand as read once and the result as written once. The benchmarks check
their results against a reference outside the timed loop. With every number you report, quote the `GPU:` line the
binary prints (it is also in the JSON context), and say whether the clocks were locked. A laptop under WSL2 cannot
lock them.

## Adding A Benchmark

Put each benchmark family in its own translation unit, `bench_<topic>.cpp` in the module directory
(`benchmarks/LU/bench_lu.cpp`). Register it in the `CMakeLists.txt` beside it with
`eigen_add_benchmark(<target> <source> [LIBRARIES ...] [DEFINITIONS ...])`, which links `benchmark_main`, compiles at
`-O3` with `NDEBUG`, and takes the include path from the tree. Do not merge families into one file: combining them
changes the code layout, and that alone has shifted timings of unchanged kernels by tens of percent. Multi-threaded
benchmarks call `UseRealTime()` on the registration to measure elapsed time. The default CPU timer measures only the
main thread and misses the worker threads. When total CPU consumption is also needed,
`MeasureProcessCPUTime()` includes those workers. See Google Benchmark's
[CPU timers](https://google.github.io/benchmark/user_guide.html#cpu-timers).

## Benchmark Design

- Benchmark the user-visible operation affected by the change, with representative scalar types, sizes, shapes,
  storage layouts, sparsity, and thread counts. Include transition sizes where a kernel or blocking strategy changes.
- Confirm that the registered arguments actually reach the changed code. A size that misses the fast path, or a grid
  with no case for the affected configuration, measures something else while the results still look fine. Check
  hand-written `bytes_per_second` and items multipliers against the operation. A wrong count silently rescales every
  reported rate.
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

Do not use `Apply()`. Its callback takes a `benchmark::internal::Benchmark*`, a library-internal name that benchmark
sources must not reference. A grid that seems to need `Apply()` can be written by listing its points with
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

When the machine cannot be made quiet enough to resolve the effect, deterministic counters are the honest measurement:
Callgrind instruction counts, allocation counts (e.g. `-Wl,--wrap=malloc`), with identical result checksums for both
variants. Report them as counter measurements that name the tool, not as timings. Such a counter measurement, plus a
statement that wall-clock timing was inconclusive, makes a complete performance claim. An unqualified ratio from a
loaded host does not.

Never infer a general speedup from one convenient size or one warm run. State the tested domain, include regressions
as well as improvements, and keep numerical accuracy results separate from performance measurements.

## Supporting Performance Evidence

A good merge request connects the hypothesis, the code change, and supporting evidence. Benchmark measurements
establish the observed performance effect. Callgrind counts, assembly analysis, or both help test whether the change
achieves the cost reduction the hypothesis predicted. For performance-critical changes, strongly prefer this
complementary evidence even when timings are stable. Choose the tools that test the hypothesis; neither Callgrind nor
assembly analysis is mandatory for every contribution.

- **Callgrind:** Compare before/after instruction counts (`Ir`) for the affected operation using identical inputs,
  compiler flags, ISA, and a fixed number of iterations. Isolate the operation from startup, unrelated allocation, input
  generation, and benchmark calibration. Pausing the benchmark timer does not pause Callgrind collection. Report
  counts per operation and the measured region, instead of comparing whole-process totals from runs that did
  different amounts of work. Use optimized builds with debug information, so counts can be attributed to source, and
  retain the commands, tool version, and relevant `callgrind_annotate` output. See the
  [Callgrind manual](https://valgrind.org/docs/manual/cl-manual.html) for collection controls. Label optional cache
  and branch simulation results as simulated events.
- **Assembly:** Compare the generated code for the same representative instantiation before and after the change, using
  the benchmark's compiler, optimization flags, and target ISA. Inspect the hot loop in the benchmark binary or a small
  reproducer that evaluates the same Eigen expression and keeps its result observable. Include a short annotated excerpt
  or diff showing the relevant change, such as removed loads, stores, shuffles, branches, spills or calls, a change in
  vectorization, or changed loop dependencies. Record the build and disassembly commands and explain how the inspected
  code relates to the benchmark case. Inspect only Eigen's code. A proprietary library that the benchmark links is
  measured from outside, never disassembled; see [`provenance.md`](provenance.md).

Connect this evidence to the measured results in the contribution's performance summary. Instruction counts and
assembly explain mechanisms. Fewer instructions alone do not establish a speedup on a particular CPU. Investigate
results that contradict the hypothesis or disagree with timings, and report unresolved uncertainty. State the
workloads and configurations over which the evidence supports the claim. If the tools cannot analyze the relevant ISA
or backend, say so and use the evidence that does apply. Do not silently switch to a different target and present
its results as if they came from the original configuration.
