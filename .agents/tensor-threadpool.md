# Tensor and Thread-Pool Changes

Use this guide for `contrib/Eigen/Tensor`, `Eigen/ThreadPool`, Core's custom GEMM thread-pool backend, and explicit
thread-pool devices. The repository-root `AGENTS.md` still applies.

## Compatibility and risk

Tensor and ThreadPool are foundational to TensorFlow and other downstream users. The `contrib/` location describes
Tensor's API-stability policy, not its importance. Changes to signatures, header layout, evaluation order, allocation,
synchronization, numerical behavior, or performance can have a large downstream impact.

- Prefer additive changes and preserve public header paths. Use `<contrib/Eigen/Tensor>` and
  `<Eigen/ThreadPool>`; never expose implementation-header includes to users.
- Paths below `unsupported/Eigen/`, including `unsupported/Eigen/CXX11/`, are backward-compatibility forwarding
  shims only. New code must use the canonical `contrib/Eigen/` headers and must not add headers under either.
- Preserve `EIGEN_DEVICE_FUNC` on code reachable by CUDA, HIP, or SYCL device evaluation.
- Treat evaluator flags, layouts, scalar/packet/block paths, zero-sized tensors, aliasing, and asynchronous object
  lifetimes as part of the behavior under test.
- Changes to contraction, reduction, convolution, morphing, scheduling, or the cost model are performance-sensitive.
  Add or update a benchmark and compare representative shapes, layouts, thread counts, and scalar types.
- Call out intentional compatibility or performance changes prominently in the merge request.

## Keep the threading mechanisms separate

### OpenMP

OpenMP is Core's primary implicit multithreading mechanism and covers the algorithms listed in
`doc/TopicMultithreading.dox`. It is controlled through the compiler's OpenMP support, `Eigen::setNbThreads`, and the
OpenMP runtime. Do not infer that every algorithm in that list is also supported by the custom GEMM thread pool.

### `EIGEN_GEMM_THREADPOOL`

This macro selects Eigen's custom thread-pool backend for general dense matrix-matrix products only. It is mutually
exclusive with OpenMP. Define it before including Eigen, create an `Eigen::ThreadPool`, and register that pool with
`Eigen::setGemmThreadPool(&pool)` before concurrent GEMM work begins.

Eigen stores the registered pointer in a process-wide global, and the caller still owns the pool. The pool must outlive
every GEMM that uses it; do not replace it while a product is running. `Eigen::setNbThreads` controls the active thread
limit, but registering a pool resets that limit to the pool's thread count. Passing `nullptr` currently returns the
registered pool; it does not clear the registration. Treat `doc/TopicMultithreading.dox` and
`Eigen/src/Core/products/Parallelizer.h` as the current API and implementation references.

### `CoreThreadPoolDevice`

`Eigen::CoreThreadPoolDevice` is an explicit device for parallel Core coefficient-wise assignment:

```cpp
#include <Eigen/ThreadPool>

Eigen::ThreadPool pool(thread_count);
Eigen::CoreThreadPoolDevice device(pool);
destination.device(device) = expression;
```

It is separate from implicit GEMM parallelization. Its tests belong with the device and evaluator tests, such as
`test/assignment_threaded.cpp`, not only with the GEMM tests.

### Tensor `ThreadPoolDevice`

Define `EIGEN_USE_THREADS` before `<contrib/Eigen/Tensor>`, then construct a `ThreadPoolDevice` over an existing
`ThreadPoolInterface` and evaluate explicitly:

```cpp
Eigen::ThreadPool pool(pool_threads);
Eigen::ThreadPoolDevice device(&pool, execution_threads);
output.device(device) = expression;
```

The device does not own the pool. The pool, allocator, input storage, output storage, and callback state must remain
alive until a synchronous evaluation returns or an asynchronous one signals completion. Tensor's executor,
contraction, reduction, and device code have code paths specific to `ThreadPoolDevice`, so a serial `DefaultDevice`
test alone is insufficient.
See `contrib/Eigen/src/Tensor/README.md` and `TensorDeviceThreadPool.h`.

## Evaluator capability flags and cost

Each evaluator capability flag is a separate promise, and the executor combines them: it vectorizes when `PacketAccess`
is set and tiles when `BlockAccess && PreferBlockAccess` holds. Setting a flag in more cases makes a broader promise.
The execution paths also treat evaluator state differently. Threaded coefficient evaluation copies the evaluator for
each worker range, while tiled evaluation shares one evaluator across concurrent block tasks. A functor with mutable
state therefore races under tiling even when its coefficient and packet paths are correct. A capability may
legitimately depend on the `Device`; prefer the conservative answer for stateful or unannotated user functors (see
rule 6 in the root `AGENTS.md`).

Eigen chooses the thread count from `costPerCoeff()`, so the cost must describe the code path actually taken. When
the packet path is taken only under a condition, apply the same condition in the cost. Where the packet path gathers
lane by lane, charge the nested evaluator's work as scalar. `TensorStriding.h` is the reference.

## Scheduling changes

- Preserve the `ThreadPoolInterface` contract, including `Schedule`, `ScheduleWithHint`, `CurrentThreadId`,
  cancellation behavior, and caller ownership.
- Test one-thread and multi-thread execution, work invoked from a worker, completion/wakeup behavior, and shutdown with
  pending or cancelled work when those paths are affected.
- Avoid blocking a worker on work that can only run on the same exhausted pool. Make callback and barrier lifetime
  rules explicit in code when they are not self-evident.
- `DenseBase::Random()` and `setRandom()` use `std::rand` and are not re-entrant. Do not call them concurrently;
  pre-generate inputs or use thread-local `<random>` generators through `NullaryExpr`.
- Cost-model and grain-size changes need both small-workload overhead measurements and large-workload throughput
  measurements. Check oversubscription and nested parallelism rather than assuming more threads are faster.
- Benchmark only on an otherwise idle system, one benchmark process at a time, and report repeated measurements rather
  than a single timing.

## Validation

- Thread-pool internals: run the affected `threads_*` target, especially event-count, run-queue, non-blocking-pool, or
  fork-join tests.
- Custom GEMM pool: run `product_threaded` and the ordinary product tests affected by the change.
- Core explicit device: build and run the test built from `test/assignment_threaded.cpp` if it is registered in the
  current test configuration.
- Tensor pool/device changes: run `tensor_thread_pool`, `tensor_executor`, and the focused operation tests such as
  contraction or reduction.
- Tensor behavior shared with accelerators: also follow `simd-gpu.md` and run the locally available device tests.
- Report unavailable sanitizers, GPU toolchains, platforms, and downstream TensorFlow validation explicitly.

Use `test/CMakeLists.txt`, `contrib/test/CMakeLists.txt`, and the checked-out CMake configuration as the source of
truth for target names. Do not maintain a duplicate test or backend inventory here.
