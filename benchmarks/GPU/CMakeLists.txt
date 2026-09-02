# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# Benchmarks of the GPU kernels Eigen generates: the Tensor module's GpuDevice executor, its launches, and the
# hand-written and vendor baselines they are measured against. (contrib/benchmarks/GPU benchmarks the
# library-dispatch module contrib/Eigen/GPU, which maps expressions onto cuBLAS/cuSOLVER calls; nothing here
# depends on it.) Built when EIGEN_BENCH_CUDA is ON, see ../CMakeLists.txt.

# eigen_add_gpu_benchmark(name source [LIBRARIES lib ...] [DEFINITIONS def ...])
# Mirrors eigen_add_benchmark. The binary carries its own main() (gpu_bench_common.h), which prints the device it
# runs on and records it in the JSON context, so benchmark::benchmark_main is not linked. NDEBUG is defined, as for
# the CPU benchmarks; runtime failures are still caught, since GpuDevice and the harness check every call through
# EIGEN_GPU_RUNTIME_CHECK, which aborts under NDEBUG instead of asserting.
function(eigen_add_gpu_benchmark name source)
  cmake_parse_arguments(BENCH "" "" "LIBRARIES;DEFINITIONS" ${ARGN})
  if(NOT IS_ABSOLUTE "${source}")
    set(source "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
  endif()
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE ${EIGEN_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(${name} PRIVATE benchmark::benchmark CUDA::cudart)
  if(BENCH_LIBRARIES)
    target_link_libraries(${name} PRIVATE ${BENCH_LIBRARIES})
  endif()
  # Every source here is a .cu compiled as CUDA.
  target_compile_options(${name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr;-lineinfo>)
  target_compile_definitions(${name} PRIVATE NDEBUG EIGEN_USE_GPU)
  if(BENCH_DEFINITIONS)
    target_compile_definitions(${name} PRIVATE ${BENCH_DEFINITIONS})
  endif()
endfunction()

# Elementwise expressions through the Tensor executor, float/half/bfloat16, aligned and unaligned maps, against a
# hand-written float4 kernel and the device-to-device copy roofline.
eigen_add_gpu_benchmark(bench_gpu_elementwise bench_gpu_elementwise.cu)
# Launch overhead: an executor launch of a tiny expression against a raw <<<>>> launch, LAUNCH_GPU_KERNEL and
# cudaLaunchKernel.
eigen_add_gpu_benchmark(bench_gpu_launch bench_gpu_launch.cu)
