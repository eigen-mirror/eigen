# GPU benchmarks require CUDA runtime + cuSOLVER.
# Build separately from the main benchmark tree since they need CUDA toolchain.
#
# Usage:
#   cmake -G Ninja -B build-bench-gpu -S benchmarks/GPU \
#         -DCMAKE_CUDA_ARCHITECTURES=89
#   cmake --build build-bench-gpu
#
# Profiling:
#   nsys profile --trace=cuda ./build-bench-gpu/bench_gpu_solvers
#   ncu --set full -o profile ./build-bench-gpu/bench_gpu_solvers --benchmark_filter=BM_GpuLLT_Compute/4096

cmake_minimum_required(VERSION 3.18)
project(EigenGpuBenchmarks CXX)

find_package(benchmark REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(EIGEN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")

function(eigen_add_gpu_benchmark name source)
  cmake_parse_arguments(BENCH "" "" "LIBRARIES;DEFINITIONS" ${ARGN})
  if(NOT IS_ABSOLUTE "${source}")
    set(source "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
  endif()
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE
    ${EIGEN_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(${name} PRIVATE
    benchmark::benchmark benchmark::benchmark_main
    CUDA::cudart CUDA::cusolver CUDA::cublas)
  if(BENCH_LIBRARIES)
    target_link_libraries(${name} PRIVATE ${BENCH_LIBRARIES})
  endif()
  target_compile_options(${name} PRIVATE -O3 -DNDEBUG)
  target_compile_definitions(${name} PRIVATE EIGEN_USE_GPU)
  if(BENCH_DEFINITIONS)
    target_compile_definitions(${name} PRIVATE ${BENCH_DEFINITIONS})
  endif()
endfunction()

# Solver benchmarks: LLT/LU compute + solve, host vs device paths, CPU baselines.
eigen_add_gpu_benchmark(bench_gpu_solvers bench_gpu_solvers.cpp)
eigen_add_gpu_benchmark(bench_gpu_solvers_float bench_gpu_solvers.cpp DEFINITIONS SCALAR=float)

# Chaining benchmarks: async pipeline efficiency, host-roundtrip vs device chain.
eigen_add_gpu_benchmark(bench_gpu_chaining bench_gpu_chaining.cpp)
eigen_add_gpu_benchmark(bench_gpu_chaining_float bench_gpu_chaining.cpp DEFINITIONS SCALAR=float)

# Batching benchmarks: multi-stream concurrency for many small systems.
eigen_add_gpu_benchmark(bench_gpu_batching bench_gpu_batching.cpp)
eigen_add_gpu_benchmark(bench_gpu_batching_float bench_gpu_batching.cpp DEFINITIONS SCALAR=float)

# FFT benchmarks: 1D/2D C2C, R2C, C2R throughput and plan reuse.
eigen_add_gpu_benchmark(bench_gpu_fft bench_gpu_fft.cpp LIBRARIES CUDA::cufft)
eigen_add_gpu_benchmark(bench_gpu_fft_double bench_gpu_fft.cpp LIBRARIES CUDA::cufft DEFINITIONS SCALAR=double)
