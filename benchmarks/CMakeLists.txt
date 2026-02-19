cmake_minimum_required(VERSION 3.10)
project(EigenBenchmarks CXX)

find_package(benchmark REQUIRED)

# Eigen is a header-only library; find it relative to this directory.
set(EIGEN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Helper: add a Google Benchmark target.
#   eigen_add_benchmark(name source [LIBRARIES lib1 lib2 ...] [DEFINITIONS def1 def2 ...])
function(eigen_add_benchmark name source)
  cmake_parse_arguments(BENCH "" "" "LIBRARIES;DEFINITIONS" ${ARGN})
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE ${EIGEN_SOURCE_DIR})
  target_link_libraries(${name} PRIVATE benchmark::benchmark benchmark::benchmark_main)
  if(BENCH_LIBRARIES)
    target_link_libraries(${name} PRIVATE ${BENCH_LIBRARIES})
  endif()
  target_compile_options(${name} PRIVATE -O3 -DNDEBUG)
  if(BENCH_DEFINITIONS)
    target_compile_definitions(${name} PRIVATE ${BENCH_DEFINITIONS})
  endif()
endfunction()

# --- Dense benchmarks ---
eigen_add_benchmark(bench_cholesky benchCholesky.cpp)
eigen_add_benchmark(bench_eigensolver benchEigenSolver.cpp)
eigen_add_benchmark(bench_fft benchFFT.cpp)
eigen_add_benchmark(bench_geometry_transforms benchGeometry.cpp)
eigen_add_benchmark(bench_vecadd benchVecAdd.cpp)
eigen_add_benchmark(bench_gemm benchGemm.cpp)
eigen_add_benchmark(bench_gemm_double benchGemm.cpp DEFINITIONS SCALAR=double)
eigen_add_benchmark(bench_gemv benchGemv.cpp)
eigen_add_benchmark(bench_move_semantics bench_move_semantics.cpp)
eigen_add_benchmark(bench_reverse bench_reverse.cpp)
eigen_add_benchmark(bench_dense_solvers dense_solvers.cpp)
eigen_add_benchmark(bench_trsm bench_trsm.cpp)
eigen_add_benchmark(bench_svd bench_svd.cpp)
eigen_add_benchmark(bench_eig33 eig33.cpp)
eigen_add_benchmark(bench_geometry geometry.cpp)
eigen_add_benchmark(bench_quatmul quatmul.cpp)

# --- Sparse benchmarks ---
eigen_add_benchmark(bench_sparse_dense_product sparse_dense_product.cpp)
eigen_add_benchmark(bench_sparse_product sparse_product.cpp)
eigen_add_benchmark(bench_sparse_transpose sparse_transpose.cpp)

# --- GEMM blocking parameter sweep ---
eigen_add_benchmark(bench_blocking_sizes benchmark_blocking_sizes.cpp)

# --- AOCL benchmark (AOCL optional) ---
eigen_add_benchmark(bench_aocl benchmark_aocl.cpp)

# Optional: BLAS GEMM comparison benchmark (requires CBLAS)
find_package(BLAS QUIET)
if(BLAS_FOUND)
  eigen_add_benchmark(bench_blas_gemm benchBlasGemm.cpp
    LIBRARIES ${BLAS_LIBRARIES}
    DEFINITIONS HAVE_BLAS)
endif()
