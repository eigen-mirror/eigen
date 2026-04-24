# GPU module tests. Included from unsupported/test/CMakeLists.txt inside the
# if(CUDA_FOUND AND EIGEN_TEST_CUDA) block.

find_package(CUDAToolkit)

# ei_add_gpu_test(<name> [EXTRA_LIBS <lib>...] [EXTRA_INCLUDES <dir>...] [EXTRA_DEFINES <def>...])
#
# Thin wrapper around ei_add_test for GPU module tests. Delegates to
# ei_add_test so CALL_SUBTEST_N splitting and EIGEN_TEST_MAX_SIZE handling
# work the same way as the rest of the test suite, then layers on the
# GPU-specific properties (EXTRA_{LIBS,INCLUDES,DEFINES}, the
# "Official;gpu" labels, SKIP_RETURN_CODE=77, buildtests_gpu dependency).
#
# GPU module tests are plain .cpp (host-compiled) by design: they
# orchestrate NVIDIA library calls and never launch kernels themselves,
# which avoids NVCC instantiating Eigen CPU packet ops for CUDA vector
# types. That keeps them outside ei_add_test's "is_gpu_test" path (which
# triggers on EIGEN_ADD_TEST_FILENAME_EXTENSION=cu), so we set the
# SKIP_RETURN_CODE and labels ourselves.
function(ei_add_gpu_test test_name)
  cmake_parse_arguments(ARG "" "" "EXTRA_LIBS;EXTRA_INCLUDES;EXTRA_DEFINES" ${ARGN})

  # Delegate to ei_add_test. The third arg is a semicolon-separated library
  # list; CUDA::cudart_static is always needed for any host test that
  # touches a device-allocated buffer.
  ei_add_test(${test_name} "" "CUDA::cudart_static;${ARG_EXTRA_LIBS}")

  # Discover targets ei_add_test created. The scan mirrors ei_add_test's own
  # scan so we see the same set of suffixes: one target per distinct
  # CALL_SUBTEST_N on EIGEN_SPLIT_LARGE_TESTS, else a single target.
  file(READ "${test_name}.cpp" _src)
  string(REGEX MATCHALL "CALL_SUBTEST_[0-9]+" _matches "${_src}")
  string(REGEX REPLACE "CALL_SUBTEST_" "" _suffixes "${_matches}")
  list(REMOVE_DUPLICATES _suffixes)
  set(_targets)
  if(EIGEN_SPLIT_LARGE_TESTS AND _suffixes)
    foreach(s ${_suffixes})
      list(APPEND _targets "${test_name}_${s}")
    endforeach()
  else()
    list(APPEND _targets "${test_name}")
  endif()

  foreach(t ${_targets})
    target_include_directories(${t} PRIVATE
      "${CUDA_TOOLKIT_ROOT_DIR}/include"
      "${CMAKE_CURRENT_BINARY_DIR}"
      ${ARG_EXTRA_INCLUDES})
    if(ARG_EXTRA_DEFINES)
      target_compile_definitions(${t} PRIVATE ${ARG_EXTRA_DEFINES})
    endif()
    add_dependencies(buildtests_gpu ${t})
    set_property(TEST ${t} APPEND PROPERTY LABELS "Official;gpu")
    set_property(TEST ${t} PROPERTY SKIP_RETURN_CODE 77)
  endforeach()
endfunction()

# DeviceMatrix core: CUDA runtime only.
ei_add_gpu_test(device_matrix)

# cuBLAS integration (BLAS-3, triangular solves, rank-k updates, ...).
option(EIGEN_TEST_CUBLAS "Test cuBLAS integration" ON)
if(EIGEN_TEST_CUBLAS AND TARGET CUDA::cublas)
  ei_add_gpu_test(cublas EXTRA_LIBS CUDA::cublas CUDA::cusolver)
endif()

# cuSOLVER dense factorizations (LLT, LU).
option(EIGEN_TEST_CUSOLVER "Test cuSOLVER integration" ON)
if(EIGEN_TEST_CUSOLVER AND TARGET CUDA::cusolver)
  foreach(_cusolver_test IN ITEMS cusolver_llt cusolver_lu)
    ei_add_gpu_test(${_cusolver_test} EXTRA_LIBS CUDA::cusolver CUDA::cublas)
  endforeach()
endif()
