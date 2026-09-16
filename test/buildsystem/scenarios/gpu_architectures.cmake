# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

include("${EIGEN_SOURCE_DIR}/cmake/EigenGpuTesting.cmake")

# Architecture selection must not require an installed toolkit or GPU.
function(check_architectures toolkit eigen_arch cmake_arch environment_arch expected)
  set(CUDAToolkit_VERSION "${toolkit}")
  set(EIGEN_CUDA_COMPUTE_ARCH "${eigen_arch}")
  set(CMAKE_CUDA_ARCHITECTURES "${cmake_arch}")
  set(ENV{CUDAARCHS} "${environment_arch}")
  ei_cuda_resolve_compute_arch()
  bs_assert_streq("${EIGEN_CUDA_COMPUTE_ARCH}" "${expected}" "CUDA architecture selection")
endfunction()

check_architectures(12.8 "" "" "" 60)
check_architectures(13.3 "" "" "" 75)
check_architectures(13.3 "" "89" "" 89)
check_architectures(13.3 "" "75;89-real" "" "75;89-real")
check_architectures(13.3 "" "native" "" native)
check_architectures(13.3 "" "OFF" "" OFF)
check_architectures(13.3 "" "" "89" 89)
check_architectures(13.3 "" "89" "75" 89)
check_architectures(13.3 "86" "89" "75" 86)

# Observe the architecture setting at language enablement, before compiler ABI detection.
macro(enable_language language)
  bs_assert_streq("${language}" "HIP" "enabled GPU language")
  bs_assert_streq("${CMAKE_HIP_ARCHITECTURES}" "${expected}" "HIP architectures before compiler detection")
endmacro()

function(check_hip_architectures eigen_arch cmake_arch expected)
  set(EIGEN_TEST_CUDA OFF)
  set(EIGEN_TEST_HIP ON)
  unset(EIGEN_HIP_ARCHITECTURES CACHE)
  if(NOT "${eigen_arch}" STREQUAL "")
    set(EIGEN_HIP_ARCHITECTURES "${eigen_arch}")
  endif()
  set(CMAKE_HIP_ARCHITECTURES "${cmake_arch}")
  ei_gpu_testing_enable()
endfunction()

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
  check_hip_architectures("" ""
    "gfx900;gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151")
  check_hip_architectures("gfx906;gfx1100" "" "gfx906;gfx1100")
  check_hip_architectures("gfx906" "gfx942" "gfx942")
  check_hip_architectures("gfx906" "OFF" "OFF")
endif()

# Trailing arguments are passed to the consumer's configure.
function(check_cuda_route route architectures expected_result expected_flags)
  string(MD5 configuration "${ARGN}")
  string(SUBSTRING "${configuration}" 0 8 configuration)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -G "${GENERATOR}"
            -S "${BS_CONSUMER_DIR}/gpu_architectures"
            -B "${WORK_DIR}/${route}-${architectures}-${configuration}"
            "-DEIGEN_SOURCE_DIR=${EIGEN_SOURCE_DIR}"
            "-DROUTE=${route}" "-DARCHITECTURES=${architectures}" "-DEXPECTED_FLAGS=${expected_flags}" ${ARGN}
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE output)
  if(expected_result STREQUAL "FAILURE")
    if(result EQUAL 0 OR NOT output MATCHES "architectures only, for example 75;89")
      bs_fail("${route} must reject ${architectures} with a numeric-architecture diagnostic:\n${output}")
    endif()
  elseif(NOT result EQUAL 0)
    bs_fail("${route} rejected ${architectures}:\n${output}")
  endif()
endfunction()

check_cuda_route(nvc 75 SUCCESS " -cuda -gpu=cc75 ")
check_cuda_route(nvc "75,89" SUCCESS " -cuda -gpu=cc75 -gpu=cc89 ")
check_cuda_route(windows-clang 75 SUCCESS " --cuda-path=/cuda --cuda-gpu-arch=sm_75 ")
check_cuda_route(windows-clang "75,89" SUCCESS " --cuda-path=/cuda --cuda-gpu-arch=sm_75 --cuda-gpu-arch=sm_89 ")
foreach(arch native OFF all all-major 89-real 89-virtual sm_89 "75,native")
  check_cuda_route(nvc "${arch}" FAILURE "")
  check_cuda_route(windows-clang "${arch}" FAILURE "")
endforeach()
check_cuda_route(cuda-language 75 SUCCESS "")
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  foreach(arch native OFF all all-major 89-real 89-virtual "75,89-real")
    check_cuda_route(cuda-language "${arch}" SUCCESS "")
  endforeach()
endif()

# An explicit nvcc selects the toolkit around it unless CUDAToolkit_ROOT does, and only nvcc does.
set(toolkit "${WORK_DIR}/cuda-12.8")
set(nvcc "${toolkit}/bin/nvcc")
if(CMAKE_HOST_WIN32)
  string(APPEND nvcc ".exe")
endif()
file(WRITE "${nvcc}" "")
check_cuda_route(cuda-language 75 SUCCESS "" "-DCMAKE_CUDA_COMPILER=${nvcc}" "-DEXPECTED_TOOLKIT_ROOT=${toolkit}")
check_cuda_route(cuda-language 75 SUCCESS "" "-DCMAKE_CUDA_COMPILER=${nvcc}"
                 "-DCUDAToolkit_ROOT=${WORK_DIR}/cuda-13.4" "-DEXPECTED_TOOLKIT_ROOT=${WORK_DIR}/cuda-13.4")
check_cuda_route(nvc 75 SUCCESS " -cuda -gpu=cc75 " "-DCMAKE_CUDA_COMPILER=${nvcc}")
